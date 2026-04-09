import logging
import time
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class StandaloneWorker(EAGLEWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.tp_rank = int(tp_rank)
        self.attn_cp_rank = int(attn_cp_rank)
        self.pp_rank = 0
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.total_verified_tokens = 0
        self.total_verified_reqs = 0
        self.total_round_forward_time_ms = 0.0
        self.total_round_forward_ct = 0

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def _is_entry_rank(self) -> bool:
        return self.pp_rank == 0 and self.tp_rank == 0 and self.attn_cp_rank == 0

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        forward_start = time.perf_counter()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(batch)
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )
            if str(batch.device).startswith("cuda"):
                torch.cuda.synchronize()
            extend_forward_time_ms = (time.perf_counter() - forward_start) * 1000
            # if self._is_entry_rank():
                # print(
                #     "[coupled-spec] "
                #     f"extend_forward_time_ms={extend_forward_time_ms:.3f}",
                #     flush=True,
                # )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            spec_info = self.draft(batch)
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = self.verify(
            batch, spec_info
        )

        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            if (
                self.server_args.enable_dp_attention
                or batch.spec_info.verified_id.shape[0] > 0
            ):
                self.forward_draft_extend_after_decode(batch)

        if str(batch.device).startswith("cuda"):
            torch.cuda.synchronize()
        round_forward_time_ms = (time.perf_counter() - forward_start) * 1000

        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
        del model_worker_batch

        verified_reqs = len(verify_output.accept_length_per_req_cpu)
        verified_tokens = int(result.num_accepted_tokens) + verified_reqs
        self.total_verified_tokens += verified_tokens
        self.total_verified_reqs += verified_reqs
        self.total_round_forward_time_ms += round_forward_time_ms
        self.total_round_forward_ct += 1
        avg_tokens_per_round = (
            self.total_verified_tokens / self.total_verified_reqs
            if self.total_verified_reqs > 0
            else 0.0
        )
        avg_round_forward_time_ms = (
            self.total_round_forward_time_ms / self.total_round_forward_ct
            if self.total_round_forward_ct > 0
            else 0.0
        )
        # if self._is_entry_rank():
        #     print(
        #         "[coupled-spec] "
        #         f"round_forward_time_ms={round_forward_time_ms:.3f} "
        #         f"accepted_this_round={int(result.num_accepted_tokens)} "
        #         f"avg_tokens_per_round={avg_tokens_per_round:.3f} "
        #         f"avg_round_forward_time_ms={avg_round_forward_time_ms:.3f}",
        #         flush=True,
        #     )
        return result
