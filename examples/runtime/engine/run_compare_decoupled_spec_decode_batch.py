#!/usr/bin/env python3
"""
Compare decoupled speculative decoding against normal decoding on one prompt batch.

The script keeps prompts exactly as read from the parquet dataset.  The
`--context-length` argument controls generation length only, and is passed as
`max_new_tokens` for both modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any

import ray
import sglang as sgl
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.speculative.decoupled_spec_io import DraftRequest, DraftResult

from run_decoupled_spec_batch import (
    DEFAULT_PROMPT_COLUMN_CANDIDATES,
    _build_chat_template_renderer,
    _normalize_prompt,
    infer_prompt_column,
)

logger = logging.getLogger(__name__)

DEFAULT_RAY_NAMESPACE = "sglang-decoupled-spec-bench"


@dataclass
class PromptSample:
    row_index: int
    prompt: str
    prompt_tokens: int


@dataclass
class ModeMetrics:
    mode: str
    generation_time_s: float
    total_generated_tokens: int
    output_throughput_tok_per_s: float
    per_request: list[dict[str, Any]]
    avg_spec_accept_length: float | None = None


def _sort_gpu_ids(gpu_ids: list[Any]) -> list[str]:
    def sort_key(value: Any) -> tuple[int, Any]:
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    return [str(value) for value in sorted(gpu_ids, key=sort_key)]


def _get_assigned_gpu_ids_from_ray() -> list[str]:
    context = ray.get_runtime_context()
    accelerator_ids = getattr(context, "get_accelerator_ids", lambda: {})()
    gpu_ids = accelerator_ids.get("GPU", [])
    if gpu_ids:
        return _sort_gpu_ids(gpu_ids)

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        return _sort_gpu_ids(
            [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        )
    return []


def _pin_actor_to_assigned_gpus(expected_num_gpus: int) -> list[str]:
    gpu_ids = _get_assigned_gpu_ids_from_ray()
    if expected_num_gpus > 0 and len(gpu_ids) < expected_num_gpus:
        raise RuntimeError(
            f"Ray assigned {len(gpu_ids)} GPUs to actor, expected at least "
            f"{expected_num_gpus}: {gpu_ids}"
        )
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    return gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation and normal decode on the same parquet "
            "prompt batch, then report latency and speedup."
        )
    )
    parser.add_argument(
        "--dataset-path",
        "--parquet-path",
        dest="dataset_path",
        required=True,
        help="Path to the parquet dataset.",
    )
    parser.add_argument(
        "--prompt-column",
        default=None,
        help=(
            "Prompt column in the parquet file. If omitted, common names are "
            f"searched in order: {DEFAULT_PROMPT_COLUMN_CANDIDATES}."
        ),
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--batch-size",
        "--bs",
        dest="batch_size",
        type=int,
        default=1,
        help="Number of valid prompts to run in one generate call.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Disable tokenizer.apply_chat_template for chat-style prompt objects.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking-style generation when building chat prompts for "
            "models such as Qwen3/Qwen3.5. Disabled by default."
        ),
    )
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Generation length. This is passed as max_new_tokens.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional prompt token upper bound. Prompts over this limit are skipped.",
    )
    parser.add_argument(
        "--target-model-path",
        required=True,
        help="Target/verifier model path.",
    )
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Draft model path.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used for prompt length filtering. Defaults to target model.",
    )
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--num-speculative-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Set sampling_params.ignore_eos=True for both decoupled speculative "
            "decoding and normal decoding. Disabled by default."
        ),
    )
    parser.add_argument(
        "--ray-address",
        default="auto",
        help="Ray cluster address. Use 'auto' for an existing cluster or local fallback on nnodes=1.",
    )
    parser.add_argument("--ray-namespace", default=DEFAULT_RAY_NAMESPACE)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--n-gpu-per-node",
        type=int,
        required=True,
        help="GPU count available on each Ray node.",
    )
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help=(
            "SGLang distributed init address. For multi-node, pass host:port "
            "or host together with --dist-init-port."
        ),
    )
    parser.add_argument(
        "--dist-init-port",
        type=int,
        default=None,
        help="Base dist init port. Decode uses base+1 to avoid reuse.",
    )
    parser.add_argument("--num-draft-replicas", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the complete benchmark result as JSON.",
    )
    return parser.parse_args()


def _build_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to count prompt tokens for this benchmark."
        ) from exc

    return AutoTokenizer.from_pretrained(tokenizer_path)


def _count_prompt_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer.encode(prompt))


def load_prompt_samples(args: argparse.Namespace) -> tuple[str, list[PromptSample], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read parquet datasets.") from exc

    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.context_length <= 0:
        raise ValueError("context-length must be positive")
    if args.max_prompt_length is not None and args.max_prompt_length <= 0:
        raise ValueError("max-prompt-length must be positive when set")

    dataset_path = Path(args.dataset_path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)
    total_rows = parquet_file.metadata.num_rows
    if args.offset >= total_rows:
        raise ValueError(
            f"offset {args.offset} is out of range for {dataset_path}; "
            f"total rows: {total_rows}"
        )

    column_names = parquet_file.schema_arrow.names
    prompt_column = args.prompt_column or infer_prompt_column(column_names)
    if prompt_column not in column_names:
        raise ValueError(
            f"prompt column {prompt_column!r} not found. Available columns: {column_names}"
        )

    tokenizer_path = args.tokenizer_path or args.target_model_path
    tokenizer = _build_tokenizer(tokenizer_path)
    chat_template_renderer = None
    if not args.disable_chat_template:
        chat_template_renderer = _build_chat_template_renderer(
            tokenizer_path, enable_thinking=args.enable_thinking
        )

    samples: list[PromptSample] = []
    skipped_empty = 0
    skipped_too_long = 0
    skipped_invalid = 0
    current_row = 0
    remaining_skip = args.offset
    reader_batch_size = max(args.batch_size, 1024)

    for record_batch in parquet_file.iter_batches(
        batch_size=reader_batch_size,
        columns=[prompt_column],
    ):
        column_values = record_batch.column(0).to_pylist()
        if remaining_skip >= len(column_values):
            remaining_skip -= len(column_values)
            current_row += len(column_values)
            continue

        start_index = remaining_skip
        for local_index in range(start_index, len(column_values)):
            row_index = current_row + local_index
            try:
                prompt = _normalize_prompt(
                    column_values[local_index],
                    row_index,
                    prompt_column,
                    chat_template_renderer,
                    enable_thinking=args.enable_thinking,
                )
            except Exception:
                skipped_invalid += 1
                continue

            if not prompt:
                skipped_empty += 1
                continue

            prompt_tokens = _count_prompt_tokens(tokenizer, prompt)
            if prompt_tokens == 0:
                skipped_empty += 1
                continue

            if (
                args.max_prompt_length is not None
                and prompt_tokens > args.max_prompt_length
            ):
                skipped_too_long += 1
                continue

            samples.append(
                PromptSample(
                    row_index=row_index,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                )
            )
            if len(samples) >= args.batch_size:
                return prompt_column, samples, total_rows

        current_row += len(column_values)
        remaining_skip = 0

    raise ValueError(
        "Not enough valid prompts were found. "
        f"requested={args.batch_size}, collected={len(samples)}, "
        f"skipped_empty={skipped_empty}, skipped_too_long={skipped_too_long}, "
        f"skipped_invalid={skipped_invalid}, offset={args.offset}, "
        f"total_rows={total_rows}, prompt_column={prompt_column!r}"
    )


def _parse_host_port(addr: str) -> tuple[str, int | None]:
    if addr.count(":") == 1:
        host, raw_port = addr.rsplit(":", 1)
        if raw_port:
            return host, int(raw_port)
    return addr, None


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _reserve_tcp_port(
    preferred_port: int | None = None,
) -> tuple[int, socket.socket]:
    def bind_port(port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.listen(1)
        return sock

    if preferred_port is not None:
        return preferred_port, bind_port(preferred_port)

    for _ in range(256):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("0.0.0.0", 0))
        probe.listen(1)
        candidate_port = int(probe.getsockname()[1])
        probe.close()
        try:
            return candidate_port, bind_port(candidate_port)
        except OSError:
            continue

    raise RuntimeError("failed to reserve a dist-init port")


@ray.remote
class DistInitBootstrapActor:
    def __init__(self):
        self._reserved_socket: socket.socket | None = None

    def reserve_port(self, preferred_port: int | None = None) -> dict[str, Any]:
        self.release_port()
        port, sock = _reserve_tcp_port(preferred_port)
        self._reserved_socket = sock
        return {
            "host": ray.util.get_node_ip_address(),
            "port": port,
        }

    def release_port(self) -> bool:
        if self._reserved_socket is not None:
            self._reserved_socket.close()
            self._reserved_socket = None
        return True


def derive_dist_init_addr(
    args: argparse.Namespace,
    *,
    port_offset: int = 0,
) -> str | None:
    if args.nnodes == 1 and args.dist_init_addr is None:
        if args.dist_init_port is not None:
            return f"127.0.0.1:{args.dist_init_port + port_offset}"
        return f"127.0.0.1:{_pick_free_local_port()}"

    if args.dist_init_addr is None:
        raise ValueError("dist-init-addr is required when nnodes > 1")

    host, parsed_port = _parse_host_port(args.dist_init_addr)
    base_port = args.dist_init_port if args.dist_init_port is not None else parsed_port
    if base_port is None:
        raise ValueError(
            "dist-init-addr must include a port or dist-init-port must be set"
        )

    return f"{host}:{base_port + port_offset}"


def derive_dist_init_addr_from_pg(
    args: argparse.Namespace,
    pg,
) -> str | None:
    if args.dist_init_addr is not None or args.nnodes == 1:
        return derive_dist_init_addr(args)

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor = DistInitBootstrapActor.options(
        num_cpus=0,
        scheduling_strategy=scheduling_strategy,
    ).remote()
    try:
        reservation = ray.get(actor.reserve_port.remote(args.dist_init_port))
        host = reservation["host"]
        port = int(reservation["port"])
        ray.get(actor.release_port.remote())
    finally:
        ray.kill(actor, no_restart=True)

    return f"{host}:{port}"


def init_ray(address: str, namespace: str, nnodes: int) -> None:
    init_kwargs = dict(
        address=address,
        namespace=namespace,
        ignore_reinit_error=True,
        log_to_driver=True,
        logging_level=logging.ERROR,
    )
    try:
        ray.init(**init_kwargs)
    except Exception:
        if address != "auto" or nnodes != 1:
            raise
        ray.init(
            namespace=namespace,
            ignore_reinit_error=True,
            log_to_driver=True,
            logging_level=logging.ERROR,
        )


def derive_target_layout(args: argparse.Namespace) -> tuple[int, int]:
    for candidate_nnodes in range(1, args.nnodes + 1):
        if args.target_tp_size % candidate_nnodes != 0:
            continue
        target_gpus_per_node = args.target_tp_size // candidate_nnodes
        if target_gpus_per_node <= args.n_gpu_per_node:
            return candidate_nnodes, target_gpus_per_node

    raise ValueError(
        f"target-tp-size ({args.target_tp_size}) cannot be packed evenly across up to "
        f"{args.nnodes} nodes with {args.n_gpu_per_node} GPUs per node"
    )


def validate_resources(args: argparse.Namespace) -> tuple[int, int]:
    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.n_gpu_per_node <= 0:
        raise ValueError("n-gpu-per-node must be positive")
    if args.target_tp_size <= 0:
        raise ValueError("target-tp-size must be positive")
    if args.draft_tp_size <= 0:
        raise ValueError("draft-tp-size must be positive")
    if args.num_draft_replicas <= 0:
        raise ValueError("num-draft-replicas must be positive")

    target_nnodes, target_gpus_per_node = derive_target_layout(args)

    residual_gpus_per_node = args.n_gpu_per_node - target_gpus_per_node
    max_free_gpus_per_node = (
        args.n_gpu_per_node if target_nnodes < args.nnodes else residual_gpus_per_node
    )
    if args.draft_tp_size > max_free_gpus_per_node:
        raise ValueError(
            f"each draft actor needs {args.draft_tp_size} GPUs on one node, "
            f"but at most {max_free_gpus_per_node} GPUs are free on any one node"
        )

    total_residual_gpus = (
        args.n_gpu_per_node * args.nnodes - args.target_tp_size
    )
    required_draft_gpus = args.draft_tp_size * args.num_draft_replicas
    if required_draft_gpus > total_residual_gpus:
        raise ValueError(
            f"draft replicas require {required_draft_gpus} GPUs, but only "
            f"{total_residual_gpus} GPUs remain after reserving target GPUs"
        )

    ray_gpus = int(ray.cluster_resources().get("GPU", 0))
    required_spec_gpus = args.target_tp_size + required_draft_gpus
    if ray_gpus and required_spec_gpus > ray_gpus:
        raise ValueError(
            f"Ray cluster reports {ray_gpus} GPUs, but spec mode requires "
            f"{required_spec_gpus}"
        )

    alive_target_nodes = [
        node
        for node in ray.nodes()
        if node.get("Alive")
        and float(node.get("Resources", {}).get("GPU", 0)) >= target_gpus_per_node
    ]
    if len(alive_target_nodes) < target_nnodes:
        raise ValueError(
            f"Ray cluster has {len(alive_target_nodes)} alive GPU nodes with at "
            f"least {target_gpus_per_node} GPUs, but target needs {target_nnodes} nodes"
        )

    return target_nnodes, target_gpus_per_node


def _get_drafter_debug_env_vars() -> dict[str, str]:
    env_vars = {}
    for key in (
        "DECOUPLED_SPEC_DEBUG_CSV_DIR",
        "DECOUPLED_SPEC_DEBUG_SUMMARY_INTERVAL_SEC",
    ):
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


@ray.remote
class DraftEngineActor:
    def __init__(
        self,
        *,
        model_path: str,
        tp_size: int,
        speculative_num_steps: int,
    ):
        self.assigned_gpu_ids = _pin_actor_to_assigned_gpus(tp_size)
        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )
        self.engine = sgl.Engine(**engine_kwargs)

    def ready(self) -> bool:
        return {
            "assigned_gpu_ids": self.assigned_gpu_ids,
        }

    async def handle_draft_request(self, request: DraftRequest) -> DraftResult:
        return await self.engine.handle_draft_request(request)

    async def handle_draft_requests(
        self, requests: list[DraftRequest]
    ) -> list[DraftResult]:
        return await self.engine.handle_draft_requests(requests)

    async def terminate_draft_request(self, request_id: str):
        await self.engine.terminate_draft_request(request_id)

    async def release_draft_session(self, request_id: str):
        await self.engine.release_draft_session(request_id)

    def shutdown(self) -> bool:
        self.engine.shutdown()
        return True


@ray.remote
class TargetEngineActor:
    def __init__(
        self,
        *,
        mode: str,
        model_path: str,
        tp_size: int,
        nnodes: int,
        node_rank: int,
        dist_init_addr: str | None,
        batch_size: int,
        speculative_num_steps: int | None = None,
        draft_actor_names: list[str] | None = None,
        draft_actor_namespace: str | None = None,
        ray_init_kwargs: dict[str, Any] | None = None,
    ):
        self.mode = mode
        self.node_rank = node_rank
        self.assigned_gpu_ids = _pin_actor_to_assigned_gpus(
            max(tp_size // nnodes, 1)
        )
        if node_rank >= 1:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

        engine_kwargs: dict[str, Any] = dict(
            model_path=model_path,
            tp_size=tp_size,
            nnodes=nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            max_running_requests=batch_size,
        )
        if mode == "decoupled_spec":
            engine_kwargs.update(
                speculative_algorithm="DECOUPLED_VERIFY",
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_steps + 1,
                draft_actor_names=draft_actor_names,
                draft_actor_namespace=draft_actor_namespace,
                draft_backend_process_kwargs={
                    "ray_init_kwargs": dict(ray_init_kwargs or {})
                },
            )
        elif mode != "decode":
            raise ValueError(f"Unsupported mode: {mode}")

        self.engine = sgl.Engine(**engine_kwargs)

    def ready(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "node_rank": self.node_rank,
            "assigned_gpu_ids": self.assigned_gpu_ids,
        }

    def generate_and_measure(
        self,
        prompts: list[str],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        if self.node_rank != 0:
            raise RuntimeError("generate_and_measure must be called on node rank 0")

        start_time = time.perf_counter()
        outputs = self.engine.generate(prompt=prompts, sampling_params=sampling_params)
        elapsed_s = time.perf_counter() - start_time
        if not isinstance(outputs, list):
            outputs = [outputs]
        return {"elapsed_s": elapsed_s, "outputs": outputs}

    def shutdown(self) -> bool:
        self.engine.shutdown()
        return True


def launch_draft_actors(
    args: argparse.Namespace,
    actor_prefix: str,
) -> tuple[list[Any], list[str]]:
    actors = []
    actor_names = []
    debug_env_vars = _get_drafter_debug_env_vars()
    for replica_index in range(args.num_draft_replicas):
        actor_name = f"{actor_prefix}-draft-{replica_index}"
        actor_options: dict[str, Any] = dict(
            name=actor_name,
            num_gpus=args.draft_tp_size,
            num_cpus=1,
            max_concurrency=128,
        )
        if debug_env_vars:
            actor_options["runtime_env"] = {"env_vars": debug_env_vars}
        actor = DraftEngineActor.options(**actor_options).remote(
            model_path=args.draft_model_path,
            tp_size=args.draft_tp_size,
            speculative_num_steps=args.num_speculative_steps,
        )
        actors.append(actor)
        actor_names.append(actor_name)
    ray.get([actor.ready.remote() for actor in actors])
    return actors, actor_names


def create_target_placement_group(target_nnodes: int, target_gpus_per_node: int):
    bundles = [
        {"CPU": 1, "GPU": target_gpus_per_node}
        for _ in range(target_nnodes)
    ]
    strategy = "PACK" if target_nnodes == 1 else "STRICT_SPREAD"
    pg = placement_group(bundles, strategy=strategy)
    ray.get(pg.ready())
    return pg


def launch_target_actors(
    *,
    args: argparse.Namespace,
    mode: str,
    dist_init_addr: str | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg,
    draft_actor_names: list[str] | None = None,
) -> list[Any]:

    ray_init_kwargs = {
        "address": args.ray_address,
        "namespace": args.ray_namespace,
        "ignore_reinit_error": True,
        "log_to_driver": False,
        "logging_level": logging.ERROR,
    }
    debug_env_vars = _get_drafter_debug_env_vars()
    actors = []
    for node_rank in range(target_nnodes):
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=node_rank,
        )
        actor_options: dict[str, Any] = dict(
            num_gpus=target_gpus_per_node,
            num_cpus=1,
            scheduling_strategy=scheduling_strategy,
        )
        if debug_env_vars:
            actor_options["runtime_env"] = {"env_vars": debug_env_vars}
        actor = TargetEngineActor.options(**actor_options).remote(
            mode=mode,
            model_path=args.target_model_path,
            tp_size=args.target_tp_size,
            nnodes=target_nnodes,
            node_rank=node_rank,
            dist_init_addr=dist_init_addr,
            batch_size=args.batch_size,
            speculative_num_steps=args.num_speculative_steps,
            draft_actor_names=draft_actor_names,
            draft_actor_namespace=args.ray_namespace,
            ray_init_kwargs=ray_init_kwargs,
        )
        actors.append(actor)

    ray.get([actor.ready.remote() for actor in actors])
    return actors


def shutdown_actors(actors: list[Any]) -> None:
    if not actors:
        return
    try:
        ray.get([actor.shutdown.remote() for actor in actors], timeout=60)
    except Exception as exc:
        logger.warning("actor shutdown failed: %s", exc)
    finally:
        for actor in actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass


def run_mode(
    *,
    args: argparse.Namespace,
    mode: str,
    prompts: list[str],
    sampling_params: dict[str, Any],
    prompt_samples: list[PromptSample],
    dist_init_addr: str | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg=None,
    draft_actor_names: list[str] | None = None,
) -> ModeMetrics:
    target_actors: list[Any] = []
    owns_pg = pg is None
    try:
        if pg is None:
            pg = create_target_placement_group(target_nnodes, target_gpus_per_node)
        target_actors = launch_target_actors(
            args=args,
            mode=mode,
            dist_init_addr=dist_init_addr,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pg=pg,
            draft_actor_names=draft_actor_names,
        )
        result = ray.get(
            target_actors[0].generate_and_measure.remote(prompts, sampling_params)
        )
    finally:
        shutdown_actors(target_actors)
        if owns_pg and pg is not None:
            remove_placement_group(pg)

    outputs = result["outputs"]
    if len(outputs) != len(prompt_samples):
        raise RuntimeError(
            f"{mode} returned {len(outputs)} outputs for {len(prompt_samples)} prompts"
        )

    total_generated_tokens = 0
    accept_lengths = []
    per_request = []
    for index, (sample, output) in enumerate(zip(prompt_samples, outputs, strict=True)):
        output_ids = output.get("output_ids", [])
        generated_tokens = len(output_ids) if isinstance(output_ids, list) else 0
        total_generated_tokens += generated_tokens

        meta_info = output.get("meta_info", {}) or {}
        accept_length = meta_info.get("spec_accept_length")
        if accept_length is not None:
            accept_lengths.append(float(accept_length))

        per_request.append(
            {
                "batch_index": index,
                "row_index": sample.row_index,
                "prompt_tokens": sample.prompt_tokens,
                "generated_tokens": generated_tokens,
                "spec_accept_length": accept_length,
            }
        )

    elapsed_s = float(result["elapsed_s"])
    throughput = total_generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    avg_accept_length = (
        sum(accept_lengths) / len(accept_lengths) if accept_lengths else None
    )
    return ModeMetrics(
        mode=mode,
        generation_time_s=elapsed_s,
        total_generated_tokens=total_generated_tokens,
        output_throughput_tok_per_s=throughput,
        per_request=per_request,
        avg_spec_accept_length=avg_accept_length,
    )


def build_result(
    *,
    args: argparse.Namespace,
    target_nnodes: int,
    target_gpus_per_node: int,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[PromptSample],
    spec_metrics: ModeMetrics,
    decode_metrics: ModeMetrics,
) -> dict[str, Any]:
    speedup = (
        decode_metrics.generation_time_s / spec_metrics.generation_time_s
        if spec_metrics.generation_time_s > 0
        else None
    )
    return {
        "config": {
            "dataset_path": args.dataset_path,
            "prompt_column": prompt_column,
            "offset": args.offset,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "max_new_tokens": args.context_length,
            "max_prompt_length": args.max_prompt_length,
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "tokenizer_path": args.tokenizer_path or args.target_model_path,
            "target_tp_size": args.target_tp_size,
            "draft_tp_size": args.draft_tp_size,
            "num_speculative_steps": args.num_speculative_steps,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "nnodes": args.nnodes,
            "n_gpu_per_node": args.n_gpu_per_node,
            "target_nnodes": target_nnodes,
            "target_gpus_per_node": target_gpus_per_node,
            "num_draft_replicas": args.num_draft_replicas,
        },
        "dataset": {
            "total_rows": total_rows,
            "loaded_rows": [sample.row_index for sample in prompt_samples],
            "total_prompt_tokens": sum(sample.prompt_tokens for sample in prompt_samples),
        },
        "decoupled_spec": asdict(spec_metrics),
        "decode": asdict(decode_metrics),
        "e2e_speedup": speedup,
    }


def print_summary(result: dict[str, Any]) -> None:
    spec = result["decoupled_spec"]
    decode = result["decode"]
    speedup = result["e2e_speedup"]
    print("=== decoupled_spec_vs_decode_batch ===")
    print(f"dataset_path: {result['config']['dataset_path']}")
    print(f"prompt_column: {result['config']['prompt_column']}")
    print(f"batch_size: {result['config']['batch_size']}")
    print(f"max_new_tokens: {result['config']['max_new_tokens']}")
    print(f"total_prompt_tokens: {result['dataset']['total_prompt_tokens']}")
    print(
        "decoupled_spec: "
        f"generation_time_s={spec['generation_time_s']:.3f}, "
        f"generated_tokens={spec['total_generated_tokens']}, "
        f"output_throughput={spec['output_throughput_tok_per_s']:.3f} tok/s, "
        f"avg_spec_accept_length={spec['avg_spec_accept_length']}"
    )
    print(
        "decode: "
        f"generation_time_s={decode['generation_time_s']:.3f}, "
        f"generated_tokens={decode['total_generated_tokens']}, "
        f"output_throughput={decode['output_throughput_tok_per_s']:.3f} tok/s"
    )
    print(f"e2e_speedup: {speedup:.4f}" if speedup is not None else "e2e_speedup: None")
    print("per_request:")
    for item in spec["per_request"]:
        print(
            "  "
            f"batch_index={item['batch_index']}, "
            f"row_index={item['row_index']}, "
            f"prompt_tokens={item['prompt_tokens']}, "
            f"generated_tokens={item['generated_tokens']}, "
            f"spec_accept_length={item['spec_accept_length']}"
        )


def main() -> None:
    args = parse_args()
    actor_prefix = f"sglang-decoupled-spec-bench-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    prompt_column, prompt_samples, total_rows = load_prompt_samples(args)
    prompts = [sample.prompt for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    draft_actors: list[Any] = []
    spec_pg = None
    try:
        init_ray(args.ray_address, args.ray_namespace, args.nnodes)
        target_nnodes, target_gpus_per_node = validate_resources(args)

        spec_pg = create_target_placement_group(target_nnodes, target_gpus_per_node)
        spec_dist_init_addr = derive_dist_init_addr_from_pg(args, spec_pg)
        draft_actors, draft_actor_names = launch_draft_actors(args, actor_prefix)
        spec_metrics = run_mode(
            args=args,
            mode="decoupled_spec",
            prompts=prompts,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addr=spec_dist_init_addr,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pg=spec_pg,
            draft_actor_names=draft_actor_names,
        )
        shutdown_actors(draft_actors)
        draft_actors = []

        decode_dist_init_addr = (
            derive_dist_init_addr(args, port_offset=1)
            if args.dist_init_addr is not None or args.dist_init_port is not None
            else derive_dist_init_addr_from_pg(args, spec_pg)
        )
        decode_metrics = run_mode(
            args=args,
            mode="decode",
            prompts=prompts,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addr=decode_dist_init_addr,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pg=spec_pg,
        )

        result = build_result(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            decode_metrics=decode_metrics,
        )
        print_summary(result)
        if args.output_json:
            output_path = Path(args.output_json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"output_json: {output_path}")
    finally:
        shutdown_actors(draft_actors)
        if spec_pg is not None:
            remove_placement_group(spec_pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
