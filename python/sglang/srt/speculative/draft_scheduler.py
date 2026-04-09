from __future__ import annotations

from sglang.srt.server_args import PortArgs, ServerArgs


def run_drafter_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank,
    pipe_writer,
) -> None:
    _ = (
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        pipe_writer,
    )
    raise NotImplementedError(
        "decoupled_draft scaffold is present, but DraftScheduler is not implemented in phase 1"
    )
