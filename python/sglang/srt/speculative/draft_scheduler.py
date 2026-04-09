from __future__ import annotations

from sglang.srt.managers.scheduler import run_scheduler_process
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
    run_scheduler_process(
        server_args=server_args,
        port_args=port_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        attn_cp_rank=attn_cp_rank,
        moe_dp_rank=moe_dp_rank,
        moe_ep_rank=moe_ep_rank,
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        pipe_writer=pipe_writer,
    )
