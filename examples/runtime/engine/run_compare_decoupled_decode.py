#!/usr/bin/env python3
# Default uses a highly repetitive fixed-format request to observe speedup under high acceptance rate.
"""
Run decoupled speculative decoding and normal decoding sequentially,
then print a summary of their metrics.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import ray
import sglang as sgl

from run_decoupled_spec import allocate_demo_gpus
from run_decoupled_spec import _endpoint_lists
from run_decoupled_spec import init_demo_ray
from run_decoupled_spec import launch_drafter_actor
from run_decoupled_spec import launch_verifier
from run_decoupled_spec import RAY_NAMESPACE
from run_decoupled_spec_batch import (
    _get_real_verify_acceptance_stats,
    _maybe_append_chatml_generation_prompt,
)
from sglang.srt.speculative.decoupled_spec_io import DraftMeshIpcConfig

DEFAULT_PROMPT = """Repeat the exact string `benchmark_token` for exactly 32 lines.

Output requirements:
- Each line must be exactly `benchmark_token`
- Output exactly 32 lines
- Do not add numbering
- Do not add code fences
- Do not add explanation
- Do not add any extra characters before or after the 32 lines
"""


@dataclass
class ModeResult:
    mode: str
    generation_time_s: float
    output_text: str
    avg_accept_length: float | None = None
    avg_accept_rate: float | None = None
    generated_tokens: int = 0
    token_throughput_tok_per_s: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run decoupled speculation and normal decoding "
            "in one script and print their performance summary."
        )
    )
    parser.add_argument(
        "--target-model-path",
        default="Qwen/Qwen3-32B",
        help="Target model path used by all modes.",
    )
    parser.add_argument(
        "--draft-model-path",
        default="Qwen/Qwen3-0.6B",
        help="Draft model path used by speculative modes.",
    )
    parser.add_argument(
        "--draft-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the decoupled drafter engine.",
    )
    parser.add_argument(
        "--target-tp-size",
        "--verify-tp-size",
        dest="target_tp_size",
        type=int,
        default=4,
        help="Tensor parallel size for the target/verifier/decode engine.",
    )
    parser.add_argument(
        "--num-speculative-steps",
        type=int,
        default=3,
        help="Number of speculative steps for speculative modes.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional context length override passed to the decode engine.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Verifier request batch size; also caps the drafter running requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used by both modes.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used by both modes.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable thinking-style generation for ChatML prompts on models such "
            "as Qwen3/Qwen3.5. Disabled by default."
        ),
    )
    return parser.parse_args()


def _generate_and_measure(engine, args: argparse.Namespace) -> ModeResult:
    start_time = time.perf_counter()
    output = engine.generate(
        prompt=_maybe_append_chatml_generation_prompt(
            args.prompt, enable_thinking=args.enable_thinking
        ),
        sampling_params={
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    elapsed_s = time.perf_counter() - start_time
    meta_info = output.get("meta_info", {})
    output_ids = output.get("output_ids", [])
    generated_tokens = len(output_ids) if isinstance(output_ids, list) else 0
    token_throughput_tok_per_s = (
        generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    )
    accept_length, accept_rate, *_ = _get_real_verify_acceptance_stats(meta_info)
    return ModeResult(
        mode="",
        generation_time_s=elapsed_s,
        output_text=output.get("text", ""),
        avg_accept_length=accept_length,
        avg_accept_rate=accept_rate,
        generated_tokens=generated_tokens,
        token_throughput_tok_per_s=token_throughput_tok_per_s,
    )


def build_decode_engine(args: argparse.Namespace) -> sgl.Engine:
    engine_kwargs = dict(
        model_path=args.target_model_path,
        tp_size=args.target_tp_size,
    )
    if args.max_model_len is not None:
        engine_kwargs["context_length"] = args.max_model_len
    return sgl.Engine(**engine_kwargs)


def run_decoupled(args: argparse.Namespace) -> ModeResult:
    drafter_actor = None
    verifier = None
    ray_runtime = None
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        draft_gpu_ids, target_gpu_ids = allocate_demo_gpus(args)
        args.draft_gpu_ids = draft_gpu_ids
        control_endpoints, result_endpoints = _endpoint_lists(
            DraftMeshIpcConfig.init_new(1)
        )
        ray_runtime = init_demo_ray(RAY_NAMESPACE)
        drafter_actor = launch_drafter_actor(
            args,
            control_endpoints,
            result_endpoints,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(target_gpu_ids)
        verifier = launch_verifier(
            args.target_model_path,
            args.target_tp_size,
            args.num_speculative_steps,
            args.num_speculative_steps + 1,
            control_endpoints,
            result_endpoints,
        )
        result = _generate_and_measure(verifier, args)
        result.mode = "decoupled_spec"
        return result
    finally:
        if verifier is not None:
            verifier.shutdown()
        if drafter_actor is not None:
            try:
                ray.get(drafter_actor.shutdown.remote())
            finally:
                ray.kill(drafter_actor, no_restart=True)
        if ray.is_initialized():
            ray.shutdown()
        if ray_runtime is not None:
            ray_runtime.stop()
        if original_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices


def run_decode(args: argparse.Namespace) -> ModeResult:
    engine = None
    try:
        engine = build_decode_engine(args)
        result = _generate_and_measure(engine, args)
        result.mode = "decode"
        return result
    finally:
        if engine is not None:
            engine.shutdown()


def print_summary(results: list[ModeResult]) -> None:
    final_output = results[0].output_text if results else ""
    print(f"output_text: {final_output}")
    print("=== performance_summary ===")
    for result in results:
        print(
            f"{result.mode}: "
            f"generation_time_s={result.generation_time_s:.3f}, "
            f"generated_tokens={result.generated_tokens}, "
            f"token_throughput={result.token_throughput_tok_per_s:.3f} tok/s, "
            f"avg_accept_length={result.avg_accept_length}, "
            f"avg_accept_rate={result.avg_accept_rate}"
        )


def main() -> None:
    args = parse_args()
    results = [
        run_decoupled(args),
        run_decode(args),
    ]
    print_summary(results)


if __name__ == "__main__":
    main()
