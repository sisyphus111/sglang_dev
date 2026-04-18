#!/usr/bin/env python3
"""
Run native SGLang speculative decoding on one prompt batch.

Unlike the decoupled examples, this script launches one normal `sgl.Engine` with
SGLang's built-in speculative decoding arguments, such as EAGLE. There are no
separate drafter/verifier engines, Ray actors, or decoupled IPC endpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sglang as sgl

import run_compare_decoupled_spec_decode_batch as compare_batch


DEFAULT_SPEC_ALGORITHM = "EAGLE"
DEFAULT_TARGET_TP_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run native non-decoupled SGLang speculative decoding on a parquet "
            "prompt batch."
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
        help="Prompt column in the parquet file. If omitted, common prompt names are searched.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "codeforces_raw", "dapo_math_17k"],
        default="auto",
        help="How to interpret the parquet rows.",
    )
    parser.add_argument(
        "--code-language",
        choices=["python", "py", "cpp", "c++"],
        default="python",
        help="Target language used when --dataset-format=codeforces_raw.",
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
        help="Enable thinking-style generation for supported chat templates.",
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
    parser.add_argument("--target-model-path", required=True, help="Target model path.")
    parser.add_argument(
        "--draft-model-path",
        required=True,
        help="Draft model path used by native speculative decoding.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path used for prompt length filtering. Defaults to target model.",
    )
    parser.add_argument(
        "--target-tp-size",
        type=int,
        default=DEFAULT_TARGET_TP_SIZE,
        help=(
            "Tensor parallel size for the native target engine. Defaults to 8 "
            "so the normal path packs the whole native spec engine onto one "
            "8-GPU machine."
        ),
    )
    parser.add_argument(
        "--speculative-algorithm",
        default=DEFAULT_SPEC_ALGORITHM,
        choices=["EAGLE", "EAGLE3", "STANDALONE", "NEXTN"],
        help="Native SGLang speculative algorithm. NEXTN is normalized to EAGLE by SGLang.",
    )
    parser.add_argument("--num-speculative-steps", type=int, default=3)
    parser.add_argument(
        "--speculative-eagle-topk",
        type=int,
        default=1,
        help="EAGLE top-k. topk=1 uses num_speculative_steps + 1 draft tokens.",
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=None,
        help=(
            "Override speculative_num_draft_tokens. Defaults to "
            "num_speculative_steps + 1 when topk=1."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic inference.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set sampling_params.ignore_eos=True.",
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument(
        "--dist-init-addr",
        default=None,
        help="SGLang distributed init address for multi-node native spec.",
    )
    parser.add_argument("--cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="Override max_running_requests. Defaults to batch_size.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the metrics/result JSON.",
    )
    return parser.parse_args()


def _get_visible_gpu_ids() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]

    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except Exception:
        return []

    return [str(index) for index in range(gpu_count)]


def validate_native_layout(args: argparse.Namespace) -> None:
    if args.target_tp_size <= 0:
        raise ValueError("target-tp-size must be positive")
    if args.nnodes <= 0:
        raise ValueError("nnodes must be positive")
    if args.node_rank < 0 or args.node_rank >= args.nnodes:
        raise ValueError(
            f"node-rank must be in [0, {args.nnodes}), got {args.node_rank}"
        )

    if args.nnodes > 1:
        if args.dist_init_addr is None:
            raise ValueError("dist-init-addr is required when nnodes > 1")
        if args.target_tp_size % args.nnodes != 0:
            raise ValueError(
                f"target-tp-size ({args.target_tp_size}) must be divisible by "
                f"nnodes ({args.nnodes})"
            )

    local_tp_size = args.target_tp_size // args.nnodes
    visible_gpu_ids = _get_visible_gpu_ids()
    if len(visible_gpu_ids) < local_tp_size:
        raise RuntimeError(
            "Insufficient visible CUDA GPUs for native speculative decoding: "
            f"node_rank={args.node_rank}, nnodes={args.nnodes}, "
            f"target_tp_size={args.target_tp_size}, local_tp_size={local_tp_size}, "
            f"visible_gpus={visible_gpu_ids or 'none'}. "
            "For the default same-machine deployment, expose 8 GPUs with "
            "CUDA_VISIBLE_DEVICES or pass a smaller --target-tp-size."
        )

    if args.nnodes == 1 and args.node_rank == 0:
        print(
            "native_layout: "
            f"same_node=True, target_tp_size={args.target_tp_size}, "
            f"visible_gpus={visible_gpu_ids[: args.target_tp_size]}"
        )


def _native_spec_num_draft_tokens(args: argparse.Namespace) -> int | None:
    if args.speculative_num_draft_tokens is not None:
        return args.speculative_num_draft_tokens
    if args.speculative_eagle_topk == 1:
        return args.num_speculative_steps + 1
    return None


def build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    engine_kwargs: dict[str, Any] = {
        "model_path": args.target_model_path,
        "tp_size": args.target_tp_size,
        "nnodes": args.nnodes,
        "node_rank": args.node_rank,
        "dist_init_addr": args.dist_init_addr,
        "max_running_requests": args.max_running_requests or args.batch_size,
        "enable_deterministic_inference": args.deterministic,
        "speculative_algorithm": args.speculative_algorithm,
        "speculative_draft_model_path": args.draft_model_path,
        "speculative_num_steps": args.num_speculative_steps,
        "speculative_eagle_topk": args.speculative_eagle_topk,
        "speculative_num_draft_tokens": _native_spec_num_draft_tokens(args),
    }
    if args.cuda_graph_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cuda_graph_max_bs
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    return engine_kwargs


def run_native_spec(
    *,
    args: argparse.Namespace,
    prompts: list[str],
    sampling_params: dict[str, Any],
    prompt_samples: list[compare_batch.PromptSample],
) -> compare_batch.ModeMetrics:
    engine = None
    try:
        engine = sgl.Engine(**build_engine_kwargs(args))
        start_time = time.perf_counter()
        outputs = engine.generate(prompt=prompts, sampling_params=sampling_params)
        elapsed_s = time.perf_counter() - start_time
    finally:
        if engine is not None:
            engine.shutdown()

    if not isinstance(outputs, list):
        outputs = [outputs]
    return compare_batch.collect_mode_metrics(
        mode="native_spec",
        elapsed_s=elapsed_s,
        outputs=outputs,
        prompt_samples=prompt_samples,
    )


def build_result(
    *,
    args: argparse.Namespace,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[compare_batch.PromptSample],
    metrics: compare_batch.ModeMetrics,
) -> dict[str, Any]:
    return {
        "config": {
            "dataset_path": args.dataset_path,
            "dataset_format": args.dataset_format,
            "prompt_column": prompt_column,
            "code_language": args.code_language,
            "offset": args.offset,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "max_new_tokens": args.context_length,
            "max_prompt_length": args.max_prompt_length,
            "target_model_path": args.target_model_path,
            "draft_model_path": args.draft_model_path,
            "tokenizer_path": args.tokenizer_path or args.target_model_path,
            "target_tp_size": args.target_tp_size,
            "speculative_algorithm": args.speculative_algorithm,
            "num_speculative_steps": args.num_speculative_steps,
            "speculative_eagle_topk": args.speculative_eagle_topk,
            "speculative_num_draft_tokens": _native_spec_num_draft_tokens(args),
            "temperature": args.temperature,
            "deterministic": args.deterministic,
            "ignore_eos": args.ignore_eos,
            "nnodes": args.nnodes,
            "node_rank": args.node_rank,
            "dist_init_addr": args.dist_init_addr,
            "local_tp_size": args.target_tp_size // args.nnodes,
            "same_node": args.nnodes == 1,
        },
        "dataset": {
            "total_rows": total_rows,
            "loaded_rows": [sample.row_index for sample in prompt_samples],
            "total_prompt_tokens": sum(sample.prompt_tokens for sample in prompt_samples),
            "prompt_samples": [
                {
                    "row_index": sample.row_index,
                    "prompt_tokens": sample.prompt_tokens,
                    "prompt_head": sample.prompt[:1024],
                    "prompt_tail": sample.prompt[-1024:],
                }
                for sample in prompt_samples
            ],
        },
        "native_spec": asdict(metrics),
    }


def print_summary(result: dict[str, Any]) -> None:
    spec = result["native_spec"]
    print("=== native_spec_batch ===")
    print(f"dataset_path: {result['config']['dataset_path']}")
    print(f"dataset_format: {result['config']['dataset_format']}")
    print(f"prompt_column: {result['config']['prompt_column']}")
    print(f"batch_size: {result['config']['batch_size']}")
    print(f"max_new_tokens: {result['config']['max_new_tokens']}")
    print(f"total_prompt_tokens: {result['dataset']['total_prompt_tokens']}")
    print(
        "native_spec: "
        f"generation_time_s={spec['generation_time_s']:.3f}, "
        f"generated_tokens={spec['total_generated_tokens']}, "
        f"output_throughput={spec['output_throughput_tok_per_s']:.3f} tok/s, "
        f"avg_spec_accept_length={spec['avg_spec_accept_length']}, "
        f"avg_spec_accept_rate={spec['avg_spec_accept_rate']}"
    )
    print("per_request:")
    for item in spec["per_request"]:
        print(
            "  "
            f"batch_index={item['batch_index']}, "
            f"row_index={item['row_index']}, "
            f"prompt_tokens={item['prompt_tokens']}, "
            f"generated_tokens={item['generated_tokens']}, "
            f"spec_accept_length={item['spec_accept_length']}, "
            f"spec_accept_rate={item['spec_accept_rate']}, "
            f"spec_verify_ct={item['spec_verify_ct']}"
        )


def main() -> None:
    args = parse_args()
    validate_native_layout(args)
    prompt_column, prompt_samples, total_rows = compare_batch.load_prompt_samples(args)
    prompts = [sample.prompt for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    metrics = run_native_spec(
        args=args,
        prompts=prompts,
        sampling_params=sampling_params,
        prompt_samples=prompt_samples,
    )
    result = build_result(
        args=args,
        prompt_column=prompt_column,
        total_rows=total_rows,
        prompt_samples=prompt_samples,
        metrics=metrics,
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


if __name__ == "__main__":
    main()
