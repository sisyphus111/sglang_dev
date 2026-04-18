#!/usr/bin/env python3
"""
Compare decoupled speculative decoding against normal decoding on one prompt batch.

This script keeps the same CLI and execution flow as
`run_compare_decoupled_spec_decode_batch.py`, but it also preserves and prints
the full response text for both decoupled-spec and normal decode runs.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ray

import run_compare_decoupled_spec_decode_batch as compare_batch


@dataclass
class ModeMetricsWithResponses:
    mode: str
    generation_time_s: float
    total_generated_tokens: int
    output_throughput_tok_per_s: float
    per_request: list[dict[str, Any]]
    avg_spec_accept_length: float | None = None
    avg_spec_accept_rate: float | None = None


def _normalize_output_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def run_mode_with_responses(
    *,
    args,
    mode: str,
    prompts: list[str],
    sampling_params: dict[str, Any],
    prompt_samples: list[compare_batch.PromptSample],
    dist_init_addr: str | None,
    target_nnodes: int,
    target_gpus_per_node: int,
    pg=None,
    control_endpoints: list[str] | None = None,
    result_endpoints: list[str] | None = None,
) -> ModeMetricsWithResponses:
    target_actors: list[Any] = []
    owns_pg = pg is None
    try:
        if pg is None:
            pg = compare_batch.create_target_placement_group(
                target_nnodes, target_gpus_per_node
            )
        target_actors = compare_batch.launch_target_actors(
            args=args,
            mode=mode,
            dist_init_addr=dist_init_addr,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pg=pg,
            control_endpoints=control_endpoints,
            result_endpoints=result_endpoints,
        )
        result = ray.get(
            target_actors[0].generate_and_measure.remote(prompts, sampling_params)
        )
    finally:
        compare_batch.shutdown_actors(target_actors)
        if owns_pg and pg is not None:
            compare_batch.remove_placement_group(pg)

    outputs = result["outputs"]
    if len(outputs) != len(prompt_samples):
        raise RuntimeError(
            f"{mode} returned {len(outputs)} outputs for {len(prompt_samples)} prompts"
        )

    total_generated_tokens = 0
    total_accepted_tokens = 0
    total_draft_tokens = 0
    total_verify_ct = 0
    per_request = []
    for index, (sample, output) in enumerate(zip(prompt_samples, outputs, strict=True)):
        output_ids = output.get("output_ids", [])
        generated_tokens = len(output_ids) if isinstance(output_ids, list) else 0
        total_generated_tokens += generated_tokens

        meta_info = output.get("meta_info", {}) or {}
        (
            accept_length,
            accept_rate,
            accepted_tokens,
            draft_tokens,
            verify_ct,
        ) = compare_batch._get_real_verify_acceptance_stats(meta_info)
        total_accepted_tokens += accepted_tokens
        total_draft_tokens += draft_tokens
        total_verify_ct += verify_ct

        output_text = _normalize_output_text(output.get("text", ""))
        finish_reason = meta_info.get("finish_reason")

        per_request.append(
            {
                "batch_index": index,
                "row_index": sample.row_index,
                "prompt_tokens": sample.prompt_tokens,
                "generated_tokens": generated_tokens,
                "spec_accept_length": accept_length,
                "spec_accept_rate": accept_rate,
                "spec_accept_token_num": accepted_tokens or None,
                "spec_draft_token_num": draft_tokens or None,
                "spec_verify_ct": verify_ct or None,
                "finish_reason": finish_reason,
                "output_text": output_text,
                "output_text_preview": output_text[:512],
                "output_ids_head": output_ids[:32]
                if isinstance(output_ids, list)
                else None,
                "output_ids_tail": output_ids[-32:]
                if isinstance(output_ids, list)
                else None,
            }
        )

    elapsed_s = float(result["elapsed_s"])
    throughput = total_generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    avg_accept_length = (
        total_accepted_tokens / total_verify_ct
        if total_verify_ct > 0
        else None
    )
    avg_accept_rate = (
        total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else None
    )
    return ModeMetricsWithResponses(
        mode=mode,
        generation_time_s=elapsed_s,
        total_generated_tokens=total_generated_tokens,
        output_throughput_tok_per_s=throughput,
        per_request=per_request,
        avg_spec_accept_length=avg_accept_length,
        avg_spec_accept_rate=avg_accept_rate,
    )


def build_result_with_responses(
    *,
    args,
    target_nnodes: int,
    target_gpus_per_node: int,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[compare_batch.PromptSample],
    spec_metrics: ModeMetricsWithResponses,
    decode_metrics: ModeMetricsWithResponses,
) -> dict[str, Any]:
    speedup = (
        decode_metrics.generation_time_s / spec_metrics.generation_time_s
        if spec_metrics.generation_time_s > 0
        else None
    )
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
            "draft_tp_size": args.draft_tp_size,
            "num_speculative_steps": args.num_speculative_steps,
            "temperature": args.temperature,
            "deterministic": args.deterministic,
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
        "decoupled_spec": asdict(spec_metrics),
        "decode": asdict(decode_metrics),
        "e2e_speedup": speedup,
    }


def _print_response_block(label: str, text: str, *, indent: str = "    ") -> None:
    print(f"{indent}{label}:")
    if not text:
        print(f"{indent}  <empty>")
        return
    for line in text.splitlines():
        print(f"{indent}  {line}")


def print_summary_with_responses(result: dict[str, Any]) -> None:
    spec = result["decoupled_spec"]
    decode = result["decode"]
    speedup = result["e2e_speedup"]
    print("=== decoupled_spec_vs_decode_batch ===")
    print(f"dataset_path: {result['config']['dataset_path']}")
    print(f"dataset_format: {result['config']['dataset_format']}")
    print(f"prompt_column: {result['config']['prompt_column']}")
    print(f"batch_size: {result['config']['batch_size']}")
    print(f"max_new_tokens: {result['config']['max_new_tokens']}")
    print(f"total_prompt_tokens: {result['dataset']['total_prompt_tokens']}")
    print(
        "decoupled_spec: "
        f"generation_time_s={spec['generation_time_s']:.3f}, "
        f"generated_tokens={spec['total_generated_tokens']}, "
        f"output_throughput={spec['output_throughput_tok_per_s']:.3f} tok/s, "
        f"avg_spec_accept_length={spec['avg_spec_accept_length']}, "
        f"avg_spec_accept_rate={spec['avg_spec_accept_rate']}"
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
            f"spec_accept_length={item['spec_accept_length']}, "
            f"spec_accept_rate={item['spec_accept_rate']}, "
            f"spec_verify_ct={item['spec_verify_ct']}"
        )

    print("responses:")
    for spec_item, decode_item in zip(
        spec["per_request"], decode["per_request"], strict=True
    ):
        if (
            spec_item["batch_index"] != decode_item["batch_index"]
            or spec_item["row_index"] != decode_item["row_index"]
        ):
            raise RuntimeError(
                "Mismatched per-request ordering between decoupled_spec and decode"
            )
        print(
            "  "
            f"batch_index={spec_item['batch_index']}, "
            f"row_index={spec_item['row_index']}"
        )
        _print_response_block(
            "decoupled_spec_response",
            spec_item.get("output_text", ""),
        )
        _print_response_block(
            "decode_response",
            decode_item.get("output_text", ""),
        )


def main() -> None:
    args = compare_batch.parse_args()
    actor_prefix = f"sglang-decoupled-spec-bench-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    prompt_column, prompt_samples, total_rows = compare_batch.load_prompt_samples(args)
    prompts = [sample.prompt for sample in prompt_samples]
    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.context_length,
        "ignore_eos": args.ignore_eos,
    }

    draft_actors: list[Any] = []
    spec_pg = None
    try:
        compare_batch.init_ray(args.ray_address, args.ray_namespace, args.nnodes)
        target_nnodes, target_gpus_per_node = compare_batch.validate_resources(args)

        spec_pg = compare_batch.create_target_placement_group(
            target_nnodes, target_gpus_per_node
        )
        spec_dist_init_addr = compare_batch.derive_dist_init_addr_from_pg(args, spec_pg)
        _, spec_dist_init_port = compare_batch._parse_host_port(spec_dist_init_addr)
        result_endpoint = compare_batch.derive_result_endpoint_from_pg(
            args,
            spec_pg,
            avoid_port=spec_dist_init_port,
        )
        draft_actors, control_endpoints = compare_batch.launch_draft_actors(
            args,
            actor_prefix,
            result_endpoint,
        )
        spec_metrics = run_mode_with_responses(
            args=args,
            mode="decoupled_spec",
            prompts=prompts,
            sampling_params=sampling_params,
            prompt_samples=prompt_samples,
            dist_init_addr=spec_dist_init_addr,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            pg=spec_pg,
            control_endpoints=control_endpoints,
            result_endpoints=[result_endpoint],
        )
        compare_batch.shutdown_actors(draft_actors)
        draft_actors = []

        decode_dist_init_addr = (
            compare_batch.derive_dist_init_addr(args, port_offset=1)
            if args.dist_init_addr is not None or args.dist_init_port is not None
            else compare_batch.derive_dist_init_addr_from_pg(args, spec_pg)
        )
        decode_metrics = run_mode_with_responses(
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

        result = build_result_with_responses(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
            decode_metrics=decode_metrics,
        )
        print_summary_with_responses(result)
        if args.output_json:
            output_path = Path(args.output_json).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"output_json: {output_path}")
    finally:
        compare_batch.shutdown_actors(draft_actors)
        if spec_pg is not None:
            compare_batch.remove_placement_group(spec_pg)
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
