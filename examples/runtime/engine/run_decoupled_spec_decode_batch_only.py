#!/usr/bin/env python3
"""
Run decoupled speculative decoding on one prompt batch without a normal decode run.

This script intentionally reuses the dataset loading, Ray placement, actor launch,
and metric collection code from `run_compare_decoupled_spec_decode_batch.py`.  The
only behavioral difference is that it stops after the decoupled-spec run.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ray

import run_compare_decoupled_spec_decode_batch as compare_batch


def build_spec_only_result(
    *,
    args,
    target_nnodes: int,
    target_gpus_per_node: int,
    prompt_column: str,
    total_rows: int,
    prompt_samples: list[compare_batch.PromptSample],
    spec_metrics: compare_batch.ModeMetrics,
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
            "enable_decoupled_spec_trace": args.enable_decoupled_spec_trace,
            "decoupled_spec_trace_dir": args.decoupled_spec_trace_dir,
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
    }


def print_spec_only_summary(result: dict[str, Any]) -> None:
    spec = result["decoupled_spec"]
    print("=== decoupled_spec_batch ===")
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
    args = compare_batch.parse_args()
    actor_prefix = f"sglang-decoupled-spec-only-{os.getpid()}-{uuid.uuid4().hex[:8]}"

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
        spec_metrics = compare_batch.run_mode(
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

        result = build_spec_only_result(
            args=args,
            target_nnodes=target_nnodes,
            target_gpus_per_node=target_gpus_per_node,
            prompt_column=prompt_column,
            total_rows=total_rows,
            prompt_samples=prompt_samples,
            spec_metrics=spec_metrics,
        )
        print_spec_only_summary(result)
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
