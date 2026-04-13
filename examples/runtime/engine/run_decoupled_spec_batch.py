"""
Batch decoupled speculative decoding demo backed by a parquet file.

This script:
1. Launches a `decoupled_draft` SGLang engine inside a Ray actor.
2. Launches a `decoupled_verify` SGLang engine that talks to the drafter actor.
3. Reads prompts from a parquet file starting at `offset` and taking `batch_size`.
4. Sends the prompt batch to the verifier in a single `generate` call.
5. Prints per-request outputs and aggregate latency stats.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import ray
import sglang as sgl
import torch
from sglang.srt.speculative.decoupled_spec_io import DraftRequest, DraftResult

ACTOR_NAME = "sglang-decoupled-drafter"
RAY_NAMESPACE = "sglang-decoupled-spec"
DEFAULT_PROMPT_COLUMN_CANDIDATES = [
    "prompt",
    "messages",
    "chat",
    "conversations",
    "text",
    "question",
    "instruction",
    "input",
    "query",
]


@dataclass
class DemoRayRuntime:
    address: str
    namespace: str
    head_process: subprocess.Popen | None = None

    def build_init_kwargs(self) -> dict[str, object]:
        return {
            "address": self.address,
            "namespace": self.namespace,
            "ignore_reinit_error": True,
            "log_to_driver": True,
            "logging_level": "ERROR",
        }

    def stop(self) -> None:
        if self.head_process is None:
            return
        if self.head_process.poll() is not None:
            return
        self.head_process.terminate()
        try:
            self.head_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.head_process.kill()
            self.head_process.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch decoupled speculative decoding from prompts stored in parquet."
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to the parquet file that stores the prompts.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start row offset inside the parquet file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts to read and generate in one batch.",
    )
    parser.add_argument(
        "--target-model-path",
        default="Qwen/Qwen3-32B",
        help="Target model path used by the drafter and verifier engines.",
    )
    parser.add_argument(
        "--draft-model-path",
        default="Qwen/Qwen3-0.6B",
        help="Draft model path used by the drafter engine. Only used for decoupled draft.",
    )
    parser.add_argument(
        "--draft-tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the draft engine.",
    )
    parser.add_argument(
        "--target-tp-size",
        type=int,
        default=4,
        help="Tensor parallel size for the verifier engine. Also treated as verifier GPU count.",
    )
    parser.add_argument(
        "--num-speculative-steps",
        type=int,
        default=3,
        help="Number of speculative steps for decoupled speculation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate from the verifier.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used for the verifier generation request.",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help=(
            "Disable automatic tokenizer.apply_chat_template for chat-style prompt "
            "objects such as a list of {role, content} messages."
        ),
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
        "--print-prompt-chars",
        type=int,
        default=120,
        help="Number of prompt characters to print per sample.",
    )
    parser.add_argument(
        "--print-output-chars",
        type=int,
        default=200,
        help="Number of generated characters to print per sample.",
    )
    return parser.parse_args()


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _start_local_ray_head(port: int) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "ray",
        "start",
        "--head",
        f"--port={port}",
        "--node-ip-address=127.0.0.1",
        "--include-dashboard=false",
        "--disable-usage-stats",
        "--block",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def init_demo_ray(namespace: str) -> DemoRayRuntime:
    port = _pick_free_local_port()
    address = f"127.0.0.1:{port}"
    head_process = _start_local_ray_head(port)
    deadline = time.monotonic() + 30
    last_error = None

    while time.monotonic() < deadline:
        try:
            ray.init(
                address=address,
                namespace=namespace,
                ignore_reinit_error=True,
                log_to_driver=True,
                logging_level="ERROR",
            )
            return DemoRayRuntime(
                address=address,
                namespace=namespace,
                head_process=head_process,
            )
        except Exception as exc:
            last_error = exc
            if head_process.poll() is not None:
                break
            time.sleep(0.5)

    head_process.terminate()
    try:
        head_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        head_process.kill()
        head_process.wait(timeout=10)
    raise RuntimeError(
        f"Failed to start demo Ray head at {address}: {last_error!r}"
    ) from last_error


@ray.remote
class DraftEngineActor:
    def __init__(
        self,
        *,
        model_path: str,
        gpu_ids: list[str],
        tp_size: int,
        speculative_num_steps: int,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        self.drafter = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="DECOUPLED_DRAFT",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
        )

    def ready(self) -> bool:
        return True

    async def handle_draft_request(self, request: DraftRequest) -> DraftResult:
        return await self.drafter.handle_draft_request(request)

    async def handle_draft_requests(
        self, requests: list[DraftRequest]
    ) -> list[DraftResult]:
        return await self.drafter.handle_draft_requests(requests)

    async def terminate_draft_request(self, request_id: str):
        await self.drafter.terminate_draft_request(request_id)

    async def release_draft_session(self, request_id: str):
        await self.drafter.release_draft_session(request_id)

    def shutdown(self) -> None:
        self.drafter.shutdown()


def get_visible_gpu_ids() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    gpu_count = torch.cuda.device_count()
    return [str(i) for i in range(gpu_count)]


def allocate_demo_gpus(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    visible_gpu_ids = get_visible_gpu_ids()
    total_visible_gpus = len(visible_gpu_ids)
    required_gpus = args.draft_tp_size + args.target_tp_size
    if total_visible_gpus == 0:
        raise RuntimeError("No visible CUDA GPUs found for decoupled spec demo")
    if args.draft_tp_size <= 0 or args.target_tp_size <= 0:
        raise ValueError("draft-tp-size and target-tp-size must both be positive")
    if required_gpus > total_visible_gpus:
        raise ValueError(
            "Insufficient visible GPUs for the demo: "
            f"need {required_gpus}, but only {total_visible_gpus} visible ({visible_gpu_ids})"
        )

    draft_gpu_ids = visible_gpu_ids[: args.draft_tp_size]
    target_gpu_ids = visible_gpu_ids[
        args.draft_tp_size : args.draft_tp_size + args.target_tp_size
    ]
    return draft_gpu_ids, target_gpu_ids


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


def launch_drafter_actor(args: argparse.Namespace) -> tuple[ray.actor.ActorHandle, str]:
    actor_options = dict(
        name=ACTOR_NAME,
        num_gpus=args.draft_tp_size,
        max_concurrency=128,
    )
    debug_env_vars = _get_drafter_debug_env_vars()
    if debug_env_vars:
        actor_options["runtime_env"] = {"env_vars": debug_env_vars}
    actor = DraftEngineActor.options(**actor_options).remote(
        model_path=args.draft_model_path,
        gpu_ids=args.draft_gpu_ids,
        tp_size=args.draft_tp_size,
        speculative_num_steps=args.num_speculative_steps,
    )
    ray.get(actor.ready.remote())
    return actor, ACTOR_NAME


def launch_verifier(
    target_model_path: str,
    draft_actor_name: str,
    draft_actor_namespace: str,
    target_tp_size: int,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    draft_backend_process_kwargs: dict[str, object] | None = None,
) -> sgl.Engine:
    return sgl.Engine(
        model_path=target_model_path,
        tp_size=target_tp_size,
        speculative_algorithm="DECOUPLED_VERIFY",
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        draft_actor_names=[draft_actor_name],
        draft_actor_namespace=draft_actor_namespace,
        draft_backend_process_kwargs=draft_backend_process_kwargs,
    )


def infer_prompt_column(
    available_columns: list[str],
) -> str:
    for candidate in DEFAULT_PROMPT_COLUMN_CANDIDATES:
        if candidate in available_columns:
            return candidate

    raise ValueError(
        "Unable to auto-detect the prompt column. "
        f"Available columns: {available_columns}"
    )


def _looks_like_chat_message(value: Any) -> bool:
    return isinstance(value, dict) and "role" in value and "content" in value


def _is_chat_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_looks_like_chat_message(item) for item in value)
    )


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_segments: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_segments.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_segments.append(item["text"])
        return "".join(text_segments)
    return str(content)


def _messages_to_fallback_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = _flatten_message_content(message.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _maybe_parse_json_prompt(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


_CHATML_ROLE_PATTERN = re.compile(r"<\|im_start\|>(system|user|assistant)\n")


def _maybe_append_chatml_generation_prompt(
    prompt: str, *, enable_thinking: bool = False
) -> str:
    stripped = prompt.rstrip()
    if "<|im_start|>" not in stripped or "<|im_end|>" not in stripped:
        return prompt

    role_matches = list(_CHATML_ROLE_PATTERN.finditer(stripped))
    if not role_matches:
        return prompt

    last_role = role_matches[-1].group(1)
    thinking_suffix = "<think>\n" if enable_thinking else ""

    # The prompt already ends with an assistant generation prefix.
    if last_role == "assistant" and not stripped.endswith("<|im_end|>"):
        if enable_thinking and not stripped.endswith("<think>\n"):
            return stripped + thinking_suffix
        return stripped

    # ChatML user/system turns should terminate with an assistant prefix for generation.
    if last_role in {"system", "user"} and stripped.endswith("<|im_end|>"):
        return stripped + "\n<|im_start|>assistant\n" + thinking_suffix

    return prompt


def _build_chat_template_renderer(model_path: str, *, enable_thinking: bool = False):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None

    def render(messages: list[dict[str, Any]]) -> str:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("tokenizer.apply_chat_template returned an empty prompt")
        return prompt

    return render


def _normalize_prompt(
    value: Any,
    row_index: int,
    column_name: str,
    chat_template_renderer,
    *,
    enable_thinking: bool = False,
) -> str:
    if value is None:
        raise ValueError(
            f"Row {row_index} in column {column_name!r} is null, cannot build a prompt."
        )
    value = _maybe_parse_json_prompt(value)
    if isinstance(value, str):
        return _maybe_append_chatml_generation_prompt(
            value, enable_thinking=enable_thinking
        )
    if _is_chat_message_list(value):
        if chat_template_renderer is not None:
            try:
                return _maybe_append_chatml_generation_prompt(
                    chat_template_renderer(value),
                    enable_thinking=enable_thinking,
                )
            except Exception:
                pass
        return _messages_to_fallback_text(value)
    return str(value)


def load_prompt_batch(
    parquet_path: str,
    target_model_path: str,
    offset: int,
    batch_size: int,
    disable_chat_template: bool,
    enable_thinking: bool,
) -> tuple[str, list[str], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read parquet prompts. "
            "Please install it in the current Python environment."
        ) from exc

    if offset < 0:
        raise ValueError("offset must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    if offset >= total_rows:
        raise ValueError(
            f"offset {offset} is out of range for {parquet_path}; total rows: {total_rows}"
        )

    selected_column = infer_prompt_column(parquet_file.schema_arrow.names)
    chat_template_renderer = None
    if not disable_chat_template:
        chat_template_renderer = _build_chat_template_renderer(
            target_model_path, enable_thinking=enable_thinking
        )

    prompts: list[str] = []
    current_row = 0
    remaining_skip = offset
    reader_batch_size = max(batch_size, 1024)

    for record_batch in parquet_file.iter_batches(
        batch_size=reader_batch_size,
        columns=[selected_column],
    ):
        column_values = record_batch.column(0).to_pylist()
        if remaining_skip >= len(column_values):
            remaining_skip -= len(column_values)
            current_row += len(column_values)
            continue

        start_index = remaining_skip
        end_index = min(len(column_values), start_index + (batch_size - len(prompts)))
        for local_index in range(start_index, end_index):
            row_index = current_row + local_index
            prompts.append(
                _normalize_prompt(
                    column_values[local_index],
                    row_index,
                    selected_column,
                    chat_template_renderer,
                    enable_thinking=enable_thinking,
                )
            )

        current_row += len(column_values)
        remaining_skip = 0
        if len(prompts) >= batch_size:
            break

    if not prompts:
        raise ValueError(
            f"No prompts were loaded from {parquet_path} using column {selected_column!r}."
        )
    return selected_column, prompts, total_rows


def run_batch_generation(
    verifier: sgl.Engine, args: argparse.Namespace
) -> None:
    prompt_column, prompts, total_rows = load_prompt_batch(
        parquet_path=args.parquet_path,
        target_model_path=args.target_model_path,
        offset=args.offset,
        batch_size=args.batch_size,
        disable_chat_template=args.disable_chat_template,
        enable_thinking=args.enable_thinking,
    )

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }

    start_time = time.perf_counter()
    outputs = verifier.generate(prompt=prompts, sampling_params=sampling_params)
    elapsed_s = time.perf_counter() - start_time

    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} outputs, but got {len(outputs)} from verifier."
        )

    print(f"parquet_path: {args.parquet_path}")
    print(f"prompt_column: {prompt_column}")
    print(f"total_rows: {total_rows}")
    print(f"offset: {args.offset}")
    print(f"requested_batch_size: {args.batch_size}")
    print(f"loaded_batch_size: {len(prompts)}")
    print(f"generation_time_s: {elapsed_s:.3f}")
    print(f"avg_time_per_request_s: {elapsed_s / len(prompts):.3f}")

    accept_lengths = []
    for batch_index, (prompt, output) in enumerate(zip(prompts, outputs)):
        meta_info = output.get("meta_info", {})
        accept_length = meta_info.get("spec_accept_length")
        if accept_length is not None:
            accept_lengths.append(float(accept_length))

        print("=" * 80)
        print(f"sample_index: {batch_index}")
        print(f"row_index: {args.offset + batch_index}")
        print(f"prompt_preview: {prompt[: args.print_prompt_chars]}")
        print(f"output_text: {output.get('text', '')[: args.print_output_chars]}")
        print(f"spec_accept_length: {accept_length}")

    if accept_lengths:
        avg_accept_length = sum(accept_lengths) / len(accept_lengths)
        print("=" * 80)
        print(f"avg_spec_accept_length: {avg_accept_length:.3f}")


def main() -> None:
    args = parse_args()
    drafter_actor = None
    verifier = None
    ray_runtime = None
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        draft_gpu_ids, target_gpu_ids = allocate_demo_gpus(args)
        args.draft_gpu_ids = draft_gpu_ids
        ray_runtime = init_demo_ray(RAY_NAMESPACE)
        drafter_actor, drafter_actor_name = launch_drafter_actor(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(target_gpu_ids)
        verifier = launch_verifier(
            args.target_model_path,
            drafter_actor_name,
            RAY_NAMESPACE,
            args.target_tp_size,
            args.num_speculative_steps,
            args.num_speculative_steps + 1,
            draft_backend_process_kwargs={
                "ray_init_kwargs": ray_runtime.build_init_kwargs()
            },
        )
        run_batch_generation(verifier, args)
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


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
