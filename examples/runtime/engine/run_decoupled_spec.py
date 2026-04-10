"""
End-to-end decoupled speculative decoding demo.

This script:
1. Launches a `decoupled_draft` SGLang engine inside a Ray actor.
2. Launches a `decoupled_verify` SGLang engine that talks to the drafter actor.
3. Runs one end-to-end generation request through the verifier.
4. Prints generation latency, final text, and average acceptance rate.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import ray
import sglang as sgl
import torch
from sglang.srt.speculative.decoupled_spec_io import DraftRequest, DraftResult

ACTOR_NAME = "sglang-decoupled-drafter"
RAY_NAMESPACE = "sglang-decoupled-spec"
DEFAULT_PROMPT = """Solve this simple math problem using C++17 and return only one complete code block.

Problem:
Given two integers a and b, output their greatest common divisor.

Input:
One line containing two integers a and b.

Output:
One integer, the greatest common divisor of a and b.

Requirements:
- Read from standard input and write to standard output.
- Return only one complete C++17 code block.
- Do not include extra explanation.
"""


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
            "log_to_driver": False,
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
        description="Run an end-to-end decoupled speculative decoding demo."
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
        "--draft-max-capture-size",
        type=int,
        default=2048,
        help="Maximum token size used by the drafter piecewise CUDA graph capture.",
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
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the end-to-end verifier generation request.",
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
                log_to_driver=False,
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
        draft_max_capture_size: int,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        self.drafter = sgl.Engine(
            model_path=model_path,
            tp_size=tp_size,
            speculative_algorithm="decoupled_draft",
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_steps + 1,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            enable_piecewise_cuda_graph=True,
            piecewise_cuda_graph_max_tokens=draft_max_capture_size,
        )

    def ready(self) -> bool:
        return True

    def handle_draft_request(self, request: DraftRequest) -> DraftResult:
        return self.drafter.handle_draft_request(request)

    def terminate_draft_request(self, request_id: str):
        self.drafter.terminate_draft_request(request_id)

    def release_draft_session(self, request_id: str):
        self.drafter.release_draft_session(request_id)

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


def launch_drafter_actor(args: argparse.Namespace) -> tuple[ray.actor.ActorHandle, str]:
    actor = DraftEngineActor.options(
        name=ACTOR_NAME,
        num_gpus=args.draft_tp_size,
    ).remote(
        model_path=args.draft_model_path,
        gpu_ids=args.draft_gpu_ids,
        tp_size=args.draft_tp_size,
        speculative_num_steps=args.num_speculative_steps,
        draft_max_capture_size=args.draft_max_capture_size,
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
        speculative_algorithm="decoupled_verify",
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        draft_actor_names=[draft_actor_name],
        draft_actor_namespace=draft_actor_namespace,
        draft_backend_process_kwargs=draft_backend_process_kwargs,
    )


def run_end_to_end_generation(
    verifier: sgl.Engine, args: argparse.Namespace
) -> None:
    start_time = time.perf_counter()
    output = verifier.generate(
        prompt=args.prompt,
        sampling_params={
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    elapsed_s = time.perf_counter() - start_time

    text = output.get("text", "")
    meta_info = output.get("meta_info", {})
    avg_tokens_per_round = meta_info.get("spec_accept_length")
    print(f"output_text: {text[:100]}...")
    print(f"generation_time_s: {elapsed_s:.3f}")
    print(f"avg_tokens_per_round: {avg_tokens_per_round}")


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
        run_end_to_end_generation(verifier, args)
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
