"""
Phase-1 launch sketch for decoupled speculative decoding.

This example intentionally mirrors the target usage shape:
1. Launch a drafter engine with `speculative_algorithm="decoupled_draft"`.
2. Launch a verifier engine with `speculative_algorithm="decoupled_verify"`.
3. Send generation requests to the verifier.

Current phase-1 boundary:
- `decoupled_draft` only exposes a scheduler scaffold and raises `NotImplementedError`.
- `decoupled_verify` expects real drafter Ray actors that implement
  `handle_draft_request(...)` and `terminate_draft_request(...)`.
"""

from __future__ import annotations

import os

import sglang as sgl


def _parse_actor_names() -> list[str]:
    raw_actor_names = os.getenv("SGLANG_DRAFT_ACTOR_NAMES", "")
    return [item.strip() for item in raw_actor_names.split(",") if item.strip()]


def launch_drafter() -> sgl.Engine:
    return sgl.Engine(
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_algorithm="decoupled_draft",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )


def launch_verifier(
    draft_actor_names: list[str],
) -> sgl.Engine:
    return sgl.Engine(
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_algorithm="decoupled_verify",
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        draft_actor_names=draft_actor_names,
    )


def main() -> None:
    drafter = None
    verifier = None

    try:
        try:
            drafter = launch_drafter()
        except NotImplementedError as exc:
            print("`decoupled_draft` scaffold reached as expected.")
            print(f"  reason: {exc}")
            print("  next step: implement DraftScheduler in a follow-up phase.")

        draft_actor_names = _parse_actor_names()
        if not draft_actor_names:
            print(
                "Skipping verifier launch because `SGLANG_DRAFT_ACTOR_NAMES` is empty."
            )
            print(
                "Provide comma-separated Ray actor names to exercise the decoupled_verify path."
            )
            return

        verifier = launch_verifier(draft_actor_names=draft_actor_names)
        outputs = verifier.generate(
            ["Explain decoupled speculative decoding in one sentence."],
            {"temperature": 0, "max_new_tokens": 32},
        )
        print(outputs)
    finally:
        if verifier is not None:
            verifier.shutdown()
        if drafter is not None:
            drafter.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
