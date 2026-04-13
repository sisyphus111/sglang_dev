import types
import unittest

from sglang.srt.speculative.decoupled_spec_io import (
    DraftRequest,
    DraftResult,
    RequestTerminateMessage,
    RequestTerminateReason,
    build_draft_scheduler_rid,
)
from sglang.srt.speculative.draft_proxy import DraftBackendManager, DraftProxy


class FakeDraftActorHandle:
    def __init__(self):
        self.submitted_batches = []
        self.terminate_calls = []
        self.handle_draft_requests = types.SimpleNamespace(remote=self._handle_remote)
        self.terminate_draft_request = types.SimpleNamespace(
            remote=self._terminate_remote
        )

    def _handle_remote(self, requests: list[DraftRequest]):
        self.submitted_batches.append(
            [(request.request_id, request.draft_round_id) for request in requests]
        )
        return object()

    def _terminate_remote(self, request_id: str):
        self.terminate_calls.append(request_id)
        return None


def make_request(request_id: str, round_id: int) -> DraftRequest:
    return DraftRequest(
        request_id=request_id,
        rid=build_draft_scheduler_rid(request_id),
        draft_round_id=round_id,
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        num_speculative_steps=4,
    )


def make_result(request_id: str, round_id: int) -> DraftResult:
    return DraftResult(
        request_id=request_id,
        rid=build_draft_scheduler_rid(request_id),
        draft_round_id=round_id,
        request_prompt_length=4,
        draft_token_ids=[4, 5],
    )


class TestDraftProxy(unittest.TestCase):
    def test_plan_routes_balances_new_requests_within_batch(self):
        actors = [FakeDraftActorHandle(), FakeDraftActorHandle()]
        proxy = DraftProxy(draft_actor_handles=actors)

        routed = proxy.plan_routes(
            [
                make_request("req-0", 0),
                make_request("req-1", 0),
                make_request("req-2", 0),
                make_request("req-3", 0),
            ]
        )

        self.assertEqual([route.draft_index for _, route in routed], [0, 1, 0, 1])

    def test_same_request_rounds_share_route_and_batch_dispatch(self):
        actors = [FakeDraftActorHandle(), FakeDraftActorHandle()]
        proxy = DraftProxy(draft_actor_handles=actors)

        requests = [
            make_request("req-0", 0),
            make_request("req-0", 1),
            make_request("req-1", 0),
        ]
        routed = proxy.plan_routes(requests)

        grouped: dict[int, list[DraftRequest]] = {}
        routes = {}
        for request, route in routed:
            grouped.setdefault(route.draft_index, []).append(request)
            routes[route.draft_index] = route
        for draft_index, grouped_requests in grouped.items():
            proxy.dispatch_batch(routes[draft_index], grouped_requests)

        self.assertEqual(actors[0].submitted_batches, [[("req-0", 0), ("req-0", 1)]])
        self.assertEqual(actors[1].submitted_batches, [[("req-1", 0)]])
        self.assertEqual(proxy.inflight_per_index, [2, 1])

    def test_complete_request_marks_ready_without_waiting_for_same_request(self):
        actor = FakeDraftActorHandle()
        proxy = DraftProxy(draft_actor_handles=[actor])

        requests = [make_request("req-0", 0), make_request("req-0", 1)]
        routed = proxy.plan_routes(requests)
        proxy.dispatch_batch(routed[0][1], requests)

        proxy.complete_request(requests[0].key, make_result("req-0", 0))
        proxy.complete_request(requests[1].key, make_result("req-0", 1))

        self.assertIn(requests[0].key, proxy.ready_results)
        self.assertIn(requests[1].key, proxy.ready_results)
        self.assertEqual(proxy.inflight_per_index, [0])

    def test_terminate_clears_all_rounds_for_request(self):
        actor = FakeDraftActorHandle()
        proxy = DraftProxy(draft_actor_handles=[actor])

        requests = [make_request("req-0", 0), make_request("req-0", 1)]
        routed = proxy.plan_routes(requests)
        proxy.dispatch_batch(routed[0][1], requests)
        proxy.ready_results[requests[0].key] = make_result("req-0", 0)

        proxy.terminate_request(
            RequestTerminateMessage(
                request_id="req-0",
                reason=RequestTerminateReason.ABORT,
            )
        )

        self.assertEqual(actor.terminate_calls, ["req-0"])
        self.assertEqual(proxy.inflight_requests, {})
        self.assertEqual(proxy.ready_results, {})
        self.assertNotIn("req-0", proxy.request_routes)


class TestDraftBackendManager(unittest.TestCase):
    def test_submit_batch_groups_requests_by_route(self):
        actor0 = FakeDraftActorHandle()
        actor1 = FakeDraftActorHandle()
        manager = DraftBackendManager.__new__(DraftBackendManager)
        manager.proxy = DraftProxy(draft_actor_handles=[actor0, actor1])
        manager.pending_poll_keys = {}

        manager._submit_draft_requests(
            [
                make_request("req-0", 0),
                make_request("req-0", 1),
                make_request("req-1", 0),
            ]
        )

        self.assertEqual(actor0.submitted_batches, [[("req-0", 0), ("req-0", 1)]])
        self.assertEqual(actor1.submitted_batches, [[("req-1", 0)]])


if __name__ == "__main__":
    unittest.main()
