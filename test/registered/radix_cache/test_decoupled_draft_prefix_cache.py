import types
import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.decoupled_draft_prefix_cache import DecoupledDraftPrefixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class FakeReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.arange(64, dtype=torch.int32).reshape(2, 32)
        self.freed_reqs = []
        self.free_slots = [1]

    def alloc(self, reqs):
        if len(self.free_slots) < len(reqs):
            return None
        for req in reqs:
            req.req_pool_idx = self.free_slots.pop(0)
        return [req.req_pool_idx for req in reqs]

    def write(self, indices, values):
        self.req_to_token[indices] = values

    def free(self, req):
        self.freed_reqs.append(req)
        self.free_slots.append(req.req_pool_idx)
        req.req_pool_idx = None


class FakeTokenToKVPoolAllocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.freed = []

    def free(self, indices):
        self.freed.append(indices.to(dtype=torch.int64, copy=True))


class FakeReq:
    def __init__(self, input_ids=None, output_ids=None, req_pool_idx=0):
        input_ids = input_ids or [1, 2, 3]
        self.rid = "draft-r0"
        self.origin_input_text = None
        self.origin_input_ids = list(input_ids)
        self.origin_input_ids_unpadded = list(input_ids)
        self.output_ids = list(output_ids or [])
        self.fill_ids = list(input_ids)
        self.req_pool_idx = req_pool_idx
        self.kv_committed_len = len(self.fill_ids) + len(self.output_ids)
        self.kv_allocated_len = self.kv_committed_len
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.cache_protected_len = 0
        self.return_logprob = False
        self.mamba_pool_idx = None
        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.draft_stateful_mode = False

    def pop_committed_kv_cache(self):
        self.kv_committed_freed = True
        return self.kv_committed_len

    def pop_overallocated_kv_cache(self):
        self.kv_overallocated_freed = True
        return self.kv_committed_len, self.kv_allocated_len


def make_cache():
    req_to_token_pool = FakeReqToTokenPool()
    token_to_kv_pool_allocator = FakeTokenToKVPoolAllocator()
    cache = DecoupledDraftPrefixCache(
        CacheInitParams(
            disable=True,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            page_size=1,
        )
    )
    return cache, req_to_token_pool, token_to_kv_pool_allocator


def match_req(cache, req):
    return cache.match_prefix(
        MatchPrefixParams(
            key=types.SimpleNamespace(token_ids=req.fill_ids[:-1], extra_key=None),
            req=req,
        )
    )


class TestDecoupledDraftPrefixCache(unittest.TestCase):
    def setUp(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.page_size = 1
        server_args.speculative_algorithm = "DECOUPLED_DRAFT"
        set_global_server_args_for_scheduler(server_args)

    def test_create_and_preserve_draft_req(self):
        cache, req_to_token_pool, _ = make_cache()
        req = FakeReq(output_ids=[4, 5])

        match_req(cache, req)

        self.assertTrue(req.draft_stateful_mode)
        self.assertTrue(cache.maybe_preserve_req(req))
        self.assertEqual(cache.session_debug_entries(), [{"session_id": "draft-r0", "kv_len": 5}])
        self.assertEqual(cache.protected_size(), 5)
        self.assertEqual(req_to_token_pool.freed_reqs, [req])
        self.assertEqual(req_to_token_pool.free_slots, [1, 0])
        self.assertIsNone(req.req_pool_idx)
        self.assertTrue(req.kv_committed_freed)
        self.assertTrue(req.kv_overallocated_freed)

    def test_prepare_next_round_reuses_prefix_and_fast_path(self):
        cache, req_to_token_pool, allocator = make_cache()
        req = FakeReq(output_ids=[4, 5])
        match_req(cache, req)
        cache.maybe_preserve_req(req)

        next_req = FakeReq(input_ids=[1, 2, 3, 4, 6], req_pool_idx=None)
        self.assertTrue(cache.can_match_draft_decode_fast_path(next_req))
        self.assertIsNone(next_req.req_pool_idx)

        match_result = match_req(cache, next_req)

        self.assertEqual(next_req.req_pool_idx, 1)
        self.assertEqual(next_req.origin_input_ids, [1, 2, 3, 4, 6])
        self.assertEqual(next_req.kv_committed_len, 4)
        self.assertEqual(next_req.kv_allocated_len, 4)
        self.assertEqual(next_req.cache_protected_len, 4)
        self.assertEqual(next_req.prefix_indices.tolist(), [0, 1, 2, 3])
        self.assertTrue(cache.can_use_draft_decode_fast_path(next_req))
        self.assertEqual(next_req.fill_ids[-1], 6)
        self.assertEqual(match_result.device_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(req_to_token_pool.req_to_token[1, :4].tolist(), [0, 1, 2, 3])
        self.assertEqual(cache.session_debug_entries(), [{"session_id": "draft-r0", "kv_len": 4}])
        self.assertEqual([item.tolist() for item in allocator.freed], [[4]])

    def test_can_match_draft_decode_fast_path_rejects_larger_delta(self):
        cache, _, _ = make_cache()
        req = FakeReq(output_ids=[4, 5])
        match_req(cache, req)
        cache.maybe_preserve_req(req)

        next_req = FakeReq(input_ids=[1, 2, 3, 7, 8, 9, 10], req_pool_idx=None)

        self.assertFalse(cache.can_match_draft_decode_fast_path(next_req))
        self.assertIsNone(next_req.req_pool_idx)

    def test_match_prefix_rejects_short_session(self):
        cache, _, _ = make_cache()
        req = FakeReq(output_ids=[4, 5])
        match_req(cache, req)
        cache.maybe_preserve_req(req)

        next_req = FakeReq(input_ids=[1, 2, 3, 4, 5, 6, 7], req_pool_idx=None)

        with self.assertRaisesRegex(
            AssertionError,
            "needs prefix len 6, but session draft-r0 only has 5",
        ):
            match_req(cache, next_req)

        self.assertEqual(cache.protected_size(), 5)
        self.assertEqual(
            cache.session_debug_entries(), [{"session_id": "draft-r0", "kv_len": 5}]
        )
        self.assertIsNone(next_req.req_pool_idx)

    def test_release_draft_session_frees_retained_session(self):
        cache, req_to_token_pool, allocator = make_cache()
        req = FakeReq(output_ids=[4])
        match_req(cache, req)
        req.output_ids = [4]
        req.kv_committed_len = 4
        req.kv_allocated_len = 4
        cache.maybe_preserve_req(req)
        waiting_queue = [req]

        cache.release_draft_session("draft-r0", waiting_queue=waiting_queue)

        self.assertEqual(waiting_queue, [req])
        self.assertEqual(cache.session_debug_entries(), [])
        self.assertEqual(req_to_token_pool.freed_reqs, [req])
        self.assertEqual([item.tolist() for item in allocator.freed], [[0, 1, 2, 3]])
        self.assertTrue(req.kv_committed_freed)
        self.assertTrue(req.kv_overallocated_freed)

    def test_preserve_updates_protected_size_by_delta(self):
        cache, _, _ = make_cache()
        req = FakeReq(output_ids=[4, 5])
        match_req(cache, req)
        cache.maybe_preserve_req(req)
        self.assertEqual(cache.protected_size(), 5)

        next_req = FakeReq(input_ids=[1, 2, 3, 4, 6], req_pool_idx=None)
        match_req(cache, next_req)
        self.assertEqual(cache.protected_size(), 4)

        next_req.output_ids = [7]
        next_req.kv_committed_len = 5
        next_req.kv_allocated_len = 5
        self.assertTrue(cache.maybe_preserve_req(next_req))
        self.assertEqual(cache.protected_size(), 5)

    def test_preserve_rejects_session_overwrite_without_reusing_old_kv(self):
        cache, req_to_token_pool, allocator = make_cache()
        req = FakeReq(output_ids=[4, 5], req_pool_idx=0)
        match_req(cache, req)
        cache.maybe_preserve_req(req)

        next_req = FakeReq(input_ids=[1, 2, 3, 4, 6], req_pool_idx=1)
        next_req.draft_stateful_mode = True
        next_req.kv_committed_len = 5
        next_req.kv_allocated_len = 5

        with self.assertRaisesRegex(
            AssertionError,
            "draft session overwrite would orphan previous KV slots",
        ):
            cache.maybe_preserve_req(next_req)

        self.assertEqual(cache.protected_size(), 5)
        self.assertEqual(
            cache.session_debug_entries(), [{"session_id": "draft-r0", "kv_len": 5}]
        )
        self.assertEqual([item.tolist() for item in allocator.freed], [])
        self.assertEqual(req_to_token_pool.freed_reqs, [req])

    def test_drop_draft_session_for_rid_updates_protected_size(self):
        cache, _, _ = make_cache()
        req = FakeReq(output_ids=[4, 5])
        match_req(cache, req)
        cache.maybe_preserve_req(req)

        self.assertEqual(cache.protected_size(), 5)
        cache.drop_draft_session_for_rid("draft-r0-3")
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(cache.session_debug_entries(), [])


if __name__ == "__main__":
    unittest.main()
