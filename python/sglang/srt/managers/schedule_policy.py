from __future__ import annotations

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Request scheduler policy"""

import os
import random
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS_ESTIMATION = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (against existing cache) less than this value,
# the scheduler runs the in-batch prefix caching check for this request.
# If we set it to -1, it means we disable in-batch prefix caching.
IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD", "32")
)

# Threshold for in-batch prefix cache.
# If a request has a matched prefix length (within the waiting queue) larger than this value,
# the scheduler deprioritizes this request
IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD = int(
    os.environ.get("IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD", "32")
)


IGNORE_EOS_RESERVE_TOKENS = 1


class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""

    LPM = "lpm"  # longest prefix match
    DFS_WEIGHT = "dfs-weight"  # depth-first search weighting


class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""

    FCFS = "fcfs"  # first come first serve
    LOF = "lof"  # longest output first
    RANDOM = "random"


class SchedulePolicy:
    Policy = Union[CacheAwarePolicy, CacheAgnosticPolicy]

    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache(
            req_to_token_pool=None,
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=False,
        )

    def calc_priority(self, waiting_queue: List[Req]) -> bool:
        if self.policy == CacheAgnosticPolicy.FCFS:
            # A shortcut for FCFS
            return False

        policy = self._determine_active_policy(waiting_queue)

        prefix_computed = False
        if isinstance(policy, CacheAwarePolicy):
            prefix_computed = True
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(waiting_queue)
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

        return prefix_computed

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy

    def _validate_and_adjust_policy(
        self, policy: str, tree_cache: BasePrefixCache
    ) -> Policy:
        """
        Validates the policy and adjusts it if necessary based on tree cache settings.
        """
        try:
            policy_enum = CacheAwarePolicy(policy)
            if getattr(tree_cache, "disable", True):
                # If tree_cache is disabled, using CacheAgnosticPolicy policy
                return CacheAgnosticPolicy.FCFS
            return policy_enum
        except ValueError:
            try:
                return CacheAgnosticPolicy(policy)
            except ValueError:
                raise ValueError(f"Unknown schedule_policy: {policy=}")

    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.adjust_max_prefix_ids()

            # NOTE: the prefix_indices must always be aligned with last_node
            r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length = (
                self.tree_cache.match_prefix(rid=r.rid, key=prefix_ids)
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                in_batch_matching_prefixes, _, _, _ = (
                    self.waiting_queue_radix_tree.match_prefix(
                        rid=r.rid, key=prefix_ids
                    )
                )
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        prefix_ids, torch.empty(len(prefix_ids), dtype=torch.bool)
                    )
        return temporary_deprioritized

    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -len(r.prefix_indices)
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )

    @staticmethod
    def _sort_by_longest_output(waiting_queue: List[Req]) -> None:
        """Sorts the waiting queue based on the longest output (max_new_tokens)."""
        waiting_queue.sort(key=lambda x: -x.sampling_params.max_new_tokens)

    @staticmethod
    def _sort_randomly(waiting_queue: List[Req]) -> None:
        """Shuffles the waiting queue randomly."""
        random.shuffle(waiting_queue)

    @staticmethod
    def _calc_weight(cur_node: TreeNode, node_to_weight: Dict[TreeNode, int]) -> None:
        for child in cur_node.children.values():
            SchedulePolicy._calc_weight(child, node_to_weight)
            node_to_weight[cur_node] += node_to_weight[child]

    @staticmethod
    def _get_dfs_priority(
        cur_node: TreeNode,
        node_to_priority: Dict[TreeNode, int],
        last_node_to_reqs: Dict[TreeNode, List[Req]],
        q: List,
    ) -> None:
        childs = [child for child in cur_node.children.values()]
        childs.sort(key=lambda x: -node_to_priority[x])
        for child in childs:
            SchedulePolicy._get_dfs_priority(
                child, node_to_priority, last_node_to_reqs, q
            )
        q.extend(last_node_to_reqs[cur_node])


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No token left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        mixed_with_decode_tokens: int = 0,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        # new_token_ratio: 这是一个动态衰减的“折损率/超分比例”系数，用于控制计算显存预算时的严苛程度。
        # 很多请求其实根本跑不到 max_new_tokens 就会提前结束(EOS)。如果全按 max_new 预留显存，会导致系统并发量(吞吐)极低。
        # 因此，系统会用 (max_new_tokens * new_token_ratio) 来作为【预计要消耗】的显存。
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - mixed_with_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= mixed_with_decode_tokens

        # rem_total_token_offset: "已承诺的显存债务 (Committed Token Budget / Debt)"
        # 物理显存池 (kv_pool) 里的 available_size 是当前的绝对空闲量，但调度器需要为“未来”做规划。
        # 当我们把一个请求放进批次时，它不仅需要消耗当前的 Prompt 长度，未来还会不断吐字消耗新的空间。
        # 这个 offset 变量记录了：我们已经向当前批次里的请求“承诺/预留”了多少个 token 槽位。
        # 算最终真正可用的逻辑额度 (rem_total_tokens) 时，就是拿当前的物理空闲量减去这个 offset 债务。
        self.rem_total_token_offset = mixed_with_decode_tokens
        self.cur_rem_token_offset = mixed_with_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0

        if running_batch is not None:
            # 初始化记账员时，计算正在跑的 running_batch 里的老请求们“未来预计还会消耗多少显存”。
            # 这里的计算直接用到了 new_token_ratio 进行打折预估，而不是傻傻地预留全量的 max_new_tokens。
            self.rem_total_token_offset += sum(
                [
                    min(
                        (r.sampling_params.max_new_tokens - len(r.output_ids)),
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    )
                    * self.new_token_ratio
                    for r in running_batch.reqs
                ]
            )

        self.is_hybrid = isinstance(
            self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        )

    @property
    def rem_total_tokens(self):
        if self.is_hybrid:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size(),
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.rem_total_token_offset

    @property
    def cur_rem_tokens(self):
        if self.is_hybrid:
            available_and_evictable = min(
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size(),
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size(),
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )

        return available_and_evictable - self.cur_rem_token_offset

    def ceil_paged_tokens(self, tokens: int) -> int:
        return -(-tokens // self.page_size) * self.page_size

    def budget_state(self):
        if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0 or (
            self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0
        ):
            return AddReqResult.OTHER

        return AddReqResult.CONTINUE

    def _update_prefill_budget(
        self, prefix_len: int, extend_input_len: int, max_new_tokens: int
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        self.rem_total_token_offset += extend_input_len + max_new_tokens
        self.cur_rem_token_offset += extend_input_len
        self.rem_input_tokens -= extend_input_len
        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len

    def add_chunked_req(self, req: Req):
        truncated = req.extend_input_len > self.rem_chunk_tokens
        req.extend_input_len = min(req.extend_input_len, self.rem_chunk_tokens)
        req.fill_ids = req.fill_ids[: len(req.prefix_indices) + req.extend_input_len]
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_input_len,
            (
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION)
                if not truncated
                else 0
            ),
        )

        # Return if chunked prefill not finished
        return req if truncated else None

    @contextmanager
    def _lock_node(self, last_node: TreeNode):
        if self.is_hybrid:
            try:
                swa_uuid_for_lock = self.tree_cache.inc_lock_ref(last_node)
                yield None
            finally:
                self.tree_cache.dec_lock_ref(last_node, swa_uuid_for_lock)
        else:
            try:
                self.tree_cache.inc_lock_ref(last_node)
                yield None
            finally:
                self.tree_cache.dec_lock_ref(last_node)

    def add_one_req_ignore_eos(self, req: Req, has_chunked_req: bool):
        # Early exit if no enough tokens for the input tokens
        if self.ceil_paged_tokens(req.extend_input_len) > min(
            self.cur_rem_tokens, self.rem_total_tokens
        ):
            return AddReqResult.NO_TOKEN

        def add_req_state(r, insert_sort=False):
            # 对设置了 ignore_eos=True 的请求，必须严格老老实实按 1.0 的满额去预留，因为它们肯定会跑到 max_new_tokens
            # 对普通请求，使用 new_token_ratio 比例打折来预估它未来还会生成多少 token
            new_token_ratio = (
                1.0 if r.sampling_params.ignore_eos else self.new_token_ratio
            )
            tokens_left = r.sampling_params.max_new_tokens * new_token_ratio - len(
                r.output_ids
            )
            tokens_occupied = len(r.origin_input_ids) + len(r.output_ids)

            if tokens_left <= 0:
                return

            if not insert_sort:
                self.req_states.append((tokens_left, tokens_occupied))
            else:
                i = 0
                for i in range(len(self.req_states)):
                    if tokens_left <= self.req_states[i][0]:
                        break
                self.req_states.insert(i, (tokens_left, tokens_occupied))

        if self.req_states is None:
            self.req_states = []
            add_req_state(req)
            if self.running_batch is not None:
                for r in self.running_batch.reqs:
                    add_req_state(r)
            for r in self.can_run_list:
                add_req_state(r)
            self.req_states.sort(key=lambda x: x[0])
        else:
            add_req_state(req, insert_sort=True)

        if not self.is_hybrid:
            # Skip this logic for swa. The SWA has different memory management, and
            # this mechanism is underestimating the memory usage.
            cur_rem_tokens = self.cur_rem_tokens - len(req.origin_input_ids)
            tokens_freed = 0
            for i, (tokens_left, tokens_occupied) in enumerate(self.req_states):
                # tokens_left gives a reservative calculation as the last token is not stored
                bs = len(self.req_states) - i
                min_free_tokens = cur_rem_tokens + tokens_freed - tokens_left * bs
                # reserve tokens for corner cases
                if min_free_tokens <= IGNORE_EOS_RESERVE_TOKENS * bs:
                    return AddReqResult.NO_TOKEN
                tokens_freed += tokens_occupied

        if (
            self.rem_chunk_tokens is None  # chunked prefill is disabled
            or req.extend_input_len <= self.rem_chunk_tokens  # it is the last chunk
        ):
            # Non-chunked prefill
            self.can_run_list.append(req)
            self._update_prefill_budget(
                0,
                req.extend_input_len,
                min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION),
            )
        else:
            if self.rem_chunk_tokens == 0:
                return AddReqResult.OTHER

            # Chunked prefill
            trunc_len = self.rem_chunk_tokens

            req.extend_input_len = trunc_len
            req.fill_ids = req.fill_ids[:trunc_len]
            self.can_run_list.append(req)
            self.new_chunked_req = req
            self._update_prefill_budget(0, trunc_len, 0)

        return self.budget_state()

    def add_one_req(self, req: Req, has_chunked_req: bool):
        # 【核心功能】：尝试将一个请求加入到当前的 Prefill 批次中。内部进行显存预算和算力预算的双重校验。
        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req, has_chunked_req)

        # 1. 预估该请求所需的总 KV Cache 数量 (用于显存约束)
        # = 还需要处理的输入长度(extend_input_len) + 预计会生成的输出长度(max_new_tokens，并做了防爆截断)
        total_tokens = req.extend_input_len + min(
            req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS_ESTIMATION
        )

        # 2. 计算实际需要耗费算力的 Input Tokens (用于算力约束)
        # adjusting the input_tokens based on host_hit_length and page_size
        # 如果命中了一些在 Host (CPU) 内存上的缓存，由于只需要传输(加载回GPU)而无需重新计算，可以减去
        real_input_tokens = req.extend_input_len - req.host_hit_length
        # 向上对齐到 page_size (因为底层内存池按页分配)
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        # 3. 粗略的第一次检查：
        # 如果预估总显存开销超过了系统剩余可用总显存(rem_total_tokens)，拒绝(NO_TOKEN)
        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        # 如果实际算力开销超过了单次 Prefill 允许的 Token 阈值上限(rem_input_tokens)，
        # 且当前批次里已经有其他请求了(len != 0)，说明装不下了，放到下一批(OTHER)
        if real_input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
            return AddReqResult.OTHER

        # 4. 锁定相关节点，准备正式接纳
        with self._lock_node(req.last_node):
            # 锁定节点可能会触发别的内部驱逐操作，导致可用显存减少，因此加锁后再检查一次
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            # 如果存在命中在 Host 内存里的缓存，启动从 Host 到 GPU 的加载操作
            if req.host_hit_length > 0:
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    req.last_host_node, req.host_hit_length
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                # 重新计算剩余真正需要计算的前向 token 长度
                req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
                prefix_len = len(req.prefix_indices)

            # 重新计算向上取整后的、精确的算力消耗
            input_tokens = self.ceil_paged_tokens(req.extend_input_len)

            # 再做一次精确的算力开销检查
            if input_tokens >= self.rem_input_tokens and len(self.can_run_list) != 0:
                return AddReqResult.OTHER

            # 5. 分配：根据是否触发 Chunked Prefill 分为两种情况
            if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # 情况A: Non-chunked prefill (完整装入)
                # 未开启 chunked 机制，或者开启了但是当前这个请求能被完全塞进本轮 chunk 预算里
                self.can_run_list.append(req)
                if self.is_hybrid:
                    swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                    req.swa_uuid_for_lock = swa_uuid_for_lock
                else:
                    self.tree_cache.inc_lock_ref(req.last_node)
                
                # 扣除相应的预算 (算力预算扣除 input_tokens，显存预算扣除 input + max_new)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS_ESTIMATION,
                    ),
                )
            else:
                # 情况B: Chunked prefill (切块装入)
                # 该请求的长度超过了本轮还能塞下的 chunk 预算，必须进行切断 (Truncate)
                
                # Make sure at least one page is available (确保截断后至少能有1个完整 page 的空间)
                trunc_len = self.rem_chunk_tokens - self.page_size + 1
                if trunc_len <= 0:
                    # 如果连一点点空间都挤不出了，就不要切了，直接放到下一轮
                    return AddReqResult.OTHER

                # 执行截断，修改请求内部游标
                req.extend_input_len = trunc_len
                req.fill_ids = req.fill_ids[: len(req.prefix_indices) + trunc_len]

                # 将截断后的请求加入批次，同时将其标记为新的 chunked_req (以便下一轮继续算它剩下的部分)
                self.can_run_list.append(req)
                self.new_chunked_req = req
                if self.is_hybrid:
                    swa_uuid_for_lock = self.tree_cache.inc_lock_ref(req.last_node)
                    req.swa_uuid_for_lock = swa_uuid_for_lock
                else:
                    self.tree_cache.inc_lock_ref(req.last_node)
                
                # 扣除相应的预算：只扣除本次计算的部分 (trunc_len)；
                # 注意 max_new_tokens 传入了 0，因为没算完，还没到分配输出显存的时候
                self._update_prefill_budget(prefix_len, trunc_len, 0)

        # 6. 最后检查状态，判断当前大批次是否已满，返回 CONTINUE(还可以接着装) 或者 NO_TOKEN/OTHER(满了)
        return self.budget_state()
