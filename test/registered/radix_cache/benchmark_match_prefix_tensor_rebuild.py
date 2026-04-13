import argparse
import math
import statistics
import time
from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey


@dataclass
class BenchmarkResult:
    matched_len: int
    num_chunks: int
    total_avg_us: float
    total_p50_us: float
    total_p95_us: float
    rebuild_avg_us: float
    rebuild_p50_us: float
    rebuild_p95_us: float


class _MockAllocator:
    def __init__(self, device: torch.device):
        self.device = device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark radix cache match_prefix and post-match tensor rebuild cost."
        )
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 512, 2048, 8192],
        help="Matched prefix lengths to benchmark.",
    )
    parser.add_argument(
        "--segment-len",
        type=int,
        default=128,
        help=(
            "Insert the cache as a chain of segments of this length. "
            "Smaller values produce more chunks and a more expensive torch.cat."
        ),
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1000,
        help="Number of warmup iterations per benchmark case.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=10000,
        help="Number of timed iterations per benchmark case.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to place cached KV index tensors on.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1,
        help="Optional page size for the simulated radix cache.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.segment_len <= 0:
        raise ValueError("--segment-len must be positive")
    if args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("--warmup-iters must be >= 0 and --iters must be > 0")
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device=cuda requested but CUDA is not available")

    if args.page_size > 1:
        if args.segment_len % args.page_size != 0:
            raise ValueError("--segment-len must be a multiple of --page-size")
        for length in args.lengths:
            if length % args.page_size != 0:
                raise ValueError(
                    f"matched length {length} must be a multiple of --page-size"
                )


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values_ns: list[int], q: float) -> float:
    if not values_ns:
        return 0.0
    sorted_values = sorted(values_ns)
    idx = min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1)
    return sorted_values[idx] / 1000.0


def build_segmented_cache(
    matched_len: int,
    segment_len: int,
    device: torch.device,
    page_size: int,
) -> tuple[RadixCache, RadixKey]:
    cache = RadixCache.create_simulated(
        disable=False,
        mock_allocator=_MockAllocator(device),
        page_size=page_size,
    )
    token_ids = list(range(matched_len))
    values = torch.arange(matched_len, dtype=torch.int64, device=device)

    end = segment_len
    while end <= matched_len:
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids[:end], extra_key=None),
                value=values[:end],
            )
        )
        end += segment_len

    if matched_len % segment_len != 0:
        cache.insert(
            InsertParams(
                key=RadixKey(token_ids=token_ids, extra_key=None),
                value=values,
            )
        )

    return cache, RadixKey(token_ids=token_ids, extra_key=None)


def get_match_chunks(cache: RadixCache, key: RadixKey) -> list[torch.Tensor]:
    lookup_key = RadixKey(token_ids=list(key.token_ids), extra_key=key.extra_key)
    lookup_key, _ = cache.maybe_bigram_convert(lookup_key)

    if cache.page_size != 1:
        page_aligned_len = len(lookup_key) // cache.page_size * cache.page_size
        lookup_key = lookup_key[:page_aligned_len]

    chunks, _ = cache._match_prefix_helper(cache.root_node, lookup_key)
    return chunks


def benchmark_match_prefix_total(
    cache: RadixCache,
    key: RadixKey,
    iters: int,
    warmup_iters: int,
    device: torch.device,
) -> list[int]:
    params = MatchPrefixParams(key=key)

    for _ in range(warmup_iters):
        cache.match_prefix(params)
    maybe_synchronize(device)

    durations_ns = []
    for _ in range(iters):
        maybe_synchronize(device)
        start_ns = time.perf_counter_ns()
        cache.match_prefix(params)
        maybe_synchronize(device)
        durations_ns.append(time.perf_counter_ns() - start_ns)
    return durations_ns


def benchmark_tensor_rebuild_only(
    chunks: list[torch.Tensor],
    matched_len: int,
    iters: int,
    warmup_iters: int,
    device: torch.device,
) -> list[int]:
    if not chunks:
        chunks = [torch.empty((0,), dtype=torch.int64, device=device)]

    for _ in range(warmup_iters):
        rebuilt = torch.cat(chunks)
        if rebuilt.numel() != matched_len:
            raise AssertionError(
                f"unexpected rebuilt tensor length: {rebuilt.numel()} != {matched_len}"
            )
    maybe_synchronize(device)

    durations_ns = []
    for _ in range(iters):
        maybe_synchronize(device)
        start_ns = time.perf_counter_ns()
        rebuilt = torch.cat(chunks)
        maybe_synchronize(device)
        if rebuilt.numel() != matched_len:
            raise AssertionError(
                f"unexpected rebuilt tensor length: {rebuilt.numel()} != {matched_len}"
            )
        durations_ns.append(time.perf_counter_ns() - start_ns)
    return durations_ns


def summarize(
    matched_len: int,
    num_chunks: int,
    total_times_ns: list[int],
    rebuild_times_ns: list[int],
) -> BenchmarkResult:
    return BenchmarkResult(
        matched_len=matched_len,
        num_chunks=num_chunks,
        total_avg_us=statistics.mean(total_times_ns) / 1000.0,
        total_p50_us=percentile(total_times_ns, 0.50),
        total_p95_us=percentile(total_times_ns, 0.95),
        rebuild_avg_us=statistics.mean(rebuild_times_ns) / 1000.0,
        rebuild_p50_us=percentile(rebuild_times_ns, 0.50),
        rebuild_p95_us=percentile(rebuild_times_ns, 0.95),
    )


def print_results(args: argparse.Namespace, results: list[BenchmarkResult]) -> None:
    print(
        f"device={args.device} page_size={args.page_size} "
        f"segment_len={args.segment_len} warmup_iters={args.warmup_iters} iters={args.iters}"
    )
    print(
        "matched_len  num_chunks  total_avg_us  total_p50_us  total_p95_us  "
        "rebuild_avg_us  rebuild_p50_us  rebuild_p95_us"
    )
    for result in results:
        print(
            f"{result.matched_len:<11d}"
            f"{result.num_chunks:<12d}"
            f"{result.total_avg_us:<14.3f}"
            f"{result.total_p50_us:<14.3f}"
            f"{result.total_p95_us:<14.3f}"
            f"{result.rebuild_avg_us:<17.3f}"
            f"{result.rebuild_p50_us:<17.3f}"
            f"{result.rebuild_p95_us:<17.3f}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = torch.device(args.device)
    results = []

    for matched_len in args.lengths:
        cache, key = build_segmented_cache(
            matched_len=matched_len,
            segment_len=args.segment_len,
            device=device,
            page_size=args.page_size,
        )
        chunks = get_match_chunks(cache, key)

        total_times_ns = benchmark_match_prefix_total(
            cache=cache,
            key=key,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
            device=device,
        )
        rebuild_times_ns = benchmark_tensor_rebuild_only(
            chunks=chunks,
            matched_len=matched_len,
            iters=args.iters,
            warmup_iters=args.warmup_iters,
            device=device,
        )
        results.append(
            summarize(
                matched_len=matched_len,
                num_chunks=len(chunks),
                total_times_ns=total_times_ns,
                rebuild_times_ns=rebuild_times_ns,
            )
        )

    print_results(args, results)


if __name__ == "__main__":
    main()
