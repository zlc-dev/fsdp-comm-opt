#!/usr/bin/env python3
"""Distributed correctness and performance benchmark for reduce-scatter.

Launch on every visible GPU with:

    torchrun --standalone --nproc-per-node=gpu benchmark_reduce_scatter.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

ReduceScatter = Callable[
    [torch.Tensor, torch.Tensor, dist.ProcessGroup], object
]


@dataclass
class Method:
    name: str
    reduce_scatter: ReduceScatter
    atol: float
    rtol: float
    reset: Callable[[], None] | None = None


@dataclass
class Accuracy:
    max_abs: float
    mean_abs: float
    rmse: float
    relative_l2: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-numel",
        type=int,
        default=4_194_304,
        help="number of output elements per rank (default: 1048576)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--zero-threshold", type=float, default=1e-6)
    parser.add_argument("--native-atol", type=float, default=1e-5)
    parser.add_argument("--native-rtol", type=float, default=1e-5)
    parser.add_argument("--compressed-atol", type=float, default=5e-2)
    parser.add_argument("--compressed-rtol", type=float, default=2e-2)
    parser.add_argument(
        "--allow-incorrect",
        action="store_true",
        help="report correctness failures without returning a non-zero status",
    )
    args = parser.parse_args()

    for name in ("shard_numel", "iters", "repeats"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.std < 0:
        parser.error("--std must be non-negative")
    if args.zero_threshold < 0:
        parser.error("--zero-threshold must be non-negative")
    return args


def native_reduce_scatter(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> object:
    return dist.reduce_scatter_tensor(
        output, input_tensor, group=group, op=dist.ReduceOp.SUM
    )


def native_all_to_all_reduce(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    group: dist.ProcessGroup,
) -> None:
    """Exchange destination shards without compression, then reduce locally."""
    world_size = dist.get_world_size(group)
    shard_numel = output.numel()
    received = torch.empty_like(input_tensor)
    dist.all_to_all_single(received, input_tensor, group=group)

    # all_to_all_single returns one shard from each source rank, ordered by
    # source. Match the compressed implementations by accumulating locally in
    # FP32 before converting to the requested output dtype.
    source_shards = received.view(world_size, shard_numel)
    reduced = source_shards.sum(dim=0, dtype=torch.float32)
    output.copy_(reduced.view(output.shape))


def generate_input(
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    mean: float,
    std: float,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    values = torch.randn(
        numel,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    values.mul_(std).add_(mean)
    return values.to(dtype)


def reference_from_all_gather(
    input_tensor: torch.Tensor,
    shard_numel: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Form this rank's BF16 result using FP32 accumulation."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    gathered = torch.empty(
        world_size * input_tensor.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_gather_into_tensor(gathered, input_tensor, group=group)
    source_destination = gathered.view(world_size, world_size, shard_numel)
    expected = source_destination[:, rank, :].sum(
        dim=0, dtype=torch.float32
    ).to(torch.bfloat16)
    del gathered, source_destination
    return expected


def global_accuracy(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    group: dist.ProcessGroup,
) -> Accuracy:
    diff = actual - expected
    max_abs = diff.abs().max()
    abs_sum = diff.abs().sum(dtype=torch.float32)
    square_sum = diff.float().square().sum()
    reference_square_sum = expected.float().square().sum()
    passed = torch.tensor(
        int(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        dtype=torch.int32,
        device=actual.device,
    )

    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(abs_sum, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(reference_square_sum, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=group)

    total_numel = actual.numel() * dist.get_world_size(group)
    rmse = torch.sqrt(square_sum / total_numel)
    relative_l2 = torch.sqrt(
        square_sum / torch.clamp(reference_square_sum, min=1e-30)
    )
    return Accuracy(
        max_abs=max_abs.item(),
        mean_abs=(abs_sum / total_numel).item(),
        rmse=rmse.item(),
        relative_l2=relative_l2.item(),
        passed=bool(passed.item()),
    )


def check_method(
    method: Method,
    input_tensor: torch.Tensor,
    expected: torch.Tensor,
    shard_numel: int,
    group: dist.ProcessGroup,
) -> Accuracy:
    if method.reset is not None:
        method.reset()
    output = torch.empty(
        shard_numel, dtype=input_tensor.dtype, device=input_tensor.device
    )
    method.reduce_scatter(output, input_tensor, group)
    torch.cuda.synchronize(input_tensor.device)
    return global_accuracy(output, expected, method.atol, method.rtol, group)


def benchmark_method(
    method: Method,
    input_tensor: torch.Tensor,
    shard_numel: int,
    warmup: int,
    iters: int,
    repeats: int,
    group: dist.ProcessGroup,
) -> list[float]:
    output = torch.empty(
        shard_numel, dtype=input_tensor.dtype, device=input_tensor.device
    )
    timings_ms = []
    for _ in range(repeats):
        if method.reset is not None:
            method.reset()
        for _ in range(warmup):
            method.reduce_scatter(output, input_tensor, group)
        torch.cuda.synchronize(input_tensor.device)
        dist.barrier(group=group)

        start = time.perf_counter()
        for _ in range(iters):
            method.reduce_scatter(output, input_tensor, group)
            torch.cuda.synchronize(input_tensor.device)
        elapsed = time.perf_counter() - start

        # Collective latency is determined by the slowest participating rank.
        elapsed_tensor = torch.tensor(
            elapsed, dtype=torch.float32, device=input_tensor.device
        )
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=group)
        timings_ms.append(elapsed_tensor.item() * 1000.0 / iters)
    return timings_ms


def print_configuration(
    args: argparse.Namespace,
    world_size: int,
    input_tensor: torch.Tensor,
) -> None:
    input_mib = input_tensor.numel() * input_tensor.element_size() / 2**20
    gathered_mib = input_mib * world_size
    print(
        f"world_size={world_size} "
        f"shard_numel={args.shard_numel:,} input_per_rank={input_mib:.2f} MiB "
        f"all_gather_buffer_per_rank={gathered_mib:.2f} MiB"
    )
    print(
        f"normal(mean={args.mean:g}, std={args.std:g}) "
        f"warmup={args.warmup} iters={args.iters} repeats={args.repeats}"
    )


def run(args: argparse.Namespace) -> bool:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "distributed environment is not initialized; launch with "
            "`torchrun --standalone --nproc-per-node=gpu "
            "benchmark_reduce_scatter.py`"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    if world_size < 2:
        raise RuntimeError(
            "this benchmark requires at least two visible GPUs; with one rank "
            "ZipCCL and ZZipCCL fall back to native reduce-scatter"
        )
    dtype = torch.bfloat16

    try:
        from main.zipccl_comm import ZipCCLReduceScatter
        from main.zzipccl_comm import ZZipCCLReduceScatter
        from main.zzipccl_ef_comm import ZZipCCLEfReduceScatter
    except ImportError as error:
        raise RuntimeError(
            "failed to import the ZipCCL CUDA extensions; rebuild zipccl and "
            "zzipccl against the active PyTorch installation"
        ) from error

    total_numel = args.shard_numel * world_size
    input_tensor = generate_input(
        total_numel,
        dtype,
        device,
        args.seed + rank,
        args.mean,
        args.std,
    )
    if rank == 0:
        print_configuration(args, world_size, input_tensor)

    expected = reference_from_all_gather(
        input_tensor, args.shard_numel, group
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    zipccl = ZipCCLReduceScatter()
    zzipccl = ZZipCCLReduceScatter(
        zero_threshold=args.zero_threshold,
    )
    zzipccl_ef = ZZipCCLEfReduceScatter(
        zero_threshold=args.zero_threshold,
    )
    methods = (
        Method(
            "Native",
            native_reduce_scatter,
            args.native_atol,
            args.native_rtol,
        ),
        Method(
            "All2All+Reduce",
            native_all_to_all_reduce,
            args.native_atol,
            args.native_rtol,
        ),
        Method(
            "ZipCCL",
            zipccl,
            args.compressed_atol,
            args.compressed_rtol,
        ),
        Method(
            "ZZipCCL",
            zzipccl,
            args.compressed_atol,
            args.compressed_rtol,
        ),
        Method(
            "ZZipCCL-EF",
            zzipccl_ef,
            args.compressed_atol,
            args.compressed_rtol,
            reset=zzipccl_ef.reset,
        ),
    )

    if rank == 0:
        print("\nCorrectness against all-gather reference:")
        print(
            f"{'method':<14} {'status':<6} {'max_abs':>12} "
            f"{'mean_abs':>12} {'rmse':>12} {'relative_l2':>12}"
        )
    all_passed = True
    for method in methods:
        accuracy = check_method(
            method, input_tensor, expected, args.shard_numel, group
        )
        all_passed &= accuracy.passed
        if rank == 0:
            status = "PASS" if accuracy.passed else "FAIL"
            print(
                f"{method.name:<14} {status:<6} {accuracy.max_abs:12.5e} "
                f"{accuracy.mean_abs:12.5e} {accuracy.rmse:12.5e} "
                f"{accuracy.relative_l2:12.5e}"
            )

    input_bytes = input_tensor.numel() * input_tensor.element_size()
    if rank == 0:
        print("\nPerformance (per-call synchronized, slowest-rank wall time):")
        print(
            f"{'method':<14} {'median_ms':>12} {'min_ms':>12} "
            f"{'input_GB/s':>12} {'speedup':>10}"
        )
    native_median_ms = None
    for method in methods:
        timings = benchmark_method(
            method,
            input_tensor,
            args.shard_numel,
            args.warmup,
            args.iters,
            args.repeats,
            group,
        )
        median_ms = statistics.median(timings)
        min_ms = min(timings)
        throughput = input_bytes / (median_ms / 1000.0) / 1e9
        if native_median_ms is None:
            native_median_ms = median_ms
        speedup = native_median_ms / median_ms
        if rank == 0:
            print(
                f"{method.name:<14} {median_ms:12.3f} {min_ms:12.3f} "
                f"{throughput:12.3f} {speedup:9.3f}x"
            )

    return all_passed


def main() -> None:
    args = parse_args()
    try:
        all_passed = run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if not all_passed and not args.allow_incorrect:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
