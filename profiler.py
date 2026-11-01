#!/usr/bin/env python3
"""Independent zzipccl/zipccl CUDA kernel and communication-size benchmark."""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
from zzipccl import zzip
from zipccl import hopper_fastzip2

ELEMS_PER_BLOCK = 128 * 32


def align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class BenchmarkInput:
    numel: int
    padded: int
    device: torch.device
    input: torch.Tensor
    n_8: torch.Tensor
    orig_n_8: torch.Tensor


def estimate_base(input_tensor: torch.Tensor, offset: float) -> torch.Tensor:
    """Estimate the exponent base from the unpadded input values."""
    std = input_tensor.std(unbiased=False).float()
    best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + offset)
    return torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)


@dataclass(frozen=True)
class CommunicationStats:
    original_bytes: int
    deterministic_bytes: int
    outlier_bytes: int
    aligned_outlier_bytes: int
    metadata_bytes: int = 4

    @property
    def payload_bytes(self):
        """Number of meaningful compressed bytes, excluding alignment."""
        return self.deterministic_bytes + self.outlier_bytes

    @property
    def wire_bytes(self):
        """Bytes sent per rank by the corresponding communication classes."""
        # The collectives use a one-byte buffer when there are no outliers.
        return (
            self.deterministic_bytes
            + max(self.aligned_outlier_bytes, 1)
            + self.metadata_bytes
        )

    @property
    def compression_ratio(self):
        return self.original_bytes / self.wire_bytes


class KernelBenchmark:
    name = "kernel"

    def __init__(self, data: BenchmarkInput):
        self.data = data

    def compress(self):
        raise NotImplementedError

    def decompress(self):
        raise NotImplementedError

    def output(self):
        raise NotImplementedError

    def outlier_count(self):
        raise NotImplementedError

    def communication_stats(self):
        outlier_bytes = self.outlier_count()
        return CommunicationStats(
            original_bytes=self.data.numel * self.data.input.element_size(),
            deterministic_bytes=self.compressed.numel(),
            outlier_bytes=outlier_bytes,
            aligned_outlier_bytes=align(outlier_bytes),
        )

    def measure(self, operation, warmup, iters, repeats):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(self.data.device)
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                operation()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / iters)
        return statistics.median(samples), statistics.mean(samples)

    def run(self, warmup, iters, repeats):
        self.compress()
        torch.cuda.synchronize(self.data.device)
        self.decompress()
        torch.cuda.synchronize(self.data.device)
        torch.cuda.nvtx.range_push(f"{self.name}_compress")
        comp = self.measure(self.compress, warmup, iters, repeats)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"{self.name}_decompress")
        decomp = self.measure(self.decompress, warmup, iters, repeats)
        torch.cuda.nvtx.range_pop()
        expected = self.data.input[: self.data.numel].float()
        error = (self.output().float() - expected).abs().max().item()
        return comp, decomp, error


class ZZipCCLBenchmark(KernelBenchmark):
    name = "zzipccl"

    def __init__(self, data):
        super().__init__(data)
        padded, blocks, device = data.padded, data.padded // ELEMS_PER_BLOCK, data.device
        det_bytes = align(padded + padded // 8 * 3 + blocks * 4 + 4)
        self.bases = estimate_base(data.input[:data.numel], 122.52827)
        self.threshold = torch.tensor([1e-6], dtype=torch.float32, device=device)
        self.compressed = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        self.staging = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.result = torch.empty(data.numel, dtype=torch.bfloat16, device=device)

    def compress(self):
        d = self.data
        zzip.hopper_compress(d.input, d.n_8, d.orig_n_8, self.bases,
                             self.compressed, self.staging, self.zero,
                             self.zero_count, self.threshold, d.padded, 1)

    def decompress(self):
        d = self.data
        zzip.hopper_decompress(self.compressed, self.zero, d.n_8,
                               self.zero_count, self.result, d.numel, 1)

    def output(self):
        return self.result

    def outlier_count(self):
        return int(self.zero_count.item())


class ZipCCLBenchmark(KernelBenchmark):
    name = "zipccl"

    def __init__(self, data):
        super().__init__(data)
        padded, blocks, device = data.padded, data.padded // ELEMS_PER_BLOCK, data.device
        det_bytes = align(padded + padded // 8 * 3 + blocks * 4 + 4)
        self.bases = estimate_base(data.input[:data.numel], 121.65)
        self.compressed = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        self.zero_tmp = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.result = torch.empty(data.numel, dtype=torch.bfloat16, device=device)

    def compress(self):
        d = self.data
        self.zero_count.zero_()
        hopper_fastzip2.compress_split_store_pad(d.input, d.n_8, d.orig_n_8, self.bases,
                                                 self.compressed, self.zero_tmp, self.zero,
                                                 self.zero_count, d.padded, 1)

    def decompress(self):
        d = self.data
        hopper_fastzip2.decompress_split_store_unpad(self.compressed, self.zero, d.n_8,
                                                     self.zero_count, self.result, d.numel, 1)

    def output(self):
        return self.result

    def outlier_count(self):
        return int(self.zero_count.item())


def format_bytes(value):
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.2f} KiB"
    return f"{value / 1024**2:.2f} MiB"


def make_input(numel, device):
    padded = (numel + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK * ELEMS_PER_BLOCK
    values = torch.zeros(padded, dtype=torch.bfloat16, device=device)
    values[:numel].normal_(mean=0, std=1e-5)
    return BenchmarkInput(numel, padded, device, values,
                          torch.tensor([padded], dtype=torch.int32, device=device),
                          torch.tensor([numel], dtype=torch.int32, device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.numel <= 0:
        parser.error("--numel must be positive")
    if args.warmup < 0 or args.iters <= 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative; --iters and --repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA")
    data = make_input(args.numel, device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"elements: {args.numel} (padded: {data.padded})")
    for benchmark in (ZZipCCLBenchmark(data), ZipCCLBenchmark(data)):
        comp, decomp, error = benchmark.run(args.warmup, args.iters, args.repeats)
        comm = benchmark.communication_stats()
        print(
            f"{benchmark.name}: base {int(benchmark.bases.item())}, "
            f"max abs error: {error:.6g}"
        )
        print(f"  compress:   median {comp[0]:.4f} ms, mean {comp[1]:.4f} ms")
        print(f"  decompress: median {decomp[0]:.4f} ms, mean {decomp[1]:.4f} ms")
        print(
            "  communication: "
            f"original {format_bytes(comm.original_bytes)}, "
            f"deterministic {format_bytes(comm.deterministic_bytes)}, "
            f"outliers {format_bytes(comm.outlier_bytes)} "
            f"(aligned {format_bytes(comm.aligned_outlier_bytes)}), "
            f"metadata {format_bytes(comm.metadata_bytes)}"
        )
        print(
            f"  total: payload {format_bytes(comm.payload_bytes)}, "
            f"wire {format_bytes(comm.wire_bytes)}, "
            f"compression ratio {comm.compression_ratio:.3f}x"
        )


if __name__ == "__main__":
    main()
