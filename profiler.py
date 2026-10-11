#!/usr/bin/env python3
"""Independent zzipccl/zipccl CUDA kernel benchmark."""
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
    bases: torch.Tensor


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

    def compression_stats(self):
        return ""

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
        det_bytes = align(padded // 8 * 3 + blocks * 4 * 2 + 4)
        self.threshold = torch.tensor([117], dtype=torch.int32, device=device)
        self.compressed = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        self.staging = torch.empty(2 * padded, dtype=torch.uint8, device=device)
        self.retained = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.retained_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.result = torch.empty(data.numel, dtype=torch.bfloat16, device=device)

    def compress(self):
        d = self.data
        zzip.compress_split_store_pad(d.input, d.n_8, d.orig_n_8, d.bases, self.threshold,
                                      self.compressed, self.staging, self.retained, self.zero,
                                      self.zero_count, self.retained_count, d.padded, 1)

    def decompress(self):
        d = self.data
        zzip.decompress_split_store_unpad(self.compressed, self.retained, self.zero, d.n_8,
                                          self.retained_count, self.zero_count, self.result,
                                          d.numel, 1)

    def output(self):
        return self.result

    def compression_stats(self):
        return f"retained bytes: {int(self.retained_count.item())}, zero bytes: {int(self.zero_count.item())}"


class ZipCCLBenchmark(KernelBenchmark):
    name = "zipccl"

    def __init__(self, data):
        super().__init__(data)
        padded, blocks, device = data.padded, data.padded // ELEMS_PER_BLOCK, data.device
        det_bytes = align(padded + padded // 8 * 3 + blocks * 4 + 4)
        self.compressed = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        self.zero_tmp = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero = torch.empty(padded, dtype=torch.uint8, device=device)
        self.zero_count = torch.zeros(1, dtype=torch.int32, device=device)
        self.result = torch.empty(data.numel, dtype=torch.bfloat16, device=device)

    def compress(self):
        d = self.data
        self.zero_count.zero_()
        hopper_fastzip2.compress_split_store_pad(d.input, d.n_8, d.orig_n_8, d.bases,
                                                 self.compressed, self.zero_tmp, self.zero,
                                                 self.zero_count, d.padded, 1)

    def decompress(self):
        d = self.data
        hopper_fastzip2.decompress_split_store_unpad(self.compressed, self.zero, d.n_8,
                                                     self.zero_count, self.result, d.numel, 1)

    def output(self):
        return self.result

    def compression_stats(self):
        return f"zero bytes: {int(self.zero_count.item())}"


def make_input(numel, device):
    padded = (numel + ELEMS_PER_BLOCK - 1) // ELEMS_PER_BLOCK * ELEMS_PER_BLOCK
    values = torch.zeros(padded, dtype=torch.bfloat16, device=device)
    values[:numel].normal_()
    return BenchmarkInput(numel, padded, device, values,
                          torch.tensor([padded], dtype=torch.int32, device=device),
                          torch.tensor([numel], dtype=torch.int32, device=device),
                          torch.tensor([124], dtype=torch.int32, device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1 << 20)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
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
        print(f"{benchmark.name}: {benchmark.compression_stats()}, max abs error: {error:.6g}")
        print(f"  compress:   median {comp[0]:.4f} ms, mean {comp[1]:.4f} ms")
        print(f"  decompress: median {decomp[0]:.4f} ms, mean {decomp[1]:.4f} ms")


if __name__ == "__main__":
    main()
