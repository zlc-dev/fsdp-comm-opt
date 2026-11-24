import math

import torch
import torch.distributed as dist

from zzipccl import zzip

from main.zzipccl_window import adaptive_window_base_tensor
from main.communication_stats import CommunicationStatsCollector

_ELEMS_PER_BLOCK = 128 * 32
_EXPONENT_BIAS = 127
_WINDOW_SIZE = 6


def _align(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class ZZipCCLAllGather:
    """Gather split-store BF16 payloads and decode directly into the output."""

    def __init__(
        self,
        min_compress_numel: int = 0,
        reuse: bool = False,
        zero_threshold: float = 2**-12,
        stats: CommunicationStatsCollector | None = None,
    ):
        if not math.isfinite(zero_threshold) or zero_threshold < 0:
            raise ValueError("zero_threshold must be a finite, non-negative value")
        self.min_compress_numel = min_compress_numel
        self.reuse = reuse
        self.zero_threshold = zero_threshold
        self.stats = stats
        self._compress = zzip.hopper_compress
        self._decompress = zzip.hopper_decompress
        if reuse:
            compress = getattr(zzip, "hopper2_compress", None)
            decompress = getattr(zzip, "hopper2_decompress", None)
            if compress is not None and decompress is not None:
                self._compress = compress
                self._decompress = decompress

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ):
        world_size = dist.get_world_size(group)

        if (
            world_size == 1
            or async_op
            or input_tensor.numel() == 0
            or input_tensor.numel() < self.min_compress_numel
            or output_tensor.numel() != input_tensor.numel() * world_size
            or input_tensor.dtype != torch.bfloat16
            or output_tensor.dtype != torch.bfloat16
            or not input_tensor.is_cuda
            or output_tensor.device != input_tensor.device
            or not output_tensor.is_contiguous()
        ):
            if self.stats is not None:
                original_bytes = input_tensor.numel() * input_tensor.element_size()
                self.stats.record(
                    outlier_count=0,
                    element_count=input_tensor.numel(),
                    actual_bytes=original_bytes,
                    original_bytes=original_bytes,
                )
            return dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )

        input_flat = input_tensor.contiguous().view(-1)
        output_flat = output_tensor.view(-1)
        orig_numel = input_flat.numel()
        padded_numel = _align(orig_numel, _ELEMS_PER_BLOCK)
        n_blocks = padded_numel // _ELEMS_PER_BLOCK

        # Sign+mantissa, three bitmaps, block offsets, and one best_i value.
        det_bytes = _align(
            padded_numel + padded_numel // 8 * 3 + n_blocks * 4 + 4
        )

        device = input_flat.device
        n_8_local = torch.tensor([padded_numel], dtype=torch.int32, device=device)
        orig_n_8_local = torch.tensor([orig_numel], dtype=torch.int32, device=device)

        std = input_flat.std(unbiased=False).float()
        std = torch.nan_to_num(std, nan=1e-30, posinf=1e30, neginf=1e-30)
        std = torch.clamp(std, min=1e-30)
        best_i = torch.round(torch.log2(std) + 122.52827)
        best_i = adaptive_window_base_tensor(
            std,
            self.zero_threshold,
            best_i - _EXPONENT_BIAS,
            _WINDOW_SIZE,
        ) + _EXPONENT_BIAS
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)

        zero_count_local = torch.zeros(1, dtype=torch.int32, device=device)
        zero_threshold = torch.tensor(
            [self.zero_threshold], dtype=torch.float32, device=device
        )
        det_send = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        zero_tmp = torch.empty(padded_numel, dtype=torch.uint8, device=device)
        zero_send_compact = torch.empty(
            padded_numel, dtype=torch.uint8, device=device
        )

        self._compress(
            input_flat,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            zero_tmp,
            zero_send_compact,
            zero_count_local,
            zero_threshold,
            padded_numel,
            1,
        )

        det_recv = torch.empty(
            world_size * det_bytes, dtype=torch.uint8, device=device
        )
        zero_counts = torch.empty(world_size, dtype=torch.int32, device=device)

        count_work = dist.all_gather_into_tensor(
            zero_counts, zero_count_local, group=group, async_op=True
        )
        det_work = dist.all_gather_into_tensor(
            det_recv, det_send, group=group, async_op=True
        )

        count_work.wait()
        zero_counts_cpu = zero_counts.cpu().tolist()
        aligned_zero_counts = [_align(count) for count in zero_counts_cpu]

        rank = dist.get_rank(group)
        max_zero_bytes = max(aligned_zero_counts)

        if self.stats is not None:
            self.stats.record(
                outlier_count=zero_counts_cpu[rank],
                element_count=orig_numel,
                actual_bytes=det_bytes + 4 + max_zero_bytes,
                original_bytes=orig_numel * input_tensor.element_size(),
            )
        zero_recv = None
        if max_zero_bytes:
            zero_send_compact[zero_counts_cpu[rank]:max_zero_bytes].zero_()
            zero_recv = torch.empty(
                world_size * max_zero_bytes, dtype=torch.uint8, device=device
            )
            zero_work = dist.all_gather_into_tensor(
                zero_recv, zero_send_compact[:max_zero_bytes], group=group, async_op=True
            )

        n_8_all = torch.full(
            (world_size,), padded_numel, dtype=torch.int32, device=device
        )

        stream = torch.cuda.current_stream(device)
        det_work.wait()
        if zero_recv is None:
            zero_input = torch.empty(1, dtype=torch.uint8, device=device)
        else:
            zero_work.wait()
            if sum(aligned_zero_counts) == zero_recv.numel():
                zero_input = zero_recv
            else:
                zero_input = torch.cat([
                    row[:size]
                    for row, size in zip(
                        zero_recv.view(world_size, max_zero_bytes), aligned_zero_counts
                    )
                ])
            zero_recv.record_stream(stream)

        self._decompress(
            det_recv,
            zero_input,
            n_8_all,
            zero_counts,
            output_flat,
            orig_numel,
            world_size,
        )

        for tensor in (
            input_flat,
            output_flat,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            zero_tmp,
            zero_send_compact,
            zero_count_local,
            zero_threshold,
            det_recv,
            zero_counts,
            n_8_all,
            zero_input,
        ):
            tensor.record_stream(stream)

        return None


class ZZipCCLReduceScatter:
    """Compressed all-to-all and local reduce, with optional sender error feedback.

    Error feedback stores one BF16 residual shard per rank, for the destination
    whose group rank matches the sender's.
    """

    def __init__(
        self,
        min_compress_numel: int = 0,
        zero_threshold: float = 2**-12,
        stats: CommunicationStatsCollector | None = None,
        *,
        error_feedback: bool = False,
    ):
        if not math.isfinite(zero_threshold) or zero_threshold < 0:
            raise ValueError("zero_threshold must be a finite, non-negative value")
        self.min_compress_numel = min_compress_numel
        self.zero_threshold = zero_threshold
        self.stats = stats
        self._compress = zzip.hopper_compress
        self._decompress = zzip.hopper_decompress
        self.error_feedback = error_feedback
        self._residual: torch.Tensor | None = None
        self._residual_world_size: int | None = None

    def reset(self) -> None:
        """Discard accumulated compression error."""
        self._residual = None
        self._residual_world_size = None

    def _get_residual(self, shard_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        residual = self._residual
        if (
            residual is None
            or residual.numel() != shard_tensor.numel()
            or residual.device != shard_tensor.device
            or residual.dtype != shard_tensor.dtype
            or self._residual_world_size != world_size
        ):
            residual = torch.zeros(
                shard_tensor.numel(), dtype=shard_tensor.dtype, device=shard_tensor.device
            )
            self._residual = residual
            self._residual_world_size = world_size
        return residual

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    @staticmethod
    def _supported_op(op) -> bool:
        return op == dist.ReduceOp.SUM or op == dist.ReduceOp.AVG

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op=dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        world_size = dist.get_world_size(group)

        if (
            world_size == 1
            or async_op
            or input_tensor.numel() == 0
            or input_tensor.numel() < self.min_compress_numel
            or input_tensor.numel() != output_tensor.numel() * world_size
            or not input_tensor.dtype == torch.bfloat16
            or not output_tensor.dtype == torch.bfloat16
            or not input_tensor.is_cuda
            or not output_tensor.is_cuda
            or not self._supported_op(op)
        ):
            if self.stats is not None:
                original_bytes = input_tensor.numel() * input_tensor.element_size()
                self.stats.record(
                    outlier_count=0,
                    element_count=input_tensor.numel(),
                    actual_bytes=original_bytes,
                    original_bytes=original_bytes,
                )
            return dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )

        device = input_tensor.device
        shard_numel = output_tensor.numel()
        padded_shard_numel = _align(shard_numel, _ELEMS_PER_BLOCK)
        padded_total_numel = padded_shard_numel * world_size
        n_blocks = padded_shard_numel // _ELEMS_PER_BLOCK

        # One deterministic compressed stream is sent to each destination.
        # It has a data-independent size, while the outlier stream is variable.
        det_bytes = _align(
            padded_shard_numel
            + padded_shard_numel // 8 * 3
            + n_blocks * 4
            + 4
        )
        input = input_tensor.contiguous().view(-1)
        n_8 = torch.full(
            (world_size,),
            padded_shard_numel,
            dtype=torch.int32,
            device=device,
        )
        orig_n_8 = torch.full(
            (world_size,), shard_numel, dtype=torch.int32, device=device
        )

        std = input.std(unbiased=False).float()
        std = torch.nan_to_num(std, nan=1e-30, posinf=1e30, neginf=1e-30)
        std = torch.clamp(std, min=1e-30)
        best_i = torch.round(torch.log2(std) + 122.52827)
        best_i = adaptive_window_base_tensor(
            std,
            self.zero_threshold,
            best_i - _EXPONENT_BIAS,
            _WINDOW_SIZE,
        ) + _EXPONENT_BIAS
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)

        zero_counts_send = torch.zeros(
            world_size, dtype=torch.int32, device=device
        )
        zero_threshold = torch.tensor(
            [self.zero_threshold], dtype=torch.float32, device=device
        )
        zero_counts_recv = torch.empty_like(zero_counts_send)
        det_send = torch.empty(
            world_size * det_bytes, dtype=torch.uint8, device=device
        )
        det_recv = torch.empty_like(det_send)
        zero_staging = torch.empty(
            padded_total_numel, dtype=torch.uint8, device=device
        )
        zero_send = torch.empty_like(zero_staging)

        residual = None
        compression_options = {}
        if self.error_feedback:
            residual = self._get_residual(output_tensor, world_size)
            compression_options = {
                "write_residual": True,
                "residual": residual,
                "residual_work": dist.get_rank(group),
            }
        self._compress(
            input,
            n_8,
            orig_n_8,
            bases_in,
            det_send,
            zero_staging,
            zero_send,
            zero_counts_send,
            zero_threshold,
            padded_total_numel,
            world_size,
            **compression_options,
        )

        # Submit the fixed payload before reading the outlier counts on the CPU.
        count_work = dist.all_to_all_single(
            zero_counts_recv,
            zero_counts_send,
            group=group,
            async_op=True,
        )
        det_work = dist.all_to_all_single(
            det_recv,
            det_send,
            group=group,
            async_op=True,
        )

        count_work.wait()
        counts = torch.cat((zero_counts_send, zero_counts_recv)).cpu().tolist()
        send_counts = counts[:world_size]
        recv_counts = counts[world_size:]
        zero_send_splits = [_align(count) for count in send_counts]
        zero_recv_splits = [_align(count) for count in recv_counts]
        zero_send_numel = sum(zero_send_splits)
        zero_recv_numel = sum(zero_recv_splits)

        if self.stats is not None:
            self.stats.record(
                outlier_count=sum(send_counts),
                element_count=input_tensor.numel(),
                actual_bytes=(
                    world_size * det_bytes
                    + world_size * zero_counts_send.element_size()
                    + zero_send_numel
                ),
                original_bytes=input_tensor.numel() * input_tensor.element_size(),
            )
        zero_recv = torch.empty(
            max(zero_recv_numel, 1), dtype=torch.uint8, device=device
        )

        zero_work = dist.all_to_all_single(
            zero_recv[:zero_recv_numel],
            zero_send[:zero_send_numel],
            output_split_sizes=zero_recv_splits,
            input_split_sizes=zero_send_splits,
            group=group,
            async_op=True,
        )

        det_work.wait()
        zero_work.wait()

        decompressed = torch.empty(
            world_size * shard_numel, dtype=torch.bfloat16, device=device
        )
        self._decompress(
            det_recv,
            zero_recv,
            n_8,
            zero_counts_recv,
            decompressed,
            shard_numel,
            world_size,
        )

        source_shards = decompressed.view(world_size, shard_numel)
        if op == dist.ReduceOp.AVG:
            reduced = source_shards.mean(dim=0, dtype=torch.float)
        else:
            reduced = source_shards.sum(dim=0, dtype=torch.float)
        output_tensor.copy_(reduced.view(output_tensor.shape))

        stream = torch.cuda.current_stream(device)
        if residual is not None:
            residual.record_stream(stream)
        for tensor in (
            n_8,
            orig_n_8,
            bases_in,
            zero_counts_send,
            zero_threshold,
            zero_counts_recv,
            det_send,
            det_recv,
            zero_staging,
            zero_send,
            zero_recv,
            decompressed,
            reduced,
        ):
            tensor.record_stream(stream)

        return None
