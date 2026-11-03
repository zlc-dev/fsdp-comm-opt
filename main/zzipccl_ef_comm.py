import math

import torch
import torch.distributed as dist

from zzipccl import zzip

from main.zzipccl_window import adaptive_window_base_tensor

_ELEMS_PER_BLOCK = 128 * 32
_EXPONENT_BIAS = 127
_WINDOW_SIZE = 6


def _align(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class ZZipCCLEfReduceScatter:
    """ZZipCCL reduce-scatter with one-shard sender error feedback.

    Each rank retains BF16 compression errors only for the destination shard
    whose index equals that rank. This bounds persistent residual storage to
    ``1 / world_size`` of the local reduce-scatter input; errors for the other
    destination shards are intentionally not fed back.
    """

    def __init__(
        self,
        min_compress_numel: int = 0,
        reuse: bool = False,
        zero_threshold: float = 2**-12,
    ):
        if not math.isfinite(zero_threshold) or zero_threshold < 0:
            raise ValueError("zero_threshold must be a finite, non-negative value")
        self.min_compress_numel = min_compress_numel
        self.reuse = reuse
        self.zero_threshold = zero_threshold
        self._compress = (
            zzip.hopper2_compress if reuse else zzip.hopper_compress
        )
        self._decompress = (
            zzip.hopper2_decompress if reuse else zzip.hopper_decompress
        )
        self._residual: torch.Tensor | None = None
        self._residual_world_size: int | None = None

    def reset(self) -> None:
        """Discard accumulated compression error."""
        self._residual = None
        self._residual_world_size = None

    def _get_residual(
        self,
        shard_tensor: torch.Tensor,
        world_size: int,
    ) -> torch.Tensor:
        residual = self._residual
        if (
            residual is None
            or residual.numel() != shard_tensor.numel()
            or residual.device != shard_tensor.device
            or residual.dtype != shard_tensor.dtype
            or self._residual_world_size != world_size
        ):
            residual = torch.zeros(
                shard_tensor.numel(),
                dtype=shard_tensor.dtype,
                device=shard_tensor.device,
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
            return dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )

        device = input_tensor.device
        rank = dist.get_rank(group)
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
        det_split_sizes = [det_bytes] * world_size

        input = input_tensor.contiguous().view(-1)
        residual = self._get_residual(output_tensor, world_size)
        n_8 = torch.full(
            (world_size,),
            padded_shard_numel,
            dtype=torch.int32,
            device=device,
        )
        orig_n_8 = torch.full(
            (world_size,), shard_numel, dtype=torch.int32, device=device
        )

        # The retained residual contains only code-7 near-zero values for one
        # shard, so using the uncompensated input for window selection avoids
        # materializing a full compensated tensor.
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
            write_residual=True,
            residual=residual,
            residual_work=rank,
        )

        # The compression kernel has updated this rank's one-shard residual;
        # only received compressed streams still require decompression.
        count_work = dist.all_to_all_single(
            zero_counts_recv,
            zero_counts_send,
            group=group,
            async_op=True,
        )
        det_work = dist.all_to_all_single(
            det_recv,
            det_send,
            output_split_sizes=det_split_sizes,
            input_split_sizes=det_split_sizes,
            group=group,
            async_op=True,
        )

        count_work.wait()
        send_counts = zero_counts_send.cpu().tolist()
        recv_counts = zero_counts_recv.cpu().tolist()
        zero_send_splits = [_align(int(count)) for count in send_counts]
        zero_recv_splits = [_align(int(count)) for count in recv_counts]
        zero_recv_numel = sum(zero_recv_splits)
        zero_recv = torch.empty(
            max(zero_recv_numel, 1), dtype=torch.uint8, device=device
        )

        zero_work = dist.all_to_all_single(
            zero_recv[:zero_recv_numel],
            zero_send[:sum(zero_send_splits)],
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
            residual,
        ):
            tensor.record_stream(stream)

        return None
