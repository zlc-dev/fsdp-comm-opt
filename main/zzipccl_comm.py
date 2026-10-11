import torch
import torch.distributed as dist

from zzipccl import zzip

_ELEMS_PER_BLOCK = 128 * 32


def _align(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class ZZipCCLAllGather:

    def __init__(self, min_compress_numel: int = 0):
        self.min_compress_numel = min_compress_numel

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
            or input_tensor.numel() < self.min_compress_numel
            or input_tensor.dtype != torch.bfloat16
            or output_tensor.dtype != torch.bfloat16
            or not input_tensor.is_cuda
            or not output_tensor.is_cuda
        ):
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
        best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + 124.08)
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)

        zero_count_local = torch.zeros(1, dtype=torch.int32, device=device)

        det_send = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        zero_tmp = torch.empty(padded_numel, dtype=torch.uint8, device=device)
        zero_send_compact = torch.empty(
            padded_numel, dtype=torch.uint8, device=device
        )

        zzip.hopper_compress(
            input_flat,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            zero_tmp,
            zero_send_compact,
            zero_count_local,
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
        aligned_zero_counts = [_align(int(count)) for count in zero_counts_cpu]

        rank = dist.get_rank(group)
        max_zero_bytes = max(max(aligned_zero_counts), 1)
        zero_send = torch.empty(max_zero_bytes, dtype=torch.uint8, device=device)
        local_zero_bytes = aligned_zero_counts[rank]
        if local_zero_bytes > 0:
            zero_send[:local_zero_bytes].copy_(
                zero_send_compact[:local_zero_bytes]
            )

        zero_recv = torch.empty(
            (world_size, max_zero_bytes), dtype=torch.uint8, device=device
        )
        zero_work = dist.all_gather_into_tensor(
            zero_recv, zero_send, group=group, async_op=True
        )

        n_8_all = torch.full(
            (world_size,), padded_numel, dtype=torch.int32, device=device
        )

        det_work.wait()
        zero_work.wait()

        total_zero_bytes = sum(aligned_zero_counts)
        zero_input = torch.empty(
            max(total_zero_bytes, 1), dtype=torch.uint8, device=device
        )

        zero_offset = 0
        for peer, zero_bytes in enumerate(aligned_zero_counts):
            if zero_bytes > 0:
                zero_input[zero_offset:zero_offset + zero_bytes].copy_(
                    zero_recv[peer, :zero_bytes]
                )
                zero_offset += zero_bytes

        zzip.hopper_decompress(
            det_recv,
            zero_input,
            n_8_all,
            zero_counts,
            output_flat,
            orig_numel,
            world_size,
        )

        stream = torch.cuda.current_stream(device)
        for tensor in (
            input_flat,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            zero_tmp,
            zero_send_compact,
            zero_count_local,
            det_recv,
            zero_counts,
            zero_send,
            zero_recv,
            n_8_all,
            zero_input,
        ):
            tensor.record_stream(stream)

        return None
