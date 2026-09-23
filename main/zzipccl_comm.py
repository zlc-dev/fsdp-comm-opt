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

        # Three bitmaps, retained/zero block offsets, and one best_i value.
        det_bytes = _align(
            padded_numel // 8 * 3 + n_blocks * 4 * 2 + 4
        )

        device = input_flat.device
        n_8_local = torch.tensor([padded_numel], dtype=torch.int32, device=device)
        orig_n_8_local = torch.tensor([orig_numel], dtype=torch.int32, device=device)

        # Estimate best_i from the original distribution, then deliberately
        # drop values below the lowest BF16 exponent represented by best_i.
        std = input_flat.std(unbiased=False).float()
        best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + 124.08)
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)
        zero_threshold = min(torch.exp2(bases_in.float() - 127), 1e-6)
        compressed_input = torch.where(
            input_flat.abs() < zero_threshold,
            torch.zeros((), dtype=input_flat.dtype, device=device),
            input_flat,
        )

        retained_count_local = torch.zeros(1, dtype=torch.int32, device=device)
        zero_count_local = torch.zeros(1, dtype=torch.int32, device=device)

        det_send = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        staging = torch.empty(2 * padded_numel, dtype=torch.uint8, device=device)
        retained_send_compact = torch.empty(
            padded_numel, dtype=torch.uint8, device=device
        )
        zero_send_compact = torch.empty(
            padded_numel, dtype=torch.uint8, device=device
        )

        zzip.compress_split_store_pad(
            compressed_input,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            staging,
            retained_send_compact,
            zero_send_compact,
            zero_count_local,
            retained_count_local,
            padded_numel,
            1,
        )

        det_recv = torch.empty(
            world_size * det_bytes, dtype=torch.uint8, device=device
        )
        counts_local = torch.cat((retained_count_local, zero_count_local))
        counts_all = torch.empty(
            world_size * 2, dtype=torch.int32, device=device
        )

        count_work = dist.all_gather_into_tensor(
            counts_all, counts_local, group=group, async_op=True
        )
        det_work = dist.all_gather_into_tensor(
            det_recv, det_send, group=group, async_op=True
        )

        count_work.wait()
        counts_cpu = counts_all.view(world_size, 2).cpu().tolist()
        retained_counts = [int(counts[0]) for counts in counts_cpu]
        zero_counts = [int(counts[1]) for counts in counts_cpu]
        aligned_retained_counts = [_align(count) for count in retained_counts]
        aligned_zero_counts = [_align(count) for count in zero_counts]

        rank = dist.get_rank(group)
        max_retained_bytes = max(max(aligned_retained_counts), 1)
        max_zero_bytes = max(max(aligned_zero_counts), 1)
        payload_bytes = max_retained_bytes + max_zero_bytes

        payload_send = torch.empty(payload_bytes, dtype=torch.uint8, device=device)
        local_retained_bytes = aligned_retained_counts[rank]
        local_zero_bytes = aligned_zero_counts[rank]
        if local_retained_bytes > 0:
            payload_send[:local_retained_bytes].copy_(
                retained_send_compact[:local_retained_bytes]
            )
        if local_zero_bytes > 0:
            payload_send[
                max_retained_bytes:max_retained_bytes + local_zero_bytes
            ].copy_(zero_send_compact[:local_zero_bytes])

        payload_recv = torch.empty(
            (world_size, payload_bytes), dtype=torch.uint8, device=device
        )
        payload_work = dist.all_gather_into_tensor(
            payload_recv, payload_send, group=group, async_op=True
        )

        n_8_all = torch.full(
            (world_size,), padded_numel, dtype=torch.int32, device=device
        )
        counts_matrix = counts_all.view(world_size, 2)
        retained_counts_tensor = counts_matrix[:, 0].contiguous()
        zero_counts_tensor = counts_matrix[:, 1].contiguous()

        det_work.wait()
        payload_work.wait()

        total_retained_bytes = sum(aligned_retained_counts)
        total_zero_bytes = sum(aligned_zero_counts)
        retained_input = torch.empty(
            max(total_retained_bytes, 1), dtype=torch.uint8, device=device
        )
        zero_input = torch.empty(
            max(total_zero_bytes, 1), dtype=torch.uint8, device=device
        )

        retained_offset = 0
        zero_offset = 0
        for peer, (retained_bytes, zero_bytes) in enumerate(
            zip(aligned_retained_counts, aligned_zero_counts)
        ):
            if retained_bytes > 0:
                retained_input[
                    retained_offset:retained_offset + retained_bytes
                ].copy_(payload_recv[peer, :retained_bytes])
                retained_offset += retained_bytes
            if zero_bytes > 0:
                zero_input[zero_offset:zero_offset + zero_bytes].copy_(
                    payload_recv[
                        peer,
                        max_retained_bytes:max_retained_bytes + zero_bytes,
                    ]
                )
                zero_offset += zero_bytes

        zzip.decompress_split_store_unpad(
            det_recv,
            retained_input,
            zero_input,
            n_8_all,
            retained_counts_tensor,
            zero_counts_tensor,
            output_flat,
            orig_numel,
            world_size,
        )

        stream = torch.cuda.current_stream(device)
        for tensor in (
            input_flat,
            compressed_input,
            n_8_local,
            orig_n_8_local,
            bases_in,
            det_send,
            staging,
            retained_send_compact,
            zero_send_compact,
            retained_count_local,
            zero_count_local,
            counts_local,
            det_recv,
            counts_all,
            payload_send,
            payload_recv,
            n_8_all,
            retained_counts_tensor,
            zero_counts_tensor,
            retained_input,
            zero_input,
        ):
            tensor.record_stream(stream)

        return None
