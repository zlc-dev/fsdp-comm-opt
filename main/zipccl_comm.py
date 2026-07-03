import torch
import torch.distributed as dist

from .work import AggregatedWork

from zipccl import hopper_fastzip2


_ELEMS_PER_BLOCK = 128 * 32


def _align(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class ZipCCLAllGather:

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

        # Split-store deterministic stream size:
        # sign+mantissa + 3 bitmaps + block_offsets + best exponent base.
        det_bytes = _align(
            padded_numel + padded_numel // 8 * 3 + n_blocks * 4 + 4
        )

        device = input_flat.device
        n_8_local = torch.tensor([padded_numel], dtype=torch.int32, device=device)
        orig_n_8_local = torch.tensor([orig_numel], dtype=torch.int32, device=device)

        std = input_flat.float().std()
        best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + 121.65)
        bases_in = torch.tensor([best_i], dtype=torch.int32, device=device)
        
        zero_count_local = torch.zeros(1, dtype=torch.int32, device=device)

        det_send = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        # The temporary outlier buffer is strided by padded_numel in the CUDA
        # kernel; output_final receives the compacted local outlier stream.
        zero_tmp = torch.empty(padded_numel, dtype=torch.uint8, device=device)
        zero_send_compact = torch.empty(padded_numel, dtype=torch.uint8, device=device)

        hopper_fastzip2.compress_split_store_pad(
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
            world_size * det_bytes,
            dtype=torch.uint8,
            device=device,
        )
        zero_counts = torch.empty(world_size, dtype=torch.int32, device=device)

        # Launch deterministic stream and outlier-count exchange immediately.
        det_work = dist.all_gather_into_tensor(
            det_recv,
            det_send,
            group=group,
            async_op=True,
        )
        count_work = dist.all_gather_into_tensor(
            zero_counts,
            zero_count_local,
            group=group,
            async_op=True,
        )

        # The second payload size depends on the exchanged outlier counts.
        count_work.wait()
        zero_counts_cpu = zero_counts.cpu().tolist()
        aligned_zero_counts = [_align(int(count)) for count in zero_counts_cpu]
        local_zero_bytes = aligned_zero_counts[dist.get_rank(group)]
        max_zero_bytes = max(max(aligned_zero_counts), 1)

        zero_send = torch.empty(max_zero_bytes, dtype=torch.uint8, device=device)
        if local_zero_bytes > 0:
            zero_send[:local_zero_bytes].copy_(
                zero_send_compact[:local_zero_bytes]
            )

        zero_recv = torch.empty(
            (world_size, max_zero_bytes),
            dtype=torch.uint8,
            device=device,
        )
        zero_work = dist.all_gather_into_tensor(
            zero_recv,
            zero_send,
            group=group,
            async_op=True,
        )

        n_8_all = torch.full(
            (world_size,),
            padded_numel,
            dtype=torch.int32,
            device=device,
        )

        def _decompress():
            total_zero_bytes = sum(aligned_zero_counts)
            if total_zero_bytes == 0:
                zero_input = torch.empty(1, dtype=torch.uint8, device=device)
            else:
                zero_input = torch.empty(
                    total_zero_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
                offset = 0
                for rank, nbytes in enumerate(aligned_zero_counts):
                    if nbytes > 0:
                        zero_input[offset:offset + nbytes].copy_(
                            zero_recv[rank, :nbytes]
                        )
                        offset += nbytes

            hopper_fastzip2.decompress_split_store_unpad(
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
                det_send,
                zero_tmp,
                zero_send_compact,
                zero_input,
                det_recv,
                zero_counts,
                zero_send,
                zero_recv,
                n_8_all,
            ):
                tensor.record_stream(stream)

        works = [det_work, zero_work]
        if not async_op:
            for work in works:
                work.wait()
            _decompress()
            return None

        return AggregatedWork(works, postprocess_fn=_decompress)

