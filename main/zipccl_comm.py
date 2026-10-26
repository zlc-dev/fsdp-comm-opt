import torch
import torch.distributed as dist

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

        # Split-store deterministic stream size:
        # sign+mantissa + 3 bitmaps + block_offsets + best exponent base.
        det_bytes = _align(
            padded_numel + padded_numel // 8 * 3 + n_blocks * 4 + 4
        )

        device = input_flat.device
        n_8_local = torch.tensor([padded_numel], dtype=torch.int32, device=device)
        orig_n_8_local = torch.tensor([orig_numel], dtype=torch.int32, device=device)

        std = input_flat.std(unbiased=False).float()
        best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + 121.65)
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)
        
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

        count_work = dist.all_gather_into_tensor(
            zero_counts,
            zero_count_local,
            group=group,
            async_op=True,
        )

        det_work = dist.all_gather_into_tensor(
            det_recv,
            det_send,
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

        det_work.wait()
        zero_work.wait()

        _decompress()

        return None


class ZipCCLReduceScatter:
    """ZipCCL reduce-scatter implemented as compressed all-to-all + reduce.

    ``input_tensor`` is laid out as one equally sized shard per destination
    rank. Each shard is compressed independently, exchanged with its
    destination, decompressed there, and finally reduced over all source
    ranks. This is the reduce-scatter construction used by ZipCCL.

    The compressed representation is bfloat16. The local reduction is carried
    out in float32 before copying to ``output_tensor`` so that FSDP's usual
    float32 gradient reduction does not accumulate in bfloat16.
    """

    def __init__(self, min_compress_numel: int = 0):
        self.min_compress_numel = min_compress_numel

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

        # ZipCCL currently implements the reduction locally for SUM and AVG.
        # In particular, NCCL's PREMUL_SUM carries a factor that is not exposed
        # through a stable Python API, so preserve its exact semantics by using
        # the native collective for unsupported operators.
        if (
            world_size == 1
            or async_op
            or input_tensor.numel() == 0
            or input_tensor.numel() < self.min_compress_numel
            or input_tensor.numel() != output_tensor.numel() * world_size
            or not input_tensor.is_floating_point()
            or not output_tensor.is_floating_point()
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
        shard_numel = output_tensor.numel()
        padded_shard_numel = _align(shard_numel, _ELEMS_PER_BLOCK)
        padded_total_numel = padded_shard_numel * world_size
        n_blocks = padded_shard_numel // _ELEMS_PER_BLOCK

        # Every destination shard has the same deterministic stream size. The
        # outlier stream remains variable and is exchanged after its per-peer
        # sizes are known.
        det_bytes = _align(
            padded_shard_numel
            + padded_shard_numel // 8 * 3
            + n_blocks * 4
            + 4
        )
        det_split_sizes = [det_bytes] * world_size

        input_bf16 = input_tensor.contiguous().view(-1).to(torch.bfloat16)
        n_8 = torch.full(
            (world_size,),
            padded_shard_numel,
            dtype=torch.int32,
            device=device,
        )
        orig_n_8 = torch.full(
            (world_size,), shard_numel, dtype=torch.int32, device=device
        )

        std = input_bf16.std(unbiased=False).float()
        std = torch.nan_to_num(std, nan=1e-30, posinf=1e30, neginf=1e-30)
        best_i = torch.round(torch.log2(torch.clamp(std, min=1e-30)) + 121.65)
        bases_in = torch.clamp(best_i, 0, 249).to(torch.int32).reshape(1)

        zero_counts_send = torch.zeros(
            world_size, dtype=torch.int32, device=device
        )
        zero_counts_recv = torch.empty_like(zero_counts_send)
        det_send = torch.empty(
            world_size * det_bytes, dtype=torch.uint8, device=device
        )
        det_recv = torch.empty_like(det_send)
        zero_tmp = torch.empty(
            padded_total_numel, dtype=torch.uint8, device=device
        )
        zero_send = torch.empty_like(zero_tmp)

        hopper_fastzip2.compress_split_store_pad(
            input_bf16,
            n_8,
            orig_n_8,
            bases_in,
            det_send,
            zero_tmp,
            zero_send,
            zero_counts_send,
            padded_total_numel,
            world_size,
        )

        # The fixed stream does not depend on the outlier counts, so overlap it
        # with the count exchange.
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

        # Received works are ordered by source rank, which is exactly the work
        # order expected by the decompressor.
        decompressed = torch.empty(
            world_size * shard_numel, dtype=torch.bfloat16, device=device
        )
        hopper_fastzip2.decompress_split_store_unpad(
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
            reduced = source_shards.float().mean(dim=0)
        else:
            reduced = source_shards.float().sum(dim=0)
        output_tensor.copy_(reduced.view(output_tensor.shape))

        stream = torch.cuda.current_stream(device)
        for tensor in (
            input_bf16,
            n_8,
            orig_n_8,
            bases_in,
            zero_counts_send,
            zero_counts_recv,
            det_send,
            det_recv,
            zero_tmp,
            zero_send,
            zero_recv,
            decompressed,
            reduced,
        ):
            tensor.record_stream(stream)

        return None
