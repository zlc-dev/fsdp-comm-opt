import ctypes
import os
import time
from functools import lru_cache

import torch
import torch.distributed as dist

from zipccl import hopper_fastzip2
from main.communication_stats import CommunicationStatsCollector

_ELEMS_PER_BLOCK = 128 * 32
_NCCL_BFLOAT16 = 9
_NCCL_SUM = 0
_NCCL_IN_PROGRESS = 7


@lru_cache(maxsize=1)
def _zipccl_nccl():
    try:
        # The communicator belongs to PyTorch's already-loaded NCCL.
        lib = ctypes.CDLL("libnccl.so.2", mode=os.RTLD_NOLOAD)
        reduce_scatter_with_sigma = lib.ncclZipCCLReduceScatterWithSigma
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "Start the process with the ZipCCL libnccl.so.2 build that accepts "
            "a device sigma pointer in LD_PRELOAD."
        ) from exc

    reduce_scatter_with_sigma.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    reduce_scatter_with_sigma.restype = ctypes.c_int
    lib.ncclCommGetAsyncError.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    ]
    lib.ncclCommGetAsyncError.restype = ctypes.c_int
    lib.ncclGetErrorString.argtypes = [ctypes.c_int]
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    return lib


class NCCLZipCCLReduceScatter:
    """Enqueue BF16 SUM with a per-input PyTorch sigma estimate by default."""

    def __init__(self, sigma: float | None = None):
        self._sigma = None if sigma is None else ctypes.c_float(sigma)
        if self._sigma is not None and (
            sigma < 0 or not self._sigma.value < float("inf")
        ):
            raise ValueError("sigma must be finite and nonnegative as a C float")

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op=dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        world_size = dist.get_world_size(group)
        device = input_tensor.device
        if (
            world_size > 1
            and not async_op
            and op == dist.ReduceOp.SUM
            and input_tensor.dtype == output_tensor.dtype == torch.bfloat16
            and device.type == "cuda"
            and output_tensor.device == device
            and output_tensor.is_contiguous()
            and output_tensor.numel() > 0
            and input_tensor.numel() == output_tensor.numel() * world_size
        ):
            process_group = dist.group.WORLD if group is None else group
            backend = process_group._get_backend(device)
            if isinstance(backend, dist.ProcessGroupNCCL):
                self._enqueue(output_tensor, input_tensor, backend)
                return None

        return dist.reduce_scatter_tensor(
            output_tensor, input_tensor, group=group, op=op, async_op=async_op
        )

    def _enqueue(self, output_tensor, input_tensor, backend):
        device = input_tensor.device
        with torch.cuda.device(device):
            lib = _zipccl_nccl()
            comm = backend._comm_ptr()
            if not comm:
                backend.eager_connect_single_device(device)
                comm = backend._comm_ptr()
                if not comm:
                    raise RuntimeError("ZipCCL could not obtain the NCCL communicator")

            stream = torch.cuda.current_stream(device)
            input_tensor = input_tensor.contiguous()
            if self._sigma is None:
                std = input_tensor.std(unbiased=False).float()
                sigma = torch.nan_to_num(std, nan=1e-30, posinf=1e30, neginf=1e-30)
            else:
                sigma = torch.full(
                    (), self._sigma.value, dtype=torch.float32, device=device
                )
            input_tensor.record_stream(stream)
            output_tensor.record_stream(stream)
            args = (
                input_tensor.data_ptr(), output_tensor.data_ptr(),
                output_tensor.numel(), _NCCL_BFLOAT16, _NCCL_SUM,
                comm, stream.cuda_stream,
            )
            # Sigma is allocated on this stream and held through NCCL submission.
            result = lib.ncclZipCCLReduceScatterWithSigma(*args, sigma.data_ptr())

            if result == _NCCL_IN_PROGRESS:
                # Wait for NCCL to submit the work before FSDP records its event.
                deadline = time.monotonic() + backend.options._timeout.total_seconds()
                state = ctypes.c_int(result)
                while result == _NCCL_IN_PROGRESS:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Timed out enqueueing NCCL ZipCCL reduce-scatter")
                    result = lib.ncclCommGetAsyncError(comm, ctypes.byref(state))
                    if result == 0:
                        result = state.value
                    if result == _NCCL_IN_PROGRESS:
                        time.sleep(0.001)
            if result != 0:
                error = lib.ncclGetErrorString(result).decode()
                raise RuntimeError(f"NCCL ZipCCL reduce-scatter failed ({result}): {error}")


def _align(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment

class ZipCCLAllGather:

    def __init__(
        self,
        min_compress_numel: int = 0,
        stats: CommunicationStatsCollector | None = None,
    ):
        self.min_compress_numel = min_compress_numel
        self.stats = stats

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

        if self.stats is not None:
            self.stats.record(
                outlier_count=int(zero_count_local.item()),
                element_count=orig_numel,
                actual_bytes=det_bytes + 4 + max_zero_bytes,
                original_bytes=orig_numel * input_tensor.element_size(),
            )

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
    rank. The split-store exchange follows ZipA2A.a2a_v_split_store:
    exchange the fixed payload, outlier counts, and variable outlier payload.
    Decode the received shards in source-rank order and reduce them locally.

    The compressed representation is bfloat16. The local reduction is carried
    out in float32 before copying to ``output_tensor`` so that FSDP's usual
    float32 gradient reduction does not accumulate in bfloat16.
    """

    def __init__(
        self,
        min_compress_numel: int = 0,
        stats: CommunicationStatsCollector | None = None,
    ):
        self.min_compress_numel = min_compress_numel
        self.stats = stats

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

        # Every destination shard has the same deterministic stream size. The
        # outlier stream remains variable and is exchanged after its per-peer
        # sizes are known.
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
            input,
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
            reduced = source_shards.mean(dim=0, dtype=torch.float)
        else:
            reduced = source_shards.sum(dim=0, dtype=torch.float)
        output_tensor.copy_(reduced.view(output_tensor.shape))

        stream = torch.cuda.current_stream(device)
        for tensor in (
            input,
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
