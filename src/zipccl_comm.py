import torch
import torch.distributed as dist

from work import AggregatedWork

from ..zipccl import hopper_fastzip2 as _zipccl

class ZipCCLAllGather:
    """AllGather with ZipCCL lossless BF16 compression (split-store approach).

    The split-store compression separates data into:
      - Deterministic stream: sign+mantissa + bitmaps + block_offsets (fixed size per rank)
      - Outlier stream: non-compressible exponents (variable size per rank)

    All-gather flow:
      1. Compress with split-store → det_out + zero_out
      2. All-gather deterministic part (fixed size, no size exchange needed)
      3. All-gather outlier counts (1 int per rank)
      4. All-gather outlier data (pad to max)
      5. Decompress into output_tensor
    """

    ELEMS_PER_BLOCK = 4096  # BLOCK_SIZE=128 * PACK=32

    def __init__(self):
        self._zipccl = _zipccl

    @staticmethod
    def _pad_to_multiple(n: int, m: int) -> int:
        return ((n + m - 1) // m) * m

    @staticmethod
    def _align128(n: int) -> int:
        return ((n + 127) // 128) * 128

    def _compute_det_bytes(self, padded_numel: int) -> int:
        """Compute deterministic stream size for a given padded element count."""
        cnb = padded_numel // self.ELEMS_PER_BLOCK
        db = padded_numel + padded_numel // 8 * 3 + cnb * 4 + 4
        return self._align128(db)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ):
        world_size = dist.get_world_size(group)
        device = input_tensor.device
        rank = dist.get_rank(group)

        # ---- prepare input ----
        x = input_tensor.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        orig_numel = x.numel()
        padded_numel = self._pad_to_multiple(orig_numel, self.ELEMS_PER_BLOCK)
        det_bytes = self._compute_det_bytes(padded_numel)

        n_works = 1  # one work per rank
        n_8 = torch.tensor([padded_numel], dtype=torch.int32, device=device)
        bases_in = torch.zeros(256, dtype=torch.int32, device=device)

        # ---- allocate compression buffers ----
        det_out = torch.empty(det_bytes, dtype=torch.uint8, device=device)
        zero_scratch = torch.empty(padded_numel, dtype=torch.uint8, device=device)
        zero_out = torch.empty(padded_numel, dtype=torch.uint8, device=device)
        gc = torch.zeros(n_works, dtype=torch.int32, device=device)

        # ---- 1. compress ----
        self._zipccl.compress_split_store(
            x.view(-1), n_8, bases_in,
            det_out, zero_scratch, zero_out, gc, n_works,
        )

        # ---- 2. all-gather deterministic part (same size across ranks) ----
        gathered_det = torch.empty(world_size, det_bytes, dtype=torch.uint8, device=device)
        det_work = dist.all_gather_into_tensor(
            gathered_det, det_out, group=group, async_op=async_op,
        )

        # ---- 3. all-gather outlier counts (small: world_size ints) ----
        all_gc_cpu = torch.empty(world_size, dtype=torch.int32)
        gc_cpu = gc.cpu()
        gc_work = dist.all_gather_into_tensor(
            all_gc_cpu, gc_cpu, group=group, async_op=async_op,
        )

        # ---- 4/5. post-gather: exchange outliers + decompress ----
        def _finish():
            # Determine max aligned outlier size across ranks
            max_gc = all_gc_cpu.max().item()
            max_gc_aligned = self._align128(max_gc)

            # Pad local outliers to max and all-gather
            local_gc = all_gc_cpu[rank].item()
            if max_gc_aligned > 0:
                zero_padded = torch.zeros(max_gc_aligned, dtype=torch.uint8, device=device)
                if local_gc > 0:
                    zero_padded[:local_gc] = zero_out[:local_gc]
                gathered_zero = torch.empty(
                    world_size, max_gc_aligned, dtype=torch.uint8, device=device,
                )
                dist.all_gather_into_tensor(gathered_zero, zero_padded, group=group)
            else:
                gathered_zero = torch.empty(world_size, 0, dtype=torch.uint8, device=device)

            # ---- decompress ----
            det_flat = gathered_det.reshape(-1)
            zero_flat = gathered_zero.reshape(-1)

            n_8_all = torch.full((world_size,), padded_numel, dtype=torch.int32, device=device)
            nbytes_8_all = all_gc_cpu.to(device)

            out_flat = output_tensor.view(-1)
            if out_flat.dtype != torch.bfloat16:
                out_bf16 = torch.empty_like(out_flat, dtype=torch.bfloat16)
            else:
                out_bf16 = out_flat

            self._zipccl.decompress_split_store_unpad(
                det_flat, zero_flat, n_8_all, nbytes_8_all,
                out_bf16, orig_numel, world_size,
            )

            if out_flat.dtype != torch.bfloat16:
                out_flat.copy_(out_bf16)

        # ---- sync path ----
        if not async_op:
            _finish()
            return None

        # ---- async path ----
        works = []
        if det_work is not None:
            works.append(det_work)
        if gc_work is not None:
            works.append(gc_work)

        return AggregatedWork(works, postprocess_fn=_finish)


