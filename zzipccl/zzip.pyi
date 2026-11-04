"""Type declarations for the ``zzip`` CUDA extension."""

from typing import Optional
import torch

def hopper_compress(
    input: torch.Tensor,
    n_8: torch.Tensor,
    orig_n_8: torch.Tensor,
    bases_in: torch.Tensor,
    compressed_output: torch.Tensor,
    staging: torch.Tensor,
    zero_output: torch.Tensor,
    zero_count_8: torch.Tensor,
    zero_threshold: torch.Tensor,
    padded_n_total: int,
    n_works: int,
    write_residual: bool = False,
    residual: Optional[torch.Tensor] = None,
    residual_work: int = -1,
) -> None:
    """Run Hopper split-store compression with in-kernel input padding.

    ``zero_count_8`` is reset and then overwritten with each work's outlier
    count. ``staging`` and ``zero_output`` each need at least
    ``padded_n_total`` bytes. ``zero_threshold`` is a scalar CUDA float32
    tensor containing the code-7 near-zero cutoff. If ``write_residual`` is
    true, ``residual`` is added to work ``residual_work`` before compression
    and then replaced in-place by that work's code-7 compression residual.
    """
    ...


def hopper_decompress(
    compressed_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    zero_count_8: torch.Tensor,
    output: torch.Tensor,
    orig_ele_num: int,
    n_works: int,
) -> None:
    """Run Hopper split-store decompression into compact BF16 works."""
    ...
