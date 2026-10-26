"""Type declarations for the ``zzip`` CUDA extension."""

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
    padded_n_total: int,
    n_works: int,
) -> None:
    """Run Hopper split-store compression with in-kernel input padding.

    ``zero_count_8`` is reset and then overwritten with each work's outlier
    count. ``staging`` and ``zero_output`` each need at least
    ``padded_n_total`` bytes.
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


def hopper2_compress(
    input: torch.Tensor,
    n_8: torch.Tensor,
    orig_n_8: torch.Tensor,
    bases_in: torch.Tensor,
    compressed_output: torch.Tensor,
    staging: torch.Tensor,
    zero_output: torch.Tensor,
    zero_count_8: torch.Tensor,
    padded_n_total: int,
    n_works: int,
) -> None:
    """Run Hopper2 compression with warp-local code-7 slot reuse.

    ``zero_count_8`` contains only code-0 exponents that could not be paired
    with a code-7 slot in the same warp. The resulting streams must be
    decoded with ``hopper2_decompress``.
    """
    ...


def hopper2_decompress(
    compressed_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    zero_count_8: torch.Tensor,
    output: torch.Tensor,
    orig_ele_num: int,
    n_works: int,
) -> None:
    """Decode Hopper2 streams into compact BF16 works."""
    ...
