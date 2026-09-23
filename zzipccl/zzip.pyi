"""Type declarations for the ``zzip`` CUDA extension."""

import torch


def compress_split_store_pad(
    input: torch.Tensor,
    n_8: torch.Tensor,
    orig_n_8: torch.Tensor,
    bases_in: torch.Tensor,
    zero_exp_threshold: torch.Tensor,
    compressed_output: torch.Tensor,
    staging: torch.Tensor,
    retained_output: torch.Tensor,
    zero_output: torch.Tensor,
    zero_count_8: torch.Tensor,
    retained_count_8: torch.Tensor,
    padded_n_total: int,
    n_works: int,
) -> None:
    """Compress dense BF16 works while dropping exact-zero mantissas.

    ``n_8`` contains padded per-work element counts and ``orig_n_8`` contains
    dense input counts. ``staging`` needs at least ``2 * padded_n_total``
    bytes. ``retained_output`` stores compacted sign/mantissas for ``code !=
    7``; ``zero_output`` stores compacted exponents for ``code == 0``.
    ``zero_count_8`` and ``retained_count_8`` are overwritten by this call.
    All tensors must be contiguous CUDA tensors; byte streams use ``uint8``,
    counts use ``int32``, and value tensors use ``bfloat16``.
    """
    ...


def decompress_split_store_pad(
    compressed_input: torch.Tensor,
    retained_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    retained_count_8: torch.Tensor,
    zero_count_8: torch.Tensor,
    output: torch.Tensor,
    padded_n_total: int,
    n_works: int,
) -> None:
    """Decompress into a padded BF16 output of ``padded_n_total`` elements."""
    ...


def decompress_split_store_unpad(
    compressed_input: torch.Tensor,
    retained_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    retained_count_8: torch.Tensor,
    zero_count_8: torch.Tensor,
    output: torch.Tensor,
    orig_ele_num: int,
    n_works: int,
) -> None:
    """Decompress into ``n_works`` compact works of ``orig_ele_num`` values."""
    ...
