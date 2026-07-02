"""
Type stubs for hopper_fastzip2 (the compiled CUDA extension).
Pylance/Pyright will use this .pyi for type checking / autocompletion.
"""

import torch


# ---------------------------------------------------------------------------
# Basic compress / decompress
# ---------------------------------------------------------------------------

def compress_8(
    input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    bases_in: torch.Tensor,
    output: torch.Tensor,
    output_final: torch.Tensor,
    global_zero_counter_8: torch.Tensor,
    n_works: int,
) -> None: ...


def decompress_8(
    compressed_input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    output: torch.Tensor,
    n_works: int,
) -> None: ...


# ---------------------------------------------------------------------------
# Padding-aware compress / decompress
# ---------------------------------------------------------------------------

def compress_8_padded(
    input: torch.Tensor,
    bases_in: torch.Tensor,
    n_works: int,
) -> torch.Tensor: ...


def decompress_8_padded(
    compressed_input: torch.Tensor,
    orig_numel: int,
    nbytes_8: torch.Tensor,
    n_works: int,
) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Decompress with compacted (un-padded) output
# ---------------------------------------------------------------------------

def decompress_8_unpad(
    compressed_input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    output: torch.Tensor,
    orig_ele_num: int,
    n_works: int,
) -> None: ...


# ---------------------------------------------------------------------------
# Split-store compress / decompress
# ---------------------------------------------------------------------------

def compress_split_store(
    input: torch.Tensor,
    n_8: torch.Tensor,
    bases_in: torch.Tensor,
    output: torch.Tensor,
    output1: torch.Tensor,
    output_final: torch.Tensor,
    global_zero_counter_8: torch.Tensor,
    n_works: int,
) -> None: ...


def decompress_split_store(
    compressed_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    output: torch.Tensor,
    n_works: int,
) -> None: ...


def decompress_split_store_unpad(
    compressed_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    output: torch.Tensor,
    orig_ele_num: int,
    n_works: int,
) -> None: ...


def decompress_split_store_unpad_v(
    compressed_input: torch.Tensor,
    zero_input: torch.Tensor,
    n_8: torch.Tensor,
    nbytes_8: torch.Tensor,
    out_off: torch.Tensor,
    orig_n_8: torch.Tensor,
    output: torch.Tensor,
    padded_n_total: int,
    n_works: int,
) -> None: ...


def compress_split_store_pad(
    input: torch.Tensor,
    n_8: torch.Tensor,
    orig_n_8: torch.Tensor,
    bases_in: torch.Tensor,
    output: torch.Tensor,
    output1: torch.Tensor,
    output_final: torch.Tensor,
    global_zero_counter_8: torch.Tensor,
    padded_n_total: int,
    n_works: int,
) -> None: ...
