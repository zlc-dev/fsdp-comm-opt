import math
import torch

ISQRT2 = 1 / math.sqrt(2)


def adaptive_window_base_tensor(
    std: torch.Tensor,
    zero_threshold: float,
    base: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Select a window base from finite, integer-valued ``base`` entries."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    zt = abs(zero_threshold)
    if zt == 0.0:
        return base

    log2_zt = math.log2(zt)
    lo = math.floor(log2_zt)
    hi = math.ceil(log2_zt)

    if lo == hi:
        return torch.clamp_min(base, hi)

    k = ISQRT2 / std
    start_hi = 2.0 ** hi
    end_lo = 2.0 ** (lo + window_size)
    end_hi = 2.0 * end_lo

    # lo < log2(zt) < hi and window_size > 0, so evaluate the four
    # unique cover terms in one batched erf call.
    erf_zt, erf_end_lo, erf_start_hi, erf_end_hi = torch.erf(
        torch.stack(
            (zt * k, end_lo * k, start_hi * k, end_hi * k),
            dim=0,
        )
    ).unbind(0)
    cover_lo = erf_end_lo
    cover_hi = erf_zt + erf_end_hi - erf_start_hi
    # hi == lo + 1 here. Subtracting the predicate avoids materializing
    # two full_like candidate tensors.
    candidate = hi - (cover_lo > cover_hi).to(base.dtype)

    return torch.maximum(base, candidate)
