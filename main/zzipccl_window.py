import math
import torch

ISQRT2 = 1 / math.sqrt(2)

def adaptive_window_base_tensor(
    std: torch.Tensor,
    zero_threshold: float,
    base: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    if window_size < 0:
        raise ValueError("window_size must be non-negative")

    zt = abs(zero_threshold)
    if zt == 0.0:
        return base

    log2_zt = math.log2(zt)
    lo = math.floor(log2_zt)
    hi = math.ceil(log2_zt)

    if lo == hi:
        candidate = torch.full_like(base, hi)
    else:
        k = ISQRT2 / std

        def tensor_cover(candidate_base: int) -> torch.Tensor:
            start = 2.0 ** candidate_base
            end = 2.0 ** (candidate_base + window_size)
            if zt > end:
                return torch.erf(zt * k)
            if zt > start:
                return torch.erf(end * k)
            return torch.erf(zt * k) + torch.erf(end * k) - torch.erf(start * k)

        cover_lo = tensor_cover(lo)
        cover_hi = tensor_cover(hi)
        candidate = torch.where(
            cover_lo > cover_hi,
            torch.full_like(base, lo),
            torch.full_like(base, hi),
        )

    return torch.where(base >= hi, base, candidate)
