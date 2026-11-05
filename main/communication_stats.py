from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class CommunicationStats:
    outlier_count: int = 0
    element_count: int = 0
    actual_bytes: int = 0
    original_bytes: int = 0


class CommunicationStatsCollector:
    """Accumulates local communication statistics between training logs."""

    def __init__(self) -> None:
        self._stats = CommunicationStats()

    def record(
        self,
        *,
        outlier_count: int,
        element_count: int,
        actual_bytes: int,
        original_bytes: int,
    ) -> None:
        self._stats = CommunicationStats(
            outlier_count=self._stats.outlier_count + int(outlier_count),
            element_count=self._stats.element_count + int(element_count),
            actual_bytes=self._stats.actual_bytes + int(actual_bytes),
            original_bytes=self._stats.original_bytes + int(original_bytes),
        )

    def pop(self) -> CommunicationStats:
        stats = self._stats
        self._stats = CommunicationStats()
        return stats


class NativeReduceScatter:
    """Native reduce-scatter wrapper that only adds optional accounting."""

    def __init__(self, stats: CommunicationStatsCollector) -> None:
        self.stats = stats

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
