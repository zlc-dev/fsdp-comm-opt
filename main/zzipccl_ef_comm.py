from main.communication_stats import CommunicationStatsCollector
from main.zzipccl_comm import ZZipCCLReduceScatter


class ZZipCCLEfReduceScatter(ZZipCCLReduceScatter):
    """Compatibility entry point for ZZipCCL with error feedback enabled."""

    def __init__(
        self,
        min_compress_numel: int = 0,
        zero_threshold: float = 2**-12,
        stats: CommunicationStatsCollector | None = None,
    ):
        super().__init__(
            min_compress_numel=min_compress_numel,
            zero_threshold=zero_threshold,
            stats=stats,
            error_feedback=True,
        )
