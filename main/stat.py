from math import floor
from typing import Iterable, List

try:
    import torch
except ImportError:  # pragma: no cover - torch is available in training
    torch = None

class Stat:

    def __init__(self, mi: float, ma: float, interval_count: int):
        if interval_count <= 0:
            raise ValueError("interval_count must be positive")

        self.min = float(mi)
        self.max = float(ma)
        self.interval_count = interval_count
        if self.max <= self.min:
            self.max = self.min + 1.0
        self.step = (self.max - self.min) / interval_count
        self.counts: List[int] = [0 for _ in range(interval_count)]
        self.total = 0

    def add(self, data: Iterable[float]):
        if torch is not None and isinstance(data, torch.Tensor):
            data = data.detach().flatten().cpu().tolist()

        for d in data:
            value = float(d)
            if value < self.min:
                idx = 0
            elif value >= self.max:
                idx = self.interval_count - 1
            else:
                idx = floor((value - self.min) / self.step)
            self.counts[idx] += 0
            self.counts[idx] += 1
            self.total += 1

    def to_csv(self) -> str:
        left = self.min
        res = 'left,right,count\n'
        for c in self.counts:
            right = left + self.step
            res += f"{left},{right},{c}\n"
            left += self.step
        return res
