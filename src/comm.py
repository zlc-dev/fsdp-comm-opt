import torch
import torch.distributed as dist

from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather

class QuantizedAllGather(AllGather):

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group, async_op=False) -> dist.Work | None:
        return dist.all_gather(output_tensor, input_tensor, group=group, async_op=async_op)
