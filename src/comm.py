import torch
import torch.distributed as dist

from work import AggregatedWork


class QuantizedAllGather:

    def allocate(self, size, *, dtype, device):
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor,
        input_tensor,
        group,
        async_op=False,
    ):
        world_size = dist.get_world_size(group)

        input_tensor = input_tensor.contiguous()

        # 1. quantize
        max_val = input_tensor.abs().max()

        scale = torch.clamp(max_val / 127.0, min=1e-8)

        q_input = torch.clamp(
            (input_tensor / scale).round(),
            -127,
            127,
        ).to(torch.int8)

        # 2. prepare gather buffers
        flat_numel = q_input.numel()

        q_output = torch.empty(
            (world_size, flat_numel),
            dtype=torch.int8,
            device=input_tensor.device,
        )

        q_output_list = list(q_output.unbind(0))

        # scale tensor
        scale_tensor = scale.reshape(1)

        scale_output = torch.empty(
            (world_size, 1),
            dtype=scale_tensor.dtype,
            device=input_tensor.device,
        )

        scale_output_list = list(scale_output.unbind(0))

        # 3. launch async all_gather
        work_q = dist.all_gather(
            q_output_list,
            q_input.view(-1),
            group=group,
            async_op=True,
        )

        work_s = dist.all_gather(
            scale_output_list,
            scale_tensor,
            group=group,
            async_op=True,
        )

        # 4. dequant
        def _dequant():

            q = q_output.view(
                world_size,
                *input_tensor.shape
            ).float()

            scales = scale_output.view(
                world_size,
                *([1] * input_tensor.dim())
            )

            dequant = q * scales

            output_tensor.copy_(
                dequant.reshape_as(output_tensor)
            )

        # 5. sync path
        if not async_op:
            work_q.wait()
            work_s.wait()

            _dequant()

            return None

        # 6. async path
        return AggregatedWork(
            [work_q, work_s],
            postprocess_fn=_dequant,
        )