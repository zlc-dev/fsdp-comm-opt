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

        # 2. prepare gather buffers - merge scale and data into single byte tensor
        scale_bytes = scale.to(torch.float32).view(torch.uint8)
        q_input_bytes = q_input.view(torch.uint8).view(-1)

        combined_size = scale_bytes.numel() + q_input_bytes.numel()
        combined_input = torch.empty(
            combined_size,
            dtype=torch.uint8,
            device=input_tensor.device,
        )
        combined_input[:scale_bytes.numel()].copy_(scale_bytes)
        combined_input[scale_bytes.numel():].copy_(q_input_bytes)

        combined_output = torch.empty(
            (world_size, combined_size),
            dtype=torch.uint8,
            device=input_tensor.device,
        )

        combined_output_list = list(combined_output.unbind(0))

        # 3. launch single async all_gather for both scale and data
        work_combined = dist.all_gather(
            combined_output_list,
            combined_input,
            group=group,
            async_op=True,
        )
        assert isinstance(work_combined, dist.Work)

        # 4. dequant
        def _dequant():
            # Separate scale bytes and quantized data from combined output
            scale_nbytes = scale_bytes.numel()
            scale_bytes_output = combined_output[:, :scale_nbytes].contiguous()
            q_output_flat = combined_output[:, scale_nbytes:]

            # Reconstruct scale by reinterpreting the gathered bytes as float32
            scales = scale_bytes_output.view(torch.float32).view(
                world_size,
                *([1] * input_tensor.dim()),
            )

            # Reshape quantized data back to original shape
            q = q_output_flat.view(torch.int8).view(
                world_size,
                *input_tensor.shape
            ).float()

            dequant = q * scales

            output_tensor.copy_(
                dequant.reshape_as(output_tensor)
            )

        # 5. sync path
        if not async_op:
            work_combined.wait()

            _dequant()

            return None

        # 6. async path
        return AggregatedWork(
            [work_combined],
            postprocess_fn=_dequant,
        )
