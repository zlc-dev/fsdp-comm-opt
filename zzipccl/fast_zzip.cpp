#include "ATen/core/TensorBody.h"
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>
#include <optional>

void hopper_compress_split_store_pad_api_8(
    __nv_bfloat16* input,
    __nv_bfloat16* residual,
    unsigned char* output,
    unsigned char* staging,
    unsigned char* zero_output,
    int* n_8,
    int* orig_n_8,
    int* zero_count_8,
    int* bases_in,
    float* zero_threshold,
    int padded_n_total,
    int residual_work,
    int n_works,
    bool write_residual);

void hopper_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* zero_count_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works);

void hopper2_compress_split_store_pad_api_8(
    __nv_bfloat16* input,
    __nv_bfloat16* residual,
    unsigned char* output,
    unsigned char* staging,
    unsigned char* zero_output,
    int* n_8,
    int* orig_n_8,
    int* zero_count_8,
    int* bases_in,
    float* zero_threshold,
    int padded_n_total,
    int residual_work,
    int n_works,
    bool write_residual);

void hopper2_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* zero_count_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works);

namespace {

constexpr int kElemsPerBlock = 128 * 32;

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    torch::ScalarType dtype)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}

void check_same_device(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name)
{
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name, " must be on the same CUDA device as input");
}

void check_common_metadata(
    const torch::Tensor& reference,
    const torch::Tensor& n_8,
    const torch::Tensor& retained_count_8,
    const torch::Tensor& zero_count_8,
    int padded_n_total,
    int n_works)
{
    TORCH_CHECK(n_works > 0, "n_works must be positive");
    TORCH_CHECK(padded_n_total >= 0, "padded_n_total must be non-negative");
    TORCH_CHECK(
        padded_n_total % kElemsPerBlock == 0,
        "padded_n_total must be a multiple of ", kElemsPerBlock);
    TORCH_CHECK(n_8.numel() >= n_works, "n_8 must contain n_works entries");
    TORCH_CHECK(
        retained_count_8.numel() >= n_works,
        "retained_count_8 must contain n_works entries");
    TORCH_CHECK(
        zero_count_8.numel() >= n_works,
        "zero_count_8 must contain n_works entries");
    check_same_device(n_8, reference, "n_8");
    check_same_device(retained_count_8, reference, "retained_count_8");
    check_same_device(zero_count_8, reference, "zero_count_8");
}

}  // namespace

void hopper_compress(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor orig_n_8,
    torch::Tensor bases_in,
    torch::Tensor compressed_output,
    torch::Tensor staging,
    torch::Tensor zero_output,
    torch::Tensor zero_count_8,
    torch::Tensor zero_threshold,
    int padded_n_total,
    int n_works,
    bool write_residual,
    std::optional<torch::Tensor> residual,
    int residual_work)
{
    check_tensor(input, "input", torch::kBFloat16);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(orig_n_8, "orig_n_8", torch::kInt32);
    check_tensor(bases_in, "bases_in", torch::kInt32);
    check_tensor(compressed_output, "compressed_output", torch::kUInt8);
    check_tensor(staging, "staging", torch::kUInt8);
    check_tensor(zero_output, "zero_output", torch::kUInt8);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(zero_threshold, "zero_threshold", torch::kFloat32);
    TORCH_CHECK(n_works > 0, "n_works must be positive");
    TORCH_CHECK(padded_n_total >= 0, "padded_n_total must be non-negative");
    TORCH_CHECK(padded_n_total % kElemsPerBlock == 0,
                "padded_n_total must be a multiple of ", kElemsPerBlock);
    TORCH_CHECK(n_8.numel() >= n_works, "n_8 must contain n_works entries");
    TORCH_CHECK(orig_n_8.numel() >= n_works,
                "orig_n_8 must contain n_works entries");
    TORCH_CHECK(zero_count_8.numel() >= n_works,
                "zero_count_8 must contain n_works entries");
    TORCH_CHECK(bases_in.numel() >= 1, "bases_in must not be empty");
    TORCH_CHECK(zero_threshold.numel() == 1,
                "zero_threshold must contain exactly one value");
    TORCH_CHECK(compressed_output.numel() >= padded_n_total,
                "compressed_output must have at least padded_n_total bytes");
    TORCH_CHECK(staging.numel() >= padded_n_total,
                "staging must have at least padded_n_total bytes");
    TORCH_CHECK(zero_output.numel() >= padded_n_total,
                "zero_output must have at least padded_n_total bytes");
    check_same_device(n_8, input, "n_8");
    check_same_device(orig_n_8, input, "orig_n_8");
    check_same_device(bases_in, input, "bases_in");
    check_same_device(compressed_output, input, "compressed_output");
    check_same_device(staging, input, "staging");
    check_same_device(zero_output, input, "zero_output");
    check_same_device(zero_count_8, input, "zero_count_8");
    check_same_device(zero_threshold, input, "zero_threshold");

    __nv_bfloat16* residual_ptr = nullptr;
    if (write_residual) {
        TORCH_CHECK(residual.has_value(),
                    "residual is required when write_residual is true");
        const torch::Tensor& residual_tensor = *residual;
        check_tensor(residual_tensor, "residual", torch::kBFloat16);
        check_same_device(residual_tensor, input, "residual");
        TORCH_CHECK(
            residual_work >= 0 && residual_work < n_works,
            "residual_work must be in [0, n_works)");
        TORCH_CHECK(
            residual_tensor.numel() * static_cast<int64_t>(n_works)
                == input.numel(),
            "residual must contain exactly one input shard");
        residual_ptr = reinterpret_cast<__nv_bfloat16*>(
            residual_tensor.data_ptr<torch::BFloat16>());
    }

    hopper_compress_split_store_pad_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        residual_ptr,
        compressed_output.data_ptr<unsigned char>(),
        staging.data_ptr<unsigned char>(),
        zero_output.data_ptr<unsigned char>(),
        n_8.data_ptr<int>(),
        orig_n_8.data_ptr<int>(),
        zero_count_8.data_ptr<int>(),
        bases_in.data_ptr<int>(),
        zero_threshold.data_ptr<float>(),
        padded_n_total,
        residual_work,
        n_works,
        write_residual);
}

void hopper_decompress(
    torch::Tensor compressed_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor zero_count_8,
    torch::Tensor output,
    int orig_ele_num,
    int n_works)
{
    TORCH_CHECK(n_works > 0, "n_works must be positive");
    TORCH_CHECK(orig_ele_num >= 0, "orig_ele_num must be non-negative");
    int64_t padded_ele_num =
        (static_cast<int64_t>(orig_ele_num) + kElemsPerBlock - 1)
        / kElemsPerBlock * kElemsPerBlock;
    int64_t padded_total_64 = padded_ele_num * n_works;
    TORCH_CHECK(padded_total_64 <= std::numeric_limits<int>::max(),
                "padded element count exceeds int range");
    int padded_n_total = static_cast<int>(padded_total_64);

    check_tensor(compressed_input, "compressed_input", torch::kUInt8);
    check_tensor(zero_input, "zero_input", torch::kUInt8);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(output, "output", torch::kBFloat16);
    TORCH_CHECK(n_8.numel() >= n_works, "n_8 must contain n_works entries");
    TORCH_CHECK(zero_count_8.numel() >= n_works,
                "zero_count_8 must contain n_works entries");
    TORCH_CHECK(output.numel() >= static_cast<int64_t>(orig_ele_num) * n_works,
                "output must have at least orig_ele_num * n_works elements");
    check_same_device(zero_input, compressed_input, "zero_input");
    check_same_device(n_8, compressed_input, "n_8");
    check_same_device(zero_count_8, compressed_input, "zero_count_8");
    check_same_device(output, compressed_input, "output");

    hopper_decompress_split_store_unpad_api_8(
        compressed_input.data_ptr<unsigned char>(),
        zero_input.data_ptr<unsigned char>(),
        zero_count_8.data_ptr<int>(),
        n_8.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        orig_ele_num,
        n_works);
}

void hopper2_compress(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor orig_n_8,
    torch::Tensor bases_in,
    torch::Tensor compressed_output,
    torch::Tensor staging,
    torch::Tensor zero_output,
    torch::Tensor zero_count_8,
    torch::Tensor zero_threshold,
    int padded_n_total,
    int n_works,
    bool write_residual,
    std::optional<torch::Tensor> residual,
    int residual_work)
{
    check_tensor(input, "input", torch::kBFloat16);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(orig_n_8, "orig_n_8", torch::kInt32);
    check_tensor(bases_in, "bases_in", torch::kInt32);
    check_tensor(compressed_output, "compressed_output", torch::kUInt8);
    check_tensor(staging, "staging", torch::kUInt8);
    check_tensor(zero_output, "zero_output", torch::kUInt8);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(zero_threshold, "zero_threshold", torch::kFloat32);
    TORCH_CHECK(n_works > 0, "n_works must be positive");
    TORCH_CHECK(padded_n_total >= 0, "padded_n_total must be non-negative");
    TORCH_CHECK(padded_n_total % kElemsPerBlock == 0,
                "padded_n_total must be a multiple of ", kElemsPerBlock);
    TORCH_CHECK(n_8.numel() >= n_works, "n_8 must contain n_works entries");
    TORCH_CHECK(orig_n_8.numel() >= n_works,
                "orig_n_8 must contain n_works entries");
    TORCH_CHECK(zero_count_8.numel() >= n_works,
                "zero_count_8 must contain n_works entries");
    TORCH_CHECK(bases_in.numel() >= 1, "bases_in must not be empty");
    TORCH_CHECK(zero_threshold.numel() == 1,
                "zero_threshold must contain exactly one value");
    TORCH_CHECK(compressed_output.numel() >= padded_n_total,
                "compressed_output must have at least padded_n_total bytes");
    TORCH_CHECK(staging.numel() >= padded_n_total,
                "staging must have at least padded_n_total bytes");
    TORCH_CHECK(zero_output.numel() >= padded_n_total,
                "zero_output must have at least padded_n_total bytes");
    check_same_device(n_8, input, "n_8");
    check_same_device(orig_n_8, input, "orig_n_8");
    check_same_device(bases_in, input, "bases_in");
    check_same_device(compressed_output, input, "compressed_output");
    check_same_device(staging, input, "staging");
    check_same_device(zero_output, input, "zero_output");
    check_same_device(zero_count_8, input, "zero_count_8");
    check_same_device(zero_threshold, input, "zero_threshold");

    __nv_bfloat16* residual_ptr = nullptr;
    if (write_residual) {
        TORCH_CHECK(residual.has_value(),
                    "residual is required when write_residual is true");
        const torch::Tensor& residual_tensor = *residual;
        check_tensor(residual_tensor, "residual", torch::kBFloat16);
        check_same_device(residual_tensor, input, "residual");
        TORCH_CHECK(
            residual_work >= 0 && residual_work < n_works,
            "residual_work must be in [0, n_works)");
        TORCH_CHECK(
            residual_tensor.numel() * static_cast<int64_t>(n_works)
                == input.numel(),
            "residual must contain exactly one input shard");
        residual_ptr = reinterpret_cast<__nv_bfloat16*>(
            residual_tensor.data_ptr<torch::BFloat16>());
    }

    hopper2_compress_split_store_pad_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        residual_ptr,
        compressed_output.data_ptr<unsigned char>(),
        staging.data_ptr<unsigned char>(),
        zero_output.data_ptr<unsigned char>(),
        n_8.data_ptr<int>(),
        orig_n_8.data_ptr<int>(),
        zero_count_8.data_ptr<int>(),
        bases_in.data_ptr<int>(),
        zero_threshold.data_ptr<float>(),
        padded_n_total,
        residual_work,
        n_works,
        write_residual);
}

void hopper2_decompress(
    torch::Tensor compressed_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor zero_count_8,
    torch::Tensor output,
    int orig_ele_num,
    int n_works)
{
    TORCH_CHECK(n_works > 0, "n_works must be positive");
    TORCH_CHECK(orig_ele_num >= 0, "orig_ele_num must be non-negative");
    int64_t padded_ele_num =
        (static_cast<int64_t>(orig_ele_num) + kElemsPerBlock - 1)
        / kElemsPerBlock * kElemsPerBlock;
    int64_t padded_total_64 = padded_ele_num * n_works;
    TORCH_CHECK(padded_total_64 <= std::numeric_limits<int>::max(),
                "padded element count exceeds int range");
    int padded_n_total = static_cast<int>(padded_total_64);

    check_tensor(compressed_input, "compressed_input", torch::kUInt8);
    check_tensor(zero_input, "zero_input", torch::kUInt8);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(output, "output", torch::kBFloat16);
    TORCH_CHECK(n_8.numel() >= n_works, "n_8 must contain n_works entries");
    TORCH_CHECK(zero_count_8.numel() >= n_works,
                "zero_count_8 must contain n_works entries");
    TORCH_CHECK(output.numel() >= static_cast<int64_t>(orig_ele_num) * n_works,
                "output must have at least orig_ele_num * n_works elements");
    check_same_device(zero_input, compressed_input, "zero_input");
    check_same_device(n_8, compressed_input, "n_8");
    check_same_device(zero_count_8, compressed_input, "zero_count_8");
    check_same_device(output, compressed_input, "output");

    hopper2_decompress_split_store_unpad_api_8(
        compressed_input.data_ptr<unsigned char>(),
        zero_input.data_ptr<unsigned char>(),
        zero_count_8.data_ptr<int>(),
        n_8.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        orig_ele_num,
        n_works);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "hopper_compress",
        &hopper_compress,
        "Hopper split-store compression with in-kernel padding",
        py::arg("input"),
        py::arg("n_8"),
        py::arg("orig_n_8"),
        py::arg("bases_in"),
        py::arg("compressed_output"),
        py::arg("staging"),
        py::arg("zero_output"),
        py::arg("zero_count_8"),
        py::arg("zero_threshold"),
        py::arg("padded_n_total"),
        py::arg("n_works"),
        py::arg("write_residual") = false,
        py::arg("residual") = py::none(),
        py::arg("residual_work") = -1);
    module.def(
        "hopper_decompress",
        &hopper_decompress,
        "Hopper split-store decompression to compact output");
    module.def(
        "hopper2_compress",
        &hopper2_compress,
        "Hopper2 split-store compression with warp-local slot reuse",
        py::arg("input"),
        py::arg("n_8"),
        py::arg("orig_n_8"),
        py::arg("bases_in"),
        py::arg("compressed_output"),
        py::arg("staging"),
        py::arg("zero_output"),
        py::arg("zero_count_8"),
        py::arg("zero_threshold"),
        py::arg("padded_n_total"),
        py::arg("n_works"),
        py::arg("write_residual") = false,
        py::arg("residual") = py::none(),
        py::arg("residual_work") = -1);
    module.def(
        "hopper2_decompress",
        &hopper2_decompress,
        "Hopper2 split-store decompression with warp-local slot reuse");
}
