#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>

void zero_compress_split_store_pad_api_8(
    __nv_bfloat16* input,
    unsigned char* output,
    unsigned char* staging,
    unsigned char* retained_output,
    unsigned char* zero_output,
    int* n_8,
    int* orig_n_8,
    int* zero_count_8,
    int* retained_count_8,
    int* bases_in,
    int padded_n_total,
    int n_works);

void zero_decompress_split_store_pad_api_8(
    unsigned char* compressed_input,
    unsigned char* retained_input,
    unsigned char* zero_input,
    int* n_8,
    int* retained_count_8,
    int* zero_count_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int n_works);

void zero_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input,
    unsigned char* retained_input,
    unsigned char* zero_input,
    int* n_8,
    int* retained_count_8,
    int* zero_count_8,
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

void compress_split_store_pad(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor orig_n_8,
    torch::Tensor bases_in,
    torch::Tensor compressed_output,
    torch::Tensor staging,
    torch::Tensor retained_output,
    torch::Tensor zero_output,
    torch::Tensor zero_count_8,
    torch::Tensor retained_count_8,
    int padded_n_total,
    int n_works)
{
    check_tensor(input, "input", torch::kBFloat16);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(orig_n_8, "orig_n_8", torch::kInt32);
    check_tensor(bases_in, "bases_in", torch::kInt32);
    check_tensor(compressed_output, "compressed_output", torch::kUInt8);
    check_tensor(staging, "staging", torch::kUInt8);
    check_tensor(retained_output, "retained_output", torch::kUInt8);
    check_tensor(zero_output, "zero_output", torch::kUInt8);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(retained_count_8, "retained_count_8", torch::kInt32);
    check_common_metadata(
        input, n_8, retained_count_8, zero_count_8,
        padded_n_total, n_works);

    TORCH_CHECK(orig_n_8.numel() >= n_works,
                "orig_n_8 must contain n_works entries");
    TORCH_CHECK(bases_in.numel() >= 1, "bases_in must not be empty");
    TORCH_CHECK(staging.numel() >= 2LL * padded_n_total,
                "staging must have at least 2 * padded_n_total bytes");
    TORCH_CHECK(retained_output.numel() >= padded_n_total,
                "retained_output must have at least padded_n_total bytes");
    TORCH_CHECK(zero_output.numel() >= padded_n_total,
                "zero_output must have at least padded_n_total bytes");
    check_same_device(orig_n_8, input, "orig_n_8");
    check_same_device(bases_in, input, "bases_in");
    check_same_device(compressed_output, input, "compressed_output");
    check_same_device(staging, input, "staging");
    check_same_device(retained_output, input, "retained_output");
    check_same_device(zero_output, input, "zero_output");

    zero_compress_split_store_pad_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        compressed_output.data_ptr<unsigned char>(),
        staging.data_ptr<unsigned char>(),
        retained_output.data_ptr<unsigned char>(),
        zero_output.data_ptr<unsigned char>(),
        n_8.data_ptr<int>(),
        orig_n_8.data_ptr<int>(),
        zero_count_8.data_ptr<int>(),
        retained_count_8.data_ptr<int>(),
        bases_in.data_ptr<int>(),
        padded_n_total,
        n_works);
}

void decompress_split_store_pad(
    torch::Tensor compressed_input,
    torch::Tensor retained_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor retained_count_8,
    torch::Tensor zero_count_8,
    torch::Tensor output,
    int padded_n_total,
    int n_works)
{
    check_tensor(compressed_input, "compressed_input", torch::kUInt8);
    check_tensor(retained_input, "retained_input", torch::kUInt8);
    check_tensor(zero_input, "zero_input", torch::kUInt8);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(retained_count_8, "retained_count_8", torch::kInt32);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(output, "output", torch::kBFloat16);
    check_common_metadata(
        compressed_input, n_8, retained_count_8, zero_count_8,
        padded_n_total, n_works);
    TORCH_CHECK(output.numel() >= padded_n_total,
                "output must have at least padded_n_total elements");
    check_same_device(retained_input, compressed_input, "retained_input");
    check_same_device(zero_input, compressed_input, "zero_input");
    check_same_device(output, compressed_input, "output");

    zero_decompress_split_store_pad_api_8(
        compressed_input.data_ptr<unsigned char>(),
        retained_input.data_ptr<unsigned char>(),
        zero_input.data_ptr<unsigned char>(),
        n_8.data_ptr<int>(),
        retained_count_8.data_ptr<int>(),
        zero_count_8.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        n_works);
}

void decompress_split_store_unpad(
    torch::Tensor compressed_input,
    torch::Tensor retained_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor retained_count_8,
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
    TORCH_CHECK(
        padded_total_64 <= std::numeric_limits<int>::max(),
        "padded element count exceeds int range");
    int padded_n_total = static_cast<int>(padded_total_64);

    check_tensor(compressed_input, "compressed_input", torch::kUInt8);
    check_tensor(retained_input, "retained_input", torch::kUInt8);
    check_tensor(zero_input, "zero_input", torch::kUInt8);
    check_tensor(n_8, "n_8", torch::kInt32);
    check_tensor(retained_count_8, "retained_count_8", torch::kInt32);
    check_tensor(zero_count_8, "zero_count_8", torch::kInt32);
    check_tensor(output, "output", torch::kBFloat16);
    check_common_metadata(
        compressed_input, n_8, retained_count_8, zero_count_8,
        padded_n_total, n_works);
    TORCH_CHECK(output.numel() >= static_cast<int64_t>(orig_ele_num) * n_works,
                "output must have at least orig_ele_num * n_works elements");
    check_same_device(retained_input, compressed_input, "retained_input");
    check_same_device(zero_input, compressed_input, "zero_input");
    check_same_device(output, compressed_input, "output");

    zero_decompress_split_store_unpad_api_8(
        compressed_input.data_ptr<unsigned char>(),
        retained_input.data_ptr<unsigned char>(),
        zero_input.data_ptr<unsigned char>(),
        n_8.data_ptr<int>(),
        retained_count_8.data_ptr<int>(),
        zero_count_8.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        orig_ele_num,
        n_works);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "compress_split_store_pad",
        &compress_split_store_pad,
        "Zero-dropping split-store compression with in-kernel padding");
    module.def(
        "decompress_split_store_pad",
        &decompress_split_store_pad,
        "Zero-dropping split-store decompression to padded output");
    module.def(
        "decompress_split_store_unpad",
        &decompress_split_store_unpad,
        "Zero-dropping split-store decompression to compact output");
}
