#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>
#include <iostream>

void hopper_compress_api_8(
    __nv_bfloat16* d_vec,
    unsigned char* output,
    unsigned char* output_final,
    int* n_8,
    int* nbytes_8,
    int* global_zero_counter_8,
    int* d_bases,
    int n,
    int n_works);

void hopper_decompress_api_8(
    unsigned char* compressed_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int n,
    int n_works);

void hopper_decompress_unpad_api_8(
    unsigned char* compressed_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works);

void hopper_compress_split_store_api_8(
    __nv_bfloat16* d_vec,
    unsigned char* output,
    unsigned char* output1,
    unsigned char* output_final,
    int* n_8,
    int* global_zero_counter_8,
    int* d_bases,
    int n,
    int n_works);

void hopper_decompress_split_store_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int n,
    int n_works);

void hopper_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works);

void hopper_decompress_split_store_unpad_v_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    int* out_off,
    int* orig_n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int n_works);

void hopper_compress_split_store_pad_api_8(
    __nv_bfloat16* d_vec,
    unsigned char* output,
    unsigned char* output1,
    unsigned char* output_final,
    int* n_8,
    int* orig_n_8,
    int* global_zero_counter_8,
    int* d_bases,
    int padded_n_total,
    int n_works);

// -------------------------------------------------------------------
// Python-facing wrappers
// -------------------------------------------------------------------

static constexpr int ELEMS_PER_BLOCK = 128 * 32;  // 4096

void compress_8(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor bases_in,
    torch::Tensor output,
    torch::Tensor output_final,
    torch::Tensor global_zero_counter_8,
    int n_works)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(n_8.device().is_cuda(),   "n_8 must be a CUDA tensor");
    TORCH_CHECK(bases_in.device().is_cuda(), "bases_in must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(),   "output must be a CUDA tensor");
    TORCH_CHECK(global_zero_counter_8.device().is_cuda(),
                "global_zero_counter_8 must be a CUDA tensor");

    int n = input.numel();

    hopper_compress_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        reinterpret_cast<unsigned char*>(output.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output_final.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(global_zero_counter_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(bases_in.data_ptr<int32_t>()),
        n,
        n_works);
}

void decompress_8(
    torch::Tensor compressed_input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor output,
    int n_works)
{
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    int n = output.numel();

    hopper_decompress_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        n,
        n_works);
}

// Padding-aware compress: accepts arbitrary numel, pads internally
torch::Tensor compress_8_padded(
    torch::Tensor input,
    torch::Tensor bases_in,
    int n_works)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "input must be bfloat16");

    int64_t orig_numel = input.numel();
    int64_t padded_numel = ((orig_numel + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

    torch::Tensor padded_input = input;
    if (padded_numel != orig_numel) {
        padded_input = torch::zeros({padded_numel}, input.options());
        padded_input.slice(0, 0, orig_numel).copy_(input);
    }

    int n = (int)padded_numel;
    auto device = input.device();

    torch::Tensor n_8_t = torch::tensor({n}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor nbytes_8_t = torch::empty({n_works}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor gc = torch::zeros({n_works}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor output_buf = torch::empty({(int64_t)padded_input.nbytes() * 2},
                                             torch::dtype(torch::kUInt8).device(device));

    hopper_compress_api_8(
        reinterpret_cast<__nv_bfloat16*>(padded_input.data_ptr<torch::BFloat16>()),
        reinterpret_cast<unsigned char*>(output_buf.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output_buf.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(n_8_t.data_ptr<int32_t>()),
        reinterpret_cast<int*>(nbytes_8_t.data_ptr<int32_t>()),
        reinterpret_cast<int*>(gc.data_ptr<int32_t>()),
        reinterpret_cast<int*>(bases_in.data_ptr<int32_t>()),
        n, n_works);

    return output_buf;
}

// Padding-aware decompress: decompresses into padded output, trims to orig_numel
torch::Tensor decompress_8_padded(
    torch::Tensor compressed_input,
    int64_t orig_numel,
    torch::Tensor nbytes_8,
    int n_works)
{
    TORCH_CHECK(compressed_input.device().is_cuda(), "compressed_input must be CUDA");

    int64_t padded_numel = ((orig_numel + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;

    auto device = compressed_input.device();
    torch::Tensor output = torch::empty({padded_numel},
        torch::dtype(torch::kBFloat16).device(device));
    torch::Tensor n_8_t = torch::tensor({(int)padded_numel},
        torch::dtype(torch::kInt32).device(device));

    hopper_decompress_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8_t.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        (int)padded_numel, n_works);

    if (padded_numel != orig_numel)
        return output.slice(0, 0, orig_numel).contiguous();
    return output;
}

// Decompress directly into compacted output, skipping padding per work.
// output must have orig_ele_num * n_works elements (bf16).
// n_8 contains padded_ele_num per work; the kernel writes only orig_ele_num
// valid elements per work, so no post-decompress copy is needed.
void decompress_8_unpad(
    torch::Tensor compressed_input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor output,
    int orig_ele_num,
    int n_works)
{
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    int64_t padded_ele_num =
        ((int64_t(orig_ele_num) + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
    int padded_n_total = (int)(padded_ele_num * n_works);

    hopper_decompress_unpad_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        orig_ele_num,
        n_works);
}

// -------------------------------------------------------------------
// Split-store wrappers
// -------------------------------------------------------------------
// Drop-in compatible with the original fastzip.compress_split_store /
// decompress_split_store signatures. The deterministic stream
// (sign_and_mantissa + bitmaps + block_offsets) is written compacted to
// `output`, while the variable-size outlier exponents are written to
// `output1` and compacted into `output_final`.

void compress_split_store(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor bases_in,
    torch::Tensor output,
    torch::Tensor output1,
    torch::Tensor output_final,
    torch::Tensor global_zero_counter_8,
    int n_works)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(n_8.device().is_cuda(),   "n_8 must be a CUDA tensor");
    TORCH_CHECK(bases_in.device().is_cuda(), "bases_in must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(),   "output must be a CUDA tensor");
    TORCH_CHECK(output1.device().is_cuda(),  "output1 must be a CUDA tensor");
    TORCH_CHECK(output_final.device().is_cuda(), "output_final must be a CUDA tensor");
    TORCH_CHECK(global_zero_counter_8.device().is_cuda(),
                "global_zero_counter_8 must be a CUDA tensor");

    int n = input.numel();

    hopper_compress_split_store_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        reinterpret_cast<unsigned char*>(output.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output1.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output_final.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(global_zero_counter_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(bases_in.data_ptr<int32_t>()),
        n,
        n_works);
}

void decompress_split_store(
    torch::Tensor compressed_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor output,
    int n_works)
{
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    int n = output.numel();

    hopper_decompress_split_store_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(zero_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        n,
        n_works);
}

// Split-store decompress writing compacted (un-padded) output: each work
// holds `n_8[i]` padded elements but only `orig_ele_num` valid elements,
// so output must have orig_ele_num * n_works bf16 elements.
void decompress_split_store_unpad(
    torch::Tensor compressed_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor output,
    int orig_ele_num,
    int n_works)
{
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    int64_t padded_ele_num =
        ((int64_t(orig_ele_num) + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK) * ELEMS_PER_BLOCK;
    int padded_n_total = (int)(padded_ele_num * n_works);

    hopper_decompress_split_store_unpad_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(zero_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        orig_ele_num,
        n_works);
}

// Split-store decompress with VARIABLE per-work valid element count.
// `n_8`     : per-work padded element count (multiple of ELEMS_PER_BLOCK).
// `nbytes_8`: per-work outlier zero-counter.
// `out_off` : per-work output start offset (in elements).
// `orig_n_8`: per-work valid element count to write.
// `output`  : flat bf16 buffer holding the concatenated valid elements.
void decompress_split_store_unpad_v(
    torch::Tensor compressed_input,
    torch::Tensor zero_input,
    torch::Tensor n_8,
    torch::Tensor nbytes_8,
    torch::Tensor out_off,
    torch::Tensor orig_n_8,
    torch::Tensor output,
    int padded_n_total,
    int n_works)
{
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    hopper_decompress_split_store_unpad_v_api_8(
        reinterpret_cast<unsigned char*>(compressed_input.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(zero_input.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(nbytes_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(out_off.data_ptr<int32_t>()),
        reinterpret_cast<int*>(orig_n_8.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        padded_n_total,
        n_works);
}

// Pad-aware split-store compress: reads a DENSE input (works tightly packed)
// and produces the padded compressed layout, padding each work to a multiple
// of ELEMS_PER_BLOCK inside the kernel.
//   n_8     : per-work PADDED element count (multiple of ELEMS_PER_BLOCK).
//   orig_n_8: per-work dense (original) element count.
void compress_split_store_pad(
    torch::Tensor input,
    torch::Tensor n_8,
    torch::Tensor orig_n_8,
    torch::Tensor bases_in,
    torch::Tensor output,
    torch::Tensor output1,
    torch::Tensor output_final,
    torch::Tensor global_zero_counter_8,
    int padded_n_total,
    int n_works)
{
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(n_8.device().is_cuda(),   "n_8 must be a CUDA tensor");
    TORCH_CHECK(orig_n_8.device().is_cuda(), "orig_n_8 must be a CUDA tensor");
    TORCH_CHECK(bases_in.device().is_cuda(), "bases_in must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(),   "output must be a CUDA tensor");

    hopper_compress_split_store_pad_api_8(
        reinterpret_cast<__nv_bfloat16*>(input.data_ptr<torch::BFloat16>()),
        reinterpret_cast<unsigned char*>(output.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output1.data_ptr<uint8_t>()),
        reinterpret_cast<unsigned char*>(output_final.data_ptr<uint8_t>()),
        reinterpret_cast<int*>(n_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(orig_n_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(global_zero_counter_8.data_ptr<int32_t>()),
        reinterpret_cast<int*>(bases_in.data_ptr<int32_t>()),
        padded_n_total,
        n_works);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_8", &compress_8, "Hopper-optimized compress_8");
    m.def("decompress_8", &decompress_8, "Hopper-optimized decompress_8");
    m.def("compress_8_padded", &compress_8_padded,
          "Hopper compress_8 with automatic padding for irregular sizes");
    m.def("decompress_8_padded", &decompress_8_padded,
          "Hopper decompress_8 with automatic padding/trim for irregular sizes");
    m.def("decompress_8_unpad", &decompress_8_unpad,
          "Hopper decompress_8 writing compacted output (no post-decompress copy)");
    m.def("compress_split_store", &compress_split_store,
          "Hopper split-store compress (deterministic stream + separate outlier stream)");
    m.def("decompress_split_store", &decompress_split_store,
          "Hopper split-store decompress (deterministic + outlier streams)");
    m.def("decompress_split_store_unpad", &decompress_split_store_unpad,
          "Hopper split-store decompress writing compacted (un-padded) output");
    m.def("decompress_split_store_unpad_v", &decompress_split_store_unpad_v,
          "Hopper split-store decompress, variable per-work valid element count");
    m.def("compress_split_store_pad", &compress_split_store_pad,
          "Hopper split-store compress with in-kernel per-work padding (dense input)");
}
