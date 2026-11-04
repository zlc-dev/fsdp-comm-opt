#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

static constexpr int BLOCK_SIZE = 128;
static constexpr int PACK       = 32;
static constexpr int WARP_SIZE  = 32;
static constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * PACK;  // 4096
static constexpr int ALIGN_BYTES = 128;

template <typename T> using rptr = T * __restrict__;

// =========================================================
// Outlier-stream compaction kernel. Each work occupies n_8[i] bytes in the
// staging buffer and align128(zero_count_8[i]) bytes in the compact output.
// =========================================================
static constexpr int COPY_BLK = 128;
static constexpr int COPY_CHUNK = COPY_BLK * 16;  // 2048 bytes per tile

__global__ void hopper_copy_zero_kernel(
    const rptr<unsigned char> input,
    rptr<unsigned char> output,
    rptr<int> n_8,
    const rptr<int> global_zero_counter_8,
    int n_works)
{
    extern __shared__ __align__(128) uint8_t copy_smem[];

    // Work lookup once per block (thread 0 -> shared).
    __shared__ int s_i, s_real_n_block, s_nbytes, s_cnb, s_input_off, s_output_off;
    if (threadIdx.x == 0) {
        int tmp_blk = 0, input_off = 0, output_off = 0, found = -1;
        for (int i = 0; i < n_works; ++i) {
            int local_n_8 = n_8[i];
            int current_n_block = local_n_8 / ELEMS_PER_BLOCK;
            int nbytes = (global_zero_counter_8[i] + ALIGN_BYTES - 1)
                       / ALIGN_BYTES * ALIGN_BYTES;
            if ((int)blockIdx.x >= tmp_blk
                && (int)blockIdx.x < tmp_blk + current_n_block) {
                found = i;
                s_real_n_block = (int)blockIdx.x - tmp_blk;
                s_nbytes = nbytes;
                s_cnb = current_n_block;
                s_input_off = input_off;
                s_output_off = output_off;
                break;
            }
            tmp_blk    += current_n_block;
            output_off += nbytes;
            input_off  += local_n_8;
        }
        s_i = found;
    }
    __syncthreads();
    if (s_i < 0) return;
    int real_n_block = s_real_n_block;
    int nbytes = s_nbytes;
    int current_n_block = s_cnb;
    int input_off = s_input_off;
    int output_off = s_output_off;

    // Each block tile-copies nbytes using shared memory
    int tiles = (nbytes + COPY_CHUNK - 1) / COPY_CHUNK;
    int tiles_per_block = (tiles + current_n_block - 1) / current_n_block;
    int tile_start = real_n_block * tiles_per_block;
    int tile_end = min(tile_start + tiles_per_block, tiles);

    for (int t = tile_start; t < tile_end; ++t) {
        int byte_start = t * COPY_CHUNK;
        int byte_end   = min(byte_start + COPY_CHUNK, nbytes);
        int chunk_size = byte_end - byte_start;

        // Load tile to shared memory (16B per thread)
        for (int off = threadIdx.x * 16; off < chunk_size; off += COPY_BLK * 16) {
            if (off + 16 <= chunk_size) {
                *reinterpret_cast<float4*>(copy_smem + off) =
                    *reinterpret_cast<const float4*>(input + input_off + byte_start + off);
            } else {
                for (int k = off; k < chunk_size; ++k)
                    copy_smem[k] = input[input_off + byte_start + k];
            }
        }
        __syncthreads();

        // Store tile from shared memory (16B per thread)
        for (int off = threadIdx.x * 16; off < chunk_size; off += COPY_BLK * 16) {
            if (off + 16 <= chunk_size) {
                *reinterpret_cast<float4*>(output + output_off + byte_start + off) =
                    *reinterpret_cast<float4*>(copy_smem + off);
            } else {
                for (int k = off; k < chunk_size; ++k)
                    output[output_off + byte_start + k] = copy_smem[k];
            }
        }
        __syncthreads();
    }
}

template <bool WRITE_RESIDUAL>
__global__ void hopper_compress_split_store_pad_kernel(
    const rptr<__nv_bfloat16> input,
    rptr<__nv_bfloat16> residual,
    rptr<unsigned char> output,
    rptr<unsigned char> output1,
    rptr<int> n_8,
    rptr<int> orig_n_8,
    rptr<int> global_zero_counter_8,
    rptr<int> bases_in,
    rptr<float> zero_threshold,
    int residual_work,
    int n_works)
{
    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int best_i  = bases_in[0];

    // --- locate which work this block belongs to + dense input offset ---
    // Done ONCE per block by thread 0 into shared memory (with an early break),
    // instead of all 128 threads redundantly running the O(n_works) scan -- this
    // is what keeps multi-work (8/16-way) compress throughput close to 1-way.
    __shared__ int s_i, s_real_n_block, s_local_n_8, s_current_n_block;
    __shared__ int s_normal_off, s_extra_off, s_orig;
    __shared__ size_t s_input_off;
    __shared__ unsigned short s_zero_threshold_bits;
    if (tid == 0) {
        const __nv_bfloat16 threshold_bf16 =
            __float2bfloat16_ru(zero_threshold[0]);
        s_zero_threshold_bits =
            *reinterpret_cast<const unsigned short*>(&threshold_bf16) & 0x7FFFu;

        int tmp_blk = 0, normal_off = 0, extra_off = 0, found = -1;
        size_t input_off = 0;
        for (int i = 0; i < n_works; ++i) {
            int local_n_8 = n_8[i];
            int orig = orig_n_8[i];
            int cnb = local_n_8 / ELEMS_PER_BLOCK;
            if ((int)blockIdx.x >= tmp_blk && (int)blockIdx.x < tmp_blk + cnb) {
                found = i;
                s_real_n_block = (int)blockIdx.x - tmp_blk;
                s_local_n_8 = local_n_8;
                s_current_n_block = cnb;
                s_normal_off = normal_off;
                s_extra_off = extra_off;
                s_input_off = input_off;       // dense element offset (size_t)
                s_orig = orig;
                break;
            }
            int det_bytes = local_n_8 + local_n_8 / 8 * 3 + cnb * 4 + 4;
            det_bytes = (local_n_8 == 0) ? 0 : ((det_bytes + 127) / 128 * 128);
            tmp_blk    += cnb;
            normal_off += det_bytes;
            extra_off  += local_n_8;
            input_off  += (size_t)orig;
        }
        s_i = found;
    }
    __syncthreads();
    const unsigned short zero_threshold_bits = s_zero_threshold_bits;
    const float zero_quant_step = __bfloat162float(
        __ushort_as_bfloat16(zero_threshold_bits)) / 127.0f;
    int my_i = s_i;
    if (my_i < 0) return;
    int my_real_n_block = s_real_n_block;
    int my_local_n_8 = s_local_n_8;
    int my_current_n_block = s_current_n_block;
    int my_normal_off = s_normal_off;
    int my_extra_off = s_extra_off;
    int my_orig = s_orig;
    size_t my_input_off = s_input_off;

    // --- load this thread's PACK elements from dense input (zero-padded) ---
    __nv_bfloat16 __align__(16) reg_in[PACK];
    int base_in_work = my_real_n_block * ELEMS_PER_BLOCK + tid * PACK;
    bool aligned = ((my_input_off & 7u) == 0);
    #pragma unroll
    for (int off = 0; off < PACK; off += 8) {
        int pad_idx = base_in_work + off;
        if (aligned && pad_idx + 8 <= my_orig) {
            *reinterpret_cast<float4*>(reg_in + off) =
                *reinterpret_cast<const float4*>(input + my_input_off + pad_idx);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int idx = pad_idx + j;
                reg_in[off + j] = (idx < my_orig)
                    ? input[my_input_off + (size_t)idx]
                    : __ushort_as_bfloat16(0);
            }
        }
    }

    if constexpr (WRITE_RESIDUAL) {
        if (my_i == residual_work) {
            #pragma unroll
            for (int j = 0; j < PACK; ++j) {
                int elem_in_work = base_in_work + j;
                if (elem_in_work < my_orig) {
                    reg_in[j] = __float2bfloat16_rn(
                        __bfloat162float(reg_in[j])
                        + __bfloat162float(residual[elem_in_work]));
                }
            }
        }
    }

    // --- extract sign+mantissa and exponent ---
    unsigned char __align__(16) sm_local[PACK];
    unsigned char exp_local[PACK];
    unsigned short* p = reinterpret_cast<unsigned short*>(reg_in);
    #pragma unroll
    for (int j = 0; j < PACK; ++j) {
        unsigned short val = p[j];
        sm_local[j]  = static_cast<unsigned char>(((val >> 8) & 0x80) | (val & 0x7F));
        exp_local[j] = static_cast<unsigned char>((val >> 7) & 0xFF);
    }

    // --- build bitmap ---
    uint32_t b0 = 0, b1 = 0, b2 = 0;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int e = static_cast<int>(exp_local[j]);
        unsigned int code = (e >= best_i && e < best_i + 6) ? (e - best_i + 1) : 0u;
        unsigned short magnitude_bits = p[j] & 0x7FFFu;
        code = (code == 0u && magnitude_bits < zero_threshold_bits) ? 7u : code;
        if (code == 7u) {
            // Code 7 does not need the BF16 exponent. Reuse its byte as a
            // sign-magnitude uniform quantizer over [0, zero_threshold].
            float magnitude = fabsf(__bfloat162float(reg_in[j]));
            int quantized = __float2int_rn(magnitude / zero_quant_step);
            quantized = min(quantized, 127);
            sm_local[j] = static_cast<unsigned char>(
                ((p[j] >> 8) & 0x80u) | quantized);
        }
        b0 |= ((code >> 0) & 1u) << j;
        b1 |= ((code >> 1) & 1u) << j;
        b2 |= ((code >> 2) & 1u) << j;
    }

    uint32_t zero_mask = ~b0 & ~b1 & ~b2;
    uint32_t overflow_mask = zero_mask;
    unsigned active_mask = __activemask();

    // --- prefix-sum of outlier spill counts ---
    int thread_spill_count = __popc(overflow_mask);
    int val_spill = thread_spill_count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(active_mask, val_spill, offset);
        if (lane >= offset) val_spill += tmp;
    }
    int warp_spill_prefix = val_spill - thread_spill_count;

    __shared__ int warp_spill_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_base_global;
    if (lane == WARP_SIZE - 1) warp_spill_totals[warp_id] = val_spill;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        int run = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            int t = warp_spill_totals[w];
            warp_spill_totals[w] = run;
            run += t;
        }
        block_base_global = atomicAdd(global_zero_counter_8 + my_i, run);
    }
    __syncthreads();

    // --- deterministic stream: sign+mantissa ---
    unsigned char* sign_and_mantissa = output + my_normal_off;
    #pragma unroll
    for (int off = 0; off < PACK; off += 16) {
        *reinterpret_cast<float4*>(
            sign_and_mantissa + (my_real_n_block * BLOCK_SIZE + tid) * PACK + off) =
            *reinterpret_cast<float4*>(sm_local + off);
    }

    // --- deterministic stream: bitmaps ---
    uint32_t* bitmap_b0 = reinterpret_cast<uint32_t*>(sign_and_mantissa + my_local_n_8);
    uint32_t* bitmap_b1 = bitmap_b0 + my_local_n_8 / 32;
    uint32_t* bitmap_b2 = bitmap_b1 + my_local_n_8 / 32;
    int bmap_idx = my_real_n_block * BLOCK_SIZE + tid;
    bitmap_b0[bmap_idx] = b0;
    bitmap_b1[bmap_idx] = b1;
    bitmap_b2[bmap_idx] = b2;

    // --- deterministic stream: block_offsets + packed work metadata ---
    // The final uint32 stores:
    //   bits  7:0  = best_i (its exponent-window base needs only 8 bits)
    //   bits 22:8  = zero_threshold_bits (sign-cleared BF16 needs 15 bits)
    //   bits 31:23 = reserved
    int* block_offsets_out = reinterpret_cast<int*>(bitmap_b2 + my_local_n_8 / 32);
    if (warp_id == 0 && lane == 0) {
        block_offsets_out[my_real_n_block] = block_base_global;
        if (my_real_n_block == 0)
            block_offsets_out[my_current_n_block] =
                (best_i & 0xFF)
                | ((static_cast<int>(zero_threshold_bits) & 0x7FFF) << 8);
    }

    // --- outlier stream ---
    unsigned char* non_compress_exp = output1 + my_extra_off;
    int pos = block_base_global + warp_spill_totals[warp_id] + warp_spill_prefix;
    uint32_t m = overflow_mask;
    while (m) {
        int bit = __ffs(m) - 1;
        non_compress_exp[pos++] = exp_local[bit];
        m &= (m - 1);
    }

    if constexpr (WRITE_RESIDUAL) {
        // Codes 0-6 are reconstructed bit-exactly. Retain the code-7
        // quantization error for only one destination shard so persistent
        // residual storage is 1 / n_works of the dense input.
        if (my_i == residual_work) {
            uint32_t seven_mask = b0 & b1 & b2;
            int elem_in_work =
                my_real_n_block * ELEMS_PER_BLOCK + tid * PACK;
            if (elem_in_work < my_orig) {
                int valid = min(PACK, my_orig - elem_in_work);
                #pragma unroll
                for (int j = 0; j < PACK; ++j) {
                    if ((seven_mask >> j) & 1u) {
                        unsigned char quantized = sm_local[j];
                        float reconstructed =
                            static_cast<float>(quantized & 0x7Fu)
                            * zero_quant_step;
                        if ((quantized & 0x80u) && (quantized & 0x7Fu))
                            reconstructed = -reconstructed;
                        __nv_bfloat16 reconstructed_bf16 =
                            __float2bfloat16_rn(reconstructed);
                        reg_in[j] = __float2bfloat16_rn(
                            __bfloat162float(reg_in[j])
                            - __bfloat162float(reconstructed_bf16));
                    } else {
                        reg_in[j] = __ushort_as_bfloat16(0);
                    }
                }
                if (valid == PACK) {
                    #pragma unroll
                    for (int off = 0; off < PACK; off += 8) {
                        *reinterpret_cast<float4*>(
                            residual + elem_in_work + off) =
                            *reinterpret_cast<float4*>(reg_in + off);
                    }
                } else {
                    for (int j = 0; j < valid; ++j) {
                        residual[elem_in_work + j] = reg_in[j];
                    }
                }
            }
        }
    }
}

// =========================================================
// Split-store decompress-unpad kernel: same decode as above but writes
// a compacted output (skipping per-work padding), so no post-decompress
// copy is needed. Each work's valid data starts at i * orig_ele_num.
// =========================================================
__global__ void hopper_decompress_split_store_unpad_kernel(
    const rptr<unsigned char> compressed_input,
    const rptr<unsigned char> zero_input,
    const rptr<int> nbytes_8,
    const rptr<int> n_8,
    rptr<__nv_bfloat16> output,
    int n_works,
    int orig_ele_num)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    unsigned char* smem_nce = smem_raw;

    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    // Work lookup once per block (thread 0 -> shared, early break) instead of a
    // 128x-redundant O(n_works) scan that advances the det/zero stream pointers.
    __shared__ int s_i, s_real_n_block, s_local_n_8, s_current_n_block;
    __shared__ long long s_base_off, s_zero_off;
    if (tid == 0) {
        int tmp_blk = 0, found = -1;
        long long base_off = 0, zero_off = 0;
        for (int i = 0; i < n_works; ++i) {
            int ln8 = n_8[i];
            int cnb = ln8 / ELEMS_PER_BLOCK;
            if ((int)blockIdx.x >= tmp_blk && (int)blockIdx.x < tmp_blk + cnb) {
                found = i; s_real_n_block = (int)blockIdx.x - tmp_blk;
                s_local_n_8 = ln8; s_current_n_block = cnb;
                s_base_off = base_off; s_zero_off = zero_off; break;
            }
            int det_bytes = ln8 + ln8 / 8 * 3 + cnb * 4 + 4;
            det_bytes = (ln8 == 0) ? 0 : ((det_bytes + 127) / 128 * 128);
            tmp_blk += cnb; base_off += det_bytes;
            zero_off += (nbytes_8[i] + 127) / 128 * 128;
        }
        s_i = found;
    }
    __syncthreads();
    int my_i = s_i;
    if (my_i < 0) return;
    {
        int real_n_block = s_real_n_block;
        int local_n_8 = s_local_n_8;
        int current_n_block = s_current_n_block;
        const unsigned char* base_arr = compressed_input + s_base_off;
        const unsigned char* zero_arr = zero_input + s_zero_off;

            const unsigned char* sm_ptr = base_arr;
            const uint32_t* bm_b0 = reinterpret_cast<const uint32_t*>(base_arr + local_n_8);
            const uint32_t* bm_b1 = bm_b0 + local_n_8 / 32;
            const uint32_t* bm_b2 = bm_b1 + local_n_8 / 32;
            const int* block_offsets = reinterpret_cast<const int*>(bm_b2 + local_n_8 / 32);
            const unsigned char* nce_ptr = zero_arr;

            int metadata = block_offsets[current_n_block];
            int best_i = metadata & 0xFF;
            unsigned short zero_threshold_bits =
                static_cast<unsigned short>((metadata >> 8) & 0x7FFF);
            float zero_quant_step = __bfloat162float(
                __ushort_as_bfloat16(zero_threshold_bits)) / 127.0f;

            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            uint32_t zero_mask = ~lb0 & ~lb1 & ~lb2;
            int thread_zero_count = __popc(zero_mask);
            unsigned active_mask = __activemask();
            int thread_spill_count = thread_zero_count;

            int val2 = thread_spill_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_spill_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_spill_count;
            if (lane == WARP_SIZE - 1)
                warp_totals[warp_id] = val2;
            __syncthreads();

            if (warp_id == 0 && lane == 0) {
                int run = 0;
                for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
                    int t = warp_totals[w];
                    warp_totals[w] = run;
                    run += t;
                }
                block_spill_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            int nce_total = block_spill_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            int zero_rank = 0;
            __nv_bfloat16 out_reg[PACK];

            #pragma unroll
            for (int bit = 0; bit < 32; ++bit) {
                unsigned char lsm = local_sm[bit];

                unsigned int code = 0;
                code |= ((lb0 >> bit) & 1u) << 0;
                code |= ((lb1 >> bit) & 1u) << 1;
                code |= ((lb2 >> bit) & 1u) << 2;

                unsigned char exp_val;
                if (code == 0) {
                    exp_val = smem_nce[thread_exp_start + zero_rank];
                    ++zero_rank;
                } else if (code == 7u) {
                    float reconstructed =
                        static_cast<float>(lsm & 0x7Fu) * zero_quant_step;
                    if ((lsm & 0x80u) && (lsm & 0x7Fu))
                        reconstructed = -reconstructed;
                    out_reg[bit] = __float2bfloat16_rn(reconstructed);
                    continue;
                } else {
                    exp_val = static_cast<unsigned char>(best_i + (code - 1));
                }

                unsigned short raw_val = 0;
                raw_val |= (static_cast<unsigned short>(lsm & 0x7F));
                raw_val |= (static_cast<unsigned short>(exp_val) << 7);
                raw_val |= (static_cast<unsigned short>(lsm >> 7) << 15);

                out_reg[bit] = __ushort_as_bfloat16(raw_val);
            }

            int elem_in_work = real_n_block * ELEMS_PER_BLOCK + tid * PACK;
            __nv_bfloat16* out_ptr = output + (size_t)my_i * orig_ele_num + elem_in_work;

            // float4 stores need 16B (=8 bf16) alignment. elem_in_work is a
            // multiple of 8, so alignment is governed by (my_i*orig_ele_num)%8.
            bool base_aligned = (((size_t)my_i * orig_ele_num) & 7u) == 0;
            if (base_aligned && elem_in_work + PACK <= orig_ele_num) {
                #pragma unroll
                for (int off = 0; off < PACK; off += 8) {
                    *reinterpret_cast<float4*>(out_ptr + off) =
                        *reinterpret_cast<float4*>(out_reg + off);
                }
            } else if (elem_in_work < orig_ele_num) {
                int valid = min(PACK, orig_ele_num - elem_in_work);
                for (int j = 0; j < valid; ++j) {
                    out_ptr[j] = out_reg[j];
                }
            }
        }
}


void hopper_compress_split_store_pad_api_8(
    __nv_bfloat16* input, __nv_bfloat16* residual, unsigned char* output,
    unsigned char* staging, unsigned char* zero_output,
    int* n_8, int* orig_n_8, int* zero_count_8, int* bases_in,
    float* zero_threshold,
    int padded_n_total, int residual_work, int n_works, bool write_residual)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = padded_n_total / ELEMS_PER_BLOCK;
    cudaMemsetAsync(zero_count_8, 0, n_works * sizeof(int), stream);
    if (blocks == 0) return;

    if (write_residual) {
        hopper_compress_split_store_pad_kernel<true><<<
            blocks, BLOCK_SIZE, 0, stream>>>(
            input, residual, output, staging, n_8, orig_n_8, zero_count_8,
            bases_in, zero_threshold, residual_work, n_works);
    } else {
        hopper_compress_split_store_pad_kernel<false><<<
            blocks, BLOCK_SIZE, 0, stream>>>(
            input, residual, output, staging, n_8, orig_n_8, zero_count_8,
            bases_in, zero_threshold, residual_work, n_works);
    }

    size_t smem = COPY_CHUNK;
    cudaFuncSetAttribute(hopper_copy_zero_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);
    hopper_copy_zero_kernel<<<blocks, COPY_BLK, smem, stream>>>(
        staging, zero_output, n_8, zero_count_8, n_works);
}

void hopper_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input, unsigned char* zero_input,
    int* zero_count_8, int* n_8, __nv_bfloat16* output,
    int padded_n_total, int orig_ele_num, int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);
    cudaFuncSetAttribute(hopper_decompress_split_store_unpad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);
    hopper_decompress_split_store_unpad_kernel<<<blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, zero_input, zero_count_8, n_8,
        output, n_works, orig_ele_num);
}
