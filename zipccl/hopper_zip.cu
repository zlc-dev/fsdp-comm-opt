/*
 * hopper_zip.cu -- SM90-optimized compress / decompress for BF16.
 *
 * Key differences from the baseline zip.cu
 * -----------------------------------------
 * 1. BLOCK_SIZE=128, PACK=32  -> 4 warps / block, 1 bitmap-word / thread.
 * 2. Cooperative shared-memory loading of non_compress_exp in decompress.
 * 3. No host-device synchronisation in the launch path (zero-overhead API).
 * 4. Irregular-size support via padding.
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include <cuda/barrier>
#include <c10/cuda/CUDAStream.h>

static constexpr int BLOCK_SIZE = 128;
static constexpr int PACK       = 32;
static constexpr int WARP_SIZE  = 32;
static constexpr int ELEMS_PER_BLOCK = BLOCK_SIZE * PACK;  // 4096

template <typename T> using rptr = T * __restrict__;

// =========================================================
// Compress kernel  (BLOCK_SIZE=128, PACK=32)
// =========================================================
__global__ void hopper_compress_kernel(
    const rptr<__nv_bfloat16> input,
    rptr<unsigned char> output,
    rptr<int> n_8,
    rptr<int> global_zero_counter_8,
    rptr<int> bases_in,
    int n_works,
    int total_n)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    unsigned char*  smem_sm   = reinterpret_cast<unsigned char*>(smem_input + ELEMS_PER_BLOCK);

    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int best_i  = bases_in[0];

    const size_t global_elem_base = (size_t)blockIdx.x * ELEMS_PER_BLOCK;

    // --- Load input to shared memory via cp.async (with bounds check) ---
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    #pragma unroll
    for (int off = 0; off < PACK; off += 8) {
        size_t elem_idx = global_elem_base + tid * PACK + off;
        if (elem_idx + 8 <= (size_t)total_n) {
            cuda::memcpy_async(
                smem_input + tid * PACK + off,
                input + elem_idx,
                cuda::aligned_size_t<16>(8 * sizeof(__nv_bfloat16)),
                pipe);
        } else {
            // Zero-fill for padding elements
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                smem_input[tid * PACK + off + j] = __ushort_as_bfloat16(0);
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();

    // --- Extract sign+mantissa and exponent ---
    unsigned char exp_local[PACK];

    #pragma unroll
    for (int off = 0; off < PACK; off += 8) {
        float4 v = *reinterpret_cast<const float4*>(smem_input + tid * PACK + off);
        unsigned short* p = reinterpret_cast<unsigned short*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short val = p[j];
            smem_sm[tid * PACK + off + j] =
                static_cast<unsigned char>(((val >> 8) & 0x80) | (val & 0x7F));
            exp_local[off + j] =
                static_cast<unsigned char>((val >> 7) & 0xFF);
        }
    }

    // --- Build bitmap (PACK=32 -> 1 uint32 per bitmap per thread) ---
    uint32_t b0 = 0, b1 = 0, b2 = 0;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int e = static_cast<int>(exp_local[j]);
        unsigned int code = (e >= best_i && e < best_i + 7) ? (e - best_i + 1) : 0u;
        b0 |= ((code >> 0) & 1u) << j;
        b1 |= ((code >> 1) & 1u) << j;
        b2 |= ((code >> 2) & 1u) << j;
    }

    // --- Prefix-sum of zero counts ---
    int thread_zero_count = __popc(~b0 & ~b1 & ~b2);

    unsigned active_mask = __activemask();
    int val = thread_zero_count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(active_mask, val, offset);
        if (lane >= offset) val += tmp;
    }
    int warp_prefix = val - thread_zero_count;

    __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_base_global;

    if (lane == WARP_SIZE - 1)
        warp_totals[warp_id] = val;
    __syncthreads();

    // --- Identify which work this block belongs to & write results ---
    // Work lookup once per block (thread 0 -> shared, early break) instead of a
    // 128x-redundant O(n_works) scan.
    __shared__ int s_i, s_real_n_block, s_local_n_8, s_current_n_block, s_normal_off;
    if (tid == 0) {
        int tmp_blk = 0, normal_off = 0, found = -1;
        for (int i = 0; i < n_works; ++i) {
            int ln8 = n_8[i];
            int cnb = ln8 / ELEMS_PER_BLOCK;
            if ((int)blockIdx.x >= tmp_blk && (int)blockIdx.x < tmp_blk + cnb) {
                found = i;
                s_real_n_block = (int)blockIdx.x - tmp_blk;
                s_local_n_8 = ln8;
                s_current_n_block = cnb;
                s_normal_off = normal_off;
                break;
            }
            tmp_blk    += cnb;
            normal_off += ln8 * 4;
        }
        s_i = found;
    }
    __syncthreads();
    int my_i = s_i;
    if (my_i < 0) return;
    int real_n_block = s_real_n_block;
    int local_n_8 = s_local_n_8;
    int current_n_block = s_current_n_block;
    int normal_off = s_normal_off;

    if (warp_id == 0 && lane == 0) {
        int run = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            int t = warp_totals[w];
            warp_totals[w] = run;
            run += t;
        }
        block_base_global = atomicAdd(global_zero_counter_8 + my_i, run);
    }
    __syncthreads();

    unsigned char* sign_and_mantissa = output + normal_off;

    // Write sign+mantissa (coalesced from smem)
    #pragma unroll
    for (int off = 0; off < PACK; off += 16) {
        *reinterpret_cast<float4*>(
            sign_and_mantissa + (real_n_block * BLOCK_SIZE + tid) * PACK + off) =
            *reinterpret_cast<float4*>(smem_sm + tid * PACK + off);
    }

    // Write bitmaps
    uint32_t* bitmap_b0 = reinterpret_cast<uint32_t*>(sign_and_mantissa + local_n_8);
    uint32_t* bitmap_b1 = bitmap_b0 + local_n_8 / 32;
    uint32_t* bitmap_b2 = bitmap_b1 + local_n_8 / 32;

    int bmap_idx = real_n_block * BLOCK_SIZE + tid;
    bitmap_b0[bmap_idx] = b0;
    bitmap_b1[bmap_idx] = b1;
    bitmap_b2[bmap_idx] = b2;

    // Write block_offsets
    int* block_offsets_out = reinterpret_cast<int*>(bitmap_b2 + local_n_8 / 32);
    if (warp_id == 0 && lane == 0) {
        block_offsets_out[real_n_block] = block_base_global;
        if (real_n_block == 0)
            block_offsets_out[current_n_block] = best_i;
    }

    // Scatter non_compress_exp
    unsigned char* non_compress_exp = reinterpret_cast<unsigned char*>(
        block_offsets_out + current_n_block + 1);

    int pos = block_base_global + warp_totals[warp_id] + warp_prefix;
    uint32_t m = ~b0 & ~b1 & ~b2;
    while (m) {
        int bit = __ffs(m) - 1;
        non_compress_exp[pos++] = exp_local[bit];
        m &= (m - 1);
    }
}

// =========================================================
// Copy kernel (shared-memory buffered, for multi-work compaction)
// Uses smem to stage data for coalesced reads and writes.
// =========================================================
static constexpr int COPY_BLK = 128;
static constexpr int COPY_CHUNK = COPY_BLK * 16;  // 2048 bytes per tile

__global__ void hopper_copy_kernel(
    const rptr<unsigned char> input,
    rptr<unsigned char> output,
    rptr<int> n_8,
    rptr<int> n_bytes_8,
    const rptr<int> global_zero_counter_8,
    int n_works)
{
    extern __shared__ __align__(128) uint8_t copy_smem[];

    // Work lookup once per block (thread 0 -> shared). Block 0 additionally
    // writes ALL works' compacted byte sizes into n_bytes_8 (single writer, no
    // race); other blocks break early once their work is found.
    __shared__ int s_i, s_real_n_block, s_nbytes, s_cnb, s_input_off, s_output_off;
    if (threadIdx.x == 0) {
        int tmp_blk = 0, input_off = 0, output_off = 0, found = -1;
        bool is0 = (blockIdx.x == 0);
        for (int i = 0; i < n_works; ++i) {
            int local_n_8 = n_8[i];
            int current_n_block = local_n_8 / ELEMS_PER_BLOCK;
            int nbytes = local_n_8 + local_n_8 / 8 * 3 + current_n_block * 4 + 4
                         + global_zero_counter_8[i];
            nbytes = (local_n_8 == 0) ? 0 : ((nbytes + 127) / 128 * 128);
            if (is0) n_bytes_8[i] = nbytes;
            if (found < 0 && (int)blockIdx.x >= tmp_blk
                          && (int)blockIdx.x < tmp_blk + current_n_block) {
                found = i;
                s_real_n_block = (int)blockIdx.x - tmp_blk;
                s_nbytes = nbytes;
                s_cnb = current_n_block;
                s_input_off = input_off;
                s_output_off = output_off;
                if (!is0) break;
            }
            tmp_blk    += current_n_block;
            output_off += nbytes;
            input_off  += local_n_8 * 4;
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

// =========================================================
// Split-store compress kernel (BLOCK_SIZE=128, PACK=32)
// -----------------------------------------------------------------
// Same exponent-window encoding as hopper_compress_kernel, but the
// output is split into two streams:
//   * `output`  : the DETERMINISTIC part (sign_and_mantissa + 3 bitmaps
//                 + block_offsets). Its size per work is known a priori
//                 (independent of the data), so it is written directly
//                 compacted (stride = align128(det_bytes)) and can be
//                 all-to-all'd immediately without waiting for the zero
//                 counter exchange.
//   * `output1` : the OUTLIER exponents (non_compress_exp), written into
//                 a per-work region of stride `local_n_8` (worst case is
//                 one outlier byte per element). A later compaction pass
//                 (hopper_copy_zero_kernel) packs it into output_final.
// =========================================================
__global__ void hopper_compress_split_store_kernel(
    const rptr<__nv_bfloat16> input,
    rptr<unsigned char> output,
    rptr<unsigned char> output1,
    rptr<int> n_8,
    rptr<int> global_zero_counter_8,
    rptr<int> bases_in,
    int n_works,
    int total_n)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __nv_bfloat16* smem_input = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    unsigned char*  smem_sm   = reinterpret_cast<unsigned char*>(smem_input + ELEMS_PER_BLOCK);

    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int best_i  = bases_in[0];

    const size_t global_elem_base = (size_t)blockIdx.x * ELEMS_PER_BLOCK;

    // --- Load input to shared memory via cp.async (with bounds check) ---
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    pipe.producer_acquire();
    #pragma unroll
    for (int off = 0; off < PACK; off += 8) {
        size_t elem_idx = global_elem_base + tid * PACK + off;
        if (elem_idx + 8 <= (size_t)total_n) {
            cuda::memcpy_async(
                smem_input + tid * PACK + off,
                input + elem_idx,
                cuda::aligned_size_t<16>(8 * sizeof(__nv_bfloat16)),
                pipe);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                smem_input[tid * PACK + off + j] = __ushort_as_bfloat16(0);
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    pipe.consumer_release();

    // --- Extract sign+mantissa and exponent ---
    unsigned char exp_local[PACK];
    #pragma unroll
    for (int off = 0; off < PACK; off += 8) {
        float4 v = *reinterpret_cast<const float4*>(smem_input + tid * PACK + off);
        unsigned short* p = reinterpret_cast<unsigned short*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            unsigned short val = p[j];
            smem_sm[tid * PACK + off + j] =
                static_cast<unsigned char>(((val >> 8) & 0x80) | (val & 0x7F));
            exp_local[off + j] =
                static_cast<unsigned char>((val >> 7) & 0xFF);
        }
    }

    // --- Build bitmap (PACK=32 -> 1 uint32 per bitmap per thread) ---
    uint32_t b0 = 0, b1 = 0, b2 = 0;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int e = static_cast<int>(exp_local[j]);
        unsigned int code = (e >= best_i && e < best_i + 7) ? (e - best_i + 1) : 0u;
        b0 |= ((code >> 0) & 1u) << j;
        b1 |= ((code >> 1) & 1u) << j;
        b2 |= ((code >> 2) & 1u) << j;
    }

    // --- Prefix-sum of zero counts ---
    int thread_zero_count = __popc(~b0 & ~b1 & ~b2);

    unsigned active_mask = __activemask();
    int val = thread_zero_count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(active_mask, val, offset);
        if (lane >= offset) val += tmp;
    }
    int warp_prefix = val - thread_zero_count;

    __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_base_global;

    if (lane == WARP_SIZE - 1)
        warp_totals[warp_id] = val;
    __syncthreads();

    // --- Identify which work this block belongs to & write results ---
    int local_n_8;
    int tmp_blk = 0;
    int normal_off = 0;   // compacted deterministic byte offset
    int extra_off  = 0;   // strided outlier-exp byte offset

    for (int i = 0; i < n_works; ++i) {
        local_n_8 = n_8[i];
        int current_n_block = local_n_8 / ELEMS_PER_BLOCK;
        int det_bytes = local_n_8 + local_n_8 / 8 * 3 + current_n_block * 4 + 4;
        det_bytes = (local_n_8 == 0) ? 0 : ((det_bytes + 127) / 128 * 128);

        if (blockIdx.x >= tmp_blk && blockIdx.x < tmp_blk + current_n_block) {
            int real_n_block = blockIdx.x - tmp_blk;

            if (warp_id == 0 && lane == 0) {
                int run = 0;
                for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
                    int t = warp_totals[w];
                    warp_totals[w] = run;
                    run += t;
                }
                block_base_global = atomicAdd(global_zero_counter_8 + i, run);
            }
            __syncthreads();

            // Deterministic stream: sign+mantissa, written compacted.
            unsigned char* sign_and_mantissa = output + normal_off;
            #pragma unroll
            for (int off = 0; off < PACK; off += 16) {
                *reinterpret_cast<float4*>(
                    sign_and_mantissa + (real_n_block * BLOCK_SIZE + tid) * PACK + off) =
                    *reinterpret_cast<float4*>(smem_sm + tid * PACK + off);
            }

            // Deterministic stream: bitmaps.
            uint32_t* bitmap_b0 = reinterpret_cast<uint32_t*>(sign_and_mantissa + local_n_8);
            uint32_t* bitmap_b1 = bitmap_b0 + local_n_8 / 32;
            uint32_t* bitmap_b2 = bitmap_b1 + local_n_8 / 32;

            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            bitmap_b0[bmap_idx] = b0;
            bitmap_b1[bmap_idx] = b1;
            bitmap_b2[bmap_idx] = b2;

            // Deterministic stream: block_offsets (per-work outlier offsets + best_i).
            int* block_offsets_out = reinterpret_cast<int*>(bitmap_b2 + local_n_8 / 32);
            if (warp_id == 0 && lane == 0) {
                block_offsets_out[real_n_block] = block_base_global;
                if (real_n_block == 0)
                    block_offsets_out[current_n_block] = best_i;
            }

            // Outlier stream: scatter non_compress_exp to the SEPARATE buffer.
            unsigned char* non_compress_exp = output1 + extra_off;
            int pos = block_base_global + warp_totals[warp_id] + warp_prefix;
            uint32_t m = ~b0 & ~b1 & ~b2;
            while (m) {
                int bit = __ffs(m) - 1;
                non_compress_exp[pos++] = exp_local[bit];
                m &= (m - 1);
            }
        }
        tmp_blk    += current_n_block;
        normal_off += det_bytes;
        extra_off  += local_n_8;
    }
}

// =========================================================
// Outlier-stream compaction kernel.
// Packs the strided per-work outlier regions of `input` (stride
// local_n_8) into `output` compacted by align128(zero_counter[i]).
// Mirrors hopper_copy_kernel's tiled shared-memory copy.
// =========================================================
__global__ void hopper_copy_zero_kernel(
    const rptr<unsigned char> input,
    rptr<unsigned char> output,
    rptr<int> n_8,
    const rptr<int> global_zero_counter_8,
    int n_works)
{
    extern __shared__ __align__(128) uint8_t copy_smem[];

    // Locate this block's work ONCE (thread 0 -> shared), early break, instead
    // of all threads scanning O(n_works) each.
    __shared__ int s_i, s_real_n_block, s_nbytes, s_cnb, s_input_off, s_output_off;
    if (threadIdx.x == 0) {
        int tmp_blk = 0, input_off = 0, output_off = 0, found = -1;
        for (int i = 0; i < n_works; ++i) {
            int local_n_8 = n_8[i];
            int current_n_block = local_n_8 / ELEMS_PER_BLOCK;
            int nbytes = (global_zero_counter_8[i] + 127) / 128 * 128;
            if ((int)blockIdx.x >= tmp_blk && (int)blockIdx.x < tmp_blk + current_n_block) {
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

    int tiles = (nbytes + COPY_CHUNK - 1) / COPY_CHUNK;
    int tiles_per_block = (tiles + current_n_block - 1) / current_n_block;
    int tile_start = real_n_block * tiles_per_block;
    int tile_end = min(tile_start + tiles_per_block, tiles);

    for (int t = tile_start; t < tile_end; ++t) {
        int byte_start = t * COPY_CHUNK;
        int byte_end   = min(byte_start + COPY_CHUNK, nbytes);
        int chunk_size = byte_end - byte_start;

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

// =========================================================
// Split-store compress (PAD) kernel.
// -----------------------------------------------------------------
// Reads a DENSE input where works are tightly packed (no gaps), but produces
// exactly the same padded compressed layout as if each work had been padded
// to a multiple of ELEMS_PER_BLOCK. This lets arbitrary hidden sizes (e.g.
// 7168) work without a host-side padded-input copy.
//   n_8[i]      : PADDED element count of work i (multiple of EPB).
//   orig_n_8[i] : true (dense) element count of work i.
// Padding elements (beyond orig) are treated as zero.
// =========================================================
__global__ void hopper_compress_split_store_pad_kernel(
    const rptr<__nv_bfloat16> input,
    rptr<unsigned char> output,
    rptr<unsigned char> output1,
    rptr<int> n_8,
    rptr<int> orig_n_8,
    rptr<int> global_zero_counter_8,
    rptr<int> bases_in,
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
    if (tid == 0) {
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
        unsigned int code = (e >= best_i && e < best_i + 7) ? (e - best_i + 1) : 0u;
        b0 |= ((code >> 0) & 1u) << j;
        b1 |= ((code >> 1) & 1u) << j;
        b2 |= ((code >> 2) & 1u) << j;
    }

    // --- prefix-sum of zero counts ---
    int thread_zero_count = __popc(~b0 & ~b1 & ~b2);
    unsigned active_mask = __activemask();
    int val = thread_zero_count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(active_mask, val, offset);
        if (lane >= offset) val += tmp;
    }
    int warp_prefix = val - thread_zero_count;

    __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_base_global;
    if (lane == WARP_SIZE - 1) warp_totals[warp_id] = val;
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        int run = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            int t = warp_totals[w]; warp_totals[w] = run; run += t;
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

    // --- deterministic stream: block_offsets ---
    int* block_offsets_out = reinterpret_cast<int*>(bitmap_b2 + my_local_n_8 / 32);
    if (warp_id == 0 && lane == 0) {
        block_offsets_out[my_real_n_block] = block_base_global;
        if (my_real_n_block == 0)
            block_offsets_out[my_current_n_block] = best_i;
    }

    // --- outlier stream ---
    unsigned char* non_compress_exp = output1 + my_extra_off;
    int pos = block_base_global + warp_totals[warp_id] + warp_prefix;
    uint32_t m = ~b0 & ~b1 & ~b2;
    while (m) {
        int bit = __ffs(m) - 1;
        non_compress_exp[pos++] = exp_local[bit];
        m &= (m - 1);
    }
}

// =========================================================
// Decompress kernel  (BLOCK_SIZE=128, PACK=32)
// Optimizations:
//  - Direct register-to-global writes (no smem intermediate for output)
//  - Cooperative smem loading of non_compress_exp
//  - Vectorised sign_mantissa loads
// =========================================================
__global__ void hopper_decompress_kernel(
    const rptr<unsigned char> compressed_input,
    const rptr<int> nbytes_8,
    const rptr<int> n_8,
    rptr<__nv_bfloat16> output,
    int n_works)
{
    // smem only needed for cooperative nce loading
    extern __shared__ __align__(128) uint8_t smem_raw[];
    unsigned char* smem_nce = smem_raw;

    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    int local_n_8;
    int local_nbytes_8 = 0;
    int tmp_blk = 0;
    const unsigned char* base_arr = nullptr;

    for (int i = 0; i < n_works; ++i) {
        local_n_8 = n_8[i];
        int current_n_block = local_n_8 / ELEMS_PER_BLOCK;

        base_arr = (i == 0) ? compressed_input : base_arr + local_nbytes_8;

        if (blockIdx.x >= tmp_blk && blockIdx.x < tmp_blk + current_n_block) {
            int real_n_block = blockIdx.x - tmp_blk;

            const unsigned char* sm_ptr = base_arr;
            const uint32_t* bm_b0 = reinterpret_cast<const uint32_t*>(base_arr + local_n_8);
            const uint32_t* bm_b1 = bm_b0 + local_n_8 / 32;
            const uint32_t* bm_b2 = bm_b1 + local_n_8 / 32;
            const int* block_offsets = reinterpret_cast<const int*>(bm_b2 + local_n_8 / 32);
            const unsigned char* nce_ptr = reinterpret_cast<const unsigned char*>(
                block_offsets + current_n_block + 1);

            int best_i = block_offsets[current_n_block];

            // Load bitmap (1 word per thread, coalesced)
            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            // Load sign_and_mantissa directly to registers (32 bytes per thread)
            // Address: sm_ptr + (real_n_block * 128 + tid) * 32
            // This is 32-byte aligned (tid * 32), so float4 (16B) loads are safe.
            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            // Compute zero count & prefix sums
            int thread_zero_count = __popc(~lb0 & ~lb1 & ~lb2);

            unsigned active_mask = __activemask();
            int val2 = thread_zero_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_zero_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_zero_count;
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
                block_zero_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            // Cooperative load of non_compress_exp into smem (coalesced)
            int nce_total = block_zero_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            // --- Decompress directly to registers, then write to global ---
            uint32_t hfi = ~(lb0 | lb1 | lb2);
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
                    uint32_t mask_before = (1u << bit) - 1;
                    int zero_before = __popc(hfi & mask_before);
                    exp_val = smem_nce[thread_exp_start + zero_before];
                } else {
                    exp_val = static_cast<unsigned char>(best_i + (code - 1));
                }

                unsigned short raw_val = 0;
                raw_val |= (static_cast<unsigned short>(lsm & 0x7F));
                raw_val |= (static_cast<unsigned short>(exp_val) << 7);
                raw_val |= (static_cast<unsigned short>(lsm >> 7) << 15);

                out_reg[bit] = __ushort_as_bfloat16(raw_val);
            }

            // Write output directly (coalesced, 16B at a time)
            __nv_bfloat16* out_ptr = output + (size_t)blockIdx.x * ELEMS_PER_BLOCK + tid * PACK;
            #pragma unroll
            for (int off = 0; off < PACK; off += 8) {
                *reinterpret_cast<float4*>(out_ptr + off) =
                    *reinterpret_cast<float4*>(out_reg + off);
            }
        }
        tmp_blk += current_n_block;
        local_nbytes_8 = nbytes_8[i];
    }
}

// =========================================================
// Decompress-unpad kernel: writes compacted output, skipping
// padding elements so no post-decompress copy is needed.
// The decompression logic is identical to hopper_decompress_kernel;
// only the output write addresses differ.
// =========================================================
__global__ void hopper_decompress_unpad_kernel(
    const rptr<unsigned char> compressed_input,
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

    int local_n_8;
    int local_nbytes_8 = 0;
    int tmp_blk = 0;
    const unsigned char* base_arr = nullptr;

    for (int i = 0; i < n_works; ++i) {
        local_n_8 = n_8[i];
        int current_n_block = local_n_8 / ELEMS_PER_BLOCK;

        base_arr = (i == 0) ? compressed_input : base_arr + local_nbytes_8;

        if (blockIdx.x >= tmp_blk && blockIdx.x < tmp_blk + current_n_block) {
            int real_n_block = blockIdx.x - tmp_blk;

            const unsigned char* sm_ptr = base_arr;
            const uint32_t* bm_b0 = reinterpret_cast<const uint32_t*>(base_arr + local_n_8);
            const uint32_t* bm_b1 = bm_b0 + local_n_8 / 32;
            const uint32_t* bm_b2 = bm_b1 + local_n_8 / 32;
            const int* block_offsets = reinterpret_cast<const int*>(bm_b2 + local_n_8 / 32);
            const unsigned char* nce_ptr = reinterpret_cast<const unsigned char*>(
                block_offsets + current_n_block + 1);

            int best_i = block_offsets[current_n_block];
 
            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            int thread_zero_count = __popc(~lb0 & ~lb1 & ~lb2);

            unsigned active_mask = __activemask();
            int val2 = thread_zero_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_zero_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_zero_count;
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
                block_zero_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            int nce_total = block_zero_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            uint32_t hfi = ~(lb0 | lb1 | lb2);
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
                    uint32_t mask_before = (1u << bit) - 1;
                    int zero_before = __popc(hfi & mask_before);
                    exp_val = smem_nce[thread_exp_start + zero_before];
                } else {
                    exp_val = static_cast<unsigned char>(best_i + (code - 1));
                }

                unsigned short raw_val = 0;
                raw_val |= (static_cast<unsigned short>(lsm & 0x7F));
                raw_val |= (static_cast<unsigned short>(exp_val) << 7);
                raw_val |= (static_cast<unsigned short>(lsm >> 7) << 15);

                out_reg[bit] = __ushort_as_bfloat16(raw_val);
            }

            // Write to compacted output: each work's data starts at
            // i * orig_ele_num instead of blockIdx.x * ELEMS_PER_BLOCK.
            int elem_in_work = real_n_block * ELEMS_PER_BLOCK + tid * PACK;
            __nv_bfloat16* out_ptr = output + (size_t)i * orig_ele_num + elem_in_work;

            // float4 stores need 16B (=8 bf16) alignment. elem_in_work is a
            // multiple of 8, so alignment is governed by (i*orig_ele_num)%8.
            bool base_aligned = (((size_t)i * orig_ele_num) & 7u) == 0;
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
        tmp_blk += current_n_block;
        local_nbytes_8 = nbytes_8[i];
    }
}

// =========================================================
// Split-store decompress kernel (BLOCK_SIZE=128, PACK=32)
// -----------------------------------------------------------------
// Decodes from two separate streams:
//   * `compressed_input` : deterministic stream (sign_and_mantissa +
//     3 bitmaps + block_offsets), stride per work = align128(det_bytes).
//   * `zero_input`        : compacted outlier exponents, stride per work
//     = align128(nbytes_8[i]) where nbytes_8 is the per-work zero counter.
// =========================================================
__global__ void hopper_decompress_split_store_kernel(
    const rptr<unsigned char> compressed_input,
    const rptr<unsigned char> zero_input,
    const rptr<int> nbytes_8,
    const rptr<int> n_8,
    rptr<__nv_bfloat16> output,
    int n_works)
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
            const unsigned char* nce_ptr = zero_arr;   // outliers from separate buffer

            int best_i = block_offsets[current_n_block];

            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            int thread_zero_count = __popc(~lb0 & ~lb1 & ~lb2);

            unsigned active_mask = __activemask();
            int val2 = thread_zero_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_zero_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_zero_count;
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
                block_zero_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            int nce_total = block_zero_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            uint32_t hfi = ~(lb0 | lb1 | lb2);
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
                    uint32_t mask_before = (1u << bit) - 1;
                    int zero_before = __popc(hfi & mask_before);
                    exp_val = smem_nce[thread_exp_start + zero_before];
                } else {
                    exp_val = static_cast<unsigned char>(best_i + (code - 1));
                }

                unsigned short raw_val = 0;
                raw_val |= (static_cast<unsigned short>(lsm & 0x7F));
                raw_val |= (static_cast<unsigned short>(exp_val) << 7);
                raw_val |= (static_cast<unsigned short>(lsm >> 7) << 15);

                out_reg[bit] = __ushort_as_bfloat16(raw_val);
            }

            __nv_bfloat16* out_ptr = output + (size_t)blockIdx.x * ELEMS_PER_BLOCK + tid * PACK;
            #pragma unroll
            for (int off = 0; off < PACK; off += 8) {
                *reinterpret_cast<float4*>(out_ptr + off) =
                    *reinterpret_cast<float4*>(out_reg + off);
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

            int best_i = block_offsets[current_n_block];

            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            int thread_zero_count = __popc(~lb0 & ~lb1 & ~lb2);

            unsigned active_mask = __activemask();
            int val2 = thread_zero_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_zero_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_zero_count;
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
                block_zero_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            int nce_total = block_zero_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            uint32_t hfi = ~(lb0 | lb1 | lb2);
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
                    uint32_t mask_before = (1u << bit) - 1;
                    int zero_before = __popc(hfi & mask_before);
                    exp_val = smem_nce[thread_exp_start + zero_before];
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

// =========================================================
// Split-store decompress-unpad (variable) kernel.
// Like the fixed unpad kernel, but supports a DIFFERENT valid element
// count per work (imbalanced MoE token distribution): work i writes its
// first `orig_n_8[i]` valid elements to output starting at element offset
// `out_off[i]` (typically prefix_tokens[i] * hidden). `n_8[i]` holds the
// per-work PADDED element count (multiple of ELEMS_PER_BLOCK).
// =========================================================
__global__ void hopper_decompress_split_store_unpad_v_kernel(
    const rptr<unsigned char> compressed_input,
    const rptr<unsigned char> zero_input,
    const rptr<int> nbytes_8,
    const rptr<int> n_8,
    const rptr<int> out_off,
    const rptr<int> orig_n_8,
    rptr<__nv_bfloat16> output,
    int n_works)
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

            int best_i = block_offsets[current_n_block];

            int bmap_idx = real_n_block * BLOCK_SIZE + tid;
            uint32_t lb0 = bm_b0[bmap_idx];
            uint32_t lb1 = bm_b1[bmap_idx];
            uint32_t lb2 = bm_b2[bmap_idx];

            const unsigned char* my_sm = sm_ptr + (real_n_block * BLOCK_SIZE + tid) * PACK;
            unsigned char local_sm[PACK];
            *reinterpret_cast<float4*>(local_sm)      = *reinterpret_cast<const float4*>(my_sm);
            *reinterpret_cast<float4*>(local_sm + 16)  = *reinterpret_cast<const float4*>(my_sm + 16);

            int thread_zero_count = __popc(~lb0 & ~lb1 & ~lb2);

            unsigned active_mask = __activemask();
            int val2 = thread_zero_count;
            #pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                int t = __shfl_up_sync(active_mask, val2, offset);
                if (lane >= offset) val2 += t;
            }
            int warp_prefix = val2 - thread_zero_count;

            __shared__ int warp_totals[BLOCK_SIZE / WARP_SIZE];
            __shared__ int block_zero_count;
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
                block_zero_count = run;
            }
            __syncthreads();

            int block_start = block_offsets[real_n_block];
            int thread_exp_start = warp_totals[warp_id] + warp_prefix;

            int nce_total = block_zero_count;
            for (int idx = tid; idx < nce_total; idx += BLOCK_SIZE) {
                smem_nce[idx] = nce_ptr[block_start + idx];
            }
            __syncthreads();

            uint32_t hfi = ~(lb0 | lb1 | lb2);
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
                    uint32_t mask_before = (1u << bit) - 1;
                    int zero_before = __popc(hfi & mask_before);
                    exp_val = smem_nce[thread_exp_start + zero_before];
                } else {
                    exp_val = static_cast<unsigned char>(best_i + (code - 1));
                }

                unsigned short raw_val = 0;
                raw_val |= (static_cast<unsigned short>(lsm & 0x7F));
                raw_val |= (static_cast<unsigned short>(exp_val) << 7);
                raw_val |= (static_cast<unsigned short>(lsm >> 7) << 15);

                out_reg[bit] = __ushort_as_bfloat16(raw_val);
            }

            int base = out_off[my_i];
            int valid_limit = orig_n_8[my_i];
            int elem_in_work = real_n_block * ELEMS_PER_BLOCK + tid * PACK;
            __nv_bfloat16* out_ptr = output + (size_t)base + elem_in_work;

            // float4 needs 16B (=8 bf16) alignment; elem_in_work is a multiple
            // of 8, so alignment is governed by base % 8.
            bool base_aligned = (((size_t)base & 7u) == 0);
            if (base_aligned && elem_in_work + PACK <= valid_limit) {
                #pragma unroll
                for (int off = 0; off < PACK; off += 8) {
                    *reinterpret_cast<float4*>(out_ptr + off) =
                        *reinterpret_cast<float4*>(out_reg + off);
                }
            } else if (elem_in_work < valid_limit) {
                int valid = min(PACK, valid_limit - elem_in_work);
                for (int j = 0; j < valid; ++j) {
                    out_ptr[j] = out_reg[j];
                }
            }
        }
}

// =========================================================
// Host launchers
// =========================================================

void hopper_compress_api_8(
    __nv_bfloat16* d_vec,
    unsigned char* output,
    unsigned char* output_final,
    int* n_8,
    int* nbytes_8,
    int* global_zero_counter_8,
    int* d_bases,
    int n,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(__nv_bfloat16)
               + ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_compress_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_compress_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        d_vec, output, n_8, global_zero_counter_8, d_bases, n_works, n);

    if (n_works > 1) {
        size_t copy_smem = COPY_CHUNK;
        cudaFuncSetAttribute(hopper_copy_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)copy_smem + 1024);
        hopper_copy_kernel<<<num_blocks, COPY_BLK, copy_smem, stream>>>(
            output, output_final, n_8, nbytes_8, global_zero_counter_8, n_works);
    }
}

void hopper_decompress_api_8(
    unsigned char* compressed_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int n,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    // smem only for cooperative nce loading (worst case: all elements are outliers)
    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_decompress_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_decompress_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, nbytes_8, n_8, output, n_works);
}

void hopper_decompress_unpad_api_8(
    unsigned char* compressed_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_decompress_unpad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_decompress_unpad_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, nbytes_8, n_8, output, n_works, orig_ele_num);
}

// =========================================================
// Split-store host launchers
// =========================================================

void hopper_compress_split_store_api_8(
    __nv_bfloat16* d_vec,
    unsigned char* output,
    unsigned char* output1,
    unsigned char* output_final,
    int* n_8,
    int* global_zero_counter_8,
    int* d_bases,
    int n,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(__nv_bfloat16)
               + ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_compress_split_store_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_compress_split_store_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        d_vec, output, output1, n_8, global_zero_counter_8, d_bases, n_works, n);

    // Compact the variable-size outlier stream (output1 -> output_final).
    size_t copy_smem = COPY_CHUNK;
    cudaFuncSetAttribute(hopper_copy_zero_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)copy_smem + 1024);
    hopper_copy_zero_kernel<<<num_blocks, COPY_BLK, copy_smem, stream>>>(
        output1, output_final, n_8, global_zero_counter_8, n_works);
}

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
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    // pad compress kernel uses register arrays -> no large dynamic smem.
    hopper_compress_split_store_pad_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_vec, output, output1, n_8, orig_n_8, global_zero_counter_8, d_bases, n_works);

    // Compact the variable-size outlier stream (output1 -> output_final).
    size_t copy_smem = COPY_CHUNK;
    cudaFuncSetAttribute(hopper_copy_zero_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)copy_smem + 1024);
    hopper_copy_zero_kernel<<<num_blocks, COPY_BLK, copy_smem, stream>>>(
        output1, output_final, n_8, global_zero_counter_8, n_works);
}

void hopper_decompress_split_store_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int n,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = (n + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_decompress_split_store_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_decompress_split_store_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, zero_input, nbytes_8, n_8, output, n_works);
}

void hopper_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int orig_ele_num,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_decompress_split_store_unpad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_decompress_split_store_unpad_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, zero_input, nbytes_8, n_8, output, n_works, orig_ele_num);
}

void hopper_decompress_split_store_unpad_v_api_8(
    unsigned char* compressed_input,
    unsigned char* zero_input,
    int* nbytes_8,
    int* n_8,
    int* out_off,
    int* orig_n_8,
    __nv_bfloat16* output,
    int padded_n_total,
    int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int num_blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (num_blocks == 0) return;

    size_t smem = ELEMS_PER_BLOCK * sizeof(unsigned char);

    cudaFuncSetAttribute(hopper_decompress_split_store_unpad_v_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);

    hopper_decompress_split_store_unpad_v_kernel<<<num_blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, zero_input, nbytes_8, n_8, out_off, orig_n_8, output, n_works);
}
