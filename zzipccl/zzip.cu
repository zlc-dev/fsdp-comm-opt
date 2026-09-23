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
static constexpr int COPY_BLK = 128;
static constexpr int COPY_CHUNK = COPY_BLK * 16;

template <typename T> using rptr = T * __restrict__;


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
__global__ void zero_compress_split_store_pad_kernel(
    const rptr<__nv_bfloat16> input,
    rptr<unsigned char> output,
    rptr<unsigned char> output1,
    rptr<int> n_8,
    rptr<int> orig_n_8,
    rptr<int> global_zero_counter_8,
    rptr<int> global_retained_counter_8,
    rptr<int> bases_in,
    rptr<int> zero_exp_threshold,
    int n_works)
{
    const int tid     = threadIdx.x;
    const int lane    = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int best_i  = bases_in[0];
    const int zero_cutoff = zero_exp_threshold[0];

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
            int det_bytes = local_n_8 / 8 * 3
                          + cnb * 4 + cnb * 4 + 4;
            det_bytes = (local_n_8 == 0) ? 0
                                        : ((det_bytes + ALIGN_BYTES - 1) /
                                           ALIGN_BYTES * ALIGN_BYTES);
            tmp_blk    += cnb;
            normal_off += det_bytes;
            extra_off  += 2 * local_n_8;
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
        int m = static_cast<int>(sm_local[j] & 0x7F);
        unsigned int code = (e >= best_i && e < best_i + 6) ? (e - best_i + 1) : 0u;
        code = (code == 0 && e < zero_cutoff) ? 7 : code;
        b0 |= ((code >> 0) & 1u) << j;
        b1 |= ((code >> 1) & 1u) << j;
        b2 |= ((code >> 2) & 1u) << j;
    }

    // --- prefix-sum of zero counts ---
    int warp_zero_prefix;
    int warp_retained_prefix;
    int thread_zero_count = __popc(~b0 & ~b1 & ~b2);
    int thread_retained_count = 32 - __popc(b0 & b1 & b2);
    unsigned active_mask = __activemask();
    int val_z = thread_zero_count;
    int val_r = thread_retained_count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp_z = __shfl_up_sync(active_mask, val_z, offset);
        int tmp_r = __shfl_up_sync(active_mask, val_r, offset);
        if (lane >= offset) {
            val_z += tmp_z;
            val_r += tmp_r;
        }
    }
    warp_zero_prefix = val_z - thread_zero_count;
    warp_retained_prefix = val_r - thread_retained_count;

    __shared__ int warp_zero_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_retained_totals[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_zero_base_global;
    __shared__ int block_retained_base_global;
    if (lane == WARP_SIZE - 1) {
        warp_zero_totals[warp_id] = val_z;
        warp_retained_totals[warp_id] = val_r;
    }
    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        int run_z = 0, run_r = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            int t = warp_zero_totals[w]; warp_zero_totals[w] = run_z; run_z += t;
            int s = warp_retained_totals[w]; warp_retained_totals[w] = run_r; run_r += s;
        }
        block_zero_base_global = atomicAdd(global_zero_counter_8 + my_i, run_z);
        block_retained_base_global = atomicAdd(global_retained_counter_8 + my_i, run_r);
    }
    __syncthreads();


    // --- deterministic stream: bitmaps ---
    uint32_t* bitmap_b0 = reinterpret_cast<uint32_t*>(output + my_normal_off);
    uint32_t* bitmap_b1 = bitmap_b0 + my_local_n_8 / 32;
    uint32_t* bitmap_b2 = bitmap_b1 + my_local_n_8 / 32;
    int bmap_idx = my_real_n_block * BLOCK_SIZE + tid;
    bitmap_b0[bmap_idx] = b0;
    bitmap_b1[bmap_idx] = b1;
    bitmap_b2[bmap_idx] = b2;

    // --- deterministic stream: block_offsets ---
    int* block_retained_offsets_out = reinterpret_cast<int*>(bitmap_b2 + my_local_n_8 / 32);
    if (warp_id == 0 && lane == 0) {
        block_retained_offsets_out[my_real_n_block] = block_retained_base_global;
    }
    int* block_zero_offsets_out = reinterpret_cast<int*>(
        block_retained_offsets_out + my_current_n_block
    );
    if (warp_id == 0 && lane == 0) {
        block_zero_offsets_out[my_real_n_block] = block_zero_base_global;
        if (my_real_n_block == 0)
            block_zero_offsets_out[my_current_n_block] = best_i;
    }

    // --- outlier stream: sign+mantissa ---
    unsigned char* sign_and_mantissa = output1 + my_extra_off;
    int retained_pos = block_retained_base_global 
        + warp_retained_totals[warp_id] 
        + warp_retained_prefix;
    uint32_t retained_mask = ~(b0 & b1 & b2);
    while(retained_mask) {
        int bit = __ffs(retained_mask) - 1;
        sign_and_mantissa[retained_pos++] = sm_local[bit];
        retained_mask &= (retained_mask - 1);
    }

    unsigned char* non_compress_exp = sign_and_mantissa + my_local_n_8;
    int zero_pos = block_zero_base_global + warp_zero_totals[warp_id] + warp_zero_prefix;
    uint32_t zero_mask = ~b0 & ~b1 & ~b2;
    while (zero_mask) {
        int bit = __ffs(zero_mask) - 1;
        non_compress_exp[zero_pos++] = exp_local[bit];
        zero_mask &= (zero_mask - 1);
    }
}

// Compact the two fixed-stride staging streams independently.  The final
// streams are packed per work and padded to 128 bytes, matching the original
// ZipCCL outlier compaction convention.
__global__ void zero_compact_two_streams_kernel(
    const rptr<unsigned char> staging,
    rptr<unsigned char> retained_output,
    rptr<unsigned char> zero_output,
    const rptr<int> n_8,
    const rptr<int> retained_count_8,
    const rptr<int> zero_count_8,
    int n_works)
{
    extern __shared__ __align__(128) uint8_t copy_smem[];

    __shared__ int s_i, s_real_n_block, s_nblocks;
    __shared__ size_t s_staging_off, s_retained_off, s_zero_off;
    if (threadIdx.x == 0) {
        int block_base = 0;
        size_t staging_off = 0, retained_off = 0, zero_off = 0;
        int found = -1;
        for (int i = 0; i < n_works; ++i) {
            int n = n_8[i];
            int blocks = n / ELEMS_PER_BLOCK;
            if ((int)blockIdx.x >= block_base &&
                (int)blockIdx.x < block_base + blocks) {
                found = i;
                s_real_n_block = (int)blockIdx.x - block_base;
                s_nblocks = blocks;
                s_staging_off = staging_off;
                s_retained_off = retained_off;
                s_zero_off = zero_off;
                break;
            }
            block_base += blocks;
            staging_off += 2ULL * n;
            retained_off +=
                (retained_count_8[i] + ALIGN_BYTES - 1) / ALIGN_BYTES
                * ALIGN_BYTES;
            zero_off +=
                (zero_count_8[i] + ALIGN_BYTES - 1) / ALIGN_BYTES
                * ALIGN_BYTES;
        }
        s_i = found;
    }
    __syncthreads();
    if (s_i < 0) return;

    int nr = retained_count_8[s_i];
    int nz = zero_count_8[s_i];
    int nr_bytes = (nr + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    int nz_bytes = (nz + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
    int tiles = max(
        (nr_bytes + COPY_CHUNK - 1) / COPY_CHUNK,
        (nz_bytes + COPY_CHUNK - 1) / COPY_CHUNK);
    int tiles_per_block = (tiles + s_nblocks - 1) / s_nblocks;
    int tile_start = s_real_n_block * tiles_per_block;
    int tile_end = min(tile_start + tiles_per_block, tiles);
    const unsigned char* retained_in = staging + s_staging_off;
    const unsigned char* zero_in = retained_in + n_8[s_i];

    for (int tile = tile_start; tile < tile_end; ++tile) {
        int byte_start = tile * COPY_CHUNK;
        int retained_size = min(COPY_CHUNK, nr_bytes - byte_start);
        int zero_size = min(COPY_CHUNK, nz_bytes - byte_start);

        for (int off = threadIdx.x * 16; off < retained_size;
             off += COPY_BLK * 16) {
            if (off + 16 <= retained_size)
                *reinterpret_cast<float4*>(copy_smem + off) =
                    *reinterpret_cast<const float4*>(retained_in + byte_start + off);
            else
                for (int k = off; k < retained_size; ++k)
                    copy_smem[k] = retained_in[byte_start + k];
        }
        __syncthreads();
        for (int off = threadIdx.x * 16; off < retained_size;
             off += COPY_BLK * 16) {
            if (off + 16 <= retained_size)
                *reinterpret_cast<float4*>(retained_output + s_retained_off + byte_start + off) =
                    *reinterpret_cast<const float4*>(copy_smem + off);
            else
                for (int k = off; k < retained_size; ++k)
                    retained_output[s_retained_off + byte_start + k] = copy_smem[k];
        }
        __syncthreads();

        for (int off = threadIdx.x * 16; off < zero_size;
             off += COPY_BLK * 16) {
            if (off + 16 <= zero_size)
                *reinterpret_cast<float4*>(copy_smem + COPY_CHUNK + off) =
                    *reinterpret_cast<const float4*>(zero_in + byte_start + off);
            else
                for (int k = off; k < zero_size; ++k)
                    copy_smem[COPY_CHUNK + k] = zero_in[byte_start + k];
        }
        __syncthreads();
        for (int off = threadIdx.x * 16; off < zero_size;
             off += COPY_BLK * 16) {
            if (off + 16 <= zero_size)
                *reinterpret_cast<float4*>(zero_output + s_zero_off + byte_start + off) =
                    *reinterpret_cast<const float4*>(copy_smem + COPY_CHUNK + off);
            else
                for (int k = off; k < zero_size; ++k)
                    zero_output[s_zero_off + byte_start + k] = copy_smem[COPY_CHUNK + k];
        }
        __syncthreads();
    }
}

// Decode the zero-dropping format.  `compressed_input` contains only the
// bitmaps and the two block-offset arrays; retained sign+mantissas and
// uncompressed exponents are supplied as separate compact streams.
__global__ void zero_decompress_split_store_pad_kernel(
    const rptr<unsigned char> compressed_input,
    const rptr<unsigned char> retained_input,
    const rptr<unsigned char> zero_input,
    const rptr<int> n_8,
    const rptr<int> retained_count_8,
    const rptr<int> zero_count_8,
    rptr<__nv_bfloat16> output,
    int n_works,
    int orig_ele_num)
{
    extern __shared__ unsigned char smem[];
    unsigned char* retained_smem = smem;
    unsigned char* zero_smem = smem + ELEMS_PER_BLOCK;

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;

    __shared__ int s_i, s_real_block, s_local_n, s_cnb;
    __shared__ long long s_base, s_ret_off, s_zero_off;
    if (tid == 0) {
        int tmp_blk = 0;
        long long base = 0, ret_off = 0, zero_off = 0;
        s_i = -1;
        for (int i = 0; i < n_works; ++i) {
            int ln = n_8[i];
            int cnb = ln / ELEMS_PER_BLOCK;
            if ((int)blockIdx.x >= tmp_blk && (int)blockIdx.x < tmp_blk + cnb) {
                s_i = i;
                s_real_block = (int)blockIdx.x - tmp_blk;
                s_local_n = ln;
                s_cnb = cnb;
                s_base = base;
                s_ret_off = ret_off;
                s_zero_off = zero_off;
                break;
            }
            int det = ln / 8 * 3 + cnb * 4 + cnb * 4 + 4;
            det = ln == 0 ? 0 : (det + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
            tmp_blk += cnb;
            base += det;
            ret_off += (retained_count_8[i] + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
            zero_off += (zero_count_8[i] + ALIGN_BYTES - 1) / ALIGN_BYTES * ALIGN_BYTES;
        }
    }
    __syncthreads();
    if (s_i < 0) return;

    int my_i = s_i;
    int local_n = s_local_n;
    int real_block = s_real_block;
    int cnb = s_cnb;
    const unsigned char* base = compressed_input + s_base;
    const uint32_t* b0p = reinterpret_cast<const uint32_t*>(base);
    const uint32_t* b1p = b0p + local_n / 32;
    const uint32_t* b2p = b1p + local_n / 32;
    const int* retained_offsets = reinterpret_cast<const int*>(b2p + local_n / 32);
    const int* zero_offsets = retained_offsets + cnb;
    int map_idx = real_block * BLOCK_SIZE + tid;
    uint32_t b0 = b0p[map_idx];
    uint32_t b1 = b1p[map_idx];
    uint32_t b2 = b2p[map_idx];

    int outlier_count = __popc(~b0 & ~b1 & ~b2);
    int retained_count = 32 - __popc(b0 & b1 & b2);
    unsigned active = __activemask();
    int oz = outlier_count, rr = retained_count;
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        int z = __shfl_up_sync(active, oz, off);
        int r = __shfl_up_sync(active, rr, off);
        if (lane >= off) { oz += z; rr += r; }
    }
    int outlier_prefix = oz - outlier_count;
    int retained_prefix = rr - retained_count;

    __shared__ int warp_outlier[BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_retained[BLOCK_SIZE / WARP_SIZE];
    __shared__ int block_outlier, block_retained;
    if (lane == WARP_SIZE - 1) {
        warp_outlier[warp_id] = oz;
        warp_retained[warp_id] = rr;
    }
    __syncthreads();
    if (warp_id == 0 && lane == 0) {
        int rz = 0, rr0 = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            int z = warp_outlier[w]; warp_outlier[w] = rz; rz += z;
            int r = warp_retained[w]; warp_retained[w] = rr0; rr0 += r;
        }
        block_outlier = rz;
        block_retained = rr0;
    }
    __syncthreads();

    int block_retained_start = retained_offsets[real_block];
    int block_outlier_start = zero_offsets[real_block];
    for (int p = tid; p < block_retained; p += BLOCK_SIZE)
        retained_smem[p] = retained_input[s_ret_off + block_retained_start + p];
    for (int p = tid; p < block_outlier; p += BLOCK_SIZE)
        zero_smem[p] = zero_input[s_zero_off + block_outlier_start + p];
    __syncthreads();

    uint32_t zero_mask = ~b0 & ~b1 & ~b2;
    uint32_t retained_mask = ~(b0 & b1 & b2);
    __nv_bfloat16 out_reg[PACK];
    #pragma unroll
    for (int bit = 0; bit < PACK; ++bit) {
        unsigned int code = ((b0 >> bit) & 1u)
                          | (((b1 >> bit) & 1u) << 1)
                          | (((b2 >> bit) & 1u) << 2);
        unsigned char lsm = 0;
        if (code != 7) {
            int before = __popc(retained_mask & ((1u << bit) - 1));
            lsm = retained_smem[warp_retained[warp_id] + retained_prefix + before];
        }
        unsigned char exp_val = 0;
        if (code == 0) {
            int before = __popc(zero_mask & ((1u << bit) - 1));
            exp_val = zero_smem[warp_outlier[warp_id] + outlier_prefix + before];
        } else if (code != 7) {
            exp_val = static_cast<unsigned char>(zero_offsets[cnb]);
            exp_val = static_cast<unsigned char>(exp_val + code - 1);
        }
        unsigned short raw = (code == 7)
            ? 0
            : static_cast<unsigned short>((lsm & 0x7f)
                | (static_cast<unsigned short>(exp_val) << 7)
                | (static_cast<unsigned short>(lsm >> 7) << 15));
        out_reg[bit] = __ushort_as_bfloat16(raw);
    }
    int elem_in_work = real_block * ELEMS_PER_BLOCK + tid * PACK;
    if (orig_ele_num < 0) {
        __nv_bfloat16* out = output
            + (size_t)blockIdx.x * ELEMS_PER_BLOCK + tid * PACK;
        #pragma unroll
        for (int off = 0; off < PACK; off += 8)
            *reinterpret_cast<float4*>(out + off) =
                *reinterpret_cast<float4*>(out_reg + off);
    } else if (elem_in_work < orig_ele_num) {
        __nv_bfloat16* out = output
            + (size_t)my_i * orig_ele_num + elem_in_work;
        int valid = min(PACK, orig_ele_num - elem_in_work);
        bool aligned = ((((size_t)my_i * orig_ele_num) & 7u) == 0);
        if (aligned && valid == PACK) {
            #pragma unroll
            for (int off = 0; off < PACK; off += 8)
                *reinterpret_cast<float4*>(out + off) =
                    *reinterpret_cast<float4*>(out_reg + off);
        } else {
            for (int j = 0; j < valid; ++j)
                out[j] = out_reg[j];
        }
    }
}

void zero_compress_split_store_pad_api_8(
    __nv_bfloat16* input, unsigned char* output,
    unsigned char* staging,
    unsigned char* retained_output, unsigned char* zero_output,
    int* n_8, int* orig_n_8, int* zero_count_8, int* retained_count_8,
    int* bases_in, int* zero_exp_threshold, int padded_n_total, int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (blocks == 0) return;
    cudaMemsetAsync(zero_count_8, 0, n_works * sizeof(int), stream);
    cudaMemsetAsync(retained_count_8, 0, n_works * sizeof(int), stream);
    zero_compress_split_store_pad_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, staging, n_8, orig_n_8, zero_count_8,
        retained_count_8, bases_in, zero_exp_threshold, n_works);
    int compact_blocks = padded_n_total / ELEMS_PER_BLOCK;
    zero_compact_two_streams_kernel<<<compact_blocks, COPY_BLK,
                                      2 * COPY_CHUNK, stream>>>(
        staging, retained_output, zero_output,
        n_8, retained_count_8, zero_count_8, n_works);
}

void zero_decompress_split_store_pad_api_8(
    unsigned char* compressed_input, unsigned char* retained_input,
    unsigned char* zero_input, int* n_8, int* retained_count_8,
    int* zero_count_8, __nv_bfloat16* output, int padded_n_total, int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (blocks == 0) return;
    size_t smem = 2 * ELEMS_PER_BLOCK * sizeof(unsigned char);
    cudaFuncSetAttribute(zero_decompress_split_store_pad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);
    zero_decompress_split_store_pad_kernel<<<blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, retained_input, zero_input, n_8, retained_count_8,
        zero_count_8, output, n_works, -1);
}

void zero_decompress_split_store_unpad_api_8(
    unsigned char* compressed_input, unsigned char* retained_input,
    unsigned char* zero_input, int* n_8, int* retained_count_8,
    int* zero_count_8, __nv_bfloat16* output, int padded_n_total,
    int orig_ele_num, int n_works)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int blocks = padded_n_total / ELEMS_PER_BLOCK;
    if (blocks == 0) return;
    size_t smem = 2 * ELEMS_PER_BLOCK * sizeof(unsigned char);
    cudaFuncSetAttribute(zero_decompress_split_store_pad_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem + 1024);
    zero_decompress_split_store_pad_kernel<<<blocks, BLOCK_SIZE, smem, stream>>>(
        compressed_input, retained_input, zero_input, n_8, retained_count_8,
        zero_count_8, output, n_works, orig_ele_num);
}
