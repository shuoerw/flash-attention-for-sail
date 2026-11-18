#include "flash_fwd_d512_swa.h"

#ifdef FLASHATTENTION_PPU_D512_SWA
#include "softmax.h"

namespace flash_d512_detail {

using namespace cute;
struct D512SwaParams {
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* o;
    float* lse;
    const float* sink;
    const int* page_table;
    const int* lengths;
    const int* cu_q;
    const int* cu_k;
    const int* used_q;
    int64_t q_token_stride;
    int64_t q_head_stride;
    int64_t q_batch_stride;
    int64_t k_token_stride;
    int64_t v_token_stride;
    int64_t k_page_stride;
    int64_t v_page_stride;
    int64_t page_table_stride;
    int q_length;
    int total_q;
    int batch;
    int page_size;
    float scale;
};

// One CTA handles one query token and all 64 local query heads. Warps 0-3
// and 4-7 compute the two D256 partial scores for the same head rows; after
// summing scores, each group accumulates its own D256 output columns.
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kTileN = 32;
constexpr int kCubeW = 64;
constexpr int kThreads = 256;
constexpr int kQElements = kHeads * kDim;
constexpr int kKvElements = kTileN * kDim;
constexpr int kScoreElements = kThreads * 16 * sizeof(float) / sizeof(uint16_t);
constexpr int kVElements = 2 * kKvElements;
// Q is cached in registers before this space becomes two V buffers plus
// partial-score scratch. K uses one buffer throughout the KV loop.
constexpr int kQvElements = kVElements + kScoreElements;
constexpr int kSmemBytes = (kQvElements + kKvElements) * sizeof(uint16_t);
static_assert(kSmemBytes <= 128 * 1024);
constexpr int kAtoms = 16;
constexpr int kHalfWarpGroup = kThreads / 2;

__device__ __forceinline__ int d512_staged_start(int q_start, int batch) {
    return (q_start + kD512GatherRows - 1) / kD512GatherRows * kD512GatherRows
        + batch * kD512GatherSuffix;
}

using BF16 = cutlass::bfloat16_t;

template <bool Transpose, bool AOperand>
__device__ __forceinline__ void read_matrix(uint32_t (&out)[4], int addr) {
    constexpr int lbo = AOperand || Transpose ? 1 : 64;
    constexpr int sbo = AOperand || Transpose ? 64 : 1;
    PPU0015_TSM_LD_SWZL_IMPL<BF16, Transpose>{}(
        reinterpret_cast<int*>(out), addr, lbo, sbo, 0);
}

__device__ __forceinline__ void mma(float (&out)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
    asm volatile(
        "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+r"(out[0]), "+r"(out[1]), "+r"(out[2]), "+r"(out[3]),
          "+r"(out[4]), "+r"(out[5]), "+r"(out[6]), "+r"(out[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
}

template <int Rows>
__device__ __forceinline__ void copy_cube(
        uint16_t* dst, uint16_t const* src, int rows, int64_t stride, int cube) {
    AiuDesc desc;
    desc.init(nullptr, rows, kDim, stride);
    PPU0015_AIU_LOAD_IMPL<Int<Rows * kCubeW * 16>, 16, Rows, kCubeW, true>{}(
        dst + cube * Rows * kCubeW, src, desc, cube * kCubeW, 0, 0);
}

template <int Rows>
__device__ __forceinline__ void load_tile(
        uint16_t* dst, uint16_t const* src, int rows, int64_t stride, int warp) {
    copy_cube<Rows>(dst, src, rows, stride, warp);
    cute::cp_async_fence();
}

__device__ __forceinline__ int operand_offset(int dim_atom, int key_atom) {
    return dim_atom / 4 * 256 + dim_atom % 4 * 2 + key_atom * 128;
}

template <int WindowLeft, bool Paged, bool VarlenQ = !Paged>
__global__ __launch_bounds__(kThreads, 2) void d512_swa_fwd(D512SwaParams p) {
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const warp = __ppu_read_firstlane(tid / 32);
    int const d_half = warp / 4;
    int const q_head0 = (warp % 4) * 16 + lane / 4;
    int batch = blockIdx.y;
    int query = blockIdx.x;
    if constexpr (VarlenQ) {
        int lo = 0, hi = p.batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (p.cu_q[mid] <= query) { lo = mid; }
            else { hi = mid; }
        }
        batch = lo;
        query -= p.cu_q[batch];
    }
    int const q_start = VarlenQ ? p.cu_q[batch] : 0;
    int const q_capacity = VarlenQ ? p.cu_q[batch + 1] - q_start : p.q_length;
    int const q_count = p.used_q ? p.used_q[batch] : q_capacity;
    if (query >= q_count) { return; }
    int const staged_start = VarlenQ ? d512_staged_start(q_start, batch) : batch * (p.q_length + WindowLeft);
    int const k_start = Paged ? staged_start : p.cu_k[batch];
    int const k_count = Paged ? min(p.lengths[batch], q_count + WindowLeft)
        : (p.lengths ? p.lengths[batch] : p.cu_k[batch + 1] - k_start);
    int const right = query + k_count - q_count;
    int const left = max(right - WindowLeft, 0);
    int const tile_origin = left;
    int const output_token = VarlenQ ? q_start + query : batch * p.q_length + query;
    auto* out = reinterpret_cast<uint4*>(p.o + int64_t(output_token) * kQElements);
    auto* lse = p.lse + (VarlenQ ? q_start + query : int64_t(batch) * kHeads * p.q_length + query);
    int const lse_stride = VarlenQ ? p.total_q : p.q_length;
    if (right < 0) {
        #pragma unroll
        for (int i = tid; i < kQElements / 8; i += kThreads) { out[i] = make_uint4(0, 0, 0, 0); }
        if (tid < kHeads) { lse[int64_t(tid) * lse_stride] = -INFINITY; }
        return;
    }

    extern __shared__ __align__(128) uint16_t shared[];
    auto* qv_smem = shared;
    auto* k_smem = shared + kQvElements;
    auto* score_smem = reinterpret_cast<float4*>(shared + kVElements);
    auto const* q_src = p.q + int64_t(batch) * p.q_batch_stride + int64_t(q_start + query) * p.q_token_stride;
    auto copy_kv = [&](uint16_t* dst, uint16_t const* src, int64_t token_stride,
                       int n) {
        int const row = tile_origin + n * kTileN;
        src += int64_t(k_start + row) * token_stride;
        int const valid_end = right + 1;
        load_tile<kTileN>(dst, src, min(kTileN, valid_end - row), token_stride, warp);
    };

    int const first = (right - tile_origin) / kTileN;
    int const last = (left - tile_origin) / kTileN;
    load_tile<kHeads>(qv_smem, q_src, kHeads, p.q_head_stride, warp);
    copy_kv(k_smem, p.k, p.k_token_stride, first);
    flash::cp_async_wait<0>();
    __syncthreads();

    int const qv_base = __ppu_read_firstlane(reinterpret_cast<uintptr_t>(qv_smem) / 16);
    int const k_base = __ppu_read_firstlane(reinterpret_cast<uintptr_t>(k_smem) / 16) + d_half * 1024;
    int const v_base = qv_base + d_half * 1024;
    int const q_base = qv_base + d_half * 2048 + (warp % 4) * 128;
    uint32_t q_reg[kAtoms][4];
    #pragma unroll
    for (int d = 0; d < kAtoms; ++d) {
        read_matrix<false, true>(q_reg[d], q_base + d / 4 * 512 + d % 4 * 2);
    }
    __syncthreads();
    copy_kv(qv_smem, p.v, p.v_token_stride, first);
    flash::cp_async_wait<0>();
    __syncthreads();

    float output[kAtoms][8];
    auto out_acc = make_tensor(make_rmem_ptr(&output[0][0]),
        make_layout(Shape<Shape<_2, _2, _2>, _1, _16>{}));
    clear(out_acc);
    flash::Softmax<2> softmax(p.scale * float(M_LOG2E));
    fill(softmax.row_max, -INFINITY);
    clear(softmax.row_sum);

    #pragma unroll 1
    for (int n = first; n >= last; --n) {
        float scores[2][8];
        auto score = make_tensor(make_rmem_ptr(&scores[0][0]),
            make_layout(Shape<Shape<_2, _2, _2>, _1, _2>{}));
        clear(score);
        int const bank = (first - n) & 1;
        int const inactive = bank ^ 1;
        int const v_addr = v_base + bank * (kKvElements / 8);

        uint32_t kb[kAtoms][2][4];
        auto load_k = [&](int d) {
            read_matrix<false, false>(kb[d][0], k_base + operand_offset(d, 0));
            read_matrix<false, false>(kb[d][1], k_base + operand_offset(d, 1));
        };
        #pragma hggc mmatiestrictly
        {
            load_k(0);
            #pragma unroll
            for (int d = 0; d < kAtoms; ++d) {
                if (d + 1 < kAtoms) { load_k(d + 1); }
                mma(scores[0], q_reg[d], kb[d][0]);
                mma(scores[1], q_reg[d], kb[d][1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int const key = i / 2;
            int const v = (i % 2) * 4;
            score_smem[i * kThreads + tid] = make_float4(
                scores[key][v], scores[key][v + 1], scores[key][v + 2], scores[key][v + 3]);
        }
        // The K readers and score producers have finished. The previous
        // V slot is free and the current V slot was filled one tile ahead.
        __syncthreads();
        if (n > last) {
            copy_kv(k_smem, p.k, p.k_token_stride, n - 1);
            copy_kv(qv_smem + inactive * kKvElements, p.v, p.v_token_stride, n - 1);
        }

        int const peer = tid ^ kHalfWarpGroup;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float4 const x = score_smem[i * kThreads + peer];
            int const key = i / 2;
            int const v = (i % 2) * 4;
            scores[key][v] += x.x;
            scores[key][v + 1] += x.y;
            scores[key][v + 2] += x.z;
            scores[key][v + 3] += x.w;
        }
        bool const needs_mask = n == first && tile_origin + (n + 1) * kTileN > right + 1;
        if (needs_mask) {
            int const key_base = tile_origin + n * kTileN + (lane % 4) * 2;
            #pragma unroll
            for (int key = 0; key < 2; ++key) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    int const idx = key_base + key * 16 + (v % 2) + (v / 4) * 8;
                    if (idx < left || idx > right) { scores[key][v] = -INFINITY; }
                }
            }
        }
        auto scales = softmax.template max_get_scale_skip<false, true>(score);
        softmax.template online_softmax<false, true>(score);
        if (softmax.need_rescale) { softmax.rescale_o(out_acc, scales); }
        uint32_t p_reg[2][4];
        auto p_bf16 = make_tensor(make_rmem_ptr(reinterpret_cast<BF16*>(&p_reg[0][0])),
                                  make_layout(Shape<Shape<_2, _2, _2>, _1, _2>{}));
        flash::convert_type_out(score, p_bf16);

        uint32_t vb[2][kAtoms][4];
        auto load_v = [&](int step) {
            int const col = step / 8 * 4 + step % 4;
            int const key = step % 8 / 4;
            read_matrix<true, false>(vb[key][col], v_addr + operand_offset(col, key));
        };
        #pragma hggc mmatiestrictly
        {
            #pragma unroll
            for (int step = 0; step < 4; ++step) { load_v(step); }
            #pragma unroll
            for (int step = 0; step < 2 * kAtoms; ++step) {
                if (step + 4 < 2 * kAtoms) { load_v(step + 4); }
                int const col = step / 8 * 4 + step % 4;
                int const key = step % 8 / 4;
                mma(output[col], p_reg[key], vb[key][col]);
            }
        }
        flash::cp_async_wait<0>();
        __syncthreads();
    }

    auto final_scale = [&] {
        if (!p.sink) { return softmax.finalize(); }
        typename flash::Softmax<2>::TensorT sink;
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            sink(row) = p.sink[q_head0 + row * 8];
        }
        return softmax.finalize_aux(sink);
    }();
    softmax.rescale_o(out_acc, final_scale);
    if (d_half == 0 && (lane % 4) == 0) {
        lse[int64_t(q_head0) * lse_stride] = softmax.row_sum(0);
        lse[int64_t(q_head0 + 8) * lse_stride] = softmax.row_sum(1);
    }
    uint32_t converted[kAtoms][4];
    auto out_bf16 = make_tensor(make_rmem_ptr(reinterpret_cast<BF16*>(&converted[0][0])), out_acc.layout());
    flash::convert_type_out(out_acc, out_bf16);
    // The former Q/V and K tiles are dead, so use the same buffer for coalesced output.
    #pragma unroll
    for (int col = 0; col < kAtoms; ++col) {
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            int const row = q_head0 + (v % 2) * 8;
            int const feature = d_half * 256 + col * 16 + (lane % 4) * 2 + (v / 2) * 8;
            reinterpret_cast<uint32_t*>(shared)[(row * kDim + feature) / 2] = converted[col][v];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i = tid; i < kQElements / 8; i += kThreads) {
        out[i] = reinterpret_cast<uint4 const*>(shared)[i];
    }
}
// Each batch stages only the union of its exact sliding windows. Inactive
// rows are outside the staged length and the AIU pads the final tile.
template <int WindowLeft, bool VarlenQ>
__global__ void gather_active_suffix(D512SwaParams p, uint16_t* k_dst, uint16_t* v_dst) {
    constexpr int Rows = kD512GatherRows;
    constexpr int VectorsPerRow = kDim / 8;
    int b = blockIdx.y;
    int row_base = blockIdx.x * Rows;
    if constexpr (VarlenQ) {
        int lo = 0, hi = p.batch;
        while (lo + 1 < hi) {
            int const mid = (lo + hi) / 2;
            if (d512_staged_start(p.cu_q[mid], mid) <= row_base) { lo = mid; }
            else { hi = mid; }
        }
        b = lo;
        row_base -= d512_staged_start(p.cu_q[b], b);
    }
    int const q_start = VarlenQ ? p.cu_q[b] : 0;
    int const q_capacity = VarlenQ ? p.cu_q[b + 1] - q_start : p.q_length;
    int const q_count = p.used_q ? p.used_q[b] : q_capacity;
    if (q_count == 0) { return; }
    int const capacity = q_count + WindowLeft;
    int const dst_start = VarlenQ ? d512_staged_start(q_start, b) : b * (p.q_length + WindowLeft);
    int const length = p.lengths[b];
    int const start = max(0, length - capacity);
    int const active = length - start;
    if (row_base >= active) { return; }
    int const shift = p.page_size == 64 ? 6 : 7;
    auto const* table = p.page_table + int64_t(b) * p.page_table_stride;
    #pragma unroll
    for (int i = threadIdx.x; i < Rows * VectorsPerRow; i += blockDim.x) {
        int const row = row_base + i / VectorsPerRow;
        if (row < active) {
            int const token = start + row;
            int const page = table[token >> shift];
            int const in_page = token & (p.page_size - 1);
            int const col = (i % VectorsPerRow) * 8;
            auto const* k = p.k + int64_t(page) * p.k_page_stride + int64_t(in_page) * p.k_token_stride + col;
            auto const* v = p.v + int64_t(page) * p.v_page_stride + int64_t(in_page) * p.v_token_stride + col;
            int64_t const dst = (int64_t(dst_start) + row) * kDim + col;
            *reinterpret_cast<uint4*>(k_dst + dst) = *reinterpret_cast<uint4 const*>(k);
            *reinterpret_cast<uint4*>(v_dst + dst) = *reinterpret_cast<uint4 const*>(v);
        }
    }
}

template <int WindowLeft>
void launch_d512_staged(D512SwaParams inputs, uint16_t* k, uint16_t* v, hggcStream_t stream) {
    if (inputs.cu_q) {
        int const rows = d512_swa_workspace_tokens(inputs.total_q, inputs.batch);
        gather_active_suffix<WindowLeft, true><<<rows / kD512GatherRows, kThreads, 0, stream>>>(inputs, k, v);
    } else {
        dim3 grid((inputs.q_length + WindowLeft + kD512GatherRows - 1) / kD512GatherRows, inputs.batch);
        gather_active_suffix<WindowLeft, false><<<grid, kThreads, 0, stream>>>(inputs, k, v);
    }
    inputs.k = k;
    inputs.v = v;
    inputs.k_token_stride = kDim;
    inputs.v_token_stride = kDim;
    if (inputs.cu_q) {
        CHECK_CUDA(hggcFuncSetAttribute(d512_swa_fwd<WindowLeft, true, true>, hggcFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        d512_swa_fwd<WindowLeft, true, true><<<inputs.total_q, kThreads, kSmemBytes, stream>>>(inputs);
    } else {
        dim3 grid(inputs.q_length, inputs.batch);
        CHECK_CUDA(hggcFuncSetAttribute(d512_swa_fwd<WindowLeft, true, false>, hggcFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
        d512_swa_fwd<WindowLeft, true, false><<<grid, kThreads, kSmemBytes, stream>>>(inputs);
    }
}

template <int WindowLeft>
void launch_d512_packed(D512SwaParams const& inputs, hggcStream_t stream) {
    CHECK_CUDA(hggcFuncSetAttribute(d512_swa_fwd<WindowLeft, false>, hggcFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    d512_swa_fwd<WindowLeft, false><<<inputs.total_q, kThreads, kSmemBytes, stream>>>(inputs);
}

}  // namespace flash_d512_detail

void run_flash_fwd_d512_swa(Flash_fwd_params const& params, void* workspace, hggcStream_t stream) {
    flash_d512_detail::D512SwaParams inputs{};
    inputs.q = static_cast<uint16_t const*>(params.q_ptr);
    inputs.k = static_cast<uint16_t const*>(params.k_ptr);
    inputs.v = static_cast<uint16_t const*>(params.v_ptr);
    inputs.o = static_cast<uint16_t*>(params.o_ptr);
    inputs.lse = static_cast<float*>(params.softmax_lse_ptr);
    inputs.sink = static_cast<float const*>(params.s_aux_ptr);
    inputs.page_table = params.page_table;
    inputs.lengths = params.seqused_k;
    inputs.cu_q = params.cu_seqlens_q;
    inputs.cu_k = params.cu_seqlens_k;
    inputs.used_q = params.seqused_q;
    inputs.q_token_stride = params.q_row_stride;
    inputs.q_head_stride = params.q_head_stride;
    inputs.q_batch_stride = params.q_batch_stride;
    inputs.k_token_stride = params.k_row_stride;
    inputs.v_token_stride = params.v_row_stride;
    inputs.k_page_stride = params.k_batch_stride;
    inputs.v_page_stride = params.v_batch_stride;
    inputs.page_table_stride = params.page_table_batch_stride;
    inputs.q_length = params.seqlen_q;
    inputs.total_q = params.total_q;
    inputs.batch = params.b;
    inputs.page_size = params.page_table ? params.page_size : 0;
    inputs.scale = params.scale_softmax;
    if (workspace) {
        auto* k = static_cast<uint16_t*>(workspace);
        int64_t const tokens = d512_swa_workspace_tokens(inputs.total_q, inputs.batch);
        if (params.window_size_left == 511) {
            flash_d512_detail::launch_d512_staged<511>(inputs, k, k + tokens * 512, stream);
        } else {
            flash_d512_detail::launch_d512_staged<512>(inputs, k, k + tokens * 512, stream);
        }
    } else {
        if (params.window_size_left == 511) {
            flash_d512_detail::launch_d512_packed<511>(inputs, stream);
        } else {
            flash_d512_detail::launch_d512_packed<512>(inputs, stream);
        }
    }
}
#endif
