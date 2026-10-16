#include <cuda_runtime.h>
#include <cmath>
#include "flash_fwd_launch_template.h"
#include "flash_fwd_combine_launch_template.h"
#include "flash_prepare_scheduler.cu"

// Native test adapter. Buffers and sequence metadata belong to replay.py.
template<bool Local, bool Causal, bool M16, bool Split>
void launch_default(Flash_fwd_params& p, cudaStream_t stream) {
    run_flash_fwd<89, 192, 128, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
        Causal, Local, false, true, true, true, 64, false, false,
        true, Split, false, false, M16, false>(p, stream);
}

extern "C" int fa_mimo_run(void** ptr, int const* cfg, cudaStream_t stream) {
    Flash_fwd_params p{};
    int const batch = cfg[0], total_q = cfg[1], max_q = cfg[2], capacity_k = cfg[3];
    int const pages = cfg[4], pages_per_request = cfg[5], window = cfg[6];
    int const requested_splits = cfg[7];
    p.q_ptr = ptr[0];
    p.k_ptr = ptr[1];
    p.v_ptr = ptr[2];
    p.o_ptr = ptr[3];
    p.softmax_lse_ptr = ptr[4];
    p.tile_count_semaphore = static_cast<int*>(ptr[5]);
    p.cu_seqlens_q = static_cast<int*>(ptr[6]);
    p.seqused_k = static_cast<int*>(ptr[7]);
    p.page_table = static_cast<int*>(ptr[8]);
    p.s_aux_ptr = ptr[9];
    p.oaccum_ptr = ptr[10];
    p.softmax_lseaccum_ptr = ptr[11];
    p.b = p.b_k = batch;
    p.h = 16;
    p.h_k = 1;
    p.seqlen_q = max_q;
    p.seqlen_k = capacity_k;
    p.total_q = total_q;
    p.d = p.d_rounded = 192;
    p.dv = p.dv_rounded = 128;
    p.seqlen_q_rounded = (max_q + 127) / 128 * 128;
    p.seqlen_k_rounded = (capacity_k + 127) / 128 * 128;
    p.q_row_stride = 16 * 192;
    p.q_head_stride = 192;
    p.k_row_stride = p.k_head_stride = 192;
    p.k_batch_stride = 64 * 192;
    p.v_row_stride = p.v_head_stride = 128;
    p.v_batch_stride = 64 * 128;
    p.v_dim_stride = 1;
    p.o_row_stride = 16 * 128;
    p.o_head_stride = 128;
    p.page_size = 64;
    p.num_pages = pages;
    p.page_table_batch_stride = pages_per_request;
    p.scale_softmax = 1.f / std::sqrt(192.f);
    p.p_dropout = p.rp_dropout = 1;
    p.is_bf16 = true;
    p.is_local = window >= 0;
    p.is_causal = !p.is_local && max_q > 1;
    p.window_size_left = p.is_local ? window : capacity_k - 1;
    p.window_size_right = 0;
    p.arch = 89;
    p.pack_gqa = true;
    p.is_varlen_q = total_q / batch != max_q;
    cudaDeviceProp properties;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    p.num_sm = properties.multiProcessorCount;

    // Mirror the standard host heuristic so the test includes scheduler setup
    // and SplitKV reduction. Kernel scheduling does not alter these choices.
    int const packed_q = max_q * 16;
    int const m64_count = std::max(1, packed_q / 64);
    p.use_kblockm_16 = (p.is_varlen_q && ((max_q < 256 && total_q / batch < 32)
        || (max_q >= 256 && total_q / batch * 16 < 16)))
        || max_q == 1 || (max_q <= 16 && batch * m64_count <= p.num_sm && capacity_k < 17408)
        || packed_q <= 16 || (packed_q <= 32 && batch <= p.num_sm * 2)
        || (packed_q < 64 && batch <= p.num_sm);
    bool const dynamic = batch > 1 && batch <= 992 && requested_splits <= 0;
    p.num_splits_dynamic_ptr = dynamic ? static_cast<int*>(ptr[5]) + 32 : nullptr;
    int const block_m = p.use_kblockm_16 ? 16 : 64;
    int const loaded_k = p.is_local ? std::min(capacity_k, window + 1 + block_m) : capacity_k;
    int const m_blocks = (packed_q + block_m - 1) / block_m;
    int const n_blocks = (loaded_k + 63) / 64;
    int const total_m = batch * ((total_q / batch * 16 + block_m - 1) / block_m);
    int const min_m = (dynamic ? 1 : batch)
        * (((dynamic ? std::min(max_q, 16) : max_q) * 16 + block_m - 1) / block_m);
    auto split_heuristic = [&](int m) {
        return num_splits_heuristic(m, p.num_sm, 5, n_blocks, m_blocks,
            capacity_k * 320 * 2, p.is_causal || p.is_local, 128);
    };
    int const average_splits = split_heuristic(total_m);
    p.num_splits = requested_splits > 0 ? requested_splits
        : dynamic ? (average_splits > 1 ? split_heuristic(min_m) : 1) : average_splits;
    if (requested_splits <= 0 && !p.use_kblockm_16) {
        float const waves = float(batch * std::max(packed_q, 64) / 64) / p.num_sm;
        if (max_q > 16 || (waves > 1 && capacity_k < 3660 * waves)) p.num_splits = 1;
    }
    p.oaccum_row_stride = 128;
    p.oaccum_head_stride = int64_t(total_q) * 128;
    p.oaccum_split_stride = int64_t(total_q) * 16 * 128;
    p.lseaccum_head_stride = total_q;
    p.lseaccum_split_stride = int64_t(total_q) * 16;
    CHECK_CUDA(cudaMemsetAsync(ptr[5], 0, (32 + (dynamic ? batch + 1 : 0)) * sizeof(int), stream));
    bool handled = false;
#ifndef SM89_PREFILL_TEST_CONTROL
    if (p.num_splits > 1) {
        run_mha_fwd_<89, cutlass::bfloat16_t, 192, 128, true, true, false, true, false>(p, stream);
    } else {
        run_mha_fwd_<89, cutlass::bfloat16_t, 192, 128, false, true, false, true, false>(p, stream);
    }
    handled = true;
#endif
    if (!handled) {
#define LAUNCH(L, C, M) \
        if (p.num_splits > 1) launch_default<L, C, M, true>(p, stream); \
        else launch_default<L, C, M, false>(p, stream)
#define SHAPE(L, C) \
        if (p.use_kblockm_16) { LAUNCH(L, C, true); } \
        else { LAUNCH(L, C, false); }
        if (p.is_local) { SHAPE(true, false); }
        else if (p.is_causal) { SHAPE(false, true); }
        else { SHAPE(false, false); }
#undef SHAPE
#undef LAUNCH
    }
    if (p.num_splits > 1) {
        if (p.num_splits <= 32) {
            run_flash_fwd_combine<89, 8, 128, 5, false, true, cutlass::bfloat16_t, float>(p, stream, false);
        } else if (p.num_splits <= 64) {
            run_flash_fwd_combine<89, 8, 128, 6, false, true, cutlass::bfloat16_t, float>(p, stream, false);
        } else if (p.num_splits <= 128) {
            run_flash_fwd_combine<89, 8, 128, 7, false, true, cutlass::bfloat16_t, float>(p, stream, false);
        } else {
            run_flash_fwd_combine<89, 8, 128, 8, false, true, cutlass::bfloat16_t, float>(p, stream, false);
        }
    }
    int const status = int(cudaGetLastError());
    return status ? -status : p.num_splits;
}
