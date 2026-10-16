#include <cuda_runtime.h>
#include <cmath>
#include "flash_fwd_launch_template.h"
#include "flash_prepare_scheduler.cu"

// Standalone dense K3 prefill adapter: mode selects an identical-input ablation.
template<bool Causal, bool M128, flash::Sm89PrefillSchedule Schedule, bool M16 = false>
void launch(Flash_fwd_params& p, cudaStream_t stream) {
    run_flash_fwd<89, 192, 128, 1, cutlass::bfloat16_t, cutlass::bfloat16_t,
        Causal, false, false, true, false, false, 1, false, false,
        false, false, false, M128, M16, false, false,
        flash::QsaConfig, Schedule>(p, stream);
}

extern "C" int fa_k3_run(void** ptr, int const* cfg, cudaStream_t stream) {
    Flash_fwd_params p{};
    int const mode = cfg[0];
    p.b = p.b_k = cfg[1];
    p.total_q = cfg[2]; p.total_k = cfg[3];
    p.seqlen_q = cfg[4]; p.seqlen_k = cfg[5];
    p.h = p.h_k = cfg[6];
    p.is_causal = cfg[7];
    p.q_ptr = ptr[0]; p.k_ptr = ptr[1]; p.v_ptr = ptr[2];
    p.o_ptr = ptr[3]; p.softmax_lse_ptr = ptr[4];
    p.tile_count_semaphore = static_cast<int*>(ptr[5]);
    p.cu_seqlens_q = static_cast<int*>(ptr[6]);
    p.cu_seqlens_k = static_cast<int*>(ptr[7]);
    p.d = p.d_rounded = 192; p.dv = p.dv_rounded = 128;
    p.q_row_stride = cfg[8]; p.q_head_stride = cfg[9];
    p.k_row_stride = cfg[10]; p.k_head_stride = cfg[11];
    p.v_row_stride = cfg[12]; p.v_head_stride = cfg[13]; p.v_dim_stride = 1;
    p.o_row_stride = p.h * 128; p.o_head_stride = 128;
    p.scale_softmax = 1.f / std::sqrt(192.f);
    p.p_dropout = p.rp_dropout = 1;
    p.is_bf16 = true; p.arch = 89; p.num_splits = 1;
    p.window_size_left = p.seqlen_k - 1;
    p.window_size_right = p.is_causal ? 0 : p.seqlen_k - 1;
    p.is_varlen_q = p.total_q / p.b != p.seqlen_q;
    cudaDeviceProp properties;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    p.num_sm = properties.multiProcessorCount;
    p.use_kblockm_16 = (p.is_varlen_q && ((p.seqlen_q < 256 && p.total_q / p.b < 32)
        || (p.seqlen_q >= 256 && p.total_q / p.b < 16))) || p.seqlen_q == 1
        || (p.seqlen_q <= 16 && p.b * p.h_k * std::max(1, p.seqlen_q / 64) <= p.num_sm && p.seqlen_k < 17408)
        || p.seqlen_q <= 16 || (p.seqlen_q <= 32 && p.b * p.h_k <= p.num_sm * 2)
        || (p.seqlen_q < 64 && p.b * p.h_k <= p.num_sm);
    // Preserve the regular BF16 D192 MHA prefill M64/M128 selection.
    bool m128 = p.is_causal && p.seqlen_q == p.seqlen_k
        && (p.is_varlen_q ? double(p.total_q) / p.b > 704
            : (p.seqlen_q > 704 && p.b * p.h * ((p.seqlen_q + 127) / 128) / p.num_sm >= 8));
    p.use_kblockm_128 = m128;
    CHECK_CUDA(cudaMemsetAsync(ptr[5], 0, 32 * sizeof(int), stream));
    if (mode == 8) {
        run_mha_fwd_<89, cutlass::bfloat16_t, 192, 128, false, false, false, false, false>(p, stream);
        int status = int(cudaGetLastError());
        bool const interleaved = flash::sm89_prefill_schedule(p) != flash::Sm89PrefillSchedule::Default;
        return status ? -status : (interleaved ? 1000 : 0) + (p.use_kblockm_16 ? 16 : m128 ? 128 : 64);
    }
    if (mode == 0 && p.use_kblockm_16) {
        if (p.is_causal) launch<true, false, flash::Sm89PrefillSchedule::Default, true>(p, stream);
        else launch<false, false, flash::Sm89PrefillSchedule::Default, true>(p, stream);
        int status = int(cudaGetLastError());
        return status ? -status : 16;
    }
#define DISPATCH(C, M) \
    if (mode == 1) launch<C, M, flash::Sm89PrefillSchedule::WarpInterleaved>(p, stream); \
    else launch<C, M, flash::Sm89PrefillSchedule::Default>(p, stream)
    if (p.is_causal) {
        if (m128) { DISPATCH(true, true); } else { DISPATCH(true, false); }
    } else {
        if (m128) { DISPATCH(false, true); } else { DISPATCH(false, false); }
    }
#undef DISPATCH
    int status = int(cudaGetLastError());
    return status ? -status : m128 ? 128 : 64;
}
