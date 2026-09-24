#pragma once

#include <cstdint>
#include <tuple>

#include "flash.h"

namespace flash {

// {block M, block N, warps, pipeline stages, Q in registers}.
inline constexpr std::tuple<int, int, int, int, bool> mimo_prefill_tile{128, 64, 8, 3, true};

inline bool use_mimo_prefill(Flash_fwd_params const& p) {
#if defined(USE_PPU) && USE_AIU && defined(FLASHATTENTION_PPU_MIMO_PREFILL)
    // Host metadata only: never read device tensor contents during dispatch.
    // This is separate from use_kblockm_128 so other large-tile paths keep
    // their existing heuristics, including FP8 and MHA prefill.
    return p.arch == 89 && p.is_bf16 && !p.is_e4m3 && !p.is_qsa &&
        p.d == 192 && p.dv == 128 && p.b > 0 && p.h_k > 0 && p.num_sm > 0 &&
        p.page_table && p.page_size == 64 && p.cu_seqlens_q &&
        !p.is_varlen_q && p.total_q == int64_t(p.b) * p.seqlen_q &&
        !p.seqused_q && !p.leftpad_k && p.v_dim_stride == 1 &&
        p.seqlen_q >= 4096 && p.seqlen_k >= 65536 && p.seqlen_k > p.seqlen_q &&
        p.h == 16 * p.h_k && p.pack_gqa && p.is_causal && !p.is_local &&
        p.attention_chunk == 0 && p.softcap == 0.f && !p.knew_ptr &&
        p.num_splits == 1 &&
        int64_t(p.b) * p.h_k * ((int64_t(p.seqlen_q) * 16 + 127) / 128) >=
            int64_t(p.num_sm) * 8;
#else
    (void)p;
    return false;
#endif
}

}  // namespace flash
