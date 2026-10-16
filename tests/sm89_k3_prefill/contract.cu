#include <cstdio>
#include "heuristics.h"

// WI candidates use the ordinary tile geometry and shared-memory stage count.
static_assert(tile_size_fwd_ppu(89, 192, 128, true, false, 2,
    false, true, false, false, false, false, false) == std::make_tuple(64, 64, 4, 1, true));
static_assert(tile_size_fwd_ppu(89, 192, 128, true, false, 2,
    false, true, false, false, false, false, true) == std::make_tuple(128, 64, 8, 1, true));

int main() {
    Flash_fwd_params p{};
    int storage = 0;
    p.arch = 89;
    p.is_bf16 = true;
    p.num_sm = 39;
    p.num_splits = 1;
    p.b = p.b_k = 1;
    p.total_q = p.seqlen_q = p.total_k = p.seqlen_k = 4096;
    p.d = 192; p.dv = 128;
    p.h = p.h_k = 12;
    p.cu_seqlens_q = p.cu_seqlens_k = &storage;
    p.q_head_stride = p.k_head_stride = 192;
    p.q_row_stride = p.k_row_stride = 12 * 192;
    p.v_dim_stride = 1;
    p.o_head_stride = 128; p.o_row_stride = 12 * 128;
    int checks = 0;
    for (bool causal : {false, true}) {
        for (int v_stride : {128, 256}) {
            p.is_causal = causal;
            p.use_kblockm_128 = causal;
            p.v_head_stride = v_stride; p.v_row_stride = 12 * v_stride;
            if (flash::sm89_prefill_schedule(p) != flash::Sm89PrefillSchedule::Default) {
                std::fprintf(stderr, "Dense prefill must retain the original schedule\n");
                return 1;
            }
            ++checks;
        }
    }
    std::printf("PASS %d dense dispatch checks and two tile static assertions\n", checks);
}
