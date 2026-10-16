#include <cstdio>
#include "heuristics.h"

// Host metadata coverage. No GPU allocation or launch is needed.
int main() {
    Flash_fwd_params valid{};
    int storage = 0;
    valid.arch = 89;
    valid.is_bf16 = true;
    valid.is_causal = true;
    valid.num_sm = 39;
    valid.num_splits = 1;
    valid.b = valid.b_k = 1;
    valid.total_q = valid.seqlen_q = 512;
    valid.total_k = valid.seqlen_k = 8192;
    valid.d = 192; valid.dv = 128;
    valid.h = valid.h_k = 12;
    valid.q_ptr = valid.k_ptr = valid.v_ptr = valid.o_ptr = &storage;
    valid.softmax_lse_ptr = &storage;
    valid.tile_count_semaphore = valid.cu_seqlens_q = valid.cu_seqlens_k = &storage;
    valid.q_head_stride = valid.k_head_stride = 192;
    valid.q_row_stride = valid.k_row_stride = 12 * 192;
    valid.v_head_stride = 256; valid.v_row_stride = 12 * 256; valid.v_dim_stride = 1;
    valid.o_head_stride = 128; valid.o_row_stride = 12 * 128;
    int checks = 0;
    auto check = [&](Flash_fwd_params const& p, bool expected, char const* label) {
        ++checks;
        bool const selected = flash::sm89_prefill_schedule(p) == flash::Sm89PrefillSchedule::WarpInterleaved;
        if (selected == expected) return true;
        std::fprintf(stderr, "Eligibility mismatch: %s\n", label);
        return false;
    };
    if (!check(valid, true, "strided V")) return 1;
#define REJECT(field, value) { auto p = valid; p.field = value; \
    if (!check(p, false, #field)) return 1; }
    REJECT(arch, 80); REJECT(is_bf16, false); REJECT(is_local, true);
    REJECT(is_qsa, true); REJECT(pack_gqa, true);
    REJECT(d, 576); REJECT(dv, 512); REJECT(h, 6); REJECT(h_k, 1);
    REJECT(num_sm, 0); REJECT(num_splits, 2);
    REJECT(use_kblockm_16, true); REJECT(use_kblockn_16, true);
    REJECT(b, 0); REJECT(total_q, 0); REJECT(seqlen_q, 16);
    REJECT(total_k, 4095);
    REJECT(cu_seqlens_q, nullptr); REJECT(cu_seqlens_k, nullptr);
    REJECT(page_table, &storage); REJECT(seqused_q, &storage); REJECT(seqused_k, &storage);
    REJECT(leftpad_k, &storage); REJECT(kv_batch_idx, &storage);
    REJECT(knew_ptr, &storage); REJECT(vnew_ptr, &storage); REJECT(qv_ptr, &storage);
    REJECT(rotary_cos_ptr, &storage); REJECT(s_aux_ptr, &storage);
    REJECT(softcap, 1.f); REJECT(attention_chunk, 128);
    REJECT(q_head_stride, 256); REJECT(q_row_stride, 12 * 256);
    REJECT(k_head_stride, 256); REJECT(k_row_stride, 12 * 256);
    REJECT(v_head_stride, 192); REJECT(v_row_stride, 12 * 192); REJECT(v_dim_stride, 2);
    REJECT(o_head_stride, 192); REJECT(o_row_stride, 12 * 192);
#undef REJECT
    auto p = valid;
    p.v_head_stride = 128; p.v_row_stride = 12 * 128;
    if (!check(p, true, "contiguous V")) return 1;
    p = valid; p.is_causal = false;
    if (!check(p, true, "prefix chunk")) return 1;
    p = valid; p.use_kblockm_128 = true;
    if (!check(p, true, "M128")) return 1;
    for (int average : {4095, 4096, 4097}) {
        p = valid; p.b = p.b_k = 32; p.total_q = 32 * 512;
        p.total_k = 32 * average; p.seqlen_k = 65536; p.is_varlen_q = true;
        if (!check(p, average >= 4096, "uneven average KV")) return 1;
    }
    p = valid; p.seqlen_k = p.total_k = 1048576;
    if (!check(p, true, "1M KV")) return 1;
    p = valid; p.total_q = p.seqlen_q = 65536; p.total_k = p.seqlen_k = 65536;
    if (!check(p, true, "64K full prefill")) return 1;
    p = valid; p.b = p.b_k = 257; p.total_q = 257 * 512; p.total_k = 257 * 8192;
    if (!check(p, true, "large request batch")) return 1;
    p = valid; p.total_q = p.seqlen_q = 65537; p.total_k = p.seqlen_k = 1048577;
    if (!check(p, true, "large query chunk and KV")) return 1;
    std::printf("PASS %d dense host eligibility checks\n", checks);
}
