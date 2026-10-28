#include <cstdio>
#include "heuristics.h"

// Host-only checks: no device, allocation, or launch is needed.
int main() {
    Flash_fwd_params valid{};
    int metadata = 0;
    valid.arch = 89;
    valid.is_bf16 = true;
    valid.is_causal = true;
    valid.pack_gqa = true;
    valid.num_sm = 39;
    valid.num_splits = 1;
    valid.b = valid.b_k = 1;
    valid.total_q = valid.seqlen_q = 512;
    valid.seqlen_k = 65536;
    valid.d = 192;
    valid.dv = 128;
    valid.h = 16;
    valid.h_k = 1;
    valid.page_table = valid.cu_seqlens_q = valid.seqused_k = &metadata;
    valid.page_size = 64;
    valid.q_row_stride = 16 * 192;
    valid.q_head_stride = 192;
    valid.k_row_stride = 192;
    valid.k_batch_stride = 64 * 192;
    valid.v_row_stride = 128;
    valid.v_batch_stride = 64 * 128;
    valid.v_dim_stride = 1;
    valid.o_row_stride = 16 * 128;
    valid.o_head_stride = 128;

    int checks = 0;
    auto check = [&](Flash_fwd_params const& p, bool expected, char const* label) {
        ++checks;
        if (flash::use_sm89_prefill_fragment_stagger(p) != expected) {
            std::fprintf(stderr, "Incorrect prefill schedule: %s\n", label);
            return false;
        }
        return true;
    };
    if (!check(valid, true, "eligible")) return 1;
#define REJECT(field, value) { \
    auto p = valid; p.field = value; \
    if (!check(p, false, #field)) return 1; \
}
    REJECT(arch, 80);
    REJECT(is_bf16, false);
    REJECT(is_causal, false);
    REJECT(is_local, true);
    REJECT(pack_gqa, false);
    REJECT(use_kblockm_16, true);
    REJECT(use_kblockm_128, true);
    REJECT(use_kblockn_16, true);
    REJECT(is_qsa, true);
    REJECT(num_sm, 0);
    REJECT(num_splits, 0);
    REJECT(num_splits, 257);
    REJECT(b, 0);
    REJECT(total_q, 0);
    REJECT(total_q, 464);
    REJECT(seqlen_q, 1);
    REJECT(seqlen_q, 4);
    REJECT(seqlen_q, 16);
    REJECT(seqlen_k, 0);
    REJECT(seqlen_k, 64);
    REJECT(seqlen_k, 512);
    REJECT(seqlen_k, 576);
    REJECT(seqlen_k, 1023);
    REJECT(d, 128);
    REJECT(dv, 192);
    REJECT(h, 8);
    REJECT(h_k, 2);
    REJECT(page_table, nullptr);
    REJECT(page_size, 128);
    REJECT(cu_seqlens_q, nullptr);
    REJECT(seqused_k, nullptr);
    REJECT(cu_seqlens_k, &metadata);
    REJECT(seqused_q, &metadata);
    REJECT(leftpad_k, &metadata);
    REJECT(kv_batch_idx, &metadata);
    REJECT(knew_ptr, &metadata);
    REJECT(vnew_ptr, &metadata);
    REJECT(qv_ptr, &metadata);
    REJECT(rotary_cos_ptr, &metadata);
    REJECT(softcap, 1.f);
    REJECT(attention_chunk, 128);
    REJECT(q_row_stride, 16 * 256);
    REJECT(q_head_stride, 256);
    REJECT(k_row_stride, 256);
    REJECT(k_batch_stride, 64 * 256);
    REJECT(v_row_stride, 192);
    REJECT(v_batch_stride, 64 * 192);
    REJECT(v_dim_stride, 2);
    REJECT(o_row_stride, 16 * 192);
    REJECT(o_head_stride, 192);
#undef REJECT
    for (int kv : {1024, 32704, 32768, 32769, 1048576, 1048577}) {
        auto p = valid;
        p.seqlen_k = kv;
        if (!check(p, true,
                   "context coverage")) return 1;
    }
    auto p = valid;
    p.b = p.b_k = 257;
    p.total_q = 257 * 512;
    if (!check(p, true, "large request batch")) return 1;
    p = valid;
    p.total_q = p.seqlen_q = 65537;
    p.seqlen_k = 1048640;
    if (!check(p, true, "large query chunk")) return 1;
    p = valid;
    p.total_q = 465;
    if (!check(p, true, "query threshold")) return 1;
    p = valid;
    p.is_varlen_q = true;
    p.num_splits = 16;
    p.num_splits_dynamic_ptr = &metadata;
    p.s_aux_ptr = &metadata;
    if (!check(p, true, "ragged split with sinks")) return 1;
    std::printf("PASS %d host eligibility checks\n", checks);
}
