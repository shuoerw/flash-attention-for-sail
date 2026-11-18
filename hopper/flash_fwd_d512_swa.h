#pragma once

#include "flash.h"

#if defined(USE_PPU) && USE_AIU && defined(FLASHATTENTION_DISABLE_SM90) \
    && !defined(FLASHATTENTION_DISABLE_SM8x) && !defined(FLASHATTENTION_DISABLE_LOCAL) \
    && !defined(FA3_HLLM_BUILD)
#define FLASHATTENTION_PPU_D512_SWA
// Reserve up to a 512-token prefix plus enough padding to align every request
// start to the 16-row gather grid, including empty requests.
constexpr int kD512GatherRows = 16;
constexpr int kD512GatherSuffix = 528;
constexpr int64_t d512_swa_workspace_tokens(int64_t total_q, int batch) {
    return (total_q + kD512GatherRows - 1) / kD512GatherRows * kD512GatherRows
        + int64_t(batch) * kD512GatherSuffix;
}
void run_flash_fwd_d512_swa(Flash_fwd_params const& params, void* workspace, hggcStream_t stream);
#endif
