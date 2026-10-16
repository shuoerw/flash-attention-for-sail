#include "flash_fwd_launch_template.h"
#include "flash_prepare_scheduler.cu"

// Instantiate unaffected dense/paged, FP16/BF16, and head-dimension paths.
// This library is compiled for template coverage; these functions are not run.
template<int HeadDim, int ValueDim, class Element, bool Paged>
void instantiate_default(Flash_fwd_params& p, hggcStream_t stream) {
    run_flash_fwd<89, HeadDim, ValueDim, 1, Element, Element,
        true, false, false, Paged, Paged, Paged, 64,
        false, false, Paged, false, false, false, false, false>(p, stream);
}

template void instantiate_default<64, 64, cutlass::bfloat16_t, false>(Flash_fwd_params&, hggcStream_t);
template void instantiate_default<128, 128, cutlass::bfloat16_t, false>(Flash_fwd_params&, hggcStream_t);
template void instantiate_default<128, 128, cutlass::half_t, false>(Flash_fwd_params&, hggcStream_t);
template void instantiate_default<192, 128, cutlass::half_t, true>(Flash_fwd_params&, hggcStream_t);
template void instantiate_default<256, 256, cutlass::bfloat16_t, true>(Flash_fwd_params&, hggcStream_t);
