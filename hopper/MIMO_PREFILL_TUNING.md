# MiMo prefill optimization port

This ports the MiMo optimization developed on the `de4a6bd` public release to
the PPU-Libraries `v2.8.2` code based on `48ab6ff`. The scheduler synchronization
fix is a separate preceding commit. No diagnostic instrumentation is included.

## What changes

For eligible BF16 cached-prefix prefill, use a 128x64 tile, eight warps, three
pipeline stages, and Q in registers. Odd warps start halfway through the QK/PV
reduction fragments and wrap around; even warps retain ascending order. Each
fragment is consumed once. This reduces contention between neighboring warps'
shared-memory matrix loads, but changes floating-point accumulation order.

The build switch `FLASH_ATTENTION_PPU_MIMO_PREFILL` defaults to `TRUE` on PPU.
Set it to `FALSE` **at build time** to compare against this repository's current
baseline while keeping the independent scheduler fix enabled.

The new dispatch requires all of the following:

- SM89, BF16, exact Q/K dimension 192 and V dimension 128.
- Paged KV with 64-token pages and contiguous V dimensions.
- Uniform query lengths with `cu_seqlens_q`, Q length at least 4,096, KV capacity
  at least 65,536 and greater than Q length, and no used-Q override or left pad.
- Packed GQA with exactly 16 query heads per KV head.
- Causal full attention, one split, no KV append, softcap, attention chunk, or QSA.
- At least eight waves of 128-row packed-query tiles under the original heuristic.

Dispatch examines only host metadata, not device tensor contents, and is safe
to use during graph capture. KV capacity is not necessarily the live KV length.

## Adaptation to the newer repository

This is a source port, not a cherry-pick: the original two changes were still
uncommitted in the public-release checkout. The newer repository already has
additional MHA/FP8 tiling, QSA, FP8 V-direct, and softmax-rescale optimizations.

The port uses a separate guarded launch instead of narrowing the shared
`use_kblockm_128` flag. The existing generic tile table, scheduler traversal,
QSA configuration, and FP8 V-direct path are retained. The compiler's
`mmatiestrictly` regions are also retained around the GEMM loops. Nonmatching
requests use the existing dispatcher and unstaggered reduction order.

## Validation

The original release measured approximately 24–25% lower attention-kernel time
on the tested long-context shapes. Those historical measurements are **not** a
performance result for this newer codebase or compiler. This port still needs
a PPU build, numerical tests, and baseline-versus-tuned performance validation.

Local Python syntax checks and `git diff --check` passed during the source
port. These checks do not compile or execute the device kernels.

The migrated `test_bf16_paged_prefill.py` covers partial pages/query tiles, pipeline
remainders, uniform and ragged query batches, short-context/decode/SWA fallbacks,
short visible prefixes, and graph replay with changed query data. Its numerical
tolerances are the original optimization's criteria, not a replacement for a
client's stricter precision requirements.

From `hopper/`, with a matching installed extension:

```bash
python -m pytest -q test_bf16_paged_prefill.py
python benchmark_mimo_prefill.py --label baseline
python benchmark_mimo_prefill.py --label tuned --profile tuned.trace.json
```

Run baseline and tuned measurements in separate processes against their
respective builds, on the same GPU and inputs. The benchmark checks selected
output rows against FP32 before timing and records the loaded extension hash.
