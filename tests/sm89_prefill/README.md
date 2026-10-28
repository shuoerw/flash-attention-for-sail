# MiMo prefill fragment staggering

This branch isolates fragment staggering on SM89/PPU890 BF16 paged GQA:
Hq16/Hkv1, Dqk192/Dv128, contiguous 64-token pages and causal full attention.
It uses the existing forward dispatcher and original M64/N64, four-warps,
one-stage kernel. Tile geometry, KV splitting and scheduler are unchanged.
The branch does not contain named-barrier warp interleaving.

Odd warps start halfway through each QK/PV reduction and wrap around; even
warps retain ascending fragment order. Every fragment is consumed exactly once.
The QK load callback still runs after the first fragment load. The purpose is
to stagger neighboring warps' shared-memory matrix accesses. Floating-point
accumulation order changes, so validation uses numerical tolerances rather than
requiring equality with the original kernel. No LLC prefetch is added.

`hopper/heuristics.h` checks layout eligibility before the existing dispatcher
selects the template specialization. Unsupported layouts and decode use the
default kernel. The inherited minimum query-work gate is three M64 tiles per SM.

## Reproduction

With the PPU SDK CUDA_SDK/bin directory on PATH:

```bash
python3 tests/sm89_prefill/build.py --source . --actlize "$ACTLIZE" --output "$BUILD/control" --control
python3 tests/sm89_prefill/build.py --source . --actlize "$ACTLIZE" --output "$BUILD/release"
python3 tests/sm89_prefill/compile_coverage.py --build "$BUILD/release"
CUDA_VISIBLE_DEVICES=0 python3 tests/sm89_prefill/validate.py --root "$BUILD" --output "$BUILD/results"
```

Use an idle GPU under its shared lock. The standalone adapter executes the
ordinary forward dispatcher, scheduler and split reduction; no package install
is needed. The installed attention operator supplies another numerical reference.
See VALIDATION.md for measured results and limits of qualification.
