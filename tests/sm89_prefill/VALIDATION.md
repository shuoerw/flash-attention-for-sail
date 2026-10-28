# MiMo prefill fragment-order staggering: isolated validation

Validated on 2026-10-28 in node78's `shuoer_mimo` container, using exclusively
locked GPU4 (ZW-M890P/SM89, 39 CUs, HGGC 13.0). Common source baseline:
`3c77a993ad20334a7b6f3cd2d9f79a58eb4e347d`.

This branch changes only fragment-order staggering. Both sides use M64/N64, four warps,
one K/V stage and the original KV split selection. Named-barrier WI is absent.
BF16 paged GQA uses Hq16/Hkv1, Dqk192/Dv128 and 64-token pages. This is the head
shard processed by one card of a TP8 replica; request batch is shared by TP ranks.

**The isolated change regresses fresh 4K–64K prefill by 2.70–3.35% in this run.**
This branch makes the technique independently reviewable; the results do not
support enabling it as a performance improvement on the original one-stage tile.
The earlier bundled M128/eight-warp/three-stage result does not apply here.

## Matched fresh measurements

B=1, Q=KV, causal attention, 31 randomized graph samples per variant, seed 2821.
Each graph contains one attention operator including scheduler setup. Native
variants share tensors and output/workspace addresses. Each sample follows
256 MiB cache eviction. The largest identical-control A/A difference is 0.123%.
Negative latency changes mean faster. These are single-run operator measurements,
not full-model throughput or a claim of universal benefit.

| Fresh tokens | Control (ms) | fragment-order staggering (ms) | Latency change |
| ---: | ---: | ---: | ---: |
| 4,096 | 0.4387 | 0.4505 | +2.70% |
| 8,192 | 1.6262 | 1.6723 | +2.83% |
| 16,384 | 6.2683 | 6.4785 | +3.35% |
| 32,768 | 24.8391 | 25.6374 | +3.21% |
| 65,536 | 108.0408 | 111.2749 | +2.99% |

## Cached-prefix checks

These additional checks use attention sinks and 11 samples. They exercise the
same isolated implementation with K/V fetching beyond the fresh-context sweep.

| New Q | Total KV | Control (ms) | Candidate (ms) | Latency change |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 32,768 | 6.2753 | 6.5412 | +4.24% |
| 4,096 | 65,536 | 14.7791 | 15.4047 | +4.23% |
| 4,096 | 131,072 | 30.1072 | 31.2845 | +3.91% |
| 512 | 32,769 | 0.9745 | 0.9870 | +1.29% |
| 1,025 | 65,539 | 4.2071 | 4.2630 | +1.33% |

## Correctness and compatibility

- 61 host eligibility checks pass.
- 167 unaffected full-dispatcher kernel instruction bodies are byte-identical
  to the validated baseline build. Only the two intended specializations differ.
- Additional dense/paged FP16/BF16 paths with head dimensions 64, 128, 192 and
  256 compile successfully.
- 22 small full-FP32 checks and 8 additional full-FP32 partial-page checks pass.
  The latter include eligible Q1024/K65539 and ragged Q1025/K65539–66001 inputs.
- All 18 operator cases pass: fresh and partial-page lengths through 64K,
  cached prefixes through 131073, mixed requests with shuffled compact pages,
  and explicit 16-way splitting.
- Eight live-length patterns pass replay of the same captured graph, including
  zero-length KV requests. Buffer and workspace guard checks pass.
- Fragment rotation changes accumulation order. FP32 output checks use
  atol=0.025/rtol=0.025; LSE uses atol=0.003/rtol=0.001. Full comparisons with
  the installed operator use output atol=0.016/rtol=0.016 and the same LSE bounds.
  Full native baseline comparisons also pass the FP32-check tolerances.

The initial graph command supplied invalid initial KV lengths and was rejected
before launching kernels; the corrected uniform initial allocation passes all
eight changed-length replays. The first compile-coverage command mixed absolute
and relative build paths; rerunning with the build manifest's relative path
passes. Raw logs retain both setup failures and successful reruns.

No sanitizer, full Python-package build or SGLang serving run was performed.
The widest checked context is 131073, with fresh contexts limited to 64K.

## Evidence and reproduction

Use the commands in README.md. To reproduce full-reference partial-page checks,
run `replay.py --partial-pages --correctness-only` with control and candidate
libraries. Use `--case 1 Q Q` for each fresh length and `--repeats 31 --seed 2821`
with `--only-libraries --shared-workspace --compare-baseline control`.

Raw JSON, samples, commands, source hashes, build logs and both validation
drivers are retained inside the container at:
`/o-work/shuoer/dev/mimo-prefill-review-20261028`.
`results/fresh.json`, `prefix.json`, `tails.json`, `mixed.json`, `split.json`,
`graph-final.json` and `full-reference.json` contain the numerical results.
`build/stagger/build.json` and `compile_coverage.json` record successful builds.

Candidate library SHA256: `b6a0b36f9d0b637471cf8da2e77ad6edc177e85d47e32c237212639af1fa190d`.
