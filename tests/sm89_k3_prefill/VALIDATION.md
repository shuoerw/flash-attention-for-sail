# Kimi K3 SM89 warp-interleaving validation

Validated on 2026-10-18 in node78's `shuoer_mimo` container, on ZW-M890P
(SM89, 39 CUs). GPU4 and GPU7 were used under their shared device locks.
Device-process checks showed no competing jobs on those devices. The resident
serving processes on GPUs 0–3 were not changed.

## Scope and decision

Every comparison uses the original N64 tile policy, original M64/four-warp
or M128/eight-warp selection, and one shared K tile plus one shared V tile.
There are no changes to tiling, stage count, split count, math, or prefetch.
B counts requests per TP8 replica; each TP rank processes its 12-head shard.

The revised WI schedule provides measurable gains on the original tiles.
It removes the previous 2–3% regression in the two repeated cached-prefix
signal cases. Across the 30-case context sweep, observed latency changes
range from -4.15% to +1.07%, including mixed 64K chunks and 1M KV.

The production dense dispatcher remains on the original schedule: short
inputs and some small ragged batches still exceed the approximately 1%
regression allowance. No new workload eligibility heuristic is introduced.
Test mode 1 exercises WI; mode 8 exercises the unchanged production dispatcher.

## Scheduling correction

Dense prefill uses three named-barrier phases in the existing core loop:

1. The leading group completes next K before joining `VReadersDone`.
   That phase also retires the trailing group's previous PV reads.
2. The leading group issues the existing V copy before entering QK and
   releases the trailing group after two reduction fragments.
3. Trailing warps wait at `KReadersDone` after QK. The leading group joins
   after completing the V copy, so this phase also publishes V. No separate
   V-ready wait is needed before PV.

This removes the dense leading-group-only barrier and separate V-ready
phase. Moving the V copy out of the QK fragment hook removes a role-dependent
branch from that instruction sequence. Paged MiMo keeps its existing protocol.

## Correctness and build validation

All 30 context cases pass complete, bit-exact output and LSE comparisons,
FP32 reference checks, and output/LSE/scheduler guard checks. Small problems
use full FP32 references; large problems use sampled query rows, all heads,
and the full live KV. Coverage includes:

- Causal and noncausal Q4K with KV from 4K to 1M.
- Fresh prompts and query chunks up to 64K.
- Mixed queries `[4096,8192,32768,20480]`, with live KV
  `[4096,8192,98304,151552]` or `[4096,8192,524288,1048576]`.
- Partial-tile Q65537/KV1048577 and 257 short requests sharing a chunk.

The cleaned build also passes 8 smoke, 21 safety, and 27 boundary cases.
These check mode 1 and production mode 8, including empty requests, M16
fallback, different head counts, and contiguous/strided V.

All 32 cleaned dense device-kernel bodies are byte-identical to the tested
candidate. All 28 baseline/scheduler bodies checked against the previous
build are unchanged. Four dense host-dispatch checks and two compile-time
original-tile assertions pass. Five unaffected template paths compile.
All 171 before/after paged MiMo kernel bodies are byte-identical, and all
61 paged selector contract checks pass. The dense correction therefore
preserves the compiled MiMo kernels and their selection policy.

## Operator performance

Numbers below are latency changes relative to the faster identical A/A
baseline median; negative means faster. Variant order is randomized, with
256 MiB eviction before each sample. Timings include semaphore initialization
and the attention operator, excluding projections, prefix assembly, Python
API overhead, and serving scheduling.

The following comparisons use 101 samples per variant in one process,
with identical inputs and original tiling:

| New Q | Live KV | Causal | Previous WI | Fixed WI |
| --- | --- | --- | ---: | ---: |
| 4K | 4K | yes | -1.28% | -3.18% |
| 16K | 16K | yes | -0.52% | -2.93% |
| 2K | 32K | yes | +2.31% | -1.29% |
| 2K | 128K | no | +2.04% | -0.90% |

A separate 51-sample run measured -0.79% and -0.91% for the two cached-prefix
cases. Within-run paired-bootstrap 95% intervals in the 101-sample run are
[-1.46%, -0.95%] and [-0.96%, -0.83%]. These intervals describe timing
uncertainty within that run, not all possible inputs or system conditions.

The 30-case context sweep uses 11 samples per variant:

| New Q | Live KV | Causal | Fixed WI |
| --- | --- | --- | ---: |
| 4K | 8K | yes | +1.07% |
| 4K | 16K | yes | -1.99% |
| 4K | 32K | yes | +0.68% |
| 4K | 128K | yes | -0.62% |
| 4K | 1M | yes | -0.17% |
| 8K | 8K | yes | -4.15% |
| 32K | 32K | yes | -3.75% |
| 64K | 64K | yes | -2.41% |
| 64K | 1M | yes | -1.53% |
| mixed 64K | 4K / 8K / 96K / 148K | yes | -1.53% |
| mixed 64K | 4K / 8K / 512K / 1M | yes | -0.85% |
| 65537 | 1048577 | yes | -0.62% |

Isolated 101-sample repeats of Q4K/KV8K and Q4K/KV32K measured -0.10% and
-0.63%. The full-sweep +1.07% observation is retained as the measured upper
regression, rather than replacing it with a favorable repeat. Selecting cases
changes the random-input stream because replay seeds once per invocation.

Short-case limitations remain: the final safety run measured +8.02% for
Q64/KV64, +4.15% for Q127/KV127, and +2.48% for Q705/KV705. The boundary run
measured +4.34% for 31 Q64/KV128 requests plus one Q512/KV64K request.
These results prevent unconditional dense enablement.

## Hardware evidence

Fixed-seed profiles of causal Q4K/KV32K use exactly the same tensors for all
three schedules. Each uses 240 registers/thread and 48 KiB shared memory,
with identical tensor and floating-point multiplication instruction counts.

| Metric | Original | Previous WI | Fixed WI |
| --- | ---: | ---: | ---: |
| Issued instructions | 734,348,125 | 763,497,805 | 752,441,053 |
| Branch instructions | 4,587,727 | 14,915,791 | 11,966,671 |
| Compute-dependency stall ratio | 0.181 | 0.083 | 0.087 |
| Instruction-fetch stall ratio | 0.061 | 0.117 | 0.105 |
| Tensor active / elapsed | 48.05% | 46.78% | 48.33% |
| FP32/FP16 ALU active / elapsed | 15.75% | 15.39% | 15.67% |
| SFU active / elapsed | 20.61% | 19.86% | 20.59% |

Stall ratios are per active issue, not utilization percentages. The measured
gain comes from retaining dependency overlap while reducing synchronization
and control-flow overhead. The compiler also overlaps softmax and PV within
a warp. These profiles do not establish saturation or a universal WI limit.
Profiler durations are not used as the operator benchmark results.

## Reproduction and evidence

Source snapshots, build hashes, complete samples, rejected scheduling probes,
and profiles are under `/o-work/shuoer/dev/fa-kimi-wi-only-20261018`.

Primary files:

- `results/winner-repeat.json`, `results/winner-bootstrap.json`.
- `results/load-setup-contexts.json`, `results/context-summary.json`.
- `results/final-{smoke,safety,boundaries,prefix-repeat}.json`.
- `results/final-body-identity.json`, `results/baseline-body-identity.json`.
- `results/mimo-body-identity.json`, `mimo-{before,after}/build.json`.
- `profiles/seed1731-{original,old-wi,final}-raw.csv`.
- `final/build.json`, `final/compile_coverage.json`.

The `load-setup` experimental library and cleaned `final` library emit identical
device instructions. Experimental variants are retained only in the external
evidence directory; they are not added to production source. Validation uses
standalone adapters, not a package installation or serving deployment.
