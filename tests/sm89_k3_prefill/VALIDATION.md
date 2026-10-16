# Kimi K3 SM89 warp-interleaving evaluation

Validated on 2026-10-16 in node78's `shuoer_mimo` container, on ZW-M890P
(SM89, 39 CUs). The final performance sweep used GPU7 under
`/tmp/shuoer-node78-ppu7.lock`; device process checks showed no competing GPU7
job. An earlier GPU4 sweep was interrupted after another user's job appeared
on that device. Its overlapping timings are excluded from the conclusions.

## Scope and decision

Both sides use the original N64 tile policy, the original M64/four-warp or
M128/eight-warp selection, and one shared K tile plus one shared V tile.
The interleaved candidate changes warp scheduling only. Dense prefill has no
LLC-prefetch extension. B counts requests per TP8 replica; every TP rank
processes its own 12-head shard.

The candidate does not meet the approximately 1% regression allowance.
The production dense dispatcher therefore retains the original kernel.
The N96 override and average-KV eligibility gate have been removed.
An explicit N64 candidate remains available as test mode 1; mode 8 exercises
the unchanged production dense dispatcher. Earlier combined N96/interleaving
gains are not evidence for this WI-only candidate.

## Correctness and build checks

The 30-case context sweep checks complete output and LSE bit for bit against
the original N64 schedule. Small workloads use full FP32 references; larger
workloads check sampled query rows and every head against the full live KV.
All 30 cases pass, including output, LSE, and scheduler guard regions.

Coverage includes causal and noncausal 4K chunks from 4K through 1M KV,
full prompts and 64K query chunks, the mixed queries
`[4096,8192,32768,20480]` with live KV
`[4096,8192,98304,151552]`, the longer-prefix variant ending at 1M KV,
and the removed upper-limit boundary cases.

The final build passes four dense dispatch checks and two compile-time
original-tiling assertions, plus all 61 paged MiMo selector checks. Five
unaffected dtype/dimension/layout templates compile. All 32 device-kernel
bodies match the N64 trial build exactly; its 27 original-schedule bodies
also match the preceding validated baseline. The final adapter preserves
M16 fallback in both baseline and candidate mode.

The final rebuilt adapter also passes eight smoke, 21 safety, and 27
mixed-length boundary cases, checking both the explicit candidate and the
restored production dispatcher. All 56 pass exact output/LSE, FP32 reference,
and guard-region checks.

The test requires exact native output/LSE comparisons; there is no
different-tile reduction-tolerance option. The explicit WI candidate is
checked directly against FP32 even when the production dispatcher takes
the original kernel.

## WI-only performance

The signal comparison uses 101 samples per variant; the 30-case context
sweep uses 11. Both randomize variant order, evict 256 MiB between samples,
and include an identical A/A baseline. The table uses the faster baseline
median and the 101-sample signal result where available. Positive values
mean slower. Measurements cover the attention operator, not model serving.

| New queries | Live KV | Attention | Latency change |
| --- | --- | --- | ---: |
| 4K | 4K | causal | -1.03% |
| 16K | 16K | causal | -0.49% |
| 64K | 64K | causal | -0.05% |
| 2K | 128K | noncausal | +2.04% |
| 4K | 128K | causal | +2.80% |
| 4K | 1M | causal | +3.25% |
| 64K | 1M | causal | +3.26% |
| mixed 64K | 4K / 8K / 96K / 148K | causal | +1.97% |
| mixed 64K | 4K / 8K / 512K / 1M | causal | +2.23% |

Short-case regressions also occur. These results do not justify enabling
the candidate generally or claiming that WI provides the previous
combined tiling/scheduling speedups.

## Scheduling diagnosis

A same-N64 profile of Q2K/KV128K noncausal attention shows identical tensor
instruction counts, 240 registers per thread, and 48 KiB shared memory
for baseline and candidate. WI increases issued instructions by 3.27%.
The profiler's compute-dependency stall ratio decreases from 0.197 to
0.084, while its instruction-fetch stall ratio increases from 0.062 to
0.119. These are ratios per active issue, not utilization percentages.
Tensor active cycles over elapsed cycles decrease from 41.21% to 40.33%.

This supports partial dependency hiding offset by additional execution and
scheduling overhead. It does not establish hardware saturation or prove
that no better WI schedule exists.

Three isolated, same-tiling scheduling probes passed their tested exact
numerical comparisons but failed to fix the performance problem:

- Earlier next-K refill improved some short cases, but Q2K/KV128K remained
  about 2% slower than the original.
- Allowing initial QK fragments before the previous-V-reader wait made that
  case 5.41% slower.
- Alternating leading/trailing warp membership reduced that regression to
  1.13%, but made Q512/KV8K 8.50% slower.

None of these probes is included in production source.

## Evidence and limits

Raw results, compiler manifests, source snapshots, and profiles are retained
under `/o-work/shuoer/dev/fa-kimi-wi-only-20261016`. Primary files are
`results/gpu7-probes-signal.json`, `results/gpu7-n64-contexts.json`,
`results/final-machine-code.json`, and `profiles/m{0,8}-raw.csv`.
The trial adapter's mode 8 selected WI; final test mode 1 instantiates the
same device body. Final production mode 8 retains the original schedule.

These are standalone adapter tests, not a full Python-package installation
or a serving deployment. They establish failure of this candidate's
performance acceptance test, not a universal limit of warp interleaving.
