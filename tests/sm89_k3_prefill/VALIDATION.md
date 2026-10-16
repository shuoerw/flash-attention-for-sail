# Kimi K3 SM89 prefill validation

Validated on 2026-10-16 on node78, container `shuoer_mimo`, GPU0 ZW-M890P
(SM89, 39 CUs), under the shared GPU lock. Branch:
`feature/sm89-fwd-warp-interleave`.

The normal forward dispatcher selects the shared one-stage named-barrier
mainloop for BF16 D192/V128 expanded MHA, with 12 local Q/KV heads on TP8.
The query tile retains the existing M64/M128 selection; the KV tile is N96.
See [README.md](README.md) for the layout and numerical contracts.

## Selector and coverage

The selector has no batch-size, query-length, or context-length upper cutoffs.
It retains the supported layout checks and the minimum average live KV length
of 4,096 tokens. The redundant maximum-KV check and checks of data pointers
already supplied by the forward API were removed. Unsupported features and
short-query M16 work retain the ordinary dispatch.

The context suite contains 30 cases:

- Causal and noncausal 4K-query chunks at every power-of-two KV context from
  4K through 1M.
- Full prefill from 8K through 64K and 64K-query chunks through 1M KV.
- Mixed query lengths `[4096, 8192, 32768, 20480]`, totaling 64K, with live KV
  `[4096, 8192, 98304, 151552]`. The first two requests have no cached prefix;
  the last two have 64K and 128K prefixes.
- The same mixed queries with live KV `[4096, 8192, 524288, 1048576]`.
- Q=65,537 / KV=1,048,577 and B=257 / Q=256 / KV=4,096, crossing the removed
  limits. B counts requests per TP8 replica, not independent batches per card.

## Validation

The final source passes 51 host selector checks. All 32 device-kernel bodies
match both the context-matrix build and the preceding combined commit byte for
byte. The final rebuild passes 27 boundary cases, eight smoke cases, and three
51-sample context repeats with bit-exact output/LSE agreement against the
context-matrix library. A 401-sample short-fallback repeat also passes: 69 case
executions including the 30-case context sweep.

Every case compares complete output and LSE against the original N64 kernel.
Small inputs use full FP32 references; larger inputs check five query rows per
request and every head against all live K/V. Guard regions around output, LSE,
and scheduler buffers remain intact. N96 changes reduction grouping, so
comparison against N64 uses the tolerances documented in README; fallback and
same-geometry library comparisons are exact.

## Performance

Timing randomizes library order, evicts 256 MiB between samples, and includes an
identical original-schedule A/A control. The context sweep uses 11 samples per
variant; boundary and selected context repeats use 51. These are single-card
attention-operator measurements, not model-serving throughput.

All 30 context cases were faster than the original schedule. Representative
latency reductions against the faster original A/A control (using the
higher-sample repeat where available):

| Queries | Live KV | Attention | Reduction |
| --- | --- | --- | ---: |
| 4K | 4K | causal | 2.62% |
| 4K | 32K | causal | 11.86% |
| 4K | 1M | causal | 20.80% |
| 64K | 64K | causal | 14.53% |
| 64K | 1M | causal | 23.25% |
| 4K | 1M | noncausal prefix | 22.22% |
| mixed 64K | 4K / 8K / 96K / 148K | causal | 16.66% |
| mixed 64K | 4K / 8K / 512K / 1M | causal | 21.19% |
| 65,537 | 1,048,577 | causal | 20.91% |
| 257 requests × 256 | 4K each | causal | 6.59% |

The 27 boundary checks had a worst measured slowdown of 0.61%. No requested
4K–1M context case regressed. An initial 17-query/17-KV fallback sample was
+2.12%; its 401-sample repeat measured +0.07%, with the identical A/A control
at +0.59%. The fallback device instructions are unchanged. These are measured
case medians, not a universal bound on individual invocation latency.

## Scope and evidence

The suite exercises the normal C++ dispatcher with one KV split. It does not
validate absorbed MLA, KDA, other TP head counts, a full Python-package build,
or serving throughput. The largest normal query chunk is 64K; Q=KV=1M in one
call was not tested. No memory/race sanitizer was available.

Remote workspace: `/o-work/shuoer/dev/fa-prefill-general-20261016`.
Final library: `final-k3/libfa.so`. SHA256:

```
07607b20cadfdd48b77f2ab44f14db6c55ad65bc267843d67d9844ad2b696a7b
```

Evidence is retained in
`/Users/shor/dev/T-HEAD/flash-attention-k3-wi-backup-20261015-evidence/prefill-general-20261016/node78`.
The context matrix is `results/dense-contexts.json`; final checks are
`results/final-k3-{boundaries,smoke,context-repeat}.json`. Build manifests record
all 41 source hashes. Device-code comparisons are retained alongside the timing
samples. Use README to reproduce the suites.
