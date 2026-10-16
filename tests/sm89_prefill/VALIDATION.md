# MiMo SM89 prefill validation

Validated on 2026-10-16 on node78, container `shuoer_mimo`, GPU0 ZW-M890P
(SM89, 39 CUs), under the shared GPU lock. Branch:
`feature/sm89-fwd-warp-interleave`.

The normal forward dispatcher selects the one-stage named-barrier mainloop for
BF16 D192/V128 paged GQA, Hq16/Hkv1, with contiguous 64-token KV pages. MiMo and
Kimi share the mainloop and existing dispatch infrastructure. See
[README.md](README.md) for the barrier and layout contracts.

## Selector

There are no batch-size, query-length, or context-length upper cutoffs. Query
work is calculated in 64-bit arithmetic. The selector retains its supported
layout requirements, minimum 1K KV capacity, and minimum three M64 query tiles
per SM. These are work-based rules, without a per-batch tuning table.

The existing 32K capacity selection for the prefetch specialization is retained.
Testing a single prefetch-capable loop for all contexts added 0.4–0.7% versus
the previous optimized path near Q465–469 and 1.1–1.2% at 1K. That probe was
rejected. Each prefetch-capable CTA still checks its actual live/split traversal
before issuing hints. No new cutoff was introduced.

## Coverage and correctness

The main matrix has 46 timing cases; targeted repeats bring this to 54 cases
and 242 full FP32 reference-case executions. Every timing case also checks
sampled FP32 rows from every request against the complete live KV. Complete
output and LSE match the original native schedule and previous optimized
library bit for bit, including newly enabled shapes. The installed FA operator
is an additional correctness reference. Output, LSE, scheduler, and split-buffer
guards remain intact; mixed cases use shuffled physical pages and poisoned
unused page-table entries.

Coverage includes:

- 4K-query chunks at every power-of-two KV context from 4K through 1M.
- Full prefill from 8K through 64K and 64K-query chunks through 1M KV.
- The exact mixed 64K chunk: queries `[4096, 8192, 32768, 20480]`, live KV
  `[4096, 8192, 98304, 151552]`, hence cached prefixes `[0, 0, 65536, 131072]`.
- The same query chunk with live KV `[4096, 8192, 524288, 1048576]`.
- The exact mixed chunk with both ordinary automatic splitting (one split in
  this workload) and explicit 16-way splitting, comparing matched split counts.
- Q=65,537 / KV capacity=1,048,640 and B=257 / Q=256 / KV=4,096, crossing the
  removed upper cutoffs. B is requests per TP8 replica; each card processes its
  head shard of those same requests.
- Small-query, decode, local-attention, empty-request, partial-page, and sink
  numerical checks; Q464–469 and KV960/1023/1024/1025 timing boundaries.

The MiMo validation snapshot passes 61 host selector checks. Its 171
device-kernel bodies match the preceding validated MiMo implementation byte
for byte. Five unaffected dtype/dimension/layout templates compile against
that snapshot, and all 41 source hashes match its build manifest. Python
syntax and diff whitespace checks pass. Kimi's separate WI-only evaluation
and current dispatch decision are recorded in
[its validation report](../sm89_k3_prefill/VALIDATION.md).

## Performance

Measurements capture one complete attention operator per graph, including
scheduler setup and split reduction. Variant order is randomized and 256 MiB
is evicted before each sample. All samples are retained. The main matrix uses
7–51 samples according to workload duration; long-query repeats use 31, the
mixed split repeat uses 51, and the short fallback repeat uses 401. The long
and split repeats include a byte-identical optimized-library copy as well as
the original-schedule A/A control.

The table uses the faster median of the two original-schedule controls and the
higher-sample repeat where available. Positive reduction means lower latency.
These are single-card attention-operator measurements, not serving throughput.
The MiMo reductions include both warp interleaving and optional LLC prefetch;
they are not an isolated measurement of interleaving alone.

| Queries | Live KV | Reduction |
| --- | --- | ---: |
| 4K | 4K | 6.50% |
| 4K | 128K | 20.96% |
| 4K | 1M | 12.53% |
| 64K | 64K | 15.13% |
| 64K | 128K | 17.49% |
| 64K | 256K | 1.59% |
| 64K | 512K | 4.99% |
| 64K | 1M | 11.69% |
| mixed 64K | 4K / 8K / 96K / 148K | 16.42% |
| mixed 64K | 4K / 8K / 512K / 1M | 5.87% |
| mixed 64K, 16 splits | 4K / 8K / 96K / 148K | 5.45% |
| 65,537 | 1,048,640 capacity | 11.67% |
| 257 requests × 256 | 4K each | 4.58% |

Every requested 4K–1M prefill case had a positive measured median reduction.
Q465–469 remains within 0.71% of the faster original control. An initial Q128 /
KV128 fallback timing was +2.27%; its 401-sample repeat is +0.64% against the
faster control. No repeated median exceeded the approximately 1% allowance.

The 64K-query/256K-KV point has substantial timing variation. An initial
seven-sample comparison appeared 6.5% slower than the previous optimized
library. With 31 samples it is 0.62% faster than that library; its identical
optimized copy differs by 1.82%. The measured median gain versus the faster
original control is only 1.59%, and the mean gain is 0.88%. This is the weakest
point, not evidence of a uniform speedup margin on every invocation.

The first explicit-split run had a 14.5% disagreement between identical
original controls. The 51-sample repeat reduces that disagreement to 0.01%,
with optimized copies also agreeing, and gives the 5.45% result above. The
initial and repeated measurements are both retained.

## Scope and evidence

The largest normal query chunk is 64K; the extra boundary case has 65,537
queries. Q=KV=1M in one call was not tested. No memory/race sanitizer was
available. The standalone adapter exercises the normal C++ dispatcher; a full
Python-package build, installation, and serving deployment were not performed.

Remote workspace: `/o-work/shuoer/dev/fa-prefill-general-20261016`.
Final source: `source`; library: `final-mimo/libfa.so`. SHA256:

```
f72484227d499c2eabfa584f4786ffbc606f0fa27556408d01d018e051fea913
```

Local evidence:
`/Users/shor/dev/T-HEAD/flash-attention-k3-wi-backup-20261015-evidence/prefill-general-20261016`.
`analysis.json` indexes the results. Raw JSON, logs, build manifests, and device
instruction comparisons are in `node78`. The main matrix is
`results/final-mimo`; repeats are `results/final-repeat`,
`results/final-split-repeat.json`, and `results/final-small-repeat.json`.
The rejected unified-prefetch probe is retained separately from the final
libraries. Use README to reproduce the suites.
