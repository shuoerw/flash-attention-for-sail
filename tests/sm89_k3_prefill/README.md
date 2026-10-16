# SM89 Kimi K3 expanded prefill validation

This suite exercises the nonpaged BF16 prefill specialization through the normal
`run_mha_fwd_` dispatcher in `hopper/flash_fwd_launch_template.h`. It does not install
a package or alter the serving process. Kimi nonpaged MHA and MiMo paged GQA
share the same interleaving implementation and normal forward dispatcher.

## Workload and scope

The inspected K3 configuration has 96 attention heads, Q/K dimension 192
(128 non-RoPE + 64 RoPE), and V dimension 128. Expanded MHA prefill on an
attention-TP8 rank has 12 Q heads and 12 KV heads. B counts requests per DP
replica; every TP rank sees that batch and processes its local head shard.

The SGLang expanded prefill path calls `flash_attn_varlen_func`. Full prefill
uses causal attention; attention against earlier prefix chunks uses noncausal
attention. Q/K are dense contiguous tensors. V can be contiguous or retain the
256-element head stride of the combined KV projection. This suite reproduces
both layouts. It does not target absorbed MLA, decode, KDA layers, or context
parallel absorbed attention.

## Kernel and dispatch

The candidate retains **one shared K tile and one shared V tile**. It uses N96
KV tiles and preserves the regular M64/M128 query-tile selection. M64 uses four
warps divided 2+2; M128 uses eight warps divided 4+4. The leading group releases
the trailing group after two QK reduction fragments. The barrier protocol is
shared with the MiMo implementation; see `../sm89_prefill/README.md` for the
K/V lifetime and barrier phases. Dense prefill does not use the MiMo LLC
prefetch extension.

The normal dispatcher calls `flash::sm89_prefill_schedule` in
`hopper/heuristics.h` after ordinary tile and split selection. Supported calls
select interleaving; other calls retain the existing launch. Tile selection is
in `hopper/tile_size.h`, and both paged GQA and nonpaged MHA use the same
barrier implementation in `hopper/mainloop_fwd_sm80.hpp`. There is no separate
model-specific launch API.

The guarded candidate requires BF16, SM89, D192/V128, 12 equal Q/KV heads,
dense varlen tensors, one KV split, and M64/M128. Average actual KV length must
be at least 4,096 tokens. There are no batch-size, query-length, or context-length
upper cutoffs. The average gate avoids selecting the
wider tile merely because a batch of short requests has one long outlier.
Short-query M16 selection, unsupported layouts, and unsupported attention
features retain the original kernel. No per-batch-size tuning table is used.

N96 changes reduction grouping relative to N64, so its numerical result need
not be bit-exact with the original. `--exact-extra` requires bit-exact output
and LSE agreement with a previously validated library during refactor checks.

## Reproduce

Use the PPU SDK CUDA-compatible compiler and the matching ACTLIZE headers.
Build outputs must be outside the source tree. For the recorded validation:

```bash
cd /o-work/shuoer/dev/fa-k3-warp-interleave-20261015
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:$PATH
python3 combined-source/tests/sm89_k3_prefill/build.py \
  --source combined-source \
  --actlize /o-work/shuoer/dev/fa-prefill-one-stage-20261015/actlize-git \
  --output combined-k3
CUDA_VISIBLE_DEVICES=0 python3 combined-source/tests/sm89_k3_prefill/replay.py \
  --library combined-k3/libfa.so --suite safety --samples 41 \
  --modes 8 --allow-reduction-difference --output results/combined-k3-safety.json
```

On node78, acquire `/tmp/shuoer-node78-ppu0.lock` around GPU work. The recorded
commands ran in `shuoer_mimo`, with the resident TP8 server left in place.

`build.py` compiles and runs host eligibility checks, records source/library
hashes, and keeps compiler intermediates.

`replay.py` suites are `smoke`, `signal`, `safety`, `boundaries`, and `contexts`.
The context suite sweeps 4K–1M KV with causal and noncausal 4K chunks, full/chunked
queries up to 64K, and mixed 64K chunks. It also crosses the removed upper cutoffs.
Modes:

| Mode | Kernel |
| --- | --- |
| 0 | Original N64 dispatch, including M16/M64/M128 |
| 1 | Raw N96 interleaved candidate |
| 8 | Normal FA dispatcher, including interleaving and fallback |

The test uses identical tensors, output addresses, and workspaces for native
variants. Full output and LSE are compared against the original kernel. Small
cases also use full FP32 references; larger cases check five query rows per
request and every head against all live K/V. The selected candidate is checked
directly against FP32. Output, LSE, and scheduler buffers have guard regions.
Fallback output/LSE and optional prior-library comparisons are checked bit-for-bit.

Timing captures one native attention call per graph, including semaphore
initialization. It randomizes variant order, evicts 256 MiB before each sample,
and includes an identical A/A control. JSON retains all timing samples. These
are attention-kernel path measurements, excluding model projections, prefix
assembly, the Python API, and serving scheduling. The inspected serving script
does not enable prefill CUDA graphs; graph capture here isolates device work.

Optional `--installed` checks the installed FA wrapper; it is not required for
the native/FP32 tests. `profile.py` isolates one warmed kernel for ACU profiling.
See `VALIDATION.md` for measured results and validation scope.
