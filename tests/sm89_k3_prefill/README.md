# SM89 Kimi K3 prefill warp-interleaving validation

This suite evaluates a named-barrier WI candidate for BF16 expanded MHA
prefill against the original kernel. The production `run_mha_fwd_` dispatcher
retains the original dense schedule. B counts requests per DP replica; each TP8 rank
processes 12 Q/KV heads with D192/V128. Q/K are dense varlen tensors. V may
be contiguous or retain the 256-element head stride of the KV projection.

## Comparison contract

The ordinary tile-selection function determines both the baseline and the
interleaved kernel: N64, M64 with four warps or M128 with eight warps, and one
shared K tile plus one shared V tile. Tile dimensions, warp count, shared-memory
stage count, and split count stay identical in the comparison. The only
candidate scheduling change is named-barrier warp interleaving.

The leading half of the warps releases the trailing half after two QK
reduction fragments. The barrier protocol is shared with paged prefill;
see `../sm89_prefill/README.md` for the K/V lifetime and barrier phases.
Dense prefill does not use the paged LLC-prefetch extension.

The explicit candidate is tested on SM89 BF16 D192/V128, 12 equal Q/KV
heads, dense varlen layouts, one KV split, and M64/M128. It preserves the
ordinary M16 fallback. Causal full prefill and noncausal attention to cached
prefixes are both covered. The candidate is not selected by the production
dense dispatcher because its measured regressions exceed the acceptance limit.
No average-KV gate is used to hide those regressions.

Output and LSE must match the original kernel bit for bit. Both schedules use
the same reduction grouping. Static assertions also check that the ordinary
M64/M128 tile policies retain N64 and one stage.

## Build and run

Use the PPU SDK CUDA-compatible compiler and matching ACTLIZE headers.
`SOURCE` is this checkout; `BUILD` is a separate output directory.

```bash
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:$PATH
python3 "$SOURCE/tests/sm89_k3_prefill/build.py" \
  --source "$SOURCE" --actlize "$ACTLIZE" --output "$BUILD"
CUDA_VISIBLE_DEVICES=0 python3 "$SOURCE/tests/sm89_k3_prefill/replay.py" \
  --library "$BUILD/libfa.so" --suite signal --samples 101 \
  --modes 1 8 --output "$BUILD/results/signal.json"
```

Acquire the shared device lock before GPU work. The tests build standalone
libraries and do not install a package or alter a serving process.
`build.py` runs host eligibility checks, records source/library hashes,
and keeps compiler intermediates.

`replay.py` suites are `smoke`, `signal`, `safety`, `boundaries`, and
`contexts`. The context suite covers 4K–1M KV, causal and noncausal 4K
query chunks, full/chunked queries up to 64K, and mixed 64K chunks.

| Mode | Kernel |
| --- | --- |
| 0 | Original N64 schedule, including M16/M64/M128 selection |
| 1 | N64 interleaving without runtime eligibility selection |
| 8 | Normal dispatcher, retaining the original dense schedule |

Native variants use identical tensors, output addresses, and workspaces.
The test compares complete output and LSE exactly, checks full FP32 references
on small inputs, and checks sampled query rows and all heads against full
live KV on large inputs. Guard regions surround output, LSE, and scheduler
buffers. Optional `--installed` checks the installed FA wrapper.
`--exact-extra` compares the dispatcher with another same-geometry library.

Timing captures one attention call per graph, including semaphore initialization,
randomizes variant order, evicts 256 MiB before each sample, and includes an
identical A/A control. JSON retains all timing samples. These are attention
operator timings, excluding model projections, prefix assembly, the Python
API, and serving scheduling. `profile.py` isolates one warmed kernel for ACU.
See `VALIDATION.md` for results and limitations.
