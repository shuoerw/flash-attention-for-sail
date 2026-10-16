# SM89 MiMo prefill validation

This suite builds and tests the PPU890 forward implementation in
the normal `run_mha_fwd_` dispatcher in `hopper/flash_fwd_launch_template.h`.
It does not install a Python package.

The supported specialization is BF16, Hq16/Hkv1, Dqk192/Dv128, causal attention,
packed GQA, and contiguous 64-token KV pages. Q is held in registers; shared
memory contains one K tile and one V tile. Each TP8 card sees the replica's
request batch and processes one head shard.

## Kernel contract

The normal forward dispatcher selects interleaving after packed GQA, query
geometry, and KV splits are known. `flash::sm89_prefill_schedule` in
`hopper/heuristics.h` checks the supported layouts and returns the schedule.
`hopper/tile_size.h` owns tile configuration; `hopper/mainloop_fwd_sm80.hpp`
owns the shared barrier sequence. The existing fallback and SplitKV reduction
remain in place. No separate prefill launch API is needed.

Q1/Q4, local attention, other dtypes/dimensions, unsupported layouts, and small
query workloads use the default kernel. Two minimum-work rules amortize the
handoff cost: KV capacity must be at least 1,024 tokens, and query work must
provide at least three M64 tiles per SM. There are no batch-size, query-length,
or context-length upper cutoffs in this selector. The ordinary dispatcher
still determines query tiling and KV splitting.

`WarpInterleaved` uses two leading and two trailing warps. The leading group
releases the trailing group after two QK reduction fragments. The following
barrier phases protect reuse of the shared slots:

| Barrier | Leading group | Trailing group | Protected operation |
| --- | --- | --- | --- |
| `QkHandoff` | Arrives during QK | Waits before QK | Staggers tensor/scalar work |
| `KReadersDone` | Waits after softmax and current V copy | Arrives after QK | Refilling the sole K slot |
| `VReady` | Arrives after publishing current V | Waits before PV | Reading the current V tile |
| `VReadersDone` | Waits before the next V copy | Arrives after PV | Refilling the sole V slot |
| `LeadingGroupReady` | Both leading warps synchronize after K load and prior PV | No participation | Leading-group copy/read ordering |

All warps participate in first-tile V zero-fill. The final iteration omits the
unused V-retirement arrival so no partial phase leaks into a persistent CTA's
next task. The scheduler also retires every reader of its shared tile index
before that storage can be reused, including empty tasks.

`WarpInterleavedPrefetch` adds one-tile-ahead K/V LLC hints. Capacity at or below
32K selects the no-prefetch specialization to avoid hint-loop overhead on short
traversals. At larger capacities, each CTA checks its live traversal, including
any split partition; fewer than 512 tiles does not issue hints. Prefetch never
allocates a second shared-memory stage.

## Build and run

Run inside a PPU SDK environment. `SOURCE` is this checkout, `ACTLIZE` is the
matching ACTLIZE tree, and `BUILD` is a separate output directory. The compiler
is the SDK's CUDA-compatible `nvcc`.

```bash
export PATH=/usr/local/PPU_SDK/CUDA_SDK/bin:$PATH
python3 "$SOURCE/tests/sm89_prefill/build.py" --source "$SOURCE" --actlize "$ACTLIZE" --output "$BUILD/control" --control
python3 "$SOURCE/tests/sm89_prefill/build.py" --source "$SOURCE" --actlize "$ACTLIZE" --output "$BUILD/release"
python3 "$SOURCE/tests/sm89_prefill/compile_coverage.py" --build "$BUILD/release"
CUDA_VISIBLE_DEVICES=0 python3 "$SOURCE/tests/sm89_prefill/validate.py" --root "$BUILD" --output "$BUILD/results"
```

On a shared node, acquire its shared GPU lock before the validation command.
The tests need an installed `flash_attn_interface` for an independent reference.
The build also compiles and runs host eligibility tests. The release adapter
calls the normal dispatcher; the control adapter instantiates the default
schedule directly. Compile coverage checks
unaffected FP16/BF16, dense/paged, and D64/D128/D192/D256 paths.

Use `validate.py --groups contexts-4k contexts-64k mixed-64k mixed-64k-1m
mixed-64k-split extended-bounds` for the 4K–1M context sweep. It tests 4K query
chunks, full/chunked queries up to 64K, and the mixed 64K chunk with query
lengths `[4096, 8192, 32768, 20480]` and cached prefixes
`[0, 0, 65536, 131072]`. The mixed tests also cover longer prefixes and 16-way
KV splitting. `replay.py --query-lengths ... --kv-lengths ...` accepts exact
per-request vectors; KV lengths include the newly processed queries. The
extended-bound cases cross the removed upper cutoffs.

`--groups contexts-repeat --interleaved-aa` repeats the longer query cases
with 31 samples and an identical interleaved-library copy, so variability in
the optimized schedule can be measured separately from the default control.

Validation compares full output and LSE against the default native schedule,
checks small tensors against FP32, samples FP32 rows in larger tensors, and
compares against the installed implementation. It checks guard regions around
output, LSE, scheduler, and split buffers. Compact-page cases poison unused page
entries. Graph replay changes live lengths without recapturing the operator.

Timing uses a single complete attention operator per CUDA graph, including
scheduler setup and split reduction, with randomized order and 256 MiB cache
eviction. A byte-identical control copy measures A/A variation. JSON files retain
all samples, outputs' error statistics, split counts, commands, and return codes.
These measurements are attention-operator timings, not model throughput.

`compare_machine_code.py` compares emitted instruction bytes between builds.
It intentionally reports differences rather than treating equivalent source
as proof of identical machine code. No package installation or service change
is needed for these checks.
