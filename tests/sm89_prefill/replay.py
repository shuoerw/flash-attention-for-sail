"""MiMo TP8 BF16 attention replay: paged GQA, Dqk=192, Dv=128.

Times the attention operator, including scheduler setup and split reduction.
Batch is the request batch of a TP8 replica; each card sees the same batch
and one head shard. This replay measures the attention operator on one card.
"""
import argparse
import ctypes
import gc
import json
import math
from pathlib import Path
import random
import statistics
import torch
from flash_attn_interface import flash_attn_with_kvcache

p = argparse.ArgumentParser()
p.add_argument('--libraries', nargs='*', default=[])
p.add_argument('--output', type=Path, required=True)
p.add_argument('--quick', action='store_true')
p.add_argument('--correctness-only', action='store_true')
p.add_argument('--native-reference-only', action='store_true', help='Correctness diagnostic: validate native libraries directly against FP32 when the installed wheel fails a case.')
p.add_argument('--skip-correctness', action='store_true')
p.add_argument('--repeats', type=int, default=11)
p.add_argument('--seed', type=int, default=193)
p.add_argument('--flush-mb', type=int, default=256)
p.add_argument('--splits', type=int, default=0)
p.add_argument('--reference-splits', type=int, help='Installed reference split count; native candidates keep --splits. Useful when the shared GPU cannot hold two large split workspaces.')
p.add_argument('--split-variants', type=int, nargs='*', default=[], help='Compare installed FA with explicit split counts against its default.')
p.add_argument('--native-split-variants', type=int, nargs='*', default=[], help='Add explicit split variants for every native library.')
p.add_argument('--workload', choices=['all', 'prefill', 'decode'], default='all')
p.add_argument('--profile', action='store_true')
p.add_argument('--compare-baseline', help='Record full output/LSE differences without changing numerical tolerances.')
p.add_argument('--exact-groups', nargs='*', default=[], help='Comma-separated same-geometry libraries that must match bitwise.')
p.add_argument('--attention-sinks', action='store_true', help='Also test attention sinks on full causal attention.')
p.add_argument('--empty-sequences', action='store_true', help='Include empty requests alongside live prefill requests.')
p.add_argument('--holdout', action='store_true', help='Use batch sizes and sequence lengths not used in the tuning sweep.')
p.add_argument('--partial-pages', action='store_true', help='Validate longer ragged prefill and partial pages under SplitKV.')
p.add_argument('--long-prefill', action='store_true', help='Exercise the launch script\'s 64K chunk limit and 128K context.')
p.add_argument('--case', type=int, nargs=3, action='append', metavar=('B', 'Q', 'K'), help='Explicit benchmark shapes; may be repeated.')
p.add_argument('--query-lengths', type=int, nargs='+', help='Exact per-request query lengths for one mixed chunk.')
p.add_argument('--kv-lengths', type=int, nargs='+', help='Exact per-request live KV lengths, including the new queries.')
p.add_argument('--windows', type=int, nargs='+', default=[-1, 128])
p.add_argument('--only-libraries', action='store_true', help='Time only native libraries, retaining installed correctness checks.')
p.add_argument('--acu-target', help='Launch one selected variant directly for external hardware profiling; requires one --case.')
p.add_argument('--kv-pattern', choices=['uniform','ramp','bimodal','one-long'], default='uniform', help='KV length distribution; mixed prefill requires KV at least as long as each query chunk.')
p.add_argument('--kv-short-length', type=int, help='Exact short KV length for mixed-length tests; also exercises short lengths during graph replay.')
p.add_argument('--q-pattern', choices=['uniform','ramp','one-short'], default='uniform', help='Query lengths for ragged prefill/verification safety tests.')
p.add_argument('--capacity-k', type=int, default=0, help='Pad page-table capacity beyond live KV lengths, as in a captured decode graph.')
p.add_argument('--compact-pages', action='store_true', help='Allocate distinct physical pages for live KV only, retaining the full page-table capacity; excludes changing-length graph stress.')
p.add_argument('--graph-replay-stress', action='store_true', help='Change live lengths across replays of the same captured graph.')
p.add_argument('--shared-workspace', action='store_true', help='Use identical native output/workspace addresses to isolate allocation-dependent timing differences.')
a = p.parse_args()
if (a.query_lengths is None) != (a.kv_lengths is None):
    p.error('--query-lengths and --kv-lengths must be supplied together')
if a.query_lengths is not None:
    if a.case or a.acu_target or a.graph_replay_stress or a.q_pattern != 'uniform' or a.kv_pattern != 'uniform':
        p.error('Exact lengths cannot be combined with --case, patterns, profiling, or changing-length replay')
    if (len(a.query_lengths) != len(a.kv_lengths) or sum(a.query_lengths) <= 0
            or any(q < 0 or k < q for q, k in zip(a.query_lengths, a.kv_lengths))):
        p.error('Exact lengths require equal-size lists, nonnegative Q, positive total Q, and K >= Q')
if a.kv_short_length is not None and (a.kv_short_length < 0 or (a.kv_short_length == 0 and not a.graph_replay_stress)):
    p.error('--kv-short-length must be positive, or zero for changing-length graph replay')
if a.native_reference_only and not a.correctness_only:
    p.error('--native-reference-only requires --correctness-only')
if a.compact_pages and a.graph_replay_stress:
    p.error('--compact-pages cannot grow live KV during graph replay')
if not -1 <= a.splits <= 256 or any(not 1 <= s <= 256 for s in a.split_variants+a.native_split_variants):
    p.error('Split counts must be 1..256; --splits also accepts 0/-1 for automatic selection.')
if a.reference_splits is not None and not 0 <= a.reference_splits <= 256:
    p.error('--reference-splits must be 0..256')
torch.manual_seed(a.seed)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
libs = {}
for path in a.libraries:
    path = Path(path).resolve()
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    lib.fa_mimo_run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.c_void_p]
    lib.fa_mimo_run.restype = ctypes.c_int
    libs[path.parent.name] = lib

def prepare(qlens, klens, window, strong_sinks=False):
    b, pages = len(qlens), (max(max(klens), a.capacity_k)+63)//64
    live_pages = [(k+63)//64 for k in klens]
    physical_pages = sum(live_pages) if a.compact_pages else b*pages
    q = torch.randn(sum(qlens), 16, 192, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(physical_pages, 64, 1, 192, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(physical_pages, 64, 1, 128, device='cuda', dtype=torch.bfloat16)
    # Physical pages are shuffled across requests, not just within a sequence.
    mapping = torch.randperm(physical_pages, device='cuda').to(torch.int32)
    if a.compact_pages:
        # Poison unused entries: the kernel must never follow them, even when
        # page-table capacity is larger than the current live KV lengths.
        table = torch.full((b, pages), -1, device='cuda', dtype=torch.int32)
        start = 0
        for i, count in enumerate(live_pages):
            table[i, :count] = mapping[start:start+count]
            start += count
    else:
        table = mapping.reshape(b, pages)
    cu = torch.tensor([0]+list(__import__('itertools').accumulate(qlens)), device='cuda', dtype=torch.int32)
    seq = torch.tensor(klens, device='cuda', dtype=torch.int32)
    sinks = torch.linspace(-20 if strong_sinks else -2, 20 if strong_sinks else 2, 16, device='cuda', dtype=torch.bfloat16) if window >= 0 or a.attention_sinks else None
    return dict(q=q, k=k, v=v, table=table, cu=cu, seq=seq, sinks=sinks,
                qlens=qlens, klens=klens, window=window)

def guarded_empty(d, shape, dtype):
    size = math.prod(shape)
    storage = torch.empty(size + 128, device='cuda', dtype=dtype)
    guards = (storage[:64], storage[-64:])
    for guard in guards:
        guard.fill_(17)
    d.setdefault('guards', []).extend(guards)
    return storage[64:-64].view(shape)


def check_guards(d):
    for guard in d.get('guards', []):
        assert bool((guard == 17).all()), 'Native kernel wrote outside its workspace'


def make_call(d, name):
    q, k, v = d['q'], d['k'], d['v']
    sq, b, tq = max(d['qlens']), len(d['qlens']), q.shape[0]
    if name.startswith('installed'):
        splits = int(name.split('-s')[1]) if '-s' in name else (a.splits if a.reference_splits is None else a.reference_splits)
        def call():
            result = flash_attn_with_kvcache(q, k, v, page_table=d['table'], cache_seqlens=d['seq'],
                cu_seqlens_q=d['cu'], max_seqlen_q=sq, causal=True,
                window_size=(d['window'], 0) if d['window'] >= 0 else (-1, -1),
                s_aux=d['sinks'], num_splits=splits, return_softmax_lse=True)
            return result[0], result[1]
        return call
    library_name, sep, split_text = name.partition('@s')
    native_splits = int(split_text) if sep else a.splits
    # The dispatcher uses rounded page-table capacity, not the longest live
    # sequence. A 65,535-token request already has 65,536-token capacity.
    ns = 256 if sq <= 16 or tq//b < 32 else max(16 if d['table'].shape[1]*64 >= 65536 else 1, native_splits)
    workspace_key = ('native_workspace', ns)
    if a.shared_workspace and workspace_key in d:
        out, lse, sem, partial, plse = d[workspace_key]
    else:
        out = guarded_empty(d, (tq, 16, 128), torch.bfloat16)
        lse = guarded_empty(d, (16, tq), torch.float32)
        sem = guarded_empty(d, (34+2*b,), torch.int32)
        partial = guarded_empty(d, (ns, 16, tq, 128), torch.float32)
        plse = guarded_empty(d, (ns, 16, tq), torch.float32)
        if a.shared_workspace: d[workspace_key] = (out, lse, sem, partial, plse)
    tensors = [q, k, v, out, lse, sem, d['cu'], d['seq'], d['table'], d['sinks'], partial, plse]
    pointers = (ctypes.c_void_p*12)(*[t.data_ptr() if t is not None else 0 for t in tensors])
    cfg = (ctypes.c_int*8)(b, tq, sq, d['table'].shape[1]*64, k.shape[0], d['table'].shape[1], d['window'], native_splits)
    def call():
        keep_alive = tensors
        status = libs[library_name].fa_mimo_run(pointers, cfg, torch.cuda.current_stream().cuda_stream)
        if status < 1:
            raise RuntimeError(f'{name}: native error {status}')
        if status > ns:
            raise RuntimeError(f'{name}: insufficient partial workspace {status} > {ns}')
        call.num_splits = status
        return out, lse
    call.scheduler_workspace = sem
    return call

def reference(d, sampled=False):
    outs, lses, offsets = [], [], []
    offset = 0
    for bi, (sq, sk) in enumerate(zip(d['qlens'], d['klens'])):
        ids = sorted(set([0, sq//2, sq-1])) if sampled and sq else list(range(sq))
        live_ids = d['table'][bi, :(sk+63)//64].long()
        ki = d['k'][live_ids].reshape(-1, 192)[:sk].float()
        vi = d['v'][live_ids].reshape(-1, 128)[:sk].float()
        for start in range(0, len(ids), 32):
            rows = torch.tensor(ids[start:start+32], device='cuda')
            qi = d['q'][rows+offset].float().transpose(0, 1)
            scores = (qi @ ki.T) / math.sqrt(192)
            pos = rows[:, None] + sk - sq
            keypos = torch.arange(sk, device='cuda')[None, :]
            mask = keypos > pos
            if d['window'] >= 0:
                mask = mask | (keypos < pos-d['window'])
            scores.masked_fill_(mask[None], float('-inf'))
            ls = scores.logsumexp(-1)
            if d['sinks'] is not None:
                ls = torch.logaddexp(ls, d['sinks'].float()[:, None])
            probs = (scores-ls[:, :, None]).exp().nan_to_num()
            outs.append((probs @ vi).transpose(0, 1))
            lses.append(ls)
            offsets.extend((rows+offset).tolist())
        offset += sq
    return torch.cat(outs), torch.cat(lses, dim=1), torch.tensor(offsets, device='cuda')

def validate(result, expected, ids=None):
    o, l = result
    if ids is not None: o, l = o[ids], l[:, ids]
    torch.testing.assert_close(o.float(), expected[0].float(), atol=0.025, rtol=0.025)
    finite = torch.isfinite(expected[1])
    torch.testing.assert_close(l[~finite], expected[1][~finite], atol=0, rtol=0)
    torch.testing.assert_close(l[finite], expected[1][finite], atol=0.003, rtol=0.001)
    assert torch.isfinite(o).all()
    return (o.float()-expected[0].float()).abs().max().item()

results = dict(device=str(torch.cuda.get_device_properties(0)), libraries=a.libraries,
    config=vars(a) | {'output': str(a.output)}, timing_method='CUDA events inside a single-operator CUDA graph, cache eviction before replay', correctness=[], benchmarks=[])
a.output.parent.mkdir(parents=True, exist_ok=True)
def save(): a.output.write_text(json.dumps(results, indent=2))
native_names = [n for lib in libs for n in [lib]+[f'{lib}@s{s}' for s in a.native_split_variants]]
names = ['installed'] + [f'installed-s{s}' for s in a.split_variants] + native_names
if a.native_reference_only: names = native_names
def explicit_lengths(b, sq, sk):
    klens = [sk]*b
    if a.kv_pattern != 'uniform':
        short_k = min(sk, a.kv_short_length) if a.kv_short_length is not None else sk-sk//2
        if a.kv_pattern == 'ramp': klens = [sk - (sk-short_k)*i//max(1,b-1) for i in range(b)]
        elif a.kv_pattern == 'bimodal': klens = [sk if i < max(1,b//4) else (short_k if a.kv_short_length is not None else min(sk,4097+i)) for i in range(b)]
        else: klens = [sk]+[(short_k if a.kv_short_length is not None else min(sk,4097+i)) for i in range(1,b)]
    qlens = [sq]*b
    if a.q_pattern == 'ramp': qlens = [max(1, sq-(sq//2)*i//max(1,b-1)) for i in range(b)]
    elif a.q_pattern == 'one-short': qlens[-1] = max(1, sq-1)
    if a.kv_pattern != 'uniform' and sq > 4:
        assert all(k >= q for q, k in zip(qlens, klens)), 'Mixed prefill requires live KV >= query chunk for every request'
    return qlens, klens

if a.acu_target:
    assert a.case and len(a.case) == 1 and a.acu_target in names
    b, sq, sk = a.case[0]
    qlens, klens = explicit_lengths(b, sq, sk)
    d = prepare(qlens, klens, a.windows[0])
    call = make_call(d, a.acu_target)
    for _ in range(5): call()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    out, lse = call()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    assert torch.isfinite(out).all()
    results['profile'] = dict(name=a.acu_target, qlens=qlens, klens=klens, num_splits=getattr(call, 'num_splits', None))
    save()
    print('PROFILE', a.acu_target, a.case[0], 'splits', getattr(call, 'num_splits', None), flush=True)
    raise SystemExit(0)
def check_exact(outputs):
    for group in a.exact_groups:
        members = group.split(',')
        for name in members[1:]:
            for actual, expected in zip(outputs[name], outputs[members[0]]):
                try:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                except AssertionError as e:
                    raise AssertionError(f'Bitwise mismatch: {name} versus {members[0]}: {e}') from e
if not a.skip_correctness:
    checks = [([1, 1], [1, 65]), ([4, 3], [129, 257]), ([63, 7], [63, 513]),
              ([65, 129], [257, 193]), ([257, 17], [513, 4097]), ([1]*8, [4096-i*63 for i in range(8)]),
              ([257, 257], [513, 577]), ([465], [465]), ([63]*8, [63]*8),
              ([65]*8, [129]*8), ([257, 257, 0], [513, 577, 0])]
    if a.quick: checks = checks[:4]
    if a.empty_sequences: checks = [([0, 257, 0, 257], [0, 513, 0, 65539]),
                                      ([0, 0, 512], [0, 0, 65539]),
                                      ([512, 0, 1], [65539, 0, 577]),
                                      ([512, 512, 512], [0, 1, 65])]
    if a.partial_pages:
        checks = [([128,128], [257,513]), ([4]*64, [4096-i*31 for i in range(64)]),
                  ([1024], [65539]), ([1025,1025], [65539,66001])]
    for qs, ks in checks:
        # Underfilled-K rows probe the optimized full-attention path. Local attention
        # with sinks has a separate empty-row convention in the default kernel.
        check_windows = [-1] if any(k < q for q, k in zip(qs, ks)) else [-1, 128]
        for window in check_windows:
            d = prepare(qs, ks, window, strong_sinks=True)
            expected = reference(d)
            installed = None if a.native_reference_only else make_call(d, 'installed')()
            row = dict(qlens=qs, klens=ks, window=window, errors={})
            outputs = {}
            for name in names:
                call = make_call(d, name)
                for _ in range(3): result = call()
                torch.cuda.synchronize()
                try:
                    row['errors'][name] = validate(result, expected)
                except AssertionError as e:
                    raise AssertionError(f'{name}: q={qs}, k={ks}, window={window}: {e}') from e
                if installed is not None:
                    torch.testing.assert_close(result[0], installed[0], atol=0.016, rtol=0.016)
                    torch.testing.assert_close(result[1], installed[1], atol=0.003, rtol=0.001)
                outputs[name] = tuple(t.clone() for t in result)
            check_exact(outputs)
            check_guards(d)
            results['correctness'].append(row)
            print('PASS', qs, ks, window, flush=True)
            save()
if a.correctness_only: raise SystemExit(0)

prefill = [('prefill', b, s, s) for b in [1, 4] for s in [512, 2048, 4096, 8192]]
prefill += [('chunked_prefill', b, 2048, 32768) for b in [1, 4]] + [('chunked_prefill', 1, 4096, 131072)]
decode = [('decode', b, 1, s) for b in [1, 8, 32, 64, 128, 256] for s in [4096, 32768]]
decode += [('verify4', b, 4, s) for b in [1, 64, 256] for s in [4096, 32768]]
decode += [('decode', 64, 1, 131072)]
if a.quick:
    prefill = [('prefill', 1, 4096, 4096), ('chunked_prefill', 1, 2048, 32768)]
    decode = [('decode', 1, 1, 4096), ('decode', 64, 1, 32768), ('verify4', 64, 4, 4096)]
if a.holdout:
    prefill = [('prefill', 2, 1024, 1024), ('prefill', 3, 1536, 1536), ('prefill', 2, 6144, 6144),
               ('chunked_prefill', 1, 1024, 65536), ('chunked_prefill', 2, 4096, 65536)]
    decode = [(kind, b, sq, s) for kind, sq in [('decode', 1), ('verify4', 4)]
              for b, s in [(16, 8192), (48, 8192), (96, 16384), (192, 65536)]]
if a.long_prefill:
    prefill = [('prefill', 1, 16384, 16384), ('prefill', 1, 65536, 65536),
               ('chunked_prefill', 1, 65536, 131072)]
    decode = []
shapes = (prefill if a.workload != 'decode' else []) + (decode if a.workload != 'prefill' else [])
if a.case:
    shapes = [('decode' if q == 1 else 'verify4' if q == 4 else 'prefill', b, q, k) for b, q, k in a.case]
if a.query_lengths is not None:
    shapes = [('mixed_chunk', len(a.query_lengths), max(a.query_lengths), max(a.kv_lengths))]
flush = torch.empty(a.flush_mb*1024*1024, dtype=torch.uint8, device='cuda')
rng = random.Random(a.seed)
for kind, b, sq, sk in shapes:
    for window in a.windows:
        qlens, klens = (a.query_lengths, a.kv_lengths) if a.query_lengths is not None else explicit_lengths(b, sq, sk)
        d = prepare(qlens, klens, window)
        expected = reference(d, sampled=True)
        graphs, calls, outputs, events, snapshots, dynamic_splits = {}, {}, {}, {}, {}, {}
        for name in names:
            call = make_call(d, name)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): call()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with torch.cuda.graph(g):
                start.record(); out = call(); end.record()
            for _ in range(3): g.replay()
            torch.cuda.synchronize()
            validate(out, expected, expected[2])
            if name != 'installed':
                torch.testing.assert_close(out[0], outputs['installed'][0], atol=0.016, rtol=0.016)
                torch.testing.assert_close(out[1], outputs['installed'][1], atol=0.003, rtol=0.001)
            graphs[name], calls[name], outputs[name] = g, call, out
            events[name] = (start, end)
            # Preserve each result before another variant reuses the buffers.
            if a.shared_workspace: snapshots[name] = tuple(t.clone() for t in out)
            if sq in (1, 4) and window == -1 and b > 1 and a.splits <= 0 and not name.startswith('installed') and '@s' not in name:
                dynamic_splits[name] = call.scheduler_workspace[32:32+b+1].cpu().tolist()
        compared = snapshots if a.shared_workspace else outputs
        comparison = {}
        if a.compare_baseline:
            expected_pair = compared[a.compare_baseline]
            for name, actual_pair in compared.items():
                if name != 'installed':
                    validate(actual_pair, expected_pair)
            for name, pair in compared.items():
                comparison[name] = []
                for actual, expected_tensor in zip(pair, expected_pair):
                    difference = (actual.float() - expected_tensor.float()).abs()
                    comparison[name].append(dict(max_abs=float(difference.max()), unequal=int((actual != expected_tensor).sum()), count=actual.numel()))
        check_exact(compared)
        for group in a.exact_groups:
            members = [name for name in group.split(',') if name in dynamic_splits]
            for name in members[1:]:
                assert dynamic_splits[name] == dynamic_splits[members[0]], 'Matched variants changed dynamic KV splits'
        timed_names = native_names if a.only_libraries else names
        times = {n: [] for n in timed_names}
        for repeat in range(a.repeats):
            order = list(timed_names); rng.shuffle(order)
            for name in order:
                # Different model layers have different KV caches. Evict the
                # previous replay's cache before timing this single operator.
                flush.zero_()
                start, end = events[name]
                graphs[name].replay(); end.synchronize()
                times[name].append(start.elapsed_time(end)*1000)
        row = dict(kind=kind, batch=b, sq=sq, sk=sk, window=window, baseline_comparison=comparison,
            q_pattern=a.q_pattern, qlens=qlens,
            capacity_k=d['table'].shape[1]*64, kv_pattern=a.kv_pattern, klens=klens,
            physical_pages=d['k'].shape[0], compact_pages=a.compact_pages,
            median_us={n: statistics.median(v) for n, v in times.items()}, samples_us=times,
            native_num_splits={n: getattr(c, 'num_splits', None) for n, c in calls.items() if not n.startswith('installed')})
        if sq in (1, 4) and window == -1 and b > 1 and a.splits <= 0:
            row['dynamic_splits_by_request'] = dynamic_splits
        if a.profile:
            row['kernel_names_by_variant'] = {}
            row['kernel_times_us_by_variant'] = {}
            for name in names:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    calls[name]()
                    torch.cuda.synchronize()
                row['kernel_names_by_variant'][name] = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                row['kernel_times_us_by_variant'][name] = [dict(name=e.name, us=e.device_time_total) for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        if a.graph_replay_stress:
            assert window == -1
            row['replay_checks'] = []
            patterns = [[sk]*b, [min(sk,4096)]*b,
                        [sk-(sk//2)*i//max(1,b-1) for i in range(b)],
                        [sk]+[min(sk,4097+i) for i in range(1,b)], [sk]*b]
            if a.kv_short_length is not None:
                short_k = min(sk, a.kv_short_length)
                patterns += [[short_k]*b, [sk]+[short_k]*(b-1), [sk]*b]
            for lengths in patterns:
                d['klens'] = lengths
                d['seq'].copy_(torch.tensor(lengths,device='cuda',dtype=torch.int32))
                expected_replay = reference(d, sampled=True)
                errors = {}
                replay_outputs = {}
                for name in names:
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    errors[name] = validate(outputs[name], expected_replay, expected_replay[2])
                    replay_outputs[name] = tuple(t.clone() for t in outputs[name])
                check_exact(replay_outputs)
                row['replay_checks'].append(dict(min_k=min(lengths),max_k=max(lengths),errors=errors))
        check_guards(d)
        results['benchmarks'].append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['samples_us', 'kernel_names_by_variant']}), flush=True)
        save()
        del d, graphs, calls, outputs, events, snapshots, dynamic_splits, expected, call, out, g
        gc.collect(); torch.cuda.empty_cache()
save()
