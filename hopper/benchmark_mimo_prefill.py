"""Reproduce MiMo's BF16 paged prefill without loading the model.

Run baseline and candidate in separate processes with their respective builds.
Output is JSON Lines; GPU timing excludes input creation and reference checks.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def make_inputs(q_lengths, kv_lengths, device="cuda", page_size=64, window_left=-1):
    import torch

    if len(q_lengths) != len(kv_lengths) or not q_lengths:
        raise ValueError("Query and KV lengths must describe the same nonempty batch")
    if any(q <= 0 or k < q for q, k in zip(q_lengths, kv_lengths)):
        raise ValueError("Require 0 < query length <= KV length")
    if page_size <= 0 or page_size % 16:
        raise ValueError("Page size must be a positive multiple of 16")
    if window_left < -1:
        raise ValueError("Window must be -1 (full) or nonnegative")
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size
    num_pages = len(q_lengths) * pages_per_seq
    # Scrambled pages catch accidental assumptions about contiguous KV storage.
    table = torch.randperm(num_pages, device=device).to(torch.int32)
    table = table.reshape(len(q_lengths), pages_per_seq)
    q = torch.randn(sum(q_lengths), 16, 192, device=device, dtype=torch.bfloat16)
    k = torch.randn(num_pages, page_size, 1, 192, device=device, dtype=q.dtype)
    v = torch.randn(num_pages, page_size, 1, 128, device=device, dtype=q.dtype)
    cu_q = [0]
    for length in q_lengths:
        cu_q.append(cu_q[-1] + length)
    return dict(
        q=q, k_cache=k, v_cache=v, page_table=table,
        cu_seqlens_q=torch.tensor(cu_q, device=device, dtype=torch.int32),
        cache_seqlens=torch.tensor(kv_lengths, device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lengths), causal=True,
        window_size=(-1, -1) if window_left == -1 else (window_left, 0),
        num_splits=1, pack_gqa=True,
    )


def check_output(inputs, out, all_rows=False):
    """Independent FP32 reference; one Q head at a time limits scratch memory."""
    import torch

    cu_q = inputs["cu_seqlens_q"].tolist()
    lengths = inputs["cache_seqlens"].tolist()
    worst_error = 0.0
    for batch, kv_len in enumerate(lengths):
        q_len = cu_q[batch + 1] - cu_q[batch]
        rows = list(range(q_len)) if all_rows else sorted({
            i for i in (0, 3, 4, 7, 8, 63, 64, 127, 128,
                        q_len // 2, q_len - 2, q_len - 1) if 0 <= i < q_len
        })
        rows = torch.tensor(rows, device=out.device, dtype=torch.long)
        pages = inputs["page_table"][batch].long()
        k = inputs["k_cache"][pages].reshape(-1, 192)[:kv_len].float()
        v = inputs["v_cache"][pages].reshape(-1, 128)[:kv_len].float()
        q = inputs["q"][cu_q[batch] + rows].float()
        right = rows[:, None] + kv_len - q_len
        keys = torch.arange(kv_len, device=out.device)[None, :]
        allowed = keys <= right
        window_left = inputs["window_size"][0]
        if window_left >= 0:
            allowed = allowed & (keys >= right - window_left)
        # Short attention spans have larger outputs and BF16 probability-rounding
        # error. Keep the tighter tolerance for the long-context optimization.
        atol = 3e-3 if window_left >= 0 or kv_len - q_len < 1024 else 3e-4
        for head in range(16):
            scores = (q[:, head] @ k.T) * (192 ** -0.5)
            scores.masked_fill_(~allowed, float("-inf"))
            expected = torch.softmax(scores, dim=-1) @ v
            actual = out[cu_q[batch] + rows, head].float()
            torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-2)
            worst_error = max(worst_error, (actual - expected).abs().max().item())
    return worst_error


def time_calls(fn, warmup, repeats):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Build identifier for A/B results")
    parser.add_argument("--q-length", type=int, default=16384)
    parser.add_argument("--kv-lengths", type=int, nargs="+",
                        default=[344064, 360448, 376832, 786432])
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--window-left", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", type=Path,
                        help="Optional untimed Chrome trace of the final shape")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("Require warmup >= 0 and repeats >= 1")

    import torch
    import flash_attn_3._C as extension
    import flash_attn_interface as fa

    if not torch.cuda.is_available():
        raise RuntimeError("A PPU CUDA-compatible PyTorch runtime is required")
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    binary_hash = hashlib.sha256()
    with open(extension.__file__, "rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            binary_hash.update(chunk)
    print(json.dumps(dict(
        label=args.label, torch=torch.__version__,
        device=torch.cuda.get_device_name(), interface=fa.__file__,
        extension=extension.__file__, extension_sha256=binary_hash.hexdigest(),
        q_heads=16, kv_heads=1, qk_dim=192, v_dim=128, dtype="bfloat16",
    )), flush=True)

    with torch.inference_mode():
        for kv_len in args.kv_lengths:
            inputs = make_inputs([args.q_length], [kv_len], page_size=args.page_size,
                                 window_left=args.window_left)
            run = lambda: fa.flash_attn_with_kvcache(**inputs)
            error = check_output(inputs, run())
            samples = time_calls(run, args.warmup, args.repeats)
            print(json.dumps(dict(
                label=args.label, q_length=args.q_length, kv_length=kv_len,
                page_size=args.page_size, window_left=args.window_left,
                median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), samples_ms=samples, checked_max_abs_error=error,
            )), flush=True)
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profile:
                run()
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(args.profile))


if __name__ == "__main__":
    main()
