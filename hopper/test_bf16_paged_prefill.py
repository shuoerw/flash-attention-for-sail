"""Numerical coverage for BF16 paged prefill with Q/K=192, V=128 and GQA=16."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="PPU GPU required")

from benchmark_mimo_prefill import check_output, make_inputs


@pytest.mark.parametrize("q_lengths,kv_lengths,window_left", [
    ([4096], [65536], -1),         # Tuned dispatch boundary.
    ([4097], [65537], -1),         # Partial Q tile and final KV page.
    ([4099], [65664], -1),         # Different three-stage pipeline remainder.
    ([4096, 4096], [65536, 65537], -1),  # Uniform Q, different live KV lengths.
    ([4096, 2049], [65536, 32769], -1),  # Ragged Q batch uses the original tile.
    ([4096], [32768], -1),         # Short-context performance fallback.
    ([1024], [65536], 128),        # Sliding-window attention keeps its tile.
    ([1], [65536], -1),           # Decode stays on its existing path.
    ([129], [257], -1),           # Small prefill and page-boundary masking.
])
@torch.inference_mode()
def test_bf16_paged_prefill(q_lengths, kv_lengths, window_left):
    from flash_attn_interface import flash_attn_with_kvcache

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_inputs(q_lengths, kv_lengths, window_left=window_left)
    out = flash_attn_with_kvcache(**inputs)
    check_output(inputs, out, all_rows=True)


@torch.inference_mode()
def test_bf16_paged_prefill_short_visible_prefix():
    from flash_attn_interface import flash_attn_with_kvcache

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_inputs([65536], [65537])
    # Early query rows see fewer KV blocks than the three pipeline stages.
    check_output(inputs, flash_attn_with_kvcache(**inputs))


@torch.inference_mode()
def test_bf16_paged_prefill_graph_replay():
    from flash_attn_interface import flash_attn_with_kvcache

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = make_inputs([4097], [65537])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flash_attn_with_kvcache(**inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flash_attn_with_kvcache(**inputs)
    # Replay must read fresh data, rather than merely preserve captured outputs.
    inputs["q"].normal_()
    graph.replay()
    torch.cuda.synchronize()
    check_output(inputs, out, all_rows=True)
