"""MiMo paged prefill regressions through the FA3 package API."""

import itertools
import os

import pytest
import torch

from flash_attn_interface import flash_attn_with_kvcache
from test_hggc_graph import capture_graph

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or any(os.getenv(f"FLASH_ATTENTION_DISABLE_{feature}", "FALSE") == "TRUE"
           for feature in ("PAGEDKV", "PACKGQA", "HDIM192", "VARLEN", "HDIMV")),
    reason="MiMo prefill requires BF16 paged GQA with D192/V128",
)


@pytest.fixture(autouse=True)
def fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def make_inputs(qlens, klens):
    torch.manual_seed(17)
    counts = [(k + 63) // 64 for k in klens]
    pages = sum(counts)
    mapping = torch.randperm(pages, device="cuda", dtype=torch.int32)
    table = torch.full((len(qlens), max(counts) + 1), -1, device="cuda", dtype=torch.int32)
    offset = 0
    for row, count in enumerate(counts):
        table[row, :count] = mapping[offset:offset + count]
        offset += count
    return dict(
        q=torch.randn(sum(qlens), 16, 192, device="cuda", dtype=torch.bfloat16),
        k_cache=torch.randn(pages, 64, 1, 192, device="cuda", dtype=torch.bfloat16),
        v_cache=torch.randn(pages, 64, 1, 128, device="cuda", dtype=torch.bfloat16),
        page_table=table,
        cu_seqlens_q=torch.tensor([0, *itertools.accumulate(qlens)], device="cuda", dtype=torch.int32),
        cache_seqlens=torch.tensor(klens, device="cuda", dtype=torch.int32),
        max_seqlen_q=max(qlens), max_seqlen_k=max(klens),
        causal=True, pack_gqa=True, return_softmax_lse=True,
    )


def check_reference(inputs, qlens, klens, result):
    out, lse = result[:2]
    offset = 0
    for batch, (sq, sk) in enumerate(zip(qlens, klens)):
        ids = inputs["page_table"][batch, :(sk + 63) // 64].long()
        k = inputs["k_cache"][ids].reshape(-1, 192)[:sk].float()
        v = inputs["v_cache"][ids].reshape(-1, 128)[:sk].float()
        # Check every query row, bounding FP32 reference memory for cached prefixes.
        for start in range(0, sq, 32):
            end = min(start + 32, sq)
            q = inputs["q"][offset + start:offset + end].float().transpose(0, 1)
            scores = (q @ k.T) * (192 ** -0.5)
            positions = torch.arange(start, end, device="cuda")[:, None] + sk - sq
            scores.masked_fill_(torch.arange(sk, device="cuda")[None, :] > positions, -torch.inf)
            expected_lse = scores.logsumexp(-1)
            probs = (scores - expected_lse[..., None]).exp().nan_to_num()
            expected = (probs @ v).transpose(0, 1)
            torch.testing.assert_close(out[offset + start:offset + end].float(), expected,
                                       atol=0.025, rtol=0.025)
            torch.testing.assert_close(lse[:, offset + start:offset + end], expected_lse,
                                       atol=0.003, rtol=0.001)
        offset += sq
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_splits", [1, 2])
@pytest.mark.parametrize("qlens,klens", [
    pytest.param([4096], [4096], id="fresh"),
    pytest.param([464], [1024], id="below-query-threshold"),
    pytest.param([465], [1024], id="query-threshold"),
    pytest.param([512], [32704], id="511-kv-tiles"),
    pytest.param([512], [32768], id="32k-capacity"),
    pytest.param([512], [32769], id="above-32k-partial-page"),
    pytest.param([512], [65536], id="512-tiles-per-split"),
    pytest.param([513, 0, 257], [65539, 0, 32769], id="mixed-empty-partial-pages"),
])
@torch.inference_mode()
def test_mimo_prefill(qlens, klens, num_splits):
    if num_splits > 1 and os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE":
        pytest.skip("SplitKV disabled in this build")
    inputs = make_inputs(qlens, klens)
    result = flash_attn_with_kvcache(**inputs, num_splits=num_splits)
    check_reference(inputs, qlens, klens, result)


@pytest.mark.parametrize("num_splits", [0, 2])
@torch.inference_mode()
def test_mimo_prefill_graph_replay(num_splits):
    if num_splits > 1 and os.getenv("FLASH_ATTENTION_DISABLE_SPLIT", "FALSE") == "TRUE":
        pytest.skip("SplitKV disabled in this build")
    qlens = [513, 257]
    inputs = make_inputs(qlens, [65539, 32769])
    table = inputs["page_table"].clone()
    graph, result = capture_graph(lambda: flash_attn_with_kvcache(**inputs, num_splits=num_splits))
    # Keep the captured capacity fixed while requests cross the live-tile cutoff.
    for klens in ([65539, 32769], [32768, 257], [0, 65], [65539, 32769]):
        inputs["cache_seqlens"].copy_(torch.tensor(klens, device="cuda", dtype=torch.int32))
        inputs["page_table"].fill_(-1)
        for row, length in enumerate(klens):
            count = (length + 63) // 64
            inputs["page_table"][row, :count] = table[row, :count]
        graph.replay()
        torch.cuda.synchronize()
        check_reference(inputs, qlens, klens, result)
