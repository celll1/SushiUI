"""SenseNova batch_size > 1: packed prefix + varlen attention (CPU-only)."""
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.attention import dispatch_attention_varlen  # noqa: E402
from core.models.sensenova.vendor.modeling_qwen3 import (  # noqa: E402
    PackedGenPlan,
    PackedSegments,
    create_block_causal_mask,
    create_packed_block_causal_mask,
    is_plain_causal_thw_index_packed,
)
from core.training.base_trainer import BaseTrainer  # noqa: E402


def _cu(*lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32)


def test_packed_segments_describe_the_layout():
    seg = PackedSegments(_cu(5, 3, 7))
    assert seg.count == 3
    assert seg.lengths == [5, 3, 7]
    assert seg.total == 15
    assert seg.max_seqlen == 7
    assert seg.bounds() == [(0, 5), (5, 8), (8, 15)]
    with pytest.raises(ValueError):
        PackedSegments(torch.tensor([0]))


def test_packed_mask_is_block_diagonal_of_the_single_masks():
    idx_a = torch.arange(5)
    idx_b = torch.tensor([0, 1, 1, 2])  # a t-index that repeats: not plain causal
    seg = PackedSegments(_cu(5, 4))
    packed = create_packed_block_causal_mask(torch.cat([idx_a, idx_b]), seg)
    assert packed.shape == (1, 1, 9, 9)
    assert torch.equal(packed[0, 0, :5, :5], create_block_causal_mask(idx_a)[0, 0])
    assert torch.equal(packed[0, 0, 5:, 5:], create_block_causal_mask(idx_b)[0, 0])
    assert torch.isinf(packed[0, 0, :5, 5:]).all()
    assert torch.isinf(packed[0, 0, 5:, :5]).all()
    assert is_plain_causal_thw_index_packed(torch.cat([idx_a, idx_a]), PackedSegments(_cu(5, 5)))
    assert not is_plain_causal_thw_index_packed(torch.cat([idx_a, idx_b]), seg)
    with pytest.raises(ValueError, match="cu_seqlens"):
        create_packed_block_causal_mask(torch.arange(8), seg)


def test_gen_plan_interleaves_each_items_prefix_with_its_own_image_tokens():
    seg = PackedSegments(_cu(3, 2))
    plan = PackedGenPlan(seg, batch=2, cur_len=4, device="cpu")
    # cat([prefix(5), cur(8)]) -> [p0 p0 p0 c0 c0 c0 c0 | p1 p1 c1 c1 c1 c1]
    assert plan.k_order.tolist() == [0, 1, 2, 5, 6, 7, 8, 3, 4, 9, 10, 11, 12]
    assert plan.cu_seqlens_q.tolist() == [0, 4, 8]
    assert plan.cu_seqlens_k.tolist() == [0, 7, 13]
    assert plan.max_seqlen_q == 4 and plan.max_seqlen_k == 7
    with pytest.raises(ValueError, match="segment"):
        PackedGenPlan(seg, batch=3, cur_len=4, device="cpu")


def test_varlen_native_fallback_matches_per_item_sdpa():
    torch.manual_seed(0)
    h, hkv, d = 4, 2, 8
    lengths_q, lengths_k = [3, 5], [7, 6]
    q = torch.randn(sum(lengths_q), h, d)
    k = torch.randn(sum(lengths_k), hkv, d)
    v = torch.randn(sum(lengths_k), hkv, d)
    out = dispatch_attention_varlen(
        q, k, v, _cu(*lengths_q), _cu(*lengths_k), max(lengths_q), max(lengths_k),
        backend="native", is_causal=False, scale=0.5,
    )
    assert out.shape == q.shape
    sq = sk = 0
    for lq, lk in zip(lengths_q, lengths_k):
        qi = q[sq:sq + lq].transpose(0, 1).unsqueeze(0)
        ki = k[sk:sk + lk].transpose(0, 1).unsqueeze(0).repeat_interleave(h // hkv, dim=1)
        vi = v[sk:sk + lk].transpose(0, 1).unsqueeze(0).repeat_interleave(h // hkv, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, scale=0.5)[0].transpose(0, 1)
        assert torch.allclose(out[sq:sq + lq], ref, atol=1e-5)
        sq += lq
        sk += lk


def test_varlen_causal_fallback_is_bottom_right_aligned():
    torch.manual_seed(1)
    q = torch.randn(2, 1, 4)
    k = torch.randn(5, 1, 4)
    v = torch.randn(5, 1, 4)
    out = dispatch_attention_varlen(q, k, v, _cu(2), _cu(5), 2, 5, backend="native", is_causal=True)
    # query row 0 may see keys 0..3, row 1 keys 0..4
    mask = torch.tensor([[True, True, True, True, False], [True] * 5])
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), attn_mask=mask
    )[0].transpose(0, 1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_batch_encode_packs_several_prompts_and_stashes_per_item_labels():
    calls = []

    class _Arch:
        def encode_prompts(self, trainer, prompts, *, requires_grad, reference_image_paths, cfg_null):
            calls.append((list(prompts), requires_grad, reference_image_paths, cfg_null))
            return "packed"

    owner = SimpleNamespace(train_text_encoder=True, arch=_Arch())
    items = [("a cat", None, False), ("a dog", ["ref.png"], True)]
    assert BaseTrainer._encode_sensenova_batch_prefix(owner, items) == "packed"
    assert calls == [(["a cat", "a dog"], True, [None, ["ref.png"]], [False, True])]
    assert owner._sensenova_prefix_cfg_null == [False, True]
    assert owner._sensenova_batch_ref_paths == [None, ["ref.png"]]


def test_packed_mnt_conditioning_reuses_matching_labels_and_memoizes_the_rest():
    calls = []

    class _Arch:
        def encode_prompts(self, trainer, prompts, *, requires_grad, reference_image_paths, cfg_null):
            calls.append((requires_grad, list(cfg_null)))
            return ("rebuilt", tuple(cfg_null))

    owner = SimpleNamespace(
        train_text_encoder=False, arch=_Arch(), sensenova_four_phase=None,
        _sensenova_prefix_cfg_null=[False, False], _sensenova_batch_ref_paths=None,
        _sensenova_alt_cfg_null_prefixes={},
    )
    owner._sensenova_mnt_conditioning_packed = MethodType(
        BaseTrainer._sensenova_mnt_conditioning_packed, owner)
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, owner)
    assembly = object()
    assert conditioning(assembly, captions=["a", "b"], mnt_index=0, cfg_null=[True, False])[3] is assembly
    assert conditioning(assembly, captions=["a", "b"], mnt_index=1, cfg_null=[False, False])[3] is assembly
    first = conditioning(assembly, captions=["a", "b"], mnt_index=1, cfg_null=[True, False])[3]
    again = conditioning(assembly, captions=["a", "b"], mnt_index=2, cfg_null=[True, False])[3]
    other = conditioning(assembly, captions=["a", "b"], mnt_index=3, cfg_null=[False, True])[3]
    assert first == ("rebuilt", (True, False)) and again is first
    assert other == ("rebuilt", (False, True))
    assert calls == [(False, [True, False]), (False, [False, True])]

    owner.train_text_encoder = True
    conditioning(assembly, captions=["a", "b"], mnt_index=1, cfg_null=[False, False])
    assert calls[-1] == (True, [False, False])


def test_packed_mnt_conditioning_refuses_a_missing_label_vector():
    owner = SimpleNamespace(train_text_encoder=False, arch=None, sensenova_four_phase=None,
                            _sensenova_prefix_cfg_null=False)
    owner._sensenova_mnt_conditioning_packed = MethodType(
        BaseTrainer._sensenova_mnt_conditioning_packed, owner)
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, owner)
    with pytest.raises(RuntimeError, match="per-item labels"):
        conditioning(object(), captions=["a", "b"], mnt_index=1, cfg_null=[True, False])
