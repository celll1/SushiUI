"""Exactness checks for Qwen-Image 2.1 packed segmented attention."""

import os
import sys

import torch
import pytest

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import core.attention
from core.attention import AttentionMode
from core.attention.dispatch import dispatch_attention_varlen as _dispatch_varlen
from core.models.qwen_image_21.vendor.transformer import QwenImage21AttnProcessor


def _native_varlen(*args, **kwargs):
    kwargs["backend"] = "native"
    kwargs["mode"] = AttentionMode.TRAINING
    return _dispatch_varlen(*args, **kwargs)


def _dense_reference(query, key, value, key_valid, segments):
    outputs = []
    prefix_len = segments[-1][1]
    for start, end, is_text in segments:
        mask = key_valid[:, None, None, :end]
        if is_text:
            seg_len = end - start
            structure = torch.cat(
                [
                    torch.ones(seg_len, start, dtype=torch.bool),
                    torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool)),
                ],
                dim=1,
            )[None, None]
            mask = mask & structure
        output = torch.nn.functional.scaled_dot_product_attention(
                query[:, start:end].transpose(1, 2),
                key[:, :end].transpose(1, 2),
                value[:, :end].transpose(1, 2),
                attn_mask=mask,
            ).transpose(1, 2)
        outputs.append(
            output.masked_fill(~key_valid[:, start:end, None, None], 0.0)
        )
    outputs.append(
        torch.nn.functional.scaled_dot_product_attention(
            query[:, prefix_len:].transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=key_valid[:, None, None],
        ).transpose(1, 2)
    )
    return torch.cat(outputs, dim=1)


def test_packed_segmented_attention_matches_dense_forward_and_backward(monkeypatch):
    monkeypatch.setattr(core.attention, "dispatch_attention_varlen", _native_varlen)
    torch.manual_seed(7)
    query = torch.randn(2, 9, 2, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(2, 9, 2, 4, dtype=torch.float64, requires_grad=True)
    value = torch.randn(2, 9, 2, 4, dtype=torch.float64, requires_grad=True)
    key_valid = torch.tensor(
        [[True, True, False, True, True, True, True, True, True],
         [True, False, False, True, True, True, True, True, True]]
    )
    segments = [(0, 3, True), (3, 5, False)]

    packed = torch.cat(
        [
            QwenImage21AttnProcessor._varlen_segment(
                query, key, value, key_valid, start, end, is_causal=is_text
            )
            for start, end, is_text in [*segments, (5, 9, False)]
        ],
        dim=1,
    )
    dense = _dense_reference(query, key, value, key_valid, segments)
    torch.testing.assert_close(packed, dense, atol=1e-10, rtol=1e-8)
    cached = QwenImage21AttnProcessor._varlen_segment(
        query, key, value, key_valid, 0, 3, is_causal=True
    )
    torch.testing.assert_close(cached, dense[:, :3], atol=1e-10, rtol=1e-8)
    assert len(key_valid._qwen_image21_varlen_indices) == 3

    chunk = QwenImage21AttnProcessor._varlen_segment(
        query, key, value, key_valid, 5, 7, key_end=9, is_causal=False
    )
    chunk_reference = torch.nn.functional.scaled_dot_product_attention(
        query[:, 5:7].transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=key_valid[:, None, None, :],
    ).transpose(1, 2)
    torch.testing.assert_close(chunk, chunk_reference, atol=1e-10, rtol=1e-8)

    grad = torch.randn_like(packed)
    packed_grads = torch.autograd.grad(packed, (query, key, value), grad, retain_graph=True)
    dense_grads = torch.autograd.grad(dense, (query, key, value), grad)
    for actual, expected in zip(packed_grads, dense_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-8)


def test_batch_one_dense_segment_uses_views_without_pack_or_scatter(monkeypatch):
    seen = {}

    def _capture(q, k, v, *args, **kwargs):
        seen.update(q=q, k=k, v=v)
        return q.square()

    monkeypatch.setattr(core.attention, "dispatch_attention_varlen", _capture)
    real_cat = torch.cat

    def _unexpected_cat(*args, **kwargs):
        raise AssertionError("dense batch-one segment must not pack with torch.cat")

    monkeypatch.setattr(torch, "cat", _unexpected_cat)
    query = torch.randn(1, 12, 2, 4, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 12, 2, 4, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 12, 2, 4, dtype=torch.float64, requires_grad=True)
    key_valid = torch.ones(1, 12, dtype=torch.bool)

    out = QwenImage21AttnProcessor._varlen_segment(
        query, key, value, key_valid, 3, 12, is_causal=False
    )
    assert seen["q"].data_ptr() == query[0, 3].data_ptr()
    assert seen["k"].data_ptr() == key[0, 0].data_ptr()
    assert seen["v"].data_ptr() == value[0, 0].data_ptr()
    torch.testing.assert_close(out, query[:, 3:].square())
    out.sum().backward()
    assert query.grad is not None
    monkeypatch.setattr(torch, "cat", real_cat)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_varlen_kernel_matches_native_segment_backward():
    pytest.importorskip("flash_attn")
    torch.manual_seed(11)
    device = torch.device("cuda")
    query = torch.randn(2, 24, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 24, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(2, 24, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    key_valid = torch.ones(2, 24, dtype=torch.bool, device=device)
    key_valid[0, 3] = False
    key_valid[0, 18:20] = False
    key_valid[1, 1:3] = False
    key_valid[1, 19] = False
    start, end = 5, 20

    flash = QwenImage21AttnProcessor._varlen_segment(
        query, key, value, key_valid, start, end, is_causal=True
    )
    seg_len = end - start
    structure = torch.cat(
        [
            torch.ones(seg_len, start, dtype=torch.bool, device=device),
            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=device)),
        ],
        dim=1,
    )[None, None]
    native = torch.nn.functional.scaled_dot_product_attention(
        query[:, start:end].transpose(1, 2),
        key[:, :end].transpose(1, 2),
        value[:, :end].transpose(1, 2),
        attn_mask=structure & key_valid[:, None, None, :end],
    ).transpose(1, 2)
    native = native.masked_fill(~key_valid[:, start:end, None, None], 0.0)
    torch.testing.assert_close(flash, native, atol=1.5e-2, rtol=1.5e-2)

    grad = torch.randn_like(flash)
    flash_grads = torch.autograd.grad(flash, (query, key, value), grad, retain_graph=True)
    native_grads = torch.autograd.grad(native, (query, key, value), grad)
    for actual, expected in zip(flash_grads, native_grads):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
