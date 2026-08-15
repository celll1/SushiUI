"""``core.models.minimax_music3.vocab_view`` -- design doc phase 10.

Pins ``FullVocabView``'s five methods bitwise identical to the pre-phase-10 inline
expressions they replaced (a synthetic, correctly-sized ``nn.Module`` LM -- no real
weights needed), so a future simplification (e.g. `mask_logits` becoming a no-op on
both views) cannot pass silently. Also pins `PrunedVocabView.decode_sample`'s row-0-EOA
/ `sampled - 1` mapping and `resolve_vocab_view`'s dispatch, including the `None` case.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_vocab_view_test.py -v
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3 import vocab_view as vv  # noqa: E402
from core.models.minimax_music3.defaults import (  # noqa: E402
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
)


def _fake_full_vocab_lm(seed=1):
    """A real (tiny-hidden-dim, real-vocab-width) nn.Module shaped like the checkpoint's
    `Qwen3ForCausalLM`: `.model.embed_tokens` and `.lm_head` span the full 200,000-row
    vocabulary, so `AUDIO_CODE_OFFSET`/`AUDIO_END_TOKEN_ID` index real, distinct rows."""
    torch.manual_seed(seed)
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = nn.Embedding(200_000, 4)
    lm.lm_head = nn.Linear(4, 200_000, bias=False)
    lm.config = SimpleNamespace(vocab_size=200_000)
    return lm


def _inline_vocab_mask(vocab_size, device):
    """The pre-phase-10 inline mask construction, verbatim."""
    mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
    mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
    mask[AUDIO_END_TOKEN_ID] = False
    return mask


def test_full_embed_text_matches_inline_expression():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)
    text_ids = torch.tensor([[1, 2, 3, 151670]])
    assert torch.equal(view.embed_text(text_ids), lm.model.embed_tokens(text_ids))


def test_full_embed_semantic_code_matches_inline_expression():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)
    code = torch.tensor([[0, 16383]])
    assert torch.equal(view.embed_semantic_code(code), lm.model.embed_tokens(code + AUDIO_CODE_OFFSET))


def test_full_audio_logits_matches_inline_expression():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)
    last_hidden = torch.randn(2, 4)

    expected = lm.lm_head(last_hidden).float()
    expected = expected.masked_fill(_inline_vocab_mask(200_000, expected.device), -float("inf"))

    got = view.audio_logits(last_hidden)
    assert torch.equal(got, expected)


def test_full_mask_logits_matches_inline_expression():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)
    guided = torch.randn(1, 200_000)
    expected = guided.masked_fill(_inline_vocab_mask(200_000, guided.device).unsqueeze(0), -float("inf"))
    got = view.mask_logits(guided)
    assert torch.equal(got, expected)


def test_full_decode_sample_matches_inline_expression():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)

    sampled_eoa = torch.tensor([AUDIO_END_TOKEN_ID])
    is_eoa, code = view.decode_sample(sampled_eoa)
    assert is_eoa is True
    assert torch.equal(code, sampled_eoa - AUDIO_CODE_OFFSET)

    sampled_code = torch.tensor([AUDIO_CODE_OFFSET + 42])
    is_eoa, code = view.decode_sample(sampled_code)
    assert is_eoa is False
    assert code.item() == 42


def test_full_vocab_mask_is_cached_by_device_not_rebuilt():
    lm = _fake_full_vocab_lm()
    view = vv.FullVocabView(lm)
    last_hidden = torch.randn(2, 4)
    view.audio_logits(last_hidden)
    cached = view._mask_by_device[torch.device("cpu")]
    view.audio_logits(last_hidden)
    assert view._mask_by_device[torch.device("cpu")] is cached


# ---------------------------------------------------------------------------
# PrunedVocabView
# ---------------------------------------------------------------------------

def _fake_pruned_vocab_lm():
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = nn.Embedding(151_675, 4)
    lm.model.embed_tokens_audio = nn.Embedding(SEMANTIC_VOCAB_SIZE, 4)
    lm.lm_head_pruned = nn.Linear(4, SEMANTIC_VOCAB_SIZE + 1, bias=False)
    lm.config = SimpleNamespace(vocab_size=151_675)
    return lm


def test_pruned_embed_semantic_code_uses_the_audio_table_directly_no_offset():
    lm = _fake_pruned_vocab_lm()
    view = vv.PrunedVocabView(lm)
    code = torch.tensor([[0, SEMANTIC_VOCAB_SIZE - 1]])
    assert torch.equal(view.embed_semantic_code(code), lm.model.embed_tokens_audio(code))


def test_pruned_mask_logits_is_identity():
    lm = _fake_pruned_vocab_lm()
    view = vv.PrunedVocabView(lm)
    logits = torch.randn(1, SEMANTIC_VOCAB_SIZE + 1)
    assert view.mask_logits(logits) is logits


def test_pruned_decode_sample_row_zero_is_end_of_audio():
    lm = _fake_pruned_vocab_lm()
    view = vv.PrunedVocabView(lm)
    is_eoa, code = view.decode_sample(torch.tensor([0]))
    assert is_eoa is True
    assert code.item() == -1  # 0 - 1; never read downstream when is_eoa is True


def test_pruned_decode_sample_row_c_plus_one_is_semantic_code_c():
    lm = _fake_pruned_vocab_lm()
    view = vv.PrunedVocabView(lm)
    is_eoa, code = view.decode_sample(torch.tensor([1]))
    assert is_eoa is False
    assert code.item() == 0
    is_eoa, code = view.decode_sample(torch.tensor([SEMANTIC_VOCAB_SIZE]))
    assert is_eoa is False
    assert code.item() == SEMANTIC_VOCAB_SIZE - 1


# ---------------------------------------------------------------------------
# resolve_vocab_view
# ---------------------------------------------------------------------------

def test_resolve_vocab_view_none_for_none():
    assert vv.resolve_vocab_view(None) is None


def test_resolve_vocab_view_selects_full_when_no_lm_head_pruned_attribute():
    lm = _fake_full_vocab_lm()
    assert isinstance(vv.resolve_vocab_view(lm), vv.FullVocabView)


def test_resolve_vocab_view_selects_pruned_when_lm_head_pruned_attribute_present():
    lm = _fake_pruned_vocab_lm()
    assert isinstance(vv.resolve_vocab_view(lm), vv.PrunedVocabView)
