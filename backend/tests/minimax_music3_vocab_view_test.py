"""Behavioral contracts for the two MiniMax Music 3 vocabulary layouts."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.models.minimax_music3 import vocab_view as vv
from core.models.minimax_music3.defaults import (
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    SEMANTIC_VOCAB_SIZE,
)


def _full_vocab_lm(seed=1):
    torch.manual_seed(seed)
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = nn.Embedding(200_000, 4)
    lm.lm_head = nn.Linear(4, 200_000, bias=False)
    lm.config = SimpleNamespace(vocab_size=200_000)
    return lm


def _pruned_vocab_lm():
    lm = nn.Module()
    lm.model = nn.Module()
    lm.model.embed_tokens = nn.Embedding(151_675, 4)
    lm.model.embed_tokens_audio = nn.Embedding(SEMANTIC_VOCAB_SIZE, 4)
    lm.lm_head_pruned = nn.Linear(4, SEMANTIC_VOCAB_SIZE + 1, bias=False)
    lm.config = SimpleNamespace(vocab_size=151_675)
    return lm


def test_full_view_uses_audio_offset_mask_and_end_token():
    lm = _full_vocab_lm()
    view = vv.FullVocabView(lm)
    codes = torch.tensor([[0, SEMANTIC_VOCAB_SIZE - 1]])
    assert torch.equal(
        view.embed_semantic_code(codes),
        lm.model.embed_tokens(codes + AUDIO_CODE_OFFSET),
    )

    logits = view.audio_logits(torch.randn(1, 4))
    allowed = logits[0, AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE]
    assert torch.isfinite(allowed).all()
    assert torch.isfinite(logits[0, AUDIO_END_TOKEN_ID])
    assert torch.isneginf(logits[0, 0])

    is_end, code = view.decode_sample(torch.tensor([AUDIO_END_TOKEN_ID]))
    assert is_end and code.item() == AUDIO_END_TOKEN_ID - AUDIO_CODE_OFFSET
    is_end, code = view.decode_sample(torch.tensor([AUDIO_CODE_OFFSET + 42]))
    assert not is_end and code.item() == 42


def test_full_view_reuses_its_device_mask():
    view = vv.FullVocabView(_full_vocab_lm())
    hidden = torch.randn(1, 4)
    view.audio_logits(hidden)
    cached = view._mask_by_device[torch.device("cpu")]
    view.audio_logits(hidden)
    assert view._mask_by_device[torch.device("cpu")] is cached


def test_pruned_view_uses_direct_codes_and_row_zero_end_token():
    lm = _pruned_vocab_lm()
    view = vv.PrunedVocabView(lm)
    codes = torch.tensor([[0, SEMANTIC_VOCAB_SIZE - 1]])
    assert torch.equal(
        view.embed_semantic_code(codes),
        lm.model.embed_tokens_audio(codes),
    )

    logits = torch.randn(1, SEMANTIC_VOCAB_SIZE + 1)
    assert view.mask_logits(logits) is logits
    is_end, _ = view.decode_sample(torch.tensor([0]))
    assert is_end
    is_end, code = view.decode_sample(torch.tensor([SEMANTIC_VOCAB_SIZE]))
    assert not is_end and code.item() == SEMANTIC_VOCAB_SIZE - 1


def test_resolver_selects_the_loaded_vocabulary_layout():
    assert isinstance(vv.resolve_vocab_view(_full_vocab_lm()), vv.FullVocabView)
    assert isinstance(vv.resolve_vocab_view(_pruned_vocab_lm()), vv.PrunedVocabView)
