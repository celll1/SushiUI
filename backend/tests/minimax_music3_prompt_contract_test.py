"""Checkpoint-contract tests for MiniMax Music 3's prompt assembly.

The prompt assembled by ``MiniMaxMusic3Pipeline.encode_text`` (and the pure
helpers it calls, ``_clean_caption`` / ``_normalize_lyrics``) is a checkpoint
contract ported from upstream ``diffusers`` PR #14456's ``encoders.py``: even
whitespace-level changes to the assembled text change the generated audio (see
``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Dependency gate"). These tests pin
that string-level behaviour so a future refactor of ``pipeline.py`` cannot
silently drift from the upstream algorithm. No model weights, tokenizer files,
or GPU are needed -- everything here is pure-Python string transforms plus a
spy tokenizer that records the exact text it was asked to tokenize.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3.defaults import AUDIO_CFG_TOKEN_ID, MAX_PROMPT_TOKENS
from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline, _clean_caption, _normalize_lyrics


# ---------------------------------------------------------------------------
# _clean_caption: markdown-strip + special-tag rewrite (upstream
# encoders.py::_clean_caption).
# ---------------------------------------------------------------------------
def test_clean_caption_rewrites_special_tags():
    assert _clean_caption("<|genre pop|>") == "genre is pop"
    # A tag with no second word (no split point) is left as its bare content.
    assert _clean_caption("<|solo|>") == "solo"


def test_clean_caption_strips_markdown_headings_and_bullets():
    caption = "# Heading\n* bullet one\n- bullet two\n+ bullet three\nplain text"
    cleaned = _clean_caption(caption)
    assert cleaned == "Heading\nbullet one\nbullet two\nbullet three\nplain text"


def test_clean_caption_strips_bold_and_italic_markers():
    assert _clean_caption("**bold** and *italic* text") == "bold and italic text"


def test_clean_caption_strips_horizontal_rules_and_bullet_dots_and_collapses_blank_lines():
    caption = "line one\n\n\n---\nline two\n• bulleted\n    indented"
    cleaned = _clean_caption(caption)
    # Horizontal rule line removed, blank-line runs collapsed to one, "• " and
    # 4-space indents stripped (upstream's literal `.replace("• ", "")` /
    # `.replace("    ", "")`).
    assert "---" not in cleaned
    assert "\n\n" not in cleaned
    assert "• " not in cleaned


# ---------------------------------------------------------------------------
# _normalize_lyrics: structure-tag normalization (upstream
# encoders.py::_normalize_lyrics).
# ---------------------------------------------------------------------------
def test_normalize_lyrics_prepends_start_tag():
    assert _normalize_lyrics("hello").startswith("[start]\n")


def test_normalize_lyrics_lowercases_structure_tags():
    assert "[verse]" in _normalize_lyrics("[VERSE]\nline one")


def test_normalize_lyrics_drops_text_sharing_a_line_with_a_leading_tag():
    # Checkpoint input contract (see design doc's generation-parameter table):
    # text on the SAME line as a leading structure tag is dropped, keeping
    # only the tag itself.
    result = _normalize_lyrics("[verse] this text is dropped")
    assert "this text is dropped" not in result
    assert "[verse]" in result


def test_normalize_lyrics_keeps_text_on_its_own_line():
    result = _normalize_lyrics("[verse]\nthis text survives")
    assert "this text survives" in result


def test_normalize_lyrics_splits_bracket_and_caret_separated_tags_onto_new_lines():
    result = _normalize_lyrics("[verse] [chorus]")
    lines = result.split("\n")
    assert "[verse]" in lines
    assert "[chorus]" in lines
    result_caret = _normalize_lyrics("line one ^ line two")
    assert "line one" in result_caret.split("\n")
    assert "line two" in result_caret.split("\n")


# ---------------------------------------------------------------------------
# encode_text: full prompt assembly + the conditional/unconditional token-id
# pair (upstream encoders.py::MiniMaxMusic3TextEncoderStep).
# ---------------------------------------------------------------------------
class _SpyTokenizer:
    """Records the exact text passed in and returns a small deterministic id sequence."""

    def __init__(self, num_tokens: int = 6):
        self.calls = []
        self._num_tokens = num_tokens

    def __call__(self, text, return_tensors="pt"):
        self.calls.append(text)
        ids = list(range(10, 10 + self._num_tokens))
        return {"input_ids": torch.tensor([ids])}


def _pipeline_with_tokenizer(tokenizer):
    # encode_text only touches `self.tokenizer` and `self.execution_device`; every other component can stay None
    # for this test (they are never read on this path).
    pipeline = MiniMaxMusic3Pipeline(
        tokenizer=tokenizer,
        language_model=None,
        rvq_depth_decoder=None,
        condition_encoder=None,
        transformer=None,
        scheduler=None,
        vocoder=None,
        execution_device=torch.device("cpu"),
    )
    return pipeline


def test_encode_text_assembles_the_pinned_special_token_prompt():
    tokenizer = _SpyTokenizer()
    pipeline = _pipeline_with_tokenizer(tokenizer)

    pipeline.encode_text("a pop song", "[verse]\nhello world")

    assert len(tokenizer.calls) == 1
    assembled = tokenizer.calls[0]
    # Pinned byte-for-byte against upstream's f-string assembly in encoders.py::MiniMaxMusic3TextEncoderStep.
    expected = (
        "<|im_start|><|caption_start|>a pop song<|caption_end|>"
        "<|lyrics_start|>[start]\n[verse]\nhello world<|lyrics_end|><|im_end|><|audio_start|>"
    )
    assert assembled == expected


def test_encode_text_returns_conditional_and_unconditional_rows():
    tokenizer = _SpyTokenizer(num_tokens=6)
    pipeline = _pipeline_with_tokenizer(tokenizer)

    text_ids = pipeline.encode_text("caption", "[verse]\nlyrics")

    assert text_ids.shape == (2, 6)
    conditional, unconditional = text_ids[0], text_ids[1]
    assert torch.equal(conditional, torch.tensor([10, 11, 12, 13, 14, 15]))
    # Every token except the first and the two trailing structure tokens is replaced by the audio-CFG token.
    assert unconditional[0].item() == 10
    assert unconditional[-1].item() == 15
    assert unconditional[-2].item() == 14
    assert torch.all(unconditional[1:-2] == AUDIO_CFG_TOKEN_ID)


def test_encode_text_rejects_empty_prompt_or_lyrics():
    pipeline = _pipeline_with_tokenizer(_SpyTokenizer())
    with pytest.raises(ValueError):
        pipeline.encode_text("", "[verse]\nlyrics")
    with pytest.raises(ValueError):
        pipeline.encode_text("caption", "")
    with pytest.raises(ValueError):
        pipeline.encode_text("caption", "   ")


def test_encode_text_rejects_prompts_over_the_token_cap():
    tokenizer = _SpyTokenizer(num_tokens=MAX_PROMPT_TOKENS + 1)
    pipeline = _pipeline_with_tokenizer(tokenizer)
    with pytest.raises(ValueError, match=str(MAX_PROMPT_TOKENS)):
        pipeline.encode_text("caption", "[verse]\nlyrics")


def test_encode_text_accepts_prompts_at_exactly_the_token_cap():
    tokenizer = _SpyTokenizer(num_tokens=MAX_PROMPT_TOKENS)
    pipeline = _pipeline_with_tokenizer(tokenizer)
    text_ids = pipeline.encode_text("caption", "[verse]\nlyrics")
    assert text_ids.shape == (2, MAX_PROMPT_TOKENS)
