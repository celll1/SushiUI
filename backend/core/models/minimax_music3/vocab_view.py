"""MiniMax Music 3 AR-loop vocabulary indirection -- design doc phase 10.

The AR loop (``core.models.minimax_music3.pipeline.MiniMaxMusic3Pipeline``)
needs three operations against "the language model's vocabulary": embed a
text-token id sequence, embed one semantic audio code, and turn a hidden
state into a distribution over "the next semantic code, or end-of-audio".
Two checkpoint layouts implement those three operations differently:

* the SHIPPED, DEFAULT layout (``official/language_model``, and the
  non-pruned flat repack) -- one ``embed_tokens`` table spanning text AND
  audio (audio code `c` lives at row `c + AUDIO_CODE_OFFSET`), one
  200,000-wide ``lm_head``, masked down to the 16,385 valid entries
  (16,384 semantic codes + end-of-audio) every call;
* the PRUNED layout (``text_encoders/minimax_music3_text_encoder_pruned_*``)
  -- ``embed_tokens`` holds ONLY text (151,675 rows), a SEPARATE
  ``embed_tokens_audio`` table holds the 16,384 semantic codes (indexed
  WITHOUT the offset), and a SEPARATE ``lm_head_pruned`` produces exactly
  16,385 logits directly (row 0 end-of-audio, rows 1..16384 semantic codes
  0..16383 -- see ``pruned_text_encoder_remap.EOA_LM_HEAD_ROW``'s docstring
  for how that was determined).

This module is the one place that difference is decided, so the AR loop in
``pipeline.py`` calls three methods (``embed_text``, ``embed_semantic_code``,
``audio_logits``) plus ``mask_logits``/``decode_sample`` and never branches on
which layout is loaded. ``resolve_vocab_view`` picks the implementation once,
from the LOADED module's own shape (``hasattr(language_model, "lm_head_pruned")``)
-- the same "detect from what IS there" convention
``flat_remap.is_pruned_flat_text_encoder`` already uses for the flat file's
header, applied here to the loaded module instead.

The full-vocabulary path (``FullVocabView``) must stay bit-identical to the
pre-phase-10 code -- pinned by ``minimax_music3_vocab_view_test.py``, not
only by regression. Seeds do NOT reproduce the same song across
``FullVocabView`` and ``PrunedVocabView`` -- see ``PrunedVocabView``'s
docstring and the design doc's phase-10 section for why (a sampler-width
RNG effect, not a numeric bug) and why that gap is not being closed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.models.minimax_music3.defaults import AUDIO_CODE_OFFSET, AUDIO_END_TOKEN_ID, SEMANTIC_VOCAB_SIZE
from core.models.minimax_music3.pruned_text_encoder_remap import EOA_LM_HEAD_ROW

__all__ = ["Music3VocabView", "FullVocabView", "PrunedVocabView", "resolve_vocab_view"]


class Music3VocabView:
    """Protocol the AR loop programs against. Not instantiated directly."""

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed_semantic_code(self, code: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def audio_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """`last_hidden` -> float32 logits, already restricted to the valid (semantic-code +
        end-of-audio) range -- see each implementation for how "restricted" is achieved."""
        raise NotImplementedError

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Idempotent re-application of the SAME restriction `audio_logits` already applied.

        The AR loop's CFG combination (`generate_ar`) computes `guided = uncond + scale *
        (cond - uncond)` from two already-restricted rows and then re-masks the COMBINED
        result before sampling -- defense against the CFG arithmetic itself producing a
        finite value at a position that was `-inf` in both operands (it cannot, for a linear
        combination of two `-inf` values plus one already-`-inf`-anchored candidate, but the
        original pre-phase-10 code applied this second mask unconditionally, so this method
        preserves that exactly for the full-vocab path -- see its own docstring)."""
        raise NotImplementedError

    def decode_sample(self, sampled: torch.Tensor):
        """A sampled index (shape `[1]`) -> `(is_end_of_audio: bool, semantic_code: Tensor)`.

        `semantic_code` is always returned (even when `is_end_of_audio` is True) so a caller
        that computes it unconditionally -- as `generate_ar` does, matching the pre-phase-10
        code's own "the subtraction is cheap; guard the BRANCH, not the arithmetic" shape --
        never needs to special-case the EOA sample itself."""
        raise NotImplementedError


class FullVocabView(Music3VocabView):
    """The shipped, default layout -- see module docstring.

    Every computation here is copied verbatim from the pre-phase-10
    `generate_ar`/`_embed_audio_frame`/`_embed_audio_frames`/
    `_generate_depth_codes`/`_replay_depth_hidden`/`recover_frame_hiddens`
    bodies, only moved behind these five methods -- the vocab MASK itself is
    now built ONCE per `(language_model, device)` pair and cached (`_mask_by_device`)
    rather than reconstructed at the top of every `generate_ar` call, which
    changes nothing numerically (the mask's VALUES are identical every time;
    `torch.ones` followed by the same two slice-assignments always produces
    the same tensor) but avoids paying for it once per RESUMED call, not just
    once per frame.
    """

    def __init__(self, language_model):
        self._lm = language_model
        self._mask_by_device: Dict[torch.device, torch.Tensor] = {}

    def _vocab_mask(self, device: torch.device) -> torch.Tensor:
        cached = self._mask_by_device.get(device)
        if cached is not None:
            return cached
        mask = torch.ones(self._lm.config.vocab_size, dtype=torch.bool, device=device)
        mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
        mask[AUDIO_END_TOKEN_ID] = False
        self._mask_by_device[device] = mask
        return mask

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self._lm.model.embed_tokens(text_ids)

    def embed_semantic_code(self, code: torch.Tensor) -> torch.Tensor:
        return self._lm.model.embed_tokens(code + AUDIO_CODE_OFFSET)

    def audio_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        logits = self._lm.lm_head(last_hidden).float()
        return logits.masked_fill(self._vocab_mask(logits.device), -float("inf"))

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(self._vocab_mask(logits.device), -float("inf"))

    def decode_sample(self, sampled: torch.Tensor):
        is_end_of_audio = bool(int(sampled.item()) == AUDIO_END_TOKEN_ID)
        semantic_code = sampled - AUDIO_CODE_OFFSET
        return is_end_of_audio, semantic_code


class PrunedVocabView(Music3VocabView):
    """The pruned-vocabulary layout -- see module docstring.

    ``audio_logits`` needs no mask: ``lm_head_pruned`` is
    ``[SEMANTIC_VOCAB_SIZE + 1, hidden]``, every row already a valid class.

    Sampled codes are NOT expected to match ``FullVocabView`` for the same
    seed, even though the restricted logits the two paths sample from are
    (to GEMM precision -- see below) the same numbers. The PRIMARY reason is
    ``_sample_top_k``'s ``torch.multinomial``: its RNG consumption depends on
    the category count, so a 200,000-wide call and a 16,385-wide call
    advance the SAME seeded generator differently and pick a different class
    most of the time even when fed bit-identical restricted logits (measured:
    152/200 GPU trials, and separately 200/200 CPU trials, mismatched --
    `minimax_music3_vocab_view_test.py`; design doc phase 10). Feeding ONE
    16,385-wide sampler the SAME restricted logits from both paths does
    agree -- this is the gate that IS meetable, and what the argmax/top-50
    check in the design doc's verification section actually established. A
    SECONDARY, smaller effect is that `lm_head_pruned(last_hidden)` and
    `lm_head(last_hidden)` restricted to the same 16,385 rows are not always
    bit-identical on GPU (bf16 GEMM output-shape-dependent rounding; CPU is
    bit-identical) -- see the design doc's phase-10 section for the measured
    numbers and the full falsification of alternative hypotheses (EOA row
    position, code offset, the GQA split, the depth-decoder split, the
    dropped mask).
    """

    def __init__(self, language_model):
        self._lm = language_model

    def embed_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self._lm.model.embed_tokens(text_ids)

    def embed_semantic_code(self, code: torch.Tensor) -> torch.Tensor:
        return self._lm.model.embed_tokens_audio(code)

    def audio_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        return self._lm.lm_head_pruned(last_hidden).float()

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits  # see class docstring -- every column here is already valid

    def decode_sample(self, sampled: torch.Tensor):
        is_end_of_audio = bool(int(sampled.item()) == EOA_LM_HEAD_ROW)
        semantic_code = sampled - 1  # row `c + 1` holds semantic code `c` -- see EOA_LM_HEAD_ROW
        return is_end_of_audio, semantic_code


def resolve_vocab_view(language_model) -> Optional[Music3VocabView]:
    """``language_model`` -> the matching ``Music3VocabView``, or ``None`` if ``language_model`` is
    itself ``None`` (a caller that only needs the flow-matching side's geometry, e.g. a test or a
    probe -- ``load_minimax_music3_from_path(..., load_language_model=False)``).

    Detects pruned by ATTRIBUTE PRESENCE (``lm_head_pruned``), the loader's own contract
    (``build_language_model_and_depth_decoder_from_pruned_flat_text_encoder`` attaches it; the
    non-pruned/`official/` builders never do) -- mirrors
    ``flat_remap.is_pruned_flat_text_encoder``'s "detect from what IS there" convention, applied
    to the LOADED module rather than a safetensors header.
    """
    if language_model is None:
        return None
    if hasattr(language_model, "lm_head_pruned"):
        return PrunedVocabView(language_model)
    return FullVocabView(language_model)
