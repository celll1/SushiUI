"""MiniMax Music 3 inference pipeline — a plain port of the upstream modular blocks.

Upstream (`huggingface/diffusers` PR #14456, commit
`dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`) ships this as five
``ModularPipelineBlocks`` driven by ``ModularPipeline``/``LoopSequentialPipelineBlocks``
and a ``guiders.ClassifierFreeGuidance`` -- all 0.40-only APIs that do not exist
in the diffusers 0.38.0 this repo pins. This module reimplements the same five
steps (text encode -> autoregressive semantic/residual-code generation ->
chunk bookkeeping -> flow-matching chunk denoise -> vocoder decode) as ordinary
methods of one plain class.

The algorithm is a checkpoint contract: the prompt assembly, the special
tokens, the AR sampling recipe (CFG scale 1.5, top-k 50), the chunk geometry
(200-frame windows / 100-frame hop / 172-latent-frame overlap blend), the flow
schedule (`sigmas = linspace(1, 1/steps, steps)`), the CFG formula
(`uncond + scale * (cond - uncond)`, zeros for the unconditional flow branch),
and the decode crop (86 leading / 258 trailing latent frames) are ported
EXACTLY from ``encoders.py``, ``before_denoise.py``, ``denoise.py`` and
``decoders.py`` in the PR. See each method's docstring for the upstream
source file it replaces, and ``docs/guides/MINIMAX_MUSIC3_DESIGN.md`` for the
architecture this implements.

Beyond the faithful port, this module adds four things the upstream blocks
have no hooks for at all (upstream drives everything through
``ModularPipeline.__call__`` with a single coarse progress bar over the flow
chunks only):

  * a ``progress_callback(step, total, stage)`` reported for BOTH stages -- AR
    progress in frames (``generate_ar``, ``stage="ar"``), then flow progress
    as ``chunk * num_inference_steps + step`` (``denoise_chunks``,
    ``stage="flow"``). The ``(step, total)`` pair alone is the shape every
    other SushiUI pipeline backend already reports, but AR and flow are two
    INDEPENDENT counters over two different totals -- reporting them through
    one 2-tuple callback would look like progress hitting 100%, resetting,
    and running again against a different denominator, with no way for a
    consumer to tell the stages apart or weight them. ``stage`` is the
    minimum addition that resolves that; the WebSocket layer threading it
    through is a phase-3 (pipeline-backend) concern, not this module's;
  * cancellation between AR frames and between flow steps, via
    ``core.inference.cancellation.raise_if_cancelled``;
  * frame-code capture and AR-resume (teacher-forced, chunked, batched replay
    to rebuild the KV cache, then continue sampling) -- see
    ``generate_ar``'s docstring for the exact contract, including the
    9000-frame / language-model-position budget guards a resume must pass.
    The routes that populate/consume the sidecar this supports land in a
    later commit (design doc phase plan, items 3 and 7); this pipeline only
    needs to expose the capability faithfully, and safely, now;
  * execution-device resolution that follows offload hooks
    (``execution_device`` / ``flow_execution_device``) rather than reading a
    parameter's resting device, which is wrong under the group-offload
    configurations the design doc explicitly targets (see
    ``execution_device``'s docstring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

from diffusers.hooks.group_offloading import _is_group_offload_enabled
from diffusers.utils.torch_utils import randn_tensor

from core.inference.cancellation import raise_if_cancelled
from core.models.minimax_music3.vocab_view import resolve_vocab_view
from core.models.minimax_music3.defaults import (
    AUDIO_CFG_TOKEN_ID,
    AUDIO_START,
    AR_CFG_SCALE,
    AR_CFG_TOP_K,
    AR_RESUME_REPLAY_CHUNK_FRAMES,
    AR_SAMPLING_TOP_K,
    CAPTION_END,
    CAPTION_START,
    CHUNK_FRAMES,
    CHUNK_HOP,
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
    FALLBACK_AUDIO_VOCAB_SIZE,
    FALLBACK_FRAME_RATE,
    FALLBACK_LATENT_HOP_LENGTH,
    FALLBACK_NUM_CHANNELS_LATENTS,
    FALLBACK_NUM_CODEBOOKS,
    FALLBACK_SAMPLING_RATE,
    IM_END,
    IM_START,
    LYRICS_END,
    LYRICS_START,
    MAX_AUDIO_FRAMES,
    MAX_PROMPT_TOKENS,
    OVERLAP_LATENT_LENGTH,
)

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


# ---------------------------------------------------------------------------
# Pure prompt-assembly helpers, ported verbatim from upstream `encoders.py`.
# ---------------------------------------------------------------------------
def _clean_caption(caption: str) -> str:
    """Markdown-strip the music description (upstream `encoders.py::_clean_caption`)."""

    def _rewrite_special_tag(match: "re.Match") -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics: str) -> str:
    """Normalize structure tags in `lyrics` (upstream `encoders.py::_normalize_lyrics`)."""
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def check_ar_resume_budget(
    prompt_tokens: int,
    total_frames_so_far: int,
    max_frames: int,
    max_position_embeddings: Optional[int],
    *,
    duration_param_name: str = "audio_duration",
    prompt_is_adjustable: bool = True,
) -> None:
    """Pure pre-flight form of `generate_ar`'s own two hard-limit guards (`MAX_AUDIO_FRAMES` total frames, the
    language model's `max_position_embeddings` total context length) -- see `generate_ar`'s docstring for the exact
    contract these enforce. Free function (no model/tensor access beyond plain ints) SO A CALLER CAN RUN IT BEFORE
    STAGING THE ~18 GB LANGUAGE MODEL + DEPTH DECODER ONTO THE ACCELERATOR (design doc phase plan item 7, "extend"):
    without this, an over-budget extend request would only be discovered after paying for that move, inside
    `generate_ar` itself, which duplicates this exact check against the SAME two numbers once the models (and a real
    `text_ids` tensor) are already resident. `generate_ar` calls this too, so the two call sites can never drift
    apart. Raises `ValueError` (never returns a bool) so a caller gets the same descriptive message either way,
    whether it is this pre-flight call or `generate_ar`'s own.

    `duration_param_name`/`prompt_is_adjustable` exist because this ONE function serves TWO callers with different
    vocabularies (audit finding F2): `generate_ar` itself, whose caller-facing duration knob is `audio_duration` and
    whose prompt IS whatever the caller just supplied, vs the extend path (`MiniMaxMusic3Mixin.
    _generate_audoutpaint_minimax_music3`), whose knob is `extend_duration_sec` and whose prompt is FORCED to the
    sidecar's own (never caller-adjustable -- see that method's docstring). Naming the wrong parameter, or telling a
    caller to shorten a prompt it does not control, is worse than a generic message: it points at a fix that does
    not exist for that caller. Defaults reproduce `generate_ar`'s original wording exactly, so its own (un-keyword)
    call site is unaffected by this addition.
    """
    if total_frames_so_far + max_frames > MAX_AUDIO_FRAMES:
        raise ValueError(
            f"This call would bring the song to {total_frames_so_far + max_frames} frames "
            f"({total_frames_so_far} previously generated + up to {max_frames} new), exceeding the checkpoint's "
            f"{MAX_AUDIO_FRAMES}-frame (six-minute) range. Shorten `{duration_param_name}`."
        )
    if max_position_embeddings is not None:
        # prompt tokens + the one ever-present warm-up token + every previous frame's feedback token + up to
        # `max_frames` new feedback tokens -- see `generate_ar`'s docstring.
        projected_positions = prompt_tokens + 1 + total_frames_so_far + max_frames
        if projected_positions > max_position_embeddings:
            if prompt_is_adjustable:
                advice = f"Shorten the prompt or the requested duration (`{duration_param_name}`)."
            else:
                advice = (
                    f"Shorten the requested duration (`{duration_param_name}`); the prompt is fixed to the "
                    f"original song's stored prompt/lyrics here and cannot be shortened for this call."
                )
            raise ValueError(
                f"This call would put {projected_positions} positions in the language model's context "
                f"(prompt {prompt_tokens} + warm-up 1 + {total_frames_so_far} previous frames + up to "
                f"{max_frames} new frames), exceeding its {max_position_embeddings}-position budget. {advice}"
            )


def _sample_top_k(logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Top-k sampling (upstream `encoders.py::_sample_top_k`)."""
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_k = min(AR_SAMPLING_TOP_K, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(torch.nn.functional.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


@dataclass
class MiniMaxMusic3ARResult:
    """Output of :meth:`MiniMaxMusic3Pipeline.generate_ar`.

    Attributes:
        frame_hiddens: `[1, frames, num_codebooks * hidden_size]` -- the newly emitted frames' condition input for
            the flow-matching stage (`generate_ar`'s only upstream-equivalent output).
        frame_codes: `[frames, num_codebooks]` `torch.long` -- the newly emitted frames' sampled codes (the
            conditional row only). This is the state-contract artifact the design doc's per-generation sidecar
            stores: 8 `int16` per frame, ~144 KB for a full six-minute song.
        prefix_codes: `[1, num_codebooks]` `torch.long` -- the codes of the "warm-up" decode step that only advances
            the language model's state past `<|audio_start|>` and is never itself an emitted audio frame (upstream
            `encoders.py`'s "the first decode step is not an emitted frame" comment). This frame is real generated
            state that MUST be replayed to reconstruct the KV cache exactly, so it is captured and returned even
            though it never contributes to `frame_hiddens` or the decoded audio. On a resumed call this is the
            passed-through `resume_prefix_codes`, unchanged -- callers extending a song across multiple resumes keep
            the FIRST call's `prefix_codes` and concatenate every call's `frame_codes` in order.
    """

    frame_hiddens: torch.Tensor
    frame_codes: torch.Tensor
    prefix_codes: torch.Tensor


@dataclass
class MiniMaxMusic3GenerationResult:
    """Output of :meth:`MiniMaxMusic3Pipeline.generate`."""

    audio: "torch.Tensor | np.ndarray"
    sample_rate: int
    frame_codes: torch.Tensor
    prefix_codes: torch.Tensor
    num_frames: int


class MiniMaxMusic3Pipeline:
    """Plain-class reimplementation of upstream's `MiniMaxMusic3Blocks` modular pipeline.

    Components are wired in by the caller (a later commit's loader); this class does not load weights itself. Every
    method mirrors one upstream block or block group 1:1 -- see each docstring.
    """

    def __init__(
        self,
        tokenizer,
        language_model,
        rvq_depth_decoder,
        condition_encoder,
        transformer,
        scheduler,
        vocoder,
        execution_device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.rvq_depth_decoder = rvq_depth_decoder
        self.condition_encoder = condition_encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.vocoder = vocoder
        self._execution_device = execution_device
        # Resolved ONCE here (design doc phase 10, "The pruned vocabulary"): which
        # checkpoint layout `language_model` is, decided from the loaded module's own
        # shape -- see `resolve_vocab_view`'s docstring. Every AR-loop text/semantic-code
        # embedding and every audio-logit computation below goes through this instead of
        # `language_model.model.embed_tokens`/`language_model.lm_head` directly, so the
        # checkpoint-contract offset/mask difference between the two layouts lives in ONE
        # place (`core.models.minimax_music3.vocab_view`) rather than at every call site.
        self._vocab = resolve_vocab_view(language_model)

    # ------------------------------------------------------------------
    # Component-derived properties. The six read-only properties below
    # (`sampling_rate` .. `num_channels_latents`) match upstream
    # `MiniMaxMusic3ModularPipeline`'s computed properties exactly
    # (`modular_pipeline.py` in the PR) -- same fallback values, same source
    # config field for each. `execution_device` / `flow_execution_device` do
    # NOT have an upstream equivalent with this name; they reimplement
    # `DiffusionPipeline._execution_device`'s offload-aware device
    # resolution (see their own docstrings), because upstream's modular
    # pipeline instead resolves devices through a `ComponentsManager` this
    # plain-class port does not have.
    # ------------------------------------------------------------------
    @staticmethod
    def _group_onload_or_hook_device(components) -> Optional[torch.device]:
        """Probe `components` (in order) for a group-offload onload device, then an `accelerate` hook device.

        Mirrors `diffusers.pipelines.pipeline_utils.DiffusionPipeline._execution_device`'s two-pass search over
        `self.components.items()`. Returns `None` (rather than a resting parameter device) when neither applies, so
        the caller can supply its own final fallback.
        """
        from diffusers.hooks.group_offloading import _get_group_onload_device

        for model in components:
            if model is None or not isinstance(model, torch.nn.Module):
                continue
            try:
                return _get_group_onload_device(model)
            except ValueError:
                pass
        for model in components:
            if model is None or not isinstance(model, torch.nn.Module) or not hasattr(model, "_hf_hook"):
                continue
            for module in model.modules():
                hook = getattr(module, "_hf_hook", None)
                if hook is not None and getattr(hook, "execution_device", None) is not None:
                    return torch.device(hook.execution_device)
        return None

    @property
    def execution_device(self) -> torch.device:
        """The device to create AR-stage intermediates (`text_ids`, embeddings) on.

        Under the leaf-level group offloading the design doc's ~8GB configuration targets (or plain `accelerate`
        sequential offload), the language model's WEIGHTS rest on CPU/meta between calls and are onloaded to the
        accelerator only for the forward -- so `next(self.language_model.parameters()).device` would return the
        resting device, not the device the next forward will actually run on. Creating `text_ids` there either
        raises a device-mismatch error inside the LM's forward or, if the resting device is `meta`, runs silently
        with no error until something calls `.item()`. This resolves the device the same way upstream's
        `DiffusionPipeline._execution_device` does: group-offload onload device, then `accelerate` hook execution
        device, and only THEN a parameter's resting device (which IS correct in the plain single-GPU-resident case).
        """
        if self._execution_device is not None:
            return self._execution_device
        components = (self.language_model, self.rvq_depth_decoder, self.condition_encoder, self.transformer, self.vocoder)
        device = self._group_onload_or_hook_device(components)
        if device is not None:
            return device
        return next(self.language_model.parameters()).device

    @property
    def flow_execution_device(self) -> torch.device:
        """Like `execution_device`, but probes the flow-stage components (transformer first).

        SushiUI addition. After a staged offload the language model and the flow-matching transformer can be
        resident on different devices at different times (the AR stage needs the LM+depth-decoder resident
        together; the flow stage needs the transformer+condition_encoder+vocoder instead, and the design doc's
        offload policy only constrains the FORMER pair to co-residency). Using the LM-derived `execution_device` to
        place `denoise_chunks`' latents would be wrong once the LM has been offloaded back off-device after the AR
        stage finishes. Falls back to the transformer's own parameter device, not the language model's.
        """
        if self._execution_device is not None:
            return self._execution_device
        components = (self.transformer, self.condition_encoder, self.vocoder, self.language_model, self.rvq_depth_decoder)
        device = self._group_onload_or_hook_device(components)
        if device is not None:
            return device
        return next(self.transformer.parameters()).device

    @property
    def sampling_rate(self) -> int:
        if self.vocoder is not None:
            return int(self.vocoder.config.sampling_rate)
        return FALLBACK_SAMPLING_RATE

    @property
    def frame_rate(self) -> float:
        if self.condition_encoder is not None:
            config = self.condition_encoder.config
            return config.input_sampling_rate / config.input_hop_length
        return FALLBACK_FRAME_RATE

    @property
    def latent_hop_length(self) -> int:
        if self.condition_encoder is not None:
            return int(self.condition_encoder.config.output_hop_length)
        return FALLBACK_LATENT_HOP_LENGTH

    @property
    def num_codebooks(self) -> int:
        if self.rvq_depth_decoder is not None:
            return int(self.rvq_depth_decoder.config.num_codebooks)
        return FALLBACK_NUM_CODEBOOKS

    @property
    def audio_vocab_size(self) -> int:
        if self.rvq_depth_decoder is not None:
            return int(self.rvq_depth_decoder.config.audio_vocab_size)
        return FALLBACK_AUDIO_VOCAB_SIZE

    @property
    def num_channels_latents(self) -> int:
        if self.transformer is not None:
            return self.transformer.config.in_channels
        return FALLBACK_NUM_CHANNELS_LATENTS

    # ------------------------------------------------------------------
    # Stage 1: text encoding. Upstream `encoders.py::MiniMaxMusic3TextEncoderStep`.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, prompt: str, lyrics: str) -> torch.Tensor:
        """Assemble and tokenize the checkpoint's special-token prompt.

        Returns `text_ids` of shape `[2, sequence_length]`: row 0 is the conditional prompt, row 1 is its
        classifier-free counterpart (every token except the first and the two trailing structure tokens replaced by
        the audio-CFG token).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"`prompt` (the music description) must be a non-empty string, got {prompt!r}")
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {lyrics!r}")

        text = (
            f"{IM_START}{CAPTION_START}{_clean_caption(prompt)}{CAPTION_END}"
            f"{LYRICS_START}{_normalize_lyrics(lyrics)}{LYRICS_END}{IM_END}{AUDIO_START}"
        )
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > MAX_PROMPT_TOKENS:
            raise ValueError(f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {MAX_PROMPT_TOKENS}")
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID
        return torch.cat((input_ids, unconditional_ids), dim=0).to(self.execution_device)

    # ------------------------------------------------------------------
    # Frame-embedding helpers, ported from upstream `encoders.py::_embed_audio_frame`. `_embed_audio_frames` is a
    # SushiUI addition: the batched (multi-frame) generalization used ONLY by the AR-resume replay path; the
    # sequential generation loop keeps calling the single-frame form so its numerics are byte-for-byte upstream's.
    # ------------------------------------------------------------------
    def _embed_audio_frame(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """frame_codes: `[2, num_codebooks]` -> `[2, 1, hidden_size]`. Verbatim upstream `_embed_audio_frame`,
        the semantic-code embedding routed through `self._vocab` (design doc phase 10) instead of
        `language_model.model.embed_tokens(... + AUDIO_CODE_OFFSET)` directly -- see `vocab_view`'s module
        docstring; numerically identical on the full-vocab path (`FullVocabView.embed_semantic_code` performs
        the exact same lookup)."""
        embeds = self._vocab.embed_semantic_code(frame_codes[:, :1])
        offsets = (torch.arange(self.num_codebooks - 1, device=frame_codes.device) * self.audio_vocab_size).unsqueeze(0)
        extra = self.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * self.num_codebooks**-0.5

    def _embed_audio_frames(self, frame_codes: torch.Tensor) -> torch.Tensor:
        """SushiUI addition: batched form of `_embed_audio_frame` for teacher-forced replay.

        frame_codes: `[2, F, num_codebooks]` -> `[2, F, hidden_size]`. Mathematically identical to calling
        `_embed_audio_frame` once per frame and concatenating along the frame axis (same embedding lookups, same
        sum, same scale); the only difference is one batched matmul/embedding pass instead of F sequential ones.
        """
        embeds = self._vocab.embed_semantic_code(frame_codes[..., 0])
        offsets = torch.arange(self.num_codebooks - 1, device=frame_codes.device) * self.audio_vocab_size
        extra = self.rvq_depth_decoder.audio_embeddings(frame_codes[..., 1:] + offsets).sum(dim=-2)
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * self.num_codebooks**-0.5

    def _generate_depth_codes(self, last_hidden: torch.Tensor, semantic_code: torch.Tensor, generator):
        """Sample the residual codes c1..c7 for one frame. Verbatim upstream `encoders.py::_generate_depth_codes`,
        plus explicit dtype casts at the two points where an LM-dtype tensor (`last_hidden`, the embedding of
        `semantic_code`) crosses into `rvq_depth_decoder.projection` -- see `_embed_audio_frame`'s `extra.to(embeds
        .dtype)` for the same crossing already guarded elsewhere in this file; this mirrors it rather than relying
        on the loader happening to hand both models the same `torch_dtype` today."""
        rvq_dtype = self.rvq_depth_decoder.dtype
        sequence = [self.rvq_depth_decoder.projection(last_hidden.to(rvq_dtype)).unsqueeze(1)]
        code_embed = self._vocab.embed_semantic_code(semantic_code)
        sequence.append(self.rvq_depth_decoder.projection(code_embed.to(rvq_dtype)).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            hidden = self.rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = self.rvq_depth_decoder.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * AR_CFG_SCALE
            code = _sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embed = self.rvq_depth_decoder.audio_embeddings(code + (index - 1) * self.audio_vocab_size)
                sequence.append(self.rvq_depth_decoder.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    # ------------------------------------------------------------------
    # Stage 2: autoregressive generation. Upstream
    # `encoders.py::MiniMaxMusic3SemanticGenerationStep`, plus SushiUI's
    # progress/cancellation/frame-code-capture/resume additions (module
    # docstring).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_ar(
        self,
        text_ids: torch.Tensor,
        audio_duration: float,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume_frame_codes: Optional[torch.Tensor] = None,
        resume_prefix_codes: Optional[torch.Tensor] = None,
    ) -> MiniMaxMusic3ARResult:
        """Frame-by-frame semantic + residual code sampling with classifier-free guidance.

        Args:
            text_ids: `[2, seq_len]` from :meth:`encode_text`.
            audio_duration: upper bound on the generated span, in seconds, for THIS call (not the whole song when
                resuming: the language model still stops on its own end-of-audio token regardless).
            progress_callback: called as `(frames_done, max_frames, "ar")` after every emitted frame.
            resume_frame_codes: `[F_prev, num_codebooks]` integer tensor, every frame previously emitted for this
                song (across however many prior calls), in order. `None` for a fresh generation. Coerced to
                `torch.long` internally; a caller storing a compact sidecar (e.g. `int16`, as the design doc's
                per-generation state contract does) does not need to cast before passing it in.
            resume_prefix_codes: `[1, num_codebooks]` integer tensor, the ORIGINAL `prefix_codes` from the very first
                call for this song. `None` for a fresh generation. Passed through unchanged into the returned
                result so a caller extending across multiple resumes never has to special-case which call was first.

        Both `resume_*` arguments must be given together or not at all: a resume with codes but no prefix (or vice
        versa) cannot reconstruct the exact original state and is rejected. Both are also shape-checked (last dim
        must be `self.num_codebooks`) before anything is device-moved or fed to an `nn.Embedding`, so a malformed
        sidecar fails fast rather than after the ~22GB model is already resident.

        A resume is rejected outright, rather than silently truncated or clamped, if it would push the song past
        either of the checkpoint's two hard limits: `MAX_AUDIO_FRAMES` (9000, six minutes) total frames, or the
        language model's `max_position_embeddings` (10240 on the released checkpoint) total context length (prompt
        tokens + the ever-present 1 warm-up token + every frame token, previous and new). The design doc notes these
        two budgets cannot both be maximized independently; extending a near-cap song is exactly where that collides.

        Returns the newly generated frames only (`MiniMaxMusic3ARResult`); a resuming caller who wants the whole
        song's `frame_hiddens` must re-run the flow stage over `resume_frame_codes` too, or (equivalently, and how
        the design doc's "extend" route is meant to work) only flow-match and decode the new tail and stitch it onto
        the previously decoded audio.
        """
        if (resume_frame_codes is None) != (resume_prefix_codes is None):
            raise ValueError("`resume_frame_codes` and `resume_prefix_codes` must both be given, or neither.")

        if audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {audio_duration}")
        max_frames = min(int(audio_duration * self.frame_rate), MAX_AUDIO_FRAMES)
        if max_frames == 0:
            raise ValueError(f"`audio_duration` {audio_duration} is shorter than one audio frame (1 / {self.frame_rate} s)")

        resuming = resume_frame_codes is not None
        if resuming:
            if resume_prefix_codes.shape[-1] != self.num_codebooks:
                raise ValueError(
                    f"`resume_prefix_codes` last dim must be num_codebooks ({self.num_codebooks}), got shape "
                    f"{list(resume_prefix_codes.shape)}"
                )
            if resume_frame_codes.shape[-1] != self.num_codebooks:
                raise ValueError(
                    f"`resume_frame_codes` last dim must be num_codebooks ({self.num_codebooks}), got shape "
                    f"{list(resume_frame_codes.shape)}"
                )
            total_frames_so_far = int(resume_frame_codes.shape[0])
        else:
            total_frames_so_far = 0

        max_position_embeddings = getattr(self.language_model.config, "max_position_embeddings", None)
        check_ar_resume_budget(
            prompt_tokens=int(text_ids.shape[1]),
            total_frames_so_far=total_frames_so_far,
            max_frames=max_frames,
            max_position_embeddings=max_position_embeddings,
        )

        language_model = self.language_model
        # Trigger CPU-offload hooks by hand (same workaround as minimax_h3/acestep): the autoregressive loop calls
        # submodules (`embed_tokens`, `lm_head`, depth-decoder heads) directly while the hook wraps only the
        # top-level `forward`. The language model goes first -- placing it can evict other models but never the
        # reverse, and both models are used on every frame, so a placement must not evict the other.
        hooked = [m for m in (language_model, self.rvq_depth_decoder) if getattr(m, "_hf_hook", None) is not None]
        for m in hooked:
            m._hf_hook.pre_forward(m)
        resident = [m for m in hooked if not _is_group_offload_enabled(m)]
        if len(resident) == 2 and resident[0].device != resident[1].device:
            raise RuntimeError(
                "The language model and the RVQ depth decoder must fit on the device together for autoregressive "
                "generation; there is not enough free device memory under CPU offloading."
            )

        device = text_ids.device
        text_embeds = self._vocab.embed_text(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        if resuming:
            # Coerce dtype/device now that the shapes are already validated above. `nn.Embedding` requires an
            # integer index dtype (`long`/`int`); a compact sidecar dtype such as `int16` raises inside
            # `_embed_audio_frames` with no context otherwise ("Expected tensor for argument #1 'indices' to have
            # ... Long, Int; but got torch.ShortTensor").
            resume_prefix_codes = resume_prefix_codes.to(device=device, dtype=torch.long)
            resume_frame_codes = resume_frame_codes.to(device=device, dtype=torch.long)

            # SushiUI addition: teacher-forced BATCHED replay of every previously emitted frame (the warm-up frame
            # first, then the emitted frames in order) to rebuild the KV cache exactly, chunked so a full 9000-frame
            # history does not become one `[2, 9001, hidden_size]` forward through the 8B language model (see
            # `AR_RESUME_REPLAY_CHUNK_FRAMES`'s docstring in `defaults.py`). The equivalence argument holds for any
            # chunk size, including 1 -- the sequential generation loop below already IS the chunk-size-1 case.
            replay_codes = torch.cat(
                (resume_prefix_codes.reshape(1, -1), resume_frame_codes.reshape(-1, resume_frame_codes.shape[-1])),
                dim=0,
            )
            replay_pair = replay_codes.unsqueeze(0).expand(2, -1, -1).contiguous()
            total_replay = replay_pair.shape[1]
            for start in range(0, total_replay, AR_RESUME_REPLAY_CHUNK_FRAMES):
                raise_if_cancelled()
                end = min(start + AR_RESUME_REPLAY_CHUNK_FRAMES, total_replay)
                feedback = self._embed_audio_frames(replay_pair[:, start:end])
                output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                last_hidden = output.last_hidden_state[:, -1]

        frame_hiddens: List[torch.Tensor] = []
        frame_codes_out: List[torch.Tensor] = []
        captured_prefix_codes: Optional[torch.Tensor] = resume_prefix_codes

        # The first decode step only advances the state past `<|audio_start|>` (or, when resuming, is skipped
        # entirely -- the replay above already consumed it) and is not an emitted frame.
        for frame_index in range(max_frames + 1):
            raise_if_cancelled()
            logits = self._vocab.audio_logits(last_hidden)
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
            threshold = torch.topk(conditional, AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = self._vocab.mask_logits(guided)
            sampled = _sample_top_k(guided, generator)
            is_end_of_audio, semantic_code = self._vocab.decode_sample(sampled)
            if is_end_of_audio:
                break

            frame_codes, depth_hidden = self._generate_depth_codes(last_hidden, semantic_code.repeat(2), generator)

            is_warmup_step = (not resuming) and frame_index == 0
            emit_this_frame = not is_warmup_step
            if emit_this_frame:
                # `depth_hidden` carries the RVQ depth decoder's dtype, `last_hidden` the language model's --
                # `torch.cat` requires an exact dtype match between operands, same crossing as
                # `_generate_depth_codes`'s two casts above.
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden.to(last_hidden.dtype)), dim=-1))
                frame_codes_out.append(frame_codes[0].clone())
                if progress_callback:
                    try:
                        progress_callback(len(frame_hiddens), max_frames, "ar")
                    except Exception as exc:
                        print(f"[MiniMaxMusic3] progress_callback raised during AR generation: {exc!r}")
                if len(frame_hiddens) >= max_frames:
                    # Matches upstream exactly: no feedback forward for this last frame. Its codes are already in
                    # `frame_codes_out`, so a FUTURE resume's teacher-forced replay (above) reconstructs the KV
                    # state through this frame from the stored codes on its own -- this call never needs to hold
                    # that state itself. Running the extra forward here (an earlier version of this port did) cost
                    # one more LM call and one more KV-cache token for a result nothing in this call or any future
                    # resume ever reads; deleted rather than kept as dead compute.
                    break
            elif captured_prefix_codes is None:
                captured_prefix_codes = frame_codes[0].clone().unsqueeze(0)

            feedback = self._embed_audio_frame(frame_codes)
            output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")

        result_frame_hiddens = torch.stack(frame_hiddens, dim=1)
        result_frame_codes = torch.stack(frame_codes_out, dim=0).to(torch.long)
        if captured_prefix_codes is None:
            # Defensive: should be unreachable (frame_hiddens non-empty implies frame_index reached >= 1, so the
            # warm-up branch always ran when not resuming), but never return `None` from a typed field.
            raise RuntimeError("MiniMax Music 3 AR generation captured no prefix (warm-up) codes")
        return MiniMaxMusic3ARResult(
            frame_hiddens=result_frame_hiddens,
            frame_codes=result_frame_codes,
            prefix_codes=captured_prefix_codes.to(torch.long),
        )

    # ------------------------------------------------------------------
    # SushiUI addition (design doc phase plan item 8, "re-render a range"
    # repaint mode): deterministic, teacher-forced recovery of `frame_hiddens`
    # for a range of ALREADY-KNOWN frames, from stored codes alone. No
    # sampling, no CFG -- both the semantic and residual codes for every
    # recovered frame are already fixed, and CFG/top-k only ever influenced
    # WHICH code was chosen during the original (live) generation, never the
    # hidden states this recomputes. Same equivalence argument as
    # `generate_ar`'s `resume_*` replay path (see its docstring), generalized
    # to capture every recovered frame's hidden state, not only the final
    # one.
    # ------------------------------------------------------------------
    def _replay_depth_hidden(self, last_hidden: torch.Tensor, frame_codes_row: torch.Tensor) -> torch.Tensor:
        """Teacher-forced (batch=1, no CFG, no sampling) recomputation of ONE frame's `depth_hidden` from its
        ALREADY-KNOWN codes -- the deterministic counterpart of `_generate_depth_codes` (verbatim upstream, live
        sampling), used by `recover_frame_hiddens`. Deterministic given fixed weights: `hidden_parts` at each
        residual-codebook step is a function of the PRIOR codes only (never of what gets sampled next), so replaying
        the known codes into the same running `sequence` reproduces hidden states identical up to floating-point
        reduction order -- no CFG batching needed because nothing is sampled. (Measured against a real
        `generate_ar` run: max abs diff ~4.768e-07 in fp32 on CPU -- the batch-1 replay here and the batch-2
        CFG-doubled original sum in a different order, so they agree to about one fp32 ulp, not bit-for-bit; the
        gap will be larger in the bf16 production dtype. Nothing functional depends on exactness here -- the
        preserved span is copied from the file, never re-derived from these hidden states.)

        Args:
            last_hidden: `[1, hidden_size]` (batch=1 -- unlike `_generate_depth_codes`'s CFG-doubled
                `[2, hidden_size]` input; no unconditional branch is needed here).
            frame_codes_row: `[num_codebooks]`, this frame's semantic + residual codes, already known.

        Returns:
            `[1, hidden_size * (num_codebooks - 1)]`, matching `_generate_depth_codes`'s second return value.
        """
        rvq_dtype = self.rvq_depth_decoder.dtype
        semantic_code = frame_codes_row[0:1]
        sequence = [self.rvq_depth_decoder.projection(last_hidden.to(rvq_dtype)).unsqueeze(1)]
        code_embed = self._vocab.embed_semantic_code(semantic_code)
        sequence.append(self.rvq_depth_decoder.projection(code_embed.to(rvq_dtype)).unsqueeze(1))
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            hidden = self.rvq_depth_decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden)
            if index < self.num_codebooks - 1:
                code = frame_codes_row[index:index + 1]
                embed = self.rvq_depth_decoder.audio_embeddings(code + (index - 1) * self.audio_vocab_size)
                sequence.append(self.rvq_depth_decoder.projection(embed).unsqueeze(1))
        return torch.cat(hidden_parts, dim=-1)

    @torch.no_grad()
    def recover_frame_hiddens(
        self,
        text_ids: torch.Tensor,
        frame_codes: torch.Tensor,
        prefix_codes: torch.Tensor,
        frame_start: int,
        frame_end: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> torch.Tensor:
        """Teacher-forced, deterministic recovery of `frame_hiddens` for emitted frames `[frame_start, frame_end)`
        from ALREADY-KNOWN codes.

        Used by MiniMax Music 3 repaint's "re-render a range" mode (design doc "Modality surfaces"): the flow
        stage needs `frame_hiddens` as its condition input, but the design doc's per-generation state contract
        deliberately does NOT store `frame_hiddens` (too large -- see design doc "Per-generation state contract"),
        only `frame_codes`. Re-rendering a range therefore first reconstructs the `frame_hiddens` the AR stage
        would have produced for that range, from the STORED codes, exactly like `generate_ar`'s own AR-resume path
        reconstructs its KV cache from stored codes -- this is the same mechanism, generalized to capture EVERY
        recovered frame's hidden state instead of only the final one.

        Mechanism: a chunked teacher-forced replay first primes the KV cache through the prompt and `frame_codes[
        :frame_start]` (context only, nothing captured -- identical to `generate_ar`'s own resume replay), then a
        second windowed forward over `frame_codes[frame_start:frame_end - 1]` captures EVERY position's own output
        hidden state (not only the final one) -- position `j` of that forward is exactly the `last_hidden`
        `generate_ar` would have used to predict emitted frame `frame_start + j` during the original, live
        generation. Each recovered `last_hidden` is then run through `_replay_depth_hidden` (teacher-forced, no
        sampling) using that SAME frame's known residual codes, to reconstruct `depth_hidden`.

        Batch shape: `text_ids` is `encode_text`'s CFG-doubled `[2, seq_len]` (row 0 conditional, row 1 the
        audio-CFG-masked row) -- this method does not need row 1 at all (no sampling, no CFG), but MUST still run
        the LM forward on the full `[2, ...]` batch throughout, because `language_model.model`'s KV cache is
        allocated at whatever batch size the FIRST call establishes and every later call in the same replay must
        match it exactly (mirrors `generate_ar`'s own `text_embeds`/`replay_pair` batch-2 shape, for the identical
        reason). Only the conditional row (index 0) is read out of each recovered `last_hidden` before it is
        combined with `depth_hidden`.

        Args:
            text_ids: `[2, seq_len]` from `encode_text`, using the SAME prompt/lyrics the song was originally
                generated with (caller's responsibility -- mirrors `_generate_audoutpaint_minimax_music3`'s
                "Prompt/lyrics" contract; a different prompt would prime the WRONG KV-cache context).
            frame_codes: every frame emitted for this song, 0-indexed, `[F_total, num_codebooks]`.
            prefix_codes: the song's original warm-up code, `[1, num_codebooks]`.
            frame_start, frame_end: half-open range of emitted frame indices to recover hidden states for
                (`0 <= frame_start < frame_end <= F_total`).
            progress_callback: called as `(recovered_count, frame_end - frame_start, "ar")` after each frame's
                `depth_hidden` is recomputed -- reuses the same `(step, total, stage)` shape `generate_ar` reports,
                so a caller can feed it through the same `compute_progress_budget`/`combined_progress` machinery.

        Returns:
            `[1, frame_end - frame_start, num_codebooks * hidden_size]`, same shape/dtype convention as
            `MiniMaxMusic3ARResult.frame_hiddens`.
        """
        total_frames = int(frame_codes.shape[0])
        if not (0 <= frame_start < frame_end <= total_frames):
            raise ValueError(
                f"recover_frame_hiddens: invalid range [{frame_start}, {frame_end}) for {total_frames} total frames."
            )

        device = text_ids.device
        frame_codes = frame_codes.to(device=device, dtype=torch.long)
        prefix_codes = prefix_codes.to(device=device, dtype=torch.long)

        language_model = self.language_model
        text_embeds = self._vocab.embed_text(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]  # predicts frame 0 (the warm-up code)

        # ---- Context replay: prefix + frames[0:frame_start] -- captures nothing, only rebuilds the KV cache.
        # Same chunked mechanism as generate_ar's resume path, batch=2 (matching text_ids/the KV cache's batch
        # size -- see this method's docstring, "Batch shape"). ----
        context_codes = torch.cat((prefix_codes.reshape(1, -1), frame_codes[:frame_start]), dim=0)
        context_pair = context_codes.unsqueeze(0).expand(2, -1, -1).contiguous()
        total_context = context_pair.shape[1]
        for start in range(0, total_context, AR_RESUME_REPLAY_CHUNK_FRAMES):
            raise_if_cancelled()
            end = min(start + AR_RESUME_REPLAY_CHUNK_FRAMES, total_context)
            feedback = self._embed_audio_frames(context_pair[:, start:end])
            output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
        # `last_hidden` now predicts frame `frame_start`.

        # ---- Windowed forward: capture EVERY position's hidden state for frames [frame_start, frame_end). ----
        hiddens_by_frame: List[torch.Tensor] = [last_hidden]
        remaining = (frame_end - 1) - frame_start  # codes still needed to advance through frame_end - 1
        if remaining > 0:
            feed_codes = frame_codes[frame_start:frame_end - 1]
            feed_pair = feed_codes.unsqueeze(0).expand(2, -1, -1).contiguous()
            total_feed = feed_pair.shape[1]
            for start in range(0, total_feed, AR_RESUME_REPLAY_CHUNK_FRAMES):
                raise_if_cancelled()
                end = min(start + AR_RESUME_REPLAY_CHUNK_FRAMES, total_feed)
                feedback = self._embed_audio_frames(feed_pair[:, start:end])
                output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                for t in range(output.last_hidden_state.shape[1]):
                    hiddens_by_frame.append(output.last_hidden_state[:, t])

        # ---- Per-frame depth-decoder teacher forcing (deterministic -- codes already known). ----
        # Only the conditional row (index 0) is read out here -- see this method's docstring, "Batch shape".
        frame_hiddens_out = []
        total_recovered = len(hiddens_by_frame)
        for offset, lm_hidden in enumerate(hiddens_by_frame):
            raise_if_cancelled()
            frame_idx = frame_start + offset
            cond_hidden = lm_hidden[:1]
            depth_hidden = self._replay_depth_hidden(cond_hidden, frame_codes[frame_idx])
            frame_hiddens_out.append(torch.cat((cond_hidden, depth_hidden.to(cond_hidden.dtype)), dim=-1))
            if progress_callback:
                try:
                    progress_callback(offset + 1, total_recovered, "ar")
                except Exception as exc:
                    print(f"[MiniMaxMusic3] progress_callback raised during frame-hidden recovery: {exc!r}")

        return torch.stack(frame_hiddens_out, dim=1)

    # ------------------------------------------------------------------
    # Stage 3: chunk bookkeeping. Upstream `before_denoise.py::MiniMaxMusic3PrepareChunksStep`.
    # ------------------------------------------------------------------
    def prepare_chunks(self, frame_hiddens: torch.Tensor) -> List[int]:
        num_frames = frame_hiddens.shape[1]
        if num_frames <= CHUNK_FRAMES:
            return [0]
        return list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))

    # ------------------------------------------------------------------
    # Stage 4: flow-matching chunk denoise. Upstream `denoise.py`'s five sub-blocks, inlined into one loop (the
    # guider is replaced by the plain CFG formula it always computed:
    # `uncond + guidance_scale * (cond - uncond)`, see the module docstring).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def denoise_chunks(
        self,
        frame_hiddens: torch.Tensor,
        num_inference_steps: int,
        flow_guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[torch.Tensor]:
        """Flow-match `frame_hiddens` into latent chunks.

        `num_inference_steps` and `flow_guidance_scale` are required, not defaulted: both are rows in the design
        doc's generation-parameter table (defaults 30 and 1.7 respectively), and the SushiUI-wide rule is that a
        user-facing default is defined exactly once, in `backend/api/param_defaults.py` (a later commit) -- never
        duplicated as a literal in a pipeline signature. `progress_callback` is called as
        `(chunk * num_inference_steps + step, num_chunks * num_inference_steps, "flow")`.
        """
        device = self.flow_execution_device
        chunk_starts = self.prepare_chunks(frame_hiddens)
        num_chunks = len(chunk_starts)
        total_steps = num_chunks * num_inference_steps

        latent_chunks: List[torch.Tensor] = []
        previous_latent: Optional[torch.Tensor] = None
        previous_condition: Optional[torch.Tensor] = None

        for k, chunk_start in enumerate(chunk_starts):
            raise_if_cancelled()
            chunk_end = min(chunk_start + CHUNK_FRAMES, frame_hiddens.shape[1])
            # `frame_hiddens` carries the language model's dtype; the loader pins
            # `condition_encoder` to float32 regardless of `torch_dtype`. `nn.Conv1d`
            # requires its input to match its weight/bias dtype exactly, so the cast
            # has to happen on the way IN, not only on the way out (`.to(self.transformer
            # .dtype)` below only fixes the output).
            condition_encoder_dtype = next(self.condition_encoder.parameters()).dtype
            condition = self.condition_encoder(
                frame_hiddens[:, chunk_start:chunk_end].to(device=device, dtype=condition_encoder_dtype)
            )
            condition = condition.to(self.transformer.dtype)

            overlap = 0
            if previous_latent is not None:
                overlap = min(previous_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = previous_condition[:, :overlap]

            latents = randn_tensor(
                (1, self.num_channels_latents, condition.shape[1]),
                generator=generator,
                device=device,
                dtype=condition.dtype,
            )
            noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None

            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
            self.scheduler.set_timesteps(sigmas=sigmas, device=device)
            timesteps = self.scheduler.timesteps

            zeros_condition = torch.zeros_like(condition)

            for i, t in enumerate(timesteps):
                raise_if_cancelled()
                if overlap > 0:
                    time_value = t.to(latents.dtype)
                    latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * time_value) * noise_prompt + (
                        time_value * previous_latent[..., :overlap]
                    )
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                cond_pred = self.transformer(
                    hidden_states=latents, timestep=timestep, encoder_hidden_states=condition, return_dict=False
                )[0]
                uncond_pred = self.transformer(
                    hidden_states=latents, timestep=timestep, encoder_hidden_states=zeros_condition, return_dict=False
                )[0]
                velocity = uncond_pred + flow_guidance_scale * (cond_pred - uncond_pred)
                latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]

                if progress_callback:
                    try:
                        progress_callback(k * num_inference_steps + i + 1, total_steps, "flow")
                    except Exception as exc:
                        print(f"[MiniMaxMusic3] progress_callback raised during flow denoise: {exc!r}")

            if overlap > 0:
                latents[..., :overlap] = previous_latent[..., :overlap]

            overlap_start = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT_LENGTH)
            overlap_end = max(overlap_start, latents.shape[-1] - OVERLAP_LATENT_LENGTH)
            previous_latent = latents[..., overlap_start:overlap_end]
            previous_condition = condition[:, overlap_start:overlap_end]

            latent_chunks.append(latents)

        return latent_chunks

    # ------------------------------------------------------------------
    # Stage 5: decode. Upstream `decoders.py::MiniMaxMusic3VocoderDecodeStep`.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, latent_chunks: List[torch.Tensor], output_type: str = "pt"):
        if output_type not in ("np", "pt"):
            raise ValueError(f"Invalid output_type: {output_type}")

        hop_length = self.latent_hop_length
        num_chunks = len(latent_chunks)
        waveform_chunks = []
        for chunk_index, latents in enumerate(latent_chunks):
            raise_if_cancelled()
            waveform = self.vocoder(latents.to(self.vocoder.dtype))
            left = 0 if chunk_index == 0 else CROP_LEFT_LATENT * hop_length
            right = 0 if chunk_index == num_chunks - 1 else CROP_RIGHT_LATENT * hop_length
            waveform_chunks.append(waveform[..., left : waveform.shape[-1] - right])

        audios = torch.cat(waveform_chunks, dim=-1).float().clamp(-1.0, 1.0)
        if output_type == "np":
            audios = audios.cpu().numpy()
        return audios

    # ------------------------------------------------------------------
    # SushiUI addition (design doc phase plan item 8, repaint's both modes):
    # `decode`, generalized to decode a SUB-RANGE of a song's chunks with the
    # crop treatment of the chunk's GLOBAL position in the whole song, rather
    # than assuming `latent_chunks[0]`/`latent_chunks[-1]` are the true first/
    # last chunks of the whole sequence (which `decode` above always assumes,
    # correctly, for both a fresh generation and extend's tail-only call).
    # Repaint needs this because it decodes only a WINDOW of chunks from the
    # middle (or a truncated end) of an already-longer song: the plain
    # `decode` above would wrongly treat that window's own first/last chunk
    # as the whole song's edges (crop 0 there) even when the song continues
    # on one or both sides.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_range(
        self,
        latent_chunks: List[torch.Tensor],
        is_global_first: bool,
        is_global_last: bool,
        output_type: str = "pt",
    ):
        """Like `decode`, but `is_global_first`/`is_global_last` tell this call whether
        `latent_chunks[0]`/`latent_chunks[-1]` are truly the first/last chunk of the WHOLE
        song (crop 0 on that side, matching `decode`'s own edge rule) or an INTERNAL window
        (crop `CROP_LEFT_LATENT`/`CROP_RIGHT_LATENT`, matching `decode`'s treatment of every
        chunk that is not at an edge). Every chunk strictly between the first and last of
        `latent_chunks` is always cropped on BOTH sides, exactly as `decode` already does --
        only the two edge chunks' treatment is overridable here.
        """
        if output_type not in ("np", "pt"):
            raise ValueError(f"Invalid output_type: {output_type}")

        hop_length = self.latent_hop_length
        num_chunks = len(latent_chunks)
        waveform_chunks = []
        for chunk_index, latents in enumerate(latent_chunks):
            raise_if_cancelled()
            waveform = self.vocoder(latents.to(self.vocoder.dtype))
            chunk_is_first = (chunk_index == 0) and is_global_first
            chunk_is_last = (chunk_index == num_chunks - 1) and is_global_last
            left = 0 if chunk_is_first else CROP_LEFT_LATENT * hop_length
            right = 0 if chunk_is_last else CROP_RIGHT_LATENT * hop_length
            waveform_chunks.append(waveform[..., left : waveform.shape[-1] - right])

        audios = torch.cat(waveform_chunks, dim=-1).float().clamp(-1.0, 1.0)
        if output_type == "np":
            audios = audios.cpu().numpy()
        return audios

    # ------------------------------------------------------------------
    # End-to-end. Upstream `modular_blocks_minimax_music3.py::MiniMaxMusic3Blocks`.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        lyrics: str,
        audio_duration: float,
        num_inference_steps: int,
        flow_guidance_scale: float,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pt",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        resume_frame_codes: Optional[torch.Tensor] = None,
        resume_prefix_codes: Optional[torch.Tensor] = None,
    ) -> MiniMaxMusic3GenerationResult:
        """Run all five stages and return audio plus the frame-code state contract.

        `audio_duration`, `num_inference_steps` and `flow_guidance_scale` are required, not defaulted -- see
        `denoise_chunks`'s docstring for why. `progress_callback` receives a `stage` discriminator
        (`"ar"`/`"flow"`; see the module docstring) so a caller can tell the two independent progress counters
        apart. See :meth:`generate_ar` for the resume contract; `resume_frame_codes`/`resume_prefix_codes` are
        forwarded to it unchanged.
        """
        text_ids = self.encode_text(prompt, lyrics)
        ar_result = self.generate_ar(
            text_ids,
            audio_duration,
            generator=generator,
            progress_callback=progress_callback,
            resume_frame_codes=resume_frame_codes,
            resume_prefix_codes=resume_prefix_codes,
        )
        latent_chunks = self.denoise_chunks(
            ar_result.frame_hiddens,
            num_inference_steps=num_inference_steps,
            flow_guidance_scale=flow_guidance_scale,
            generator=generator,
            progress_callback=progress_callback,
        )
        audio = self.decode(latent_chunks, output_type=output_type)
        return MiniMaxMusic3GenerationResult(
            audio=audio,
            sample_rate=self.sampling_rate,
            frame_codes=ar_result.frame_codes,
            prefix_codes=ar_result.prefix_codes,
            num_frames=ar_result.frame_codes.shape[0],
        )
