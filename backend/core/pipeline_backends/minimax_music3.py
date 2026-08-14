"""MiniMax Music 3 txt2aud backend mixin for DiffusionPipelineManager.

Design doc phase plan item 3: drives the ported plain-class pipeline
(``core.models.minimax_music3.pipeline.MiniMaxMusic3Pipeline``, phase 1) over
the components the loader built (``core.models.minimax_music3.loader``,
phase 2) -- staged GPU/CPU placement, one combined progress series over the
pipeline's two independently-counted stages, cancellation (handled inside the
ported pipeline; this mixin must not swallow it), and the frame-code state
contract (``core.models.minimax_music3.frame_codes``). The route that
persists the returned codes alongside the saved audio file, and the
``arch_capabilities``/``param_defaults`` wiring that lets a real request
reach this code, land in a later commit (design doc phase plan item 4) --
see ``MiniMaxMusic3Txt2AudResult``'s docstring.

Staged offload
---------------
The AR stage calls the language model (~17 GB bf16) and RVQ depth decoder
(~1.3 GB) directly, submodule by submodule, so both must be resident on the
same device. The flow stage needs the transformer (~4.9 GB) and condition
encoder (tiny, fp32) together; the vocoder (~0.2 GB, fp32) only for decode.
The two pairs are never both wanted on the GPU at once, so staging is
sequential, offloading between stages only, never within one:

  1. language_model + rvq_depth_decoder -> GPU; ``encode_text`` +
     ``generate_ar``; -> CPU.
  2. transformer + condition_encoder -> GPU; ``denoise_chunks``; -> CPU.
  3. vocoder -> GPU; ``decode``; -> CPU.

Each transition is a ``try/finally`` (mirrors ``AceStepMixin``/
``MiniMaxH3Mixin``'s staging pattern) so an exception or cancellation mid-
stage still offloads that stage's components before propagating; the
pipeline's own ``raise_if_cancelled()`` calls are frequent enough that a
cancel request lands within one stage, never mid-transition.

``keep_models_hot`` (``core.keep_hot``) is not wired in here -- a scope
decision, not a structural impossibility. It tracks residency per component,
so it would let a subsequent QUEUED generation skip a resident component's
own ->GPU staging call. The obvious candidate is the language model: kept
resident across a queue, every song after the first could skip its ~17 GB
transfer at stage 1. Wiring it in needs stage 1's ``finally`` to consult
``keep_hot.should_keep_resident`` instead of unconditionally offloading, and
confirming the flow stage's transformer can be staged alongside a kept-hot
LM within budget (plausible here; the two pairs together are still well
under a modern card's capacity). This is a different situation from
``MiniMaxH3Mixin``'s exclusion (its text encoder is never resident as a
whole module -- it streams layer by layer, so keep-hot has nothing to
track there); noted so a future integration is not modeled on the wrong
precedent.

Progress weighting
--------------------
No timing harness exists in this repo to measure the two stages, so this
mixin counts MODEL FORWARD CALLS per stage instead, using the pipeline's own
report granularity (one tick per AR frame; one tick per flow scheduler
step):

  * AR tick: 1 language-model forward + ``num_codebooks - 1`` (7) RVQ
    depth-decoder forwards (each batches cond+uncond into one call).
  * Flow tick: 2 transformer forwards (``cond_pred``/``uncond_pred`` are
    separate calls, not batched).

Each stage's tick count times its per-tick call count gives a "total forward
calls" figure; the two stages' shares of that total become fixed budgets
(out of ``PROGRESS_TOTAL_UNITS``), sized once before generation starts from
``audio_duration``/``num_inference_steps`` alone (``compute_progress_
budget``). This ignores model size and per-call token count, so it is a
coarse proxy, not a FLOPs or wall-clock model -- but it reproduces the
qualitatively correct behavior (a 300 s song's AR call count, ~60,000,
dwarfs a typical flow configuration's, ~180, so AR correctly claims most of
the bar). Every caller only sees the resulting ``(step, total)`` pair via
the ordinary WebSocket ``progress`` message (``WS_PROTOCOL.md``); no new
message type is added here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, NamedTuple, Optional

import random

import torch

# ---------------------------------------------------------------------------
# Progress weighting: pure, weight-free functions (no component access), so
# they are unit-testable without a loaded model. See module docstring.
# ---------------------------------------------------------------------------

# A single fixed total reported for the WHOLE generation (both stages), so
# the WS layer's `(step, total)` pair stays one stable series across a stage
# transition instead of resetting to a new denominator mid-generation (see
# the module docstring's citation of the pipeline's own module docstring on
# why a raw two-stage callback cannot be forwarded as-is).
PROGRESS_TOTAL_UNITS = 10_000

# denoise_chunks makes `cond_pred` and `uncond_pred` as two SEPARATE
# transformer() calls per scheduler step (see MiniMaxMusic3Pipeline.
# denoise_chunks) -- unlike the AR stage's language-model/depth-decoder
# calls, which batch cond+uncond into ONE call each.
_FLOW_FORWARD_CALLS_PER_STEP = 2

# MiniMaxMusic3Vocoder.forward's `waveform.reshape(batch_size, 2, -1)` is unconditional -- every song this
# checkpoint produces is stereo, regardless of config. Extend's channel-count sanity check
# (`_generate_audoutpaint_minimax_music3`, audit finding F3) compares the SOURCE FILE against this fixed constant
# rather than against the newly generated audio, so it can run immediately after the file is read instead of only
# after a full (staged, GPU) generation completes.
_MINIMAX_MUSIC3_EXPECTED_CHANNELS = 2


def estimate_num_chunks(num_frames: int, chunk_frames: int, chunk_hop: int) -> int:
    """Mirrors `MiniMaxMusic3Pipeline.prepare_chunks`'s chunk-count arithmetic
    without needing a `frame_hiddens` tensor -- used to size the flow-stage
    progress budget BEFORE the AR stage has produced one. Kept as a free
    function (not a pipeline method) so `compute_progress_budget` can be
    tested with no model loaded at all.
    """
    if num_frames <= chunk_frames:
        return 1
    return len(range(0, num_frames - chunk_hop, chunk_hop))


def compute_progress_budget(
    max_frames: int,
    num_codebooks: int,
    num_inference_steps: int,
    chunk_frames: int,
    chunk_hop: int,
    total_units: int = PROGRESS_TOTAL_UNITS,
) -> "tuple[int, int]":
    """`(ar_budget, flow_budget)`, summing to `total_units`.

    `max_frames` MUST be computed the same way `MiniMaxMusic3Pipeline.
    generate_ar` computes it (`min(int(audio_duration * frame_rate),
    MAX_AUDIO_FRAMES)`) -- see the module docstring: the AR stage's own
    reported `total` (the denominator in every `(step, total, "ar")`
    callback) is exactly this number, so sizing the budget from a MATCHING
    value is what keeps the mapped AR fraction reaching 1.0 exactly when the
    AR stage's own progress does (barring early stop -- see
    `_generate_txt2aud_minimax_music3`'s docstring).
    """
    ar_calls_per_frame = 1 + max(int(num_codebooks) - 1, 0)
    estimated_chunks = estimate_num_chunks(max(int(max_frames), 0), chunk_frames, chunk_hop)
    ar_cost = max(int(max_frames), 0) * ar_calls_per_frame
    flow_cost = estimated_chunks * max(int(num_inference_steps), 0) * _FLOW_FORWARD_CALLS_PER_STEP
    total_cost = ar_cost + flow_cost
    if total_cost <= 0:
        # Degenerate (zero-length request) -- give the whole budget to
        # whichever stage would still run rather than divide by zero.
        return total_units, 0
    ar_budget = int(round(total_units * ar_cost / total_cost))
    ar_budget = max(0, min(ar_budget, total_units))
    flow_budget = total_units - ar_budget
    return ar_budget, flow_budget


def combined_progress(
    stage: str,
    step: int,
    total: int,
    ar_budget: int,
    flow_budget: int,
    total_units: int = PROGRESS_TOTAL_UNITS,
) -> int:
    """Map one `(step, total, stage)` tick from the pipeline's own two
    independent counters onto the single combined `(0, total_units]` series.

    Monotonic WITHIN a stage by construction (`step` is non-decreasing and
    `total`/the budget are fixed for the call); monotonic ACROSS the AR ->
    flow transition because the flow bucket starts exactly where the AR
    bucket's own maximum (`ar_budget`) ends, regardless of how large a
    fraction of its OWN budget the AR stage actually reached (see
    `_generate_txt2aud_minimax_music3`'s docstring on early stop).
    """
    if stage not in ("ar", "flow"):
        raise ValueError(f"Unknown MiniMax Music 3 progress stage {stage!r}; expected 'ar' or 'flow'.")
    total = max(int(total), 1)
    step = max(0, min(int(step), total))
    fraction = step / total
    if stage == "ar":
        combined = ar_budget * fraction
    else:
        combined = ar_budget + flow_budget * fraction
    return max(0, min(int(round(combined)), total_units))


class MiniMaxMusic3Txt2AudResult(NamedTuple):
    """Return shape of `_generate_txt2aud_minimax_music3`.

    Deliberately NOT the plain `(waveform, sample_rate, actual_seed)`
    3-tuple `AceStepMixin._generate_txt2aud_acestep` returns (and that
    `routes.py`'s `generate_txt2aud` currently unpacks positionally): the
    design doc's "Per-generation state contract" requires the frame codes to
    survive past this call so a LATER commit's route can write them into the
    sidecar (`core.models.minimax_music3.frame_codes`) alongside the saved
    audio file -- "this must ship with txt2aud, not after it" (design doc).
    Silently cramming that into a 3-tuple, or dropping it and reconstructing
    it later, was rejected: the codes are cheap to carry now and expensive
    (a second full AR pass) to reconstruct later.

    `routes.py` does not call `generate_txt2aud` for a MiniMax Music 3 model
    today (`_reject_if_music3_model_not_yet_wired` gates it) -- design doc
    phase plan item 4 is what removes that gate and starts branching on this
    richer return shape (mirroring how it already branches on
    `pipeline_manager.is_minimax_music3_model` for the rejection itself), so
    this is not a breaking change to any code path that runs today.
    """

    waveform: torch.Tensor  # [2, samples], CPU, float32, [-1, 1]
    sample_rate: int
    actual_seed: int
    frame_codes: torch.Tensor  # [num_frames, num_codebooks], CPU, int64
    prefix_codes: torch.Tensor  # [1, num_codebooks], CPU, int64
    num_frames: int
    frame_rate: float
    prompt: str
    lyrics: str


class MiniMaxMusic3ExtendResult(NamedTuple):
    """Return shape of `_generate_audoutpaint_minimax_music3` (design doc phase plan item 7).

    Mirrors `MiniMaxMusic3Txt2AudResult` field-for-field (same rationale: the state contract must survive the call
    so a route can persist a NEW sidecar for the extended song, so extend-of-an-extend keeps working), plus
    `appended_num_frames` -- diagnostic/gallery-metadata surface for how many NEW frames THIS call actually added
    (may be less than requested: `audio_duration` is an upper bound, see `_generate_txt2aud_minimax_music3`'s
    docstring on early stop).

    `frame_codes`/`prefix_codes`/`num_frames` here are the FULL, concatenated (previous + newly generated) code
    sequence for the WHOLE song -- not just this call's new tail -- so a caller writing a sidecar for the extended
    file does not have to separately track and re-concatenate the previous sidecar's codes itself.
    """

    waveform: torch.Tensor  # [2, samples], CPU, float32, [-1, 1] -- FULL song (preserved span + new tail)
    sample_rate: int
    actual_seed: int
    frame_codes: torch.Tensor  # [num_frames, num_codebooks], CPU, int64 -- FULL song
    prefix_codes: torch.Tensor  # [1, num_codebooks], CPU, int64 -- unchanged from the original generation
    num_frames: int  # FULL song's frame count
    frame_rate: float
    prompt: str
    lyrics: str
    appended_num_frames: int  # how many of `num_frames` are NEW (this call only)


class MiniMaxMusic3Mixin:
    """MiniMaxMusic3Mixin: lyrics- and caption-conditioned music generation
    (8B Qwen3 language model + 0.6B RVQ depth decoder -> 2.4B flow-matching
    DiT -> vocoder). See module docstring for the staged-offload contract."""

    # ------------------------------------------------------------------
    # Component staging
    # ------------------------------------------------------------------

    def _minimax_music3_move(self, names, device, *, allow_partial_failure: bool = False) -> None:
        """Move each named component to `device`.

        `allow_partial_failure` must stay `False` (the default) for every
        ->GPU staging call: `nn.Module.to()` moves its parameters one at a
        time, so a failure partway through (e.g. a CUDA OOM on the ~17 GB
        language model) leaves the component split across two devices, and
        letting that failure raise here is what stops generation from
        continuing into an unrelated device-mismatch error inside a forward
        pass later. Only the ->CPU cleanup calls in a stage's `finally`
        block pass `allow_partial_failure=True`: raising there would replace
        whatever exception is already propagating, so a failure is instead
        recorded (`self._minimax_music3_stranded`) and logged, not silently
        dropped to a print with no trace of which component may still be
        holding GPU memory.
        """
        components = getattr(self, "minimax_music3_components", None) or {}
        for name in names:
            comp = components.get(name)
            if comp is None or not hasattr(comp, "to"):
                continue
            try:
                comp.to(device)
            except Exception as exc:
                if not allow_partial_failure:
                    raise
                stranded = getattr(self, "_minimax_music3_stranded", None)
                if stranded is None:
                    stranded = set()
                    self._minimax_music3_stranded = stranded
                stranded.add(name)
                print(
                    f"[MiniMaxMusic3] WARNING: could not move {name!r} to {device!r} during cleanup; "
                    f"it may be partially resident on its previous device: {exc}"
                )

    def _minimax_music3_empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # txt2aud
    # ------------------------------------------------------------------

    def _generate_txt2aud_minimax_music3(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> MiniMaxMusic3Txt2AudResult:
        """Generate a music waveform from a caption + lyrics (MiniMax Music 3).

        Args:
            params: `prompt` (the music description, required, non-empty),
                `lyrics` (required, non-empty -- see
                `MiniMaxMusic3Pipeline.encode_text`'s validation), `seed`
                (int, -1/None = random), `audio_duration`/
                `num_inference_steps`/`flow_guidance_scale` -- all THREE
                REQUIRED with no fallback here (see design doc's generation-
                parameter table; the user-facing defaults belong in
                `backend/api/param_defaults.py`, a later commit).
            progress_callback: called as `(step, total)` with `total ==
                PROGRESS_TOTAL_UNITS` fixed for the whole call -- the
                ordinary WS `progress` message shape every other backend
                already reports (module docstring: no new message type).
            step_callback: reserved, unused (mirrors every other audio
                backend's contract).

        Returns:
            `MiniMaxMusic3Txt2AudResult` -- see its own docstring for why
            this is NOT the plain ACE-Step 3-tuple.

        AR early stop. `audio_duration` is an UPPER BOUND (design doc): the
        language model may emit its end-of-audio token before `max_frames`
        is reached, in which case the AR stage's own progress never reaches
        its full reported total and `result.num_frames` will be less than
        `int(audio_duration * frame_rate)`. The combined progress series
        still reaches `PROGRESS_TOTAL_UNITS` exactly at the end of this
        call (the final unconditional callback below), regardless.
        """
        from api.error_handlers import ValidationError
        from core.models.minimax_music3.defaults import (
            CHUNK_FRAMES,
            CHUNK_HOP,
            FALLBACK_FRAME_RATE,
            FALLBACK_NUM_CODEBOOKS,
            MAX_AUDIO_FRAMES,
        )
        from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline

        if not getattr(self, "is_minimax_music3_model", False) or not self.minimax_music3_components:
            raise ValidationError(
                "Text-to-music generation requires a MiniMax Music 3 model",
                detail="The currently loaded model is not a MiniMax Music 3 audio model.",
            )

        comps = self.minimax_music3_components
        tokenizer = comps.get("tokenizer")
        language_model = comps.get("language_model")
        rvq_depth_decoder = comps.get("rvq_depth_decoder")
        condition_encoder = comps.get("condition_encoder")
        transformer = comps.get("transformer")
        scheduler = comps.get("scheduler")
        vocoder = comps.get("vocoder")

        missing = [
            name for name, comp in (
                ("tokenizer", tokenizer),
                ("language_model", language_model),
                ("rvq_depth_decoder", rvq_depth_decoder),
                ("condition_encoder", condition_encoder),
                ("transformer", transformer),
                ("scheduler", scheduler),
                ("vocoder", vocoder),
            ) if comp is None
        ]
        if missing:
            detail = f"missing component(s): {', '.join(missing)}."
            if "language_model" in missing:
                detail += " The language model is required to generate audio; reload the full model."
            raise ValidationError("MiniMax Music 3 model is missing a required component", detail=detail)

        prompt = params.get("prompt") or ""
        lyrics = params.get("lyrics") or ""
        if not prompt.strip():
            raise ValidationError(
                "`prompt` (the music description) is required",
                detail="`prompt` must be a non-empty string.",
            )
        if not lyrics.strip():
            raise ValidationError(
                "`lyrics` is required",
                detail="`lyrics` must be a non-empty string (the checkpoint's input contract "
                       "requires it; instrumental tracks are expressed through caption/structure "
                       "tags in `prompt`, not by omitting `lyrics`).",
            )

        # audio_duration / num_inference_steps / flow_guidance_scale: required,
        # no fallback -- see this method's docstring.
        for required_key in ("audio_duration", "num_inference_steps", "flow_guidance_scale"):
            if params.get(required_key) is None:
                raise ValidationError(
                    f"`{required_key}` is required",
                    detail=f"`{required_key}` must be provided explicitly; no default is available yet.",
                )

        audio_duration = float(params["audio_duration"])
        num_inference_steps = int(params["num_inference_steps"])
        flow_guidance_scale = float(params["flow_guidance_scale"])

        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        frame_rate = float(comps.get("frame_rate", FALLBACK_FRAME_RATE))
        num_codebooks = int((comps.get("rvq_depth_decoder_config") or {}).get("num_codebooks", FALLBACK_NUM_CODEBOOKS))
        # MUST mirror MiniMaxMusic3Pipeline.generate_ar's own clamp exactly
        # (module docstring, compute_progress_budget's own note) so the
        # progress budget is sized against the SAME number the pipeline
        # itself reports as the AR stage's total.
        max_frames = min(int(audio_duration * frame_rate), MAX_AUDIO_FRAMES)

        # `generate_ar` re-checks both of these, but only AFTER the ~18 GB
        # language-model + depth-decoder GPU staging below -- cheap to check
        # here first, before that move, so an invalid request never pays for
        # a staging round trip.
        if audio_duration <= 0:
            raise ValidationError(
                "`audio_duration` must be positive", detail=f"Got {audio_duration}.",
            )
        if max_frames == 0:
            raise ValidationError(
                "`audio_duration` is too short to produce a single audio frame",
                detail=f"{audio_duration}s is shorter than one frame at {frame_rate} frames/sec; "
                       f"increase `audio_duration`.",
            )

        ar_budget, flow_budget = compute_progress_budget(
            max_frames, num_codebooks, num_inference_steps, CHUNK_FRAMES, CHUNK_HOP,
        )

        def _combined_progress(step, total, stage) -> None:
            if progress_callback is None:
                return
            try:
                combined = combined_progress(stage, step, total, ar_budget, flow_budget)
                progress_callback(combined, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        pipeline = MiniMaxMusic3Pipeline(
            tokenizer=tokenizer,
            language_model=language_model,
            rvq_depth_decoder=rvq_depth_decoder,
            condition_encoder=condition_encoder,
            transformer=transformer,
            scheduler=scheduler,
            vocoder=vocoder,
        )

        # ---- Stage 1: autoregressive (LM + depth decoder co-resident) ----
        # See module docstring "Staged offload". The ->GPU move here uses the
        # default allow_partial_failure=False (must raise on a partial move,
        # not continue with a split-device component).
        self._minimax_music3_move(("language_model", "rvq_depth_decoder"), device)
        # The pipeline's own co-residency guard (generate_ar) only inspects
        # modules carrying an accelerate `_hf_hook`; under this backend's
        # manual staging there are none, so that guard can never fire here.
        # Assert it ourselves instead of relying on a check that is
        # structurally disabled for this call path.
        lm_device = next(language_model.parameters()).device
        depth_device = next(rvq_depth_decoder.parameters()).device
        if lm_device != depth_device:
            raise RuntimeError(
                f"MiniMax Music 3's language model ({lm_device}) and RVQ depth decoder "
                f"({depth_device}) are not on the same device after staging; the autoregressive "
                f"stage requires both co-resident."
            )
        try:
            text_ids = pipeline.encode_text(prompt, lyrics)
            ar_result = pipeline.generate_ar(
                text_ids,
                audio_duration,
                generator=generator,
                progress_callback=_combined_progress,
            )
        finally:
            self._minimax_music3_move(("language_model", "rvq_depth_decoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        # `frame_hiddens` ([1, num_frames, num_codebooks * hidden_size], the
        # AR stage's LM-dtype output) would otherwise stay GPU-resident
        # through the entire flow stage -- at the 9,000-frame cap that is the
        # ~590 MB tensor the design doc's state contract calls too big to
        # STORE, held live for no reason while it sits idle. `denoise_chunks`
        # already pulls each chunk's own slice back to its execution device
        # per iteration, so moving the whole tensor to CPU here costs one
        # slice-back per chunk and frees the rest between chunks.
        ar_result.frame_hiddens = ar_result.frame_hiddens.detach().to("cpu")

        # ---- Stage 2: flow-matching (transformer + condition encoder) ----
        self._minimax_music3_move(("transformer", "condition_encoder"), device)
        try:
            latent_chunks = pipeline.denoise_chunks(
                ar_result.frame_hiddens,
                num_inference_steps=num_inference_steps,
                flow_guidance_scale=flow_guidance_scale,
                generator=generator,
                progress_callback=_combined_progress,
            )
        finally:
            self._minimax_music3_move(("transformer", "condition_encoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        # ---- Stage 3: decode (vocoder) ----
        self._minimax_music3_move(("vocoder",), device)
        try:
            audio = pipeline.decode(latent_chunks, output_type="pt")
        finally:
            self._minimax_music3_move(("vocoder",), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        if progress_callback is not None:
            try:
                progress_callback(PROGRESS_TOTAL_UNITS, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        if torch.isnan(audio).any() or torch.isinf(audio).any():
            raise RuntimeError(
                f"MiniMax Music 3 generation produced NaN/Inf audio (shape={list(audio.shape)})."
            )
        if audio.numel() > 0 and audio.abs().sum() == 0:
            raise RuntimeError("MiniMax Music 3 generation produced all-silent (all-zero) audio.")

        waveform = audio[0].detach().to("cpu").float()
        sample_rate = int(pipeline.sampling_rate)

        return MiniMaxMusic3Txt2AudResult(
            waveform=waveform,
            sample_rate=sample_rate,
            actual_seed=seed,
            frame_codes=ar_result.frame_codes.detach().to("cpu"),
            prefix_codes=ar_result.prefix_codes.detach().to("cpu"),
            num_frames=int(ar_result.frame_codes.shape[0]),
            frame_rate=float(pipeline.frame_rate),
            prompt=prompt,
            lyrics=lyrics,
        )

    # ------------------------------------------------------------------
    # Extend (design doc phase plan item 7 -- "outpaint / extend",
    # `POST /generate/outpaint/audio`). The route/`arch_capabilities`/
    # `param_defaults` wiring that lets a real request reach this method
    # land in a separate commit; this is the mechanism only.
    # ------------------------------------------------------------------

    @staticmethod
    def _minimax_music3_load_source_waveform(path: str) -> "tuple[torch.Tensor, int]":
        """Load the FULL waveform at `path`, `[channels, samples]` float32 CPU, plus its sample rate.

        Unlike `AceStepMixin._acestep_load_reference_audio` (used for an UPLOADED reference clip, which gets
        resampled/renormalized to a fixed stereo/48kHz target before use), this does NOT resample or reshape
        channels: extend's preserved span must be spliced back sample-exact to whatever is ALREADY on disk -- the
        file MiniMax Music 3 itself wrote for the song being extended -- so a mismatch against the sidecar's
        recorded `sample_rate` is a signal the file is not the one the sidecar belongs to (checked by the caller via
        `MiniMaxMusic3FrameCodes.matches`), not something to silently coerce away.
        """
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
        wav = torch.from_numpy(data.T).contiguous()  # [channels, samples]
        return wav, int(sr)

    @staticmethod
    def _minimax_music3_apply_extend_waveform_splice(
        original_wave: torch.Tensor,
        new_wave: torch.Tensor,
        sample_rate: int,
        crossfade_ms: float = 10.0,
    ) -> torch.Tensor:
        """Concatenate `new_wave` (the newly generated tail) onto `original_wave` (the preserved span, read
        verbatim from the source file) SAMPLE-EXACTLY -- every sample of `original_wave` is returned unmodified.
        This is the design doc's chosen preservation property for extend: option (b) from the design brief ("keep
        the original waveform for its span and splice the newly generated tail on"), NOT option (a) ("re-decode
        everything"). Only `original_wave` is exact to its ON-DISK (decoded) representation -- see
        `AceStepMixin._acestep_apply_outpaint_waveform_splice`'s docstring for the same "sample-exact to the
        decoded representation, not to a pre-encode float tensor" caveat, which applies identically here.

        Extend has only ONE boundary (unlike ACE-Step's outpaint, which can have generated audio on BOTH sides of a
        held span): original -> new, with nothing preserved after the new tail. A short declick ramp is applied
        ENTIRELY WITHIN `new_wave`'s own leading samples -- never touching `original_wave` -- fading its amplitude
        from the exact last preserved sample's level up to its own natural level. This is the same "no reference
        audio on the far side of the boundary, so level-match rather than content-blend" reasoning
        `_acestep_apply_outpaint_waveform_splice` already uses for its own un-referenced boundaries; a genuine
        CONTENT crossfade (blending `original_wave`'s tail samples with `new_wave`'s head samples) was rejected
        because it would necessarily overwrite/alter some of `original_wave`'s own samples, which the "preserved
        span must be sample-exact" gate does not allow.

        Note on what this ramp does and does not fix -- and what the seam actually is (see
        `_generate_audoutpaint_minimax_music3`'s docstring, "Flow-stage scope", for the full accounting): the new
        tail's FIRST flow chunk is decoded with `decode()`'s `chunk_index == 0` crop rule (`left = 0`) -- i.e.
        exactly like the start of a brand-new song, with no left context and none of the overlap blend the
        200/100/172 chunk geometry exists to provide everywhere else in this pipeline. This ramp is a plain
        amplitude/DC declick across a fixed millisecond window; it cannot and does not disguise that the new
        tail begins as a fresh onset -- it only removes the audible AMPLITUDE step at the boundary, not any
        difference in timbre/dynamics/onset character between the two sides. Say so plainly rather than implying
        this is a content-aware crossfade: it is not one.
        """
        original_wave = original_wave.to(dtype=torch.float32)
        new_wave = new_wave.to(dtype=torch.float32).clone()
        if original_wave.shape[-1] == 0:
            return new_wave
        if new_wave.shape[-1] == 0:
            return original_wave

        crossfade_samples = max(0, int(round((crossfade_ms / 1000.0) * sample_rate)))
        n = min(crossfade_samples, new_wave.shape[-1])
        if n > 0:
            boundary_value = original_wave[..., -1:]  # exact last preserved sample, per channel
            frac = torch.linspace(0.0, 1.0, n + 2, device=new_wave.device, dtype=new_wave.dtype)[1:-1]
            seg = new_wave[..., :n]
            new_wave[..., :n] = seg * frac + boundary_value.to(new_wave.dtype) * (1.0 - frac)

        return torch.cat([original_wave, new_wave], dim=-1)

    def _generate_audoutpaint_minimax_music3(
        self,
        params: Dict[str, Any],
        reference_audio,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> MiniMaxMusic3ExtendResult:
        """Forward-extend a MiniMax-Music3-generated song by resuming the autoregressive stage from its stored
        frame-code sidecar (design doc "Modality surfaces" / "Per-generation state contract").

        Mechanism: `core.models.minimax_music3.frame_codes.read_frame_codes_sidecar_for_audio` locates and reads
        the `<song>.mm3frames.json` sidecar next to `reference_audio`, `MiniMaxMusic3FrameCodes.matches` validates
        it against the currently loaded model AND the audio file's own identity (sample rate/frame rate/codebook
        count, the file's actual sample count, AND a content hash computed HERE from the file just read -- see
        "Identity validation" below for why that hash is never taken from the caller) before anything is trusted,
        then `MiniMaxMusic3Pipeline.generate_ar`'s `resume_frame_codes`/`resume_prefix_codes` path teacher-force
        replays the stored history to rebuild the language model's KV cache and continues sampling -- proven
        bit-exact for the shared prefix by an independent audit (see the design doc and
        `backend/tests/minimax_music3_ar_resume_test.py`).

        Identity validation (audit finding F4). `content_hash` passed to `MiniMaxMusic3FrameCodes.matches` is
        ALWAYS computed here, server-side, from `reference_audio` itself (`utils.image_utils.calculate_file_hash`
        -- the same hash `routes.py`'s save path already writes into a NEW sidecar's `content_hash` field, see
        `routes.py`'s frame-code-sidecar-write block), NEVER taken from `params`. A caller-suppliable hash is an
        opt-out: `matches()` skips a `None` content-hash check entirely, so a request that simply omits it (or a
        route that never threads a caller-supplied one through) would fall back to `num_samples` alone -- and two
        songs generated from the SAME `audio_duration` that both run to `MAX_AUDIO_FRAMES`/early-stop identically
        have IDENTICAL sample counts by construction, not by coincidence, meaning a completely different song's
        audio would be silently extended and returned as if it were a continuation of the requested one. Computing
        the hash from the file this call itself just read closes that hole unconditionally; it costs one more read
        of a file already on local disk.

        Placement. Only `params["placement"] == "extend_forward"` is supported and REQUIRED (no default is assumed)
        -- the AR stage is a CAUSAL language model: it can only continue a sequence forward from an existing KV
        cache, never backward or into the middle. BOTH refusal paths -- an omitted `placement` and an explicitly
        unsupported one (e.g. a caller wanting to extend BACKWARD from the song's start) -- carry this same causal-
        LM reason (audit finding F6: the omitted-`placement` message previously named only the supported value with
        no explanation, so a caller/log reader who only saw that message had no way to tell this was a structural
        limit rather than an unfinished feature), mirroring how `MiniMaxH3Mixin`'s video outpaint enumerates its
        supported placements and refuses the rest rather than silently approximating one placement with another.

        Preservation property actually delivered. This implements the design brief's option (b): the ORIGINAL
        waveform, read verbatim from `reference_audio` (the file on disk), is returned UNMODIFIED for its own span;
        only the newly generated tail is appended. This is sample-exact to the file's OWN decoded representation
        (matching `_acestep_apply_outpaint_waveform_splice`'s identical caveat for ACE-Step), not to some other
        precision. It is explicitly NOT option (a) ("re-decode everything") -- the preserved span is never run back
        through the flow stage or the vocoder. See `_minimax_music3_apply_extend_waveform_splice`'s docstring for
        the boundary treatment.

        Flow-stage scope. Flow (denoising) work is restricted to EXACTLY the newly generated frame region --
        `pipeline.denoise_chunks` is called with ONLY `ar_result.frame_hiddens` (the AR-resume call's own return,
        which already covers only the new frames), never any frame-hiddens from the previously generated span.

        The alternative considered -- re-deriving ~`CHUNK_FRAMES` (200) frames of the PRECEDING song's own
        frame-hiddens so the new tail's leading flow chunk could be seeded with `denoise_chunks`'s own
        `previous_latent`/`previous_condition` continuity mechanism -- was rejected for this phase, but NOT because
        it would cost as much as generating that many new frames (an earlier draft of this docstring claimed that;
        it is wrong on all three counts and is corrected here so a follow-up phase costs this accurately):
          * the language-model HALF of a context frame's `frame_hiddens` is already computed and thrown away by the
            existing resume replay -- `generate_ar`'s batched KV-cache-rebuild pass runs the LM over the WHOLE
            history and keeps only the final hidden state, so recovering ~200 more of them is not "free" but is
            also not a new LM cost class: it is the SAME batched-forward mechanism already paid for, sized larger;
          * the RVQ depth-decoder HALF is batchable, not the sequential 7-step CFG sampling chain `_generate_depth_
            codes` runs during live generation: with the codes already KNOWN (no sampling, no CFG needed), it is
            one batched forward per frame, not a per-frame sampling loop;
          * only ~200 frames of context (one chunk) are ever needed, never the whole song's history.
        The genuinely irreducible cost is `previous_latent` itself: it is an OUTPUT of actually running one context
        chunk through the flow-matching transformer for `num_inference_steps` steps -- there is no way to obtain it
        without paying for that one chunk's flow-stage compute, a BOUNDED constant (one chunk, independent of song
        length) rather than the unbounded-with-history cost this docstring previously implied. This was still not
        implemented for this phase (added complexity/risk against a benefit this docstring is now explicit about --
        see the next paragraph for what that benefit actually is and is not), but the reason is a real complexity
        tradeoff, not the inflated cost argument the earlier draft gave.

        `denoise_chunks`'s OWN internal windowing/overlap-blend still applies IF the new region itself spans
        multiple flow chunks (i.e. the extend duration exceeds one ~8s chunk), since that mechanism operates
        entirely within the frame-hiddens this call passes it, independent of anything from the original
        generation.

        What the seam actually is (recorded honestly, not glossed over): `MiniMaxMusic3Pipeline.decode` applies
        `left = 0` for `chunk_index == 0` -- the new tail's FIRST flow chunk is decoded with NO left context and
        NONE of the overlap blend the 200/100/172 chunk geometry exists to provide EVERYWHERE ELSE in this
        pipeline. It is decoded exactly as if it were the start of a brand-new song. The 10ms declick ramp
        (`_minimax_music3_apply_extend_waveform_splice`) removes the audible AMPLITUDE step at the boundary; it
        cannot and does not make the two sides acoustically continuous in timbre or onset character, because that
        would require the flow stage to have actually been conditioned on the true preceding audio (the rejected
        alternative above), which this implementation does not do. A follow-up phase choosing to close this gap
        should budget for one context chunk's flow-stage compute, not the whole song.

        Prompt/lyrics. The song's ORIGINAL `prompt`/`lyrics` (from the sidecar) are ALWAYS reused for the
        continuation, never a caller-supplied replacement -- but NOT because a caller-supplied prompt would corrupt
        a resumed KV cache (audit finding F1: an earlier draft of this docstring claimed that, and it is factually
        wrong -- `generate_ar` builds NO KV cache from disk at all; every call constructs a FRESH one via
        `embed_tokens(text_ids)` -> `language_model.model(inputs_embeds=text_embeds, ...)`, and only AFTER that
        replays the STORED CODES (never a stored KV cache) on top via `_embed_audio_frames`. Nothing but frame
        codes comes off disk. A different prompt for the continuation is therefore MECHANICALLY SUPPORTED, at zero
        extra cost, by this pipeline today). The real reason this is refused is a PRODUCT decision, not a
        mechanical impossibility: the checkpoint was trained to condition an entire song's generation on ONE
        caption/lyrics pair from the start; switching the caption mid-sequence (predicting frames 401+ from a KV
        cache built on prompt A, guided by a CFG contrast built from prompt B) is untrained-distribution behavior
        with no evidence either way about output quality, so it is refused here as a conservative default rather
        than as a limit that cannot be lifted. A future phase revisiting this should test it, not assume it is
        impossible. If `params` supplies a non-empty `prompt`/`lyrics` that differs from the sidecar's own, it is
        ignored and a warning is surfaced (`api.generation_status.add_warning`) rather than silently dropped with
        no trace.

        Budget guards. `core.models.minimax_music3.pipeline.check_ar_resume_budget` (the same pure function
        `generate_ar` itself calls, with `duration_param_name="extend_duration_sec"` and `prompt_is_adjustable=
        False` so its message names THIS path's actual parameter and never tells a caller to shorten a prompt it
        does not control -- audit finding F2) is run here BEFORE either GPU staging move, using the sidecar's own
        `num_frames` and a pre-staging tokenization of the (reused) prompt -- an over-budget request is rejected
        before paying for the ~18 GB language-model + depth-decoder move, not after. The `encode_text` preflight
        call itself is wrapped so an unusable sidecar prompt/lyrics (e.g. an empty string in a hand-edited or
        corrupted sidecar) surfaces as a `ValidationError`, not a raw `ValueError` reaching the caller as an
        unrelated HTTP 500 (audit finding F5).

        Args:
            params: `placement` (required, see above), `extend_duration_sec` (float, required, > 0 -- the
                UPPER BOUND on how much MORE audio to generate, same "duration is an upper bound" semantics as
                `audio_duration` elsewhere in this module), `num_inference_steps`/`flow_guidance_scale` (required,
                no fallback, same convention as `_generate_txt2aud_minimax_music3`), `seed` (int, -1/None =
                random), `prompt`/`lyrics` (optional -- see "Prompt/lyrics" above; NEVER used to override the
                sidecar's own). NOTE: there is no `content_hash` param -- see "Identity validation" above for why
                that value is always computed server-side rather than accepted from a caller.
            reference_audio: a filesystem PATH (`str`) to the song being extended -- NOT raw upload bytes. Extend
                fundamentally requires the sidecar file already sitting next to `reference_audio` on this server
                (see module docstring's per-generation state contract); an arbitrary in-memory upload has no
                sidecar to find, so this refuses anything that is not an existing path rather than accepting bytes
                and failing later with a confusing "sidecar not found" error.
            progress_callback: same `(step, total)` contract as `_generate_txt2aud_minimax_music3`, reusing its
                `PROGRESS_TOTAL_UNITS`/`compute_progress_budget`/`combined_progress` machinery unchanged (sized
                against `extend_duration_sec`'s own `max_new_frames`, exactly as a fresh generation would be).
            step_callback: reserved, unused.

        Returns:
            `MiniMaxMusic3ExtendResult` -- the FULL (preserved + new) waveform and code sequence, so a caller can
            write a new sidecar for the extended file and extend it again later.
        """
        from api.error_handlers import ValidationError
        from core.models.minimax_music3.defaults import (
            CHUNK_FRAMES,
            CHUNK_HOP,
            MAX_AUDIO_FRAMES,
        )
        from core.models.minimax_music3.frame_codes import read_frame_codes_sidecar_for_audio
        from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline, check_ar_resume_budget

        if not getattr(self, "is_minimax_music3_model", False) or not self.minimax_music3_components:
            raise ValidationError(
                "MiniMax Music 3 audio extend requires a MiniMax Music 3 model",
                detail="The currently loaded model is not a MiniMax Music 3 audio model.",
            )

        comps = self.minimax_music3_components
        tokenizer = comps.get("tokenizer")
        language_model = comps.get("language_model")
        rvq_depth_decoder = comps.get("rvq_depth_decoder")
        condition_encoder = comps.get("condition_encoder")
        transformer = comps.get("transformer")
        scheduler = comps.get("scheduler")
        vocoder = comps.get("vocoder")

        missing = [
            name for name, comp in (
                ("tokenizer", tokenizer),
                ("language_model", language_model),
                ("rvq_depth_decoder", rvq_depth_decoder),
                ("condition_encoder", condition_encoder),
                ("transformer", transformer),
                ("scheduler", scheduler),
                ("vocoder", vocoder),
            ) if comp is None
        ]
        if missing:
            detail = f"missing component(s): {', '.join(missing)}."
            if "language_model" in missing:
                detail += " The language model is required to extend audio; reload the full model."
            raise ValidationError("MiniMax Music 3 model is missing a required component", detail=detail)

        # ---- placement: only extend_forward, and it must be requested explicitly ----
        # Both refusal messages below carry the SAME causal-LM reason (audit finding F6): the missing-`placement`
        # message previously said only "no default is assumed" with no explanation, indistinguishable from an
        # unimplemented feature rather than a structural limit.
        _causal_lm_reason = (
            "MiniMax Music 3's autoregressive stage is a causal language model: it can only continue a song "
            "forward from its existing end, never backward from its start or into the middle. The only "
            "supported placement is 'extend_forward'."
        )
        placement = params.get("placement")
        if placement is None:
            raise ValidationError(
                "`placement` is required for MiniMax Music 3 audio extend",
                detail=f"No default is assumed. {_causal_lm_reason}",
            )
        if placement != "extend_forward":
            raise ValidationError(
                f"Unsupported placement {placement!r} for MiniMax Music 3 audio extend",
                detail=_causal_lm_reason,
            )

        # ---- reference_audio must be a server-side path (the sidecar lives next to it) ----
        if not isinstance(reference_audio, str) or not reference_audio:
            raise ValidationError(
                "MiniMax Music 3 audio extend requires a server-side audio file path",
                detail="Extend resumes the autoregressive stage from a frame-code sidecar stored next to the "
                       "original audio file; an in-memory upload has no such sidecar to find. Select an existing "
                       "MiniMax Music 3 song (e.g. from the gallery) rather than uploading a new file.",
            )
        import os as _os
        if not _os.path.isfile(reference_audio):
            raise ValidationError(
                "MiniMax Music 3 audio extend: source audio file not found",
                detail=f"No file at {reference_audio!r}.",
            )

        # ---- locate + validate the sidecar ----
        try:
            sidecar = read_frame_codes_sidecar_for_audio(reference_audio)
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 audio extend: the frame-code sidecar is unreadable",
                detail=str(exc),
            )
        if sidecar is None:
            raise ValidationError(
                "MiniMax Music 3 audio extend: no frame-code sidecar found",
                detail=f"No sidecar next to {reference_audio!r}. This song either predates the frame-code "
                       f"sidecar feature or was not generated by MiniMax Music 3; it cannot be extended.",
            )

        pipeline = MiniMaxMusic3Pipeline(
            tokenizer=tokenizer,
            language_model=language_model,
            rvq_depth_decoder=rvq_depth_decoder,
            condition_encoder=condition_encoder,
            transformer=transformer,
            scheduler=scheduler,
            vocoder=vocoder,
        )

        try:
            original_wave, original_sr = self._minimax_music3_load_source_waveform(reference_audio)
        except Exception as exc:
            raise ValidationError(
                "MiniMax Music 3 audio extend: could not read the source audio file",
                detail=f"{reference_audio!r}: {exc}",
            )

        current_model_info = getattr(self, "current_model_info", None) or {}
        model_hash = current_model_info.get("model_hash") or None

        # Audit finding F4: the content-hash half of identity validation is computed HERE, from the file this call
        # just read, and is NEVER accepted from `params` -- see this method's docstring, "Identity validation", for
        # why a caller-suppliable hash is an opt-out (a `None` is silently skipped by `matches()`, and two songs
        # generated from the same `audio_duration` that both reach the same stop condition have IDENTICAL sample
        # counts by construction, not coincidence -- `num_samples` alone is not a strong enough check on its own).
        from utils.image_utils import calculate_file_hash
        source_content_hash = calculate_file_hash(reference_audio) or None

        if not sidecar.matches(
            sample_rate=int(pipeline.sampling_rate),
            frame_rate=float(pipeline.frame_rate),
            num_codebooks=int(pipeline.num_codebooks),
            model_hash=model_hash,
            num_samples=int(original_wave.shape[-1]),
            content_hash=source_content_hash,
        ):
            raise ValidationError(
                "MiniMax Music 3 audio extend: the sidecar does not match this audio file or the loaded model",
                detail=f"sidecar sample_rate={sidecar.sample_rate}, frame_rate={sidecar.frame_rate}, "
                       f"num_codebooks={sidecar.num_codebooks}, num_samples={sidecar.num_samples} vs the currently "
                       f"loaded model ({pipeline.sampling_rate} Hz, {pipeline.frame_rate} frames/s, "
                       f"{pipeline.num_codebooks} codebooks) and the file on disk ({original_wave.shape[-1]} "
                       f"samples @ {original_sr} Hz, content hash {source_content_hash!r}). This sidecar may "
                       f"belong to a different file, a different model checkpoint, or the file may have been "
                       f"overwritten since it was generated.",
            )
        if original_sr != sidecar.sample_rate:
            raise ValidationError(
                "MiniMax Music 3 audio extend: source file sample rate does not match its sidecar",
                detail=f"File is {original_sr} Hz; sidecar declares {sidecar.sample_rate} Hz.",
            )
        # Audit finding F3: the channel count is knowable from the SOURCE FILE ALONE (MiniMax Music 3's vocoder
        # always decodes to stereo -- `MiniMaxMusic3Vocoder.forward`'s `waveform.reshape(batch_size, 2, -1)` is
        # unconditional), so check it HERE, immediately after the file is read, rather than only after the full
        # generation completes (the channel-count comparison against the newly generated audio used to happen
        # post-decode, at the very end of this method -- on the real checkpoint that is minutes of GPU time and
        # three staging round trips spent on a request this check could have refused for free at load time).
        if original_wave.shape[0] != _MINIMAX_MUSIC3_EXPECTED_CHANNELS:
            raise ValidationError(
                "MiniMax Music 3 audio extend: source file is not stereo",
                detail=f"Source file has {original_wave.shape[0]} channel(s); MiniMax Music 3 always decodes to "
                       f"{_MINIMAX_MUSIC3_EXPECTED_CHANNELS} (stereo), so this file cannot be the vocoder's own "
                       f"output for the sidecar next to it.",
            )

        # ---- prompt/lyrics: always reused from the sidecar -- see this method's docstring ----
        prompt = sidecar.prompt
        lyrics = sidecar.lyrics
        requested_prompt = params.get("prompt")
        requested_lyrics = params.get("lyrics")
        if (requested_prompt and requested_prompt != prompt) or (requested_lyrics and requested_lyrics != lyrics):
            # Audit finding F8: no bare `except Exception: pass` around this -- `add_warning` is documented as
            # unable to raise for a normal (str, code=str) call, so swallowing an exception here could only ever
            # hide a REAL bug (e.g. an import error) with no trace, not a legitimate failure mode of this call.
            from api.generation_status import add_warning
            add_warning(
                "MiniMax Music 3 extend reuses the original song's prompt/lyrics (required for the "
                "autoregressive resume to be well-defined); the prompt/lyrics supplied with this request "
                "were ignored.",
                code="minimax_music3_extend_prompt_ignored",
            )

        # ---- required, no-fallback generation params (same convention as txt2aud) ----
        for required_key in ("extend_duration_sec", "num_inference_steps", "flow_guidance_scale"):
            if params.get(required_key) is None:
                raise ValidationError(
                    f"`{required_key}` is required",
                    detail=f"`{required_key}` must be provided explicitly; no default is available yet.",
                )

        extend_duration_sec = float(params["extend_duration_sec"])
        num_inference_steps = int(params["num_inference_steps"])
        flow_guidance_scale = float(params["flow_guidance_scale"])
        if extend_duration_sec <= 0:
            raise ValidationError(
                "`extend_duration_sec` must be positive", detail=f"Got {extend_duration_sec}.",
            )

        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)

        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        frame_rate = float(pipeline.frame_rate)
        max_new_frames = min(int(extend_duration_sec * frame_rate), MAX_AUDIO_FRAMES)
        if max_new_frames == 0:
            raise ValidationError(
                "`extend_duration_sec` is too short to produce a single audio frame",
                detail=f"{extend_duration_sec}s is shorter than one frame at {frame_rate} frames/sec; "
                       f"increase `extend_duration_sec`.",
            )

        # ---- budget guards BEFORE staging (design doc requirement) ----
        # A cheap pre-staging tokenization of the (reused) prompt/lyrics, purely to get the prompt's token
        # count for the position-budget check; the real, on-device `text_ids` used for generation is recomputed
        # AFTER staging below (mirrors `_generate_txt2aud_minimax_music3`'s ordering).
        #
        # Audit finding F5: wrapped -- `encode_text` raises a raw `ValueError` for an empty/unusable prompt or
        # lyrics string (or a tokenizer failure), which would otherwise surface as an unrelated HTTP 500 blaming a
        # prompt the CALLER of this request never supplied (the prompt/lyrics here always come from the sidecar,
        # never from `params` -- see this method's docstring, "Prompt/lyrics").
        try:
            preflight_text_ids = pipeline.encode_text(prompt, lyrics)
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 audio extend: the sidecar's stored prompt/lyrics could not be tokenized",
                detail=f"{exc} This sidecar's prompt/lyrics may be empty or corrupted; the song cannot be "
                       f"extended until it is re-generated.",
            )
        max_position_embeddings = getattr(language_model.config, "max_position_embeddings", None)
        try:
            check_ar_resume_budget(
                prompt_tokens=int(preflight_text_ids.shape[1]),
                total_frames_so_far=sidecar.num_frames,
                max_frames=max_new_frames,
                max_position_embeddings=max_position_embeddings,
                duration_param_name="extend_duration_sec",
                prompt_is_adjustable=False,
            )
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 audio extend request exceeds the checkpoint's limits",
                detail=str(exc),
            )

        ar_budget, flow_budget = compute_progress_budget(
            max_new_frames, int(pipeline.num_codebooks), num_inference_steps, CHUNK_FRAMES, CHUNK_HOP,
        )

        def _combined_progress(step, total, stage) -> None:
            if progress_callback is None:
                return
            try:
                combined = combined_progress(stage, step, total, ar_budget, flow_budget)
                progress_callback(combined, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        # ---- Stage 1: autoregressive resume (LM + depth decoder co-resident) ----
        self._minimax_music3_move(("language_model", "rvq_depth_decoder"), device)
        lm_device = next(language_model.parameters()).device
        depth_device = next(rvq_depth_decoder.parameters()).device
        if lm_device != depth_device:
            raise RuntimeError(
                f"MiniMax Music 3's language model ({lm_device}) and RVQ depth decoder "
                f"({depth_device}) are not on the same device after staging; the autoregressive "
                f"stage requires both co-resident."
            )
        try:
            try:
                text_ids = pipeline.encode_text(prompt, lyrics)
            except ValueError as exc:
                # Same F5 wrapping as the preflight call above; kept independent (not "trust the preflight and
                # skip this one") because this call runs against the real, on-device execution path.
                raise ValidationError(
                    "MiniMax Music 3 audio extend: the sidecar's stored prompt/lyrics could not be tokenized",
                    detail=str(exc),
                )
            ar_result = pipeline.generate_ar(
                text_ids,
                extend_duration_sec,
                generator=generator,
                progress_callback=_combined_progress,
                resume_frame_codes=sidecar.frame_codes,
                resume_prefix_codes=sidecar.prefix_codes,
            )
        finally:
            self._minimax_music3_move(("language_model", "rvq_depth_decoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        appended_num_frames = int(ar_result.frame_codes.shape[0])
        ar_result.frame_hiddens = ar_result.frame_hiddens.detach().to("cpu")

        # ---- Stage 2: flow-matching -- restricted to the NEW frame region only; see this method's docstring
        # "Flow-stage scope" for why the preceding song's own frame-hiddens are not re-derived here. ----
        self._minimax_music3_move(("transformer", "condition_encoder"), device)
        try:
            latent_chunks = pipeline.denoise_chunks(
                ar_result.frame_hiddens,
                num_inference_steps=num_inference_steps,
                flow_guidance_scale=flow_guidance_scale,
                generator=generator,
                progress_callback=_combined_progress,
            )
        finally:
            self._minimax_music3_move(("transformer", "condition_encoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        # ---- Stage 3: decode (vocoder) -- new tail only ----
        self._minimax_music3_move(("vocoder",), device)
        try:
            new_audio = pipeline.decode(latent_chunks, output_type="pt")
        finally:
            self._minimax_music3_move(("vocoder",), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        if progress_callback is not None:
            try:
                progress_callback(PROGRESS_TOTAL_UNITS, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        if torch.isnan(new_audio).any() or torch.isinf(new_audio).any():
            raise RuntimeError(
                f"MiniMax Music 3 extend produced NaN/Inf audio (shape={list(new_audio.shape)})."
            )
        # Audit finding F7: parity with `_generate_txt2aud_minimax_music3`'s own all-silent guard -- without this,
        # a silent new tail spliced onto a perfectly good preserved span would ship as a "successful" extend.
        if new_audio.numel() > 0 and new_audio.abs().sum() == 0:
            raise RuntimeError("MiniMax Music 3 extend produced all-silent (all-zero) audio for the new tail.")

        new_waveform = new_audio[0].detach().to("cpu").float()
        sample_rate = int(pipeline.sampling_rate)

        # Internal-invariant check, not a user-facing ValidationError: the source file's channel count was already
        # validated against `_MINIMAX_MUSIC3_EXPECTED_CHANNELS` right after it was read (audit finding F3), and
        # the vocoder unconditionally decodes to that same channel count, so this can only fire on a genuine bug.
        if original_wave.shape[0] != new_waveform.shape[0]:
            raise RuntimeError(
                f"MiniMax Music 3 audio extend: channel count mismatch between the (already-validated) source "
                f"file ({original_wave.shape[0]} channel(s)) and the newly generated audio "
                f"({new_waveform.shape[0]} channel(s)) -- this should be unreachable."
            )

        full_waveform = self._minimax_music3_apply_extend_waveform_splice(
            original_wave, new_waveform, sample_rate,
        )

        full_frame_codes = torch.cat([sidecar.frame_codes.to(torch.long), ar_result.frame_codes.detach().to("cpu")], dim=0)

        return MiniMaxMusic3ExtendResult(
            waveform=full_waveform,
            sample_rate=sample_rate,
            actual_seed=seed,
            frame_codes=full_frame_codes,
            prefix_codes=sidecar.prefix_codes.to(torch.long),
            num_frames=int(full_frame_codes.shape[0]),
            frame_rate=frame_rate,
            prompt=prompt,
            lyrics=lyrics,
            appended_num_frames=appended_num_frames,
        )
