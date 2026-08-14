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
