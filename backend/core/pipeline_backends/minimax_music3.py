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


# ---------------------------------------------------------------------------
# Chunk/decode geometry: pure, weight-free functions (no component access),
# used by repaint's "re-render a range" mode (design doc phase plan item 8,
# "Modality surfaces") to work out, IN SAMPLES, exactly which span of an
# already-decoded audio file a given range of chunk indices corresponds to --
# without running the model. This is what makes a byte-exact splice possible:
# `MiniMaxMusic3ConditionEncoder.forward`'s frame -> latent resample (nearest-
# neighbor, `input_sampling_rate`/`input_hop_length` -> `output_sampling_rate`/
# `output_hop_length`) and `MiniMaxMusic3Pipeline.decode`'s crop (`CROP_LEFT_
# LATENT`/`CROP_RIGHT_LATENT` latent frames per window) are BOTH deterministic
# functions of a chunk's FRAME COUNT alone, never of the tensor's CONTENT --
# so replaying that exact arithmetic here, with no model loaded, reproduces
# the real decode's per-chunk sample counts precisely. `_generate_txt2aud_
# minimax_music3`'s own `frame_hiddens` -> `latents` -> `decode` path is the
# ground truth this mirrors; `minimax_music3_chunk_geometry_test.py` proves
# the same crop arithmetic against a real (tiny, synthetic) pipeline.
# ---------------------------------------------------------------------------
def prepare_chunk_starts(num_frames: int, chunk_frames: int, chunk_hop: int) -> "list[int]":
    """Pure mirror of `MiniMaxMusic3Pipeline.prepare_chunks`'s chunk-start arithmetic (which needs a real
    `frame_hiddens` tensor only to read its own frame-axis length) -- identical output for the same `num_frames`.
    """
    if num_frames <= chunk_frames:
        return [0]
    return list(range(0, num_frames - chunk_hop, chunk_hop))


def compute_condition_latent_length(
    num_frames_in_chunk: int,
    input_sampling_rate: int,
    input_hop_length: int,
    output_sampling_rate: int,
    output_hop_length: int,
) -> int:
    """Pure mirror of `MiniMaxMusic3ConditionEncoder.forward`'s `latent_length` formula -- the exact integer
    `F.interpolate(..., size=latent_length, mode="nearest")` target it computes from a chunk's frame count alone,
    with NO dependency on the tensor's content. Reproduced here so repaint's geometry accounting needs no model.
    """
    return max(
        1,
        int(num_frames_in_chunk * output_sampling_rate / input_sampling_rate * input_hop_length / output_hop_length),
    )


def compute_chunk_geometry(
    num_frames_total: int,
    chunk_frames: int,
    chunk_hop: int,
    input_sampling_rate: int,
    input_hop_length: int,
    output_sampling_rate: int,
    output_hop_length: int,
) -> "list[dict]":
    """Every chunk `MiniMaxMusic3Pipeline.prepare_chunks`/`denoise_chunks` would produce for a song of
    `num_frames_total` emitted frames, as `{"start", "end", "latent_length"}` dicts (AR-frame start/end, and the
    condition encoder's own deterministic latent-frame count for that chunk) -- pure, no model needed.
    """
    chunk_starts = prepare_chunk_starts(num_frames_total, chunk_frames, chunk_hop)
    geometry = []
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_frames, num_frames_total)
        n = chunk_end - chunk_start
        latent_length = compute_condition_latent_length(
            n, input_sampling_rate, input_hop_length, output_sampling_rate, output_hop_length,
        )
        geometry.append({"start": chunk_start, "end": chunk_end, "latent_length": latent_length})
    return geometry


def compute_cumulative_samples(
    num_frames_total: int,
    hop_length: int,
    chunk_frames: int,
    chunk_hop: int,
    crop_left_latent: int,
    crop_right_latent: int,
    input_sampling_rate: int,
    input_hop_length: int,
    output_sampling_rate: int,
    output_hop_length: int,
) -> "list[int]":
    """`cumulative[k]` = the exact number of DECODED AUDIO SAMPLES `MiniMaxMusic3Pipeline.decode` would produce
    from chunks `[0, k)` of a `num_frames_total`-frame song's FULL chunk sequence -- i.e. the sample offset at
    which chunk `k`'s own (cropped) output begins in the full song's decoded waveform. `cumulative[len(chunk_
    starts)]` equals the FULL decoded sample count (a strong self-check: it must match the sidecar's own recorded
    `num_samples` for the file this is computed against -- see the callers below, which assert exactly that).

    Uses the SAME left/right crop exception `decode` uses -- crop 0 on the side that is chunk index 0 / the LAST
    chunk of the WHOLE `num_frames_total`-frame sequence, `crop_left_latent`/`crop_right_latent` otherwise -- so
    this is only meaningful when `num_frames_total` is the REAL total frame count of the song being spliced
    against (not an arbitrary sub-range's local frame count, which would misapply the edge exception).

    Geometry self-check (why matching `cumulative[-1]` against a file's real sample count is enough, even though
    it only checks the TOTAL length, not any individual boundary). Every INTERNAL chunk keeps exactly
    `CHUNK_HOP` frames' worth of latents (`CROP_RIGHT_LATENT` is defined as `344 - CROP_LEFT_LATENT`, and a
    100-frame hop is exactly 344 latents at the checkpoint's frame->latent ratio -- see `defaults.py`), so the
    per-chunk kept spans TELESCOPE: chunk `k`'s kept span starts exactly where chunk `k-1`'s ends, for every
    internal `k`. This means the individual `cumulative[k]` boundaries are not independent guesses that happen to
    sum to the right total -- once the SUM matches the file's real length, every intermediate entry is a genuine
    segment boundary on disk (a coarser subgrid of chunk starts, never a misaligned one). Checked directly against
    a real (tiny, synthetic) pipeline across 19 frame counts spanning every 200/100-hop boundary in
    `minimax_music3_repaint_test.py`, and by construction (not sampling) for every case the self-check accepts.
    """
    geometry = compute_chunk_geometry(
        num_frames_total, chunk_frames, chunk_hop,
        input_sampling_rate, input_hop_length, output_sampling_rate, output_hop_length,
    )
    num_chunks = len(geometry)
    cumulative = [0]
    for idx, g in enumerate(geometry):
        left = 0 if idx == 0 else crop_left_latent
        right = 0 if idx == num_chunks - 1 else crop_right_latent
        kept = max(0, g["latent_length"] - left - right)
        cumulative.append(cumulative[-1] + kept * hop_length)
    return cumulative


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


class MiniMaxMusic3RepaintResult(NamedTuple):
    """Return shape of `_generate_aud2aud_minimax_music3` (design doc phase plan item 8, "repaint").

    Mirrors `MiniMaxMusic3ExtendResult` field-for-field for the same reason (a route must be able to write a NEW
    sidecar for the repainted file so it can itself be extended or repainted again) -- `frame_codes`/`prefix_codes`/
    `num_frames` here are always the FULL song's code sequence after this call, for BOTH repaint sub-modes:

      * "regenerate" -- `frame_codes` is `codes[:T]` (the preserved prefix) concatenated with the freshly
        AR-resumed new tail; strictly SHORTER than the original song's codes when `T` is not the very end.
      * "rerender" -- `frame_codes` is UNCHANGED from the sidecar (the whole point of this mode is that the codes
        never change, only their flow-stage rendering does), returned here anyway so both sub-modes share one
        result shape and one sidecar-writing call site in `routes.py`.

    `repaint_mode` records which sub-mode actually ran (diagnostic/gallery-metadata surface, mirrors
    `MiniMaxMusic3ExtendResult.appended_num_frames`'s role) -- `"regenerate"` or `"rerender"`.
    """

    waveform: torch.Tensor  # [2, samples], CPU, float32, [-1, 1] -- FULL song after repaint
    sample_rate: int
    actual_seed: int
    frame_codes: torch.Tensor  # [num_frames, num_codebooks], CPU, int64 -- FULL song after repaint
    prefix_codes: torch.Tensor  # [1, num_codebooks], CPU, int64
    num_frames: int
    frame_rate: float
    prompt: str
    lyrics: str
    repaint_mode: str  # "regenerate" | "rerender"


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

    # ------------------------------------------------------------------
    # Repaint (design doc phase plan item 8 -- "inpaint / repaint",
    # `POST /generate/aud2aud` with `mode="repaint"`). Two honest modes
    # (design doc "Modality surfaces"):
    #
    #   * "regenerate" -- AR-resume with the prefix codes as context and a
    #     NEW tail, exactly like extend above, except the truncation point is
    #     somewhere in the MIDDLE of the song (discarding everything after it,
    #     including any content the song used to have there) rather than at
    #     the original end.
    #   * "rerender" -- the codes never change; only a WINDOW's flow-stage
    #     rendering is redone with a new seed (new timbre/mix; melody/timing/
    #     lyrics unchanged, since the autoregressive stage never runs).
    #
    # Mid-span infill with a preserved tail (changing codes in the middle
    # while an ORIGINAL, different-content tail after it stays intact) is not
    # offered by either mode and is not reachable through this dispatcher --
    # see the design doc's "Capability verdict": "the global LM is causal;
    # there is no infilling contract." "regenerate" discards everything after
    # its cut point; "rerender" never touches the codes at all.
    # ------------------------------------------------------------------

    def _minimax_music3_load_repaint_source(self, reference_audio_path):
        """Shared setup for both repaint sub-modes: component fetch, sidecar read + identity validation, source
        waveform read. Factored out of `_generate_audoutpaint_minimax_music3`'s equivalent opening section (same
        checks, same reasons) because both repaint sub-modes need EXACTLY this, not because extend also needs it
        again -- extend's own copy is untouched, so a change here cannot silently affect it.

        Returns `(pipeline, sidecar, original_wave, original_sr, model_hash)`. Raises `ValidationError` for every
        failure mode `_generate_audoutpaint_minimax_music3`'s docstring documents for its own identical checks
        (missing component, non-path reference_audio, missing/unreadable file, missing/mismatched sidecar,
        sample-rate mismatch, non-stereo source).
        """
        from api.error_handlers import ValidationError
        from core.models.minimax_music3.frame_codes import read_frame_codes_sidecar_for_audio
        from core.models.minimax_music3.pipeline import MiniMaxMusic3Pipeline

        if not getattr(self, "is_minimax_music3_model", False) or not self.minimax_music3_components:
            raise ValidationError(
                "MiniMax Music 3 audio repaint requires a MiniMax Music 3 model",
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
                detail += " The language model is required to repaint audio; reload the full model."
            raise ValidationError("MiniMax Music 3 model is missing a required component", detail=detail)

        if not isinstance(reference_audio_path, str) or not reference_audio_path:
            raise ValidationError(
                "MiniMax Music 3 audio repaint requires a server-side audio file path",
                detail="Repaint resumes/re-renders from a frame-code sidecar stored next to the original audio "
                       "file; an in-memory upload has no such sidecar to find. Select an existing MiniMax Music 3 "
                       "song (e.g. from the gallery) rather than uploading a new file.",
            )
        import os as _os
        if not _os.path.isfile(reference_audio_path):
            raise ValidationError(
                "MiniMax Music 3 audio repaint: source audio file not found",
                detail=f"No file at {reference_audio_path!r}.",
            )

        try:
            sidecar = read_frame_codes_sidecar_for_audio(reference_audio_path)
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 audio repaint: the frame-code sidecar is unreadable",
                detail=str(exc),
            )
        if sidecar is None:
            raise ValidationError(
                "MiniMax Music 3 audio repaint: no frame-code sidecar found",
                detail=f"No sidecar next to {reference_audio_path!r}. This song either predates the frame-code "
                       f"sidecar feature or was not generated by MiniMax Music 3; it cannot be repainted. Repaint "
                       f"only works on a song this server generated -- cover/style transfer from arbitrary "
                       f"uploaded audio is refused (the RVQ tokenizer's encoder is not published in this release, "
                       f"so no audio can be turned into semantic codes).",
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
            original_wave, original_sr = self._minimax_music3_load_source_waveform(reference_audio_path)
        except Exception as exc:
            raise ValidationError(
                "MiniMax Music 3 audio repaint: could not read the source audio file",
                detail=f"{reference_audio_path!r}: {exc}",
            )

        current_model_info = getattr(self, "current_model_info", None) or {}
        model_hash = current_model_info.get("model_hash") or None

        # Same "identity validation" reasoning as `_generate_audoutpaint_minimax_music3` -- see its docstring: the
        # content hash is ALWAYS computed here, server-side, from the file just read, never accepted from `params`.
        from utils.image_utils import calculate_file_hash
        source_content_hash = calculate_file_hash(reference_audio_path) or None

        if not sidecar.matches(
            sample_rate=int(pipeline.sampling_rate),
            frame_rate=float(pipeline.frame_rate),
            num_codebooks=int(pipeline.num_codebooks),
            model_hash=model_hash,
            num_samples=int(original_wave.shape[-1]),
            content_hash=source_content_hash,
        ):
            raise ValidationError(
                "MiniMax Music 3 audio repaint: the sidecar does not match this audio file or the loaded model",
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
                "MiniMax Music 3 audio repaint: source file sample rate does not match its sidecar",
                detail=f"File is {original_sr} Hz; sidecar declares {sidecar.sample_rate} Hz.",
            )
        if original_wave.shape[0] != _MINIMAX_MUSIC3_EXPECTED_CHANNELS:
            raise ValidationError(
                "MiniMax Music 3 audio repaint: source file is not stereo",
                detail=f"Source file has {original_wave.shape[0]} channel(s); MiniMax Music 3 always decodes to "
                       f"{_MINIMAX_MUSIC3_EXPECTED_CHANNELS} (stereo), so this file cannot be the vocoder's own "
                       f"output for the sidecar next to it.",
            )

        return pipeline, sidecar, original_wave, original_sr, model_hash

    @staticmethod
    def _minimax_music3_resolve_prompt_lyrics(sidecar, params):
        """Prompt/lyrics are ALWAYS reused from the sidecar for repaint, exactly like extend -- see
        `_generate_audoutpaint_minimax_music3`'s docstring, "Prompt/lyrics", for the full reasoning (mechanically
        supported at zero extra cost, refused as a product decision because the checkpoint was trained to condition
        a whole song on one caption/lyrics pair from the start). Returns `(prompt, lyrics)` and surfaces a warning
        (never silently drops) if `params` supplied a different, non-empty value for either.
        """
        prompt = sidecar.prompt
        lyrics = sidecar.lyrics
        requested_prompt = params.get("prompt")
        requested_lyrics = params.get("lyrics")
        if (requested_prompt and requested_prompt != prompt) or (requested_lyrics and requested_lyrics != lyrics):
            from api.generation_status import add_warning
            add_warning(
                "MiniMax Music 3 repaint reuses the original song's prompt/lyrics (required for both the "
                "autoregressive resume and the frame-hidden recovery to be well-defined); the prompt/lyrics "
                "supplied with this request were ignored.",
                code="minimax_music3_repaint_prompt_ignored",
            )
        return prompt, lyrics

    @staticmethod
    def _minimax_music3_snap_seconds_to_chunk_start(
        requested_seconds: float,
        frame_rate: float,
        chunk_starts: "list[int]",
        *,
        min_index: int = 0,
    ) -> "tuple[int, int]":
        """Snap a user-requested time (seconds) to the NEAREST entry of `chunk_starts` (AR-frame indices,
        `MiniMaxMusic3Pipeline.prepare_chunks`'s own chunk-window starts) at or after `min_index` (an index into
        `chunk_starts`). Repaint's boundary math (both sub-modes -- see `compute_cumulative_samples`'s docstring
        and the two callers below) is only sample-exact AT a chunk-window start, so an arbitrary requested second
        is always coerced to one of these, never used verbatim.

        No upper bound: both callers snap against the LAST entry of `chunk_starts` too (`_minimax_music3_repaint_
        regenerate`'s `T` may be the song's last chunk's own start -- see its docstring; `_minimax_music3_repaint_
        rerender` snaps its OWN end candidates separately, against exclusive chunk-count boundaries rather than
        through this function a second time).

        Returns `(chunk_index, frame_index)` -- `chunk_index` into `chunk_starts`, `frame_index = chunk_starts[
        chunk_index]`.
        """
        if not chunk_starts:
            raise ValueError("chunk_starts must be non-empty")
        min_index = max(0, min(min_index, len(chunk_starts) - 1))
        candidates = chunk_starts[min_index:]
        requested_frame = requested_seconds * frame_rate
        best_offset = min(range(len(candidates)), key=lambda i: abs(candidates[i] - requested_frame))
        chunk_index = min_index + best_offset
        return chunk_index, chunk_starts[chunk_index]

    def _minimax_music3_repaint_regenerate(
        self,
        params: Dict[str, Any],
        reference_audio_path: str,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> MiniMaxMusic3RepaintResult:
        """Repaint sub-mode "regenerate from T onward" (design doc "Modality surfaces").

        Mechanism: `params["repaint_start"]` (seconds) is snapped to the nearest chunk-window start `T` that is
        NOT the song's very first chunk (see `_minimax_music3_snap_seconds_to_chunk_start`'s `min_index=1` below
        -- `T` MAY be the song's LAST chunk's own start; regenerating just the final chunk is well-defined and
        `is_global_first=False` still holds, since a preserved predecessor still exists); everything before `T`
        is preserved VERBATIM from the source file (`_minimax_music3_
        apply_extend_waveform_splice` reused unchanged -- same "sample-exact to the decoded representation, one
        boundary, declick confined to the new side" contract as extend); `sidecar.frame_codes[:T]` becomes the
        AR-resume context for a NEW tail via `MiniMaxMusic3Pipeline.generate_ar`'s `resume_*` path, exactly like
        `_generate_audoutpaint_minimax_music3` -- codes from `T` onward in the ORIGINAL song are discarded, never
        read again.

        Splice alignment (the reason `T` must be a chunk-window start, not an arbitrary frame). Because
        `MiniMaxMusic3ConditionEncoder`'s frame -> latent resample is nearest-neighbor at a NON-integer ratio
        (`output_sampling_rate / input_sampling_rate * input_hop_length / output_hop_length` ~= 3.445), a chunk's
        KEPT (post-crop) audio span, measured in AR-frame-equivalent width, is NOT the clean `CHUNK_HOP` (100)
        frames the window geometry might suggest -- see `compute_cumulative_samples`'s docstring. The preserved
        prefix's LAST kept chunk therefore ends at frame `T + ~25` (an internal chunk's crop keeps roughly frames
        `[window_start + 25, window_start + 125)` of its own 200-frame window), not at `T` itself. The new tail's
        OWN first flow chunk covers the SAME window `[T, T + 200)` -- decoded with `CROP_LEFT_LATENT` applied
        (`is_global_first=False` on `MiniMaxMusic3Pipeline.decode_range`) rather than `decode`'s ordinary
        chunk-index-0 rule (`left=0`, what extend's tail uses) -- so its own kept span begins at the SAME `T + ~25`
        frame boundary the preserved prefix ends at: BOTH sides of the splice are cropped by the identical,
        purely-geometric (content-independent) amount, so they tile with no gap and no overlap in TIME COVERAGE.
        `is_global_first=False` is unconditional here because `T` is always required to be an INTERNAL chunk start
        (never the song's first), so there is always a preserved predecessor to align against.

        This alignment is exact in TIME COVERAGE, not in CONTENT -- the two sides are independently rendered audio
        that happen to tile without gap/overlap, so a short (10ms) declick ramp, confined entirely to the new
        tail's own leading samples, is still applied (same mechanism and same reasoning as extend's single
        boundary) to remove any audible amplitude step.

        Known limitation, tested rather than silently wrong (`minimax_music3_repaint_test.py`'s
        `test_rerender_after_a_short_regenerate_result_is_refused_not_mis_spliced`): this method's `decode_range`
        call always forces `is_global_first=False` for the new tail's own leading chunk, deliberately NOT the
        standard "chunk 0 of a fresh decode" rule -- the whole point is alignment against the preserved
        predecessor. This means the RESULT FILE's tail is not decoded the way a from-scratch decode of the SAME
        frame range would be. If a LATER request against this result file needs to recompute standard chunk
        geometry from ITS OWN (possibly now shorter) total frame count -- "rerender", or a second "regenerate" --
        and that recomputed geometry does not, by coincidence, describe the tail's actual crop treatment, the
        geometry self-check both sibling methods run (`cumulative[-1]` against the file's real sample count)
        refuses with a `RuntimeError` rather than mis-splicing. Extending the result (`_generate_audoutpaint_
        minimax_music3`) is NEVER affected: it only appends after the file's end and never recomputes any
        INTERNAL geometry of the existing audio.

        Args:
            params: `repaint_start` (seconds, required), `repaint_end` (seconds, required -- an UPPER BOUND on the
                new tail's total song length, same "duration is an upper bound" semantics as `audio_duration`/
                `extend_duration_sec` elsewhere in this module: the new tail's own duration is `repaint_end -
                T_seconds`, and the language model may stop earlier), `num_inference_steps`/`flow_guidance_scale`
                (required, no fallback), `seed` (int, -1/None = random). `prompt`/`lyrics` -- see
                `_minimax_music3_resolve_prompt_lyrics`.
            reference_audio_path: server-side file path, resolved by the caller (mirrors `_generate_audoutpaint_
                minimax_music3`'s identical contract).

        Returns:
            `MiniMaxMusic3RepaintResult` with `repaint_mode="regenerate"`.
        """
        from api.error_handlers import ValidationError
        from core.models.minimax_music3.defaults import (
            CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT, MAX_AUDIO_FRAMES,
        )
        from core.models.minimax_music3.pipeline import check_ar_resume_budget

        pipeline, sidecar, original_wave, original_sr, model_hash = self._minimax_music3_load_repaint_source(
            reference_audio_path,
        )
        prompt, lyrics = self._minimax_music3_resolve_prompt_lyrics(sidecar, params)

        for required_key in ("repaint_start", "repaint_end", "num_inference_steps", "flow_guidance_scale"):
            if params.get(required_key) is None:
                raise ValidationError(
                    f"`{required_key}` is required",
                    detail=f"`{required_key}` must be provided explicitly; no default is available yet.",
                )
        repaint_start_sec = float(params["repaint_start"])
        repaint_end_sec = float(params["repaint_end"])
        num_inference_steps = int(params["num_inference_steps"])
        flow_guidance_scale = float(params["flow_guidance_scale"])
        if repaint_end_sec <= repaint_start_sec:
            raise ValidationError(
                "Invalid repaint range",
                detail=f"repaint_end ({repaint_end_sec}) must be greater than repaint_start ({repaint_start_sec}) "
                       f"for MiniMax Music 3's 'regenerate' repaint mode (repaint_end is an upper bound on the "
                       f"new tail's total song length).",
            )

        chunk_starts = prepare_chunk_starts(sidecar.num_frames, CHUNK_FRAMES, CHUNK_HOP)
        if len(chunk_starts) < 2:
            raise ValidationError(
                "MiniMax Music 3 'regenerate' repaint requires a longer source song",
                detail=f"The source song is {sidecar.num_frames} frames ({sidecar.num_frames / sidecar.frame_rate:.2f}s "
                       f"at {sidecar.frame_rate} frames/s) -- shorter than two flow-matching chunk windows, so "
                       f"there is no chunk boundary to regenerate from other than the song's very first, which "
                       f"would leave nothing preserved. 'regenerate' needs a boundary strictly after the song's "
                       f"first chunk (it may be the LAST chunk's own start -- regenerating just the final chunk "
                       f"is well-defined).",
            )
        # min_index=1: T must be an INTERNAL-OR-LAST chunk start (never chunk_starts[0]==0, since "regenerate
        # from the very start" leaves nothing preserved -- see this method's docstring, "Splice alignment").
        # max_index is left at its default (the last entry of chunk_starts): T MAY be the last chunk's own start,
        # regenerating just the final chunk -- that is well-defined and is_global_first=False still holds, since
        # a preserved predecessor still exists.
        chunk_index, T = self._minimax_music3_snap_seconds_to_chunk_start(
            repaint_start_sec, sidecar.frame_rate, chunk_starts, min_index=1,
        )

        hop_length = int(pipeline.latent_hop_length)
        ce_config = pipeline.condition_encoder.config
        cumulative = compute_cumulative_samples(
            sidecar.num_frames, hop_length, CHUNK_FRAMES, CHUNK_HOP,
            CROP_LEFT_LATENT, CROP_RIGHT_LATENT,
            ce_config.input_sampling_rate, ce_config.input_hop_length,
            ce_config.output_sampling_rate, ce_config.output_hop_length,
        )
        # Self-check (module docstring, "Geometry self-check"): the full cumulative table must reproduce the
        # sidecar's own recorded sample count exactly, or this geometry does not describe the file actually on
        # disk. USER-REACHABLE, not only a defensive assertion: a "regenerate" result's tail is decoded with
        # continuity-preserving crop treatment (see `_minimax_music3_repaint_regenerate`'s "Known limitation"
        # paragraph), so a LATER repaint request against such a file can legitimately land here -- a
        # ValidationError (400), not a RuntimeError (500), because a caller can reach this without a bug.
        if cumulative[-1] != int(original_wave.shape[-1]):
            raise ValidationError(
                "MiniMax Music 3 repaint: this file's chunk geometry does not match its own sidecar",
                detail=f"Computed chunk geometry predicts {cumulative[-1]} total samples for this source file, "
                       f"but it actually has {int(original_wave.shape[-1])}. This can happen for a file that was "
                       f"itself produced by a 'regenerate' repaint whose tail used non-standard (continuity-"
                       f"preserving) crop treatment -- see that method's docstring, 'Known limitation'. Extending "
                       f"this file (rather than repainting it) is unaffected and still works.",
            )
        preserved_samples = cumulative[chunk_index]

        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        new_tail_duration_sec = repaint_end_sec - (T / sidecar.frame_rate)
        max_new_frames = min(int(new_tail_duration_sec * sidecar.frame_rate), MAX_AUDIO_FRAMES)
        if max_new_frames == 0:
            raise ValidationError(
                "MiniMax Music 3 'regenerate' repaint: the requested range is too short to produce a single "
                "audio frame",
                detail=f"repaint_end ({repaint_end_sec}s) minus the snapped repaint_start ({T / sidecar.frame_rate:.3f}"
                       f"s) is shorter than one frame at {sidecar.frame_rate} frames/sec; widen the range.",
            )

        comps = self.minimax_music3_components
        language_model = comps["language_model"]

        try:
            preflight_text_ids = pipeline.encode_text(prompt, lyrics)
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 repaint: the sidecar's stored prompt/lyrics could not be tokenized",
                detail=f"{exc} This sidecar's prompt/lyrics may be empty or corrupted; the song cannot be "
                       f"repainted until it is re-generated.",
            )
        max_position_embeddings = getattr(language_model.config, "max_position_embeddings", None)
        try:
            check_ar_resume_budget(
                prompt_tokens=int(preflight_text_ids.shape[1]),
                total_frames_so_far=T,
                max_frames=max_new_frames,
                max_position_embeddings=max_position_embeddings,
                duration_param_name="repaint_end",
                prompt_is_adjustable=False,
            )
        except ValueError as exc:
            raise ValidationError(
                "MiniMax Music 3 repaint request exceeds the checkpoint's limits",
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

        rvq_depth_decoder = comps["rvq_depth_decoder"]
        transformer = comps["transformer"]
        condition_encoder = comps["condition_encoder"]
        vocoder = comps["vocoder"]

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
                raise ValidationError(
                    "MiniMax Music 3 repaint: the sidecar's stored prompt/lyrics could not be tokenized",
                    detail=str(exc),
                )
            ar_result = pipeline.generate_ar(
                text_ids,
                new_tail_duration_sec,
                generator=generator,
                progress_callback=_combined_progress,
                resume_frame_codes=sidecar.frame_codes[:T],
                resume_prefix_codes=sidecar.prefix_codes,
            )
        finally:
            self._minimax_music3_move(("language_model", "rvq_depth_decoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        ar_result.frame_hiddens = ar_result.frame_hiddens.detach().to("cpu")

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

        self._minimax_music3_move(("vocoder",), device)
        try:
            # is_global_first=False: crop the new tail's own first chunk exactly like an INTERNAL chunk (see this
            # method's docstring, "Splice alignment") -- T is always an internal chunk start (min_index=1 above),
            # so there is always a preserved predecessor to align against. is_global_last=True: the new tail's
            # last chunk is the true end of the repainted song (nothing follows it), same as a fresh generation.
            new_audio = pipeline.decode_range(latent_chunks, is_global_first=False, is_global_last=True, output_type="pt")
        finally:
            self._minimax_music3_move(("vocoder",), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        if progress_callback is not None:
            try:
                progress_callback(PROGRESS_TOTAL_UNITS, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        if torch.isnan(new_audio).any() or torch.isinf(new_audio).any():
            raise RuntimeError(f"MiniMax Music 3 repaint produced NaN/Inf audio (shape={list(new_audio.shape)}).")
        if new_audio.numel() > 0 and new_audio.abs().sum() == 0:
            raise RuntimeError("MiniMax Music 3 repaint produced all-silent (all-zero) audio for the new tail.")

        new_waveform = new_audio[0].detach().to("cpu").float()
        sample_rate = int(pipeline.sampling_rate)

        preserved_prefix = original_wave[..., :preserved_samples]
        full_waveform = self._minimax_music3_apply_extend_waveform_splice(preserved_prefix, new_waveform, sample_rate)

        full_frame_codes = torch.cat(
            [sidecar.frame_codes[:T].to(torch.long), ar_result.frame_codes.detach().to("cpu")], dim=0,
        )

        return MiniMaxMusic3RepaintResult(
            waveform=full_waveform,
            sample_rate=sample_rate,
            actual_seed=seed,
            frame_codes=full_frame_codes,
            prefix_codes=sidecar.prefix_codes.to(torch.long),
            num_frames=int(full_frame_codes.shape[0]),
            frame_rate=float(sidecar.frame_rate),
            prompt=prompt,
            lyrics=lyrics,
            repaint_mode="regenerate",
        )

    @staticmethod
    def _minimax_music3_apply_rerender_waveform_splice(
        original_wave: torch.Tensor,
        new_middle_wave: torch.Tensor,
        start_sample: int,
        end_sample: int,
        sample_rate: int,
        crossfade_ms: float = 10.0,
    ) -> torch.Tensor:
        """Splice a re-rendered middle span into `original_wave` at `[start_sample, end_sample)`, sample-exact
        OUTSIDE that range (`original_wave[..., :start_sample]`/`original_wave[..., end_sample:]` are returned
        byte-identical -- never touched). Two boundaries (unlike extend's/regenerate's one), so up to two short
        (10ms) declick ramps are applied, EACH confined entirely to `new_middle_wave`'s own leading/trailing
        samples (never to `original_wave`'s) -- same "level-match, not content-blend" reasoning as
        `_minimax_music3_apply_extend_waveform_splice`'s single ramp. The left ramp is skipped when `start_sample
        == 0` (nothing precedes it to blend against, mirrors `_generate_audoutpaint_minimax_music3`'s "no
        reference on the far side" case); the right ramp is skipped when `end_sample == original_wave.shape[-1]`
        (nothing follows it).
        """
        original_wave = original_wave.to(dtype=torch.float32)
        new_middle_wave = new_middle_wave.to(dtype=torch.float32).clone()
        total_original_samples = original_wave.shape[-1]

        crossfade_samples = max(0, int(round((crossfade_ms / 1000.0) * sample_rate)))

        if start_sample > 0 and new_middle_wave.shape[-1] > 0:
            n = min(crossfade_samples, new_middle_wave.shape[-1])
            if n > 0:
                boundary_value = original_wave[..., start_sample - 1 : start_sample]
                frac = torch.linspace(0.0, 1.0, n + 2, device=new_middle_wave.device, dtype=new_middle_wave.dtype)[1:-1]
                seg = new_middle_wave[..., :n]
                new_middle_wave[..., :n] = seg * frac + boundary_value.to(new_middle_wave.dtype) * (1.0 - frac)

        if end_sample < total_original_samples and new_middle_wave.shape[-1] > 0:
            n = min(crossfade_samples, new_middle_wave.shape[-1])
            if n > 0:
                boundary_value = original_wave[..., end_sample : end_sample + 1]
                frac = torch.linspace(0.0, 1.0, n + 2, device=new_middle_wave.device, dtype=new_middle_wave.dtype)[1:-1]
                seg = new_middle_wave[..., -n:]
                # frac ramps 0->1 moving FORWARD through the segment; the trailing ramp needs the boundary value
                # approached at the END, so its blend weight runs 1->0 (reverse of the leading ramp's 0->1).
                new_middle_wave[..., -n:] = seg * frac.flip(0) + boundary_value.to(new_middle_wave.dtype) * (1.0 - frac.flip(0))

        return torch.cat(
            [original_wave[..., :start_sample], new_middle_wave, original_wave[..., end_sample:]], dim=-1,
        )

    def _minimax_music3_repaint_rerender(
        self,
        params: Dict[str, Any],
        reference_audio_path: str,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> MiniMaxMusic3RepaintResult:
        """Repaint sub-mode "re-render a range" (design doc "Modality surfaces"): the codes never change; only the
        flow-matching stage's rendering of a WINDOW is redone with a new seed. Timbre/mix change; lyrics/melody/
        timing do not, because the autoregressive stage never runs (no sampling, no CFG -- see `MiniMaxMusic3
        Pipeline.recover_frame_hiddens`'s docstring for why the recovered condition input is identical, up to
        floating-point reduction order, to what the original generation would have produced for these frames --
        not bit-for-bit, since the batch shapes differ, but nothing here depends on bit-exactness).

        Window geometry (design doc: "the pipeline denoises in 200-frame windows with a 100-frame hop ... [so]
        re-rendering 'a range' therefore touches more windows than the range itself"). `params["repaint_start"]`/
        `params["repaint_end"]` (seconds) are each snapped to the nearest chunk-window start (`_minimax_music3_
        snap_seconds_to_chunk_start`), giving chunk indices `[k1, k2)`. The frame_hiddens needed to re-render
        EXACTLY those chunks span AR frames `[chunk_starts[k1], min(chunk_starts[k2 - 1] + CHUNK_FRAMES,
        num_frames))` -- i.e. up to `CHUNK_FRAMES - CHUNK_HOP` (100) frames PAST the nominal end of the range, since
        chunk `k2 - 1`'s own 200-frame window extends that far -- recovered via `MiniMaxMusic3Pipeline.
        recover_frame_hiddens` (teacher-forced, deterministic, no sampling).

        Splice alignment. `MiniMaxMusic3Pipeline.decode_range` is called with `is_global_first=(k1 == 0)` and
        `is_global_last=(k2 == num_chunks_total)`: when the range is fully INTERNAL to the song, both edges get the
        SAME crop treatment (`CROP_LEFT_LATENT`/`CROP_RIGHT_LATENT`) the ORIGINAL decode gave those exact chunk
        positions -- purely geometric, content-independent (see `compute_cumulative_samples`'s docstring) -- so the
        re-rendered span's sample count and boundary positions are IDENTICAL to the original span's
        (`compute_cumulative_samples` computes the exact `[cumulative[k1], cumulative[k2])` sample range being
        replaced). Both boundaries get a short declick ramp (`_minimax_music3_apply_rerender_waveform_splice`,
        confined to the new middle span's own edges), for the same reason regenerate's single boundary does: exact
        alignment in TIME COVERAGE does not guarantee AMPLITUDE continuity across two independently-rendered spans.

        Args:
            params: `repaint_start`/`repaint_end` (seconds, required, `repaint_end > repaint_start`),
                `num_inference_steps`/`flow_guidance_scale` (required, no fallback), `seed` (int, -1/None =
                random). `prompt`/`lyrics` -- see `_minimax_music3_resolve_prompt_lyrics` (needed only to rebuild
                the KV-cache context for `recover_frame_hiddens`; no sampling/CFG happens in this mode at all).
            reference_audio_path: server-side file path, resolved by the caller.

        Returns:
            `MiniMaxMusic3RepaintResult` with `repaint_mode="rerender"`; `frame_codes`/`num_frames` are UNCHANGED
            from the sidecar (see `MiniMaxMusic3RepaintResult`'s own docstring).
        """
        from api.error_handlers import ValidationError
        from core.models.minimax_music3.defaults import CHUNK_FRAMES, CHUNK_HOP, CROP_LEFT_LATENT, CROP_RIGHT_LATENT

        pipeline, sidecar, original_wave, original_sr, model_hash = self._minimax_music3_load_repaint_source(
            reference_audio_path,
        )
        prompt, lyrics = self._minimax_music3_resolve_prompt_lyrics(sidecar, params)

        for required_key in ("repaint_start", "repaint_end", "num_inference_steps", "flow_guidance_scale"):
            if params.get(required_key) is None:
                raise ValidationError(
                    f"`{required_key}` is required",
                    detail=f"`{required_key}` must be provided explicitly; no default is available yet.",
                )
        repaint_start_sec = float(params["repaint_start"])
        repaint_end_sec = float(params["repaint_end"])
        num_inference_steps = int(params["num_inference_steps"])
        flow_guidance_scale = float(params["flow_guidance_scale"])
        if repaint_end_sec <= repaint_start_sec:
            raise ValidationError(
                "Invalid repaint range",
                detail=f"repaint_end ({repaint_end_sec}) must be greater than repaint_start ({repaint_start_sec}).",
            )

        chunk_starts = prepare_chunk_starts(sidecar.num_frames, CHUNK_FRAMES, CHUNK_HOP)
        num_chunks_total = len(chunk_starts)
        if num_chunks_total < 1:
            raise ValidationError(
                "MiniMax Music 3 'rerender' repaint: source song has no flow-matching chunks",
                detail="This should be unreachable for a non-empty sidecar.",
            )
        k1, frame_start = self._minimax_music3_snap_seconds_to_chunk_start(
            repaint_start_sec, sidecar.frame_rate, chunk_starts,
        )
        # k2 (exclusive end) is a chunk COUNT, not an index into chunk_starts -- candidates run from k1+1 (at
        # least one chunk re-rendered) through num_chunks_total (the whole rest of the song). Each candidate's
        # VALUE is `chunk_starts[k2]` (the frame at which chunk k2 -- the first chunk NOT re-rendered -- would
        # itself start), except for k2 == num_chunks_total, where `chunk_starts` has no such entry (there is no
        # chunk k2) and `sidecar.num_frames` (the song's own end) stands in for it -- so the same "nearest
        # chunk-window start, or the song's end" rule governs both ends symmetrically. This list is NEVER empty
        # (it always contains at least the appended `sidecar.num_frames`), so there is no "nothing after
        # repaint_start" case to refuse here -- k1 being the song's last chunk still leaves exactly one valid
        # candidate, k2 == num_chunks_total, re-rendering that one last chunk.
        end_candidates = chunk_starts[k1 + 1:] + [sidecar.num_frames]
        requested_end_frame = repaint_end_sec * sidecar.frame_rate
        best_end_offset = min(
            range(len(end_candidates)), key=lambda i: abs(end_candidates[i] - requested_end_frame),
        )
        k2 = k1 + 1 + best_end_offset  # exclusive chunk-count end

        is_global_first = (k1 == 0)
        is_global_last = (k2 == num_chunks_total)

        hop_length = int(pipeline.latent_hop_length)
        ce_config = pipeline.condition_encoder.config
        cumulative = compute_cumulative_samples(
            sidecar.num_frames, hop_length, CHUNK_FRAMES, CHUNK_HOP,
            CROP_LEFT_LATENT, CROP_RIGHT_LATENT,
            ce_config.input_sampling_rate, ce_config.input_hop_length,
            ce_config.output_sampling_rate, ce_config.output_hop_length,
        )
        # Same self-check as `_minimax_music3_repaint_regenerate`'s identical block -- see there for why this is
        # USER-REACHABLE (a "regenerate" result's own tail geometry) and therefore a ValidationError, not a
        # RuntimeError.
        if cumulative[-1] != int(original_wave.shape[-1]):
            raise ValidationError(
                "MiniMax Music 3 repaint: this file's chunk geometry does not match its own sidecar",
                detail=f"Computed chunk geometry predicts {cumulative[-1]} total samples for this source file, "
                       f"but it actually has {int(original_wave.shape[-1])}. This can happen for a file that was "
                       f"itself produced by a 'regenerate' repaint whose tail used non-standard (continuity-"
                       f"preserving) crop treatment -- see that method's docstring, 'Known limitation'. Extending "
                       f"this file (rather than repainting it) is unaffected and still works.",
            )
        start_sample = cumulative[k1]
        end_sample = cumulative[k2]

        # Frame-hidden recovery span: the union of chunks [k1, k2)'s own 200-frame windows -- see this method's
        # docstring "Window geometry".
        recover_frame_start = frame_start  # == chunk_starts[k1], already computed above
        recover_frame_end = min(chunk_starts[k2 - 1] + CHUNK_FRAMES, sidecar.num_frames)

        seed = params.get("seed", -1)
        if seed is None or int(seed) < 0:
            seed = random.randint(0, 2**32 - 1)
        seed = int(seed)
        device = self.device
        generator = torch.Generator(device=device).manual_seed(seed)

        num_recovered_frames = recover_frame_end - recover_frame_start
        num_rerendered_chunks = k2 - k1
        ar_budget, flow_budget = compute_progress_budget(
            num_recovered_frames, int(pipeline.num_codebooks), num_inference_steps, CHUNK_FRAMES, CHUNK_HOP,
        )

        def _combined_progress(step, total, stage) -> None:
            if progress_callback is None:
                return
            try:
                combined = combined_progress(stage, step, total, ar_budget, flow_budget)
                progress_callback(combined, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        comps = self.minimax_music3_components
        language_model = comps["language_model"]
        rvq_depth_decoder = comps["rvq_depth_decoder"]

        self._minimax_music3_move(("language_model", "rvq_depth_decoder"), device)
        lm_device = next(language_model.parameters()).device
        depth_device = next(rvq_depth_decoder.parameters()).device
        if lm_device != depth_device:
            raise RuntimeError(
                f"MiniMax Music 3's language model ({lm_device}) and RVQ depth decoder "
                f"({depth_device}) are not on the same device after staging; frame-hidden recovery "
                f"requires both co-resident."
            )
        try:
            try:
                text_ids = pipeline.encode_text(prompt, lyrics)
            except ValueError as exc:
                raise ValidationError(
                    "MiniMax Music 3 repaint: the sidecar's stored prompt/lyrics could not be tokenized",
                    detail=str(exc),
                )
            recovered_frame_hiddens = pipeline.recover_frame_hiddens(
                text_ids,
                sidecar.frame_codes,
                sidecar.prefix_codes,
                recover_frame_start,
                recover_frame_end,
                progress_callback=_combined_progress,
            )
        finally:
            self._minimax_music3_move(("language_model", "rvq_depth_decoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        recovered_frame_hiddens = recovered_frame_hiddens.detach().to("cpu")

        self._minimax_music3_move(("transformer", "condition_encoder"), device)
        try:
            latent_chunks = pipeline.denoise_chunks(
                recovered_frame_hiddens,
                num_inference_steps=num_inference_steps,
                flow_guidance_scale=flow_guidance_scale,
                generator=generator,
                progress_callback=_combined_progress,
            )
        finally:
            self._minimax_music3_move(("transformer", "condition_encoder"), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        if len(latent_chunks) != num_rerendered_chunks:
            raise RuntimeError(
                f"MiniMax Music 3 'rerender' repaint: denoise_chunks produced {len(latent_chunks)} latent "
                f"chunk(s) from {num_recovered_frames} recovered frames, expected {num_rerendered_chunks} -- the "
                f"recovered frame-hiddens span does not tile the way this mode's own geometry assumed."
            )

        self._minimax_music3_move(("vocoder",), device)
        try:
            new_middle_audio = pipeline.decode_range(
                latent_chunks, is_global_first=is_global_first, is_global_last=is_global_last, output_type="pt",
            )
        finally:
            self._minimax_music3_move(("vocoder",), "cpu", allow_partial_failure=True)
            self._minimax_music3_empty_cache()

        if progress_callback is not None:
            try:
                progress_callback(PROGRESS_TOTAL_UNITS, PROGRESS_TOTAL_UNITS)
            except Exception as exc:
                print(f"[MiniMaxMusic3] progress_callback raised: {exc!r}")

        if torch.isnan(new_middle_audio).any() or torch.isinf(new_middle_audio).any():
            raise RuntimeError(
                f"MiniMax Music 3 repaint produced NaN/Inf audio (shape={list(new_middle_audio.shape)})."
            )
        if new_middle_audio.numel() > 0 and new_middle_audio.abs().sum() == 0:
            raise RuntimeError("MiniMax Music 3 repaint produced all-silent (all-zero) audio for the re-rendered range.")

        new_middle_waveform = new_middle_audio[0].detach().to("cpu").float()
        sample_rate = int(pipeline.sampling_rate)

        if new_middle_waveform.shape[0] != original_wave.shape[0]:
            raise RuntimeError(
                f"MiniMax Music 3 repaint: channel count mismatch between the (already-validated) source file "
                f"({original_wave.shape[0]} channel(s)) and the re-rendered range "
                f"({new_middle_waveform.shape[0]} channel(s)) -- this should be unreachable."
            )

        full_waveform = self._minimax_music3_apply_rerender_waveform_splice(
            original_wave, new_middle_waveform, start_sample, end_sample, sample_rate,
        )

        return MiniMaxMusic3RepaintResult(
            waveform=full_waveform,
            sample_rate=sample_rate,
            actual_seed=seed,
            frame_codes=sidecar.frame_codes.to(torch.long),
            prefix_codes=sidecar.prefix_codes.to(torch.long),
            num_frames=sidecar.num_frames,
            frame_rate=float(sidecar.frame_rate),
            prompt=prompt,
            lyrics=lyrics,
            repaint_mode="rerender",
        )

    def _generate_aud2aud_minimax_music3(
        self,
        params: Dict[str, Any],
        reference_audio_path: str,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ) -> MiniMaxMusic3RepaintResult:
        """Dispatch `/generate/aud2aud` for a loaded MiniMax Music 3 model (design doc phase plan item 8).

        `params["mode"]` must be `"repaint"` -- ACE-Step's other `aud2aud` mode, `"cover"` (re-render the WHOLE
        reference under a new caption), is refused here for MiniMax Music 3: it would require turning arbitrary
        reference audio into semantic codes to condition the autoregressive stage, and the RVQ tokenizer's encoder
        is not published in this release (design doc "Capability verdict": "Cover / repaint of arbitrary user
        audio -- Not in phase 1; not proven"). Repaint only ever operates on a song THIS server generated,
        identified by its frame-code sidecar (`_minimax_music3_load_repaint_source`'s content-hash-matched lookup,
        same mechanism `/generate/outpaint/audio`'s extend already uses).

        `params["music3_repaint_mode"]` selects the sub-mode: `"regenerate"` (`_minimax_music3_repaint_
        regenerate`) or `"rerender"` (`_minimax_music3_repaint_rerender`) -- see this module's "Repaint" section
        docstring above both for the honest description of each, and for why mid-span infill with a preserved
        tail is not offered by either.
        """
        from api.error_handlers import ValidationError

        mode = params.get("mode")
        if mode != "repaint":
            raise ValidationError(
                f"MiniMax Music 3 does not support aud2aud mode {mode!r}",
                detail="Only mode='repaint' is available for MiniMax Music 3. Re-rendering arbitrary reference "
                       "audio under a new caption ('cover') would require turning that audio into the "
                       "autoregressive stage's semantic codes, and the RVQ tokenizer's encoder that would do that "
                       "is not published in this release -- see docs/guides/MINIMAX_MUSIC3_DESIGN.md, "
                       "\"Capability verdict\". Repaint only works on a song this server itself generated.",
            )

        repaint_mode = params.get("music3_repaint_mode")
        if repaint_mode == "regenerate":
            return self._minimax_music3_repaint_regenerate(params, reference_audio_path, progress_callback, step_callback)
        if repaint_mode == "rerender":
            return self._minimax_music3_repaint_rerender(params, reference_audio_path, progress_callback, step_callback)
        # Same causal-LM reason `_generate_audoutpaint_minimax_music3`'s "Placement" enumeration gives for
        # backward extension -- named explicitly here (rather than falling into the generic "invalid value" branch
        # below) for any request that names what it actually wants: mid-span infill with a preserved tail.
        if repaint_mode in ("infill", "inpaint", "mid_span", "preserve_tail"):
            raise ValidationError(
                f"MiniMax Music 3 does not support music3_repaint_mode {repaint_mode!r}",
                detail="Mid-song infill with a preserved tail is not offered: the autoregressive stage is a "
                       "causal language model, so changing codes in the middle of a song while an ORIGINAL, "
                       "different-content tail after them stays intact has no infilling contract. 'regenerate' "
                       "discards everything after its cut point instead of preserving a differing tail; "
                       "'rerender' never changes the codes at all, only their flow-stage rendering.",
            )
        raise ValidationError(
            f"Invalid music3_repaint_mode {repaint_mode!r}",
            detail="music3_repaint_mode must be 'regenerate' (AR-resume with a new tail from a point onward) or "
                   "'rerender' (keep the codes, redraw the flow stage over a window with a new seed).",
        )
