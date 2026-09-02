"""MiniMax-H3 video backend mixin for DiffusionPipelineManager.

Video-with-audio generation against the pruned MiniMax-H3 checkpoint, from a
prompt alone (``t2va``) or, on ``fl2va``, from keyframes placed at named frames
and/or an uploaded audio track the video is generated against (ia2v: the track's
rows are pinned clean for the whole clip and the sampler never writes them).
The loop itself lives in ``core.models.minimax_h3.h3_pipeline_ops`` (this repo
owns it — upstream ships a Modular pipeline only); this mixin is the staging
layer: it sequences the components on and off the GPU, resolves the request, and
returns the LTX-2.3 video tuple contract so the route, the mux and the gallery
need no new plumbing.

Output contract of ``_generate_txt2vid_minimax_h3`` /
``_generate_img2vid_minimax_h3`` (identical to LTX-2.3's):
    ``(frames, audio, audio_sample_rate, actual_seed)`` where
    ``frames`` is ``np.uint8 [T, H, W, 3]`` RGB,
    ``audio`` is a ``torch.FloatTensor [2, samples]`` on CPU (or None when
    ``audio_enable`` is false), ``audio_sample_rate`` 32000 (or None),
    ``actual_seed`` the concrete seed (a random draw resolved when seed < 0).

WHY THE STAGING IS STRICTLY SEQUENTIAL
--------------------------------------
Nothing here is co-resident by choice: the components are 21 GB (DiT, weight-
only FP8), 51.5 GB (Qwen3-VL-32B text encoder, bf16), 5.2 GB (video VAE, fp16)
and 0.6 GB (audio VAE, fp32) against a 48 GB card. So a generation runs three
phases and each one gives the GPU back before the next starts:

  1. **Text encode.** The encoder is never moved: its parameters are memory-
     mapped from the 48 GiB file and each decoder layer is materialised on the
     GPU for the length of one call (``h3_pipeline_ops.encode_prompt``). Moving
     the module instead costs +23 GB of resident RAM and 3.4x the wall time
     (K0.7). Only the layer-50 hidden state survives — kilobytes.
  1b. **Keyframe encode (fl2va only).** The video VAE encodes each keyframe as a
     single-frame clip and goes straight back to the CPU. It is the same
     autoencoder phase 3 uses, run before the DiT is staged rather than after.
     An ia2v track is encoded here too, by the audio VAE alone (0.6 GB), staged
     after the video VAE has gone back.
  2. **Denoise.** The DiT alone on the GPU, plus the packed sequence's
     activations.
  3. **Decode.** The DiT goes back to the CPU FIRST, then the video VAE decodes
     (its 36-layer ViT decoder is the heavy one), then the small audio VAE.

The DiT round trip costs real seconds per generation. It is not optimised away
by keeping it resident: this arch is excluded from ``keep_models_hot`` for the
same reason LTX-2.3 is, and there is no configuration in which the text encoder
and the DiT are both wanted at once.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os
import random
import time
import weakref

import numpy as np
import torch
from PIL import Image

from config.settings import settings
from core.inference.generation_timing import generation_timer
from core.inference.substep_progress import attach_block_substep_hooks


def _is_lora_target(module) -> bool:
    """Whether ``module`` is a Linear a MiniMax-H3 LoRA may wrap.

    Delegates to the ONE shared predicate,
    ``core.training.adapters.base_adapter.is_lora_wrappable_linear``, and exists
    as a named module-level function for the reason
    ``backend/tests/quantized_capability_parity_test.py`` states: the released
    MiniMax-H3 DiT ships weight-only FP8, so 300 of its Linears are
    ``Fp8Linear`` -- an ``nn.Module`` that is NOT an ``nn.Linear`` subclass. A
    target predicate written as ``isinstance(m, nn.Linear)`` would drop every one
    of them silently, and the run would "succeed" with a target count that looks
    like a narrower scope. That defect has been found on four architectures in
    this repo already.

    Declared in THIS phase because this is the phase in which ``minimax_h3``
    joins ``QUANTIZED_LINEAR_ARCHS`` and therefore comes under that parity test.
    The generation-side LoRA application and the training adapter that will call
    it land with the training phase; until then this is the arch's declared
    predicate and nothing else.
    """
    from core.training.adapters.base_adapter import is_lora_wrappable_linear

    return is_lora_wrappable_linear(module)


def select_minimax_h3_decode_vae(
    components: Dict[str, Any], latent_frames: int,
) -> Tuple[Any, str, bool]:
    """The VAE to decode ``latent_frames`` with: ``(module, component_name, used_fallback)``.

    At ``latent_frames == 1`` (the still-image case), prefers the optional
    community ``image_vae`` (measured 14-18 dB PSNR above the base video
    VAE's own T=1 decode) and falls back to the video VAE's T=1 branch when
    it is absent. Any other ``latent_frames`` always decodes through the
    video VAE with ``used_fallback=False``.
    """
    if latent_frames == 1:
        image_vae = components.get("image_vae")
        if image_vae is not None:
            return image_vae, "image_vae", False
        return components["vae"], "vae", True
    return components["vae"], "vae", False


def build_outpaint_references(
    head: np.ndarray, generated_frames: int, frame_rate: float, reference_images: Sequence[Image.Image],
) -> Tuple[Any, ...]:
    """The ref2va reference tuple for a video outpaint extend_forward request.

    A module-level pure function (no model components, no I/O) so the row
    order and the tail-truncation arithmetic are unit-testable without a
    loaded checkpoint. The LAST row is ALWAYS the source clip -- its own last
    ``min(len(head), generated_frames)`` frames (soundtrack excluded), at
    ``frame_rate`` rather than the source's own probed fps: outpaint's own
    convention already treats the preserved span as running at the declared
    output rate, so this keeps the two conventions consistent instead of
    letting ``normalize_reference_video`` resample a clip that outpaint
    itself never resamples. The rows BEFORE it are the optional image
    references, in request order.

    THE ROW ORDER IS THE FIX FOR THE ROTARY COLLISION BETWEEN AN IMAGE
    REFERENCE AND THE BOUNDARY ANCHOR, and this paragraph is the arithmetic
    for it. ``build_ref2va_packed_layout`` places an image reference at
    exactly 1.0 rotary unit before wherever its own block ends, and the
    boundary anchor sits at the rotary origin the *whole* reference loop
    leaves behind (``h3_pipeline_ops._anchor_rotary_time``, C5). The video
    reference is deliberately ``ROPE_FRAME_RESCALE``-contiguous with that
    anchor (measured in ``minimax_h3_outpaint_refs_design.md`` §1: the source
    clip's own last frame sits one pixel-frame step, 5/3 units, before the
    anchor -- a documented, accepted one-step hold). An image reference is
    NOT: it is only 1.0 unit from whatever follows it, closer than a single
    generated frame's own 5/3-unit spacing and well inside the anchor's own
    measured binding radius (A1: argmin within +/-2 frames, ~3.33 rotary
    units). Packed AFTER the video reference (the order this function used to
    return), an image reference lands 1.0 unit before the anchor -- inside
    that radius, competing with the anchor for the same instant, which is the
    mechanism behind the "reference is read as a keyframe of the join" defect
    (visual A/B: pre-paste boundary-anchor RMS 5.226 with no image reference
    vs 28.679 with one, at otherwise identical geometry). Packed BEFORE the
    video reference instead, an image reference sits near the text origin --
    ``video_span`` rotary units (the whole preserved-span's worth, ~207 at the
    125-frame/640x384 A/B geometry) away from the anchor, far outside the
    measured binding radius, while the video reference keeps its own
    documented one-step contiguity with the anchor unperturbed. Verified in
    ``minimax_h3_outpaint_reference_gate_test.py`` against
    ``h3_pipeline_ops.build_ref2va_packed_layout`` directly (a rotary-distance
    assertion, not a visual one -- the visual claim above is the recorded A/B
    result, not re-measured here). The *visual* effect of this reordering
    (whether the reference now reads as whole-span content conditioning
    rather than a keyframe) has not been re-confirmed on the GPU; that A/B
    rerun is the next step, not a claim this docstring makes.
    """
    from core.models.minimax_h3 import h3_references as refs

    source_ref_frames = head[-min(head.shape[0], generated_frames):]
    return tuple(
        refs.MiniMaxH3Reference(kind="image", image=image, label=f"reference {i + 1}")
        for i, image in enumerate(reference_images)
    ) + (
        refs.MiniMaxH3Reference(
            kind="video", frames=source_ref_frames, fps=frame_rate,
            video_canvas=(int(head.shape[1]), int(head.shape[2])),
            label="source clip (auto-referenced)"),
    )


class MiniMaxH3Mixin:
    """MiniMaxH3Mixin: joint video + audio generation with MiniMax-H3."""

    # ------------------------------------------------------------------
    # Component staging
    # ------------------------------------------------------------------

    def _minimax_h3_empty_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _minimax_h3_move(self, name: str, device):
        """Move one component of the H3 component dict.

        The TEXT ENCODER is refused rather than moved, and that refusal is the
        point: ``.to()`` on it — in either direction, with or without a dtype —
        detaches all 902 of its tensors from the file mapping and turns a
        memory-mapped 48 GiB module into an anonymous resident copy (73.08 GB
        peak RSS against 49.82 GB, MEASURED). Its GPU work is done a layer at a
        time by ``h3_pipeline_ops.encode_prompt`` instead.
        """
        if name == "text_encoder":
            raise RuntimeError(
                "The MiniMax-H3 text encoder must not be moved: its weights are memory-mapped "
                "from a 48 GiB file and `.to()` detaches every one of them. Use "
                "h3_pipeline_ops.encode_prompt, which streams one layer at a time.")
        components = getattr(self, "minimax_h3_components", None) or {}
        module = components.get(name)
        if module is None or not hasattr(module, "to"):
            return
        module.to(device)

    def _minimax_h3_assert_components_off_cuda(self, *names: str) -> None:
        """Refuse a phase transition while an inactive component is still on CUDA."""
        components = getattr(self, "minimax_h3_components", None) or {}
        offenders = []
        for name in names:
            module = components.get(name)
            if module is None:
                continue
            tensors = list(module.named_parameters()) + list(module.named_buffers())
            cuda_names = [tensor_name for tensor_name, tensor in tensors if tensor.device.type == "cuda"]
            if cuda_names:
                offenders.append(f"{name} ({', '.join(cuda_names[:3])})")
        if offenders:
            raise RuntimeError(
                "MiniMax-H3 phase transition found inactive CUDA component(s): "
                + "; ".join(offenders)
            )

    @staticmethod
    def _minimax_h3_fit_keyframe(image, width: int, height: int, anchor):
        """Put one keyframe onto the canvas, the way the released model does.

        THE ANCHORS ARE NOT TREATED THE SAME, and this asymmetry is the
        reference implementation's, in both independent ports of it:

        * the FRAME-0 keyframe is the geometry anchor: MiniMax derives the
          canvas from it when the request omits width/height, so when a canvas
          is given the frame is simply STRETCHED onto it (diffusers
          ``MiniMaxH3ResizeStep``: a plain PIL ``resize((w, h), LANCZOS)``; the
          other port spells it "no crop");
        * every OTHER keyframe is a FOLLOWER and is aspect-preserving
          centre-cover-cropped (the other port: "center"), because it
          has no say in the geometry and stretching it would hand the model a
          distorted anchor it is then pinned to for the whole loop.

        ``anchor`` is the placement, not the position in the list: ``"first"``
        or the integer ``0`` stretches, ``"last"`` and every other integer
        cover-crops. THAT IS A DELIBERATE CHANGE from "the packed-first keyframe
        stretches", made when placement shipped, and it is a change of rule
        rather than of behaviour:

        * with placement, a request can have no frame-0 anchor at all (a
          mid-only or last-only one). Under the old rule its single anchor would
          be stretched -- i.e. the model would be pinned mid-clip to a distorted
          frame -- purely because it was first in the list;
        * the one shipped path that feeds a lone non-zero anchor is video
          outpaint's ``extend_backward`` (``("last", head[0])``), and it is
          provably unaffected: ``_generate_vidoutpaint_minimax_h3`` passes
          frames straight out of ``center_crop_resize_frames(..., width,
          height)``, so every anchor it sends is ALREADY exactly
          ``(width, height)`` and returns at the identity check below before
          either branch runs. ``minimax_h3_layout_test`` asserts that, on the
          outpaint anchor shapes, both rules produce identical pixels.

        The arithmetic below is MiniMax's own, kept verbatim rather than
        expressed through ``VaeImageProcessor(resize_mode="crop")``: that helper
        sizes with floor division and centres with ``w // 2 - src_w // 2``,
        where this rounds and centres with ``(src_w - w) // 2``. The two agree
        on some aspect ratios and differ BY ONE PIXEL on others (106 of 218
        sampled, per the diffusers block's own note), which moves the
        conditioning latents off the reference implementation.
        """
        from core.models.minimax_h3.h3_pipeline_ops import is_frame_index_anchor

        image = image.convert("RGB")
        if image.size == (width, height):
            return image
        # ONE predicate for "this anchor is a frame index", shared with the
        # layout builder: an anchor type the layout PLACES at frame 0 must be
        # the one this stretches, or a `np.int64(0)` would silently be
        # cover-cropped instead.
        if anchor == "first" or (is_frame_index_anchor(anchor) and int(anchor) == 0):
            return image.resize((width, height), Image.LANCZOS)
        source_width, source_height = image.size
        scale = max(width / source_width, height / source_height)
        resized_size = (max(width, round(source_width * scale)),
                        max(height, round(source_height * scale)))
        left = max(0, (resized_size[0] - width) // 2)
        top = max(0, (resized_size[1] - height) // 2)
        resized = image.resize(resized_size, Image.LANCZOS)
        return resized.crop((left, top, left + width, top + height))

    def _minimax_h3_peak_vram(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 2 ** 30

    def _minimax_h3_vram_stats(self) -> Tuple[float, float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        scale = float(2 ** 30)
        return (
            torch.cuda.memory_allocated() / scale,
            torch.cuda.memory_reserved() / scale,
            torch.cuda.max_memory_allocated() / scale,
        )

    def _minimax_h3_reset_peak_vram(self) -> None:
        # Fold first: this reset exists for the per-phase logging below, and
        # without the fold it would also truncate the GENERATION-level peak the
        # route reports (generation_timing.peak_vram_dict).
        from core.inference.generation_timing import generation_timer

        generation_timer.note_peak_vram()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _minimax_h3_dump_outpaint_ref_debug(placement, *, frames_gen, head, tail=None):
        """Dump the PRE-PASTE generated boundary frame(s) alongside their
        preserved anchor, for A-V8 criterion 1 (boundary-anchor RMS).

        Env-gated (``MINIMAX_H3_OUTPAINT_REF_DEBUG_DUMP``) or sentinel-gated
        (``outputs/debug_latents/.enable_minimax_h3_outpaint_ref``), same
        convention as ``custom_sampling.py``'s ``OUTPAINT_DEBUG_LATENT_DUMP``.
        Zero-cost when disabled. Runs for every placement/variant (not just
        ref2va-with-references) so N, R and R+I share one instrument.

        Writes ``<run_dir>/frame0_generated_pre_paste.png`` +
        ``anchor_frame.png`` for extend_forward, the mirrored pair for
        extend_backward, or both pairs for a bridge -- read directly, no
        video decode needed, since the paste (``np.concatenate``) has not
        run yet when this is called.
        """
        import os

        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "outputs", "debug_latents",
        )
        enabled = bool(os.environ.get("MINIMAX_H3_OUTPAINT_REF_DEBUG_DUMP")) or os.path.exists(
            os.path.join(base, ".enable_minimax_h3_outpaint_ref"))
        if not enabled:
            return
        import time as _time
        run_dir = os.path.join(base, f"minimax_h3_outpaint_ref_{_time.time_ns()}_{placement}")
        os.makedirs(run_dir, exist_ok=True)
        pairs = []
        if placement in ("extend_forward", "bridge"):
            pairs.append(("frame0", frames_gen[0], head[-1]))
        if placement in ("extend_backward", "bridge"):
            anchor = tail[0] if placement == "bridge" else head[0]
            pairs.append(("frame_last", frames_gen[-1], anchor))
        for name, generated, anchor in pairs:
            Image.fromarray(generated).save(os.path.join(run_dir, f"{name}_generated_pre_paste.png"))
            Image.fromarray(anchor).save(os.path.join(run_dir, f"{name}_anchor.png"))
        print(f"[MiniMax-H3] vid_outpaint: pre-paste debug dump -> {run_dir}")

    @staticmethod
    def _minimax_h3_dump_residual_probe(records, *, prompt, num_frames, seed, num_inference_steps):
        """Dump Experiment B's per-block residual-contribution recording as JSON.

        Unlike ``_minimax_h3_dump_outpaint_ref_debug``'s own env-var/sentinel
        gate, there is no separate dump toggle here: the request field
        (``_minimax_h3_debug_probe_residuals``, threaded through
        ``_ensure_minimax_h3_swap_and_offload``) IS the gate -- a run that
        asked for the recording gets it written unconditionally, no-op when
        ``records`` is empty or ``None`` (probing was never requested).
        Called from the denoise ``finally``, so a failed generation still
        dumps whatever steps it reached before raising -- best-effort, not a
        guaranteed on-failure dump: an exception raised by this function
        itself is left to propagate rather than swallowed, since a broken
        dump on a research-only path is something the caller should see.
        NOTE: because this runs inside a ``finally``, an exception raised
        here REPLACES (not appends to) an in-flight generation exception --
        Python's own ``finally`` semantics, not a bug in this function -- so
        a disk-full/permissions error while dumping would surface instead of
        whatever actually broke the generation. Realistically rare (a few
        hundred KB write) and this repo has no existing convention for a
        finally-safe secondary write, so accepted rather than engineered
        around here.
        """
        if not records:
            return
        import json
        import os
        import time as _time

        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "outputs", "debug_latents",
        )
        os.makedirs(base, exist_ok=True)
        out_path = os.path.join(base, f"minimax_h3_residual_probe_{_time.time_ns()}.json")
        payload = {
            "prompt": prompt,
            "num_frames": num_frames,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "records": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"[MiniMax-H3] Residual probe (debug/research): {len(records)} record(s) -> {out_path}")

    # ------------------------------------------------------------------
    # Attention backend
    # ------------------------------------------------------------------

    def _minimax_h3_apply_attention_backend(self, transformer, params: Dict[str, Any]) -> str:
        """Stamp the inference attention backend on the transformer. Returns it.

        The vendored ``MiniMaxH3Transformer3DModel`` propagates ``_attn_backend``
        to every ``MiniMaxH3Attention`` once per forward
        (``_stamp_attention_backend``), so setting it on the model is enough for
        the 50 blocks AND the 2-layer token refiner.

        MiniMax-H3 attention is unmasked full self-attention over one packed
        document with ``head_dim = 128`` and equal q/kv head counts, so no
        conduit guard fires: sage and flash both run here rather than being
        downgraded. Measure, do not assume -- the conduit logs the backend it
        actually used.
        """
        from core.attention import normalize_backend

        requested = params.get("attention_type", settings.attention_type)
        backend = normalize_backend(requested)
        inner = getattr(transformer, "transformer", transformer)
        inner._attn_backend = backend
        print(f"[MiniMax-H3] Attention backend: {backend} (from attention_type={requested!r})")
        return backend

    # ------------------------------------------------------------------
    # Block swap (the block-loop wrapper)
    # ------------------------------------------------------------------

    @staticmethod
    def _minimax_h3_build_residual_recorder():
        """Build a residual-probe recorder plus the list it accumulates into.

        See ``MiniMaxH3BlockLoopWrapper.attach_residual_probe`` (Experiment B,
        debug/research only). Returns ``(records, recorder)``; ``records`` is
        the list the caller dumps to disk once the generation completes.
        """
        records: List[Dict[str, Any]] = []

        def _recorder(block_idx: int, step_idx: int, rel_residual: float) -> None:
            records.append({
                "block_idx": block_idx,
                "step_idx": step_idx,
                "rel_residual": rel_residual,
            })

        return records, _recorder

    def _ensure_minimax_h3_swap_and_offload(
        self, params: Dict[str, Any], device: torch.device,
    ):
        """Stage the DiT onto ``device`` for the denoise loop. Returns the callable.

        Returns ``(module, offloader, probe_records)``: the object the sampler
        calls (the raw transformer, or a ``MiniMaxH3BlockLoopWrapper`` when
        block swap/FBCache/block-skip/residual-probe needs the re-owned block
        loop), the block offloader to tear down afterwards (``None`` when
        there is none), and the residual-probe recording list (``None`` when
        ``params["_minimax_h3_debug_probe_residuals"]`` is not set -- see
        ``_minimax_h3_build_residual_recorder``). A truthy
        ``params["_minimax_h3_debug_skip_blocks"]`` (Phase 1c ablation knob) forces
        the wrapper even when block swap and FBCache are both off, so the raw
        transformer is never returned while a skip set is attached; the same is
        true of a truthy ``params["_minimax_h3_debug_probe_residuals"]``.

        TWO STATES:

        * ``blocks_to_swap == 0`` (the default): the transformer is moved to the
          device whole. The sampler calls it directly unless opt-in FBCache needs
          the wrapper's first-block decision point; the disabled default remains
          byte-identical to the raw path.
        * ``blocks_to_swap > 0``: the NON-block modules (the three input
          projections, the RoPE buffer, the output norm and the two heads) are
          moved to the device, then ``TransformerBlockOffloader`` places the
          block stack -- the first ``50 - blocks_to_swap`` blocks resident and
          the rest weight-on-CPU. The whole-model ``.to(device)`` is
          deliberately NOT used here: it would put all 21 GB on the card and
          only then take some of it back off, which is the opposite of what
          the request asked for.

          THE TOKEN REFINER IS THE ONE EXCEPTION and is deliberately left off
          this loop (measured 1.4356 GiB, the only bf16 -- i.e. unquantized --
          module left in the w4a8 checkpoint, 12.2% of the file). It is left
          CPU-resident here and staged onto ``device`` for the length of its
          own call by ``MiniMaxH3BlockLoopWrapper._custom_forward`` instead,
          under the SAME ``blocks_to_swap > 0`` condition that gates block
          swap itself (the wrapper's local ``swap_on``), because it has
          exactly one call site, runs once per forward on ~500 text rows
          (negligible compute), and is otherwise unconditionally resident at
          every ``blocks_to_swap`` setting even though the block stack -- 9.12
          of the 11.82 GiB the DiT holds outside the block loop -- IS
          swappable. Not fed to ``TransformerBlockOffloader`` itself: that
          offloader is scoped to ``transformer.transformer_blocks`` and staged
          against wait-ahead prefetch depth, machinery this module does not
          need for a single call per forward.

        WHY THE OFFLOADER IS PER-GENERATION rather than persistent wrapper state
        (LTX-2.3's shape, and the source of the stale-offloader defect the
        parity suite exists for): this architecture's DiT is moved off the GPU
        at the end of EVERY generation -- the video VAE's 36-layer ViT decoder
        and the DiT do not fit together -- so an offloader that survived the
        generation would be holding device buffers for a model that is no longer
        on the device. Building and cleaning it up inside one generation is not
        a simplification, it is the only lifetime that matches the staging.

        ``h2d_only`` is deliberately off. It coalesces each swappable block's
        Linear weights into ONE flat buffer, which requires a single dtype
        across them; a MiniMax-H3 block holds ``Fp8Linear`` weights
        (``float8_e4m3fn``) next to the float32 ``adaln_proj.linear``, so the
        offloader would detect the mixed dtype and fall back to the standard
        swap anyway. Asking for the standard swap directly keeps the module's
        own weights the owner of their storage, which is what makes the
        end-of-generation ``.to("cpu")`` restore the model correctly.
        """
        from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper

        components = self.minimax_h3_components
        transformer = components["transformer"]
        # Defensive: a previous generation that was killed between the wrap and
        # its `finally` would leave a wrapper here.
        if isinstance(transformer, MiniMaxH3BlockLoopWrapper):
            transformer = transformer.transformer
            components["transformer"] = transformer

        # Phase 1c debug/ablation knob (see Txt2VidRequest.minimax_h3_debug_skip_blocks
        # and MiniMaxH3BlockLoopWrapper.attach_block_skip). Internal only -- the
        # request-level gate in routes.py already refused this on a non-minimax_h3
        # arch, so any truthy value reaching here belongs to this generation.
        skip_blocks = params.get("_minimax_h3_debug_skip_blocks")

        # Experiment B instrumentation knob (see
        # Txt2VidRequest.minimax_h3_debug_probe_residuals and
        # MiniMaxH3BlockLoopWrapper.attach_residual_probe). Same internal-only
        # contract as skip_blocks above.
        probe_on = bool(params.get("_minimax_h3_debug_probe_residuals"))

        # SushiUI addition: opt-in, not bit-exact -- see `adaln_chunking.py`'s "Head fusion" note.
        # Set on the INNER transformer (not the wrapper) so both call sites -- the stock forward's
        # (fast path, wrapper delegates to it) and `MiniMaxH3BlockLoopWrapper._custom_forward`'s
        # (block swap / FBCache) -- read the same flag off the same object.
        transformer.fuse_output_proj = bool(params.get("fuse_output_proj", False))

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        from core.inference.fbcache import fbcache_active
        fbcache_on = fbcache_active(params) and not params.get("spectrum_enable", False)
        num_blocks = len(transformer.transformer_blocks)
        if skip_blocks:
            # Validated HERE, before the device move below: raising inside
            # `attach_block_skip` after the 21 GB DiT is already on the GPU
            # would leave it resident with no denoise loop left to stage it
            # back off (nothing downstream of this function catches that raise).
            bad = sorted(i for i in skip_blocks if not (0 <= i < num_blocks))
            if bad:
                raise ValueError(f"minimax_h3_debug_skip_blocks out of range [0, {num_blocks}): {bad}")
        if blocks_to_swap >= num_blocks:
            print(f"[MiniMax-H3] blocks_to_swap={blocks_to_swap} >= {num_blocks} blocks; "
                  f"clamping to {num_blocks - 1} (at least one block must stay resident)")
            blocks_to_swap = num_blocks - 1

        if blocks_to_swap <= 0:
            self._minimax_h3_move("transformer", device)
            if fbcache_on or skip_blocks or probe_on:
                wrapper = MiniMaxH3BlockLoopWrapper(transformer)
                components["transformer"] = wrapper
                if fbcache_on:
                    print("[FBCache] MiniMax-H3 wrapper armed (cache is opt-in)")
                if skip_blocks:
                    wrapper.attach_block_skip(skip_blocks)
                    print(f"[MiniMax-H3] Block skip (debug/ablation): {sorted(skip_blocks)}")
                probe_records = None
                if probe_on:
                    probe_records, recorder = self._minimax_h3_build_residual_recorder()
                    wrapper.attach_residual_probe(recorder)
                    print("[MiniMax-H3] Residual probe (debug/research) armed")
                return wrapper, None, probe_records
            return transformer, None, None

        from core.memory_management import TransformerBlockOffloader

        for name, child in transformer.named_children():
            if name in ("transformer_blocks", "token_refiner"):
                continue
            child.to(device)
        # Buffers registered directly on the model (the AdaLN curve table) are
        # not children and would otherwise stay on the CPU.
        for _name, buf in transformer.named_buffers(recurse=False):
            buf.data = buf.data.to(device)

        offloader = TransformerBlockOffloader(
            blocks=transformer.transformer_blocks,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=transformer.dtype,
            use_pinned_memory=False,
            transformer=transformer,
            supports_backward=False,
            h2d_only=False,
        )
        offloader.prepare_block_devices_before_forward()

        wrapper = MiniMaxH3BlockLoopWrapper(transformer, block_offloader=offloader)
        components["transformer"] = wrapper
        print(f"[MiniMax-H3] Block Swap enabled: {blocks_to_swap} of {num_blocks} blocks "
              f"swapped (MiniMaxH3BlockLoopWrapper active)")
        if skip_blocks:
            wrapper.attach_block_skip(skip_blocks)
            print(f"[MiniMax-H3] Block skip (debug/ablation): {sorted(skip_blocks)} "
                  f"(with Block Swap)")
        probe_records = None
        if probe_on:
            probe_records, recorder = self._minimax_h3_build_residual_recorder()
            wrapper.attach_residual_probe(recorder)
            print("[MiniMax-H3] Residual probe (debug/research) armed (with Block Swap)")
        return wrapper, offloader, probe_records

    def _unstage_minimax_h3_transformer(self, offloader) -> None:
        """Tear the block-loop wrapper down and put the DiT back on the CPU.

        Runs in the generation's ``finally``, so a cancelled or failed denoise
        cannot leave ``minimax_h3_components["transformer"]`` holding a wrapper
        whose offloader references device buffers -- the stale-offloader state
        that ``quantized_capability_parity_test`` exists over on the
        architectures whose wrapper IS persistent.
        """
        from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper

        components = self.minimax_h3_components or {}
        current = components.get("transformer")
        if isinstance(current, MiniMaxH3BlockLoopWrapper):
            components["transformer"] = current.transformer
        if offloader is not None:
            try:
                offloader.cleanup()
            except Exception as exc:  # teardown must never take a generation down
                print(f"[MiniMax-H3] block offloader cleanup raised: {exc}")
        # Whole-model move: with block swap the swappable blocks' weights are on
        # the CPU already and this is a no-op for them; the resident blocks and
        # the auxiliary modules come back.
        self._minimax_h3_move("transformer", "cpu")

    # ------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------

    def _minimax_h3_lora_state(self, transformer):
        """The ``(originals, wrapped_keys)`` maps for THIS transformer.

        Reset when the model was reloaded: the maps hold the OLD transformer's
        Linears, ``apply_lora_group`` keeps them (setdefault) and
        ``restore_originals`` would then splice them into the new transformer.
        Keyed by weakref rather than id() because a freed object's id is reusable.
        """
        ref = getattr(self, "_minimax_h3_lora_transformer_ref", None)
        if ref is None or ref() is not transformer:
            self._minimax_h3_lora_original_modules: Dict[str, torch.nn.Module] = {}
            self._minimax_h3_lora_wrapped_keys: set = set()
            self._minimax_h3_lora_transformer_ref = weakref.ref(transformer)
        return self._minimax_h3_lora_original_modules, self._minimax_h3_lora_wrapped_keys

    def _load_lora_minimax_h3(self, lora_configs: list, params: Dict[str, Any]) -> int:
        """Wrap target Linear modules of the MiniMax-H3 DiT with LoRA adapters.

        Applied PER GENERATION, immediately before
        ``_ensure_minimax_h3_swap_and_offload`` -- never at model load time.
        ``TransformerBlockOffloader``/``swap_linears_to_w4a8`` raise on a
        non-``nn.Linear`` child, so a load-time LoRA application (which wraps a
        Linear in a ``MiniMaxH3LoRALinearLayer``, an ``nn.Module`` that is not
        an ``nn.Linear`` subclass) would break the block-swap staging path.
        The transformer must already be the raw (un-wrapped-by-block-loop)
        module -- call this before ``_ensure_minimax_h3_swap_and_offload``
        replaces ``minimax_h3_components["transformer"]`` with a
        ``MiniMaxH3BlockLoopWrapper``.

        Several LoRAs may be selected, but they must target DISJOINT modules
        (see ``apply_lora_group``): an overlap warns, a full shadow refuses.
        Any refusal raises here, before the denoise loop, so a LoRA that could
        not be applied never returns a successful generation.
        """
        from core.models.minimax_h3.minimax_h3_lora import (
            normalise_lora_state_dict, apply_lora_group,
            check_variant_compatibility, detect_rank_variation,
        )
        from core.extensions.lora_manager import lora_manager

        # Unconditional, and BEFORE the empty-config exit: this is what re-keys
        # the state to the live transformer, and a restore that failed in an
        # earlier request must not leak its wrappers into this generation.
        self._unload_lora_minimax_h3()

        if not lora_configs:
            return 0
        components = getattr(self, "minimax_h3_components", None)
        if not components:
            print("[MiniMax-H3 LoRA] WARNING: components not loaded")
            return 0

        transformer = components["transformer"]
        # Defensive: a previous generation killed between wrap and its
        # `finally` could have left the block-loop wrapper installed.
        from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
        if isinstance(transformer, MiniMaxH3BlockLoopWrapper):
            transformer = transformer.transformer

        originals, wrapped_keys = self._minimax_h3_lora_state(transformer)

        try:
            from api.generation_status import add_warning
        except Exception:  # pragma: no cover - status module always present in-process
            add_warning = None

        def warn(message: str, code: str) -> None:
            print(f"[MiniMax-H3 LoRA] WARNING: {message}")
            if add_warning is not None:
                try:
                    add_warning(message, code=code)
                except Exception:
                    pass

        current_variant = components.get("variant")
        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        from core.inference.fbcache import fbcache_active
        distillation_cache_active = fbcache_active(params) or bool(params.get("spectrum_enable", False))
        num_inference_steps = int(params.get("num_inference_steps", 0) or 0)

        total_applied = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            # Warnings ride into the PNG metadata chunk, so never an absolute path.
            lora_file = os.path.basename(str(lora_path))
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                message = (
                    f"LoRA '{lora_file}' was requested but no such file exists in the "
                    f"registered LoRA directories -- refusing to generate without it."
                )
                warn(message, "lora_not_found")
                raise FileNotFoundError(message)
            raw, metadata = self._minimax_h3_read_lora_file(str(resolved), lora_file, warn)
            try:
                check_variant_compatibility(metadata, lora_file, current_variant, warn)
            except ValueError as exc:
                # Declared-variant mismatch, the one refusal the guard raises
                # itself. Surfaced through warn() too, so it reaches the
                # generation's warnings[] like every other refusal here.
                warn(str(exc), "minimax_h3_lora_variant_mismatch")
                raise
            try:
                targets = normalise_lora_state_dict(raw, metadata)
                print(f"[MiniMax-H3 LoRA] {i + 1}/{len(lora_configs)}: {lora_file} "
                      f"keys={len(raw)} matched_targets={len(targets)} strength={strength}")

                if blocks_to_swap > 0:
                    rank_variation = detect_rank_variation(targets)
                    varying_leaves = [leaf for leaf, varies in rank_variation.items() if varies]
                    if varying_leaves:
                        warn(
                            f"LoRA '{lora_file}' has a rank that varies across blocks for "
                            f"{varying_leaves} while blocks_to_swap={blocks_to_swap} is active. "
                            f"TransformerBlockOffloader pairs an incoming/outgoing block's Linear "
                            f"weights by name+shape+dtype; a rank-varying LoRA's lora_down/lora_up "
                            f"Linears fail to pair (different shapes) and the offloader falls back "
                            f"to moving the incoming block's LoRA weights onto the GPU WITHOUT "
                            f"evicting the outgoing block's -- these accumulate on the GPU over the "
                            f"denoise loop instead of being swapped.",
                            "minimax_h3_lora_rank_varies_with_block_swap",
                        )

                student_steps = metadata.get("student_steps")
                if student_steps is not None and num_inference_steps > 0:
                    try:
                        student_steps_int = int(float(student_steps))
                    except (TypeError, ValueError):
                        student_steps_int = None
                    if student_steps_int is not None:
                        expected_steps = student_steps_int + 1
                        if num_inference_steps != expected_steps:
                            warn(
                                f"LoRA '{lora_file}' is a {student_steps_int}-step distillation "
                                f"checkpoint (num_inference_steps counts sigma grid points "
                                f"including the terminal 0, so {expected_steps} is the matching "
                                f"value); this generation is requesting "
                                f"num_inference_steps={num_inference_steps}.",
                                "minimax_h3_lora_step_count_mismatch",
                            )
                        if distillation_cache_active:
                            warn(
                                f"LoRA '{lora_file}' is a {student_steps_int}-step distillation "
                                f"checkpoint and FBCache/Spectrum forecasting is also active. With "
                                f"only ~{expected_steps} model evaluations, warmup alone can consume "
                                f"the whole trajectory and leave nothing for the cache/forecast to "
                                f"act on.",
                                "minimax_h3_lora_distillation_with_cache",
                            )

                shadowed: list = []
                applied, missing = apply_lora_group(
                    transformer, targets, strength, originals, wrapped_keys, shadowed,
                )
            except Exception as exc:
                print(f"[MiniMax-H3 LoRA] ERROR loading {lora_file}: {exc}")
                import traceback
                traceback.print_exc()
                # Type + basename only: this rides into the PNG text chunk and the API
                # response, and an OSError's str() carries the absolute resolved path.
                message = (f"MiniMax-H3 LoRA '{lora_file}' could not be applied "
                           f"({type(exc).__name__}); see the server log for details")
                warn(message, "lora_load_failed")
                raise RuntimeError(message) from exc

            print(f"[MiniMax-H3 LoRA]   wrapped {applied} module(s)")
            if applied == 0:
                if shadowed:
                    message = (
                        f"LoRA '{lora_file}': every one of its {len(shadowed)} target modules is "
                        f"already wrapped by an earlier LoRA in this request. MiniMax-H3 applies "
                        f"one LoRA per target; select a single MiniMax-H3 LoRA."
                    )
                    code = "lora_stacking_unsupported"
                else:
                    message = (
                        f"LoRA '{lora_file}': 0 of {len(targets)} target(s) applied to the loaded "
                        f"MiniMax-H3 transformer ({len(missing)} unresolved against the live module "
                        f"tree) -- unrecognized key format or a different model. Expected either "
                        f"'diffusion_model.*.lora_A.weight' (ComfyUI) or "
                        f"'lora_unet_transformer_blocks_<N>_*.lora_down.weight' (SushiUI-trained). "
                        f"Sample keys in file: {list(raw.keys())[:5]}"
                    )
                    code = "lora_incompatible"
                warn(message, code)
                raise RuntimeError(message)

            if missing or shadowed:
                warn(
                    f"LoRA '{lora_file}': applied {applied} of {len(targets)} target(s) "
                    f"({len(missing)} unresolved against the loaded MiniMax-H3 transformer, first "
                    f"few: {missing[:5]}; {len(shadowed)} already wrapped by an earlier LoRA).",
                    "lora_partial",
                )
            total_applied += applied

        return total_applied

    @staticmethod
    def _minimax_h3_read_lora_file(resolved: str, lora_file: str, warn):
        """Read a LoRA safetensors file, or refuse the generation."""
        from core.models.minimax_h3.minimax_h3_lora import load_lora_safetensors

        try:
            return load_lora_safetensors(resolved)
        except Exception as exc:
            print(f"[MiniMax-H3 LoRA] ERROR reading {lora_file}: {exc}")
            # Type + basename only: this rides into the PNG text chunk and the API
            # response, and an OSError's str() carries the absolute resolved path.
            message = (f"MiniMax-H3 LoRA '{lora_file}' could not be applied "
                       f"({type(exc).__name__}); see the server log for details")
            warn(message, "lora_load_failed")
            raise RuntimeError(message) from exc

    def _unload_lora_minimax_h3(self) -> int:
        """Restore every MiniMax-H3 transformer Linear to its pre-LoRA original.

        Drops the original-module map with the wrappers: it is per-generation
        state. Which transformer that map belongs to is decided by
        ``_minimax_h3_lora_state``, which this must consult rather than
        restoring straight into ``components["transformer"]``: a model swap
        with wrappers still live would otherwise install model A's Linears
        into model B.
        """
        from core.models.minimax_h3.minimax_h3_lora import restore_originals

        components = getattr(self, "minimax_h3_components", None)
        transformer = components.get("transformer") if components else None
        if transformer is None:
            # Model unloaded: drop the maps so a later load cannot inherit them.
            self._minimax_h3_lora_original_modules = {}
            self._minimax_h3_lora_wrapped_keys = set()
            self._minimax_h3_lora_transformer_ref = None
            return 0
        from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
        if isinstance(transformer, MiniMaxH3BlockLoopWrapper):
            transformer = transformer.transformer
        originals, wrapped_keys = self._minimax_h3_lora_state(transformer)
        if not wrapped_keys:
            originals.clear()
            return 0
        restored = restore_originals(transformer, originals, wrapped_keys)
        originals.clear()
        print(f"[MiniMax-H3 LoRA] Unloaded {restored} LoRA wrapper(s)")
        return restored

    def _minimax_h3_report_te_provenance(
        self, components: Dict[str, Any], params: Dict[str, Any],
    ) -> None:
        """``params`` provenance + the TE-substitution warning, per GENERATION.

        Split out of ``_minimax_h3_project_prompt_embeds`` so a
        ``prompt_cache`` hit -- which skips that function's encode-dependent
        projection math entirely -- can still call just this half. These
        fields and the warning describe what model is generating THIS
        request, not what work happened to produce the prompt embedding, so
        they must fire on every generation, cached or not.

        A released encoder carries no projection and this returns after
        setting ``text_encoder_file`` alone -- that path emits no warning and
        runs no extra arithmetic.
        """
        params["text_encoder_file"] = os.path.basename(
            str(components.get("text_encoder_path") or ""))
        projection = components.get("te_projection")
        if not projection:
            return

        from core.models.minimax_h3.te_projection import (
            TE_SUBSTITUTION_WARNING_CODE, describe_te_substitution,
        )

        te_path = str(components.get("text_encoder_path") or "")
        projection_path = str(projection.get("path") or "")
        params["clip_projection_file"] = os.path.basename(projection_path)

        message = describe_te_substitution(te_path, projection_path)
        print(f"[MiniMax-H3] {message}")
        from api.generation_status import add_warning
        add_warning(message, code=TE_SUBSTITUTION_WARNING_CODE)

    def _minimax_h3_project_prompt_embeds(
        self,
        prompt_embeds_cpu: torch.Tensor,
        components: Dict[str, Any],
        params: Dict[str, Any],
        *,
        device,
    ) -> torch.Tensor:
        """Project a substituted encoder's hidden state, and report that it was.

        A released encoder carries no projection and this returns its argument
        untouched -- that path emits no warning and runs no extra arithmetic.
        """
        self._minimax_h3_report_te_provenance(components, params)
        projection = components.get("te_projection")
        if not projection:
            return prompt_embeds_cpu

        from core.models.minimax_h3 import h3_pipeline_ops as ops

        te_path = str(components.get("text_encoder_path") or "")
        projection_path = str(projection.get("path") or "")
        # A component switch installs encoder and projection together, so this
        # fires only if some other path desynchronises them.
        # `apply_te_projection`'s own d_in guard would catch it too, but it
        # cannot name the cause.
        d_in = int(projection["spec"]["d_in"])
        if prompt_embeds_cpu.shape[-1] != d_in:
            raise RuntimeError(
                f"MiniMax-H3's loaded text encoder ({os.path.basename(te_path)}) produced "
                f"{prompt_embeds_cpu.shape[-1]}-wide hidden states but the paired projection "
                f"{os.path.basename(projection_path)} takes d_in={d_in}. The two no longer belong "
                f"to each other -- reload the model to re-resolve the pairing.")
        projected = ops.project_prompt_embeds(
            prompt_embeds_cpu, projection,
            text_dim=int(components["transformer_config"]["text_dim"]), device=device,
        )
        return projected

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_txt2vid_minimax_h3(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Text-to-video (+ joint audio) generation with MiniMax-H3 (``t2va``).

        Returns ``(frames, audio, audio_sample_rate, actual_seed)`` — see the
        module docstring.
        """
        return self._generate_minimax_h3(
            params, keyframes=(), label="txt2vid",
            progress_callback=progress_callback, step_callback=step_callback)

    def _generate_img2vid_minimax_h3(
        self,
        params: Dict[str, Any],
        input_image,
        last_frame_image=None,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        keyframes: Optional[Sequence[Tuple[Any, Any]]] = None,
        input_audio=None,
    ):
        """Keyframe-conditioned generation with MiniMax-H3 (``fl2va``).

        Each keyframe is an ordinary PIL image and becomes one single-frame
        visual condition, anchored at its own frame's rotary position
        (``num_text_tokens + (5/3)*f``, exact for every pixel frame).

        ``keyframes`` — the RESOLVED placement plan, ``(anchor, image)`` in
        packed order, where anchor is ``"first"``, ``"last"`` or an integer
        pixel frame. The route builds it with
        ``generation_utils.plan_keyframe_placements``, which is where the
        ``-1`` sentinel is resolved against the snapped clip length and the two
        ends are mapped onto the string anchors.

        ``input_image`` / ``last_frame_image`` — the pre-placement call shape,
        used when ``keyframes`` is None (internal callers, and any caller that
        has not been through the route). Equivalent to
        ``[("first", input_image)]`` plus ``("last", last_frame_image)``.

        ``input_audio`` — an ia2v track, already 32 kHz stereo at the exact
        length this clip needs (``h3_references.prepare_pinned_audio``). Its
        rows are pinned clean for the whole clip and the video is generated
        against them. WITH ONE SUPPLIED, KEYFRAMES ARE OPTIONAL: an imageless
        request is audio + prompt conditioning, which is measured working, and
        it is the one shape of this route that carries no image.

        Same return contract as ``_generate_txt2vid_minimax_h3``.
        """
        if keyframes is None:
            if input_image is None and input_audio is None:
                raise RuntimeError("img2vid requires an input image for the first-frame keyframe")
            keyframes = [] if input_image is None else [("first", input_image)]
            if last_frame_image is not None:
                keyframes.append(("last", last_frame_image))
        if not keyframes and input_audio is None:
            raise RuntimeError(
                "img2vid requires at least one keyframe image, or an input audio track to "
                "condition on")
        return self._generate_minimax_h3(
            params, keyframes=tuple(keyframes), label="img2vid", input_audio=input_audio,
            progress_callback=progress_callback, step_callback=step_callback)

    def _generate_ref2vid_minimax_h3(
        self,
        params: Dict[str, Any],
        references,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        keyframes: Sequence[Tuple[Any, Any]] = (),
    ):
        """Omni-reference generation with MiniMax-H3 (``ref2va``).

        ``references`` is a sequence of
        ``core.models.minimax_h3.h3_references.MiniMaxH3Reference`` **in the
        order the model should read them**: that order labels them in the prompt
        presentation (``<Picture i>`` / ``<Audio j>`` / ``<Video k>``) and lays
        them out on the packed sequence's shared rotary clock, so a different
        order is a different request. Nothing here sorts or regroups them.

        THIS NEEDS THE ``ref2va`` TRANSFORMER, and refuses rather than running on
        the other one. The two released single files are structurally identical
        (same config, byte-identical size, no distinguishing key), so a mismatch
        cannot be detected from the weights and would simply produce a bad video:
        reference conditioning is a trained behaviour of ``transformer_ref``, and
        the ``fl2va`` partition was never trained to read reference rows.

        ``keyframes`` (C5) is optional: the same ``(anchor, PIL.Image)`` plan
        ``_generate_img2vid_minimax_h3`` takes, laid out AFTER the reference
        blocks by ``h3_pipeline_ops.build_ref2va_packed_layout``. Empty by
        default, which is the pre-C5 ref2vid request.

        Same return contract as ``_generate_txt2vid_minimax_h3``.
        """
        if not references:
            raise RuntimeError("ref2vid requires at least one reference")
        return self._generate_minimax_h3(
            params, references=tuple(references), keyframes=tuple(keyframes or ()),
            label="ref2vid", progress_callback=progress_callback, step_callback=step_callback)

    def _generate_vidoutpaint_minimax_h3(
        self,
        params: Dict[str, Any],
        video_frames: np.ndarray,
        fps: float,
        input_audio: Optional[bytes],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        *,
        bridge_frames: Optional[np.ndarray] = None,
        bridge_fps: Optional[float] = None,
        bridge_audio: Optional[bytes] = None,
        reference_images: Sequence[Image.Image] = (),
    ):
        """Temporal outpaint with MiniMax-H3: extend a clip, or bridge two.

        WHAT THIS PATH CONDITIONS ON, because the shape of this function is
        entirely that fact: it hands the model the FIRST and/or the LAST frame
        of the span it asks for, as fl2va keyframe anchors, and concatenates
        the preserved clip(s) around the result. It therefore serves exactly
        the three placements those two anchors describe -- extend-forward,
        extend-backward, bridge -- and refuses everything else
        (``generation_utils.plan_video_outpaint_placement``) rather than
        approximating it with a nearby placement.

        That is a property of THIS path, not of the architecture, and the
        distinction is worth stating because this docstring used to blur it:

        * The packed sequence's temporal axis is pixel-frame time
          (``t(f) = num_text_tokens + (5/3)*f``, measured exact), and
          ``h3_pipeline_ops.build_packed_layout`` accepts an integer frame
          index per anchor, so index-addressable conditioning does exist here.
          It is measured to bind on the released fl2va weights (an anchor at
          frame 60 is the per-frame RMS argmin, at 640x384 and at 1344x768).
          What it does NOT yet have is a route that reaches it: nothing below
          passes an index, and a mid-timeline placement -- anchoring a
          preserved clip's two boundary frames at their own indices inside one
          generated span -- is unmeasured and is not offered.
        * The ref2va partition conditions on a whole clip rather than on a
          boundary frame (``/generate/ref2vid`` with a reference video, which
          MiniMax documents as video continuation). It gives the model the
          source's motion, and it gives up the exact preservation below: a
          reference is conditioning, not a concatenated span, so every output
          frame is generated. Neither is a better version of the other.

        Consequences worth stating plainly:

        * **The preserved frames are not generated at all.** LTX-2.3 generates
          the whole timeline and this repo pastes the input back over its span;
          here the model is asked only for the missing span, and the output is
          a concatenation. Exactness is by construction, not by a corrective
          paste -- and it holds for every frame of the input, with no
          frame-grid rounding of the preserved side (the ``17n + 5`` rule binds
          the GENERATED span only, which is what the model actually samples).
        * **The anchor frame is not emitted twice.** The generated clip's frame
          0 IS the anchor, i.e. the same instant as the last preserved frame it
          was taken from, so it is dropped from the concatenation; a bridge
          drops both ends. An extend of a P-frame clip by a G-frame span is
          ``P + G - 1`` frames long.
        * **Interior source motion is not provided to the model.** Only the
          boundary frame is. Nothing here compensates for that, and the seam is
          measured rather than hidden (design K5).

        ``reference_images`` (ref2va, extend_forward only): the preserved
        clip's own trailing frames become an automatic video reference
        (soundtrack excluded, tail-truncated to the generated length --
        ``h3_references.normalize_reference_video`` truncates from the HEAD,
        so this hands it the source's LAST frames rather than its first) and
        ``reference_images`` add optional image references BEFORE it, in
        request order -- ``build_outpaint_references``'s docstring has the
        rotary-collision arithmetic for why the image references are packed
        first rather than after the video reference. Both ride through
        ``build_ref2va_packed_layout`` with the boundary anchor placed AFTER
        every reference block (C5). Refused
        (not silently ignored) on ``fl2va`` or on any placement other than
        ``extend_forward`` -- see ``minimax_h3_outpaint_refs_design.md`` §3;
        the route enforces the same table before this function is ever
        called, so these are defensive re-checks.

        Args:
            params: see ``OUTPAINT_VIDEO_DEFAULTS``. ``total_frames`` is a
                REQUESTED output length; the generated span is solved for and
                rounded up to the arch's grid, and the effective values are
                written back into ``params`` (so the route's ``params.copy()``
                carries them to the gallery row) and warned about.
            video_frames: np.uint8 [T, H, W, 3] decoded HEAD clip.
            fps: the head clip's own probed frame rate. Used ONLY to cut its
                original audio track for ``preserve_input``.
            input_audio: WAV bytes of the head clip's original audio, or None.
            bridge_frames / bridge_fps / bridge_audio: the same three things for
                the optional TAIL clip, which turns the request into a bridge.
                ``input_trim_*`` apply to the head clip only.
            reference_images: optional PIL images, ref2va extend_forward only
                (see above).

        Returns:
            ``(frames, audio, audio_sample_rate, actual_seed)`` -- the same
            tuple contract as every other video generate path.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import (
            MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE,
            plan_video_continuation_context,
            plan_video_outpaint_placement,
        )
        from core.inference.outpaint_utils import center_crop_resize_frames

        if not getattr(self, "minimax_h3_components", None):
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")
        if video_frames is None or len(video_frames) == 0:
            raise RuntimeError("vid_outpaint requires a decoded input video clip")

        width = int(params.get("width", 960))
        height = int(params.get("height", 544))

        # ---- Trim the head clip (pixel frames), then preprocess ONCE ----
        trim_start = max(0, int(params.get("input_trim_start_frames", 0) or 0))
        trim_end = max(0, int(params.get("input_trim_end_frames", 0) or 0))
        total_src = video_frames.shape[0]
        end_idx = total_src - trim_end if trim_end > 0 else total_src
        trimmed = video_frames[trim_start:end_idx]
        if trimmed.shape[0] < 1:
            raise ValidationError(
                "vid_outpaint input trim leaves no frames",
                detail=f"input has {total_src} frames; trim_start={trim_start}, trim_end={trim_end}",
            )
        # The RESULT of this preprocessing -- not the raw upload -- is the
        # exact-preserved content (the same convention the image and LTX-2.3
        # outpaint paths state), and it is the same shared helper they call.
        head = center_crop_resize_frames(trimmed, width, height)
        tail = None
        if bridge_frames is not None and len(bridge_frames) > 0:
            tail = center_crop_resize_frames(bridge_frames, width, height)

        arch = (getattr(self, "current_model_info", None) or {}).get("type")
        # Defensive re-check of the route's own gate (a caller that bypasses the
        # route reaches the same answer), and the source of the shared-overlap
        # length the placement is solved with.
        continuation = plan_video_continuation_context(
            params.get("continuation_mode"), params.get("continuation_overlap_frames"),
            arch or "minimax_h3",
            (self.minimax_h3_components.get("variant") or "") or None,
            params.get("continuation_anchor_count"),
        )
        overlap_frames = int(continuation["overlap_frames"])
        motion_anchor_frames = tuple(continuation.get("anchor_local_frames") or ())
        plan = plan_video_outpaint_placement(
            params, arch or "minimax_h3",
            head_frames=int(head.shape[0]),
            tail_frames=int(tail.shape[0]) if tail is not None else None,
            overlap_frames=overlap_frames,
        )
        placement = plan["placement"]
        generated_frames = int(plan["generated_frames"])
        out_frames_total = int(plan["total_frames"])
        frame_rate = float(params.get("frame_rate", 24.0)) or 24.0

        try:
            from api.generation_status import add_warning
        except Exception:  # pragma: no cover - status module always present in-process
            add_warning = None

        def warn(message: str, code: str) -> None:
            print(f"[MiniMax-H3] vid_outpaint: {message}")
            if add_warning is not None:
                try:
                    add_warning(message, code=code)
                except Exception:
                    pass

        if out_frames_total != int(plan["requested_total_frames"]):
            warn(
                f"total_frames={plan['requested_total_frames']} is not reachable by generating a "
                f"valid clip length next to a {plan['head_frames'] + plan['tail_frames']}-frame "
                f"preserved span; generated {generated_frames} frame(s) (17n+5) for an effective "
                f"output of {out_frames_total} frames. The anchor frame is shared with the "
                f"preserved clip and is emitted once.",
                code="outpaint_video_total_frames_adjusted",
            )

        # ---- The conditioning anchors: boundary frames of the preserved
        # clip(s), in PACKED order (first, then last). `pinned_tail` takes none
        # of them: its overlap already includes the boundary frame (latent frame
        # 0 covers exactly it), and an anchor would claim the same conditioning
        # prefix the pin needs.
        #
        # `motion_preroll` takes the OPPOSITE branch of that same exclusivity:
        # it pins nothing and instead places several of the preserved clip's
        # tail frames as anchors ON the overlap, which it then regenerates and
        # discards (design §7.3). ----
        keyframes = []
        if continuation["mode"] == "pinned_tail":
            pass
        elif continuation["mode"] == "motion_preroll":
            keyframes = self._minimax_h3_motion_preroll_keyframes(
                head, generated_frames=generated_frames, overlap_frames=overlap_frames,
                anchor_local_frames=motion_anchor_frames)
        elif placement == "extend_forward":
            keyframes.append(("first", Image.fromarray(head[-1])))
        elif placement == "extend_backward":
            keyframes.append(("last", Image.fromarray(head[0])))
        else:  # bridge: both ends are anchored, in packed order
            keyframes.append(("first", Image.fromarray(head[-1])))
            keyframes.append(("last", Image.fromarray(tail[0])))

        print(f"[MiniMax-H3] vid_outpaint: {placement} {width}x{height} "
              f"preserved head={plan['head_frames']} tail={plan['tail_frames']} "
              f"generated={generated_frames} -> {out_frames_total} frame(s) @ {frame_rate} fps"
              + (f" continuation={continuation['mode']} overlap={overlap_frames}"
                 if continuation["mode"] != "boundary_frame" else ""))

        # ---- ref2va: the source clip is ALWAYS the sole video reference on
        # extend_forward (decision table, minimax_h3_outpaint_refs_design.md
        # §3); every other row is refused. The route enforces the same table
        # (`resolve_minimax_h3_outpaint_reference_gate`, shared here) before
        # this function is ever reached -- this call is a defensive re-check
        # for an internal caller that bypasses the route. Tail-truncated from
        # the HEAD's own end (not the front `normalize_reference_video`
        # keeps): the frames nearest the join are what matter for an anchor.
        variant = (self.minimax_h3_components.get("variant") or "").lower()
        from api.generation_utils import resolve_minimax_h3_outpaint_reference_gate
        resolve_minimax_h3_outpaint_reference_gate(
            variant, has_reference_images=bool(reference_images), placement=placement,
            generated_frames=generated_frames)
        references: tuple = ()
        if variant == "ref2va":
            references = build_outpaint_references(head, generated_frames, frame_rate, reference_images)

        pinned_video_frames: Tuple[int, ...] = ()
        pinned_video_source = None
        pinned_audio = None
        pinned_audio_latents: Tuple[int, ...] = ()
        if continuation["mode"] == "pinned_tail":
            pinned_video_frames, pinned_video_source = self._minimax_h3_pinned_tail_video(
                head, arch or "minimax_h3", generated_frames=generated_frames,
                overlap_frames=overlap_frames)
            pinned_audio, pinned_audio_latents = self._minimax_h3_pinned_tail_audio(
                input_audio, params,
                head_frames=int(plan["head_frames"]), generated_frames=generated_frames,
                overlap_frames=overlap_frames, source_fps=float(fps or frame_rate),
                trim_start=trim_start, frame_rate=frame_rate, warn=warn)
            warn(
                "This request conditions MiniMax-H3 outside the documented shape "
                f"(continuation_mode 'pinned_tail': the preserved clip's last {overlap_frames} "
                f"frame(s) are pinned as the generated span's own leading latent frames"
                + (f", with {len(pinned_audio_latents)} audio latent(s) of the same physical time"
                   if pinned_audio_latents else "")
                + f"). {MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE}; the same pinning mechanism as "
                  "/generate/inpaint/video is used here at the head of the span.",
                code="minimax_h3_undocumented_conditioning",
            )
        elif continuation["mode"] == "motion_preroll":
            warn(
                "This request conditions MiniMax-H3 outside the documented shape "
                f"(continuation_mode 'motion_preroll': {len(keyframes)} anchors on frames "
                f"{', '.join(str(f) for f in motion_anchor_frames)} of the generated span, taken "
                f"from the preserved clip's last {overlap_frames} frame(s)). "
                f"{MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE}",
                code="minimax_h3_undocumented_conditioning",
            )
            warn(
                f"continuation_mode 'motion_preroll' generates {overlap_frames} frame(s) that are "
                f"then discarded: of the {generated_frames} frames sampled, "
                f"{generated_frames - overlap_frames} reach the output. The preserved clip is "
                f"concatenated over that span unchanged, and the anchors add "
                f"{len(keyframes)} frame(s) worth of conditioning rows to every denoise step.",
                code="minimax_h3_motion_preroll_discarded_frames",
            )

        if keyframes and pinned_video_frames:  # pragma: no cover - the resolver refuses it first
            raise ValidationError(
                "a continuation cannot both pin its overlap and anchor it",
                detail="An anchor reserves conditioning rows ahead of the clip and a pin re-uses "
                       "that same prefix for rows of the clip itself, so continuation_mode "
                       "'pinned_tail' and 'motion_preroll' are mutually exclusive.",
            )

        # Only the generated span is sampled; everything else about the run is
        # an ordinary fl2va (or, with references, ref2va) generation, so it
        # goes through the ONE generation path rather than a second copy of
        # the staging/denoise/decode sequence.
        sub_params = dict(params)
        sub_params["num_frames"] = generated_frames
        frames_gen, audio_gen, audio_sample_rate, seed = self._generate_minimax_h3(
            sub_params, keyframes=tuple(keyframes), references=references, label="vid_outpaint",
            pinned_video_frames=pinned_video_frames, pinned_video_source=pinned_video_source,
            input_audio=pinned_audio, pinned_audio_latents=pinned_audio_latents,
            progress_callback=progress_callback, step_callback=step_callback,
        )
        params.update({
            key: value for key, value in sub_params.items()
            if key.startswith("minimax_h3_")
        })
        self._minimax_h3_dump_outpaint_ref_debug(
            placement, frames_gen=frames_gen, head=head, tail=tail)
        if frames_gen.shape[0] != generated_frames:  # pragma: no cover - decode guarantees it
            raise RuntimeError(
                f"MiniMax-H3 returned {frames_gen.shape[0]} generated frame(s) where the placement "
                f"plan expects {generated_frames}.")

        # ---- Assemble. The SHARED frame(s) at the head of the GENERATED span
        # are dropped: they are the model's reconstruction of frames we are
        # preserving exactly, at the same instants. With `pinned_tail` that is
        # the whole pinned overlap, which is why the pin costs nothing in
        # exactness -- pin for conditioning, concatenate for exactness. ----
        if placement == "extend_forward":
            shared = int(plan["shared_anchor_frames"])
            frames_out = np.concatenate([head, frames_gen[shared:]], axis=0)
            preserved_spans = [(0, plan["head_frames"], input_audio, fps, trim_start)]
            gen_audio_start_frame = plan["head_frames"] - shared
        elif placement == "extend_backward":
            frames_out = np.concatenate([frames_gen[:-1], head], axis=0)
            preserved_spans = [(generated_frames - 1, out_frames_total, input_audio, fps, trim_start)]
            gen_audio_start_frame = 0
        else:  # bridge
            frames_out = np.concatenate([head, frames_gen[1:-1], tail], axis=0)
            preserved_spans = [
                (0, plan["head_frames"], input_audio, fps, trim_start),
                (out_frames_total - plan["tail_frames"], out_frames_total,
                 bridge_audio, float(bridge_fps or frame_rate), 0),
            ]
            gen_audio_start_frame = plan["head_frames"] - 1

        if frames_out.shape[0] != out_frames_total:  # pragma: no cover - arithmetic above
            raise RuntimeError(
                f"MiniMax-H3 vid_outpaint assembled {frames_out.shape[0]} frame(s) where the plan "
                f"expects {out_frames_total}.")

        # Recorded in place so routes.py's `params.copy()` -> gallery metadata /
        # DB path picks them up without knowing this arch exists.
        params["outpaint_video_placement"] = placement
        params["outpaint_generated_frames"] = generated_frames
        params["outpaint_effective_preserved_frames"] = plan["head_frames"] + plan["tail_frames"]
        params["outpaint_effective_pixel_start"] = int(preserved_spans[0][0])
        params["outpaint_effective_pixel_end"] = int(preserved_spans[0][1])
        params["total_frames"] = out_frames_total
        params["num_frames"] = out_frames_total
        # The EFFECTIVE continuation context (design sec.4: the effective values
        # are the authoritative ones, not the requested ones). The audio figure
        # is what was actually pinned, which is 0 whenever the clip had no track
        # to pin -- a warning above says so rather than this number implying it.
        params["continuation_mode"] = continuation["mode"]
        params["continuation_overlap_frames"] = overlap_frames
        # Non-zero for every mode that takes an overlap OF the predecessor,
        # whether it pins it (`pinned_tail`) or regenerates and drops it
        # (`motion_preroll`): it is the length the output arithmetic used.
        params["continuation_effective_overlap_frames"] = (
            0 if continuation["mode"] == "boundary_frame" else overlap_frames)
        params["continuation_anchor_count"] = (
            len(keyframes) if continuation["mode"] == "motion_preroll" else 0)
        params["continuation_anchor_frames"] = list(motion_anchor_frames)
        params["continuation_effective_overlap_samples"] = int(
            len(pinned_audio_latents)
            * int(self.minimax_h3_components.get("audio_sample_rate", 32000))
            / float(self.minimax_h3_components.get("audio_latent_rate", 40.0))
        ) if pinned_audio_latents else 0

        audio_out = audio_gen
        if audio_gen is not None and audio_sample_rate:
            audio_out = self._minimax_h3_outpaint_audio(
                audio_gen, audio_sample_rate, params,
                total_frames=out_frames_total, frame_rate=frame_rate,
                gen_audio_start_frame=gen_audio_start_frame,
                preserved_spans=preserved_spans, warn=warn,
            )

        return frames_out, audio_out, audio_sample_rate, seed

    def _minimax_h3_motion_preroll_keyframes(
        self,
        head: np.ndarray,
        *,
        generated_frames: int,
        overlap_frames: int,
        anchor_local_frames: Sequence[int],
    ) -> List[Tuple[Any, Image.Image]]:
        """`motion_preroll`'s anchors: ``(anchor, image)`` in PACKED order.

        The generated span opens on the overlap, so its local frame ``k < m``
        is the same instant as the preserved clip's frame ``len(head) - m + k``
        -- and local ``m - 1`` is the boundary frame, the one instant a
        `boundary_frame` continuation anchors. Those frames are handed to the
        model at their own indices and then regenerated; the caller keeps the
        preserved pixels by concatenation, exactly as it does for every other
        mode, so nothing here is load-bearing for exactness.

        The index resolution, the ends -> ``"first"``/``"last"`` mapping and the
        duplicate/range refusals are ``plan_keyframe_placements``', shared with
        /generate/img2vid rather than repeated. The images are already exactly
        ``width x height`` (`center_crop_resize_frames` ran on the whole clip),
        so `_minimax_h3_fit_keyframe`'s stretch/cover branches are both the
        identity here.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import plan_keyframe_placements

        head_frames = int(head.shape[0])
        if not anchor_local_frames:  # pragma: no cover - the resolver fills them
            raise ValidationError(
                "continuation_mode 'motion_preroll' has no anchors to place",
                detail="The anchor positions are resolved from the pre-roll length and the "
                       "anchor count before this point.",
            )
        if overlap_frames >= generated_frames:
            raise ValidationError(
                "the motion pre-roll leaves nothing to generate",
                detail=f"A {overlap_frames}-frame pre-roll covers the whole "
                       f"{generated_frames}-frame generated span, so the continuation would add "
                       f"no new frames.",
            )
        plan = plan_keyframe_placements(
            [(f"motion_preroll[{position}]", int(frame))
             for position, frame in enumerate(anchor_local_frames)],
            generated_frames,
        )
        return [
            (entry["anchor"],
             Image.fromarray(head[head_frames - overlap_frames + int(entry["frame"])]))
            for entry in plan["anchors"]
        ]

    def _minimax_h3_pinned_tail_video(
        self,
        head: np.ndarray,
        arch: str,
        *,
        generated_frames: int,
        overlap_frames: int,
    ) -> Tuple[Tuple[int, ...], np.ndarray]:
        """``(pinned latent frames, the clip they are encoded from)``, `pinned_tail`.

        The generated span's own timeline starts AT the overlap: its first
        ``overlap_frames`` pixel frames are the preserved clip's last ones, so
        the latent frames covering them are latent frames 0..m-1 and the pin is
        a leading prefix -- an identity permutation, which is why this composes
        with FBCache's whole-latent-frame assumption while an interior pin does
        not (`minimax_h3_block_loop_wrapper.py`).

        Putting the imported history at the START of the new clip is also what
        keeps the rotary clock consistent: attention depends on position
        DIFFERENCES, so history laid down as frames 0..m-1 of this clip needs no
        knowledge of the clock it had in its own segment.

        The frames after the overlap are held at the last preserved frame rather
        than left black: only the pinned rows are substituted into the sampler,
        but the VAE encodes the whole clip, so a cut to black at the overlap
        boundary would be the one thing the pinned latents could see.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import latent_frame_spans
        from core.models.components.wiring import temporal_spec_for_arch
        from core.models.minimax_h3.loader import minimax_h3_latent_frames

        spec = temporal_spec_for_arch(arch)
        spans = latent_frame_spans(spec, minimax_h3_latent_frames(generated_frames)) if spec else []
        pinned = tuple(index for index, (_lo, hi) in enumerate(spans) if hi <= overlap_frames)
        # Both of these are refused at the route (`plan_video_continuation_context`
        # for the alignment, the 17-frame ceiling against the 124-frame floor for
        # the emptiness); they are restated because getting either wrong pins a
        # different span than the caller was told.
        if not pinned or spans[pinned[-1]][1] != overlap_frames:
            raise ValidationError(
                "the continuation overlap does not land on a video-VAE group boundary",
                detail=f"{overlap_frames} frame(s) cannot be pinned whole on this clip's "
                       f"latent grid.",
            )
        if len(pinned) >= len(spans):
            raise ValidationError(
                "the continuation overlap leaves nothing to generate",
                detail=f"{overlap_frames} pinned frame(s) cover the whole {generated_frames}-frame "
                       f"generated span.",
            )
        filler = np.repeat(head[-1:], generated_frames - overlap_frames, axis=0)
        return pinned, np.concatenate([head[-overlap_frames:], filler], axis=0)

    def _minimax_h3_pinned_tail_audio(
        self,
        input_audio: Optional[bytes],
        params: Dict[str, Any],
        *,
        head_frames: int,
        generated_frames: int,
        overlap_frames: int,
        source_fps: float,
        trim_start: int,
        frame_rate: float,
        warn: Callable[[str, str], None],
    ):
        """The audio half of `pinned_tail`: the same physical time, or nothing.

        The pinned audio latents are the whole ones that fit INSIDE the video
        overlap (``plan_audio_pin_latents``' inward snap, the same helper
        ``regenerate_range`` uses), so the two pins never describe different
        spans of time -- the audio grid is finer than the video one, so it can
        only be a subset, never an overhang.

        Returns ``(waveform | None, pinned latent indices)``. ``None`` means the
        continuation runs with a video-only pin, which happens only when there
        is no track to pin at all (the clip carries none, or the request did not
        ask for the input's audio); that is warned, never assumed. A track that
        EXISTS but cannot be cut is an error, not a downgrade: the caller asked
        for a joint pin and would otherwise get a different, unannounced one.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import plan_audio_pin_latents
        from core.models.minimax_h3 import h3_pipeline_ops as ops

        components = self.minimax_h3_components
        if not input_audio:
            warn(
                "continuation_mode='pinned_tail' pinned video only: the clip's own soundtrack was "
                f"not available to pin (outpaint_video_audio_mode="
                f"{str(params.get('outpaint_video_audio_mode'))!r}, audio_enable="
                f"{bool(params.get('audio_enable', True))}). The generated span's audio starts "
                "from noise while its video continues the pinned frames.",
                code="minimax_h3_pinned_tail_video_only",
            )
            return None, ()

        latent_rate = float(components.get("audio_latent_rate", 40.0))
        # The MODEL's fixed fps, for the reason `regenerate_range` states: this
        # layer must not depend on the route having rewritten `frame_rate`.
        model_fps = float(components.get("fps", 24.0))
        num_audio_latents = ops.audio_latent_frames(
            generated_frames, fps=model_fps, latents_per_second=latent_rate)
        _free, pinned = plan_audio_pin_latents(
            overlap_frames, generated_frames, num_audio_latents,
            fps=model_fps, latents_per_second=latent_rate)
        if not pinned:
            warn(
                f"continuation_mode='pinned_tail' pinned video only: a {overlap_frames}-frame "
                f"overlap contains no whole audio latent at {latent_rate} latents/s.",
                code="minimax_h3_pinned_tail_video_only",
            )
            return None, ()

        waveform = self._minimax_h3_inpaint_pinned_audio(
            input_audio, clip_frames=generated_frames, source_fps=source_fps,
            trim_start=trim_start, frame_rate=frame_rate, warn=warn,
            mode="pinned_tail", source_start_frame=head_frames - overlap_frames,
            fallback_clause="the request is refused rather than pinning the video against an "
                            "unpinned soundtrack",
        )
        if waveform is None:
            raise ValidationError(
                "the continuation's audio tail could not be pinned",
                detail="continuation_mode='pinned_tail' pins video and audio for the same "
                       "physical time; the uploaded clip's track could not be cut for it. Retry "
                       "with continuation_mode='boundary_frame', or with "
                       "outpaint_video_audio_mode='regenerate' to accept a video-only pin.",
            )
        # Past the overlap the window is edge-padded material this run never
        # pins; silence it so the audio VAE sees the track end rather than a
        # held DC level bleeding into the boundary latent.
        overlap_samples = int(round(
            overlap_frames / float(frame_rate or model_fps)
            * int(components.get("audio_sample_rate", 32000))))
        if 0 < overlap_samples < waveform.shape[-1]:
            waveform[:, overlap_samples:] = 0.0
        return waveform, pinned

    def _minimax_h3_outpaint_audio(
        self,
        audio_gen,
        sample_rate: int,
        params: Dict[str, Any],
        *,
        total_frames: int,
        frame_rate: float,
        gen_audio_start_frame: int,
        preserved_spans,
        warn: Callable[[str, str], None],
    ):
        """Lay the generated audio onto the OUTPUT timeline, per audio mode.

        WHY THIS IS NOT LTX-2.3's AUDIO PATH, and cannot be: MiniMax-H3
        generates audio and video jointly, in one packed sequence, for one
        span. It produced audio for the GENERATED frames and for nothing else,
        because it was never asked to generate the preserved frames. So the two
        modes cannot both mean what they mean on LTX-2.3, where the pipeline
        hands back a whole-timeline track:

        * ``regenerate`` -- "do not carry the input clip's audio over" -- is
          honoured literally: the generated track is placed at the generated
          span's own position and the preserved span is left SILENT, with a
          warning saying so and naming the mode that fills it. Quietly
          substituting the input's audio here would make the two modes the same
          thing while claiming to be different. Because that outcome (extend a
          clip that has sound, get back a video whose original half is silent)
          is the wrong thing to hand someone who expressed no preference, it is
          NOT this architecture's default -- ``OUTPAINT_VIDEO_ARCH_OVERLAYS``
          defaults MiniMax-H3 to ``preserve_input``, so reaching this branch
          means the caller asked for ``regenerate`` by name.
        * ``preserve_input`` splices each preserved span's ORIGINAL audio over
          it, through exactly the LTX-2.3 helpers (``extract_audio_window`` ->
          ``mux_audio_over_span``, 50 ms crossfade confined to the generated
          side), so an input audio sample is never resampled twice or
          crossfaded away.

        The generated track is placed at ``gen_audio_start_frame`` rather than
        at the first NEW frame: the anchor frame is part of the generated span's
        own clock, so aligning on it is what keeps A and V in sync.
        """
        import numpy as _np

        # The fallback is this ARCHITECTURE's resolved default, not the base
        # map's -- `params` always carries the key when the route built it, and
        # a caller that assembled `params` by hand should land on the same
        # answer an omitted form field lands on.
        from api.param_defaults import outpaint_video_defaults_for_arch

        audio_mode = (params.get("outpaint_video_audio_mode")
                      or outpaint_video_defaults_for_arch("minimax_h3")["outpaint_video_audio_mode"])
        generated = audio_gen.numpy() if hasattr(audio_gen, "numpy") else _np.asarray(audio_gen)
        channels = generated.shape[0]
        total_samples = int(round((total_frames / frame_rate) * sample_rate)) if frame_rate else generated.shape[1]

        full = _np.zeros((channels, total_samples), dtype=generated.dtype)
        start = int(round((gen_audio_start_frame / frame_rate) * sample_rate)) if frame_rate else 0
        start = max(0, min(start, total_samples))
        width = min(generated.shape[1], total_samples - start)
        if width > 0:
            full[:, start:start + width] = generated[:, :width]

        if audio_mode != "preserve_input":
            warn(
                "outpaint_video_audio_mode='regenerate' was requested explicitly (MiniMax-H3 "
                "defaults to 'preserve_input'): this architecture generates audio only for the "
                "frames it generates, so the preserved span carries no audio and is silent. Omit "
                "the field, or send 'preserve_input', to carry the input clip's own audio across "
                "it.",
                code="outpaint_video_audio_preserved_span_silent",
            )
            return torch.from_numpy(full)

        from utils.video_utils import extract_audio_window, mux_audio_over_span

        for span_start, span_end, src_audio, src_fps, src_trim_start in preserved_spans:
            span_frames = int(span_end) - int(span_start)
            if span_frames <= 0:
                continue
            if src_audio is None:
                warn(
                    "preserve_input audio mode requested but a preserved clip has no audio stream; "
                    "its span is left silent",
                    code="outpaint_video_no_input_audio",
                )
                continue
            offset_sec = span_start / frame_rate
            target_dur_sec = span_frames / frame_rate
            src_fps = float(src_fps or frame_rate)
            src_start_sec = (src_trim_start / src_fps) if src_fps else 0.0
            src_dur_sec = (span_frames / src_fps) if src_fps else target_dur_sec
            if target_dur_sec > 0 and abs(src_dur_sec - target_dur_sec) / target_dur_sec > 0.005:
                warn(
                    f"preserve_input audio was time-stretched ({src_dur_sec:.3f}s -> "
                    f"{target_dur_sec:.3f}s) because a preserved clip's frame rate ({src_fps:.3f}) "
                    f"differs from MiniMax-H3's fixed {frame_rate:.3f} fps",
                    code="outpaint_video_audio_stretched",
                )
            try:
                window = extract_audio_window(
                    src_audio, src_start_sec, src_dur_sec, target_dur_sec,
                    sample_rate=sample_rate, channels=channels,
                )
            except Exception as exc:
                window = None
                print(f"[MiniMax-H3] vid_outpaint audio window extraction raised: {exc}")
            if window is None:
                # NEVER overwrite with silence on a failure -- leave whatever is
                # already there (the generated track, or the silence the mode
                # already warned about).
                warn(
                    "preserve_input audio window extraction failed; that span was left as generated",
                    code="outpaint_video_audio_extract_failed",
                )
                continue
            full = mux_audio_over_span(
                full, window, offset_sec=offset_sec, dur_sec=target_dur_sec,
                sample_rate=sample_rate, crossfade_ms=50.0,
            )

        return torch.from_numpy(full)

    def _generate_vidinpaint_minimax_h3(
        self,
        params: Dict[str, Any],
        video_frames: np.ndarray,
        fps: float,
        input_audio: Optional[bytes],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
        spatial_mask_timeline=None,
        spatial_mask_arrays=None,
        references: Sequence[Any] = (),
    ):
        """Temporal inpaint with MiniMax-H3: regenerate one time range in place.

        PIN FOR CONDITIONING, PASTE FOR EXACTNESS -- the two halves are separate
        and each is needed:

        * the clip is encoded once and the latent frames OUTSIDE the range are
          pinned at ``VISUAL_COND_TIMESTEP`` and never denoised, so the model
          generates the range against the rest of the clip. The released fl2va
          weights honour that at the decoder's floor (preserved-span RMS 3.12
          against a VAE round-trip floor of 3.15, control 75.69 --
          ``scratchpad/minimax_h3_ti_probe_results.md``);
        * the source pixels are then pasted back over the preserved region. The
          pin alone returns those frames through a VAE round trip (3.20 RMS,
          38.04 dB) plus up to 2.97 RMS of decoder bleed within ~15 frames of a
          boundary, so "preserved" is true with the paste and false without it.
          The paste is not a toggle for that reason.

        The output is the same length as the TRIMMED input, which is why this is
        not a placement of the outpaint endpoint: there, ``17n + 5`` binds the
        generated span; here every frame of the clip has a row, so the trimmed
        clip itself must be a valid length. It is never snapped.

        Exactness is claimed of the returned frames and of an FFV1 encode
        (``video_lossless``); the default H.264 mp4 is lossy for preserved and
        generated frames alike, the same scoping the outpaint path states.

        Args:
            params: see ``INPAINT_VIDEO_DEFAULTS``. ``regenerate_start_frame`` /
                ``regenerate_end_frame`` are pixel frames of the trimmed clip,
                start inclusive and end exclusive; the effective (latent-group
                aligned) span is written back into ``params``.
            video_frames: np.uint8 [T, H, W, 3] decoded input clip.
            fps: the clip's own probed frame rate, used only to cut its original
                audio track for ``preserve_input``.
            input_audio: WAV bytes of the clip's original audio, or None.
            references: a ``ref2va`` reference list (``h3_references.
                MiniMaxH3Reference`` objects, packed order), same convention
                as ``_generate_ref2vid_minimax_h3``. Refused on ``fl2va``/
                ``hybrid`` by ``resolve_minimax_h3_inpaint_reference_gate``;
                allowed on ``ref2va`` (see that function's docstring for the
                unmeasured-shape caveat).

        Returns:
            ``(frames, audio, audio_sample_rate, actual_seed)`` -- the same
            tuple contract as every other video generate path.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import (
            MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE,
            latent_frame_spans,
            plan_audio_pin_latents,
            plan_video_inpaint_span,
            resolve_minimax_h3_inpaint_reference_gate,
        )
        from core.inference.outpaint_utils import center_crop_resize_frames
        from core.models.minimax_h3 import h3_pipeline_ops as ops

        if not getattr(self, "minimax_h3_components", None):
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")
        if video_frames is None or len(video_frames) == 0:
            raise RuntimeError("vid_inpaint requires a decoded input video clip")

        references = tuple(references or ())
        # Defensive re-check of the route's gate (mirroring
        # `_generate_vidoutpaint_minimax_h3`'s own re-check), run before any
        # VAE/DiT work even when `references` is empty. PHASE B-3-open:
        # `ref2va` now passes -- fl2va still refuses any reference, and
        # hybrid still refuses every request. `has_vision_conditioning=True`:
        # this endpoint always pins the frames outside the regenerate range,
        # so an audio-only reference set is never refused by the pairing
        # rule alone here.
        _variant = ((getattr(self, "minimax_h3_components", None) or {}).get("variant") or "").lower()
        resolve_minimax_h3_inpaint_reference_gate(
            _variant,
            has_reference_images=any(getattr(r, "kind", "") == "image" for r in references),
            has_reference_videos=any(getattr(r, "kind", "") == "video" for r in references),
            has_reference_audios=any(getattr(r, "kind", "") == "audio" for r in references),
            has_vision_conditioning=True,
        )
        if references and (spatial_mask_timeline is not None or spatial_mask_arrays is not None):
            # `build_ref2va_packed_layout`'s pin extension (B-1) only carries
            # `pinned_video_frames` (a whole-frame pin), not
            # `pinned_video_row_indices` (spatial inpaint's row-level pin).
            # The route already refuses this combination as a 400
            # (`generate_inpaint_video`'s own check); this is the defensive
            # re-check for a caller that bypasses the route, kept as a
            # ValidationError (400) rather than a RuntimeError (500) now that
            # the ref2va gate is open and this branch is reachable in
            # practice, not just in theory.
            raise ValidationError(
                "MiniMax-H3 cannot combine a spatial mask with reference conditioning: the "
                "extended ref2va layout builder only carries a frame-level temporal-inpaint pin "
                "alongside references, not a row-level spatial-mask pin.")

        width = int(params.get("width", 960))
        height = int(params.get("height", 544))

        trim_start = max(0, int(params.get("input_trim_start_frames", 0) or 0))
        trim_end = max(0, int(params.get("input_trim_end_frames", 0) or 0))
        total_src = video_frames.shape[0]
        end_idx = total_src - trim_end if trim_end > 0 else total_src
        trimmed = video_frames[trim_start:end_idx]
        if trimmed.shape[0] < 1:
            raise ValidationError(
                "vid_inpaint input trim leaves no frames",
                detail=f"input has {total_src} frames; trim_start={trim_start}, trim_end={trim_end}",
            )
        # The RESULT of this preprocessing is the preserved content, the same
        # convention the image and video outpaint paths state.
        clip = center_crop_resize_frames(trimmed, width, height)

        spatial_mask_mode = (spatial_mask_timeline is not None or
                             spatial_mask_arrays is not None)
        full_soft_masks = None
        pinned_video_row_indices: Tuple[int, ...] = ()

        arch = (getattr(self, "current_model_info", None) or {}).get("type")
        plan = plan_video_inpaint_span(
            params,
            arch or "minimax_h3",
            clip_frames=int(clip.shape[0]),
            allow_full_range=spatial_mask_mode,
        )
        start_frame = int(plan["start_frame"])
        end_frame = int(plan["end_frame"])
        clip_frames = int(plan["clip_frames"])
        frame_rate = float(params.get("frame_rate", 24.0)) or 24.0

        try:
            from api.generation_status import add_warning
        except Exception:  # pragma: no cover - status module always present in-process
            add_warning = None

        def warn(message: str, code: str) -> None:
            print(f"[MiniMax-H3] vid_inpaint: {message}")
            if add_warning is not None:
                try:
                    add_warning(message, code=code)
                except Exception:
                    pass

        if spatial_mask_mode:
            from core.inference.video_mask_timeline import (
                MaskTimelineManifest,
                VideoMaskTimelineError,
                build_spatial_mask_plan,
                composite_masked_frames,
            )
            from core.models.components.wiring import temporal_spec_for_arch

            if spatial_mask_timeline is None or spatial_mask_arrays is None:
                raise ValidationError(
                    "spatial mask timeline and arrays must be supplied together",
                    detail="Pass both spatial_mask_timeline and spatial_mask_arrays, or neither.",
                )
            # H-2: enforced at the route (a 400 before this call is even
            # reached), restated here so a caller that bypasses the route
            # (an internal script, a future second caller) cannot hit the
            # RuntimeError this combination raises deep in the denoise loop
            # (`MiniMaxH3BlockLoopWrapper._custom_forward`'s FBCache guard
            # indicator) after the text encode, VAE encode and DiT staging
            # have already run.
            #
            # Low-2 (final audit): this uses `fbcache_active(params)`
            # (threshold-aware -- FBCache never actually runs at
            # `fbcache_threshold=0`), which is a DIFFERENT, looser basis than
            # the route's raw `fbcache_enable` check (`api/routes.py`). That
            # is intentional layering, not drift: the route refuses a
            # checkbox state the UI can produce regardless of whether it
            # would matter, while this restatement only needs to protect the
            # actual invariant this layer cares about --
            # `minimax_h3_spatial_mask_fbcache_test.py::
            # test_spatial_mask_with_fbcache_threshold_zero_is_not_refused`
            # pins the difference. Do not "align" the two without re-reading
            # that test.
            from core.inference.fbcache import fbcache_active
            if fbcache_active(params):
                raise ValidationError(
                    "spatial mask is incompatible with FBCache",
                    detail="FBCache's per-frame guard indicator assumes the free video rows "
                           "tile into whole latent frames, which a spatial mask's row-level pin "
                           "does not guarantee. Disable fbcache_enable, or drop the spatial mask.",
                )
            if not isinstance(spatial_mask_timeline, MaskTimelineManifest):
                raise ValidationError(
                    "invalid spatial mask timeline",
                    detail="spatial_mask_timeline must be a MaskTimelineManifest.",
                )
            if clip.ndim != 4 or clip.shape[0] != clip_frames or clip.shape[1:3] != (height, width):
                raise ValidationError(
                    "invalid inpaint source clip geometry",
                    detail=f"Expected [{clip_frames}, {height}, {width}, 3], got {clip.shape}.",
                )
            if clip.shape[-1] != 3 or clip.dtype != np.uint8:
                raise ValidationError(
                    "invalid inpaint source clip format",
                    detail=f"Expected uint8 RGB frames, got dtype={clip.dtype}, shape={clip.shape}.",
                )
            if (spatial_mask_timeline.canvas.width != width or
                    spatial_mask_timeline.canvas.height != height):
                raise ValidationError(
                    "spatial mask canvas does not match the output canvas",
                    detail=(
                        f"Mask canvas is {spatial_mask_timeline.canvas.width}x"
                        f"{spatial_mask_timeline.canvas.height}; output is {width}x{height}."
                    ),
                )

            components = self.minimax_h3_components
            spatial_scale = int(components.get("vae_scale_factor_spatial", 0) or 0)
            transformer_config = components.get("transformer_config") or {}
            patch_size = tuple(transformer_config.get("patch_size") or ())
            if spatial_scale <= 0 or len(patch_size) != 3:
                raise ValidationError(
                    "MiniMax-H3 spatial mask geometry is unavailable",
                    detail="The loaded components do not expose VAE scale and transformer patch size.",
                )
            patch_h = int(patch_size[1])
            patch_w = int(patch_size[2])
            spec = temporal_spec_for_arch(arch or "minimax_h3")
            spans = latent_frame_spans(spec, int(plan["latent_frames"])) if spec else []
            _mask_plan_warnings: list = []
            try:
                full_soft_masks, pinned_video_row_indices = build_spatial_mask_plan(
                    spatial_mask_timeline,
                    spatial_mask_arrays,
                    clip_frames=clip_frames,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    latent_frame_spans=spans,
                    spatial_scale=spatial_scale,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    warnings=_mask_plan_warnings,
                )
            except VideoMaskTimelineError as exc:
                # M-1: a MEMORY error or an internal numpy bug from this call
                # (e.g. H-3's `composite_masked_frames`, called later, or a
                # bug in the pooling/rasterization arithmetic itself) is not
                # the caller's input being wrong, and must not be reported to
                # them as if it were. Only the timeline module's OWN input-
                # validation exceptions are treated as a 400 here; anything
                # else propagates as a 500.
                raise ValidationError(
                    "invalid spatial mask timeline",
                    detail=str(exc),
                ) from exc
            for _mask_warning in _mask_plan_warnings:
                warn(_mask_warning, code="minimax_h3_spatial_mask_warning")

        if plan["snapped"]:
            warn(
                f"frames {plan['requested_start']}-{plan['requested_end']} were expanded to "
                f"{start_frame}-{end_frame}: the video VAE stores frames in groups of up to 4 and "
                f"a group is regenerated or preserved as a whole, so a range is expanded outward "
                f"to group boundaries rather than shrunk.",
                code="inpaint_video_range_snapped",
            )
        warn(
            "This request conditions MiniMax-H3 outside the documented shape (frames of the clip "
            "pinned at interior positions while a range between them is regenerated). "
            f"{MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE}; the same pinning mechanism is used here at "
            "other positions.",
            code="minimax_h3_undocumented_conditioning",
        )
        if _variant == "ref2va":
            # Phase B-3-open (audit M2): §6.2 arm P says the INTERIOR PIN is
            # unmeasured on ref2va -- that is true with zero references too
            # (the resolver's ref2va row is open unconditionally, not only
            # when a reference is present), so this sentence must not be
            # gated on `references`. The reference clause is appended only
            # when one is actually present, since that is a SECOND, distinct
            # unmeasured fact (arm P-ref), not a restatement of the first.
            message = (
                "This request conditions MiniMax-H3 outside the documented shape on the ref2va "
                "transformer (a mid-clip pin, generation-only). The mid-clip pin is measured on the "
                "fl2va partition (preserved-span RMS 3.12, floor 3.15, control 75.69) and unmeasured "
                "on ref2va; whether ref2va holds an interior pin the way fl2va does has not been "
                "measured."
            )
            if references:
                message += (
                    " This request also combines the pin with reference conditioning: reading "
                    "reference rows is a trained behaviour of ref2va, not of fl2va, and whether "
                    "ref2va reads a reference while pinned has not been measured either -- this "
                    "combination has not been measured at all."
                )
            warn(message, code="minimax_h3_undocumented_conditioning")

        # ---- Audio. `preserve_input` pins the clip's own track across the WHOLE
        # clip through the shipped ia2v machinery and muxes it back verbatim, so
        # the regenerated span has the original soundtrack both to condition on
        # and in the output; the pin and the exact mux are both
        # `_generate_minimax_h3`'s ia2v behaviour. `regenerate_range` now pins
        # the PRESERVED spans only -- the same partial-audio-pin mechanism
        # `build_packed_layout`'s `pinned_audio_latents` builds. That mechanism
        # in isolation, with video left free, is what
        # `scratchpad/minimax_h3_ai_probe_results.md` measured (its own §4: the
        # SAME shape with video frames also pinned on the same range -- the
        # actual configuration below, which pins both -- was not measured) --
        # so the range itself still generates against real conditioning either
        # side of it, rather than blind; the input's own track is ALSO spliced back in over those
        # spans AFTER generation (`_minimax_h3_splice_inpaint_range_audio`,
        # below), because a pinned latent still comes back through an
        # audio-VAE round trip -- "pin for conditioning, paste for exactness",
        # the same reason the pixel path pastes. ----
        audio_mode = str(params.get("inpaint_video_audio_mode") or "regenerate")
        pinned_audio = None
        pinned_audio_latents: Tuple[int, ...] = ()
        if audio_mode == "preserve_input":
            pinned_audio = self._minimax_h3_inpaint_pinned_audio(
                input_audio, clip_frames=clip_frames, source_fps=float(fps or frame_rate),
                trim_start=trim_start, frame_rate=frame_rate, warn=warn)
            if pinned_audio is None:
                audio_mode = "regenerate"
                params["inpaint_video_audio_mode"] = "regenerate"
            elif not params.get("audio_enable", True):
                warn("audio_enable is false: the clip's own track still conditions the generation "
                     "(its rows ride the packed sequence at t = 1.0), and nothing is muxed into "
                     "the output file.",
                     code="minimax_h3_input_audio_not_muxed")
        elif audio_mode == "regenerate_range" and not input_audio:
            # `regenerate_range` needs `input_audio` both to pin the preserved
            # spans as conditioning and to splice them back in after
            # generation, so a clip with no audio stream has nothing to fall
            # back to but plain `regenerate`.
            warn("inpaint_video_audio_mode='regenerate_range' was requested but the uploaded "
                 "clip has no audio stream; the whole clip's soundtrack is generated instead",
                 code="inpaint_video_no_input_audio")
            audio_mode = "regenerate"
            params["inpaint_video_audio_mode"] = "regenerate"
        elif audio_mode == "regenerate_range":
            components = self.minimax_h3_components
            audio_latent_rate = float(components.get("audio_latent_rate", 40.0))
            # The MODEL's fixed fps, not the request's `frame_rate` -- the same
            # basis `_generate_minimax_h3` derives `num_audio_latents` from
            # (`float(components.get("fps", 24.0))`). The two only happened to
            # agree because the route rewrites `params["frame_rate"]` to
            # `fps_fixed` before this runs; this layer must not depend on that.
            model_fps = float(components.get("fps", 24.0))
            num_audio_latents = ops.audio_latent_frames(
                clip_frames, fps=model_fps, latents_per_second=audio_latent_rate)
            _free_audio_latents, pinned_audio_latent_span = plan_audio_pin_latents(
                start_frame, end_frame, num_audio_latents,
                fps=model_fps, latents_per_second=audio_latent_rate)
            if not pinned_audio_latent_span:
                # The snapped range covers the whole clip's audio grid: there
                # is nothing outside it to pin, so this degenerates to plain
                # `regenerate` -- recorded the same way the no-audio-stream
                # branch above records it, so the mode written to gallery
                # metadata and echoed to the client matches what actually ran
                # (the output-level splice below is skipped entirely once
                # `audio_mode` reads "regenerate", since both preserved spans
                # are empty anyway).
                warn("regenerate_range's regenerate range covers the whole clip's audio grid, so "
                     "there is nothing to pin; the soundtrack generates unconditioned exactly as "
                     "plain 'regenerate' does",
                     code="inpaint_video_audio_range_full")
                audio_mode = "regenerate"
                params["inpaint_video_audio_mode"] = "regenerate"
            else:
                pinned_audio = self._minimax_h3_inpaint_pinned_audio(
                    input_audio, clip_frames=clip_frames, source_fps=float(fps or frame_rate),
                    trim_start=trim_start, frame_rate=frame_rate, warn=warn,
                    mode="regenerate_range")
                if pinned_audio is not None:
                    pinned_audio_latents = pinned_audio_latent_span
                    warn(
                        "inpaint_video_audio_mode='regenerate_range' conditions the regenerate "
                        f"range's audio on the input track outside it (frames {start_frame}-"
                        f"{end_frame}) via the same mid-clip audio pin mechanism as the video "
                        "pin -- an even less documented shape than that pin alone.",
                        code="minimax_h3_undocumented_audio_conditioning",
                    )
                    if not params.get("audio_enable", True):
                        # Same stance as `preserve_input`: the pinned rows still
                        # ride the packed sequence and still condition the video
                        # through self-attention, so they are worth extracting
                        # and encoding even though nothing gets decoded or muxed.
                        warn("audio_enable is false: the preserved spans' own track still "
                             "conditions the regenerate range (its rows ride the packed sequence "
                             "at t = 1.0), and nothing is muxed into the output file.",
                             code="minimax_h3_input_audio_not_muxed")
                # else: `_minimax_h3_inpaint_pinned_audio` already warned (no
                # stream, or the window extraction failed); `pinned_audio`
                # stays None and the range generates unconditioned, exactly the
                # pre-partial-pin `regenerate_range` behaviour. The output-level
                # splice below still runs regardless.

        print(f"[MiniMax-H3] vid_inpaint: {width}x{height} clip={clip_frames} frame(s) "
              f"regenerate {start_frame}..{end_frame} "
              f"({len(plan['regenerate_latent_frames'])} of {plan['latent_frames']} latent "
              f"frames) audio={audio_mode} @ {frame_rate} fps"
              + (f" spatial_pinned_rows={len(pinned_video_row_indices)}"
                 if spatial_mask_mode else "")
              + (f" audio_pinned_latents={len(pinned_audio_latents)}"
                 if pinned_audio_latents else ""))

        sub_params = dict(params)
        sub_params["num_frames"] = clip_frames
        if spatial_mask_mode:
            frames_gen, audio_out, audio_sample_rate, seed = self._generate_minimax_h3(
                sub_params,
                pinned_video_frames=(),
                pinned_video_row_indices=pinned_video_row_indices,
                pinned_video_source=clip, input_audio=pinned_audio,
                pinned_audio_latents=pinned_audio_latents, label="vid_inpaint",
                progress_callback=progress_callback, step_callback=step_callback,
            )
        else:
            frames_gen, audio_out, audio_sample_rate, seed = self._generate_minimax_h3(
                sub_params,
                pinned_video_frames=plan["pinned_latent_frames"],
                pinned_video_source=clip, input_audio=pinned_audio,
                pinned_audio_latents=pinned_audio_latents, references=references,
                label="vid_inpaint",
                progress_callback=progress_callback, step_callback=step_callback,
            )
        if frames_gen.shape[0] != clip_frames:  # pragma: no cover - decode guarantees it
            raise RuntimeError(
                f"MiniMax-H3 returned {frames_gen.shape[0]} frame(s) where this clip is "
                f"{clip_frames}.")

        if spatial_mask_mode:
            try:
                frames_out = composite_masked_frames(clip, frames_gen, full_soft_masks)
            except VideoMaskTimelineError as exc:
                # M-1: only this module's OWN input-validation exceptions are
                # a 400; a MemoryError (H-3 is a mitigation, not a guarantee
                # on every host) or an internal bug must propagate as a 500.
                raise ValidationError(
                    "invalid MiniMax-H3 spatial mask composite",
                    detail=str(exc),
                ) from exc
            _preserved_fraction = float(np.count_nonzero(full_soft_masks < 0.5)) / float(full_soft_masks.size)
            params["inpaint_video_spatial_mask"] = True
            params["inpaint_video_spatial_mask_preserved_fraction"] = round(_preserved_fraction, 4)
            params["inpaint_video_spatial_mask_keyframe_count"] = len(spatial_mask_timeline.keyframes)
            params["inpaint_video_spatial_mask_feather_px"] = spatial_mask_timeline.composite_feather_px
        else:
            # ---- The paste. Everything outside the regenerated range is the
            # input's own pixels; the range itself is untouched.
            frames_out = np.array(frames_gen, dtype=np.uint8, copy=True)
            frames_out[:start_frame] = clip[:start_frame]
            frames_out[end_frame:] = clip[end_frame:]

        params["num_frames"] = clip_frames
        params["inpaint_video_effective_start_frame"] = start_frame
        params["inpaint_video_effective_end_frame"] = end_frame
        params["inpaint_video_preserved_frames"] = clip_frames - (end_frame - start_frame)

        if audio_mode == "regenerate_range":
            # The output-level half of this mode: pin for conditioning, paste
            # for exactness -- the preserved spans were pinned above as
            # conditioning WHEN `pinned_audio` was available (see the
            # `elif audio_mode == "regenerate_range":` branch), but a pinned
            # latent still comes back through an audio-VAE round trip, so the
            # input's own audio is spliced back over the two preserved spans
            # -- everything outside [start_frame, end_frame) -- with the same
            # helpers the outpaint path splices with, leaving the range itself
            # as generated.
            audio_out = self._minimax_h3_splice_inpaint_range_audio(
                audio_out, input_audio, audio_sample_rate,
                clip_frames=clip_frames, start_frame=start_frame, end_frame=end_frame,
                source_fps=float(fps or frame_rate), trim_start=trim_start,
                frame_rate=frame_rate, warn=warn,
                conditioned=bool(pinned_audio_latents),
            )

        return frames_out, audio_out, audio_sample_rate, seed

    def _minimax_h3_splice_inpaint_range_audio(
        self,
        audio_out,
        input_audio: Optional[bytes],
        sample_rate: int,
        *,
        clip_frames: int,
        start_frame: int,
        end_frame: int,
        source_fps: float,
        trim_start: int,
        frame_rate: float,
        warn: Callable[[str, str], None],
        conditioned: bool = False,
    ):
        """Splice the input clip's own audio back over the PRESERVED spans of
        an already-generated whole-clip track.

        This is the output-level half of ``regenerate_range``, and it runs on
        top of the conditioning half rather than instead of it: the preserved
        spans are pinned as conditioning during generation, but a pinned latent
        still comes back through an audio-VAE round trip, so they are pasted
        here for exactness -- the same "pin for conditioning, paste for
        exactness" split the pixel path already uses. The decoded track covers
        the whole clip; the audio inside ``[start_frame, end_frame)`` stays
        exactly that generated track, and only the two spans outside it
        (``[0, start_frame)`` and ``[end_frame, clip_frames)``) are
        overwritten, through the same ``extract_audio_window`` ->
        ``mux_audio_over_span`` pair (50 ms crossfade confined to the
        generated side) that ``_minimax_h3_outpaint_audio``'s
        ``preserve_input`` branch and ``_minimax_h3_inpaint_pinned_audio``
        both use.

        ``conditioned`` states whether the range that is NOT overwritten by
        this splice was actually generated against the pinned preserved spans
        (``bool(pinned_audio_latents)`` at the call site), because the pin is
        conditional -- ``_minimax_h3_inpaint_pinned_audio`` can return ``None``
        (no audio stream, or window extraction failed) and leave the range
        generating unconditioned even though ``audio_mode`` is still
        ``regenerate_range``. The warning below must say which of the two
        actually happened rather than always claiming the pin held.

        Assumes the caller already handled the no-audio-stream fallback (this
        mode is downgraded to ``regenerate`` before generation when
        ``input_audio`` is empty), so this only defends against it, it does
        not warn about it again.
        """
        import numpy as _np
        from utils.video_utils import extract_audio_window, mux_audio_over_span

        if not input_audio or audio_out is None or not sample_rate:
            return audio_out

        generated = audio_out.numpy() if hasattr(audio_out, "numpy") else _np.asarray(audio_out)
        channels = generated.shape[0]
        full = _np.array(generated, copy=True)
        source_fps = float(source_fps or frame_rate)

        spliced_any = False
        for span_start, span_end in ((0, start_frame), (end_frame, clip_frames)):
            span_frames = int(span_end) - int(span_start)
            if span_frames <= 0:
                continue
            offset_sec = span_start / frame_rate
            target_dur_sec = span_frames / frame_rate
            # Same convention as `_minimax_h3_inpaint_pinned_audio`: pixel
            # frames are not resampled between source and output, so a span's
            # SOURCE duration is stretched/compressed to the OUTPUT frame
            # rate rather than read at 1:1 time.
            src_start_sec = (trim_start + span_start) / source_fps
            src_dur_sec = target_dur_sec * (frame_rate / source_fps)
            if target_dur_sec > 0 and abs(src_dur_sec - target_dur_sec) / target_dur_sec > 0.005:
                warn(f"regenerate_range preserved audio was time-stretched ({src_dur_sec:.3f}s -> "
                     f"{target_dur_sec:.3f}s) because the uploaded clip's frame rate "
                     f"({source_fps:.3f}) differs from MiniMax-H3's fixed {frame_rate:.3f} fps",
                     code="inpaint_video_audio_stretched")
            try:
                window = extract_audio_window(
                    input_audio, src_start_sec, src_dur_sec, target_dur_sec,
                    sample_rate=sample_rate, channels=channels,
                )
            except Exception as exc:
                window = None
                print(f"[MiniMax-H3] vid_inpaint regenerate_range audio window extraction "
                      f"raised: {exc}")
            if window is None:
                # NEVER overwrite with silence on a failure -- leave the
                # generated track already in that span.
                warn("regenerate_range preserved audio window extraction failed; that span was "
                     "left as generated", code="inpaint_video_audio_extract_failed")
                continue
            full = mux_audio_over_span(
                full, window, offset_sec=offset_sec, dur_sec=target_dur_sec,
                sample_rate=sample_rate, crossfade_ms=50.0,
            )
            spliced_any = True

        if spliced_any:
            condition_clause = (
                "audio inside the range was generated with the surrounding track pinned as "
                "conditioning" if conditioned else
                "audio inside the range was generated unconditioned (the preserved-span pin "
                "was not available for this request)"
            )
            warn(
                "inpaint_video_audio_mode='regenerate_range': audio outside the regenerate "
                f"range ({start_frame}-{end_frame}) is the input clip's own track, spliced back "
                f"in with a crossfade; {condition_clause}.",
                code="inpaint_video_audio_range_spliced",
            )

        return torch.from_numpy(full)

    def _minimax_h3_inpaint_pinned_audio(
        self,
        input_audio: Optional[bytes],
        *,
        clip_frames: int,
        source_fps: float,
        trim_start: int,
        frame_rate: float,
        warn: Callable[[str, str], None],
        mode: str = "preserve_input",
        source_start_frame: int = 0,
        fallback_clause: Optional[str] = None,
    ):
        """The clip's own track as the ia2v condition, or None to fall back.

        Cut out of the uploaded clip with the SAME helper the outpaint path
        splices with (``extract_audio_window``: trim, pitch-preserving stretch
        when the source frame rate differs from this model's fixed 24 fps, then
        resample), and handed to ``prepare_pinned_audio`` -- so a track this
        endpoint pins and a track ``/generate/img2vid`` pins are the same object
        by the time the model sees it.

        The window asked for is the model's AUDIO GRID duration, which is a few
        milliseconds longer than the clip's own (124 frames -> 207 latents ->
        5.175 s against 5.167 s): both the encoded slice and the muxed slice
        come out of this one waveform, so the longer of the two is what has to
        be filled.

        ``mode`` names the caller (``"preserve_input"``, ``"regenerate_range"``
        or the outpaint chain's ``"pinned_tail"``) so the warnings below neither
        hardcode a mode the caller may not have selected nor claim the same
        fallback for all of them: on ``preserve_input`` a failure here means the
        WHOLE clip's soundtrack is generated (there is no splice afterward); on
        ``regenerate_range`` only the conditioning PIN is dropped -- the
        regenerate range's audio generates unconditioned, but the preserved
        spans are still spliced back from the input track after generation
        (``_minimax_h3_splice_inpaint_range_audio`` runs regardless). A caller
        whose failure mode is neither of those states its own
        ``fallback_clause``.

        ``source_start_frame`` is where in the (trimmed) SOURCE clip the window
        starts, in that clip's own pixel frames. 0 -- the inpaint callers' case
        -- is the clip's start, i.e. the whole clip's own track; the chain's
        ``pinned_tail`` uses it to cut the tail the continuation pins.

        Returns None -- with a warning -- for every recoverable failure, and the
        caller then generates the audio instead of pinning it.
        """
        from core.models.minimax_h3 import h3_references as refs
        from utils.video_utils import extract_audio_window

        fallback_clause = fallback_clause or (
            "the soundtrack is generated instead" if mode == "preserve_input" else
            "the regenerate range's audio is generated unconditioned (the preserved spans are "
            "still spliced back from the input track after generation)"
        )

        if not input_audio:
            warn(f"inpaint_video_audio_mode='{mode}' was requested but the uploaded clip "
                 f"has no audio stream; {fallback_clause}",
                 code="inpaint_video_no_input_audio")
            return None

        components = self.minimax_h3_components
        sample_rate = int(components.get("audio_sample_rate", 32000))
        required, _grid, _clip = refs.pinned_audio_sample_counts(
            clip_frames, fps=frame_rate, sample_rate=sample_rate,
            latent_rate=float(components.get("audio_latent_rate", 40.0)))
        target_dur_sec = required / float(sample_rate)
        source_fps = float(source_fps or frame_rate)
        # The frames occupy `target_dur_sec` of OUTPUT time but were captured
        # over `frame_rate / source_fps` as much SOURCE time.
        src_dur_sec = target_dur_sec * (frame_rate / source_fps)
        if abs(src_dur_sec - target_dur_sec) / target_dur_sec > 0.005:
            warn(f"{mode} audio was time-stretched ({src_dur_sec:.3f}s -> "
                 f"{target_dur_sec:.3f}s) because the uploaded clip's frame rate "
                 f"({source_fps:.3f}) differs from MiniMax-H3's fixed {frame_rate:.3f} fps",
                 code="inpaint_video_audio_stretched")
        try:
            window = extract_audio_window(
                input_audio, (trim_start + max(0, int(source_start_frame))) / source_fps,
                src_dur_sec, target_dur_sec,
                sample_rate=sample_rate, channels=2,
            )
        except Exception as exc:
            window = None
            print(f"[MiniMax-H3] vid_inpaint audio window extraction raised: {exc}")
        if window is None:
            warn(f"{mode} audio window extraction failed; {fallback_clause}",
                 code="inpaint_video_audio_extract_failed")
            return None
        try:
            return refs.prepare_pinned_audio(
                torch.from_numpy(np.ascontiguousarray(window)), sample_rate,
                num_frames=clip_frames, fps=frame_rate, target_sample_rate=sample_rate,
                latent_rate=float(components.get("audio_latent_rate", 40.0)))
        except ValueError as exc:
            warn(f"{mode} audio could not condition this clip ({exc}); {fallback_clause}",
                 code="inpaint_video_audio_extract_failed")
            return None

    def _generate_minimax_h3(
        self,
        params: Dict[str, Any],
        *,
        keyframes: Sequence[Tuple[Any, Any]] = (),
        references: Sequence[Any] = (),
        input_audio=None,
        pinned_audio_latents: Sequence[int] = (),
        pinned_video_frames: Sequence[int] = (),
        pinned_video_row_indices: Sequence[int] = (),
        pinned_video_source: Optional[np.ndarray] = None,
        label: str = "txt2vid",
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """The one MiniMax-H3 generation path, for all three workflows.

        ``keyframes`` is a sequence of ``(anchor, PIL.Image)`` in PACKED ORDER,
        where anchor is ``"first"``, ``"last"`` or an integer PIXEL frame index
        (see ``h3_pipeline_ops._anchor_rotary_time``). Empty is ``t2va``; one or
        more entries is ``fl2va``.

        ``references`` is a ``ref2va`` request's reference list, also in packed
        order (see ``_generate_ref2vid_minimax_h3``). ``keyframes`` MAY be
        combined with ``references`` (C5): an anchor's rows are then laid out
        AFTER every reference block, from the rotary origin the reference loop
        leaves behind (``build_ref2va_packed_layout``'s ``keyframe_anchors``).
        With no references, ``keyframes`` alone selects ``build_packed_layout``
        (``fl2va``) exactly as before -- the merge is additive, not a
        replacement of the single-track path.

        ``input_audio`` is ia2v: a ``[2, samples]`` float32 waveform, already at
        the audio VAE's rate and at the exact length this clip needs
        (``h3_references.prepare_pinned_audio``). Its VAE encoding becomes the
        clip's OWN audio rows, pinned at ``AUDIO_COND_TIMESTEP`` = 1.0 -- exactly
        clean, since the forward process is ``x_t = t*x0 + (1-t)*noise`` -- so
        the sampler never writes them and the video is generated against a fixed
        soundtrack. THE AUDIO NOISE IS STILL DRAWN and then discarded, which is
        what keeps the video noise bit-identical to a free-audio run at the same
        seed (K0.6's recorded order). Mutually exclusive with ``references``:
        ref2va reaches an audio track through its own reference block, at a
        different rotary offset.

        ``pinned_audio_latents`` narrows ``input_audio`` from a WHOLE-track pin
        to a PARTIAL one: the audio-latent indices (per channel, into
        ``[0, num_audio_latents)``) that are pinned, with every other latent
        generated freely against them. Needs ``input_audio`` (there is nothing
        to substitute without a source) and is empty by default, which
        reproduces the whole-track ia2v behaviour exactly. See
        ``h3_pipeline_ops.build_packed_layout``'s ``pinned_audio_latents`` for
        the permutation this builds on and
        ``scratchpad/minimax_h3_ai_probe_results.md`` for what is measured
        about it -- an audio-only pin, video left free (its own §4); the same
        shape with video ALSO pinned on the same range, which is what
        ``pinned_video_frames``/``pinned_video_row_indices`` combined with this
        parameter produces, was not measured there.

        ``pinned_video_frames`` / ``pinned_video_row_indices`` /
        ``pinned_video_source`` are temporal inpaint:
        the LATENT frames of the clip that are supplied at (near) their true
        value and never denoised, and the trimmed source clip
        (``uint8 [T, H, W, 3]``) they are taken from. Their rows are permuted to
        the head of the video block by ``build_packed_layout``, the substitution
        below FOLLOWS the noise draw exactly as the ia2v one does, and the
        decode un-permutes -- so at one seed the generated frames see the same
        noise a t2va run would. The pinned frames come back through a VAE round
        trip (3.20 RMS, measured); what makes them exact is the caller's paste
        (``_generate_vidinpaint_minimax_h3``), never this path. Mutually
        exclusive with keyframes and references, which claim the same prefix.

        What actually differs between the three is small and local — which
        presentation the conditioner reads, what gets VAE-encoded as
        conditioning, and which layout builder lays the rows out. The draw
        order, the packed denoise loop, the offload sequencing and the decode
        are ONE implementation, which is why they share this function rather
        than having a copy each.

        Returns ``(frames, audio, audio_sample_rate, actual_seed)`` — see the
        module docstring.
        """
        from core.models.minimax_h3 import h3_pipeline_ops as ops
        from core.models.minimax_h3 import h3_references as refs
        from core.models.minimax_h3 import prompt_cache
        from core.models.minimax_h3.loader import minimax_h3_latent_frames

        components = getattr(self, "minimax_h3_components", None)
        if not components:
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")
        if pinned_video_frames is None:
            pinned_video_frames = ()
        if pinned_video_row_indices is None:
            pinned_video_row_indices = ()
        if pinned_audio_latents is None:
            pinned_audio_latents = ()
        pinned_video_frames = tuple(pinned_video_frames)
        pinned_video_row_indices = tuple(pinned_video_row_indices)
        pinned_audio_latents = tuple(int(t) for t in pinned_audio_latents)
        if pinned_video_frames and pinned_video_row_indices:
            raise RuntimeError(
                "MiniMax-H3 cannot combine pinned video frames with pinned video row indices: "
                "pass one video pinning scheme only.")
        if (pinned_video_frames or pinned_video_row_indices) and keyframes:
            raise RuntimeError(
                "MiniMax-H3 cannot combine pinned video pins with keyframes: the pin re-uses the "
                "video block's conditioning prefix for rows of the clip itself, and an anchor "
                "reserves that same prefix for rows of its own.")
        if pinned_video_row_indices and references:
            # L8 (audit): deliberately still a RuntimeError (500), unlike the
            # `pinned_video_frames`+references check the route/backend both
            # already turn into a 400 before this function is ever reached --
            # `pinned_video_row_indices` is never built from a raw request
            # directly (it only exists after spatial-mask processing, which
            # the route and `_generate_vidinpaint_minimax_h3`'s own re-check
            # both already refuse alongside references), so hitting this line
            # means an internal caller bypassed BOTH of those layers, which is
            # a programming error, not a request a user can construct.
            raise RuntimeError(
                "MiniMax-H3 cannot combine spatial-mask row pins with references: the extended "
                "ref2va layout builder (h3_pipeline_ops.build_ref2va_packed_layout) only carries a "
                "frame-level pin (pinned_video_frames) alongside references, not a row-level one.")
        # B-2 (minimax_h3_inpaint_refs_design.md, Option B): a frame-level pin
        # and references can now share the conditioning prefix, but ONLY for
        # `_generate_vidinpaint_minimax_h3`'s own pin (`label == "vid_inpaint"`)
        # on the ref2va partition -- `build_ref2va_packed_layout` (h3_pipeline_
        # ops.py) lays out [reference/anchor rows | pinned target rows | free
        # target rows] for that request shape specifically. Scoped to `label`
        # rather than to `variant` alone so `_generate_vidoutpaint_minimax_h3`'s
        # OWN pin (`continuation_mode="pinned_tail"`) cannot silently reuse
        # this relaxation if its own (currently exhaustive, but independent)
        # refusals ever change -- that combination has never been gated,
        # measured, or requested, and stays refused regardless of variant.
        if pinned_video_frames and references and not (
                label == "vid_inpaint" and (components.get("variant") or "").lower() == "ref2va"):
            raise RuntimeError(
                "MiniMax-H3 cannot combine pinned video pins with references on this request: "
                "the pin and a reference share the same conditioning-prefix mechanism, and only "
                "temporal inpaint's own pin (label='vid_inpaint') on the ref2va partition's "
                "layout builder carries both at once.")
        if (pinned_video_frames or pinned_video_row_indices) and pinned_video_source is None:
            raise RuntimeError(
                "MiniMax-H3 temporal inpaint needs the source clip the pinned content is taken "
                "from: pinned video frames/rows name conditioning content, and "
                "pinned_video_source supplies the pixels they are encoded from.")
        # `label == "vid_inpaint"` is temporal inpaint's OWN audio pin
        # (preserve_input/regenerate_range, `_generate_vidinpaint_minimax_h3`)
        # reusing this same `input_audio` parameter for a different reason
        # than img2vid's ia2v: it preserves the CLIP's OWN track outside the
        # regenerate range, not an externally supplied steering track, and it
        # is the design's arm P shape (Gate registration (B) §6.2) -- so it
        # stays allowed alongside references. Every other caller of
        # `input_audio` (img2vid's true ia2v) keeps the original exclusion:
        # a reference soundtrack already occupies its own block, ia2v pins
        # the TARGET's audio rows, and the two would collide.
        if input_audio is not None and references and label != "vid_inpaint":
            raise RuntimeError(
                "MiniMax-H3 cannot pin an input audio track on a ref2va request: a reference "
                "soundtrack already occupies its own block at its own rotary offset, while ia2v "
                "pins the TARGET's audio rows. Send the track as an audio reference instead.")
        if pinned_audio_latents and input_audio is None:
            raise RuntimeError(
                "MiniMax-H3 partial audio pin needs input_audio: pinned_audio_latents names which "
                "of its encoded rows are kept, and there is no track to encode without one.")
        if references and (components.get("variant") or "") != "ref2va":
            raise RuntimeError(
                f"ref2vid needs the MiniMax-H3 ref2va transformer, but the loaded checkpoint is "
                f"{components.get('variant') or 'an unidentified variant'} "
                f"({components.get('dit_path')}). Load "
                f"diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors -- reference "
                f"conditioning is a trained behaviour of that partition alone, and the two files "
                f"are otherwise indistinguishable, so running it here would silently produce a bad "
                f"video rather than fail.")
        # Defensive re-check of the route's gate, for a caller that bypasses it.
        # Prompt-only requests fall through; everything else is refused when the
        # loaded encoder is a converted text-only one.
        if (keyframes or references or input_audio is not None
                or pinned_video_frames or pinned_video_row_indices):
            from api.generation_utils import resolve_minimax_h3_text_only_te_gate
            resolve_minimax_h3_text_only_te_gate(
                components,
                workflow=f"{label} conditioning",
                has_vision_references=any(
                    getattr(reference, "kind", "") in ("image", "video")
                    for reference in references),
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device)

        # Every default this backend reads comes from the ONE resolver in
        # `param_defaults` (a route-level or backend-level literal is exactly
        # what that file exists to prevent). The route already resolved the
        # request against this same map; resolving it again here only covers an
        # internal caller (a smoke script, a training preview) that built the
        # dict by hand.
        from api.param_defaults import video_defaults_for_arch
        defaults = video_defaults_for_arch("minimax_h3")

        prompt = (params.get("prompt") or "").strip()
        width = int(params.get("width", defaults["width"]))
        height = int(params.get("height", defaults["height"]))
        num_frames = int(params.get("num_frames", defaults["num_frames"]))
        num_inference_steps = int(params.get("num_inference_steps", defaults["num_inference_steps"]))
        audio_enable = bool(params.get("audio_enable", defaults["audio_enable"]))

        # The route has already validated and, where needed, snapped these (see
        # `TemporalSpec` + the txt2vid route). Re-derive the geometry rather than
        # trusting a caller-supplied latent shape: the VAE's chunking is what
        # decides it.
        latent_frames = minimax_h3_latent_frames(num_frames)
        spatial = int(components.get("vae_scale_factor_spatial", 16))
        latent_height = height // spatial
        latent_width = width // spatial
        num_audio_latents = ops.audio_latent_frames(num_frames, fps=float(components.get("fps", 24.0)))

        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2 ** 32 - 1)

        # Visual conditioning anchors, in packed order, put onto the generation
        # canvas here (the VAE encodes exactly what it is given, and the packed
        # layout reserves `rows_per_frame` rows per anchor). The two anchors are
        # NOT treated the same way -- see `_minimax_h3_fit_keyframe`.
        anchors = tuple(anchor for anchor, _image in keyframes)
        keyframe_pixels = [
            np.asarray(self._minimax_h3_fit_keyframe(image, width, height, anchor), dtype=np.uint8)
            for anchor, image in keyframes
        ]

        # ---- ref2va: normalise the references onto the model's own rates and
        # resolutions FIRST. Everything downstream reads the normalised media:
        # the presentation labels it, the VAEs encode it, and the layout is
        # built from the shapes those encodes produce.
        normalized_references: list = []
        if references:
            image_canvas = None
            reference_image_size = str(params.get("reference_image_size")
                                       or defaults.get("reference_image_size", "max")).lower()
            if reference_image_size == "match":
                image_canvas = (height, width)
            normalized_references = refs.normalize_references(
                references,
                num_frames=num_frames,
                fps=float(components.get("fps", 24.0)),
                audio_sample_rate=int(components.get("audio_sample_rate", 32000)),
                image_canvas=image_canvas,
            )

        print(f"[MiniMax-H3] {label}: {width}x{height} num_frames={num_frames} "
              f"(latent {latent_frames}x{latent_height}x{latent_width}, "
              f"{num_audio_latents} audio latents/ch) steps={num_inference_steps} "
              f"seed={seed} audio={audio_enable} "
              f"conditions={list(anchors) if anchors else 'none (t2va)'}"
              + (f" references=[{refs.describe_references(normalized_references)}]"
                 if normalized_references else "")
              + (f" input_audio={int(input_audio.shape[-1])} sample(s) "
                 f"{(str(len(pinned_audio_latents)) + '/' + str(num_audio_latents) + ' latents') if pinned_audio_latents else 'whole track'} pinned"
                 if input_audio is not None else ""))

        phase_peaks: Dict[str, float] = {}
        self._minimax_h3_reset_peak_vram()
        wall_start = time.perf_counter()

        # ---- Phase 1: text encode (layer-streamed; nothing else on the GPU) ----
        text_encoder = components.get("text_encoder")
        tokenizer = components.get("tokenizer")
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "MiniMax-H3 is missing its text encoder or tokenizer, so a prompt cannot be "
                "encoded. Load the model with its text_encoders/ and official/tokenizer/ trees.")
        encode_start = time.perf_counter()
        text_token_tags = None
        prompt_cache_hit = False
        with generation_timer.phase("text_encode"):
            if normalized_references:
                # The ref2va PRESENTATION: a label and a vision block per
                # reference, then the prompt verbatim. The vision blocks' rows
                # are tagged VIDEO, not text, which is what the transformer's
                # AdaLN modulation keys off -- so the tags travel with the
                # embeddings into the layout. NOT prompt-cached: see
                # `core.models.minimax_h3.prompt_cache`'s module docstring for
                # why this branch is out of that cache's scope.
                token_ids, text_token_tags, vision_inputs = refs.build_ref2va_presentation(
                    tokenizer, components.get("processor"), prompt, normalized_references,
                    fps=float(components.get("fps", 24.0)),
                    text_tag=ops.TEXT_TAG, video_tag=ops.VIDEO_TAG,
                )
                prompt_embeds_cpu = ops.encode_presentation(
                    text_encoder, token_ids, vision_inputs=vision_inputs, device=device,
                    dtype=torch.bfloat16, layer=ops.TEXT_ENCODER_LAYER,
                )
                num_text_tokens = len(token_ids)
                # A substituted encoder's hidden state is not conditioning until it
                # is projected: this is the one seam where that is true, and the
                # projection is a per-token map, so `num_text_tokens` is unchanged.
                prompt_embeds_cpu = self._minimax_h3_project_prompt_embeds(
                    prompt_embeds_cpu, components, params, device=device)
            else:
                te_path = str(components.get("text_encoder_path") or "")
                te_projection_path = (
                    str((components.get("te_projection") or {}).get("path") or "") or None)

                def _encode_and_project_prompt():
                    embeds, tokens = ops.encode_prompt(
                        text_encoder, tokenizer, prompt, device=device,
                        dtype=torch.bfloat16, layer=ops.TEXT_ENCODER_LAYER,
                    )
                    # Same projection seam as the ref2va branch above, applied
                    # BEFORE the pair reaches the cache: the cache key is
                    # (encoder, projection, prompt), so what it stores must
                    # already be that key's final, projection-dependent
                    # conditioning.
                    embeds = self._minimax_h3_project_prompt_embeds(
                        embeds, components, params, device=device)
                    return embeds, tokens

                ((prompt_embeds_cpu, num_text_tokens), prompt_cache_hit) = (
                    prompt_cache.get_or_encode_prompt(
                        te_path, te_projection_path, prompt,
                        int(components["transformer_config"]["text_dim"]),
                        _encode_and_project_prompt))
                if prompt_cache_hit:
                    # A hit skips `_encode_and_project_prompt` (and therefore
                    # `_minimax_h3_project_prompt_embeds`) entirely, but that
                    # function's params provenance and substitution warning are
                    # per-GENERATION, not per-prompt -- report them here so a
                    # cache hit still surfaces them.
                    self._minimax_h3_report_te_provenance(components, params)
        self._minimax_h3_empty_cache()
        text_allocated, text_reserved, text_peak = self._minimax_h3_vram_stats()
        phase_peaks["text_encode"] = text_peak
        print(f"[MiniMax-H3] prompt encoded: {num_text_tokens} token(s) in "
              f"{time.perf_counter() - encode_start:.1f}s "
              f"(VRAM allocated {text_allocated:.2f} GB, reserved {text_reserved:.2f} GB, "
              f"phase peak {text_peak:.2f} GB)"
              + (" (cache hit)" if prompt_cache_hit else ""))
        self._minimax_h3_reset_peak_vram()

        # ---- Visual (and, for ref2va, audio) conditioning: VAE-encode it ----
        # On the VAEs, BEFORE the DiT is staged: the two do not fit together.
        # The conditioning latents' shapes are what the layout reserves rows for
        # and what the noise draws are sized from, so this runs first for every
        # conditioned workflow.
        patch_size = tuple(components["transformer_config"]["patch_size"])
        latent_channels = int(components.get("latent_channels", 24))
        # `condition_latents` is references THEN anchors, in the order
        # `build_condition_rows`'s draw and `build_ref2va_packed_layout`'s row
        # placement both read it in (C5). `reference_condition_latents` is the
        # subset the ref2va layout call describes per-reference shapes from --
        # it must not include the anchors, which that call places itself.
        condition_latents: list = []
        reference_condition_latents: list = []
        anchor_condition_latents: list = []
        audio_condition_rows: list = []
        # The ia2v track's rows, once encoded. `None` (not an empty list) is what
        # every other path downstream tests against: an empty list would be
        # indistinguishable from "a track that encoded to nothing".
        pinned_audio_rows = None
        # Temporal inpaint's source rows, in FRAME-MAJOR order (the permutation
        # is applied to the whole video block after the noise draw, not here).
        pinned_video_rows = None
        if pinned_video_source is not None:
            clip_start = time.perf_counter()
            with generation_timer.phase("condition_encode"):
                self._minimax_h3_move("vae", torch_device)
                try:
                    # ONE encode of the whole clip, through the same recipe a ref2va
                    # video reference takes -- the T > 1 branch, so the temporally
                    # chunked latents line up frame for frame with the target grid.
                    clip_latents = ops.encode_visual_condition(
                        components["vae"], np.asarray(pinned_video_source, dtype=np.uint8),
                        latents_mean=components["latents_mean"],
                        latents_std=components["latents_std"],
                        pixel_mean=components["pixel_mean"],
                        pixel_std=components["pixel_std"],
                        device=device,
                    )
                finally:
                    self._minimax_h3_move("vae", "cpu")
                    self._minimax_h3_empty_cache()
            if tuple(clip_latents.shape[2:5]) != (latent_frames, latent_height, latent_width):
                raise RuntimeError(
                    f"MiniMax-H3 temporal inpaint encoded its source clip to "
                    f"{tuple(clip_latents.shape[2:5])} latent frames/height/width where this "
                    f"request's grid is {(latent_frames, latent_height, latent_width)} -- the clip "
                    f"handed over is not the clip the layout is built from.")
            pinned_video_rows = ops.patchify_video_latents(clip_latents, patch_size)[0]
            del clip_latents
            print(f"[MiniMax-H3] source clip encoded in {time.perf_counter() - clip_start:.1f}s: "
                  f"{pinned_video_source.shape[0]} frame(s) -> {pinned_video_rows.shape[0]} row(s) "
                  f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")
        if keyframe_pixels or normalized_references:
            cond_start = time.perf_counter()
            with generation_timer.phase("condition_encode"):
                self._minimax_h3_move("vae", torch_device)
                try:
                    if normalized_references:
                        reference_condition_latents = refs.encode_reference_visuals(
                            components["vae"], normalized_references,
                            latents_mean=components["latents_mean"],
                            latents_std=components["latents_std"],
                            pixel_mean=components["pixel_mean"],
                            pixel_std=components["pixel_std"],
                            device=device,
                        )
                    if keyframe_pixels:
                        anchor_condition_latents = ops.encode_condition_images(
                            components["vae"], keyframe_pixels,
                            latents_mean=components["latents_mean"],
                            latents_std=components["latents_std"],
                            pixel_mean=components["pixel_mean"],
                            pixel_std=components["pixel_std"],
                            device=device,
                        )
                    condition_latents = reference_condition_latents + anchor_condition_latents
                finally:
                    self._minimax_h3_move("vae", "cpu")
                    self._minimax_h3_empty_cache()
                if any(getattr(reference, "has_audio", False) for reference in normalized_references):
                    # The audio VAE is 0.6 GB and is the ONLY component needed here,
                    # so it is staged on its own after the video VAE has gone back.
                    self._minimax_h3_move("audio_vae", torch_device)
                    try:
                        audio_condition_rows = refs.encode_reference_audio_rows(
                            components["audio_vae"], normalized_references,
                            latents_mean=components["audio_latents_mean"],
                            latents_std=components["audio_latents_std"],
                            audio_latent_channels=int(components.get("audio_latent_channels", 32)),
                            device=device,
                        )
                    finally:
                        self._minimax_h3_move("audio_vae", "cpu")
                        self._minimax_h3_empty_cache()
            print(f"[MiniMax-H3] conditioning encoded in "
                  f"{time.perf_counter() - cond_start:.1f}s: "
                  f"{len(condition_latents)} visual, {len(audio_condition_rows)} audio "
                  f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")

        # ---- ia2v: the pinned track, through the SAME audio-VAE recipe a
        # reference soundtrack takes (posterior MODE, never a sample, then the
        # per-channel normalisation), so the two conditioning paths cannot
        # drift. The clip's audio grid is `num_audio_latents` latents per
        # channel and the encoder emits one per 800 samples, so exactly that
        # many samples are handed over -- the rest of the prepared waveform is
        # the mux's, not the model's. ----
        if input_audio is not None:
            audio_start_time = time.perf_counter()
            _required, grid_samples, _clip_samples = refs.pinned_audio_sample_counts(
                num_frames,
                fps=float(components.get("fps", 24.0)),
                sample_rate=int(components.get("audio_sample_rate", 32000)),
                latent_rate=float(components.get("audio_latent_rate", 40.0)),
            )
            with generation_timer.phase("condition_encode"):
                self._minimax_h3_move("audio_vae", torch_device)
                try:
                    pinned_audio_rows = refs.encode_reference_audio_rows(
                        components["audio_vae"],
                        [refs.MiniMaxH3Reference(
                            kind="audio", audio=input_audio[:, :grid_samples],
                            sample_rate=int(components.get("audio_sample_rate", 32000)),
                            label="input_audio")],
                        latents_mean=components["audio_latents_mean"],
                        latents_std=components["audio_latents_std"],
                        audio_latent_channels=int(components.get("audio_latent_channels", 32)),
                        device=device,
                    )[0]
                finally:
                    self._minimax_h3_move("audio_vae", "cpu")
                    self._minimax_h3_empty_cache()
            expected = num_audio_latents * ops.AUDIO_CHANNELS
            if pinned_audio_rows.shape[0] != expected:
                raise RuntimeError(
                    f"MiniMax-H3 input audio encoded to {pinned_audio_rows.shape[0]} row(s) where "
                    f"this clip's audio grid is {expected} ({num_audio_latents} latent(s) x "
                    f"{ops.AUDIO_CHANNELS} channels) -- the prepared waveform does not match the "
                    f"geometry the layout is built from.")
            print(f"[MiniMax-H3] input audio encoded in "
                  f"{time.perf_counter() - audio_start_time:.1f}s: "
                  f"{grid_samples} sample(s) -> {pinned_audio_rows.shape[0]} pinned row(s) "
                  f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")

        condition_allocated, condition_reserved, condition_peak = self._minimax_h3_vram_stats()
        phase_peaks["condition_encode"] = condition_peak
        print(f"[MiniMax-H3] condition phase VRAM: allocated {condition_allocated:.2f} GB, "
              f"reserved {condition_reserved:.2f} GB, phase peak {condition_peak:.2f} GB")
        self._minimax_h3_reset_peak_vram()

        # ---- Layout + noise (drawn on the generation device, before staging) ----
        # Manual accumulate (`generation_timer.add()` below) rather than a
        # `with` block: this ~250-line span already has its own peak-VRAM
        # reset/read pair (`phase_peaks["prepare"]`).
        prepare_time_start = time.perf_counter()
        if normalized_references:
            layout = ops.build_ref2va_packed_layout(
                text_token_tags,
                [(reference.kind, reference.has_audio) for reference in normalized_references],
                [tuple(latent.shape[2:5]) for latent in reference_condition_latents],
                [rows.shape[0] for rows in audio_condition_rows],
                latent_frames, latent_height, latent_width, num_audio_latents,
                patch_size=patch_size,
                # C5: anchors placed after the reference blocks. Empty when
                # `keyframes` is empty, which reproduces the pre-C5 layout.
                keyframe_anchors=anchors,
                # The temporal-inpaint pin, carried through unchanged from the
                # caller -- opened for `label == "vid_inpaint"` on this
                # partition (`resolve_minimax_h3_inpaint_reference_gate`'s
                # `ref2va` row).
                pinned_video_frames=tuple(pinned_video_frames),
                # Same whole-track shorthand as the fl2va branch below, and
                # for the SAME reason: `preserve_input`'s whole-track pin
                # must actually be counted as conditioning here too, or the
                # clean re-encoded track substituted in below (`pinned_audio_
                # rows`) gets read as noise at t=T by a layout that thinks
                # every target audio row is still free (H1: this was missing
                # entirely -- ref2va had no `pin_target_audio` at all, so
                # `preserve_input` silently corrupted the audio conditioning
                # on this partition with no crash and no warning).
                pin_target_audio=pinned_audio_rows is not None and not pinned_audio_latents,
                pinned_audio_latents=tuple(pinned_audio_latents),
                device=torch_device,
            )
        else:
            layout_kwargs = {
                "patch_size": patch_size,
                "keyframe_anchors": anchors,
                "pinned_video_frames": tuple(pinned_video_frames),
                # ia2v needs no rows of its own: the target audio rows are
                # already on the target's clock, and this flag only moves them
                # from "generated" to "conditioning" in the row-timestep plan.
                # A PARTIAL pin (pinned_audio_latents) is a different flag on
                # the same builder -- the two are mutually exclusive there --
                # so the whole-track shorthand only applies when nothing was
                # asked to stay free.
                "pin_target_audio": pinned_audio_rows is not None and not pinned_audio_latents,
                "device": torch_device,
            }
            if pinned_video_row_indices:
                layout_kwargs["pinned_video_row_indices"] = tuple(pinned_video_row_indices)
            if pinned_audio_latents:
                layout_kwargs["pinned_audio_latents"] = pinned_audio_latents
            layout = ops.build_packed_layout(
                num_text_tokens, latent_frames, latent_height, latent_width, num_audio_latents,
                **layout_kwargs,
            )
        row_counts = ops.packed_row_counts(layout)
        params["minimax_h3_packed_rows"] = row_counts["total"]
        params["minimax_h3_conditioning_rows"] = (
            row_counts["condition_video"] + row_counts["condition_audio"])
        target_rows = row_counts["target_video"] + row_counts["target_audio"]
        expansion = row_counts["total"] / max(1, row_counts["text"] + target_rows)
        print(
            "[MiniMax-H3] packed rows: "
            f"text={row_counts['text']}, "
            f"video={row_counts['target_video']} target + {row_counts['condition_video']} condition, "
            f"audio={row_counts['target_audio']} target + {row_counts['condition_audio']} condition, "
            f"total={row_counts['total']} ({expansion:.2f}x target-only sequence)"
        )
        # Sequence length is a WARNING, never a refusal (owner correction,
        # design doc §6). See `minimax_h3_inpaint_reference_row_count_warning`
        # for what the numbers mean.
        if normalized_references and label == "vid_inpaint":
            from api.generation_status import add_warning
            from api.generation_utils import minimax_h3_inpaint_reference_row_count_warning
            _message, _code = minimax_h3_inpaint_reference_row_count_warning(
                row_counts, num_references=len(normalized_references),
                num_pinned_video_rows=int(layout.get("num_pinned_video_rows", 0) or 0),
                num_pinned_audio_rows=int(layout.get("num_pinned_audio_rows", 0) or 0),
            )
            add_warning(_message, code=_code)
        generator = torch.Generator(device=device).manual_seed(seed)
        # ONE draw per visual condition FIRST, at that condition's own latent
        # shape (they do NOT share one on ref2va), then the video noise, then
        # the audio noise -- the recorded order (K0.6). A condition that is not
        # drawn, or drawn later, changes the video for the same seed. Reference
        # SOUNDTRACKS take no draw at all: they condition clean.
        condition_noises, video_noise, audio_rows = ops.draw_noise(
            generator,
            video_latent_shape=(1, latent_channels, latent_frames, latent_height, latent_width),
            num_audio_latents=num_audio_latents,
            condition_shapes=tuple(tuple(latent.shape) for latent in condition_latents),
            device=device,
            audio_latent_channels=int(components.get("audio_latent_channels", 32)),
        )
        video_rows = ops.patchify_video_latents(video_noise, patch_size)[0]
        del video_noise

        if pinned_video_rows is not None and (pinned_video_frames or pinned_video_row_indices):
            # Substitute AFTER the draw, for the reason the audio substitution
            # below states: the draw order is what makes the generated frames'
            # noise the same at one seed whether or not anything is pinned. Each
            # pinned row is noised to VISUAL_COND_TIMESTEP with ITS OWN row of
            # that draw, through the scheduler's own forward process -- the
            # recipe `build_condition_rows` uses for a keyframe anchor.
            if pinned_video_row_indices:
                pin_rows = torch.tensor(
                    pinned_video_row_indices,
                    dtype=torch.long,
                    device=video_rows.device,
                )
            else:
                rows_per_frame = int(layout["rows_per_frame"])
                pin_rows = torch.cat([
                    torch.arange(frame * rows_per_frame, (frame + 1) * rows_per_frame)
                    for frame in sorted(int(f) for f in pinned_video_frames)
                ]).to(video_rows.device)
            video_rows = ops.pin_video_rows(
                video_rows,
                pinned_video_rows,
                tuple(int(row) for row in pin_rows.tolist()),
                components["scheduler"],
                ops.VISUAL_COND_TIMESTEP,
            )
            del pinned_video_rows
            # Frame-major -> packed. The layout permuted `video_indices` the same
            # way, and the transformer addresses rows by that index list, so the
            # two permutations cancel inside the forward.
            video_rows = video_rows[layout["video_row_permutation"].to(video_rows.device)]
            if pinned_video_row_indices:
                pin_description = "spatial mask rows"
            else:
                pin_description = (
                    f"latent frames {list(pinned_video_frames)[:4]}"
                    f"{'...' if len(pinned_video_frames) > 4 else ''}"
                )
            print(f"[MiniMax-H3] temporal inpaint: {pin_rows.numel()} of {video_rows.shape[0]} "
                  f"video row(s) pinned at t={ops.VISUAL_COND_TIMESTEP} ({pin_description})")

        if pinned_audio_rows is not None:
            # THE AUDIO DRAW ABOVE HAPPENED AND IS DISCARDED HERE, deliberately.
            # `draw_noise` is one generator drawing three things in a recorded
            # order (K0.6), so skipping the audio draw would not save anything
            # -- it would move the generator's state and change the VIDEO noise
            # of every ia2v request. Substituting after the fact is what makes a
            # pinned run and a free-audio run bit-identical in video noise at
            # the same seed; `minimax_h3_ia2v_test` asserts exactly that.
            if pinned_audio_rows.shape != audio_rows.shape:
                raise RuntimeError(
                    f"MiniMax-H3 input audio packs into {tuple(pinned_audio_rows.shape)} where "
                    f"this clip's audio rows are {tuple(audio_rows.shape)}.")
            if pinned_audio_latents:
                # PARTIAL pin: only the rows named by `pinned_audio_latents` are
                # the clean source; every other row keeps its own draw.
                # `substitute_and_permute_audio_rows` does the substitution in
                # ORIGINAL (unpermuted, channel-major) row space -- exactly
                # like `pin_video_rows` above -- so the free rows' draw order
                # (K0.6) is untouched, then applies the layout's DRAW-time
                # permutation (`audio_row_permutation`, NOT `audio_row_order`
                # -- that one is the decode-time inverse, below), so a
                # permuted row and its permuted index address the same
                # sequence slot.
                num_pinned_rows = len(pinned_audio_latents) * ops.AUDIO_CHANNELS
                audio_rows = ops.substitute_and_permute_audio_rows(
                    audio_rows, pinned_audio_rows, pinned_audio_latents, num_audio_latents,
                    layout["audio_row_permutation"],
                )
                print(f"[MiniMax-H3] partial audio pin: {num_pinned_rows} of {audio_rows.shape[0]} "
                      f"audio row(s) pinned at t={ops.AUDIO_COND_TIMESTEP} "
                      f"(latents {list(pinned_audio_latents)[:4]}"
                      f"{'...' if len(pinned_audio_latents) > 4 else ''})")
            else:
                # Whole-track pin (ia2v): every row is the source, so there is
                # nothing to substitute into a draw and no permutation to apply
                # -- bitwise unchanged from the pre-partial-pin behaviour.
                audio_rows = pinned_audio_rows.to(audio_rows.device, audio_rows.dtype)
            del pinned_audio_rows

        if condition_latents:
            condition_rows = ops.build_condition_rows(
                components["scheduler"], condition_latents, condition_noises,
                patch_size=patch_size,
            ).to(video_rows.device, video_rows.dtype)
            # `num_condition_video_rows` also counts the pin's own rows on
            # this builder (B-2), which `condition_rows` never includes -- the
            # pin substitutes into `video_rows` itself, above, not through
            # `condition_latents`.
            expected_rows = (int(layout["num_condition_video_rows"])
                             - int(layout.get("num_pinned_video_rows", 0) or 0))
            if condition_rows.shape[0] != expected_rows:
                raise RuntimeError(
                    f"MiniMax-H3 conditioning produced {condition_rows.shape[0]} row(s) where the "
                    f"packed layout reserves {expected_rows} -- the conditioning latents do not "
                    f"match the geometry the layout was built from.")
            # The conditioning rows LEAD the video block; the loop protects them
            # by never writing the first `num_condition_video_rows` entries.
            video_rows = torch.cat([condition_rows, video_rows], dim=0)
            del condition_rows
        if audio_condition_rows:
            reference_audio_rows = torch.cat(
                [rows.to(audio_rows.device, audio_rows.dtype) for rows in audio_condition_rows])
            # Same pin-share correction as the video branch above.
            expected_audio_rows = (int(layout["num_condition_audio_rows"])
                                   - int(layout.get("num_pinned_audio_rows", 0) or 0))
            if reference_audio_rows.shape[0] != expected_audio_rows:
                raise RuntimeError(
                    f"MiniMax-H3 reference soundtracks pack into {reference_audio_rows.shape[0]} "
                    f"row(s) where the packed layout reserves {expected_audio_rows}.")
            # Same invariant on the audio side: reference rows lead, and the
            # loop never writes them, so a soundtrack rides through at t = 1.0.
            audio_rows = torch.cat([reference_audio_rows, audio_rows], dim=0)
            del reference_audio_rows
        del condition_latents, reference_condition_latents, anchor_condition_latents
        del audio_condition_rows, condition_noises
        generation_timer.add("prepare", time.perf_counter() - prepare_time_start)

        # ---- Phase 2: denoise (DiT resident) ----
        self._minimax_h3_assert_components_off_cuda("text_encoder", "vae", "audio_vae", "image_vae")
        prepare_allocated, prepare_reserved, prepare_peak = self._minimax_h3_vram_stats()
        phase_peaks["prepare"] = prepare_peak
        print(f"[MiniMax-H3] packed preparation VRAM: allocated {prepare_allocated:.2f} GB, "
              f"reserved {prepare_reserved:.2f} GB, phase peak {prepare_peak:.2f} GB")
        self._minimax_h3_reset_peak_vram()
        prompt_embeds = prompt_embeds_cpu.to(torch_device)
        denoise_start = time.perf_counter()
        # LoRA wraps Linear modules on the RAW transformer, before staging
        # replaces `minimax_h3_components["transformer"]` with a block-loop
        # wrapper (or hands `swap_linears_to_w4a8`/TransformerBlockOffloader a
        # tree containing a non-nn.Linear module, which they reject). Timed as
        # its own phase: this is where the 21 GB DiT actually moves onto the
        # GPU (`_ensure_minimax_h3_swap_and_offload`), previously unaccounted
        # for between the `text_encode` and `denoise` phases.
        # Staging is INSIDE the try: a raise between the LoRA wrap and the
        # denoise loop (a refused LoRA, an out-of-range block-skip set, an OOM
        # during the device move) otherwise left the transformer wrapped and
        # GPU-resident for the NEXT generation.
        transformer = None
        offloader = None
        probe_records = None
        substep_reporter = None
        try:
            with generation_timer.phase("stage_transformer"):
                lora_configs = params.get("loras") or []
                if lora_configs:
                    self._load_lora_minimax_h3(lora_configs, params)
                # Staging owns the device move: with block swap it places the block stack
                # itself rather than moving all 21 GB on and some of it back off.
                transformer, offloader, probe_records = self._ensure_minimax_h3_swap_and_offload(
                    params, torch_device)
                self._minimax_h3_apply_attention_backend(transformer, params)
                # ~150s per step means the per-step callback alone looks like a hang, so
                # block-level forward hooks tick progress from inside the step. Removed
                # in the `finally` below: a surviving hook would fire on the next
                # generation against this generation's callback.
                substep_reporter = attach_block_substep_hooks(
                    transformer, progress_callback, label="MiniMax-H3")
            with generation_timer.phase("denoise"):
                video_rows, audio_rows = ops.denoise(
                    transformer,
                    components["scheduler"],
                    components["audio_scheduler"],
                    prompt_embeds=prompt_embeds,
                    layout=layout,
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    num_inference_steps=num_inference_steps,
                    device=device,
                    progress_callback=progress_callback,
                    substep_reporter=substep_reporter,
                    step_callback=step_callback,
                    # Preview geometry: the loop hands the callback LATENTS, not
                    # packed rows, so it needs the shape to unpatchify into.
                    preview_latent_shape=(latent_frames, latent_height, latent_width),
                    # Non-None only for a pinned request, and read only by the
                    # preview: with pinned frames the conditioning prefix IS clip
                    # content, so the preview shows the whole clip in frame order.
                    video_row_order=layout["video_row_order"],
                    latent_channels=int(components.get("latent_channels", 24)),
                    patch_size=tuple(components["transformer_config"]["patch_size"]),
                    spectrum_params=params,
                    block_swap_on=int(params.get("blocks_to_swap", 0) or 0) > 0,
                )
        finally:
            if substep_reporter is not None:
                substep_reporter.close()
            # Mirrors `stage_transformer` above: the DiT's round trip back to
            # the CPU is real seconds too (module docstring, "WHY THE STAGING
            # IS STRICTLY SEQUENTIAL"), previously folded silently into
            # whatever the caller's own wall-clock measurement was.
            with generation_timer.phase("unstage_transformer"):
                # Unconditional: a partial application that then raised wraps
                # modules without returning a count, and a second unload is a
                # no-op once `_minimax_h3_lora_wrapped_keys` is empty.
                self._unload_lora_minimax_h3()
                # Back to the CPU before ANY decode: the video VAE's ViT decoder is
                # the second-largest allocation of the generation and the two do not
                # fit together. This also unwraps the block-loop wrapper and cleans
                # up the block offloader.
                self._unstage_minimax_h3_transformer(offloader)
                del transformer
                del prompt_embeds
                self._minimax_h3_empty_cache()
            self._minimax_h3_dump_residual_probe(
                probe_records, prompt=prompt, num_frames=num_frames,
                seed=seed, num_inference_steps=num_inference_steps,
            )
        denoise_seconds = time.perf_counter() - denoise_start
        denoise_allocated, denoise_reserved, peak_after_denoise = self._minimax_h3_vram_stats()
        phase_peaks["denoise"] = peak_after_denoise
        print(f"[MiniMax-H3] denoise: {num_inference_steps} step(s) in {denoise_seconds:.1f}s "
              f"({denoise_seconds / max(num_inference_steps, 1):.2f}s/step, "
              f"VRAM allocated {denoise_allocated:.2f} GB, reserved {denoise_reserved:.2f} GB, "
              f"phase peak {peak_after_denoise:.2f} GB)")

        self._minimax_h3_assert_components_off_cuda("transformer")

        if not torch.isfinite(video_rows).all():
            raise RuntimeError("MiniMax-H3 produced non-finite video latents.")

        # ---- Phase 3: decode ----
        n_cond_video = layout["num_condition_video_rows"]
        n_cond_audio = layout["num_condition_audio_rows"]
        video_row_order = layout["video_row_order"]
        # `num_condition_video_rows` is reference/anchor rows PLUS a pin's own
        # rows (ref2va's extended builder ADDS them; fl2va's builder REPLACES
        # them with the pin, so this is 0 there); `num_pinned_video_rows` is
        # the pin's share alone, so the difference is the reference/anchor
        # prefix that leads the (possibly permuted) target block in this
        # FULL-row-space tensor -- 0 on fl2va and on any reference-only ref2va
        # request, nonzero on temporal inpaint's own pin (`label ==
        # "vid_inpaint"`) alongside references on ref2va, which reaches the
        # route as of phase B-3-open (`resolve_minimax_h3_inpaint_reference_gate`).
        n_cond_reference_video_rows = n_cond_video - int(layout.get("num_pinned_video_rows", 0) or 0)
        # With pinned frames the conditioning rows are rows of THIS clip, so the
        # decode takes every video row and restores frame-major order; with
        # anchors or references they are separate content and the tail is the
        # clip. `video_row_order` is None in the second case, which is the same
        # test the preview makes.
        clip_rows = (video_rows[n_cond_video:] if video_row_order is None
                     else video_rows[n_cond_reference_video_rows:][video_row_order.to(video_rows.device)])
        latents = ops.unpatchify_video_rows(
            clip_rows, latent_frames, latent_height, latent_width,
            latent_channels=int(components.get("latent_channels", 24)),
            patch_size=tuple(components["transformer_config"]["patch_size"]),
        )
        del clip_rows
        del video_rows

        decode_vae, decode_vae_name, decode_vae_is_fallback = select_minimax_h3_decode_vae(
            components, latent_frames)
        if decode_vae_is_fallback:
            from api.generation_status import add_warning

            add_warning(
                "This still-image request decoded through the video VAE's T=1 branch because "
                "the optional MiniMax-H3 image VAE checkpoint is not installed. Internal "
                "testing measured this decode path 14-18 dB lower PSNR than the image VAE "
                "checkpoint.",
                code="minimax_h3_still_image_default_vae_fallback",
            )

        decode_start = time.perf_counter()
        self._minimax_h3_reset_peak_vram()
        # The move onto/off the device is inside the timed phase now too (not
        # just the decode call): staging the video VAE (5.2 GB) is real
        # seconds, the same reason `stage_transformer`/`unstage_transformer`
        # exist for the DiT above, and it shares `vae_decode`'s own VRAM-peak
        # span (`phase_peaks["video_decode"]` below already reads across both).
        with generation_timer.phase("vae_decode"):
            self._minimax_h3_move(decode_vae_name, torch_device)
            try:
                frames = ops.decode_video(
                    decode_vae, latents,
                    latents_mean=components["latents_mean"],
                    latents_std=components["latents_std"],
                    pixel_mean=components["pixel_mean"],
                    pixel_std=components["pixel_std"],
                    device=device,
                )
            finally:
                self._minimax_h3_move(decode_vae_name, "cpu")
                del latents
                self._minimax_h3_empty_cache()
        video_decode_allocated, video_decode_reserved, video_decode_peak = self._minimax_h3_vram_stats()
        phase_peaks["video_decode"] = video_decode_peak
        print(f"[MiniMax-H3] video decode: {frames.shape[0]} frame(s) in "
              f"{time.perf_counter() - decode_start:.1f}s "
              f"(VRAM allocated {video_decode_allocated:.2f} GB, "
              f"reserved {video_decode_reserved:.2f} GB, phase peak {video_decode_peak:.2f} GB)")

        # `audio_enable=False` skips the DECODE and the mux -- the audio rows
        # still rode the packed sequence and still influenced the video through
        # self-attention, and they still consumed their noise draw, so the video
        # is bit-identical to the same seed with audio enabled. This is an
        # H3-specific behaviour: on LTX-2.3 the flag only discards audio the
        # pipeline already produced.
        audio_out = None
        audio_sample_rate = None
        if audio_enable and input_audio is not None and not pinned_audio_latents:
            # ia2v (WHOLE-track pin only -- a partial pin falls through to the
            # decode branch below, since most of the track was denoised and has
            # no source samples to hand back): the SOURCE waveform is handed
            # back, sample for sample. The pinned rows were never written, so
            # decoding them would return a VAE round trip of the input and
            # nothing else -- strictly worse than the samples that are already
            # in hand. This is the same exact-preservation stance the outpaint
            # path's `preserve_input` takes. The trim is the one the decode
            # path uses, so the muxed track ends with the last frame either way.
            #
            # WHAT THIS DOES NOT PROMISE: the mp4. `save_video_with_metadata`
            # encodes audio as AAC unless a caller asks for lossless, so the
            # FILE carries a lossy encoding of these samples -- exactly as it
            # does for a generated soundtrack. The exactness is of the handoff.
            audio_sample_rate = int(components.get("audio_sample_rate", 32000))
            audio_out = ops.trim_audio_to_video(
                input_audio, num_frames, fps=float(components.get("fps", 24.0)),
                sample_rate=audio_sample_rate)
            print(f"[MiniMax-H3] input audio muxed unchanged: {audio_out.shape[-1]} sample(s) @ "
                  f"{audio_sample_rate} Hz (the pinned rows are never denoised, so there is "
                  f"nothing to decode)")
        elif audio_enable:
            # A partial pin permuted the audio rows so the pinned prefix could
            # share the video pin's write-slice trick; the decode needs the
            # WHOLE block back in channel-major order, not the generated
            # suffix -- exactly the video pin's `video_row_order` un-permute
            # above. A caller with NO pin at all (or a ref2va reference count)
            # leaves `audio_row_order` `None` and takes the unpinned slice
            # unchanged -- but a WHOLE-track ia2v pin (`pin_target_audio`) also
            # leaves an IDENTITY permutation in `audio_row_order`
            # (`build_packed_layout`'s degenerate case, K0.6/§3), and is kept
            # out of THIS branch only by the `if ... and not pinned_audio_latents`
            # guard above sending it to the source-mux branch instead. If that
            # guard is ever relaxed, a whole-track ia2v request would fall
            # through here and silently decode a VAE round trip of the pinned
            # rows instead of returning the uploaded samples verbatim.
            audio_row_order = layout["audio_row_order"]
            # Same reference/anchor-prefix offset as the video branch above,
            # mirrored on the audio index list.
            n_cond_reference_audio_rows = n_cond_audio - int(layout.get("num_pinned_audio_rows", 0) or 0)
            full_audio_rows = (audio_rows[n_cond_reference_audio_rows:][audio_row_order.to(audio_rows.device)]
                               if audio_row_order is not None else audio_rows[n_cond_audio:])
            audio_latents = ops.unpack_audio_rows(full_audio_rows, num_audio_latents)
            self._minimax_h3_reset_peak_vram()
            # Same widening as the video decode above: the audio VAE's move
            # onto/off the device is inside `vae_decode` now too, and it
            # accumulates (`generation_timer.phase` sums repeat calls) onto
            # whatever the video decode already recorded this generation.
            with generation_timer.phase("vae_decode"):
                self._minimax_h3_move("audio_vae", torch_device)
                try:
                    waveform = ops.decode_audio(
                        components["audio_vae"], audio_latents,
                        latents_mean=components["audio_latents_mean"],
                        latents_std=components["audio_latents_std"],
                        device=device,
                    )
                finally:
                    self._minimax_h3_move("audio_vae", "cpu")
                    del audio_latents
                    self._minimax_h3_empty_cache()
            audio_sample_rate = int(components.get("audio_sample_rate", 32000))
            audio_out = ops.trim_audio_to_video(
                waveform, num_frames, fps=float(components.get("fps", 24.0)),
                sample_rate=audio_sample_rate)
            _audio_allocated, _audio_reserved, audio_decode_peak = self._minimax_h3_vram_stats()
            phase_peaks["audio_decode"] = audio_decode_peak
            # The waveform is handed over AS THE DECODER PRODUCED IT. An earlier
            # revision divided by the peak whenever it exceeded full scale, on
            # the premise that the 16-bit mux would wrap; it does not --
            # `utils/video_utils.py` clips to [-1, 1] before the int16 cast. A
            # peak normalisation is therefore an unmeasured global gain change
            # that neither MiniMax's reference implementation nor the LTX-2.3
            # path applies, i.e. a silent divergence on any loud clip.
            print(f"[MiniMax-H3] audio decode: {audio_out.shape[-1]} sample(s) @ "
                  f"{audio_sample_rate} Hz")
        else:
            print("[MiniMax-H3] audio_enable=false: skipping the audio decode and mux "
                  "(the audio rows still took part in generation)")
        del audio_rows

        overall_peak = max(phase_peaks.values(), default=0.0)
        peak_phase = max(phase_peaks, key=phase_peaks.get) if phase_peaks else "none"
        params["minimax_h3_vram_phase_peaks_gb"] = {
            name: round(value, 3) for name, value in phase_peaks.items()
        }
        params["minimax_h3_vram_peak_gb"] = round(overall_peak, 3)
        params["minimax_h3_vram_peak_phase"] = peak_phase
        print(f"[MiniMax-H3] total {time.perf_counter() - wall_start:.1f}s, "
              f"peak VRAM {overall_peak:.2f} GB in {peak_phase}; phase peaks "
              + ", ".join(f"{name}={value:.2f}" for name, value in phase_peaks.items()))
        self._minimax_h3_empty_cache()

        if frames.dtype != np.uint8:  # pragma: no cover - decode_video guarantees it
            frames = frames.astype(np.uint8)
        return frames, audio_out, audio_sample_rate, seed
