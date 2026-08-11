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

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import random
import time

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
          ``MiniMaxH3ResizeStep``: a plain PIL ``resize((w, h), LANCZOS)``;
          ComfyUI: ``_resize(..., "disabled")``);
        * every OTHER keyframe is a FOLLOWER and is aspect-preserving
          centre-cover-cropped (ComfyUI: ``_resize(..., "center")``), because it
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

    def _ensure_minimax_h3_swap_and_offload(
        self, params: Dict[str, Any], device: torch.device,
    ):
        """Stage the DiT onto ``device`` for the denoise loop. Returns the callable.

        Returns ``(module, offloader)``: the object the sampler calls (the raw
        transformer, or a ``MiniMaxH3BlockLoopWrapper`` when block swap/FBCache needs the
        re-owned block loop) and the block offloader to tear down afterwards
        (``None`` when there is none).

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

        # SushiUI addition: opt-in, not bit-exact -- see `adaln_chunking.py`'s "Head fusion" note.
        # Set on the INNER transformer (not the wrapper) so both call sites -- the stock forward's
        # (fast path, wrapper delegates to it) and `MiniMaxH3BlockLoopWrapper._custom_forward`'s
        # (block swap / FBCache) -- read the same flag off the same object.
        transformer.fuse_output_proj = bool(params.get("fuse_output_proj", False))

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        from core.inference.fbcache import fbcache_active
        fbcache_on = fbcache_active(params) and not params.get("spectrum_enable", False)
        num_blocks = len(transformer.transformer_blocks)
        if blocks_to_swap >= num_blocks:
            print(f"[MiniMax-H3] blocks_to_swap={blocks_to_swap} >= {num_blocks} blocks; "
                  f"clamping to {num_blocks - 1} (at least one block must stay resident)")
            blocks_to_swap = num_blocks - 1

        if blocks_to_swap <= 0:
            self._minimax_h3_move("transformer", device)
            if fbcache_on:
                wrapper = MiniMaxH3BlockLoopWrapper(transformer)
                components["transformer"] = wrapper
                print("[FBCache] MiniMax-H3 wrapper armed (cache is opt-in)")
                return wrapper, None
            return transformer, None

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
        return wrapper, offloader

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

        Supports stacking multiple LoRAs on the same module (later configs
        wrap the module the earlier one already wrapped).
        """
        from core.models.minimax_h3.minimax_h3_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
            check_variant_compatibility, detect_rank_variation,
        )
        from core.extensions.lora_manager import lora_manager

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

        if not hasattr(self, "_minimax_h3_lora_original_modules"):
            self._minimax_h3_lora_original_modules: Dict[str, torch.nn.Module] = {}
            self._minimax_h3_lora_wrapped_keys: set = set()

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
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                warn(f"LoRA file not found: {lora_path}", "minimax_h3_lora_not_found")
                continue
            try:
                raw, metadata = load_lora_safetensors(str(resolved))
                check_variant_compatibility(metadata, lora_path, current_variant, warn)
                targets = normalise_lora_state_dict(raw)
                print(f"[MiniMax-H3 LoRA] {i + 1}/{len(lora_configs)}: {lora_path} "
                      f"keys={len(raw)} matched_targets={len(targets)} strength={strength}")

                if blocks_to_swap > 0:
                    rank_variation = detect_rank_variation(targets)
                    varying_leaves = [leaf for leaf, varies in rank_variation.items() if varies]
                    if varying_leaves:
                        warn(
                            f"LoRA '{lora_path}' has a rank that varies across blocks for "
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
                                f"LoRA '{lora_path}' is a {student_steps_int}-step distillation "
                                f"checkpoint (num_inference_steps counts sigma grid points "
                                f"including the terminal 0, so {expected_steps} is the matching "
                                f"value); this generation is requesting "
                                f"num_inference_steps={num_inference_steps}.",
                                "minimax_h3_lora_step_count_mismatch",
                            )
                        if distillation_cache_active:
                            warn(
                                f"LoRA '{lora_path}' is a {student_steps_int}-step distillation "
                                f"checkpoint and FBCache/Spectrum forecasting is also active. With "
                                f"only ~{expected_steps} model evaluations, warmup alone can consume "
                                f"the whole trajectory and leave nothing for the cache/forecast to "
                                f"act on.",
                                "minimax_h3_lora_distillation_with_cache",
                            )

                applied, missing = apply_lora_group(
                    transformer, targets, strength,
                    self._minimax_h3_lora_original_modules, self._minimax_h3_lora_wrapped_keys,
                )
                if missing:
                    warn(
                        f"LoRA '{lora_path}' has {len(missing)} target(s) that could not be "
                        f"resolved against the loaded MiniMax-H3 transformer (first few: "
                        f"{missing[:5]}) -- skipped.",
                        "minimax_h3_lora_targets_unresolved",
                    )
                print(f"[MiniMax-H3 LoRA]   wrapped {applied} module(s)")
                total_applied += applied
            except Exception as e:
                print(f"[MiniMax-H3 LoRA] ERROR loading {lora_path}: {e}")
                import traceback
                traceback.print_exc()
                warn(f"Failed to load LoRA '{lora_path}': {e}", "minimax_h3_lora_load_failed")

        return total_applied

    def _unload_lora_minimax_h3(self) -> int:
        """Restore every MiniMax-H3 transformer Linear to its pre-LoRA original."""
        from core.models.minimax_h3.minimax_h3_lora import restore_originals

        if not getattr(self, "_minimax_h3_lora_wrapped_keys", None):
            return 0
        components = getattr(self, "minimax_h3_components", None)
        if not components:
            return 0
        transformer = components["transformer"]
        from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
        if isinstance(transformer, MiniMaxH3BlockLoopWrapper):
            transformer = transformer.transformer
        restored = restore_originals(
            transformer, self._minimax_h3_lora_original_modules, self._minimax_h3_lora_wrapped_keys,
        )
        print(f"[MiniMax-H3 LoRA] Unloaded {restored} LoRA wrapper(s)")
        return restored

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
        from api.generation_utils import plan_video_outpaint_placement
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
        plan = plan_video_outpaint_placement(
            params, arch or "minimax_h3",
            head_frames=int(head.shape[0]),
            tail_frames=int(tail.shape[0]) if tail is not None else None,
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
        # clip(s), in PACKED order (first, then last). ----
        keyframes = []
        if placement == "extend_forward":
            keyframes.append(("first", Image.fromarray(head[-1])))
        elif placement == "extend_backward":
            keyframes.append(("last", Image.fromarray(head[0])))
        else:  # bridge: both ends are anchored, in packed order
            keyframes.append(("first", Image.fromarray(head[-1])))
            keyframes.append(("last", Image.fromarray(tail[0])))

        print(f"[MiniMax-H3] vid_outpaint: {placement} {width}x{height} "
              f"preserved head={plan['head_frames']} tail={plan['tail_frames']} "
              f"generated={generated_frames} -> {out_frames_total} frame(s) @ {frame_rate} fps")

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

        # Only the generated span is sampled; everything else about the run is
        # an ordinary fl2va (or, with references, ref2va) generation, so it
        # goes through the ONE generation path rather than a second copy of
        # the staging/denoise/decode sequence.
        sub_params = dict(params)
        sub_params["num_frames"] = generated_frames
        frames_gen, audio_gen, audio_sample_rate, seed = self._generate_minimax_h3(
            sub_params, keyframes=tuple(keyframes), references=references, label="vid_outpaint",
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

        # ---- Assemble. The anchor frame(s) of the GENERATED span are dropped:
        # they are the model's reconstruction of a frame we are preserving
        # exactly, at the same instant. ----
        if placement == "extend_forward":
            frames_out = np.concatenate([head, frames_gen[1:]], axis=0)
            preserved_spans = [(0, plan["head_frames"], input_audio, fps, trim_start)]
            gen_audio_start_frame = plan["head_frames"] - 1
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

        audio_out = audio_gen
        if audio_gen is not None and audio_sample_rate:
            audio_out = self._minimax_h3_outpaint_audio(
                audio_gen, audio_sample_rate, params,
                total_frames=out_frames_total, frame_rate=frame_rate,
                gen_audio_start_frame=gen_audio_start_frame,
                preserved_spans=preserved_spans, warn=warn,
            )

        return frames_out, audio_out, audio_sample_rate, seed

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

        Returns:
            ``(frames, audio, audio_sample_rate, actual_seed)`` -- the same
            tuple contract as every other video generate path.
        """
        from api.error_handlers import ValidationError
        from api.generation_utils import (
            MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE,
            latent_frame_spans,
            plan_video_inpaint_span,
        )
        from core.inference.outpaint_utils import center_crop_resize_frames

        if not getattr(self, "minimax_h3_components", None):
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")
        if video_frames is None or len(video_frames) == 0:
            raise RuntimeError("vid_inpaint requires a decoded input video clip")

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

        # ---- Audio. `preserve_input` pins the clip's own track across the WHOLE
        # clip through the shipped ia2v machinery and muxes it back verbatim, so
        # the regenerated span has the original soundtrack both to condition on
        # and in the output. There is no second audio path here: the pin and the
        # exact mux are both `_generate_minimax_h3`'s ia2v behaviour. ----
        audio_mode = str(params.get("inpaint_video_audio_mode") or "regenerate")
        pinned_audio = None
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

        print(f"[MiniMax-H3] vid_inpaint: {width}x{height} clip={clip_frames} frame(s) "
              f"regenerate {start_frame}..{end_frame} "
              f"({len(plan['regenerate_latent_frames'])} of {plan['latent_frames']} latent "
              f"frames) audio={audio_mode} @ {frame_rate} fps"
              + (f" spatial_pinned_rows={len(pinned_video_row_indices)}"
                 if spatial_mask_mode else ""))

        sub_params = dict(params)
        sub_params["num_frames"] = clip_frames
        if spatial_mask_mode:
            frames_gen, audio_out, audio_sample_rate, seed = self._generate_minimax_h3(
                sub_params,
                pinned_video_frames=(),
                pinned_video_row_indices=pinned_video_row_indices,
                pinned_video_source=clip, input_audio=pinned_audio, label="vid_inpaint",
                progress_callback=progress_callback, step_callback=step_callback,
            )
        else:
            frames_gen, audio_out, audio_sample_rate, seed = self._generate_minimax_h3(
                sub_params,
                pinned_video_frames=plan["pinned_latent_frames"],
                pinned_video_source=clip, input_audio=pinned_audio, label="vid_inpaint",
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

        return frames_out, audio_out, audio_sample_rate, seed

    def _minimax_h3_inpaint_pinned_audio(
        self,
        input_audio: Optional[bytes],
        *,
        clip_frames: int,
        source_fps: float,
        trim_start: int,
        frame_rate: float,
        warn: Callable[[str, str], None],
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

        Returns None -- with a warning -- for every recoverable failure, and the
        caller then generates the audio instead of pinning it.
        """
        from core.models.minimax_h3 import h3_references as refs
        from utils.video_utils import extract_audio_window

        if not input_audio:
            warn("inpaint_video_audio_mode='preserve_input' was requested but the uploaded clip "
                 "has no audio stream; the soundtrack is generated instead",
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
            warn(f"preserve_input audio was time-stretched ({src_dur_sec:.3f}s -> "
                 f"{target_dur_sec:.3f}s) because the uploaded clip's frame rate "
                 f"({source_fps:.3f}) differs from MiniMax-H3's fixed {frame_rate:.3f} fps",
                 code="inpaint_video_audio_stretched")
        try:
            window = extract_audio_window(
                input_audio, trim_start / source_fps, src_dur_sec, target_dur_sec,
                sample_rate=sample_rate, channels=2,
            )
        except Exception as exc:
            window = None
            print(f"[MiniMax-H3] vid_inpaint audio window extraction raised: {exc}")
        if window is None:
            warn("preserve_input audio window extraction failed; the soundtrack is generated "
                 "instead", code="inpaint_video_audio_extract_failed")
            return None
        try:
            return refs.prepare_pinned_audio(
                torch.from_numpy(np.ascontiguousarray(window)), sample_rate,
                num_frames=clip_frames, fps=frame_rate, target_sample_rate=sample_rate,
                latent_rate=float(components.get("audio_latent_rate", 40.0)))
        except ValueError as exc:
            warn(f"preserve_input audio could not condition this clip ({exc}); the soundtrack is "
                 f"generated instead", code="inpaint_video_audio_extract_failed")
            return None

    def _generate_minimax_h3(
        self,
        params: Dict[str, Any],
        *,
        keyframes: Sequence[Tuple[Any, Any]] = (),
        references: Sequence[Any] = (),
        input_audio=None,
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
        from core.models.minimax_h3.loader import minimax_h3_latent_frames

        components = getattr(self, "minimax_h3_components", None)
        if not components:
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")
        if pinned_video_frames is None:
            pinned_video_frames = ()
        if pinned_video_row_indices is None:
            pinned_video_row_indices = ()
        pinned_video_frames = tuple(pinned_video_frames)
        pinned_video_row_indices = tuple(pinned_video_row_indices)
        if pinned_video_frames and pinned_video_row_indices:
            raise RuntimeError(
                "MiniMax-H3 cannot combine pinned video frames with pinned video row indices: "
                "pass one video pinning scheme only.")
        if (pinned_video_frames or pinned_video_row_indices) and (keyframes or references):
            raise RuntimeError(
                "MiniMax-H3 cannot combine pinned video pins with keyframes or references: the "
                "pin re-uses the video block's conditioning prefix for rows of the clip itself, "
                "and an anchor or a reference reserves that same prefix for rows of its own.")
        if (pinned_video_frames or pinned_video_row_indices) and pinned_video_source is None:
            raise RuntimeError(
                "MiniMax-H3 temporal inpaint needs the source clip the pinned content is taken "
                "from: pinned video frames/rows name conditioning content, and "
                "pinned_video_source supplies the pixels they are encoded from.")
        if input_audio is not None and references:
            raise RuntimeError(
                "MiniMax-H3 cannot pin an input audio track on a ref2va request: a reference "
                "soundtrack already occupies its own block at its own rotary offset, while ia2v "
                "pins the TARGET's audio rows. Send the track as an audio reference instead.")
        if references and (components.get("variant") or "") != "ref2va":
            raise RuntimeError(
                f"ref2vid needs the MiniMax-H3 ref2va transformer, but the loaded checkpoint is "
                f"{components.get('variant') or 'an unidentified variant'} "
                f"({components.get('dit_path')}). Load "
                f"diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors -- reference "
                f"conditioning is a trained behaviour of that partition alone, and the two files "
                f"are otherwise indistinguishable, so running it here would silently produce a bad "
                f"video rather than fail.")

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
              + (f" input_audio={int(input_audio.shape[-1])} sample(s) pinned"
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
        with generation_timer.phase("text_encode"):
            if normalized_references:
                # The ref2va PRESENTATION: a label and a vision block per
                # reference, then the prompt verbatim. The vision blocks' rows
                # are tagged VIDEO, not text, which is what the transformer's
                # AdaLN modulation keys off -- so the tags travel with the
                # embeddings into the layout.
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
            else:
                prompt_embeds_cpu, num_text_tokens = ops.encode_prompt(
                    text_encoder, tokenizer, prompt, device=device,
                    dtype=torch.bfloat16, layer=ops.TEXT_ENCODER_LAYER,
                )
        self._minimax_h3_empty_cache()
        text_allocated, text_reserved, text_peak = self._minimax_h3_vram_stats()
        phase_peaks["text_encode"] = text_peak
        print(f"[MiniMax-H3] prompt encoded: {num_text_tokens} token(s) in "
              f"{time.perf_counter() - encode_start:.1f}s "
              f"(VRAM allocated {text_allocated:.2f} GB, reserved {text_reserved:.2f} GB, "
              f"phase peak {text_peak:.2f} GB)")
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
                "pin_target_audio": pinned_audio_rows is not None,
                "device": torch_device,
            }
            if pinned_video_row_indices:
                layout_kwargs["pinned_video_row_indices"] = tuple(pinned_video_row_indices)
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
            audio_rows = pinned_audio_rows.to(audio_rows.device, audio_rows.dtype)
            del pinned_audio_rows

        if condition_latents:
            condition_rows = ops.build_condition_rows(
                components["scheduler"], condition_latents, condition_noises,
                patch_size=patch_size,
            ).to(video_rows.device, video_rows.dtype)
            expected_rows = int(layout["num_condition_video_rows"])
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
            expected_audio_rows = int(layout["num_condition_audio_rows"])
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

        # ---- Phase 2: denoise (DiT resident) ----
        self._minimax_h3_assert_components_off_cuda("text_encoder", "vae", "audio_vae")
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
        # tree containing a non-nn.Linear module, which they reject).
        lora_configs = params.get("loras") or []
        applied_lora_count = self._load_lora_minimax_h3(lora_configs, params) if lora_configs else 0
        # Staging owns the device move: with block swap it places the block stack
        # itself rather than moving all 21 GB on and some of it back off.
        transformer, offloader = self._ensure_minimax_h3_swap_and_offload(params, torch_device)
        self._minimax_h3_apply_attention_backend(transformer, params)
        # ~150s per step means the per-step callback alone looks like a hang, so
        # block-level forward hooks tick progress from inside the step. Removed
        # in the `finally` below: a surviving hook would fire on the next
        # generation against this generation's callback.
        substep_reporter = attach_block_substep_hooks(
            transformer, progress_callback, label="MiniMax-H3")
        try:
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
            substep_reporter.close()
            if applied_lora_count:
                self._unload_lora_minimax_h3()
            # Back to the CPU before ANY decode: the video VAE's ViT decoder is
            # the second-largest allocation of the generation and the two do not
            # fit together. This also unwraps the block-loop wrapper and cleans
            # up the block offloader.
            self._unstage_minimax_h3_transformer(offloader)
            del transformer
            del prompt_embeds
            self._minimax_h3_empty_cache()
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
        # With pinned frames the conditioning rows are rows of THIS clip, so the
        # decode takes every video row and restores frame-major order; with
        # anchors or references they are separate content and the tail is the
        # clip. `video_row_order` is None in the second case, which is the same
        # test the preview makes.
        clip_rows = (video_rows[n_cond_video:] if video_row_order is None
                     else video_rows[video_row_order.to(video_rows.device)])
        latents = ops.unpatchify_video_rows(
            clip_rows, latent_frames, latent_height, latent_width,
            latent_channels=int(components.get("latent_channels", 24)),
            patch_size=tuple(components["transformer_config"]["patch_size"]),
        )
        del clip_rows
        del video_rows

        decode_start = time.perf_counter()
        self._minimax_h3_reset_peak_vram()
        self._minimax_h3_move("vae", torch_device)
        try:
            with generation_timer.phase("vae_decode"):
                frames = ops.decode_video(
                    components["vae"], latents,
                    latents_mean=components["latents_mean"],
                    latents_std=components["latents_std"],
                    pixel_mean=components["pixel_mean"],
                    pixel_std=components["pixel_std"],
                    device=device,
                )
        finally:
            self._minimax_h3_move("vae", "cpu")
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
        if audio_enable and input_audio is not None:
            # ia2v: the SOURCE waveform is handed back, sample for sample. The
            # pinned rows were never written, so decoding them would return a
            # VAE round trip of the input and nothing else -- strictly worse
            # than the samples that are already in hand. This is the same
            # exact-preservation stance the outpaint path's `preserve_input`
            # takes. The trim is the one the decode path uses, so the muxed
            # track ends with the last frame either way.
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
            audio_latents = ops.unpack_audio_rows(audio_rows[n_cond_audio:], num_audio_latents)
            self._minimax_h3_reset_peak_vram()
            self._minimax_h3_move("audio_vae", torch_device)
            try:
                with generation_timer.phase("vae_decode"):
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
