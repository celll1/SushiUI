"""MiniMax-H3 video backend mixin for DiffusionPipelineManager.

Video-with-audio generation against the pruned MiniMax-H3 checkpoint, from a
prompt alone (``t2va``) or from first/last keyframes as well (``fl2va``).
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

    @staticmethod
    def _minimax_h3_fit_keyframe(image, width: int, height: int, index: int):
        """Put one keyframe onto the canvas, the way the released model does.

        THE TWO ANCHORS ARE NOT TREATED THE SAME, and this asymmetry is the
        reference implementation's, in both independent ports of it:

        * the FIRST keyframe (``index == 0``) is the geometry anchor: MiniMax
          derives the canvas from it when the request omits width/height, so
          when a canvas is given the frame is simply STRETCHED onto it
          (diffusers ``MiniMaxH3ResizeStep``: a plain PIL
          ``resize((w, h), LANCZOS)``; ComfyUI: ``_resize(..., "disabled")``);
        * every LATER keyframe is a FOLLOWER and is aspect-preserving
          centre-cover-cropped (ComfyUI: ``_resize(..., "center")``), because it
          has no say in the geometry and stretching it would hand the model a
          distorted anchor it is then pinned to for the whole loop.

        The arithmetic below is MiniMax's own, kept verbatim rather than
        expressed through ``VaeImageProcessor(resize_mode="crop")``: that helper
        sizes with floor division and centres with ``w // 2 - src_w // 2``,
        where this rounds and centres with ``(src_w - w) // 2``. The two agree
        on some aspect ratios and differ BY ONE PIXEL on others (106 of 218
        sampled, per the diffusers block's own note), which moves the
        conditioning latents off the reference implementation.
        """
        image = image.convert("RGB")
        if image.size == (width, height):
            return image
        if index == 0:
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
        transformer, or a ``MiniMaxH3BlockLoopWrapper`` when block swap needs the
        re-owned block loop) and the block offloader to tear down afterwards
        (``None`` when there is none).

        TWO STATES:

        * ``blocks_to_swap == 0`` (the default): the transformer is moved to the
          device whole and the sampler calls it directly -- byte-identical to
          the pre-Phase-4 path, with no wrapper in the call chain at all. Block
          swap is the ONLY thing that wraps: FBCache was measured against the K3
          protocol and dropped for this architecture (see the block-loop
          wrapper's module docstring for the numbers), and Spectrum is declared
          unsupported, so there is no second reason to wrap.
        * ``blocks_to_swap > 0``: the NON-block modules (the three input
          projections, the token refiner, the RoPE buffer, the output norm and
          the two heads) are moved to the device, then
          ``TransformerBlockOffloader`` places the block stack -- the first
          ``50 - blocks_to_swap`` blocks resident and the rest weight-on-CPU.
          The whole-model ``.to(device)`` is deliberately NOT used here: it would
          put all 21 GB on the card and only then take some of it back off,
          which is the opposite of what the request asked for.

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

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        num_blocks = len(transformer.transformer_blocks)
        if blocks_to_swap >= num_blocks:
            print(f"[MiniMax-H3] blocks_to_swap={blocks_to_swap} >= {num_blocks} blocks; "
                  f"clamping to {num_blocks - 1} (at least one block must stay resident)")
            blocks_to_swap = num_blocks - 1

        if blocks_to_swap <= 0:
            self._minimax_h3_move("transformer", device)
            return transformer, None

        from core.memory_management import TransformerBlockOffloader

        for name, child in transformer.named_children():
            if name == "transformer_blocks":
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
    ):
        """Keyframe-conditioned generation with MiniMax-H3 (``fl2va``).

        ``input_image`` is the FIRST frame and the optional ``last_frame_image``
        is the LAST one — MiniMax-H3 conditions on the two ends of the clip and
        on nothing in between (its conditioning anchors are addressed by the
        rotary clock's first and last frame positions, not by an arbitrary frame
        index). Both are ordinary PIL images; each becomes one single-frame
        visual condition.

        Same return contract as ``_generate_txt2vid_minimax_h3``.
        """
        if input_image is None:
            raise RuntimeError("img2vid requires an input image for the first-frame keyframe")
        keyframes = [("first", input_image)]
        if last_frame_image is not None:
            keyframes.append(("last", last_frame_image))
        return self._generate_minimax_h3(
            params, keyframes=tuple(keyframes), label="img2vid",
            progress_callback=progress_callback, step_callback=step_callback)

    def _generate_minimax_h3(
        self,
        params: Dict[str, Any],
        *,
        keyframes: Sequence[Tuple[str, Any]] = (),
        label: str = "txt2vid",
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """The one MiniMax-H3 generation path, with 0-2 visual conditions.

        ``keyframes`` is a sequence of ``(anchor, PIL.Image)`` in PACKED ORDER,
        where anchor is ``"first"`` or ``"last"``. Empty is ``t2va``; one or two
        entries is ``fl2va``. Nothing else about the run differs between the two
        workflows — same layout builder, same draw order, same loop — which is
        why they share this function rather than having one copy each.

        Returns ``(frames, audio, audio_sample_rate, actual_seed)`` — see the
        module docstring.
        """
        from core.models.minimax_h3 import h3_pipeline_ops as ops
        from core.models.minimax_h3.loader import minimax_h3_latent_frames

        components = getattr(self, "minimax_h3_components", None)
        if not components:
            raise RuntimeError("MiniMax-H3 components are not loaded. Load a MiniMax-H3 model first.")

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
            np.asarray(self._minimax_h3_fit_keyframe(image, width, height, index), dtype=np.uint8)
            for index, (_anchor, image) in enumerate(keyframes)
        ]

        print(f"[MiniMax-H3] {label}: {width}x{height} num_frames={num_frames} "
              f"(latent {latent_frames}x{latent_height}x{latent_width}, "
              f"{num_audio_latents} audio latents/ch) steps={num_inference_steps} "
              f"seed={seed} audio={audio_enable} "
              f"conditions={list(anchors) if anchors else 'none (t2va)'}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()

        # ---- Phase 1: text encode (layer-streamed; nothing else on the GPU) ----
        text_encoder = components.get("text_encoder")
        tokenizer = components.get("tokenizer")
        if text_encoder is None or tokenizer is None:
            raise RuntimeError(
                "MiniMax-H3 is missing its text encoder or tokenizer, so a prompt cannot be "
                "encoded. Load the model with its text_encoders/ and official/tokenizer/ trees.")
        encode_start = time.perf_counter()
        with generation_timer.phase("text_encode"):
            prompt_embeds_cpu, num_text_tokens = ops.encode_prompt(
                text_encoder, tokenizer, prompt, device=device,
                dtype=torch.bfloat16, layer=ops.TEXT_ENCODER_LAYER,
            )
        self._minimax_h3_empty_cache()
        print(f"[MiniMax-H3] prompt encoded: {num_text_tokens} token(s) in "
              f"{time.perf_counter() - encode_start:.1f}s "
              f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")

        # ---- Layout + noise (drawn on the generation device, before staging) ----
        patch_size = tuple(components["transformer_config"]["patch_size"])
        latent_channels = int(components.get("latent_channels", 24))
        layout = ops.build_packed_layout(
            num_text_tokens, latent_frames, latent_height, latent_width, num_audio_latents,
            patch_size=patch_size,
            keyframe_anchors=anchors,
            device=torch_device,
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        # ONE draw per visual condition FIRST, at that condition's own latent
        # shape, then the video noise, then the audio noise -- the recorded
        # order (K0.6). A condition that is not drawn, or drawn later, changes
        # the video for the same seed.
        condition_noises, video_noise, audio_rows = ops.draw_noise(
            generator,
            video_latent_shape=(1, latent_channels, latent_frames, latent_height, latent_width),
            num_audio_latents=num_audio_latents,
            condition_shapes=tuple(
                (1, latent_channels, 1, latent_height, latent_width) for _ in anchors),
            device=device,
            audio_latent_channels=int(components.get("audio_latent_channels", 32)),
        )
        video_rows = ops.patchify_video_latents(video_noise, patch_size)[0]
        del video_noise

        # ---- Visual conditioning (fl2va): VAE-encode the keyframes ----
        # On the VIDEO VAE, before the DiT is staged: the two do not fit
        # together, and this is a single-frame spatial encode (cheap) rather
        # than the ViT decode at the end.
        if keyframe_pixels:
            cond_start = time.perf_counter()
            self._minimax_h3_move("vae", torch_device)
            try:
                condition_latents = ops.encode_condition_images(
                    components["vae"], keyframe_pixels,
                    latents_mean=components["latents_mean"],
                    latents_std=components["latents_std"],
                    pixel_mean=components["pixel_mean"],
                    pixel_std=components["pixel_std"],
                    device=device,
                )
            finally:
                self._minimax_h3_move("vae", "cpu")
                self._minimax_h3_empty_cache()
            condition_rows = ops.build_condition_rows(
                components["scheduler"], condition_latents, condition_noises,
                patch_size=patch_size,
            ).to(video_rows.device, video_rows.dtype)
            expected_rows = len(anchors) * layout["rows_per_frame"]
            if condition_rows.shape[0] != expected_rows:
                raise RuntimeError(
                    f"MiniMax-H3 conditioning produced {condition_rows.shape[0]} row(s) where the "
                    f"packed layout reserves {expected_rows} -- the keyframe latents do not match "
                    f"the generation canvas.")
            # The conditioning rows LEAD the video block; the loop protects them
            # by never writing the first `num_condition_video_rows` entries.
            video_rows = torch.cat([condition_rows, video_rows], dim=0)
            del condition_latents, condition_rows
            print(f"[MiniMax-H3] conditioned on {len(anchors)} keyframe(s) {list(anchors)} in "
                  f"{time.perf_counter() - cond_start:.1f}s "
                  f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")
        del condition_noises

        # ---- Phase 2: denoise (DiT resident) ----
        prompt_embeds = prompt_embeds_cpu.to(torch_device)
        denoise_start = time.perf_counter()
        # Staging owns the device move: with block swap it places the block stack
        # itself rather than moving all 21 GB on and some of it back off.
        transformer, offloader = self._ensure_minimax_h3_swap_and_offload(params, torch_device)
        self._minimax_h3_apply_attention_backend(transformer, params)
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
                    step_callback=step_callback,
                    # Preview geometry: the loop hands the callback LATENTS, not
                    # packed rows, so it needs the shape to unpatchify into.
                    preview_latent_shape=(latent_frames, latent_height, latent_width),
                    latent_channels=int(components.get("latent_channels", 24)),
                    patch_size=tuple(components["transformer_config"]["patch_size"]),
                )
        finally:
            # Back to the CPU before ANY decode: the video VAE's ViT decoder is
            # the second-largest allocation of the generation and the two do not
            # fit together. This also unwraps the block-loop wrapper and cleans
            # up the block offloader.
            self._unstage_minimax_h3_transformer(offloader)
            del transformer
            del prompt_embeds
            self._minimax_h3_empty_cache()
        denoise_seconds = time.perf_counter() - denoise_start
        peak_after_denoise = self._minimax_h3_peak_vram()
        print(f"[MiniMax-H3] denoise: {num_inference_steps} step(s) in {denoise_seconds:.1f}s "
              f"({denoise_seconds / max(num_inference_steps, 1):.2f}s/step, "
              f"peak VRAM {peak_after_denoise:.2f} GB)")

        if not torch.isfinite(video_rows).all():
            raise RuntimeError("MiniMax-H3 produced non-finite video latents.")

        # ---- Phase 3: decode ----
        n_cond_video = layout["num_condition_video_rows"]
        n_cond_audio = layout["num_condition_audio_rows"]
        latents = ops.unpatchify_video_rows(
            video_rows[n_cond_video:], latent_frames, latent_height, latent_width,
            latent_channels=int(components.get("latent_channels", 24)),
            patch_size=tuple(components["transformer_config"]["patch_size"]),
        )
        del video_rows

        decode_start = time.perf_counter()
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
        print(f"[MiniMax-H3] video decode: {frames.shape[0]} frame(s) in "
              f"{time.perf_counter() - decode_start:.1f}s "
              f"(peak VRAM {self._minimax_h3_peak_vram():.2f} GB)")

        # `audio_enable=False` skips the DECODE and the mux -- the audio rows
        # still rode the packed sequence and still influenced the video through
        # self-attention, and they still consumed their noise draw, so the video
        # is bit-identical to the same seed with audio enabled. This is an
        # H3-specific behaviour: on LTX-2.3 the flag only discards audio the
        # pipeline already produced.
        audio_out = None
        audio_sample_rate = None
        if audio_enable:
            audio_latents = ops.unpack_audio_rows(audio_rows[n_cond_audio:], num_audio_latents)
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

        print(f"[MiniMax-H3] total {time.perf_counter() - wall_start:.1f}s, "
              f"peak VRAM {self._minimax_h3_peak_vram():.2f} GB")
        self._minimax_h3_empty_cache()

        if frames.dtype != np.uint8:  # pragma: no cover - decode_video guarantees it
            frames = frames.astype(np.uint8)
        return frames, audio_out, audio_sample_rate, seed
