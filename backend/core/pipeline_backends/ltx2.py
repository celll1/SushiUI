"""LTX-2.3 video backend mixin for DiffusionPipelineManager.

Holds the text-to-video generation entry point for the LTX-2.3 joint
audio+video MM-DiT. The assembled `LTX2Pipeline` is stored in
`self.ltx2_components["pipeline"]` by the loader (P1a). This mixin stages the
components onto the GPU via `enable_model_cpu_offload` and drives the pipeline's
`__call__` to produce a video (and optional audio) tensor.

Output contract of `_generate_txt2vid_ltx2`:
    (frames, audio, audio_sample_rate, actual_seed)
where:
    frames  = np.uint8 array of shape [T, H, W, 3] (RGB, value range 0-255).
              Derived from the pipeline's `output_type="np"` result
              (shape [B, T, H, W, C], float in [0, 1]); batch index 0 is taken.
    audio   = torch.FloatTensor of shape [channels, samples] on CPU, or None
              when audio is disabled / unavailable. From `audio[0]`.
    audio_sample_rate = int (vocoder output sampling rate, typically 24000) or None.
    actual_seed = the concrete seed used (random draw resolved when seed < 0).
"""

from typing import Dict, Any, Optional, Callable
import random

import numpy as np
import torch
from PIL import Image


class LTX2Mixin:
    """LTX2Mixin: LTX-2.3 text-to-video generation backend."""

    def _ensure_ltx2_offload(self, blocks_to_swap: int = 0):
        """Attach model_cpu_offload hooks, in either the stock ("normal") mode or
        the block-swap-compatible ("block_swap") mode, re-attaching whenever the
        requested mode differs from what is currently attached.

        Stock mode (``blocks_to_swap == 0``, UNCHANGED): `enable_model_cpu_offload`
        stages the offload sequence (text_encoder -> connectors -> transformer ->
        vae -> audio_vae -> vocoder) so the 19B transformer + Gemma-3 text encoder
        move through the GPU one component at a time.

        Block-swap mode (``blocks_to_swap > 0``): `enable_model_cpu_offload`
        normally moves each staged component to GPU via an accelerate forward
        hook that owns the WHOLE module (`cpu_offload_with_hook`), which
        conflicts with block-swap (some transformer blocks must stay resident on
        CPU while the rest stream through GPU). diffusers 0.38.0's
        `enable_model_cpu_offload` (`pipeline_utils.py` ~1189-1282) builds its
        per-model hook chain by walking ``model_cpu_offload_seq.split("->")``;
        remaining `self.components` entries are either given a plain
        `.to(device)` (if listed in `self._exclude_from_cpu_offload`) or another
        accelerate hook. Both are plain-Python attributes read off the pipeline
        INSTANCE (falling back to the class default), so reassigning them on the
        instance does not mutate the class or other pipeline instances. We drop
        the ``"transformer"`` token from the instance's ``model_cpu_offload_seq``
        AND add ``"transformer"`` to ``self._exclude_from_cpu_offload`` so it
        takes the plain-`.to(device)` branch (one unconditional move, not an
        accelerate hook) instead of being hook-managed. The block offloader
        (built afterward in `_ensure_ltx2_block_swap_wrapper`) then repositions
        the swappable blocks' weights back to CPU. Every other component keeps
        its stock accelerate-hook behavior in both modes.
        """
        pipeline = self.ltx2_components.get("pipeline")
        if pipeline is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        desired_mode = "block_swap" if blocks_to_swap > 0 else "normal"
        current_mode = getattr(self, "_ltx2_offload_mode", None)
        if getattr(self, "_ltx2_offload_enabled", False) and current_mode == desired_mode:
            return pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                # model_cpu_offload_seq is a CLASS attribute; reading it via
                # type(pipeline) always yields the stock diffusers-defined
                # string, independent of any previous instance-level override.
                stock_seq = type(pipeline).model_cpu_offload_seq
                exclude = list(getattr(pipeline, "_exclude_from_cpu_offload", []) or [])
                if desired_mode == "block_swap":
                    pipeline.model_cpu_offload_seq = "->".join(
                        tok for tok in stock_seq.split("->") if tok != "transformer"
                    )
                    if "transformer" not in exclude:
                        exclude.append("transformer")
                else:
                    pipeline.model_cpu_offload_seq = stock_seq
                    exclude = [tok for tok in exclude if tok != "transformer"]
                pipeline._exclude_from_cpu_offload = exclude

                pipeline.enable_model_cpu_offload(device=device)
                print(f"[LTX-2.3] enable_model_cpu_offload attached (mode={desired_mode}, "
                      f"seq={pipeline.model_cpu_offload_seq})")
            except Exception as e:
                print(f"[LTX-2.3] enable_model_cpu_offload failed ({e}); "
                      f"falling back to whole-pipeline .to(cuda)")
                pipeline.to(device)
        else:
            print("[LTX-2.3] CUDA unavailable; running pipeline on CPU")

        # VRAM: enable the LTX video VAE's temporal/spatial tiling so decoding the
        # full [T,H,W] latent does not spike VRAM (the dominant generation peak,
        # especially for long clips). Idempotent; diffusers picks tile sizes.
        vae = self.ltx2_components.get("vae")
        if vae is not None and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
                print("[LTX-2.3] video VAE tiling enabled (decode VRAM headroom)")
            except Exception as e:
                print(f"[LTX-2.3] VAE tiling enable failed ({e}); continuing")

        self._ltx2_offload_enabled = True
        self._ltx2_offload_mode = desired_mode
        return pipeline

    def _ensure_ltx2_block_swap_wrapper(self, blocks_to_swap: int):
        """Wrap (or unwrap) the LTX-2.3 transformer for AP1 block-swap GENERATION.

        ``blocks_to_swap <= 0``: unwraps back to the stock
        ``LTX2VideoTransformer3DModel`` (byte-identical current behavior — the
        wrapper is NOT applied in this case).

        ``blocks_to_swap > 0``: builds a ``TransformerBlockOffloader`` over
        ``transformer.transformer_blocks`` (generic block_offloading.py,
        ``supports_backward=False`` — inference only) and wraps the transformer
        with ``Ltx2BlockLoopWrapper``. Both ``pipeline.transformer`` and
        ``self.ltx2_components["transformer"]`` are updated to the wrapper so
        every consumer (base pipeline and a later-built i2v pipeline) sees the
        same object. An already-cached i2v pipeline (if any) has its
        ``transformer`` ref updated too, since it shares every module with the
        base pipeline rather than owning its own weights.

        Must be called AFTER `_ensure_ltx2_offload(blocks_to_swap)` so the
        transformer's device placement (whole-GPU via the offload-exclusion
        above) is settled before the offloader repositions swappable blocks.
        """
        pipeline = self.ltx2_components.get("pipeline")
        if pipeline is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        from core.models.ltx2_block_loop_wrapper import Ltx2BlockLoopWrapper

        current = pipeline.transformer
        inner = current.transformer if isinstance(current, Ltx2BlockLoopWrapper) else current

        if blocks_to_swap <= 0:
            if isinstance(current, Ltx2BlockLoopWrapper):
                offloader = current._block_offloader
                if offloader is not None:
                    offloader.cleanup()
                pipeline.transformer = inner
                self.ltx2_components["transformer"] = inner
                i2v = self.ltx2_components.get("i2v_pipeline")
                if i2v is not None:
                    i2v.transformer = inner
                print("[LTX-2.3] Block Swap disabled; transformer unwrapped (stock forward)")
            self._ltx2_block_swap_count = 0
            return

        prev_count = getattr(self, "_ltx2_block_swap_count", 0)
        if isinstance(current, Ltx2BlockLoopWrapper) and prev_count == blocks_to_swap:
            return  # already wired for this exact swap count

        if isinstance(current, Ltx2BlockLoopWrapper) and current._block_offloader is not None:
            current._block_offloader.cleanup()

        from core.memory_management import TransformerBlockOffloader
        device = next(inner.parameters()).device
        offloader = TransformerBlockOffloader(
            blocks=inner.transformer_blocks,
            blocks_to_swap=blocks_to_swap,
            device=device,
            target_dtype=inner.dtype,
            use_pinned_memory=False,
            transformer=inner,
            supports_backward=False,
            # Generation weights are frozen: use the H2D-only fast path (permanent
            # pinned CPU masters + coalesced single copy per block) so we skip the
            # pointless device->host eviction of read-only weights (halves PCIe
            # traffic vs the standard swap). Auto-disables if backward is ever on.
            h2d_only=True,
        )
        offloader.prepare_block_devices_before_forward()

        wrapper = Ltx2BlockLoopWrapper(inner, block_offloader=offloader)
        pipeline.transformer = wrapper
        self.ltx2_components["transformer"] = wrapper
        i2v = self.ltx2_components.get("i2v_pipeline")
        if i2v is not None:
            i2v.transformer = wrapper
        self._ltx2_block_swap_count = blocks_to_swap
        print(f"[LTX-2.3] Block Swap enabled: {blocks_to_swap} blocks to swap "
              f"(Ltx2BlockLoopWrapper active)")

    def _ensure_ltx2_swap_and_offload(self, blocks_to_swap: int):
        """Bring the shared transformer to the requested block-swap state with the
        CORRECT ordering relative to the model-offload hook attach, and return the
        base pipeline.

        Enabling (``blocks_to_swap > 0``): offload FIRST (which excludes the
        transformer from the accelerate hook chain and gives it a plain
        `.to(device)`), THEN wrap + build the block offloader that repositions the
        swappable blocks to CPU.

        Disabling (``blocks_to_swap <= 0``): UNWRAP FIRST, then re-attach offload.
        `enable_model_cpu_offload` moves the whole pipeline to CPU and binds a
        streaming forward-hook to ``pipeline.transformer``; if we re-offloaded while
        the wrapper were still installed, the hook would bind to the wrapper object
        that the subsequent unwrap discards, leaving the inner transformer stranded
        on CPU with no hook (device-mismatch on the next call). Unwrapping first
        makes the hook bind to the restored inner transformer.
        """
        if blocks_to_swap > 0:
            pipeline = self._ensure_ltx2_offload(blocks_to_swap=blocks_to_swap)
            self._ensure_ltx2_block_swap_wrapper(blocks_to_swap)
        else:
            self._ensure_ltx2_block_swap_wrapper(0)
            pipeline = self._ensure_ltx2_offload(blocks_to_swap=0)
        return pipeline

    def _ensure_ltx2_i2v_pipeline(self):
        """Build (once) and return the LTX2ImageToVideoPipeline.

        The image-to-video pipeline shares EVERY module with the base
        `LTX2Pipeline` — it only changes the denoise loop (first-frame keyframe
        conditioning). We therefore construct it from the already-loaded
        components WITHOUT reloading any weights, and cache it under
        `ltx2_components["i2v_pipeline"]` so it survives across calls and is
        freed by the load_model eviction loop (which iterates the components
        dict) on a model swap.

        Offload: we never call `enable_model_cpu_offload` on the i2v pipeline.
        The base pipeline owns the cpu-offload hooks (attached to the shared
        module objects via `_ensure_ltx2_offload`). Because the i2v pipeline
        drives those same module objects, the per-module accelerate hooks fire
        during its `__call__` exactly as they do for the base pipeline. This
        avoids double-hooking the shared modules.
        """
        cached = self.ltx2_components.get("i2v_pipeline")
        if cached is not None:
            return cached

        base = self.ltx2_components.get("pipeline")
        if base is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        from core.models.ltx2 import LTX2ImageToVideoPipeline

        i2v = LTX2ImageToVideoPipeline(
            scheduler=self.ltx2_components.get("scheduler"),
            vae=self.ltx2_components.get("vae"),
            audio_vae=self.ltx2_components.get("audio_vae"),
            text_encoder=self.ltx2_components.get("text_encoder"),
            tokenizer=self.ltx2_components.get("tokenizer"),
            connectors=self.ltx2_components.get("connectors"),
            transformer=self.ltx2_components.get("transformer"),
            vocoder=self.ltx2_components.get("vocoder"),
            processor=getattr(base, "processor", None),
        )
        self.ltx2_components["i2v_pipeline"] = i2v
        print("[LTX-2.3] LTX2ImageToVideoPipeline constructed from shared components "
              "(no weight reload; offload owned by base pipeline)")
        return i2v

    def _generate_img2vid_ltx2(
        self,
        params: Dict[str, Any],
        input_image,
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Image-to-video generation with LTX-2.3 (first-frame keyframe).

        `input_image` is a PIL.Image used as the first-frame keyframe. The
        pipeline's `video_processor.preprocess(image, height, width)` resizes it
        to the (÷32) target resolution and the denoise loop pins it as frame 0
        via `conditioning_mask[:, :, 0] = 1`.

        Returns (frames, audio, audio_sample_rate, actual_seed) — identical
        contract to `_generate_txt2vid_ltx2` (see module docstring).
        """
        if not self.ltx2_components:
            raise RuntimeError("LTX-2.3 components not loaded. Please load an LTX-2.3 model first.")

        if input_image is None:
            raise RuntimeError("img2vid requires an input image for the first-frame keyframe")

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)

        # Base pipeline owns the offload hooks on the shared modules. This brings
        # the shared transformer to the requested block-swap state (wrap/unwrap +
        # offload) in the correct order, BEFORE the i2v pipeline is built (or
        # re-cached) so it always references the correct (wrapped or stock) object.
        self._ensure_ltx2_swap_and_offload(blocks_to_swap)
        pipeline = self._ensure_ltx2_i2v_pipeline()

        # Normalize the keyframe to RGB PIL; the pipeline's video_processor
        # handles the resize/fit to (width, height).
        if not isinstance(input_image, Image.Image):
            raise RuntimeError("img2vid input_image must be a PIL.Image")
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Resolve parameters (mirrors _generate_txt2vid_ltx2).
        prompt = params.get("prompt", "") or ""
        negative_prompt = params.get("negative_prompt", "") or ""
        width = int(params.get("width", 768))
        height = int(params.get("height", 512))
        num_frames = int(params.get("num_frames", 121))
        frame_rate = float(params.get("frame_rate", 24.0))
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        num_videos_per_prompt = int(params.get("num_videos_per_prompt", 1))
        max_sequence_length = int(params.get("max_sequence_length", 1024))
        audio_enable = bool(params.get("audio_enable", True))

        # Seed: -1 (or negative/None) -> random draw, recorded back for the caller.
        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        callback = None
        if progress_callback is not None:
            def _cb(pipe, step_index, timestep, callback_kwargs):
                try:
                    progress_callback(step_index + 1, num_inference_steps)
                except Exception:
                    pass
                return callback_kwargs
            callback = _cb

        print(f"[LTX-2.3] img2vid: {width}x{height} num_frames={num_frames} "
              f"fps={frame_rate} steps={num_inference_steps} cfg={guidance_scale} "
              f"seed={seed} audio={audio_enable}")

        video, audio = pipeline(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="np",
            return_dict=False,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=callback,
        )

        frames_np = video[0]  # [T, H, W, C]
        frames = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        audio_out = None
        audio_sample_rate = None
        if audio_enable and audio is not None:
            try:
                audio_sample_rate = int(pipeline.vocoder.config.output_sampling_rate)
            except Exception:
                audio_sample_rate = 24000
            try:
                audio_out = audio[0].detach().float().cpu()
            except Exception as e:
                print(f"[LTX-2.3] audio extraction failed ({e}); saving video without audio")
                audio_out = None
                audio_sample_rate = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames, audio_out, audio_sample_rate, seed

    def _generate_txt2vid_ltx2(
        self,
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        step_callback: Optional[Callable] = None,
    ):
        """Text-to-video generation with LTX-2.3.

        Returns (frames, audio, audio_sample_rate, actual_seed) — see module docstring.
        """
        if not self.ltx2_components:
            raise RuntimeError("LTX-2.3 components not loaded. Please load an LTX-2.3 model first.")

        blocks_to_swap = int(params.get("blocks_to_swap", 0) or 0)
        pipeline = self._ensure_ltx2_swap_and_offload(blocks_to_swap)

        # Resolve parameters
        prompt = params.get("prompt", "") or ""
        negative_prompt = params.get("negative_prompt", "") or ""
        width = int(params.get("width", 768))
        height = int(params.get("height", 512))
        num_frames = int(params.get("num_frames", 121))
        frame_rate = float(params.get("frame_rate", 24.0))
        num_inference_steps = int(params.get("num_inference_steps", 8))
        guidance_scale = float(params.get("guidance_scale", 1.0))
        num_videos_per_prompt = int(params.get("num_videos_per_prompt", 1))
        max_sequence_length = int(params.get("max_sequence_length", 1024))
        audio_enable = bool(params.get("audio_enable", True))

        # Seed: -1 (or negative/None) -> random draw, recorded back for the caller.
        seed = params.get("seed", -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Progress: LTX2Pipeline invokes callback_on_step_end(pipe, i, t, kwargs)
        # at the end of every denoise step and expects a dict back.
        callback = None
        if progress_callback is not None:
            def _cb(pipe, step_index, timestep, callback_kwargs):
                try:
                    progress_callback(step_index + 1, num_inference_steps)
                except Exception:
                    pass
                return callback_kwargs
            callback = _cb

        print(f"[LTX-2.3] txt2vid: {width}x{height} num_frames={num_frames} "
              f"fps={frame_rate} steps={num_inference_steps} cfg={guidance_scale} "
              f"seed={seed} audio={audio_enable}")

        video, audio = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            output_type="np",
            return_dict=False,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=callback,
        )

        # video: np.ndarray [B, T, H, W, C] float in [0, 1] (output_type="np").
        frames_np = video[0]  # [T, H, W, C]
        frames = (np.clip(frames_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)

        # audio: torch.Tensor [B, channels, samples]; take batch 0 -> [channels, samples].
        audio_out = None
        audio_sample_rate = None
        if audio_enable and audio is not None:
            try:
                audio_sample_rate = int(pipeline.vocoder.config.output_sampling_rate)
            except Exception:
                audio_sample_rate = 24000
            try:
                audio_out = audio[0].detach().float().cpu()
            except Exception as e:
                print(f"[LTX-2.3] audio extraction failed ({e}); saving video without audio")
                audio_out = None
                audio_sample_rate = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames, audio_out, audio_sample_rate, seed
