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

    def _ensure_ltx2_offload(self):
        """Attach model_cpu_offload hooks once.

        `enable_model_cpu_offload` stages the offload sequence
        (text_encoder -> connectors -> transformer -> vae -> audio_vae ->
        vocoder) so the 19B transformer + Gemma-3 text encoder move through the
        GPU one component at a time. Re-calling it would re-attach hooks, so this
        is guarded by a flag.
        """
        pipeline = self.ltx2_components.get("pipeline")
        if pipeline is None:
            raise RuntimeError("LTX-2.3 pipeline reference missing from components")

        if getattr(self, "_ltx2_offload_enabled", False):
            return pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload(device=device)
                print("[LTX-2.3] enable_model_cpu_offload attached "
                      "(text_encoder->connectors->transformer->vae->audio_vae->vocoder)")
            except Exception as e:
                print(f"[LTX-2.3] enable_model_cpu_offload failed ({e}); "
                      f"falling back to whole-pipeline .to(cuda)")
                pipeline.to(device)
        else:
            print("[LTX-2.3] CUDA unavailable; running pipeline on CPU")

        self._ltx2_offload_enabled = True
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

        # Base pipeline owns the offload hooks on the shared modules.
        self._ensure_ltx2_offload()
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

        pipeline = self._ensure_ltx2_offload()

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
