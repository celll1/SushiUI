"""On-demand generation using the in-training model.

The API writes a request file in ``<output_dir>/.preview_request_<id>.json``
and the trainer picks it up at the next batch boundary (next to the
existing ``.stop_training`` flag).  This module handles the trainer
side: read params, run inference using the in-training UNet + VAE +
text encoders (plus optional additional LoRAs / ControlNets), write
result PNG + meta JSON back to disk.

Supported modes (Phase 2+):
    * ``mode = "txt2img"`` — uses ``custom_sampling_loop``
    * ``mode = "img2img"`` — uses ``custom_img2img_sampling_loop``
    * ``mode = "inpaint"`` — uses ``custom_inpaint_sampling_loop``

Optional extras:
    * LoRA stack on top of the in-training LoRA (SD1.5 / SDXL only; see
      ``_apply_additional_loras``)
    * ControlNet conditioning (builds a real diffusers CN pipeline
      from the TempPipeline's components dict, then runs the same
      sampling loop)

For txt2img we still delegate to the trainer's existing
``generate_sample()`` since that path is well-tested.  img2img /
inpaint paths are implemented here directly because the existing
trainer code only handles txt2img.
"""
from __future__ import annotations

import base64
import io
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from .training_preview_rpc import write_result

if TYPE_CHECKING:
    from PIL import Image
    from .base_trainer import BaseTrainer


# ---------------------------------------------------------------------------
# Helpers — base64 ↔ PIL.Image
# ---------------------------------------------------------------------------

def _decode_b64_image(b64: Optional[str]) -> Optional["Image.Image"]:
    """Decode a base64-encoded PNG/JPEG into a PIL Image, or None."""
    if not b64:
        return None
    from PIL import Image as _Image
    try:
        # Tolerate "data:image/png;base64," prefixes
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return _Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:   # noqa: BLE001
        raise ValueError(f"Could not decode base64 image: {e}")


def _decode_b64_mask(b64: Optional[str]) -> Optional["Image.Image"]:
    """Decode a base64 mask — preserves single-channel or alpha-mode mask."""
    if not b64:
        return None
    from PIL import Image as _Image
    try:
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return _Image.open(io.BytesIO(raw)).convert("L")
    except Exception as e:   # noqa: BLE001
        raise ValueError(f"Could not decode base64 mask: {e}")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class TrainingPreviewGenerator:
    """Bound to a BaseTrainer; processes one preview request at a time.

    Caller (the trainer loop) invokes :meth:`process_request` at a batch
    boundary.  No state is retained between requests — every call sets
    up eval-mode + temp pipeline, runs, then restores train-mode.
    """

    def __init__(self, trainer: "BaseTrainer"):
        self.trainer = trainer
        # Set while a preview holds stacked LoRAs, and holding the exact
        # pipeline + wrapper sites the load touched, so a detach undoes what
        # was attached even if the trainer swapped a module meanwhile.
        # Cleared by ``_detach_additional_loras``.
        self._lora_stack: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API — invoked by the trainer at batch boundaries
    # ------------------------------------------------------------------

    def process_request(self, request_id: str, params: Dict[str, Any]) -> None:
        """Run one preview generation and write result files to disk.

        Every exception path produces a result file with ``ok=False``
        and an error message, so a broken request never crashes
        training.
        """
        output_dir = str(self.trainer.output_dir)
        meta: Dict[str, Any] = {"request_id": request_id, "ok": False}
        png_bytes: Optional[bytes] = None
        # Live-preview frames written during sampling are keyed by this id (read+decoded
        # by the API and broadcast to the WebSocket).
        self._current_request_id = request_id
        try:
            mode = (params.get("mode") or "txt2img").lower()
            self._enter_eval_mode()
            try:
                # Stack any additional LoRAs the user passed on top of
                # the in-training LoRA.  Failure here is fatal for the
                # request (we don't want to silently generate without
                # them).
                self._apply_additional_loras(params.get("loras") or [])
                if mode == "txt2img":
                    image, seed = self._generate_txt2img(params)
                elif mode == "img2img":
                    image, seed = self._generate_img2img(params)
                elif mode == "inpaint":
                    image, seed = self._generate_inpaint(params)
                else:
                    raise ValueError(f"Unsupported preview mode: {mode!r}")
            except Exception as gen_error:
                # A detach failure below would otherwise replace this one, and
                # a model left mid-detach is the more urgent of the two.
                print(f"[TrainingPreview:{request_id}] generation failed: "
                      f"{type(gen_error).__name__}: {gen_error}")
                raise
            finally:
                # Train mode must come back even if the detach raises.
                try:
                    self._detach_additional_loras()
                finally:
                    self._exit_eval_mode()

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            meta.update({
                "ok": True,
                "seed": seed,
                "width": image.width,
                "height": image.height,
                "mode": mode,
            })
        except Exception as e:   # noqa: BLE001
            print(f"[TrainingPreview:{request_id}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            meta.update({
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            })
            png_bytes = None
        finally:
            try:
                write_result(output_dir, request_id, png_bytes, meta)
            except Exception as we:   # noqa: BLE001
                print(f"[TrainingPreview:{request_id}] failed to write result: {we}")
            try:
                from .training_preview_rpc import cleanup_preview_frame
                cleanup_preview_frame(output_dir, request_id)
            except Exception:
                pass
            self._current_request_id = None

    # ------------------------------------------------------------------
    # eval/train mode helpers (idempotent)
    # ------------------------------------------------------------------

    def _enter_eval_mode(self) -> None:
        t = self.trainer
        if hasattr(t, "unet") and t.unet is not None:
            t.unet.eval()
        if hasattr(t, "vae") and t.vae is not None:
            t.vae.eval()
        if hasattr(t, "text_encoder") and t.text_encoder is not None:
            t.text_encoder.eval()
        if getattr(t, "text_encoder_2", None) is not None:
            t.text_encoder_2.eval()

    def _exit_eval_mode(self) -> None:
        t = self.trainer
        if hasattr(t, "unet") and t.unet is not None:
            t.unet.train()
        if hasattr(t, "vae") and t.vae is not None:
            t.vae.train()
        if hasattr(t, "text_encoder") and t.text_encoder is not None:
            t.text_encoder.train()
        if getattr(t, "text_encoder_2", None) is not None:
            t.text_encoder_2.train()

    # ------------------------------------------------------------------
    # LoRA stack: attach / detach additional adapters
    # ------------------------------------------------------------------

    def _apply_additional_loras(self, loras: List[Dict[str, Any]]) -> None:
        """Attach user-specified LoRAs on top of the in-training LoRA.

        Runs the production ``lora_manager.load_loras`` against a TempPipeline
        that subclasses the real diffusers loader mixin, with the trainer's
        ``LoRALinearLayer`` wrappers detoured around the PEFT injection.
        Refuses (rather than generating a picture without them) whenever the
        stack cannot be applied or could not be undone afterwards.
        """
        if not loras:
            return
        from .temp_pipeline import (
            build_temp_pipeline_for_trainer,
            collect_training_lora_sites,
            components_with_peft_adapters,
            training_lora_detour,
        )
        from core.extensions.lora_manager import lora_manager

        # lora_manager never touches the scheduler; the trainer's own is fine.
        sched = getattr(self.trainer, "original_scheduler", None)
        temp = build_temp_pipeline_for_trainer(self.trainer, sched)
        temp.assert_can_stack_loras()
        components = temp.lora_components()

        occupied = components_with_peft_adapters(components)
        if occupied:
            raise RuntimeError(
                "Refusing to stack preview LoRAs: PEFT adapters are already "
                f"installed on {occupied}. The trainer does not create those, "
                "so detaching afterwards could not restore the model to the "
                "state training left it in."
            )

        sites = collect_training_lora_sites(components)
        # Recorded BEFORE the load: a load that raises half way through a stack
        # still has adapters to detach.
        self._lora_stack = {
            "pipeline": temp,
            "sites": sites,
            "names": [f"lora_{i}" for i in range(len(loras))],
        }
        with training_lora_detour(sites):
            lora_manager.load_loras(temp, loras)

    def _detach_additional_loras(self) -> None:
        """Remove everything the preview stacked, restoring module identity.

        ``unload_lora_weights`` strips the PEFT layers and puts the original
        module objects back (the optimizer holds their parameters), which is
        exact only because the apply refused to run over pre-existing PEFT
        adapters. Raises on failure: a model still carrying preview adapters
        would train against a forward the user never asked for.
        """
        stack = self._lora_stack
        self._lora_stack = None
        if stack is None:
            return
        from .temp_pipeline import training_lora_detour
        try:
            with training_lora_detour(stack["sites"]):
                stack["pipeline"].unload_lora_weights()
        except Exception as e:   # noqa: BLE001
            traceback.print_exc()
            raise RuntimeError(
                f"Preview LoRAs {stack['names']} could not be detached "
                f"({type(e).__name__}: {e}); the in-training model may still "
                "carry them. Stop the run before trusting further steps."
            ) from e

    # ------------------------------------------------------------------
    # Mode dispatchers — each path decodes its mode-specific images
    # then routes through the unified ``_run_sampling`` below
    # ------------------------------------------------------------------

    def _generate_txt2img(self, params: Dict[str, Any]):
        return self._run_sampling(params, init_image=None, mask_image=None)

    def _generate_img2img(self, params: Dict[str, Any]):
        init_image = _decode_b64_image(params.get("init_image_base64"))
        if init_image is None:
            raise ValueError("img2img preview missing 'init_image_base64'")
        return self._run_sampling(params, init_image=init_image, mask_image=None)

    def _generate_inpaint(self, params: Dict[str, Any]):
        init_image = _decode_b64_image(params.get("init_image_base64"))
        mask_image = _decode_b64_mask(params.get("mask_image_base64"))
        if init_image is None:
            raise ValueError("inpaint preview missing 'init_image_base64'")
        if mask_image is None:
            raise ValueError("inpaint preview missing 'mask_image_base64'")
        return self._run_sampling(params, init_image=init_image, mask_image=mask_image)

    def _make_preview_callback(
        self, *, request_id, output_dir, width, height,
        preview_interval, preview_predicted_x0, preview_decoder,
    ):
        """Build a custom_sampling_loop progress_callback that writes live-preview frames
        (small latent + meta) to disk for the API to decode + broadcast. Returns None when
        previews aren't applicable (no request id, or FLUX.2 latents which TAESD can't decode)."""
        if not request_id:
            return None
        t = self.trainer
        if getattr(t, "is_flux2", False):
            return None
        from .training_preview_rpc import write_preview_frame
        flags = {
            "is_sdxl": bool(getattr(t, "is_sdxl", False)),
            "is_zimage": bool(getattr(t, "is_zimage", False)),
            "is_anima": bool(getattr(t, "is_anima", False)),
            "is_lens": bool(getattr(t, "is_lens", False)),
            "is_ideogram4": bool(getattr(t, "is_ideogram4", False)),
            "is_minit2i": bool(getattr(t, "is_minit2i", False)),
        }
        seq = [0]

        def _cb(step, total_steps, latents, cfg_metrics=None, pred_original_sample=None):
            try:
                is_last = (step == total_steps - 1)
                if not (step == -1 or step == 0 or is_last
                        or (step > 0 and preview_interval > 0 and step % preview_interval == 0)):
                    return
                lat = (pred_original_sample
                       if (preview_predicted_x0 and pred_original_sample is not None)
                       else latents)
                if lat is None:
                    return
                seq[0] += 1
                write_preview_frame(output_dir, request_id, {
                    "seq": seq[0],
                    "step": step,
                    "total": total_steps,
                    "latents": lat.detach().to("cpu", dtype=torch.float32),
                    "image_width": width,
                    "image_height": height,
                    "preview_decoder": preview_decoder,
                    **flags,
                })
            except Exception:
                pass  # never break generation for a preview frame

        return _cb

    # ------------------------------------------------------------------
    # Unified sampling — used by txt2img / img2img / inpaint.  Dispatches
    # to the appropriate ``custom_*_sampling_loop`` based on which
    # images are present.  Shares text-encoding, scheduler build,
    # ControlNet pipeline construction, and advanced CFG/NAG passthrough.
    # ------------------------------------------------------------------

    def _run_sampling(
        self,
        params: Dict[str, Any],
        *,
        init_image: Optional["Image.Image"],
        mask_image: Optional["Image.Image"],
    ):
        import random
        from PIL import Image as _Image
        from core.inference.schedulers import get_scheduler
        from core.inference.custom_sampling import (
            custom_sampling_loop,
            custom_img2img_sampling_loop,
            custom_inpaint_sampling_loop,
        )
        from .temp_pipeline import build_temp_pipeline_for_trainer

        t = self.trainer
        prompt = params.get("prompt") or ""
        if not prompt:
            raise ValueError("preview request missing 'prompt'")
        width  = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        # Snap to multiples of 8 (VAE requirement)
        width  -= width  % 8
        height -= height % 8
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid size after snap-to-8: {width}x{height}")

        # ----- resize init / mask (if present) -----
        if init_image is not None:
            init_image = init_image.resize((width, height), _Image.LANCZOS)
        if mask_image is not None:
            mask_image = mask_image.resize((width, height), _Image.LANCZOS)

        # ----- scheduler -----
        schedule_type_mapped = params.get("schedule_type", "uniform")
        if schedule_type_mapped == "sgm_uniform":
            schedule_type_mapped = "uniform"
        # get_scheduler expects a "pipeline" with .scheduler attr
        class _SchedHolder:
            def __init__(self, sch): self.scheduler = sch
        scheduler = get_scheduler(
            pipeline=_SchedHolder(t.original_scheduler),
            sampler=params.get("sampler", "euler"),
            schedule_type=schedule_type_mapped,
        )

        # ----- build pipeline shim -----
        pipeline = build_temp_pipeline_for_trainer(t, scheduler)

        # ----- text encoding (uses trainer's encoder, eval-mode already) -----
        t.move_text_encoder_to_gpu()
        negative_prompt = params.get("negative_prompt") or ""
        if pipeline.is_sdxl:
            prompt_embeds, pooled_prompt_embeds = t.encode_prompt(prompt, requires_grad=False)
            negative_prompt_embeds, negative_pooled_prompt_embeds = \
                t.encode_prompt(negative_prompt, requires_grad=False)
        else:
            prompt_embeds = t.encode_prompt(prompt, requires_grad=False)
            negative_prompt_embeds = t.encode_prompt(negative_prompt, requires_grad=False)
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None
        # Pad negative to match positive length (prompt chunking)
        if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
            diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
            if diff > 0:
                pad = torch.zeros(
                    (negative_prompt_embeds.shape[0], diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device,
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)
        t.move_text_encoder_to_cpu()
        torch.cuda.empty_cache()

        # ----- seed -----
        seed_in = int(params.get("seed", -1))
        actual_seed = random.randint(0, 2**32 - 1) if seed_in < 0 else seed_in
        generator = torch.Generator(device=t.device).manual_seed(actual_seed)

        # ----- ControlNet (Phase 4) -----
        # If controlnets present, swap the temp pipeline for a real
        # diffusers ControlNet pipeline built from its components.
        cn_pipeline = self._maybe_build_controlnet_pipeline(
            pipeline, params.get("controlnets") or [], width, height,
        )
        runtime_pipeline = cn_pipeline or pipeline

        # ----- move main model + VAE to GPU for inference -----
        t.move_main_model_to_gpu()
        t.move_vae_to_gpu()

        is_v_prediction = scheduler.config.get("prediction_type") == "v_prediction"
        guidance_rescale = 0.7 if is_v_prediction else 0.0

        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 7.0))

        # ----- live-preview frame callback -----
        # Write a small latent + meta to disk per preview interval; the API decodes it
        # (TAESD) and broadcasts to the WebSocket, so the user sees a live preview during
        # the in-training generation. Best-effort: never raise into the sampling loop.
        preview_cb = self._make_preview_callback(
            request_id=getattr(self, "_current_request_id", None),
            output_dir=str(t.output_dir),
            width=width, height=height,
            preview_interval=int(params.get("preview_interval", 4) or 4),
            preview_predicted_x0=bool(params.get("preview_predicted_x0", False)),
            preview_decoder=str(params.get("preview_decoder", "matrix")),
        )

        # ----- params shared by all 3 sampling loops -----
        common_kwargs: Dict[str, Any] = dict(
            progress_callback=preview_cb,
            pipeline=runtime_pipeline,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            generator=generator,
            width=width,
            height=height,
            cfg_schedule_type=params.get("cfg_schedule_type", "constant"),
            cfg_schedule_min=float(params.get("cfg_schedule_min", 1.0)),
            cfg_schedule_max=params.get("cfg_schedule_max"),
            cfg_schedule_power=float(params.get("cfg_schedule_power", 2.0)),
            cfg_rescale_snr_alpha=float(params.get("cfg_rescale_snr_alpha", 0.0)),
            dynamic_threshold_percentile=float(params.get("dynamic_threshold_percentile", 0.0)),
            dynamic_threshold_mimic_scale=float(params.get("dynamic_threshold_mimic_scale", 1.0)),
            nag_enable=bool(params.get("nag_enable", False)),
            nag_scale=float(params.get("nag_scale", 5.0)),
            nag_tau=float(params.get("nag_tau", 3.5)),
            nag_alpha=float(params.get("nag_alpha", 0.25)),
            nag_sigma_end=float(params.get("nag_sigma_end", 0.0)),
            attention_type=params.get("attention_type", "normal"),
        )

        # ControlNet image args (when in CN mode) — shared across all 3 modes
        if cn_pipeline is not None:
            cn_configs = params.get("controlnets") or []
            cn_images = []
            cn_scales = []
            for cn in cn_configs:
                img = _decode_b64_image(cn.get("image_base64"))
                if img is None:
                    continue
                img = img.resize((width, height), _Image.LANCZOS)
                cn_images.append(img)
                cn_scales.append(float(cn.get("strength", 1.0)))
            common_kwargs["controlnet_images"] = cn_images
            common_kwargs["controlnet_conditioning_scale"] = (
                cn_scales if len(cn_scales) > 1 else (cn_scales[0] if cn_scales else 1.0)
            )

        # ----- dispatch to the right sampling loop -----
        try:
            with torch.autocast(device_type=t.device.type, dtype=t.training_dtype):
                if init_image is None:
                    # txt2img — no init / strength
                    image = custom_sampling_loop(**common_kwargs)
                elif mask_image is None:
                    # img2img — init_image + strength
                    image = custom_img2img_sampling_loop(
                        init_image=init_image,
                        strength=float(params.get("denoising_strength", 0.75)),
                        **common_kwargs,
                    )
                else:
                    # inpaint — init + mask + fill controls
                    image = custom_inpaint_sampling_loop(
                        init_image=init_image,
                        mask_image=mask_image,
                        strength=float(params.get("denoising_strength", 0.75)),
                        inpaint_fill_mode=params.get("inpaint_fill_mode", "original"),
                        inpaint_fill_strength=float(params.get("inpaint_fill_strength", 1.0)),
                        inpaint_blur_strength=float(params.get("inpaint_blur_strength", 1.0)),
                        **common_kwargs,
                    )
        finally:
            t.move_main_model_to_cpu()
            t.move_vae_to_cpu()
            try:
                from core.extensions.controlnet_manager import controlnet_manager
                controlnet_manager.remove_lllite_patches()
                controlnet_manager.offload_controlnets_to_cpu()
            except Exception as e:
                print(f"[TrainingInference] WARNING: ControlNet offload failed: {e}")
            torch.cuda.empty_cache()

        return image, actual_seed

    # ------------------------------------------------------------------
    # Phase 4: ControlNet — build a real diffusers CN pipeline
    # ------------------------------------------------------------------

    def _maybe_build_controlnet_pipeline(
        self,
        temp_pipeline,
        controlnets: List[Dict[str, Any]],
        width: int, height: int,
    ):
        """Return a StableDiffusion(XL)ControlNetPipeline if any
        ``controlnets`` are configured; else ``None`` (caller uses the
        plain TempPipeline).

        The returned pipeline shares unet/vae/text_encoders with the
        trainer — no weight copy.  After generation, the caller doesn't
        need to dispose anything (Python GC handles the wrapper).
        """
        if not controlnets:
            return None
        from core.extensions.controlnet_manager import controlnet_manager

        loaded_cns = []
        for cn_cfg in controlnets:
            model_path = cn_cfg.get("model") or cn_cfg.get("path")
            if not model_path:
                continue
            cn = controlnet_manager.load_controlnet(
                model_path=model_path,
                device="cuda",
                dtype=self.trainer.training_dtype,
                is_lllite=False,
            )
            if cn is not None:
                # load_controlnet only stages on the FIRST load; a cache hit
                # returns the module exactly as it was left, and this path's own
                # finally offloads it to CPU after every preview. Without this
                # re-stage, preview #2 onward dies on a device mismatch.
                cn = cn.to(device=self.trainer.device, dtype=self.trainer.training_dtype)
                loaded_cns.append(cn)
        if not loaded_cns:
            return None

        cn_arg = loaded_cns[0] if len(loaded_cns) == 1 else loaded_cns
        if temp_pipeline.is_sdxl:
            from diffusers import StableDiffusionXLControlNetPipeline
            cn_pipe = StableDiffusionXLControlNetPipeline(
                vae=temp_pipeline.vae,
                text_encoder=temp_pipeline.text_encoder,
                text_encoder_2=temp_pipeline.text_encoder_2,
                tokenizer=temp_pipeline.tokenizer,
                tokenizer_2=temp_pipeline.tokenizer_2,
                unet=temp_pipeline.unet,
                controlnet=cn_arg,
                scheduler=temp_pipeline.scheduler,
            )
        else:
            from diffusers import StableDiffusionControlNetPipeline
            cn_pipe = StableDiffusionControlNetPipeline(
                vae=temp_pipeline.vae,
                text_encoder=temp_pipeline.text_encoder,
                tokenizer=temp_pipeline.tokenizer,
                unet=temp_pipeline.unet,
                controlnet=cn_arg,
                scheduler=temp_pipeline.scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        # custom_sampling_loop needs vae_scale_factor / image_processor
        cn_pipe.vae_scale_factor = temp_pipeline.vae_scale_factor
        return cn_pipe
