from typing import Any, Dict, Optional
import random

import torch


class SenseNovaMixin:
    """SenseNovaMixin: SenseNova-U1.5-8B-MoT generation methods.

    txt2img, img2img (SDEdit) and inpaint (RePaint) -- all three share the
    same encode_prompt() prefix + Euler denoise loop; img2img/inpaint call
    core/models/sensenova/sensenova_pipeline_ops.py's denoise_loop_img2img/
    denoise_loop_inpaint instead of denoise_loop. Reference-image editing and
    spatial outpaint remain unimplemented (refused at the route, see
    routes.py's `_reject_if_sensenova_unsupported`). Pixel-space (no VAE): the
    transformer is the ONLY component (self.sensenova_components["transformer"]),
    all-or-nothing residency, the same shape MiniT2I uses.
    """

    def _sensenova_move(self, component_name: str, target_device: str):
        # Takes the component KEY, not the component. Passing the module itself
        # made `.get()` return None and silently skipped the move, leaving the
        # transformer on CPU -- invisible to the smoke script, which moves the
        # model itself and never calls this.
        if not isinstance(component_name, str):
            raise TypeError(
                f"_sensenova_move expects a component key, got {type(component_name).__name__}")
        comp = (self.sensenova_components or {}).get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[SenseNova] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _sensenova_apply_attention_backend(self, transformer, params: Dict[str, Any]):
        """Stamp the conduit attention backend (native/flash/sage/tq) onto every
        Qwen3Attention module, mirroring the other DiT mixins' attention_type
        wiring."""
        from core.attention import normalize_backend
        from core.models.sensenova import sensenova_pipeline_ops as ops

        attn_backend = normalize_backend(params.get("attention_type"))
        count = ops.set_attention_backend(transformer, attn_backend)
        print(f"[SenseNova] Attention backend: {attn_backend} "
              f"(from attention_type={params.get('attention_type')!r}, {count} module(s) stamped)")

    def _sensenova_common_params(self, params: Dict[str, Any], default_w: int, default_h: int) -> Dict[str, Any]:
        from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS
        from core.models.sensenova import sensenova_pipeline_ops as ops

        seed = params.get("seed", -1)
        if seed is None or seed == -1:
            seed = random.randint(0, 2**32 - 1)
        req_w = int(params.get("width") or default_w)
        req_h = int(params.get("height") or default_h)
        width, height = ops.normalize_resolution(req_w, req_h)
        if (width, height) != (req_w, req_h):
            print(f"[SenseNova] Resolution aligned to the /32 token grid: {req_w}x{req_h} -> {width}x{height}")

        # Structural /32 alignment is enforced above (snap, never refuse). The
        # ~4MP bucket range is a documented-preference range, not a bound --
        # informational only (upstream's own examples/t2i/inference.py only
        # warns on an off-bucket size too; see SENSENOVA_FACTS scratchpad note).
        area_mp = (width * height) / 1_000_000.0
        if area_mp < 3.0 or area_mp > 5.0:
            try:
                from api.generation_status import add_warning
                add_warning(
                    f"{width}x{height} ({area_mp:.2f} MP) is outside SenseNova U1.5's documented "
                    f"~4 MP resolution buckets. The request is not refused (the model's own "
                    f"structural range is ~256x256 to ~16.7 MP), but this size is untested.",
                    code="sensenova_resolution",
                )
            except Exception:
                pass

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "num_inference_steps": int(params.get("steps") or SENSENOVA_GENERATION_DEFAULTS["steps"]),
            "cfg_scale": float(params.get("cfg_scale", SENSENOVA_GENERATION_DEFAULTS["cfg_scale"])),
            "timestep_shift": float(params.get("timestep_shift", SENSENOVA_GENERATION_DEFAULTS["timestep_shift"])),
            "width": width,
            "height": height,
        }

    def _generate_txt2img_sensenova(self, params, progress_callback=None, step_callback=None) -> tuple:
        if not self.sensenova_components:
            raise RuntimeError("SenseNova components not loaded.")
        from core.models.sensenova import sensenova_pipeline_ops as ops

        print("[SenseNova] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._sensenova_common_params(params, 1024, 1024)
        transformer = self.sensenova_components["transformer"]
        tokenizer = self.sensenova_components["tokenizer"]

        self._sensenova_apply_attention_backend(transformer, params)
        self._sensenova_move("transformer", device)
        prefix = None
        try:
            def _prefill_note():
                # A real, multi-second stall (the prefix KV-cache forward pass)
                # that must not read as a hang before step 0 -- see
                # sensenova_pipeline_ops.encode_prompt's docstring.
                if progress_callback is not None:
                    try:
                        progress_callback(
                            0, 1, None,
                            phase_label="Encoding prompt (SenseNova prefix pass -- this can take several seconds)")
                    except Exception as exc:
                        print(f"[SenseNova] prefill progress callback raised: {exc}")

            prefix = ops.encode_prompt(
                transformer, tokenizer, cfg["prompt"], cfg["height"], cfg["width"], cfg["cfg_scale"],
                prefill_callback=_prefill_note,
            )

            def _step_bridge(j, total, image_prediction, _mask, _extra):
                # sensenova_pipeline_ops's step_callback carries the tensor,
                # unlike its progress_callback(step, total); bridge it into
                # SushiUI's unified progress_callback(step, total, latents,
                # cfg_metrics, pred_original_sample). image_prediction IS the
                # model's current x0-parameterized estimate, so it stands in
                # for both positions.
                if progress_callback is None:
                    return
                try:
                    progress_callback(j, total, image_prediction, None, image_prediction)
                except Exception as exc:
                    print(f"[SenseNova] progress_callback raised: {exc}")

            x = ops.denoise_loop(
                transformer, prefix,
                seed=cfg["seed"], cfg_scale=cfg["cfg_scale"], timestep_shift=cfg["timestep_shift"],
                num_inference_steps=cfg["num_inference_steps"],
                step_callback=_step_bridge,
            )
        finally:
            # Defence in depth: _euler_run already clears both prefix KV caches
            # in its own finally, but that path is only entered once
            # denoise_loop reaches it. Idempotent (clear_flash_kv_cache guards
            # with hasattr), so a second call here after a clean run is a no-op.
            if prefix is not None:
                try:
                    ops.clear_prefix_caches(prefix)
                except Exception:
                    pass
            self._sensenova_move("transformer", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = ops.tensor_to_image(x)
        print("[SenseNova] txt2img completed")
        return image, cfg["seed"], 0

    def _generate_img2img_sensenova(self, params, init_image, progress_callback=None, step_callback=None) -> tuple:
        if not self.sensenova_components:
            raise RuntimeError("SenseNova components not loaded.")
        from api.param_defaults import GENERATION_DEFAULTS
        from core.models.sensenova import sensenova_pipeline_ops as ops

        print("[SenseNova] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._sensenova_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", GENERATION_DEFAULTS["denoising_strength"]))
        transformer = self.sensenova_components["transformer"]
        tokenizer = self.sensenova_components["tokenizer"]

        self._sensenova_apply_attention_backend(transformer, params)
        self._sensenova_move("transformer", device)
        prefix = None
        try:
            def _prefill_note():
                if progress_callback is not None:
                    try:
                        progress_callback(
                            0, 1, None,
                            phase_label="Encoding prompt (SenseNova prefix pass -- this can take several seconds)")
                    except Exception as exc:
                        print(f"[SenseNova] prefill progress callback raised: {exc}")

            prefix = ops.encode_prompt(
                transformer, tokenizer, cfg["prompt"], cfg["height"], cfg["width"], cfg["cfg_scale"],
                prefill_callback=_prefill_note,
            )

            def _step_bridge(j, total, image_prediction, _mask, _extra):
                if progress_callback is None:
                    return
                try:
                    progress_callback(j, total, image_prediction, None, image_prediction)
                except Exception as exc:
                    print(f"[SenseNova] progress_callback raised: {exc}")

            x = ops.denoise_loop_img2img(
                transformer, prefix, init_image, denoising_strength,
                seed=cfg["seed"], cfg_scale=cfg["cfg_scale"], timestep_shift=cfg["timestep_shift"],
                num_inference_steps=cfg["num_inference_steps"],
                step_callback=_step_bridge,
            )
        finally:
            # Same defence-in-depth as txt2img: _euler_run already clears both
            # prefix KV caches in its own finally.
            if prefix is not None:
                try:
                    ops.clear_prefix_caches(prefix)
                except Exception:
                    pass
            self._sensenova_move("transformer", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = ops.tensor_to_image(x)
        print("[SenseNova] img2img completed")
        return image, cfg["seed"], 0

    def _generate_inpaint_sensenova(self, params, init_image, mask_image, progress_callback=None,
                                    step_callback=None) -> tuple:
        if not self.sensenova_components:
            raise RuntimeError("SenseNova components not loaded.")
        from api.param_defaults import GENERATION_DEFAULTS
        from core.models.sensenova import sensenova_pipeline_ops as ops

        print("[SenseNova] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._sensenova_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", GENERATION_DEFAULTS["denoising_strength"]))
        mask_blur = int(params.get("mask_blur", GENERATION_DEFAULTS["mask_blur"]) or 0)
        transformer = self.sensenova_components["transformer"]
        tokenizer = self.sensenova_components["tokenizer"]

        self._sensenova_apply_attention_backend(transformer, params)
        self._sensenova_move("transformer", device)
        prefix = None
        try:
            def _prefill_note():
                if progress_callback is not None:
                    try:
                        progress_callback(
                            0, 1, None,
                            phase_label="Encoding prompt (SenseNova prefix pass -- this can take several seconds)")
                    except Exception as exc:
                        print(f"[SenseNova] prefill progress callback raised: {exc}")

            prefix = ops.encode_prompt(
                transformer, tokenizer, cfg["prompt"], cfg["height"], cfg["width"], cfg["cfg_scale"],
                prefill_callback=_prefill_note,
            )

            def _step_bridge(j, total, image_prediction, _mask, _extra):
                if progress_callback is None:
                    return
                try:
                    progress_callback(j, total, image_prediction, None, image_prediction)
                except Exception as exc:
                    print(f"[SenseNova] progress_callback raised: {exc}")

            x = ops.denoise_loop_inpaint(
                transformer, prefix, init_image, mask_image, denoising_strength,
                seed=cfg["seed"], cfg_scale=cfg["cfg_scale"], timestep_shift=cfg["timestep_shift"],
                num_inference_steps=cfg["num_inference_steps"], mask_blur=mask_blur,
                step_callback=_step_bridge,
            )
        finally:
            if prefix is not None:
                try:
                    ops.clear_prefix_caches(prefix)
                except Exception:
                    pass
            self._sensenova_move("transformer", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = ops.tensor_to_image(x)
        print("[SenseNova] inpaint completed")
        return image, cfg["seed"], 0
