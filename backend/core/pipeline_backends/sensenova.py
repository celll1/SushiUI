from typing import Any, Dict, List, Optional
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

    def _load_lora_sensenova(self, lora_configs: List[Dict]) -> int:
        """Wrap the SenseNova gen-branch Linears with runtime LoRA adapters.

        Never merges into the base (see sensenova_lora.py's module docstring)
        -- restore_originals must always be called in a finally, mirroring
        Ideogram4Mixin._load_lora_ideogram4/_unload_lora_ideogram4.
        """
        from core.models.sensenova.sensenova_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs or not self.sensenova_components:
            return 0

        transformer = self.sensenova_components["transformer"]
        if not hasattr(self, "_sensenova_lora_orig"):
            self._sensenova_lora_orig: Dict[str, torch.nn.Module] = {}
            self._sensenova_lora_keys: set = set()

        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", cfg.get("weight", 1.0)))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[SenseNova LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._sensenova_lora_orig, self._sensenova_lora_keys,
                )
                print(f"[SenseNova LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} modules={len(grouped)} wrapped={applied} strength={strength}")
                total += applied
            except Exception as e:
                print(f"[SenseNova LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_sensenova(self) -> int:
        """Restore every SenseNova transformer Linear to its pre-LoRA original."""
        from core.models.sensenova.sensenova_lora import restore_originals
        if not self.sensenova_components:
            return 0
        restored = 0
        if getattr(self, "_sensenova_lora_keys", None):
            restored += restore_originals(
                self.sensenova_components["transformer"],
                self._sensenova_lora_orig, self._sensenova_lora_keys,
            )
        if restored:
            print(f"[SenseNova LoRA] Unloaded {restored} LoRA wrappers")
        return restored

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
            # Write the snapped size back: `params` is what the route later hands
            # to prepare_params_for_db() and the PNG metadata writer, so leaving
            # the request size here would record a resolution that was never
            # generated -- and "reuse parameters" from the gallery would replay it.
            params["width"] = width
            params["height"] = height

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
        applied_lora = 0
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])

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
            if applied_lora:
                self._unload_lora_sensenova()
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
        applied_lora = 0
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])

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
            if applied_lora:
                self._unload_lora_sensenova()
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
        applied_lora = 0
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])

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
            if applied_lora:
                self._unload_lora_sensenova()
            self._sensenova_move("transformer", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = ops.tensor_to_image(x)
        print("[SenseNova] inpaint completed")
        return image, cfg["seed"], 0
