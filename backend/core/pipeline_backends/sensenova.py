from typing import Any, Dict, List, Optional
import random

import torch


# Upstream's largest demonstrated reference count for `it2i_generate`
# (reference-image editing). Bounds the request, not a hard architectural
# limit; enforced at the route (routes.py's ref_images cap check).
SENSENOVA_MAX_REFERENCE_IMAGES = 5


class SenseNovaMixin:
    """SenseNovaMixin: SenseNova-U1.5-8B-MoT generation methods.

    txt2img, img2img (SDEdit) and inpaint (RePaint) -- all three share the
    same encode_prompt() prefix + Euler denoise loop; img2img/inpaint call
    core/models/sensenova/sensenova_pipeline_ops.py's denoise_loop_img2img/
    denoise_loop_inpaint instead of denoise_loop. Reference-image editing
    (`ref_images`/`img_cfg_scale`, passed through to encode_prompt()) is a
    PREFIX-phase concern, orthogonal to and combinable with all three denoise
    loops. Spatial outpaint remains unimplemented (refused at the route, see
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

    def _sensenova_maybe_install_mot_eviction(self, params: Dict[str, Any], transformer, device):
        """Install the MoT phase-exclusive half-weight evictor when requested.

        See mot_phase_eviction.py for the mechanism; `blocks_to_swap` is NOT
        wired here (it is inert for SenseNova today, and this boolean is not
        an alias for it).
        """
        from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

        enabled = bool(params.get(
            "sensenova_mot_phase_eviction",
            SENSENOVA_GENERATION_DEFAULTS["sensenova_mot_phase_eviction"],
        ))
        if not enabled:
            return None
        from core.models.sensenova import mot_phase_eviction
        return mot_phase_eviction.install(transformer, device)

    def _sensenova_teardown_mot_eviction(self, transformer, evictor) -> None:
        """Always call from the generation's ``finally``, AFTER the whole-model
        `_sensenova_move("transformer", "cpu")` restore -- robust to a mid-run
        exception/cancellation (the full-model move already normalizes device
        placement regardless of which phase the evictor last saw). Does NOT
        un-pin (see mot_phase_eviction.py); this only clears the callback so a
        later, eviction-off generation on the same loaded transformer never
        sees a stale hook."""
        from core.models.sensenova import mot_phase_eviction
        mot_phase_eviction.uninstall(transformer, evictor)

    def _sensenova_maybe_install_kv_streaming(self, params: Dict[str, Any], transformer, device):
        """Install the 2-slot flash-KV prefix streamer when requested.

        See kv_cache_streaming.py for the mechanism. Independent of MoT phase
        eviction (disjoint tensors, disjoint hook points, no shared
        coordinator) -- installing both together is expected to compose.
        """
        from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

        enabled = bool(params.get(
            "sensenova_kv_cache_streaming",
            SENSENOVA_GENERATION_DEFAULTS["sensenova_kv_cache_streaming"],
        ))
        if not enabled:
            return None
        from core.models.sensenova import kv_cache_streaming
        return kv_cache_streaming.install(transformer, device)

    def _sensenova_teardown_kv_streaming(self, transformer, streamer) -> None:
        """Always call from the generation's ``finally``, AFTER the whole-model
        restore -- idempotent with ``clear_prefix_caches``'s own defence-in-
        depth teardown call (``sensenova_pipeline_ops.py``), so a normal run
        that already tore the streamer down here is a no-op."""
        from core.models.sensenova import kv_cache_streaming
        kv_cache_streaming.uninstall(transformer, streamer)

    def _sensenova_style_triple(self, style_dict: Dict[str, Any], transformer, device, prefix,
                                seed: Optional[int], ref_index: int = 0):
        """Build a single (StyleTransferConfig, ref_x0, eps_ref) triple from one
        style_transfer dict, mirroring Krea2/Lens's own ``_style_triple``.

        ``axes_dims`` is the t/h/w RoPE split from ``Qwen3Attention.__init__``
        (t gets ``head_dim//2``, h and w ``head_dim//4`` each), and SenseNova
        rotates halves rather than interleaved pairs, hence ``rope_layout``.

        Pixel-space, no VAE, so the reference is resized outright to the exact
        target size (capture and inject must yield the same token count) and a
        non-matching aspect ratio is warned about. ``ref_index`` decorrelates
        the fixed noise across simultaneous references (``seed+991+ref_index``,
        as Krea2/Lens)."""
        from core.inference.reference_style import style_config_from_dict
        from core.models.sensenova import sensenova_pipeline_ops as ops

        image = style_dict["image"]
        width, height = prefix.image_size
        if image.width > 0 and image.height > 0:
            target_ratio = width / height
            ref_ratio = image.width / image.height
            if abs(target_ratio - ref_ratio) > 0.01 * target_ratio:
                msg = (f"[SenseNova] style reference {image.width}x{image.height} does not match this "
                       f"generation's {width}x{height} aspect ratio; it will be resized (not cropped) to "
                       f"fit -- the capture and inject forwards must produce the same image-token count.")
                print(msg)
                try:
                    from api.generation_status import add_warning
                    add_warning(msg, code="sensenova_style_reference_resized")
                except Exception:
                    pass

        cfg = style_config_from_dict(style_dict)
        cfg.resolve_default_block_range(len(transformer.language_model.model.layers))
        head_dim = transformer.language_model.model.layers[0].self_attn.head_dim
        cfg.axes_dims = (head_dim // 2, head_dim // 4, head_dim // 4)
        cfg.rope_layout = "rotate_half"

        noise_scale = ops.compute_noise_scale(transformer, prefix.grid_h, prefix.grid_w, prefix.merge_size)
        ref_seed = None if seed is None or seed < 0 else (int(seed) + 991 + ref_index) % (2**32)
        ref_x0, eps_ref = ops.prepare_style_reference(
            image, height, width, device, prefix.dtype, ref_seed, noise_scale)
        return cfg, ref_x0, eps_ref

    def _sensenova_style_config(self, params: Dict[str, Any], transformer, device, prefix, seed: Optional[int]):
        """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
        ``params["style_transfer"]``, or ``(None, None, None)`` when no style
        reference is attached. Single-reference path, delegates to
        ``_sensenova_style_triple`` with ``ref_index=0``."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None
        return self._sensenova_style_triple(style_dict, transformer, device, prefix, seed, ref_index=0)

    def _sensenova_style_configs(self, params: Dict[str, Any], transformer, device, prefix, seed: Optional[int]):
        """Build the full style-transfer configuration for SenseNova generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple) and the multi-reference path
        (``style_refs``, populated ONLY when ``params["style_transfers"]`` has
        more than one entry), mirroring Krea2/Lens exactly.

        Returns ``(style_cfg, style_ref_x0, style_eps_ref, style_refs,
        style_combine_mode)``."""
        style_list = params.get("style_transfers")
        if style_list and len(style_list) > 1:
            combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
            refs = []
            for idx, style_dict in enumerate(style_list):
                if not style_dict or not style_dict.get("image"):
                    continue
                refs.append(self._sensenova_style_triple(style_dict, transformer, device, prefix, seed, ref_index=idx))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._sensenova_style_config(params, transformer, device, prefix, seed)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

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

        ref_images = params.get("ref_images") or []
        img_cfg_scale = float(params.get("img_cfg_scale", SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]))

        # The <image>-placeholder/reference count check lives at the route
        # (routes.py's _reject_if_sensenova_ref_placeholders_exceed_refs):
        # raising ValidationError here would be re-wrapped as a 500 by the
        # route's generic except, since it is GenerationError's sibling.
        if not ref_images and img_cfg_scale != SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]:
            msg = (f"[SenseNova] img_cfg_scale={img_cfg_scale} has no effect without ref_images "
                   f"(it only applies to reference-image editing).")
            print(msg)
            try:
                from api.generation_status import add_warning
                add_warning(msg, code="sensenova_img_cfg_scale_no_refs")
            except Exception:
                pass

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            # Feeds the CFG uncond branch in encode_prompt(); None/empty falls
            # back to the original empty-string uncond (see MODEL_FACTS.md).
            "negative_prompt": params.get("negative_prompt") or None,
            "num_inference_steps": int(params.get("steps") or SENSENOVA_GENERATION_DEFAULTS["steps"]),
            "cfg_scale": float(params.get("cfg_scale", SENSENOVA_GENERATION_DEFAULTS["cfg_scale"])),
            "timestep_shift": float(params.get("timestep_shift", SENSENOVA_GENERATION_DEFAULTS["timestep_shift"])),
            "img_cfg_scale": img_cfg_scale,
            "cfg_norm": params.get("cfg_norm", SENSENOVA_GENERATION_DEFAULTS["cfg_norm"]),
            # routes.py already decodes uploads into PIL Image objects before
            # putting them in params["ref_images"]; nothing to decode here.
            "ref_images": ref_images,
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
        evictor = None
        kv_streamer = None
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])
            # Installed AFTER LoRA (if any) has wrapped the gen-branch Linears
            # in place -- see mot_phase_eviction.py's MotPhaseEvictor docstring.
            evictor = self._sensenova_maybe_install_mot_eviction(params, transformer, device)
            # Independent of MoT eviction (disjoint tensors/hooks); must be
            # installed before encode_prompt() so _finalize_prefix_caches sees it.
            kv_streamer = self._sensenova_maybe_install_kv_streaming(params, transformer, device)

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
                prefill_callback=_prefill_note, negative_prompt=cfg["negative_prompt"],
                ref_images=cfg["ref_images"], img_cfg_scale=cfg["img_cfg_scale"],
            )

            # Training-free reference-style transfer. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None, None,
            # "stack"), no-op below). Mechanically independent of it2i reference
            # images above (prefix tokens vs runtime K/V concat), so both can be
            # active simultaneously. Built AFTER `prefix` exists -- needs
            # prefix.grid_h/grid_w/merge_size (noise_scale) and prefix.dtype.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._sensenova_style_configs(params, transformer, device, prefix, cfg["seed"])

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
                cfg_norm=cfg["cfg_norm"],
                num_inference_steps=cfg["num_inference_steps"],
                step_callback=_step_bridge,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
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
            # Defence in depth (L3): _style_ctx is a CLASS attribute on
            # Qwen3Attention that outlives this generation (the module lives in
            # sys.modules) -- _euler_run already clears it in its own finally on
            # every exit path, but a raise before _euler_run is ever reached
            # (e.g. during _sensenova_style_configs) could not have armed it in
            # the first place, so this is a pure safety net, not a required path.
            try:
                from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Attention
                Qwen3Attention._style_ctx = None
            except Exception:
                pass
            if applied_lora:
                self._unload_lora_sensenova()
            self._sensenova_move("transformer", "cpu")
            self._sensenova_teardown_mot_eviction(transformer, evictor)
            self._sensenova_teardown_kv_streaming(transformer, kv_streamer)
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
        evictor = None
        kv_streamer = None
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])
            evictor = self._sensenova_maybe_install_mot_eviction(params, transformer, device)
            kv_streamer = self._sensenova_maybe_install_kv_streaming(params, transformer, device)

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
                prefill_callback=_prefill_note, negative_prompt=cfg["negative_prompt"],
                ref_images=cfg["ref_images"], img_cfg_scale=cfg["img_cfg_scale"],
            )

            # Training-free reference-style transfer -- same wiring as txt2img.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._sensenova_style_configs(params, transformer, device, prefix, cfg["seed"])

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
                cfg_norm=cfg["cfg_norm"],
                num_inference_steps=cfg["num_inference_steps"],
                step_callback=_step_bridge,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
            )
        finally:
            # Same defence-in-depth as txt2img: _euler_run already clears both
            # prefix KV caches in its own finally.
            if prefix is not None:
                try:
                    ops.clear_prefix_caches(prefix)
                except Exception:
                    pass
            try:
                from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Attention
                Qwen3Attention._style_ctx = None
            except Exception:
                pass
            if applied_lora:
                self._unload_lora_sensenova()
            self._sensenova_move("transformer", "cpu")
            self._sensenova_teardown_mot_eviction(transformer, evictor)
            self._sensenova_teardown_kv_streaming(transformer, kv_streamer)
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
        evictor = None
        kv_streamer = None
        try:
            applied_lora = self._load_lora_sensenova(params.get("loras") or [])
            evictor = self._sensenova_maybe_install_mot_eviction(params, transformer, device)
            kv_streamer = self._sensenova_maybe_install_kv_streaming(params, transformer, device)

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
                prefill_callback=_prefill_note, negative_prompt=cfg["negative_prompt"],
                ref_images=cfg["ref_images"], img_cfg_scale=cfg["img_cfg_scale"],
            )

            # Training-free reference-style transfer -- same wiring as txt2img.
            # Mechanically independent of RePaint's mask blend (see
            # denoise_loop_inpaint's docstring), so style + inpaint compose.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._sensenova_style_configs(params, transformer, device, prefix, cfg["seed"])

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
                cfg_norm=cfg["cfg_norm"],
                num_inference_steps=cfg["num_inference_steps"], mask_blur=mask_blur,
                step_callback=_step_bridge,
                style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                style_refs=style_refs, style_combine_mode=style_combine_mode,
            )
        finally:
            if prefix is not None:
                try:
                    ops.clear_prefix_caches(prefix)
                except Exception:
                    pass
            try:
                from core.models.sensenova.vendor.modeling_qwen3 import Qwen3Attention
                Qwen3Attention._style_ctx = None
            except Exception:
                pass
            if applied_lora:
                self._unload_lora_sensenova()
            self._sensenova_move("transformer", "cpu")
            self._sensenova_teardown_mot_eviction(transformer, evictor)
            self._sensenova_teardown_kv_streaming(transformer, kv_streamer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        image = ops.tensor_to_image(x)
        print("[SenseNova] inpaint completed")
        return image, cfg["seed"], 0
