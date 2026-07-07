from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import sys
import gc
import random
from pathlib import Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel
)
from config.settings import settings
from extensions.base_extension import BaseExtension
from core.model_loader import ModelLoader, ModelSource
from core.prompts.processors import PromptEditingProcessor
from core.inference.schedulers import get_scheduler
from core.inference.custom_sampling import custom_sampling_loop, custom_img2img_sampling_loop, custom_inpaint_sampling_loop

class MiniT2IMixin:
    """MiniT2IMixin: minit2i backend methods extracted verbatim from pipeline.py."""

    def _minit2i_move(self, component_name: str, target_device: str):
        comp = self.minit2i_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[MiniT2I] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _minit2i_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        from core.models.minit2i.minit2i_pipeline_ops import normalize_resolution
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        req_w = int(params.get("width", default_w))
        req_h = int(params.get("height", default_h))
        width, height = normalize_resolution(req_w, req_h)
        if (width, height) != (req_w, req_h):
            print(f"[MiniT2I] Resolution aligned: {req_w}x{req_h} -> {width}x{height}")
        cfg = self.minit2i_components["transformer"].mmjit_config
        scheduler = self.minit2i_components["scheduler"]
        vae_type = self.minit2i_components.get("vae_type", "none")
        vsf = int(self.minit2i_components.get("vae_scale_factor", 8))
        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", "") or "",
            "num_inference_steps": int(params.get("steps", scheduler.config.num_inference_steps)),
            "cfg_scale": float(params.get("cfg_scale", 6.0)),
            "cfg_interval": tuple(cfg.cfg_interval),
            "prompt_length": int(cfg.prompt_length),
            "width": width,
            "height": height,
            # Latent-space fields (vae_type != "none"): work in [1, C, H/vsf, W/vsf].
            "vae_type": vae_type,
            "is_latent": vae_type != "none",
            "channels": int(cfg.in_channels),
            "noise_scale": float(getattr(cfg, "noise_scale", 2.0)),
            "vae_scale_factor": vsf,
            "latent_h": height // vsf,
            "latent_w": width // vsf,
        }

    def _minit2i_decode(self, x, cfg):
        """Pixel: tensor_to_image. Latent: move VAE to GPU, decode, VAE back to CPU."""
        from core.models.minit2i.minit2i_pipeline_ops import tensor_to_image, vae_decode_latent
        _cf = getattr(self, "_color_flatten_strength", 0)
        if not cfg["is_latent"]:
            return tensor_to_image(x, color_flatten_strength=_cf)
        vae = self._minit2i_move("vae", self.device)
        try:
            return vae_decode_latent(vae, x, color_flatten_strength=_cf)
        finally:
            self._minit2i_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _minit2i_encode(self, prompt, negative_prompt, prompt_length, device, dtype,
                        nag_negative_prompt=None):
        """Encode prompt (+ optional negative, + optional NAG-negative) with FLAN-T5,
        then free TE to CPU. NAG-negative uses the same encoder path as the negative
        prompt; returns (nag_text, nag_mask) or (None, None) when not requested."""
        from core.models.minit2i.minit2i_pipeline_ops import encode_prompt
        from core.inference.negpip_minit2i import negpip_eligible, clean_prompt
        # NegPip auto-activation: when a negative emphasis weight is present, strip the
        # emphasis syntax so T5 encodes CLEAN text; the signed weights are carried on the
        # attention value (V) instead. Positive-only prompts are left untouched (the
        # default path stays byte-identical).
        if negpip_eligible(prompt, negative_prompt):
            enc_prompt = clean_prompt(prompt)
            enc_negative = clean_prompt(negative_prompt) if (negative_prompt and negative_prompt.strip()) else negative_prompt
            enc_nag = clean_prompt(nag_negative_prompt) if (nag_negative_prompt and str(nag_negative_prompt).strip()) else nag_negative_prompt
        else:
            enc_prompt, enc_negative, enc_nag = prompt, negative_prompt, nag_negative_prompt
        self._minit2i_move("text_encoder", device)
        te = self.minit2i_components["text_encoder"]
        tok = self.minit2i_components["tokenizer"]
        text, mask = encode_prompt(te, tok, enc_prompt, prompt_length, device)
        neg_text = neg_mask = None
        if enc_negative and enc_negative.strip():
            neg_text, neg_mask = encode_prompt(te, tok, enc_negative, prompt_length, device)
            neg_text = neg_text.to(dtype)
        nag_text = nag_mask = None
        if enc_nag is not None and str(enc_nag).strip():
            nag_text, nag_mask = encode_prompt(te, tok, enc_nag, prompt_length, device)
            nag_text = nag_text.to(dtype)
        self._minit2i_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text.to(dtype), mask, neg_text, neg_mask, nag_text, nag_mask

    def _minit2i_nag_wrap(self, params, transformer, nag_text, nag_mask):
        """Install the MiniT2I NAG wrapper when NAG is active; returns (call_target,
        wrapper_or_None). The call target is what the euler loop uses as ``transformer``:
        the NAG wrapper if active, else the transformer itself (byte-identical path)."""
        from core.inference.nag_dit import nag_active
        nag_enable = bool(params.get("nag_enable", False))
        nag_scale = float(params.get("nag_scale", 1.0))
        if not nag_active(nag_enable, nag_scale, nag_text):
            return transformer, None
        from core.inference.nag_minit2i import MiniT2INAGWrapper
        wrapper = MiniT2INAGWrapper(
            transformer, nag_text, nag_mask,
            nag_scale=nag_scale,
            nag_tau=float(params.get("nag_tau", 2.5)),
            nag_alpha=float(params.get("nag_alpha", 0.25)),
        )
        print(f"[MiniT2I] NAG active: scale={nag_scale} tau={params.get('nag_tau', 2.5)} "
              f"alpha={params.get('nag_alpha', 0.25)}")
        return wrapper, wrapper

    def _minit2i_negpip_wrap(self, params, call_target, nag_wrapper, transformer,
                             prompt, negative_prompt, seq_len, device, dtype):
        """Install the MiniT2I NegPip wrapper when the prompt has a negative emphasis
        weight (auto-activation). Returns (new_call_target, negpip_wrapper_or_None).

        When eligible, NegPip wraps the current ``call_target`` (the NAG wrapper if NAG
        is active, else the transformer). Signed per-token weight vectors are built for
        the positive prompt, the negative prompt, and (if present) the NAG-negative prompt,
        aligned to the FLAN-T5 token sequence. When NOT eligible the call target is
        returned unchanged -> the positive-only default path is byte-identical to before.
        """
        from core.inference.negpip_minit2i import (
            negpip_eligible, build_signed_weight_vector_t5, MiniT2INegPipWrapper,
        )
        if not negpip_eligible(prompt, negative_prompt):
            return call_target, None
        tokenizer = self.minit2i_components["tokenizer"]

        def _wv(p):
            return build_signed_weight_vector_t5(p, tokenizer, seq_len, device, dtype)

        pos_w = _wv(prompt)
        neg_w = _wv(negative_prompt) if (negative_prompt and negative_prompt.strip()) else None
        nag_neg = params.get("nag_negative_prompt")
        nag_w = _wv(nag_neg) if (nag_wrapper is not None and nag_neg and str(nag_neg).strip()) else None
        wrapper = MiniT2INegPipWrapper(
            transformer, pos_w, neg_weights=neg_w, nag_neg_weights=nag_w,
            nag_wrapper=nag_wrapper,
        )
        print(f"[MiniT2I] NegPip active (negative emphasis weight detected): "
              f"pos+neg{'+nag' if nag_w is not None else ''} signed V scaling, seq_len={seq_len}")
        return wrapper, wrapper

    def _minit2i_setup_block_swap(self, transformer, blocks_to_swap: int,
                                  use_pinned_memory: bool, device: str,
                                  h2d_only: bool = False, ring_size: int = 2):
        """Attach a block-swap offloader that streams only the heavy MMJiT
        ``double_blocks`` CPU<->GPU per forward.

        MiniT2I's swappable list lives at ``transformer.model.net.double_blocks``.
        The offloader's built-in aux-move only knows Z-Image module names, so ALL
        other modules (``txt_preamble_blocks``, embedders, final layer, top-level
        params/buffers) are moved to GPU here first; the offloader then pushes the
        swappable ``double_blocks`` back to CPU in prepare_block_devices_before_forward.
        The offloader is stored on the MMJiT net (whose forward drives swapping) and
        mirrored on the top-level transformer for teardown.
        """
        from core.memory_management import create_block_offloader_for_model

        net = transformer.model.net  # MMJiT instance carrying double_blocks

        # Auxiliary modules (everything except the swappable double_blocks) stay on GPU.
        for name, child in net.named_children():
            if name != "double_blocks":
                child.to(device)
        for p in net.parameters(recurse=False):
            if p.device.type != "cuda":
                p.data = p.data.to(device)
        for b in net.buffers(recurse=False):
            if b.device.type != "cuda":
                b.data = b.data.to(device)
        # Outer wrapper (MiniT2IMMJiTModel / DiffusionModel) non-recursive params/buffers.
        for holder in (transformer, transformer.model):
            for p in holder.parameters(recurse=False):
                if p.device.type != "cuda":
                    p.data = p.data.to(device)
            for b in holder.buffers(recurse=False):
                if b.device.type != "cuda":
                    b.data = b.data.to(device)

        offloader = create_block_offloader_for_model(
            transformer=net,
            blocks_to_swap=blocks_to_swap,
            device=torch.device(device),
            target_dtype=torch.bfloat16,
            use_pinned_memory=use_pinned_memory,
            h2d_only=h2d_only,
            ring_size=ring_size,
            block_list=net.double_blocks,
        )
        net._block_offloader = offloader
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _minit2i_apply_attention_backend(self, transformer, params: Dict[str, Any]):
        """Select the attention backend (native / flash / sage) for inference.

        Reads the inference key ``attention_type`` (``normal``|``sage``|``flash``;
        ``normal``/None -> ``native``), normalizes it to the canonical vocabulary,
        and stamps it on BOTH the transformer wrapper (``transformer._attn_backend``)
        and the underlying MMJiT net (``transformer.model.net._attn_backend``). The
        MMJiT forward reads the net-level attr each step and propagates it to every
        attention-bearing block, so the vendored and NAG/NegPip block forwards route
        their ``mem_efficient_sdpa`` call through the unified conduit. The default
        ``native`` keeps the prior SDPA path byte-identical.

        Guards handled downstream by the conduit: l16 head_dim 52 (padded to 56) is
        excluded from sage's allowed set -> auto-downgrade to native; flash accepts
        the padded 56. b16 head_dim 64 is accepted by both flash and sage.
        """
        from core.attention import normalize_backend
        attn_backend = normalize_backend(params.get("attention_type"))
        transformer._attn_backend = attn_backend
        net = getattr(getattr(transformer, "model", None), "net", None)
        if net is not None:
            net._attn_backend = attn_backend
        print(f"[MiniT2I] Attention backend: {attn_backend} "
              f"(from attention_type={params.get('attention_type')!r})")

    def _minit2i_stage_transformer(self, device: str, params: Optional[Dict[str, Any]] = None):
        """Place the MiniT2I transformer on GPU for the denoise loop.

        With block swap enabled, only the heavy ``double_blocks`` are streamed
        (the rest stays resident); otherwise the whole transformer moves to GPU
        (default path, unchanged)."""
        params = params or {}
        transformer = self.minit2i_components["transformer"]
        self._minit2i_apply_attention_backend(transformer, params)
        enable_block_swap = bool(params.get("enable_block_swap", False))
        num_blocks = len(transformer.model.net.double_blocks)
        blocks_to_swap = int(params.get("blocks_to_swap", 0))
        blocks_to_swap = max(0, min(blocks_to_swap, num_blocks - 1))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))
        h2d_only = bool(params.get("block_swap_h2d_only", False))
        ring_size = int(params.get("block_swap_ring_size", 2))

        self._minit2i_offloader = None
        if enable_block_swap and blocks_to_swap > 0:
            print(f"[MiniT2I] Block swap enabled: {blocks_to_swap}/{num_blocks} double_blocks "
                  f"(pinned_memory={use_pinned_memory}, h2d_only={h2d_only}, ring_size={ring_size})")
            self._minit2i_offloader = self._minit2i_setup_block_swap(
                transformer, blocks_to_swap, use_pinned_memory, device,
                h2d_only=h2d_only, ring_size=ring_size)
        else:
            self._minit2i_move("transformer", device)
        return transformer

    def _minit2i_unstage_transformer(self):
        """Tear down any block-swap offloader, then return the transformer to CPU."""
        off = getattr(self, "_minit2i_offloader", None)
        transformer = (self.minit2i_components or {}).get("transformer")
        if transformer is not None:
            net = getattr(getattr(transformer, "model", None), "net", None)
            for holder in (transformer, net):
                if holder is not None and hasattr(holder, "_block_offloader"):
                    try:
                        delattr(holder, "_block_offloader")
                    except Exception:
                        pass
        if off is not None:
            cleanup = getattr(off, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._minit2i_offloader = None
        self._minit2i_move("transformer", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_lora_minit2i(self, lora_configs: List[Dict]) -> int:
        from core.models.minit2i.minit2i_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
            apply_te_lora_group, TE_NAMESPACE,
        )
        from core.extensions.lora_manager import lora_manager
        if not lora_configs or not self.minit2i_components:
            return 0
        transformer = self.minit2i_components["transformer"]
        text_encoder = self.minit2i_components.get("text_encoder")
        if not hasattr(self, "_minit2i_lora_orig"):
            self._minit2i_lora_orig: Dict[str, torch.nn.Module] = {}
            self._minit2i_lora_keys: set = set()
        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[MiniT2I LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                # Transformer LoRA (lora_unet_) and TE LoRA (lora_te_) auto-route by key.
                applied = apply_lora_group(transformer, grouped, strength,
                                           self._minit2i_lora_orig, self._minit2i_lora_keys)
                applied_te = 0
                has_te_keys = any(k.startswith(TE_NAMESPACE) for k in grouped)
                if has_te_keys and text_encoder is not None:
                    applied_te = apply_te_lora_group(text_encoder, grouped, strength,
                                                     self._minit2i_lora_orig, self._minit2i_lora_keys)
                elif has_te_keys:
                    print(f"[MiniT2I LoRA] WARNING: {lora_path} has TE-LoRA keys but no text encoder is loaded; "
                          f"TE-LoRA skipped")
                print(f"[MiniT2I LoRA] {i+1}/{len(lora_configs)}: {lora_path} fmt={fmt} "
                      f"matched={len(grouped)} wrapped(transformer)={applied} wrapped(te)={applied_te} "
                      f"strength={strength}")
                total += applied + applied_te
            except Exception as e:
                print(f"[MiniT2I LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_minit2i(self) -> int:
        from core.models.minit2i.minit2i_lora import restore_originals
        if not self.minit2i_components or not getattr(self, "_minit2i_lora_keys", None):
            return 0
        restored = restore_originals(
            self.minit2i_components["transformer"], self._minit2i_lora_orig, self._minit2i_lora_keys,
            text_encoder=self.minit2i_components.get("text_encoder"),
        )
        if restored:
            print(f"[MiniT2I LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _minit2i_cleanup(self):
        try:
            self._unload_lora_minit2i()
        except Exception:
            pass
        # Strip any leftover block-swap offloader (e.g. if staging raised mid-way).
        transformer = (self.minit2i_components or {}).get("transformer")
        if transformer is not None:
            net = getattr(getattr(transformer, "model", None), "net", None)
            for holder in (transformer, net):
                if holder is not None and hasattr(holder, "_block_offloader"):
                    try:
                        delattr(holder, "_block_offloader")
                    except Exception:
                        pass
        self._minit2i_offloader = None
        for _c in ("text_encoder", "transformer", "vae"):
            try:
                self._minit2i_move(_c, "cpu")
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_minit2i(self, params, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import denoise_loop, tensor_to_image
        print("[MiniT2I] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, 512, 512)
        try:
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"))
            transformer = self._minit2i_stage_transformer(device, params)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            call_target, nag_wrapper = self._minit2i_nag_wrap(params, transformer, nag_text, nag_mask)
            call_target, negpip_wrapper = self._minit2i_negpip_wrap(
                params, call_target, nag_wrapper, transformer,
                cfg["prompt"], cfg["negative_prompt"], text.shape[1], device, dtype)
            try:
                if cfg["is_latent"]:
                    # work in latent space: [1, C, H/vsf, W/vsf]
                    x = denoise_loop(
                        call_target, text, mask, cfg["latent_h"], cfg["latent_w"],
                        cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                        device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                        progress_callback=progress_callback,
                        channels=cfg["channels"], noise_scale=cfg["noise_scale"], clamp_output=False,
                        spectrum_params=params,
                    )
                else:
                    x = denoise_loop(
                        call_target, text, mask, cfg["height"], cfg["width"],
                        cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                        device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                        progress_callback=progress_callback,
                        spectrum_params=params,
                    )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer()
            image = self._minit2i_decode(x, cfg)
            print("[MiniT2I] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()

    def _generate_img2img_minit2i(self, params, init_image, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import (
            denoise_loop_img2img, image_to_tensor, vae_encode_image,
        )
        print("[MiniT2I] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        try:
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"))
            if cfg["is_latent"]:
                vae = self._minit2i_move("vae", device)
                init_t = vae_encode_image(vae, init_image, cfg["height"], cfg["width"], device, dtype)
                self._minit2i_move("vae", "cpu")
            else:
                init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_stage_transformer(device, params)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            call_target, nag_wrapper = self._minit2i_nag_wrap(params, transformer, nag_text, nag_mask)
            call_target, negpip_wrapper = self._minit2i_negpip_wrap(
                params, call_target, nag_wrapper, transformer,
                cfg["prompt"], cfg["negative_prompt"], text.shape[1], device, dtype)
            try:
                x = denoise_loop_img2img(
                    call_target, init_t, denoising_strength, text, mask,
                    cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                    device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                    progress_callback=progress_callback,
                    noise_scale=cfg["noise_scale"], clamp_output=not cfg["is_latent"],
                    spectrum_params=params,
                )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer()
            image = self._minit2i_decode(x, cfg)
            print("[MiniT2I] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()

    def _generate_inpaint_minit2i(self, params, init_image, mask_image, progress_callback=None, step_callback=None) -> tuple:
        if not self.minit2i_components:
            raise RuntimeError("MiniT2I components not loaded.")
        from core.models.minit2i.minit2i_pipeline_ops import (
            denoise_loop_inpaint, image_to_tensor, vae_encode_image, prepare_mask,
        )
        print("[MiniT2I] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._minit2i_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        try:
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"))
            if cfg["is_latent"]:
                vae = self._minit2i_move("vae", device)
                init_t = vae_encode_image(vae, init_image, cfg["height"], cfg["width"], device, dtype)
                self._minit2i_move("vae", "cpu")
                # mask at latent resolution (1=regenerate, 0=keep)
                mask_latent = prepare_mask(mask_image, cfg["latent_h"], cfg["latent_w"], device, dtype)
            else:
                init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
                mask_latent = prepare_mask(mask_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_stage_transformer(device, params)
            applied_lora = self._load_lora_minit2i(params.get("loras") or [])
            call_target, nag_wrapper = self._minit2i_nag_wrap(params, transformer, nag_text, nag_mask)
            call_target, negpip_wrapper = self._minit2i_negpip_wrap(
                params, call_target, nag_wrapper, transformer,
                cfg["prompt"], cfg["negative_prompt"], text.shape[1], device, dtype)
            try:
                x = denoise_loop_inpaint(
                    call_target, init_t, mask_latent, denoising_strength, text, mask,
                    cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                    device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                    progress_callback=progress_callback,
                    noise_scale=cfg["noise_scale"], clamp_output=not cfg["is_latent"],
                    spectrum_params=params,
                )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer()
            image = self._minit2i_decode(x, cfg)
            print("[MiniT2I] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup()
