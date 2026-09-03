from typing import Dict, Any, Optional, List, Callable
from PIL import Image
import torch
import json
import os
import sys
import gc
import random
import weakref
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

    def _minit2i_kh_setup(self, params: Dict[str, Any]):
        """Compute keep-models-hot eligibility for this generation (see core/keep_hot.py).

        MiniT2I applies LoRA in place to BOTH the transformer and (when TE-LoRA
        keys are present) the text encoder -- see _load_lora_minit2i /
        _apply_te_lora_minit2i -- so both are gated off whenever any LoRA is
        requested (the reference's "no LoRA" hazard, extended here to the TE for
        the same in-place-mutation reason: a resident TE would carry a stale
        LoRA-wrapped state into the next generation). The transformer is
        additionally gated off under block-swap streaming. The VAE is absent for
        pixel-space checkpoints (``vae_type == "none"``) -- component_nbytes(None)
        is 0 and ``_minit2i_move`` no-ops on a missing component, so it is simply
        never eligible in that case.

        Returns (model_key, keep_te, keep_transformer, keep_vae, is_block_swapped).
        """
        from core.keep_hot import (
            invalidate_if_model_changed, should_keep_resident, compute_model_key,
            component_nbytes, keep_hot_requested,
        )
        requested = keep_hot_requested(params)
        model_key = compute_model_key(self, params)
        has_loras = bool(params.get("loras") or [])
        has_vae = self.minit2i_components.get("vae") is not None

        enable_block_swap = bool(params.get("enable_block_swap", False))
        transformer = self.minit2i_components.get("transformer")
        num_blocks = len(transformer.model.net.double_blocks) if transformer is not None else 0
        blocks_to_swap = int(params.get("blocks_to_swap", 0))
        blocks_to_swap = max(0, min(blocks_to_swap, max(num_blocks - 1, 0)))
        is_block_swapped = enable_block_swap and blocks_to_swap > 0

        # If a resident set exists from a previous generation but is no longer valid
        # for THIS request's model_key, force a full offload before staging anything.
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._minit2i_move("text_encoder", "cpu"),
                self._minit2i_move("transformer", "cpu"),
                self._minit2i_move("vae", "cpu"),
            ),
        )

        total_bytes = 0
        if requested:
            if not has_loras:
                total_bytes += component_nbytes(self.minit2i_components.get("text_encoder"))
                if not is_block_swapped:
                    total_bytes += component_nbytes(self.minit2i_components.get("transformer"))
            if has_vae:
                total_bytes += component_nbytes(self.minit2i_components.get("vae"))
        guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=total_bytes,
        ) if requested else False

        keep_te = requested and guard_ok and not has_loras
        keep_transformer = requested and guard_ok and not is_block_swapped and not has_loras
        keep_vae = requested and guard_ok and has_vae
        return model_key, keep_te, keep_transformer, keep_vae, is_block_swapped

    def _minit2i_decode(self, x, cfg, model_key: Optional[str] = None, keep_vae: bool = False):
        """Pixel: tensor_to_image. Latent: move VAE to GPU (unless already kept
        resident), decode, then either leave it resident (keep-models-hot) or
        move it back to CPU (default)."""
        from core.models.minit2i.minit2i_pipeline_ops import tensor_to_image, vae_decode_latent
        from core.keep_hot import is_resident, mark_resident, discard_resident
        _cf = getattr(self, "_color_flatten_strength", 0)
        if not cfg["is_latent"]:
            # Pixel-space checkpoint: no VAE exists, so a requested tiling toggle
            # is a real runtime no-op, not the arch-level "unsupported" case.
            if getattr(self, "_vae_tiling", False):
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "vae_tiling requested but the loaded MiniT2I checkpoint is pixel-space "
                        "(no VAE); the toggle has nothing to apply to",
                        code="minit2i_vae_tiling_no_vae",
                    )
                except Exception:
                    pass
            return tensor_to_image(x, color_flatten_strength=_cf)
        if model_key is None or not is_resident(self, "vae", model_key):
            vae = self._minit2i_move("vae", self.device)
        else:
            vae = self.minit2i_components["vae"]
        self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
        try:
            return vae_decode_latent(vae, x, color_flatten_strength=_cf)
        finally:
            if keep_vae and model_key is not None:
                mark_resident(self, "vae", model_key)
            else:
                self._minit2i_move("vae", "cpu")
                if model_key is not None:
                    discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def _minit2i_encode(self, prompt, negative_prompt, prompt_length, device, dtype,
                        nag_negative_prompt=None, model_key: Optional[str] = None,
                        keep_te: bool = False):
        """Encode prompt (+ optional negative, + optional NAG-negative) with FLAN-T5,
        then free TE to CPU (default) or leave it resident (keep-models-hot).
        NAG-negative uses the same encoder path as the negative prompt; returns
        (nag_text, nag_mask) or (None, None) when not requested."""
        from core.models.minit2i.minit2i_pipeline_ops import encode_prompt
        from core.inference.negpip_minit2i import negpip_eligible, clean_prompt
        from core.keep_hot import is_resident, mark_resident, discard_resident
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
        if model_key is None or not is_resident(self, "text_encoder", model_key):
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
        if keep_te and model_key is not None:
            mark_resident(self, "text_encoder", model_key)
        else:
            self._minit2i_move("text_encoder", "cpu")
            if model_key is not None:
                discard_resident(self, "text_encoder")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return text.to(dtype), mask, neg_text, neg_mask, nag_text, nag_mask

    def _minit2i_style_triple(self, style_dict: Dict[str, Any], cfg: Dict[str, Any], device, dtype,
                              model_key: Optional[str] = None, ref_index: int = 0):
        """Build a single ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from one
        style_transfer dict.

        No ``axes_dims`` is set here (unlike Krea2/Z-Image): MiniT2I uses interleaved
        (rotate_half) RoPE, not the ``repeat_interleave_real=True`` layout
        ``frequency_scale_vector`` assumes, so ``core.inference.style_minit2i`` bypasses
        the frequency-scale machinery entirely (all-ones vector) rather than calling
        ``get_freq_scale_vector`` -- see that module's docstring. This is unaffected by
        multi-reference support: ``StyleContext.collect_block_refs`` falls back to the
        SAME all-ones vector whenever a config's ``axes_dims`` is unset (it never is,
        for any ref, here).

        The reference is encoded the SAME way this generation consumes images: pixel-
        space checkpoints (``cfg["is_latent"] is False``, the MiniT2I default) get the
        raw normalized pixel tensor; the optional latent-VAE variant gets a VAE-encoded
        normalized latent, exactly like the target's own init_image path in img2img/
        inpaint. Resized to this generation's own PIXEL (height, width) -- vae_encode_image
        downsamples by 8 internally, so pixel dims are passed for BOTH the pixel and the
        latent-VAE path (never the already-divided latent_h/latent_w).

        ``ref_index`` decorrelates the fixed re-noising noise tensor across multiple
        simultaneous references (each ref would otherwise draw the EXACT same noise from
        ``prepare_style_reference``'s internal ``seed+991`` offset applied to the SAME
        ``cfg["seed"]``, since that offset does not depend on which reference is being
        prepared). ``ref_index=0`` (the default, used by the single-ref path) reproduces
        the pre-multi-ref ``cfg["seed"]`` exactly.
        """
        from core.inference.reference_style import style_config_from_dict
        from core.models.minit2i.minit2i_pipeline_ops import prepare_style_reference
        from core.keep_hot import is_resident, discard_resident

        style_cfg = style_config_from_dict(style_dict)
        ref_seed = cfg["seed"] if ref_index == 0 else cfg["seed"] + ref_index

        is_latent = cfg["is_latent"]
        if is_latent:
            # PIXEL dims: vae_encode_image resizes the PIL image to (width,height) pixels
            # then downsamples by 8, yielding the [latent_h, latent_w] latent that matches
            # the target's working latent (same as the target's own init-image encode).
            height, width = cfg["height"], cfg["width"]
            # VAE encode needs the VAE on-device; mirrors the img2img/inpaint init-image
            # encode path (brief on-GPU use, offloaded again right after -- the VAE is
            # not part of this generation's keep-models-hot residency decision here).
            if model_key is not None and is_resident(self, "vae", model_key):
                vae = self.minit2i_components["vae"]
            else:
                vae = self._minit2i_move("vae", device)
            ref_x0, eps_ref = prepare_style_reference(
                vae, style_dict["image"], height, width, device, dtype,
                is_latent=True, channels=cfg["channels"], noise_scale=cfg["noise_scale"], seed=ref_seed,
            )
            self._minit2i_move("vae", "cpu")
            if model_key is not None:
                discard_resident(self, "vae")
        else:
            height, width = cfg["height"], cfg["width"]
            ref_x0, eps_ref = prepare_style_reference(
                None, style_dict["image"], height, width, device, dtype,
                is_latent=False, channels=cfg["channels"], noise_scale=cfg["noise_scale"], seed=ref_seed,
            )
        return style_cfg, ref_x0, eps_ref

    def _minit2i_style_config(self, params: Dict[str, Any], cfg: Dict[str, Any], device, dtype,
                              model_key: Optional[str] = None):
        """Build a ``(StyleTransferConfig, ref_x0, eps_ref)`` triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None, None)``
        when no style reference is attached (byte-identical default path -- the
        caller never installs the style block patch in that case). Single-reference
        path, BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_minit2i_style_triple`` with ``ref_index=0``, which reproduces the original
        ``cfg["seed"]`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        return self._minit2i_style_triple(style_dict, cfg, device, dtype, model_key=model_key, ref_index=0)

    def _minit2i_style_configs(self, params: Dict[str, Any], cfg: Dict[str, Any], device, dtype,
                               model_key: Optional[str] = None):
        """Build the full style-transfer configuration for MiniT2I generation,
        covering both the single-reference path (legacy ``(style_cfg, style_ref_x0,
        style_eps_ref)`` triple, exactly as ``_minit2i_style_config`` would return)
        and the multi-reference path (``style_refs``, a list of per-ref triples,
        populated ONLY when ``params["style_transfers"]`` has more than one entry).
        A single-entry ``style_transfers`` list is intentionally routed through the
        single-ref triple instead (``style_refs`` stays ``None``), so the
        pre-multi-ref code path executes byte-identically end to end.

        Returns ``(style_cfg, style_ref_x0, style_eps_ref, style_refs,
        style_combine_mode)``.
        """
        style_list = params.get("style_transfers")
        if style_list and len(style_list) > 1:
            combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
            refs = []
            for idx, style_dict in enumerate(style_list):
                if not style_dict or not style_dict.get("image"):
                    continue
                refs.append(self._minit2i_style_triple(style_dict, cfg, device, dtype, model_key=model_key, ref_index=idx))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                style_cfg_single, x0, eps = refs[0]
                return style_cfg_single, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._minit2i_style_config(params, cfg, device, dtype, model_key=model_key)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

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

    def _minit2i_stage_transformer(self, device: str, params: Optional[Dict[str, Any]] = None,
                                   model_key: Optional[str] = None):
        """Place the MiniT2I transformer on GPU for the denoise loop.

        With block swap enabled, only the heavy ``double_blocks`` are streamed
        (the rest stays resident); otherwise the whole transformer moves to GPU
        (default path, unchanged) -- unless it is already kept GPU-resident from
        a previous successful generation (keep-models-hot), in which case the
        ->GPU move is skipped entirely."""
        from core.keep_hot import is_resident

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
            if model_key is None or not is_resident(self, "transformer", model_key):
                self._minit2i_move("transformer", device)
        return transformer

    def _minit2i_unstage_transformer(self, keep_transformer: bool = False,
                                     model_key: Optional[str] = None):
        """Tear down any block-swap offloader, then either leave the transformer
        GPU-resident (keep-models-hot: mark_resident) or return it to CPU
        (default, or when keep_transformer is False)."""
        from core.keep_hot import mark_resident, discard_resident

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
        if keep_transformer and model_key is not None:
            mark_resident(self, "transformer", model_key)
        else:
            self._minit2i_move("transformer", "cpu")
            if model_key is not None:
                discard_resident(self, "transformer")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _minit2i_lora_warn(message: str, code: str) -> None:
        """Record a user-visible generation warning (best effort)."""
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    def _minit2i_lora_state(self, transformer, text_encoder):
        """The shared (originals, wrapped_keys) maps, minus any reloaded component's
        entries.

        Transformer keys are bare module paths and text-encoder keys carry
        TE_NAMESPACE, so each component's half is dropped independently -- a module
        path can never contain "::", so the partition is total and ONE map pair
        expresses two component lifecycles. Keyed by weakref rather than id(), which
        is reusable after a free: a map that outlived its model would have
        restore_originals graft the old model's Linears into the new one,
        undetectably, since the shapes match.
        """
        from core.models.minit2i.minit2i_lora import TE_NAMESPACE
        if not hasattr(self, "_minit2i_lora_orig"):
            self._minit2i_lora_orig: Dict[str, torch.nn.Module] = {}
            self._minit2i_lora_keys: set = set()

        def reloaded(attr, module):
            ref = getattr(self, attr, None)
            if module is not None and ref is not None and ref() is module:
                return False
            setattr(self, attr, weakref.ref(module) if module is not None else None)
            return True

        drop_tf = reloaded("_minit2i_lora_transformer_ref", transformer)
        drop_te = reloaded("_minit2i_lora_te_ref", text_encoder)
        if drop_tf or drop_te:
            stale = {k for k in (self._minit2i_lora_keys | set(self._minit2i_lora_orig))
                     if (drop_te if k.startswith(TE_NAMESPACE) else drop_tf)}
            self._minit2i_lora_keys -= stale
            for key in stale:
                self._minit2i_lora_orig.pop(key, None)
        return self._minit2i_lora_orig, self._minit2i_lora_keys

    def _minit2i_prepare_loras(self, lora_configs: List[Dict]) -> List[Dict]:
        """Resolve and parse every requested LoRA, before any of it is applied.

        MiniT2I wraps in two places at two different times -- the text encoder BEFORE
        _minit2i_encode, the transformer after it is staged -- so each file is read
        once here and the parsed groups are carried across both passes, including the
        branch name, which must be the same in both. A missing or unreadable file
        refuses the generation here, with nothing yet wrapped.
        """
        import os
        from core.models.minit2i.minit2i_lora import (
            load_lora_safetensors, normalise_lora_state_dict, alpha_from_metadata,
            TE_NAMESPACE,
        )
        from core.extensions.lora_manager import lora_manager
        from api.error_handlers import with_error_code

        # Unconditional, and BEFORE the empty-config exit: a restore that failed in
        # an earlier request must not leak wrappers into this one. `_minit2i_cleanup`
        # swallows restore failures, and now that a second branch SUMS rather than
        # being refused, a leaked wrapper would silently double-apply.
        self._unload_lora_minit2i()

        components = self.minit2i_components or {}
        # Bookkeeping that outlived a model reload would splice the previous model's
        # Linears into this one; each component's half is dropped separately.
        self._minit2i_lora_state(components.get("transformer"), components.get("text_encoder"))
        if not lora_configs or not components:
            return []

        prepared: List[Dict] = []
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            lora_file = os.path.basename(str(lora_path))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                message = f"MiniT2I LoRA '{lora_file}' not found in any configured LoRA directory"
                print(f"[MiniT2I LoRA] ERROR: {message}")
                self._minit2i_lora_warn(message, code="lora_not_found")
                raise with_error_code(FileNotFoundError(message), "lora_not_found")
            try:
                raw, fmt, metadata = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
            except Exception as e:
                print(f"[MiniT2I LoRA] ERROR loading {lora_file}: {e}")
                import traceback; traceback.print_exc()
                # Type + basename only: this rides into the PNG text chunk and the API
                # response, and an OSError's str() carries the absolute resolved path.
                message = (f"MiniT2I LoRA '{lora_file}' could not be applied "
                           f"({type(e).__name__}); see the server log for details")
                self._minit2i_lora_warn(message, code="lora_load_failed")
                raise with_error_code(RuntimeError(message), "lora_load_failed") from e
            prepared.append({
                "file": lora_file,
                # Unique within the request, so selecting the SAME file twice is two
                # branches, not a duplicate-name refusal; shared by both passes so a
                # file's transformer and TE branches carry one name.
                "branch": f"{i}:{lora_file}",
                "strength": float(cfg.get("strength", 1.0)),
                "fmt": fmt,
                "grouped": grouped,
                "alpha": alpha_from_metadata(metadata),
                "sample_keys": list(raw.keys())[:5],
                "te_pairs": sum(1 for k in grouped if k.startswith(TE_NAMESPACE)),
                "applied_te": 0,
            })
        return prepared

    def _apply_te_lora_minit2i(self, prepared: List[Dict]) -> int:
        """Cover the text encoder for every prepared LoRA carrying lora_te_ keys.

        MUST run BEFORE _minit2i_encode: the embeddings are produced there and never
        recomputed, so a TE wrapper installed after it would report a count and change
        nothing. The per-file verdict is deferred to _load_lora_minit2i, which sees
        both halves.

        The text encoder's composites are separate objects from the transformer's,
        and their bookkeeping is separate by key namespace, so stacking on one
        component leaves the other alone.
        """
        from core.models.minit2i.minit2i_lora import apply_te_lora_group
        components = self.minit2i_components or {}
        text_encoder = components.get("text_encoder")
        originals, keys = self._minit2i_lora_state(
            components.get("transformer"), text_encoder)
        total = 0
        for entry in prepared:
            if not entry["te_pairs"]:
                continue
            if text_encoder is None:
                print(f"[MiniT2I LoRA] WARNING: {entry['file']} has TE-LoRA keys but no text "
                      f"encoder is loaded; TE-LoRA skipped")
                continue
            applied = apply_te_lora_group(
                text_encoder, entry["grouped"], entry["strength"], originals, keys,
                default_alpha=entry["alpha"], branch_name=entry["branch"])
            entry["applied_te"] = applied
            print(f"[MiniT2I LoRA] {entry['file']}: wrapped(te)={applied} of "
                  f"{entry['te_pairs']} TE pair(s)")
            total += applied
        return total

    def _load_lora_minit2i(self, prepared: List[Dict]) -> int:
        """Cover the transformer, then pass the per-file verdict on BOTH halves.

        Takes _minit2i_prepare_loras' output, whose text-encoder half
        _apply_te_lora_minit2i has already applied. Each target is covered ONCE by a
        ``CompositeAdapterLayer`` and each selected LoRA adds a NAMED branch, so two
        MiniT2I LoRAs over the same module SUM instead of being refused. A LoRA that
        matches no target module in EITHER component must not produce a successful
        generation of the base model's image, so that raises; a file that binds only
        some of its modules is reported through add_warning.
        """
        from core.models.minit2i.minit2i_lora import apply_lora_group
        from api.error_handlers import with_error_code
        components = self.minit2i_components or {}
        transformer = components.get("transformer")
        originals, keys = self._minit2i_lora_state(
            transformer, components.get("text_encoder"))
        if not prepared:
            return 0
        if transformer is None:
            raise RuntimeError("MiniT2I LoRA requested but no transformer is loaded.")
        total = 0
        for i, entry in enumerate(prepared):
            lora_file = entry["file"]
            grouped = entry["grouped"]
            fmt = entry["fmt"]
            applied = apply_lora_group(
                transformer, grouped, entry["strength"], originals, keys,
                default_alpha=entry["alpha"], branch_name=entry["branch"])
            print(f"[MiniT2I LoRA] {i+1}/{len(prepared)}: {lora_file} fmt={fmt} "
                  f"matched={len(grouped)} wrapped(transformer)={applied} "
                  f"wrapped(te)={entry['applied_te']} strength={entry['strength']}")

            applied_all = applied + entry["applied_te"]
            # An occupied target is no longer one of the ways to get here: the
            # composite adds a branch beside the earlier LoRA's.
            if applied_all == 0:
                message = (
                    f"LoRA '{lora_file}': 0 of {len(grouped)} down/up pairs applied to the "
                    f"loaded MiniT2I model (format={fmt}) -- unrecognized key format or a "
                    f"different model. Sample keys in file: {entry['sample_keys']}"
                )
                print(f"[MiniT2I LoRA] ERROR: {message}")
                self._minit2i_lora_warn(message, code="lora_incompatible")
                raise with_error_code(RuntimeError(message), "lora_incompatible")

            if applied_all < len(grouped):
                self._minit2i_lora_warn(
                    f"LoRA '{lora_file}': applied {applied_all} of {len(grouped)} down/up pairs "
                    f"({len(grouped) - applied_all} matched no target module).",
                    code="lora_partial",
                )
            total += applied_all
        return total

    def _unload_lora_minit2i(self) -> int:
        """Restore both components, each through its own half of the bookkeeping.

        A component whose model was reloaded has already had its keys dropped by
        `_minit2i_lora_state`, and the new one carries no composites, so it
        restores nothing while the other component's wrappers survive.
        """
        from core.models.minit2i.minit2i_lora import restore_originals
        components = self.minit2i_components or {}
        transformer = components.get("transformer")
        text_encoder = components.get("text_encoder")
        # Same guard as the load path: maps that outlived their component must be
        # dropped, not restored into whatever now sits at the same module paths.
        originals, keys = self._minit2i_lora_state(transformer, text_encoder)
        if not keys:
            return 0
        restored = restore_originals(
            transformer, originals, keys, text_encoder=text_encoder,
        )
        if restored:
            print(f"[MiniT2I LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _minit2i_cleanup(self, model_key: Optional[str] = None,
                        keep_te: bool = False, keep_transformer: bool = False,
                        keep_vae: bool = False, gen_succeeded: bool = False):
        """End-of-generation teardown. On failure (or when called with no
        keep-hot context, e.g. old call sites), force a full offload and clear
        any tentative residency -- including any transformer/TE mark_resident
        made optimistically by an inner stage before the failure occurred (see
        _minit2i_unstage_transformer / _minit2i_encode). On success, components
        already left resident by their own stage are trusted as-is;
        discard_resident on the rest just keeps the tracked set in sync
        (idempotent no-op otherwise)."""
        from core.keep_hot import clear_resident, discard_resident
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
        if not gen_succeeded:
            clear_resident(self)
            for _c in ("text_encoder", "transformer", "vae"):
                try:
                    self._minit2i_move(_c, "cpu")
                except Exception:
                    pass
        else:
            if not keep_te:
                discard_resident(self, "text_encoder")
            if not keep_transformer:
                discard_resident(self, "transformer")
            if not keep_vae:
                discard_resident(self, "vae")
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
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae, _kh_is_block_swapped = \
            self._minit2i_kh_setup(params)
        _kh_gen_succeeded = False
        try:
            # The text-encoder half goes on BEFORE the encode below, which produces the
            # embeddings once: a TE wrapper installed after it would change nothing.
            prepared_loras = self._minit2i_prepare_loras(params.get("loras") or [])
            self._apply_te_lora_minit2i(prepared_loras)
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"),
                model_key=_kh_model_key, keep_te=_kh_keep_te)
            transformer = self._minit2i_stage_transformer(device, params, model_key=_kh_model_key)
            applied_lora = self._load_lora_minit2i(prepared_loras)
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                self._minit2i_style_configs(params, cfg, device, dtype, model_key=_kh_model_key)
            style_active = style_cfg is not None or style_refs is not None
            # Style transfer is mutually exclusive with NAG/NegPip: both ALSO monkey-patch
            # DoubleStreamDiTBlock.forward (core.inference.style_minit2i), so installing
            # both would have one wrapper silently clobber the other's patch. Skip the
            # NAG/NegPip wrap entirely for a style-active generation.
            if style_active:
                call_target, nag_wrapper, negpip_wrapper = transformer, None, None
                print("[MiniT2I] Style transfer active: NAG/NegPip wrapping skipped for this generation")
            else:
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
                        style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                        style_refs=style_refs, style_combine_mode=style_combine_mode,
                    )
                else:
                    x = denoise_loop(
                        call_target, text, mask, cfg["height"], cfg["width"],
                        cfg["num_inference_steps"], cfg["cfg_scale"], cfg["cfg_interval"],
                        device, dtype, seed=cfg["seed"], neg_text=neg_text, neg_mask=neg_mask,
                        progress_callback=progress_callback,
                        spectrum_params=params,
                        style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                        style_refs=style_refs, style_combine_mode=style_combine_mode,
                    )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer(keep_transformer=_kh_keep_transformer, model_key=_kh_model_key)
            image = self._minit2i_decode(x, cfg, model_key=_kh_model_key, keep_vae=_kh_keep_vae)
            _kh_gen_succeeded = True
            print("[MiniT2I] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                  _kh_keep_vae, _kh_gen_succeeded)

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
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae, _kh_is_block_swapped = \
            self._minit2i_kh_setup(params)
        _kh_gen_succeeded = False
        try:
            # The text-encoder half goes on BEFORE the encode below, which produces the
            # embeddings once: a TE wrapper installed after it would change nothing.
            prepared_loras = self._minit2i_prepare_loras(params.get("loras") or [])
            self._apply_te_lora_minit2i(prepared_loras)
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"),
                model_key=_kh_model_key, keep_te=_kh_keep_te)
            if cfg["is_latent"]:
                from core.keep_hot import is_resident, discard_resident
                # First use of VAE this generation only: honor cross-generation residency
                # on entry, but always offload after (VAE is reused again at decode, so
                # this is an intermediate step, not the generation's final exit point).
                if not is_resident(self, "vae", _kh_model_key):
                    vae = self._minit2i_move("vae", device)
                else:
                    vae = self.minit2i_components["vae"]
                init_t = vae_encode_image(vae, init_image, cfg["height"], cfg["width"], device, dtype)
                self._minit2i_move("vae", "cpu")
                discard_resident(self, "vae")
            else:
                init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_stage_transformer(device, params, model_key=_kh_model_key)
            applied_lora = self._load_lora_minit2i(prepared_loras)
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                self._minit2i_style_configs(params, cfg, device, dtype, model_key=_kh_model_key)
            style_active = style_cfg is not None or style_refs is not None
            if style_active:
                call_target, nag_wrapper, negpip_wrapper = transformer, None, None
                print("[MiniT2I] Style transfer active: NAG/NegPip wrapping skipped for this generation")
            else:
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
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer(keep_transformer=_kh_keep_transformer, model_key=_kh_model_key)
            image = self._minit2i_decode(x, cfg, model_key=_kh_model_key, keep_vae=_kh_keep_vae)
            _kh_gen_succeeded = True
            print("[MiniT2I] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                  _kh_keep_vae, _kh_gen_succeeded)

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
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae, _kh_is_block_swapped = \
            self._minit2i_kh_setup(params)
        _kh_gen_succeeded = False
        try:
            # The text-encoder half goes on BEFORE the encode below, which produces the
            # embeddings once: a TE wrapper installed after it would change nothing.
            prepared_loras = self._minit2i_prepare_loras(params.get("loras") or [])
            self._apply_te_lora_minit2i(prepared_loras)
            text, mask, neg_text, neg_mask, nag_text, nag_mask = self._minit2i_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg["prompt_length"], device, dtype,
                nag_negative_prompt=params.get("nag_negative_prompt"),
                model_key=_kh_model_key, keep_te=_kh_keep_te)
            if cfg["is_latent"]:
                from core.keep_hot import is_resident, discard_resident
                # First use of VAE this generation only: honor cross-generation residency
                # on entry, but always offload after (VAE is reused again at decode, so
                # this is an intermediate step, not the generation's final exit point).
                if not is_resident(self, "vae", _kh_model_key):
                    vae = self._minit2i_move("vae", device)
                else:
                    vae = self.minit2i_components["vae"]
                init_t = vae_encode_image(vae, init_image, cfg["height"], cfg["width"], device, dtype)
                self._minit2i_move("vae", "cpu")
                discard_resident(self, "vae")
                # mask at latent resolution (1=regenerate, 0=keep)
                mask_latent = prepare_mask(mask_image, cfg["latent_h"], cfg["latent_w"], device, dtype)
            else:
                init_t = image_to_tensor(init_image, cfg["height"], cfg["width"], device, dtype)
                mask_latent = prepare_mask(mask_image, cfg["height"], cfg["width"], device, dtype)
            transformer = self._minit2i_stage_transformer(device, params, model_key=_kh_model_key)
            applied_lora = self._load_lora_minit2i(prepared_loras)
            style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                self._minit2i_style_configs(params, cfg, device, dtype, model_key=_kh_model_key)
            style_active = style_cfg is not None or style_refs is not None
            if style_active:
                call_target, nag_wrapper, negpip_wrapper = transformer, None, None
                print("[MiniT2I] Style transfer active: NAG/NegPip wrapping skipped for this generation")
            else:
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
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                if negpip_wrapper is not None:
                    negpip_wrapper.restore()
                if nag_wrapper is not None:
                    nag_wrapper.restore()
                if applied_lora:
                    self._unload_lora_minit2i()
                self._minit2i_unstage_transformer(keep_transformer=_kh_keep_transformer, model_key=_kh_model_key)
            image = self._minit2i_decode(x, cfg, model_key=_kh_model_key, keep_vae=_kh_keep_vae)
            _kh_gen_succeeded = True
            print("[MiniT2I] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[MiniT2I] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._minit2i_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                  _kh_keep_vae, _kh_gen_succeeded)
