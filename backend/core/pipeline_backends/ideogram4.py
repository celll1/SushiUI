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


def set_ideogram4_attention_backend(transformer, uncond_transformer, backend: str) -> str:
    """Stamp the inference attention backend on every Ideogram4Attention processor.

    ``backend`` is the canonical inference selector (``normal|flash|sage``, or the
    app-wide default). It is normalized and mapped to the diffusers
    AttentionBackendName via ``to_diffusers_backend`` and set as
    ``_attention_backend`` on:

      1. each ``Ideogram4Attention`` module's currently-installed base processor
         (scanning BOTH the conditional and unconditional transformers), and
      2. the ``Ideogram4NAGAttnProcessor`` / ``Ideogram4NegPipAttnProcessor``
         CLASSES, so NAG / NegPip instances created later (by the NAG wrapper and
         the NegPip installer) -- and the dynamic NegPip+NAG subclass, which
         inherits from the NAG processor -- pick up the same backend.

    The vendored ``ideogram4_dispatch_attention`` reads this attribute: ``flash``
    engages ``flash_attn_varlen_func`` over the block-diagonal segment mask (D2),
    ``sage`` falls back to native for Ideogram 4's head_dim=256, and ``native``
    is byte-identical to the legacy diffusers default path.
    """
    from core.attention import normalize_backend, to_diffusers_backend
    from core.inference.nag_ideogram4 import Ideogram4NAGAttnProcessor
    from core.inference.negpip_ideogram4 import Ideogram4NegPipAttnProcessor

    canonical = normalize_backend(backend)
    diff_backend = to_diffusers_backend(canonical)

    n = 0
    for t in (transformer, uncond_transformer):
        if t is None:
            continue
        for m in t.modules():
            if type(m).__name__ == "Ideogram4Attention":
                proc = getattr(m, "processor", None)
                if proc is not None:
                    proc._attention_backend = diff_backend
                n += 1

    Ideogram4NAGAttnProcessor._attention_backend = diff_backend
    Ideogram4NegPipAttnProcessor._attention_backend = diff_backend

    print(f"[Ideogram4] Attention backend '{canonical}' -> diffusers '{diff_backend}' set on {n} module(s)")
    return diff_backend


class Ideogram4Mixin:
    """Ideogram4Mixin: ideogram4 backend methods extracted verbatim from pipeline.py."""

    def _load_lora_ideogram4(self, lora_configs: List[Dict]) -> int:
        """Wrap Ideogram 4 transformer Linear/Fp8Linear modules with LoRA adapters.

        Applies the conditional-branch LoRA to `transformer`; if the checkpoint
        also carries unconditional-branch keys (lora_uncond_*) and the
        unconditional transformer is loaded, those are applied to it too.
        Must be called after the transformer(s) are staged on GPU.
        """
        from core.models.ideogram4.ideogram4_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
        )
        from core.extensions.lora_manager import lora_manager

        if not lora_configs or not self.ideogram4_components:
            return 0

        transformer = self.ideogram4_components["transformer"]
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if not hasattr(self, "_ideogram4_lora_orig"):
            self._ideogram4_lora_orig: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys: set = set()
            self._ideogram4_lora_orig_uncond: Dict[str, torch.nn.Module] = {}
            self._ideogram4_lora_keys_uncond: set = set()

        total = 0
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                print(f"[Ideogram4 LoRA] WARNING: file not found: {lora_path}")
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw, branch="cond")
                applied = apply_lora_group(
                    transformer, grouped, strength,
                    self._ideogram4_lora_orig, self._ideogram4_lora_keys,
                )
                print(f"[Ideogram4 LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} cond_modules={len(grouped)} wrapped={applied} strength={strength}")
                total += applied

                if uncond is not None:
                    grouped_u = normalise_lora_state_dict(raw, branch="uncond")
                    if grouped_u:
                        applied_u = apply_lora_group(
                            uncond, grouped_u, strength,
                            self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
                        )
                        print(f"[Ideogram4 LoRA]   uncond wrapped {applied_u} module(s)")
                        total += applied_u
            except Exception as e:
                print(f"[Ideogram4 LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
        return total

    def _unload_lora_ideogram4(self) -> int:
        """Restore every Ideogram 4 transformer Linear to its pre-LoRA original."""
        from core.models.ideogram4.ideogram4_lora import restore_originals
        if not self.ideogram4_components:
            return 0
        restored = 0
        if getattr(self, "_ideogram4_lora_keys", None):
            restored += restore_originals(
                self.ideogram4_components["transformer"],
                self._ideogram4_lora_orig, self._ideogram4_lora_keys,
            )
        uncond = self.ideogram4_components.get("unconditional_transformer")
        if uncond is not None and getattr(self, "_ideogram4_lora_keys_uncond", None):
            restored += restore_originals(
                uncond, self._ideogram4_lora_orig_uncond, self._ideogram4_lora_keys_uncond,
            )
        if restored:
            print(f"[Ideogram4 LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    def _ideogram4_move(self, component_name: str, target_device: str):
        """Move a named Ideogram 4 component to the target device.

        Weights are already weight-only FP8 (Fp8Linear buffers move with .to());
        plain .to() is sufficient — no extra quantization step is applied.
        """
        comp = self.ideogram4_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[Ideogram4] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    @staticmethod
    def _ideogram4_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs for Ideogram 4 generation.

        Consumed by ideogram4_pipeline_ops._blend_guidance (standard CFG blend
        plus optional schedule / SNR-rescale / dynamic thresholding).
        """
        return {
            "cfg_schedule_type": params.get("cfg_schedule_type", "constant"),
            "cfg_schedule_min": params.get("cfg_schedule_min", 1.0),
            "cfg_schedule_max": params.get("cfg_schedule_max"),
            "cfg_schedule_power": params.get("cfg_schedule_power", 2.0),
            "cfg_rescale_snr_alpha": params.get("cfg_rescale_snr_alpha", 0.0),
            "dynamic_threshold_percentile": params.get("dynamic_threshold_percentile", 0.0),
            "dynamic_threshold_mimic_scale": params.get("dynamic_threshold_mimic_scale", 1.0),
            "developer_mode": params.get("developer_mode", False),
        }

    def _ideogram4_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        """Resolve shared Ideogram 4 generation parameters."""
        from core.models.ideogram4.ideogram4_resolution import normalize_resolution, latent_grid

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        req_width = int(params.get("width", default_w))
        req_height = int(params.get("height", default_h))
        width, height = normalize_resolution(req_width, req_height)
        if (width, height) != (req_width, req_height):
            print(f"[Ideogram4] Resolution aligned: {req_width}x{req_height} -> {width}x{height}")
        grid_h, grid_w = latent_grid(width, height)

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "num_inference_steps": int(params.get("steps", 28)),
            "guidance_scale": float(params.get("cfg_scale", 7.0)),
            "mu": float(params.get("ideogram4_mu", 0.0)),
            "std": float(params.get("ideogram4_std", 1.5)),
            "max_sequence_length": int(params.get("ideogram4_max_seq_len", 512)),
            "width": width,
            "height": height,
            "grid_h": grid_h,
            "grid_w": grid_w,
        }

    @staticmethod
    def _ideogram4_negpip_clean_prompt(prompt: str, params: Dict[str, Any]) -> str:
        """Return the emphasis-syntax-stripped prompt when NegPip will auto-activate, else the
        prompt unchanged.

        NegPip carries ALL the signed emphasis weights via V scaling (like the SDXL
        skip_emphasis clean-embeds path), so when a negative weight is present the CLEAN text
        (parentheses/weights removed) must be fed to ``encode_prompt`` — otherwise Qwen3-VL
        would tokenize the literal ``(worst quality:-1)`` characters. The clean text matches
        the token->weight alignment in ``build_ideogram4_text_weights``. Positive-only prompts
        return unchanged so the default encode path is byte-identical.
        """
        from core.prompts.prompt_parser import prompt_has_negative_weight, parse_prompt_attention
        # NegPip is a GLOBAL decision (main prompt OR, with NAG, the nag-negative prompt has a
        # negative weight). Once active, strip emphasis from ANY text fed to the encoder so V
        # scaling carries all weights consistently.
        main_prompt = params.get("prompt", "")
        is_nag = bool(params.get("nag_enable", False)) and float(params.get("nag_scale", 5.0)) > 1.0
        nag_neg = (params.get("nag_negative_prompt", "") or params.get("negative_prompt", "") or "")
        negpip_active = prompt_has_negative_weight(main_prompt) or (
            is_nag and prompt_has_negative_weight(nag_neg)
        )
        if not negpip_active:
            return prompt
        parsed = parse_prompt_attention(prompt) if prompt else []
        return "".join(t for t, _ in parsed)

    @torch.no_grad()
    def _ideogram4_encode(self, prompt, grid_h, grid_w, max_sequence_length, device, dtype,
                          skip_gpu_stage: bool = False, skip_cpu_offload: bool = False):
        """Stage the text encoder to GPU, encode the prompt, then free it back to CPU.

        ``skip_gpu_stage``/``skip_cpu_offload`` let a keep-models-hot caller skip the
        ->GPU stage (already resident from a previous generation) and/or the ->CPU
        offload (kept hot for the next queued generation) around this single encode
        call. Both default False, so the default behaviour is byte-identical.
        """
        from core.models.ideogram4.ideogram4_pipeline_ops import encode_prompt

        if not skip_gpu_stage:
            self._ideogram4_move("text_encoder", device)
        text_encoder = self.ideogram4_components["text_encoder"]
        tokenizer = self.ideogram4_components["tokenizer"]
        cond = encode_prompt(
            text_encoder, tokenizer, prompt,
            grid_h=grid_h, grid_w=grid_w,
            max_sequence_length=max_sequence_length, device=device,
        )
        if not skip_cpu_offload:
            self._ideogram4_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Cast conditioning to the transformer compute dtype (halves memory; matches RMSNorm dtype).
        cond["llm_features"] = cond["llm_features"].to(dtype)
        cond["neg_llm_features"] = cond["neg_llm_features"].to(dtype)
        return cond

    @torch.no_grad()
    def _ideogram4_encode_nag_negative(self, params, cfg, cond, device, dtype,
                                       skip_gpu_stage: bool = False, skip_cpu_offload: bool = False):
        """Encode the NAG-negative prompt into a packed ``nag_llm_features`` tensor and store
        it on ``cond`` — only when NAG is active. Byte-identical (returns early) otherwise.

        NAG is gated on ``nag_enable`` AND ``nag_scale > 1`` (nag-negative defaults to the
        empty prompt like FLUX.2). The negative features share the positive prompt's packed
        layout (same ``encode_prompt`` path), so the conditional transformer can run a
        doubled ``[positive; nag_negative]`` text batch.

        ``skip_gpu_stage``/``skip_cpu_offload``: see ``_ideogram4_encode`` docstring —
        same keep-models-hot semantics, applied to this (second, optional) text-encoder use.
        """
        from core.models.ideogram4.ideogram4_pipeline_ops import encode_prompt

        nag_enable = bool(params.get("nag_enable", False))
        nag_scale = float(params.get("nag_scale", 5.0))
        if not (nag_enable and nag_scale > 1.0):
            return None
        nag_neg_prompt = params.get("nag_negative_prompt", "") or params.get("negative_prompt", "") or ""
        # When NegPip is active, strip emphasis syntax so the nag-negative text tokenizes
        # cleanly; the signed nag_neg weights are carried by NegPip's V scaling instead.
        nag_neg_prompt = self._ideogram4_negpip_clean_prompt(nag_neg_prompt, params)

        if not skip_gpu_stage:
            self._ideogram4_move("text_encoder", device)
        text_encoder = self.ideogram4_components["text_encoder"]
        tokenizer = self.ideogram4_components["tokenizer"]
        nag_cond = encode_prompt(
            text_encoder, tokenizer, nag_neg_prompt,
            grid_h=cfg["grid_h"], grid_w=cfg["grid_w"],
            max_sequence_length=cfg["max_sequence_length"], device=device,
        )
        if not skip_cpu_offload:
            self._ideogram4_move("text_encoder", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Packed features with the nag-negative TEXT region (image region stays zero-padded),
        # same shape/layout as cond["llm_features"] so it feeds the doubled cond forward.
        cond["nag_llm_features"] = nag_cond["llm_features"].to(dtype)
        return {"nag_scale": nag_scale,
                "nag_tau": float(params.get("nag_tau", 2.5)),
                "nag_alpha": float(params.get("nag_alpha", 0.25))}

    def _ideogram4_wrap_nag(self, transformer, nag_cfg):
        """Wrap the conditional transformer with the NAG wrapper (in place of the raw
        transformer for the denoise loop). Returns the wrapper, or the transformer unchanged
        when NAG is inactive."""
        if nag_cfg is None:
            return transformer
        from core.inference.nag_ideogram4 import Ideogram4NAGWrapper
        print(f"[Ideogram4] NAG enabled: scale={nag_cfg['nag_scale']}, "
              f"tau={nag_cfg['nag_tau']}, alpha={nag_cfg['nag_alpha']}")
        return Ideogram4NAGWrapper(
            transformer,
            nag_scale=nag_cfg["nag_scale"], nag_tau=nag_cfg["nag_tau"], nag_alpha=nag_cfg["nag_alpha"],
        )

    @staticmethod
    def _ideogram4_unwrap_nag(transformer):
        """Restore the original attention processors if ``transformer`` is a NAG wrapper.
        Returns the underlying transformer."""
        if transformer.__class__.__name__ == "Ideogram4NAGWrapper":
            transformer.restore()
            return transformer.transformer
        return transformer

    def _ideogram4_maybe_negpip(self, transformer, params, cfg, cond, device, dtype):
        """Auto-activate NegPip on the conditional transformer when the prompt (or, with NAG,
        the nag-negative prompt) carries a negative emphasis weight.

        Returns a handle dict (with ``restore``) to undo the processor swap after denoising,
        or ``None`` when NegPip is not active. Byte-identical (returns early) when the prompt
        has no negative weight -- the positive-only default path never installs processors.

        Ideogram 4 has no CLIP text encoder: the signed per-token weight vector is built with
        the model's own tokenizer + chat template (matching ``encode_prompt``), aligned to the
        left-padded ``[text][image]`` packed sequence. Because Ideogram 4's CFG is dual-branch
        with a ZEROED-text unconditional branch, only the conditional branch's text V is
        scaled (there is no unconditional text context to double-negate). When NAG is active
        the doubled ``[pos; nag_neg]`` batch gets per-half weights so a negative weight in the
        nag-negative prompt re-affirms.
        """
        from core.prompts.prompt_parser import prompt_has_negative_weight
        from core.inference.negpip_ideogram4 import (
            build_ideogram4_negpip_weights, install_negpip,
        )

        # Weights are built from the ORIGINAL prompt (with emphasis syntax); the CLEAN prompt
        # was already fed to encode_prompt so the token positions line up.
        prompt = cfg.get("negpip_prompt", cfg["prompt"])
        is_nag = transformer.__class__.__name__ == "Ideogram4NAGWrapper"
        nag_neg_prompt = None
        if is_nag:
            nag_neg_prompt = (
                params.get("nag_negative_prompt", "")
                or params.get("negative_prompt", "")
                or ""
            )

        has_neg = prompt_has_negative_weight(prompt) or (
            is_nag and prompt_has_negative_weight(nag_neg_prompt)
        )
        if not has_neg:
            return None

        tokenizer = self.ideogram4_components["tokenizer"]
        weights = build_ideogram4_negpip_weights(
            prompt=prompt,
            tokenizer=tokenizer,
            max_sequence_length=cfg["max_sequence_length"],
            grid_h=cfg["grid_h"],
            grid_w=cfg["grid_w"],
            device=device,
            dtype=dtype,
            nag_negative_prompt=nag_neg_prompt if is_nag else None,
        )

        if is_nag:
            import torch as _torch
            token_weights = _torch.stack([weights["pos"], weights["nag_neg"]], dim=0)
        else:
            token_weights = weights["pos"].unsqueeze(0)

        print(f"[NegPip/Ideogram4] Negative emphasis detected -> signed V scaling active "
              f"(nag={'on' if is_nag else 'off'})")
        return install_negpip(transformer, token_weights)

    def _ideogram4_style_triple(self, params: Dict[str, Any], style_dict: Dict[str, Any],
                                height: int, width: int, device, dtype,
                                model_key: Optional[str] = None, ref_index: int = 0):
        """Build a single (StyleTransferConfig, ref_x0, eps_ref) triple from one
        style_transfer dict.

        ``axes_dims`` is deliberately left unset (``None``): Ideogram 4's MRoPE is
        INTERLEAVED (``Ideogram4MRoPE`` splices H/W frequencies into every-3rd
        channel), which ``frequency_scale_vector``'s concatenated-per-axis-block
        layout does not match -- ``core.models.ideogram4.style_ideogram4``'s hook
        passes an all-ones frequency vector straight to ``inject_kv``/
        ``inject_kv_multi`` instead of calling ``StyleTransferConfig.get_freq_scale_vector``
        (which requires ``axes_dims``; it is never read here).

        Reuses ``ideogram4_pipeline_ops.vae_encode`` (already produces the exact
        packed, patchified, BN-normalized token layout the transformer's image
        region expects) rather than duplicating that encode path.

        ``ref_index`` decorrelates the fixed re-noising noise tensor across
        multiple simultaneous references (each ref would otherwise draw the
        EXACT same noise from the ``seed+991`` offset, since that offset does
        not depend on which reference is being prepared). ``ref_index=0`` (the
        default, used by the single-ref path) reproduces the pre-multi-ref
        ``seed+991`` offset exactly.

        Moves the VAE to ``device`` (if not already resident) for THIS ref's
        encode, then back to CPU afterwards -- the same round-trip the
        original single-ref implementation performed. For N>1 references this
        means N independent VAE round-trips rather than one shared residency
        window across all refs; functionally correct, just not the most
        efficient sequencing (a known perf follow-up, mirrors the "default
        per-ref strength tuning pending" disclosure on the Anima N-ref
        commit).
        """
        from diffusers.utils.torch_utils import randn_tensor
        from core.inference.reference_style import style_config_from_dict
        from core.models.ideogram4.ideogram4_pipeline_ops import vae_encode
        from core.keep_hot import is_resident, discard_resident

        cfg = style_config_from_dict(style_dict)

        if not is_resident(self, "vae", model_key):
            self._ideogram4_move("vae", device)
        vae_gpu = self.ideogram4_components["vae"]
        ref_x0 = vae_encode(vae_gpu, style_dict["image"], height, width, device=device, dtype=dtype)
        self._ideogram4_move("vae", "cpu")
        discard_resident(self, "vae")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed = params.get("seed", -1)
        ref_seed = None if seed is None or seed < 0 else (int(seed) + 991 + ref_index) % (2**32)
        generator = torch.Generator(device=device).manual_seed(ref_seed) if ref_seed is not None else None
        eps_ref = randn_tensor(ref_x0.shape, generator=generator, device=device, dtype=ref_x0.dtype)
        return cfg, ref_x0, eps_ref

    def _ideogram4_style_config(self, params: Dict[str, Any], height: int, width: int,
                                device, dtype, model_key: Optional[str] = None):
        """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None, None)``
        when no style reference is attached. Single-reference path,
        BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_ideogram4_style_triple`` with ``ref_index=0``, which reproduces the
        original ``seed+991`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        return self._ideogram4_style_triple(
            params, style_dict, height, width, device, dtype, model_key=model_key, ref_index=0,
        )

    def _ideogram4_style_configs(self, params: Dict[str, Any], height: int, width: int,
                                 device, dtype, model_key: Optional[str] = None):
        """Build the full style-transfer configuration for Ideogram 4 generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple, exactly as ``_ideogram4_style_config``
        would return) and the multi-reference path (``style_refs``, a list of
        per-ref triples, populated ONLY when ``params["style_transfers"]`` has
        more than one entry). A single-entry ``style_transfers`` list is
        intentionally routed through the single-ref triple instead (``style_refs``
        stays ``None``), so the pre-multi-ref code path executes
        byte-identically end to end.

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
                refs.append(self._ideogram4_style_triple(
                    params, style_dict, height, width, device, dtype,
                    model_key=model_key, ref_index=idx,
                ))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._ideogram4_style_config(
            params, height, width, device, dtype, model_key=model_key,
        )
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    def _ideogram4_setup_block_swap(self, transformer, blocks_to_swap: int,
                                    use_pinned_memory: bool, device: str,
                                    h2d_only: bool = False, ring_size: int = 2):
        """Attach a block-swap offloader to one Ideogram 4 transformer.

        The transformer starts on CPU; the offloader keeps the first
        (num_layers - blocks_to_swap) blocks resident on GPU and streams the rest
        per forward. Non-block (auxiliary) modules are moved to GPU here since the
        shared offloader only auto-moves Z-Image-named aux modules.
        """
        from core.memory_management import create_block_offloader_for_model

        # Auxiliary modules (everything except the swappable block list) stay on GPU.
        for name, child in transformer.named_children():
            if name != "layers":
                child.to(device)
        for p in transformer.parameters(recurse=False):
            if p.device.type != "cuda":
                p.data = p.data.to(device)
        for b in transformer.buffers(recurse=False):
            if b.device.type != "cuda":
                b.data = b.data.to(device)

        offloader = create_block_offloader_for_model(
            transformer=transformer,
            blocks_to_swap=blocks_to_swap,
            device=torch.device(device),
            target_dtype=torch.bfloat16,
            use_pinned_memory=use_pinned_memory,
            h2d_only=h2d_only,
            ring_size=ring_size,
        )
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _ideogram4_stage_transformers(self, device: str, params: Optional[Dict[str, Any]] = None):
        """Place both transformers on GPU for the denoise loop.

        With block swap enabled, each transformer streams its blocks (per-model
        offloader) instead of being fully resident, roughly halving the resident
        footprint of the two 9.3B FP8 transformers at the cost of CPU<->GPU traffic.
        """
        params = params or {}
        enable_block_swap = bool(params.get("enable_block_swap", False))
        num_layers = len(self.ideogram4_components["transformer"].layers)
        blocks_to_swap = int(params.get("blocks_to_swap", 20))
        blocks_to_swap = max(0, min(blocks_to_swap, num_layers - 1))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))
        h2d_only = bool(params.get("block_swap_h2d_only", False))
        ring_size = int(params.get("block_swap_ring_size", 2))

        self._ideogram4_offloaders = []
        if enable_block_swap and blocks_to_swap > 0:
            print(f"[Ideogram4] Block swap enabled: {blocks_to_swap}/{num_layers} blocks per transformer "
                  f"(pinned_memory={use_pinned_memory}, h2d_only={h2d_only}, ring_size={ring_size})")
            for comp_name in ("transformer", "unconditional_transformer"):
                t = self.ideogram4_components[comp_name]
                off = self._ideogram4_setup_block_swap(t, blocks_to_swap, use_pinned_memory, device,
                                                       h2d_only=h2d_only, ring_size=ring_size)
                self._ideogram4_offloaders.append((comp_name, off))
        else:
            self._ideogram4_move("transformer", device)
            self._ideogram4_move("unconditional_transformer", device)
        return (
            self.ideogram4_components["transformer"],
            self.ideogram4_components["unconditional_transformer"],
        )

    def _ideogram4_unstage_transformers(self):
        # Tear down any block-swap offloaders, then return both transformers to CPU.
        offloaders = getattr(self, "_ideogram4_offloaders", [])
        for comp_name, off in offloaders:
            t = self.ideogram4_components.get(comp_name)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
            cleanup = getattr(off, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._ideogram4_offloaders = []
        self._ideogram4_move("transformer", "cpu")
        self._ideogram4_move("unconditional_transformer", "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ideogram4_cleanup(self, model_key: Optional[str] = None, gen_succeeded: bool = False,
                           keep_te: bool = False, keep_transformer: bool = False,
                           keep_vae: bool = False):
        """Final cross-generation boundary: strip any leftover block-swap offloaders,
        then ensure components are back on CPU -- EXCEPT components kept hot on a
        SUCCESSFUL generation (see core/keep_hot.py). On an exception (``gen_succeeded``
        False) or when keep-hot bookkeeping was never engaged (``model_key`` None),
        this forces a full offload of every component, matching the pre-keep-hot
        behaviour byte-for-byte.

        Ideogram 4 stages BOTH transformers (cond + uncond) as one logical residency
        unit named ``"transformer"`` (they are always staged/unstaged together by
        ``_ideogram4_stage_transformers``/``_ideogram4_unstage_transformers``), so
        ``keep_transformer`` gates offloading BOTH.
        """
        from core.keep_hot import mark_resident, discard_resident, clear_resident

        # Strip any leftover block-swap offloaders (e.g. if setup raised mid-way).
        for _comp in ("transformer", "unconditional_transformer"):
            t = (self.ideogram4_components or {}).get(_comp)
            if t is not None and hasattr(t, "_block_offloader"):
                try:
                    delattr(t, "_block_offloader")
                except Exception:
                    pass
        self._ideogram4_offloaders = []

        if not gen_succeeded or model_key is None:
            clear_resident(self)
            for _comp in ("text_encoder", "transformer", "unconditional_transformer", "vae"):
                try:
                    self._ideogram4_move(_comp, "cpu")
                except Exception:
                    pass
        else:
            if keep_te:
                mark_resident(self, "text_encoder", model_key)
            else:
                try:
                    self._ideogram4_move("text_encoder", "cpu")
                except Exception:
                    pass
                discard_resident(self, "text_encoder")
            if keep_transformer:
                mark_resident(self, "transformer", model_key)
            else:
                try:
                    self._ideogram4_move("transformer", "cpu")
                    self._ideogram4_move("unconditional_transformer", "cpu")
                except Exception:
                    pass
                discard_resident(self, "transformer")
            if keep_vae:
                mark_resident(self, "vae", model_key)
            else:
                try:
                    self._ideogram4_move("vae", "cpu")
                except Exception:
                    pass
                discard_resident(self, "vae")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_txt2img_ideogram4(self, params: Dict[str, Any],
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded. Please load an Ideogram 4 model first.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            prepare_latents, denoise_loop, vae_decode,
        )

        print("[Ideogram4] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, 1024, 1024)
        cfg["negpip_prompt"] = cfg["prompt"]  # original (with emphasis syntax) for weight build
        cfg["prompt"] = self._ideogram4_negpip_clean_prompt(cfg["prompt"], params)
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident,
            should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._ideogram4_move("text_encoder", "cpu"),
                self._ideogram4_move("transformer", "cpu"),
                self._ideogram4_move("unconditional_transformer", "cpu"),
                self._ideogram4_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("transformer"))
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("unconditional_transformer"))
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        # Ideogram 4 has no CPU-text-encoding mode, so TE eligibility is guard-only.
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
                skip_gpu_stage=is_resident(self, "text_encoder", _kh_model_key),
                skip_cpu_offload=_kh_keep_te,
            )

            print("[Ideogram4] Stage 2: Prepare latents...")
            latents = prepare_latents(
                cfg["grid_h"], cfg["grid_w"], dtype=torch.float32, device=device, seed=cfg["seed"],
            )

            # Training-free reference-style transfer: mutually exclusive with NAG/NegPip for
            # the WHOLE generation (both rewrite the attention-time token/value layout, same
            # conflict as FBCache below) -- decided BEFORE either is set up so neither text
            # encode nor auto-activation ever runs when style is active.
            style_active = bool(params.get("style_transfer") and params["style_transfer"].get("image"))
            if style_active:
                print("[Ideogram4] Style transfer active: disabling NAG/NegPip for this generation")
                nag_cfg = None
            else:
                nag_cfg = self._ideogram4_encode_nag_negative(
                    params, cfg, cond, device, dtype,
                    skip_gpu_stage=_kh_keep_te, skip_cpu_offload=_kh_keep_te,
                )

            print("[Ideogram4] Stage 3: Denoising (dual-branch)...")
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.ideogram4_components["transformer"]
                uncond_transformer = self.ideogram4_components["unconditional_transformer"]
                self._ideogram4_offloaders = []
            else:
                transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            set_ideogram4_attention_backend(
                transformer, uncond_transformer,
                params.get("attention_type", settings.attention_type),
            )
            if style_active:
                # Mask-extension correctness (see core.models.ideogram4.style_ideogram4) is only
                # implemented for the dense (B,1,L,L) boolean-mask "native" path -- the "flash"
                # backend bypasses attention_mask entirely via cu_seqlens, which would silently
                # miss the appended reference-K columns. Force native for the whole generation.
                set_ideogram4_attention_backend(transformer, uncond_transformer, "normal")
                print("[Ideogram4] Style transfer active: forcing native attention backend "
                      "(flash's cu_seqlens path cannot see the appended reference-K columns)")
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            transformer = self._ideogram4_wrap_nag(transformer, nag_cfg)
            negpip_handle = None if style_active else self._ideogram4_maybe_negpip(
                transformer, params, cfg, cond, device, dtype)

            style_processors: List[Any] = []
            style_saved: List[Any] = []
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if style_active:
                # ``style_refs`` is populated (and style_cfg/style_ref_x0/style_eps_ref
                # left None) ONLY when ``params["style_transfers"]`` carries 2+
                # references -- a single reference always resolves through the
                # style_cfg/style_ref_x0/style_eps_ref triple, so that code path
                # (both here and inside denoise_loop/_run_loop) is untouched.
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._ideogram4_style_configs(
                        params, cfg["height"], cfg["width"], device, dtype, model_key=_kh_model_key,
                    )
                if style_cfg is not None or style_refs is not None:
                    from core.models.ideogram4.style_ideogram4 import install_ideogram4_style_processors
                    style_processors, style_saved = install_ideogram4_style_processors(transformer)

            try:
                latents = denoise_loop(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, latents=latents, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    spectrum_params=params,
                    style_processors=style_processors, style_cfg=style_cfg,
                    style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                if style_saved:
                    from core.models.ideogram4.style_ideogram4 import restore_ideogram4_style_processors
                    restore_ideogram4_style_processors(style_saved)
                if negpip_handle is not None:
                    negpip_handle["restore"]()
                transformer = self._ideogram4_unwrap_nag(transformer)
                if applied_lora:
                    self._unload_lora_ideogram4()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._ideogram4_unstage_transformers()
            # Transformer marked optimistically; _kh_gen_succeeded is set only AFTER
            # decode below, so a decode failure routes to the finally exception
            # branch (clear_resident + full offload), undoing this mark.
            del cond

            print("[Ideogram4] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded.
            _kh_gen_succeeded = True

            print("[Ideogram4] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup(
                model_key=_kh_model_key, gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te, keep_transformer=_kh_keep_transformer, keep_vae=_kh_keep_vae,
            )

    def _generate_img2img_ideogram4(self, params: Dict[str, Any], init_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_img2img, vae_decode,
        )

        print("[Ideogram4] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        cfg["negpip_prompt"] = cfg["prompt"]
        cfg["prompt"] = self._ideogram4_negpip_clean_prompt(cfg["prompt"], params)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident,
            should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._ideogram4_move("text_encoder", "cpu"),
                self._ideogram4_move("transformer", "cpu"),
                self._ideogram4_move("unconditional_transformer", "cpu"),
                self._ideogram4_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("transformer"))
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("unconditional_transformer"))
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
                skip_gpu_stage=is_resident(self, "text_encoder", _kh_model_key),
                skip_cpu_offload=_kh_keep_te,
            )

            print("[Ideogram4] Stage 2: Encoding init image...")
            # First use of the VAE this generation: the cross-generation entry point.
            if not is_resident(self, "vae", _kh_model_key):
                self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, cfg["height"], cfg["width"],
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            style_active = bool(params.get("style_transfer") and params["style_transfer"].get("image"))
            if style_active:
                print("[Ideogram4] Style transfer active: disabling NAG/NegPip for this generation")
                nag_cfg = None
            else:
                nag_cfg = self._ideogram4_encode_nag_negative(
                    params, cfg, cond, device, dtype,
                    skip_gpu_stage=_kh_keep_te, skip_cpu_offload=_kh_keep_te,
                )

            print("[Ideogram4] Stage 3: Denoising (SDEdit)...")
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.ideogram4_components["transformer"]
                uncond_transformer = self.ideogram4_components["unconditional_transformer"]
                self._ideogram4_offloaders = []
            else:
                transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            set_ideogram4_attention_backend(
                transformer, uncond_transformer,
                params.get("attention_type", settings.attention_type),
            )
            if style_active:
                set_ideogram4_attention_backend(transformer, uncond_transformer, "normal")
                print("[Ideogram4] Style transfer active: forcing native attention backend "
                      "(flash's cu_seqlens path cannot see the appended reference-K columns)")
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            transformer = self._ideogram4_wrap_nag(transformer, nag_cfg)
            negpip_handle = None if style_active else self._ideogram4_maybe_negpip(
                transformer, params, cfg, cond, device, dtype)

            style_processors: List[Any] = []
            style_saved: List[Any] = []
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if style_active:
                # See the txt2img comment above for the single-ref/multi-ref
                # routing invariant.
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._ideogram4_style_configs(
                        params, cfg["height"], cfg["width"], device, dtype, model_key=_kh_model_key,
                    )
                if style_cfg is not None or style_refs is not None:
                    from core.models.ideogram4.style_ideogram4 import install_ideogram4_style_processors
                    style_processors, style_saved = install_ideogram4_style_processors(transformer)

            try:
                latents = denoise_loop_img2img(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, denoising_strength=denoising_strength,
                    cond=cond, guidance_scale=cfg["guidance_scale"],
                    num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=cfg["height"], width=cfg["width"],
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    spectrum_params=params,
                    style_processors=style_processors, style_cfg=style_cfg,
                    style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                if style_saved:
                    from core.models.ideogram4.style_ideogram4 import restore_ideogram4_style_processors
                    restore_ideogram4_style_processors(style_saved)
                if negpip_handle is not None:
                    negpip_handle["restore"]()
                transformer = self._ideogram4_unwrap_nag(transformer)
                if applied_lora:
                    self._unload_lora_ideogram4()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._ideogram4_unstage_transformers()
            # Transformer marked optimistically; _kh_gen_succeeded is set only AFTER
            # decode below, so a decode failure routes to the finally exception
            # branch (clear_resident + full offload), undoing this mark.
            del cond, init_latents

            print("[Ideogram4] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded.
            _kh_gen_succeeded = True

            print("[Ideogram4] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup(
                model_key=_kh_model_key, gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te, keep_transformer=_kh_keep_transformer, keep_vae=_kh_keep_vae,
            )

    def _generate_inpaint_ideogram4(self, params: Dict[str, Any],
                                    init_image: Image.Image, mask_image: Image.Image,
                                    progress_callback=None, step_callback=None) -> tuple:
        if not self.ideogram4_components:
            raise RuntimeError("Ideogram 4 components not loaded.")

        from core.models.ideogram4.ideogram4_pipeline_ops import (
            vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )

        print("[Ideogram4] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._ideogram4_common_params(params, init_image.width, init_image.height)
        cfg["negpip_prompt"] = cfg["prompt"]
        cfg["prompt"] = self._ideogram4_negpip_clean_prompt(cfg["prompt"], params)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        scheduler = self.ideogram4_components["scheduler"]
        advanced_cfg = self._ideogram4_advanced_cfg(params)

        width, height = cfg["width"], cfg["height"]
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident,
            should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._ideogram4_move("text_encoder", "cpu"),
                self._ideogram4_move("transformer", "cpu"),
                self._ideogram4_move("unconditional_transformer", "cpu"),
                self._ideogram4_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("transformer"))
                _kh_total_bytes += component_nbytes(self.ideogram4_components.get("unconditional_transformer"))
            _kh_total_bytes += component_nbytes(self.ideogram4_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            print("[Ideogram4] Stage 1: Text encoding...")
            cond = self._ideogram4_encode(
                cfg["prompt"], cfg["grid_h"], cfg["grid_w"],
                cfg["max_sequence_length"], device, dtype,
                skip_gpu_stage=is_resident(self, "text_encoder", _kh_model_key),
                skip_cpu_offload=_kh_keep_te,
            )

            print("[Ideogram4] Stage 2: Encoding init image + mask...")
            # First use of the VAE this generation: the cross-generation entry point.
            if not is_resident(self, "vae", _kh_model_key):
                self._ideogram4_move("vae", device)
            init_latents = vae_encode(
                self.ideogram4_components["vae"], init_image, height, width,
                device=device, dtype=torch.float32,
            )
            self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_latent = prepare_mask_latent(
                mask_image, cfg["grid_h"], cfg["grid_w"], device=device, dtype=torch.float32,
            )

            style_active = bool(params.get("style_transfer") and params["style_transfer"].get("image"))
            if style_active:
                print("[Ideogram4] Style transfer active: disabling NAG/NegPip for this generation")
                nag_cfg = None
            else:
                nag_cfg = self._ideogram4_encode_nag_negative(
                    params, cfg, cond, device, dtype,
                    skip_gpu_stage=_kh_keep_te, skip_cpu_offload=_kh_keep_te,
                )

            print("[Ideogram4] Stage 3: Denoising (repaint)...")
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.ideogram4_components["transformer"]
                uncond_transformer = self.ideogram4_components["unconditional_transformer"]
                self._ideogram4_offloaders = []
            else:
                transformer, uncond_transformer = self._ideogram4_stage_transformers(device, params)
            set_ideogram4_attention_backend(
                transformer, uncond_transformer,
                params.get("attention_type", settings.attention_type),
            )
            if style_active:
                set_ideogram4_attention_backend(transformer, uncond_transformer, "normal")
                print("[Ideogram4] Style transfer active: forcing native attention backend "
                      "(flash's cu_seqlens path cannot see the appended reference-K columns)")
            applied_lora = self._load_lora_ideogram4(params.get("loras") or [])
            transformer = self._ideogram4_wrap_nag(transformer, nag_cfg)
            negpip_handle = None if style_active else self._ideogram4_maybe_negpip(
                transformer, params, cfg, cond, device, dtype)

            style_processors: List[Any] = []
            style_saved: List[Any] = []
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if style_active:
                # See the txt2img comment above for the single-ref/multi-ref
                # routing invariant.
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._ideogram4_style_configs(
                        params, height, width, device, dtype, model_key=_kh_model_key,
                    )
                if style_cfg is not None or style_refs is not None:
                    from core.models.ideogram4.style_ideogram4 import install_ideogram4_style_processors
                    style_processors, style_saved = install_ideogram4_style_processors(transformer)

            try:
                latents = denoise_loop_inpaint(
                    transformer=transformer, unconditional_transformer=uncond_transformer,
                    scheduler=scheduler, init_latents=init_latents, mask_latent=mask_latent,
                    denoising_strength=denoising_strength, cond=cond,
                    guidance_scale=cfg["guidance_scale"], num_inference_steps=cfg["num_inference_steps"],
                    grid_h=cfg["grid_h"], grid_w=cfg["grid_w"], height=height, width=width,
                    mu=cfg["mu"], std=cfg["std"], seed=cfg["seed"],
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    spectrum_params=params,
                    style_processors=style_processors, style_cfg=style_cfg,
                    style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                if style_saved:
                    from core.models.ideogram4.style_ideogram4 import restore_ideogram4_style_processors
                    restore_ideogram4_style_processors(style_saved)
                if negpip_handle is not None:
                    negpip_handle["restore"]()
                transformer = self._ideogram4_unwrap_nag(transformer)
                if applied_lora:
                    self._unload_lora_ideogram4()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._ideogram4_unstage_transformers()
            # Transformer marked optimistically; _kh_gen_succeeded is set only AFTER
            # decode below, so a decode failure routes to the finally exception
            # branch (clear_resident + full offload), undoing this mark.
            del cond, init_latents, mask_latent

            print("[Ideogram4] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._ideogram4_move("vae", device)
            self._apply_vae_tiling(self.ideogram4_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.ideogram4_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._ideogram4_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded.
            _kh_gen_succeeded = True

            print("[Ideogram4] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Ideogram4] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._ideogram4_cleanup(
                model_key=_kh_model_key, gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te, keep_transformer=_kh_keep_transformer, keep_vae=_kh_keep_vae,
            )
