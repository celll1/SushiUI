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

class AnimaMixin:
    """AnimaMixin: anima backend methods extracted verbatim from pipeline.py."""

    def _anima_resolve_dtype(self, dtype_str: Optional[str] = None) -> torch.dtype:
        if dtype_str == "fp16":
            return torch.float16
        if dtype_str == "fp32":
            return torch.float32
        return torch.bfloat16

    @staticmethod
    def _anima_lora_warn(message: str, code: str) -> None:
        """Record a user-visible generation warning. ``message`` rides into the
        output PNG's text chunk and the API's ``warnings[]``, so it must never
        carry an absolute path."""
        print(f"[Anima LoRA] WARNING: {message}")
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    def _anima_lora_state(self, transformer):
        """The (originals, wrapped_keys) maps for THIS transformer.

        Reset when the DiT was reloaded: the maps hold the OLD transformer's
        Linears, ``apply_lora_group`` keeps them (setdefault) and
        ``restore_originals`` would then splice them into the new transformer.
        Keyed by weakref rather than id() because a freed object's id is reusable.
        """
        ref = getattr(self, "_anima_lora_transformer_ref", None)
        if ref is None or ref() is not transformer:
            self._anima_lora_original_modules: Dict[str, torch.nn.Module] = {}
            self._anima_lora_wrapped_keys: set = set()
            self._anima_lora_transformer_ref = weakref.ref(transformer)
        return self._anima_lora_original_modules, self._anima_lora_wrapped_keys

    def _load_lora_anima(self, lora_configs: List[Dict]) -> int:
        """Wrap target Linear modules of the Anima DiT with LoRA adapters.

        The wrapped scope is derived per file from its own keys, so an
        attention-only LoRA and a full attention+mlp+llm_adapter one both apply
        in full. Stacking is additive across DISJOINT modules only; a later file
        whose targets are ALL already wrapped is refused, because each wrap
        rebuilds from the true original and would discard the earlier branch
        instead of summing it (summing needs the composite wrapper,
        docs/guides/LYCORIS_ADAPTER_DESIGN.md Phase 1). Unload always returns to
        the un-LoRA'd model.

        Raises RuntimeError when a requested LoRA cannot be applied: a
        generation that silently ignores a selected LoRA is not a success.
        """
        from core.models.anima.anima_lora import (
            load_lora_safetensors, normalise_lora_state_dict, apply_lora_group,
            derive_scope_from_keys, unmatched_source_keys,
        )
        from core.extensions.lora_manager import lora_manager

        # Unconditional, and BEFORE the empty-config exit: a model reload or a
        # restore that failed in an earlier request must not leak the previous
        # DiT's modules into this generation.
        self._unload_lora_anima()

        if not lora_configs:
            return 0

        if not self.anima_components:
            raise RuntimeError(
                "[Anima LoRA] cannot apply the selected LoRA(s): Anima components are not loaded"
            )

        transformer = self.anima_components["transformer"]
        originals, wrapped_keys = self._anima_lora_state(transformer)

        total_applied = 0
        failures: List[str] = []
        for i, cfg in enumerate(lora_configs):
            lora_path = cfg.get("path", "")
            # Warnings ride into the PNG metadata chunk, so never an absolute path.
            lora_file = os.path.basename(str(lora_path))
            strength = float(cfg.get("strength", 1.0))
            resolved = lora_manager._resolve_lora_path(lora_path)
            if resolved is None:
                message = f"LoRA '{lora_file}': file not found"
                self._anima_lora_warn(message, "lora_not_found")
                failures.append(message)
                continue
            try:
                raw, fmt = load_lora_safetensors(str(resolved))
                grouped = normalise_lora_state_dict(raw)
                scope = derive_scope_from_keys(grouped.keys())
                dropped = unmatched_source_keys(raw, grouped)
                print(f"[Anima LoRA] {i+1}/{len(lora_configs)}: {lora_path} "
                      f"format={fmt} keys={len(raw)} matched_modules={len(grouped)} "
                      f"scope={sorted(k for k, v in scope.items() if v)} strength={strength}")
                if dropped:
                    self._anima_lora_warn(
                        f"LoRA '{lora_file}' has {len(dropped)} tensor key(s) in no "
                        f"recognised Anima LoRA format, or missing their down/up pair "
                        f"(first few: {dropped[:5]}) -- not applied.",
                        "anima_lora_keys_unrecognised")
                overlap = wrapped_keys & set(grouped)
                applied, unmatched = apply_lora_group(
                    transformer, grouped, strength, originals, wrapped_keys,
                    scope=scope,
                )
                if unmatched:
                    self._anima_lora_warn(
                        f"LoRA '{lora_file}' targets {len(unmatched)} module(s) that the "
                        f"loaded Anima DiT does not expose (first few: {unmatched[:5]}) "
                        f"-- skipped.",
                        "anima_lora_targets_unresolved")
                    if applied:
                        # A 0-applied file is incompatible, not partial; that
                        # branch below carries its own code.
                        self._anima_lora_warn(
                            f"LoRA '{lora_file}': applied {applied} of the {len(grouped)} "
                            f"module(s) the file carries.",
                            "lora_partial")
                print(f"[Anima LoRA]   wrapped {applied} module(s)")
                if applied == 0:
                    message = (
                        f"LoRA '{lora_file}': matched 0 target(s) out of {len(raw)} key(s) "
                        f"against the loaded Anima DiT (wrong architecture or key format?)"
                    )
                    self._anima_lora_warn(message, "lora_incompatible")
                    failures.append(message)
                elif len(overlap) == applied:
                    message = (
                        f"LoRA '{lora_file}': every one of its {applied} target module(s) is "
                        f"already wrapped by an earlier LoRA in this request. Anima applies "
                        f"one LoRA per module; select LoRAs with disjoint targets."
                    )
                    self._anima_lora_warn(message, "lora_stacking_unsupported")
                    failures.append(message)
                elif overlap:
                    self._anima_lora_warn(
                        f"LoRA '{lora_file}': {len(overlap)} module(s) were already wrapped by "
                        f"an earlier LoRA in this request; the earlier branch is replaced, "
                        f"not summed.",
                        "lora_partial")
                total_applied += applied
            except Exception as e:
                print(f"[Anima LoRA] ERROR loading {lora_path}: {e}")
                import traceback; traceback.print_exc()
                # Type + basename only: this rides into the PNG text chunk and the API
                # response, and an OSError's str() carries the absolute resolved path.
                message = (f"Anima LoRA '{lora_file}' could not be applied "
                           f"({type(e).__name__}); see the server log for details")
                self._anima_lora_warn(message, "lora_load_failed")
                failures.append(message)

        if failures:
            # Refuse before denoising rather than generate with a silently
            # partial LoRA set; restore the DiT first so the failure is clean.
            self._unload_lora_anima()
            raise RuntimeError("[Anima LoRA] " + "; ".join(failures))
        return total_applied

    def _unload_lora_anima(self) -> int:
        """Restore every Anima DiT Linear to its pre-LoRA original.

        Routed through ``_anima_lora_state``, so a reloaded or dropped DiT
        discards the previous model's modules instead of splicing them in.
        """
        from core.models.anima.anima_lora import restore_originals
        transformer = (self.anima_components or {}).get("transformer")
        if transformer is None:
            # Model unloaded: drop the maps so a later load cannot inherit them.
            self._anima_lora_original_modules = {}
            self._anima_lora_wrapped_keys = set()
            self._anima_lora_transformer_ref = None
            return 0
        originals, wrapped_keys = self._anima_lora_state(transformer)
        if not wrapped_keys:
            return 0
        restored, unresolved = restore_originals(transformer, originals, wrapped_keys)
        if unresolved:
            self._anima_lora_warn(
                f"{len(unresolved)} Anima LoRA wrapper(s) could not be removed (first few: "
                f"{unresolved[:5]}); the DiT may still carry LoRA weights.",
                "lora_unload_failed")
        print(f"[Anima LoRA] Unloaded {restored} LoRA wrappers")
        return restored

    @staticmethod
    def _anima_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect Advanced-CFG knobs from a generation params dict.

        Returns a dict consumed by anima_pipeline_ops._apply_advanced_cfg.
        Missing keys fall back to no-op defaults (constant CFG, no rescale,
        no threshold).
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

    @staticmethod
    def _anima_encode_nag_neg(params, encode_prompt, text_encoder, qwen3_tokenizer,
                              t5_tokenizer, enc_device, compute_dtype):
        """Encode the NAG-negative prompt using the SAME encoder path as the
        positive / negative prompt. Returns the embeds dict, or None when NAG is
        not active (so the generation path is unchanged by default).

        NAG is active only when nag_enable is set, nag_scale > 1, and a
        NAG-negative prompt string is provided (falls back to the regular
        negative_prompt when nag_negative_prompt is empty, matching the other
        backends' behaviour of guiding away from the negative context).
        """
        from core.inference.nag_dit import nag_active

        nag_enable = params.get("nag_enable", False)
        nag_scale = float(params.get("nag_scale", 1.0) or 1.0)
        nag_negative_prompt = params.get("nag_negative_prompt", "") or ""
        if not nag_negative_prompt:
            nag_negative_prompt = params.get("negative_prompt", "") or ""

        # nag_active gates on enable + scale; require a non-empty negative text too.
        if not (nag_enable and abs(nag_scale - 1.0) > 1e-5 and nag_negative_prompt):
            return None

        neg = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                            nag_negative_prompt, device=enc_device, dtype=compute_dtype)
        # Sanity: only proceed if nag_active agrees (defensive, mirrors reference).
        return neg if nag_active(nag_enable, nag_scale, neg.get("prompt_embeds")) else None

    @staticmethod
    def _anima_build_nag_wrapper(params, transformer, nag_neg_embeds):
        """Build an AnimaNAGWrapper for the conditional pass, or None when NAG
        is inactive (nag_neg_embeds is None). OFF by default."""
        if nag_neg_embeds is None:
            return None
        from core.inference.nag_anima import AnimaNAGWrapper
        nag_scale = float(params.get("nag_scale", 5.0) or 5.0)
        nag_tau = float(params.get("nag_tau", 2.5) or 2.5)
        nag_alpha = float(params.get("nag_alpha", 0.25) or 0.25)
        return AnimaNAGWrapper(
            transformer, nag_neg_embeds,
            nag_scale=nag_scale, nag_tau=nag_tau, nag_alpha=nag_alpha,
        )

    @staticmethod
    def _anima_negpip_active(params) -> bool:
        """True when NegPip should auto-activate: the positive OR the negative
        prompt carries a NEGATIVE emphasis weight. OFF by default otherwise, so
        positive-only prompts take the byte-identical default path."""
        from core.inference.negpip_anima import negpip_active
        return negpip_active(params.get("prompt", "") or "",
                             params.get("negative_prompt", "") or "")

    @staticmethod
    def _anima_build_negpip(params, cond_transformer, uncond_transformer,
                            cond_embeds, uncond_embeds, t5_tokenizer,
                            device, compute_dtype):
        """Build the (cond, uncond) NegPip transformer wrappers, or (cond, None).

        ``cond_transformer`` is whatever already drives the COND pass (the raw
        transformer, or an ``AnimaNAGWrapper`` when NAG is active) — the returned
        cond wrapper arms the POSITIVE prompt's signed weights around it, folding
        into NAG when present. ``uncond_transformer`` (raw transformer) is wrapped
        with the NEGATIVE prompt's signed weights (a negative weight there is a
        double-negative that re-affirms). Returns the possibly-wrapped
        (cond, uncond) transformers; wrappers must be ``.restore()``d after use.

        Only called when ``_anima_negpip_active`` is true, so the default path is
        untouched. If neither prompt yields a non-unit weight vector aligned to
        its T5 tokens, the corresponding pass is left as-is.
        """
        from core.inference.negpip_anima import (
            build_anima_negpip_weights, AnimaNegPipWrapper,
        )

        pos_w = build_anima_negpip_weights(
            params.get("prompt", "") or "", cond_embeds["t5_input_ids"],
            t5_tokenizer, device, compute_dtype,
        )
        cond_wrapped = cond_transformer
        if pos_w is not None:
            cond_wrapped = AnimaNegPipWrapper(cond_transformer, pos_w)

        uncond_wrapped = None
        if uncond_embeds is not None:
            neg_w = build_anima_negpip_weights(
                params.get("negative_prompt", "") or "", uncond_embeds["t5_input_ids"],
                t5_tokenizer, device, compute_dtype,
            )
            if neg_w is not None:
                uncond_wrapped = AnimaNegPipWrapper(uncond_transformer, neg_w)

        return cond_wrapped, uncond_wrapped

    def _anima_move(self, component_name: str, target_device: str,
                     quantization: Optional[str] = None):
        """Move a named Anima component to the given device.

        For GPU moves the specialized helpers in core.vram_optimization apply
        optional FP8 quantization to text_encoder / transformer; CPU moves use
        the plain .to('cpu') path. The (possibly quantized) component is
        written back into self.anima_components so subsequent calls see the
        quantized copy.
        """
        from core.vram_optimization import (
            move_anima_text_encoder_to_gpu, move_anima_text_encoder_to_cpu,
            move_anima_transformer_to_gpu, move_anima_transformer_to_cpu,
            move_anima_vae_to_gpu, move_anima_vae_to_cpu,
        )

        comp = self.anima_components.get(component_name)
        if comp is None:
            return comp

        try:
            if component_name == "text_encoder":
                if target_device == "cpu":
                    move_anima_text_encoder_to_cpu(comp)
                else:
                    comp = move_anima_text_encoder_to_gpu(comp, quantization)
                    self.anima_components["text_encoder"] = comp
            elif component_name == "transformer":
                if target_device == "cpu":
                    move_anima_transformer_to_cpu(comp)
                else:
                    comp = move_anima_transformer_to_gpu(comp, quantization)
                    self.anima_components["transformer"] = comp
            elif component_name == "vae":
                if target_device == "cpu":
                    move_anima_vae_to_cpu(comp)
                else:
                    move_anima_vae_to_gpu(comp)
            else:
                # Fallback for unknown components
                if hasattr(comp, "to"):
                    comp.to(target_device)
        except Exception as e:
            print(f"[Anima] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _anima_setup_block_swap(self, transformer, blocks_to_swap: int,
                                use_pinned_memory: bool, device: str,
                                h2d_only: bool = False, ring_size: int = 2):
        """Attach a block-swap offloader to the Anima DiT transformer.

        The transformer starts on CPU; the offloader keeps the first
        (num_blocks - blocks_to_swap) blocks resident on GPU and streams the
        rest per forward. Non-block (auxiliary) modules are moved to GPU here
        since the shared offloader only auto-moves Z-Image-named aux modules.
        """
        from core.memory_management import create_block_offloader_for_model

        # Auxiliary modules (everything except the swappable 'blocks' list) stay
        # on GPU. Anima's heavy list is 'blocks' (the LLMAdapter's own 'blocks'
        # lives under the 'llm_adapter' child, which is moved wholesale here).
        for name, child in transformer.named_children():
            if name != "blocks":
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
            block_list=transformer.blocks,
        )
        transformer._block_offloader = offloader
        offloader.prepare_block_devices_before_forward()
        return offloader

    def _anima_runtime_int8(self, transformer_quantization, progress_callback=None):
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        Runs BEFORE the transformer is staged (so the module is still on CPU and
        no second module copy is built; host RSS still ends at ~1.6x the source,
        see quantize_linears_in_place) and BEFORE LoRA wrapping (the converter
        refuses a LoRA-wrapped module, because wrappers would hide Linears and
        silently change the selection). Returns the quantization string the rest
        of staging should see: ``None`` once a conversion has happened, since
        ``_anima_quantize_fp8`` has nothing left to do on quantized Linears.
        """
        from core.vram_optimization import (
            apply_runtime_int8_quantization, runtime_int8_requested,
        )
        transformer = self.anima_components.get("transformer")
        if transformer is None:
            return transformer_quantization
        model, converted = apply_runtime_int8_quantization(
            self, transformer, "anima", transformer_quantization,
            label="Anima Transformer", progress_callback=progress_callback)
        self.anima_components["transformer"] = model
        if converted or runtime_int8_requested(transformer_quantization) \
                or getattr(self, "_runtime_int8_converted", False) \
                or getattr(self, "_runtime_int8_from_checkpoint", False) \
                or getattr(self, "_runtime_int8_partial", False):
            # ``_runtime_int8_partial`` too: a half-converted transformer already
            # carries Int8Linear modules, which would make _anima_quantize_fp8
            # refuse with its CHECKPOINT-provenance message (false here) and
            # leave the still-bf16 remainder unquantized anyway.
            # apply_runtime_int8_quantization has already warned accurately.
            # ``_runtime_int8_from_checkpoint`` is listed for the same reason it
            # used to be covered by ``_runtime_int8_converted`` (which the
            # already-quantized-checkpoint branch no longer sets): the Linears
            # are Int8Linear/Fp8Linear, so _anima_quantize_fp8 has nothing to do.
            return None
        return transformer_quantization

    def _anima_stage_transformer(self, device: str, transformer_quantization,
                                 params: Dict[str, Any], progress_callback=None):
        """Place the Anima transformer on GPU for the denoise loop.

        Default (block swap disabled): full GPU move via _anima_move, byte-identical
        to the pre-block-swap behaviour. With block swap enabled, the transformer's
        blocks are streamed per forward by a per-model offloader instead of being
        fully resident. Returns the (possibly reassigned) transformer.

        ``unet_quantization="int8"`` is handled first, in place and once per model
        load; every other quantization value falls through to the pre-existing
        FP8 paths unchanged.
        """
        transformer_quantization = self._anima_runtime_int8(
            transformer_quantization, progress_callback=progress_callback)
        enable_block_swap = bool(params.get("enable_block_swap", False))
        blocks_to_swap = int(params.get("blocks_to_swap", 20))
        use_pinned_memory = bool(params.get("use_pinned_memory", False))
        h2d_only = bool(params.get("block_swap_h2d_only", False))
        ring_size = int(params.get("block_swap_ring_size", 2))

        self._anima_offloader = None
        if not (enable_block_swap and blocks_to_swap > 0):
            # Default full-GPU placement (with optional FP8 quantization).
            return self._anima_move("transformer", device, transformer_quantization)

        # Block-swap mode: apply optional FP8 quantization on CPU first (produces
        # plain tensors the offloader can stream), then attach the offloader. The
        # offloader — not a full .to(device) — handles placing the swappable blocks.
        transformer = self.anima_components["transformer"]
        if transformer_quantization not in (None, "", "none"):
            from core.vram_optimization import _anima_quantize_fp8
            try:
                if next(transformer.parameters()).device.type != "cpu":
                    transformer.to("cpu")
                transformer = _anima_quantize_fp8(transformer, transformer_quantization, "Transformer")
                self.anima_components["transformer"] = transformer
            except Exception as e:
                print(f"[Anima] Warning: block-swap FP8 quantization failed: {e}")
                transformer = self.anima_components["transformer"]

        num_blocks = len(transformer.blocks)
        clamped = max(0, min(blocks_to_swap, num_blocks - 1))
        print(f"[Anima] Block swap enabled: {clamped}/{num_blocks} blocks "
              f"(pinned_memory={use_pinned_memory}, h2d_only={h2d_only}, ring_size={ring_size})")
        self._anima_offloader = self._anima_setup_block_swap(
            transformer, clamped, use_pinned_memory, device,
            h2d_only=h2d_only, ring_size=ring_size,
        )
        return transformer

    def _anima_unstage_transformer(self):
        """Tear down any block-swap offloader, then return the transformer to CPU."""
        offloader = getattr(self, "_anima_offloader", None)
        transformer = (self.anima_components or {}).get("transformer")
        if transformer is not None and hasattr(transformer, "_block_offloader"):
            try:
                delattr(transformer, "_block_offloader")
            except Exception:
                pass
        if offloader is not None:
            cleanup = getattr(offloader, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
        self._anima_offloader = None
        self._anima_move("transformer", "cpu")

    def _anima_set_attention_backend(self, params: Dict[str, Any]) -> None:
        """Route Anima's attention kernel through the unified conduit.

        Mirrors ``pipeline_backends/zimage.py``: the selected backend
        (``attention_type`` from params, else the global setting) is pushed onto
        the ``anima_attention`` module-global that ``anima_attention.attention()``
        reads. Both the image self-attention (``anima_models.Attention.forward``)
        and the cross-attention (patched by ``nag_anima`` / ``negpip_anima``)
        call that one primitive, so NAG/NegPip honor the selection too.

        Anima has no dedicated sage kernel; a ``sage`` request is handled by the
        conduit (sage->native guard) without crashing.
        """
        from core.attention import normalize_backend
        from core.models.anima import anima_attention

        attention_type = params.get("attention_type", settings.attention_type)
        backend = normalize_backend(attention_type)
        if backend != getattr(self, "current_attention_type", None):
            print(f"[Anima] Switching attention backend: "
                  f"{getattr(self, 'current_attention_type', None)} -> {backend}")
            self.current_attention_type = backend
        anima_attention.set_attention_backend(backend)

    def _anima_style_triple(self, style_dict: Dict[str, Any], width: int, height: int, device,
                             seed, ref_index: int = 0):
        """Build a single (StyleTransferConfig, ref_x0, eps_ref) triple from one
        style_transfer dict.

        ``axes_dims`` is intentionally left UNSET (``None``): Anima's 3D video
        RoPE (``apply_rotary_pos_emb(..., interleaved=False)``) uses the
        "rotate-half" convention, not the per-axis interleave-real layout that
        ``reference_style.frequency_scale_vector`` assumes (Krea2/FLUX-style).
        Deriving a correct per-axis frequency-suppression curve for Anima's
        RoPE layout is a separate adaptation; until then the attention hook
        (``anima_models.Attention.forward`` / ``StyleContext.collect_block_refs``)
        uses an all-ones frequency vector instead of calling
        ``cfg.get_freq_scale_vector`` -- this only disables the
        RoPE-frequency-content suppression (a quality knob), NOT the
        ``ref_k_strength`` scale or AdaIN alignment, which still apply in full.

        ``width``/``height`` must be the TARGET generation's already-snapped
        resolution (not the style image's own size) so the encoded reference
        latent aligns token-for-token with the target latent grid at every
        denoise step.

        ``ref_index`` decorrelates the fixed re-noising noise tensor across
        multiple simultaneous references (each ref would otherwise draw the
        EXACT same noise from ``prepare_style_reference``'s ``seed+991``
        offset, since that offset does not depend on which reference is being
        prepared). ``ref_index=0`` (the default, used by the single-ref path)
        reproduces the pre-multi-ref ``seed+991`` offset exactly.
        """
        from core.inference.reference_style import style_config_from_dict
        from core.models.anima.anima_pipeline_ops import prepare_style_reference

        cfg = style_config_from_dict(style_dict)

        ref_seed = seed if seed is None or seed < 0 else int(seed) + ref_index
        ref_x0, eps_ref = prepare_style_reference(
            self.anima_components["vae"], style_dict["image"], height, width,
            device=device, dtype=torch.bfloat16, seed=ref_seed,
        )
        return cfg, ref_x0, eps_ref

    def _anima_style_config(self, params: Dict[str, Any], width: int, height: int, device):
        """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None,
        None)`` when no style reference is attached. Single-reference path,
        BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_anima_style_triple`` with ``ref_index=0``, which reproduces the
        original ``seed+991`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        seed = params.get("seed", -1)
        return self._anima_style_triple(style_dict, width, height, device, seed, ref_index=0)

    def _anima_style_configs(self, params: Dict[str, Any], width: int, height: int, device):
        """Build the full style-transfer configuration for Anima generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple, exactly as ``_anima_style_config``
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
            seed = params.get("seed", -1)
            combine_mode = str(params.get("style_combine_mode", "stack") or "stack")
            refs = []
            for idx, style_dict in enumerate(style_list):
                if not style_dict or not style_dict.get("image"):
                    continue
                refs.append(self._anima_style_triple(style_dict, width, height, device, seed, ref_index=idx))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._anima_style_config(params, width, height, device)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    def _generate_txt2img_anima(self, params: Dict[str, Any],
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded. Please load an Anima model first.")

        print("[Anima] Starting txt2img generation")
        self._anima_set_attention_backend(params)
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_txt2img, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        # Seed
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        ancestral_seed = params.get("ancestral_seed", -1)
        if ancestral_seed == -1:
            ancestral_seed = random.randint(0, 2147483647)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        height = int(params.get("height", 512))
        width = int(params.get("width", 512))
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))

        # Optional FP8 quantization (matches FLUX.2 / Z-Image pattern)
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Snap to patch_spatial * vae_scale_factor
        snap = transformer.patch_spatial * 8
        height = (height // snap) * snap
        width = (width // snap) * snap

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        # If a resident set exists from a previous generation but is no longer valid
        # for THIS request's model_key (checkpoint/LoRA/quantization/dtype changed),
        # force a full offload before staging anything.
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._anima_move("text_encoder", "cpu"),
                self._anima_move("transformer", "cpu"),
                self._anima_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not cpu_text_encoding:
                _kh_total_bytes += component_nbytes(self.anima_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.anima_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.anima_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not cpu_text_encoding
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            # Stage 1: text encoding
            if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            # NegPip auto-activates on any negative emphasis weight. When active
            # we encode CLEAN embeddings (skip_emphasis) so the signed V scaling
            # carries all the emphasis; otherwise the default emphasis path runs.
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            # Offload text encoder after encoding (unless kept hot for the next
            # queued generation on the same model_key).
            if _kh_keep_te:
                mark_resident(self, "text_encoder", _kh_model_key)
            elif not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                # Move CPU-encoded embeddings to GPU for denoising
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: denoising
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.anima_components["transformer"]
                self._anima_offloader = None
            else:
                transformer = self._anima_stage_transformer(device, transformer_quantization, params,
                                                            progress_callback=progress_callback)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default; folds
            # into NAG on the cond pass and wraps the raw transformer on uncond.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            # Restore only the NegPip wrappers we actually created (cond_driver
            # differs from base_cond only when a NegPip cond wrapper was built).
            negpip_cond = cond_driver if cond_driver is not base_cond else None

            # Training-free reference-style transfer. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None,
            # None, "stack"), no-op below). ``style_refs`` is populated (and
            # style_cfg/style_ref_x0/style_eps_ref left None) ONLY when
            # ``params["style_transfers"]`` carries 2+ references -- a single
            # reference (via either key) always resolves through the
            # style_cfg/style_ref_x0/style_eps_ref triple, so that code path
            # (both here and inside sample_*) is untouched.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                if not is_resident(self, "vae", _kh_model_key):
                    self._anima_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._anima_style_configs(params, width, height, device)
                self._anima_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = sample_txt2img(
                    transformer=transformer, scheduler=scheduler,
                    cond_embeds=cond, uncond_embeds=uncond,
                    height=height, width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            # Optimistically mark/offload the transformer here; the success flag is
            # NOT set until AFTER the VAE decode below, so a decode failure routes to
            # the finally's exception branch (clear_resident + full offload), undoing
            # this mark. (Decode is a separate GPU op here, unlike the SDXL reference
            # where decode is inside the sampling call.)
            if applied_lora_count:
                self._unload_lora_anima()
            if _kh_keep_transformer:
                mark_resident(self, "transformer", _kh_model_key)
            else:
                self._anima_unstage_transformer()
            del cond, uncond
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: VAE decode
            if not is_resident(self, "vae", _kh_model_key):
                self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents, color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded: only now is the generation a
            # success for keep-hot purposes.
            _kh_gen_succeeded = True

            print("[Anima] txt2img completed")
            return images[0], seed, ancestral_seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Strip any leftover block-swap offloader (e.g. if setup raised mid-way),
            # then ensure all components are back on CPU -- EXCEPT components kept
            # hot on a SUCCESSFUL generation. On an exception, ALWAYS force a full
            # offload + clear residency (never trust the pipeline state after an
            # error going into the next generation).
            _t = (self.anima_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._anima_offloader = None
            # A sampling/decode failure must not leave LoRA wrappers on the DiT;
            # no-op when the success path already unwrapped.
            try:
                self._unload_lora_anima()
            except Exception:
                pass
            if not _kh_gen_succeeded:
                clear_resident(self)
                for _comp in ("text_encoder", "transformer", "vae"):
                    try:
                        self._anima_move(_comp, "cpu")
                    except Exception:
                        pass
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    try:
                        self._anima_move("text_encoder", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    try:
                        self._anima_move("transformer", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    try:
                        self._anima_move("vae", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "vae")

    def _generate_img2img_anima(self, params: Dict[str, Any], init_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting img2img generation")
        self._anima_set_attention_backend(params)
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_img2img, vae_encode_image, vae_decode_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.7))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        # Resize init image to match desired width/height (snapped)
        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._anima_move("text_encoder", "cpu"),
                self._anima_move("transformer", "cpu"),
                self._anima_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not cpu_text_encoding:
                _kh_total_bytes += component_nbytes(self.anima_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.anima_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.anima_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not cpu_text_encoding
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            # Encode init image. This is the generation's first use of the VAE, so
            # this is the cross-generation entry point for it (the later decode-stage
            # move below is an intra-generation re-stage, unaffected by keep-hot).
            if not is_resident(self, "vae", _kh_model_key):
                self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Text encoding
            if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            if _kh_keep_te:
                mark_resident(self, "text_encoder", _kh_model_key)
            elif not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.anima_components["transformer"]
                self._anima_offloader = None
            else:
                transformer = self._anima_stage_transformer(device, transformer_quantization, params,
                                                            progress_callback=progress_callback)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            negpip_cond = cond_driver if cond_driver is not base_cond else None

            # Training-free reference-style transfer. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None,
            # None, "stack"), no-op below). ``style_refs`` is populated (and
            # style_cfg/style_ref_x0/style_eps_ref left None) ONLY when
            # ``params["style_transfers"]`` carries 2+ references -- a single
            # reference (via either key) always resolves through the
            # style_cfg/style_ref_x0/style_eps_ref triple, so that code path
            # (both here and inside sample_*) is untouched.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                if not is_resident(self, "vae", _kh_model_key):
                    self._anima_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._anima_style_configs(params, width, height, device)
                self._anima_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = sample_img2img(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents,
                    cond_embeds=cond, uncond_embeds=uncond,
                    num_inference_steps=num_inference_steps,
                    denoising_strength=denoising_strength,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            # Optimistically mark/offload the transformer; _kh_gen_succeeded is set
            # only AFTER decode below, so a decode failure routes to the finally
            # exception branch (clear_resident + full offload), undoing this mark.
            if applied_lora_count:
                self._unload_lora_anima()
            if _kh_keep_transformer:
                mark_resident(self, "transformer", _kh_model_key)
            else:
                self._anima_unstage_transformer()
            del cond, uncond, init_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            if not is_resident(self, "vae", _kh_model_key):
                self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents, color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded.
            _kh_gen_succeeded = True

            print("[Anima] img2img completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Strip any leftover block-swap offloader (e.g. if setup raised mid-way),
            # then ensure all components are back on CPU -- EXCEPT components kept
            # hot on a SUCCESSFUL generation. On an exception, ALWAYS force a full
            # offload + clear residency.
            _t = (self.anima_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._anima_offloader = None
            # A sampling/decode failure must not leave LoRA wrappers on the DiT;
            # no-op when the success path already unwrapped.
            try:
                self._unload_lora_anima()
            except Exception:
                pass
            if not _kh_gen_succeeded:
                clear_resident(self)
                for _comp in ("text_encoder", "transformer", "vae"):
                    try:
                        self._anima_move(_comp, "cpu")
                    except Exception:
                        pass
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    try:
                        self._anima_move("text_encoder", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    try:
                        self._anima_move("transformer", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    try:
                        self._anima_move("vae", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "vae")

    def _generate_inpaint_anima(self, params: Dict[str, Any],
                                 init_image: Image.Image, mask_image: Image.Image,
                                 progress_callback=None, step_callback=None
                                 ) -> tuple[Image.Image, int]:
        if not self.anima_components:
            raise RuntimeError("Anima components not loaded.")

        print("[Anima] Starting inpaint generation")
        self._anima_set_attention_backend(params)
        from core.models.anima.anima_pipeline_ops import (
            encode_prompt, sample_inpaint, vae_encode_image, vae_decode_latents,
            make_mask_latents,
        )

        device = self.device
        compute_dtype = torch.bfloat16

        transformer = self.anima_components["transformer"]
        text_encoder = self.anima_components["text_encoder"]
        qwen3_tokenizer = self.anima_components["tokenizer"]
        t5_tokenizer = self.anima_components["t5_tokenizer"]
        vae = self.anima_components["vae"]
        scheduler = self.anima_components["scheduler"]

        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        num_inference_steps = int(params.get("steps", 28))
        guidance_scale = float(params.get("cfg_scale", 4.0))
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        transformer_quantization = params.get("unet_quantization")
        text_encoder_quantization = params.get("text_encoder_quantization")

        width = int(params.get("width", init_image.width))
        height = int(params.get("height", init_image.height))
        snap = transformer.patch_spatial * 8
        width = (width // snap) * snap
        height = (height // snap) * snap
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)

        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        cpu_text_encoding = params.get("cpu_text_encoding", False)
        enc_device = "cpu" if cpu_text_encoding else device

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 20)) > 0
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._anima_move("text_encoder", "cpu"),
                self._anima_move("transformer", "cpu"),
                self._anima_move("vae", "cpu"),
            ),
        )
        _kh_total_bytes = 0
        if _kh_requested:
            if not cpu_text_encoding:
                _kh_total_bytes += component_nbytes(self.anima_components.get("text_encoder"))
            if not (_kh_is_block_swapped or _kh_has_loras):
                _kh_total_bytes += component_nbytes(self.anima_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.anima_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok and not cpu_text_encoding
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_is_block_swapped and not _kh_has_loras
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

        try:
            # Encode init image. This is the generation's first use of the VAE, so
            # this is the cross-generation entry point for it (the later decode-stage
            # move below is an intra-generation re-stage, unaffected by keep-hot).
            if not is_resident(self, "vae", _kh_model_key):
                self._anima_move("vae", device)
            init_latents = vae_encode_image(vae, init_image, device, compute_dtype)
            self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            mask_latents = make_mask_latents(
                mask_image, init_latents.shape[-2], init_latents.shape[-1],
                device, compute_dtype,
            )

            # Text encoding
            if not cpu_text_encoding and not is_resident(self, "text_encoder", _kh_model_key):
                text_encoder = self._anima_move("text_encoder", device, text_encoder_quantization)
            use_negpip = self._anima_negpip_active(params)
            cond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                  prompt, device=enc_device, dtype=compute_dtype,
                                  skip_emphasis=use_negpip)
            uncond = None
            if guidance_scale > 1.0:
                uncond = encode_prompt(text_encoder, qwen3_tokenizer, t5_tokenizer,
                                       negative_prompt, device=enc_device, dtype=compute_dtype,
                                       skip_emphasis=use_negpip)
            nag_neg = self._anima_encode_nag_neg(
                params, encode_prompt, text_encoder, qwen3_tokenizer, t5_tokenizer,
                enc_device, compute_dtype,
            )
            if _kh_keep_te:
                mark_resident(self, "text_encoder", _kh_model_key)
            elif not cpu_text_encoding:
                self._anima_move("text_encoder", "cpu")
            if cpu_text_encoding:
                def _embeds_to_gpu(d):
                    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
                cond = _embeds_to_gpu(cond)
                if uncond is not None:
                    uncond = _embeds_to_gpu(uncond)
                if nag_neg is not None:
                    nag_neg = _embeds_to_gpu(nag_neg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Denoise
            if is_resident(self, "transformer", _kh_model_key):
                transformer = self.anima_components["transformer"]
                self._anima_offloader = None
            else:
                transformer = self._anima_stage_transformer(device, transformer_quantization, params,
                                                            progress_callback=progress_callback)

            # Apply user-supplied LoRAs after the transformer is on GPU (and
            # after any optional quantization). LoRA wrappers point at the
            # current Linear modules; they survive .to() but not deepcopy,
            # so the order must be: quantize -> wrap LoRA -> sample -> unwrap.
            lora_configs = params.get("loras") or []
            applied_lora_count = self._load_lora_anima(lora_configs) if lora_configs else 0
            transformer = self.anima_components["transformer"]

            # Optional NAG (Normalized Attention Guidance). OFF by default.
            nag_wrapper = self._anima_build_nag_wrapper(params, transformer, nag_neg)
            # Optional NegPip (signed per-token V scale). OFF by default.
            base_cond = nag_wrapper if nag_wrapper is not None else transformer
            cond_driver = base_cond
            negpip_uncond = None
            if use_negpip:
                cond_driver, negpip_uncond = self._anima_build_negpip(
                    params, base_cond, transformer, cond, uncond,
                    t5_tokenizer, device, compute_dtype,
                )
            negpip_cond = cond_driver if cond_driver is not base_cond else None

            # Training-free reference-style transfer. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None,
            # None, "stack"), no-op below). ``style_refs`` is populated (and
            # style_cfg/style_ref_x0/style_eps_ref left None) ONLY when
            # ``params["style_transfers"]`` carries 2+ references -- a single
            # reference (via either key) always resolves through the
            # style_cfg/style_ref_x0/style_eps_ref triple, so that code path
            # (both here and inside sample_*) is untouched.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                if not is_resident(self, "vae", _kh_model_key):
                    self._anima_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._anima_style_configs(params, width, height, device)
                self._anima_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = sample_inpaint(
                    transformer=transformer, scheduler=scheduler,
                    init_latents=init_latents, mask_latents=mask_latents,
                    cond_embeds=cond, uncond_embeds=uncond,
                    num_inference_steps=num_inference_steps,
                    denoising_strength=denoising_strength,
                    guidance_scale=guidance_scale,
                    generator=generator, device=device, dtype=compute_dtype,
                    step_callback=(progress_callback or step_callback),
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                    advanced_cfg=self._anima_advanced_cfg(params),
                    spectrum_params=params,
                    nag_transformer=cond_driver if cond_driver is not transformer else None,
                    negpip_uncond_transformer=negpip_uncond,
                )
            finally:
                for w in (negpip_uncond, negpip_cond, nag_wrapper):
                    if w is not None and hasattr(w, "restore"):
                        w.restore()
            # Optimistically mark/offload the transformer; _kh_gen_succeeded is set
            # only AFTER decode below, so a decode failure routes to the finally
            # exception branch (clear_resident + full offload), undoing this mark.
            if applied_lora_count:
                self._unload_lora_anima()
            if _kh_keep_transformer:
                mark_resident(self, "transformer", _kh_model_key)
            else:
                self._anima_unstage_transformer()
            del cond, uncond, init_latents, mask_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Decode
            if not is_resident(self, "vae", _kh_model_key):
                self._anima_move("vae", device)
            self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
            images = vae_decode_latents(vae, latents, color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            del latents
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._anima_move("vae", "cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # All GPU work (incl. decode) succeeded.
            _kh_gen_succeeded = True

            print("[Anima] inpaint completed")
            return images[0], seed
        except Exception as e:
            print(f"[Anima] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            # Strip any leftover block-swap offloader (e.g. if setup raised mid-way),
            # then ensure all components are back on CPU -- EXCEPT components kept
            # hot on a SUCCESSFUL generation. On an exception, ALWAYS force a full
            # offload + clear residency.
            _t = (self.anima_components or {}).get("transformer")
            if _t is not None and hasattr(_t, "_block_offloader"):
                try:
                    delattr(_t, "_block_offloader")
                except Exception:
                    pass
            self._anima_offloader = None
            # A sampling/decode failure must not leave LoRA wrappers on the DiT;
            # no-op when the success path already unwrapped.
            try:
                self._unload_lora_anima()
            except Exception:
                pass
            if not _kh_gen_succeeded:
                clear_resident(self)
                for _comp in ("text_encoder", "transformer", "vae"):
                    try:
                        self._anima_move(_comp, "cpu")
                    except Exception:
                        pass
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    try:
                        self._anima_move("text_encoder", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    try:
                        self._anima_move("transformer", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    try:
                        self._anima_move("vae", "cpu")
                    except Exception:
                        pass
                    discard_resident(self, "vae")
