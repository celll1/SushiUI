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


def _set_flux2_nag_negpip_backend(diffusers_backend: str) -> None:
    """Set ``_attention_backend`` on the 6 vendored NAG/NegPip Flux.2 processor classes.

    The vendored NAG/NegPip processors funnel every attention call through their
    ``_sdpa`` helper (``nag_flux2._sdpa``), which forwards ``self._attention_backend``
    to diffusers' ``dispatch_attention_fn``. All six classes read the per-class
    attribute, so setting it once on each class covers every instance the wrappers
    install. ``dispatch_attention_fn`` normalizes the string itself
    (``AttentionBackendName(backend)``), so the diffusers backend string is exactly
    what each ``_sdpa`` needs.

    Imported lazily so a FLUX.2 run that never touches NAG/NegPip pays nothing, and so
    an import hiccup inside those modules cannot break the default (registry) path.
    """
    from core.inference.nag_flux2 import (
        NAGFlux2AttnProcessor,
        NAGFlux2ParallelSelfAttnProcessor,
    )
    from core.inference.negpip_flux2 import (
        NegPipFlux2AttnProcessor,
        NegPipFlux2ParallelSelfAttnProcessor,
        NegPipNAGFlux2AttnProcessor,
        NegPipNAGFlux2ParallelSelfAttnProcessor,
    )
    for cls in (
        NAGFlux2AttnProcessor,
        NAGFlux2ParallelSelfAttnProcessor,
        NegPipFlux2AttnProcessor,
        NegPipFlux2ParallelSelfAttnProcessor,
        NegPipNAGFlux2AttnProcessor,
        NegPipNAGFlux2ParallelSelfAttnProcessor,
    ):
        cls._attention_backend = diffusers_backend


def _install_flux2_conduit_processors(transformer, canonical_backend, mode) -> int:
    """Install ConduitFlux2* processors on the NON-KV default attention modules.

    Gated on the CURRENT processor type so the reference-image KV-cache processors
    (Flux2KVAttnProcessor / Flux2KVParallelSelfAttnProcessor) are NOT clobbered --
    they stay on the diffusers registry (run native after the caller resets the
    diffusers global). Returns the count of modules migrated.
    """
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Attention,
        Flux2AttnProcessor,
        Flux2ParallelSelfAttention,
        Flux2ParallelSelfAttnProcessor,
    )
    from core.inference.conduit_flux2 import (
        ConduitFlux2AttnProcessor,
        ConduitFlux2ParallelSelfAttnProcessor,
    )
    migrated = 0
    for module in transformer.modules():
        proc = getattr(module, "processor", None)
        if isinstance(module, Flux2Attention) and type(proc) is Flux2AttnProcessor:
            module.set_processor(ConduitFlux2AttnProcessor(canonical_backend, mode))
            migrated += 1
        elif isinstance(module, Flux2ParallelSelfAttention) and type(proc) is Flux2ParallelSelfAttnProcessor:
            module.set_processor(ConduitFlux2ParallelSelfAttnProcessor(canonical_backend, mode))
            migrated += 1
    return migrated


def set_flux2_attention_backend(transformer, backend, attention_impl="diffusers") -> str:
    """Honor the selected attention backend for a FLUX.2 inference run.

    FLUX.2 uses diffusers' OWN attention registry (``dispatch_attention_fn``), which we
    keep intact (NOT rerouted through SushiUI's unified conduit) so diffusers' context-
    parallel + varlen machinery keeps working. Instead we drive that registry from the
    SAME canonical backend string the rest of SushiUI uses, setting two things from ONE
    source string:

      1. The DEFAULT processors (``Flux2AttnProcessor`` / ``Flux2ParallelSelfAttnProcessor``)
         via ``transformer.set_attention_backend`` -- this also sets diffusers' global
         active backend so the registry propagates to any processor left at ``None``.
      2. The vendored NAG / NegPip processor classes, whose ``_sdpa`` choke point reads
         the per-class ``_attention_backend`` attribute (see
         :func:`_set_flux2_nag_negpip_backend`).

    ``set_attention_backend`` is wrapped in try/except: some diffusers builds reject
    ``flash``/``sage`` (missing flash-attn / sageattention, unsupported head_dim, etc.),
    in which case we fall back to ``native`` for BOTH the default path and the NAG/NegPip
    processors. Selecting ``normal``/``none``/``native`` (or leaving it unset) maps to
    ``native`` and is byte-identical to the pre-wiring behavior (attention_type was
    previously ignored and FLUX.2 always ran native).

    Returns the diffusers backend string actually applied ("native" on any fallback).
    """
    from core.attention import normalize_backend, to_diffusers_backend

    canonical = normalize_backend(backend)

    # attention_impl='conduit' (new default): route the DEFAULT + NAG/NegPip attention
    # through SushiUI's unified conduit (enables conduit-only backends such as tq on FLUX.2).
    # attention_impl='diffusers': byte-identical legacy path via diffusers' registry.
    if attention_impl == "conduit":
        from core.attention import AttentionMode
        from core.inference.nag_flux2 import set_flux2_nag_negpip_conduit
        # Reset diffusers' global active backend to native so any residual KV-cache
        # (ref-image) processors left on the diffusers registry run deterministically.
        try:
            transformer.set_attention_backend("native")
        except Exception:
            pass
        migrated = _install_flux2_conduit_processors(transformer, canonical, AttentionMode.INFERENCE)
        # NAG/NegPip choke point (_sdpa) -> conduit, reading the CANONICAL string.
        _set_flux2_nag_negpip_backend(canonical)
        set_flux2_nag_negpip_conduit(True)
        print(f"[FLUX.2] Attention impl: conduit backend='{canonical}' "
              f"(requested '{backend}'); {migrated} default attn modules migrated")
        return f"conduit:{canonical}"

    # attention_impl='diffusers' (legacy): drive diffusers' own registry.
    from core.inference.nag_flux2 import set_flux2_nag_negpip_conduit
    set_flux2_nag_negpip_conduit(False)
    diffusers_backend = to_diffusers_backend(canonical)  # 'native' | 'flash' | 'sage'

    # (1) Default processors + diffusers' global active backend.
    applied = diffusers_backend
    try:
        transformer.set_attention_backend(diffusers_backend)
    except Exception as e:
        if diffusers_backend != "native":
            print(f"[FLUX.2] Attention backend '{diffusers_backend}' unavailable "
                  f"({e}); falling back to native")
        applied = "native"
        try:
            transformer.set_attention_backend("native")
        except Exception:
            # Even native rejected (very old diffusers lacking the registry API):
            # leave diffusers' default in place -- dispatch stays native.
            pass

    # (2) NAG / NegPip processor class-attrs (choke point: nag_flux2._sdpa).
    _set_flux2_nag_negpip_backend(applied)

    print(f"[FLUX.2] Attention backend: {applied} "
          f"(requested '{backend}' -> canonical '{canonical}')")
    return applied


class Flux2Mixin:
    """Flux2Mixin: flux2 backend methods extracted verbatim from pipeline.py."""

    def _load_lora_flux2(self, lora_configs: List[Dict]):
        """Load LoRAs for FLUX.2 Transformer

        Args:
            lora_configs: List of LoRA configurations

        Note:
            FLUX.2 uses component-based architecture (not pipeline-based).
            LoRAs wrap original linear layers (forward-time addition, not weight merging).
            This allows LoRAs to be unloaded by restoring original modules.
            Based on training implementation in flux2_adapter.py

            FLUX.2 has two block types:
            1. Dual stream blocks: Flux2Attention (to_q, to_k, to_v, to_out[0], add_q_proj, add_k_proj, add_v_proj, to_add_out)
            2. Single stream blocks: Flux2ParallelSelfAttention (to_qkv_mlp_proj, to_out)
        """
        if not lora_configs:
            return

        if not self.flux2_components:
            print("[FLUX.2 LoRA] WARNING: FLUX.2 components not loaded")
            return

        transformer = self.flux2_components["transformer"]

        # Store original modules for unloading (first time only)
        if not hasattr(self, '_flux2_lora_original_modules'):
            self._flux2_lora_original_modules = {}
            self._flux2_lora_wrapped_modules = set()

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[FLUX.2 LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            lora_strength = lora_config.get("strength", 1.0)
            layer_weights = lora_config.get("unet_layer_weights", {})

            # Resolve path using LoRAManager
            resolved_path = lora_manager._resolve_lora_path(lora_path)

            if resolved_path is None:
                print(f"[FLUX.2 LoRA] WARNING: LoRA file not found: {lora_path}")
                print(f"[FLUX.2 LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[FLUX.2 LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                continue

            print(f"[FLUX.2 LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")
            if layer_weights:
                print(f"[FLUX.2 LoRA] Layer weights: {layer_weights}")

            # Load LoRA weights
            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[FLUX.2 LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # Apply LoRA to transformer modules
                applied_count = 0

                # Debug: Print first few LoRA keys
                lora_keys_sample = list(lora_state_dict.keys())[:5]
                print(f"[FLUX.2 LoRA] Sample LoRA keys: {lora_keys_sample}")

                # Debug: Print module class names found
                module_classes_found = set()
                for name, module in transformer.named_modules():
                    module_classes_found.add(module.__class__.__name__)
                print(f"[FLUX.2 LoRA] Module classes in transformer: {module_classes_found}")

                for name, module in transformer.named_modules():
                    # Flux2Attention (dual stream blocks)
                    if module.__class__.__name__ == "Flux2Attention":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        # Standard QKV projections
                        for attr_name in ["to_q", "to_k", "to_v"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    # Build LoRA key using training adapter's naming convention
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                        # to_out (ModuleList) - uses same effective_strength computed above
                        if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.ModuleList):
                            if len(module.to_out) > 0 and isinstance(module.to_out[0], torch.nn.Linear):
                                lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out_0"
                                lora_down_key = f"{lora_name}.lora_down.weight"
                                lora_up_key = f"{lora_name}.lora_up.weight"

                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]
                                    lora_alpha_key = f"{lora_name}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    module_key = f"{name}.to_out.0"
                                    wrapped = self._wrap_with_lora_flux2(
                                        module.to_out, 0, module.to_out[0],
                                        lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                    )
                                    if wrapped:
                                        applied_count += 1

                        # Additional projections for encoder cross attention - uses same effective_strength
                        for attr_name in ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                    # Flux2ParallelSelfAttention (single stream blocks)
                    elif module.__class__.__name__ == "Flux2ParallelSelfAttention":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        # Fused QKV + MLP projection
                        if hasattr(module, "to_qkv_mlp_proj"):
                            original_linear = module.to_qkv_mlp_proj
                            if isinstance(original_linear, torch.nn.Linear):
                                lora_name = f"lora_transformer_{name.replace('.', '_')}_to_qkv_mlp_proj"
                                lora_down_key = f"{lora_name}.lora_down.weight"
                                lora_up_key = f"{lora_name}.lora_up.weight"

                                if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                    lora_down_weight = lora_state_dict[lora_down_key]
                                    lora_up_weight = lora_state_dict[lora_up_key]
                                    lora_alpha_key = f"{lora_name}.alpha"
                                    lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                    module_key = f"{name}.to_qkv_mlp_proj"
                                    wrapped = self._wrap_with_lora_flux2(
                                        module, "to_qkv_mlp_proj", original_linear,
                                        lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                    )
                                    if wrapped:
                                        applied_count += 1

                        # Output projection (fused attention + MLP) - uses same effective_strength
                        if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.Linear):
                            lora_name = f"lora_transformer_{name.replace('.', '_')}_to_out"
                            lora_down_key = f"{lora_name}.lora_down.weight"
                            lora_up_key = f"{lora_name}.lora_up.weight"

                            if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                lora_down_weight = lora_state_dict[lora_down_key]
                                lora_up_weight = lora_state_dict[lora_up_key]
                                lora_alpha_key = f"{lora_name}.alpha"
                                lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                module_key = f"{name}.to_out"
                                wrapped = self._wrap_with_lora_flux2(
                                    module, "to_out", module.to_out,
                                    lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                )
                                if wrapped:
                                    applied_count += 1

                    # Flux2FeedForward (dual stream blocks)
                    elif module.__class__.__name__ == "Flux2FeedForward":
                        # Get block name for layer-wise weight lookup
                        block_name = self._get_flux2_block_name(name)
                        block_weight = layer_weights.get(block_name, 1.0)
                        effective_strength = lora_strength * block_weight

                        for attr_name in ["linear_in", "linear_out"]:
                            if hasattr(module, attr_name):
                                original_linear = getattr(module, attr_name)
                                if isinstance(original_linear, torch.nn.Linear):
                                    lora_name = f"lora_transformer_{name.replace('.', '_')}_{attr_name}"
                                    lora_down_key = f"{lora_name}.lora_down.weight"
                                    lora_up_key = f"{lora_name}.lora_up.weight"

                                    if lora_down_key in lora_state_dict and lora_up_key in lora_state_dict:
                                        lora_down_weight = lora_state_dict[lora_down_key]
                                        lora_up_weight = lora_state_dict[lora_up_key]
                                        lora_alpha_key = f"{lora_name}.alpha"
                                        lora_alpha = lora_state_dict.get(lora_alpha_key, None)

                                        module_key = f"{name}.{attr_name}"
                                        wrapped = self._wrap_with_lora_flux2(
                                            module, attr_name, original_linear,
                                            lora_down_weight, lora_up_weight, effective_strength, lora_alpha, module_key
                                        )
                                        if wrapped:
                                            applied_count += 1

                print(f"[FLUX.2 LoRA] Applied LoRA to {applied_count} modules")

            except Exception as e:
                print(f"[FLUX.2 LoRA] ERROR: Failed to load LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()

    def _get_flux2_block_name(self, module_name: str) -> str:
        """Get the block name (DUAL{XX} or SING{XX}) from module name for layer-wise weight lookup

        Args:
            module_name: Module name like 'transformer_blocks.0.attn' or 'single_transformer_blocks.5.attn'

        Returns:
            Block name like 'DUAL00', 'SING05', or 'BASE' if no match
        """
        import re

        # Dual stream blocks: transformer_blocks.X.* (but not single_transformer_blocks)
        if 'transformer_blocks' in module_name and 'single_transformer_blocks' not in module_name:
            match = re.search(r'transformer_blocks\.(\d+)', module_name)
            if match:
                block_num = int(match.group(1))
                return f"DUAL{block_num:02d}"

        # Single stream blocks: single_transformer_blocks.X.*
        match = re.search(r'single_transformer_blocks\.(\d+)', module_name)
        if match:
            block_num = int(match.group(1))
            return f"SING{block_num:02d}"

        return "BASE"

    def _wrap_with_lora_flux2(self, parent_module, attr_name, original_linear, lora_down_weight, lora_up_weight, strength, alpha, module_key):
        """Wrap a linear layer with LoRA for FLUX.2

        Args:
            parent_module: Parent module containing the linear layer
            attr_name: Attribute name or index (for ModuleList)
            original_linear: Original linear layer
            lora_down_weight: LoRA down projection weight
            lora_up_weight: LoRA up projection weight
            strength: LoRA strength multiplier (already adjusted with layer weight)
            alpha: LoRA alpha parameter
            module_key: Unique key for tracking

        Returns:
            True if wrapped successfully, False otherwise
        """
        # Import LoRALinearLayer from training adapters (model-agnostic wrapper class)
        from core.training.adapters.sd15_adapter import LoRALinearLayer

        # Handle already wrapped modules
        if isinstance(original_linear, LoRALinearLayer):
            true_original = original_linear.original_module
        else:
            true_original = original_linear

        # Save original module (first time only)
        if module_key not in self._flux2_lora_original_modules:
            self._flux2_lora_original_modules[module_key] = true_original

        # Compute rank and alpha value
        rank = lora_down_weight.shape[0]
        alpha_value = alpha.item() if alpha is not None else rank

        # Create LoRA wrapper
        # lora_name is required parameter, use module_key for identification
        lora_wrapper = LoRALinearLayer(
            true_original, rank=rank, alpha=alpha_value, lora_name=module_key
        )

        # Load pretrained weights
        device = true_original.weight.device
        dtype = true_original.weight.dtype

        with torch.no_grad():
            lora_wrapper.lora_down.weight.data = lora_down_weight.to(device=device, dtype=dtype)
            lora_wrapper.lora_up.weight.data = lora_up_weight.to(device=device, dtype=dtype)

        # Apply strength (override the default scale)
        lora_wrapper.scale = (alpha_value / rank) * strength

        # Replace in parent module
        if isinstance(attr_name, int):
            parent_module[attr_name] = lora_wrapper
        else:
            setattr(parent_module, attr_name, lora_wrapper)

        self._flux2_lora_wrapped_modules.add(module_key)
        return True

    def _unload_lora_flux2(self):
        """Unload LoRAs from FLUX.2 Transformer"""
        if not hasattr(self, '_flux2_lora_original_modules'):
            print("[FLUX.2 LoRA] No LoRAs loaded")
            return

        if not self.flux2_components:
            print("[FLUX.2 LoRA] WARNING: FLUX.2 components not loaded")
            return

        transformer = self.flux2_components["transformer"]
        unloaded_count = 0

        print(f"[FLUX.2 LoRA] Unloading LoRAs ({len(self._flux2_lora_wrapped_modules)} modules)...")

        for name, module in transformer.named_modules():
            # Flux2Attention
            if module.__class__.__name__ == "Flux2Attention":
                for attr_name in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

                # to_out (ModuleList)
                module_key = f"{name}.to_out.0"
                if module_key in self._flux2_lora_original_modules and hasattr(module, "to_out"):
                    module.to_out[0] = self._flux2_lora_original_modules[module_key]
                    unloaded_count += 1

            # Flux2ParallelSelfAttention
            elif module.__class__.__name__ == "Flux2ParallelSelfAttention":
                for attr_name in ["to_qkv_mlp_proj", "to_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

            # Flux2FeedForward
            elif module.__class__.__name__ == "Flux2FeedForward":
                for attr_name in ["linear_in", "linear_out"]:
                    module_key = f"{name}.{attr_name}"
                    if module_key in self._flux2_lora_original_modules:
                        setattr(module, attr_name, self._flux2_lora_original_modules[module_key])
                        unloaded_count += 1

        self._flux2_lora_wrapped_modules.clear()
        print(f"[FLUX.2 LoRA] Unloaded {unloaded_count} LoRA modules")

    def _generate_txt2img_flux2(self, params: Dict[str, Any], progress_callback=None, step_callback=None) -> tuple[Image.Image, int, int]:
        """Generate image from text using FLUX.2 Klein

        Args:
            params: Generation parameters
            progress_callback: Callback for progress (step, total_steps, latent)
            step_callback: Step callback (not used for FLUX.2)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting txt2img generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            print(f"[FLUX.2] DEBUG: lora_configs from params = {lora_configs}")
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()
                else:
                    print(f"[FLUX.2] DEBUG: No LoRAs in params, skipping LoRA loading")

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Honor the selected attention backend for this run. FLUX.2 drives diffusers'
            # own attention registry (dispatch_attention_fn) from our canonical backend
            # string: default processors via transformer.set_attention_backend, and the
            # NAG/NegPip processor classes via their _attention_backend choke point. This
            # was previously always native (attention_type was ignored). try/except inside
            # the helper falls back to native if the diffusers build rejects flash/sage.
            attention_type = params.get("attention_type", settings.attention_type)
            attention_impl = params.get("attention_impl", getattr(settings, "attention_impl", "conduit"))
            set_flux2_attention_backend(transformer, attention_type, attention_impl)

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed (for stochastic samplers)
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
                print(f"[FLUX.2] Generated random ancestral seed: {actual_ancestral_seed}")
            else:
                actual_ancestral_seed = ancestral_seed
                print(f"[FLUX.2] Using specified ancestral seed: {ancestral_seed}")

            # FLUX.2 parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            height = params.get("height", 1024)
            width = params.get("width", 1024)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            max_sequence_length = 512  # FLUX.2 uses Qwen3 with max 512 tokens

            # Check if distilled model (no CFG)
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            print(f"[FLUX.2] Generating {width}x{height} image")
            print(f"[FLUX.2] Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
            print(f"[FLUX.2] CFG enabled: {do_classifier_free_guidance}")
            print(f"[FLUX.2] Prompt: {prompt[:100]}...")

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")  # Transformer (U-Net equivalent)
            text_encoder_quantization = params.get("text_encoder_quantization")  # Text Encoder (Qwen3)

            # ============================================================
            # Stage 1: Text Encoding (Qwen3)
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            # NAG (Normalized Attention Guidance): encode the nag-negative prompt so image
            # tokens can be guided away from it in attention space. Works with CFG (text
            # batch [cfg_neg, cfg_pos, nag_neg]) and distilled (text [pos, nag_neg]).
            nag_active = params.get("nag_enable", False) and params.get("nag_scale", 5.0) > 1.0
            nag_negative_prompt_embeds = None
            nag_negative_text_ids = None
            nag_wrapper = None
            nag_neg_prompt = params.get("nag_negative_prompt", "") or negative_prompt or ""
            if nag_active:
                nag_negative_prompt_embeds, nag_negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, nag_neg_prompt, max_sequence_length
                )

            # NegPip: auto-activate on a negative emphasis weight (e.g. (worst:-1)) in
            # either prompt. Signed per-token V weighting; positive-only prompts skip
            # this entirely (byte-identical default path). Builds a [txt_b, seq] signed
            # weight tensor aligned to the Qwen3 chat-template token sequence, per CFG
            # context (and nag_neg row when NAG is active).
            negpip_active = self._flux2_negpip_eligible(prompt, negative_prompt)
            negpip_weights = None
            negpip_wrapper = None
            if negpip_active:
                negpip_weights = self._build_flux2_negpip_weights(
                    prompt, negative_prompt, tokenizer, prompt_embeds,
                    prompt_embeds.dtype, do_classifier_free_guidance, nag_active,
                    nag_neg_prompt, max_sequence_length,
                )
                print(f"[FLUX.2] NegPip auto-activated (negative emphasis weight detected); "
                      f"weights {tuple(negpip_weights.shape)}")

            # Offload text encoder to CPU
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Prepare Latents
            # ============================================================
            print("[FLUX.2] Stage 2: Preparing latents...")

            # VAE scale factor (8) * patch size (2) = 16
            vae_scale_factor = 8
            patch_size = 2

            # Ensure height/width divisible by vae_scale_factor * patch_size
            latent_height = 2 * (int(height) // (vae_scale_factor * patch_size))
            latent_width = 2 * (int(width) // (vae_scale_factor * patch_size))

            # FLUX.2 has 32 latent channels, but patchified to 128
            num_channels_latents = transformer.config.in_channels // 4  # 32

            # Create random latents
            latent_shape = (1, num_channels_latents * 4, latent_height // 2, latent_width // 2)
            latents = torch.randn(latent_shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype)

            # Prepare latent position IDs
            latent_ids = self._flux2_prepare_latent_ids(latents).to(self.device)

            # Pack latents: (B, C, H, W) -> (B, H*W, C)
            latents = self._flux2_pack_latents(latents)

            print(f"[FLUX.2] Latents shape: {latents.shape}, Latent IDs shape: {latent_ids.shape}")

            # ============================================================
            # Stage 3: Denoising Loop
            # ============================================================
            print("[FLUX.2] Stage 3: Denoising loop...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                # Create block offloader
                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )

                # Prepare block devices
                block_offloader.prepare_block_devices_before_forward()

                # NAG / NegPip now compose with Block Swap: install the matching attention
                # processors and build ONE unified wrapper holding both the offloader and
                # the single-stream processors.
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                #   plain        -> Flux2BlockSwapWrapper
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip + Block Swap enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG + Block Swap enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(
                        transformer, negpip_weights, block_offloader=block_offloader
                    )
                    transformer_wrapper = negpip_wrapper
                    print("[FLUX.2] NegPip + Block Swap enabled")
                else:
                    # Wrap transformer (block swap only)
                    transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                    print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                # This is important when switching from Block Swap ON to OFF
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                # Move all block weights to GPU (in case they were on CPU from previous Block Swap)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

                # NAG / NegPip (no block swap): swap in a forward wrapper (installs
                # attention processors; built with no offloader here). The same wrappers
                # compose with block swap in the branch above. Restored after the loop via
                # nag_wrapper.restore() / negpip_wrapper.restore().
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(transformer, negpip_weights)
                    transformer_wrapper = negpip_wrapper

            # First Block Cache (FBCache): dynamic per-step image-residual reuse. Mutually
            # exclusive with (a) Spectrum (same trajectory redundancy; combining compounds
            # error) and (b) Block Swap (a cache hit skips the block loops -> desyncs the
            # per-block swap rotation). Active only when BOTH are off. When active, we must
            # route through the unified Flux2BlockSwapWrapper (offloader=None) so its custom
            # forward intercepts the dual+single block loops; the raw diffusers forward
            # (fast path) does not. If NAG/NegPip already installed a wrapper above, reuse it.
            fbcache = self._flux2_build_fbcache(
                params, enable_block_swap and blocks_to_swap > 0
            )
            if fbcache is not None:
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
                _unified = getattr(transformer_wrapper, "_unified", None)
                if isinstance(transformer_wrapper, Flux2BlockSwapWrapper):
                    fbcache_target = transformer_wrapper
                elif isinstance(_unified, Flux2BlockSwapWrapper):
                    # NAG/NegPip wrapper delegates forward to its internal _unified
                    # Flux2BlockSwapWrapper (whose forward has the FBCache branch); attach
                    # there so the NAG/NegPip wrapper is preserved (do NOT replace it).
                    fbcache_target = _unified
                else:
                    fbcache_target = Flux2BlockSwapWrapper(transformer, block_offloader=None)
                    transformer_wrapper = fbcache_target
                fbcache_target._fbcache = fbcache
            else:
                fbcache_target = None

            # Prepare timesteps
            image_seq_len = latents.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)

            # Set timesteps with sigmas
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps
            scheduler.set_begin_index(0)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            # Denoising loop
            # Spectrum output-mode acceleration (forecast per-step model output)
            spectrum = None
            if params.get("spectrum_enable", False):
                from core.inference.spectrum_forecaster import build_output_forecaster
                spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    # Cleanup block offloader if used
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                # Expand timestep
                preview_pred_x0 = None  # set by the eval branch; None on Spectrum skip steps
                # Spectrum: forecast the model output on skip steps (skip transformer + CFG)
                spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
                if spectrum_skip:
                    noise_pred = spectrum.forecast(i)
                else:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # FBCache: hand the wrapper the current step index (warmup + per-step gate).
                    if fbcache_target is not None:
                        fbcache_target._fbcache_step = i

                    latent_model_input = latents.to(transformer_input_dtype)
                    latent_image_ids = latent_ids

                    # Concatenate reference tokens/IDs if present (Image Edit)
                    if ref_tokens is not None:
                        # Temporarily move to GPU for concatenation
                        ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                        ref_ids = ref_ids.to(device=latent_image_ids.device)
                        latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                        latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                    # Batch CFG: Concatenate unconditional and conditional for single forward pass
                    if do_classifier_free_guidance:
                        # Double the batch: [uncond, cond]
                        latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                        timestep_doubled = torch.cat([timestep, timestep], dim=0)
                        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                        if nag_wrapper is not None:
                            # CFG+NAG: text batch [cfg_neg, cfg_pos, nag_neg]; image stays 2x
                            prompt_embeds_combined = torch.cat([prompt_embeds_combined, nag_negative_prompt_embeds], dim=0)
                            text_ids_combined = torch.cat([text_ids_combined, nag_negative_text_ids], dim=0)
                        latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                        # Single forward pass for both unconditional and conditional
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred_combined = transformer_wrapper(
                                        hidden_states=latent_model_input_doubled,
                                        timestep=timestep_doubled / 1000,
                                        guidance=None,
                                        encoder_hidden_states=prompt_embeds_combined,
                                        txt_ids=text_ids_combined,
                                        img_ids=latent_image_ids_doubled,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                        # Split and apply CFG formula
                        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        # Distilled model: Use guidance vector (not CFG)
                        guidance_vec = torch.full(
                            (latent_model_input.shape[0],),
                            guidance_scale,
                            device=latent_model_input.device,
                            dtype=latent_model_input.dtype
                        )
                        # NAG (distilled): text batch [pos, nag_neg]; image stays 1x
                        _nag_enc = prompt_embeds
                        _nag_tids = text_ids
                        if nag_wrapper is not None:
                            _nag_enc = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)
                            _nag_tids = torch.cat([text_ids, nag_negative_text_ids], dim=0)
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred = transformer_wrapper(
                                        hidden_states=latent_model_input,
                                        timestep=timestep / 1000,
                                        guidance=guidance_vec,
                                        encoder_hidden_states=_nag_enc,
                                        txt_ids=_nag_tids,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=_nag_enc,
                                    txt_ids=_nag_tids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred = noise_pred[:, :seq_len, :]

                    # Predicted clean latent for preview, computed from the
                    # pre-step latents + noise_pred. x_t = (1-σ)·x_0 + σ·noise,
                    # v = noise - x_0, σ = t / 1000 -> pred_x0 = x_t - σ·v.
                    # The progress callback receives this as the 5th positional
                    # arg (pred_original_sample) and the factory uses it when
                    # preview_predicted_x0=True (defaulted on for FLUX.2 below).
                    try:
                        sigma = (
                            t.float() / 1000.0 if isinstance(t, torch.Tensor)
                            else float(t) / 1000.0
                        )
                        preview_pred_x0 = (latents.float() - sigma * noise_pred.float()).to(latents.dtype)
                    except Exception:
                        preview_pred_x0 = None

                    # Scheduler step
                    if spectrum is not None:
                        spectrum.record(i, noise_pred)
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents, None, preview_pred_x0)
                    except Exception as e:
                        print(f"[FLUX.2] Progress callback error: {e}")

                if (i + 1) % 10 == 0 or i == len(timesteps) - 1:
                    print(f"[FLUX.2] Step {i + 1}/{len(timesteps)}")

            # FBCache cleanup: detach the cache + step so it never leaks into a later forward.
            if fbcache_target is not None:
                print(f"[FBCache] FLUX.2 summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target._fbcache = None
                fbcache_target._fbcache_step = 0

            # Cleanup block offloader and offload transformer to CPU
            if block_offloader is not None:
                block_offloader.cleanup()
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 4: VAE Decode
            # ============================================================
            print("[FLUX.2] Stage 4: VAE decoding...")
            vae = vae.to(self.device)

            # Unpack latents with IDs
            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Apply BatchNorm scaling (FLUX.2-specific)
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            # Unpatchify
            latents = self._flux2_unpatchify_latents(latents)

            # Decode - convert latents to VAE dtype (bfloat16 -> float32)
            latents = latents.to(dtype=vae.dtype)
            with torch.no_grad():
                self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
                image = vae.decode(latents, return_dict=False)[0]

            # Convert to PIL
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Offload VAE to CPU
            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] Generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 generation failed: {str(e)}")

    def _flux2_negpip_eligible(self, prompt: str, negative_prompt: str) -> bool:
        """Auto-activate NegPip iff either prompt carries a negative emphasis weight.

        Positive-only prompts return False so the default path is byte-identical.
        """
        try:
            from core.prompts.prompt_parser import prompt_has_negative_weight
        except Exception:
            return False
        return bool(prompt_has_negative_weight(prompt) or
                    prompt_has_negative_weight(negative_prompt or ""))

    def _build_flux2_negpip_weights(self, prompt, negative_prompt, tokenizer,
                                    prompt_embeds, dtype, do_cfg, nag_active,
                                    nag_negative_prompt, max_sequence_length=512):
        """Signed per-token weight tensor [txt_b, seq] matching the transformer text batch."""
        from core.inference.negpip_flux2 import build_flux2_negpip_weights
        device = prompt_embeds.device
        return build_flux2_negpip_weights(
            prompt, negative_prompt or "", tokenizer, device, dtype,
            embed_seq_len=prompt_embeds.shape[1],
            nag_negative_prompt=nag_negative_prompt,
            do_cfg=do_cfg, nag_active=nag_active,
            max_length=max_sequence_length,
        )

    def _flux2_encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple = (9, 18, 27),
    ):
        """Encode prompt using Qwen3 text encoder

        FLUX.2 extracts hidden states from layers 9, 18, 27 of Qwen3 and concatenates them.
        """
        device = text_encoder.device

        # Check if Text Encoder has FP8 weights
        has_fp8_weights = False
        for module in text_encoder.modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    has_fp8_weights = True
                    break

        # For FP8 quantized models, use BF16 for output dtype (not FP8)
        if has_fp8_weights:
            dtype = torch.bfloat16
        else:
            dtype = text_encoder.dtype

        print(f"[FLUX.2] FP8 weight detection: has_fp8_weights = {has_fp8_weights}, output dtype = {dtype}")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        # For FP8 quantized Text Encoder, use autocast for mixed precision
        with torch.no_grad():
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
            else:
                output = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

        # Extract and stack hidden states from specified layers
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        # Prepare text IDs (4D position coordinates)
        text_ids = self._flux2_prepare_text_ids(prompt_embeds).to(device)

        return prompt_embeds, text_ids

    def _flux2_prepare_text_ids(self, x: torch.Tensor):
        """Prepare 4D position IDs for text embeddings"""
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _flux2_prepare_latent_ids(self, latents: torch.Tensor):
        """Prepare 4D position IDs for latents"""
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    def _flux2_pack_latents(self, latents: torch.Tensor):
        """Pack latents: (B, C, H, W) -> (B, H*W, C)"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _flux2_unpack_latents_with_ids(self, x: torch.Tensor, x_ids: torch.Tensor):
        """Unpack latents using position IDs"""
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def _flux2_patchify_latents(self, latents: torch.Tensor):
        """Patchify latents for 2x2 patches"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    def _flux2_unpatchify_latents(self, latents: torch.Tensor):
        """Unpatchify latents from 2x2 patches"""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def _flux2_build_fbcache(self, params, block_swap_on: bool):
        """Build a FirstBlockCache for FLUX.2, or None when inactive/guarded.

        FBCache is mutually exclusive with Spectrum (same trajectory-redundancy target) and
        Block Swap (a cache hit skips the block loops, desyncing the per-block swap rotation),
        so it is force-disabled (with a logged reason) when either is enabled."""
        from core.inference.fbcache import build_fbcache, fbcache_active
        if not fbcache_active(params):
            return None
        if params.get("spectrum_enable", False):
            print("[FBCache] FLUX.2 disabled: Spectrum is enabled (same redundancy target)")
            return None
        if block_swap_on:
            print("[FBCache] FLUX.2 disabled: Block Swap is enabled (layer skip desyncs rotation)")
            return None
        return build_fbcache(params, label="FLUX.2")

    def _flux2_compute_empirical_mu(self, image_seq_len: int, num_steps: int) -> float:
        """Compute empirical mu for FLUX.2 scheduler"""
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)

    def encode_flux2_image_refs(self, images: List[Image.Image], device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode reference images for FLUX.2 Image Edit feature

        This encodes reference images into latent tokens with position IDs,
        allowing them to be used as sequence-level conditioning in the transformer.
        Reference images are concatenated with generation latents in the sequence dimension.

        Args:
            images: List of reference images (max 10)
            device: Device to encode on

        Returns:
            ref_tokens: [1, K, 128] Encoded reference image tokens
            ref_ids: [1, K, 4] Position IDs [t, h, w, l]
                     Returns (None, None) if no images provided
        """
        if not images:
            return None, None

        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded")

        import numpy as np

        # Pixel limits based on number of images
        limit_pixels = 2024**2 if len(images) == 1 else 1024**2

        vae = self.flux2_components["vae"]
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        print(f"[FLUX.2 Image Edit] Encoding {len(images)} reference image(s)...")

        # Preprocess and encode each image
        encoded_refs = []
        for idx, img in enumerate(images[:10]):  # Max 10 images
            # Convert to RGB
            img = img.convert("RGB")

            # Resize to fit pixel limit (preserve aspect ratio)
            w, h = img.size
            if w * h > limit_pixels:
                scale = (limit_pixels / (w * h)) ** 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                print(f"[FLUX.2 Image Edit] Image {idx+1}: Resized from {w}x{h} to {new_w}x{new_h}")

            # Crop to multiple of 16
            w, h = img.size
            new_w = (w // 16) * 16
            new_h = (h // 16) * 16
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            img = img.crop((left, top, left + new_w, top + new_h))

            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) * 2.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

            # VAE encode
            with torch.no_grad():
                latent_dist = vae.encode(img_tensor).latent_dist
                encoded = latent_dist.sample()

                # Patchify: (1, 32, H, W) -> (1, 128, H/2, W/2)
                encoded = self._flux2_patchify_latents(encoded)

                # BatchNorm normalization
                latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(encoded.device, encoded.dtype)
                latents_bn_std = torch.sqrt(
                    vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
                ).to(encoded.device, encoded.dtype)
                encoded = (encoded - latents_bn_mean) / latents_bn_std

                encoded_refs.append(encoded[0])  # [128, H, W]
                print(f"[FLUX.2 Image Edit] Image {idx+1}: Encoded to latent {encoded[0].shape}")

        # Generate position IDs for each reference image
        ref_tokens_list = []
        ref_ids_list = []

        scale = 10  # Time offset scale
        for idx, encoded in enumerate(encoded_refs):
            c, h, w = encoded.shape

            # Time offset: 10, 20, 30, ...
            t_coord = torch.tensor([scale + scale * idx], dtype=torch.long, device=device)

            # Position IDs: [t, h, w, l]
            t_ids = t_coord.expand(h * w)
            h_ids = torch.arange(h, device=device).repeat_interleave(w)
            w_ids = torch.arange(w, device=device).repeat(h)
            l_ids = torch.zeros(h * w, dtype=torch.long, device=device)

            pos_ids = torch.stack([t_ids, h_ids, w_ids, l_ids], dim=1)  # [H*W, 4]

            # Flatten spatial dimensions
            tokens = encoded.view(c, -1).permute(1, 0)  # [H*W, 128]

            ref_tokens_list.append(tokens)
            ref_ids_list.append(pos_ids)

        # Concatenate all references
        ref_tokens = torch.cat(ref_tokens_list, dim=0)  # [K, 128]
        ref_ids = torch.cat(ref_ids_list, dim=0)        # [K, 4]

        # Add batch dimension
        ref_tokens = ref_tokens.unsqueeze(0)  # [1, K, 128]
        ref_ids = ref_ids.unsqueeze(0)        # [1, K, 4]

        print(f"[FLUX.2 Image Edit] Total reference tokens: {ref_tokens.shape[1]}, shape: {ref_tokens.shape}")

        # Offload VAE to CPU after encoding reference images
        vae.to("cpu")
        torch.cuda.empty_cache()

        return ref_tokens, ref_ids

    def _generate_img2img_flux2(self, params: Dict[str, Any], init_image: Image.Image, progress_callback=None, step_callback=None) -> tuple[Image.Image, int, int]:
        """Generate image from image using FLUX.2 Klein

        FLUX.2 supports image conditioning by encoding input images to latents
        and using them as reference during denoising.

        Args:
            params: Generation parameters
            init_image: Input PIL image
            progress_callback: Callback for progress
            step_callback: Step callback (not used)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting img2img generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Honor the selected attention backend for this run. FLUX.2 drives diffusers'
            # own attention registry (dispatch_attention_fn) from our canonical backend
            # string: default processors via transformer.set_attention_backend, and the
            # NAG/NegPip processor classes via their _attention_backend choke point. This
            # was previously always native (attention_type was ignored). try/except inside
            # the helper falls back to native if the diffusers build rejects flash/sage.
            attention_type = params.get("attention_type", settings.attention_type)
            attention_impl = params.get("attention_impl", getattr(settings, "attention_impl", "conduit"))
            set_flux2_attention_backend(transformer, attention_type, attention_impl)

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
            else:
                actual_ancestral_seed = ancestral_seed

            # Parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            denoising_strength = params.get("denoising_strength", 0.75)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            max_sequence_length = 512

            # Get image dimensions (use input image size)
            width, height = init_image.size

            # VAE scale factor
            vae_scale_factor = 8
            patch_size = 2
            multiple_of = vae_scale_factor * patch_size

            # Resize if needed
            width = (width // multiple_of) * multiple_of
            height = (height // multiple_of) * multiple_of
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)

            print(f"[FLUX.2] img2img: {width}x{height}, strength: {denoising_strength}")

            # Check CFG
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            # NAG (Normalized Attention Guidance): encode the nag-negative prompt so image
            # tokens can be guided away from it in attention space. Works with CFG (text
            # batch [cfg_neg, cfg_pos, nag_neg]) and distilled (text [pos, nag_neg]).
            nag_active = params.get("nag_enable", False) and params.get("nag_scale", 5.0) > 1.0
            nag_negative_prompt_embeds = None
            nag_negative_text_ids = None
            nag_wrapper = None
            nag_neg_prompt = params.get("nag_negative_prompt", "") or negative_prompt or ""
            if nag_active:
                nag_negative_prompt_embeds, nag_negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, nag_neg_prompt, max_sequence_length
                )

            # NegPip: auto-activate on a negative emphasis weight in either prompt.
            negpip_active = self._flux2_negpip_eligible(prompt, negative_prompt)
            negpip_weights = None
            negpip_wrapper = None
            if negpip_active:
                negpip_weights = self._build_flux2_negpip_weights(
                    prompt, negative_prompt, tokenizer, prompt_embeds,
                    prompt_embeds.dtype, do_classifier_free_guidance, nag_active,
                    nag_neg_prompt, max_sequence_length,
                )
                print(f"[FLUX.2] NegPip auto-activated (negative emphasis weight detected); "
                      f"weights {tuple(negpip_weights.shape)}")

            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Encode input image
            # ============================================================
            print("[FLUX.2] Stage 2: Encoding input image...")
            vae = vae.to(self.device)

            # Preprocess image
            image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(self.device, dtype=vae.dtype)

            # Encode
            with torch.no_grad():
                latent_dist = vae.encode(image_tensor).latent_dist
                init_latents = latent_dist.mode()  # Use mode for img2img

            # Patchify
            init_latents = self._flux2_patchify_latents(init_latents)

            # Apply BatchNorm normalization
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(init_latents.device, init_latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            init_latents = (init_latents - latents_bn_mean) / latents_bn_std

            vae.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 3: Prepare latents with noise
            # ============================================================
            print("[FLUX.2] Stage 3: Preparing latents...")

            # Prepare position IDs
            latent_ids = self._flux2_prepare_latent_ids(init_latents).to(self.device)

            # Pack latents
            init_latents = self._flux2_pack_latents(init_latents)

            # Prepare timesteps
            image_seq_len = init_latents.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps

            # Calculate start timestep based on denoising strength
            t_start = max(int(len(timesteps) * (1 - denoising_strength)), 1)
            timesteps = timesteps[t_start:]

            # Add noise at start timestep (Flow Matching linear interpolation)
            # t ranges from 1.0 (pure noise) to 0.0 (clean image)
            # scheduler.timesteps is in [0, 1000] range, normalize to [0, 1]
            t_value = timesteps[0].item() / 1000.0
            noise = torch.randn(init_latents.shape, generator=generator, device=init_latents.device, dtype=init_latents.dtype)
            latents = (1 - t_value) * init_latents + t_value * noise

            print(f"[FLUX.2] Denoising from step {t_start} ({len(timesteps)} steps, t={t_value:.4f})")

            # ============================================================
            # Stage 4: Denoising Loop
            # ============================================================
            print("[FLUX.2] Stage 4: Denoising loop...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )
                block_offloader.prepare_block_devices_before_forward()
                # NAG / NegPip now compose with Block Swap: install the matching attention
                # processors and build ONE unified wrapper holding both the offloader and
                # the single-stream processors.
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                #   plain        -> Flux2BlockSwapWrapper
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip + Block Swap enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG + Block Swap enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(
                        transformer, negpip_weights, block_offloader=block_offloader
                    )
                    transformer_wrapper = negpip_wrapper
                    print("[FLUX.2] NegPip + Block Swap enabled")
                else:
                    transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                    print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

                # NAG / NegPip (no block swap): swap in a forward wrapper (installs
                # attention processors; built with no offloader here). The same wrappers
                # compose with block swap in the branch above. Restored after the loop via
                # nag_wrapper.restore() / negpip_wrapper.restore().
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(transformer, negpip_weights)
                    transformer_wrapper = negpip_wrapper

            # First Block Cache (FBCache): dynamic per-step image-residual reuse. Mutually
            # exclusive with Spectrum and Block Swap (see _flux2_build_fbcache). When active,
            # route through the unified Flux2BlockSwapWrapper (offloader=None) so its custom
            # forward intercepts the dual+single block loops (the raw diffusers forward /
            # fast path does not). Reuse the NAG/NegPip wrapper if one was installed above.
            fbcache = self._flux2_build_fbcache(
                params, enable_block_swap and blocks_to_swap > 0
            )
            if fbcache is not None:
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
                _unified = getattr(transformer_wrapper, "_unified", None)
                if isinstance(transformer_wrapper, Flux2BlockSwapWrapper):
                    fbcache_target = transformer_wrapper
                elif isinstance(_unified, Flux2BlockSwapWrapper):
                    # NAG/NegPip wrapper delegates forward to its internal _unified
                    # Flux2BlockSwapWrapper (whose forward has the FBCache branch); attach
                    # there so the NAG/NegPip wrapper is preserved (do NOT replace it).
                    fbcache_target = _unified
                else:
                    fbcache_target = Flux2BlockSwapWrapper(transformer, block_offloader=None)
                    transformer_wrapper = fbcache_target
                fbcache_target._fbcache = fbcache
            else:
                fbcache_target = None

            scheduler.set_begin_index(t_start)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            # Spectrum output-mode acceleration (forecast per-step model output)
            spectrum = None
            if params.get("spectrum_enable", False):
                from core.inference.spectrum_forecaster import build_output_forecaster
                spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                preview_pred_x0 = None  # set by the eval branch; None on Spectrum skip steps
                # Spectrum: forecast the model output on skip steps (skip transformer + CFG)
                spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
                if spectrum_skip:
                    noise_pred = spectrum.forecast(i)
                else:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # FBCache: hand the wrapper the current step index (warmup + per-step gate).
                    if fbcache_target is not None:
                        fbcache_target._fbcache_step = i

                    latent_model_input = latents.to(transformer_input_dtype)
                    latent_image_ids = latent_ids

                    # Concatenate reference tokens/IDs if present (Image Edit)
                    if ref_tokens is not None:
                        # Temporarily move to GPU for concatenation
                        ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                        ref_ids = ref_ids.to(device=latent_image_ids.device)
                        latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                        latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                    # Batch CFG: Concatenate unconditional and conditional for single forward pass
                    if do_classifier_free_guidance:
                        # Double the batch: [uncond, cond]
                        latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                        timestep_doubled = torch.cat([timestep, timestep], dim=0)
                        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                        if nag_wrapper is not None:
                            # CFG+NAG: text batch [cfg_neg, cfg_pos, nag_neg]; image stays 2x
                            prompt_embeds_combined = torch.cat([prompt_embeds_combined, nag_negative_prompt_embeds], dim=0)
                            text_ids_combined = torch.cat([text_ids_combined, nag_negative_text_ids], dim=0)
                        latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                        # Single forward pass for both unconditional and conditional
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred_combined = transformer_wrapper(
                                        hidden_states=latent_model_input_doubled,
                                        timestep=timestep_doubled / 1000,
                                        guidance=None,
                                        encoder_hidden_states=prompt_embeds_combined,
                                        txt_ids=text_ids_combined,
                                        img_ids=latent_image_ids_doubled,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                        # Split and apply CFG formula
                        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        # Distilled model: Use guidance vector (not CFG)
                        guidance_vec = torch.full(
                            (latent_model_input.shape[0],),
                            guidance_scale,
                            device=latent_model_input.device,
                            dtype=latent_model_input.dtype
                        )
                        # NAG (distilled): text batch [pos, nag_neg]; image stays 1x
                        _nag_enc = prompt_embeds
                        _nag_tids = text_ids
                        if nag_wrapper is not None:
                            _nag_enc = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)
                            _nag_tids = torch.cat([text_ids, nag_negative_text_ids], dim=0)
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred = transformer_wrapper(
                                        hidden_states=latent_model_input,
                                        timestep=timestep / 1000,
                                        guidance=guidance_vec,
                                        encoder_hidden_states=_nag_enc,
                                        txt_ids=_nag_tids,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=_nag_enc,
                                    txt_ids=_nag_tids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred = noise_pred[:, :seq_len, :]

                    # Step
                    if spectrum is not None:
                        spectrum.record(i, noise_pred)
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents)
                    except Exception:
                        pass

            # FBCache cleanup: detach the cache + step so it never leaks into a later forward.
            if fbcache_target is not None:
                print(f"[FBCache] FLUX.2 summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target._fbcache = None
                fbcache_target._fbcache_step = 0

            # Cleanup block offloader and offload transformer to CPU (img2img)
            if block_offloader is not None:
                block_offloader.cleanup()
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (img2img)
            # ============================================================
            print("[FLUX.2] Stage 5: VAE decoding...")
            vae = vae.to(self.device)

            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Denormalize
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            latents = self._flux2_unpatchify_latents(latents)

            with torch.no_grad():
                self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
                image = vae.decode(latents, return_dict=False)[0]

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] img2img generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] img2img error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 img2img failed: {str(e)}")

    def _generate_inpaint_flux2(
        self,
        params: Dict[str, Any],
        init_image: Image.Image,
        mask_image: Image.Image,
        progress_callback=None,
        step_callback=None
    ) -> tuple[Image.Image, int, int]:
        """Generate inpainted image using FLUX.2 Klein

        FLUX.2 inpainting works by blending masked regions during denoising.

        Args:
            params: Generation parameters
            init_image: Input PIL image
            mask_image: Mask PIL image (white = inpaint, black = keep)
            progress_callback: Callback for progress
            step_callback: Step callback (not used)

        Returns:
            tuple: (image, actual_seed, actual_ancestral_seed)
        """
        if not self.flux2_components:
            raise RuntimeError("FLUX.2 components not loaded. Please load a FLUX.2 model first.")

        print("[FLUX.2] Starting inpaint generation")

        try:
            import numpy as np

            # Load LoRAs if specified
            lora_configs = params.get("loras", [])
            if lora_configs:
                # Unload previous LoRAs first (if any)
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    self._unload_lora_flux2()
                # Load new LoRAs
                print(f"[FLUX.2] Loading {len(lora_configs)} LoRA(s)...")
                self._load_lora_flux2(lora_configs)
            else:
                # No LoRAs requested - unload if any are loaded
                if hasattr(self, '_flux2_lora_wrapped_modules') and self._flux2_lora_wrapped_modules:
                    print(f"[FLUX.2] No LoRAs in params, unloading existing LoRAs")
                    self._unload_lora_flux2()

            # Extract components
            transformer = self.flux2_components["transformer"]
            vae = self.flux2_components["vae"]
            text_encoder = self.flux2_components["text_encoder"]
            tokenizer = self.flux2_components["tokenizer"]
            scheduler = self.flux2_components["scheduler"]
            config = self.flux2_components.get("config", {})

            # Honor the selected attention backend for this run. FLUX.2 drives diffusers'
            # own attention registry (dispatch_attention_fn) from our canonical backend
            # string: default processors via transformer.set_attention_backend, and the
            # NAG/NegPip processor classes via their _attention_backend choke point. This
            # was previously always native (attention_type was ignored). try/except inside
            # the helper falls back to native if the diffusers build rejects flash/sage.
            attention_type = params.get("attention_type", settings.attention_type)
            attention_impl = params.get("attention_impl", getattr(settings, "attention_impl", "conduit"))
            set_flux2_attention_backend(transformer, attention_type, attention_impl)

            # Prepare generator
            seed = params.get("seed", -1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)

            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

            # Ancestral seed
            ancestral_seed = params.get("ancestral_seed", -1)
            if ancestral_seed == -1:
                actual_ancestral_seed = random.randint(0, 2147483647)
            else:
                actual_ancestral_seed = ancestral_seed

            # Parameters
            prompt = params.get("prompt", "")
            negative_prompt = params.get("negative_prompt", "")
            denoising_strength = params.get("denoising_strength", 1.0)
            num_inference_steps = params.get("steps", 50)
            guidance_scale = params.get("cfg_scale", 4.0)
            mask_blur = params.get("mask_blur", 4)
            max_sequence_length = 512

            # Get dimensions
            width, height = init_image.size

            vae_scale_factor = 8
            patch_size = 2
            multiple_of = vae_scale_factor * patch_size

            # Resize if needed
            width = (width // multiple_of) * multiple_of
            height = (height // multiple_of) * multiple_of
            if init_image.size != (width, height):
                init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
                mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)

            print(f"[FLUX.2] inpaint: {width}x{height}, strength: {denoising_strength}")

            # Apply mask blur
            if mask_blur > 0:
                from PIL import ImageFilter
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=mask_blur))

            # Check CFG
            is_distilled = config.get("is_distilled", False)
            do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

            # Import VRAM optimization functions
            from core.vram_optimization import (
                move_flux2_text_encoder_to_gpu,
                move_flux2_transformer_to_gpu
            )

            # Get quantization parameters
            transformer_quantization = params.get("unet_quantization")
            text_encoder_quantization = params.get("text_encoder_quantization")

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            text_encoder = move_flux2_text_encoder_to_gpu(text_encoder, text_encoder_quantization)

            prompt_embeds, text_ids = self._flux2_encode_prompt(
                text_encoder, tokenizer, prompt, max_sequence_length
            )

            if do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, negative_prompt, max_sequence_length
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

            # NAG (Normalized Attention Guidance): encode the nag-negative prompt so image
            # tokens can be guided away from it in attention space. Works with CFG (text
            # batch [cfg_neg, cfg_pos, nag_neg]) and distilled (text [pos, nag_neg]).
            nag_active = params.get("nag_enable", False) and params.get("nag_scale", 5.0) > 1.0
            nag_negative_prompt_embeds = None
            nag_negative_text_ids = None
            nag_wrapper = None
            nag_neg_prompt = params.get("nag_negative_prompt", "") or negative_prompt or ""
            if nag_active:
                nag_negative_prompt_embeds, nag_negative_text_ids = self._flux2_encode_prompt(
                    text_encoder, tokenizer, nag_neg_prompt, max_sequence_length
                )

            # NegPip: auto-activate on a negative emphasis weight in either prompt.
            negpip_active = self._flux2_negpip_eligible(prompt, negative_prompt)
            negpip_weights = None
            negpip_wrapper = None
            if negpip_active:
                negpip_weights = self._build_flux2_negpip_weights(
                    prompt, negative_prompt, tokenizer, prompt_embeds,
                    prompt_embeds.dtype, do_classifier_free_guidance, nag_active,
                    nag_neg_prompt, max_sequence_length,
                )
                print(f"[FLUX.2] NegPip auto-activated (negative emphasis weight detected); "
                      f"weights {tuple(negpip_weights.shape)}")

            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            ref_images = params.get("ref_images", [])
            ref_tokens = None
            ref_ids = None

            if ref_images:
                print(f"[FLUX.2 Image Edit] Encoding {len(ref_images)} reference image(s)...")
                ref_tokens, ref_ids = self.encode_flux2_image_refs(ref_images, device=self.device)
                if ref_tokens is not None:
                    ref_tokens = ref_tokens.to(prompt_embeds.dtype)
                    ref_ids = ref_ids.to(self.device)
                    print(f"[FLUX.2 Image Edit] Reference tokens: {ref_tokens.shape}, IDs: {ref_ids.shape}")

            # ============================================================
            # Stage 2: Encode input image and prepare mask
            # ============================================================
            print("[FLUX.2] Stage 2: Encoding input image and mask...")
            vae = vae.to(self.device)

            # Preprocess image
            image_tensor = torch.from_numpy(np.array(init_image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            image_tensor = (image_tensor - 0.5) * 2
            image_tensor = image_tensor.to(self.device, dtype=vae.dtype)

            # Encode
            with torch.no_grad():
                latent_dist = vae.encode(image_tensor).latent_dist
                init_latents = latent_dist.mode()

            # Prepare mask in latent space
            mask_tensor = torch.from_numpy(np.array(mask_image.convert("L"))).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Resize mask to latent size
            latent_h = height // vae_scale_factor
            latent_w = width // vae_scale_factor
            mask_latent = torch.nn.functional.interpolate(
                mask_tensor, size=(latent_h, latent_w), mode='bilinear', align_corners=False
            )
            mask_latent = mask_latent.to(self.device, dtype=init_latents.dtype)

            # Patchify
            init_latents = self._flux2_patchify_latents(init_latents)

            # Apply BatchNorm normalization
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(init_latents.device, init_latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
            init_latents_normalized = (init_latents - latents_bn_mean) / latents_bn_std

            vae.to("cpu")
            torch.cuda.empty_cache()

            # ============================================================
            # Stage 3: Prepare latents
            # ============================================================
            print("[FLUX.2] Stage 3: Preparing latents...")

            # Patchify mask (same spatial transform as latents)
            # Mask for patchified latents needs special handling
            mask_patchified = torch.nn.functional.interpolate(
                mask_latent, size=(latent_h // 2, latent_w // 2), mode='bilinear', align_corners=False
            )

            # Prepare position IDs
            latent_ids = self._flux2_prepare_latent_ids(init_latents).to(self.device)

            # Pack latents
            init_latents_packed = self._flux2_pack_latents(init_latents_normalized)

            # Pack mask (1, 1, H/2, W/2) -> (1, H*W/4, 1)
            mask_packed = mask_patchified.reshape(1, 1, -1).permute(0, 2, 1)

            # Prepare timesteps
            image_seq_len = init_latents_packed.shape[1]
            mu = self._flux2_compute_empirical_mu(image_seq_len, num_inference_steps)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
            timesteps = scheduler.timesteps

            # Calculate start timestep
            t_start = max(int(len(timesteps) * (1 - denoising_strength)), 1)
            timesteps = timesteps[t_start:]

            # Add noise (Flow Matching linear interpolation)
            # t ranges from 1.0 (pure noise) to 0.0 (clean image)
            # scheduler.timesteps is in [0, 1000] range, normalize to [0, 1]
            t_value = timesteps[0].item() / 1000.0
            noise = torch.randn(init_latents_packed.shape, generator=generator, device=init_latents_packed.device, dtype=init_latents_packed.dtype)
            latents = (1 - t_value) * init_latents_packed + t_value * noise

            print(f"[FLUX.2] Inpainting from step {t_start} ({len(timesteps)} steps, t={t_value:.4f})")

            # ============================================================
            # Stage 4: Denoising Loop with mask blending
            # ============================================================
            print("[FLUX.2] Stage 4: Denoising loop with mask blending...")

            # Block Swap setup
            enable_block_swap = params.get("enable_block_swap", False)
            blocks_to_swap = params.get("blocks_to_swap", 0) if enable_block_swap else 0
            use_pinned_memory = params.get("use_pinned_memory", False)
            block_swap_h2d_only = params.get("block_swap_h2d_only", False)
            block_swap_ring_size = int(params.get("block_swap_ring_size", 2))
            block_offloader = None

            if enable_block_swap and blocks_to_swap > 0:
                print(f"[FLUX.2] Block Swap enabled: {blocks_to_swap} blocks to swap")
                from core.memory_management import create_flux_block_offloader
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

                block_offloader = create_flux_block_offloader(
                    transformer=transformer,
                    blocks_to_swap=blocks_to_swap,
                    device=torch.device(self.device),
                    target_dtype=torch.bfloat16,
                    use_pinned_memory=use_pinned_memory,
                    supports_backward=False,
                    h2d_only=block_swap_h2d_only,
                    ring_size=block_swap_ring_size,
                )
                block_offloader.prepare_block_devices_before_forward()
                # NAG / NegPip now compose with Block Swap: install the matching attention
                # processors and build ONE unified wrapper holding both the offloader and
                # the single-stream processors.
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                #   plain        -> Flux2BlockSwapWrapper
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip + Block Swap enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                        block_offloader=block_offloader,
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG + Block Swap enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(
                        transformer, negpip_weights, block_offloader=block_offloader
                    )
                    transformer_wrapper = negpip_wrapper
                    print("[FLUX.2] NegPip + Block Swap enabled")
                else:
                    transformer_wrapper = Flux2BlockSwapWrapper(transformer, block_offloader)
                    print("[FLUX.2] Using Block Swap wrapper for denoising")
            else:
                # No Block Swap - ensure ALL weights are on GPU
                from core.memory_management.block_offloading import weighs_to_device
                transformer = move_flux2_transformer_to_gpu(transformer, transformer_quantization)
                for block in transformer.transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                for block in transformer.single_transformer_blocks:
                    weighs_to_device(block, torch.device(self.device))
                transformer_wrapper = transformer

                # NAG / NegPip (no block swap): swap in a forward wrapper (installs
                # attention processors; built with no offloader here). The same wrappers
                # compose with block swap in the branch above. Restored after the loop via
                # nag_wrapper.restore() / negpip_wrapper.restore().
                #   NAG + NegPip -> Flux2NegPipNAGWrapper (signed V folded into NAG's V)
                #   NAG only     -> Flux2NAGWrapper
                #   NegPip only  -> Flux2NegPipWrapper (signed text-V, no extra forward)
                if nag_active and negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipNAGWrapper
                    nag_wrapper = Flux2NegPipNAGWrapper(
                        transformer,
                        negpip_weights,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print("[FLUX.2] NAG + NegPip enabled")
                elif nag_active:
                    from core.inference.nag_flux2 import Flux2NAGWrapper
                    nag_wrapper = Flux2NAGWrapper(
                        transformer,
                        nag_scale=params.get("nag_scale", 5.0),
                        nag_tau=params.get("nag_tau", 2.5),
                        nag_alpha=params.get("nag_alpha", 0.25),
                    )
                    transformer_wrapper = nag_wrapper
                    print(f"[FLUX.2] NAG enabled: scale={params.get('nag_scale', 5.0)}, "
                          f"tau={params.get('nag_tau', 2.5)}, alpha={params.get('nag_alpha', 0.25)}")
                elif negpip_active:
                    from core.inference.negpip_flux2 import Flux2NegPipWrapper
                    negpip_wrapper = Flux2NegPipWrapper(transformer, negpip_weights)
                    transformer_wrapper = negpip_wrapper

            # First Block Cache (FBCache): dynamic per-step image-residual reuse. Mutually
            # exclusive with Spectrum and Block Swap (see _flux2_build_fbcache). When active,
            # route through the unified Flux2BlockSwapWrapper (offloader=None) so its custom
            # forward intercepts the dual+single block loops (the raw diffusers forward /
            # fast path does not). Reuse the NAG/NegPip wrapper if one was installed above.
            fbcache = self._flux2_build_fbcache(
                params, enable_block_swap and blocks_to_swap > 0
            )
            if fbcache is not None:
                from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper
                _unified = getattr(transformer_wrapper, "_unified", None)
                if isinstance(transformer_wrapper, Flux2BlockSwapWrapper):
                    fbcache_target = transformer_wrapper
                elif isinstance(_unified, Flux2BlockSwapWrapper):
                    # NAG/NegPip wrapper delegates forward to its internal _unified
                    # Flux2BlockSwapWrapper (whose forward has the FBCache branch); attach
                    # there so the NAG/NegPip wrapper is preserved (do NOT replace it).
                    fbcache_target = _unified
                else:
                    fbcache_target = Flux2BlockSwapWrapper(transformer, block_offloader=None)
                    transformer_wrapper = fbcache_target
                fbcache_target._fbcache = fbcache
            else:
                fbcache_target = None

            scheduler.set_begin_index(t_start)

            # Determine input dtype for transformer (FP8 quantized uses BF16 input)
            transformer_has_fp8 = False
            for module in transformer.modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    if module.weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        transformer_has_fp8 = True
                        break

            if transformer_has_fp8:
                transformer_input_dtype = torch.bfloat16
            else:
                transformer_input_dtype = transformer.dtype

            print(f"[FLUX.2] Transformer FP8 detection: {transformer_has_fp8}, input dtype = {transformer_input_dtype}")

            # Spectrum output-mode acceleration (forecast per-step model output)
            spectrum = None
            if params.get("spectrum_enable", False):
                from core.inference.spectrum_forecaster import build_output_forecaster
                spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            for i, t in enumerate(timesteps):
                if self.cancel_requested:
                    print("[FLUX.2] Generation cancelled")
                    self.cancel_requested = False
                    if block_offloader is not None:
                        block_offloader.cleanup()
                    raise RuntimeError("Generation cancelled by user")

                preview_pred_x0 = None  # set by the eval branch; None on Spectrum skip steps
                # Spectrum: forecast the model output on skip steps (skip transformer + CFG)
                spectrum_skip = spectrum is not None and not spectrum.is_anchor(i)
                if spectrum_skip:
                    noise_pred = spectrum.forecast(i)
                else:
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    # FBCache: hand the wrapper the current step index (warmup + per-step gate).
                    if fbcache_target is not None:
                        fbcache_target._fbcache_step = i

                    latent_model_input = latents.to(transformer_input_dtype)
                    latent_image_ids = latent_ids

                    # Concatenate reference tokens/IDs if present (Image Edit)
                    if ref_tokens is not None:
                        # Temporarily move to GPU for concatenation
                        ref_tokens = ref_tokens.to(device=latent_model_input.device, dtype=transformer_input_dtype)
                        ref_ids = ref_ids.to(device=latent_image_ids.device)
                        latent_model_input = torch.cat([latent_model_input, ref_tokens], dim=1)
                        latent_image_ids = torch.cat([latent_image_ids, ref_ids], dim=1)

                    # Batch CFG: Concatenate unconditional and conditional for single forward pass
                    if do_classifier_free_guidance:
                        # Double the batch: [uncond, cond]
                        latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                        timestep_doubled = torch.cat([timestep, timestep], dim=0)
                        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                        if nag_wrapper is not None:
                            # CFG+NAG: text batch [cfg_neg, cfg_pos, nag_neg]; image stays 2x
                            prompt_embeds_combined = torch.cat([prompt_embeds_combined, nag_negative_prompt_embeds], dim=0)
                            text_ids_combined = torch.cat([text_ids_combined, nag_negative_text_ids], dim=0)
                        latent_image_ids_doubled = torch.cat([latent_image_ids, latent_image_ids], dim=0)

                        # Single forward pass for both unconditional and conditional
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred_combined = transformer_wrapper(
                                        hidden_states=latent_model_input_doubled,
                                        timestep=timestep_doubled / 1000,
                                        guidance=None,
                                        encoder_hidden_states=prompt_embeds_combined,
                                        txt_ids=text_ids_combined,
                                        img_ids=latent_image_ids_doubled,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred_combined = transformer_wrapper(
                                    hidden_states=latent_model_input_doubled,
                                    timestep=timestep_doubled / 1000,
                                    guidance=None,
                                    encoder_hidden_states=prompt_embeds_combined,
                                    txt_ids=text_ids_combined,
                                    img_ids=latent_image_ids_doubled,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred_combined = noise_pred_combined[:, :seq_len, :]

                        # Split and apply CFG formula
                        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        # Distilled model: Use guidance vector (not CFG)
                        guidance_vec = torch.full(
                            (latent_model_input.shape[0],),
                            guidance_scale,
                            device=latent_model_input.device,
                            dtype=latent_model_input.dtype
                        )
                        # NAG (distilled): text batch [pos, nag_neg]; image stays 1x
                        _nag_enc = prompt_embeds
                        _nag_tids = text_ids
                        if nag_wrapper is not None:
                            _nag_enc = torch.cat([prompt_embeds, nag_negative_prompt_embeds], dim=0)
                            _nag_tids = torch.cat([text_ids, nag_negative_text_ids], dim=0)
                        # For FP8 quantized models, use autocast for mixed precision
                        with torch.no_grad():
                            if transformer_has_fp8:
                                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                    noise_pred = transformer_wrapper(
                                        hidden_states=latent_model_input,
                                        timestep=timestep / 1000,
                                        guidance=guidance_vec,
                                        encoder_hidden_states=_nag_enc,
                                        txt_ids=_nag_tids,
                                        img_ids=latent_image_ids,
                                        return_dict=False,
                                    )[0]
                            else:
                                noise_pred = transformer_wrapper(
                                    hidden_states=latent_model_input,
                                    timestep=timestep / 1000,
                                    guidance=guidance_vec,
                                    encoder_hidden_states=_nag_enc,
                                    txt_ids=_nag_tids,
                                    img_ids=latent_image_ids,
                                    return_dict=False,
                                )[0]

                        # Extract generation part only (remove reference tokens)
                        if ref_tokens is not None:
                            seq_len = latents.shape[1]
                            noise_pred = noise_pred[:, :seq_len, :]

                    # Step
                    if spectrum is not None:
                        spectrum.record(i, noise_pred)
                latents_dtype = latents.dtype
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                # Blend with original in unmasked regions
                # Noise original latents to current timestep using Flow Matching interpolation
                if i < len(timesteps) - 1:
                    # Flow Matching: normalize timestep [0, 1000] -> [0.0, 1.0]
                    t_value = timesteps[i + 1].item() / 1000.0
                    # Linear interpolation: x_t = (1 - t) * x_0 + t * noise
                    init_latents_noised = (1 - t_value) * init_latents_packed + t_value * noise
                else:
                    init_latents_noised = init_latents_packed

                # Blend: mask=1 -> use new latents, mask=0 -> use original
                latents = mask_packed * latents + (1 - mask_packed) * init_latents_noised

                # Progress callback (step is 0-indexed, generation_utils will add +1 for display)
                if progress_callback:
                    try:
                        progress_callback(i, len(timesteps), latents)
                    except Exception:
                        pass

            # FBCache cleanup: detach the cache + step so it never leaks into a later forward.
            if fbcache_target is not None:
                print(f"[FBCache] FLUX.2 summary: {fbcache.n_hits} hit(s), {fbcache.n_miss} miss(es)")
                fbcache_target._fbcache = None
                fbcache_target._fbcache_step = 0

            # Cleanup block offloader and offload transformer to CPU (inpaint)
            if block_offloader is not None:
                block_offloader.cleanup()
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            transformer.to("cpu")
            torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (inpaint)
            # ============================================================
            print("[FLUX.2] Stage 5: VAE decoding...")
            vae = vae.to(self.device)

            latents = self._flux2_unpack_latents_with_ids(latents, latent_ids)

            # Denormalize
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean

            latents = self._flux2_unpatchify_latents(latents)

            with torch.no_grad():
                self._apply_vae_tiling(vae, getattr(self, "_vae_tiling", False))
                image = vae.decode(latents, return_dict=False)[0]

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            vae.to("cpu")
            torch.cuda.empty_cache()

            print("[FLUX.2] inpaint generation completed")
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] inpaint error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 inpaint failed: {str(e)}")
