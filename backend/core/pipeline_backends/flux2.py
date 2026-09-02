from typing import Dict, Any, Optional, List, Callable, Tuple
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
import time as _time
from core.inference.generation_timing import generation_timer


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
            try:
                from api.generation_status import add_warning
                add_warning(
                    f"FLUX.2 attention backend '{diffusers_backend}' unavailable "
                    f"({e}); falling back to native",
                    code="attention_downgrade",
                )
            except Exception:
                pass
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

    def _flux2_runtime_int8(self, params: Dict[str, Any], transformer,
                            progress_callback=None):
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        Returns the transformer (the SAME object -- the conversion replaces child
        modules in place and never builds a second one; a 3.6 GB bf16 Klein 4B
        would not tolerate the ``copy.deepcopy`` the legacy FP8 path uses, and a
        9B still less).

        ORDERING, which is load-bearing rather than incidental:

        * BEFORE the block-swap wrapper and ``create_flux_block_offloader``. The
          offloader captures references to each block's Linear modules and builds
          pinned CPU masters from their weights; converting afterwards would
          replace the modules it holds and strand it on the pre-conversion ones.
          Checked rather than assumed, because the failure is silent -- and
          checked as the converter's ``precheck``, i.e. only when a conversion is
          really about to touch the first layer. FLUX.2 clears
          ``_flux2_active_block_offloader`` in a ``finally`` after every
          generation, so an unconditional guard here happens to be safe today;
          it is not safe as a PATTERN (the identical-looking guard on LTX-2.3,
          whose offloader is persistent wrapper state, refused every second
          block-swap generation even with no quantization requested), and the
          safety of this one should not rest on a ``finally`` in three other
          functions. Same shape as ``_ideogram4_runtime_int8``'s
          ``runtime_int8_requested`` guard.
        * BEFORE staging. ``move_flux2_transformer_to_gpu`` is only reached in the
          NO-block-swap branch, so quantizing there would leave a block-swapped
          generation unquantized; here it happens on whatever device the module
          currently sits on (the converter is device-aware).
        * AFTER LoRA loading, which is where LoRAs are applied for FLUX.2. A
          LoRA-wrapped transformer is REFUSED by the converter (the wrappers hide
          the Linears, so the selection would differ from the offline audit) and
          the user gets a warning; that is the same contract Krea 2 has.
        """
        from core.vram_optimization import apply_runtime_int8_quantization

        if transformer is None:
            return transformer

        # Checked, not asserted: `python -O` strips an assert, and this is the one
        # invariant whose violation is invisible (a conversion that "succeeded"
        # while the offloader still streams the pre-conversion weights).
        def _refuse_if_offloader_live():
            if getattr(self, "_flux2_active_block_offloader", None) is None:
                return
            raise RuntimeError(
                "FLUX.2 INT8 conversion was reached while a block offloader is still "
                "active. It must run BEFORE the offloader is created: the offloader "
                "holds references to each block's Linear modules, and the conversion "
                "replaces those modules, so afterwards it would stream the original "
                "bf16 weights into modules nothing reads.")

        model, _converted = apply_runtime_int8_quantization(
            self, transformer, "flux2", params.get("unet_quantization"),
            label="FLUX.2 Transformer", progress_callback=progress_callback,
            precheck=_refuse_if_offloader_live)
        if self.flux2_components is not None:
            self.flux2_components["transformer"] = model
        return model

    @staticmethod
    def _flux2_lora_warn(message: str, code: str) -> None:
        """Record a user-visible generation warning (best effort)."""
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    def _flux2_lora_state(self):
        """The (originals, wrapped_keys) maps for the CURRENTLY loaded components.

        Reset when the transformer or the text encoder was reloaded: the maps
        hold the OLD model's Linears, and restoring them would splice them into
        the new one. Keyed by weakref rather than id() because a freed object's
        id is reusable.
        """
        components = self.flux2_components or {}
        transformer = components.get("transformer")
        text_encoder = components.get("text_encoder")
        previous = getattr(self, "_flux2_lora_component_refs", None)

        def _same(ref, obj):
            return obj is None if ref is None else ref() is obj

        if previous is None or not _same(previous[0], transformer) or not _same(previous[1], text_encoder):
            self._flux2_lora_original_modules: Dict[str, torch.nn.Module] = {}
            self._flux2_lora_wrapped_modules: set = set()
            self._flux2_te_lora_wrapped: List[tuple] = []
            self._flux2_lora_component_refs = (
                weakref.ref(transformer) if transformer is not None else None,
                weakref.ref(text_encoder) if text_encoder is not None else None,
            )
        return self._flux2_lora_original_modules, self._flux2_lora_wrapped_modules

    def _load_lora_flux2(self, lora_configs: List[Dict]):
        """Load LoRAs for the FLUX.2 Transformer and Qwen3 text encoder

        Args:
            lora_configs: List of LoRA configurations

        Note:
            FLUX.2 uses component-based architecture (not pipeline-based).
            LoRAs wrap original linear layers (forward-time addition, not weight merging).
            This allows LoRAs to be unloaded by restoring original modules.
            Both key codecs come from the training adapter (flux2_adapter.py):
            ``lora_transformer_*`` for the transformer and ``lora_te_*`` for the
            Qwen3 encoder (written whenever a run set ``train_text_encoder``).

            FLUX.2 has two block types:
            1. Dual stream blocks: Flux2Attention (to_q, to_k, to_v, to_out[0], add_q_proj, add_k_proj, add_v_proj, to_add_out)
            2. Single stream blocks: Flux2ParallelSelfAttention (to_qkv_mlp_proj, to_out)

            ``apply_to_unet`` / ``apply_to_text_encoder`` select the components;
            a file whose only tensors belong to a disabled component applies zero
            modules and is a warning, while an ENABLED component that carries
            keys and matches nothing is an error, per component rather than on
            the sum (a base-model image must not be returned as a LoRA one).
        """
        # Unconditional, and BEFORE the empty-config exit: state left over from a
        # previously loaded model must never be restored into this one.
        self._flux2_lora_state()

        if not lora_configs:
            return

        if not self.flux2_components:
            print("[FLUX.2 LoRA] WARNING: FLUX.2 components not loaded")
            return

        transformer = self.flux2_components["transformer"]
        text_encoder = self.flux2_components.get("text_encoder")

        # Use global lora_manager instance (has user-configured additional_dirs)
        from core.extensions.lora_manager import lora_manager

        print(f"[FLUX.2 LoRA] Loading {len(lora_configs)} LoRA(s)...")

        for i, lora_config in enumerate(lora_configs):
            lora_path = lora_config.get("path", "")
            # Warnings ride into the PNG metadata chunk, so never an absolute path.
            lora_file = os.path.basename(str(lora_path))
            lora_strength = lora_config.get("strength", 1.0)
            layer_weights = lora_config.get("unet_layer_weights", {})
            apply_to_unet = lora_config.get("apply_to_unet", True)
            apply_to_te = lora_config.get("apply_to_text_encoder", True)

            # Resolve path using LoRAManager
            resolved_path = lora_manager._resolve_lora_path(lora_path)

            if resolved_path is None:
                message = (
                    f"LoRA '{lora_file}' was requested but no such file exists in the "
                    f"registered LoRA directories -- refusing to generate without it.")
                print(f"[FLUX.2 LoRA] ERROR: {message}")
                print(f"[FLUX.2 LoRA]   Searched in: {lora_manager.lora_dir}")
                print(f"[FLUX.2 LoRA]   Additional dirs: {lora_manager.additional_dirs}")
                self._flux2_lora_warn(message, code="lora_not_found")
                raise FileNotFoundError(message)

            print(f"[FLUX.2 LoRA] Loading LoRA {i+1}/{len(lora_configs)}: {lora_path} (strength={lora_strength})")
            if layer_weights:
                print(f"[FLUX.2 LoRA] Layer weights: {layer_weights}")
            if not apply_to_unet or not apply_to_te:
                print(f"[FLUX.2 LoRA] Components: transformer={apply_to_unet}, text_encoder={apply_to_te}")

            # Load LoRA weights
            from safetensors import safe_open

            try:
                with safe_open(str(resolved_path), framework="pt", device="cpu") as f:
                    lora_state_dict = {key: f.get_tensor(key) for key in f.keys()}

                print(f"[FLUX.2 LoRA] Loaded {len(lora_state_dict)} tensors from {lora_path}")

                # Per component, never summed: see the refusal gate below.
                unet_counts: Dict[str, int] = {}
                te_counts: Dict[str, int] = {}

                # Debug: Print first few LoRA keys
                lora_keys_sample = list(lora_state_dict.keys())[:5]
                print(f"[FLUX.2 LoRA] Sample LoRA keys: {lora_keys_sample}")

                unet_keys_present = any(k.startswith("lora_transformer_") for k in lora_state_dict)
                te_keys_present = any(k.startswith("lora_te_") for k in lora_state_dict)

                # Debug: Print module class names found
                module_classes_found = set()
                for name, module in transformer.named_modules():
                    module_classes_found.add(module.__class__.__name__)
                print(f"[FLUX.2 LoRA] Module classes in transformer: {module_classes_found}")

                # Target sets per block class; the key stem is the module path with
                # dots replaced, exactly as flux2_adapter writes it.
                targets_by_class = {
                    "Flux2Attention": ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj",
                                       "add_v_proj", "to_add_out"],
                    "Flux2ParallelSelfAttention": ["to_qkv_mlp_proj", "to_out"],
                    "Flux2FeedForward": ["linear_in", "linear_out"],
                }

                walk = transformer.named_modules() if (apply_to_unet and unet_keys_present) else ()
                for name, module in walk:
                    attrs = targets_by_class.get(module.__class__.__name__)
                    if attrs is None:
                        continue

                    block_weight = layer_weights.get(self._get_flux2_block_name(name), 1.0)
                    effective_strength = lora_strength * block_weight
                    stem = f"lora_transformer_{name.replace('.', '_')}"

                    for attr_name in attrs:
                        target = getattr(module, attr_name, None)
                        if target is None or isinstance(target, torch.nn.ModuleList):
                            continue
                        self._flux2_apply_lora_branch(
                            unet_counts, module, attr_name, target, f"{name}.{attr_name}",
                            lora_state_dict, f"{stem}_{attr_name}", effective_strength)

                    # to_out as a ModuleList (Flux2Attention); the parallel block's
                    # to_out is a plain Linear and was handled above.
                    to_out = getattr(module, "to_out", None)
                    if isinstance(to_out, torch.nn.ModuleList) and len(to_out) > 0:
                        self._flux2_apply_lora_branch(
                            unet_counts, to_out, 0, to_out[0], f"{name}.to_out.0",
                            lora_state_dict, f"{stem}_to_out_0", effective_strength)

                if te_keys_present and apply_to_te:
                    te_counts = self._apply_lora_to_flux2_text_encoder(
                        text_encoder, lora_state_dict, lora_strength)
                    print(f"[FLUX.2 LoRA] Applied LoRA to {te_counts.get('applied', 0)} "
                          f"Qwen3 text encoder modules")

            except Exception as e:
                print(f"[FLUX.2 LoRA] ERROR: Failed to load LoRA {lora_file}: {e}")
                import traceback
                traceback.print_exc()
                # Type + basename only: this rides into the PNG text chunk and the API
                # response, and an OSError's str() carries the absolute resolved path.
                message = (f"FLUX.2 LoRA '{lora_file}' could not be applied "
                           f"({type(e).__name__}); see the server log for details")
                self._flux2_lora_warn(message, code="lora_load_failed")
                raise RuntimeError(message) from e

            if not unet_keys_present and not te_keys_present:
                message = (f"LoRA '{lora_file}': no FLUX.2 LoRA tensors found (expected "
                           f"SushiUI-trained 'lora_transformer_*' and/or 'lora_te_*' keys). "
                           f"Sample keys in file: {lora_keys_sample}")
                print(f"[FLUX.2 LoRA] ERROR: {message}")
                self._flux2_lora_warn(message, code="lora_incompatible")
                raise RuntimeError(message)

            unet_pairs = sum(1 for k in lora_state_dict
                             if k.startswith("lora_transformer_") and k.endswith(".lora_down.weight"))
            te_pairs = sum(1 for k in lora_state_dict
                           if k.startswith("lora_te_") and k.endswith(".lora_down.weight"))
            applied_count = unet_counts.get("applied", 0) + te_counts.get("applied", 0)
            print(f"[FLUX.2 LoRA] Applied LoRA to {applied_count} modules "
                  f"(transformer {unet_counts.get('applied', 0)}/{unet_pairs}, "
                  f"text encoder {te_counts.get('applied', 0)}/{te_pairs})")

            components = (("transformer", apply_to_unet, unet_keys_present, unet_pairs, unet_counts),
                          ("text encoder", apply_to_te, te_keys_present, te_pairs, te_counts))

            # Per component, NOT on the sum: a file whose transformer half matches
            # nothing must not pass because its text-encoder half applied.
            dead = [(label, pairs, counts) for label, enabled, present, pairs, counts in components
                    if enabled and present and not counts.get("applied", 0)]
            if dead:
                detail = "; ".join(
                    f"{label}: 0 of {pairs} down/up pairs applied "
                    f"({counts.get('already_wrapped', 0)} already wrapped, "
                    f"{counts.get('shape_mismatch', 0)} shape mismatch)"
                    for label, pairs, counts in dead)
                if any(counts.get("already_wrapped", 0) for _, _, counts in dead):
                    # This backend replaces the target Linear, so a second file could
                    # only overwrite the first; stacking needs the composite wrapper
                    # (LYCORIS_ADAPTER_DESIGN Phase 1).
                    message = (f"LoRA '{lora_file}': {detail}. The targets are already wrapped "
                               f"by an earlier LoRA in this request; FLUX.2 applies one LoRA at "
                               f"a time, so select a single FLUX.2 LoRA.")
                    code = "lora_stacking_unsupported"
                else:
                    message = (f"LoRA '{lora_file}': {detail}. Unrecognized key format or a "
                               f"different model. Sample keys in file: {lora_keys_sample}")
                    code = "lora_incompatible"
                print(f"[FLUX.2 LoRA] ERROR: {message}")
                self._flux2_lora_warn(message, code=code)
                raise RuntimeError(message)

            if applied_count == 0:
                # Every tensor the file carries belongs to a component the request
                # switched off: an explicit no-op, not a failure.
                message = (f"LoRA '{lora_file}' applied 0 modules: its tensors target only the "
                           f"component(s) disabled by apply_to_unet={apply_to_unet} / "
                           f"apply_to_text_encoder={apply_to_te}")
                print(f"[FLUX.2 LoRA] WARNING: {message}")
                self._flux2_lora_warn(message, code="lora_no_targets")
            else:
                skipped = "; ".join(
                    f"{label}: applied {counts.get('applied', 0)} of {pairs} down/up pairs "
                    f"({counts.get('shape_mismatch', 0)} shape mismatch, "
                    f"{counts.get('already_wrapped', 0)} already wrapped)"
                    for label, enabled, present, pairs, counts in components
                    if enabled and present and counts.get("applied", 0) < pairs)
                if skipped:
                    self._flux2_lora_warn(f"LoRA '{lora_file}': {skipped}.", code="lora_partial")

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

    def _apply_lora_to_flux2_text_encoder(self, text_encoder, lora_state_dict, strength) -> Dict[str, int]:
        """Apply one LoRA's ``lora_te_*`` tensors to the Qwen3 text encoder.

        Key codec is exactly the one flux2_adapter.apply_lora_to_text_encoders
        writes: ``lora_te_model_layers_{i}_{mlp|self_attn}_{proj}``, where ``i``
        is the position in ``text_encoder.model.layers`` (the adapter enumerates
        the same list, so the index is the shared identifier -- there is no
        dotted module path in the key).

        Strength is the plain request strength: unet_layer_weights is a
        transformer-block map and has no text-encoder counterpart.

        Returns the per-status counts (see ``_flux2_apply_lora_branch``).
        """
        counts: Dict[str, int] = {}
        if text_encoder is None:
            print("[FLUX.2 LoRA] WARNING: text encoder not loaded; lora_te_* tensors skipped")
            return counts

        layers = None
        if hasattr(text_encoder, "model") and hasattr(text_encoder.model, "layers"):
            layers = text_encoder.model.layers
        elif hasattr(text_encoder, "layers"):
            layers = text_encoder.layers
        if layers is None:
            print("[FLUX.2 LoRA] WARNING: could not find Qwen3 layers; lora_te_* tensors skipped")
            return counts

        groups = (("mlp", ("gate_proj", "up_proj", "down_proj")),
                  ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")))

        for layer_idx, layer in enumerate(layers):
            for group, attrs in groups:
                parent = getattr(layer, group, None)
                if parent is None:
                    continue
                for attr in attrs:
                    original_linear = getattr(parent, attr, None)
                    if original_linear is None:
                        continue
                    module_key = f"text_encoder.model.layers.{layer_idx}.{group}.{attr}"
                    status = self._flux2_apply_lora_branch(
                        counts, parent, attr, original_linear, module_key, lora_state_dict,
                        f"lora_te_model_layers_{layer_idx}_{group}_{attr}", strength,
                    )
                    if status == "applied":
                        self._flux2_te_lora_wrapped.append((parent, attr, module_key))

        return counts

    def _restore_flux2_te_lora(self) -> int:
        """Put the Qwen3 text encoder's original Linears back. Idempotent."""
        wrapped = getattr(self, "_flux2_te_lora_wrapped", None)
        if not wrapped:
            return 0

        originals = getattr(self, "_flux2_lora_original_modules", {})
        live = getattr(self, "_flux2_lora_wrapped_modules", set())
        restored = 0
        for parent, attr, module_key in reversed(wrapped):
            original = originals.get(module_key)
            if original is None:
                continue
            setattr(parent, attr, original)
            live.discard(module_key)
            restored += 1
        wrapped.clear()
        if restored:
            print(f"[FLUX.2 LoRA] Restored {restored} Qwen3 text encoder modules")
        return restored

    def _flux2_te_quantization_with_lora(self, text_encoder_quantization):
        """Drop text-encoder quantization for a run that has a text-encoder LoRA.

        ``_quantize_text_encoder`` deep-copies the encoder and casts every
        ``nn.Linear`` weight to FP8 -- which, on a wrapped encoder, includes the
        wrapper's own lora_down/lora_up and so quantizes the ADAPTER (~2.6e-02
        relative error on e4m3) rather than only the base.
        """
        if not text_encoder_quantization or text_encoder_quantization == "none":
            return text_encoder_quantization
        if not getattr(self, "_flux2_te_lora_wrapped", None):
            return text_encoder_quantization

        msg = (f"FLUX.2 text encoder quantization '{text_encoder_quantization}' was ignored: "
               f"a text-encoder LoRA is applied, and the quantizer would cast the adapter's "
               f"own weights to FP8 along with the base. The Qwen3 encoder runs at its full "
               f"weight precision for this generation, so VRAM use is higher than requested")
        print(f"[FLUX.2 LoRA] {msg}")
        self._flux2_lora_warn(msg, code="quantization_fallback")
        return None

    def _flux2_apply_lora_branch(self, counts, parent_module, attr_name, original_linear,
                                 module_key, lora_state_dict, lora_name, strength) -> str:
        """Wrap one target with its branch from ``lora_state_dict``, tallying into
        ``counts``. Returns "applied", "absent", "already_wrapped", "not_linear"
        or "shape_mismatch".

        A shape mismatch is skipped rather than assigned: ``.data = tensor``
        replaces the parameter wholesale and would only fail later, inside text
        encoding or the denoise loop.
        """
        from core.training.adapters.sd15_adapter import LoRALinearLayer
        # NOT ``isinstance(x, torch.nn.Linear)``: after a runtime INT8 conversion
        # (unet_quantization="int8") the very layers a LoRA targets are Int8Linear /
        # Fp8Linear, which are nn.Module but NOT nn.Linear subclasses.
        from core.training.adapters.base_adapter import is_lora_wrappable_linear

        def _tally(status):
            counts[status] = counts.get(status, 0) + 1
            return status

        down = lora_state_dict.get(f"{lora_name}.lora_down.weight")
        up = lora_state_dict.get(f"{lora_name}.lora_up.weight")
        if down is None or up is None:
            return _tally("absent")
        if isinstance(original_linear, LoRALinearLayer):
            return _tally("already_wrapped")
        if not is_lora_wrappable_linear(original_linear):
            return _tally("not_linear")

        in_features = getattr(original_linear, "in_features", None)
        out_features = getattr(original_linear, "out_features", None)
        if (down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]
                or down.shape[1] != in_features or up.shape[0] != out_features):
            print(f"[FLUX.2 LoRA] WARNING: shape mismatch at {module_key}: "
                  f"down{tuple(down.shape)} up{tuple(up.shape)} vs Linear"
                  f"({in_features} -> {out_features}); skipping this module")
            return _tally("shape_mismatch")

        self._wrap_with_lora_flux2(
            parent_module, attr_name, original_linear, down, up, strength,
            lora_state_dict.get(f"{lora_name}.alpha"), module_key)
        return _tally("applied")

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

        # Load pretrained weights.
        # The LoRA branch computes in the BASE weight's dtype -- except when the
        # base is weight-only quantized, where that dtype is int8 or e4m3 and
        # copying the adapter into it would quantize the adapter itself (int8:
        # 254 levels over its own amax; e4m3: ~2.6e-02 relative error). Both
        # quantized bases produce bf16 from a bf16 activation, so the branch uses
        # bf16 too. Same rule as krea2_lora.apply_lora_group.
        from core.training.adapters.base_adapter import lora_branch_dtype

        device = true_original.weight.device
        dtype = lora_branch_dtype(true_original)

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
        """Unload LoRAs from the FLUX.2 Transformer and Qwen3 text encoder"""
        # Unconditional and first: after a model switch these maps hold the OLD
        # model's Linears, and restoring them here would splice them into the new one.
        originals, _wrapped = self._flux2_lora_state()

        # Text encoder first: it is normally already restored by _flux2_cleanup's
        # finally, so this is the fallback path (and the only one for a caller
        # that unloads outside a generation).
        self._restore_flux2_te_lora()

        if not originals:
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

    def _flux2_cleanup(self, gen_succeeded=True, keep_te=False, keep_transformer=False, keep_vae=False):
        """Safety-net CPU offload for FLUX.2 components.

        On the happy path each generate function already offloads text_encoder,
        transformer and vae to CPU inline, and tears down its block-swap offloader.
        This helper is called from a `finally` block in every generate entry point
        so that an exception raised mid-generation (denoise loop, VAE decode, etc.)
        cannot leave the transformer (largest component) or TE resident on GPU.

        Idempotent: re-running after the happy-path cleanup is a cheap no-op
        (.to("cpu") on an already-CPU module, offloader.cleanup() on an
        already-cleaned-up offloader). Never raises - a failure here must not
        mask the original exception from the caller's try/except.

        Keep-models-hot (see core/keep_hot.py): ``gen_succeeded`` and the
        ``keep_*`` flags let a successful generation that opted into
        keep_models_hot skip forcing a component back to CPU here when the
        caller already decided (and is tracking via keep_hot.mark_resident)
        that it should stay GPU-resident for the next generation. On any
        failed generation (``gen_succeeded=False``) the keep flags are
        ignored entirely and every component is force-offloaded, exactly as
        before this feature existed -- never trust GPU residency after an
        exception. Defaults are chosen so a call with no arguments reproduces
        the pre-keep-hot behavior (force-offload everything).

        The block-swap offloader teardown is NEVER conditioned on the keep
        flags: ``keep_transformer`` can only be True when block swap was not
        active for this generation (see the keep-hot eligibility gate in each
        generate function), so whenever the transformer is kept hot,
        ``_flux2_active_block_offloader`` is already None here and this
        teardown is a no-op -- the two states never coexist.
        """
        try:
            # Text-encoder LoRA wrappers are per-generation state, restored here
            # because this helper IS the finally of every generate entry point.
            # The transformer's wrappers keep their existing longer lifetime
            # (unloaded by the next generation's _unload_lora_flux2).
            try:
                self._restore_flux2_te_lora()
            except Exception as e:
                print(f"[FLUX.2] cleanup: text encoder LoRA restore failed: {e}")

            offloader = getattr(self, "_flux2_active_block_offloader", None)
            if offloader is not None:
                try:
                    offloader.cleanup()
                except Exception as e:
                    print(f"[FLUX.2] cleanup: block offloader teardown failed: {e}")
                self._flux2_active_block_offloader = None

            # Style-transfer attention processors: on the happy path these are already
            # restored in the generate function's try body (which also clears this attr
            # to None). If an exception fired mid-denoise, restore is skipped there, so
            # this safety net catches it here -- otherwise the style-stamped processors
            # would leak onto the persistent transformer and silently affect the NEXT
            # generation (see style_flux2 module docstring).
            style_saved = getattr(self, "_flux2_active_style_saved", None)
            if style_saved:
                try:
                    from core.inference.style_flux2 import restore_flux2_style_processors
                    restore_flux2_style_processors(style_saved)
                except Exception as e:
                    print(f"[FLUX.2] cleanup: style processor restore failed: {e}")
                self._flux2_active_style_saved = None

            _kh_skip = set()
            if gen_succeeded:
                if keep_te:
                    _kh_skip.add("text_encoder")
                if keep_transformer:
                    _kh_skip.add("transformer")
                if keep_vae:
                    _kh_skip.add("vae")

            components = getattr(self, "flux2_components", None) or {}
            for key in ("text_encoder", "transformer", "vae"):
                if key in _kh_skip:
                    continue
                comp = components.get(key)
                if comp is None:
                    continue
                try:
                    comp.to("cpu")
                except Exception as e:
                    print(f"[FLUX.2] cleanup: failed to offload {key} to CPU: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[FLUX.2] cleanup: unexpected error during safety-net cleanup: {e}")

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

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        # FLUX.2 has no cpu_text_encoding option (the text encoder always runs on
        # GPU), so unlike SD1.5/SDXL there is no is_cpu_inference gate for TE here.
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 0) or 0) > 0

        def _kh_offload_flux2():
            comps = getattr(self, "flux2_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_flux2)

        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.flux2_components.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(self.flux2_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.flux2_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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
            text_encoder_quantization = self._flux2_te_quantization_with_lora(text_encoder_quantization)

            # ============================================================
            # Stage 1: Text Encoding (Qwen3)
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            if not is_resident(self, "text_encoder", _kh_model_key):
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

            # Offload text encoder to CPU (unless kept hot -- TE is not touched
            # again in this generation, so this is also TE's keep-hot exit point;
            # see core/keep_hot.py).
            if not _kh_keep_te:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            # Style transfer and Image-Edit reference images are mutually exclusive
            # (see core.inference.style_flux2 module docstring) -- style takes
            # precedence and ref_images is dropped for this generation when both
            # are requested.
            style_requested = bool((params.get("style_transfer") or {}).get("image"))
            # Style transfer's attention hook only replaces Flux2AttnProcessor /
            # ConduitFlux2AttnProcessor instances (see style_flux2 module docstring). If
            # NAG or NegPip already swapped in their own processor/wrapper, style would
            # silently no-op (its hook never sees the batch) while the NAG/NegPip machinery
            # still ran -- so NAG/NegPip takes precedence and style is dropped explicitly.
            if style_requested and (nag_active or negpip_active):
                print("[FLUX.2] Style transfer requested: disabling (NAG/NegPip is active and "
                      "takes precedence) for this generation -- the two features are mutually exclusive.")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 style transfer disabled: NAG/NegPip is active",
                        code="style_disabled_by_nag_negpip",
                    )
                except Exception:
                    pass
                style_requested = False
            ref_images = params.get("ref_images", []) if not style_requested else []
            if style_requested and params.get("ref_images"):
                print("[FLUX.2] Style transfer requested: ignoring ref_images (Image-Edit) "
                      "for this generation -- the two features are mutually exclusive.")
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
            _t_denoise = _time.perf_counter()

            # One-time in-place INT8 conversion (unet_quantization="int8"). MUST be
            # here: before the block offloader is built (it captures the Linear
            # modules this replaces) and before staging (move_flux2_transformer_to_gpu
            # is only reached in the no-block-swap branch below). No-op for every
            # other value and for an already-converted / already-quantized model.
            transformer = self._flux2_runtime_int8(
                params, transformer, progress_callback=progress_callback)

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
                # Track the active offloader on self so the finally-block safety net
                # (_flux2_cleanup) can tear it down even if an exception is raised
                # before the normal-path cleanup below runs.
                self._flux2_active_block_offloader = block_offloader

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
                if not is_resident(self, "transformer", _kh_model_key):
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
            # error), (b) Block Swap (a cache hit skips the block loops -> desyncs the
            # per-block swap rotation), and (c) style transfer (its capture-forward +
            # inject_kv steps would run through the FBCache wrappers at the same step_idx,
            # storing the REF pass's residual and corrupting the COND pass -- see
            # core.inference.style_flux2). Active only when ALL of these are off. When
            # active, we must route through the unified Flux2BlockSwapWrapper
            # (offloader=None) so its custom forward intercepts the dual+single block
            # loops; the raw diffusers forward (fast path) does not. If NAG/NegPip already
            # installed a wrapper above, reuse it.
            if style_requested:
                print("[FLUX.2] FBCache disabled: style transfer is active (capture-forward cache pollution)")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 FBCache disabled: style transfer is active",
                        code="style_disables_fbcache",
                    )
                except Exception:
                    pass
                fbcache = None
            else:
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

            # Training-free reference-style transfer setup (no-op / None when no
            # style reference / style reference list is attached -- byte-identical
            # default path below). Gated on style_requested, which the NAG/NegPip
            # precedence check above may already have forced to False (see Stage 1.5).
            # ``style_refs`` is populated (and style_cfg/style_ref_x0/style_eps_ref
            # left None) ONLY when ``params["style_transfers"]`` carries 2+
            # references -- a single reference (via either key) always resolves
            # through the style_cfg/style_ref_x0/style_eps_ref triple, so that
            # code path (both here and in the per-step branch below) is untouched.
            style_refs = None
            style_combine_mode = "stack"
            if style_requested:
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = self._flux2_style_configs(
                    params, transformer, height, width, self.device
                )
            else:
                style_cfg, style_ref_x0, style_eps_ref = None, None, None
            style_processors: List[Any] = []
            style_saved_processors: List[Any] = []
            if style_cfg is not None or style_refs is not None:
                from core.attention import AttentionMode, normalize_backend
                from core.inference.style_flux2 import install_flux2_style_processors
                style_canonical_backend = normalize_backend(params.get("attention_type", settings.attention_type))
                style_processors, style_saved_processors = install_flux2_style_processors(
                    transformer, style_canonical_backend, AttentionMode.INFERENCE
                )
                print(f"[FLUX.2] Style transfer active: {len(style_processors)} attention modules stamped")
                # Stash for _flux2_cleanup's exception safety net (see Bug 1): if an
                # exception fires mid-denoise, the happy-path restore below (in the try
                # body) is skipped, and this attr tells cleanup to restore instead. On the
                # happy path this is cleared back to None right after the in-try restore.
                self._flux2_active_style_saved = style_saved_processors
                # CFG-decoupled style guidance (style_guidance_scale) needs a real
                # uncond/cond CFG split to decouple lambda from (see
                # _flux2_style_step); a distilled model (do_classifier_free_guidance
                # False) has no uncond pass at all, so the knob is a silent no-op
                # there. Single-ref only (style_cfg is None on the multi-ref path).
                if (
                    style_cfg is not None
                    and style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and not do_classifier_free_guidance
                ):
                    print("[FLUX.2] style_guidance_scale has no effect: model is distilled "
                          "(no classifier-free guidance split to decouple from)")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 style_guidance_scale ignored: distilled model has no CFG split",
                            code="style_guidance_scale_needs_cfg",
                        )
                    except Exception:
                        pass

            # Denoising loop
            # Spectrum output-mode acceleration (forecast per-step model output). Also
            # yields to style transfer: Spectrum records the final noise_pred and skips
            # transformer+CFG on forecast steps, which would starve the style-active steps
            # of the REF/COND/UNCOND forwards _flux2_style_step depends on.
            spectrum = None
            if params.get("spectrum_enable", False):
                if style_requested:
                    print("[FLUX.2] Spectrum disabled: style transfer is active")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 Spectrum disabled: style transfer is active",
                            code="style_disables_spectrum",
                        )
                    except Exception:
                        pass
                else:
                    from core.inference.spectrum_forecaster import build_output_forecaster
                    spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            total_steps = len(timesteps)
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

                    if style_refs is not None:
                        # Multi-reference (N>1): step-active if ANY ref's own
                        # StyleTransferConfig is step-active (mirrors the
                        # single-ref gate below, applied per-ref instead of
                        # globally -- see _flux2_style_step_multi).
                        style_active_step = any(
                            cfg_i.is_step_active(i, total_steps) for cfg_i, _, _ in style_refs
                        )
                    else:
                        style_active_step = style_cfg is not None and style_cfg.is_step_active(i, total_steps)
                    if style_active_step:
                        # Training-free reference-style transfer: bypasses the Image-Edit
                        # ref-token concat + batched-CFG fast path below (mutually exclusive
                        # with NAG/NegPip/FBCache -- see core.inference.style_flux2).
                        style_guidance_vec = None
                        if not do_classifier_free_guidance:
                            style_guidance_vec = torch.full(
                                (latents.shape[0],), guidance_scale,
                                device=latents.device, dtype=transformer_input_dtype,
                            )
                        if style_refs is not None:
                            noise_pred = self._flux2_style_step_multi(
                                transformer_wrapper, style_refs, style_combine_mode, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                        else:
                            noise_pred = self._flux2_style_step(
                                transformer_wrapper, style_cfg, style_ref_x0, style_eps_ref, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                    else:
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
                self._flux2_active_block_offloader = None
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            if style_saved_processors:
                from core.inference.style_flux2 import restore_flux2_style_processors
                restore_flux2_style_processors(style_saved_processors)
                # Happy path: already restored above, so clear the exception-safety-net
                # attr (set at install time) to make _flux2_cleanup's finally-block
                # restore a no-op (see Bug 1 fix in _flux2_cleanup).
                self._flux2_active_style_saved = None
            # Offload transformer to CPU (unless kept hot -- only possible when
            # block swap was NOT active this generation, see keep-hot setup above;
            # this is also transformer's keep-hot exit point, it is not touched
            # again in this generation).
            if not _kh_keep_transformer:
                transformer.to("cpu")
                torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 4: VAE Decode
            # ============================================================
            generation_timer.add("denoise", _time.perf_counter() - _t_denoise)
            print("[FLUX.2] Stage 4: VAE decoding...")
            _t_decode = _time.perf_counter()
            if not is_resident(self, "vae", _kh_model_key):
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
            _cf = getattr(self, "_color_flatten_strength", 0)
            if _cf and _cf > 0:
                from core.inference.color_flatten import flatten_chroma
                image = flatten_chroma(image, _cf)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Offload VAE to CPU (unless kept hot -- this is VAE's keep-hot exit
            # point for this generation)
            if not _kh_keep_vae:
                vae.to("cpu")
                torch.cuda.empty_cache()

            generation_timer.add("vae_decode", _time.perf_counter() - _t_decode)
            print("[FLUX.2] Generation completed")
            _kh_gen_succeeded = True
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 generation failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._flux2_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )

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
        _t_phase = _time.perf_counter()
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

        generation_timer.add("text_encode", _time.perf_counter() - _t_phase)
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

    def _flux2_prepare_style_reference(self, style_image: Image.Image, height: int, width: int, device) -> torch.Tensor:
        """VAE-encode + patchify + BatchNorm-normalize a style reference image to the
        EXACT SAME packed-token layout as the target latents (same ``height``/``width``
        -> same grid), reusing the target's own ``latent_ids``/``img_ids`` (NOT the
        Image-Edit ``ref_ids`` scheme, whose separate rope "time" axis offset would
        desync the reference's positions from the target -- StyleAligned-style transfer
        needs the reference at the SAME rope positions as the target it's stylizing).
        Mirrors the encode steps in ``encode_flux2_image_refs`` (patchify + BN norm)
        without that method's multi-image / separate-position-id machinery."""
        vae = self.flux2_components["vae"]
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        img = style_image.convert("RGB").resize((int(width), int(height)), Image.LANCZOS)
        import numpy as np
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - 0.5) * 2.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latent = vae.encode(img_tensor).latent_dist.mode()
            latent = self._flux2_patchify_latents(latent)
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
            latents_bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
            ).to(latent.device, latent.dtype)
            latent = (latent - latents_bn_mean) / latents_bn_std

        ref_x0 = self._flux2_pack_latents(latent).to(device=device, dtype=torch.float32)
        return ref_x0

    def _flux2_style_triple(
        self, style_dict: Dict[str, Any], transformer, height: int, width: int, device, seed, ref_index: int = 0,
    ):
        """Build a single (StyleTransferConfig, ref_x0, eps_ref) triple from one
        style_transfer dict. ``axes_dims`` is filled in from the loaded
        transformer's own RoPE config (``axes_dims_rope``, default
        ``(32, 32, 32, 32)`` -- sums to ``attention_head_dim`` == 128).

        ``ref_index`` decorrelates the fixed re-noising noise tensor across
        multiple simultaneous references (each ref would otherwise draw the
        EXACT same noise from the ``seed+991`` offset, since that offset does
        not depend on which reference is being prepared). ``ref_index=0``
        (the default, used by the single-ref path) reproduces the pre-multi-ref
        ``seed+991`` offset exactly.
        """
        from diffusers.utils.torch_utils import randn_tensor
        from core.inference.reference_style import style_config_from_dict

        cfg = style_config_from_dict(style_dict)
        transformer_config = getattr(transformer, "config", None)
        axes_dims = tuple(transformer_config.axes_dims_rope) if transformer_config is not None else (32, 32, 32, 32)
        cfg.axes_dims = axes_dims

        ref_x0 = self._flux2_prepare_style_reference(style_dict["image"], height, width, device)

        ref_seed = None if seed is None or seed < 0 else (int(seed) + 991 + ref_index) % (2**32)
        generator = torch.Generator(device=device).manual_seed(ref_seed) if ref_seed is not None else None
        eps_ref = randn_tensor(ref_x0.shape, generator=generator, device=device, dtype=ref_x0.dtype)
        return cfg, ref_x0, eps_ref

    def _flux2_style_config(self, params: Dict[str, Any], transformer, height: int, width: int, device):
        """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None, None)``
        when no style reference is attached. Single-reference path,
        BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_flux2_style_triple`` with ``ref_index=0``, which reproduces the
        original ``seed+991`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        seed = params.get("seed", -1)
        return self._flux2_style_triple(style_dict, transformer, height, width, device, seed, ref_index=0)

    def _flux2_style_configs(self, params: Dict[str, Any], transformer, height: int, width: int, device):
        """Build the full style-transfer configuration for FLUX.2 generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple, exactly as ``_flux2_style_config``
        would return) and the multi-reference path (``style_refs``, a list of
        per-ref triples, populated ONLY when ``params["style_transfers"]`` has
        more than one entry). A single-entry ``style_transfers`` list is
        intentionally routed through the single-ref triple instead (``style_refs``
        stays ``None``), so the pre-multi-ref code path executes byte-identically
        end to end.

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
                refs.append(
                    self._flux2_style_triple(style_dict, transformer, height, width, device, seed, ref_index=idx)
                )
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._flux2_style_config(params, transformer, height, width, device)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    def _flux2_style_step(
        self,
        transformer_wrapper,
        style_cfg,
        style_ref_x0: torch.Tensor,
        style_eps_ref: torch.Tensor,
        style_processors: List[Any],
        step_idx: int,
        total_steps: int,
        t,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        negative_prompt_embeds,
        negative_text_ids,
        latent_ids: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_vec,
        transformer_input_dtype,
    ) -> torch.Tensor:
        """One style-active denoise step for FLUX.2: a REF capture forward (the style
        reference re-noised to this step's CURRENT sigma, using the TARGET's own
        prompt_embeds/text_ids/latent_ids so the image-token offsets line up exactly)
        stashes post-RoPE image-token Q/K/V per block; the COND forward then reads/
        injects them via ``inject_kv``. The UNCOND forward (when CFG is active) is
        always run with the style context disarmed (untouched), matching the Krea2
        wiring. Bypasses the Image-Edit ref-token concatenation, NAG, NegPip and
        FBCache fast paths for this step (see ``core.inference.style_flux2`` module
        docstring for why); block swap still applies since ``transformer_wrapper`` is
        called unchanged. ``guidance_vec`` is ``None`` on the CFG path (the model's
        guidance-embed input is unused when true CFG is active) and the distilled
        guidance tensor otherwise -- passed to BOTH the capture and cond forward so
        the reference sees the same guidance-embed conditioning as the target.

        Noising convention (verified against this loop's own scheduler stepping,
        ``pipeline_backends/flux2.py``'s ``t_value = timesteps[0]/1000`` + linear
        interpolation): flow-matching ``x_t = (1 - sigma) * x0 + sigma * eps``,
        ``sigma = t / 1000``, matching Krea2's identical convention.
        """
        from core.inference.reference_style import StyleContext
        from core.inference.style_flux2 import set_flux2_style_context

        sigma_now = float(t.item()) / 1000.0
        ref_t = (1.0 - sigma_now) * style_ref_x0 + sigma_now * style_eps_ref
        progress = style_cfg.step_progress(step_idx, total_steps)

        text_seq_len = text_ids.shape[1]
        image_seq_len = latents.shape[1]
        timestep = t.expand(latents.shape[0]).to(transformer_input_dtype) / 1000

        capture_ctx = StyleContext(mode="capture", config=style_cfg, progress=progress)
        capture_ctx.img_start = text_seq_len
        capture_ctx.img_end = text_seq_len + image_seq_len
        set_flux2_style_context(style_processors, capture_ctx)
        with torch.no_grad():
            transformer_wrapper(
                hidden_states=ref_t.to(transformer_input_dtype),
                timestep=timestep,
                guidance=guidance_vec,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )

        inject_ctx = StyleContext(mode="inject", config=style_cfg, store=capture_ctx.store, progress=progress)
        inject_ctx.img_start = capture_ctx.img_start
        inject_ctx.img_end = capture_ctx.img_end
        set_flux2_style_context(style_processors, inject_ctx)
        with torch.no_grad():
            noise_pred_cond = transformer_wrapper(
                hidden_states=latents.to(transformer_input_dtype),
                timestep=timestep,
                guidance=guidance_vec,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
        set_flux2_style_context(style_processors, None)

        # --- CFG-decoupled style guidance (FLUX.2) --- see the SDXL prototype
        # (core.inference.custom_sampling) for the full derivation; identical
        # mechanism, adapted to this function's own combine below. Disabled by
        # default (style_cfg.style_guidance_scale is None/<=0): this block is
        # skipped and noise_pred_cond stays exactly the styled cond pred above
        # (cond_s) -- byte-identical to before this feature (zero extra forwards).
        #
        # Only applicable when ``do_classifier_free_guidance`` is True -- i.e. a
        # real uncond/cond CFG split exists for this generation (non-distilled
        # model, guidance_scale > 1.0). When the model is distilled
        # (``do_classifier_free_guidance`` False), FLUX.2 has no uncond pass at
        # all: ``guidance_scale`` feeds the transformer's own distilled
        # guidance-embed input instead of a classic CFG combine, so there is no
        # (uncond, cond) delta for a SECOND lambda-scaled delta to decouple
        # from -- style_guidance_scale has no defined effect there and is left
        # a no-op (warned about once at generation setup; see the style_cfg
        # setup block above the denoise loop).
        #
        # Enabled (>0) AND CFG-active: run one extra cond forward -- SAME
        # latents/timestep/guidance/encoder_hidden_states/txt_ids/img_ids as the
        # styled cond forward above -- with the style context already cleared
        # (line above) to get the cond prediction WITHOUT style (cond_ns), then
        # rewrite noise_pred_cond so the UNCHANGED combine below
        # (noise_pred = uncond + cfg*(cond - uncond)) reproduces the
        # style-guidance target:
        #   uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
        # Algebra: let cond' = cond_ns + (lambda/cfg)*(cond_s - cond_ns).
        # Substituting into the combine:
        #   uncond + cfg*(cond' - uncond)
        # = uncond + cfg*(cond_ns - uncond) + cfg*(lambda/cfg)*(cond_s - cond_ns)
        # = uncond + cfg*(cond_ns - uncond) + lambda*(cond_s - cond_ns)
        # which is exactly the target above -- so assigning
        # noise_pred_cond = cond' lets the untouched combine line below produce
        # style guidance decoupled from cfg. cfg is guarded (>1e-6) even though
        # do_classifier_free_guidance implies guidance_scale > 1.0 here; if it
        # were ever ~0 we skip the rewrite and keep noise_pred_cond = cond_s.
        if (
            do_classifier_free_guidance
            and style_cfg.style_guidance_scale is not None
            and style_cfg.style_guidance_scale > 0
        ):
            cond_s = noise_pred_cond
            with torch.no_grad():
                cond_ns = transformer_wrapper(
                    hidden_states=latents.to(transformer_input_dtype),
                    timestep=timestep,
                    guidance=guidance_vec,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            lam = style_cfg.style_guidance_scale
            if guidance_scale > 1e-6:
                noise_pred_cond = cond_ns + (lam / guidance_scale) * (cond_s - cond_ns)

        if do_classifier_free_guidance:
            with torch.no_grad():
                noise_pred_uncond = transformer_wrapper(
                    hidden_states=latents.to(transformer_input_dtype),
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        return noise_pred

    def _flux2_style_step_multi(
        self,
        transformer_wrapper,
        style_refs: List[Tuple[Any, torch.Tensor, torch.Tensor]],
        style_combine_mode: str,
        style_processors: List[Any],
        step_idx: int,
        total_steps: int,
        t,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        negative_prompt_embeds,
        negative_text_ids,
        latent_ids: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_vec,
        transformer_input_dtype,
    ) -> torch.Tensor:
        """Multi-reference (N>1) generalization of ``_flux2_style_step``: one REF
        capture forward PER reference (each with ITS OWN ``StyleTransferConfig``
        -- block_range, strengths, freq curve, step gating -- all independent),
        skipping refs that are not step-active at this step (mirrors the
        single-ref caller's ``style_cfg.is_step_active`` gate, applied per-ref
        instead of globally). The COND forward then reads ALL active refs'
        stores via a single ``StyleContext(mode="inject", refs=...,
        combine_mode=...)`` (see ``reference_style.StyleContext.collect_block_refs``
        / ``inject_kv_multi``). The UNCOND forward (CFG) is always run with the
        style context disarmed, same as the single-ref path. Only ever called
        when ``len(style_refs) > 1`` -- the denoise loop's ``style_refs is not
        None`` branch routes ``len(style_refs) <= 1`` through the byte-identical
        ``_flux2_style_step`` instead (see ``_flux2_style_configs``).
        """
        from core.inference.reference_style import StyleContext
        from core.inference.style_flux2 import set_flux2_style_context

        sigma_now = float(t.item()) / 1000.0
        text_seq_len = text_ids.shape[1]
        image_seq_len = latents.shape[1]
        timestep = t.expand(latents.shape[0]).to(transformer_input_dtype) / 1000

        active_refs = []
        overall_progress = 0.0
        for cfg_i, x0_i, eps_i in style_refs:
            if not cfg_i.is_step_active(step_idx, total_steps):
                continue
            progress_i = cfg_i.step_progress(step_idx, total_steps)
            overall_progress = progress_i
            ref_t = (1.0 - sigma_now) * x0_i + sigma_now * eps_i

            capture_ctx_i = StyleContext(mode="capture", config=cfg_i, progress=progress_i)
            capture_ctx_i.img_start = text_seq_len
            capture_ctx_i.img_end = text_seq_len + image_seq_len
            set_flux2_style_context(style_processors, capture_ctx_i)
            with torch.no_grad():
                transformer_wrapper(
                    hidden_states=ref_t.to(transformer_input_dtype),
                    timestep=timestep,
                    guidance=guidance_vec,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )
            active_refs.append((capture_ctx_i.store, cfg_i))

        if active_refs:
            inject_ctx = StyleContext(
                mode="inject", config=active_refs[0][1], refs=active_refs,
                combine_mode=style_combine_mode, progress=overall_progress,
            )
            inject_ctx.img_start = text_seq_len
            inject_ctx.img_end = text_seq_len + image_seq_len
            set_flux2_style_context(style_processors, inject_ctx)
        else:
            # No ref is step-active this step (mirrors the single-ref path's
            # ``style_active_step`` gate never even entering this function in
            # that case) -- run the cond forward with the style context
            # disarmed, i.e. a plain forward.
            set_flux2_style_context(style_processors, None)

        with torch.no_grad():
            noise_pred_cond = transformer_wrapper(
                hidden_states=latents.to(transformer_input_dtype),
                timestep=timestep,
                guidance=guidance_vec,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
        set_flux2_style_context(style_processors, None)

        if do_classifier_free_guidance:
            with torch.no_grad():
                noise_pred_uncond = transformer_wrapper(
                    hidden_states=latents.to(transformer_input_dtype),
                    timestep=timestep,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        return noise_pred

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

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 0) or 0) > 0

        def _kh_offload_flux2():
            comps = getattr(self, "flux2_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_flux2)

        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.flux2_components.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(self.flux2_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.flux2_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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
            text_encoder_quantization = self._flux2_te_quantization_with_lora(
                params.get("text_encoder_quantization"))

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            if not is_resident(self, "text_encoder", _kh_model_key):
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

            # Offload text encoder to CPU (unless kept hot -- TE is not touched
            # again in this generation, so this is also TE's keep-hot exit point;
            # see core/keep_hot.py).
            if not _kh_keep_te:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            # Style transfer and Image-Edit reference images are mutually exclusive
            # (see core.inference.style_flux2 module docstring) -- style takes
            # precedence and ref_images is dropped for this generation when both
            # are requested.
            style_requested = bool((params.get("style_transfer") or {}).get("image"))
            # Style transfer's attention hook only replaces Flux2AttnProcessor /
            # ConduitFlux2AttnProcessor instances (see style_flux2 module docstring). If
            # NAG or NegPip already swapped in their own processor/wrapper, style would
            # silently no-op (its hook never sees the batch) while the NAG/NegPip machinery
            # still ran -- so NAG/NegPip takes precedence and style is dropped explicitly.
            if style_requested and (nag_active or negpip_active):
                print("[FLUX.2] Style transfer requested: disabling (NAG/NegPip is active and "
                      "takes precedence) for this generation -- the two features are mutually exclusive.")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 style transfer disabled: NAG/NegPip is active",
                        code="style_disabled_by_nag_negpip",
                    )
                except Exception:
                    pass
                style_requested = False
            ref_images = params.get("ref_images", []) if not style_requested else []
            if style_requested and params.get("ref_images"):
                print("[FLUX.2] Style transfer requested: ignoring ref_images (Image-Edit) "
                      "for this generation -- the two features are mutually exclusive.")
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
            if not is_resident(self, "vae", _kh_model_key):
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

            # NOTE: this offload is a within-generation VRAM-relief step (VAE is
            # needed again for decode after denoising), not the keep-hot exit
            # boundary -- intentionally left unconditional; see core/keep_hot.py.
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
            _t_denoise = _time.perf_counter()

            # One-time in-place INT8 conversion (unet_quantization="int8"). MUST be
            # here: before the block offloader is built (it captures the Linear
            # modules this replaces) and before staging (move_flux2_transformer_to_gpu
            # is only reached in the no-block-swap branch below). No-op for every
            # other value and for an already-converted / already-quantized model.
            transformer = self._flux2_runtime_int8(
                params, transformer, progress_callback=progress_callback)

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
                # Track the active offloader on self so the finally-block safety net
                # (_flux2_cleanup) can tear it down even if an exception is raised
                # before the normal-path cleanup below runs.
                self._flux2_active_block_offloader = block_offloader
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
                if not is_resident(self, "transformer", _kh_model_key):
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
            # exclusive with Spectrum and Block Swap (see _flux2_build_fbcache), and also with
            # style transfer (its capture-forward + inject_kv steps would run through the
            # FBCache wrappers at the same step_idx, storing the REF pass's residual and
            # corrupting the COND pass -- see core.inference.style_flux2). When active, route
            # through the unified Flux2BlockSwapWrapper (offloader=None) so its custom
            # forward intercepts the dual+single block loops (the raw diffusers forward /
            # fast path does not). Reuse the NAG/NegPip wrapper if one was installed above.
            if style_requested:
                print("[FLUX.2] FBCache disabled: style transfer is active (capture-forward cache pollution)")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 FBCache disabled: style transfer is active",
                        code="style_disables_fbcache",
                    )
                except Exception:
                    pass
                fbcache = None
            else:
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

            # Training-free reference-style transfer setup (no-op / None when no
            # style reference / style reference list is attached -- byte-identical
            # default path below). Gated on style_requested, which the NAG/NegPip
            # precedence check above may already have forced to False (see Stage 1.5).
            # ``style_refs`` is populated (and style_cfg/style_ref_x0/style_eps_ref
            # left None) ONLY when ``params["style_transfers"]`` carries 2+
            # references -- a single reference (via either key) always resolves
            # through the style_cfg/style_ref_x0/style_eps_ref triple, so that
            # code path (both here and in the per-step branch below) is untouched.
            style_refs = None
            style_combine_mode = "stack"
            if style_requested:
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = self._flux2_style_configs(
                    params, transformer, height, width, self.device
                )
            else:
                style_cfg, style_ref_x0, style_eps_ref = None, None, None
            style_processors: List[Any] = []
            style_saved_processors: List[Any] = []
            if style_cfg is not None or style_refs is not None:
                from core.attention import AttentionMode, normalize_backend
                from core.inference.style_flux2 import install_flux2_style_processors
                style_canonical_backend = normalize_backend(params.get("attention_type", settings.attention_type))
                style_processors, style_saved_processors = install_flux2_style_processors(
                    transformer, style_canonical_backend, AttentionMode.INFERENCE
                )
                print(f"[FLUX.2] Style transfer active: {len(style_processors)} attention modules stamped")
                # Stash for _flux2_cleanup's exception safety net (see Bug 1): if an
                # exception fires mid-denoise, the happy-path restore below (in the try
                # body) is skipped, and this attr tells cleanup to restore instead. On the
                # happy path this is cleared back to None right after the in-try restore.
                self._flux2_active_style_saved = style_saved_processors
                # CFG-decoupled style guidance (style_guidance_scale) needs a real
                # uncond/cond CFG split to decouple lambda from (see
                # _flux2_style_step); a distilled model (do_classifier_free_guidance
                # False) has no uncond pass at all, so the knob is a silent no-op
                # there. Single-ref only (style_cfg is None on the multi-ref path).
                if (
                    style_cfg is not None
                    and style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and not do_classifier_free_guidance
                ):
                    print("[FLUX.2] style_guidance_scale has no effect: model is distilled "
                          "(no classifier-free guidance split to decouple from)")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 style_guidance_scale ignored: distilled model has no CFG split",
                            code="style_guidance_scale_needs_cfg",
                        )
                    except Exception:
                        pass

            # Spectrum output-mode acceleration (forecast per-step model output). Also
            # yields to style transfer: Spectrum records the final noise_pred and skips
            # transformer+CFG on forecast steps, which would starve the style-active steps
            # of the REF/COND/UNCOND forwards _flux2_style_step depends on.
            spectrum = None
            if params.get("spectrum_enable", False):
                if style_requested:
                    print("[FLUX.2] Spectrum disabled: style transfer is active")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 Spectrum disabled: style transfer is active",
                            code="style_disables_spectrum",
                        )
                    except Exception:
                        pass
                else:
                    from core.inference.spectrum_forecaster import build_output_forecaster
                    spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            total_steps = len(timesteps)
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

                    if style_refs is not None:
                        # Multi-reference (N>1): step-active if ANY ref's own
                        # StyleTransferConfig is step-active (mirrors the
                        # single-ref gate below, applied per-ref instead of
                        # globally -- see _flux2_style_step_multi).
                        style_active_step = any(
                            cfg_i.is_step_active(i, total_steps) for cfg_i, _, _ in style_refs
                        )
                    else:
                        style_active_step = style_cfg is not None and style_cfg.is_step_active(i, total_steps)
                    if style_active_step:
                        # Training-free reference-style transfer: bypasses the Image-Edit
                        # ref-token concat + batched-CFG fast path below (mutually exclusive
                        # with NAG/NegPip/FBCache -- see core.inference.style_flux2).
                        style_guidance_vec = None
                        if not do_classifier_free_guidance:
                            style_guidance_vec = torch.full(
                                (latents.shape[0],), guidance_scale,
                                device=latents.device, dtype=transformer_input_dtype,
                            )
                        if style_refs is not None:
                            noise_pred = self._flux2_style_step_multi(
                                transformer_wrapper, style_refs, style_combine_mode, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                        else:
                            noise_pred = self._flux2_style_step(
                                transformer_wrapper, style_cfg, style_ref_x0, style_eps_ref, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                    else:
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
                self._flux2_active_block_offloader = None
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            if style_saved_processors:
                from core.inference.style_flux2 import restore_flux2_style_processors
                restore_flux2_style_processors(style_saved_processors)
                # Happy path: already restored above, so clear the exception-safety-net
                # attr (set at install time) to make _flux2_cleanup's finally-block
                # restore a no-op (see Bug 1 fix in _flux2_cleanup).
                self._flux2_active_style_saved = None
            # Offload transformer to CPU (unless kept hot -- only possible when
            # block swap was NOT active this generation; also transformer's
            # keep-hot exit point, it is not touched again in this generation).
            if not _kh_keep_transformer:
                transformer.to("cpu")
                torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (img2img)
            # ============================================================
            generation_timer.add("denoise", _time.perf_counter() - _t_denoise)
            print("[FLUX.2] Stage 5: VAE decoding...")
            _t_decode = _time.perf_counter()
            # NOTE: VAE was already staged to GPU once for input-image encoding
            # (Stage 2) and unconditionally offloaded again there -- so this
            # reload always runs (never resident-skipped); see keep-hot NOTE above.
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
            _cf = getattr(self, "_color_flatten_strength", 0)
            if _cf and _cf > 0:
                from core.inference.color_flatten import flatten_chroma
                image = flatten_chroma(image, _cf)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Offload VAE to CPU (unless kept hot -- this is VAE's keep-hot exit
            # point for this generation)
            if not _kh_keep_vae:
                vae.to("cpu")
                torch.cuda.empty_cache()

            generation_timer.add("vae_decode", _time.perf_counter() - _t_decode)
            print("[FLUX.2] img2img generation completed")
            _kh_gen_succeeded = True
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] img2img error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 img2img failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._flux2_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )

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

        # ===== Keep-models-hot (opt-in queue optimization; see core/keep_hot.py) =====
        from core.keep_hot import (
            invalidate_if_model_changed, is_resident, mark_resident, clear_resident,
            discard_resident, should_keep_resident, compute_model_key, component_nbytes,
            keep_hot_requested,
        )
        _kh_requested = keep_hot_requested(params)
        _kh_model_key = compute_model_key(self, params)
        _kh_has_loras = bool(params.get("loras") or [])
        _kh_is_block_swapped = bool(params.get("enable_block_swap", False)) and int(params.get("blocks_to_swap", 0) or 0) > 0

        def _kh_offload_flux2():
            comps = getattr(self, "flux2_components", None) or {}
            for _kh_key in ("text_encoder", "transformer", "vae"):
                _kh_comp = comps.get(_kh_key)
                if _kh_comp is not None:
                    try:
                        _kh_comp.to("cpu")
                    except Exception:
                        pass

        invalidate_if_model_changed(self, params, offload_fn=_kh_offload_flux2)

        _kh_total_bytes = 0
        if _kh_requested:
            _kh_total_bytes += component_nbytes(self.flux2_components.get("text_encoder"))
            if not _kh_has_loras and not _kh_is_block_swapped:
                _kh_total_bytes += component_nbytes(self.flux2_components.get("transformer"))
            _kh_total_bytes += component_nbytes(self.flux2_components.get("vae"))
        _kh_guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=_kh_total_bytes,
        ) if _kh_requested else False
        _kh_keep_te = _kh_requested and _kh_guard_ok
        _kh_keep_transformer = _kh_requested and _kh_guard_ok and not _kh_has_loras and not _kh_is_block_swapped
        _kh_keep_vae = _kh_requested and _kh_guard_ok
        _kh_gen_succeeded = False

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
            text_encoder_quantization = self._flux2_te_quantization_with_lora(
                params.get("text_encoder_quantization"))

            # ============================================================
            # Stage 1: Text Encoding
            # ============================================================
            print("[FLUX.2] Stage 1: Text encoding...")
            if not is_resident(self, "text_encoder", _kh_model_key):
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

            # Offload text encoder to CPU (unless kept hot -- TE is not touched
            # again in this generation, so this is also TE's keep-hot exit point;
            # see core/keep_hot.py).
            if not _kh_keep_te:
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 1.5: Encode Reference Images (Image Edit)
            # ============================================================
            # Style transfer and Image-Edit reference images are mutually exclusive
            # (see core.inference.style_flux2 module docstring) -- style takes
            # precedence and ref_images is dropped for this generation when both
            # are requested.
            style_requested = bool((params.get("style_transfer") or {}).get("image"))
            # Style transfer's attention hook only replaces Flux2AttnProcessor /
            # ConduitFlux2AttnProcessor instances (see style_flux2 module docstring). If
            # NAG or NegPip already swapped in their own processor/wrapper, style would
            # silently no-op (its hook never sees the batch) while the NAG/NegPip machinery
            # still ran -- so NAG/NegPip takes precedence and style is dropped explicitly.
            if style_requested and (nag_active or negpip_active):
                print("[FLUX.2] Style transfer requested: disabling (NAG/NegPip is active and "
                      "takes precedence) for this generation -- the two features are mutually exclusive.")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 style transfer disabled: NAG/NegPip is active",
                        code="style_disabled_by_nag_negpip",
                    )
                except Exception:
                    pass
                style_requested = False
            ref_images = params.get("ref_images", []) if not style_requested else []
            if style_requested and params.get("ref_images"):
                print("[FLUX.2] Style transfer requested: ignoring ref_images (Image-Edit) "
                      "for this generation -- the two features are mutually exclusive.")
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
            if not is_resident(self, "vae", _kh_model_key):
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

            # NOTE: this offload is a within-generation VRAM-relief step (VAE is
            # needed again for decode after denoising), not the keep-hot exit
            # boundary -- intentionally left unconditional; see core/keep_hot.py.
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
            _t_denoise = _time.perf_counter()

            # One-time in-place INT8 conversion (unet_quantization="int8"). MUST be
            # here: before the block offloader is built (it captures the Linear
            # modules this replaces) and before staging (move_flux2_transformer_to_gpu
            # is only reached in the no-block-swap branch below). No-op for every
            # other value and for an already-converted / already-quantized model.
            transformer = self._flux2_runtime_int8(
                params, transformer, progress_callback=progress_callback)

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
                # Track the active offloader on self so the finally-block safety net
                # (_flux2_cleanup) can tear it down even if an exception is raised
                # before the normal-path cleanup below runs.
                self._flux2_active_block_offloader = block_offloader
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
                if not is_resident(self, "transformer", _kh_model_key):
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
            # exclusive with Spectrum and Block Swap (see _flux2_build_fbcache), and also with
            # style transfer (its capture-forward + inject_kv steps would run through the
            # FBCache wrappers at the same step_idx, storing the REF pass's residual and
            # corrupting the COND pass -- see core.inference.style_flux2). When active, route
            # through the unified Flux2BlockSwapWrapper (offloader=None) so its custom
            # forward intercepts the dual+single block loops (the raw diffusers forward /
            # fast path does not). Reuse the NAG/NegPip wrapper if one was installed above.
            if style_requested:
                print("[FLUX.2] FBCache disabled: style transfer is active (capture-forward cache pollution)")
                try:
                    from api.generation_status import add_warning
                    add_warning(
                        "FLUX.2 FBCache disabled: style transfer is active",
                        code="style_disables_fbcache",
                    )
                except Exception:
                    pass
                fbcache = None
            else:
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

            # Training-free reference-style transfer setup (no-op / None when no
            # style reference / style reference list is attached -- byte-identical
            # default path below). Gated on style_requested, which the NAG/NegPip
            # precedence check above may already have forced to False (see Stage 1.5).
            # ``style_refs`` is populated (and style_cfg/style_ref_x0/style_eps_ref
            # left None) ONLY when ``params["style_transfers"]`` carries 2+
            # references -- a single reference (via either key) always resolves
            # through the style_cfg/style_ref_x0/style_eps_ref triple, so that
            # code path (both here and in the per-step branch below) is untouched.
            style_refs = None
            style_combine_mode = "stack"
            if style_requested:
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = self._flux2_style_configs(
                    params, transformer, height, width, self.device
                )
            else:
                style_cfg, style_ref_x0, style_eps_ref = None, None, None
            style_processors: List[Any] = []
            style_saved_processors: List[Any] = []
            if style_cfg is not None or style_refs is not None:
                from core.attention import AttentionMode, normalize_backend
                from core.inference.style_flux2 import install_flux2_style_processors
                style_canonical_backend = normalize_backend(params.get("attention_type", settings.attention_type))
                style_processors, style_saved_processors = install_flux2_style_processors(
                    transformer, style_canonical_backend, AttentionMode.INFERENCE
                )
                print(f"[FLUX.2] Style transfer active: {len(style_processors)} attention modules stamped")
                # Stash for _flux2_cleanup's exception safety net (see Bug 1): if an
                # exception fires mid-denoise, the happy-path restore below (in the try
                # body) is skipped, and this attr tells cleanup to restore instead. On the
                # happy path this is cleared back to None right after the in-try restore.
                self._flux2_active_style_saved = style_saved_processors
                # CFG-decoupled style guidance (style_guidance_scale) needs a real
                # uncond/cond CFG split to decouple lambda from (see
                # _flux2_style_step); a distilled model (do_classifier_free_guidance
                # False) has no uncond pass at all, so the knob is a silent no-op
                # there. Single-ref only (style_cfg is None on the multi-ref path).
                if (
                    style_cfg is not None
                    and style_cfg.style_guidance_scale is not None
                    and style_cfg.style_guidance_scale > 0
                    and not do_classifier_free_guidance
                ):
                    print("[FLUX.2] style_guidance_scale has no effect: model is distilled "
                          "(no classifier-free guidance split to decouple from)")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 style_guidance_scale ignored: distilled model has no CFG split",
                            code="style_guidance_scale_needs_cfg",
                        )
                    except Exception:
                        pass

            # Spectrum output-mode acceleration (forecast per-step model output). Also
            # yields to style transfer: Spectrum records the final noise_pred and skips
            # transformer+CFG on forecast steps, which would starve the style-active steps
            # of the REF/COND/UNCOND forwards _flux2_style_step depends on.
            spectrum = None
            if params.get("spectrum_enable", False):
                if style_requested:
                    print("[FLUX.2] Spectrum disabled: style transfer is active")
                    try:
                        from api.generation_status import add_warning
                        add_warning(
                            "FLUX.2 Spectrum disabled: style transfer is active",
                            code="style_disables_spectrum",
                        )
                    except Exception:
                        pass
                else:
                    from core.inference.spectrum_forecaster import build_output_forecaster
                    spectrum = build_output_forecaster(params, len(timesteps), label="FLUX.2")
            total_steps = len(timesteps)
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

                    if style_refs is not None:
                        # Multi-reference (N>1): step-active if ANY ref's own
                        # StyleTransferConfig is step-active (mirrors the
                        # single-ref gate below, applied per-ref instead of
                        # globally -- see _flux2_style_step_multi).
                        style_active_step = any(
                            cfg_i.is_step_active(i, total_steps) for cfg_i, _, _ in style_refs
                        )
                    else:
                        style_active_step = style_cfg is not None and style_cfg.is_step_active(i, total_steps)
                    if style_active_step:
                        # Training-free reference-style transfer: bypasses the Image-Edit
                        # ref-token concat + batched-CFG fast path below (mutually exclusive
                        # with NAG/NegPip/FBCache -- see core.inference.style_flux2).
                        style_guidance_vec = None
                        if not do_classifier_free_guidance:
                            style_guidance_vec = torch.full(
                                (latents.shape[0],), guidance_scale,
                                device=latents.device, dtype=transformer_input_dtype,
                            )
                        if style_refs is not None:
                            noise_pred = self._flux2_style_step_multi(
                                transformer_wrapper, style_refs, style_combine_mode, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                        else:
                            noise_pred = self._flux2_style_step(
                                transformer_wrapper, style_cfg, style_ref_x0, style_eps_ref, style_processors,
                                i, total_steps, t, latents, prompt_embeds, text_ids,
                                negative_prompt_embeds, negative_text_ids, latent_ids,
                                do_classifier_free_guidance, guidance_scale, style_guidance_vec,
                                transformer_input_dtype,
                            )
                    else:
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
                self._flux2_active_block_offloader = None
            if nag_wrapper is not None:
                nag_wrapper.restore()  # restore original attention processors
            if negpip_wrapper is not None:
                negpip_wrapper.restore()  # restore original attention processors
            if style_saved_processors:
                from core.inference.style_flux2 import restore_flux2_style_processors
                restore_flux2_style_processors(style_saved_processors)
                # Happy path: already restored above, so clear the exception-safety-net
                # attr (set at install time) to make _flux2_cleanup's finally-block
                # restore a no-op (see Bug 1 fix in _flux2_cleanup).
                self._flux2_active_style_saved = None
            # Offload transformer to CPU (unless kept hot -- only possible when
            # block swap was NOT active this generation; also transformer's
            # keep-hot exit point, it is not touched again in this generation).
            if not _kh_keep_transformer:
                transformer.to("cpu")
                torch.cuda.empty_cache()

            # Clean up reference tokens/IDs (Image Edit)
            if ref_tokens is not None:
                del ref_tokens, ref_ids
                torch.cuda.empty_cache()

            # ============================================================
            # Stage 5: VAE Decode (inpaint)
            # ============================================================
            generation_timer.add("denoise", _time.perf_counter() - _t_denoise)
            print("[FLUX.2] Stage 5: VAE decoding...")
            _t_decode = _time.perf_counter()
            # NOTE: VAE was already staged to GPU once for input-image/mask
            # encoding (Stage 2) and unconditionally offloaded again there -- so
            # this reload always runs (never resident-skipped).
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
            _cf = getattr(self, "_color_flatten_strength", 0)
            if _cf and _cf > 0:
                from core.inference.color_flatten import flatten_chroma
                image = flatten_chroma(image, _cf)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image[0] * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Offload VAE to CPU (unless kept hot -- this is VAE's keep-hot exit
            # point for this generation)
            if not _kh_keep_vae:
                vae.to("cpu")
                torch.cuda.empty_cache()

            generation_timer.add("vae_decode", _time.perf_counter() - _t_decode)
            print("[FLUX.2] inpaint generation completed")
            _kh_gen_succeeded = True
            return pil_image, seed, actual_ancestral_seed

        except Exception as e:
            print(f"[FLUX.2] inpaint error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"FLUX.2 inpaint failed: {str(e)}")
        finally:
            if not _kh_gen_succeeded:
                clear_resident(self)
            else:
                if _kh_keep_te:
                    mark_resident(self, "text_encoder", _kh_model_key)
                else:
                    discard_resident(self, "text_encoder")
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    discard_resident(self, "transformer")
                if _kh_keep_vae:
                    mark_resident(self, "vae", _kh_model_key)
                else:
                    discard_resident(self, "vae")
            self._flux2_cleanup(
                gen_succeeded=_kh_gen_succeeded,
                keep_te=_kh_keep_te,
                keep_transformer=_kh_keep_transformer,
                keep_vae=_kh_keep_vae,
            )
