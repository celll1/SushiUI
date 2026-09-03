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

    # -- LoRA lifetime -------------------------------------------------------
    #
    # Owned by ``core.adapters.AdapterSession``: it resolves, parses and plans
    # every selected file against BOTH live transformers before mutating a slot,
    # then installs the whole request or none of it, and holds the weakref-keyed
    # bookkeeping -- one state per transformer, which is what makes a
    # cross-branch splice impossible once the restore order can vary (one shared
    # map only survives because the conditional restore pops every key before
    # the unconditional restore reads it). What stays here is only what is
    # Ideogram 4's: the key codec, how one branch is built, and the two refusal
    # texts whose wording is this architecture's.

    @staticmethod
    def _ideogram4_lora_warn(message: str, code: str) -> None:
        """Record a user-visible generation warning (best effort).

        The session cannot do this itself: ``core.adapters`` may not import
        ``api`` (``adapter_layering_test``), so the backend passes this in as the
        warning callback.
        """
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    @staticmethod
    def _ideogram4_resolve_lora_path(raw_path):
        from core.extensions.lora_manager import lora_manager

        return lora_manager._resolve_lora_path(raw_path)

    @staticmethod
    def _ideogram4_missing_lora(name, _raw_path):
        """Ideogram 4's own wording for an unresolvable path; the session logs,
        warns and raises it."""
        from core.adapters import AdapterFileMissing

        return AdapterFileMissing(
            f"Ideogram 4 LoRA '{name}' not found in any configured LoRA directory")

    @staticmethod
    def _ideogram4_declared_branches(tensors, _components) -> int:
        from core.models.ideogram4.ideogram4_lora import count_declared_pairs

        return count_declared_pairs(tensors)

    @property
    def _ideogram4_lora_session(self):
        """The per-backend session, created on first use.

        The mixin has no ``__init__`` of its own, so this cannot be a constructor
        assignment.
        """
        session = getattr(self, "_ideogram4_lora_session_instance", None)
        if session is None:
            from core.adapters import AdapterSession

            session = AdapterSession(
                resolve_path=self._ideogram4_resolve_lora_path,
                warn=self._ideogram4_lora_warn,
                architecture="ideogram4",
                label="Ideogram 4 LoRA",
                count_declared_branches=self._ideogram4_declared_branches,
                missing_file=self._ideogram4_missing_lora,
                prepare_file=self._ideogram4_prepare_lora_file,
                describe_zero_targets=self._ideogram4_zero_target_message,
            )
            self._ideogram4_lora_session_instance = session
        return session

    def _ideogram4_lora_components(self):
        """The two transformers Ideogram 4 LoRAs touch, as separate components.

        Rebuilt from ``ideogram4_components`` on every call, so a model swap of
        either half reaches the session's weakref reset instead of being
        remembered here.
        """
        from core.adapters import AdapterComponent
        from core.models.ideogram4.ideogram4_lora import iter_ideogram4_lora_slots

        components = self.ideogram4_components or {}
        cond = components.get("transformer")
        uncond = components.get("unconditional_transformer")
        if uncond is not None and uncond is cond:
            # Two components over one object would enumerate the same slots twice
            # and install one file's branch under one name twice. The loader always
            # builds two, so this is an impossible state rather than a mode.
            raise RuntimeError(
                "Ideogram 4's conditional and unconditional transformers are the "
                "same object; the asymmetric-CFG branches must be distinct models.")
        return [
            AdapterComponent(name="transformer", module=cond,
                             iter_targets=iter_ideogram4_lora_slots,
                             build_branch=self._ideogram4_build_lora_branch),
            AdapterComponent(name="unconditional_transformer", module=uncond,
                             iter_targets=iter_ideogram4_lora_slots,
                             build_branch=self._ideogram4_build_lora_branch),
        ]

    @property
    def _ideogram4_lora_orig(self):
        """Module path -> the pre-LoRA Linear of the CONDITIONAL transformer.

        Deliberately read WITHOUT the reload check: the model-reload gate has to
        be able to observe a stale map rather than one that resets itself on
        being looked at. ``AdapterSession.bind`` performs the reset on the load
        and unload paths instead.
        """
        return self._ideogram4_lora_session.state("transformer").originals

    @property
    def _ideogram4_lora_keys(self):
        return self._ideogram4_lora_session.state("transformer").wrapped

    @property
    def _ideogram4_lora_orig_uncond(self):
        return self._ideogram4_lora_session.state(
            "unconditional_transformer").originals

    @property
    def _ideogram4_lora_keys_uncond(self):
        return self._ideogram4_lora_session.state(
            "unconditional_transformer").wrapped

    @staticmethod
    def _ideogram4_prepare_lora_file(file):
        """One file's per-branch key groups, format label and metadata alpha.

        Computed once per file by the session: without it the codec would
        re-group every tensor once per target, twice over.
        """
        from core.models.ideogram4.ideogram4_lora import (
            alpha_from_metadata, detect_lora_format, normalise_lora_state_dict,
        )

        return {
            "transformer": normalise_lora_state_dict(file.tensors, branch="cond"),
            "unconditional_transformer": normalise_lora_state_dict(
                file.tensors, branch="uncond"),
            "format": detect_lora_format(file.tensors),
            "alpha": alpha_from_metadata(file.metadata),
        }

    def _ideogram4_build_lora_branch(self, request):
        """Build the branch for one target of one transformer, or say there is none.

        Which half of the checkpoint is read is decided by the COMPONENT, not by
        the module path: both transformers carry identical paths, and only the key
        namespace (``lora_unet_`` / ``lora_uncond_``) tells them apart.
        """
        from core.adapters import PreparedBranch
        from core.models.ideogram4.ideogram4_lora import build_lora_branch

        groups = request.prepared
        weights = groups[request.component].get(request.module_path)
        if weights is None:
            return None
        branch = build_lora_branch(request.base, weights, request.module_path,
                                   default_alpha=groups["alpha"])
        # Strength is folded into the branch's own scale by ``add_branch``, never
        # multiplied onto its delta -- a post-multiply is different arithmetic and
        # loses bit-identity with the single-LoRA numerics this replaces.
        return PreparedBranch(branch, request.file.strength)

    def _ideogram4_zero_target_message(self, file, counts):
        """The zero-target refusal text, or Ideogram 4's one dedicated code.

        The uncond-only case returns the REFUSAL rather than text: the session
        would otherwise tag it ``lora_incompatible``, and the whole point of the
        separate code is that the keys WERE recognized.
        """
        from core.adapters import AdapterIncompatible

        groups = file.prepared
        cond = groups["transformer"]
        uncond = groups["unconditional_transformer"]
        if uncond and not cond and (self.ideogram4_components or {}).get(
                "unconditional_transformer") is None:
            return AdapterIncompatible(
                f"LoRA '{file.name}': all {len(uncond)} of its down/up pairs target "
                f"the unconditional branch, and no unconditional Ideogram 4 transformer "
                f"is loaded, so none of them could be applied.",
                code="lora_uncond_unavailable")
        return (
            f"LoRA '{file.name}': 0 of {file.declared_branches} down/up pairs applied to "
            f"the loaded Ideogram 4 transformer (format={groups['format']}) -- "
            f"unrecognized key format or a different model. Sample keys in file: "
            f"{list(file.tensors.keys())[:5]}"
        )

    def _load_lora_ideogram4(self, lora_configs: List[Dict]) -> int:
        """Cover Ideogram 4 transformer Linear/Fp8Linear modules with LoRA adapters.

        Applies the conditional-branch keys to `transformer` and the
        unconditional-branch ones to `unconditional_transformer` when it is
        loaded, as ONE atomic request: nothing is installed until every file has
        been resolved, parsed and validated against both halves, so a refusal on
        the unconditional half leaves the conditional one unwrapped too.

        Each target Linear is covered ONCE by a ``CompositeAdapterLayer`` and each
        selected LoRA adds a NAMED branch to it, so two Ideogram 4 LoRAs over the
        same module SUM. The two transformers hold SEPARATE composites and
        separate bookkeeping, so a stack on one branch cannot reach the other.

        Raises:
            AdapterFileMissing (a FileNotFoundError) / AdapterLoadFailed /
            AdapterIncompatible (both RuntimeErrors), each carrying its warning
            ``code``. A requested-but-ineffective LoRA must not produce a
            successful generation.
        """
        components = self._ideogram4_lora_components()
        # Unconditional, and BEFORE the empty-config exit: a restore that failed in
        # an earlier request must not leak wrappers into this one. `_ideogram4_cleanup`
        # swallows restore failures, and now that a second branch SUMS rather than
        # being refused, a leaked wrapper would silently double-apply. This also
        # performs the weakref reset both halves need.
        self._ideogram4_lora_session.unload(components)

        if components[0].module is None:
            if lora_configs:
                print("[Ideogram 4 LoRA] WARNING: no Ideogram 4 transformer is loaded")
            return 0
        result = self._ideogram4_lora_session.load(lora_configs, components)
        self._ideogram4_report_lora_files(result, components)
        return result.applied

    def _ideogram4_report_lora_files(self, result, components) -> None:
        """Ideogram 4's own console breadcrumbs: the format label and the split
        between the two branches, neither of which the session can know."""
        for index, (file, counts) in enumerate(result.files):
            groups = file.prepared
            cond_applied = counts.per_component.get("transformer", (0, 0))[0]
            uncond_applied = counts.per_component.get(
                "unconditional_transformer", (0, 0))[0]
            print(f"[Ideogram 4 LoRA] {index + 1}/{len(result.files)}: {file.name} "
                  f"format={groups['format']} cond_modules={len(groups['transformer'])} "
                  f"wrapped={cond_applied} uncond_wrapped={uncond_applied} "
                  f"strength={file.strength}")
            if groups["unconditional_transformer"] and components[1].module is None:
                print(f"[Ideogram 4 LoRA]   WARNING: {file.name} carries "
                      f"{len(groups['unconditional_transformer'])} unconditional-branch "
                      f"module(s) but no unconditional Ideogram 4 transformer is "
                      f"loaded; those are skipped")

    def _unload_lora_ideogram4(self) -> int:
        """Restore every Ideogram 4 transformer Linear to its pre-LoRA original.

        PER BRANCH: each transformer is restored through its own component state,
        so a reload of one branch (whose state the session has just reset) leaves
        the other branch's wrappers installed and restorable.
        """
        return self._ideogram4_lora_session.unload(self._ideogram4_lora_components())

    def _ideogram4_runtime_int8(self, params: Dict[str, Any], progress_callback=None) -> None:
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        BOTH TRANSFORMERS, in one call. Ideogram 4 runs asymmetric CFG: the
        conditional and the unconditional branch are separate 9.28 G-parameter
        transformers and both run every step, so converting one of them would put
        the two halves of a single denoise step at different precisions --
        invisibly, since both would still produce finite images. That is why the
        conversion goes through ``apply_runtime_int8_quantization_multi``: the
        manager-level ``_runtime_int8_converted`` latch is set once, for the set,
        and calling the single-module function twice would convert the
        conditional branch, latch, and silently skip the unconditional one.

        ORDERING. Called from ``_ideogram4_stage_transformers``, which is the one
        choke point every generation path goes through, and BEFORE:

        * the per-transformer block-swap offloaders are built. They capture each
          block's Linear modules and build CPU masters from their weights;
          converting afterwards would replace the modules they hold and leave
          them streaming the pre-conversion bf16 weights into modules nothing
          reads. Checked below rather than assumed.
        * the ``.to(device)`` staging, so the conversion runs on CPU (the
          converter is device-aware) and a block-swapped generation -- which
          never stages the transformers at all -- is converted just the same.

        LoRAs are loaded AFTER staging on this architecture and unloaded in the
        generation's ``finally``, so the modules here are unwrapped; the
        converter refuses a wrapped one anyway, for the whole set at once.

        No-op for every value other than ``"int8"``, and for a model that is
        already weight-only quantized -- which the published Ideogram 4
        checkpoints are (FP8 or nf4), so on those the request is reported as
        superseded rather than applied.
        """
        from core.vram_optimization import (
            apply_runtime_int8_quantization_multi, runtime_int8_requested,
        )

        components = self.ideogram4_components or {}
        # Scoped to a request that would actually convert something: a stale
        # offloader is a problem for THIS conversion, and refusing a generation
        # that never asked for INT8 would turn a leftover attribute into a
        # crash.
        for name in ("transformer", "unconditional_transformer"):
            t = components.get(name)
            if (runtime_int8_requested(params.get("unet_quantization"))
                    and t is not None
                    and getattr(t, "_block_offloader", None) is not None):
                raise RuntimeError(
                    f"Ideogram 4 INT8 conversion was reached while a block offloader is "
                    f"still attached to '{name}'. It must run BEFORE the offloaders are "
                    f"created: they hold references to each block's Linear modules, and the "
                    f"conversion replaces those modules, so afterwards they would stream the "
                    f"original bf16 weights into modules nothing reads.")

        targets = [
            ("transformer", "Ideogram 4 Transformer (conditional)",
             components.get("transformer")),
            ("unconditional_transformer", "Ideogram 4 Transformer (unconditional)",
             components.get("unconditional_transformer")),
        ]
        if not any(mod is not None for _n, _l, mod in targets):
            return

        present = [t for t in targets if t[2] is not None]
        models, _converted = apply_runtime_int8_quantization_multi(
            self, present, "ideogram4", params.get("unet_quantization"),
            progress_callback=progress_callback)
        # The converter replaces child modules in place and returns the same
        # objects, so this is bookkeeping rather than a swap; written back
        # anyway so the components dict is authoritative either way.
        for (name, _label, _module), model in zip(present, models):
            components[name] = model

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

    def _ideogram4_stage_transformers(self, device: str, params: Optional[Dict[str, Any]] = None,
                                      progress_callback=None):
        """Place both transformers on GPU for the denoise loop.

        With block swap enabled, each transformer streams its blocks (per-model
        offloader) instead of being fully resident, roughly halving the resident
        footprint of the two 9.3B FP8 transformers at the cost of CPU<->GPU traffic.
        """
        params = params or {}
        # One-time in-place INT8 conversion (unet_quantization="int8"), for BOTH
        # transformers. MUST be here: before the block offloaders are built (they
        # capture the Linear modules this replaces) and before the ->GPU move,
        # which the block-swap branch below never performs at all. No-op for every
        # other value and for an already-quantized checkpoint.
        self._ideogram4_runtime_int8(params, progress_callback=progress_callback)
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

        # LoRA unwrap belongs HERE, not only in the denoise finally: the wrappers
        # are installed before NAG/NegPip/style setup, so an exception in any of
        # those would otherwise carry them into the NEXT generation.
        try:
            self._unload_lora_ideogram4()
        except Exception:
            pass

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
                transformer, uncond_transformer = self._ideogram4_stage_transformers(
                    device, params, progress_callback=progress_callback)
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
                transformer, uncond_transformer = self._ideogram4_stage_transformers(
                    device, params, progress_callback=progress_callback)
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
                transformer, uncond_transformer = self._ideogram4_stage_transformers(
                    device, params, progress_callback=progress_callback)
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
