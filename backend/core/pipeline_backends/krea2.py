"""Krea 2 generation backend (Krea2Mixin) for PipelineManager.

Single-stream flow-matching MMDiT with a Qwen3-VL text encoder (12-layer
hidden-state stack) and the Qwen-Image VAE (16 latent channels, f8). VRAM staging
mirrors the other large DiT archs: encode text (TE -> GPU -> free to CPU), stage
the ~26GB transformer on GPU for the denoise loop, then VAE-decode. CFG uses the
standard UI ``cfg_scale`` mapped to the Krea guidance convention
(``guidance = cfg_scale - 1``); the distilled/turbo checkpoint disables CFG and
pins a fixed timestep shift.
"""

from typing import Dict, Any, Optional, List
from PIL import Image
import random
import torch

from config.settings import settings

# Latent geometry: Qwen-Image VAE 8x downscale + 2x2 patchify => 16px token grid.
GRID_ALIGN = 16

def _dropped_lora_keys(raw: Dict[str, Any], grouped: Dict[str, Any]) -> List[str]:
    """Source keys that carry nothing into ``grouped``: unparseable, foreign
    (``lora_te_*``), or a down without its up. Counting ``grouped`` alone hides
    them, and a dropped key is a silently weaker LoRA rather than a no-op
    (mirrors ``anima_lora.unmatched_source_keys``).

    Asks the parser what each key IS rather than re-spelling each group back
    into ``.lora_down.weight``: the shared suffix table accepts several
    spellings per tensor, so a reconstruction reports a file that applied in
    full as entirely dropped.
    """
    from core.models.krea2.krea2_lora import _parse_key
    dropped = []
    for key in raw:
        parsed = _parse_key(key)
        if parsed is None or parsed[0] not in grouped:
            dropped.append(key)
    return sorted(dropped)


def _normalize_resolution(width: int, height: int) -> tuple:
    """Round width/height up to a multiple of the 16px token grid."""
    w = ((int(width) + GRID_ALIGN - 1) // GRID_ALIGN) * GRID_ALIGN
    h = ((int(height) + GRID_ALIGN - 1) // GRID_ALIGN) * GRID_ALIGN
    return max(w, GRID_ALIGN), max(h, GRID_ALIGN)


class Krea2Mixin:
    """Krea 2 backend methods for PipelineManager."""

    def _krea2_move(self, component_name: str, target_device: str):
        comp = self.krea2_components.get(component_name)
        if comp is None or not hasattr(comp, "to"):
            return comp
        try:
            comp.to(target_device)
        except Exception as e:
            print(f"[Krea2] Warning: could not move {component_name} to {target_device}: {e}")
        return comp

    def _krea2_kh_setup(self, params: Dict[str, Any]):
        """Compute keep-models-hot eligibility for this generation (see core/keep_hot.py).

        Krea 2 has no block-swap streaming, so the LoRA hazard is the only gate
        that applies: ``_load_lora_krea2`` mutates the transformer in place, so a
        LoRA generation never leaves it resident (same rule as anima/flux2/
        minit2i/zimage). The TE is frozen for Krea 2 training and a Krea 2 LoRA
        can never carry TE keys, so TE residency is not gated.
        Returns (model_key, keep_te, keep_transformer, keep_vae).
        """
        from core.keep_hot import (
            invalidate_if_model_changed, should_keep_resident, compute_model_key,
            component_nbytes, keep_hot_requested,
        )
        requested = keep_hot_requested(params)
        model_key = compute_model_key(self, params)
        has_loras = bool(params.get("loras") or [])

        # If a resident set exists from a previous generation but is no longer valid
        # for THIS request's model_key, force a full offload before staging anything.
        invalidate_if_model_changed(
            self, params,
            offload_fn=lambda: (
                self._krea2_move("text_encoder", "cpu"),
                self._krea2_move("transformer", "cpu"),
                self._krea2_move("vae", "cpu"),
            ),
        )

        total_bytes = 0
        if requested:
            total_bytes += component_nbytes(self.krea2_components.get("text_encoder"))
            if not has_loras:
                total_bytes += component_nbytes(self.krea2_components.get("transformer"))
            total_bytes += component_nbytes(self.krea2_components.get("vae"))
        guard_ok = should_keep_resident(
            self, "combined", params,
            is_block_swapped=False, is_cpu_inference=False,
            component_bytes=total_bytes,
        ) if requested else False

        keep_te = requested and guard_ok
        keep_transformer = requested and guard_ok and not has_loras
        keep_vae = requested and guard_ok
        return model_key, keep_te, keep_transformer, keep_vae

    def _krea2_runtime_int8(self, params: Dict[str, Any], progress_callback=None) -> None:
        """Apply the one-time in-place INT8 conversion, if this request asks for it.

        Krea 2's quantization is otherwise decided by the CHECKPOINT FORMAT at
        load time (a bf16 or a weight-only FP8/INT8 file); this is the one
        per-generation value it honours, and only this one. Called before the
        transformer is staged, so the module is still on CPU and no second module
        copy is built -- the bf16 transformer is ~24 GB and a deep copy is not an
        option. Host RSS still ends around 1.6x the source (~36 GB here) because
        the source checkpoint's mapping stays referenced; see
        ``quantize_linears_in_place`` and docs/guides/MODEL_FACTS.md.
        """
        from core.vram_optimization import apply_runtime_int8_quantization

        transformer = (self.krea2_components or {}).get("transformer")
        if transformer is None:
            return
        model, _converted = apply_runtime_int8_quantization(
            self, transformer, "krea2", params.get("unet_quantization"),
            label="Krea 2 Transformer", progress_callback=progress_callback)
        self.krea2_components["transformer"] = model

    def _krea2_apply_attention_backend(self, transformer, params: Dict[str, Any]):
        """Stamp the inference attention backend (native/flash/sage) on the transformer.

        The vendored forward propagates ``_attn_backend`` to every Krea2Attention each
        step; the conduit handles the GQA (48Q/12KV) downgrade rules per backend.
        """
        from core.attention import normalize_backend
        backend = normalize_backend(params.get("attention_type", settings.attention_type))
        transformer._attn_backend = backend
        print(f"[Krea2] Attention backend: {backend} "
              f"(from attention_type={params.get('attention_type')!r})")

    @staticmethod
    def _krea2_lora_warn(message: str, code: str) -> None:
        """Record a user-visible generation warning. ``message`` is embedded in the
        output PNG's text chunk, so it must never carry an absolute path."""
        print(f"[Krea2 LoRA] WARNING: {message}")
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    @staticmethod
    def _krea2_lora_ignored_options(cfg: Dict) -> List[str]:
        """LoRA knobs this backend does not implement (LYCORIS_ADAPTER_DESIGN
        Phase 1 owns them). ``apply_to_text_encoder`` is absent on purpose: Krea 2
        never touches the TE, so either value is honoured."""
        ignored = []
        if not cfg.get("apply_to_unet", True):
            ignored.append("apply_to_unet=false")
        if cfg.get("unet_layer_weights"):
            ignored.append("unet_layer_weights")
        step_range = cfg.get("step_range")
        if step_range is not None and list(step_range) != [0, 1000]:
            ignored.append(f"step_range={list(step_range)}")
        return ignored

    # -- LoRA lifetime -------------------------------------------------------
    #
    # Owned by ``core.adapters.AdapterSession``: it resolves, parses and plans
    # every selected file against the live transformer BEFORE mutating a slot,
    # then installs the whole request or none of it, and holds the weakref-keyed
    # bookkeeping and its reset. What stays here is Krea 2's -- the target scope,
    # the key codec, this backend's own per-file warnings and one branch.

    @staticmethod
    def _krea2_resolve_lora_path(raw_path):
        from core.extensions.lora_manager import lora_manager

        return lora_manager._resolve_lora_path(raw_path)

    @staticmethod
    def _krea2_missing_lora(lora_file, _raw_path):
        """Krea 2's own refusal for an unresolvable path.

        A ``RuntimeError``, not the session's ``FileNotFoundError``: this
        backend's callers and its gate catch that type. The searched directories
        stay on the console -- the refusal rides into the PNG text chunk.
        """
        from api.error_handlers import with_error_code
        from core.extensions.lora_manager import lora_manager

        print(f"[Krea2 LoRA] ERROR: {lora_file} not found; searched "
              f"{lora_manager.lora_dir} and {lora_manager.additional_dirs}")
        return with_error_code(RuntimeError(
            f"Krea 2 LoRA file not found: '{lora_file}' -- no such file in any "
            f"registered LoRA directory."), "lora_not_found")

    @property
    def _krea2_lora_session(self):
        """The per-backend session, created on first use.

        The mixin has no ``__init__`` of its own, so this cannot be a
        constructor assignment.
        """
        session = getattr(self, "_krea2_lora_session_instance", None)
        if session is None:
            from core.adapters import AdapterSession

            session = AdapterSession(
                resolve_path=self._krea2_resolve_lora_path,
                warn=self._krea2_lora_warn,
                architecture="krea2",
                # Bound on the composed PipelineManager, not on this
                # mixin; `getattr` keeps a bare-mixin unit test
                # constructible (adapter_key_normalization_gate).
                base_latent=getattr(self, "base_latent_identity", None),
                # The console prefix this backend has always used, and the
                # noun its user-visible failure text spells with a space.
                label="Krea2 LoRA",
                message_label="Krea 2 LoRA",
                count_declared_branches=self._krea2_declared_branches,
                missing_file=self._krea2_missing_lora,
                prepare_file=self._krea2_prepare_lora_file,
                describe_zero_targets=self._krea2_zero_target_message,
            )
            self._krea2_lora_session_instance = session
        return session

    def _krea2_lora_components(self):
        """The one component Krea 2 LoRAs touch.

        Transformer-only: there is no TE apply path, and no Krea 2 LoRA can
        carry TE keys because the Qwen3-VL encoder is frozen for training.
        Rebuilt from ``krea2_components`` on every call, so a model swap reaches
        the session's weakref reset instead of being remembered here.
        """
        from core.adapters import AdapterComponent
        from core.models.krea2.krea2_lora import iter_krea2_lora_slots

        components = getattr(self, "krea2_components", None) or {}
        return [AdapterComponent(
            name="transformer",
            module=components.get("transformer"),
            iter_targets=iter_krea2_lora_slots,
            build_branch=self._krea2_build_lora_branch,
        )]

    @property
    def _krea2_lora_original_modules(self):
        """Module path -> the pre-LoRA Linear of the transformer in hand.

        Read WITHOUT the reload check on purpose: the reload gate has to observe
        a stale map, not one that resets itself on being looked at.
        ``AdapterSession.bind`` does the reset, on the load and unload paths.
        """
        return self._krea2_lora_session.state("transformer").originals

    @property
    def _krea2_lora_wrapped_keys(self):
        return self._krea2_lora_session.state("transformer").wrapped

    @staticmethod
    def _krea2_declared_branches(tensors, _components) -> int:
        """Complete factor GROUPS, not ``.lora_down.weight`` keys: a foreign
        (``lora_te_*``) or unpaired key would inflate the count the session
        compares against and warn ``lora_partial`` on a file that applied in
        full, and a LyCORIS file has no such key at all.
        """
        from core.models.krea2.krea2_lora import declared_branch_count

        return declared_branch_count(tensors)

    def _krea2_prepare_lora_file(self, file):
        """This file's grouped tensors, parsed and reported once.

        The session runs this before the file is planned, so a file that matches
        nothing still reports its dropped keys before its refusal.
        """
        from core.models.krea2.krea2_lora import (detect_lora_format,
                                                  normalise_lora_state_dict)

        grouped = normalise_lora_state_dict(file.tensors)
        print(f"[Krea2 LoRA] {file.name} format={detect_lora_format(file.tensors)} "
              f"keys={len(file.tensors)} matched_modules={len(grouped)} "
              f"strength={file.strength}")

        dropped = _dropped_lora_keys(file.tensors, grouped)
        if dropped:
            self._krea2_lora_warn(
                f"LoRA '{file.name}': {len(dropped)} of its {len(file.tensors)} tensor "
                f"key(s) are not part of a complete Krea 2 'lora_unet_*' factor "
                f"group and were dropped "
                f"(first few: {dropped[:5]}).",
                "krea2_lora_keys_unrecognised")

        ignored = self._krea2_lora_ignored_options(file.config)
        if ignored:
            self._krea2_lora_warn(
                f"LoRA '{file.name}': {', '.join(ignored)} is not implemented for Krea 2 "
                f"and was ignored; strength {file.strength} applies to every matched "
                f"module for the whole denoise loop.",
                "krea2_lora_options_ignored")
        return grouped

    def _krea2_build_lora_branch(self, request):
        """The branch for one target, ``None`` when this file names no key for it,
        or ``SHAPE_MISMATCH`` when its factors do not fit. Nothing is installed
        here."""
        from core.adapters import SHAPE_MISMATCH, PreparedBranch
        from core.models.krea2.krea2_lora import build_lora_branch

        group = request.prepared.get(request.module_path)
        if group is None:
            return None
        branch = build_lora_branch(request.base, group, request.module_path)
        if branch is SHAPE_MISMATCH:
            return branch
        # Strength is folded into the branch's own scale by ``add_branch``, never
        # multiplied onto its delta.
        return PreparedBranch(branch, request.file.strength)

    def _krea2_zero_target_message(self, file, counts) -> str:
        """The zero-target refusal text: it names Krea 2's own key convention.

        The session owns the DECISION to refuse; the text is Krea 2's because
        the only actionable part of it is the key format this loader expects.
        """
        from core.models.krea2.krea2_lora import detect_lora_format

        if file.declared_branches == 0:
            return (f"Krea 2 LoRA '{file.name}': none of its {len(file.tensors)} tensors "
                    f"form a complete 'lora_unet_*' factor group "
                    f"(format={detect_lora_format(file.tensors)}) -- not a Krea 2 LoRA.")
        return (f"LoRA '{file.name}': 0 of {file.declared_branches} modules matched the "
                f"loaded Krea 2 transformer -- wrong architecture or an unsupported "
                f"target scope.")

    def _load_lora_krea2(self, lora_configs: List[Dict]) -> int:
        """Cover Krea 2 transformer Linears with the requested LoRA(s).

        Each target Linear is covered ONCE by a ``CompositeAdapterLayer`` and
        each selected LoRA adds a NAMED branch to it, so two Krea 2 LoRAs over
        the same module sum instead of being refused. Nothing is installed until
        every file has been resolved, parsed and validated.

        Must run AFTER the transformer is staged on GPU and after any runtime
        INT8 conversion: the branches reference the CURRENT Linear modules and
        copy their device. Raises on a file that resolves to zero targets -- a
        LoRA that had no effect must not pass as a successful generation.
        """
        # Unconditional, and BEFORE the empty-config exit: a restore that failed
        # in an earlier request must not leak its wrappers into this generation,
        # and this is also what resets bookkeeping a model reload invalidated.
        self._unload_lora_krea2()

        if not lora_configs:
            return 0
        if not self.krea2_components:
            raise RuntimeError("Krea 2 LoRA requested but no Krea 2 model is loaded.")

        return self._krea2_lora_session.load(
            lora_configs, self._krea2_lora_components()).applied

    def _unload_lora_krea2(self) -> int:
        """Restore every wrapped Krea 2 Linear to its pre-LoRA original.

        The session drops each original as restore lands it, so a later one-time
        INT8 conversion cannot reinstate a stale pre-conversion module.
        """
        return self._krea2_lora_session.unload(self._krea2_lora_components())

    def _krea2_style_triple(self, params: Dict[str, Any], style_dict: Dict[str, Any],
                             transformer, device, ref_index: int = 0):
        """Build a single (StyleTransferConfig, ref_x0, eps_ref) triple from one
        style_transfer dict. ``axes_dims`` is filled in from the loaded
        transformer's own RoPE config (arch-specific; the shared
        ``reference_style`` module has no universal default for it).

        ``ref_index`` decorrelates the fixed re-noising noise tensor across
        multiple simultaneous references (each ref would otherwise draw the
        EXACT same noise from ``prepare_style_reference``'s ``seed+991``
        offset applied to the SAME ``common["seed"]``, since that offset does
        not depend on which reference is being prepared). ``ref_index=0``
        (the default, used by the single-ref path) reproduces the
        pre-multi-ref ``common["seed"]`` offset exactly."""
        from core.inference.reference_style import style_config_from_dict
        from core.models.krea2.krea2_pipeline_ops import prepare_style_reference

        cfg = style_config_from_dict(style_dict)
        cfg.axes_dims = tuple(transformer.config.axes_dims_rope)

        common = self._krea2_common_params(params, style_dict["image"].width, style_dict["image"].height)
        ref_seed = common["seed"] if ref_index == 0 else common["seed"] + ref_index
        ref_x0, eps_ref = prepare_style_reference(
            self.krea2_components["vae"], style_dict["image"],
            common["height"], common["width"], common["patch_size"],
            device=device, seed=ref_seed,
        )
        return cfg, ref_x0, eps_ref

    def _krea2_style_config(self, params: Dict[str, Any], transformer, device):
        """Build a (StyleTransferConfig, ref_x0, eps_ref) triple from
        ``params["style_transfer"]`` (assembled by
        ``generation_utils.process_controlnet_configs``), or ``(None, None, None)``
        when no style reference is attached. Single-reference path,
        BYTE-IDENTICAL to the pre-multi-ref implementation (delegates to
        ``_krea2_style_triple`` with ``ref_index=0``, which reproduces the
        original ``common["seed"]`` re-noising offset exactly)."""
        style_dict = params.get("style_transfer")
        if not style_dict or not style_dict.get("image"):
            return None, None, None

        return self._krea2_style_triple(params, style_dict, transformer, device, ref_index=0)

    def _krea2_style_configs(self, params: Dict[str, Any], transformer, device):
        """Build the full style-transfer configuration for Krea 2 generation,
        covering both the single-reference path (legacy ``(style_cfg,
        style_ref_x0, style_eps_ref)`` triple, exactly as ``_krea2_style_config``
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
                refs.append(self._krea2_style_triple(params, style_dict, transformer, device, ref_index=idx))
            if len(refs) > 1:
                return None, None, None, refs, combine_mode
            if len(refs) == 1:
                cfg, x0, eps = refs[0]
                return cfg, x0, eps, None, combine_mode
            return None, None, None, None, combine_mode

        style_cfg, style_ref_x0, style_eps_ref = self._krea2_style_config(params, transformer, device)
        return style_cfg, style_ref_x0, style_eps_ref, None, "stack"

    @staticmethod
    def _krea2_advanced_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
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

    def _krea2_common_params(self, params: Dict[str, Any], default_w: int, default_h: int):
        seed = params.get("seed", -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        req_w = int(params.get("width", default_w))
        req_h = int(params.get("height", default_h))
        width, height = _normalize_resolution(req_w, req_h)
        if (width, height) != (req_w, req_h):
            print(f"[Krea2] Resolution aligned: {req_w}x{req_h} -> {width}x{height}")

        vae_scale = int(self.krea2_components.get("vae_scale_factor", 8))
        patch_size = int(self.krea2_components.get("patch_size", 2))
        grid_h = height // (vae_scale * patch_size)
        grid_w = width // (vae_scale * patch_size)
        is_distilled = bool(self.krea2_components.get("is_distilled", False))

        # UI cfg_scale -> Krea guidance (cfg_scale - 1). Distilled checkpoint: no CFG.
        cfg_scale = float(params.get("cfg_scale", 4.5))
        guidance = 0.0 if is_distilled else max(cfg_scale - 1.0, 0.0)
        default_steps = 8 if is_distilled else 28

        return {
            "seed": seed,
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", "") or "",
            "num_inference_steps": int(params.get("steps", default_steps)),
            "guidance": guidance,
            "max_sequence_length": int(params.get("krea2_max_seq_len", 512)),
            "width": width,
            "height": height,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "patch_size": patch_size,
            "is_distilled": is_distilled,
            "num_channels_latents": int(self.krea2_components["transformer"].config["in_channels"]) // (patch_size ** 2),
        }

    @torch.no_grad()
    def _krea2_encode(self, prompt, negative_prompt, cfg, device, dtype,
                       model_key: Optional[str] = None, keep_te: bool = False):
        """Stage the TE to GPU (unless already kept resident), encode positive
        (+ negative when CFG on), then either free TE to CPU (default) or leave
        it resident (keep-models-hot: mark_resident, deferred final decision is
        still corrected by the caller's outer cleanup on exception)."""
        from core.models.krea2.krea2_pipeline_ops import encode_prompt
        from core.keep_hot import is_resident, mark_resident, discard_resident

        select_layers = self.krea2_components["text_encoder_select_layers"]
        max_len = cfg["max_sequence_length"]

        if model_key is None or not is_resident(self, "text_encoder", model_key):
            self._krea2_move("text_encoder", device)
        te = self.krea2_components["text_encoder"]
        tok = self.krea2_components["tokenizer"]

        prompt_embeds, prompt_mask = encode_prompt(te, tok, prompt, select_layers, max_len, device)
        neg_embeds = neg_mask = None
        if cfg["guidance"] > 0.0:
            neg_prompt = negative_prompt if (negative_prompt and negative_prompt.strip()) else ""
            neg_embeds, neg_mask = encode_prompt(te, tok, neg_prompt, select_layers, max_len, device)
            neg_embeds = neg_embeds.to(dtype)

        if keep_te and model_key is not None:
            mark_resident(self, "text_encoder", model_key)
        else:
            self._krea2_move("text_encoder", "cpu")
            if model_key is not None:
                discard_resident(self, "text_encoder")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return prompt_embeds.to(dtype), prompt_mask, neg_embeds, neg_mask

    def _krea2_cleanup(self, model_key: Optional[str] = None,
                       keep_te: bool = False, keep_transformer: bool = False,
                       keep_vae: bool = False, gen_succeeded: bool = False):
        """End-of-generation teardown. On failure (or when called with no
        keep-hot context, e.g. old call sites), force a full offload and clear
        any tentative residency. On success, components already left resident
        by their own stage (keep_X=True) are trusted as-is; the rest were
        already offloaded at their own stage -- discard_resident just keeps
        the tracked set in sync (idempotent no-op otherwise)."""
        from core.keep_hot import clear_resident, discard_resident
        # Idempotent: the denoise finally already unwrapped on the normal path.
        # This is the net for a failure between apply and that block.
        try:
            self._unload_lora_krea2()
        except Exception as e:
            # Swallowing this is what makes a leak silent: wrappers stay installed
            # and the next generation would denoise through them. It is retried
            # (and raises) at the top of _load_lora_krea2, so do not fail here.
            print(f"[Krea2 LoRA] ERROR: could not restore the transformer: {e}")
            import traceback; traceback.print_exc()
            self._krea2_lora_warn(
                f"Krea 2 LoRA wrappers could not be removed after this generation ({e}); "
                f"the next generation retries the restore before denoising.",
                "lora_unload_failed")
        if not gen_succeeded:
            clear_resident(self)
            for _c in ("text_encoder", "transformer", "vae"):
                try:
                    self._krea2_move(_c, "cpu")
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

    def _generate_txt2img_krea2(self, params: Dict[str, Any],
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded. Please load a Krea 2 model first.")
        from core.models.krea2.krea2_pipeline_ops import (
            prepare_latents_txt2img, denoise_loop, vae_decode,
        )

        print("[Krea2] Starting txt2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, 1024, 1024)
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        from core.keep_hot import is_resident, mark_resident, discard_resident
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae = self._krea2_kh_setup(params)
        _kh_gen_succeeded = False

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype,
                model_key=_kh_model_key, keep_te=_kh_keep_te)

            print("[Krea2] Stage 2: Prepare latents...")
            latents = prepare_latents_txt2img(
                cfg["num_channels_latents"], cfg["grid_h"], cfg["grid_w"], cfg["patch_size"],
                dtype=torch.float32, device=device, seed=cfg["seed"])

            print("[Krea2] Stage 3: Denoising...")
            # One-time in-place INT8 conversion (unet_quantization="int8"), while
            # the transformer is still on CPU. No-op for every other value and
            # for an already-converted / already-quantized model.
            self._krea2_runtime_int8(params, progress_callback=progress_callback)
            if not is_resident(self, "transformer", _kh_model_key):
                transformer = self._krea2_move("transformer", device)
            else:
                transformer = self.krea2_components["transformer"]
            self._krea2_apply_attention_backend(transformer, params)

            # LoRA wrappers hold the current Linear modules, so this must follow
            # both the INT8 conversion and the GPU stage above.
            self._load_lora_krea2(params.get("loras") or [])

            # Training-free reference-style transfer. OFF by default
            # (style_transfer/style_transfers absent -> (None, None, None,
            # None, "stack"), no-op below). ``style_refs`` is populated (and
            # style_cfg/style_ref_x0/style_eps_ref left None) ONLY when
            # ``params["style_transfers"]`` carries 2+ references -- a single
            # reference (via either key) always resolves through the
            # style_cfg/style_ref_x0/style_eps_ref triple, so that code path
            # (both here and inside _run_loop) is untouched.
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                if not is_resident(self, "vae", _kh_model_key):
                    self._krea2_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._krea2_style_configs(params, transformer, device)
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = denoise_loop(
                    transformer, scheduler, latents, prompt_embeds, prompt_mask,
                    neg_embeds, neg_mask, cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                # Unconditional: a guard here would skip the restore in exactly the
                # cases that leak wrappers into the next generation.
                self._unload_lora_krea2()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._krea2_move("transformer", "cpu")
                    discard_resident(self, "transformer")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _kh_gen_succeeded = True
            print("[Krea2] txt2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] Generation error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                 _kh_keep_vae, _kh_gen_succeeded)

    def _generate_img2img_krea2(self, params: Dict[str, Any], init_image: Image.Image,
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded.")
        from core.models.krea2.krea2_pipeline_ops import (
            vae_encode, denoise_loop_img2img, vae_decode,
        )

        print("[Krea2] Starting img2img generation")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.7))
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        from core.keep_hot import is_resident, mark_resident, discard_resident
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae = self._krea2_kh_setup(params)
        _kh_gen_succeeded = False

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype,
                model_key=_kh_model_key, keep_te=_kh_keep_te)

            print("[Krea2] Stage 2: Encoding init image...")
            # First use of VAE this generation only: honor cross-generation residency
            # on entry, but always offload after (VAE is reused again at Stage 4, so
            # this is an intermediate step, not the generation's final exit point).
            if not is_resident(self, "vae", _kh_model_key):
                self._krea2_move("vae", device)
            init_latents = vae_encode(
                self.krea2_components["vae"], init_image, cfg["height"], cfg["width"],
                cfg["patch_size"], device=device, dtype=torch.float32)
            self._krea2_move("vae", "cpu")
            discard_resident(self, "vae")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("[Krea2] Stage 3: Denoising (SDEdit)...")
            # One-time in-place INT8 conversion (unet_quantization="int8"), while
            # the transformer is still on CPU. No-op for every other value and
            # for an already-converted / already-quantized model.
            self._krea2_runtime_int8(params, progress_callback=progress_callback)
            if not is_resident(self, "transformer", _kh_model_key):
                transformer = self._krea2_move("transformer", device)
            else:
                transformer = self.krea2_components["transformer"]
            self._krea2_apply_attention_backend(transformer, params)

            # Must follow the INT8 conversion and the GPU stage (see txt2img).
            self._load_lora_krea2(params.get("loras") or [])

            # Training-free reference-style transfer (see the txt2img comment
            # above for the single-ref/multi-ref routing invariant).
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                self._krea2_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._krea2_style_configs(params, transformer, device)
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = denoise_loop_img2img(
                    transformer, scheduler, init_latents, denoising_strength,
                    prompt_embeds, prompt_mask, neg_embeds, neg_mask,
                    cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    seed=cfg["seed"], progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                self._unload_lora_krea2()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._krea2_move("transformer", "cpu")
                    discard_resident(self, "transformer")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _kh_gen_succeeded = True
            print("[Krea2] img2img completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] img2img error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                 _kh_keep_vae, _kh_gen_succeeded)

    def _generate_inpaint_krea2(self, params: Dict[str, Any],
                                init_image: Image.Image, mask_image: Image.Image,
                                progress_callback=None, step_callback=None) -> tuple:
        if not self.krea2_components:
            raise RuntimeError("Krea 2 components not loaded.")
        from core.models.krea2.krea2_pipeline_ops import (
            vae_encode, denoise_loop_inpaint, vae_decode, prepare_mask_latent,
        )

        print("[Krea2] Starting inpaint generation (repaint)")
        device = self.device
        dtype = torch.bfloat16
        cfg = self._krea2_common_params(params, init_image.width, init_image.height)
        denoising_strength = float(params.get("denoising_strength", 0.8))
        mask_blur = int(params.get("mask_blur", 4))
        scheduler = self.krea2_components["scheduler"]
        advanced_cfg = self._krea2_advanced_cfg(params)

        width, height = cfg["width"], cfg["height"]
        if (init_image.width, init_image.height) != (width, height):
            init_image = init_image.resize((width, height), Image.LANCZOS)
        if (mask_image.width, mask_image.height) != (width, height):
            mask_image = mask_image.resize((width, height), Image.NEAREST)
        if mask_blur > 0:
            from PIL import ImageFilter
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(mask_blur))

        from core.keep_hot import is_resident, mark_resident, discard_resident
        _kh_model_key, _kh_keep_te, _kh_keep_transformer, _kh_keep_vae = self._krea2_kh_setup(params)
        _kh_gen_succeeded = False

        try:
            print("[Krea2] Stage 1: Text encoding...")
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._krea2_encode(
                cfg["prompt"], cfg["negative_prompt"], cfg, device, dtype,
                model_key=_kh_model_key, keep_te=_kh_keep_te)

            print("[Krea2] Stage 2: Encoding init image + mask...")
            # First use of VAE this generation only: honor cross-generation residency
            # on entry, but always offload after (VAE is reused again at Stage 4, so
            # this is an intermediate step, not the generation's final exit point).
            if not is_resident(self, "vae", _kh_model_key):
                self._krea2_move("vae", device)
            init_latents = vae_encode(
                self.krea2_components["vae"], init_image, height, width,
                cfg["patch_size"], device=device, dtype=torch.float32)
            self._krea2_move("vae", "cpu")
            discard_resident(self, "vae")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_latent = prepare_mask_latent(
                mask_image, cfg["grid_h"], cfg["grid_w"], device=device, dtype=torch.float32)

            print("[Krea2] Stage 3: Denoising (repaint)...")
            # One-time in-place INT8 conversion (unet_quantization="int8"), while
            # the transformer is still on CPU. No-op for every other value and
            # for an already-converted / already-quantized model.
            self._krea2_runtime_int8(params, progress_callback=progress_callback)
            if not is_resident(self, "transformer", _kh_model_key):
                transformer = self._krea2_move("transformer", device)
            else:
                transformer = self.krea2_components["transformer"]
            self._krea2_apply_attention_backend(transformer, params)

            # Must follow the INT8 conversion and the GPU stage (see txt2img).
            self._load_lora_krea2(params.get("loras") or [])

            # Training-free reference-style transfer (see the txt2img comment
            # above for the single-ref/multi-ref routing invariant).
            style_cfg = style_ref_x0 = style_eps_ref = None
            style_refs = None
            style_combine_mode = "stack"
            if params.get("style_transfer") or params.get("style_transfers"):
                self._krea2_move("vae", device)
                style_cfg, style_ref_x0, style_eps_ref, style_refs, style_combine_mode = \
                    self._krea2_style_configs(params, transformer, device)
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            try:
                latents = denoise_loop_inpaint(
                    transformer, scheduler, init_latents, mask_latent, denoising_strength,
                    prompt_embeds, prompt_mask, neg_embeds, neg_mask,
                    cfg["guidance"], cfg["num_inference_steps"],
                    cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], cfg["is_distilled"], device,
                    seed=cfg["seed"], progress_callback=progress_callback, advanced_cfg=advanced_cfg,
                    style_cfg=style_cfg, style_ref_x0=style_ref_x0, style_eps_ref=style_eps_ref,
                    style_refs=style_refs, style_combine_mode=style_combine_mode,
                )
            finally:
                self._unload_lora_krea2()
                if _kh_keep_transformer:
                    mark_resident(self, "transformer", _kh_model_key)
                else:
                    self._krea2_move("transformer", "cpu")
                    discard_resident(self, "transformer")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print("[Krea2] Stage 4: VAE decode...")
            if not is_resident(self, "vae", _kh_model_key):
                self._krea2_move("vae", device)
            self._apply_vae_tiling(self.krea2_components["vae"], getattr(self, "_vae_tiling", False))
            image = vae_decode(self.krea2_components["vae"], latents, cfg["grid_h"], cfg["grid_w"], cfg["patch_size"], color_flatten_strength=getattr(self, "_color_flatten_strength", 0))
            if _kh_keep_vae:
                mark_resident(self, "vae", _kh_model_key)
            else:
                self._krea2_move("vae", "cpu")
                discard_resident(self, "vae")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _kh_gen_succeeded = True
            print("[Krea2] inpaint completed")
            return image, cfg["seed"], 0
        except Exception as e:
            print(f"[Krea2] inpaint error: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            self._krea2_cleanup(_kh_model_key, _kh_keep_te, _kh_keep_transformer,
                                 _kh_keep_vae, _kh_gen_succeeded)
