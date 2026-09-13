"""YuE2 audio generation dispatch."""
import random

from core.models.yue2.pipeline import YuE2Txt2AudResult


class YuE2Mixin:
    @staticmethod
    def _yue2_resolve_lora_path(raw_path):
        from core.extensions.lora_manager import lora_manager
        return lora_manager._resolve_lora_path(raw_path)

    @staticmethod
    def _yue2_lora_warn(message, code):
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    @staticmethod
    def _yue2_prepare_lora_file(file):
        from core.adapters import AdapterIncompatible
        from core.models.yue2.yue2_lora import (normalise_yue2_lora_state_dict,
                                                normalize_yue2_stages)
        declared = str(file.metadata.get("model_type") or
                       file.metadata.get("modelspec.architecture") or "").lower()
        if declared != "yue2":
            raise AdapterIncompatible(
                f"YuE2 LoRA '{file.name}' must declare model_type='yue2', got {declared or 'none'}"
            )
        stages = normalize_yue2_stages(str(file.metadata.get("yue2_apply_stages") or ""))
        grouped = normalise_yue2_lora_state_dict(file.tensors)
        if not grouped:
            raise AdapterIncompatible(f"YuE2 LoRA '{file.name}' has no recognized complete branches")
        return {"stages": stages, "grouped": grouped}

    @staticmethod
    def _yue2_declared_branches(tensors, _components):
        from core.models.yue2.yue2_lora import normalise_yue2_lora_state_dict
        return len(normalise_yue2_lora_state_dict(tensors))

    @property
    def _yue2_lora_session(self):
        session = getattr(self, "_yue2_lora_session_instance", None)
        if session is None:
            from core.adapters import AdapterSession
            session = AdapterSession(
                resolve_path=self._yue2_resolve_lora_path,
                warn=self._yue2_lora_warn,
                architecture="yue2",
                label="YuE2 LoRA",
                prepare_file=self._yue2_prepare_lora_file,
                count_declared_branches=self._yue2_declared_branches,
            )
            self._yue2_lora_session_instance = session
        return session

    @staticmethod
    def _yue2_build_lora_branch(request):
        from core.adapters import PreparedBranch, SHAPE_MISMATCH, build_adapter_branch, lora_branch_dtype
        stem = request.module_path.replace(".", "_")
        group = request.prepared["grouped"].get(stem)
        if group is None:
            return None
        branch = build_adapter_branch(
            request.base, group,
            metadata_alpha=float(request.file.metadata["lora_alpha"])
            if request.file.metadata.get("lora_alpha") else None,
            lora_dtype=lora_branch_dtype(request.base), lora_name=request.module_path,
        )
        if branch is SHAPE_MISMATCH:
            return branch
        return PreparedBranch(branch, request.file.strength)

    def _yue2_component(self, half):
        from core.adapters import AdapterComponent
        from core.models.yue2.yue2_lora import iter_yue2_lora_targets
        model = (self.yue2_components or {}).get("transformer")

        def targets(transformer):
            for target in iter_yue2_lora_targets(
                    transformer, half=half, scope={"attention": True, "mlp": True}):
                yield target.parent, target.attr, target.path

        return AdapterComponent(name="transformer", module=model,
                                iter_targets=targets,
                                build_branch=self._yue2_build_lora_branch)

    def _prepare_yue2_loras(self, configs):
        session = self._yue2_lora_session
        session.unload([self._yue2_component("ar")])
        return session.parse(configs or [])

    def _set_yue2_adapter_stage(self, files, stage):
        from core.models.yue2.yue2_lora import normalize_yue2_stages, stage_is_active
        session = self._yue2_lora_session
        session.unload([self._yue2_component("ar")])
        if stage is None:
            return 0
        selected = [
            file for file in files
            if stage_is_active(normalize_yue2_stages(
                str(file.metadata.get("yue2_apply_stages") or "")
            ), stage)
        ]
        if not selected:
            return 0
        half = "nar" if stage == "nar" else "ar"
        return session.load(selected, [self._yue2_component(half)]).applied

    def _generate_txt2aud_yue2(self, params, progress_callback=None, step_callback=None):
        from core.models.yue2.pipeline import generate
        from core.inference.cancellation import raise_if_cancelled
        if not self.yue2_components:
            raise ValueError("Load a complete YuE2 checkpoint before generating music")
        required = ("prompt", "lyrics", "audio_duration", "yue2_cot", "yue2_abc", "yue2_abc_max_tokens",
                    "temperature", "top_p", "top_k", "repetition_penalty", "guidance_scale",
                    "vae_decode_mode", "vae_tile_frames")
        missing = [name for name in required if name not in params]
        if missing:
            raise ValueError(f"YuE2 requires resolved API parameters: {missing}")
        if not str(params["lyrics"]).strip() or not str(params["prompt"]).strip():
            raise ValueError("YuE2 requires a music style and structured lyrics")
        if params.get("negative_prompt"):
            raise ValueError("YuE2 uses a protocol-defined negative branch")
        seed = params.get("seed")
        seed = random.randrange(2**32) if seed is None or int(seed) < 0 else int(seed)
        def cancelled():
            raise_if_cancelled()
            return False
        def progress(stage, n, total):
            if progress_callback:
                offsets = {"abc": (0, 1500), "semantic": (1500, 5500), "nar": (7000, 2500), "vae": (9500, 500)}
                start, width = offsets[stage]
                progress_callback(start + int(width * n / max(total, 1)), 10000)
        try:
            lora_files = self._prepare_yue2_loras(params.get("loras") or [])
            return generate(
                self.yue2_components, params, self.device, seed, cancelled, progress,
                set_adapter_stage=lambda stage: self._set_yue2_adapter_stage(lora_files, stage),
            )
        except ValueError as exc:
            if str(exc).startswith(("Prefix + requested generation budget exceeds",
                                    "Negative prefix + generation budget exceeds")):
                from api.error_handlers import ValidationError
                raise ValidationError("YuE2 prompt and duration exceed the model context", detail=str(exc)) from exc
            raise
