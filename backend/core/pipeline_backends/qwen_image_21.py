"""Qwen-Image 2.1 generation backend."""

from __future__ import annotations

import random
from typing import Any, Dict

import torch
from PIL import Image, ImageFilter


GRID_ALIGN = 32


def _aligned(value: int, alignment: int = GRID_ALIGN) -> int:
    return max(alignment, int(value) // alignment * alignment)


class QwenImage21Mixin:
    def _validate_qwen_partition_arch(self, params):
        config = params.get("qwen_image_21_partition")
        if config is None:
            return
        from core.inference.qwen_image_21_partition import normalize_partition_config
        if normalize_partition_config(config)["mode"] != "off" and not self.is_qwen_image_21_model:
            from api.error_handlers import ValidationError
            raise ValidationError("Partition generation is only available for Qwen-Image 2.1")

    @staticmethod
    def _qwen21_lora_warn(message, code):
        print(f"[Qwen-Image 2.1 LoRA] WARNING: {message}")
        try:
            from api.generation_status import add_warning
            add_warning(message, code=code)
        except Exception:
            pass

    @staticmethod
    def _qwen21_resolve_lora_path(path):
        from core.extensions.lora_manager import lora_manager
        return lora_manager._resolve_lora_path(path)

    @property
    def _qwen21_lora_session(self):
        session = getattr(self, "_qwen21_lora_session_instance", None)
        if session is None:
            from core.adapters import AdapterSession
            session = AdapterSession(
                resolve_path=self._qwen21_resolve_lora_path,
                warn=self._qwen21_lora_warn,
                architecture="qwen_image_21",
                base_latent=getattr(self, "base_latent_identity", None),
                label="Qwen-Image 2.1 LoRA",
                message_label="Qwen-Image 2.1 LoRA",
                count_declared_branches=lambda tensors, _components: __import__(
                    "core.models.qwen_image_21.lora", fromlist=["declared_branch_count"]
                ).declared_branch_count(tensors),
                prepare_file=self._qwen21_prepare_lora_file,
                describe_zero_targets=self._qwen21_zero_target_message,
            )
            self._qwen21_lora_session_instance = session
        return session

    def _qwen21_prepare_lora_file(self, file):
        from core.models.qwen_image_21.lora import normalise_lora_state_dict
        from core.models.qwen_image_21.partition_global import read_partition_global_payload
        file.qwen_global_payload = read_partition_global_payload(
            self.qwen_image_21_components.get("transformer"), file.tensors, file.metadata,
            allow_alpha=True,
        )
        base_forward = str(file.metadata.get("qwen_base_forward") or "")
        if base_forward:
            if base_forward != "convrot_int8_bf16_backward_v1":
                raise ValueError(
                    f"Qwen-Image 2.1 LoRA '{file.name}' declares unsupported "
                    f"base forward {base_forward!r}"
                )
            components = self.qwen_image_21_components or {}
            variant = str(components.get("transformer_variant") or "bf16")
            if variant != "int8_convrot":
                raise ValueError(
                    f"Qwen-Image 2.1 LoRA '{file.name}' was trained against the "
                    "ConvRot INT8 base forward and requires an int8_convrot model; "
                    f"the loaded transformer is {variant!r}"
                )
        return normalise_lora_state_dict(file.tensors)

    @staticmethod
    def _qwen21_zero_target_message(file, counts):
        if (file.declared_branches == 0 and file.apply_to_unet
                and all(key.startswith("qwen_partition_global_adapter.") for key in file.tensors)
                and getattr(file, "qwen_global_payload", None) is not None):
            return None
        return (
            f"Qwen-Image 2.1 LoRA '{file.name}' matched no transformer attention targets; "
            "the file is for another architecture or uses an unsupported key layout."
        )

    def _qwen21_build_lora_branch(self, request):
        from core.adapters import PreparedBranch, SHAPE_MISMATCH
        from core.models.qwen_image_21.lora import build_lora_branch
        group = request.prepared.get(request.module_path)
        if group is None:
            return None
        branch = build_lora_branch(request.base, group, request.module_path)
        return branch if branch is SHAPE_MISMATCH else PreparedBranch(branch, request.file.strength)

    def _qwen21_lora_components(self):
        from core.adapters import AdapterComponent
        from core.models.qwen_image_21.lora import iter_lora_slots
        components = self.qwen_image_21_components or {}
        return [AdapterComponent(
            name="transformer",
            module=components.get("transformer"),
            iter_targets=iter_lora_slots,
            build_branch=self._qwen21_build_lora_branch,
        )]

    def _load_lora_qwen21(self, configs):
        self._unload_lora_qwen21()
        if not configs:
            return 0
        from core.adapters import AdapterIncompatible
        from core.rewiring.adapters import resolve_adapter_topology, HybridPayloadTransaction
        from core.models.qwen_image_21.residual_unet import restore_topology, validate_topology_identity
        model = self.qwen_image_21_components["transformer"]
        files = self._qwen21_lora_session.parse(configs)
        try:
            required = resolve_adapter_topology(model, files, validate_topology_identity)
            if required is not None and (getattr(model, "_qwen_partition_global_payload", None) is not None
                    or any(file.metadata.get("qwen_partition_global_adapter") for file in files)):
                raise ValueError("Residual topology excludes partition-global adapters")
        except ValueError as exc:
            raise AdapterIncompatible(str(exc), code="lora_rewiring_incompatible") from exc
        transaction = HybridPayloadTransaction(model, files, required, restore_topology)
        components = self._qwen21_lora_components()
        try:
            result = self._qwen21_lora_session.load(files, components)
            transaction.install()
            self._qwen21_lora_session.own_cleanup(components[0], transaction.restore)
        except Exception:
            try:
                self._qwen21_lora_session.unload(components)
            finally:
                transaction.restore()
            raise
        self._qwen21_loaded_files = [file for file, _counts in result.files]
        return result.applied

    def _unload_lora_qwen21(self):
        self._qwen21_loaded_files = []
        try:
            return self._qwen21_lora_session.unload(self._qwen21_lora_components())
        finally:
            saved = self.__dict__.pop("_qwen21_topology_restore", None)
            if saved is not None and saved[0]() is not None:
                from core.models.qwen_image_21.residual_unet import restore_topology
                restore_topology(saved[0](), saved[1])

    def _qwen_image_21_pipe(self):
        components = self.qwen_image_21_components
        pipe = components.get("pipeline")
        if pipe is None:
            from core.models.qwen_image_21.loader import build_pipeline

            pipe = build_pipeline(components)
            if self.device == "cuda":
                pipe.enable_model_cpu_offload(device=self.device)
            else:
                pipe.to(self.device)
            components["pipeline"] = pipe
        return pipe

    def _qwen_image_21_run(
        self, params: Dict[str, Any], *, images=None, progress_callback=None, step_callback=None
    ):
        pipe = self._qwen_image_21_pipe()
        vae_scale_factor = getattr(pipe, "vae_scale_factor", GRID_ALIGN // 2)
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        steps = int(params.get("steps", 40) or 40)
        width = _aligned(params.get("width", 1024) or 1024, vae_scale_factor * 2)
        height = _aligned(params.get("height", 1024) or 1024, vae_scale_factor * 2)
        cfg_scale = float(params.get("cfg_scale", 1.0) or 1.0)
        negative_prompt = (params.get("negative_prompt") or "") if cfg_scale > 1 else None

        def _flag(item, name):
            if isinstance(item, dict):
                return bool(item.get(name))
            return bool(getattr(item, name, False))

        requested_controlnets = params.get("controlnets") or []
        real_controlnets = [
            item for item in requested_controlnets
            if not _flag(item, "is_reference_guide")
            and not _flag(item, "is_style_transfer")
        ]
        if real_controlnets:
            from api.error_handlers import ValidationError
            raise ValidationError(
                "ControlNet is not available for Qwen-Image 2.1",
                detail=(
                    "Qwen-Image 2.1 has no compatible ControlNet module. Use its native "
                    "Reference Images, Reference Guide, or Style Transfer conditioning instead."
                ),
            )

        reference_guides = [
            item for item in (params.get("controlnet_images") or [])
            if item.get("is_reference_guide")
        ]
        style_transfers = [
            item for item in (params.get("style_transfers") or [])
            if item.get("image") is not None
        ]
        from core.inference.qwen_image_21_partition import resolve_partition_plan
        try:
            plan = resolve_partition_plan(params.get("qwen_image_21_partition"), height // vae_scale_factor, width // vae_scale_factor)
        except ValueError as exc:
            from api.error_handlers import ValidationError
            raise ValidationError(str(exc)) from exc
        if plan is not None and (reference_guides or style_transfers):
            from api.error_handlers import ValidationError
            raise ValidationError("Qwen partitioned generation currently excludes Reference Guide and Style Transfer")

        def callback(_pipe, index, timestep, callback_kwargs):
            if self.cancel_requested:
                _pipe._interrupt = True
            latents = callback_kwargs.get("latents")
            pred_original_sample = callback_kwargs.get("pred_original_sample")
            if progress_callback is not None:
                progress_callback(
                    index, steps, latents, None, pred_original_sample)
            if step_callback is not None:
                step_callback(index, timestep, latents)
            return callback_kwargs

        executor = None
        try:
            self._load_lora_qwen21(params.get("loras") or [])
            model = self.qwen_image_21_components["transformer"]
            topology = getattr(model, "_residual_unet_spec", None)
            if topology is not None:
                from api.error_handlers import ValidationError
                from core.rewiring.spec import digest, preview_plan
                if plan is not None or images or reference_guides or style_transfers:
                    raise ValidationError("Residual U-Net generation currently requires full-frame txt2img without references or styles")
                try:
                    preview_plan(topology, height // vae_scale_factor, width // vae_scale_factor)
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                params["dit_rewiring_actual"] = {"spec": topology, "spec_sha256": digest(topology), "kv_cache": False}
            else:
                params.pop("dit_rewiring_actual", None)
            partition_kwargs = {}
            before_step = self._qwen21_lora_session.set_step
            if plan is not None:
                from core.inference.qwen_image_21_partition import PartitionExecutor
                model = self.qwen_image_21_components["transformer"]
                payloads = []
                base_payload = getattr(model, "_qwen_partition_global_payload", None)
                if base_payload is not None:
                    payloads.append((base_payload, None))
                files = getattr(self, "_qwen21_loaded_files", [])
                payloads.extend((file.qwen_global_payload, file) for file in files
                                if file.apply_to_unet and file.qwen_global_payload is not None)
                executor = PartitionExecutor(
                    model, plan, payloads, step_sources=files,
                    use_cache=bool(params.get("qwen_image_21_kv_cache", True)),
                    cancelled=lambda: self.cancel_requested or pipe.interrupt,
                )
                if not payloads:
                    self._qwen21_lora_warn(
                        "Qwen partitioned generation has no global adapter; cross-region attention is absent.",
                        "qwen_partition_no_global_adapter",
                    )
                params["qwen_image_21_partition_actual"] = {
                    **plan.metadata(), "global_adapters": [
                        {"name": "full model" if source is None else source.name,
                         "strength": 1.0 if source is None else source.strength,
                         "step_range": None if source is None else source.step_range}
                        for _payload, source in payloads
                    ],
                }

                def before_step(index, total):
                    self._qwen21_lora_session.set_step(index, total)
                    executor.set_step(index, total)

                partition_kwargs["partition_executor"] = executor
            else:
                params.pop("qwen_image_21_partition_actual", None)
                for file in getattr(self, "_qwen21_loaded_files", []):
                    if file.qwen_global_payload is not None and file.apply_to_unet:
                        self._qwen21_lora_warn(
                            f"Qwen LoRA '{file.name}': full-frame generation ignores its global adapter.",
                            "qwen_partition_global_adapter_ignored",
                        )
            result = pipe(
                prompt=params.get("prompt", ""), image=images,
                negative_prompt=negative_prompt, true_cfg_scale=cfg_scale,
                height=height, width=width, num_inference_steps=steps,
                generator=generator, callback_on_step_end=callback,
                callback_on_step_end_tensor_inputs=["latents", "pred_original_sample"],
                use_kv_cache=topology is None and bool(params.get("qwen_image_21_kv_cache", True)),
                reference_guides=reference_guides,
                style_transfers=style_transfers,
                style_combine_mode=str(params.get("style_combine_mode", "stack") or "stack"),
                before_step_callback=before_step,
                **partition_kwargs,
            ).images[0]
        finally:
            if executor is not None:
                executor.close()
            self._unload_lora_qwen21()
        return result, seed, seed

    def _generate_txt2img_qwen_image_21(self, params, progress_callback=None, step_callback=None):
        images = params.get("ref_images") or None
        if images and len(images) > 10:
            raise ValueError("Qwen-Image 2.1 accepts at most 10 reference images")
        return self._qwen_image_21_run(
            params, images=images, progress_callback=progress_callback, step_callback=step_callback
        )

    def _generate_img2img_qwen_image_21(
        self, params, init_image, progress_callback=None, step_callback=None
    ):
        images = [init_image, *(params.get("ref_images") or [])]
        if len(images) > 10:
            raise ValueError("Qwen-Image 2.1 accepts at most 10 reference images")
        return self._qwen_image_21_run(
            params, images=images, progress_callback=progress_callback, step_callback=step_callback
        )

    def _generate_inpaint_qwen_image_21(
        self, params, init_image: Image.Image, mask_image: Image.Image,
        progress_callback=None, step_callback=None,
    ):
        rgba = init_image.convert("RGBA")
        mask = mask_image.convert("L").resize(rgba.size)
        rgba.putalpha(Image.eval(mask, lambda value: 255 - value))
        images = [rgba, *(params.get("ref_images") or [])]
        if len(images) > 10:
            raise ValueError("Qwen-Image 2.1 accepts at most 10 reference images")
        generated, seed, ancestral_seed = self._qwen_image_21_run(
            params, images=images, progress_callback=progress_callback, step_callback=step_callback
        )
        generated = generated.convert("RGB").resize(init_image.size)
        blur = float(params.get("mask_blur", 0) or 0)
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        generated.paste(init_image.convert("RGB"), mask=Image.eval(mask, lambda value: 255 - value))
        return generated, seed, ancestral_seed
