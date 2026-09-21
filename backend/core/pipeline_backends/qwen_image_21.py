"""Qwen-Image 2.1 generation backend."""

from __future__ import annotations

import random
from typing import Any, Dict

import torch
from PIL import Image, ImageFilter


GRID_ALIGN = 32


def _aligned(value: int) -> int:
    return max(GRID_ALIGN, int(value) // GRID_ALIGN * GRID_ALIGN)


class QwenImage21Mixin:
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
        return normalise_lora_state_dict(file.tensors)

    @staticmethod
    def _qwen21_zero_target_message(file, counts):
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
        return self._qwen21_lora_session.load(configs, self._qwen21_lora_components()).applied

    def _unload_lora_qwen21(self):
        return self._qwen21_lora_session.unload(self._qwen21_lora_components())

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
        seed = int(params.get("seed", -1))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        steps = int(params.get("steps", 40) or 40)
        width = _aligned(params.get("width", 1024) or 1024)
        height = _aligned(params.get("height", 1024) or 1024)
        negative_prompt = params.get("negative_prompt") or None
        cfg_scale = float(params.get("cfg_scale", 1.0) or 1.0)

        def callback(_pipe, index, timestep, callback_kwargs):
            if self.cancel_requested:
                _pipe._interrupt = True
            if progress_callback is not None:
                progress_callback(index + 1, steps)
            if step_callback is not None:
                step_callback(index, timestep, callback_kwargs.get("latents"))
            return callback_kwargs

        self._load_lora_qwen21(params.get("loras") or [])
        try:
            result = pipe(
                prompt=params.get("prompt", ""), image=images,
                negative_prompt=negative_prompt, true_cfg_scale=cfg_scale,
                height=height, width=width, num_inference_steps=steps,
                generator=generator, callback_on_step_end=callback,
                use_kv_cache=bool(params.get("qwen_image_21_kv_cache", True)),
            ).images[0]
        finally:
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
