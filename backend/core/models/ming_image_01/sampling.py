"""Ming task validation, native sampling and ordered layer results."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import torch

from .results import MingResultGroup
from .conditioning import MingConditioning
from .vendor.inference_profile import InferenceProfile
from .vendor.diffusion.pipeline import ImageGenerationPipeline


class MingGenerationCancelled(RuntimeError):
    pass


class MingTransformerBridge(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self.config = transformer.config
        self.in_channels = transformer.in_channels
        self.partition_executor = None
        self.style_executor = None

    @property
    def dtype(self):
        return self.transformer.dtype

    @property
    def device(self):
        return self.transformer.device

    def forward(self, hidden_states, timestep, encoder_hidden_states, *,
                ref_hidden_states=None, encoder_hidden_states_2=None, return_dict=False):
        if self.style_executor is not None:
            return self.style_executor.predict(hidden_states,timestep,encoder_hidden_states,
                ref_hidden_states=ref_hidden_states,encoder_hidden_states_2=encoder_hidden_states_2,return_dict=return_dict)
        if self.partition_executor is not None:
            return self.partition_executor.predict(hidden_states,timestep,encoder_hidden_states,
                ref_hidden_states=ref_hidden_states,encoder_hidden_states_2=encoder_hidden_states_2,return_dict=return_dict)
        return self.transformer(x=hidden_states, t=timestep, cap_feats=encoder_hidden_states,
            ref_x=ref_hidden_states, cap_feats_2=encoder_hidden_states_2, return_dict=return_dict)


def build_pipeline(components) -> ImageGenerationPipeline:
    pipe = ImageGenerationPipeline(scheduler=components.scheduler, vae=components.vae,
                                   transformer=MingTransformerBridge(components.transformer),
                                   text_encoder=None, tokenizer=None)
    pipe.set_progress_bar_config(disable=True)
    return pipe


class MingSampler:
    def __init__(self, components):
        self.components = components
        self.pipe = build_pipeline(components)
        config = components.artifact.configs["transformer"]
        self.profile = InferenceProfile.from_dict({
            "schema_version": 1, "inference_profile": components.artifact.profile,
            "alignment_padding_mode": config["alignment_padding_mode"],
            "multi_frame_output": config["multi_frame_output"],
            "vae_input_channels": components.artifact.configs["vae"]["input_channels"],
            "vae_sample_mode": "argmax"})

    @torch.no_grad()
    def run(self, *, task: str, conditioning: MingConditioning, width: int, height: int,
            steps: int, cfg: float, seed: int, requested_layers: int,
            reference_pixels=None, negative_prompt: str = "", output_type: str = "pil",
            cancelled: Callable[[], bool] | None = None, step_callback=None, reference_guides=None, partition_plan=None, styles=None, style_combine_mode=None, preview_callback=None) -> list[MingResultGroup]:
        if negative_prompt:
            raise ValueError("Ming native guidance uses zero conditioning; negative prompts are unsupported")
        if output_type not in {"pil", "latent"}:
            raise ValueError("Ming output_type must be pil or latent")
        if type(requested_layers) is not int or requested_layers < 1:
            raise ValueError("Ming requested layer count must be an integer >= 1")
        if any(type(value) is not int or value <= 0 for value in (width, height)):
            raise ValueError("Ming dimensions must be positive integers")
        self.profile.validate_task(task, has_reference_image=reference_pixels is not None,
                                   num_layers=requested_layers)
        if type(steps) is not int or steps <= 0 or cfg is None:
            raise ValueError("Ming sampling parameters must be explicitly resolved before sampling")
        sampling = self.profile.resolve_sampling_parameters(steps=steps, cfg=cfg)
        frames = requested_layers + 1 if task == "layer-decompose" else 1
        if conditioning.queries.ndim != 3 or conditioning.direct.ndim != 3:
            raise ValueError("Ming conditioning streams must have batch/token/feature axes")
        config = self.components.artifact.configs["transformer"]
        if (conditioning.queries.shape[-1] != config["cap_feat_dim"]
                or conditioning.direct.shape[-1] != config["dim"]):
            raise ValueError("Ming conditioning feature dimensions disagree with the checkpoint")
        batch = conditioning.queries.shape[0]
        if batch <= 0:
            raise ValueError("Ming conditioning batch must not be empty")
        if conditioning.direct.shape[0] != batch:
            raise ValueError("Ming conditioning batch sizes differ")
        if reference_pixels is not None and (reference_pixels.ndim != 4
                or reference_pixels.shape[0] != batch or reference_pixels.shape[1] % 4
                or reference_pixels.shape[1] == 0 or reference_pixels.shape[-2:] != (height, width)):
            raise ValueError("Ming references must be matching-size RGBA channel groups with the same batch")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Ming seed must be an integer in [0, 2**63)")

        def check_cancel(*_args):
            if cancelled is not None and cancelled():
                raise MingGenerationCancelled("Ming generation cancelled")

        preview_state={"index":0,"clean":None}

        def callback(_pipe, index, timestep, values):
            check_cancel()
            if not torch.isfinite(values["latents"]).all():
                raise ValueError(f"Ming denoising produced nonfinite latents at step {index + 1}")
            if reference_guides:
                from .reference_guide import blend_guides
                values["latents"] = blend_guides(values["latents"], reference_guides,
                    index=index, total=sampling.steps, sigma=_pipe.scheduler.sigmas[index+1],
                    profile=self.profile.inference_profile)
            if preview_callback is not None:
                preview_callback(index,sampling.steps,timestep,values["latents"],preview_state["clean"])
                preview_state["clean"]=None
                preview_state["index"]=index+1
            if step_callback is not None:
                step_callback(index, sampling.steps, timestep, values["latents"])
            return values

        check_cancel()
        negative = conditioning.zero()
        self.pipe._interrupt = False
        handle = self.components.transformer.register_forward_pre_hook(check_cancel)
        executor = style_executor = None
        try:
            if styles:
                if partition_plan is not None:
                    raise ValueError("Ming partition generation excludes style transfer")
                from .style_generation import MingStyleExecutor
                style_executor = MingStyleExecutor(self.components.transformer,self.pipe.scheduler,styles,
                    batch=batch,total=sampling.steps,combine_mode=style_combine_mode,cancelled=cancelled or (lambda:False))
                self.pipe.transformer.style_executor = style_executor
            if partition_plan is not None:
                from .partition_generation import MingPartitionExecutor
                executor = MingPartitionExecutor(self.components.transformer,partition_plan,cancelled=cancelled or (lambda:False))
                self.pipe.transformer.partition_executor = executor
            from .generation_preview import scheduler_preview
            from core.inference.callback_utils import callback_requests
            requested=lambda index:callback_requests(preview_callback,"wants_predicted_x0",index,sampling.steps)
            preview_scope=scheduler_preview(self.pipe.scheduler,preview_state,requested) if preview_callback is not None else nullcontext()
            with preview_scope:
                result = self.pipe(
                    height=height, width=width, num_inference_steps=sampling.steps,
                    guidance_scale=sampling.cfg, generator=torch.Generator(device="cpu").manual_seed(seed),
                    num_frames_per_prompt=frames, ref_hidden_states=reference_pixels,
                    prompt_embeds=list(conditioning.queries.unbind(0)),
                    negative_prompt_embeds=list(negative.queries.unbind(0)),
                    prompt_embeds_2=list(conditioning.direct.unbind(0)),
                    negative_prompt_embeds_2=list(negative.direct.unbind(0)),
                    sample_mode="argmax", output_type=output_type,
                    callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
            check_cancel()
            outputs = result.images
            if len(outputs) != batch * frames:
                raise ValueError("Ming native output count disagrees with target-frame contract")
            roles = ["auxiliary_canvas"] + [f"layer_{i}" for i in range(1, frames)] if frames > 1 else ["image"]
            groups = []
            for batch_index in range(batch):
                values = [outputs[frame * batch + batch_index] for frame in range(frames)]
                if output_type == "pil" and any(image.mode != "RGBA" for image in values):
                    raise ValueError("Ming native output lost RGBA channels")
                groups.append(MingResultGroup(values, roles.copy(), self.profile.inference_profile,
                    requested_layers, frames, seed, width, height, output_type))
            return groups
        finally:
            if style_executor is not None:
                self.pipe.transformer.style_executor = None
                style_executor.close()
            if executor is not None:
                self.pipe.transformer.partition_executor = None
                executor.close()
            handle.remove()
            self.pipe._interrupt = False
            self.pipe.maybe_free_model_hooks()
