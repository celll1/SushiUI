"""Native Ming generation orchestration shared by API and trainer samples."""

from contextlib import nullcontext

from .conditioning import MingConditioning
from .prompting import prepare_inputs, encode_inputs
from .sampling import MingSampler, MingGenerationCancelled


class MingGenerator:
    def __init__(self, components):
        self.components = components
        self.sampler = MingSampler(components)

    def run(self, *, prompts, references, task, width, height, resolution,
            steps, cfg, seed, requested_layers, negative_prompt="", output_type="pil",
            cancelled=None, step_callback=None, phase_context=None, reference_guides=None, partition_plan=None):
        """Residency contexts belong to the caller; every exit releases their resources."""
        def check_cancel():
            if cancelled is not None and cancelled():
                raise MingGenerationCancelled("Ming generation cancelled")

        check_cancel()
        if negative_prompt:
            raise ValueError("Ming native guidance uses zero conditioning; negative prompts are unsupported")
        from .reference_guide import normalize_guides, prepare_guides
        guides = normalize_guides(reference_guides)
        prepared = prepare_inputs(self.components.processor, prompts, references=references,
                                  width=width, height=height, resolution=resolution)
        reference_pixels = prepared.generation_references()
        for images in references:
            self.sampler.profile.validate_task(task, has_reference_image=bool(images), num_layers=requested_layers)
        context = phase_context if phase_context is not None else lambda _phase: nullcontext()
        check_cancel()
        with context("conditioner"):
            check_cancel()
            condition = encode_inputs(self.components.conditioner, prepared, requires_grad=False)
        check_cancel()
        with context("denoising"):
            transformer = self.components.transformer
            if self.components.vae.device != transformer.device:
                raise ValueError("Ming denoising context must place VAE and DiT on the same execution device")
            condition = MingConditioning(condition.queries.to(device=transformer.device, dtype=transformer.dtype),
                                         condition.direct.to(device=transformer.device, dtype=transformer.dtype))
            if reference_pixels is not None:
                reference_pixels = reference_pixels.to(device=transformer.device, dtype=self.components.vae.dtype)
            check_cancel()
            prepared_guides = prepare_guides(self.components.vae, guides, width=width, height=height,
                batch=len(prompts), seed=seed, check_cancel=check_cancel)
            return self.sampler.run(task=task, conditioning=condition, width=width, height=height,
                steps=steps, cfg=cfg, seed=seed, requested_layers=requested_layers,
                reference_pixels=reference_pixels, negative_prompt=negative_prompt, output_type=output_type,
                cancelled=cancelled, step_callback=step_callback, reference_guides=prepared_guides,partition_plan=partition_plan)
