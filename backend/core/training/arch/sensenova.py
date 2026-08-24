"""SenseNova B1 LoRA training architecture handler."""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import SENSENOVA_WIRING


class SenseNovaArchHandler(ArchHandler):
    name = "sensenova"
    wiring = SENSENOVA_WIRING
    pixel_align = 32

    def load_components(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        raise NotImplementedError("SenseNova training block swap is not implemented")

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(
        self, trainer, prompt, *, requires_grad: bool = False, reference_image_paths=None
    ):
        from core.training.ops import sensenova_ops

        return sensenova_ops.encode_prompt(
            trainer,
            prompt,
            requires_grad=requires_grad,
            reference_image_paths=reference_image_paths,
        )

    def vae_encode(
        self,
        trainer,
        image_tensor,
        *,
        image=None,
        width=None,
        height=None,
        vae_device=None,
        debug_preprocessing=False,
    ):
        from core.training.ops import sensenova_ops

        return sensenova_ops.vae_encode(trainer, image_tensor)

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("SenseNova is pixel-space and has no VAE decoder")

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import sensenova_ops

        return sensenova_ops.train_step(
            trainer,
            images=ctx.latents,
            prefix=ctx.sensenova_prefix,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import sensenova_ops

        return sensenova_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
            reference_image_path=sample_ctx.reference_image_path,
            condition_image_path=sample_ctx.condition_image_path,
        )
