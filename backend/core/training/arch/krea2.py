"""
Krea2ArchHandler — P0/P1 stub handler for arch "krea2".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import KREA2_WIRING


class Krea2ArchHandler(ArchHandler):
    name = "krea2"
    wiring = KREA2_WIRING
    pixel_align = 16  # vae_scale(8) * patch(2); latent grid = pixel/16
    # noisy = (1-sigma)*latents + sigma*noise (ops/krea2_ops.py, "sigma=1 ->
    # noise" comment). sampler t=0 is clean.
    timestep_convention = "t0"

    def load_components(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with the base_trainer load-time
        # dispatcher + _load_checkpoint_as_base, which cannot route via self.arch —
        # self.arch binds after loading; see the construction-order note in krea2_ops).
        from core.training.ops import krea2_ops
        krea2_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with the base_trainer
        # setup_krea2_block_swap delegator, called late by mode subclasses).
        from core.training.ops import krea2_ops
        krea2_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with base_trainer delegator).
        from core.training.ops import krea2_ops
        krea2_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/krea2_ops (shared with base_trainer delegator).
        from core.training.ops import krea2_ops
        return krea2_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import krea2_ops
        return krea2_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("krea2.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6c: verbatim body in ops/krea2_ops.train_step. ctx fields map 1:1 to
        # the previous train_step_krea2 kwargs bundle.
        from core.training.ops import krea2_ops
        return krea2_ops.train_step(
            trainer,
            latents=ctx.latents,
            encoder_features=ctx.encoder_features,
            encoder_mask=ctx.encoder_mask,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/krea2_ops.generate_sample.
        from core.training.ops import krea2_ops
        return krea2_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
