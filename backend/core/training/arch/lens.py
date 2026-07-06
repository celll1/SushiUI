"""
LensArchHandler — P0/P1 stub handler for arch "lens".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import LENS_WIRING


class LensArchHandler(ArchHandler):
    name = "lens"
    wiring = LENS_WIRING

    def load_components(self, trainer) -> None:
        # P3b: body lives in ops/lens_ops (shared with the base_trainer
        # load-time dispatcher, which cannot route via self.arch — self.arch
        # binds after loading; see the construction-order note in lens_ops).
        from core.training.ops import lens_ops
        lens_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3b: body lives in ops/lens_ops (shared with the base_trainer
        # setup_lens_block_swap delegator, called late by mode subclasses).
        from core.training.ops import lens_ops
        lens_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        # P3b: body lives in ops/lens_ops (shared with base_trainer delegator).
        from core.training.ops import lens_ops
        lens_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/lens_ops (shared with base_trainer delegator).
        from core.training.ops import lens_ops
        return lens_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import lens_ops
        return lens_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("lens.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6b: verbatim body in ops/lens_ops.train_step. encoder_features rides in
        # ctx.encoder_features, encoder mask in ctx.encoder_mask, latent geometry in
        # ctx.latent_h/latent_w (from lens_latent_shape).
        from core.training.ops import lens_ops
        return lens_ops.train_step(
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
        raise NotImplementedError("lens.sample: phase P7")
