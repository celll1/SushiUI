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
    pixel_align = 16  # vae_scale(8) * patch(2); latent grid = pixel/16
    text_seq_axis = 2  # encode_prompt returns [1, num_layers, L, D]
    # noisy_latents = (1-sigma)*latents + sigma*noise (ops/lens_ops.py train_step).
    # sampler t=0 is clean.
    timestep_convention = "t0"
    # The inference uncond branch for a blank negative is zero features plus an
    # all-false mask at the positive's own length
    # (lens_pipeline_ops.encode_prompt), so the aligned null is reachable by
    # rewriting the already-collated conditioning. Mirrored for the API process
    # by api/arch_capabilities.CFG_NULL_STAGE_BY_ARCH.
    cfg_null_stage = "collated"

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

    def apply_cfg_null_collated(self, trainer, conditioning, auxiliary,
                                drop_mask):
        from core.training.ops import lens_ops
        return lens_ops.apply_cfg_null_collated(
            conditioning, auxiliary, drop_mask)

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6b: verbatim body in ops/lens_ops.train_step. encoder_features rides in
        # ctx.encoder_features, encoder mask in ctx.encoder_mask, latent geometry in
        # ctx.latent_h/latent_w (from lens_latent_shape).
        from core.training.ops import lens_ops
        # Collated-stage rewrite, out of place: these conditioning tensors are
        # the batch's, reused by every MNT iteration.
        encoder_features, encoder_mask = self.apply_cfg_null_step(
            trainer, ctx, ctx.encoder_features, ctx.encoder_mask)
        return lens_ops.train_step(
            trainer,
            latents=ctx.latents,
            encoder_features=encoder_features,
            encoder_mask=encoder_mask,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/lens_ops.generate_sample.
        from core.training.ops import lens_ops
        return lens_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
