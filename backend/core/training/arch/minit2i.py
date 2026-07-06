"""
MiniT2IArchHandler — P0/P1 stub handler for arch "minit2i".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import MINIT2I_WIRING


class MiniT2IArchHandler(ArchHandler):
    name = "minit2i"
    wiring = MINIT2I_WIRING

    def load_components(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with the base_trainer load-time
        # dispatcher + _load_checkpoint_as_base, which cannot route via self.arch —
        # self.arch binds after loading; see the construction-order note in minit2i_ops).
        from core.training.ops import minit2i_ops
        minit2i_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with the base_trainer
        # setup_minit2i_block_swap delegator, called late by mode subclasses).
        from core.training.ops import minit2i_ops
        minit2i_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with base_trainer delegator).
        from core.training.ops import minit2i_ops
        minit2i_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/minit2i_ops (shared with base_trainer delegator).
        from core.training.ops import minit2i_ops
        return minit2i_ops.encode_prompt(trainer, prompt, requires_grad=requires_grad)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        # P5: both minit2i sub-branches (pixel-space no-VAE + latent-space) live in
        # ops/minit2i_ops; self-contained (dispatched before the shared VAE staging).
        from core.training.ops import minit2i_ops
        return minit2i_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("minit2i.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6c: verbatim body in ops/minit2i_ops.train_step. ctx fields map 1:1 to
        # the previous train_step_minit2i kwargs bundle (mnt_latents is the
        # pixel-space image tensor; text_embeds/attention_mask carry FLAN-T5).
        from core.training.ops import minit2i_ops
        return minit2i_ops.train_step(
            trainer,
            images=ctx.latents,
            text_embeds=ctx.text_embeddings,
            attention_mask=ctx.attention_mask,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            repa_pixels=ctx.repa_pixels,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/minit2i_ops.generate_sample.
        from core.training.ops import minit2i_ops
        return minit2i_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
