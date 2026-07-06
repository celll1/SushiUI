"""
Flux2ArchHandler — P0/P1 stub handler for arch "flux2".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import FLUX2_WIRING


class Flux2ArchHandler(ArchHandler):
    name = "flux2"
    wiring = FLUX2_WIRING

    def load_components(self, trainer) -> None:
        # P3c: body lives in ops/flux2_ops (shared with the base_trainer load-time
        # dispatcher, which cannot route via self.arch — self.arch binds after
        # loading; see the construction-order note in flux2_ops).
        from core.training.ops import flux2_ops
        flux2_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3c: FLUX.2 has NO dedicated late setup_*_block_swap method — block swap
        # is wired INSIDE the loader (block_swap_h2d_args + wire_block_swap_driver),
        # not via a post-adapter conductor call from the mode subclasses. No-op.
        return None

    def setup_attention_backend(self, trainer) -> None:
        # P3c: body lives in ops/flux2_ops (shared with base_trainer delegator).
        from core.training.ops import flux2_ops
        flux2_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        raise NotImplementedError("flux2.encode_prompt: phase P4")

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import flux2_ops
        return flux2_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("flux2.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6c: verbatim body in ops/flux2_ops.train_step. The ctx-build branch in
        # _execute_forward_backward already produced packed latents / img_ids /
        # txt_ids / detached reference_latents_nested via the spine packing
        # helpers; ctx fields map 1:1 to the previous train_step_flux2 kwargs.
        from core.training.ops import flux2_ops
        return flux2_ops.train_step(
            trainer,
            latents=ctx.latents,
            prompt_embeds=ctx.text_embeddings,
            img_ids=ctx.img_ids,
            txt_ids=ctx.txt_ids,
            timesteps=ctx.timesteps,
            guidance=ctx.guidance,
            reference_latents_nested=ctx.reference_latents_nested,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("flux2.sample: phase P7")
