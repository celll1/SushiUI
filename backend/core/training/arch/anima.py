"""
AnimaArchHandler — P0/P1 stub handler for arch "anima".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import ANIMA_WIRING


class AnimaArchHandler(ArchHandler):
    name = "anima"
    wiring = ANIMA_WIRING
    pixel_align = 16  # vae_scale(8) * patch_spatial(2); patchify asserts on non-/16 dims

    def load_components(self, trainer) -> None:
        # P3b: body lives in ops/anima_ops (shared with the base_trainer
        # load-time dispatcher, which cannot route via self.arch — self.arch
        # binds after loading; see the construction-order note in anima_ops).
        from core.training.ops import anima_ops
        anima_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3b: body lives in ops/anima_ops (shared with the base_trainer
        # setup_anima_block_swap delegator, called late by mode subclasses).
        from core.training.ops import anima_ops
        anima_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        # P3b: body lives in ops/anima_ops (shared with base_trainer delegator).
        from core.training.ops import anima_ops
        anima_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/anima_ops (shared with base_trainer delegator).
        from core.training.ops import anima_ops
        return anima_ops.encode_prompt(trainer, prompt)

    def collate_aux(self, trainer, batch) -> dict:
        # P4: body lives in ops/anima_ops (shared with base_trainer call sites).
        # Overrides the base_arch no-op default; the train-loop dispatches here
        # via ``self.arch.collate_aux`` only on the ``is_anima`` branch.
        from core.training.ops import anima_ops
        return anima_ops.collate_aux(trainer, batch)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import anima_ops
        return anima_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("anima.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6b: verbatim body in ops/anima_ops.train_step. The LLM-adapter payload
        # rides in ctx.anima_aux (the dispatcher extracts it from mnt_attention_mask
        # when that is a dict); text tensor is ctx.text_embeddings.
        from core.training.ops import anima_ops
        return anima_ops.train_step(
            trainer,
            latents=ctx.latents,
            prompt_embeds=ctx.text_embeddings,
            anima_aux=ctx.anima_aux,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/anima_ops.generate_sample.
        from core.training.ops import anima_ops
        return anima_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
