"""
ZImageArchHandler — P0/P1 stub handler for arch "zimage".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import ZIMAGE_WIRING


class ZImageArchHandler(ArchHandler):
    name = "zimage"
    wiring = ZIMAGE_WIRING
    wires_sample_step_progress = True
    pixel_align = 16  # vae_scale(8) * patch(2)
    # noisy = (1-t)*latents + t*noise via add_noise_unified(noise_process="flow")
    # (ops/zimage_ops.py train_step; default and only supported noise_process).
    # sampler t=0 is clean.
    timestep_convention = "t0"

    def load_components(self, trainer) -> None:
        # P3a: body lives in ops/zimage_ops (shared with the base_trainer
        # load-time dispatcher, which cannot route via self.arch — see the
        # construction-order note in zimage_ops).
        from core.training.ops import zimage_ops
        zimage_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3a: Z-Image has NO dedicated setup_*_block_swap method — block-swap
        # wiring (LayerOffloadConductor) is done inline inside load_components.
        # No-op here (nothing to move); block swap is already set up at load.
        return None

    def setup_attention_backend(self, trainer) -> None:
        # P3a: body lives in ops/zimage_ops (shared with base_trainer delegator).
        from core.training.ops import zimage_ops
        zimage_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/zimage_ops (shared with base_trainer delegator).
        # Z-Image encode is grad-free; requires_grad is unused (matches the
        # original encode_prompt_zimage signature).
        from core.training.ops import zimage_ops
        return zimage_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import zimage_ops
        return zimage_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("zimage.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6b: verbatim body in ops/zimage_ops.train_step. ctx fields map 1:1 to
        # the previous train_step_zimage kwargs bundle (attention_mask is a tensor).
        from core.training.ops import zimage_ops
        return zimage_ops.train_step(
            trainer,
            latents=ctx.latents,
            prompt_embeds=ctx.text_embeddings,
            attention_mask=ctx.attention_mask,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/zimage_ops.generate_sample (+ its private
        # helpers _run_zimage_denoising_loop / _decode_zimage_latents).
        from core.training.ops import zimage_ops
        return zimage_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
            step_progress_callback=sample_ctx.step_progress_callback,
        )
