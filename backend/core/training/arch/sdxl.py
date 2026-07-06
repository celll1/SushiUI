"""
SDXLArchHandler — P0/P1 stub handler for arch "sdxl".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import SDXL_WIRING


class SDXLArchHandler(ArchHandler):
    name = "sdxl"
    wiring = SDXL_WIRING

    def load_components(self, trainer) -> None:
        # P3a: ONE loader serves both SD1.5 and SDXL; it SETS trainer.is_sdxl.
        # Body lives in ops/sd_sdxl_ops (shared with the base_trainer load-time
        # dispatcher, which cannot route via self.arch — is_sdxl is only final
        # after this returns; see the construction-order note in sd_sdxl_ops).
        from core.training.ops import sd_sdxl_ops
        sd_sdxl_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3a: SD/SDXL has NO dedicated setup_*_block_swap method (U-Net path
        # uses component offload, not the block-swap conductor). No-op.
        return None

    def setup_attention_backend(self, trainer) -> None:
        # P3a: body lives in ops/sd_sdxl_ops (shared with base_trainer delegator).
        from core.training.ops import sd_sdxl_ops
        sd_sdxl_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: the SD/SDXL top dispatcher (BaseTrainer.encode_prompt) STAYS in the
        # spine — it selects custom-TE / simple / chunked and the three bodies
        # live in ops/sd_sdxl_ops. This handler routes back through that
        # dispatcher (no circularity: the dispatcher calls ops, never self.arch).
        return trainer.encode_prompt(prompt, requires_grad=requires_grad)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        # P5: SDXL shares the SD/SDXL VAE branch body with SD1.5.
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("sdxl.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6a: SDXL shares the SD/SDXL train_step body with SD1.5 (verbatim in
        # ops/sd_sdxl_ops). ctx fields map 1:1 to the previous kwargs bundle.
        from core.training.ops import sd_sdxl_ops
        return sd_sdxl_ops.train_step(
            trainer,
            latents=ctx.latents,
            text_embeddings=ctx.text_embeddings,
            pooled_embeddings=ctx.pooled_embeddings,
            time_ids=ctx.time_ids,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("sdxl.sample: phase P7")
