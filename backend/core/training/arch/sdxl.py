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
        raise NotImplementedError("sdxl.encode_prompt: phase P4")

    def vae_encode(self, trainer, image_tensor, *, width, height):
        raise NotImplementedError("sdxl.vae_encode: phase P5")

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("sdxl.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        raise NotImplementedError("sdxl.train_step: phase P6")

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("sdxl.sample: phase P7")
