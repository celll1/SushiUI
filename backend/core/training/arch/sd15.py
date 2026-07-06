"""
SD15ArchHandler — P0/P1 stub handler for arch "sd15".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import SD15_WIRING


class SD15ArchHandler(ArchHandler):
    name = "sd15"
    wiring = SD15_WIRING

    def load_components(self, trainer) -> None:
        raise NotImplementedError("sd15.load_components: phase P3")

    def setup_block_swap(self, trainer) -> None:
        raise NotImplementedError("sd15.setup_block_swap: phase P3")

    def setup_attention_backend(self, trainer) -> None:
        raise NotImplementedError("sd15.setup_attention_backend: phase P3")

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        raise NotImplementedError("sd15.encode_prompt: phase P4")

    def vae_encode(self, trainer, image_tensor, *, width, height):
        raise NotImplementedError("sd15.vae_encode: phase P5")

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("sd15.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        raise NotImplementedError("sd15.train_step: phase P6")

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("sd15.sample: phase P7")
