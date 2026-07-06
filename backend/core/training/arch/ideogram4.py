"""
Ideogram4ArchHandler — P0/P1 stub handler for arch "ideogram4".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import ArchHandler, SampleContext, TrainStepContext
from core.training.components.wiring import IDEOGRAM4_WIRING


class Ideogram4ArchHandler(ArchHandler):
    name = "ideogram4"
    wiring = IDEOGRAM4_WIRING

    def load_components(self, trainer) -> None:
        raise NotImplementedError("ideogram4.load_components: phase P3")

    def setup_block_swap(self, trainer) -> None:
        raise NotImplementedError("ideogram4.setup_block_swap: phase P3")

    def setup_attention_backend(self, trainer) -> None:
        raise NotImplementedError("ideogram4.setup_attention_backend: phase P3")

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        raise NotImplementedError("ideogram4.encode_prompt: phase P4")

    def vae_encode(self, trainer, image_tensor, *, width, height):
        raise NotImplementedError("ideogram4.vae_encode: phase P5")

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("ideogram4.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        raise NotImplementedError("ideogram4.train_step: phase P6")

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("ideogram4.sample: phase P7")
