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
        raise NotImplementedError("minit2i.encode_prompt: phase P4")

    def vae_encode(self, trainer, image_tensor, *, width, height):
        raise NotImplementedError("minit2i.vae_encode: phase P5")

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("minit2i.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        raise NotImplementedError("minit2i.train_step: phase P6")

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("minit2i.sample: phase P7")
