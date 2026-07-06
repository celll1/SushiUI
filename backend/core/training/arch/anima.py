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
        raise NotImplementedError("anima.train_step: phase P6")

    def sample(self, trainer, sample_ctx: SampleContext):
        raise NotImplementedError("anima.sample: phase P7")
