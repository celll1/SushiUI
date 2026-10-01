"""Caller-owned encoder phases for the native shared training batch."""

from contextlib import contextmanager, ExitStack

import torch

from .cache import configured_modes


class MingTrainingResidency:
    def __init__(self, components, *, device, latent_mode, text_mode):
        modes = configured_modes({"latent_encoding_mode": latent_mode, "text_encoding_mode": text_mode})
        self.components, self.device, self.modes = components, torch.device(device), modes
        self.active = None

    @contextmanager
    def __call__(self, phase):
        if phase not in {"vae", "conditioner"}:
            raise ValueError("Unknown Ming training encoder phase")
        if self.active is not None:
            raise RuntimeError("Ming training encoder phases must not overlap")
        encoder = getattr(self.components, phase)
        mode = self.modes["latent_encoding_mode" if phase == "vae" else "text_encoding_mode"]
        if mode == "cpu_prefetch":
            raise NotImplementedError("Ming CPU prefetch requires its native worker connection")
        swap = mode in {"swap_onthefly", "pre_encoded_cache"} and self.device.type != "cpu"
        release_encoder = phase == "vae" or encoder.cacheable
        self.active = phase
        try:
            if swap:
                self.components.transformer.to(device="cpu")
            encoder.to(device=self.device)
            yield
        finally:
            try:
                if release_encoder:
                    encoder.to(device="cpu")
            finally:
                try:
                    if swap:
                        self.components.transformer.to(device=self.device)
                finally:
                    self.active = None


def configure_training_residency(trainer, *, latent_mode, text_mode):
    components = trainer.ming_image_01_components
    policy = MingTrainingResidency(components, device=trainer.device,
                                   latent_mode=latent_mode, text_mode=text_mode)
    if text_mode == "cpu_prefetch" and not components.conditioner.cacheable:
        raise ValueError("Ming CPU prefetch requires frozen conditioner ownership")
    components.vae.to(device="cpu")
    components.conditioner.to(device="cpu")
    trainer.ming_training_residency = policy


@contextmanager
def grouped_encoder_phase(policy, phase):
    """Open one phase lazily, only if a grouped item actually needs encoding."""
    if policy is None:
        yield None
        return
    started = active = closed = False
    with ExitStack() as stack:
        @contextmanager
        def enter(requested):
            nonlocal started, active
            if closed or active or requested != phase:
                raise RuntimeError("Invalid Ming grouped encoder phase use")
            if not started:
                stack.enter_context(policy(phase))
                started = True
            active = True
            try:
                yield
            finally:
                active = False
        try:
            yield enter
        finally:
            closed = True
