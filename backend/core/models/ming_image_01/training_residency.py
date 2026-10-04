"""Caller-owned encoder phases for the native shared training batch."""

from contextlib import contextmanager, ExitStack

import torch

from .block_offload import move_component

from .cache import configured_modes


class MingTrainingResidency:
    def __init__(self, components, *, device, latent_mode, text_mode):
        modes = configured_modes({"latent_encoding_mode": latent_mode, "text_encoding_mode": text_mode})
        self.components, self.device, self.modes = components, torch.device(device), modes
        self.active = None
        self.defer_restore = False

    @contextmanager
    def continuous_refill(self):
        if self.active is not None or self.defer_restore:
            raise RuntimeError("Ming refill phases must not overlap")
        self.defer_restore = True
        try:
            yield
        finally:
            self.defer_restore = False
            if self.device.type != "cpu":
                move_component(self.components.transformer, self.device)

    def _report_memory(self, phase, stage):
        if self.device.type == "cuda":
            print(f"[Ming Residency] {phase} {stage}: "
                  f"CUDA allocated={torch.cuda.memory_allocated(self.device) / 2**30:.2f} GiB "
                  f"reserved={torch.cuda.memory_reserved(self.device) / 2**30:.2f} GiB "
                  f"peak={torch.cuda.max_memory_allocated(self.device) / 2**30:.2f} GiB", flush=True)

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
                move_component(self.components.transformer, "cpu")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            encoder.to(device=self.device)
            self._report_memory(phase, "loaded")
            yield
        finally:
            try:
                self._report_memory(phase, "finished")
                if release_encoder:
                    encoder.to(device="cpu")
                    if self.device.type == "cuda":
                        # Mixed-resolution encoder work must not retain allocator pools across phases.
                        torch.cuda.empty_cache()
            finally:
                try:
                    if swap and not self.defer_restore:
                        move_component(self.components.transformer, self.device)
                    self._report_memory(phase, "released")
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
