"""Sequential native conditioner/DiT residency without implicit dtype changes."""

from contextlib import contextmanager

import torch

from .block_offload import move_component


class MingPhaseResidency:
    def __init__(self, components, *, device, cpu_text_encoding):
        self.components = components
        self.device = torch.device(device)
        self.cpu_text_encoding = cpu_text_encoding
        self.active = None

    def _cpu_all(self):
        errors = []
        for module in (self.components.conditioner, self.components.transformer, self.components.vae):
            try:
                move_component(module, "cpu")
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Ming phase cleanup failed") from errors[0]

    @contextmanager
    def __call__(self, phase):
        if phase not in {"conditioner", "denoising"}:
            raise ValueError("Unknown Ming residency phase")
        if self.active is not None:
            raise RuntimeError("Ming residency phases cannot overlap")
        self.active = phase
        try:
            self._cpu_all()
            if phase == "conditioner":
                target = "cpu" if self.cpu_text_encoding else self.device
                self.components.conditioner.to(device=target)
            else:
                move_component(self.components.transformer, self.device)
                self.components.vae.to(device=self.device)
            yield
        finally:
            try:
                self._cpu_all()
            finally:
                self.active = None
