"""Borrow a resident frozen base as teacher while fitting hybrid modules."""
from contextlib import contextmanager

import torch

from .hybrid_spec import validate_hybrid_spec


class NativeTeacher:
    def __init__(self, model, declaration, parameters):
        self.model = model
        self.declaration = declaration
        self.parameters = parameters
        self._active = False

    def __call__(self, *args, **kwargs):
        if not self._active or getattr(self.model, "_hybrid_teacher_owner", None) is not self:
            raise RuntimeError("Native teacher cannot be used outside its initialization context")
        if self.model._residual_unet_spec != self.declaration:
            raise RuntimeError("Teacher requires exclusive ownership of the hybrid topology")
        original = self.model._residual_unet_spec
        self.model._residual_unet_spec = None
        try:
            with torch.no_grad():
                return self.model(*args, **kwargs)
        finally:
            self.model._residual_unet_spec = original


@contextmanager
def frozen_native_teacher(model):
    """Restore flags, modes, RNG and topology on success, cancellation or failure.

    The caller owns residency, optimizer, fit inputs and artifact provenance.
    This context does not roll back fitted module weights.
    """
    declaration = validate_hybrid_spec(getattr(model, "_residual_unet_spec", None))
    modules = getattr(model, "rewiring_modules", None)
    if modules is None or modules.declaration != declaration:
        raise ValueError("Teacher initialization requires matching installed hybrid modules")
    if getattr(model, "_block_offloader", None) is not None:
        raise ValueError("Teacher initialization requires resident native blocks")
    if getattr(model, "_hybrid_teacher_owner", None) is not None:
        raise ValueError("Hybrid teacher initialization is already active")
    parameters = tuple(model.parameters())
    if any(parameter.grad is not None for parameter in parameters):
        raise ValueError("Teacher initialization must not discard existing training gradients")
    learned = tuple(modules.parameters())
    learned_ids = {id(parameter) for parameter in learned}
    if not learned or any(id(parameter) in learned_ids and not name.startswith("rewiring_modules.")
                          for name, parameter in model.named_parameters(remove_duplicate=False)):
        raise ValueError("Hybrid parameters must be owned separately from native parameters")
    flags = [(parameter, parameter.requires_grad) for parameter in parameters]
    modes = [(module, module.training) for module in model.modules()]
    topology = model._residual_unet_spec
    devices = sorted({value.device.index for value in (*parameters, *model.buffers()) if value.is_cuda})
    teacher = NativeTeacher(model, declaration, learned)
    with torch.random.fork_rng(devices=devices):
        try:
            model._hybrid_teacher_owner = teacher
            for parameter in parameters:
                parameter.requires_grad_(id(parameter) in learned_ids)
            teacher._active = True
            yield teacher
        finally:
            teacher._active = False
            teacher.model = None
            teacher.parameters = ()
            model.__dict__.pop("_hybrid_teacher_owner", None)
            model._residual_unet_spec = topology
            for parameter, requires_grad in flags:
                parameter.grad = None
                parameter.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
