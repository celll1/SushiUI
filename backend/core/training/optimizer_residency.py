"""Temporary optimizer-state residency without changing optimizer semantics."""

from contextlib import contextmanager

import torch


@contextmanager
def park_optimizer_state(optimizers, parameters):
    """Park CUDA state for selected parameters, then restore its original devices."""
    owned = {id(parameter) for parameter in parameters}
    moved = []

    def park(container):
        for key, value in list(container.items() if isinstance(container, dict) else enumerate(container)):
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                device = value.device
                container[key] = value.to("cpu")
                moved.append((container, key, device))
            elif isinstance(value, (dict, list)):
                park(value)

    try:
        for optimizer in optimizers:
            for parameter, state in optimizer.state.items():
                if id(parameter) in owned:
                    park(state)
        yield
    finally:
        for container, key, device in reversed(moved):
            container[key] = container[key].to(device)


@contextmanager
def encoder_refill_residency(trainer, *, text, vae, restore_main=True):
    """Keep main-model weights and state off GPU throughout a shared refill."""
    if not (text or vae):
        yield
        return
    from .base_trainer import all_optimizers

    model = trainer._main_model_module()
    parameters = tuple(model.parameters()) if model is not None else ()
    trainer.move_main_model_to_cpu()
    with park_optimizer_state(all_optimizers(trainer), parameters):
        try:
            torch.cuda.empty_cache()
            yield
        finally:
            if text:
                trainer.move_text_encoder_to_cpu()
            if vae:
                trainer.move_vae_to_cpu()
            torch.cuda.empty_cache()
            if restore_main:
                trainer.move_main_model_to_gpu()
