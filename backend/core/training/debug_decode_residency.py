"""Exclusive VAE residency for deferred, non-training debug decode."""

from contextlib import contextmanager

import torch


class DebugDecodeRestorationError(RuntimeError):
    pass


@contextmanager
def debug_decode_residency(trainer, vae, *, dtype=None):
    from .base_trainer import all_optimizers
    from .optimizer_residency import park_optimizer_state

    parameter = next(vae.parameters())
    original_device, original_dtype = parameter.device, parameter.dtype
    was_training = vae.training
    device = torch.device(getattr(trainer, "device", original_device))
    moved, seen, restoration_errors = [], set(), []
    optimizer_scope = None
    vae_ids = {id(value) for value in (*vae.parameters(), *vae.buffers())}

    def park(value):
        if id(value) in seen or id(value) in vae_ids:
            return
        seen.add(id(value))
        if value.device.type == "cuda":
            location = value.device
            value.data = value.data.to("cpu")
            moved.append((value, location))
            gradient = value.grad if isinstance(value, torch.nn.Parameter) else None
            if gradient is not None:
                value.grad = gradient.to("cpu")

    def park_tree(value):
        if isinstance(value, torch.Tensor):
            park(value)
        elif isinstance(value, dict):
            for item in value.values():
                park_tree(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                park_tree(item)

    optimizers = all_optimizers(trainer)
    parameters = [p for optimizer in optimizers for group in optimizer.param_groups
                  for p in group["params"]]
    try:
        if device.type == "cuda":
            conductors = getattr(trainer, "_layer_offload_conductors", lambda: ())()
            for conductor in conductors:
                conductor.flush()
                # Idle slots have no live layer views; CPU master bindings stay intact.
                park_tree(conductor.engine.slots)
            main = getattr(trainer, "_main_model_module", lambda: None)()
            modules = [main] + [getattr(trainer, name, None) for name in (
                "transformer", "transformer_original", "unet", "text_encoder",
                "text_encoder_2", "vision_encoder", "image_encoder", "network",
                "lora_network", "repa_encoder", "repa_projector", "repa_latent_stem")]
            for module in modules:
                if isinstance(module, torch.nn.Module):
                    for value in (*module.parameters(), *module.buffers()):
                        park(value)
            for value in parameters:
                park(value)
            park_tree(getattr(trainer, "ema_shadow", None))
        scope = park_optimizer_state(optimizers, parameters)
        scope.__enter__()
        optimizer_scope = scope
        if device.type == "cuda":
            torch.cuda.empty_cache()
        vae.to(device=device, dtype=dtype or original_dtype).eval()
        if device.type == "cuda":
            print(f"{getattr(trainer, 'log_prefix', '[Trainer]')} [debug_latents] "
                  f"VAE-only decode: allocated={torch.cuda.memory_allocated(device)/2**30:.2f} GiB",
                  flush=True)
        yield
    finally:
        try:
            vae.to(device="cpu", dtype=original_dtype).train(was_training)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except BaseException as exc:
            restoration_errors.append(exc)
        if optimizer_scope is not None:
            try:
                optimizer_scope.__exit__(None, None, None)
            except BaseException as exc:
                restoration_errors.append(exc)
        with torch.no_grad():
            for value, location in reversed(moved):
                try:
                    value.data = value.data.to(location)
                    if isinstance(value, torch.nn.Parameter) and value.grad is not None:
                        value.grad = value.grad.to(location)
                except BaseException as exc:
                    restoration_errors.append(exc)
        try:
            vae.to(device=original_device, dtype=original_dtype).train(was_training)
        except BaseException as exc:
            restoration_errors.append(exc)
        if restoration_errors:
            raise DebugDecodeRestorationError(
                "Failed to restore training residency after debug decode; training must stop"
            ) from restoration_errors[0]
