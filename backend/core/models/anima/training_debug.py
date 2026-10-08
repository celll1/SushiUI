"""Deferred image previews for native and swapped Anima VAEs."""
from pathlib import Path

import torch

from api.param_defaults import TRAINING_DEFAULTS


def queue_debug_previews(trainer, path, data):
    if getattr(trainer, "config", {}).get(
            "qwen_debug_latent_view", TRAINING_DEFAULTS["qwen_debug_latent_view"]) != "pixel":
        return
    trainer._pending_anima_debug_previews = {
        "path": Path(path), "timestep": float(data["timestep"]),
        "previews": [(name, data[key][:1].detach().float().cpu()) for name, key in (
            ("target", "latents"), ("noisy", "noisy_latents"), ("pred_x0", "predicted_latent"))],
    }


def flush_pending_debug_previews(trainer):
    pending = getattr(trainer, "_pending_anima_debug_previews", None)
    if pending is None:
        return
    del trainer._pending_anima_debug_previews
    from core.training.debug_decode_residency import debug_decode_residency
    from .anima_pipeline_ops import vae_decode_latents
    with debug_decode_residency(trainer, trainer.vae, dtype=trainer.vae_dtype), torch.inference_mode():
        for name, latent in pending["previews"]:
            image = vae_decode_latents(trainer.vae,
                latent.to(device=trainer.device, dtype=trainer.vae_dtype))[0]
            image.save(pending["path"] / f"decode_t{pending['timestep']:.4f}_{name}.webp",
                       "WEBP", quality=80, method=4)
            del image
