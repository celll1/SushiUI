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
    from core.training.debug_decode_residency import debug_decode_residency, DebugDecodeRestorationError
    from .anima_pipeline_ops import vae_decode_latents
    outputs = [(pending["path"] / f"decode_t{pending['timestep']:.4f}_{name}.webp", latent)
               for name, latent in pending["previews"]]
    published = False
    try:
        with debug_decode_residency(trainer, trainer.vae, dtype=trainer.vae_dtype), torch.inference_mode():
            for path, latent in outputs:
                image = vae_decode_latents(trainer.vae,
                    latent.to(device=trainer.device, dtype=trainer.vae_dtype))[0]
                image.save(path.with_suffix(".webp.tmp"), "WEBP", quality=80, method=4)
                del image
        for path, _ in outputs:
            path.with_suffix(".webp.tmp").replace(path)
        published = True
    except DebugDecodeRestorationError:
        raise
    except Exception as error:
        print(f"{getattr(trainer, 'log_prefix', '[Training]')} [debug_latents] "
              f"Decode failed; using latent previews: {type(error).__name__}: {error}")
    finally:
        for path, _ in outputs:
            path.with_suffix(".webp.tmp").unlink(missing_ok=True)
            if not published:
                path.unlink(missing_ok=True)
