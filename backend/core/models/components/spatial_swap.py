"""Shared image swap artifacts and bounded paired calibration."""
import json
from dataclasses import replace

import torch

from .latent_adapter import (
    SpatialAlignmentPair, NormalizedImageVAE, PairedSpatialCalibration,
    validate_declaration,
)
from .vae_registry import normalize

NAME = "vae_latent_bridge"
META_KEY = "component.vae.alignment"


def encode_image(module, pixels, *, generator=None):
    """Preserve legacy unclamped sampling; normalized facades own their posterior."""
    if isinstance(module, NormalizedImageVAE):
        return module.encode(pixels).latent_dist.sample(generator=generator)
    from .vae_registry import normalize
    value = module.encoder(pixels)
    if module.quant_conv is not None:
        value = module.quant_conv(value)
    mean, logvar = value.chunk(2, dim=1)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return normalize(mean + (logvar * .5).exp() * noise, module)


def read_declaration(metadata):
    raw = (metadata or {}).get(META_KEY)
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
        raise ValueError("Invalid spatial VAE alignment metadata")
    return dict(validate_declaration(json.loads(raw)))


def artifact_metadata(model):
    pair = getattr(model, NAME, None)
    if pair is None:
        return {}
    if not pair.ready:
        raise ValueError("Cannot save an uncalibrated spatial VAE swap")
    return {META_KEY: json.dumps(pair.declaration, sort_keys=True, allow_nan=False)}


def install(model, declaration):
    if declaration is not None:
        model.add_module(NAME, SpatialAlignmentPair(declaration))


def validate_loaded(model):
    pair = getattr(model, NAME, None)
    if pair is not None and not pair.ready:
        raise ValueError("Saved spatial VAE alignment is not calibrated")


def restored_image_codec(module, resolved, declaration):
    validate_declaration(declaration)
    if (resolved.latent_channels, resolved.scale_factor, resolved.norm, resolved.norm_pack, resolved.ndim) != (
            declaration["new_channels"], declaration["new_scale"], declaration["norm"],
            declaration["norm_pack"], declaration["ndim"]):
        raise ValueError("Saved image codec disagrees with its spatial alignment")
    return NormalizedImageVAE(module, declaration), replace(resolved, state_dict=None)


def calibrate_images(trainer, datasets, *, model, pending_key):
    pending = getattr(trainer, pending_key, None)
    if pending is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from PIL import Image, ImageOps
    import numpy as np
    from core.training.optimizer_residency import encoder_refill_residency, park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in (128, 256, 512):
        raise ValueError("Invalid VAE spatial calibration count/resolution")
    paths = list(dict.fromkeys(item["image_path"] for dataset in datasets for item in dataset.items
                               if item.get("image_path")))
    if not paths:
        raise ValueError("Spatial VAE calibration requires training images")
    paths = [paths[i] for i in torch.linspace(0, len(paths)-1, min(count, len(paths))).long().tolist()]
    old, source_spec = pending
    pair = getattr(model, NAME)
    fit = PairedSpatialCalibration(pair, ridge=1e-2)
    text_modules = [module for module in (getattr(trainer, "text_encoder", None),
                    getattr(trainer, "text_encoder_2", None)) if module is not None]
    text_devices = [next(module.parameters()).device for module in text_modules]
    text_parameters = [parameter for module in text_modules for parameter in module.parameters()]
    callback = getattr(trainer, "_vae_alignment_progress_callback", None)

    def encode(module, path, source=False):
        with Image.open(path) as image:
            pixels = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (side, side))
            pixels = torch.from_numpy(np.array(pixels).copy()).permute(2, 0, 1).float()/127.5-1
        pixels = pixels[None].to(device=trainer.device, dtype=trainer.vae_dtype)
        if source and source_spec.latent_ndim == 5:
            pixels = pixels.unsqueeze(2)
        z = module.encode(pixels).latent_dist.mode()
        if source:
            z = normalize(z, module, source_spec)
            if z.ndim == 5:
                if z.shape[2] != 1:
                    raise ValueError("Image calibration codec changed the independent frame count")
                z = z.squeeze(2)
        return z.cpu()

    with encoder_refill_residency(trainer, text=False, vae=True), park_optimizer_state(
            all_optimizers(trainer), text_parameters):
        for module in text_modules:
            module.to("cpu")
        try:
            old.to(trainer.device)
            paired = []
            with torch.no_grad():
                for i, path in enumerate(paths):
                    trainer._check_stop_requested()
                    paired.append(encode(old, path, source=True))
                    if callback:
                        callback(phase="latent_cache", step=i+1, total=2*len(paths))
            old.to("cpu")
            if torch.device(trainer.device).type == "cuda":
                torch.cuda.empty_cache()
            trainer.vae.to(trainer.device)
            with torch.no_grad():
                for i, (path, legacy) in enumerate(zip(paths, paired)):
                    trainer._check_stop_requested()
                    fit.add(legacy, encode(trainer.vae, path))
                    if callback:
                        callback(phase="latent_cache", step=len(paths)+i+1, total=2*len(paths))
            fit.finish()
            delattr(trainer, pending_key)
        finally:
            old.to("cpu")
            trainer.vae.to("cpu")
            for module, device in zip(text_modules, text_devices):
                module.to(device)
