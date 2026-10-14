"""Shared image swap artifacts and bounded paired calibration."""
import json
from dataclasses import replace

import torch

from .latent_adapter import (
    SpatialAlignmentPair, NormalizedImageVAE, PairedSpatialCalibration,
    ImageCodecContract, SpatialSwapPlan, validate_declaration,
)
from .vae_registry import normalize

NAME = "vae_latent_bridge"
META_KEY = "component.vae.alignment"


def image_token_stride(vae, architecture, patch_size):
    from core.models.component_registry import _WIRING_BY_ARCH
    scale = getattr(vae, "spatial_scale_factor", _WIRING_BY_ARCH[architecture].vae_scale_factor)
    if type(scale) is not int or scale < 1 or type(patch_size) is not int or patch_size < 1:
        raise ValueError("Image token stride requires positive codec and patch scales")
    return scale * patch_size


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


def initialize_image_swap(trainer, resolved, *, handler, model, old_channels,
                          source_wiring, patch_size, resize, pending_key, topology_config=None,
                          noise_policy=None, passthrough=None, noise_estimate=None):
    """Install target-native bridges before optimizer construction; geometry stays arch-owned."""
    from core.models.common.module_identity import module_fingerprint
    from core.models.common.vae_source import normalization_config
    from core.models.components.latent_io import ResizeReport
    codec = ImageCodecContract.from_resolved(resolved)
    old = trainer.vae
    target = resolved.load_module(torch_dtype=trainer.vae_dtype).requires_grad_(False).eval()
    identity = module_fingerprint(old)
    if identity == module_fingerprint(target) and normalization_config(old.config) == normalization_config(target.config):
        return ResizeReport((), old_channels, old_channels, old_channels, 0, 0)
    declaration = codec.declaration(source=resolved.source, source_identity=identity,
        target_identity=resolved.latent_hash, old_channels=old_channels, old_scale=source_wiring.vae_scale_factor,
        passthrough=passthrough)
    if noise_policy is not None:
        declaration["noise_policy"] = noise_policy
        if noise_estimate is not None:
            declaration["noise_estimate"] = noise_estimate
        validate_declaration(declaration)
    plan = SpatialSwapPlan.build(declaration, patch_size=patch_size)
    finish_topology = None
    if trainer.config.get("dit_rewiring_spec") is not None:
        if topology_config is None:
            raise ValueError("Explicit VAE-swap topology requires an architecture native-config provider")
        from core.rewiring.training import prepare_topology_io_rebase
        finish_topology = prepare_topology_io_rebase(
            trainer.config, handler.name, topology_config(), declaration=declaration)
    report = resize(plan.backbone_channels)
    if finish_topology is not None:
        finish_topology(topology_config())
    install(model, declaration)
    getattr(model, NAME).to(dtype=trainer.weight_dtype)
    setattr(trainer, pending_key, (old.to("cpu").requires_grad_(False).eval(), source_wiring))
    trainer.vae = NormalizedImageVAE(target, declaration)
    trainer.vae_identity = replace(resolved, state_dict=None)
    trainer.vae_latent_channels = codec.latent_channels
    trainer.wiring = handler.resolve_wiring(trainer)
    return report


def map_packed_image(pair, states, grid, *, sigma=None, output=False, patch_size=2,
                     channel_order="outer", convention="eps_minus_x0", noise=None):
    """The spatial bridge changes neither grid size nor packing order."""
    height, width = grid
    batch, tokens, packed = states.shape
    p = patch_size
    if height < 1 or width < 1 or tokens != height*width or packed % (p*p):
        raise ValueError("Packed image alignment requires a rectangular divisible grid")
    if channel_order not in ("outer", "inner"):
        raise ValueError("Packed image alignment requires an explicit channel order")
    if channel_order == "inner":
        states = states.reshape(batch,tokens,p,p,-1).permute(0,1,4,2,3).reshape(batch,tokens,packed)
    image = states.reshape(batch,height,width,packed//(p*p),p,p).permute(0,3,1,4,2,5)
    image = image.reshape(batch,packed//(p*p),height*p,width*p)
    if noise is not None:
        if noise.ndim != 3 or noise.shape[:2] != (batch,tokens) or noise.shape[-1] != pair.declaration["new_channels"]*p*p:
            raise ValueError("Packed alignment noise must share the external image grid")
        if channel_order == "inner":
            noise = noise.reshape(batch,tokens,p,p,-1).permute(0,1,4,2,3).reshape(batch,tokens,-1)
        noise = noise.reshape(batch,height,width,-1,p,p).permute(0,3,1,4,2,5)
        noise = noise.reshape(batch,pair.declaration["new_channels"],height*p,width*p)
    image = (pair.flow_velocity(image, convention=convention, noise=noise) if output
             else pair.flow_state(image, sigma, noise=noise))
    states = image.reshape(batch,image.shape[1],height,p,width,p).permute(0,2,4,1,3,5)
    if channel_order == "inner":
        states = states.permute(0,1,2,4,5,3)
    return states.reshape(batch,tokens,-1)


def calibrate_images(trainer, datasets, *, model, pending_key,
                     target_observer=None, on_complete=None):
    pending = getattr(trainer, pending_key, None)
    if pending is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS, VAE_SWAP_CALIBRATION_RESOLUTIONS
    from PIL import Image, ImageOps
    import numpy as np
    from core.training.optimizer_residency import encoder_refill_residency, park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in VAE_SWAP_CALIBRATION_RESOLUTIONS:
        raise ValueError("Invalid VAE spatial calibration count/resolution")
    paths = list(dict.fromkeys(item["image_path"] for dataset in datasets for item in dataset.items
                               if item.get("image_path")))
    if not paths:
        raise ValueError("Spatial VAE calibration requires training images")
    paths = [paths[i] for i in torch.linspace(0, len(paths)-1, min(count, len(paths))).long().tolist()]
    old, source_spec = pending
    pair = getattr(model, NAME)
    from .latent_adapter import calibration_source_side
    source_side = calibration_source_side(side, pair.declaration)
    fit = PairedSpatialCalibration(pair, ridge=1e-2)
    if getattr(trainer,"is_lens",False):
        # Lens's mxfp4 allocations are released only by discarding the encoder.
        trainer.move_text_encoder_to_cpu()
    text_modules = [module for module in (getattr(trainer, "text_encoder", None),
                    getattr(trainer, "text_encoder_2", None)) if module is not None]
    text_devices = [next(module.parameters()).device for module in text_modules]
    text_parameters = [parameter for module in text_modules for parameter in module.parameters()]
    callback = getattr(trainer, "_vae_alignment_progress_callback", None)
    # A private stream samples target statistics without altering training RNG.
    statistic_generator = (torch.Generator(device=trainer.device).manual_seed(0)
                           if target_observer is not None else None)

    def encode(module, path, source=False):
        with Image.open(path) as image:
            image_side = source_side if source else side
            pixels = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (image_side, image_side))
            pixels = torch.from_numpy(np.array(pixels).copy()).permute(2, 0, 1).float()/127.5-1
        pixels = pixels[None].to(device=trainer.device, dtype=trainer.vae_dtype)
        if source and source_spec.latent_ndim == 5:
            pixels = pixels.unsqueeze(2)
        posterior = module.encode(pixels).latent_dist
        z = posterior.mode()
        if not source and target_observer is not None:
            target_observer(posterior.sample(generator=statistic_generator).cpu())
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
            prior = getattr(pair, "noise_prior", None)
            targets = []
            with torch.no_grad():
                for i, (path, legacy) in enumerate(zip(paths, paired)):
                    trainer._check_stop_requested()
                    target = encode(trainer.vae, path)
                    fit.add(legacy, target)
                    if prior is not None:
                        targets.append(target)
                    if callback:
                        callback(phase="latent_cache", step=len(paths)+i+1, total=2*len(paths))
            fit.finish()
            if prior is not None:
                prior.capture(targets)
            if on_complete is not None:
                on_complete()
            delattr(trainer, pending_key)
        finally:
            old.to("cpu")
            trainer.vae.to("cpu")
            for module, device in zip(text_modules, text_devices):
                module.to(device)
