"""Ming's external VAE domain, calibrated bridges and portable restoration."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from core.models.components.latent_adapter import (
    SpatialAlignmentPair, NormalizedFrameVAE, PairedSpatialCalibration,
    validate_declaration, calibration_grid, calibration_source_side, SpatialSwapPlan, ImageCodecContract)

MingVAEAlignment = SpatialAlignmentPair

NAME = "vae_latent_bridge"
WEIGHTS = "vae_alignment.safetensors"
_noise_contexts = ContextVar("ming_alignment_noise", default=())


def uses_gaussian(model):
    return getattr(getattr(model, NAME, None), "noise_alignment", None) is not None


def forward_with_noise(model, noise, *args, **kwargs):
    if not uses_gaussian(model):
        return model(*args, **kwargs)
    token = _noise_contexts.set(_noise_contexts.get() + ((model, noise),))
    try:
        return model(*args, **kwargs)
    finally:
        _noise_contexts.reset(token)


def validate_noise_contract(model, scheduler):
    if not uses_gaussian(model):
        return
    from diffusers import FlowMatchEulerDiscreteScheduler
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler) or scheduler.config.get("stochastic_sampling", False):
        raise ValueError("Gaussian Ming alignment requires deterministic FlowMatchEuler sampling")


@contextmanager
def sampling_noise_scope(pipe, *, batch, frames):
    bridge = pipe.transformer
    if not uses_gaussian(bridge.transformer):
        yield
        return
    validate_noise_contract(bridge.transformer, pipe.scheduler)
    original = pipe.prepare_latents
    previous = pipe.__dict__.get("prepare_latents")
    def prepare(*args, **kwargs):
        latents = original(*args, **kwargs)
        if latents.ndim != 4 or latents.shape[0] != batch * frames:
            raise ValueError("Ming initial noise must preserve frame-major sampling order")
        bridge.alignment_noise = torch.stack(latents.chunk(frames, dim=0), dim=2).clone()
        return latents
    pipe.prepare_latents = prepare
    try:
        yield
    finally:
        bridge.alignment_noise = None
        if previous is None:
            pipe.__dict__.pop("prepare_latents", None)
        else:
            pipe.prepare_latents = previous


def expanded_noise(noise, count):
    if noise is None or noise.ndim != 5 or noise.shape[0] < 1 or count % noise.shape[0]:
        raise ValueError("Gaussian Ming alignment requires this target exposure's B/C/F/H/W noise")
    return noise.repeat(count // noise.shape[0], 1, 1, 1, 1) if count != noise.shape[0] else noise


def resize_backbone(model, channels):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import MING_IMAGE_01_LATENT_IO
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import rebase_spec
    old = model.in_channels
    if old == channels:
        return
    topology = getattr(model, "_residual_unet_spec", None)
    resize_latent_io(model, MING_IMAGE_01_LATENT_IO, channels)
    model.in_channels = model.out_channels = channels
    if topology is not None:
        restore_topology(model, rebase_spec(topology, native_config(model)))
    from .partition_global import GLOBAL_NAME, from_payload, resize_channels
    adapter = getattr(model, GLOBAL_NAME, None)
    payload = getattr(model, "_ming_partition_global_payload", None)
    if adapter is not None:
        setattr(model, GLOBAL_NAME, resize_channels(adapter, channels))
    elif payload is not None:
        adapter = from_payload(payload["metadata"], payload["state"], latent_channels=old,
            hidden_dim=model.dim, dtype=next(model.parameters()).dtype)
        adapter = resize_channels(adapter, channels)
        model._ming_partition_global_payload = {"metadata": adapter.metadata(), "state": adapter.export_tensors()}


def aligned_forward(model, x, t, cap_feats, **kwargs):
    alignment = getattr(model, NAME)
    if not alignment.ready:
        raise RuntimeError("Ming VAE swap needs paired calibration before training or generation")
    refs = kwargs.pop("ref_x", None)
    noise = next((value for owner, value in reversed(_noise_contexts.get()) if owner is model), None)
    if uses_gaussian(model):
        noise = expanded_noise(noise, len(x))
        if any(tuple(value.shape) != tuple(noise[i].shape) for i, value in enumerate(x)):
            raise ValueError("Ming exposure noise disagrees with the target grid/frame layout")
    converted = [alignment.flow_state(value[None], 1-t[i], noise=None if noise is None else noise[i:i+1]).squeeze(0)
                 for i, value in enumerate(x)]
    if refs is not None:
        refs = [None if value is None else alignment.input(value[None]).squeeze(0) for value in refs]
    result = model._native_forward(converted, t, cap_feats, ref_x=refs, **kwargs)
    native = result.sample if kwargs.get("return_dict", True) else result[0]
    values = []
    for i, value in enumerate(native):
        exposure = None if noise is None else noise[i:i+1]
        if exposure is not None and value.shape[1] > exposure.shape[2]:
            # The native head also emits clean reference frames, excluded from target noise/loss.
            exposure = torch.cat((exposure, exposure.new_zeros(1, exposure.shape[1],
                value.shape[1]-exposure.shape[2], *exposure.shape[3:])), dim=2)
        values.append(alignment.flow_velocity(value[None], convention="x0_minus_eps", noise=exposure).squeeze(0))
    if kwargs.get("return_dict", True):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        return Transformer2DModelOutput(sample=values)
    return (values,)


def restore(model, artifact, dtype):
    declaration = artifact.configs["transformer"].get("vae_alignment")
    if declaration is None:
        return None
    from safetensors.torch import load_file
    alignment = MingVAEAlignment(declaration).to(dtype=dtype)
    alignment.load_state_dict(load_file(str(artifact.root / WEIGHTS), device="cpu"), strict=True)
    if not alignment.ready:
        raise ValueError("Saved Ming VAE alignment is not calibrated")
    model.add_module(NAME, alignment)
    model.register_to_config(vae_alignment=declaration)
    return declaration


def load_saved_vae(directory, declaration, dtype):
    from dataclasses import replace
    from core.models.common.vae_source import resolve_vae_source
    resolved = resolve_vae_source(f"file:{directory}", load_weights=True)
    if resolved.latent_channels != declaration["new_channels"] or resolved.scale_factor != declaration["new_scale"]:
        raise ValueError("Saved target VAE disagrees with alignment geometry")
    if (resolved.norm, resolved.norm_pack, resolved.ndim) != (declaration["norm"], declaration["norm_pack"], declaration["ndim"]):
        raise ValueError("Saved target VAE disagrees with alignment normalization/frame layout")
    vae = NormalizedFrameVAE(resolved.load_module(torch_dtype=dtype), declaration)
    vae._sushi_vae_identity = replace(resolved, state_dict=None, identity_native=False,
        struct_native=False, provenance=declaration["source"])
    return vae


def configure_swap(trainer):
    from core.training.vae_swap import resolve_vae_swap_source, check_swap_method, initialization_mode
    from core.training.ops.training_method import resolve_training_method
    from core.models.common.vae_source import resolve_vae_source
    source = resolve_vae_swap_source(trainer.config)
    if not source:
        return
    check_swap_method(source, resolve_training_method(trainer))
    model = trainer.transformer
    existing = getattr(model, NAME, None)
    if existing is not None:
        if source == existing.declaration["source"]:
            return
        raise ValueError("Changing an already aligned Ming VAE requires a fresh base checkpoint")
    method = initialization_mode(trainer.config, "ming_image_01")
    if method != "paired_spatial":
        raise ValueError("Ming VAE swaps require paired_spatial initialization")
    from core.training.arch.ming_image_01 import MingImage01ArchHandler
    # Component loading precedes BaseTrainer's final handler binding.
    handler = getattr(trainer, "arch", None) or MingImage01ArchHandler(trainer)
    resolved = resolve_vae_source(source)
    compatible, reason = handler.check_vae_compatibility(resolved.facts(), trainer=trainer)
    if not compatible:
        raise ValueError(reason)
    old = trainer.vae
    target = resolved.load_module(torch_dtype=trainer.vae_dtype).requires_grad_(False).eval()
    from .cache import module_fingerprint
    from core.models.common.vae_source import normalization_config
    source_identity = module_fingerprint(old)
    if (source_identity == module_fingerprint(target)
            and normalization_config(old.config) == normalization_config(target.config)):
        print("[VAE alignment] Byte-identical VAE: no initialization or calibration", flush=True)
        return
    declaration = ImageCodecContract.from_resolved(resolved).declaration(
        source=source, source_identity=source_identity, target_identity=resolved.latent_hash,
        old_channels=model.in_channels, old_scale=handler.wiring.vae_scale_factor)
    if resolved.scale_factor > declaration["old_scale"] and resolved.latent_channels >= model.in_channels:
        from core.models.components.gaussian_alignment import POLICY
        declaration["noise_policy"] = POLICY
    plan = SpatialSwapPlan.build(declaration, patch_size=2)
    from core.rewiring.training import prepare_topology_io_rebase
    from .residual_unet import native_config
    finish_topology = prepare_topology_io_rebase(trainer.config,"ming_image_01",native_config(model),declaration=declaration)
    alignment = MingVAEAlignment(declaration).to(dtype=trainer.weight_dtype)
    resize_backbone(model, plan.backbone_channels)
    if finish_topology is not None:
        finish_topology(native_config(model))
    model.add_module(NAME, alignment)
    model.register_to_config(vae_alignment=declaration)
    trainer._ming_vae_calibration_source = old.to("cpu").requires_grad_(False).eval()
    trainer.vae = trainer.ming_image_01_components.vae = NormalizedFrameVAE(target, declaration)
    trainer.wiring = handler.wiring.replace(latent_channels=resolved.latent_channels,
        vae_scale_factor=resolved.scale_factor, vae_norm="shift_scale")
    handler.pixel_align = plan.pixel_alignment


def _calibration_paths(datasets):
    def candidates():
        for dataset in datasets:
            for item in dataset.items:
                record = item.get("_ming_record")
                if record is not None and record.profile == "layer_decompose":
                    for asset, weight in zip(record.targets, record.weights):
                        if weight > 0:
                            yield str(asset.path)
                elif item.get("image_path"):
                    yield item["image_path"]
        for dataset in datasets:
            for item in dataset.items:
                record = item.get("_ming_record")
                if record is not None and record.profile == "layer_decompose":
                    yield from (str(asset.path) for asset in record.references)
    return list(dict.fromkeys(candidates()))


def calibrate(trainer, datasets):
    old = getattr(trainer, "_ming_vae_calibration_source", None)
    if old is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS, VAE_SWAP_CALIBRATION_RESOLUTIONS
    from PIL import Image, ImageOps
    import numpy as np
    from .training_data import encode_frames
    from .block_offload import move_component, execution_device
    from core.training.optimizer_residency import park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in VAE_SWAP_CALIBRATION_RESOLUTIONS:
        raise ValueError("Invalid VAE spatial calibration count/resolution")
    # Layer primaries can be opaque composites; calibration must include supervised RGBA cutouts.
    paths = _calibration_paths(datasets)
    if not paths:
        raise ValueError("Ming VAE calibration requires training images")
    indices = torch.linspace(0, len(paths)-1, min(count, len(paths))).long().tolist()
    paths = [paths[i] for i in indices]
    callback = getattr(trainer, "_vae_alignment_progress_callback", None)
    def progress(stage, step):
        if callback is not None:
            callback(phase="latent_cache", step=step + (len(paths) if stage == "target" else 0), total=2*len(paths))
    components, device = trainer.ming_image_01_components, trainer.device
    model = components.transformer
    alignment = getattr(model, NAME)
    source_side = calibration_source_side(side, alignment.declaration)
    fit = PairedSpatialCalibration(alignment, ridge=1e-2)
    previous = execution_device(model)
    conditioner_device = execution_device(components.conditioner)
    parameters = list(model.parameters()) + list(components.conditioner.parameters())
    with park_optimizer_state(all_optimizers(trainer), parameters):
        move_component(model, "cpu")
        components.conditioner.to("cpu")
        try:
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
            # Retain only the small paired latent subset, with one VAE resident at a time.
            paired = []
            old.to(device)
            with torch.no_grad():
                for i, path in enumerate(paths):
                    trainer._check_stop_requested()
                    with Image.open(path) as image:
                        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (source_side, source_side))
                        pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()/127.5-1
                    paired.append(encode_frames(old, pixels[None, :, None]).cpu())
                    print(f"[VAE alignment] Source calibration {i+1}/{len(paths)}", flush=True)
                    progress("source", i+1)
            old.to("cpu")
            if torch.device(device).type == "cuda": torch.cuda.empty_cache()
            trainer.vae.to(device)
            with torch.no_grad():
                for i, (path, legacy) in enumerate(zip(paths, paired)):
                    trainer._check_stop_requested()
                    with Image.open(path) as image:
                        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (side, side))
                        pixels = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()/127.5-1
                    target = encode_frames(trainer.vae, pixels[None, :, None]).cpu()
                    fit.add(legacy, target)
                    print(f"[VAE alignment] Target calibration {i+1}/{len(paths)}", flush=True)
                    progress("target", i+1)
            fit.finish()
            del trainer._ming_vae_calibration_source
        finally:
            old.to("cpu")
            trainer.vae.to("cpu")
            if torch.device(device).type == "cuda": torch.cuda.empty_cache()
            move_component(model, previous)
            components.conditioner.to(conditioner_device)
