"""Ming's external VAE domain, calibrated bridges and portable restoration."""

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from core.models.components.vae_alignment import SpatialLatentBridge, SpatialBridgeCalibrator
from core.models.components.vae_registry import normalize, denormalize

NAME = "vae_latent_bridge"
WEIGHTS = "vae_alignment.safetensors"


def validate_declaration(value):
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("method") != "paired_spatial":
        raise ValueError("Unsupported Ming VAE alignment declaration")
    if value.get("token_policy") != "target_native":
        raise ValueError("Ming VAE alignment must use the target VAE token grid")
    for key in ("old_channels", "new_channels", "old_scale", "new_scale", "input_channels"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError(f"Invalid VAE alignment {key}")
    if value["input_channels"] not in (3, 4):
        raise ValueError("Ming aligned VAE requires RGB or RGBA pixels")
    if value["old_scale"] % value["new_scale"] and value["new_scale"] % value["old_scale"]:
        raise ValueError("VAE alignment needs integer spatial packing")
    if not isinstance(value.get("source"), str) or not value["source"]:
        raise ValueError("VAE alignment source identity is missing")
    if value.get("norm") not in ("shift_scale", "per_channel", "batchnorm"):
        raise ValueError("VAE alignment normalization is missing")
    if value.get("ndim") not in (4, 5):
        raise ValueError("Ming VAE alignment needs image-capable latents")
    if type(value.get("norm_pack")) is not int or value["norm_pack"] < 1:
        raise ValueError("Ming VAE alignment normalization packing is invalid")
    return value


class MingVAEAlignment(nn.Module):
    def __init__(self, declaration):
        super().__init__()
        self.declaration = dict(validate_declaration(declaration))
        old, new = declaration["old_channels"], declaration["new_channels"]
        width = max(old, new)
        self.input = SpatialLatentBridge(new, width)
        self.output = SpatialLatentBridge(width, new)

    @property
    def ready(self):
        return self.input._ready and self.output._ready

    def initialize_from_pair(self, incoming, outgoing):
        old, new = self.declaration["old_channels"], self.declaration["new_channels"]
        width = max(old, new)
        iw = incoming.projection.weight.detach().new_zeros(width, new, 3, 3)
        ow = outgoing.projection.weight.detach().new_zeros(new, width, 3, 3)
        ib = incoming.projection.bias.detach().new_zeros(width)
        iw[:old], ib[:old] = incoming.projection.weight.detach(), incoming.projection.bias.detach()
        ow[:, :old] = outgoing.projection.weight.detach()
        # Extra channels bypass the calibrated legacy subspace and remain learnable.
        for c in range(old, width):
            iw[c, c, 1, 1] = ow[c, c, 1, 1] = 1
        self.input.initialize(iw.flatten(1).T, ib)
        self.output.initialize(ow.flatten(1).T, outgoing.projection.bias.detach())


def resize_backbone(model, channels):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import ZIMAGE_LATENT_IO
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec
    old = model.in_channels
    if old == channels:
        return
    topology = getattr(model, "_residual_unet_spec", None)
    resize_latent_io(model, ZIMAGE_LATENT_IO, channels)
    model.in_channels = model.out_channels = channels
    if topology is not None:
        restore_topology(model, build_spec("ming_image_01",
            [{"name": "main", "block_count": len(model.layers)}], topology["regions"], native_config(model)))
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


class _Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
    __setattr__ = dict.__setitem__


class NormalizedFrameVAE(nn.Module):
    """Expose normalized independent RGBA frames to Ming's VAE callers."""
    def __init__(self, module, declaration):
        super().__init__()
        self.inner = module
        self.declaration = dict(validate_declaration(declaration))
        self.norm_spec = SimpleNamespace(vae_norm=declaration["norm"], vae_norm_pack=declaration["norm_pack"])
        self.spatial_scale_factor = declaration["new_scale"]
        self.input_channels = 4
        self.config = _Config(dict(module.config), input_channels=4,
            z_dim=declaration["new_channels"], latent_channels=declaration["new_channels"],
            scaling_factor=1.0, shift_factor=0.0, temperal_downsample=[])

    @property
    def device(self):
        return next(self.inner.parameters()).device

    @property
    def dtype(self):
        return next(self.inner.parameters()).dtype

    def _encode_one(self, pixels):
        pixels = pixels[:, :self.declaration["input_channels"]]
        if self.declaration["ndim"] == 5:
            pixels = pixels.unsqueeze(2)
        raw = self.inner.encode(pixels).latent_dist.mode()
        if raw.ndim == 5:
            if raw.shape[2] != 1:
                raise ValueError("VAE must preserve each independent Ming image frame")
            raw = raw.squeeze(2)
        return normalize(raw, self.inner, self.norm_spec)

    def encode(self, pixels, **kwargs):
        if pixels.ndim != 5 or pixels.shape[1] != 4:
            raise ValueError("Aligned Ming VAE requires independent RGBA frames")
        z = torch.stack([self._encode_one(frame) for frame in pixels.unbind(2)], dim=2)
        # Native Ming uses the deterministic posterior mode for all image frames.
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))

    def decode(self, latents, return_dict=True, **kwargs):
        if latents.ndim != 5:
            raise ValueError("Aligned Ming decode requires independent frames")
        decoded = []
        for frame in latents.unbind(2):
            raw = denormalize(frame, self.inner, self.norm_spec)
            if self.declaration["ndim"] == 5:
                raw = raw.unsqueeze(2)
            result = self.inner.decode(raw, return_dict=False)[0]
            if result.ndim == 5:
                if result.shape[2] != 1:
                    raise ValueError("Aligned VAE decode changed the independent frame count")
                result = result.squeeze(2)
            if result.shape[1] == 3:
                result = torch.cat((result, torch.ones_like(result[:, :1])), dim=1)
            if result.shape[1] != 4:
                raise ValueError("Aligned Ming decoder must return RGB or RGBA")
            decoded.append(result)
        value = torch.stack(decoded, dim=2)
        return SimpleNamespace(sample=value) if return_dict else (value,)

    def save_pretrained(self, path, **kwargs):
        return self.inner.save_pretrained(path, **kwargs)


def aligned_forward(model, x, t, cap_feats, **kwargs):
    alignment = getattr(model, NAME)
    if not alignment.ready:
        raise RuntimeError("Ming VAE swap needs paired calibration before training or generation")
    refs = kwargs.pop("ref_x", None)
    converted = [alignment.input.flow_state(value[None], 1-t[i]).squeeze(0) for i, value in enumerate(x)]
    if refs is not None:
        refs = [None if value is None else alignment.input(value[None]).squeeze(0) for value in refs]
    result = model._native_forward(converted, t, cap_feats, ref_x=refs, **kwargs)
    native = result.sample if kwargs.get("return_dict", True) else result[0]
    values = [alignment.output.flow_velocity(value[None], convention="x0_minus_eps").squeeze(0) for value in native]
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
    from core.models.common.vae_source import resolve_vae_source
    resolved = resolve_vae_source(f"file:{directory}", load_weights=True)
    if resolved.latent_channels != declaration["new_channels"] or resolved.scale_factor != declaration["new_scale"]:
        raise ValueError("Saved target VAE disagrees with alignment geometry")
    if (resolved.norm, resolved.norm_pack, resolved.ndim) != (declaration["norm"], declaration["norm_pack"], declaration["ndim"]):
        raise ValueError("Saved target VAE disagrees with alignment normalization/frame layout")
    return NormalizedFrameVAE(resolved.load_module(torch_dtype=dtype), declaration)


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
    resolved = resolve_vae_source(source)
    compatible, reason = trainer.arch.check_vae_compatibility(resolved.facts(), trainer=trainer)
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
    channels = int(target.config.get("in_channels", target.config.get("input_channels", 3)))
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native", source=source,
        source_identity=source_identity, target_identity=resolved.latent_hash, old_channels=model.in_channels, old_scale=8,
        new_channels=resolved.latent_channels, new_scale=resolved.scale_factor,
        input_channels=channels, ndim=resolved.ndim, norm=resolved.norm, norm_pack=resolved.norm_pack)
    alignment = MingVAEAlignment(declaration).to(dtype=trainer.weight_dtype)
    resize_backbone(model, max(declaration["old_channels"], declaration["new_channels"]))
    model.add_module(NAME, alignment)
    model.register_to_config(vae_alignment=declaration)
    trainer._ming_vae_calibration_source = old.to("cpu").requires_grad_(False).eval()
    trainer.vae = trainer.ming_image_01_components.vae = NormalizedFrameVAE(target, declaration)
    trainer.wiring = trainer.arch.wiring.replace(latent_channels=resolved.latent_channels,
        vae_scale_factor=resolved.scale_factor, vae_norm="shift_scale")
    trainer.arch.pixel_align = resolved.scale_factor * 2


def calibrate(trainer, datasets):
    old = getattr(trainer, "_ming_vae_calibration_source", None)
    if old is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from PIL import Image, ImageOps
    import numpy as np
    from .training_data import encode_frames
    from .block_offload import move_component, execution_device
    from core.training.optimizer_residency import park_optimizer_state
    from core.training.base_trainer import all_optimizers
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in (128, 256, 512):
        raise ValueError("Invalid VAE spatial calibration count/resolution")
    paths = list(dict.fromkeys(item["image_path"] for dataset in datasets for item in dataset.items
                               if item.get("image_path")))
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
    incoming = SpatialLatentBridge(alignment.declaration["new_channels"], alignment.declaration["old_channels"])
    outgoing = SpatialLatentBridge(alignment.declaration["old_channels"], alignment.declaration["new_channels"])
    input_fit = SpatialBridgeCalibrator(incoming, ridge=1e-2)
    output_fit = SpatialBridgeCalibrator(outgoing, ridge=1e-2)
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
                        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (side, side))
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
                    legacy = calibration_grid(legacy, target.shape[-2:])
                    input_fit.add(target, legacy)
                    output_fit.add(legacy, target)
                    print(f"[VAE alignment] Target calibration {i+1}/{len(paths)}", flush=True)
                    progress("target", i+1)
            input_fit.finish()
            output_fit.finish()
            alignment.initialize_from_pair(incoming, outgoing)
            del trainer._ming_vae_calibration_source
        finally:
            old.to("cpu")
            trainer.vae.to("cpu")
            if torch.device(device).type == "cuda": torch.cuda.empty_cache()
            move_component(model, previous)
            components.conditioner.to(conditioner_device)


def calibration_grid(latent, shape):
    """Match the old clean latent to the target grid; never expand model tokens."""
    if latent.ndim != 5:
        raise ValueError("Ming calibration requires independent-frame latents")
    b, c, frames, h, w = latent.shape
    flat = latent.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
    if tuple(shape) != (h, w):
        if shape[0] <= h and shape[1] <= w:
            flat = F.interpolate(flat.float(), size=shape, mode="area")
        else:
            flat = F.interpolate(flat.float(), size=shape, mode="bilinear", align_corners=False)
    return flat.reshape(b, frames, c, *shape).permute(0, 2, 1, 3, 4)
