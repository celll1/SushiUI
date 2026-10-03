"""Z-Image residual execution with native image-prefix and padding semantics."""
import inspect

import torch
from torch.utils.checkpoint import checkpoint

from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import preview_plan, validate_model_identity


def native_model(model):
    return getattr(model, "transformer", model)


def native_config(config):
    from .zimage_transformer import ZImageTransformer2DModel
    fields = {name: parameter.default for name, parameter in
        inspect.signature(ZImageTransformer2DModel.__init__).parameters.items() if name != "self"}
    model = config if isinstance(config, dict) else native_model(config)
    values = {key: model.get(key, value) if isinstance(model, dict) else getattr(model, key, value)
              for key, value in fields.items()}
    for key in ("all_patch_size", "all_f_patch_size", "axes_dims", "axes_lens"):
        sequence = values[key]
        if not isinstance(sequence, (list, tuple)) or not sequence or any(type(value) is not int or value < 1 for value in sequence):
            raise ValueError(f"Z-Image {key} must contain positive integers")
        values[key] = list(sequence)
    if len(values["all_patch_size"]) != len(values["all_f_patch_size"]):
        raise ValueError("Z-Image patch banks must have paired spatial/temporal factors")
    if len(values["axes_dims"]) != 3 or len(values["axes_lens"]) != 3:
        raise ValueError("Z-Image native RoPE requires three axes")
    return values


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "zimage",
        [{"name": "main", "block_count": resolved["n_layers"]}], resolved)


def restore_topology(model, spec):
    model = native_model(model)
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
    else:
        model._residual_unet_spec = validate_topology_identity(spec, model)


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    model = native_model(model)
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model))


def validate_forward(model, spec, images, patch_size, f_patch_size):
    validate_topology_identity(spec, model)
    for name in ("_nag_request", "_negpip_request", "_style_ctx", "_fbcache",
                 "_tread_config", "_block_skip_config", "_repa_tap_depth"):
        if getattr(model, name, None) is not None:
            raise ValueError(f"Z-Image rewiring is incompatible with {name}")
    for attention_type in {type(layer.attention) for layer in model.layers}:
        if getattr(attention_type, "_nag_ctx", None) is not None or getattr(attention_type, "_negpip_ctx", None) is not None:
            raise ValueError("Z-Image rewiring requires inactive attention guidance contexts")
    if f_patch_size != 1 or not images or any(image.ndim != 4 or image.shape[1] != 1 for image in images):
        raise ValueError("Z-Image rewiring requires single-frame 2D images")
    shape = tuple(images[0].shape)
    if any(tuple(image.shape) != shape for image in images):
        raise ValueError("Z-Image rewiring requires one image grid per batch")
    if type(patch_size) is not int or patch_size < 1 or shape[2] % patch_size or shape[3] % patch_size:
        raise ValueError("Z-Image rewiring requires a divisible native patch grid")
    preview_plan(spec, shape[2] // patch_size, shape[3] // patch_size)


def run_rewired_zimage(model, hidden, modulation, rotary, mask, image_sizes,
                       image_positions, patch_size, spec):
    spec = validate_topology_identity(spec, model)
    batch = hidden.shape[0]
    if len(image_sizes) != batch or len(image_positions) != batch or len(model.layers) != spec["streams"][0]["block_count"]:
        raise ValueError("Z-Image rewiring batch or block layout mismatch")
    frames, height, width = image_sizes[0]
    if frames != 1 or any(size != image_sizes[0] for size in image_sizes):
        raise ValueError("Z-Image rewiring requires one 2D image grid per batch")
    height, width = height // patch_size, width // patch_size
    count = height * width
    if mask.shape != hidden.shape[:2] or not mask[:, :count].bool().all() or rotary.shape[:2] != hidden.shape[:2]:
        raise ValueError("Z-Image rewiring requires unmasked native image-prefix tokens")
    offsets = torch.stack([positions[0, 0] for positions in image_positions])

    def positions(h, w):
        ids = image_positions[0].new_zeros(batch, h*w, 3)
        ids[..., 0] = offsets[:, None]
        ids[..., 1] = torch.arange(h, device=ids.device).repeat_interleave(w)
        ids[..., 2] = torch.arange(w, device=ids.device).repeat(h)
        return ids

    if not torch.equal(torch.stack([ids[:count] for ids in image_positions]), positions(height, width)):
        raise ValueError("Z-Image rewiring requires native row-major image coordinates")
    # Padding and captions evolve through every block; only actual image tokens are pooled.
    extras = hidden.shape[1] - count
    auxiliary_rotary, auxiliary_mask = rotary[:, count:], mask[:, count:]
    transported = torch.cat((hidden[:, count:], hidden[:, :count]), dim=1)
    offloader = getattr(model, "_block_offloader", None)

    def coarse_context(context, value, h, w, prefix):
        coarse = model.rope_embedder(positions(h, w).reshape(batch*h*w, 3)).reshape(batch, h*w, -1)
        return (torch.cat((coarse, auxiliary_rotary), dim=1),
                torch.cat((auxiliary_mask.new_ones(batch, h*w), auxiliary_mask), dim=1))

    def execute(index, value, context):
        frequencies, current_mask = context
        native = torch.cat((value[:, extras:], value[:, :extras]), dim=1)
        if offloader is not None:
            offloader.wait_for_block(index)
        layer = model.layers[index]
        if model.gradient_checkpointing and model.training:
            output = checkpoint(layer, native, current_mask, frequencies, modulation, use_reentrant=False)
        else:
            output = layer(native, current_mask, frequencies, modulation)
        if offloader is not None:
            offloader.submit_move_blocks_forward(index)
        image_count = value.shape[1] - extras
        return torch.cat((output[:, image_count:], output[:, :image_count]), dim=1)

    output = run_rewired_stream(transported, spec, "main", height, width, extras,
                               (rotary, mask), execute, coarse_context)
    return torch.cat((output[:, extras:], output[:, :extras]), dim=1)


def preflight_artifact(path):
    import json
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    source = Path(path)
    if not source.exists() and Path(str(source)+".index.json").is_file():
        source = Path(str(source)+".index.json")
    topology = read_artifact_topology(source)
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            metadata = json.load(handle).get("metadata") or {}
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    declaration = metadata.get("transformer_config")
    config = None
    if declaration:
        if not isinstance(declaration, str):
            raise ValueError("Z-Image native config metadata must be JSON text")
        config = json.loads(declaration)
        if not isinstance(config, dict):
            raise ValueError("Z-Image native config must be an object")
        config = native_config(config)
    elif topology is not None:
        raise ValueError("Rewired Z-Image artifact requires native config metadata")
    if topology is not None:
        topology = validate_topology_identity(topology, config)
    return topology, config


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(native_model(model), config, "zimage",
        validate_topology_identity, restore_topology, default_topology, resuming=resuming,
        pending_adapter_resume=pending_adapter_resume, resume_spec=resume_spec, install=install)


def default_topology(config):
    from copy import deepcopy
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    preset = deepcopy(DIT_REWIRING_PRESETS["zimage"])
    resolved = native_config(config)
    if resolved["n_layers"] != preset["block_count"]:
        return None
    return validate_topology_identity(build_spec("zimage",
        [{"name": "main", "block_count": preset["block_count"]}],
        [dict(region, **OPERATORS) for region in preset["regions"]], resolved), config)


def generation_token_stride(vae):
    blocks = getattr(getattr(vae, "config", None), "block_out_channels", None)
    return (2 ** (len(blocks) - 1) if blocks else 8) * 2
