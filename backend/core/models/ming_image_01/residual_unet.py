"""Ming target-only residual transport with native reference/padding state."""

import inspect
import math

import torch

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import preview_plan, validate_model_identity
from core.rewiring.training import NO_RESUME_SPEC


def spatial_alignment(model):
    """Cumulative downsample multiples for the native spatial token grid."""
    spec = getattr(model, "_residual_unet_spec", None)
    factors = {}
    height = width = 1
    for region in (spec or {}).get("regions", []):
        parent_h, parent_w = factors.get(region["parent"], (1, 1))
        h, w = region["down_factor"]
        factors[region["id"]] = (parent_h*h, parent_w*w)
        height, width = math.lcm(height, parent_h*h), math.lcm(width, parent_w*w)
    return height, width


def native_config(config):
    from .vendor.diffusion.transformer import DiffusionTransformer
    source = dict(config.config) if hasattr(config, "config") else dict(config)
    fields = inspect.signature(DiffusionTransformer.__init__).parameters
    resolved = {name: source.get(name, field.default) for name, field in fields.items() if name != "self"}
    for key in ("all_patch_size", "all_f_patch_size", "axes_dims", "axes_lens"):
        resolved[key] = list(resolved[key])
    pair = resolved["alignment_padding_mode"], resolved["multi_frame_output"]
    if type(resolved["multi_frame_output"]) is not bool or pair not in (("zero_masked", False), ("learned", True)):
        raise ValueError("Ming topology requires a verified Design/Layer profile")
    resolved["native_layout"] = "target_reference_padding_caption_v1"
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    spec = validate_model_identity(spec, "ming_image_01",
        [{"name": "main", "block_count": resolved["n_layers"]}], resolved)
    if spec["method"] == "hybrid_unet_ssdit" and spec["stream_widths"] != {"main": resolved["dim"]}:
        raise ValueError("Hybrid Ming stream width differs from native blocks")
    return spec


def restore_topology(model, spec):
    model = getattr(model, "transformer", model)
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
        model.__dict__.pop("_hybrid_initialization", None)
        if hasattr(model, "rewiring_modules"):
            del model.rewiring_modules
    else:
        spec = validate_topology_identity(spec, model)
        if spec["method"] == "hybrid_unet_ssdit":
            from core.rewiring.hybrid_modules import HybridModules
            existing = getattr(model, "rewiring_modules", None)
            if existing is None:
                reference = next(model.parameters())
                with torch.device(reference.device):
                    model.add_module("rewiring_modules", HybridModules(spec).to(dtype=reference.dtype))
            elif existing.declaration["module_layout_sha256"] != spec["module_layout_sha256"]:
                raise ValueError("Changing hybrid modules requires an explicit fresh model")
            else:
                existing.declaration = spec
        elif hasattr(model, "rewiring_modules"):
            del model.rewiring_modules
            model.__dict__.pop("_hybrid_initialization", None)
        model._residual_unet_spec = spec


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    from core.rewiring.provenance import model_metadata
    model = getattr(model, "transformer", model)
    spec = getattr(model, "_residual_unet_spec", None)
    metadata = {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model))
    return dict(metadata, **model_metadata(model))


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    resolved = native_config(config)
    preset = DIT_REWIRING_PRESETS["ming_image_01"]
    if resolved["n_layers"] != preset["block_count"]:
        return None
    return validate_topology_identity(build_spec("ming_image_01",
        [{"name": "main", "block_count": preset["block_count"]}],
        [dict(region, **OPERATORS) for region in preset["regions"]], resolved), config)


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(model, config, "ming_image_01", validate_topology_identity,
        restore_topology, default_topology, resuming=resuming,
        pending_adapter_resume=pending_adapter_resume, resume_spec=resume_spec, install=install)


def validate_forward(model, images, patch_size, f_patch_size, spec):
    validate_topology_identity(spec, model)
    if not images or any(image.ndim != 4 for image in images):
        raise ValueError("Ming rewiring requires C/F/H/W target tensors")
    shape = tuple(images[0].shape)
    if any(tuple(image.shape) != shape for image in images):
        raise ValueError("Ming rewiring requires equal target grids/frame counts per batch")
    if f_patch_size != 1 or patch_size != 2 or shape[2] % 2 or shape[3] % 2:
        raise ValueError("Ming rewiring requires the native divisible spatial patch grid")
    if shape[1] < 1 or (not model.multi_frame_output and shape[1] != 1):
        raise ValueError("Ming target frame count conflicts with checkpoint profile")
    preview_plan(spec, shape[2] // patch_size, shape[3] // patch_size)


def run_rewired_ming(model, hidden, modulation, rotary, mask, image_positions,
                     target_shape, patch_size, spec):
    spec = validate_topology_identity(spec, model)
    _, frames, pixel_h, pixel_w = target_shape
    h, w = pixel_h // patch_size, pixel_w // patch_size
    count = frames * h * w
    batch = hidden.shape[0]
    if (mask.shape != hidden.shape[:2] or rotary.shape[:2] != hidden.shape[:2]
            or count > hidden.shape[1] or not mask[:, :count].all()):
        raise ValueError("Ming target layout/mask is inconsistent")
    offsets = torch.stack([ids[0, 0] for ids in image_positions])
    spatial_origins = torch.stack([ids[0, 1:] for ids in image_positions])

    def positions(height, width):
        per_frame_h = height // frames
        ids = image_positions[0].new_zeros(batch, height * width, 3)
        ids[..., 0] = offsets[:, None] + torch.arange(frames, device=ids.device).repeat_interleave(per_frame_h * width)
        origin = spatial_origins // spatial_origins.new_tensor([h // per_frame_h, w // width])
        ids[..., 1] = origin[:, :1] + torch.arange(per_frame_h, device=ids.device).repeat_interleave(width).repeat(frames)
        ids[..., 2] = origin[:, 1:] + torch.arange(width, device=ids.device).repeat(frames * per_frame_h)
        return ids

    if not torch.equal(torch.stack([ids[:count] for ids in image_positions]), positions(frames*h, w)):
        raise ValueError("Ming target tokens do not have native frame/spatial coordinates")
    prefix = hidden.shape[1] - count
    auxiliary_rotary, auxiliary_mask = rotary[:, count:], mask[:, count:]
    transported = torch.cat((hidden[:, count:], hidden[:, :count]), dim=1)
    offloader = getattr(model, "_block_offloader", None)

    def coarse_context(context, value, height, width, extras):
        ids = positions(height, width)
        coarse = model.rope_embedder(ids.flatten(0, 1)).reshape(batch, height*width, -1)
        return (torch.cat((coarse, auxiliary_rotary), dim=1),
                torch.cat((auxiliary_mask.new_ones(batch, height*width), auxiliary_mask), dim=1))

    def execute(index, value, context):
        frequencies, current_mask = context
        native = torch.cat((value[:, prefix:], value[:, :prefix]), dim=1)
        if offloader is not None:
            offloader.wait_for_block(index)
        block = model.layers[index]
        style_info = getattr(block.attention, "_ming_style_info", None)
        if style_info is not None:
            style_info["count"] = value.shape[1] - prefix
        limit = getattr(model, "_training_gradient_checkpointing_blocks", len(model.layers))
        if torch.is_grad_enabled() and model.gradient_checkpointing and index < limit:
            output = model._gradient_checkpointing_func(block, native, current_mask, frequencies, modulation)
        else:
            output = block(native, current_mask, frequencies, modulation)
        if offloader is not None:
            offloader.submit_move_blocks_forward(index)
        targets = value.shape[1] - prefix
        return torch.cat((output[:, targets:], output[:, :targets]), dim=1)

    if spec["method"] == "hybrid_unet_ssdit":
        modules = getattr(model, "rewiring_modules", None)
        if modules is None or modules.declaration != spec:
            raise ValueError("Ming hybrid modules do not match the installed declaration")
        output = modules.forward_stream(transported, "main", frames*h, w, prefix,
            (rotary, mask), execute, coarse_context, grid_groups=frames,
            allow_encoder_replay=offloader is None,
            checkpoint_modules=model.gradient_checkpointing and getattr(model, "_training_gradient_checkpointing_blocks", len(model.layers)) > 0)
    else:
        output = run_rewired_stream(transported, spec, "main", frames*h, w, prefix,
            (rotary, mask), execute, coarse_context, grid_groups=frames)
    return torch.cat((output[:, prefix:], output[:, :prefix]), dim=1)
