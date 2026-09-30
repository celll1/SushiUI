"""Lens residual execution with native evolving text and centered image RoPE."""
import inspect

import torch

from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity, preview_plan, checkpoint_metadata


def native_config(config):
    from .vendor.transformer import LensTransformer2DModel
    source = config if isinstance(config, dict) else dict(config.config)
    signature = inspect.signature(LensTransformer2DModel.__init__)
    resolved = {key: source.get(key, value.default) for key, value in signature.parameters.items()
                if key != "self" and value.default is not inspect.Parameter.empty}
    for key in ("patch_size", "in_channels", "num_layers", "attention_head_dim",
                "num_attention_heads", "enc_hidden_dim"):
        if type(resolved[key]) is not int or resolved[key] < 1:
            raise ValueError(f"Lens native {key} must be a positive integer")
    if resolved["out_channels"] is not None and (type(resolved["out_channels"]) is not int or resolved["out_channels"] < 1):
        raise ValueError("Lens native output channels must be positive or null")
    axes = resolved["axes_dims_rope"]
    if not isinstance(axes, (list, tuple)) or len(axes) != 3 or any(type(d) is not int or d < 2 or d % 2 for d in axes):
        raise ValueError("Lens rewiring requires three even RoPE axes")
    if sum(axes) != resolved["attention_head_dim"]:
        raise ValueError("Lens RoPE axes must match the native attention head")
    resolved["axes_dims_rope"] = list(axes)
    indices = resolved["selected_layer_index"]
    if not isinstance(indices, (list, tuple)) or any(type(i) is not int for i in indices):
        raise ValueError("Lens selected text layers must be integer indices")
    if resolved["multi_layer_encoder_feature"] and not indices:
        raise ValueError("Lens multilayer text features require selected layers")
    resolved["selected_layer_index"] = list(indices)
    for key in ("gate_mlp", "rms_norm", "multi_layer_encoder_feature"):
        if type(resolved[key]) is not bool:
            raise ValueError(f"Lens native {key} must be a boolean")
    resolved["inner_dim"] = resolved["num_attention_heads"] * resolved["attention_head_dim"]
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "lens", [{"name": "main", "block_count": resolved["num_layers"]}], resolved)


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
    else:
        model._residual_unet_spec = validate_topology_identity(spec, dict(model.config))


def model_topology_metadata(model):
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, dict(model.config)))


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(model, config, "lens", validate_topology_identity,
        restore_topology, resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=install)


def preflight_checkpoint(path):
    import json
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    source = Path(path)
    if not source.exists() and Path(str(source) + ".index.json").is_file():
        source = Path(str(source) + ".index.json")
    spec = read_artifact_topology(source)
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            metadata = json.load(handle).get("metadata") or {}
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    declaration = metadata.get("transformer_config")
    if declaration is None:
        if spec is not None:
            raise ValueError("Rewired Lens checkpoint requires native transformer_config")
        return spec, None
    if not isinstance(declaration, str):
        raise ValueError("Lens transformer_config must be JSON text")
    config = json.loads(declaration)
    if not isinstance(config, dict):
        raise ValueError("Lens transformer_config must be an object")
    config = native_config(config)
    if spec is not None:
        validate_topology_identity(spec, config)
    return spec, config


def preflight_directory(path):
    import json
    from pathlib import Path
    from core.rewiring.artifacts import read_directory_topology
    directory = Path(path) / "transformer"
    if not directory.is_dir():
        return None, None, None
    spec = read_directory_topology(directory)
    if spec is None:
        return None, None, None
    with (directory / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Lens directory transformer config must be an object")
    config = native_config(config)
    validate_topology_identity(spec, config)
    source = directory / "diffusion_pytorch_model.safetensors.index.json"
    if not source.is_file():
        source = directory / "diffusion_pytorch_model.safetensors"
    if not source.is_file():
        raise ValueError("Rewired Lens directory requires native safetensors weights")
    return spec, config, source


def _grid(shapes, batch, count):
    if not isinstance(shapes, (list, tuple)) or not shapes:
        raise ValueError("Lens rewiring requires a native image grid")
    if len(shapes) == 3 and all(type(v) is int for v in shapes):
        shapes = [shapes]
    if len(shapes) == 1 and isinstance(shapes[0], list):
        shapes = shapes[0]
    if len(shapes) not in (1, batch):
        raise ValueError("Lens rewiring requires one shared image grid")
    for shape in shapes:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3 or any(type(v) is not int or v < 1 for v in shape):
            raise ValueError("Lens image grid must contain positive integer T/H/W")
        if tuple(shape) != tuple(shapes[0]):
            raise ValueError("Lens rewiring requires identical image grids across the batch")
    frames, height, width = shapes[0]
    if frames != 1 or height * width != count:
        raise ValueError("Lens rewiring requires a contiguous single-frame image grid without references")
    return height, width


def run_rewired_lens(model, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                     timestep, img_shapes, attention_kwargs=None):
    spec = validate_topology_identity(model._residual_unet_spec, dict(model.config))
    for name in ("_style_ctx", "_fbcache", "_repa_tap_depth", "_tread_config", "_blockskip_config", "_block_skip_config"):
        if getattr(model, name, None) is not None:
            raise ValueError(f"Lens residual execution is incompatible with {name}")
    if attention_kwargs:
        raise ValueError("Lens residual execution requires native attention without extra arguments")
    for block in model.transformer_blocks:
        if getattr(block.attn, "_style_ctx", None) is not None or any(
                getattr(block.attn, name, False) for name in ("_nag_enabled", "_negpip_enabled")):
            raise ValueError("Lens residual execution requires inactive attention guidance/style")
    if hidden_states.ndim != 3:
        raise ValueError("Lens residual execution requires packed image tokens")
    batch, count, _ = hidden_states.shape
    height, width = _grid(img_shapes, batch, count)
    preview_plan(spec, height, width)
    if len(model.transformer_blocks) != model.config.num_layers:
        raise ValueError("Lens residual execution block layout mismatch")
    features = encoder_hidden_states
    if model.multi_layer_encoder_feature:
        if not isinstance(features, (list, tuple)) or len(features) != len(model.selected_layer_index):
            raise ValueError("Lens residual execution requires all selected text feature layers")
    elif isinstance(features, torch.Tensor):
        features = [features]
    else:
        raise ValueError("Lens residual execution requires a single text feature tensor")
    if any(not isinstance(value, torch.Tensor) or value.ndim != 3 for value in features):
        raise ValueError("Lens text features must be rank-three tensors")
    prefix = features[0].shape[1]
    if any(value.shape[:2] != (batch, prefix) for value in features):
        raise ValueError("Lens text feature batches and lengths must agree")
    if encoder_hidden_states_mask.shape != (batch, prefix):
        raise ValueError("Lens text mask must match the native text sequence")
    if not model.pos_embed.scale_rope or model.pos_embed.theta != 10000:
        raise ValueError("Lens residual execution requires native centered RoPE")
    # Native centered-image RoPE offsets text by the full grid, even in coarse blocks.
    offset = max(height // 2, width // 2)
    if offset + prefix > len(model.pos_embed.pos_freqs) or max(height, width) > 2 * len(model.pos_embed.pos_freqs):
        raise ValueError("Lens image/text coordinates exceed the native RoPE table")
    image = model.img_in(hidden_states)
    model._stamp_style_context()
    if model.multi_layer_encoder_feature:
        text = model.txt_in(torch.cat([model.txt_norm[i](value) for i, value in enumerate(features)], dim=-1))
    else:
        text = model.txt_in(model.txt_norm(features[0]))
    temb = model.time_text_embed(timestep.to(image.dtype), image)
    image_rotary, text_rotary = model.pos_embed([(1, height, width)], [prefix], device=image.device)

    def context(h, w):
        rotary, _ = model.pos_embed([(1, h, w)], [prefix], device=image.device)
        return (rotary, text_rotary), model._build_joint_attention_mask(encoder_hidden_states_mask, h * w)

    def coarse_context(previous, value, h, w, extras):
        return context(h, w)

    offloader = getattr(model, "_block_offloader", None)
    def execute(index, value, current):
        if offloader is not None:
            offloader.wait_for_block(index)
        text_out, image_out = model.transformer_blocks[index](value[:, prefix:], value[:, :prefix], temb,
                                                              current[0], current[1])
        if offloader is not None:
            offloader.submit_move_blocks_forward(index)
        return torch.cat((text_out, image_out), dim=1)

    initial = ((image_rotary, text_rotary), model._build_joint_attention_mask(encoder_hidden_states_mask, count))
    value = run_rewired_stream(torch.cat((text, image), dim=1), spec, "main", height, width, prefix,
                               initial, execute, coarse_context)
    return model.proj_out(model.norm_out(value[:, prefix:], temb))
