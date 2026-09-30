"""MiniT2I residual image transport with native text preamble and joint blocks."""
from dataclasses import asdict
import math

import torch
from torch.utils.checkpoint import checkpoint

from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity, preview_plan, checkpoint_metadata


def native_model(model):
    holder = getattr(model, "model", model)
    return getattr(holder, "net", holder)


def native_config(config):
    from .vendor.mmjit import MMJiTConfig
    source = config if isinstance(config, dict) else asdict(native_model(config).cfg)
    resolved = {key: source.get(key, value) for key, value in asdict(MMJiTConfig()).items()}
    for key in ("image_size", "patch_size", "in_channels", "txt_input_size", "hidden_size",
                "txt_hidden_size", "cond_vec_size", "depth_double", "num_heads", "head_dim",
                "pca_channels", "prompt_length", "n_T", "cfg_channels"):
        if type(resolved[key]) is not int or resolved[key] < 1:
            raise ValueError(f"MiniT2I native {key} must be a positive integer")
    if type(resolved["txt_preamble_depth"]) is not int or resolved["txt_preamble_depth"] < 0:
        raise ValueError("MiniT2I text preamble depth must be nonnegative")
    if resolved["head_dim"] % 4 or resolved["hidden_size"] % 4:
        raise ValueError("MiniT2I native rotary and absolute position widths must be divisible by four")
    if resolved["hidden_size"] != resolved["txt_hidden_size"]:
        raise ValueError("MiniT2I native final joint projection requires equal image/text widths")
    for key in ("mlp_ratio", "noise_scale"):
        if type(resolved[key]) not in (int, float) or not math.isfinite(resolved[key]) or resolved[key] <= 0:
            raise ValueError(f"MiniT2I native {key} must be finite and positive")
        resolved[key] = float(resolved[key])
    interval = resolved["cfg_interval"]
    if not isinstance(interval, (list, tuple)) or len(interval) != 2 or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in interval):
        raise ValueError("MiniT2I cfg interval requires two finite numbers")
    resolved["cfg_interval"] = [float(value) for value in interval]
    if resolved["cfg_interval"][0] > resolved["cfg_interval"][1]:
        raise ValueError("MiniT2I cfg interval must be ordered")
    for key in ("prediction", "sampler", "llm", "vae_type"):
        if not isinstance(resolved[key], str) or not resolved[key]:
            raise ValueError(f"MiniT2I native {key} must be a nonempty string")
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "minit2i",
        [{"name": "main", "block_count": resolved["depth_double"]}], resolved)


def restore_topology(model, spec):
    net = native_model(model)
    resolved = None if spec is None else validate_topology_identity(spec, net)
    for target in (model, net):
        if resolved is None:target.__dict__.pop("_residual_unet_spec", None)
        else:target._residual_unet_spec = resolved


def model_topology_metadata(model):
    net = native_model(model)
    spec = getattr(net, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, net))


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(model, config, "minit2i", validate_topology_identity,
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
    if spec is None:
        return None, None
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            metadata = json.load(handle).get("metadata") or {}
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    declaration = metadata.get("mmjit_config")
    if not isinstance(declaration, str):
        raise ValueError("Rewired MiniT2I checkpoint requires native mmjit_config JSON")
    config = json.loads(declaration)
    if not isinstance(config, dict):
        raise ValueError("MiniT2I mmjit_config must be an object")
    config = native_config(config)
    validate_topology_identity(spec, config)
    return spec, config


def load_directory_transformer(path, torch_dtype):
    import json
    from pathlib import Path
    from .vendor.transformer import MiniT2IMMJiTModel
    from core.rewiring.artifacts import read_directory_topology
    from core.models.common.single_file_format import read_state_dict
    from core.models.common.quantized_checkpoint_guard import refuse_quantized_state_dict
    directory = Path(path)
    spec = read_directory_topology(directory)
    if spec is None:
        return MiniT2IMMJiTModel.from_pretrained(path, torch_dtype=torch_dtype)
    with (directory / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("MiniT2I native directory config must be an object")
    config = native_config(config)
    validate_topology_identity(spec, config)
    source = directory / "diffusion_pytorch_model.safetensors.index.json"
    if not source.is_file():source = directory / "diffusion_pytorch_model.safetensors"
    state, _ = read_state_dict(str(source))
    refuse_quantized_state_dict(state, arch="minit2i", path=str(source))
    model = MiniT2IMMJiTModel(**config)
    info = model.load_state_dict(state, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError("Rewired MiniT2I directory weights do not match native config")
    model.to(torch_dtype).to("cpu").eval()
    restore_topology(model, spec)
    return model


def run_rewired_minit2i(net, img, t, context, attn_mask):
    from .vendor.mmjit import get_2d_sincos_pos_embed, AttentionMode
    spec = validate_topology_identity(net._residual_unet_spec, net)
    for name in ("_style_ctx", "_fbcache", "_repa_tap_depth", "_tread_config", "_blockskip_config", "_block_skip_config"):
        if getattr(net, name, None) is not None:
            raise ValueError(f"MiniT2I residual execution is incompatible with {name}")
    for block in net.double_blocks:
        if getattr(block, "_style_ctx", None) is not None or any(
                getattr(block, name, False) for name in ("_nag_active", "_negpip_active")):
            raise ValueError("MiniT2I residual execution requires inactive guidance/style hooks")
    if img.ndim != 4:
        raise ValueError("MiniT2I residual execution requires a rank-four image/latent tensor")
    if img.shape[1] != net.cfg.in_channels:
        img = img.permute(0, 3, 1, 2)
    batch, channels, height, width = img.shape
    patch = net.cfg.patch_size
    if channels != net.cfg.in_channels or height % patch or width % patch:
        raise ValueError("MiniT2I image geometry must match complete native patches")
    gh, gw = height // patch, width // patch
    preview_plan(spec, gh, gw)
    if len(net.double_blocks) != net.cfg.depth_double or len(net.txt_preamble_blocks) != net.cfg.txt_preamble_depth:
        raise ValueError("MiniT2I residual execution native block layout mismatch")
    if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != net.cfg.txt_input_size or not context.shape[1]:
        raise ValueError("MiniT2I residual execution requires native batched text features")
    if attn_mask.shape != context.shape[:2] or t.shape != (batch,):
        raise ValueError("MiniT2I text mask and timestep must match their native batch")
    attn_mask = attn_mask.to(device=context.device)
    context = torch.where(attn_mask[:, :, None] > 0.5, context, net.mask_token.to(dtype=context.dtype))
    image, embedded_h, embedded_w = net.img_embedder(img)
    if (embedded_h, embedded_w) != (gh, gw):
        raise ValueError("MiniT2I embedder changed its native patch grid")
    image = image + get_2d_sincos_pos_embed(net.cfg.hidden_size, gh, gw, image.device, image.dtype)[None]
    text = net.txt_embedder(context.to(dtype=next(net.txt_embedder.parameters()).dtype))
    vec = net.t_embedder(t) + net.pooled_embedder(context.mean(dim=1).to(dtype=next(net.pooled_embedder.parameters()).dtype))
    use_checkpoint = net.gradient_checkpointing and net.training and torch.is_grad_enabled()
    backend = getattr(net, "_attn_backend", "native")
    mode = AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE
    for block in (*net.txt_preamble_blocks, *net.double_blocks):
        block._attn_backend = backend; block._attn_mode = mode
    for block in net.txt_preamble_blocks:
        text = checkpoint(block, text, use_reentrant=False) if use_checkpoint else block(text)
    net._repa_tap_out = None
    prefix = text.shape[1]
    offloader = getattr(net, "_block_offloader", None)

    def coarse_context(previous, hidden, h, w, extras):
        return h, w

    def execute(index, hidden, grid):
        if offloader is not None:offloader.wait_for_block(index)
        block = net.double_blocks[index]
        image, text = hidden[:, prefix:], hidden[:, :prefix]
        if use_checkpoint:
            image, text = checkpoint(block, image, text, vec, *grid, use_reentrant=False)
        else:
            image, text = block(image, text, vec, *grid)
        if offloader is not None:offloader.submit_move_blocks_forward(index)
        return torch.cat((text, image), dim=1)

    hidden = run_rewired_stream(torch.cat((text, image), dim=1), spec, "main", gh, gw, prefix,
                                (gh, gw), execute, coarse_context)
    projected = net.final_layer(hidden, vec)
    return net.unpatchify(projected[:, prefix:], gh, gw)
