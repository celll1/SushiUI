"""Temporal residual transport for ACE-Step's native diffusion decoder."""
from copy import deepcopy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import checkpoint_metadata, validate_model_identity, preview_plan


def native_config(config):
    from .vendor.configuration_acestep_v15 import AceStepConfig
    if isinstance(config, dict):
        config = AceStepConfig(**deepcopy(config))
    if not isinstance(config, AceStepConfig):
        raise ValueError("ACE-Step residual transport requires native decoder config")
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "hidden_act", "rms_norm_eps", "rope_theta",
            "rope_scaling", "rope_parameters", "max_position_embeddings", "attention_bias",
            "attention_dropout", "use_sliding_window", "sliding_window", "layer_types",
            "patch_size", "in_channels", "audio_acoustic_hidden_dim")
    resolved = {key: deepcopy(getattr(config, key)) for key in keys if hasattr(config, key)}
    resolved["encoder_hidden_size"] = getattr(config, "encoder_hidden_size", None) or config.hidden_size
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "patch_size", "in_channels",
                "audio_acoustic_hidden_dim", "encoder_hidden_size", "max_position_embeddings"):
        if type(resolved[key]) is not int or resolved[key] < 1:
            raise ValueError(f"ACE-Step native {key} must be a positive integer")
    if resolved["head_dim"] % 2 or resolved["num_attention_heads"] % resolved["num_key_value_heads"]:
        raise ValueError("ACE-Step residual transport requires even RoPE heads and integral GQA")
    layers = resolved["layer_types"]
    if not isinstance(layers, (list, tuple)) or len(layers) != resolved["num_hidden_layers"] or any(
            kind not in ("full_attention", "sliding_attention") for kind in layers):
        raise ValueError("ACE-Step residual layer types must match native decoder depth")
    if "sliding_attention" in layers and (not resolved["use_sliding_window"] or
            type(resolved["sliding_window"]) is not int or resolved["sliding_window"] < 1):
        raise ValueError("ACE-Step sliding layers require a positive native window")
    for key in ("rms_norm_eps", "rope_theta"):
        value = resolved[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"ACE-Step native {key} must be finite and positive")
    return resolved


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    graph = validate_model_identity(spec, "acestep",
        [{"name": "main", "block_count": resolved["num_hidden_layers"]}], resolved)
    if any(region["down_factor"][1] != 1 for region in graph["regions"]):
        raise ValueError("ACE-Step residual regions shrink the temporal axis only: [factor,1]")
    return graph


def restore_topology(model, spec):
    decoder = getattr(model, "decoder", model)
    if spec is None:
        for module in (model, decoder):
            module.__dict__.pop("_residual_unet_spec", None)
        return
    graph = validate_topology_identity(spec, decoder.config)
    if model is not decoder and native_config(model.config) != native_config(decoder.config):
        raise ValueError("ACE-Step owner and decoder configs disagree")
    if len(decoder.layers) != native_config(decoder.config)["num_hidden_layers"]:
        raise ValueError("ACE-Step actual decoder depth disagrees with native config")
    for module in (model, decoder):
        module._residual_unet_spec = deepcopy(graph)


def model_topology_metadata(model):
    decoder = getattr(model, "decoder", model)
    spec = getattr(model, "_residual_unet_spec", None)
    if spec != getattr(decoder, "_residual_unet_spec", None):
        raise ValueError("ACE-Step owner and decoder residual topologies disagree")
    if spec is None:
        return {}
    config = native_config(decoder.config)
    if len(decoder.layers) != config["num_hidden_layers"] or (
            model is not decoder and native_config(model.config) != config):
        raise ValueError("ACE-Step export owner/decoder geometry disagrees")
    return checkpoint_metadata(validate_topology_identity(spec, decoder.config))


def preflight_forward(model, length, attention_mask, encoder_mask, *, batch_size, encoder_length, past_key_values,
                      output_attentions, custom_layers_config, enable_early_exit, return_hidden_states):
    graph = validate_topology_identity(model._residual_unet_spec, model.config)
    if len(model.layers) != native_config(model.config)["num_hidden_layers"]:
        raise ValueError("ACE-Step actual decoder depth disagrees with native config")
    preview_plan(graph, (length + model.patch_size - 1) // model.patch_size, 1)
    if model.config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("ACE-Step residual execution currently requires eager or SDPA attention")
    if past_key_values is not None or output_attentions or custom_layers_config is not None or enable_early_exit or return_hidden_states:
        raise ValueError("ACE-Step residual execution refuses cache, attention extraction and early-exit probes")
    if attention_mask is not None and (not isinstance(attention_mask, torch.Tensor) or
            attention_mask.shape != (batch_size, length) or not bool((attention_mask == 1).all())):
        raise ValueError("ACE-Step residual execution currently requires unpadded audio")
    if encoder_mask is not None and (not isinstance(encoder_mask, torch.Tensor) or
            encoder_mask.shape != (batch_size, encoder_length) or
            not bool(((encoder_mask == 0) | (encoder_mask == 1)).all()) or
            not bool(encoder_mask.bool().any(dim=1).all())):
        raise ValueError("ACE-Step residual conditioning mask requires binary native B/L with a valid token per row")


def run_rewired_decoder(model, hidden, positions, timestep_proj, encoder_hidden, encoder_mask=None, **kwargs):
    spec = validate_topology_identity(model._residual_unet_spec, model.config)
    if (positions.ndim != 2 or positions.shape[1] != hidden.shape[1]
            or positions.shape[0] not in (1, hidden.shape[0]) or not torch.isfinite(positions).all()
            or not bool((positions[:,1:] - positions[:,:-1] == 1).all())):
        raise ValueError("ACE-Step residual positions must be a regular native temporal grid")
    cross_mask = None
    if encoder_mask is not None and not bool((encoder_mask == 1).all()):
        # Conditioning stays at native length; broadcast its key mask across every query grid.
        cross_mask = hidden.new_zeros((hidden.shape[0], 1, 1, encoder_hidden.shape[1]))
        cross_mask.masked_fill_(~encoder_mask.to(device=hidden.device, dtype=torch.bool)[:, None, None, :],
                                torch.finfo(hidden.dtype).min)
    def make_context(current, states):
        mask = None
        if "sliding_attention" in model.config.layer_types:
            distances = (current.unsqueeze(2) - current.unsqueeze(1)).abs()
            mask = states.new_zeros((current.shape[0],1,current.shape[1],current.shape[1]))
            mask.masked_fill_(distances.unsqueeze(1) > model.config.sliding_window, torch.finfo(states.dtype).min)
        return dict(positions=current, rotary=model.rotary_emb(states, current), sliding_mask=mask)
    context = make_context(positions.float(), hidden)

    def coarse_context(current, states, height, width, prefix):
        positions = current["positions"]
        factor = positions.shape[1] // height
        return make_context(F.avg_pool1d(positions.unsqueeze(1), factor).squeeze(1), states)

    def execute(index, states, current):
        rotary = current["rotary"]
        count = states.shape[1]
        mask = current["sliding_mask"] if model.layers[index].attention_type == "sliding_attention" else None
        positions = current["positions"]
        cache_position = torch.arange(count, device=states.device)
        def block(value):
            return nn.Module.__call__(model.layers[index], value, rotary, timestep_proj,
                mask, positions, None, False, False, cache_position, encoder_hidden,
                cross_mask, **kwargs)[0]
        if model.training and model.gradient_checkpointing:
            return checkpoint(block, states, use_reentrant=False)
        return block(states)

    return run_rewired_stream(hidden, spec, "main", hidden.shape[1], 1, 0,
                              context, execute, coarse_context)


def export_metadata(model):
    import json
    metadata = model_topology_metadata(model)
    from .component_sources import export_sources
    metadata.update(export_sources(model))
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "to_dict"):
        raw = config.to_dict()
        if metadata:
            validate_topology_identity(model._residual_unet_spec, raw)
        metadata["acestep_config"] = json.dumps(raw, allow_nan=False)
    elif metadata:
        raise ValueError("Rewired ACE-Step export requires native wrapper config")
    return metadata


def preflight_checkpoint(path):
    import json
    import re
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    graph = read_artifact_topology(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    if any(key.startswith("net.") for key in keys) and not all(key.startswith("net.") for key in keys):
        raise ValueError("ACE-Step checkpoint mixes native and net-prefixed weights")
    config = None
    if metadata.get("acestep_config"):
        config = json.loads(metadata["acestep_config"])
        if not isinstance(config, dict):
            raise ValueError("ACE-Step embedded config must be an object")
    if graph is not None:
        if config is None or type(config.get("num_hidden_layers")) is not int:
            raise ValueError("Rewired ACE-Step requires embedded native config and explicit depth")
        validate_topology_identity(graph, config)
        native_keys = [key.removeprefix("net.") for key in keys]
        blocks = {int(match.group(1)) for key in native_keys
                  if (match := re.match(r"decoder\.layers\.(\d+)\.", key))}
        if (blocks != set(range(config["num_hidden_layers"])) or
                "decoder.proj_in.1.weight" not in native_keys or
                "decoder.proj_out.1.weight" not in native_keys or
                "null_condition_emb" not in native_keys):
            raise ValueError("Rewired ACE-Step requires complete native wrapper headers, not adapter weights")
    return graph, config


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from types import SimpleNamespace
    from api.param_defaults import TRAINING_DEFAULTS
    from core.rewiring.training import resolve_training_topology
    model_topology_metadata(model)
    existing = getattr(model, "_residual_unet_spec", None)
    declared = config.get("dit_rewiring_spec", TRAINING_DEFAULTS["dit_rewiring_spec"])
    needs_identity = existing is not None or declared is not None or (
        resume_spec is not NO_RESUME_SPEC and resume_spec is not None)
    native = native_config(model.config) if needs_identity else {}
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    resolved = resolve_training_topology(view, dict(config), "acestep", validate_topology_identity,
        lambda owner, spec: None, resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=False)
    if resolved is not None:
        decoder = getattr(model, "decoder", model)
        if decoder.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("ACE-Step residual training currently requires eager or SDPA attention")
        if native_config(decoder.config) != native:
            raise ValueError("ACE-Step training owner/decoder geometry disagrees")
        if len(decoder.layers) != native["num_hidden_layers"]:
            raise ValueError("ACE-Step actual decoder depth disagrees with native config")
    if install:
        restore_topology(model, resolved)
        config["dit_rewiring_mode"] = config.get("dit_rewiring_mode", TRAINING_DEFAULTS["dit_rewiring_mode"])
        config["dit_rewiring_spec"] = resolved
    return resolved


def resolve_generation_topology(model, files):
    from types import SimpleNamespace
    from core.rewiring.spec import read_checkpoint_metadata
    from core.rewiring.adapters import resolve_adapter_topology
    model_topology_metadata(model)
    existing = getattr(model, "_residual_unet_spec", None)
    needs_identity = existing is not None or any(read_checkpoint_metadata(file.metadata) is not None for file in files)
    native = native_config(model.config) if needs_identity else {}
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    required = resolve_adapter_topology(view, files, validate_topology_identity)
    if required is not None:
        decoder = getattr(model, "decoder", model)
        if native_config(decoder.config) != native or len(decoder.layers) != native["num_hidden_layers"]:
            raise ValueError("ACE-Step generation owner/decoder geometry disagrees")
        if decoder.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("ACE-Step residual generation currently requires eager or SDPA attention")
    return required


def generation_metadata(model, length):
    from core.rewiring.spec import digest
    topology = resolve_generation_topology(model, [])
    if topology is None:
        return None
    if type(length) is not int or length < 1:
        raise ValueError("ACE-Step residual audio length must be a positive native frame count")
    patch = native_config(model.config)["patch_size"]
    tokens = (length + patch - 1) // patch
    preview_plan(topology, tokens, 1)
    return dict(spec=topology, spec_sha256=digest(topology), kv_cache=False,
                input_frames=length, token_grid=[tokens, 1])
