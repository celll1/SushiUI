"""Residual transport restricted to YuE2 acoustic NAR blocks."""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import checkpoint_metadata, digest, preview_plan, validate_model_identity

GRID_POLICY = "zero_hidden_tail_v1"


def native_config(config):
    resolved = {key: deepcopy(getattr(config, key)) for key in config._inference_fields}
    resolved["sushi_residual_grid_policy"] = GRID_POLICY
    return resolved


def validate_topology_identity(spec, config):
    graph = validate_model_identity(spec, "yue2",
        [{"name": "nar", "block_count": config.num_hidden_layers}], native_config(config))
    if config.latent_dim != 64 or config.latent_type != "vae":
        raise ValueError("YuE2 residual NAR requires native 64-channel VAE latents")
    if any(region["down_factor"][1] != 1 for region in graph["regions"]):
        raise ValueError("YuE2 residual NAR shrinks the acoustic timeline only: [factor,1]")
    return graph


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
        return
    graph = validate_topology_identity(spec, model.config)
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise ValueError("YuE2 actual depth disagrees with native config")
    model._residual_unet_spec = deepcopy(graph)


def model_topology_metadata(model):
    graph = getattr(model, "_residual_unet_spec", None)
    if graph is None:
        return {}
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise ValueError("YuE2 actual depth disagrees with native config")
    return checkpoint_metadata(validate_topology_identity(graph, model.config))


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    preset = DIT_REWIRING_PRESETS["yue2"]
    if config.num_hidden_layers != preset["block_count"]:
        return None
    return build_spec("yue2", [{"name": "nar", "block_count": config.num_hidden_layers}],
        [{**region, **OPERATORS} for region in preset["regions"]], native_config(config))



def temporal_geometry(config):
    from api.param_defaults import audio_defaults_for_arch
    if config.latent_dim != 64 or config.latent_type != "vae":
        raise ValueError("YuE2 residual geometry requires native 64-channel VAE latents")
    return dict(frame_rate=25, patch_size=1,
        preview_frames=max(1, round(audio_defaults_for_arch("yue2")["audio_duration"] * 25)),
        grid_policy=GRID_POLICY)


def model_info(components):
    model = components.get("transformer")
    graph = getattr(model, "_residual_unet_spec", None)
    if graph is None:
        return {"dit_rewiring_spec": None}
    model_topology_metadata(model)
    if components.get("frame_rate") != 25 or components.get("latent_channels") != 64:
        raise ValueError("YuE2 residual model-info requires native 25 Hz / 64-channel geometry")
    return {"dit_rewiring_spec": deepcopy(graph),
            "dit_rewiring_temporal_geometry": temporal_geometry(model.config)}


def preflight_engine(model, native_tokens, ar_tokens):
    graph = getattr(model, "_residual_unet_spec", None)
    if graph is None:
        return None
    model_topology_metadata(model)
    graph = validate_topology_identity(graph, model.config)
    factors, alignment = {}, 1
    for region in graph["regions"]:
        factor = factors.get(region["parent"], 1) * region["down_factor"][0]
        factors[region["id"]] = factor
        alignment = math.lcm(alignment, factor)
    padded = ((native_tokens + alignment - 1) // alignment) * alignment
    preview_plan(graph, padded, 1)
    if ar_tokens + padded + 2 > model.config.max_position_embeddings:
        raise ValueError("YuE2 padded acoustic chunk exceeds the model context")
    return {"graph": deepcopy(graph), "sha256": digest(graph), "padded_tokens": padded}


def assert_engine_topology(engine):
    graph = getattr(engine.model, "_residual_unet_spec", None)
    state = engine.residual_state
    if (graph is None) != (state is None):
        raise ValueError("YuE2 NAR topology changed after AR prefill")
    if state is not None:
        model_topology_metadata(engine.model)
        if digest(graph) != state["sha256"]:
            raise ValueError("YuE2 NAR topology changed after AR prefill")


def run_native_blocks(engine, hidden):
    state = engine.residual_state
    native, padded = engine.nar_length - 2, state["padded_tokens"]
    # Both boundary tokens travel as prefixes, outside the acoustic grid.
    hidden = torch.cat((hidden[:, :1], hidden[:, -1:],
                        F.pad(hidden[:, 1:-1], (0, 0, 0, padded - native))), dim=1)
    def rotary(height):
        positions = torch.cat((torch.tensor([engine.ar_length, engine.ar_length + native + 1],
                                            device=hidden.device),
                               torch.arange(engine.ar_length + 1, engine.ar_length + 1 + min(height, native),
                                            device=hidden.device),
                               torch.arange(engine.ar_length + native + 2,
                                            engine.ar_length + native + 2 + max(0, height - native),
                                            device=hidden.device)))[None]
        return engine.model.model.rotary_emb(positions)
    def coarse_context(context, states, height, width, prefix):
        if width != 1 or prefix != 2:
            raise ValueError("YuE2 residual NAR requires two preserved boundary tokens")
        return rotary(height)
    def execute(index, states, context):
        layer = engine.model.model.layers[index]
        ar_k, ar_v = engine.cache[index]
        q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(states), *context)
        k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
        update = engine._attention(q[0], k, v)
        states = states + layer.nar_self_attn.o_proj(update.flatten(1)[None])
        return states + layer.nar_mlp(layer.nar_pre_mlp_layernorm(states))
    hidden = run_rewired_stream(hidden, state["graph"], "nar", padded, 1, 2,
                                rotary(padded), execute, coarse_context)
    return torch.cat((hidden[:, :1], hidden[:, 2:2 + native], hidden[:, 1:2]), dim=1)
