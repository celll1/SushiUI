"""SenseNova generation-only residual transport over immutable understanding KV."""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import checkpoint_metadata, validate_model_identity


def _decoder_config(config):
    keys = ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
            "max_position_embeddings", "max_position_embeddings_hw", "rope_theta",
            "rope_theta_hw", "rms_norm_eps", "attention_bias", "mlp_bias",
            "attention_dropout", "layer_types", "rope_parameters", "tie_word_embeddings")
    result = {key: deepcopy(getattr(config, key)) for key in keys if hasattr(config, key)}
    for key in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim"):
        if type(result.get(key)) is not int or result[key] < 1:
            raise ValueError(f"SenseNova native {key} must be a positive integer")
    if result["head_dim"] % 4 or result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("SenseNova residual transport requires native quarter-width spatial RoPE and GQA")
    if result.get("layer_types") != ["full_attention"] * result["num_hidden_layers"]:
        raise ValueError("SenseNova residual transport requires full-attention decoder layers")
    for key in ("rope_theta", "rope_theta_hw", "rms_norm_eps"):
        value = result.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"SenseNova native {key} must be finite and positive")
    return result


def native_config(config):
    from .vendor.configuration_neo_chat import NEOChatConfig
    if isinstance(config, dict):
        config = NEOChatConfig(**config)
    if not isinstance(config, NEOChatConfig) or config.is_moe_llm:
        raise ValueError("SenseNova residual transport requires the dense native NEOChat config")
    ratio = config.downsample_ratio
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("SenseNova residual transport requires a finite positive merge ratio")
    merge = int(1 / ratio)
    patch = config.gen_patch_size or config.vision_config.patch_size * merge
    channels = config.gen_in_channels or 3
    if (merge < 1 or config.downsample_ratio != 1 / merge or type(patch) is not int
            or patch < 4 or patch % 4 or patch % merge or type(channels) is not int or channels < 1):
        raise ValueError("SenseNova residual transport requires valid native generation geometry")
    return dict(llm_config=_decoder_config(config.llm_config), gen_patch_size=patch,
                gen_in_channels=channels, merge_size=merge,
                use_pixel_head=bool(getattr(config, "use_pixel_head", False)))


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "sensenova",
        [{"name": "main", "block_count": resolved["llm_config"]["num_hidden_layers"]}], resolved)


def restore_topology(model, spec):
    resolved = native_config(model.config)
    decoder = model.language_model.model
    if (_decoder_config(decoder.config) != resolved["llm_config"]
            or len(decoder.layers) != resolved["llm_config"]["num_hidden_layers"]):
        raise ValueError("SenseNova owner and decoder configs disagree")
    graph = None if spec is None else validate_topology_identity(spec, model.config)
    for module in (model, decoder):
        if graph is None:
            module.__dict__.pop("_residual_unet_spec", None)
            module.__dict__.pop("_residual_unet_native_config", None)
        else:
            module._residual_unet_spec = deepcopy(graph)
            module._residual_unet_native_config = deepcopy(resolved)


def model_topology_metadata(model):
    spec = getattr(model, "_residual_unet_spec", None)
    decoder_spec = getattr(model.language_model.model, "_residual_unet_spec", None)
    if spec != decoder_spec:
        raise ValueError("SenseNova owner and generation decoder topologies disagree")
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model.config))


def run_rewired_generation(model, hidden, indexes, prefix_cache, *,
                           attention_mask=None, checkpoint_layers=False, packed_gen=None):
    spec = getattr(model, "_residual_unet_spec", None)
    config = getattr(model, "_residual_unet_native_config", None)
    if config is None or _decoder_config(model.config) != config["llm_config"]:
        raise ValueError("SenseNova residual decoder requires its restored native config")
    validate_model_identity(spec, "sensenova",
        [{"name": "main", "block_count": len(model.layers)}], config)
    if packed_gen is not None or getattr(prefix_cache, "packed", None) is not None:
        raise ValueError("SenseNova residual transport does not yet support packed image batches")
    if attention_mask is not None:
        raise ValueError("SenseNova residual transport requires unmasked generation attention")
    if getattr(model, "_repa_tap_depth", None) is not None:
        raise ValueError("SenseNova residual transport does not support REPA taps")
    if any(getattr(layer.self_attn, "_style_ctx", None) is not None for layer in model.layers):
        raise ValueError("SenseNova residual transport does not support reference-style attention")
    if hidden.ndim != 3 or not isinstance(indexes, torch.Tensor) or indexes.shape != (3, hidden.shape[1]):
        raise ValueError("SenseNova residual transport requires one contiguous native 2D image grid")
    height = int(torch.unique(indexes[1]).numel())
    width = int(torch.unique(indexes[2]).numel())
    rows = torch.arange(height, device=indexes.device).repeat_interleave(width)
    cols = torch.arange(width, device=indexes.device).repeat(height)
    if (height * width != hidden.shape[1] or height == 0 or width == 0
            or not torch.equal(indexes[1], rows + indexes[1, 0])
            or not torch.equal(indexes[2], cols + indexes[2, 0])
            or not bool(torch.isfinite(indexes).all())
            or not bool((indexes[0] == indexes[0, 0]).all())):
        raise ValueError("SenseNova residual transport requires regular row-major spatial positions")
    if prefix_cache is not None:
        if len(prefix_cache.layers) != len(model.layers):
            raise ValueError("SenseNova residual prefix cache must have every native decoder layer")
        for layer in prefix_cache.layers:
            keys, values = layer.keys, layer.values
            llm = config["llm_config"]
            if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
                    or keys.ndim != 4 or keys.shape != values.shape or keys.shape[2] < 1
                    or keys.shape[0] != hidden.shape[0] or keys.shape[1] != llm["num_key_value_heads"]
                    or keys.shape[3] != llm["head_dim"]):
                raise ValueError("SenseNova residual prefix KV must match native batch/head geometry")
    context = (indexes.float(), height, width)

    def coarse_context(current, states, new_h, new_w, prefix):
        positions, old_h, old_w = current
        factor = (old_h // new_h, old_w // new_w)
        coarse = F.avg_pool2d(positions.reshape(1, 3, old_h, old_w), factor).reshape(3, -1)
        return coarse, new_h, new_w

    def execute(index, states, current):
        positions = current[0]
        indicators = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device)

        def block(value):
            return nn.Module.__call__(model.layers[index], value,
                image_gen_indicators=indicators, exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True, indexes=positions, attention_mask=None,
                past_key_values=prefix_cache, use_cache=False, update_cache=False)

        return checkpoint(block, states, use_reentrant=False) if checkpoint_layers else block(states)

    return run_rewired_stream(hidden, spec, "main", height, width, 0, context, execute, coarse_context)
