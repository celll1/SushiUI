"""SenseNova generation-only residual transport over immutable understanding KV."""
from copy import deepcopy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from core.rewiring.training import NO_RESUME_SPEC
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
        if "merge_size" in config and "vision_config" not in config:
            from types import SimpleNamespace
            merge = config.get("merge_size")
            patch = config.get("gen_patch_size")
            channels = config.get("gen_in_channels")
            if (type(merge) is not int or merge < 1 or type(patch) is not int
                    or patch < 4 or patch % 4 or patch % merge or type(channels) is not int
                    or channels < 1 or type(config.get("use_pixel_head")) is not bool
                    or not isinstance(config.get("llm_config"), dict)):
                raise ValueError("SenseNova canonical generation geometry is invalid")
            return dict(llm_config=_decoder_config(SimpleNamespace(**config["llm_config"])),
                        gen_patch_size=patch, gen_in_channels=channels, merge_size=merge,
                        use_pixel_head=config["use_pixel_head"])
        try:
            config = NEOChatConfig(**config)
        except Exception as exc:
            raise ValueError("SenseNova native NEOChat config is invalid") from exc
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


def default_topology(config):
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    resolved = native_config(config)
    preset = DIT_REWIRING_PRESETS["sensenova"]
    count = resolved["llm_config"]["num_hidden_layers"]
    if count != preset["block_count"]:
        return None
    return build_spec("sensenova", [{"name": "main", "block_count": count}],
                      [dict(region, **OPERATORS) for region in preset["regions"]], resolved)


def restore_topology(model, spec):
    decoder = getattr(getattr(model, "language_model", None), "model", None)
    if spec is None:
        for module in (model, decoder):
            if module is not None:
                module.__dict__.pop("_residual_unet_spec", None)
                module.__dict__.pop("_residual_unet_native_config", None)
        return
    graph = validate_owner_topology(spec, model)
    resolved = native_config(model.config)
    for module in (model, decoder):
        module._residual_unet_spec = deepcopy(graph)
        module._residual_unet_native_config = deepcopy(resolved)


def validate_owner_topology(spec, model):
    resolved = native_config(model.config)
    decoder = getattr(getattr(model, "language_model", None), "model", None)
    if (decoder is None or _decoder_config(decoder.config) != resolved["llm_config"]
            or len(decoder.layers) != resolved["llm_config"]["num_hidden_layers"]):
        raise ValueError("SenseNova owner and decoder configs disagree")
    return validate_topology_identity(spec, model.config)


def model_topology_metadata(model):
    spec = getattr(model, "_residual_unet_spec", None)
    decoder = getattr(getattr(model, "language_model", None), "model", None)
    decoder_spec = getattr(decoder, "_residual_unet_spec", None)
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
        streamer = getattr(prefix_cache, "_kv_cache_streamer", None)
        if streamer is not None:
            branch = getattr(prefix_cache, "_kv_cache_streamer_branch", None)
            masters = getattr(streamer, "_masters", {}).get(branch)
            if (not isinstance(masters, list) or len(masters) != len(model.layers)
                    or getattr(streamer, "_current_branch", None) != branch):
                raise ValueError("SenseNova residual streaming prefix branch must match the active branch")
            tensors = masters
            token_axis, head_axis = 1, 2
        else:
            tensors = [(layer.keys, layer.values) for layer in prefix_cache.layers]
            token_axis, head_axis = 2, 1
        for keys, values in tensors:
            llm = config["llm_config"]
            if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
                    or keys.ndim != 4 or keys.shape != values.shape or keys.shape[token_axis] < 1
                    or keys.shape[0] != hidden.shape[0] or keys.shape[head_axis] != llm["num_key_value_heads"]
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


def preflight_checkpoint(path, *, _seen=None):
    """Bind every shard and refiner dependency before reading model tensors."""
    import json
    import re
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    from .loader import sensenova_checkpoint_metadata, SENSENOVA_REFINER_DELTA_KIND
    source = Path(path)
    if not source.is_file() and Path(str(source) + ".index.json").is_file():
        source = Path(str(source) + ".index.json")
    source = source.resolve()
    seen = set() if _seen is None else set(_seen)
    if source in seen:
        raise ValueError("SenseNova residual artifact dependency contains a cycle")
    if len(seen) >= 64:
        raise ValueError("SenseNova residual artifact dependency chain exceeds 64 files")
    seen.add(source)
    spec = read_artifact_topology(source)
    metadata = sensenova_checkpoint_metadata(str(source))
    if spec is not None:
        declaration = metadata.get("sensenova_config")
        if not isinstance(declaration, str) or not declaration.strip():
            raise ValueError("Rewired SenseNova requires embedded native sensenova_config JSON")
        config = json.loads(declaration)
        if (not isinstance(config, dict) or not isinstance(config.get("llm_config"), dict)
                or "architectures" not in config["llm_config"]):
            raise ValueError("SenseNova checkpoint requires raw native NEOChat config")
        validate_topology_identity(spec, config)
    if metadata.get("sensenova_checkpoint_kind") == SENSENOVA_REFINER_DELTA_KIND:
        base = str(metadata.get("sensenova_refiner_base_model_path") or "").strip()
        if not base:
            raise ValueError("SenseNova refiner delta requires a base checkpoint path")
        base = Path(base)
        if not base.is_absolute():
            base = source.parent / base
        base_spec = preflight_checkpoint(base, _seen=seen)
        if spec != base_spec:
            raise ValueError("SenseNova refiner delta and base residual topologies disagree")
    elif spec is not None:
        if str(source).endswith(".safetensors.index.json"):
            with source.open(encoding="utf-8") as handle:
                keys = list(json.load(handle)["weight_map"])
        else:
            with safe_open(str(source), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
        prefix = "transformer.language_model.model.layers."
        blocks = {int(match.group(1)) for key in keys
                  if (match := re.match(re.escape(prefix) + r"(\d+)\.", key))}
        count = native_config(config)["llm_config"]["num_hidden_layers"]
        if blocks != set(range(count)) or not any(
                key.startswith("transformer.fm_modules.fm_head.") for key in keys) or not any(
                key.endswith("self_attn.q_proj_mot_gen.weight") for key in keys):
            raise ValueError("Rewired SenseNova requires complete native model blocks, not adapter weights")
    return spec


def inspect_artifact(path):
    import json
    import re
    from pathlib import Path
    from safetensors import safe_open
    from core.models.common.single_file_format import parse_component_metadata
    from .loader import sensenova_checkpoint_metadata, SENSENOVA_REFINER_DELTA_KIND, _load_sensenova_config
    source = Path(path)
    if not source.is_file() and Path(str(source) + ".index.json").is_file():
        source = Path(str(source) + ".index.json")
    topology = preflight_checkpoint(source)
    metadata = sensenova_checkpoint_metadata(str(source))
    declaration = metadata.get("sensenova_config")
    if declaration is None:
        _, config = _load_sensenova_config(metadata, str(source.parent))
    else:
        if not isinstance(declaration, str) or not declaration.strip():
            raise ValueError("SenseNova inspection requires native config JSON")
        config = json.loads(declaration)
    if (not isinstance(config, dict) or not isinstance(config.get("llm_config"), dict)
            or type(config["llm_config"].get("num_hidden_layers")) is not int
            or "architectures" not in config["llm_config"]):
        raise ValueError("SenseNova inspection requires an explicit native block count")
    resolved = native_config(config)
    if metadata.get("sensenova_checkpoint_kind") == SENSENOVA_REFINER_DELTA_KIND:
        base = Path(metadata["sensenova_refiner_base_model_path"])
        if not base.is_absolute():
            base = source.parent / base
        base_config, _, stride = inspect_artifact(base)
        if native_config(base_config) != resolved:
            raise ValueError("SenseNova refiner and base generation geometry disagree")
        return config, topology, stride
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            keys = list(json.load(handle)["weight_map"])
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    key_prefix = "transformer." if any(key.startswith("transformer.") for key in keys) else ""
    prefix = key_prefix + "language_model.model.layers."
    blocks = {int(match.group(1)) for key in keys
              if (match := re.match(re.escape(prefix) + r"(\d+)\.", key))}
    count = resolved["llm_config"]["num_hidden_layers"]
    if (blocks != set(range(count)) or not any(
            key.startswith(key_prefix + "fm_modules.fm_head.") for key in keys) or not any(
            key.endswith("self_attn.q_proj_mot_gen.weight") for key in keys)):
        raise ValueError("SenseNova inspection requires complete native blocks and generation head")
    vae = parse_component_metadata(metadata).get("vae") or {}
    if config.get("gen_patch_size") is None and config.get("gen_in_channels") is None:
        if vae:
            raise ValueError("SenseNova pixel config cannot declare a swapped VAE")
        scale = 1
    else:
        if config.get("gen_patch_size") is None or config.get("gen_in_channels") is None:
            raise ValueError("SenseNova latent config requires both generation patch and channels")
        scale = vae.get("scale_factor")
        try:
            channels = int(vae.get("channels", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("SenseNova editor requires declared VAE channels") from exc
        from core.models.components.spatial_swap import read_declaration
        alignment = read_declaration(metadata)
        expected = channels
        if alignment is not None:
            if (channels, scale) != (alignment["new_channels"], alignment["new_scale"]):
                raise ValueError("SenseNova editor VAE geometry disagrees with spatial alignment")
            expected = max(alignment["old_channels"], channels)
        if type(scale) is not int or scale < 1 or expected != resolved["gen_in_channels"]:
            raise ValueError("SenseNova editor requires matching VAE channels and positive spatial compression")
    return config, topology, resolved["gen_patch_size"] * scale


def export_topology_metadata(model, config_dict, *, base_path=None, export_metadata=None):
    metadata = model_topology_metadata(model)
    spec = getattr(model, "_residual_unet_spec", None)
    if spec is not None:
        validate_topology_identity(spec, config_dict)
        if export_metadata is not None:
            import json
            declaration = export_metadata.get("sensenova_config")
            if not isinstance(declaration, str) or not declaration.strip():
                raise ValueError("Rewired SenseNova export requires embedded native config")
            validate_topology_identity(spec, json.loads(declaration))
    if base_path is not None and preflight_checkpoint(base_path) != spec:
        raise ValueError("SenseNova refiner export and base residual topologies disagree")
    return metadata


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from types import SimpleNamespace
    from core.rewiring.training import resolve_training_topology
    from api.param_defaults import TRAINING_DEFAULTS
    model_topology_metadata(model)
    existing = getattr(model, "_residual_unet_spec", None)
    declared = config.get("dit_rewiring_spec", TRAINING_DEFAULTS["dit_rewiring_spec"])
    needs_identity = config.get("dit_rewiring_mode") == "configure" or existing is not None or declared is not None or (
        resume_spec is not NO_RESUME_SPEC and resume_spec is not None)
    native = {}
    if needs_identity:
        from .loader import _serializable_sensenova_config
        native = _serializable_sensenova_config(model.config)
    view = SimpleNamespace(config=native, _residual_unet_spec=existing)
    staged = dict(config)
    resolved = resolve_training_topology(view, staged, "sensenova", validate_topology_identity,
        lambda owner, spec: None, default_topology, resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=False)
    if resolved is not None:
        validate_owner_topology(resolved, model)
    if resolved is not None and int(config.get("batch_size", TRAINING_DEFAULTS["batch_size"])) != 1:
        raise ValueError("SenseNova residual training currently requires batch_size 1")
    if install:
        restore_topology(model, resolved)
        config["dit_rewiring_mode"] = config.get("dit_rewiring_mode", TRAINING_DEFAULTS["dit_rewiring_mode"])
        config["dit_rewiring_spec"] = resolved
    return resolved
