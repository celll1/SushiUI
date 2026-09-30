"""Krea image-tail residual execution with native position and block semantics."""
import torch

from core.rewiring.training import NO_RESUME_SPEC as _NO_RESUME_SPEC

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity


def native_config(config):
    from .vendor.single_file import KREA2_DEFAULT_CONFIG
    return {key: config.get(key, value) for key, value in KREA2_DEFAULT_CONFIG.items()}


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    return validate_model_identity(spec, "krea2",
        [{"name": "main", "block_count": resolved["num_layers"]}], resolved)


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
    else:
        model._residual_unet_spec = validate_topology_identity(spec, dict(model.config))


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(
        validate_topology_identity(spec, dict(model.config)))


def preflight_artifact_topology(path):
    import json
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology

    spec = read_artifact_topology(path)
    if spec is None:
        return None
    source = Path(path)
    if not source.exists() and Path(str(source) + ".index.json").is_file():
        source = Path(str(source) + ".index.json")
    if str(source).endswith(".safetensors.index.json"):
        with source.open(encoding="utf-8") as handle:
            metadata = json.load(handle)["metadata"]
    else:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    if metadata.get("model_type") != "krea2" or not metadata.get("krea2_config"):
        raise ValueError("Rewired Krea artifact requires native config metadata")
    config = json.loads(metadata["krea2_config"])
    if not isinstance(config, dict):
        raise ValueError("Native Krea config must be an object")
    return validate_topology_identity(spec, config)


def run_rewired_krea(model, hidden, modulation, rotary, mask, position_ids, prefix, spec):
    spec = validate_topology_identity(spec, dict(model.config))
    if getattr(model, "_style_ctx", None) is not None:
        raise ValueError("Krea rewiring does not support style injection")
    if model._repa_tap_depth is not None:
        raise ValueError("Krea rewiring does not support REPA taps")
    if position_ids.shape != (hidden.shape[1], 3) or prefix < 0:
        raise ValueError("Krea rewiring requires native text-prefix/image-tail positions")
    image_ids = position_ids[prefix:]
    if image_ids.shape[0] == 0:
        raise ValueError("Krea rewiring requires a nonempty image grid")
    height = int(image_ids[:, 1].max().item()) + 1
    width = int(image_ids[:, 2].max().item()) + 1
    if height < 1 or width < 1 or height * width != image_ids.shape[0]:
        raise ValueError("Krea rewiring requires a rectangular native image grid")

    def positions(h, w):
        result = position_ids.new_zeros(prefix + h * w, 3)
        result[prefix:, 1] = torch.arange(h, device=result.device).repeat_interleave(w)
        result[prefix:, 2] = torch.arange(w, device=result.device).repeat(h)
        return result

    if not torch.equal(position_ids, positions(height, width)):
        raise ValueError("Krea rewiring requires native row-major origin-based positions")
    if mask is not None and (mask.shape != (hidden.shape[0], 1, 1, hidden.shape[1])
                             or not mask[..., prefix:].bool().all()):
        raise ValueError("Krea rewiring requires unmasked image tokens")

    def coarse_context(context, value, h, w, text_count):
        _, old_mask = context
        new_mask = None if old_mask is None else torch.cat((
            old_mask[..., :text_count], old_mask.new_ones(value.shape[0], 1, 1, h*w)), dim=-1)
        return model.rotary_emb(positions(h, w)), new_mask

    def execute(index, value, context):
        block = model.transformer_blocks[index]
        current_rotary, current_mask = context
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            return model._gradient_checkpointing_func(
                block, value, modulation, current_rotary, current_mask)
        return block(value, modulation, current_rotary, current_mask)

    return run_rewired_stream(hidden, spec, "main", height, width, prefix,
                              (rotary, mask), execute, coarse_context)




def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=_NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(
        model, config, "krea2", validate_topology_identity, restore_topology,
        resuming=resuming, pending_adapter_resume=pending_adapter_resume,
        resume_spec=resume_spec, install=install)
