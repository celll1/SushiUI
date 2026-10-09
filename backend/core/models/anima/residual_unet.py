"""Anima image-grid residual execution through native cross-attention blocks."""
from .config import ANIMA_DIT_CONFIG

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.training import NO_RESUME_SPEC
from core.rewiring.spec import validate_model_identity


def native_config(config):
    fields = ANIMA_DIT_CONFIG
    if isinstance(config, dict):
        return {key: config.get(key, value) for key, value in fields.items()}
    return {key: getattr(config, key, value) for key, value in fields.items()}


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    if resolved["patch_temporal"] != 1 or resolved["extra_per_block_abs_pos_emb"]:
        raise ValueError("Anima rewiring requires image patches without extra absolute positions")
    if resolved["pos_emb_cls"] != "rope3d":
        raise ValueError("Anima rewiring requires native 3D RoPE")
    spec = validate_model_identity(spec, "anima",
        [{"name": "main", "block_count": resolved["num_blocks"]}], resolved)
    if spec["method"] == "hybrid_unet_ssdit" and spec["stream_widths"] != {"main": resolved["model_channels"]}:
        raise ValueError("Hybrid Anima stream width differs from native blocks")
    return spec


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
        if hasattr(model, "rewiring_modules"):
            del model.rewiring_modules
    else:
        spec = validate_topology_identity(spec, model)
        if spec["method"] == "hybrid_unet_ssdit":
            from core.rewiring.hybrid_modules import HybridModules
            existing = getattr(model, "rewiring_modules", None)
            if existing is None:
                reference = next(model.parameters())
                model.add_module("rewiring_modules", HybridModules(spec).to(device=reference.device, dtype=reference.dtype))
            elif existing.declaration["module_layout_sha256"] != spec["module_layout_sha256"]:
                raise ValueError("Changing hybrid modules requires an explicit fresh model")
            else:
                existing.declaration = spec
        elif hasattr(model, "rewiring_modules"):
            del model.rewiring_modules
        model._residual_unet_spec = spec


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model))


def run_rewired_anima(model, hidden, embedding, text, attention, use_fp32,
                       rotary, adaln, fps, spec, *, partition_box=None):
    spec = validate_topology_identity(spec, model)
    if hidden.ndim != 5 or hidden.shape[1] != 1:
        raise ValueError("Anima rewiring requires a single 2D image frame")
    for name in ("_style_ctx", "_fbcache", "_tread_config", "_blockskip_config",
                 "_block_skip_config", "_repa_tap_depth"):
        if getattr(model, name, None) is not None:
            raise ValueError(f"Anima rewiring is incompatible with {name}")
    batch, _, height, width, channels = hidden.shape
    if len(model.blocks) != spec["streams"][0]["block_count"]:
        raise ValueError("Anima rewiring block count mismatch")
    offloader = getattr(model, "_block_offloader", None)

    def coarse_context(context, value, h, w, prefix):
        grid = value.reshape(batch, 1, h, w, channels)
        if partition_box is not None:
            from .partition import region_rotary
            return region_rotary(model.pos_embedder, grid, partition_box,
                factor=(height // h, width // w), fps=fps), h, w
        return model.pos_embedder(grid, fps=fps), h, w

    def execute(index, value, context):
        current_rotary, h, w = context
        if offloader is not None:
            offloader.wait_for_block(index)
        output = model.blocks[index](value.reshape(batch, 1, h, w, channels),
            embedding, text, attention, use_fp32, rope_emb_L_1_1_D=current_rotary,
            adaln_lora_B_T_3D=adaln, extra_per_block_pos_emb=None)
        if offloader is not None:
            offloader.submit_move_blocks_forward(index)
        return output.reshape(batch, h*w, channels)

    if spec["method"] == "hybrid_unet_ssdit":
        modules = getattr(model, "rewiring_modules", None)
        if modules is None or modules.declaration != spec:
            raise ValueError("Hybrid Anima modules are missing or disagree with the topology")
        output = modules.forward_stream(hidden.reshape(batch, height*width, channels),
            "main", height, width, 0, (rotary, height, width), execute, coarse_context,
            checkpoint_modules=any(getattr(block, "gradient_checkpoint_mode", "none") != "none" for block in model.blocks),
            checkpoint_offload=any(getattr(block, "gradient_checkpoint_mode", "none") in ("cpu_offload", "async_cpu_offload") for block in model.blocks),
            allow_encoder_replay=offloader is None)
    else:
        output = run_rewired_stream(hidden.reshape(batch, height*width, channels),
            spec, "main", height, width, 0, (rotary, height, width), execute, coarse_context)
    return output.reshape(batch, 1, height, width, channels)


def preflight_artifact(path, latent_channels=None, *, include_alignment=False):
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
    config = dict(ANIMA_DIT_CONFIG)
    declaration = metadata.get("transformer_config")
    if declaration:
        if not isinstance(declaration, str):
            raise ValueError("Anima native config metadata must be JSON text")
        declared = json.loads(declaration)
        if not isinstance(declared, dict):
            raise ValueError("Anima native config must be an object")
        config.update(declared)
    elif topology is not None:
        raise ValueError("Rewired Anima artifact requires native config metadata")
    from core.models.components.spatial_swap import read_declaration
    alignment=read_declaration(metadata)
    if alignment is not None:
        if not declaration:
            raise ValueError("Aligned Anima requires native config metadata")
        if latent_channels and int(latent_channels)!=alignment["new_channels"]:
            raise ValueError("Anima codec declaration disagrees with spatial alignment")
        width=max(alignment["old_channels"],alignment["new_channels"])
        if config["in_channels"]!=width or config["out_channels"]!=width:
            raise ValueError("Aligned Anima native config disagrees with internal latent width")
    elif latent_channels:
        config["in_channels"] = config["out_channels"] = int(latent_channels)
    if topology is not None:
        topology = validate_topology_identity(topology, config)
    result=(topology,native_config(config))
    return (*result,alignment) if include_alignment else result


def configure_training_topology(model, config, *, resuming=False, pending_adapter_resume=False,
                                resume_spec=NO_RESUME_SPEC, install=True):
    from core.rewiring.training import resolve_training_topology
    return resolve_training_topology(model, config, "anima", validate_topology_identity,
        restore_topology, default_topology, resuming=resuming,
        pending_adapter_resume=pending_adapter_resume, resume_spec=resume_spec, install=install)


def default_topology(config):
    from copy import deepcopy
    from api.param_defaults import DIT_REWIRING_PRESETS
    from core.rewiring.spec import OPERATORS, build_spec
    preset = deepcopy(DIT_REWIRING_PRESETS["anima"])
    resolved = native_config(config)
    if resolved["num_blocks"] != preset["block_count"]:
        return None
    return validate_topology_identity(build_spec("anima",
        [{"name": "main", "block_count": preset["block_count"]}],
        [dict(region, **OPERATORS) for region in preset["regions"]], resolved), config)
