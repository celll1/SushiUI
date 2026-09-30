"""Anima image-grid residual execution through native cross-attention blocks."""
import inspect

from core.rewiring.runtime import run_rewired_stream
from core.rewiring.spec import validate_model_identity


def native_config(config):
    from .anima_models import Anima
    fields = {name: parameter.default for name, parameter in
              inspect.signature(Anima.__init__).parameters.items()
              if name not in {"self", "attn_mode", "split_attn"}}
    if isinstance(config, dict):
        return {key: config.get(key, value) for key, value in fields.items()}
    return {key: getattr(config, key, value) for key, value in fields.items()}


def validate_topology_identity(spec, config):
    resolved = native_config(config)
    if resolved["patch_temporal"] != 1 or resolved["extra_per_block_abs_pos_emb"]:
        raise ValueError("Anima rewiring requires image patches without extra absolute positions")
    if resolved["pos_emb_cls"] != "rope3d":
        raise ValueError("Anima rewiring requires native 3D RoPE")
    return validate_model_identity(spec, "anima",
        [{"name": "main", "block_count": resolved["num_blocks"]}], resolved)


def restore_topology(model, spec):
    if spec is None:
        model.__dict__.pop("_residual_unet_spec", None)
    else:
        model._residual_unet_spec = validate_topology_identity(spec, model)


def model_topology_metadata(model):
    from core.rewiring.spec import checkpoint_metadata
    spec = getattr(model, "_residual_unet_spec", None)
    return {} if spec is None else checkpoint_metadata(validate_topology_identity(spec, model))


def run_rewired_anima(model, hidden, embedding, text, attention, use_fp32,
                       rotary, adaln, fps, spec):
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

    output = run_rewired_stream(hidden.reshape(batch, height*width, channels),
        spec, "main", height, width, 0, (rotary, height, width), execute, coarse_context)
    return output.reshape(batch, 1, height, width, channels)


def preflight_artifact(path, latent_channels=None):
    import json
    from pathlib import Path
    from safetensors import safe_open
    from core.rewiring.artifacts import read_artifact_topology
    from .anima_models import ANIMA_DIT_CONFIG
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
    if latent_channels:
        config["in_channels"] = config["out_channels"] = int(latent_channels)
    if topology is not None:
        topology = validate_topology_identity(topology, config)
    return topology, native_config(config)
