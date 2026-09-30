"""Architecture inspection for a topology editor; reads only config and headers."""
import json
from pathlib import Path

from safetensors import safe_open

from .artifacts import read_artifact_topology, read_directory_topology
from .spec import OPERATORS, build_spec, digest


def _native_adapter(architecture):
    if architecture == "qwen_image_21":
        from core.models.qwen_image_21 import residual_unet as adapter
        return adapter, 16, 32
    if architecture == "krea2":
        from core.models.krea2 import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "anima":
        from core.models.anima import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "zimage":
        from core.models import zimage_residual_unet as adapter
        return adapter, 16, 16
    raise ValueError("Residual inspection requires a supported native DiT artifact")


def inspect_model(model_path):
    path = Path(model_path)
    if path.is_dir():
        from core.models.anima.discovery import detect_anima_split_layout
        layout = detect_anima_split_layout(str(path))
        if layout is not None:
            path = Path(layout["dit"])
    if path.is_dir():
        if (path / "manifest.json").is_file():
            from core.models.qwen_image_21.artifact import load_manifest
            manifest = load_manifest(str(path))
            architecture = "qwen_image_21"
            config = manifest.transformer_config
            topology = read_artifact_topology(manifest.transformer)
        else:
            transformer_dir = path / "transformer" if (path / "transformer").is_dir() else path
            with (transformer_dir / "config.json").open(encoding="utf-8") as handle:
                config = json.load(handle)
            if not isinstance(config, dict):
                raise ValueError("Native model config must be an object")
            class_name = config.get("_class_name")
            if not isinstance(class_name, str):
                raise ValueError("Native model class must be a string")
            architecture = {"QwenImage21Transformer2DModel": "qwen_image_21",
                            "Krea2Transformer2DModel": "krea2",
                            "ZImageTransformer2DModel": "zimage"}.get(class_name)
            _native_adapter(architecture)
            topology = read_directory_topology(transformer_dir)
    else:
        if not path.exists() and Path(str(path)+".index.json").exists():
            path = Path(str(path)+".index.json")
        if str(path).endswith(".safetensors.index.json"):
            with path.open(encoding="utf-8") as handle:
                index = json.load(handle)
                if not isinstance(index, dict):
                    raise ValueError("Checkpoint index must be an object")
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("Checkpoint index requires a nonempty weight_map")
                keys = list(weight_map)
                metadata = index.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError("Checkpoint metadata must be an object")
        else:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                keys = list(handle.keys())
        architecture = metadata.get("model_type")
        from core.models.anima.discovery import is_anima_state_dict_keys
        if architecture is None and is_anima_state_dict_keys(keys):
            architecture = "anima"
        _native_adapter(architecture)
        if architecture == "qwen_image_21":
            if metadata.get("component") not in {"transformer", "training_bundle"}:
                raise ValueError("Residual inspection requires a Qwen-Image 2.1 model artifact")
            config = json.loads(metadata.get("config") or "{}")
        elif architecture == "anima":
            if not is_anima_state_dict_keys(keys):
                raise ValueError("Residual inspection requires an Anima DiT artifact, not adapter weights")
            from core.models.anima.residual_unet import preflight_artifact
            topology, config = preflight_artifact(path)
        elif architecture == "zimage":
            if not any(key.startswith(("all_x_embedder.", "transformer.all_x_embedder.", "x_embedder.")) for key in keys):
                raise ValueError("Residual inspection requires a Z-Image model artifact, not adapter weights")
            from core.models.zimage_residual_unet import preflight_artifact
            topology, config = preflight_artifact(path)
            if config is None:
                raise ValueError("Z-Image file inspection requires native config metadata")
        else:
            if not metadata.get("krea2_config"):
                raise ValueError("Residual inspection requires a Krea model config declaration")
            config = json.loads(metadata["krea2_config"])
        topology = read_artifact_topology(path)
    if not isinstance(config, dict):
        raise ValueError("Native model config must be an object")
    adapter, token_scale, alignment = _native_adapter(architecture)
    config = adapter.native_config(config)
    if topology is not None:
        topology = adapter.validate_topology_identity(topology, config)
    count_key = {"anima": "num_blocks", "zimage": "n_layers"}.get(architecture, "num_layers")
    block_count = config[count_key]
    if type(block_count) is not int or not 3 <= block_count <= 4096:
        raise ValueError("Native stream block count must be an integer between 3 and 4096")
    streams = [{"name": "main", "block_count": block_count}]
    if architecture == "anima":
        patch = config["patch_spatial"]
        if type(patch) is not int or patch < 1:
            raise ValueError("Anima spatial patch size must be a positive integer")
        token_scale = alignment = patch * 8
    if architecture == "zimage" and (2 not in config["all_patch_size"] or 1 not in config["all_f_patch_size"]):
        raise ValueError("Z-Image editor requires the native 2-1 generation patch bank")
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    edges = DIT_REWIRING_REGION_DEFAULTS["edge_blocks"]
    initial = build_spec(architecture, streams, [dict(
        id="r0", stream=streams[0]["name"], parent=None,
        start_block=edges, end_block=block_count-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS["down_factor"], **OPERATORS)], config)
    return dict(architecture=architecture, streams=streams,
                base_config_sha256=digest(config), block_layout_sha256=digest(streams),
                topology=topology, topology_sha256=None if topology is None else digest(topology),
                default_spec=adapter.default_topology(config), initial_spec=initial,
                latent_scale_factor=token_scale, pixel_alignment=alignment)
