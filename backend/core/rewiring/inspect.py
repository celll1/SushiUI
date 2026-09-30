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
    raise ValueError("Residual inspection requires a supported native DiT artifact")


def inspect_model(model_path):
    path = Path(model_path)
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
                            "Krea2Transformer2DModel": "krea2"}.get(class_name)
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
                metadata = index.get("metadata") or {}
                if not isinstance(metadata, dict):
                    raise ValueError("Checkpoint metadata must be an object")
        else:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
        architecture = metadata.get("model_type")
        _native_adapter(architecture)
        if architecture == "qwen_image_21":
            if metadata.get("component") not in {"transformer", "training_bundle"}:
                raise ValueError("Residual inspection requires a Qwen-Image 2.1 model artifact")
            config = json.loads(metadata.get("config") or "{}")
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
    if type(config["num_layers"]) is not int or not 3 <= config["num_layers"] <= 4096:
        raise ValueError("Native stream block count must be an integer between 3 and 4096")
    streams = [{"name": "main", "block_count": config["num_layers"]}]
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    edges = DIT_REWIRING_REGION_DEFAULTS["edge_blocks"]
    initial = build_spec(architecture, streams, [dict(
        id="r0", stream=streams[0]["name"], parent=None,
        start_block=edges, end_block=config["num_layers"]-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS["down_factor"], **OPERATORS)], config)
    return dict(architecture=architecture, streams=streams,
                base_config_sha256=digest(config), block_layout_sha256=digest(streams),
                topology=topology, topology_sha256=None if topology is None else digest(topology),
                default_spec=adapter.default_topology(config), initial_spec=initial,
                latent_scale_factor=token_scale, pixel_alignment=alignment)
