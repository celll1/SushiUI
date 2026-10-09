"""Architecture inspection for a topology editor; reads only config and headers."""
import json
from pathlib import Path

from safetensors import safe_open

from .artifacts import read_artifact_topology, read_directory_topology
from .spec import OPERATORS, build_spec, digest


def _native_adapter(architecture):
    if architecture == "minimax_h3":
        from core.models.minimax_h3 import residual_unet as adapter
        return adapter, 32, 32
    if architecture == "ltx2":
        from core.models.ltx2 import residual_unet as adapter
        return adapter, 32, 32
    if architecture == "acestep":
        from core.models.acestep import residual_unet as adapter
        return adapter, 1, 1
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
    if architecture == "flux2":
        from core.models.flux2 import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "ideogram4":
        from core.models.ideogram4 import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "minit2i":
        from core.models.minit2i import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "lens":
        from core.models.lens import residual_unet as adapter
        return adapter, 16, 16
    if architecture == "sensenova":
        from core.models.sensenova import residual_unet as adapter
        return adapter, 32, 32
    raise ValueError("Residual inspection requires a supported native DiT artifact")


def inspect_model(model_path):
    path = Path(model_path)
    ming_marker = path / "transformer" / "config.json" if path.is_dir() else None
    if ming_marker is not None and ming_marker.is_file():
        with ming_marker.open(encoding="utf-8") as handle:
            if json.load(handle).get("_class_name") == "DiffusionTransformer":
                return _inspect_ming(path)
    manifest_marker = path / "manifest.json" if path.is_dir() else path
    if manifest_marker.name == "manifest.json" and manifest_marker.is_file():
        with manifest_marker.open(encoding="utf-8") as handle:
            if json.load(handle).get("model_type") == "ming_image_01":
                return _inspect_ming(path)
    from core.models.minimax_music3.residual_unet import find_native_directory
    if path.suffix.lower() == ".gguf" or find_native_directory(path) is not None:
        return _inspect_music3(path)
    if path.is_dir():
        from core.models.anima.discovery import detect_anima_split_layout
        layout = detect_anima_split_layout(str(path))
        if layout is not None:
            path = Path(layout["dit"])
    if path.is_dir():
        from core.models.acestep.loader import detect_acestep_layout
        layout = detect_acestep_layout(str(path))
        if layout is not None:
            path = Path(layout["dit"])
    if path.is_dir():
        from core.models.minimax_h3.loader import detect_minimax_h3_layout
        layout = detect_minimax_h3_layout(str(path))
        if layout is not None and layout.get("dit"):
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
                            "ZImageTransformer2DModel": "zimage",
                            "Flux2Transformer2DModel": "flux2",
                            "LensTransformer2DModel": "lens",
                            "MiniT2IMMJiTModel": "minit2i",
                            "Ideogram4Transformer2DModel": "ideogram4",
                            "LTX2VideoTransformer3DModel": "ltx2"}.get(class_name)
            _native_adapter(architecture)
            topology = read_directory_topology(transformer_dir)
            if architecture == "ltx2":
                from core.models.ltx2.loader import _transformer_shards
                from core.models.ltx2.residual_unet import preflight_checkpoint, native_config
                _, _, source = _transformer_shards(str(transformer_dir))
                if source is None:
                    raise ValueError("LTX-2 inspection requires native AV transformer headers")
                declared, saved_config = preflight_checkpoint(source, directory_config=config)
                if native_config(saved_config) != native_config(config) or declared != topology:
                    raise ValueError("LTX-2 directory and checkpoint native configs disagree")
            if architecture == "ideogram4":
                from core.models.ideogram4.residual_unet import inspect_artifact
                config,topology = inspect_artifact(path)
            if architecture in ("lens", "minit2i"):
                native_keys = []
                for source in transformer_dir.glob("*.safetensors"):
                    with safe_open(str(source), framework="pt", device="cpu") as handle:
                        native_keys.extend(handle.keys())
                if architecture == "minit2i" and "model.net.img_embedder.proj1.weight" not in native_keys:
                    raise ValueError("MiniT2I directory inspection requires native model weights")
                if architecture == "lens" and "img_in.weight" not in native_keys:
                    raise ValueError("Lens directory inspection requires native model weights")
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
        from core.models.yue2.loader import keys_look_yue2
        if architecture == "yue2" or keys_look_yue2(keys):
            return _inspect_yue2(path)
        if architecture == "minimax_music3" or "diffusion_transformer.transformer.project_in.weight" in keys:
            return _inspect_music3(path)
        if architecture is None and {"video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight"}.issubset(keys):
            architecture = "minimax_h3"
        if architecture is None and any(key.startswith(("transformer_blocks.", "single_transformer_blocks.",
                "model.diffusion_model.transformer_blocks.", "model.diffusion_model.single_transformer_blocks.")) for key in keys):
            from core.models.flux2.single_file import count_flux2_blocks
            if all(count_flux2_blocks(key.removeprefix("model.diffusion_model.") for key in keys)):
                architecture = "flux2"
        from core.models.anima.discovery import is_anima_state_dict_keys
        if architecture is None and is_anima_state_dict_keys(keys):
            architecture = "anima"
        if architecture is None and {"decoder.proj_in.1.weight", "decoder.proj_out.1.weight", "null_condition_emb"}.issubset(
                key.removeprefix("net.") for key in keys):
            architecture = "acestep"
        _native_adapter(architecture)
        if architecture == "minimax_h3":
            from core.models.minimax_h3.residual_unet import inspect_artifact
            config, topology = inspect_artifact(path)
        elif architecture == "ltx2":
            from core.models.ltx2.residual_unet import preflight_checkpoint
            topology, config = preflight_checkpoint(path)
            if config is None:
                raise ValueError("LTX-2 file inspection requires native transformer_config metadata")
        elif architecture == "acestep":
            from core.models.acestep.residual_unet import inspect_artifact
            config, topology = inspect_artifact(path)
        elif architecture == "qwen_image_21":
            if metadata.get("component") not in {"transformer", "training_bundle"}:
                raise ValueError("Residual inspection requires a Qwen-Image 2.1 model artifact")
            config = json.loads(metadata.get("config") or "{}")
        elif architecture == "sensenova":
            from core.models.sensenova.residual_unet import inspect_artifact
            config, topology, sensenova_stride = inspect_artifact(path)
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
        elif architecture == "flux2":
            native_keys = [key.removeprefix("model.diffusion_model.") for key in keys]
            if "x_embedder.weight" not in native_keys:
                raise ValueError("Residual inspection requires a Flux2 model artifact, not adapter weights")
            from core.models.flux2.residual_unet import preflight_checkpoint
            from core.models.flux2.single_file import count_flux2_blocks
            topology, config = preflight_checkpoint(path)
            if config is None:
                raise ValueError("Flux2 file inspection requires native config metadata")
            if count_flux2_blocks(native_keys) != (config.get("num_layers"), config.get("num_single_layers")):
                raise ValueError("Flux2 native config and checkpoint block counts disagree")
        elif architecture == "ideogram4":
            from core.models.ideogram4.residual_unet import inspect_artifact
            config,topology = inspect_artifact(path)
        elif architecture == "minit2i":
            native_keys = [key.removeprefix("transformer.") for key in keys]
            if "model.net.img_embedder.proj1.weight" not in native_keys:
                raise ValueError("Residual inspection requires a MiniT2I model artifact, not adapter weights")
            declaration = metadata.get("mmjit_config")
            if not isinstance(declaration, str):
                raise ValueError("MiniT2I file inspection requires native mmjit_config metadata")
            config = json.loads(declaration)
            if not isinstance(config, dict) or "depth_double" not in config:
                raise ValueError("MiniT2I inspection requires an explicit native block count")
        elif architecture == "lens":
            native_keys = [key.removeprefix("net.") for key in keys]
            if "img_in.weight" not in native_keys:
                raise ValueError("Residual inspection requires a Lens model artifact, not adapter weights")
            from core.models.lens.residual_unet import preflight_checkpoint
            topology, config = preflight_checkpoint(path)
            if config is None:
                raise ValueError("Lens file inspection requires native config metadata")
        else:
            if not metadata.get("krea2_config"):
                raise ValueError("Residual inspection requires a Krea model config declaration")
            config = json.loads(metadata["krea2_config"])
        topology = read_artifact_topology(path)
    if not isinstance(config, dict):
        raise ValueError("Native model config must be an object")
    adapter, token_scale, alignment = _native_adapter(architecture)
    config = adapter.native_config(config)
    if architecture == "lens":
        import re
        blocks = {int(match.group(1)) for key in native_keys
                  if (match := re.match(r"^transformer_blocks\.(\d+)\.", key))}
        if len(blocks) != config["num_layers"] or blocks != set(range(len(blocks))):
            raise ValueError("Lens native config and checkpoint block counts disagree")
    if architecture == "minit2i":
        import re
        blocks = {int(match.group(1)) for key in native_keys
                  if (match := re.match(r"^model\.net\.double_blocks\.(\d+)\.", key))}
        if len(blocks) != config["depth_double"] or blocks != set(range(len(blocks))):
            raise ValueError("MiniT2I native config and checkpoint block counts disagree")
        from core.models.minit2i.minit2i_vae import is_latent_vae, VAE_SCALE_FACTOR
        if config["vae_type"] != "none" and not is_latent_vae(config["vae_type"]):
            raise ValueError("MiniT2I editor requires a known native VAE scale")
        token_scale = alignment = config["patch_size"] * (1 if config["vae_type"] == "none" else VAE_SCALE_FACTOR)
    if topology is not None:
        topology = adapter.validate_topology_identity(topology, config)
    count_key = {"anima": "num_blocks", "zimage": "n_layers", "minit2i": "depth_double", "acestep": "num_hidden_layers"}.get(architecture, "num_layers")
    block_count = config["llm_config"]["num_hidden_layers"] if architecture == "sensenova" else config[count_key]
    if architecture == "sensenova":
        token_scale = alignment = sensenova_stride
    if type(block_count) is not int or not 3 <= block_count <= 4096:
        raise ValueError("Native stream block count must be an integer between 3 and 4096")
    streams = [{"name": "main", "block_count": block_count}]
    if architecture == "flux2":
        single_count = config["num_single_layers"]
        if type(single_count) is not int or not 3 <= single_count <= 4096:
            raise ValueError("Native single stream block count must be an integer between 3 and 4096")
        streams = [{"name": "double", "block_count": block_count}, {"name": "single", "block_count": single_count}]
    if architecture == "anima":
        patch = config["patch_spatial"]
        if type(patch) is not int or patch < 1:
            raise ValueError("Anima spatial patch size must be a positive integer")
        token_scale = alignment = patch * 8
    if architecture == "minimax_h3":
        geometry = adapter.video_geometry(config)
        import math
        token_scale = geometry["height_scale_factor"]
        alignment = math.lcm(geometry["height_scale_factor"], geometry["width_scale_factor"])
    if architecture == "ltx2":
        token_scale = config["vae_scale_factors"][1]
        import math
        alignment = math.lcm(*config["vae_scale_factors"][1:])
    if architecture == "lens" and config["patch_size"] != 2:
        raise ValueError("Lens editor requires native packed 2x2 patches")
    if architecture == "zimage" and (2 not in config["all_patch_size"] or 1 not in config["all_f_patch_size"]):
        raise ValueError("Z-Image editor requires the native 2-1 generation patch bank")
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    edges = DIT_REWIRING_REGION_DEFAULTS["edge_blocks"]
    initial = build_spec(architecture, streams, [dict(
        id="r0", stream=streams[0]["name"], parent=None,
        start_block=edges, end_block=block_count-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS["temporal_down_factor" if architecture == "acestep" else "down_factor"], **OPERATORS)], config)
    from .parameters import parameter_counts
    info = dict(parameter_counts=parameter_counts(architecture, config),
                architecture=architecture, streams=streams,
                base_config_sha256=digest(config), block_layout_sha256=digest(streams),
                topology=topology, topology_sha256=None if topology is None else digest(topology),
                default_spec=adapter.default_topology(config), initial_spec=initial,
                latent_scale_factor=token_scale, pixel_alignment=alignment)
    if architecture in ("ltx2", "minimax_h3"):
        info["video_geometry"] = adapter.video_geometry(config)
    if architecture == "acestep":
        info["temporal_geometry"] = adapter.temporal_geometry(config)
    if architecture in ("anima", "qwen_image_21"):
        from .hybrid_spec import default_hybrid_spec
        width = config["model_channels"] if architecture == "anima" else config["num_attention_heads"] * config["attention_head_dim"]
        info["hybrid_default_spec"] = default_hybrid_spec(info["default_spec"] or initial, {"main": width})
    return info


def _inspect_music3(path):
    from core.models.minimax_music3 import residual_unet as adapter
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    config,topology,geometry = adapter.inspect_artifact(path)
    count = config['num_layers']
    if not 3 <= count <= 4096:
        raise ValueError('Native stream block count must be between 3 and 4096')
    streams = [{'name':'main','block_count':count}]
    edges = DIT_REWIRING_REGION_DEFAULTS['edge_blocks']
    initial = build_spec('minimax_music3', streams, [dict(id='r0',stream='main',parent=None,
        start_block=edges,end_block=count-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS['temporal_down_factor'],**OPERATORS)],config)
    return dict(architecture='minimax_music3',streams=streams,base_config_sha256=digest(config),
        block_layout_sha256=digest(streams),topology=topology,
        topology_sha256=None if topology is None else digest(topology),
        default_spec=adapter.default_topology(config),initial_spec=initial,
        latent_scale_factor=1,pixel_alignment=1,temporal_geometry=geometry)


def _inspect_yue2(path):
    from core.models.yue2 import residual_unet as adapter
    from core.models.yue2.loader import preflight_yue2
    from core.models.yue2.vendor.modeling_yue2 import YuE2Config
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    inspected = preflight_yue2(path, load_tokenizer=False)
    config = YuE2Config(**inspected["model_config"])
    resolved = adapter.native_config(config)
    topology = inspected["residual_topology"]
    streams = [{"name": "nar", "block_count": config.num_hidden_layers}]
    edges = DIT_REWIRING_REGION_DEFAULTS["edge_blocks"]
    initial = build_spec("yue2", streams, [dict(id="r0", stream="nar", parent=None,
        start_block=edges, end_block=config.num_hidden_layers-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS["temporal_down_factor"], **OPERATORS)], resolved)
    return dict(architecture="yue2", streams=streams, base_config_sha256=digest(resolved),
        block_layout_sha256=digest(streams), topology=topology,
        topology_sha256=None if topology is None else digest(topology),
        default_spec=adapter.default_topology(config), initial_spec=initial,
        latent_scale_factor=1, pixel_alignment=1, temporal_geometry=adapter.temporal_geometry(config))



def _inspect_ming(path):
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.models.ming_image_01 import residual_unet as adapter
    from api.param_defaults import DIT_REWIRING_REGION_DEFAULTS
    from .parameters import parameter_counts
    artifact = inspect_artifact(path)
    config = adapter.native_config(artifact.configs["transformer"])
    topology = read_directory_topology(artifact.components["transformer"])
    if topology is not None:
        topology = adapter.validate_topology_identity(topology, config)
    count = config["n_layers"]
    streams = [{"name": "main", "block_count": count}]
    edges = DIT_REWIRING_REGION_DEFAULTS["edge_blocks"]
    initial = build_spec("ming_image_01", streams, [dict(id="r0", stream="main", parent=None,
        start_block=edges, end_block=count-edges,
        down_factor=DIT_REWIRING_REGION_DEFAULTS["down_factor"], **OPERATORS)], config)
    from .hybrid_spec import default_hybrid_spec
    return dict(architecture="ming_image_01", streams=streams, base_config_sha256=digest(config),
        block_layout_sha256=digest(streams), topology=topology,
        topology_sha256=None if topology is None else digest(topology),
        default_spec=adapter.default_topology(config), initial_spec=initial,
        hybrid_default_spec=default_hybrid_spec(adapter.default_topology(config) or initial, {"main": config["dim"]}),
        latent_scale_factor=16, pixel_alignment=16, parameter_counts=parameter_counts("ming_image_01", config),
        ming_partition_global=None if artifact.global_summary is None else
            {**{key:artifact.global_summary[key] for key in ('metadata','parameters','storage_bytes')},'loaded_in_model':False})
