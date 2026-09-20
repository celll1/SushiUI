"""Atomic bootstrap builder for SenseNova SDXL Chimera artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from core.models.common.single_file_format import dedup_tensors, save_single_file_state
from core.models.common.vae_source import ResolvedVAE
from core.models.sensenova.loader import is_sensenova_state_dict_keys

from .artifact import (
    CONFIG_NAME,
    FORMAT_VERSION,
    V3_FORMAT_VERSION,
    V4_FORMAT_VERSION,
    MANIFEST_NAME,
    MODEL_TYPE,
    WEIGHTS_BASENAME,
    checkpoint_content_hash,
    checkpoint_headers,
    config_hash,
    infer_bridge_config,
    manifest_template,
    prefixed_state,
    runtime_config,
    prediction_contract,
)
from .conditioning_bridge import ConditioningBridge
from .flow import (
    FLOW_V3_ANGULAR_SCHEDULE,
    FLOW_V3_PREDICTION,
    FLOW_V4_PREDICTION,
)
from .unet import build_donor_equal_unet, install_polar_radial_head


def _source_config(source: str) -> dict[str, Any]:
    shapes, metadata = checkpoint_headers(source)
    if not is_sensenova_state_dict_keys(shapes):
        raise ValueError(f"understanding source is not a supported SenseNova checkpoint: {source}")
    embedded = metadata.get("sensenova_config")
    if embedded:
        raw = json.loads(embedded)
    else:
        config_path = Path(source).resolve().parent / CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(
                f"SenseNova source has no sensenova_config metadata or sibling config.json: {source}"
            )
        with config_path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"SenseNova config must be a JSON object: {source}")
    return raw


def _vae_facts(vae: ResolvedVAE) -> dict[str, Any]:
    facts = vae.facts()
    facts.update({"embedded": True, "config_hash": config_hash(vae.config)})
    return facts


def build_chimera_artifact_from_components(
    output_directory: str,
    *,
    understanding_source: str,
    sdxl_source: str,
    donor_unet: torch.nn.Module,
    vae: ResolvedVAE,
    unet_initialization: str = "scratch",
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build into an empty directory; callers own atomic target publication."""
    root = Path(output_directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Chimera target must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if vae.state_dict is None:
        raise ValueError("builder requires a VAE resolved with load_weights=True")
    if vae.latent_channels != 4 or vae.scale_factor != 8 or vae.ndim != 4:
        raise ValueError(
            "Chimera requires an SDXL-compatible 4-channel, 8x, 2-D VAE; "
            f"got {vae.latent_channels}ch/{vae.scale_factor}x/ndim={vae.ndim}"
        )

    und_config = _source_config(understanding_source)
    bridge_config = infer_bridge_config(und_config)
    bridge = ConditioningBridge(bridge_config)
    declared_prediction = dict(prediction or prediction_contract())
    is_polar = declared_prediction.get("type") in {
        FLOW_V3_PREDICTION,
        FLOW_V4_PREDICTION,
    }
    unet, _report = build_donor_equal_unet(
        donor_unet,
        initialization=unet_initialization,
        seed=initialization_seed,
        output_initialization="default" if is_polar else "zero",
    )
    radial_head_config = None
    if is_polar:
        radial_head = install_polar_radial_head(unet)
        radial_head_config = {
            "version": radial_head.VERSION,
            "source": "conditioned_mid_block",
            "mid_channels": radial_head.mid_channels,
        }
    unet_config = dict(unet.config)
    artifact_format = (
        V4_FORMAT_VERSION
        if declared_prediction.get("type") == FLOW_V4_PREDICTION
        else V3_FORMAT_VERSION
        if is_polar
        else FORMAT_VERSION
    )
    config = runtime_config(
        unet_config=unet_config,
        bridge_config=bridge_config,
        vae_config=vae.config,
        format_version=artifact_format,
        polar_radial_head=radial_head_config,
    )
    manifest = manifest_template(
        understanding_source=understanding_source,
        understanding_hash=checkpoint_content_hash(understanding_source),
        understanding_config_hash=config_hash(und_config),
        sdxl_source=sdxl_source,
        sdxl_hash=checkpoint_content_hash(sdxl_source),
        initialization=unet_initialization,
        unet_config=unet_config,
        parameter_count=sum(parameter.numel() for parameter in unet.parameters()),
        bridge_config=bridge_config,
        vae_facts=_vae_facts(vae),
        prediction=declared_prediction,
    )

    tensors, dropped = dedup_tensors((
        *prefixed_state(bridge, "condition_bridge."),
        *prefixed_state(unet, "unet."),
        *((f"vae.{name}", tensor) for name, tensor in vae.state_dict.items()),
    ))
    metadata = {
        "model_type": MODEL_TYPE,
        "format": "pt",
        "format_version": str(artifact_format),
        "tied_weights_dropped": json.dumps(dropped),
    }
    written = save_single_file_state(
        tensors, metadata, str(root / WEIGHTS_BASENAME), max_shard_bytes=max_shard_bytes
    )
    with (root / CONFIG_NAME).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    # The manifest is the completion marker and is deliberately written last.
    with (root / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return {"directory": str(root), "weights": written, "manifest": manifest, "config": config}


def initialize_chimera_atomically(
    model_root: str,
    output_name: str,
    *,
    build_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be one relative directory name")
    target = (root / output_name).resolve()
    if target.parent != root:
        raise ValueError("Chimera target escapes the configured model root")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"Chimera target already exists and is not empty: {target}")
        target.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.building-", dir=str(root)))
    try:
        result = build_chimera_artifact_from_components(str(temporary), **dict(build_kwargs))
        from .loader import load_chimera_artifact

        load_chimera_artifact(str(temporary), load_understanding=False)
        os.replace(temporary, target)
        result["directory"] = str(target)
        result["weights"] = str(target / Path(result["weights"]).name)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def initialize_chimera_from_paths(
    model_root: str,
    output_name: str,
    *,
    understanding_source: str,
    sdxl_source: str,
    unet_initialization: str = "scratch",
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
    torch_dtype: torch.dtype = torch.float32,
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load production source paths and publish one complete artifact atomically."""
    from .source import load_sdxl_donor_components

    donor_unet, vae = load_sdxl_donor_components(sdxl_source, torch_dtype=torch_dtype)
    return initialize_chimera_atomically(
        model_root,
        output_name,
        build_kwargs={
            "understanding_source": understanding_source,
            "sdxl_source": sdxl_source,
            "donor_unet": donor_unet,
            "vae": vae,
            "unet_initialization": unet_initialization,
            "initialization_seed": initialization_seed,
            "max_shard_bytes": max_shard_bytes,
            "prediction": prediction,
        },
    )


def build_chimera_v3_warmstart(
    output_directory: str,
    *,
    source_directory: str,
    prediction: Mapping[str, Any],
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    """Convert a v1/v2 Chimera checkpoint into a step-zero v3 artifact."""
    from .loader import load_chimera_artifact

    root = Path(output_directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Chimera target must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = load_chimera_artifact(source_directory, load_understanding=False)
    if source["prediction"]["type"] == FLOW_V3_PREDICTION:
        raise ValueError("Chimera v3 warm-start source must be a v1/v2 artifact")
    declared = prediction_contract(
        str(prediction.get("type") or ""),
        latent_mean=prediction.get("latent_mean"),
        latent_centered_second_moment=prediction.get(
            "latent_centered_second_moment"
        ),
        angular_schedule=prediction.get(
            "angular_schedule", FLOW_V3_ANGULAR_SCHEDULE
        ),
        angular_endpoint_slope=prediction.get("angular_endpoint_slope", 2.0),
        radius_floor=prediction.get("radius_floor", 1e-8),
        angular_singularity_threshold=prediction.get(
            "angular_singularity_threshold", 1e-6
        ),
        angular_step_limit=prediction.get("angular_step_limit"),
    )
    if declared["type"] != FLOW_V3_PREDICTION:
        raise ValueError("Chimera warm-start output must use polar_tangent_flow")

    unet = source["unet"]
    with torch.random.fork_rng(
        devices=[] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    ):
        torch.manual_seed(int(initialization_seed))
        unet.conv_out.reset_parameters()
        radial_head = install_polar_radial_head(unet)
    bridge = source["condition_bridge"]
    vae_state = source["frozen_vae_state"]
    runtime = json.loads(json.dumps(source["config"]))
    runtime["format_version"] = V3_FORMAT_VERSION
    runtime["polar_radial_head"] = {
        "version": radial_head.VERSION,
        "source": "conditioned_mid_block",
        "mid_channels": radial_head.mid_channels,
    }
    manifest = json.loads(json.dumps(source["manifest"]))
    source_training = dict(manifest.pop("training", {}) or {})
    manifest["format_version"] = V3_FORMAT_VERSION
    manifest["prediction"] = declared
    manifest["unet"]["initialization"] = "chimera_v2_warmstart"
    manifest["unet"]["parameter_count"] = sum(
        parameter.numel() for parameter in unet.parameters()
    )
    manifest["unet"]["warmstart"] = {
        "source": str(Path(source_directory).resolve()),
        "source_format_version": int(source["manifest"]["format_version"]),
        "source_prediction_type": source["prediction"]["type"],
        "source_step": source_training.get("step"),
        "reset": ["conv_out", "chimera_polar_radial_head"],
    }

    tensors, dropped = dedup_tensors((
        *prefixed_state(bridge, "condition_bridge."),
        *prefixed_state(unet, "unet."),
        *((f"vae.{name}", tensor.detach().cpu().contiguous())
          for name, tensor in vae_state.items()),
    ))
    metadata = {
        "model_type": MODEL_TYPE,
        "format": "pt",
        "format_version": str(V3_FORMAT_VERSION),
        "tied_weights_dropped": json.dumps(dropped),
    }
    written = save_single_file_state(
        tensors, metadata, str(root / WEIGHTS_BASENAME),
        max_shard_bytes=max_shard_bytes,
    )
    with (root / CONFIG_NAME).open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, ensure_ascii=False)
    with (root / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return {
        "directory": str(root),
        "weights": written,
        "manifest": manifest,
        "config": runtime,
    }


def initialize_chimera_v3_warmstart_atomically(
    model_root: str,
    output_name: str,
    *,
    source_directory: str,
    prediction: Mapping[str, Any],
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be one relative directory name")
    target = (root / output_name).resolve()
    if target.parent != root:
        raise ValueError("Chimera target escapes the configured model root")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"Chimera target already exists and is not empty: {target}")
        target.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.building-", dir=str(root)))
    try:
        result = build_chimera_v3_warmstart(
            str(temporary),
            source_directory=source_directory,
            prediction=prediction,
            initialization_seed=initialization_seed,
            max_shard_bytes=max_shard_bytes,
        )
        from .loader import load_chimera_artifact

        load_chimera_artifact(str(temporary), load_understanding=False)
        os.replace(temporary, target)
        result["directory"] = str(target)
        result["weights"] = str(target / Path(result["weights"]).name)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_chimera_v4_warmstart(
    output_directory: str,
    *,
    source_directory: str,
    prediction: Mapping[str, Any],
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    """Convert a v3 checkpoint into a step-zero v4 artifact."""
    from .loader import load_chimera_artifact

    root = Path(output_directory).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Chimera target must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source = load_chimera_artifact(source_directory, load_understanding=False)
    if source["prediction"]["type"] != FLOW_V3_PREDICTION:
        raise ValueError("Chimera v4 warm-start source must be a v3 polar artifact")
    declared = prediction_contract(
        str(prediction.get("type") or ""),
        latent_mean=prediction.get("latent_mean"),
        latent_centered_second_moment=prediction.get(
            "latent_centered_second_moment"
        ),
        radius_floor=prediction.get("radius_floor", 1e-8),
        angular_singularity_threshold=prediction.get(
            "angular_singularity_threshold", 1e-6
        ),
        angular_step_limit=prediction.get("angular_step_limit"),
    )
    if declared["type"] != FLOW_V4_PREDICTION:
        raise ValueError(
            "Chimera v4 warm-start output must use destruction_coordinate_polar_flow"
        )

    unet = source["unet"]
    with torch.random.fork_rng(
        devices=[] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))
    ):
        torch.manual_seed(int(initialization_seed))
        unet.conv_out.reset_parameters()
    bridge = source["condition_bridge"]
    vae_state = source["frozen_vae_state"]
    runtime = json.loads(json.dumps(source["config"]))
    runtime["format_version"] = V4_FORMAT_VERSION
    manifest = json.loads(json.dumps(source["manifest"]))
    source_training = dict(manifest.pop("training", {}) or {})
    manifest["format_version"] = V4_FORMAT_VERSION
    manifest["prediction"] = declared
    manifest["unet"]["initialization"] = "chimera_v3_warmstart"
    manifest["unet"]["parameter_count"] = sum(
        parameter.numel() for parameter in unet.parameters()
    )
    manifest["unet"]["warmstart"] = {
        "source": str(Path(source_directory).resolve()),
        "source_format_version": int(source["manifest"]["format_version"]),
        "source_prediction_type": source["prediction"]["type"],
        "source_step": source_training.get("step"),
        "inherited": ["unet_trunk", "time_embedding", "condition_bridge", "radial_head"],
        "reset": ["conv_out"],
    }
    source_repa = Path(f"{Path(source_directory).resolve()}.repa.safetensors")
    if source_repa.is_file():
        shutil.copy2(source_repa, root / "repa_projector.safetensors")
        manifest["unet"]["warmstart"]["inherited"].append("repa_projector")

    tensors, dropped = dedup_tensors((
        *prefixed_state(bridge, "condition_bridge."),
        *prefixed_state(unet, "unet."),
        *((f"vae.{name}", tensor.detach().cpu().contiguous())
          for name, tensor in vae_state.items()),
    ))
    metadata = {
        "model_type": MODEL_TYPE,
        "format": "pt",
        "format_version": str(V4_FORMAT_VERSION),
        "tied_weights_dropped": json.dumps(dropped),
    }
    written = save_single_file_state(
        tensors,
        metadata,
        str(root / WEIGHTS_BASENAME),
        max_shard_bytes=max_shard_bytes,
    )
    with (root / CONFIG_NAME).open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2, ensure_ascii=False)
    with (root / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return {
        "directory": str(root),
        "weights": written,
        "manifest": manifest,
        "config": runtime,
    }


def initialize_chimera_v4_warmstart_atomically(
    model_root: str,
    output_name: str,
    *,
    source_directory: str,
    prediction: Mapping[str, Any],
    initialization_seed: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    if not output_name or Path(output_name).name != output_name or output_name in {".", ".."}:
        raise ValueError("output_name must be one relative directory name")
    target = (root / output_name).resolve()
    if target.parent != root:
        raise ValueError("Chimera target escapes the configured model root")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise FileExistsError(f"Chimera target already exists and is not empty: {target}")
        target.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.building-", dir=str(root)))
    try:
        result = build_chimera_v4_warmstart(
            str(temporary),
            source_directory=source_directory,
            prediction=prediction,
            initialization_seed=initialization_seed,
            max_shard_bytes=max_shard_bytes,
        )
        from .loader import load_chimera_artifact

        load_chimera_artifact(str(temporary), load_understanding=False)
        os.replace(temporary, target)
        result["directory"] = str(target)
        result["weights"] = str(target / Path(result["weights"]).name)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
