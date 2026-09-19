"""On-disk contract and CPU preflight for SenseNova SDXL Chimera."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors import safe_open

from core.models.common.single_file_format import is_index_path
from core.models.common.vae_source import content_hash_for_state_dict
from core.models.sensenova.loader import is_sensenova_state_dict_keys

from .conditioning_bridge import ChimeraBridgeConfig
from .flow import FLOW_V1_PREDICTION, FLOW_V2_PATH, FLOW_V2_PREDICTION
from .positional import POSITION_LAYOUT_VERSION, SPATIAL_UNIT_PIXELS

MODEL_TYPE = "sensenova_sdxl_chimera"
LEGACY_FORMAT_VERSION = 2
FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (LEGACY_FORMAT_VERSION, FORMAT_VERSION)
MANIFEST_NAME = "chimera.json"
CONFIG_NAME = "config.json"
WEIGHTS_BASENAME = "model.safetensors"


class ChimeraArtifactError(ValueError):
    """A Chimera artifact or one of its pinned sources is invalid."""


def prediction_contract(
    prediction_type: str = FLOW_V1_PREDICTION,
    *,
    latent_mean: Iterable[float] | None = None,
    latent_centered_second_moment: float | None = None,
) -> dict[str, Any]:
    """Build and validate one explicit Chimera prediction contract."""
    kind = str(prediction_type).strip().lower()
    if kind == FLOW_V1_PREDICTION:
        return {
            "type": FLOW_V1_PREDICTION,
            "time_direction": "zero_noise_to_one_clean",
        }
    if kind != FLOW_V2_PREDICTION:
        raise ChimeraArtifactError(f"unsupported Chimera prediction type: {prediction_type!r}")
    mean_source = () if latent_mean is None else latent_mean
    mean = [float(value) for value in mean_source]
    q = float(
        0.0 if latent_centered_second_moment is None
        else latent_centered_second_moment
    )
    if len(mean) != 4 or not all(math.isfinite(value) for value in mean):
        raise ChimeraArtifactError("Chimera v2 latent_mean must contain four finite values")
    if not math.isfinite(q) or q <= 0.0:
        raise ChimeraArtifactError(
            "Chimera v2 latent_centered_second_moment must be finite and > 0"
        )
    return {
        "type": FLOW_V2_PREDICTION,
        "version": 1,
        "time_direction": "zero_noise_to_one_clean",
        "path": FLOW_V2_PATH,
        "latent_mean": mean,
        "latent_centered_second_moment": q,
    }


def validated_prediction_contract(
    manifest: Mapping[str, Any], *, artifact_format: int | None = None
) -> dict[str, Any]:
    version = int(artifact_format or manifest.get("format_version", 0))
    declared = dict(manifest.get("prediction") or {})
    if version == LEGACY_FORMAT_VERSION:
        if declared.get("type") != FLOW_V1_PREDICTION:
            raise ChimeraArtifactError("Chimera format v2 requires flow_velocity prediction")
        return prediction_contract()
    if version != FORMAT_VERSION:
        raise ChimeraArtifactError(f"unsupported Chimera format version: {version}")
    return prediction_contract(
        str(declared.get("type") or ""),
        latent_mean=declared.get("latent_mean"),
        latent_centered_second_moment=declared.get(
            "latent_centered_second_moment"
        ),
    )


def migrate_manifest_prediction(
    manifest: Mapping[str, Any],
    *,
    latent_mean: Iterable[float],
    latent_centered_second_moment: float,
) -> dict[str, Any]:
    """Return a format-v3 manifest using v2 residual semantics."""
    migrated = json.loads(json.dumps(dict(manifest)))
    migrated["format_version"] = FORMAT_VERSION
    migrated["prediction"] = prediction_contract(
        FLOW_V2_PREDICTION,
        latent_mean=latent_mean,
        latent_centered_second_moment=latent_centered_second_moment,
    )
    return migrated


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _checkpoint_files(path: str | os.PathLike[str]) -> tuple[Path, ...]:
    entry = Path(path).resolve()
    if entry.is_dir():
        files = tuple(sorted(item for item in entry.rglob("*") if item.is_file()))
        if not files:
            raise ChimeraArtifactError(f"checkpoint directory is empty: {entry}")
        return files
    if not entry.is_file():
        raise FileNotFoundError(f"checkpoint not found: {entry}")
    if is_index_path(str(entry)):
        with entry.open(encoding="utf-8") as handle:
            index = json.load(handle)
        shards = sorted(set((index.get("weight_map") or {}).values()))
        if not shards:
            raise ChimeraArtifactError(f"shard index has no weight_map: {entry}")
        files = (entry, *(entry.parent / shard for shard in shards))
        missing = [str(item) for item in files if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"checkpoint is missing shard(s): {missing[:5]}")
        return tuple(files)
    return (entry,)


def checkpoint_content_hash(path: str | os.PathLike[str]) -> str:
    """Hash checkpoint bytes, including an index and every referenced shard."""
    root = Path(path).resolve()
    digest = hashlib.sha256()
    for item in _checkpoint_files(path):
        if root.is_dir():
            relative = item.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def checkpoint_headers(path: str | os.PathLike[str]) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Read tensor shapes and metadata without materialising payload tensors."""
    entry = Path(path).resolve()
    shapes: dict[str, tuple[int, ...]] = {}
    metadata: dict[str, str] = {}
    files = _checkpoint_files(entry)
    if is_index_path(str(entry)):
        with entry.open(encoding="utf-8") as handle:
            metadata.update({k: str(v) for k, v in (json.load(handle).get("metadata") or {}).items()})
        files = files[1:]
    for item in files:
        with safe_open(str(item), framework="pt", device="cpu") as handle:
            if not metadata:
                metadata.update(dict(handle.metadata() or {}))
            for key in handle.keys():
                shapes[key] = tuple(handle.get_slice(key).get_shape())
    return shapes, metadata


def resolve_understanding_source(
    declaration: Mapping[str, Any], *, override: str | None = None
) -> str:
    locator = str(declaration.get("locator") or "")
    candidate = override or (locator[len("model:") :] if locator.startswith("model:") else "")
    if not candidate:
        raise ChimeraArtifactError("understanding locator must use model:<path>")
    if not Path(candidate).is_file():
        raise FileNotFoundError(f"referenced SenseNova checkpoint not found: {candidate}")
    expected = str(declaration.get("content_hash") or "")
    actual = checkpoint_content_hash(candidate)
    if not expected or actual != expected:
        raise ChimeraArtifactError(
            f"understanding source hash mismatch: expected {expected or '<missing>'}, got {actual}"
        )
    shapes, _ = checkpoint_headers(candidate)
    if not is_sensenova_state_dict_keys(shapes):
        raise ChimeraArtifactError(f"understanding source is not a supported SenseNova checkpoint: {candidate}")
    return str(Path(candidate).resolve())


def infer_bridge_config(sensenova_config: Mapping[str, Any]) -> ChimeraBridgeConfig:
    llm = dict(sensenova_config.get("llm_config") or {})
    hidden = int(llm.get("hidden_size") or 0)
    layers = int(llm.get("num_hidden_layers") or 0)
    kv_heads = int(llm.get("num_key_value_heads") or 0)
    heads = int(llm.get("num_attention_heads") or 0)
    head_dim = int(llm.get("head_dim") or (hidden // heads if heads else 0))
    if not all((hidden, layers, kv_heads, head_dim)):
        raise ChimeraArtifactError("SenseNova config does not declare hidden/layer/KV geometry")
    from .conditioning_bridge import selected_layer_indices

    return ChimeraBridgeConfig(
        hidden_size=hidden,
        kv_width=kv_heads * head_dim,
        selected_layers=selected_layer_indices(layers),
    )


def manifest_template(
    *,
    understanding_source: str,
    understanding_hash: str,
    understanding_config_hash: str,
    sdxl_source: str,
    sdxl_hash: str,
    initialization: str,
    unet_config: Mapping[str, Any],
    parameter_count: int,
    bridge_config: ChimeraBridgeConfig,
    vae_facts: Mapping[str, Any],
    prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    declared = dict(prediction or prediction_contract())
    declared = prediction_contract(
        str(declared.get("type") or ""),
        latent_mean=declared.get("latent_mean"),
        latent_centered_second_moment=declared.get(
            "latent_centered_second_moment"
        ),
    )
    return {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "understanding": {
            "locator": f"model:{Path(understanding_source).resolve()}",
            "basename": Path(understanding_source).name,
            "content_hash": understanding_hash,
            "config_hash": understanding_config_hash,
            "model_type": "sensenova",
            "branch": "understanding",
        },
        "sdxl_donor": {
            "provenance": f"model:{Path(sdxl_source).resolve()}",
            "basename": Path(sdxl_source).name,
            "content_hash": sdxl_hash,
        },
        "unet": {
            "initialization": initialization,
            "config_hash": config_hash(unet_config),
            "parameter_count": int(parameter_count),
        },
        "vae": dict(vae_facts),
        "conditioning": {
            "context_length": "native_prefix",
            "context_dim": bridge_config.context_dim,
            "pooled_dim": bridge_config.pooled_dim,
            "alignment_tokens": bridge_config.alignment_tokens,
            "attention_mask": True,
            "bridge_state": "unaligned",
            "position_encoding": {
                "mode": "sensenova_3d_rope",
                "layout": "rotate_half",
                "axes_ratio": [2, 1, 1],
                "spatial_unit_pixels": int(SPATIAL_UNIT_PIXELS),
                "version": POSITION_LAYOUT_VERSION,
            },
        },
        "prediction": declared,
    }


def runtime_config(
    *, unet_config: Mapping[str, Any], bridge_config: ChimeraBridgeConfig, vae_config: Mapping[str, Any]
) -> dict[str, Any]:
    bridge = asdict(bridge_config)
    bridge["selected_layers"] = list(bridge["selected_layers"])
    return {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "unet": dict(unet_config),
        "conditioning_bridge": bridge,
        "vae": dict(vae_config),
    }


def read_artifact_documents(directory: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory).resolve()
    try:
        with (root / MANIFEST_NAME).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (root / CONFIG_NAME).open(encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ChimeraArtifactError(f"cannot read Chimera artifact documents at {root}: {exc}") from exc
    versions = set()
    for label, value in (("manifest", manifest), ("config", config)):
        version = int(value.get("format_version", 0))
        if value.get("model_type") != MODEL_TYPE or version not in SUPPORTED_FORMAT_VERSIONS:
            raise ChimeraArtifactError(f"unsupported {label} model_type/format_version at {root}")
        versions.add(version)
    if len(versions) != 1:
        raise ChimeraArtifactError(f"manifest/config format version mismatch at {root}")
    version = versions.pop()
    validated_prediction_contract(manifest, artifact_format=version)
    conditioning = manifest.get("conditioning") or {}
    expected = {
        "context_length": "native_prefix",
        "context_dim": 2048,
        "pooled_dim": 1280,
        "alignment_tokens": 77,
        "attention_mask": True,
    }
    mismatches = {
        key: conditioning.get(key)
        for key, value in expected.items()
        if conditioning.get(key) != value
    }
    if mismatches:
        raise ChimeraArtifactError(
            f"invalid Chimera conditioning contract at {root}: {mismatches}"
        )
    bridge = config.get("conditioning_bridge") or {}
    if "context_tokens" in bridge or bridge.get("alignment_tokens") != 77:
        raise ChimeraArtifactError(
            f"invalid Chimera conditioning bridge config at {root}"
        )
    return manifest, config


def find_weights_entry(directory: str | os.PathLike[str]) -> str:
    root = Path(directory).resolve()
    index = root / "model.safetensors.index.json"
    single = root / WEIGHTS_BASENAME
    if index.is_file():
        return str(index)
    if single.is_file():
        return str(single)
    raise FileNotFoundError(f"Chimera artifact has no model weights at {root}")


def prefixed_state(module: torch.nn.Module, prefix: str) -> Iterable[tuple[str, torch.Tensor]]:
    for name, tensor in module.state_dict().items():
        yield f"{prefix}{name}", tensor.detach().cpu().contiguous()


def save_chimera_checkpoint(
    output_directory: str | os.PathLike[str],
    *,
    base_manifest: Mapping[str, Any],
    runtime: Mapping[str, Any],
    bridge: torch.nn.Module,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    frozen_vae_state: Mapping[str, torch.Tensor] | None = None,
    stage: str,
    step: int,
    epoch: int,
    alignment_metrics: Mapping[str, float] | None = None,
    alignment_passed: bool = False,
    stage_plan: Iterable[str] | None = None,
    bridge_align_steps: int = 0,
    max_shard_bytes: int = 10 * 1024**3,
) -> Path:
    """Atomically save a production-loadable Chimera training checkpoint."""
    from core.models.common.single_file_format import dedup_tensors, save_single_file_state

    target = Path(output_directory).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        manifest, _config = read_artifact_documents(target)
        if int((manifest.get("training") or {}).get("step", -1)) == int(step):
            return target
        raise FileExistsError(f"Chimera checkpoint target already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.saving-", dir=str(target.parent)))
    try:
        manifest = json.loads(json.dumps(dict(base_manifest)))
        vae_state = dict(frozen_vae_state or vae.state_dict())
        if content_hash_for_state_dict(vae_state) != manifest["vae"]["content_hash"]:
            raise ValueError("checkpoint VAE state differs from the pinned donor VAE")
        metrics = {key: float(value) for key, value in (alignment_metrics or {}).items()}
        if alignment_passed:
            manifest["conditioning"]["bridge_state"] = "aligned"
        manifest["training"] = {
            "stage": str(stage),
            "stage_plan": list(stage_plan or (stage,)),
            "bridge_align_steps": int(bridge_align_steps),
            "step": int(step),
            "epoch": int(epoch),
            "alignment_metrics": metrics,
            "alignment_passed": bool(alignment_passed),
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
            "format_version": str(int(manifest["format_version"])),
            "training_stage": str(stage),
            "training_step": str(int(step)),
            "training_epoch": str(int(epoch)),
            "tied_weights_dropped": json.dumps(dropped),
        }
        save_single_file_state(
            tensors,
            metadata,
            str(temporary / WEIGHTS_BASENAME),
            max_shard_bytes=max_shard_bytes,
        )
        with (temporary / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(dict(runtime), handle, indent=2, ensure_ascii=False)
        with (temporary / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, target)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
