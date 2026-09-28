"""Versioned inference artifacts and resumable checkpoints for FoveaTag."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .encoder import FoveaEncoderConfig
from .views import ViewConfig


FOVEATAG_FORMAT_VERSION = 1
COORDINATE_CONVENTION = "normalized-source-xy-content-aware-v1"


@dataclass(frozen=True)
class InferenceMode:
    overview_size: int
    local_views: int

    def __post_init__(self) -> None:
        if self.overview_size <= 0 or self.local_views < 0:
            raise ValueError("inference mode dimensions must be non-negative")


@dataclass(frozen=True)
class FoveaArtifactConfig:
    num_tags: int
    feature_dim: int = 512
    evidence_slots_per_view: int = 2
    pool_heads: int = 8
    temperature: float = 0.1
    encoder: FoveaEncoderConfig = field(default_factory=FoveaEncoderConfig)
    views: ViewConfig = field(default_factory=ViewConfig)
    routing_slots: int = 4
    routing_dim: int = 256
    inference_modes: Mapping[str, InferenceMode] = field(default_factory=lambda: {
        "fast": InferenceMode(384, 2),
        "balanced": InferenceMode(448, 4),
        "detail": InferenceMode(448, 8),
    })

    def __post_init__(self) -> None:
        if self.num_tags <= 0 or min(self.feature_dim, self.evidence_slots_per_view, self.pool_heads) <= 0:
            raise ValueError("model dimensions and num_tags must be positive")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["format_version"] = FOVEATAG_FORMAT_VERSION
        value["coordinate_convention"] = COORDINATE_CONVENTION
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FoveaArtifactConfig":
        version = int(value.get("format_version", 0))
        if version != FOVEATAG_FORMAT_VERSION:
            raise ValueError(f"unsupported FoveaTag format_version: {version}")
        if value.get("coordinate_convention") != COORDINATE_CONVENTION:
            raise ValueError("unsupported FoveaTag coordinate convention")
        encoder_value = dict(value["encoder"])
        encoder_value["widths"] = tuple(encoder_value["widths"])
        encoder_value["depths"] = tuple(encoder_value["depths"])
        encoder_value["attention_heads"] = tuple(encoder_value["attention_heads"])
        encoder = FoveaEncoderConfig(**encoder_value)
        views_value = dict(value["views"])
        views_value["field_levels"] = tuple(views_value["field_levels"])
        views_value["scale_jitter"] = tuple(views_value["scale_jitter"])
        views_value["pad_color"] = tuple(views_value["pad_color"])
        views = ViewConfig(**views_value)
        modes = {
            name: InferenceMode(**mode)
            for name, mode in value["inference_modes"].items()
        }
        return cls(
            num_tags=int(value["num_tags"]),
            feature_dim=int(value["feature_dim"]),
            evidence_slots_per_view=int(value["evidence_slots_per_view"]),
            pool_heads=int(value["pool_heads"]),
            temperature=float(value["temperature"]),
            encoder=encoder,
            views=views,
            routing_slots=int(value["routing_slots"]),
            routing_dim=int(value["routing_dim"]),
            inference_modes=modes,
        )


def save_inference_artifact(
    path: os.PathLike[str] | str,
    model: nn.Module,
    config: FoveaArtifactConfig,
    *,
    router: Optional[nn.Module] = None,
    vocabulary: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save only inference-time weights; training-only heads are not accepted."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError("inference artifact path must end in .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _prefixed_state(model, "model")
    if router is not None:
        state.update(_prefixed_state(router, "router"))
    save_file(state, str(path))
    metadata = {
        "config": config.to_dict(),
        "vocabulary": dict(vocabulary or {}),
        "provenance": dict(provenance or {"initialization": "scratch"}),
        "contains_training_only_state": False,
    }
    _write_json(_metadata_path(path), metadata)
    return path


def load_inference_artifact(
    path: os.PathLike[str] | str,
    model: nn.Module,
    *,
    router: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    path = Path(path)
    metadata = _read_json(_metadata_path(path))
    FoveaArtifactConfig.from_dict(metadata["config"])
    state = load_file(str(path), device="cpu")
    model.load_state_dict(_strip_prefix(state, "model"), strict=True)
    router_state = _strip_prefix(state, "router")
    if router is None and router_state:
        raise ValueError("artifact contains router weights but no router was supplied")
    if router is not None:
        if not router_state:
            raise ValueError("router was supplied but artifact has no router weights")
        router.load_state_dict(router_state, strict=True)
    return metadata


def save_training_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: FoveaArtifactConfig,
    *,
    step: int,
    router: Optional[nn.Module] = None,
    distillation_head: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
) -> Path:
    """Save all state needed to resume, including optional diagnostic modules."""
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError("training checkpoint path must end in .pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FOVEATAG_FORMAT_VERSION,
        "config": config.to_dict(),
        "step": int(step),
        "model": model.state_dict(),
        "router": router.state_dict() if router is not None else None,
        "distillation_head": distillation_head.state_dict() if distillation_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint(
    path: os.PathLike[str] | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    router: Optional[nn.Module] = None,
    distillation_head: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != FOVEATAG_FORMAT_VERSION:
        raise ValueError("unsupported FoveaTag training checkpoint")
    FoveaArtifactConfig.from_dict(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    _load_optional_state("router", payload.get("router"), router)
    _load_optional_state("distillation_head", payload.get("distillation_head"), distillation_head)
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise ValueError("scheduler was supplied but checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    return payload


def _prefixed_state(module: nn.Module, prefix: str) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }


def _strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    marker = f"{prefix}."
    return {name[len(marker):]: value for name, value in state.items() if name.startswith(marker)}


def _load_optional_state(name: str, state: Any, module: Optional[nn.Module]) -> None:
    if state is None and module is None:
        return
    if state is None or module is None:
        raise ValueError(f"checkpoint and runtime disagree about optional {name}")
    module.load_state_dict(state, strict=True)


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
