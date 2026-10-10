"""Side-effect-free residual topology schema and preview endpoints."""
from copy import deepcopy

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal
from safetensors import SafetensorError

from api.param_defaults import DIT_REWIRING_PRESETS, DIT_HYBRID_DEFAULTS, DIT_HYBRID_INITIALIZATION_DEFAULTS as INIT_DEFAULTS
from api.param_defaults import DIT_HYBRID_INITIALIZATION_PHASE_DEFAULTS as PHASE_DEFAULTS
from database import get_datasets_db
from core.rewiring.capability import RUNTIME_ARCHITECTURES
from core.rewiring.spec import digest, preview_plan, validate_spec

router = APIRouter()


class RewiringPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    spec: dict
    token_height: int = Field(ge=1, le=65536)
    token_width: int = Field(ge=1, le=65536)


@router.get("/schema/dit-rewiring")
def rewiring_schema():
    from api.hybrid_initialization_job import INITIALIZER_ARCHITECTURES
    return {"presets": deepcopy(DIT_REWIRING_PRESETS), "hybrid_defaults": deepcopy(DIT_HYBRID_DEFAULTS),
            "runtime_architectures": list(RUNTIME_ARCHITECTURES),
            "initialization_defaults": deepcopy(INIT_DEFAULTS),
            "initializer_architectures": list(INITIALIZER_ARCHITECTURES)}


class HybridInitializationPhaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["data_noising", "native_trajectory"]
    updates: int = Field(ge=1, le=10000)
    learning_rate: float = Field(gt=0, le=1)
    skip_learning_rate_multiplier: float = Field(
        default=PHASE_DEFAULTS["skip_learning_rate_multiplier"], gt=0, le=1)


class HybridInitializationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_path: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    dataset_id: int = Field(gt=0)
    spec: dict | None = INIT_DEFAULTS["spec"]
    item_ids: list[int] | None = Field(default=INIT_DEFAULTS["item_ids"], max_length=64)
    sample_count: int = Field(default=INIT_DEFAULTS["sample_count"], ge=1, le=64)
    holdout_count: int = Field(default=INIT_DEFAULTS["holdout_count"], ge=0, le=63)
    resolution: int = Field(default=INIT_DEFAULTS["resolution"], ge=64, le=1024)
    seed: int = Field(default=INIT_DEFAULTS["seed"], ge=0, le=2147483647)
    caption_type: str = Field(default=INIT_DEFAULTS["caption_type"], min_length=1, max_length=64)
    teacher_steps: int = Field(default=INIT_DEFAULTS["teacher_steps"], ge=1, le=100)
    teacher_cfg: float = Field(default=INIT_DEFAULTS["teacher_cfg"], ge=1, le=20)
    conditioning_dropout: float = Field(default=INIT_DEFAULTS["conditioning_dropout"], ge=0, le=1)
    clip_norm: float = Field(default=INIT_DEFAULTS["clip_norm"], gt=0, le=100)
    optimizer: Literal["adamw8bit", "adamw"] = INIT_DEFAULTS["optimizer"]
    stochastic_rounding: bool = INIT_DEFAULTS["stochastic_rounding"]
    weight_decay: float = Field(default=INIT_DEFAULTS["weight_decay"], ge=0, le=1)
    attention_backend: Literal["flash", "native"] = INIT_DEFAULTS["attention_backend"]
    phases: list[HybridInitializationPhaseRequest] = Field(
        default_factory=lambda: [HybridInitializationPhaseRequest(**phase) for phase in INIT_DEFAULTS["phases"]],
        min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_plan(self):
        if self.resolution % 64 or self.holdout_count >= self.sample_count:
            raise ValueError("Resolution must be divisible by 64 and holdout_count must be smaller than sample_count")
        if self.item_ids is not None and (len(self.item_ids) != self.sample_count
                or len(set(self.item_ids)) != len(self.item_ids) or any(i <= 0 for i in self.item_ids)):
            raise ValueError("item_ids must contain sample_count distinct positive IDs")
        if self.spec is not None and validate_spec(self.spec).get("method") != "hybrid_unet_ssdit":
            raise ValueError("Initialization requires a hybrid U-Net declaration")
        for phase in self.phases:
            if phase.learning_rate * phase.skip_learning_rate_multiplier <= 0:
                raise ValueError("Effective skip learning rate must be positive")
            if (phase.skip_learning_rate_multiplier < 1 and self.spec is not None
                    and not self.spec.get("skips")):
                raise ValueError("Skip learning-rate multiplier requires compressed skips")
        return self


@router.get("/models/rewiring/initialize")
def initialization_status():
    from api.hybrid_initialization_job import status
    return status()


@router.post("/models/rewiring/initialize", status_code=202)
def start_initialization(request: HybridInitializationRequest, db=Depends(get_datasets_db)):
    from pathlib import Path
    from sqlalchemy.orm import selectinload
    from database.models import Dataset, DatasetItem
    from core.pipeline import pipeline_manager
    from core.model_state_coordinator import ModelStateBusyError
    from api.hybrid_initialization_job import start
    if db.get(Dataset, request.dataset_id) is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    query = db.query(DatasetItem).options(selectinload(DatasetItem.captions)).filter(
        DatasetItem.dataset_id == request.dataset_id)
    if request.item_ids is not None:
        query = query.filter(DatasetItem.id.in_(request.item_ids))
    rows = query.order_by(DatasetItem.id).limit(request.sample_count).all()
    if len(rows) != request.sample_count:
        raise HTTPException(status_code=422, detail="Dataset does not contain all requested calibration items")
    if request.item_ids is not None:
        by_id = {row.id: row for row in rows}
        rows = [by_id[item_id] for item_id in request.item_ids]
    try:
        items = []
        for row in rows:
            path = Path(row.image_path).resolve(strict=True)
            if not path.is_file() or row.item_type != "single":
                raise ValueError("Calibration requires individual image files")
            captions = sorted((c for c in row.captions if c.caption_type == request.caption_type),
                              key=lambda c: c.id, reverse=True)
            if not captions:
                raise ValueError(f"Calibration item {row.id} has no {request.caption_type!r} caption")
            items.append({"image_path": str(path), "caption": captions[0].content})
        return start(pipeline_manager, request.model_dump(), items)
    except (ModelStateBusyError, FileExistsError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, OSError, SafetensorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/models/rewiring/initialize/{job_id}")
def cancel_initialization(job_id: str):
    from api.hybrid_initialization_job import cancel
    from core.model_state_coordinator import ModelStateBusyError
    try:
        return cancel(job_id)
    except ModelStateBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class HybridBuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    base_spec: dict
    stream_widths: dict
    boundaries: list[dict] = Field(max_length=128)
    refiners: list[dict] = Field(max_length=128)
    skips: list[dict] = Field(max_length=128)


@router.post("/models/rewiring/hybrid/build")
def build_hybrid(request: HybridBuildRequest):
    from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
    try:
        base = request.base_spec
        if base.get("method") == "hybrid_unet_ssdit":
            base = legacy_topology(base)
        spec = build_hybrid_spec(base, stream_widths=request.stream_widths,
                                boundaries=request.boundaries, refiners=request.refiners, skips=request.skips)
        return {"spec": spec, "spec_sha256": digest(spec)}
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/models/rewiring/validate")
def validate_rewiring(request: RewiringPreviewRequest):
    try:
        spec = validate_spec(request.spec)
        return {"spec": spec, "spec_sha256": digest(spec),
                "plan": preview_plan(spec, request.token_height, request.token_width)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RewiringInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_path: str = Field(min_length=1)


@router.post("/models/rewiring/inspect")
def inspect_rewiring(request: RewiringInspectRequest):
    from core.rewiring.inspect import inspect_model
    try:
        return inspect_model(request.model_path)
    except (ValueError, OSError, SafetensorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class RewiringExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_path: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    spec: dict


@router.post("/models/rewiring/export")
def export_rewiring(request: RewiringExportRequest):
    from core.rewiring.export import export_acoustic_topology
    try:
        return export_acoustic_topology(request.model_path, request.output_name, request.spec)
    except (ValueError, OSError, SafetensorError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
