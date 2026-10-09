"""Side-effect-free residual topology schema and preview endpoints."""
from copy import deepcopy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from safetensors import SafetensorError

from api.param_defaults import DIT_REWIRING_PRESETS, DIT_HYBRID_DEFAULTS
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
    return {"presets": deepcopy(DIT_REWIRING_PRESETS), "hybrid_defaults": deepcopy(DIT_HYBRID_DEFAULTS),
            "runtime_architectures": list(RUNTIME_ARCHITECTURES)}


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
