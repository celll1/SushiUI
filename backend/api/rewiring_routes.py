"""Side-effect-free residual topology schema and preview endpoints."""
from copy import deepcopy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from api.param_defaults import DIT_REWIRING_PRESETS
from core.rewiring.spec import digest, preview_plan, validate_spec

router = APIRouter()


class RewiringPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    spec: dict
    token_height: int = Field(ge=1, le=65536)
    token_width: int = Field(ge=1, le=65536)


@router.get("/schema/dit-rewiring")
def rewiring_schema():
    return {"presets": deepcopy(DIT_REWIRING_PRESETS), "runtime_architectures": []}


@router.post("/models/rewiring/validate")
def validate_rewiring(request: RewiringPreviewRequest):
    try:
        spec = validate_spec(request.spec)
        return {"spec": spec, "spec_sha256": digest(spec),
                "plan": preview_plan(spec, request.token_height, request.token_width)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
