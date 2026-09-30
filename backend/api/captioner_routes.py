"""Compact captioner REST integration."""

import copy
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy.orm import Session

from api.error_handlers import APIError, NotFoundError, ValidationError
from api.param_defaults import CAPTIONER_DEFAULTS, CAPTIONER_TRAINING_DEFAULTS
from core.captioner.storage import file_hash
from core.datasets.captions import select_caption, update_caption
from database import get_datasets_db
from database.models import Dataset, DatasetCaption, DatasetItem

router = APIRouter()


class CaptionerSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: StrictInt = Field(ge=1)
    item_id: StrictInt = Field(ge=1)
    dataset_revision: StrictInt = Field(ge=0)
    image_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    caption: str = Field(min_length=1, max_length=16384)
    caption_type: str = Field(CAPTIONER_DEFAULTS["save_caption_type"], min_length=1, max_length=128)


class CaptionerSaveResponse(BaseModel):
    caption_id: int
    dataset_revision: int
    sidecar_path: str
    source_field: Literal["compact_caption"]


@router.get("/schema/captioner-defaults")
def captioner_defaults() -> dict:
    return copy.deepcopy(CAPTIONER_DEFAULTS)


@router.get("/schema/captioner-training-defaults")
def captioner_training_defaults() -> dict:
    return copy.deepcopy(CAPTIONER_TRAINING_DEFAULTS)


@router.post("/captioner/save", response_model=CaptionerSaveResponse)
def save_caption(request: CaptionerSaveRequest, db: Session = Depends(get_datasets_db)):
    if request.caption_type == "tags" or not request.caption_type.strip() or not request.caption.strip():
        raise ValidationError("Select a nonempty natural caption and a non-tag caption type", code="captioner_invalid_input")
    try:
        # Serialize revision checks with other SQLite dataset writers.
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).one_or_none()
        item = db.query(DatasetItem).filter(DatasetItem.id == request.item_id,
                                          DatasetItem.dataset_id == request.dataset_id).one_or_none()
        if dataset is None or item is None or item.item_type in {"audio", "video"}:
            raise NotFoundError("Dataset image item not found", code="captioner_item_not_found")
        if dataset.revision != request.dataset_revision:
            raise APIError("Dataset changed after caption preview", status_code=409, code="captioner_stale_revision")
        root = Path(dataset.path).resolve()
        path = Path(item.image_path)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_relative_to(root) or any(not path.with_suffix(suffix).resolve().is_relative_to(root)
                                               for suffix in (".txt", ".json")):
            raise ValidationError("Image or sidecar escapes registered dataset root", code="captioner_unsafe_path")
        if file_hash(path) != request.image_hash:
            raise APIError("Image changed after caption preview", status_code=409, code="captioner_stale_image")
        if db.query(DatasetCaption.id).filter(DatasetCaption.item_id == item.id,
                DatasetCaption.is_tags_format == True,
                (DatasetCaption.caption_type == request.caption_type) |
                (DatasetCaption.source_field == "compact_caption")).first():
            raise ValidationError("Cannot replace a tag-formatted caption", code="captioner_tag_destination")
        caption = select_caption(db, item.id, request.caption_type, caption_id=None, source_field="compact_caption")
        if db.query(DatasetCaption.id).filter(DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == request.caption_type,
                (DatasetCaption.source_field != "compact_caption") |
                (DatasetCaption.source_field == None)).first():
            raise ValidationError("Choose a separate caption type to preserve existing captions", code="captioner_field_conflict")
        if db.query(DatasetCaption.id).filter(DatasetCaption.item_id == item.id,
                DatasetCaption.source_field == "compact_caption",
                DatasetCaption.caption_type != request.caption_type).first():
            raise ValidationError("compact_caption is already indexed with another type", code="captioner_field_conflict")
        result = update_caption(db, item_id=item.id, dataset_id=dataset.id,
            caption_type=request.caption_type, caption_id=caption.id if caption else None,
            source_field="compact_caption", content=request.caption.strip(), source="compact_captioner",
            persist_sidecar=True, sidecar_json_only=True, sidecar_image_path=str(path))
        return {"caption_id": result.caption.id, "dataset_revision": dataset.revision,
                "sidecar_path": result.sidecar.path, "source_field": "compact_caption"}
    except APIError:
        db.rollback()
        raise
    except (ValueError, OSError) as exc:
        db.rollback()
        raise ValidationError(str(exc), code="captioner_save_failed") from exc
    except Exception:
        db.rollback()
        raise
