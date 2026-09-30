"""Compact captioner REST integration."""

import copy
import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
import json
from pathlib import Path
from typing import Literal
import threading
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy.orm import Session

from api.error_handlers import APIError, NotFoundError, ValidationError
from api.param_defaults import CAPTIONER_DEFAULTS, CAPTIONER_TRAINING_DEFAULTS
from core.captioner.storage import file_hash
from core.captioner.inference import CaptionInferenceManager
from core.gpu_coordinator import gpu_coordinator
from core.model_state_coordinator import model_state_coordinator, ModelStateBusyError
from core.tagger.siglip2_inference_manager import get_siglip2_inference_manager
from core.datasets.captions import select_caption, update_caption
from database import get_datasets_db
from database.models import Dataset, DatasetCaption, DatasetItem

router = APIRouter()
_caption_manager = None
_requests = {}
_request_lock = threading.Lock()


def get_caption_manager():
    global _caption_manager
    with _request_lock:
        if _caption_manager is None:
            _caption_manager = CaptionInferenceManager(get_siglip2_inference_manager())
        return _caption_manager


class CaptionerLoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)


class CaptionerItemPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: StrictInt = Field(ge=1)
    item_id: StrictInt = Field(ge=1)
    request_id: UUID
    tag_source: Literal["predicted", "provided", "merged"] = CAPTIONER_DEFAULTS["tag_source"]
    tags: list[str] = CAPTIONER_DEFAULTS["tags"]
    max_new_tokens: StrictInt = Field(CAPTIONER_DEFAULTS["max_new_tokens"], ge=1)
    device: Literal["cpu", "cuda:0"] = CAPTIONER_DEFAULTS["device"]


class CaptionerStatusResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    loaded: bool


class CaptionerUsedTag(BaseModel):
    name: str
    source: Literal["predicted", "provided"]
    score: float | None
    score_calibrated: bool


class CaptionerPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    caption: str
    used_tags: list[CaptionerUsedTag]
    warnings: list[str]
    termination: Literal["eos", "length_limit"]
    token_count: int
    image_hash: str
    model_identity: str
    timings_ms: dict[str, float]


def _caption_error(exc, manager, *, loading=False):
    if isinstance(exc, InterruptedError):
        return APIError(str(exc), status_code=409, code="captioner_cancelled")
    if isinstance(exc, FileNotFoundError):
        return NotFoundError(str(exc), code="captioner_artifact_not_found" if loading else "captioner_item_not_found")
    if manager.tagger.model_type == "onnx":
        return ValidationError("ONNX taggers do not expose caption features", code="captioner_unsupported_backend")
    if manager.tagger.model is None:
        return ValidationError("Load a PyTorch SigLIP2 tagger first", code="captioner_tagger_not_loaded")
    if "tagger" in str(exc).lower() and "match" in str(exc).lower():
        return ValidationError(str(exc), code="captioner_tagger_mismatch")
    return ValidationError(str(exc), code="captioner_invalid_artifact" if loading else "captioner_invalid_input")


@router.post("/captioner/load", response_model=CaptionerStatusResponse)
async def load_captioner(request: CaptionerLoadRequest):
    manager = get_caption_manager()
    try:
        return await asyncio.to_thread(manager.load, Path(request.path).expanduser().resolve())
    except (ValueError, OSError) as exc:
        raise _caption_error(exc, manager, loading=True) from exc


@router.get("/captioner/status", response_model=CaptionerStatusResponse)
async def captioner_status():
    return await asyncio.to_thread(get_caption_manager().status)


@router.post("/captioner/unload", response_model=CaptionerStatusResponse)
async def unload_captioner():
    manager = get_caption_manager()
    await asyncio.to_thread(manager.unload)
    return await asyncio.to_thread(manager.status)


@router.post("/captioner/requests/{request_id}/cancel")
def cancel_caption_request(request_id: UUID):
    with _request_lock:
        stop = _requests.get(str(request_id))
        if stop is None:
            raise NotFoundError("Caption request is no longer active", code="captioner_request_not_found")
        stop.set()
    return {"cancel_requested": True}


@asynccontextmanager
async def _caption_slot(device):
    try:
        model_state_coordinator.begin_generation()
    except ModelStateBusyError as exc:
        raise APIError(str(exc), status_code=409, code="captioner_model_busy") from exc
    try:
        if device == "cuda:0":
            async with AsyncExitStack() as stack:
                try:
                    await stack.enter_async_context(gpu_coordinator.generation_slot(estimated_peak_gb=3.0,
                        timeout=CAPTIONER_DEFAULTS["gpu_pause_timeout"], require_pause_ack=True))
                except RuntimeError as exc:
                    raise APIError(str(exc), status_code=409, code="captioner_gpu_busy") from exc
                yield
        else:
            yield
    finally:
        model_state_coordinator.end_generation()


async def _predict_caption(http_request, request_id, image_bytes, *, tag_source, tags, max_new_tokens, device,
                           manager=None, bundle_path=None):
    manager = manager if manager is not None else get_caption_manager()
    stop = threading.Event()
    with _request_lock:
        if str(request_id) in _requests:
            raise APIError("Caption request id is already active", status_code=409, code="captioner_request_busy")
        _requests[str(request_id)] = stop
    try:
        if bundle_path is None and not (await asyncio.to_thread(manager.status))["loaded"]:
            raise ValidationError("Load a captioner bundle first", code="captioner_not_loaded")
        async with _caption_slot(device):
            def predict():
                try:
                    if bundle_path is not None:
                        if stop.is_set():
                            raise InterruptedError("Caption generation cancelled")
                        try:
                            manager.load(bundle_path)
                        except (ValueError, OSError) as exc:
                            raise _caption_error(exc, manager, loading=True) from exc
                    return manager.predict(image_bytes, tag_source=tag_source, tags=tags,
                        max_new_tokens=max_new_tokens, device=device, cancelled=stop.is_set)
                finally:
                    if bundle_path is not None:
                        manager.unload()
            worker = asyncio.create_task(asyncio.to_thread(predict))
            async def watch_disconnect():
                while not worker.done():
                    if await http_request.is_disconnected():
                        stop.set()
                        return
                    await asyncio.sleep(0.2)
            monitor = asyncio.create_task(watch_disconnect())
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                stop.set()
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                raise
            except (ValueError, OSError, InterruptedError) as exc:
                raise _caption_error(exc, manager) from exc
            finally:
                monitor.cancel()
                try:
                    await monitor
                except asyncio.CancelledError:
                    pass
    finally:
        with _request_lock:
            _requests.pop(str(request_id), None)


@router.post("/captioner/predict", response_model=CaptionerPredictionResponse)
async def predict_caption(request: Request, image: UploadFile = File(...), request_id: UUID = Form(...),
        tag_source: Literal["predicted", "provided", "merged"] = Form(CAPTIONER_DEFAULTS["tag_source"]),
        tags: str = Form(json.dumps(CAPTIONER_DEFAULTS["tags"])),
        max_new_tokens: int = Form(CAPTIONER_DEFAULTS["max_new_tokens"], ge=1),
        device: Literal["cpu", "cuda:0"] = Form(CAPTIONER_DEFAULTS["device"])):
    try:
        provided = json.loads(tags)
    except json.JSONDecodeError as exc:
        raise ValidationError("tags must be a JSON array", code="captioner_invalid_input") from exc
    try:
        content = await image.read(CAPTIONER_DEFAULTS["max_image_bytes"] + 1)
    finally:
        await image.close()
    if len(content) > CAPTIONER_DEFAULTS["max_image_bytes"]:
        raise APIError("Caption image is too large", status_code=413, code="captioner_image_too_large")
    return await _predict_caption(request, request_id, content, tag_source=tag_source, tags=provided,
                                 max_new_tokens=max_new_tokens, device=device)


def read_caption_image(dataset_id, item_id, session_factory=None):
    if session_factory is None:
        from database import DatasetsSessionLocal
        session_factory = DatasetsSessionLocal
    try:
        with session_factory() as db:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).one_or_none()
            item = db.query(DatasetItem).filter(DatasetItem.id == item_id,
                DatasetItem.dataset_id == dataset_id).one_or_none()
            if dataset is None or item is None or item.item_type in {"audio", "video"}:
                raise NotFoundError("Dataset image item not found", code="captioner_item_not_found")
            root = Path(dataset.path).resolve()
            path = Path(item.image_path)
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_relative_to(root):
                raise ValidationError("Image escapes registered root", code="captioner_unsafe_path")
            with path.open("rb") as stream:
                content = stream.read(CAPTIONER_DEFAULTS["max_image_bytes"] + 1)
            if len(content) > CAPTIONER_DEFAULTS["max_image_bytes"]:
                raise APIError("Caption image is too large", status_code=413, code="captioner_image_too_large")
            return content, dataset.revision
    except OSError as exc:
        raise NotFoundError(str(exc), code="captioner_item_not_found") from exc


@router.post("/captioner/predict-item", response_model=CaptionerPredictionResponse)
async def predict_caption_item(http_request: Request, request: CaptionerItemPredictRequest):
    content, revision = await asyncio.to_thread(read_caption_image, request.dataset_id, request.item_id)
    result = await _predict_caption(http_request, request.request_id, content, tag_source=request.tag_source,
        tags=request.tags, max_new_tokens=request.max_new_tokens, device=request.device)
    return {**result, "dataset_id": request.dataset_id, "item_id": request.item_id, "dataset_revision": revision}


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
