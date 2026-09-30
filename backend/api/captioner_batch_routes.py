"""Durable, resumable caption previews without automatic sidecar writes."""

import asyncio
import copy
from pathlib import Path
import threading
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from sqlalchemy import func

from api.captioner_routes import _predict_caption, read_caption_image
from api.error_handlers import APIError, NotFoundError, ValidationError
from api.param_defaults import CAPTIONER_BATCH_DEFAULTS as BD, CAPTIONER_DEFAULTS as CD
from core.captioner.artifact import load_bundle
from core.captioner.inference import CaptionInferenceManager
from core.captioner.storage import file_hash
from core.tagger.siglip2_inference_manager import get_siglip2_inference_manager
from database import DatasetsSessionLocal, TrainingSessionLocal
from database.models import CaptionerBatch as Batch, CaptionerBatchItem as Item, Dataset, DatasetItem

router = APIRouter()


class CaptionerBatchCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bundle_path: str = Field(min_length=1)
    dataset_id: StrictInt = Field(ge=1)
    item_ids: list[StrictInt] = Field(min_length=1, max_length=BD["max_items"])
    tag_source: Literal["predicted", "provided", "merged"] = CD["tag_source"]
    tags: list[str] = CD["tags"]
    max_new_tokens: StrictInt = Field(CD["max_new_tokens"], ge=1)
    device: Literal["cpu", "cuda:0"] = CD["device"]


class CaptionerBatchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry_failed: StrictBool = BD["retry_failed"]


class BackgroundRequest:
    async def is_disconnected(self):
        return False


async def _thread(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


class CaptionBatchService:
    def __init__(self, session_factory, dataset_session_factory, manager_factory=None):
        self.db, self.datasets = session_factory, dataset_session_factory
        self.manager_factory = manager_factory or (lambda: CaptionInferenceManager(get_siglip2_inference_manager()))
        self.lock = threading.RLock()
        self.workers = {}

    def _get(self, db, job_id):
        job = db.get(Batch, str(job_id))
        if job is None:
            raise NotFoundError("Caption batch not found")
        if job.status in {"running", "stopping"} and job.job_id not in self.workers:
            job.status = "interrupted"
            db.query(Item).filter(Item.job_id == job.job_id, Item.status == "running").update({Item.status: "pending"})
            db.commit()
        return job

    @staticmethod
    def _describe(db, job):
        counts = dict(db.query(Item.status, func.count()).filter(Item.job_id == job.job_id).group_by(Item.status).all())
        return {**job.to_dict(), "counts": counts, "total": sum(counts.values())}

    def list(self):
        with self.lock, self.db() as db:
            jobs = db.query(Batch).order_by(Batch.created_at.desc()).all()
            return {"batches": [self._describe(db, self._get(db, job.job_id)) for job in jobs]}

    def get(self, job_id):
        with self.lock, self.db() as db:
            return self._describe(db, self._get(db, job_id))

    def results(self, job_id, offset, limit):
        with self.lock, self.db() as db:
            job = self._get(db, job_id)
            rows = db.query(Item).filter(Item.job_id == job.job_id).order_by(Item.position).offset(offset).limit(limit).all()
            return {"items": [row.to_dict() for row in rows], "offset": offset, "total": self._describe(db, job)["total"]}

    def create(self, request):
        if len(set(request.item_ids)) != len(request.item_ids) or any(key < 1 for key in request.item_ids):
            raise ValidationError("Item IDs must be unique positive integers")
        if request.tag_source == "predicted" and request.tags:
            raise ValidationError("predicted mode does not accept provided tags")
        path = Path(request.bundle_path).expanduser().resolve()
        manifest_hash = file_hash(path / "manifest.json")
        model, _, _, _, _ = load_bundle(path)
        if request.max_new_tokens > model.config.max_length:
            raise ValidationError("Generation limit exceeds caption bundle")
        del model
        if manifest_hash != file_hash(path / "manifest.json"):
            raise ValidationError("Bundle changed while creating batch")
        with self.datasets() as db:
            dataset = db.get(Dataset, request.dataset_id)
            if dataset is None:
                raise NotFoundError("Dataset not found")
            items = db.query(DatasetItem).filter(DatasetItem.dataset_id == request.dataset_id,
                DatasetItem.id.in_(request.item_ids)).all()
            if len(items) != len(request.item_ids) or any(item.item_type in {"audio", "video"} for item in items):
                raise NotFoundError("Dataset image item not found")
        job_id = str(uuid4())
        with self.lock, self.db() as db:
            job = Batch(job_id=job_id, dataset_id=request.dataset_id, bundle_path=str(path),
                manifest_hash=manifest_hash, options=request.model_dump(exclude={"bundle_path", "dataset_id", "item_ids"}),
                status="pending", error=None, error_code=None)
            db.add(job)
            db.flush()
            db.add_all([Item(job_id=job_id, item_id=key, position=position, status="pending", attempts=0)
                        for position, key in enumerate(request.item_ids)])
            db.commit()
            return self._describe(db, job)

    def start(self, job_id, retry_failed):
        with self.lock, self.db() as db:
            job = self._get(db, job_id)
            if self.workers:
                raise APIError("A caption batch worker is already active", 409)
            if retry_failed:
                db.query(Item).filter(Item.job_id == job.job_id, Item.status == "failed").update(
                    {Item.status: "pending", Item.error: None, Item.error_code: None})
            if not db.query(Item).filter(Item.job_id == job.job_id, Item.status == "pending").first():
                raise APIError("No pending caption batch items", 409)
            stop = threading.Event()
            job.status, job.error, job.error_code = "running", None, None
            db.commit()
            data = self._describe(db, job)
            self.workers[job.job_id] = {"stop": stop, "task": None}
            try:
                task = asyncio.create_task(self._work(data, stop), name=f"caption-batch-{job.job_id}")
                self.workers[job.job_id]["task"] = task
            except BaseException:
                self.workers.pop(job.job_id, None)
                raise
            return data

    def stop(self, job_id):
        with self.lock, self.db() as db:
            job = self._get(db, job_id)
            record = self.workers.get(job.job_id)
            if record:
                record["stop"].set()
                job.status = "stopping"
            elif job.status == "pending":
                job.status = "stopped"
            db.commit()
            return self._describe(db, job)

    def _next(self, job_id):
        with self.lock, self.db() as db:
            row = db.query(Item).filter(Item.job_id == job_id, Item.status == "pending").order_by(Item.position).first()
            if row is None:
                return None
            row.status, row.attempts = "running", row.attempts + 1
            db.commit()
            return row.item_id

    def _item(self, job_id, item_id, status, preview=None, error=None, code=None):
        with self.lock, self.db() as db:
            row = db.get(Item, (job_id, item_id))
            row.status, row.preview, row.error, row.error_code = status, preview, error, code
            db.commit()

    def _finish(self, job_id, status, error=None, code=None):
        with self.lock, self.db() as db:
            job = db.get(Batch, job_id)
            job.status, job.error, job.error_code = status, error, code
            db.query(Item).filter(Item.job_id == job_id, Item.status == "running").update({Item.status: "pending"})
            db.commit()

    async def _work(self, data, stop):
        key = data["job_id"]
        manager = None
        try:
            if await _thread(file_hash, Path(data["bundle_path"]) / "manifest.json") != data["manifest_hash"]:
                raise ValidationError("Batch bundle manifest changed", code="captioner_batch_bundle_changed")
            manager = self.manager_factory()
            while not stop.is_set():
                item_id = await _thread(self._next, key)
                if item_id is None:
                    break
                try:
                    content, revision = await _thread(read_caption_image, data["dataset_id"], item_id, self.datasets)
                    if stop.is_set():
                        break
                    result = await _predict_caption(BackgroundRequest(), uuid4(), content, **data["options"],
                        manager=manager, bundle_path=None if manager.status()["loaded"] else data["bundle_path"],
                        unload_bundle=False, stop_event=stop)
                    result = {**result, "dataset_id": data["dataset_id"], "item_id": item_id, "dataset_revision": revision}
                    await _thread(self._item, key, item_id, "completed", result)
                except APIError as exc:
                    if exc.code == "captioner_cancelled" and stop.is_set():
                        break
                    if exc.code in {"captioner_invalid_artifact", "captioner_tagger_not_loaded", "captioner_tagger_mismatch",
                                    "captioner_unsupported_backend", "captioner_gpu_busy", "captioner_model_busy"}:
                        raise
                    await _thread(self._item, key, item_id, "failed", None, str(exc), exc.code)
            await _thread(self._finish, key, "stopped" if stop.is_set() else "completed")
        except asyncio.CancelledError:
            stop.set()
            await _thread(self._finish, key, "interrupted")
            raise
        except Exception as exc:
            await _thread(self._finish, key, "failed", str(exc), getattr(exc, "code", None))
        finally:
            try:
                if manager is not None:
                    await _thread(manager.unload)
            finally:
                with self.lock:
                    self.workers.pop(key, None)


service = CaptionBatchService(TrainingSessionLocal, DatasetsSessionLocal)


@router.get("/schema/captioner-batch-defaults")
def batch_defaults():
    return copy.deepcopy(BD)


@router.get("/captioner/batches")
def list_batches():
    return service.list()


@router.post("/captioner/batches")
def create_batch(request: CaptionerBatchCreateRequest):
    try:
        return service.create(request)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ValidationError(str(exc), code="captioner_invalid_artifact") from exc


@router.get("/captioner/batches/{job_id}")
def batch_status(job_id: UUID):
    return service.get(job_id)


@router.post("/captioner/batches/{job_id}/start")
async def start_batch(job_id: UUID, request: CaptionerBatchStartRequest):
    return service.start(job_id, request.retry_failed)


@router.post("/captioner/batches/{job_id}/stop")
def stop_batch(job_id: UUID):
    return service.stop(job_id)


@router.get("/captioner/batches/{job_id}/results")
def batch_results(job_id: UUID, offset: int = Query(BD["offset"], ge=0),
                  limit: int = Query(BD["limit"], ge=1, le=1000)):
    return service.results(job_id, offset, limit)
