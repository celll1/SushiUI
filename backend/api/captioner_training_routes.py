"""Durable caption training runs with one private encoder worker per process."""

import copy
import asyncio
from pathlib import Path
import threading
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, create_model

from api.error_handlers import APIError, NotFoundError, ValidationError
from api.param_defaults import CAPTIONER_RUN_DEFAULTS as RD, CAPTIONER_TRAINING_DEFAULTS as TD
from api.training_run_helpers import find_output_dir_owner
from api.captioner_routes import (CaptionerItemPredictRequest, CaptionerPredictionResponse,
                                 _predict_caption, read_caption_image)
from config.settings import settings
from core.captioner.artifact import export_bundle
from core.captioner.storage import read_json, write_json
from core.captioner.inference import CaptionInferenceManager
from core.tagger.siglip2_inference_manager import get_siglip2_inference_manager
from core.model_state_coordinator import model_state_coordinator
from database import DatasetsSessionLocal, TrainingSessionLocal
from database.models import CaptionerTrainingEvent as Event, CaptionerTrainingRun as Run

router = APIRouter()


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _config_type(value):
    if isinstance(value, bool):
        return StrictBool
    if isinstance(value, int):
        return StrictInt
    if isinstance(value, float):
        return StrictFloat
    if isinstance(value, list):
        return list[str]
    if isinstance(value, dict):
        return dict[str, StrictFloat]
    return str


CaptionerTrainingConfig = create_model("CaptionerTrainingConfig", __base__=StrictRequest,
    **{key: (_config_type(value), copy.deepcopy(value)) for key, value in TD.items()})


class CaptionerTaggerConfig(StrictRequest):
    checkpoint_path: str = Field(min_length=1)
    vocab_path: str = Field(min_length=1)
    vision_encoder_path: str = RD["vision_encoder_path"]
    lora_rank: StrictInt = Field(RD["lora_rank"], ge=1)
    lora_alpha: StrictFloat = Field(RD["lora_alpha"], gt=0, allow_inf_nan=False)


class CaptionerDatasetConfig(StrictRequest):
    dataset_id: StrictInt = Field(ge=1)
    input_tag_caption_types: list[str]
    target_caption_types: list[str] = Field(min_length=1)
    filters: dict


class CaptionerRunCreateRequest(StrictRequest):
    run_name: str = Field(RD["run_name"], min_length=1)
    output_dir: str | None = RD["output_dir"]
    tagger: CaptionerTaggerConfig
    dataset_configs: list[CaptionerDatasetConfig] = Field(min_length=1)
    config: CaptionerTrainingConfig


class CaptionerRunStartRequest(StrictRequest):
    resume: StrictBool = RD["resume"]
    checkpoint: str = RD["checkpoint"]


class CaptionerExportRequest(StrictRequest):
    checkpoint: str = RD["checkpoint"]


class CaptionerSampleRequest(CaptionerItemPredictRequest):
    checkpoint: str = RD["checkpoint"]


def _load_private_tagger(config):
    from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
    manager = SigLIP2InferenceManager()
    manager.device = "cpu"
    try:
        manager.load_model(**config)
    except BaseException:
        manager.unload()
        raise
    return manager


def _make_trainer(**kwargs):
    from core.captioner.trainer import CaptionTrainer
    return CaptionTrainer(**kwargs)


class CaptionRunService:
    def __init__(self, session_factory, dataset_session_factory, root, *,
                 loader=_load_private_tagger, trainer=_make_trainer):
        self.db = session_factory
        self.datasets = dataset_session_factory
        self.root = Path(root)
        self.loader, self.trainer = loader, trainer
        self.lock = threading.RLock()
        self.workers = {}

    def _get(self, db, run_id):
        run = db.get(Run, str(run_id))
        if run is None:
            raise NotFoundError("Caption training run not found")
        if run.status in {"starting", "running", "stopping"} and run.run_id not in self.workers:
            run.status = "interrupted"
            self._event(db, run, {"status": "interrupted"})
            db.commit()
        return run

    @staticmethod
    def _event(db, run, payload):
        run.progress = {**run.progress, **payload}
        if "checkpoint" in payload:
            run.checkpoints = list(dict.fromkeys([*run.checkpoints, payload["checkpoint"]]))
        db.add(Event(run_id=run.run_id, resume_seq=run.resume_seq, payload=payload))

    def list(self):
        with self.lock, self.db() as db:
            ids = [row.run_id for row in db.query(Run).order_by(Run.created_at.desc()).all()]
            return {"runs": [self._get(db, key).to_dict() for key in ids]}

    def get(self, run_id):
        with self.lock, self.db() as db:
            return self._get(db, run_id).to_dict()

    def create(self, request):
        from core.captioner.trainer import validate_config
        cfg = validate_config(request.config.model_dump())
        datasets = [item.model_dump() for item in request.dataset_configs]
        if len({item["dataset_id"] for item in datasets}) != len(datasets):
            raise ValidationError("Duplicate dataset IDs")
        for item in datasets:
            for key in ("input_tag_caption_types", "target_caption_types"):
                if any(not value.strip() for value in item[key]) or len(set(item[key])) != len(item[key]):
                    raise ValidationError("Caption column names must be nonempty and unique")
            if set(item["input_tag_caption_types"]) & set(item["target_caption_types"]):
                raise ValidationError("Tag inputs and caption targets must be separate columns")
            if set(item["filters"]) - {"item_ids", "min_width", "min_height"}:
                raise ValidationError("Unsupported dataset filter")
            for key, value in item["filters"].items():
                values = value if key == "item_ids" else [value]
                if not isinstance(values, list) or any(type(v) is not int or v < 1 for v in values):
                    raise ValidationError("Dataset filters require positive integers")
        tagger = request.tagger.model_dump()
        if tagger["vision_encoder_path"]:
            tagger["vision_encoder_path"] = str(Path(tagger["vision_encoder_path"]).resolve())
        for key in ("checkpoint_path", "vocab_path"):
            if not Path(tagger[key]).is_file():
                raise ValidationError(f"{key} must name an existing file")
            tagger[key] = str(Path(tagger[key]).resolve())
        alias_path = self.root / "tagother" / "tag_aliases.json"
        aliases = read_json(alias_path) if cfg["use_tag_aliases"] and alias_path.exists() else {}
        if not isinstance(aliases, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in aliases.items()):
            raise ValidationError("Invalid tag alias table")
        run_id = str(uuid4())
        output = Path(request.output_dir).resolve() if request.output_dir else self.root / "training" / "captioner" / run_id
        with self.lock, self.db() as db:
            if output.exists() or find_output_dir_owner(db, str(output)):
                raise APIError("Output directory must be new and exclusive", 409)
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "owner.json", {"run_id": run_id})
            run = Run(run_id=run_id, run_name=request.run_name, output_dir=str(output), status="pending",
                      config=cfg, tagger_config=tagger, dataset_configs=datasets, aliases=aliases,
                      progress={}, checkpoints=[], resume_seq=0, error=None)
            db.add(run)
            db.flush()
            self._event(db, run, {"status": "pending"})
            db.commit()
            return run.to_dict()

    @staticmethod
    def _checkpoint(run, name):
        root = Path(run.output_dir).resolve()
        if not name:
            marker = root / "latest.json"
            if not marker.is_file():
                raise ValidationError("No resumable checkpoint exists")
            name = read_json(marker)["checkpoint"]
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
            raise ValidationError("Checkpoint must be a run-owned directory name")
        path = (root / name).resolve()
        if path.parent != root or not path.is_dir() or not (path / "manifest.json").is_file():
            raise ValidationError("Checkpoint is missing or outside this run")
        return path

    def start(self, run_id, request):
        with self.lock, self.db() as db:
            run = self._get(db, run_id)
            if self.workers:
                raise APIError("A caption training worker is already active", 409)
            if read_json(Path(run.output_dir) / "owner.json") != {"run_id": run.run_id}:
                raise ValidationError("Output directory ownership changed")
            if not request.resume and (run.checkpoints or (Path(run.output_dir) / "snapshot.json").exists()):
                raise APIError("This run must resume from a checkpoint", 409)
            if request.checkpoint and not request.resume:
                raise ValidationError("Checkpoint selection requires resume")
            checkpoint = str(self._checkpoint(run, request.checkpoint)) if request.resume else None
            stop = threading.Event()
            run.resume_seq += 1
            run.status, run.error = "starting", None
            run.progress = {}
            self._event(db, run, {"status": "starting"})
            data = run.to_dict()
            aliases = copy.deepcopy(run.aliases)
            worker = threading.Thread(target=self._work, args=(data, aliases, checkpoint, stop),
                                      name=f"captioner-{run.run_id}", daemon=True)
            self.workers[run.run_id] = (stop, worker)
            try:
                db.commit()
                worker.start()
            except BaseException:
                self.workers.pop(run.run_id, None)
                raise
            return data

    def _progress(self, run_id, payload, status=None, error=None):
        with self.lock, self.db() as db:
            run = db.get(Run, run_id)
            if status:
                run.status = status
            run.error = error
            self._event(db, run, payload)
            db.commit()

    def _work(self, data, aliases, checkpoint, stop):
        key = data["run_id"]
        manager = None
        activity = f"caption training {key}"
        registered = False
        try:
            model_state_coordinator.begin_activity(activity)
            registered = True
            if stop.is_set():
                raise InterruptedError("Stopped before model load")
            manager = self.loader(data["tagger"])
            if stop.is_set():
                raise InterruptedError("Stopped during model load")
            self._progress(key, {"status": "running"}, status="running")
            result = self.trainer(run_id=key, output_dir=data["output_dir"], config=data["config"],
                dataset_configs=data["dataset_configs"], tagger_manager=manager,
                session_factory=self.datasets, progress=lambda payload: self._progress(key, payload),
                stop_event=stop, alias_table=aliases, resume_path=checkpoint).run()
            status = "stopped" if result["stopped"] else "completed"
            self._progress(key, {**result, "status": status}, status=status)
        except InterruptedError:
            self._progress(key, {"status": "stopped"}, status="stopped")
        except Exception as exc:
            self._progress(key, {"status": "failed", "error": str(exc)}, status="failed", error=str(exc))
        finally:
            try:
                if manager is not None:
                    manager.unload()
            finally:
                if registered:
                    model_state_coordinator.end_activity(activity)
                with self.lock:
                    self.workers.pop(key, None)

    def stop(self, run_id):
        with self.lock, self.db() as db:
            run = self._get(db, run_id)
            if run.run_id in self.workers:
                self.workers[run.run_id][0].set()
                run.status = "stopping"
            elif run.status == "pending":
                run.status = "stopped"
            self._event(db, run, {"status": run.status})
            db.commit()
            return run.to_dict()

    def events(self, run_id, after_id, limit):
        with self.lock, self.db() as db:
            self._get(db, run_id)
            rows = db.query(Event).filter(Event.run_id == str(run_id), Event.id > after_id).order_by(Event.id).limit(limit).all()
            return {"events": [row.to_dict() for row in rows]}

    def export(self, run_id, name):
        with self.lock, self.db() as db:
            run = self._get(db, run_id)
            source = self._checkpoint(run, name)
            root = Path(run.output_dir).resolve()
            destination = root / "exports" / str(uuid4())
            if not destination.resolve().is_relative_to(root):
                raise ValidationError("Export directory escaped run ownership")
            export_bundle(source, destination)
            self._event(db, run, {"export": str(destination), "source_checkpoint": source.name})
            db.commit()
            return {"path": str(destination)}

    def checkpoint(self, run_id, name):
        with self.lock, self.db() as db:
            return self._checkpoint(self._get(db, run_id), name)

    def record_sample(self, run_id, result):
        with self.lock, self.db() as db:
            run = self._get(db, run_id)
            self._event(db, run, {"sample": result})
            db.commit()


service = CaptionRunService(TrainingSessionLocal, DatasetsSessionLocal, settings.root_dir)


@router.get("/schema/captioner-run-defaults")
def run_defaults():
    return copy.deepcopy(RD)


@router.get("/captioner-training/runs")
def list_runs():
    return service.list()


@router.post("/captioner-training/runs")
def create_run(request: CaptionerRunCreateRequest):
    try:
        return service.create(request)
    except (ValueError, TypeError) as exc:
        raise ValidationError(str(exc)) from exc


@router.get("/captioner-training/runs/{run_id}")
def get_run(run_id: UUID):
    return service.get(run_id)


@router.post("/captioner-training/runs/{run_id}/start")
def start_run(run_id: UUID, request: CaptionerRunStartRequest):
    try:
        return service.start(run_id, request)
    except (ValueError, KeyError, OSError) as exc:
        raise ValidationError(str(exc)) from exc


@router.post("/captioner-training/runs/{run_id}/stop")
def stop_run(run_id: UUID):
    return service.stop(run_id)


@router.get("/captioner-training/runs/{run_id}/events")
def run_events(run_id: UUID, after_id: int = Query(RD["after_id"], ge=0),
               limit: int = Query(RD["event_limit"], ge=1, le=5000)):
    return service.events(run_id, after_id, limit)


@router.post("/captioner-training/runs/{run_id}/export")
def export_run(run_id: UUID, request: CaptionerExportRequest):
    try:
        return service.export(run_id, request.checkpoint)
    except (ValueError, KeyError, OSError) as exc:
        raise ValidationError(str(exc)) from exc


@router.post("/captioner-training/runs/{run_id}/sample", response_model=CaptionerPredictionResponse)
async def sample_run(run_id: UUID, http_request: Request, request: CaptionerSampleRequest):
    try:
        checkpoint = await asyncio.to_thread(service.checkpoint, run_id, request.checkpoint)
    except (ValueError, KeyError, OSError) as exc:
        raise ValidationError(str(exc)) from exc
    content, revision = await asyncio.to_thread(read_caption_image, request.dataset_id, request.item_id, service.datasets)
    manager = CaptionInferenceManager(get_siglip2_inference_manager())
    result = await _predict_caption(http_request, request.request_id, content, tag_source=request.tag_source,
        tags=request.tags, max_new_tokens=request.max_new_tokens, device=request.device,
        manager=manager, bundle_path=checkpoint)
    result = {**result, "dataset_id": request.dataset_id, "item_id": request.item_id,
              "dataset_revision": revision, "run_id": str(run_id), "checkpoint": checkpoint.name}
    await asyncio.to_thread(service.record_sample, run_id, result)
    return result
