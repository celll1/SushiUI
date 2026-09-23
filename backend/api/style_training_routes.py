"""REST lifecycle of style encoder training runs (docs/plans/STYLE_EMBEDDING_DESIGN_2026-09.md §12).

Runs train in a backend thread like tagger runs. The in-process registry `_active` is authoritative for
whether a run is live: a catalogue status of starting/running/stopping without a registry entry is left
over from a previous backend process.
"""

from __future__ import annotations

import copy
import os
import threading
import traceback
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, create_model
from sqlalchemy.orm import Session

from api.error_handlers import APIError, NotFoundError, ValidationError
from api.param_defaults import STYLE_TRAINING_DEFAULTS, STYLE_TRAINING_METRICS_QUERY_DEFAULTS
from api.training_run_helpers import (
    LazySerialExecutor,
    decimate_metric_rows,
    find_output_dir_owner,
    next_resume_seq,
    upsert_metric_row,
    validate_gpu_index,
)
from api.websocket import manager
from config.settings import settings
from database import get_training_db
from database.models import StyleTrainingMetrics, StyleTrainingRun
from database.training_cleanup import delete_style_training_run_record
from database.training_detail_store import (
    RUN_DB_SCHEMA_VERSION,
    RUN_DB_V2,
    STYLE_RUN_DB_FILENAME,
    DetailStoreError,
    initialize_style_detail_database,
    open_style_detail_session,
    open_style_history_session,
)

router = APIRouter()
_D = STYLE_TRAINING_DEFAULTS
_Q = STYLE_TRAINING_METRICS_QUERY_DEFAULTS

# Request fields that are not trainer config keys.
RUN_FIELDS = ("run_name", "output_dir")
ACTIVE_STATUSES = ("starting", "running", "stopping")


class StyleTrainingRunCreateRequest(BaseModel):
    """Every STYLE_TRAINING_DEFAULTS key plus the run fields. Pydantic owns only JSON types and unknown keys
    (422); every value bound and cross-field constraint is the trainer's StyleTrainingConfig (400)."""
    model_config = ConfigDict(extra="forbid")

    run_name: Optional[str] = None
    output_dir: Optional[str] = None
    dataset_configs: List[Dict[str, Any]]

    vision_encoder_path: str = _D["vision_encoder_path"]
    vision_encoder_repo: str = _D["vision_encoder_repo"]
    training_method: str = _D["training_method"]
    lora_rank: StrictInt = _D["lora_rank"]
    lora_alpha: float = _D["lora_alpha"]
    lora_dropout: float = _D["lora_dropout"]
    attn_implementation: str = _D["attn_implementation"]
    mixed_precision: str = _D["mixed_precision"]
    gradient_checkpointing: StrictBool = _D["gradient_checkpointing"]
    gpu_index: Optional[StrictInt] = _D["gpu_index"]
    optimizer: str = _D["optimizer"]
    learning_rate: float = _D["learning_rate"]
    head_lr_multiplier: float = _D["head_lr_multiplier"]
    weight_decay: float = _D["weight_decay"]
    warmup_steps: StrictInt = _D["warmup_steps"]
    min_lr_ratio: float = _D["min_lr_ratio"]
    max_grad_norm: float = _D["max_grad_norm"]
    epochs: StrictInt = _D["epochs"]
    num_workers: StrictInt = _D["num_workers"]
    save_every_n_steps: StrictInt = _D["save_every_n_steps"]
    keep_last_n_checkpoints: StrictInt = _D["keep_last_n_checkpoints"]
    validate_every: StrictInt = _D["validate_every"]
    log_every_n_steps: StrictInt = _D["log_every_n_steps"]
    style_seed: StrictInt = _D["style_seed"]
    style_split_seed: StrictInt = _D["style_split_seed"]
    style_use_tag_aliases: StrictBool = _D["style_use_tag_aliases"]
    style_use_gelbooru_categories: StrictBool = _D["style_use_gelbooru_categories"]
    style_signature_tags: List[str] = _D["style_signature_tags"]
    style_non_copyright_tags: List[str] = _D["style_non_copyright_tags"]
    style_val_artist_ratio: float = _D["style_val_artist_ratio"]
    style_val_image_ratio: float = _D["style_val_image_ratio"]
    style_min_images_per_artist: StrictInt = _D["style_min_images_per_artist"]
    style_artists_per_batch: StrictInt = _D["style_artists_per_batch"]
    style_images_per_artist: StrictInt = _D["style_images_per_artist"]
    style_same_copyright_batch_ratio: float = _D["style_same_copyright_batch_ratio"]
    style_artist_sampling_power: float = _D["style_artist_sampling_power"]
    style_mask_ratio_min: float = _D["style_mask_ratio_min"]
    style_mask_ratio_max: float = _D["style_mask_ratio_max"]
    style_max_num_patches: StrictInt = _D["style_max_num_patches"]
    style_temperature: float = _D["style_temperature"]
    style_view_positive_weight: float = _D["style_view_positive_weight"]
    style_positive_weight_min: float = _D["style_positive_weight_min"]
    style_feature_layers: List[StrictInt] = _D["style_feature_layers"]
    style_token_dim: StrictInt = _D["style_token_dim"]
    style_pool_heads: StrictInt = _D["style_pool_heads"]
    style_projector_hidden_dim: StrictInt = _D["style_projector_hidden_dim"]
    style_embed_dim: StrictInt = _D["style_embed_dim"]
    style_val_max_items: StrictInt = _D["style_val_max_items"]
    style_val_batch_size: StrictInt = _D["style_val_batch_size"]
    style_eval_chunk_size: StrictInt = _D["style_eval_chunk_size"]


# The create fields without defaults (an omitted field is unchanged) and with the same types, so null for a
# non-nullable key is a 422 here too. output_dir holds the detail database and checkpoints: fixed at creation.
StyleTrainingRunUpdateRequest = create_model(
    "StyleTrainingRunUpdateRequest",
    __config__=ConfigDict(extra="forbid"),
    # field.annotation alone drops Strict() (kept in field.metadata), so re-attach the metadata.
    **{name: (Annotated[(field.annotation, *field.metadata)] if field.metadata else field.annotation, None)
       for name, field in StyleTrainingRunCreateRequest.model_fields.items() if name != "output_dir"},
)


class _ActiveRun:
    def __init__(self) -> None:
        self.thread: Optional[threading.Thread] = None
        self.trainer_holder: list = []
        self.stop_requested = False

    def forward_stop(self) -> None:
        if self.stop_requested and self.trainer_holder:
            self.trainer_holder[0].stop()


_active: Dict[str, _ActiveRun] = {}
_active_lock = threading.Lock()
_detail_writer = LazySerialExecutor("style_detail_db")


def _session_factory():
    from database import TrainingSessionLocal
    return TrainingSessionLocal


def _trainer_entry():
    # Deferred: the trainer stack is heavy and only a started run needs it.
    from core.style_encoder.style_trainer import run_style_training
    return run_style_training


def _validate_trainer_config(config: dict) -> None:
    from core.style_encoder.style_trainer import StyleTrainingConfig
    try:
        StyleTrainingConfig.from_dict(config)
    except ValueError as exc:
        raise ValidationError("Invalid style training configuration", detail=str(exc)) from exc


def _trainer_config(values: dict) -> dict:
    return {key: value for key, value in values.items() if key not in RUN_FIELDS}


def _get_run(db: Session, run_id: str) -> StyleTrainingRun:
    run = db.query(StyleTrainingRun).filter(StyleTrainingRun.run_id == run_id).first()
    if run is None:
        raise NotFoundError("Style training run not found", detail=f"run_id={run_id!r}")
    return run


def _current_run(db: Session, run_id: str) -> StyleTrainingRun:
    """The committed row as of now; call under `_active_lock` before deciding on it."""
    run = _get_run(db, run_id)
    db.refresh(run)
    return run


def _complete_config(run: StyleTrainingRun, config: dict) -> tuple[dict, list[str]]:
    """A stored config from before a key existed gets that key's default, except a key a resume refuses to
    change once the run has checkpoints: its default need not be what the run trained with."""
    missing = [key for key in _D if key not in config]
    if not missing:
        return config, []
    from core.style_encoder.style_artifacts import resumable_bundle_files
    from core.style_encoder.style_trainer import resume_refusing_keys
    refusing = [k for k in missing if k in resume_refusing_keys(config.get("training_method", _D["training_method"]))]
    if refusing and run.output_dir and resumable_bundle_files(run.output_dir):
        raise APIError("The stored configuration lacks keys its checkpoints depend on", 409,
                       detail=f"run {run.run_id} has checkpoints in {run.output_dir} but no stored value for "
                              f"{refusing}; PATCH them to the values the run was trained with")
    return {**{key: copy.deepcopy(_D[key]) for key in missing}, **config}, missing


def _record_filled_defaults(run_id: str, filled: list[str]) -> None:
    if not filled:
        return
    message = f"Stored configuration had no value for {filled}; filled from the current defaults"
    print(f"[StyleTraining] {run_id}: {message}")
    warning = {"message": message, "timestamp": datetime.now().isoformat(), "filled_defaults": filled}
    _detail_writer.get().submit(_write_detail, _session_factory(), run_id, 0, None, {"warning": warning})


def _run_payload(run: StyleTrainingRun) -> dict:
    return {**run.to_dict(), "active": run.run_id in _active}


def _apply_config(run: StyleTrainingRun, config: dict) -> None:
    run.config = config
    run.training_method = config["training_method"]
    run.vision_encoder_path = config["vision_encoder_path"]
    run.vision_encoder_repo = config["vision_encoder_repo"]
    run.dataset_configs = config["dataset_configs"]
    run.total_epochs = config["epochs"]


# --------------------------------------------------------------------- progress


def _write_detail(factory, run_id: str, resume_seq: int, metric: Optional[tuple], detail: dict) -> None:
    catalog_db = factory()
    history_db, owns_history_db = None, False
    try:
        history_db, owns_history_db, _ = open_style_history_session(catalog_db, run_id)
        if metric is not None:
            upsert_metric_row(history_db, StyleTrainingMetrics, run_id, resume_seq, metric[0], metric[1])
        if detail:
            local = history_db.query(StyleTrainingRun).filter(StyleTrainingRun.run_id == run_id).one()
            if "config" in detail:
                local.config = detail["config"]
            if "dataset_stats" in detail:
                local.dataset_stats = detail["dataset_stats"]
            if "warning" in detail:
                local.warnings = [*(local.warnings or []), detail["warning"]]
        history_db.commit()
    except Exception as exc:  # noqa: BLE001 - history loss must not end the run
        if history_db is not None:
            history_db.rollback()
        print(f"[StyleTraining] history write error for {run_id}: {exc}")
    finally:
        if owns_history_db and history_db is not None:
            history_db.close()
        catalog_db.close()


def _make_progress_callback(run_id: str, factory, entry: _ActiveRun):
    """Catalogue updates inline, history rows through one ordered writer. Raises if the run's detail store
    cannot be opened, so the run fails instead of training without history."""
    catalog_db = factory()
    try:
        history_db, owns_history_db, _ = open_style_history_session(catalog_db, run_id)
        try:
            resume_seq = next_resume_seq(history_db, StyleTrainingMetrics, run_id)
        finally:
            if owns_history_db:
                history_db.close()
    finally:
        catalog_db.close()
    state: Dict[str, Any] = {"index_hash": None, "last_map_at_r": None}

    def submit(metric: Optional[tuple] = None, detail: Optional[dict] = None) -> None:
        _detail_writer.get().submit(_write_detail, factory, run_id, resume_seq, metric, detail or {})

    def callback(rid: str, event: str, data: dict) -> None:
        entry.forward_stop()
        metric, detail = None, {}
        db = factory()
        try:
            run = db.query(StyleTrainingRun).filter(StyleTrainingRun.run_id == rid).first()
            if run is None:
                return
            if event == "step":
                run.current_step = data["step"]
                run.current_epoch = data.get("epoch", run.current_epoch)
                run.total_steps = data.get("total_steps", run.total_steps)
                run.progress = data.get("progress", run.progress)
                run.latest_loss = data.get("loss")
                run.latest_lr = data.get("lr")
                run.status_message = None
                index_hash = data.get("index_hash") or state["index_hash"]
                metric = (data["step"], {"epoch": data.get("epoch"), "index_hash": index_hash,
                                         "loss": data.get("loss"), "supcon_loss": data.get("supcon_loss"),
                                         "learning_rate": data.get("lr"), "grad_norm": data.get("grad_norm")})
                manager.send_style_metrics(rid, "step", data["step"], resume_seq, epoch=data.get("epoch"),
                                           loss=data.get("loss"), lr=data.get("lr"),
                                           grad_norm=data.get("grad_norm"), progress=data.get("progress"),
                                           index_hash=index_hash)
            elif event == "epoch":
                run.current_epoch = data.get("epoch", run.current_epoch)
                metric = (data["step"], {"epoch": data.get("epoch"), "epoch_loss": data.get("loss"),
                                         "index_hash": state["index_hash"]})
                manager.send_style_metrics(rid, "epoch", data["step"], resume_seq, epoch=data.get("epoch"),
                                           epoch_loss=data.get("loss"), index_hash=state["index_hash"])
            elif event == "validation":
                held_out = data.get("held_out_artists") or {}
                state["last_map_at_r"] = held_out.get("map_at_r")
                values = {"val_map_at_r": held_out.get("map_at_r"), "val_recall_at_1": held_out.get("recall_at_1"),
                          "val_recall_at_5": held_out.get("recall_at_5")}
                metric = (data["global_step"], {"epoch": data.get("epoch"), "index_hash": data.get("index_hash"),
                                                "evaluation": data, **values})
                manager.send_style_metrics(rid, "validation", data["global_step"], resume_seq,
                                           epoch=data.get("epoch"), index_hash=data.get("index_hash"), **values)
            elif event == "checkpoint":
                if data.get("name") == "best":
                    run.best_checkpoint_path = data.get("path")
                    run.best_step = data.get("step")
                    run.best_map_at_r = state["last_map_at_r"]
                elif data.get("name") == "latest":
                    run.latest_checkpoint_path = data.get("path")
            elif event == "dataset_stats":
                state["index_hash"] = data.get("index_hash")
                run.index_hash = data.get("index_hash")
                detail["dataset_stats"] = data
            elif event == "resume":
                run.resumed_from_step = data.get("resumed_from_step")
                run.last_resume_mode = data.get("mode")
                run.last_resumed_at = datetime.now()
            elif event == "warning":
                detail["warning"] = {"message": data.get("message"), "timestamp": datetime.now().isoformat(),
                                     **({"ignored": data["ignored"]} if "ignored" in data else {})}
            elif event == "phase":
                run.status_message = data.get("message") or data.get("phase")
            elif event == "dataset_progress":
                message = data.get("message") or "Loading dataset..."
                manager.send_progress_sync(int(data.get("step", 0)), int(data.get("total", 1)), message)
                run.status_message = message
            db.commit()
        except Exception as exc:  # noqa: BLE001 - a progress write must not end the run
            db.rollback()
            print(f"[StyleTraining] progress write error for {rid}: {exc}")
        finally:
            db.close()
        if metric is not None or detail:
            submit(metric, detail)

    callback.submit = submit
    return callback


def _set_run_columns(db: Session, run_id: str, only_if: Optional[tuple] = None, **values) -> int:
    """One UPDATE; with `only_if`, applied only while the status is one of those (the check and the write
    cannot interleave with another writer). Returns the rows changed; the caller commits."""
    query = db.query(StyleTrainingRun).filter(StyleTrainingRun.run_id == run_id)
    if only_if is not None:
        query = query.filter(StyleTrainingRun.status.in_(only_if))
    return query.update(values, synchronize_session=False)


def _update_run(factory, run_id: str, only_if: Optional[tuple] = None, **values) -> None:
    db = factory()
    try:
        _set_run_columns(db, run_id, only_if, **values)
        db.commit()
    finally:
        db.close()


def _run_thread(run_id: str, config: dict, output_dir: str, entry: _ActiveRun, activity: str) -> None:
    from core.model_state_coordinator import model_state_coordinator
    factory = _session_factory()
    try:
        callback = _make_progress_callback(run_id, factory, entry)
        callback.submit(detail={"config": config})
        _update_run(factory, run_id, only_if=("starting",), status="running")
        result = _trainer_entry()(run_id, config, output_dir, callback, entry.trainer_holder)
        best = result.get("best") or {}
        final = {"status": "stopped" if result.get("stopped") else "completed", "status_message": None,
                 "current_step": result.get("global_step"), "total_steps": result.get("total_steps")}
        if best:
            final.update(best_map_at_r=best.get("map_at_r"), best_step=best.get("global_step"))
        if final["status"] == "completed":
            final.update(progress=1.0, completed_at=datetime.now())
        _update_run(factory, run_id, **final)
    except Exception as exc:  # noqa: BLE001 - reported on the run
        print(f"[StyleTraining] Run {run_id} failed: {exc}")
        traceback.print_exc()
        try:
            _update_run(factory, run_id, status="failed", error_message=str(exc), status_message=None)
        except Exception as db_exc:  # noqa: BLE001
            print(f"[StyleTraining] could not record the failure of {run_id}: {db_exc}")
    finally:
        with _active_lock:
            _active.pop(run_id, None)
        model_state_coordinator.end_activity(activity)


# ----------------------------------------------------------------------- routes


@router.post("/style-training/runs")
def create_style_training_run(request: StyleTrainingRunCreateRequest,
                              training_db: Session = Depends(get_training_db)):
    """Create a style training run (catalogue row + per-run detail database)."""
    config = _trainer_config(request.model_dump())
    _validate_trainer_config(config)
    run_id = str(uuid.uuid4())
    output_dir = os.path.abspath(request.output_dir) if request.output_dir else \
        os.path.join(settings.root_dir, "style_models", run_id)
    owner = find_output_dir_owner(training_db, output_dir)
    if owner is not None:
        raise APIError("Output directory belongs to another training run", 409,
                       detail=f"{output_dir} is the output directory of {owner}")
    run = StyleTrainingRun(
        run_id=run_id,
        run_name=request.run_name or f"style-{run_id[:8]}",
        status="pending",
        output_dir=output_dir,
        detail_store=RUN_DB_V2,
        detail_schema_version=RUN_DB_SCHEMA_VERSION,
        detail_state="ready",
        detail_db_name=STYLE_RUN_DB_FILENAME,
    )
    _apply_config(run, config)
    training_db.add(run)
    training_db.flush()
    try:
        initialize_style_detail_database(run)
    except DetailStoreError as exc:
        training_db.rollback()
        raise APIError("Output directory holds another run's detail database", 409, detail=str(exc)) from exc
    training_db.commit()
    training_db.refresh(run)
    return _run_payload(run)


@router.get("/style-training/runs")
def list_style_training_runs(training_db: Session = Depends(get_training_db)):
    """List style training runs from the catalogue only (no detail database is opened)."""
    runs = training_db.query(StyleTrainingRun).order_by(StyleTrainingRun.created_at.desc()).all()
    return [_run_payload(r) for r in runs]


@router.get("/style-training/runs/{run_id}")
def get_style_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Catalogue row plus the detail store's dataset statistics and warnings."""
    run = _get_run(training_db, run_id)
    payload = {**_run_payload(run), "dataset_stats": None, "warnings": [], "detail_error": None}
    try:
        detail_db = open_style_detail_session(run)
    except DetailStoreError as exc:
        payload["detail_error"] = str(exc)
        return payload
    try:
        local = detail_db.query(StyleTrainingRun).filter(StyleTrainingRun.run_id == run_id).one()
        payload["dataset_stats"] = local.dataset_stats
        payload["warnings"] = local.warnings or []
    finally:
        detail_db.close()
    return payload


@router.patch("/style-training/runs/{run_id}")
def update_style_training_run(run_id: str, request: StyleTrainingRunUpdateRequest,
                              training_db: Session = Depends(get_training_db)):
    """Change the given fields of a run that is not active; the merged config is re-validated. A resumed
    run keeps its model identity and schedule (the trainer refuses or warns)."""
    _get_run(training_db, run_id)
    changes = request.model_dump(exclude_unset=True)
    run_name = changes.pop("run_name", None)
    with _active_lock:  # merged onto the committed row, so concurrent PATCHes do not lose updates
        run = _current_run(training_db, run_id)
        if run_id in _active:
            raise APIError("Cannot edit an active style training run", 409,
                           detail=f"run {run_id} is {run.status}; stop it first")
        config, filled = _complete_config(run, {**(run.config or {}), **changes})
        _validate_trainer_config(config)
        _apply_config(run, config)
        if run_name:
            run.run_name = run_name
        training_db.commit()
    _record_filled_defaults(run_id, filled)
    training_db.refresh(run)
    return {**_run_payload(run), "filled_defaults": filled}


@router.post("/style-training/runs/{run_id}/start")
def start_style_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Start or resume a run in a backend thread; the trainer resumes from the furthest complete bundle in
    the output directory."""
    from core.model_state_coordinator import ModelStateBusyError, model_state_coordinator

    _get_run(training_db, run_id)
    activity = f"style training run {run_id}"
    with _active_lock:  # the config launched is the committed one, not a read a concurrent PATCH overtook
        run = _current_run(training_db, run_id)
        if run_id in _active:
            raise APIError("Style training run is already active", 409,
                           detail=f"run {run_id} is {run.status}")
        config, filled = _complete_config(run, dict(run.config or {}))
        _validate_trainer_config(config)
        validate_gpu_index(config.get("gpu_index"))
        try:
            model_state_coordinator.begin_activity(activity)
        except ModelStateBusyError as exc:
            raise APIError("The loaded model is changing", 409, detail=str(exc)) from exc
        previous = {"status": run.status, "started_at": run.started_at, "error_message": run.error_message,
                    "completed_at": run.completed_at, "status_message": run.status_message, "config": run.config}
        entry = _ActiveRun()
        entry.thread = threading.Thread(target=_run_thread, args=(run_id, config, run.output_dir, entry, activity),
                                        daemon=True, name=f"style-{run_id[:8]}")
        _active[run_id] = entry
        try:
            if filled:
                _apply_config(run, config)
            run.status = "starting"
            run.started_at = datetime.now()
            run.error_message = None
            run.completed_at = None
            run.status_message = None
            training_db.commit()
            entry.thread.start()
        except BaseException:
            _active.pop(run_id, None)
            model_state_coordinator.end_activity(activity)
            training_db.rollback()
            for key, value in previous.items():
                setattr(run, key, value)
            training_db.commit()
            raise
    _record_filled_defaults(run_id, filled)
    training_db.refresh(run)
    return {"message": "started", "run": _run_payload(run), "filled_defaults": filled}


@router.post("/style-training/runs/{run_id}/stop")
def stop_style_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Ask an active run to stop; it saves a resumable checkpoint when it has trained at least one step."""
    run = _get_run(training_db, run_id)
    with _active_lock:
        entry = _active.get(run_id)
        if entry is None:
            marked = _set_run_columns(training_db, run_id, ACTIVE_STATUSES, status="stopped",
                                      status_message="No live trainer in this backend process; marked stopped")
            training_db.commit()
            training_db.refresh(run)
            if not marked:
                raise APIError("Style training run is not active", 409, detail=f"run {run_id} is {run.status}")
            return {"message": "marked_stopped", "run": _run_payload(run)}
        entry.stop_requested = True
        entry.forward_stop()
        # Conditional: the trainer thread may already have written its final status.
        _set_run_columns(training_db, run_id, ("starting", "running"), status="stopping")
        training_db.commit()
    training_db.refresh(run)
    return {"message": "stop_requested", "run": _run_payload(run)}


@router.delete("/style-training/runs/{run_id}")
def delete_style_training_run(run_id: str, training_db: Session = Depends(get_training_db)):
    """Delete the catalogue row; the output directory, its checkpoints and detail database are kept."""
    run = _get_run(training_db, run_id)
    with _active_lock:
        if run_id in _active:
            raise APIError("Cannot delete an active style training run", 409,
                           detail=f"run {run_id} is {run.status}; stop it first")
        delete_style_training_run_record(training_db, run)
        training_db.commit()
    return {"deleted": run_id}


@router.get("/style-training/runs/{run_id}/metrics")
def get_style_training_metrics(run_id: str,
                               since_step: int = Query(_Q["since_step"], ge=0),
                               max_points: int = Query(_Q["max_points"], ge=0),
                               training_db: Session = Depends(get_training_db)):
    """Per-step and evaluation rows from the run's detail database, ordered by (resume_seq, step)."""
    _get_run(training_db, run_id)
    try:
        history_db, owns_history_db, _ = open_style_history_session(training_db, run_id)
    except DetailStoreError as exc:
        raise NotFoundError("Style training history is unavailable", detail=str(exc)) from exc
    try:
        rows = (history_db.query(StyleTrainingMetrics)
                .filter(StyleTrainingMetrics.run_id == run_id, StyleTrainingMetrics.step >= since_step)
                .order_by(StyleTrainingMetrics.resume_seq, StyleTrainingMetrics.step)
                .all())
        return decimate_metric_rows(rows, max_points, lambda m: m.evaluation is not None)
    finally:
        if owns_history_db:
            history_db.close()
