"""Catalog lifecycle for the owner-reviewed joint route, separate from A/B/C."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import uuid
from types import SimpleNamespace

from fastapi import HTTPException
from api.foveatag_training import _lifecycle_lock


def register_joint_review(request, db):
    import torch
    from safetensors.torch import save_file
    from database.models import TaggerTrainingRun
    from database.training_detail_store import RUN_DB_SCHEMA_VERSION, RUN_DB_V2, TAGGER_RUN_DB_FILENAME, initialize_tagger_detail_database
    from core.tagger.foveatag.artifact import FoveaArtifactConfig, InferenceMode
    from core.tagger.foveatag.supervised_run import _file_sha256, _write_json_atomic
    from api.tagger_diagnostics import configure
    from api.param_defaults import TAGGER_DIAGNOSTIC_DEFAULTS
    with _lifecycle_lock:
        source = db.query(TaggerTrainingRun).filter_by(run_id=request["source_run_id"]).first()
        if source is None:
            raise HTTPException(404, "Source run not found")
        if source.status != "stopped" or source.config.get("architecture") != "foveatag_phase3":
            raise HTTPException(409, "A stopped scratch FoveaTag comparison is required as initialization")
        root = Path(request["output_dir"]).resolve()
        if root.exists():
            raise HTTPException(409, "Use a new output directory for the new training contract")
        source_path = Path(source.latest_checkpoint_path)
        original = source.config["foveatag"]["arguments"]
        init = json.loads((Path(source.output_dir) / "shared_initialization_metadata.json").read_text())
        cfg = FoveaArtifactConfig.from_dict(init["initialization_contract"]["model"])
        cfg = replace(cfg, views=replace(cfg.views, overview_resize_mode="stretch", local_border_mode="shift_inside"),
            inference_modes={"balanced": InferenceMode(cfg.views.overview_size, request["selection_budget"])})
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "foveatag_supervised_training":
            raise HTTPException(422, "Source is not a scratch supervised checkpoint")
        root.mkdir(parents=True)
        bootstrap = root / "bootstrap.safetensors"
        save_file({"model." + name: tensor.contiguous() for name, tensor in payload["model"].items()}, str(bootstrap))
        metadata = bootstrap.with_name("bootstrap_metadata.json")
        _write_json_atomic(metadata, {"config": cfg.to_dict(), "vocabulary": {},
            "provenance": {"initialization": "scratch", "external_weights": [], "source_run_id": source.run_id,
                "source_step": payload["global_step"], "source_checkpoint_sha256": _file_sha256(source_path),
                "optimizer_reused": False, "input_policy_changed": True}})
        del payload
        keys = ("datasets_db", "vocabulary_path", "train_counts", "validation_counts", "domain_weights",
                "holdout_manifests", "photo_dataset_id", "photo_loss_weight", "seed", "max_source_pixels")
        plan = {key: original[key] for key in keys}
        plan.update({key: value for key, value in request.items() if key not in ("source_run_id", "run_name")})
        plan.update(kind="foveatag_direct_joint_review_v1", acceptance="owner_visual_review", output_dir=str(root),
            bootstrap=str(bootstrap), bootstrap_metadata=str(metadata), artifact_config=cfg.to_dict())
        plan["hashes"] = {key: _file_sha256(Path(plan[key])) for key in ("bootstrap", "bootstrap_metadata", "vocabulary_path", "datasets_db")}
        plan["holdout_hashes"] = {path: _file_sha256(Path(path)) for path in plan["holdout_manifests"]}
        _write_json_atomic(root / "review_plan.json", plan)
        nominal = math.ceil(plan["epoch_samples"] / plan["batch_size"]) * plan["epochs"]
        run = TaggerTrainingRun(run_id=str(uuid.uuid4()), run_name=request.get("run_name") or "FoveaTag learned router / owner review", status="pending",
            training_method="full", vision_encoder_path=str(bootstrap), output_dir=str(root),
            config={"architecture": "foveatag_joint_review", "training_method": "full", "epochs": plan["epochs"],
                "batch_size": plan["batch_size"], "learning_rate": plan["learning_rate"], "joint_review": plan},
            dataset_configs=source.dataset_configs, total_epochs=plan["epochs"], total_steps=nominal,
            num_tags=cfg.num_tags, detail_store=RUN_DB_V2, detail_schema_version=RUN_DB_SCHEMA_VERSION,
            detail_state="ready", detail_db_name=TAGGER_RUN_DB_FILENAME)
        db.add(run)
        db.flush()
        initialize_tagger_detail_database(run)
        configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True})
        db.commit()
        db.refresh(run)
        return run.to_list_dict()


def start_joint_review(run, db, threads, callback_factory, *, continuous=False):
    from database import TrainingSessionLocal
    from database.models import TaggerTrainingRun
    from core.model_state_coordinator import model_state_coordinator, ModelStateBusyError
    from core.tagger.foveatag.run_control import FoveaRunControl
    from core.tagger.foveatag.direct_joint import run_direct_joint
    from api.tagger_diagnostics import training_due, schedule_training
    run_id = run.run_id
    activity = "FoveaTag joint review " + run_id
    with _lifecycle_lock:
        db.refresh(run)
        plan = run.config["joint_review"]
        diagnostic_run = SimpleNamespace(run_id=run_id, output_dir=run.output_dir, config=run.config)
        if run.status in ("running", "starting", "completed") or run_id in threads:
            raise HTTPException(409, "Joint review is already active or complete")
        try:
            model_state_coordinator.begin_activity(activity)
        except ModelStateBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        control = FoveaRunControl(run_id=run_id, output_dir=run.output_dir)
        callback = callback_factory(run_id, TrainingSessionLocal)

        def emit(event, data):
            if event == "train":
                callback(run_id, "step", {"step": data["global_step"], "epoch": data["epoch"] + 1,
                    "loss": data["classification"], "lr": data.get("lr"), "progress": (data["epoch"] + data["sample_offset"] / plan["epoch_samples"]) / plan["epochs"]})
            elif event == "validation":
                callback(run_id, "epoch", {"step": data["global_step"], "epoch": data["epoch"] + 1,
                    "loss": data["mean_loss"], "f1": data["macro_f1"], "precision": data["macro_precision"],
                    "recall": data["macro_recall"], "threshold": data["threshold"]})
            elif event == "resumed":
                callback(run_id, "resume", {"resumed_from_step": data["global_step"]})
            elif event == "train_f1":
                callback(run_id, "train_f1", data)
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status_message = f"FoveaTag learned routing: {event}"
                if event == "checkpoint_saved":
                    import torch
                    saved = torch.load(data["path"], map_location="cpu", weights_only=False)
                    row.current_step, row.current_epoch = saved["global_step"], saved["epoch"] + 1
                    row.progress = (saved["epoch"] + saved["sample_offset"] / plan["epoch_samples"]) / plan["epochs"]
                    row.latest_checkpoint_path = data["path"]
                    row.best_f1, row.best_threshold = saved["best_validation_f1"], saved["best_threshold"]
                    del saved
                session.commit()
            finally:
                session.close()

        def diagnostic(snapshot):
            try:
                schedule_training(diagnostic_run, snapshot)
            except Exception as exc:
                from api.tagger_diagnostics import directory, write_json
                try:
                    write_json(directory(diagnostic_run) / "error.json", {"error": f"Training-step diagnostic failed ({type(exc).__name__})"})
                except OSError:
                    pass

        def worker():
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status = "running"
                session.commit()
                checkpoint = Path(row.output_dir) / "latest.pt"
                from api.param_defaults import TAGGER_TRAINING_DEFAULTS
                metric_keys = ("save_tag_metrics", "train_f1_eval_every_n_steps", "train_f1_threshold_search_every_n_steps",
                    "train_f1_initial_threshold", "train_f1_buffer_batches")
                result = run_direct_joint(plan, progress_callback=emit, run_control=control,
                    resume=checkpoint if checkpoint.exists() else None,
                    segment_steps=None if continuous else plan["review_steps"],
                    diagnostic_due=lambda step: training_due(diagnostic_run, step), diagnostic_callback=diagnostic,
                    common_metrics_config={key: plan.get(key, TAGGER_TRAINING_DEFAULTS[key]) for key in metric_keys})
                session.expire_all()
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status = "completed" if result["status"] == "complete" else "stopped"
                row.status_message = "Saved routed checkpoint; awaiting owner visual review"
                row.current_step = result["global_step"]
                session.commit()
            except Exception as exc:
                session.rollback()
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status, row.error_message = "failed", f"Joint training failed ({type(exc).__name__})"
                session.commit()
            finally:
                control.detach()
                session.close()
                with _lifecycle_lock:
                    threads.pop(run_id, None)
                model_state_coordinator.end_activity(activity)

        thread = threading.Thread(target=worker, daemon=True, name="fovea-joint-" + run_id[:8])
        threads[run_id] = {"thread": thread, "fovea_control": control, "output_dir": run.output_dir}
        run.status, run.error_message = "starting", None
        run.status_message = "FoveaTag learned routing: preparing"
        run.started_at = run.started_at or datetime.now()
        run.last_resumed_at = datetime.now()
        run.resumed_from_step = run.current_step
        db.commit()
        try:
            thread.start()
        except Exception:
            threads.pop(run_id, None)
            model_state_coordinator.end_activity(activity)
            run.status = "failed"
            db.commit()
            raise
        return {"message": "started", "run": run.to_list_dict()}


def edit_joint_config(run_id, updates, db):
    from api.routes import FoveaJointReviewRequest
    from api.param_defaults import TAGGER_JOINT_REVIEW_DEFAULTS, TAGGER_TRAINING_DEFAULTS
    from core.tagger.foveatag.joint_checkpoint import EDITABLE_JOINT_FIELDS
    from core.tagger.foveatag.supervised_run import _write_json_atomic
    from database.models import TaggerTrainingRun
    from pydantic import ValidationError
    with _lifecycle_lock:
        run = db.query(TaggerTrainingRun).filter_by(run_id=run_id).first()
        if run is None:
            raise HTTPException(404, "Run not found")
        if run.config.get("architecture") != "foveatag_joint_review":
            raise HTTPException(422, "A FoveaTag joint run is required")
        if run.status not in ("pending", "stopped", "failed"):
            raise HTTPException(409, "Stop the run before saving training settings")
        if set(updates) - set(EDITABLE_JOINT_FIELDS):
            raise HTTPException(422, "Dataset, model topology and input policy cannot be edited in this run")
        original = run.config["joint_review"]
        try:
            request = FoveaJointReviewRequest(source_run_id="config-edit", output_dir=run.output_dir,
                **{**{key: original[key] for key in (*TAGGER_JOINT_REVIEW_DEFAULTS, *EDITABLE_JOINT_FIELDS) if key in original}, **updates})
        except ValidationError as exc:
            raise HTTPException(422, "Invalid training values or routing budget") from exc
        if request.epochs < (run.current_epoch or 0):
            raise HTTPException(422, "Epoch target cannot precede the saved epoch")
        plan = {**original, **{key: request.model_dump()[key] for key in EDITABLE_JOINT_FIELDS}}
        changes = {key: {"before": original.get(key, TAGGER_TRAINING_DEFAULTS.get(key)), "after": plan[key]}
            for key in EDITABLE_JOINT_FIELDS if original.get(key, TAGGER_TRAINING_DEFAULTS.get(key)) != plan[key]}
        if not changes:
            return run.to_list_dict()
        revisions = [*run.config.get("joint_config_history", []),
            {"saved_step": run.current_step, "updated_at": datetime.now().isoformat(), "changes": changes}]
        run.config = {**run.config, "joint_review": plan, "joint_config_history": revisions,
            "epochs": plan["epochs"], "batch_size": plan["batch_size"], "learning_rate": plan["learning_rate"]}
        run.total_epochs = plan["epochs"]
        run.total_steps = math.ceil(plan["epoch_samples"] / plan["batch_size"]) * plan["epochs"]
        root = Path(run.output_dir)
        try:
            _write_json_atomic(root / "review_plan.json", plan)
            _write_json_atomic(root / "config_revisions.json", revisions)
            db.commit()
        except Exception:
            db.rollback()
            _write_json_atomic(root / "review_plan.json", original)
            raise
        db.refresh(run)
        return run.to_list_dict()
