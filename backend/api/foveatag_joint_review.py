"""Catalog lifecycle for the owner-reviewed joint route, separate from A/B/C."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import uuid

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


def start_joint_review(run, db, threads, callback_factory):
    from database import TrainingSessionLocal
    from database.models import TaggerTrainingRun
    from core.model_state_coordinator import model_state_coordinator, ModelStateBusyError
    from core.tagger.foveatag.run_control import FoveaRunControl
    from core.tagger.foveatag.direct_joint import run_direct_joint
    from api.tagger_diagnostics import schedule
    run_id, plan = run.run_id, run.config["joint_review"]
    activity = "FoveaTag joint review " + run_id
    with _lifecycle_lock:
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
                    "loss": data["classification"], "progress": (data["epoch"] + data["sample_offset"] / plan["epoch_samples"]) / plan["epochs"]})
            elif event == "validation":
                callback(run_id, "epoch", {"step": data["global_step"], "epoch": data["epoch"] + 1,
                    "loss": data["mean_loss"], "f1": data["macro_f1"], "precision": data["macro_precision"],
                    "recall": data["macro_recall"], "threshold": data["threshold"]})
            elif event == "resumed":
                callback(run_id, "resume", {"resumed_from_step": data["global_step"]})
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status_message = f"FoveaTag learned routing: {event}; owner review pending"
                if event == "checkpoint_saved":
                    import torch
                    saved = torch.load(data["path"], map_location="cpu", weights_only=False)
                    row.current_step, row.current_epoch = saved["global_step"], saved["epoch"] + 1
                    row.progress = (saved["epoch"] + saved["sample_offset"] / plan["epoch_samples"]) / plan["epochs"]
                    row.latest_checkpoint_path = data["path"]
                    row.best_f1, row.best_threshold = saved["best_validation_f1"], saved["best_threshold"]
                    del saved
                session.commit()
                if event == "checkpoint_saved":
                    try:
                        schedule(row)
                    except Exception:
                        pass
            finally:
                session.close()

        def worker():
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status = "running"
                session.commit()
                checkpoint = Path(row.output_dir) / "latest.pt"
                result = run_direct_joint(plan, progress_callback=emit, run_control=control,
                    resume=checkpoint if checkpoint.exists() else None, segment_steps=plan["review_steps"])
                session.expire_all()
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status = "completed" if result["status"] == "complete" else "stopped"
                row.status_message = "Saved routed checkpoint; awaiting owner visual review"
                row.current_step = result["global_step"]
                session.commit()
                from api.tagger_diagnostics import status as diagnostic_status
                if diagnostic_status(row)["settings"]["enabled"] and checkpoint.exists():
                    schedule(row, force=True)
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
