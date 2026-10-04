"""Manage experimental FoveaTag runs without claiming production acceptance."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import threading
import uuid

from fastapi import HTTPException

_lifecycle_lock = threading.RLock()


def register_experiment(request, db):
    from core.tagger.foveatag.ui_bridge import import_experiment, MODES
    from database.models import TaggerTrainingRun, TaggerTrainingMetrics
    from database.training_detail_store import (
        RUN_DB_SCHEMA_VERSION, RUN_DB_V2, TAGGER_RUN_DB_FILENAME,
        initialize_tagger_detail_database, open_tagger_history_session,
    )
    directory = str(Path(request["comparison_dir"]).resolve())
    with _lifecycle_lock:
        if db.query(TaggerTrainingRun).filter(TaggerTrainingRun.output_dir == directory).first():
            raise HTTPException(409, "Comparison directory is already registered")
        try:
            imported = import_experiment(request)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise HTTPException(422, f"Cannot import experiment: {exc}") from exc
        args, saved = imported["arguments"], imported["snapshot"]
        branch = MODES.index(saved["mode"])
        nominal = math.ceil(args["epoch_samples"] / args["batch_size"]) * args["epochs"]
        vocab = saved["vocabulary"]
        config = {
            "architecture": "foveatag_phase3", "training_method": "full",
            "epochs": args["epochs"] * 3, "batch_size": args["batch_size"],
            "learning_rate": args["learning_rate"], "weight_decay": args["weight_decay"],
            "num_workers": args["workers"], "warmup_steps": args["warmup_steps"],
            "optimizer": "adamw", "loss_function": "asl", "head_lr_multiplier": 1.0,
            "foveatag": {"arguments": args, "active_mode": saved["mode"], "registered_experiment": True},
        }
        run = TaggerTrainingRun(
            run_id=str(uuid.uuid4()), run_name=request.get("run_name") or f"FoveaTag A/B/C — {Path(directory).name}",
            status="stopped", training_method="full", vision_encoder_path=args["pretrained_encoder"],
            output_dir=directory, config=config,
            dataset_configs=[{"dataset_id": int(key)} for key in args["train_counts"]],
            total_epochs=args["epochs"] * 3, total_steps=nominal * 3,
            current_epoch=branch * args["epochs"] + saved["epoch"],
            current_step=branch * nominal + saved["step"],
            progress=(branch * args["epochs"] + saved["epoch"] + saved["sample_offset"] / args["epoch_samples"]) / (3 * args["epochs"]),
            num_tags=vocab["num_tags"], best_f1=saved["best_f1"], best_threshold=saved["best_threshold"],
            latest_checkpoint_path=saved["checkpoint"],
            best_checkpoint_path=str(Path(directory) / saved["mode"] / "best_f1.safetensors"),
            status_message=f"FoveaTag {saved['mode']}: paused at saved step {saved['step']}",
            detail_store=RUN_DB_V2, detail_schema_version=RUN_DB_SCHEMA_VERSION,
            detail_state="ready", detail_db_name=TAGGER_RUN_DB_FILENAME,
        )
        if saved["history"]:
            run.latest_loss = saved["history"][-1]["validation"]["mean_loss"]
            run.latest_lr = saved["history"][-1]["learning_rate"]
        db.add(run)
        db.flush()
        initialize_tagger_detail_database(run)
        history_db, owns, detail_run = open_tagger_history_session(db, run.run_id)
        try:
            detail_run.tag_vocabulary = vocab
            for row in saved["history"]:
                validation = row["validation"]
                history_db.add(TaggerTrainingMetrics(
                    run_id=run.run_id, resume_seq=0, step=branch * nominal + row["global_step"],
                    epoch=branch * args["epochs"] + row["epoch"] + 1,
                    loss=validation["mean_loss"], f1=validation["macro_f1"],
                    precision=validation["macro_precision"], recall=validation["macro_recall"],
                    threshold=validation["threshold"], learning_rate=row["learning_rate"],
                ))
            history_db.commit()
        finally:
            if owns:
                history_db.close()
        db.commit()
        db.refresh(run)
        return run.to_list_dict()


def start_experiment(run, db, threads, callback_factory):
    from database import TrainingSessionLocal
    from database.models import TaggerTrainingRun
    from core.tagger.foveatag.run_control import FoveaRunControl
    from core.tagger.foveatag.ui_bridge import run_catalog_comparison, project_progress, require_no_external_worker
    from core.model_state_coordinator import ModelStateBusyError, model_state_coordinator

    run_id = run.run_id
    config = run.config
    arguments = config["foveatag"]["arguments"]
    activity = f"FoveaTag comparison {run_id}"
    with _lifecycle_lock:
        if run.status in ("running", "starting") or run_id in threads:
            raise HTTPException(409, "Comparison is already running")
        if run.status == "completed":
            raise HTTPException(409, "Comparison already completed")
        if any(entry.get("output_dir") == run.output_dir for entry in threads.values()):
            raise HTTPException(409, "Output directory already has a training worker")
        try:
            require_no_external_worker(Path(run.output_dir))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        callback = callback_factory(run_id, TrainingSessionLocal)
        try:
            model_state_coordinator.begin_activity(activity)
        except ModelStateBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        control = FoveaRunControl(run_id=run_id, output_dir=run.output_dir)
        prior_status = run.status

        def emit(event, data):
            for kind, projected in project_progress(event, data, arguments):
                callback(run_id, kind, projected)
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                cfg = dict(row.config)
                fovea = dict(cfg["foveatag"])
                fovea["active_mode"] = data["input_mode"]
                cfg["foveatag"] = fovea
                row.config = cfg
                row.status_message = f"FoveaTag {data['input_mode']}: {event}"
                if event == "checkpoint_saved":
                    from core.tagger.foveatag.ui_bridge import MODES
                    branch = MODES.index(data["input_mode"])
                    nominal = math.ceil(arguments["epoch_samples"] / arguments["batch_size"]) * arguments["epochs"]
                    row.current_step = branch * nominal + data["global_step"]
                    row.current_epoch = branch * arguments["epochs"] + data["epoch"]
                    row.progress = (row.current_epoch + data["sample_offset"] / arguments["epoch_samples"]) / (3 * arguments["epochs"])
                    row.latest_checkpoint_path = data["path"]
                    row.best_checkpoint_path = data["best_path"] if Path(data["best_path"]).is_file() else None
                    row.best_f1 = data["best_f1"]
                    row.best_threshold = data["best_threshold"]
                session.commit()
                if event == "checkpoint_saved":
                    try:
                        from api.tagger_diagnostics import schedule
                        schedule(row)
                    except Exception:
                        # Optional visualization must not fail the training run.
                        pass
            finally:
                session.close()

        def worker():
            session = TrainingSessionLocal()
            try:
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                row.status = "running"
                session.commit()
                result = run_catalog_comparison(config, emit, control)
                session.expire_all()
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).one()
                if result["status"] == "complete":
                    row.status, row.progress = "completed", 1.0
                    row.completed_at = datetime.now()
                else:
                    row.status = "stopped"
                row.status_message = f"FoveaTag: {result['status']}"
                session.commit()
            except Exception as exc:
                session.rollback()
                row = session.query(TaggerTrainingRun).filter_by(run_id=run_id).first()
                if row:
                    row.status, row.error_message = "failed", str(exc)
                    session.commit()
            finally:
                session.close()
                with _lifecycle_lock:
                    threads.pop(run_id, None)
                model_state_coordinator.end_activity(activity)

        thread = threading.Thread(target=worker, daemon=True, name=f"foveatag-{run_id[:8]}")
        threads[run_id] = {"thread": thread, "fovea_control": control, "output_dir": run.output_dir}
        try:
            run.status = "starting"
            run.started_at = run.started_at or datetime.now()
            run.last_resumed_at = datetime.now()
            run.resumed_from_step = run.current_step
            run.error_message = None
            db.commit()
            thread.start()
        except Exception:
            threads.pop(run_id, None)
            model_state_coordinator.end_activity(activity)
            db.rollback()
            run.status = prior_status
            db.commit()
            raise
        db.refresh(run)
        return {"message": "started", "run": run.to_list_dict()}


def stop_experiment(run, db, threads):
    with _lifecycle_lock:
        entry = threads.get(run.run_id)
        if entry and entry.get("fovea_control"):
            entry["fovea_control"].request_stop()
            run.status_message = "Saving FoveaTag resume checkpoint before stopping..."
            db.commit()
        elif run.status in ("running", "starting"):
            raise HTTPException(409, "Training handle is missing; cannot confirm a safe stop")
        db.refresh(run)
        return {"message": "stop_requested", "run": run.to_list_dict()}
