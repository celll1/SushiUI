"""Resolve the detailed-history store declared by a training run."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os

from sqlalchemy import create_engine, event, or_, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


CENTRAL_V1 = "central_v1"
RUN_DB_V2 = "run_db_v2"
RUN_DB_FILENAME = "training_run.db"
TAGGER_RUN_DB_FILENAME = "tagger_training_run.db"
SUPPORTED_DETAIL_STORES = frozenset((CENTRAL_V1, RUN_DB_V2))
RUN_DB_SCHEMA_VERSION = 2
TERMINAL_RUN_STATUSES = frozenset(("completed", "failed", "stopped"))


class DetailStoreError(ValueError):
    """The catalogue points at an invalid or unsupported detail store."""


@dataclass(frozen=True)
class DetailStoreLocation:
    kind: str
    path: Optional[Path]
    schema_version: Optional[int]
    state: Optional[str]


def detail_store_kind(run) -> str:
    """Return a normalized store kind; NULL is the backward-compatible v1."""
    kind = getattr(run, "detail_store", None) or CENTRAL_V1
    if kind not in SUPPORTED_DETAIL_STORES:
        raise DetailStoreError(f"Unsupported training detail store: {kind!r}")
    return kind


def detail_db_path(run) -> Path:
    """Resolve the fixed run DB name beneath the run's output directory."""
    output_dir = getattr(run, "output_dir", None)
    if not output_dir:
        raise DetailStoreError("Training run has no output directory")
    configured_name = getattr(run, "detail_db_name", None)
    name = RUN_DB_FILENAME if configured_name is None else configured_name
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute() or name in ("", ".", ".."):
        raise DetailStoreError(f"Invalid training detail database name: {name!r}")
    return Path(output_dir) / name


def tagger_detail_db_path(run) -> Path:
    """Resolve the fixed tagger DB name beneath its output directory."""
    output_dir = getattr(run, "output_dir", None)
    if not output_dir:
        raise DetailStoreError("Tagger training run has no output directory")
    configured_name = getattr(run, "detail_db_name", None)
    name = TAGGER_RUN_DB_FILENAME if configured_name is None else configured_name
    candidate = Path(name)
    if candidate.name != name or candidate.is_absolute() or name in ("", ".", ".."):
        raise DetailStoreError(f"Invalid tagger detail database name: {name!r}")
    return Path(output_dir) / name


def resolve_detail_store(run) -> DetailStoreLocation:
    """Resolve catalogue metadata without opening or creating any database."""
    kind = detail_store_kind(run)
    return DetailStoreLocation(
        kind=kind,
        path=detail_db_path(run) if kind == RUN_DB_V2 else None,
        schema_version=getattr(run, "detail_schema_version", None),
        state=getattr(run, "detail_state", None),
    )


def _configure_run_db(dbapi_conn, _) -> None:
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")
    dbapi_conn.execute("PRAGMA busy_timeout=30000")


@lru_cache(maxsize=64)
def _run_db_session_factory(path_text: str):
    engine = create_engine(
        URL.create("sqlite", database=path_text),
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=NullPool,
    )
    event.listen(engine, "connect", _configure_run_db)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def run_db_session_factory(path: Path):
    """Return a cached factory; constructing it does not open the database."""
    return _run_db_session_factory(str(Path(path).resolve()))


def _copy_run_columns(run) -> dict:
    from .models import TrainingRun

    return {
        column.name: getattr(run, column.name)
        for column in TrainingRun.__table__.columns
    }


def initialize_run_detail_database(run):
    """Create and identify one v2 run database, returning its session factory.

    The file uses the existing ORM tables so partial updates, resume cleanup,
    and API serialization keep exactly the legacy row contract.
    """
    from .models import (
        TrainingCheckpoint,
        TrainingMetrics,
        TrainingRun,
        TrainingSample,
    )

    path = detail_db_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    factory = run_db_session_factory(path)
    engine = factory.kw["bind"]
    for table in (
        TrainingRun.__table__,
        TrainingMetrics.__table__,
        TrainingCheckpoint.__table__,
        TrainingSample.__table__,
    ):
        table.create(bind=engine, checkfirst=True)

    db = factory()
    try:
        local = db.query(TrainingRun).filter(TrainingRun.id == run.id).first()
        if local is None:
            db.add(TrainingRun(**_copy_run_columns(run)))
            db.commit()
        elif local.run_id != run.run_id:
            raise DetailStoreError(
                f"Training detail database identity mismatch at {path}"
            )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return factory


def open_run_detail_session(run):
    """Open an existing v2 run DB and verify its catalogue UUID."""
    from .models import TrainingRun

    location = resolve_detail_store(run)
    if location.kind != RUN_DB_V2 or location.path is None:
        raise DetailStoreError("Training run does not use a v2 detail database")
    if not location.path.is_file():
        raise DetailStoreError(f"Training detail database is unavailable: {location.path}")
    db = run_db_session_factory(location.path)()
    try:
        local = db.query(TrainingRun).filter(TrainingRun.id == run.id).first()
        if local is None or local.run_id != run.run_id:
            raise DetailStoreError(
                f"Training detail database identity mismatch at {location.path}"
            )
        return db
    except Exception:
        db.close()
        raise


def open_training_history_session(catalog_db, run_id: int):
    """Return the authoritative history session and whether the caller owns it."""
    from .models import TrainingRun

    run = catalog_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run is None:
        raise DetailStoreError(f"Training run {run_id} is missing from the catalogue")
    if detail_store_kind(run) == CENTRAL_V1:
        return catalog_db, False, run
    if run.detail_state != "ready":
        raise DetailStoreError(
            f"Training run {run_id} detail store is not ready: {run.detail_state!r}"
        )
    return open_run_detail_session(run), True, run


def mirror_metrics_to_run_database(run, central_db, touched_steps) -> None:
    """Bring a dual-written run DB through the newest central metric batch.

    A previous process may have committed centrally and exited before its mirror.
    Copying the missing tail as well as the touched steps repairs that gap on the
    next flush and preserves same-step partial updates.
    """
    from sqlalchemy import func

    from .models import TrainingMetrics

    steps = {int(step) for step in touched_steps}
    if not steps:
        return
    local_db = open_run_detail_session(run)
    try:
        local_max = local_db.query(func.max(TrainingMetrics.step)).filter(
            TrainingMetrics.run_id == run.id
        ).scalar()
        predicate = TrainingMetrics.step.in_(steps)
        if local_max is not None:
            predicate = or_(TrainingMetrics.step > int(local_max), predicate)
        rows = central_db.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run.id,
            predicate,
        ).order_by(TrainingMetrics.step.asc()).all()
        columns = [
            column.name for column in TrainingMetrics.__table__.columns
            if column.name != "id"
        ]
        for source in rows:
            target = local_db.query(TrainingMetrics).filter(
                TrainingMetrics.run_id == run.id,
                TrainingMetrics.step == source.step,
            ).first()
            values = {name: getattr(source, name) for name in columns}
            if target is None:
                local_db.add(TrainingMetrics(**values))
            else:
                for name, value in values.items():
                    setattr(target, name, value)
        local_db.commit()
    except Exception:
        local_db.rollback()
        raise
    finally:
        local_db.close()


def delete_run_database_metrics_after(run, step: int) -> int:
    """Delete stale v2 history after a checkpoint rewind."""
    from .models import TrainingMetrics

    local_db = open_run_detail_session(run)
    try:
        deleted = local_db.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run.id,
            TrainingMetrics.step > int(step),
        ).delete(synchronize_session=False)
        local_db.commit()
        return int(deleted)
    except Exception:
        local_db.rollback()
        raise
    finally:
        local_db.close()


def _copy_owned_rows(central_db, local_db, model, run_id: int,
                     batch_size: int) -> int:
    """Resume an ID-ordered copy into a terminal run's temporary database."""
    from sqlalchemy import func

    local_max = local_db.query(func.max(model.id)).scalar() or 0
    copied = 0
    columns = [column.name for column in model.__table__.columns]
    while True:
        rows = central_db.query(model).filter(
            model.run_id == run_id,
            model.id > local_max,
        ).order_by(model.id.asc()).limit(batch_size).all()
        if not rows:
            break
        local_db.bulk_insert_mappings(model, [
            {name: getattr(row, name) for name in columns}
            for row in rows
        ])
        local_db.commit()
        local_max = rows[-1].id
        copied += len(rows)
    return copied


def _verify_migrated_counts(central_db, local_db, run_id: int) -> dict:
    from .models import TrainingCheckpoint, TrainingMetrics, TrainingSample

    counts = {}
    for model in (TrainingMetrics, TrainingCheckpoint, TrainingSample):
        central_count = central_db.query(model).filter(
            model.run_id == run_id
        ).count()
        local_count = local_db.query(model).filter(
            model.run_id == run_id
        ).count()
        if central_count != local_count:
            raise DetailStoreError(
                f"{model.__tablename__} count mismatch: "
                f"central={central_count}, run_db={local_count}"
            )
        counts[model.__tablename__] = local_count
    integrity = local_db.execute(text("PRAGMA integrity_check")).scalar()
    if integrity != "ok":
        raise DetailStoreError(f"Run database integrity_check failed: {integrity}")
    return counts


def migrate_terminal_run_to_v2(central_db, run, *, batch_size: int = 5000) -> dict:
    """Copy one terminal legacy run and flip its catalogue pointer last.

    Central detail rows are deliberately retained. Re-running after an
    interrupted copy resumes from each temporary table's largest source ID.
    """
    from .models import (
        TrainingCheckpoint,
        TrainingMetrics,
        TrainingRun,
        TrainingSample,
    )

    if run.status not in TERMINAL_RUN_STATUSES:
        raise DetailStoreError(
            f"Run {run.id} is {run.status!r}; only terminal runs can migrate"
        )
    if detail_store_kind(run) == RUN_DB_V2 and run.detail_state == "ready":
        local_db = open_run_detail_session(run)
        try:
            counts = _verify_migrated_counts(central_db, local_db, run.id)
        finally:
            local_db.close()
        return {"run_id": run.id, "already_migrated": True, **counts}

    original = {
        "detail_store": run.detail_store,
        "detail_schema_version": run.detail_schema_version,
        "detail_state": run.detail_state,
        "detail_db_name": run.detail_db_name,
    }
    final_name = RUN_DB_FILENAME
    temp_name = RUN_DB_FILENAME + ".migrating"
    final_path = Path(run.output_dir) / final_name
    if final_path.exists():
        # Recovery seam: the atomic rename may have completed immediately
        # before the catalogue flip failed. Accept only the matching, complete
        # database; an unrelated file is still refused by identity/count checks.
        probe_values = _copy_run_columns(run)
        probe_values.update({
            "detail_store": RUN_DB_V2,
            "detail_schema_version": RUN_DB_SCHEMA_VERSION,
            "detail_state": "ready",
            "detail_db_name": final_name,
        })
        probe_run = TrainingRun(**probe_values)
        local_db = open_run_detail_session(probe_run)
        try:
            counts = _verify_migrated_counts(central_db, local_db, run.id)
        finally:
            local_db.close()
        run.detail_store = RUN_DB_V2
        run.detail_schema_version = RUN_DB_SCHEMA_VERSION
        run.detail_state = "ready"
        run.detail_db_name = final_name
        central_db.commit()
        return {"run_id": run.id, "recovered_final_file": True, **counts}

    run.detail_state = "migrating"
    central_db.commit()
    temp_values = _copy_run_columns(run)
    temp_values.update({
        "detail_store": RUN_DB_V2,
        "detail_schema_version": RUN_DB_SCHEMA_VERSION,
        "detail_state": "migrating",
        "detail_db_name": temp_name,
    })
    temp_run = TrainingRun(**temp_values)

    try:
        factory = initialize_run_detail_database(temp_run)
        local_db = factory()
        try:
            copied = {}
            for model in (TrainingMetrics, TrainingCheckpoint, TrainingSample):
                copied[model.__tablename__] = _copy_owned_rows(
                    central_db, local_db, model, run.id, batch_size
                )
            counts = _verify_migrated_counts(central_db, local_db, run.id)
            local_run = local_db.query(TrainingRun).filter(
                TrainingRun.id == run.id
            ).one()
            run.detail_store = RUN_DB_V2
            run.detail_schema_version = RUN_DB_SCHEMA_VERSION
            run.detail_state = "ready"
            run.detail_db_name = final_name
            for name, value in _copy_run_columns(run).items():
                setattr(local_run, name, value)
            local_db.commit()
            local_db.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        finally:
            local_db.close()
        factory.kw["bind"].dispose()
        os.replace(str(Path(run.output_dir) / temp_name), str(final_path))
        _run_db_session_factory.cache_clear()
        central_db.commit()
        return {
            "run_id": run.id,
            "already_migrated": False,
            "copied": copied,
            **counts,
        }
    except Exception:
        central_db.rollback()
        for name, value in original.items():
            setattr(run, name, value)
        central_db.commit()
        raise


def initialize_tagger_detail_database(run):
    """Create and identify a tagger run's isolated detail database."""
    from .models import TaggerTrainingMetrics, TaggerTrainingRun

    path = tagger_detail_db_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    factory = run_db_session_factory(path)
    engine = factory.kw["bind"]
    for table in (TaggerTrainingRun.__table__, TaggerTrainingMetrics.__table__):
        table.create(bind=engine, checkfirst=True)
    values = {
        column.name: getattr(run, column.name)
        for column in TaggerTrainingRun.__table__.columns
    }
    db = factory()
    try:
        local = db.query(TaggerTrainingRun).filter(
            TaggerTrainingRun.run_id == run.run_id
        ).first()
        if local is None:
            if db.query(TaggerTrainingRun).count():
                raise DetailStoreError(
                    f"Tagger detail database identity mismatch at {path}"
                )
            db.add(TaggerTrainingRun(**values))
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return factory


def open_tagger_detail_session(run):
    """Open an existing tagger run DB and verify its UUID identity."""
    from .models import TaggerTrainingRun

    if detail_store_kind(run) != RUN_DB_V2:
        raise DetailStoreError("Tagger run does not use a v2 detail database")
    path = tagger_detail_db_path(run)
    if not path.is_file():
        raise DetailStoreError(f"Tagger detail database is unavailable: {path}")
    db = run_db_session_factory(path)()
    try:
        local = db.query(TaggerTrainingRun).filter(
            TaggerTrainingRun.run_id == run.run_id
        ).first()
        if local is None:
            raise DetailStoreError(
                f"Tagger detail database identity mismatch at {path}"
            )
        return db
    except Exception:
        db.close()
        raise


def open_tagger_history_session(catalog_db, run_id: str):
    """Return the authoritative tagger history session and ownership flag."""
    from .models import TaggerTrainingRun

    run = catalog_db.query(TaggerTrainingRun).filter(
        TaggerTrainingRun.run_id == run_id
    ).first()
    if run is None:
        raise DetailStoreError(
            f"Tagger training run {run_id!r} is missing from the catalogue"
        )
    if detail_store_kind(run) == CENTRAL_V1:
        return catalog_db, False, run
    if run.detail_state != "ready":
        raise DetailStoreError(
            f"Tagger run {run_id!r} detail store is not ready: {run.detail_state!r}"
        )
    return open_tagger_detail_session(run), True, run


def mirror_tagger_metrics_to_run_database(run, central_db, touched_keys) -> None:
    """Mirror committed tagger metric rows, including same-step updates."""
    from sqlalchemy import and_, func

    from .models import TaggerTrainingMetrics

    keys = {(int(resume), int(step)) for resume, step in touched_keys}
    if not keys:
        return
    local_db = open_tagger_detail_session(run)
    try:
        local_max = local_db.query(func.max(TaggerTrainingMetrics.id)).scalar() or 0
        touched = or_(*[
            and_(TaggerTrainingMetrics.resume_seq == resume,
                 TaggerTrainingMetrics.step == step)
            for resume, step in keys
        ])
        rows = central_db.query(TaggerTrainingMetrics).filter(
            TaggerTrainingMetrics.run_id == run.run_id,
            or_(TaggerTrainingMetrics.id > local_max, touched),
        ).order_by(TaggerTrainingMetrics.id.asc()).all()
        columns = [
            column.name for column in TaggerTrainingMetrics.__table__.columns
            if column.name != "id"
        ]
        for source in rows:
            target = local_db.query(TaggerTrainingMetrics).filter(
                TaggerTrainingMetrics.run_id == run.run_id,
                TaggerTrainingMetrics.resume_seq == source.resume_seq,
                TaggerTrainingMetrics.step == source.step,
            ).first()
            values = {name: getattr(source, name) for name in columns}
            if target is None:
                local_db.add(TaggerTrainingMetrics(**values))
            else:
                for name, value in values.items():
                    setattr(target, name, value)
        local_db.commit()
    except Exception:
        local_db.rollback()
        raise
    finally:
        local_db.close()


def _verify_tagger_migrated_counts(central_db, local_db, run_id: str) -> dict:
    from .models import TaggerTrainingMetrics

    central_count = central_db.query(TaggerTrainingMetrics).filter(
        TaggerTrainingMetrics.run_id == run_id
    ).count()
    local_count = local_db.query(TaggerTrainingMetrics).filter(
        TaggerTrainingMetrics.run_id == run_id
    ).count()
    if central_count != local_count:
        raise DetailStoreError(
            f"tagger_training_metrics count mismatch: "
            f"central={central_count}, run_db={local_count}"
        )
    integrity = local_db.execute(text("PRAGMA integrity_check")).scalar()
    if integrity != "ok":
        raise DetailStoreError(f"Tagger run database integrity_check failed: {integrity}")
    return {"tagger_training_metrics": local_count}


def migrate_terminal_tagger_run_to_v2(central_db, run,
                                      *, batch_size: int = 5000) -> dict:
    """Copy one terminal tagger run, retaining its central rollback rows."""
    from sqlalchemy import func

    from .models import TaggerTrainingMetrics, TaggerTrainingRun

    if run.status not in TERMINAL_RUN_STATUSES:
        raise DetailStoreError(
            f"Tagger run {run.run_id} is {run.status!r}; "
            "only terminal runs can migrate"
        )
    if detail_store_kind(run) == RUN_DB_V2 and run.detail_state == "ready":
        local_db = open_tagger_detail_session(run)
        try:
            counts = _verify_tagger_migrated_counts(
                central_db, local_db, run.run_id
            )
        finally:
            local_db.close()
        return {"run_id": run.run_id, "already_migrated": True, **counts}

    original = {
        "detail_store": run.detail_store,
        "detail_schema_version": run.detail_schema_version,
        "detail_state": run.detail_state,
        "detail_db_name": run.detail_db_name,
    }
    final_name = TAGGER_RUN_DB_FILENAME
    temp_name = TAGGER_RUN_DB_FILENAME + ".migrating"
    final_path = Path(run.output_dir or "") / final_name
    if not run.output_dir:
        raise DetailStoreError(f"Tagger run {run.run_id} has no output directory")
    if final_path.exists():
        probe_values = {
            column.name: getattr(run, column.name)
            for column in TaggerTrainingRun.__table__.columns
        }
        probe_values.update({
            "detail_store": RUN_DB_V2,
            "detail_schema_version": RUN_DB_SCHEMA_VERSION,
            "detail_state": "ready",
            "detail_db_name": final_name,
        })
        probe_run = TaggerTrainingRun(**probe_values)
        local_db = open_tagger_detail_session(probe_run)
        try:
            counts = _verify_tagger_migrated_counts(
                central_db, local_db, run.run_id
            )
        finally:
            local_db.close()
        run.detail_store = RUN_DB_V2
        run.detail_schema_version = RUN_DB_SCHEMA_VERSION
        run.detail_state = "ready"
        run.detail_db_name = final_name
        central_db.commit()
        return {"run_id": run.run_id, "recovered_final_file": True, **counts}

    run.detail_state = "migrating"
    central_db.commit()
    values = {
        column.name: getattr(run, column.name)
        for column in TaggerTrainingRun.__table__.columns
    }
    values.update({
        "detail_store": RUN_DB_V2,
        "detail_schema_version": RUN_DB_SCHEMA_VERSION,
        "detail_state": "migrating",
        "detail_db_name": temp_name,
    })
    temp_run = TaggerTrainingRun(**values)
    try:
        factory = initialize_tagger_detail_database(temp_run)
        local_db = factory()
        try:
            local_max = local_db.query(
                func.max(TaggerTrainingMetrics.id)
            ).scalar() or 0
            copied = 0
            columns = [
                column.name
                for column in TaggerTrainingMetrics.__table__.columns
            ]
            while True:
                rows = central_db.query(TaggerTrainingMetrics).filter(
                    TaggerTrainingMetrics.run_id == run.run_id,
                    TaggerTrainingMetrics.id > local_max,
                ).order_by(TaggerTrainingMetrics.id.asc()).limit(batch_size).all()
                if not rows:
                    break
                local_db.bulk_insert_mappings(TaggerTrainingMetrics, [
                    {name: getattr(row, name) for name in columns}
                    for row in rows
                ])
                local_db.commit()
                local_max = rows[-1].id
                copied += len(rows)
            counts = _verify_tagger_migrated_counts(
                central_db, local_db, run.run_id
            )
            run.detail_store = RUN_DB_V2
            run.detail_schema_version = RUN_DB_SCHEMA_VERSION
            run.detail_state = "ready"
            run.detail_db_name = final_name
            local_run = local_db.query(TaggerTrainingRun).filter(
                TaggerTrainingRun.run_id == run.run_id
            ).one()
            for column in TaggerTrainingRun.__table__.columns:
                setattr(local_run, column.name, getattr(run, column.name))
            local_db.commit()
            local_db.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        finally:
            local_db.close()
        factory.kw["bind"].dispose()
        os.replace(str(Path(run.output_dir) / temp_name), str(final_path))
        _run_db_session_factory.cache_clear()
        central_db.commit()
        return {
            "run_id": run.run_id,
            "already_migrated": False,
            "copied": copied,
            **counts,
        }
    except Exception:
        central_db.rollback()
        for name, value in original.items():
            setattr(run, name, value)
        central_db.commit()
        raise
