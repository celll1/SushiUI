"""Resolve the detailed-history store declared by a training run."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, event, or_
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


CENTRAL_V1 = "central_v1"
RUN_DB_V2 = "run_db_v2"
RUN_DB_FILENAME = "training_run.db"
SUPPORTED_DETAIL_STORES = frozenset((CENTRAL_V1, RUN_DB_V2))
RUN_DB_SCHEMA_VERSION = 2


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
