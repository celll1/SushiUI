"""Resolve the detailed-history store declared by a training run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CENTRAL_V1 = "central_v1"
RUN_DB_V2 = "run_db_v2"
RUN_DB_FILENAME = "training_run.db"
SUPPORTED_DETAIL_STORES = frozenset((CENTRAL_V1, RUN_DB_V2))


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
