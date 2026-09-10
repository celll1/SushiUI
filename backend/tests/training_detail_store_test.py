"""Compatibility and path-safety tests for training detail-store routing."""

from types import SimpleNamespace

import pytest

from database.training_detail_store import (
    CENTRAL_V1,
    RUN_DB_V2,
    DetailStoreError,
    detail_db_path,
    resolve_detail_store,
)


def _run(**overrides):
    values = {
        "detail_store": None,
        "detail_schema_version": None,
        "detail_state": None,
        "detail_db_name": None,
        "output_dir": "output/run",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_null_store_is_legacy_without_touching_output_directory():
    location = resolve_detail_store(_run(output_dir=None))
    assert location.kind == CENTRAL_V1
    assert location.path is None


def test_v2_uses_fixed_name_beneath_output_directory():
    location = resolve_detail_store(_run(
        detail_store=RUN_DB_V2,
        detail_schema_version=2,
        detail_state="ready",
    ))
    assert location.path == detail_db_path(_run(detail_store=RUN_DB_V2))
    assert location.path.name == "training_run.db"
    assert location.schema_version == 2
    assert location.state == "ready"


@pytest.mark.parametrize("name", ("../other.db", "nested/run.db", "", ".", ".."))
def test_v2_rejects_non_local_database_names(name):
    with pytest.raises(DetailStoreError):
        resolve_detail_store(_run(detail_store=RUN_DB_V2, detail_db_name=name))


def test_unknown_store_is_not_silently_treated_as_legacy():
    with pytest.raises(DetailStoreError):
        resolve_detail_store(_run(detail_store="future_v9"))
