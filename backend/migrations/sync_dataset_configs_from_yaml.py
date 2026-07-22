"""
One-time backfill migration: sync TrainingRun.dataset_configs from config_yaml.

Background
----------
``TrainingRun.dataset_configs`` (JSON column) is a denormalized cache of
which datasets a run trains on. ``TrainingRun.config_yaml`` is the actual
source of truth -- it's what the trainer subprocess reads
(core/training/train_runner.py). Historically the two could diverge:

  * The PUT /training/runs/{id} endpoint built a fresh dataset_configs list
    but never assigned it back to the run (fixed alongside this migration).
  * PATCH /config and POST /config/reload only ever touched config_yaml.

This script re-derives dataset_configs from config_yaml for every run and
overwrites the column where it disagrees, so the DB column, the pre-flight
rescan (routes.py start_training_run), and list/edit views all agree with
what the trainer actually loads.

Safety
------
Before touching anything, this script scans every row's CURRENT
dataset_configs for a non-empty ``filters`` value. The design here assumes
``filters`` is a dead field (the frontend always sends ``{}`` and there are
no backend consumers -- see routes.py's "TODO: Apply filters here when
filter logic is implemented"). If any row has a real filters value, that
assumption is wrong and blindly proceeding could silently discard user
data, so the script aborts with a report instead of migrating anything.

Idempotent: running this script multiple times is safe -- rows that already
match their derived value are left untouched (no-op update), and the
filters pre-check re-runs every time.
"""

import sys
from pathlib import Path

# Add backend directory to path so `core.*` / `database.*` imports resolve
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database import TrainingSessionLocal, DatasetsSessionLocal  # noqa: E402
from database.models import TrainingRun  # noqa: E402


def _check_no_real_filters(training_db) -> bool:
    """Pre-check: abort if any run has a non-empty ``filters`` value.

    Returns True if it's safe to proceed, False (and prints a report) if a
    real filters value was found anywhere.
    """
    offenders = []
    runs = training_db.query(TrainingRun).all()
    for run in runs:
        for entry in (run.dataset_configs or []):
            filters = entry.get("filters") if isinstance(entry, dict) else None
            if filters:  # non-empty dict (or any truthy non-{} value)
                offenders.append((run.id, run.run_name, entry.get("dataset_id"), filters))

    if offenders:
        print("=" * 70)
        print("[Migration] ABORTED: found non-empty 'filters' value(s).")
        print("[Migration] This migration assumes 'filters' is a dead field;")
        print("[Migration] a real value here means that assumption is wrong")
        print("[Migration] and proceeding could silently discard data.")
        print("=" * 70)
        for run_id, run_name, dataset_id, filters in offenders:
            print(f"  run_id={run_id} run_name={run_name!r} dataset_id={dataset_id} filters={filters!r}")
        print("=" * 70)
        return False

    print(f"[Migration] Pre-check OK: no non-empty 'filters' found across {len(runs)} run(s)")
    return True


def migrate():
    training_db = TrainingSessionLocal()
    datasets_db = DatasetsSessionLocal()

    try:
        if not _check_no_real_filters(training_db):
            return

        from core.training.dataset_params import resolve_dataset_configs_from_yaml

        runs = training_db.query(TrainingRun).all()
        print(f"[Migration] Checking {len(runs)} training run(s)...")

        updated = 0
        unresolved = 0
        unchanged = 0

        for run in runs:
            derived = resolve_dataset_configs_from_yaml(run.config_yaml, datasets_db)

            if derived is None:
                # Can't derive from YAML (no datasets section, unparsable
                # YAML, or every entry failed to resolve -- e.g. a
                # folder_path that's no longer present in datasets.db).
                # Keep the current column value; just warn.
                unresolved += 1
                print(f"[Migration] WARN: run_id={run.id} run_name={run.run_name!r} -- "
                      f"could not derive dataset_configs from config_yaml, keeping current column "
                      f"({run.dataset_configs!r})")
                continue

            current_ids = sorted({
                int(c["dataset_id"]) for c in (run.dataset_configs or []) if c.get("dataset_id")
            })
            derived_ids = sorted({c["dataset_id"] for c in derived})

            if current_ids == derived_ids:
                unchanged += 1
                continue

            print(f"[Migration] run_id={run.id} run_name={run.run_name!r}: "
                  f"dataset_configs {current_ids} -> {derived_ids}")
            run.dataset_configs = derived
            updated += 1

        if updated:
            training_db.commit()

        print("=" * 70)
        print(f"[Migration] Done. updated={updated} unchanged={unchanged} "
              f"unresolved(kept-as-is)={unresolved} total={len(runs)}")
        print("=" * 70)

    except Exception:
        training_db.rollback()
        raise
    finally:
        training_db.close()
        datasets_db.close()


if __name__ == "__main__":
    migrate()
