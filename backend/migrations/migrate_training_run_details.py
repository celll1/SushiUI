"""Copy terminal legacy runs into per-run SQLite databases.

Dry-run is the default. Pass ``--apply`` with explicit ``--run-id`` values or
``--all-terminal`` to write files and flip catalogue pointers. Central detail
rows are retained for rollback; this command never purges or vacuums them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import TrainingSessionLocal  # noqa: E402
from database.models import TrainingMetrics, TrainingRun  # noqa: E402
from database.training_detail_store import (  # noqa: E402
    TERMINAL_RUN_STATUSES,
    detail_store_kind,
    migrate_terminal_run_to_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id", type=int, action="append")
    selection.add_argument("--all-terminal", action="store_true")
    parser.add_argument("--apply", action="store_true",
                        help="perform the migration; otherwise only report")
    parser.add_argument("--batch-size", type=int, default=5000)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    db = TrainingSessionLocal()
    try:
        query = db.query(TrainingRun)
        if args.all_terminal:
            query = query.filter(TrainingRun.status.in_(TERMINAL_RUN_STATUSES))
        else:
            query = query.filter(TrainingRun.id.in_(args.run_id))
        runs = query.order_by(TrainingRun.id.asc()).all()
        if not runs:
            print("No matching runs.")
            return 0
        for run in runs:
            count = db.query(TrainingMetrics).filter(
                TrainingMetrics.run_id == run.id
            ).count()
            print(f"run={run.id} name={run.run_name!r} status={run.status} "
                  f"store={detail_store_kind(run)} metrics={count} "
                  f"output={run.output_dir}")
            if args.apply:
                result = migrate_terminal_run_to_v2(
                    db, run, batch_size=args.batch_size
                )
                print(f"  migrated: {result}")
        if not args.apply:
            print("Dry run only; pass --apply to migrate these runs.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
