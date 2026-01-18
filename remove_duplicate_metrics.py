"""
Remove duplicate training metrics records

Keeps the most recent record for each (run_id, step) combination.

Run from project root:
    python remove_duplicate_metrics.py
"""

import sys
from pathlib import Path
from sqlalchemy import text

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from database import training_engine, TrainingSessionLocal


def remove_duplicates():
    """Remove duplicate (run_id, step) records, keeping most recent timestamp."""

    session = TrainingSessionLocal()

    try:
        print("="*80)
        print("Removing Duplicate Training Metrics")
        print("="*80)
        print()

        # Find all duplicates
        duplicates_query = text("""
            SELECT run_id, step, COUNT(*) as count
            FROM training_metrics
            GROUP BY run_id, step
            HAVING COUNT(*) > 1
            ORDER BY run_id, step
        """)

        duplicates = session.execute(duplicates_query).fetchall()

        if not duplicates:
            print("No duplicates found.")
            return

        print(f"Found {len(duplicates)} duplicate (run_id, step) combinations")
        print()

        total_deleted = 0

        for run_id, step, count in duplicates:
            # Delete all but the most recent record for this (run_id, step)
            # SQLite doesn't support DELETE with JOIN, so use subquery
            delete_query = text("""
                DELETE FROM training_metrics
                WHERE id IN (
                    SELECT id FROM training_metrics
                    WHERE run_id = :run_id AND step = :step
                    ORDER BY timestamp DESC
                    LIMIT -1 OFFSET 1
                )
            """)

            result = session.execute(delete_query, {"run_id": run_id, "step": step})
            deleted_count = result.rowcount

            print(f"  run_id={run_id}, step={step}: Deleted {deleted_count} duplicate(s) (kept most recent)")
            total_deleted += deleted_count

        session.commit()

        print()
        print(f"Total deleted: {total_deleted} records")
        print()

        # Verify no duplicates remain
        remaining = session.execute(duplicates_query).fetchall()

        if remaining:
            print(f"WARNING: {len(remaining)} duplicates still remain!")
        else:
            print("SUCCESS: All duplicates removed")

        print()
        print("="*80)

    except Exception as e:
        print(f"ERROR: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    remove_duplicates()
