"""
Migration: Add unique constraint to training_metrics table

This migration:
1. Removes duplicate (run_id, step) records (keeps most recent timestamp)
2. Adds unique constraint if not exists
3. Creates composite index for fast queries

Run from project root:
    python migrate_training_metrics_unique_constraint.py
"""

import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from database import TrainingSessionLocal, training_engine
from database.models import TrainingMetrics


def remove_duplicate_metrics(session):
    """Remove duplicate (run_id, step) records, keeping most recent timestamp."""

    # Find duplicates
    duplicates_query = text("""
        SELECT run_id, step, COUNT(*) as count
        FROM training_metrics
        GROUP BY run_id, step
        HAVING COUNT(*) > 1
    """)

    duplicates = session.execute(duplicates_query).fetchall()

    if not duplicates:
        print("[Migration] No duplicate metrics found")
        return 0

    print(f"[Migration] Found {len(duplicates)} duplicate (run_id, step) combinations")

    total_deleted = 0

    for run_id, step, count in duplicates:
        # Get all records for this (run_id, step)
        records = session.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run_id,
            TrainingMetrics.step == step
        ).order_by(TrainingMetrics.timestamp.desc()).all()

        # Keep the most recent, delete others
        keep_record = records[0]
        delete_records = records[1:]

        print(f"[Migration]   (run_id={run_id}, step={step}): Keeping most recent (timestamp={keep_record.timestamp}), deleting {len(delete_records)} older records")

        for record in delete_records:
            session.delete(record)
            total_deleted += 1

    session.commit()
    print(f"[Migration] Deleted {total_deleted} duplicate metrics")
    return total_deleted


def check_unique_constraint(engine):
    """Check if unique constraint exists on training_metrics table."""
    inspector = inspect(engine)
    constraints = inspector.get_unique_constraints("training_metrics")

    for constraint in constraints:
        if set(constraint['column_names']) == {'run_id', 'step'}:
            return True
    return False


def add_unique_constraint(session):
    """Add unique constraint to training_metrics table if not exists."""

    # SQLite-specific: Check and add constraint
    # Note: SQLite doesn't support ALTER TABLE ADD CONSTRAINT for unique constraints
    # We need to recreate the table

    try:
        # Try to create unique index (SQLite way to enforce uniqueness)
        session.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_run_step
            ON training_metrics (run_id, step)
        """))
        session.commit()
        print("[Migration] Added unique constraint (via unique index) to training_metrics")
        return True
    except Exception as e:
        print(f"[Migration] Unique constraint may already exist: {e}")
        session.rollback()
        return False


def main():
    print("="*80)
    print("[Migration] Training Metrics Unique Constraint Migration")
    print("="*80)
    print()

    # Get engine and session
    engine = training_engine
    session = TrainingSessionLocal()

    try:
        # Step 1: Remove duplicates
        print("[Migration] Step 1: Removing duplicate metrics...")
        deleted = remove_duplicate_metrics(session)
        print()

        # Step 2: Check if constraint exists
        print("[Migration] Step 2: Checking unique constraint...")
        has_constraint = check_unique_constraint(engine)

        if has_constraint:
            print("[Migration] Unique constraint already exists")
        else:
            print("[Migration] Unique constraint not found, adding...")
            add_unique_constraint(session)
        print()

        # Step 3: Verify
        print("[Migration] Step 3: Verification...")
        duplicates_query = text("""
            SELECT run_id, step, COUNT(*) as count
            FROM training_metrics
            GROUP BY run_id, step
            HAVING COUNT(*) > 1
        """)
        remaining_duplicates = session.execute(duplicates_query).fetchall()

        if remaining_duplicates:
            print(f"[Migration] WARNING: {len(remaining_duplicates)} duplicates still remain!")
            for run_id, step, count in remaining_duplicates:
                print(f"[Migration]   (run_id={run_id}, step={step}): {count} records")
        else:
            print("[Migration] ✓ No duplicates found")

        print()
        print("="*80)
        print("[Migration] Migration completed successfully")
        print("="*80)

    except Exception as e:
        print(f"[Migration] ERROR: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
