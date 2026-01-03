"""
Cleanup old metrics that haven't been overwritten by resumed training

When training resumes from step 0, only steps that have been reached
will be overwritten via UPSERT. Old steps beyond current progress remain.

This script removes metrics beyond the latest timestamp to clean up old data.

Run from project root:
    python cleanup_old_metrics.py [run_id]
"""

import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from database import TrainingSessionLocal
from database.models import TrainingMetrics


def cleanup_old_metrics(run_id):
    """Remove old metrics that haven't been overwritten by resumed training."""

    session = TrainingSessionLocal()

    try:
        print("="*80)
        print(f"Cleanup Old Metrics for run_id={run_id}")
        print("="*80)
        print()

        # Get all metrics for this run
        all_metrics = session.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run_id
        ).order_by(TrainingMetrics.step).all()

        if not all_metrics:
            print(f"No metrics found for run_id={run_id}")
            return

        print(f"Total metrics: {len(all_metrics)}")
        print(f"Step range: {all_metrics[0].step} - {all_metrics[-1].step}")
        print()

        # Find the latest timestamp (most recent training session)
        latest_timestamp = max(m.timestamp for m in all_metrics if m.timestamp)
        print(f"Latest timestamp: {latest_timestamp}")

        # Find the cutoff point: step with latest timestamp
        # All steps after this with older timestamps should be deleted
        latest_step_with_latest_timestamp = max(
            m.step for m in all_metrics if m.timestamp == latest_timestamp
        )
        print(f"Latest step reached: {latest_step_with_latest_timestamp}")
        print()

        # Find old metrics (steps beyond latest progress with older timestamps)
        old_metrics = [
            m for m in all_metrics
            if m.step > latest_step_with_latest_timestamp
        ]

        if not old_metrics:
            print("No old metrics to clean up")
            return

        print(f"Old metrics to delete: {len(old_metrics)}")
        print(f"Steps to delete: {old_metrics[0].step} - {old_metrics[-1].step}")
        print()

        # Show sample of what will be deleted
        print("Sample (first 5):")
        for m in old_metrics[:5]:
            loss_str = f"{m.loss:.6f}" if m.loss is not None else "None"
            print(f"  step={m.step}, timestamp={m.timestamp}, loss={loss_str}")
        print()

        # Confirm deletion
        response = input(f"Delete {len(old_metrics)} old metrics? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled")
            return

        # Delete old metrics
        for m in old_metrics:
            session.delete(m)

        session.commit()

        print()
        print(f"✓ Deleted {len(old_metrics)} old metrics")
        print()

        # Verify
        remaining = session.query(TrainingMetrics).filter(
            TrainingMetrics.run_id == run_id
        ).count()
        print(f"Remaining metrics: {remaining}")
        print(f"Step range: 1 - {latest_step_with_latest_timestamp}")

        print()
        print("="*80)

    finally:
        session.close()


if __name__ == "__main__":
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 55
    cleanup_old_metrics(run_id)
