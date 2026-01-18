"""
Add phase tracking columns to training_runs table.

This migration adds:
- phase: Current training phase (initializing, latent_cache, text_encoder_cache, training)
- phase_progress: Progress within current phase (0.0-100.0)
- phase_detail: Detailed status message

Run with: python backend/database/migrations/add_training_phase.py
"""

import sqlite3
import sys
from pathlib import Path

def migrate():
    """Add phase tracking columns to training_runs table."""
    # Get database path
    db_path = Path(__file__).parent.parent.parent.parent / "training.db"

    print(f"[Migration] Migrating database: {db_path}")

    if not db_path.exists():
        print(f"[Migration] ERROR: Database not found at {db_path}")
        print(f"[Migration] Please create the database first by starting the application.")
        return False

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(training_runs)")
        columns = [row[1] for row in cursor.fetchall()]

        columns_to_add = []
        if "phase" not in columns:
            columns_to_add.append(("phase", "VARCHAR DEFAULT 'initializing'"))
        if "phase_progress" not in columns:
            columns_to_add.append(("phase_progress", "REAL DEFAULT 0.0"))
        if "phase_detail" not in columns:
            columns_to_add.append(("phase_detail", "VARCHAR"))

        if not columns_to_add:
            print("[Migration] All columns already exist. Nothing to do.")
            return True

        # Add columns
        for col_name, col_type in columns_to_add:
            print(f"[Migration] Adding column: {col_name} ({col_type})")
            cursor.execute(f"ALTER TABLE training_runs ADD COLUMN {col_name} {col_type}")

        conn.commit()
        print(f"[Migration] Successfully added {len(columns_to_add)} column(s)")

        # Verify
        cursor.execute("PRAGMA table_info(training_runs)")
        columns_after = [row[1] for row in cursor.fetchall()]
        print(f"[Migration] Columns after migration: {', '.join(columns_after)}")

        return True

    except Exception as e:
        print(f"[Migration] ERROR: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()

if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
