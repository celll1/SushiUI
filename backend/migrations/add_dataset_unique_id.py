"""
Migration: Add unique_id column to datasets table

This migration adds a unique_id (UUID) column to the datasets table for stable cache directory naming.
Existing datasets will have UUIDs automatically generated.
"""

import sys
import uuid
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from database.database import get_datasets_db
from database.models import Dataset
from sqlalchemy import text


def main():
    print("[Migration] Adding unique_id column to datasets table...")

    db = next(get_datasets_db())

    try:
        # Check if column already exists
        result = db.execute(text("PRAGMA table_info(datasets)")).fetchall()
        columns = [row[1] for row in result]

        if "unique_id" in columns:
            print("[Migration] Column 'unique_id' already exists, skipping...")
            return

        # Add column
        print("[Migration] Adding 'unique_id' column...")
        db.execute(text("ALTER TABLE datasets ADD COLUMN unique_id VARCHAR"))
        db.commit()

        # Generate UUIDs for existing datasets
        datasets = db.query(Dataset).all()
        print(f"[Migration] Generating UUIDs for {len(datasets)} existing datasets...")

        for dataset in datasets:
            dataset.unique_id = str(uuid.uuid4())
            print(f"  Dataset '{dataset.name}' (id={dataset.id}) -> unique_id={dataset.unique_id}")

        db.commit()

        # Add unique constraint and index
        print("[Migration] Adding unique constraint and index...")
        db.execute(text("CREATE UNIQUE INDEX idx_datasets_unique_id ON datasets(unique_id)"))
        db.commit()

        print("[Migration] Migration completed successfully!")

    except Exception as e:
        print(f"[Migration] ERROR: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
