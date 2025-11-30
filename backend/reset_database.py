"""
Reset database with new schema.
This script drops all tables and recreates them with the current models.
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import engine, Base
from database.models import (
    UserSettings,
    GeneratedImage,
    Dataset,
    DatasetItem,
    DatasetCaption,
    TagDictionary,
    TrainingRun,
    TrainingCheckpoint,
    TrainingSample,
)

def reset_database():
    """Drop all tables and recreate them."""
    print("[Database] Dropping all tables...")
    Base.metadata.drop_all(bind=engine)

    print("[Database] Creating all tables with new schema...")
    Base.metadata.create_all(bind=engine)

    print("[Database] Database reset complete!")
    print("\n[Database] New schema includes:")
    print("  - UserSettings")
    print("  - GeneratedImage")
    print("  - Dataset")
    print("  - DatasetItem")
    print("  - DatasetCaption")
    print("  - TagDictionary")
    print("  - TrainingRun (with run_id UUID field)")
    print("  - TrainingCheckpoint")
    print("  - TrainingSample")

if __name__ == "__main__":
    print("\nWARNING: This will delete all data in the database!")
    print("This includes:")
    print("  - Generated images metadata")
    print("  - Dataset configurations")
    print("  - Training runs")
    print("  - All other data\n")

    response = input("Are you sure you want to continue? (yes/no): ")

    if response.lower() == "yes":
        reset_database()
    else:
        print("Database reset cancelled.")
