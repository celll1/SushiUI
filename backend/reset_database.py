"""
Reset databases with new schema.
This script drops all tables and recreates them with the current models.

NOTE: This will reset all three databases:
- gallery.db (generated images, user settings)
- datasets.db (datasets, items, captions, tag dictionary)
- training.db (training runs, checkpoints, samples)
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import gallery_engine, datasets_engine, training_engine
from database.models import (
    GalleryBase,
    DatasetBase,
    TrainingBase,
)

def reset_all_databases():
    """Drop all tables and recreate them in all databases."""

    print("\n" + "=" * 60)
    print("RESETTING ALL DATABASES")
    print("=" * 60)

    # Reset gallery.db
    print("\n[Gallery DB] Dropping all tables...")
    GalleryBase.metadata.drop_all(bind=gallery_engine)
    print("[Gallery DB] Creating all tables with new schema...")
    GalleryBase.metadata.create_all(bind=gallery_engine)
    print("[Gallery DB] Reset complete!")
    print("  - UserSettings")
    print("  - GeneratedImage")

    # Reset datasets.db
    print("\n[Datasets DB] Dropping all tables...")
    DatasetBase.metadata.drop_all(bind=datasets_engine)
    print("[Datasets DB] Creating all tables with new schema...")
    DatasetBase.metadata.create_all(bind=datasets_engine)
    print("[Datasets DB] Reset complete!")
    print("  - Dataset")
    print("  - DatasetItem")
    print("  - DatasetCaption")
    print("  - TagDictionary")

    # Reset training.db
    print("\n[Training DB] Dropping all tables...")
    TrainingBase.metadata.drop_all(bind=training_engine)
    print("[Training DB] Creating all tables with new schema...")
    TrainingBase.metadata.create_all(bind=training_engine)
    print("[Training DB] Reset complete!")
    print("  - TrainingRun (with run_id UUID field)")
    print("  - TrainingCheckpoint")
    print("  - TrainingSample")

    print("\n" + "=" * 60)
    print("ALL DATABASES RESET COMPLETE!")
    print("=" * 60)

def reset_gallery_only():
    """Reset only gallery.db"""
    print("\n[Gallery DB] Dropping all tables...")
    GalleryBase.metadata.drop_all(bind=gallery_engine)
    print("[Gallery DB] Creating all tables...")
    GalleryBase.metadata.create_all(bind=gallery_engine)
    print("[Gallery DB] Reset complete!")

def reset_datasets_only():
    """Reset only datasets.db"""
    print("\n[Datasets DB] Dropping all tables...")
    DatasetBase.metadata.drop_all(bind=datasets_engine)
    print("[Datasets DB] Creating all tables...")
    DatasetBase.metadata.create_all(bind=datasets_engine)
    print("[Datasets DB] Reset complete!")

def reset_training_only():
    """Reset only training.db"""
    print("\n[Training DB] Dropping all tables...")
    TrainingBase.metadata.drop_all(bind=training_engine)
    print("[Training DB] Creating all tables...")
    TrainingBase.metadata.create_all(bind=training_engine)
    print("[Training DB] Reset complete!")

if __name__ == "__main__":
    print("\nDATABASE RESET UTILITY")
    print("=" * 60)
    print("\nWhich database(s) would you like to reset?")
    print("  1. All databases (gallery + datasets + training)")
    print("  2. Gallery only (generated images, settings)")
    print("  3. Datasets only (datasets, items, captions)")
    print("  4. Training only (training runs, checkpoints)")
    print("  5. Cancel")

    choice = input("\nEnter your choice (1-5): ")

    if choice == "1":
        print("\nWARNING: This will delete ALL data in ALL databases!")
        print("This includes:")
        print("  - Generated images metadata")
        print("  - Dataset configurations")
        print("  - Training runs")
        print("  - All other data\n")
        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() == "yes":
            reset_all_databases()
        else:
            print("Operation cancelled.")

    elif choice == "2":
        print("\nWARNING: This will delete all data in gallery.db!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            reset_gallery_only()
        else:
            print("Operation cancelled.")

    elif choice == "3":
        print("\nWARNING: This will delete all data in datasets.db!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            reset_datasets_only()
        else:
            print("Operation cancelled.")

    elif choice == "4":
        print("\nWARNING: This will delete all data in training.db!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            reset_training_only()
        else:
            print("Operation cancelled.")

    else:
        print("Operation cancelled.")
