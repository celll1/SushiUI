from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.settings import settings
import os

# Import bases from models
from .models import GalleryBase, DatasetBase, TrainingBase

# Create separate engines
gallery_db_path = os.path.join(settings.root_dir, "gallery.db")
datasets_db_path = os.path.join(settings.root_dir, "datasets.db")
training_db_path = os.path.join(settings.root_dir, "training.db")

gallery_engine = create_engine(f"sqlite:///{gallery_db_path}", connect_args={"check_same_thread": False})
datasets_engine = create_engine(f"sqlite:///{datasets_db_path}", connect_args={"check_same_thread": False})
training_engine = create_engine(f"sqlite:///{training_db_path}", connect_args={"check_same_thread": False})

# Create separate session factories
GallerySessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=gallery_engine)
DatasetsSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=datasets_engine)
TrainingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=training_engine)

# Legacy compatibility (default to gallery)
# DEPRECATED: Use specific session factories (GallerySessionLocal, DatasetsSessionLocal, TrainingSessionLocal)
# These aliases are kept for backward compatibility only
engine = gallery_engine
SessionLocal = GallerySessionLocal
Base = GalleryBase

def init_db():
    """Initialize all databases"""
    from .models import (
        GeneratedImage, UserSettings,  # Gallery
        Dataset, DatasetItem, DatasetCaption, TagDictionary,  # Datasets
        TrainingRun, TrainingCheckpoint, TrainingSample  # Training
    )
    import uuid

    # Create tables for each database
    print("[Database] Initializing gallery.db...")
    GalleryBase.metadata.create_all(bind=gallery_engine)

    print("[Database] Initializing datasets.db...")
    DatasetBase.metadata.create_all(bind=datasets_engine)

    print("[Database] Initializing training.db...")
    TrainingBase.metadata.create_all(bind=training_engine)

    # Migration: Add unique_id to existing datasets
    print("[Database] Running migrations...")
    datasets_db = DatasetsSessionLocal()
    try:
        datasets_without_unique_id = datasets_db.query(Dataset).filter(
            (Dataset.unique_id == None) | (Dataset.unique_id == "")
        ).all()

        if datasets_without_unique_id:
            print(f"[Database] Migrating {len(datasets_without_unique_id)} datasets to add unique_id...")
            for dataset in datasets_without_unique_id:
                dataset.unique_id = str(uuid.uuid4())
                print(f"[Database]   Dataset {dataset.id} ({dataset.name}): {dataset.unique_id}")

            datasets_db.commit()
            print(f"[Database] Migration complete: {len(datasets_without_unique_id)} datasets updated")
        else:
            print("[Database] No migration needed: All datasets have unique_id")
    except Exception as e:
        print(f"[Database] Migration warning: {e}")
        datasets_db.rollback()
    finally:
        datasets_db.close()

def get_db():
    """Get gallery database session (legacy compatibility)"""
    db = GallerySessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_gallery_db():
    """Get gallery database session"""
    db = GallerySessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_datasets_db():
    """Get datasets database session"""
    db = DatasetsSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_training_db():
    """Get training database session"""
    db = TrainingSessionLocal()
    try:
        yield db
    finally:
        db.close()
