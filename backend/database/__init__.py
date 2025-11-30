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

    # Create tables for each database
    print("[Database] Initializing gallery.db...")
    GalleryBase.metadata.create_all(bind=gallery_engine)

    print("[Database] Initializing datasets.db...")
    DatasetBase.metadata.create_all(bind=datasets_engine)

    print("[Database] Initializing training.db...")
    TrainingBase.metadata.create_all(bind=training_engine)

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
