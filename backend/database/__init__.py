from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from config.settings import settings
import os
import shutil

# Import bases from models
from .models import GalleryBase, DatasetBase, TrainingBase

# Create separate engines
# NullPool: no connection pooling — each session gets a fresh connection that is
# closed immediately on session.close().  SQLite connections are cheap to create
# and the database has its own writer lock, so pooling provides no benefit and
# causes QueuePool exhaustion when training threads + API requests run concurrently.
gallery_db_path = os.path.join(settings.root_dir, "gallery.db")
datasets_db_path = os.path.join(settings.root_dir, "datasets.db")
training_db_path = os.path.join(settings.root_dir, "training.db")

_sqlite_kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": NullPool}
gallery_engine  = create_engine(f"sqlite:///{gallery_db_path}",  **_sqlite_kwargs)
datasets_engine = create_engine(f"sqlite:///{datasets_db_path}", **_sqlite_kwargs)
training_engine = create_engine(f"sqlite:///{training_db_path}", **_sqlite_kwargs)

def _set_wal_mode(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA synchronous=NORMAL")

for _engine in (gallery_engine, datasets_engine, training_engine):
    event.listen(_engine, "connect", _set_wal_mode)

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

_db_initialized = False

def init_db():
    """Initialize all databases (no-op if already called)."""
    global _db_initialized
    if _db_initialized:
        return
    _db_initialized = True
    from .models import (
        GeneratedImage, StudioRenderJob, UserSettings,  # Gallery
        Dataset, DatasetItem, DatasetCaption, TagDictionary,  # Datasets
        TrainingRun, TrainingCheckpoint, TrainingSample,  # Training
        TaggerTrainingRun, TaggerTrainingMetrics  # Tagger Training
    )
    from .auto_migrate import auto_migrate_all_databases
    import uuid

    # Create tables for each database
    print("[Database] Initializing gallery.db...")
    GalleryBase.metadata.create_all(bind=gallery_engine)

    # A server restart must not leave a render job looking active forever.
    # Queued jobs are safe to resume only after their staging directory has
    # been revalidated, so both transient states are surfaced as interrupted.
    gallery_db = GallerySessionLocal()
    try:
        from datetime import datetime
        interrupted_staging_dirs = []
        interrupted = gallery_db.query(StudioRenderJob).filter(
            StudioRenderJob.state.in_(["queued", "running", "cancel_requested"])
        ).all()
        for job in interrupted:
            interrupted_staging_dirs.append(job.input_dir)
            job.state = "failed"
            job.error = "Backend restarted before the Studio render completed."
            job.finished_at = datetime.now()
        if interrupted:
            gallery_db.commit()
        staging_root = os.path.realpath(os.path.join(settings.cache_dir, "studio_render_jobs"))
        for staging_dir in interrupted_staging_dirs:
            if not isinstance(staging_dir, str):
                continue
            target = os.path.realpath(staging_dir)
            try:
                inside_root = os.path.commonpath([staging_root, target]) == staging_root
            except ValueError:
                inside_root = False
            if inside_root and target != staging_root:
                shutil.rmtree(target, ignore_errors=True)
    except Exception as exc:
        gallery_db.rollback()
        print(f"[Database] Studio render recovery warning: {exc}")
    finally:
        gallery_db.close()

    print("[Database] Initializing datasets.db...")
    DatasetBase.metadata.create_all(bind=datasets_engine)

    print("[Database] Initializing training.db...")
    TrainingBase.metadata.create_all(bind=training_engine)

    # Run auto-migration to add any missing columns
    auto_migrate_all_databases()

    # Migration: Add unique_id to existing datasets
    print("[Database] Running data migrations...")
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


def get_gallery_db_sync():
    """Get gallery database session (synchronous, non-generator version).

    Use this when you need a database session outside of FastAPI dependency injection.
    IMPORTANT: Caller is responsible for closing the session with db.close()

    Returns:
        Session: SQLAlchemy session for gallery database
    """
    return GallerySessionLocal()
