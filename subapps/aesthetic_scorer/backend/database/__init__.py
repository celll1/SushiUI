"""
Database models and session management for Aesthetic Scorer.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent.parent / "aesthetic_scorer.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite specific
    echo=False,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Get database session (dependency for FastAPI)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database (create tables)."""
    from .models import Base, AestheticScorerSettings
    Base.metadata.create_all(bind=engine)
    print(f"[Database] Initialized at {DB_PATH}")

    # Initialize settings if not exists
    session = SessionLocal()
    try:
        settings = session.query(AestheticScorerSettings).first()
        if not settings:
            settings = AestheticScorerSettings()
            session.add(settings)
            session.commit()
            print(f"[Database] Created default settings")
    finally:
        session.close()


def get_settings(db: Session):
    """Get application settings (create if not exists)."""
    from .models import AestheticScorerSettings

    settings = db.query(AestheticScorerSettings).first()
    if not settings:
        settings = AestheticScorerSettings()
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings
