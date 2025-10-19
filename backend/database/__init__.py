from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
from config.settings import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database with automatic migration"""
    # First, try to create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Then run automatic migration to add any missing columns
    try:
        from .auto_migrate import auto_migrate
        auto_migrate()
    except Exception as e:
        print(f"[Database] Warning: Auto-migration failed: {e}")
        # Continue even if migration fails - let the app try to run

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
