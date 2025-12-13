"""
Training Utility Functions

This module contains utility functions for training-related operations.
"""

from pathlib import Path


def get_training_base_dir() -> str:
    """
    Get the base training directory from user settings.

    Returns:
        Base training directory path (default: "training")
    """
    try:
        from database import get_gallery_db
        from database.models import UserSettings

        db = next(get_gallery_db())
        try:
            settings = db.query(UserSettings).first()
            if settings and settings.training_dir:
                # User configured training directory
                print(f"[Training] Using user-configured training_dir: {settings.training_dir}")
                return settings.training_dir
            else:
                print(f"[Training] No training_dir in UserSettings, using default")
        finally:
            db.close()
    except Exception as e:
        # Fallback to default if database query fails
        print(f"[Training] Warning: Failed to get training_dir from settings: {e}")

    # Default training directory (relative to project root)
    print(f"[Training] Using default training_dir: training")
    return "training"
