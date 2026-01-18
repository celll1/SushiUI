"""
Training Utility Functions

This module contains utility functions for training-related operations.
"""

from pathlib import Path


def get_training_base_dir() -> str:
    """
    Get the base training directory from user settings.

    Priority:
    1. UserSettings.training_dir (if set in database)
    2. Default: {root_dir}/training

    Returns:
        Base training directory path
    """
    try:
        from database import get_gallery_db
        from database.models import UserSettings

        db = next(get_gallery_db())
        try:
            user_settings = db.query(UserSettings).first()
            if user_settings and user_settings.training_dir:
                # User configured training directory (from database)
                return user_settings.training_dir
        finally:
            db.close()
    except Exception as e:
        # Fallback to default if database query fails
        print(f"[Training] Warning: Failed to get training_dir from UserSettings: {e}")

    # Default training directory
    try:
        from config.settings import settings
        import os
        return os.path.join(settings.root_dir, "training")
    except Exception as e:
        print(f"[Training] Warning: Failed to get root_dir from settings: {e}")
        # Ultimate fallback
        return "training"
