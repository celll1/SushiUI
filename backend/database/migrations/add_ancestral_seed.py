"""
Migration script to add ancestral_seed field to generated_images table

Run this script to update existing database:
python backend/database/migrations/add_ancestral_seed.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import create_engine, text
from config.settings import settings

def migrate():
    """Add ancestral_seed column to generated_images table"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("PRAGMA table_info(generated_images)"))
        existing_columns = {row[1] for row in result}

        # Add ancestral_seed column if it doesn't exist
        if 'ancestral_seed' not in existing_columns:
            print("Adding ancestral_seed column...")
            conn.execute(text("ALTER TABLE generated_images ADD COLUMN ancestral_seed INTEGER"))
            conn.commit()
            print("ancestral_seed column added successfully!")
        else:
            print("ancestral_seed column already exists, skipping...")

        print("Migration completed successfully!")

if __name__ == "__main__":
    migrate()
