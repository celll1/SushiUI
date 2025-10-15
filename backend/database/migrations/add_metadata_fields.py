"""
Migration script to add new metadata fields to generated_images table

Run this script to update existing database:
python backend/database/migrations/add_metadata_fields.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlalchemy import create_engine, text
from config.settings import settings

def migrate():
    """Add new metadata columns to generated_images table"""
    engine = create_engine(settings.database_url)

    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("PRAGMA table_info(generated_images)"))
        existing_columns = {row[1] for row in result}

        # Add new columns if they don't exist
        if 'image_hash' not in existing_columns:
            print("Adding image_hash column...")
            conn.execute(text("ALTER TABLE generated_images ADD COLUMN image_hash VARCHAR"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_generated_images_image_hash ON generated_images(image_hash)"))
            conn.commit()

        if 'source_image_hash' not in existing_columns:
            print("Adding source_image_hash column...")
            conn.execute(text("ALTER TABLE generated_images ADD COLUMN source_image_hash VARCHAR"))
            conn.commit()

        if 'mask_data' not in existing_columns:
            print("Adding mask_data column...")
            conn.execute(text("ALTER TABLE generated_images ADD COLUMN mask_data VARCHAR"))
            conn.commit()

        if 'lora_names' not in existing_columns:
            print("Adding lora_names column...")
            conn.execute(text("ALTER TABLE generated_images ADD COLUMN lora_names VARCHAR"))
            conn.commit()

        # Add index on created_at if not exists
        print("Adding index on created_at...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_generated_images_created_at ON generated_images(created_at)"))
        conn.commit()

        print("Migration completed successfully!")

if __name__ == "__main__":
    migrate()
