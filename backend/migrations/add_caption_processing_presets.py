"""
Migration: Add caption_processing_presets table

This migration adds a new table for storing reusable caption processing presets.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_datasets_db
from database.models import DatasetBase, CaptionProcessingPreset
from sqlalchemy import inspect


def migrate():
    """Run migration"""
    print("[Migration] Adding caption_processing_presets table...")

    db = next(get_datasets_db())
    engine = db.get_bind()

    # Check if table already exists
    inspector = inspect(engine)
    if "caption_processing_presets" in inspector.get_table_names():
        print("[Migration] Table 'caption_processing_presets' already exists, skipping...")
        return

    # Create table
    DatasetBase.metadata.create_all(bind=engine, tables=[CaptionProcessingPreset.__table__])
    print("[Migration] Created table 'caption_processing_presets'")

    # Add some default presets
    default_presets = [
        {
            "name": "Conservative (Light Processing)",
            "description": "Light caption processing - minimal dropout, simple shuffle",
            "config": {
                "caption_dropout_rate": 0.0,
                "token_dropout_rate": 0.05,
                "keep_tokens": 1,
                "shuffle_tokens": True,
                "shuffle_per_epoch": True,
                "shuffle_keep_first_n": 1,
                "tag_dropout_rate": 0.0,
                "tag_dropout_per_epoch": False,
                "tag_dropout_keep_first_n": 0,
                "tag_dropout_category_rates": {},
                "tag_dropout_exclude_person_count": False,
            }
        },
        {
            "name": "Aggressive (Heavy Processing)",
            "description": "Aggressive processing - high dropout, category-specific rates",
            "config": {
                "caption_dropout_rate": 0.1,
                "token_dropout_rate": 0.0,
                "keep_tokens": 0,
                "shuffle_tokens": True,
                "shuffle_per_epoch": True,
                "shuffle_keep_first_n": 0,
                "shuffle_tag_groups": ["Character", "General"],
                "shuffle_groups_together": False,
                "tag_group_dir": "taggroup",
                "exclude_person_count_from_shuffle": True,
                "tag_dropout_rate": 0.1,
                "tag_dropout_per_epoch": True,
                "tag_dropout_keep_first_n": 0,
                "tag_dropout_category_rates": {
                    "Character": 0.05,
                    "General": 0.3,
                },
                "tag_dropout_exclude_person_count": True,
            }
        },
        {
            "name": "Tag Group Focused",
            "description": "Tag group-based shuffle with category dropout (requires tag groups)",
            "config": {
                "caption_dropout_rate": 0.0,
                "token_dropout_rate": 0.0,
                "keep_tokens": 0,
                "shuffle_tokens": True,
                "shuffle_per_epoch": True,
                "shuffle_keep_first_n": 0,
                "shuffle_tag_groups": ["Character", "General", "Copyright"],
                "shuffle_groups_together": False,
                "tag_group_dir": "taggroup",
                "exclude_person_count_from_shuffle": True,
                "tag_dropout_rate": 0.2,
                "tag_dropout_per_epoch": True,
                "tag_dropout_keep_first_n": 0,
                "tag_dropout_category_rates": {
                    "Character": 0.1,
                    "General": 0.25,
                    "Copyright": 0.0,
                },
                "tag_dropout_exclude_person_count": True,
            }
        },
    ]

    for preset_data in default_presets:
        preset = CaptionProcessingPreset(**preset_data)
        db.add(preset)

    db.commit()
    print(f"[Migration] Added {len(default_presets)} default presets")
    print("[Migration] Migration complete!")


if __name__ == "__main__":
    migrate()
