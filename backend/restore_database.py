"""
Restore gallery database from existing files.
This script recreates database entries from:
- Generated images in outputs/ directory (with PNG metadata)
"""

import os
import sys
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from datetime import datetime

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import GallerySessionLocal, gallery_engine
from database.models import GeneratedImage, GalleryBase

def restore_generated_images(db, outputs_dir):
    """Restore generated images from outputs/ directory."""
    print("\n[Restore] Scanning generated images...")

    outputs_path = Path(outputs_dir)
    image_files = list(outputs_path.glob("*.png"))

    print(f"[Restore] Found {len(image_files)} PNG files")

    restored = 0
    skipped = 0

    for img_path in image_files:
        try:
            # Extract parameters from metadata
            filename = img_path.name

            # Check if already exists (skip duplicates)
            existing = db.query(GeneratedImage).filter(GeneratedImage.filename == filename).first()
            if existing:
                skipped += 1
                continue

            # Read PNG metadata
            with Image.open(img_path) as img:
                metadata = img.info or {}

            prompt = metadata.get("prompt", "")
            negative_prompt = metadata.get("negative_prompt", "")
            model_name = metadata.get("model", "unknown")
            sampler = metadata.get("sampler", "unknown")
            steps = int(metadata.get("steps", 20))
            cfg_scale = float(metadata.get("cfg_scale", 7.0))
            seed = int(metadata.get("seed", 0))
            width = int(metadata.get("width", 1024))
            height = int(metadata.get("height", 1024))

            # Determine generation type from filename
            if filename.startswith("txt2img"):
                generation_type = "txt2img"
            elif filename.startswith("img2img"):
                generation_type = "img2img"
            elif filename.startswith("inpaint"):
                generation_type = "inpaint"
            else:
                generation_type = "unknown"

            # Get file creation time
            created_at = datetime.fromtimestamp(img_path.stat().st_mtime)

            # Create database entry
            db_image = GeneratedImage(
                filename=filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_name=model_name,
                sampler=sampler,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                width=width,
                height=height,
                generation_type=generation_type,
                parameters=metadata,
                created_at=created_at,
                is_favorite=False
            )

            db.add(db_image)
            restored += 1

            if restored % 100 == 0:
                print(f"[Restore] Progress: {restored}/{len(image_files)}")
                db.commit()

        except Exception as e:
            print(f"[Restore] Error processing {img_path.name}: {e}")
            db.rollback()  # Rollback on error and continue
            skipped += 1

    db.commit()
    print(f"[Restore] Restored {restored} images, skipped {skipped}")

    return restored

def main():
    print("[Restore] Starting gallery database restoration...")

    # Create all tables
    print("[Restore] Creating gallery database schema...")
    GalleryBase.metadata.create_all(bind=gallery_engine)

    # Create session
    db = GallerySessionLocal()

    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        outputs_dir = project_root / "outputs"

        # Restore generated images
        if outputs_dir.exists():
            restore_generated_images(db, outputs_dir)
        else:
            print(f"[Restore] Outputs directory not found: {outputs_dir}")

        print("\n[Restore] Gallery database restoration complete!")

    except Exception as e:
        print(f"[Restore] Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
