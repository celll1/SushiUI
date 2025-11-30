"""
Restore database from existing files.
This script recreates database entries from:
- Generated images in outputs/ directory (with PNG metadata)
- Dataset directories
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

from database import engine, Base, SessionLocal
from database.models import GeneratedImage, Dataset, DatasetItem, DatasetCaption

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
            # Read PNG metadata
            with Image.open(img_path) as img:
                metadata = img.info or {}

            # Extract parameters from metadata
            filename = img_path.name
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
            skipped += 1

    db.commit()
    print(f"[Restore] Restored {restored} images, skipped {skipped}")

    return restored

def restore_datasets(db, dataset_paths):
    """Restore datasets from directory paths."""
    print("\n[Restore] Restoring datasets...")

    restored = 0

    for dataset_path in dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            print(f"[Restore] Dataset not found: {dataset_path}")
            continue

        # Create dataset entry
        dataset_name = path.name

        # Check if already exists
        existing = db.query(Dataset).filter(Dataset.name == dataset_name).first()
        if existing:
            print(f"[Restore] Dataset '{dataset_name}' already exists, skipping")
            continue

        dataset = Dataset(
            name=dataset_name,
            path=str(path),
            created_at=datetime.utcnow()
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        # Scan for images in dataset
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.webp']:
            image_files.extend(path.glob(f"*{ext}"))

        print(f"[Restore] Found {len(image_files)} images in {dataset_name}")

        # Add dataset items
        for img_path in image_files:
            item = DatasetItem(
                dataset_id=dataset.id,
                image_path=str(img_path),
                created_at=datetime.utcnow()
            )
            db.add(item)
            db.commit()
            db.refresh(item)

            # Try to find caption file
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()

                caption = DatasetCaption(
                    item_id=item.id,
                    caption_type="tags",
                    content=caption_text,
                    created_at=datetime.utcnow()
                )
                db.add(caption)

        db.commit()
        restored += 1
        print(f"[Restore] Restored dataset: {dataset_name}")

    print(f"[Restore] Restored {restored} datasets")
    return restored

def main():
    print("[Restore] Starting database restoration...")

    # Create all tables
    print("[Restore] Creating database schema...")
    Base.metadata.create_all(bind=engine)

    # Create session
    db = SessionLocal()

    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        outputs_dir = project_root / "outputs"

        # Restore generated images
        if outputs_dir.exists():
            restore_generated_images(db, outputs_dir)
        else:
            print(f"[Restore] Outputs directory not found: {outputs_dir}")

        # Restore datasets
        dataset_paths = [
            "/m/dataset_control/cref",
            "/m/dataset_control/cref-d",
            "/m/dataset_control/test",
        ]
        restore_datasets(db, dataset_paths)

        print("\n[Restore] Database restoration complete!")

    except Exception as e:
        print(f"[Restore] Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
