"""
Extract verification samples from dataset DB for layer pruning evaluation.

This script extracts random samples from a dataset in the SushiUI database
and saves them as a JSON file for layer pruning evaluation.
"""

import sys
import os
import json
import argparse
import sqlite3
import random
from pathlib import Path
from typing import List, Dict, Any

# Add backend to path for database models
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from database.models import Dataset, DatasetItem, DatasetCaption
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def extract_samples(
    db_path: str,
    dataset_id: str,
    num_samples: int,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Extract random samples from dataset.

    Args:
        db_path: Path to datasets.db
        dataset_id: Dataset unique ID
        num_samples: Number of samples to extract
        random_seed: Random seed for reproducibility

    Returns:
        List of sample dicts with image_path, caption, etc.
    """
    # Create database connection
    engine = create_engine(f"sqlite:///{db_path}")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Get dataset
        dataset = session.query(Dataset).filter(Dataset.unique_id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in database")

        print(f"[Extract] Dataset: {dataset.unique_id}")
        print(f"[Extract] Total items: {len(dataset.items)}")

        # Get all items
        items = list(dataset.items)

        if len(items) < num_samples:
            print(f"[Extract] WARNING: Requested {num_samples} samples, but only {len(items)} available")
            num_samples = len(items)

        # Random sampling
        random.seed(random_seed)
        sampled_items = random.sample(items, num_samples)

        # Extract data
        samples = []
        for item in sampled_items:
            # Get primary caption (tags)
            primary_caption = session.query(DatasetCaption).filter(
                DatasetCaption.item_id == item.id,
                DatasetCaption.caption_type == "tags"
            ).first()

            caption = primary_caption.content if primary_caption else ""

            sample = {
                "image_path": item.image_path,
                "caption": caption,
                "width": item.width,
                "height": item.height,
                "dataset_id": dataset_id,
            }
            samples.append(sample)

        print(f"[Extract] Extracted {len(samples)} samples")
        return samples

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Extract samples from dataset DB for layer pruning")
    parser.add_argument("--dataset-db", type=str, default="datasets.db", help="Path to datasets.db")
    parser.add_argument("--dataset-id", type=str, required=True, help="Dataset unique ID")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to extract")
    parser.add_argument("--output", type=str, default="samples.json", help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("Layer Pruning - Sample Extraction")
    print("=" * 60)
    print(f"Database: {args.dataset_db}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Num samples: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Check database exists
    if not Path(args.dataset_db).exists():
        print(f"ERROR: Database '{args.dataset_db}' not found")
        sys.exit(1)

    # Extract samples
    samples = extract_samples(
        db_path=args.dataset_db,
        dataset_id=args.dataset_id,
        num_samples=args.num_samples,
        random_seed=args.seed
    )

    # Validate image paths exist
    valid_samples = []
    for sample in samples:
        if Path(sample["image_path"]).exists():
            valid_samples.append(sample)
        else:
            print(f"[Extract] WARNING: Image not found: {sample['image_path']}")

    print(f"[Extract] Valid samples: {len(valid_samples)}/{len(samples)}")

    if len(valid_samples) == 0:
        print("ERROR: No valid samples found")
        sys.exit(1)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_samples, f, indent=2, ensure_ascii=False)

    print(f"[Extract] Saved {len(valid_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
