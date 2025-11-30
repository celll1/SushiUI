"""
Training Runner for SushiUI

Entry point for training processes. Reads YAML config and executes training.
Can be run as: python -m core.train_runner config.yaml run_id
"""

import sys
import yaml
import os
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_training_db, get_datasets_db
from database.models import TrainingRun, Dataset, DatasetItem, DatasetCaption
from sqlalchemy.orm import Session


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_items(db: Session, dataset_id: int) -> list:
    """Get all items from dataset."""
    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

    dataset_items = []
    for item in items:
        # Get primary caption from dataset_captions table
        primary_caption = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.caption_type == "tags"
        ).first()

        caption = primary_caption.content if primary_caption else ""

        dataset_items.append({
            "image_path": item.image_path,
            "caption": caption,
        })

    return dataset_items


def update_training_progress(
    db: Session,
    run_id: int,
    step: int,
    loss: float,
    lr: float,
    total_steps: int
):
    """Update training run progress in database."""
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        run.current_step = step
        run.loss = loss
        run.learning_rate = lr
        run.progress = (step / total_steps) * 100.0
        db.commit()


def main():
    """Main training entry point."""
    if len(sys.argv) < 3:
        print("Usage: python -m core.train_runner <config_path> <run_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_id = int(sys.argv[2])

    print(f"[TrainRunner] Starting training")
    print(f"[TrainRunner] Config: {config_path}")
    print(f"[TrainRunner] Run ID: {run_id}")

    # Load config
    config = load_config(config_path)
    print(f"[TrainRunner] Loaded config: {config['job']}")

    # Get database sessions (separate DBs for training and datasets)
    training_db_gen = get_training_db()
    training_db = next(training_db_gen)

    datasets_db_gen = get_datasets_db()
    datasets_db = next(datasets_db_gen)

    try:
        # Get training run info (from training.db)
        run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run:
            print(f"[TrainRunner] ERROR: Training run {run_id} not found")
            sys.exit(1)

        # Get dataset (from datasets.db)
        dataset = datasets_db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        if not dataset:
            print(f"[TrainRunner] ERROR: Dataset {run.dataset_id} not found")
            sys.exit(1)

        print(f"[TrainRunner] Dataset: {dataset.name} ({dataset.path})")

        # Get dataset items (from datasets.db)
        dataset_items = get_dataset_items(datasets_db, dataset.id)
        print(f"[TrainRunner] Dataset items: {len(dataset_items)}")

        if len(dataset_items) == 0:
            print("[TrainRunner] ERROR: Dataset is empty")
            sys.exit(1)

        # Extract training parameters from config
        process_config = config['config']['process'][0]
        train_config = process_config['train']
        network_config = process_config.get('network', {})

        # Determine training method
        network_type = network_config.get('type', 'lora')

        if network_type == 'lora':
            print("[TrainRunner] Training method: LoRA")
            from core.lora_trainer import LoRATrainer

            # Initialize trainer
            trainer = LoRATrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                lora_rank=network_config.get('linear', 16),
                lora_alpha=network_config.get('linear_alpha', 16),
                learning_rate=train_config.get('lr', 1e-4),
            )

            # Setup optimizer
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')
            trainer.setup_optimizer(optimizer_type, lr_scheduler_type)

            # Determine epochs or steps
            num_epochs = train_config.get('epochs', None)
            total_steps_config = train_config.get('steps', None)

            if num_epochs:
                print(f"[TrainRunner] Training for {num_epochs} epochs")
            elif total_steps_config:
                num_epochs = max(1, total_steps_config // len(dataset_items))
                print(f"[TrainRunner] Training for {total_steps_config} steps (~{num_epochs} epochs)")
            else:
                num_epochs = 1

            # Progress callback (update DB only, no print to avoid cluttering tqdm output)
            def progress_callback(step: int, loss: float, lr: float):
                update_training_progress(training_db, run_id, step, loss, lr, run.total_steps)

            # Start training
            trainer.train(
                dataset_items=dataset_items,
                num_epochs=num_epochs,
                batch_size=train_config.get('batch_size', 1),
                save_every=process_config['save'].get('save_every', 100),
                save_every_unit=process_config['save'].get('save_every_unit', 'steps'),
                sample_every=process_config['sample'].get('sample_every', 100),
                progress_callback=progress_callback,
                resume_from_checkpoint=train_config.get('resume_from_checkpoint'),
            )

            print("[TrainRunner] Training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        else:
            print(f"[TrainRunner] ERROR: Unsupported network type: {network_type}")
            sys.exit(1)

    except Exception as e:
        print(f"[TrainRunner] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Update run status to failed (in training.db)
        run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(e)
            training_db.commit()

        sys.exit(1)

    finally:
        training_db.close()
        datasets_db.close()


if __name__ == "__main__":
    main()
