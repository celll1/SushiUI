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
            "width": item.width,
            "height": item.height,
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
        model_config = process_config.get('model', {})

        # Determine training method
        network_type = network_config.get('type', 'lora')

        if network_type == 'lora':
            print("[TrainRunner] Training method: LoRA")
            from core.lora_trainer import LoRATrainer

            # Get dtype settings from config
            weight_dtype = train_config.get('weight_dtype', 'fp16')
            training_dtype = train_config.get('dtype', 'fp16')  # 'dtype' is legacy name for training_dtype
            output_dtype = train_config.get('output_dtype', 'fp32')
            vae_dtype = model_config.get('vae_dtype', 'fp16')  # VAE-specific dtype (SDXL VAE works with fp16)
            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)  # Debug VRAM profiling (default: False)
            use_flash_attention = train_config.get('use_flash_attention', False)  # Flash Attention (default: False)
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)  # Min-SNR gamma weighting (default: 5.0)

            # Initialize trainer
            trainer = LoRATrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                lora_rank=network_config.get('linear', 16),
                lora_alpha=network_config.get('linear_alpha', 16),
                learning_rate=train_config.get('lr', 1e-4),
                weight_dtype=weight_dtype,
                training_dtype=training_dtype,
                output_dtype=output_dtype,
                vae_dtype=vae_dtype,
                mixed_precision=mixed_precision,
                debug_vram=debug_vram,
                use_flash_attention=use_flash_attention,
                min_snr_gamma=min_snr_gamma,
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
                # Pass total_steps_config to trainer; it will calculate epochs based on actual batch count
                # (batch count depends on bucketing, which is only known after dataset processing)
                num_epochs = None  # Will be calculated by trainer
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback (update DB only, no print to avoid cluttering tqdm output)
            def progress_callback(step: int, loss: float, lr: float):
                update_training_progress(training_db, run_id, step, loss, lr, run.total_steps)

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            # Prepare sample configuration
            # Note: YAML uses 'prompts', 'width', etc. (not 'sample_prompts', 'sample_width')
            sample_prompts = process_config['sample'].get('prompts', process_config['sample'].get('sample_prompts', []))
            sample_config = {
                'width': process_config['sample'].get('width', 1024),
                'height': process_config['sample'].get('height', 1024),
                'steps': process_config['sample'].get('sample_steps', 20),
                'cfg_scale': process_config['sample'].get('guidance_scale', 7.0),
                'sampler': process_config['sample'].get('sampler', 'euler'),
                'schedule_type': process_config['sample'].get('schedule_type', 'sgm_uniform'),
                'seed': process_config['sample'].get('seed', -1),
            }

            # Get debug parameters from config
            debug_latents = train_config.get('debug_latents', False)
            debug_latents_every = train_config.get('debug_latents_every', 50)

            # Get bucketing parameters from config
            enable_bucketing = train_config.get('enable_bucketing', False)
            base_resolutions = train_config.get('base_resolutions', [1024])
            bucket_strategy = train_config.get('bucket_strategy', 'resize')
            multi_resolution_mode = train_config.get('multi_resolution_mode', 'max')

            # Get latent caching parameters
            # Check datasets config first, then fall back to train config
            cache_latents_to_disk = True  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)

            # Start training
            trainer.train(
                dataset_items=dataset_items,
                num_epochs=num_epochs,
                target_steps=total_steps_config,  # Pass target steps for dynamic epoch calculation
                batch_size=train_config.get('batch_size', 1),
                save_every=process_config['save'].get('save_every', 100),
                save_every_unit=process_config['save'].get('save_every_unit', 'steps'),
                sample_every=process_config['sample'].get('sample_every', 100),
                sample_prompts=sample_prompts if sample_prompts else None,
                sample_config=sample_config if sample_prompts else None,
                progress_callback=progress_callback,
                resume_from_checkpoint=train_config.get('resume_from_checkpoint'),
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                # Bucketing parameters
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy=bucket_strategy,
                multi_resolution_mode=multi_resolution_mode,
                # Latent caching
                cache_latents_to_disk=cache_latents_to_disk,
                dataset_id=dataset.id,  # Use dataset.id from line 104
                # Checkpoint management
                max_step_saves_to_keep=process_config['save'].get('max_step_saves_to_keep'),
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
