"""
Training Runner for SushiUI

Entry point for training processes. Reads YAML config and executes training.
Can be run as: python -m core.train_runner config.yaml run_id
"""

import sys
import yaml
import os
import signal
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add backend directory to path for imports (extensions, database, etc.)
backend_dir = Path(__file__).parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

from database import get_training_db, get_datasets_db
from database.models import TrainingRun, Dataset, DatasetItem, DatasetCaption
from sqlalchemy.orm import Session
from core.training.caption_processor import process_caption, get_default_caption_processing_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_items(db: Session, dataset_id: int, epoch_num: int = 0) -> list:
    """
    Get all items from dataset with caption processing applied.

    Args:
        db: Database session
        dataset_id: Dataset ID
        epoch_num: Current epoch number (for per-epoch shuffle/dropout)

    Returns:
        List of dataset items with processed captions
    """
    # Get dataset and its caption processing config
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise ValueError(f"Dataset {dataset_id} not found")

    # Get caption processing config (or defaults)
    caption_config = dataset.caption_processing or get_default_caption_processing_config()

    # Debug: Log caption config for first item only
    if epoch_num == 0:
        print(f"[TrainRunner] Caption config for dataset {dataset_id}:")
        print(f"  category_order: {caption_config.get('category_order', None)}")
        print(f"  normalize_tags: {caption_config.get('normalize_tags', True)}")
        print(f"  shuffle_tokens: {caption_config.get('shuffle_tokens', False)}")

    items = db.query(DatasetItem).filter(DatasetItem.dataset_id == dataset_id).all()

    dataset_items = []
    for item in items:
        # Get primary caption from dataset_captions table
        primary_caption = db.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.caption_type == "tags"
        ).first()

        raw_caption = primary_caption.content if primary_caption else ""

        # Apply caption processing (dropout, shuffle, etc.)
        processed_caption = process_caption(
            caption=raw_caption,
            epoch_num=epoch_num,
            item_path=item.image_path,
            normalize_tags=caption_config.get("normalize_tags", True),
            category_order=caption_config.get("category_order", None),
            caption_dropout_rate=caption_config.get("caption_dropout_rate", 0.0),
            token_dropout_rate=caption_config.get("token_dropout_rate", 0.0),
            keep_tokens=caption_config.get("keep_tokens", 0),
            shuffle_tokens=caption_config.get("shuffle_tokens", False),
            shuffle_per_epoch=caption_config.get("shuffle_per_epoch", False),
            shuffle_keep_first_n=caption_config.get("shuffle_keep_first_n", 0),
            shuffle_tag_groups=caption_config.get("shuffle_tag_groups", None),
            shuffle_groups_together=caption_config.get("shuffle_groups_together", False),
            tag_group_dir=caption_config.get("tag_group_dir", "taglist"),
            exclude_person_count_from_shuffle=caption_config.get("exclude_person_count_from_shuffle", False),
            tag_dropout_rate=caption_config.get("tag_dropout_rate", 0.0),
            tag_dropout_per_epoch=caption_config.get("tag_dropout_per_epoch", False),
            tag_dropout_keep_first_n=caption_config.get("tag_dropout_keep_first_n", 0),
            tag_dropout_category_rates=caption_config.get("tag_dropout_category_rates", {}),
            tag_dropout_exclude_person_count=caption_config.get("tag_dropout_exclude_person_count", False),
        )

        dataset_items.append({
            "image_path": item.image_path,
            "caption": processed_caption,
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

    # Set up signal handlers to convert SIGTERM to KeyboardInterrupt
    # This allows graceful shutdown with checkpoint saving when user stops training
    def signal_handler(signum, frame):
        print(f"\n[TrainRunner] Received signal {signum}, converting to KeyboardInterrupt for graceful shutdown...")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Also handle Ctrl+C explicitly
    print(f"[TrainRunner] Signal handlers registered (SIGTERM, SIGINT)")

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

        # Get dataset configs (support multiple datasets)
        dataset_configs = run.dataset_configs or []
        if not dataset_configs and run.dataset_id:
            # Fallback to legacy single dataset
            dataset_configs = [{"dataset_id": run.dataset_id, "caption_types": [], "filters": {}}]

        if not dataset_configs:
            print("[TrainRunner] ERROR: No datasets configured")
            sys.exit(1)

        print(f"[TrainRunner] Loading {len(dataset_configs)} dataset(s)...")

        # Load all datasets and combine items
        all_dataset_items = []
        dataset_unique_ids = []  # Collect unique IDs for cache management
        for i, ds_config in enumerate(dataset_configs):
            dataset_id = ds_config["dataset_id"]
            dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                print(f"[TrainRunner] ERROR: Dataset {dataset_id} not found")
                sys.exit(1)

            print(f"[TrainRunner] Dataset {i+1}: {dataset.name} ({dataset.path})")
            dataset_unique_ids.append(dataset.unique_id)

            # Get dataset items and tag with dataset_unique_id for cache management
            dataset_items = get_dataset_items(datasets_db, dataset_id)
            print(f"[TrainRunner]   Items: {len(dataset_items)}")

            # Add dataset_unique_id to each item for cache management
            for item in dataset_items:
                item["dataset_unique_id"] = dataset.unique_id

            all_dataset_items.extend(dataset_items)

        print(f"[TrainRunner] Total dataset items: {len(all_dataset_items)}")

        if len(all_dataset_items) == 0:
            print("[TrainRunner] ERROR: All datasets are empty")
            sys.exit(1)

        # Use combined dataset items
        dataset_items = all_dataset_items

        # Extract training parameters from config
        process_config = config['config']['process'][0]
        train_config = process_config['train']
        network_config = process_config.get('network', {})
        model_config = process_config.get('model', {})

        # Determine training method
        network_type = network_config.get('type', 'lora')

        if network_type == 'lora':
            print("[TrainRunner] Training method: LoRA")
            from core.training.lora_trainer import LoRATrainer

            # Get dtype settings from config
            weight_dtype = train_config.get('weight_dtype', 'fp16')
            training_dtype = train_config.get('dtype', 'fp16')  # 'dtype' is legacy name for training_dtype
            output_dtype = train_config.get('output_dtype', 'fp32')
            vae_dtype = model_config.get('vae_dtype', 'fp16')  # VAE-specific dtype (SDXL VAE works with fp16)

            # Z-Image requires BFloat16 for numerical stability (trained with bf16)
            if 'z-image' in run.base_model_path.lower() or 'zimage' in run.base_model_path.lower():
                print("[TrainRunner] Z-Image model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)  # Debug VRAM profiling (default: False)
            use_flash_attention = train_config.get('use_flash_attention', False)  # Flash Attention (default: False)
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)  # Min-SNR gamma weighting (default: 5.0)

            # Get component-specific learning rates from train_config
            unet_lr = train_config.get('unet_lr')
            text_encoder_lr = train_config.get('text_encoder_lr')
            text_encoder_1_lr = train_config.get('text_encoder_1_lr')
            text_encoder_2_lr = train_config.get('text_encoder_2_lr')

            # Initialize trainer
            trainer = LoRATrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,  # Pass run_name for checkpoint filename generation
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
                # Component-specific learning rates
                unet_lr=unet_lr,
                text_encoder_lr=text_encoder_lr,
                text_encoder_1_lr=text_encoder_1_lr,
                text_encoder_2_lr=text_encoder_2_lr,
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

            # Total steps callback (called once when actual total_steps is determined)
            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

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
            force_recache = False  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Create reload_dataset_callback for per-epoch caption processing
            def reload_dataset_for_epoch(epoch_num: int) -> list:
                """Reload all datasets with caption processing for the current epoch"""
                all_items = []
                for ds_config in dataset_configs:
                    dataset_id = ds_config["dataset_id"]
                    dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
                    if not dataset:
                        print(f"[TrainRunner] ERROR: Dataset {dataset_id} not found during reload")
                        continue

                    items = get_dataset_items(datasets_db, dataset_id, epoch_num=epoch_num)

                    # Add dataset_unique_id to each item for cache management
                    for item in items:
                        item["dataset_unique_id"] = dataset.unique_id

                    all_items.extend(items)
                return all_items

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
                update_total_steps_callback=update_total_steps_callback,
                reload_dataset_callback=reload_dataset_for_epoch,  # Reload dataset per epoch for caption processing
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
                dataset_unique_ids=dataset_unique_ids,
                force_recache=force_recache,
                # Checkpoint management
                max_step_saves_to_keep=process_config['save'].get('max_step_saves_to_keep'),
                # DB tracking
                run_id=run_id,
            )

            print("[TrainRunner] Training completed successfully!")

            # Update run status
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            training_db.commit()

        elif network_type == 'full_finetune':
            print("[TrainRunner] Training method: Full Parameter Fine-Tuning")
            from core.training.full_parameter_trainer import FullParameterTrainer

            # Get dtype settings from config
            weight_dtype = train_config.get('weight_dtype', 'fp16')
            training_dtype = train_config.get('dtype', 'fp16')  # 'dtype' is legacy name for training_dtype
            output_dtype = train_config.get('output_dtype', 'fp32')
            vae_dtype = model_config.get('vae_dtype', 'fp16')  # VAE-specific dtype (SDXL VAE works with fp16)

            # Z-Image requires BFloat16 for numerical stability (trained with bf16)
            if 'z-image' in run.base_model_path.lower() or 'zimage' in run.base_model_path.lower():
                print("[TrainRunner] Z-Image model detected: forcing training_dtype=bf16 for numerical stability")
                training_dtype = 'bf16'
                weight_dtype = 'bf16'

            mixed_precision = train_config.get('mixed_precision', True)
            debug_vram = train_config.get('debug_vram', False)  # Debug VRAM profiling (default: False)
            use_flash_attention = train_config.get('use_flash_attention', False)  # Flash Attention (default: False)
            min_snr_gamma = train_config.get('min_snr_gamma', 5.0)  # Min-SNR gamma weighting (default: 5.0)

            # Initialize trainer
            trainer = FullParameterTrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
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
                num_epochs = None  # Will be calculated by trainer
                print(f"[TrainRunner] Training for {total_steps_config} steps (epochs will be calculated by trainer)")
            else:
                num_epochs = 1

            # Progress callback
            def progress_callback(step: int, loss: float, lr: float):
                update_training_progress(training_db, run_id, step, loss, lr, run.total_steps)

            # Total steps callback
            def update_total_steps_callback(total_steps: int):
                print(f"[TrainRunner] Updating total_steps in DB: {total_steps}")
                run.total_steps = total_steps
                training_db.commit()

            # Update status to running
            run.status = "running"
            training_db.commit()
            print("[TrainRunner] Status updated to 'running'")

            # Prepare sample configuration
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
            cache_latents_to_disk = True  # Default
            force_recache = False  # Default
            if 'datasets' in process_config and len(process_config['datasets']) > 0:
                cache_latents_to_disk = process_config['datasets'][0].get('cache_latents_to_disk', True)
                force_recache = process_config['datasets'][0].get('force_recache', False)

            # Create reload_dataset_callback for per-epoch caption processing
            def reload_dataset_for_epoch(epoch_num: int) -> list:
                """Reload all datasets with caption processing for the current epoch"""
                all_items = []
                for ds_config in dataset_configs:
                    dataset_id = ds_config["dataset_id"]
                    dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
                    if not dataset:
                        print(f"[TrainRunner] ERROR: Dataset {dataset_id} not found during reload")
                        continue

                    items = get_dataset_items(datasets_db, dataset_id, epoch_num=epoch_num)

                    # Add dataset_unique_id to each item for cache management
                    for item in items:
                        item["dataset_unique_id"] = dataset.unique_id

                    all_items.extend(items)
                return all_items

            # Start training
            trainer.train(
                dataset_items=dataset_items,
                num_epochs=num_epochs,
                target_steps=total_steps_config,
                batch_size=train_config.get('batch_size', 1),
                save_every=process_config['save'].get('save_every', 100),
                save_every_unit=process_config['save'].get('save_every_unit', 'steps'),
                sample_every=process_config['sample'].get('sample_every', 100),
                sample_prompts=sample_prompts if sample_prompts else None,
                sample_config=sample_config if sample_prompts else None,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                reload_dataset_callback=reload_dataset_for_epoch,
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
                dataset_unique_ids=dataset_unique_ids,
                force_recache=force_recache,
                # Checkpoint management
                max_step_saves_to_keep=process_config['save'].get('max_step_saves_to_keep'),
                # DB tracking
                run_id=run_id,
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
