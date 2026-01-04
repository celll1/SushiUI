"""
Training Runner for SushiUI

Entry point for training processes. Reads YAML config and executes training.
Can be run as: python -m core.train_runner config.yaml run_id
"""

import sys
import yaml
import os
import signal
import torch
import gc
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add backend directory to path for imports (extensions, database, etc.)
backend_dir = Path(__file__).parent.parent.parent  # backend/
sys.path.insert(0, str(backend_dir))

from database import get_training_db, get_datasets_db
from database.models import TrainingRun, Dataset, DatasetItem, DatasetCaption
from sqlalchemy.orm import Session
from core.training.caption_processor import process_caption, get_default_caption_processing_config


class TeeOutput:
    """
    Redirects output to both console and file (like Unix tee command).
    """
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.console.write(message)
        self.console.flush()
        if self.file:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        self.console.flush()
        if self.file:
            self.file.flush()


class TrainingLogger:
    """
    Logger for training that supports both console+file and file-only output.

    Usage:
        logger.info("This goes to both console and file")
        logger.log_only("This goes only to file (verbose logs)")
    """
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.original_stdout = sys.stdout

    def info(self, message):
        """Print to both console and log file."""
        print(message)

    def log_only(self, message):
        """Print only to log file, not to console (for verbose logs)."""
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()
        # If no log file, silently ignore (don't spam console)


# Global logger instance (initialized in main)
logger: TrainingLogger = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataset_items(db: Session, dataset_id: int, epoch_num: int = 0, run_id: int = None, caption_types: list = None) -> list:
    """
    Get all items from dataset with caption processing applied.

    Args:
        db: Database session
        dataset_id: Dataset ID
        epoch_num: Current epoch number (for per-epoch shuffle/dropout)
        run_id: Training run ID (for phase progress updates)
        caption_types: List of caption types to use (e.g., ["tags", "natural_language"]). If None/empty, auto-select.

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
    total_items = len(items)
    print(f"[TrainRunner] Processing {total_items} items from dataset {dataset_id}...")

    # Helper function to update phase progress
    def update_phase_progress(phase: str, progress: float, detail: str = None):
        if run_id is None:
            return
        try:
            # Create separate training DB session (db is for datasets.db)
            training_db = next(get_training_db())
            try:
                run = training_db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
                if run:
                    run.phase = phase
                    run.phase_progress = progress
                    if detail:
                        run.phase_detail = detail
                    training_db.commit()
            finally:
                training_db.close()
        except Exception as e:
            print(f"[TrainRunner] Warning: Failed to update phase progress: {e}")

    dataset_items = []
    # Check if category_order is enabled
    has_category_order = caption_config.get("category_order") and len(caption_config.get("category_order", [])) > 0

    # Determine which caption types to use
    # Priority: 1) caption_types parameter (from dataset_configs, legacy)
    #           2) caption_config.caption_types (from dataset.caption_processing, new standard)
    #           3) Auto-select (priority: tags > natural_language > others)
    if caption_types:
        # Legacy: from dataset_configs (Training Config page)
        selected_caption_types = caption_types
        print(f"[TrainRunner] Using selected caption types (from dataset_configs): {selected_caption_types}")
    elif caption_config.get("caption_types"):
        # New standard: from caption_processing (Dataset Management page)
        selected_caption_types = caption_config.get("caption_types")
        print(f"[TrainRunner] Using selected caption types (from caption_processing): {selected_caption_types}")
    else:
        # Auto-select: priority order: tags > natural_language > others
        selected_caption_types = None  # Will auto-select per item
        print(f"[TrainRunner] No caption types specified - will auto-select per item (priority: tags > natural_language)")

    # Update phase to "initializing" for dataset loading
    update_phase_progress("initializing", 0.0, f"Loading dataset: 0/{total_items} items")

    for idx, item in enumerate(items):
        # Phase update every 1000 items (for UI responsiveness)
        if (idx + 1) % 1000 == 0:
            progress_pct = ((idx + 1) / total_items) * 100.0
            update_phase_progress("initializing", progress_pct, f"Loading dataset: {idx + 1}/{total_items} items")

        # Console log every 10000 items (to reduce log spam)
        if (idx + 1) % 10000 == 0:
            progress_pct = ((idx + 1) / total_items) * 100.0
            print(f"[TrainRunner] Processed {idx + 1}/{total_items} items ({progress_pct:.1f}%)")

        # Get caption based on selected caption_types
        primary_caption = None
        if selected_caption_types:
            # Try each selected caption type in order
            for caption_type in selected_caption_types:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id,
                    DatasetCaption.caption_type == caption_type
                ).first()
                if primary_caption:
                    break
        else:
            # Auto-select: try "tags" first, then "natural_language", then any other
            for caption_type in ["tags", "natural_language"]:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id,
                    DatasetCaption.caption_type == caption_type
                ).first()
                if primary_caption:
                    break

            # If still not found, use any caption type
            if not primary_caption:
                primary_caption = db.query(DatasetCaption).filter(
                    DatasetCaption.item_id == item.id
                ).first()

        raw_caption = primary_caption.content if primary_caption else ""

        # Check if caption is tags format (Danbooru tags) or natural language
        is_tags_format = primary_caption.is_tags_format if primary_caption and hasattr(primary_caption, 'is_tags_format') else True  # Default to True for backward compatibility

        if is_tags_format:
            # Tags format: Apply tag processing (normalization, shuffle, dropout, etc.)
            # Check if tag_data is available (pre-categorized tags for fast processing)
            tag_data_available = primary_caption and primary_caption.tag_data

            if tag_data_available:
                # Fast path: Use pre-categorized tag_data
                import json
                try:
                    tag_data = json.loads(primary_caption.tag_data)
                except:
                    tag_data = None
                    tag_data_available = False

            if tag_data_available and tag_data:
                # Fast per-epoch shuffle/dropout using pre-categorized tags
                from core.training.caption_processor import process_caption_with_tag_data
                processed_caption = process_caption_with_tag_data(
                    tag_data=tag_data,
                    epoch_num=epoch_num,
                    item_path=item.image_path,
                    caption_config=caption_config,
                )
            else:
                # Legacy path: Use process_caption with category lookup
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
        else:
            # Natural language: Use caption as-is (no tag processing)
            processed_caption = raw_caption
            print(f"[TrainRunner] Natural language caption (skipping tag processing): {raw_caption[:50]}...")

        dataset_items.append({
            "image_path": item.image_path,
            "caption": processed_caption,
            "width": item.width,
            "height": item.height,
        })

    # Mark dataset loading as complete
    update_phase_progress("initializing", 100.0, f"Loaded {total_items}/{total_items} items")
    print(f"[TrainRunner] Completed processing {total_items} items from dataset {dataset_id}")
    return dataset_items


def update_training_progress(
    db: Session,
    run_id: int,
    phase: str,
    step: int,
    total: int,
    epoch: int = 0,
    loss: float = None,
    lr: float = None,
):
    """
    Update training run progress in database with phase-based progress.

    Args:
        db: Database session
        run_id: Training run ID
        phase: Current phase ("initializing", "latent_cache", "text_encoder_cache", "training")
        step: Current step within phase
        total: Total steps in phase
        epoch: Current epoch (training phase only)
        loss: Current loss (training phase only)
        lr: Learning rate (training phase only)
    """
    run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
    if run:
        # Update phase
        run.phase = phase

        # Calculate phase progress (cap at 100% to prevent exceeding due to mid-epoch resume)
        phase_progress = (step / total * 100.0) if total > 0 else 0.0
        phase_progress = min(phase_progress, 100.0)  # Cap at 100%
        run.phase_progress = phase_progress

        # Update phase detail
        if phase == "initializing":
            run.phase_detail = f"Loading dataset: {step}/{total} items"
        elif phase == "latent_cache":
            run.phase_detail = f"Generating latent cache: {step}/{total} items"
        elif phase == "text_encoder_cache":
            run.phase_detail = f"Encoding captions: {step}/{total} captions"
        elif phase == "training":
            run.phase_detail = f"Epoch {epoch}, Step {step}/{total}"
            run.current_step = step
            if loss is not None:
                run.loss = loss
            if lr is not None:
                run.learning_rate = lr
            # Overall progress = phase_progress during training (capped at 100%)
            run.progress = phase_progress

        db.commit()


def main():
    """Main training entry point."""
    # Fix Windows cp932 encoding issue: force UTF-8 for stdout/stderr
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    if len(sys.argv) < 3:
        print("Usage: python -m core.train_runner <config_path> <run_id>")
        sys.exit(1)

    config_path = sys.argv[1]
    run_id = int(sys.argv[2])

    print(f"[TrainRunner] Starting training")
    print(f"[TrainRunner] Config: {config_path}")
    print(f"[TrainRunner] Run ID: {run_id}")

    # Set up training log file (will be created after we load config and get output_dir)
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Declare global logger
    global logger

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

    # ============================================================
    # Set Up Training Log File
    # ============================================================
    try:
        # Get training folder from config
        training_folder = config['config']['process'][0].get('training_folder')
        if training_folder:
            training_folder_path = Path(training_folder)

            # Create logs directory
            logs_dir = training_folder_path / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"training_{timestamp}.log"
            log_file_path = logs_dir / log_filename

            # Open log file
            log_file = open(log_file_path, 'w', encoding='utf-8')

            # Redirect stdout and stderr to both console and file
            sys.stdout = TeeOutput(original_stdout, log_file)
            sys.stderr = TeeOutput(original_stderr, log_file)

            # Initialize global logger
            logger = TrainingLogger(log_file=log_file)

            print(f"[TrainRunner] Training log will be saved to: {log_file_path}")
        else:
            print(f"[TrainRunner] Warning: training_folder not found in config, log file not created")
            # Initialize logger without file
            logger = TrainingLogger(log_file=None)
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to set up log file: {e}")
        # Initialize logger without file
        logger = TrainingLogger(log_file=None)

    # ============================================================
    # Unload Generate Pipeline to Free Memory
    # ============================================================
    # The main application loads models for inference (txt2img/img2img/inpaint).
    # Training uses a separate model instance, so we unload the generate pipeline
    # to free CPU/GPU memory (Z-Image 6B model = ~15 GB on CPU).
    print(f"[TrainRunner] Unloading generate pipeline to free memory...")
    try:
        from core.pipeline import pipeline_manager

        # Unload all generate pipelines
        if pipeline_manager.txt2img_pipeline is not None:
            print(f"[TrainRunner] Unloading txt2img pipeline...")
            del pipeline_manager.txt2img_pipeline
            pipeline_manager.txt2img_pipeline = None

        if pipeline_manager.img2img_pipeline is not None:
            print(f"[TrainRunner] Unloading img2img pipeline...")
            del pipeline_manager.img2img_pipeline
            pipeline_manager.img2img_pipeline = None

        if pipeline_manager.inpaint_pipeline is not None:
            print(f"[TrainRunner] Unloading inpaint pipeline...")
            del pipeline_manager.inpaint_pipeline
            pipeline_manager.inpaint_pipeline = None

        # Unload Z-Image components if present
        if pipeline_manager.zimage_components is not None:
            print(f"[TrainRunner] Unloading Z-Image components...")
            del pipeline_manager.zimage_components
            pipeline_manager.zimage_components = None

        # Reset current model tracking
        pipeline_manager.current_model = None
        pipeline_manager.current_model_info = None
        pipeline_manager.is_zimage_model = False

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[TrainRunner] Generate pipeline unloaded successfully")
    except Exception as e:
        print(f"[TrainRunner] Warning: Failed to unload generate pipeline: {e}")
        # Continue training even if unload fails

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
            caption_types = ds_config.get("caption_types", [])
            dataset_items = get_dataset_items(datasets_db, dataset_id, run_id=run_id, caption_types=caption_types)
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

        # ============================================================
        # Dataset Wrapper Class for New Interface
        # ============================================================
        class TrainRunnerDataset:
            """
            Dataset wrapper for train_runner.py to use new BaseTrainer.train() interface.

            This wrapper converts the old dataset_items format (list of dicts) to
            the new Dataset object format expected by BaseTrainer.train().
            """
            def __init__(self, unique_id: str, items: List[Dict], dataset_config: Dict):
                self.unique_id = unique_id
                self.items = items
                self.dataset_config = dataset_config
                self.cache_dir = Path(f"./latent_cache/{unique_id}")

                # Extract caption configuration from first item (all items share same config)
                if items:
                    self.caption_config = {
                        "normalize_tags": items[0].get("normalize_tags", True),
                        "shuffle_tokens": items[0].get("shuffle_tokens", True),
                        "category_order": items[0].get("category_order", []),
                    }
                else:
                    self.caption_config = {
                        "normalize_tags": True,
                        "shuffle_tokens": True,
                        "category_order": [],
                    }

            def reload_for_epoch(self, epoch_num: int, run_id: int) -> List[Dict]:
                """
                Reload dataset items with caption processing for the current epoch.

                This method is called by the trainer at the start of each epoch to
                get freshly processed captions (with shuffling, etc.).
                """
                dataset_id = self.dataset_config["dataset_id"]
                caption_types = self.dataset_config.get("caption_types", [])
                items = get_dataset_items(datasets_db, dataset_id, epoch_num=epoch_num, run_id=run_id, caption_types=caption_types)

                # Add dataset_unique_id for cache management
                for item in items:
                    item["dataset_unique_id"] = self.unique_id

                return items

        # ============================================================
        # Prepare Datasets for New Interface
        # ============================================================
        print(f"[TrainRunner] Preparing {len(dataset_configs)} dataset(s) for training...")

        # Convert dataset_items to Dataset objects, grouped by unique_id
        from collections import defaultdict
        items_by_dataset = defaultdict(lambda: {"items": [], "config": None})

        for item in dataset_items:
            unique_id = item.get("dataset_unique_id", "default")
            items_by_dataset[unique_id]["items"].append(item)

        # Match dataset configs to items
        for ds_config in dataset_configs:
            dataset_id = ds_config["dataset_id"]
            dataset = datasets_db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset and dataset.unique_id in items_by_dataset:
                items_by_dataset[dataset.unique_id]["config"] = ds_config

        # Create Dataset wrapper objects
        training_datasets = [
            TrainRunnerDataset(unique_id, data["items"], data["config"])
            for unique_id, data in items_by_dataset.items()
            if data["config"] is not None
        ]

        print(f"[TrainRunner] Created {len(training_datasets)} dataset wrapper(s)")
        for ds in training_datasets:
            print(f"  Dataset {ds.unique_id}: {len(ds.items)} items")

        # ============================================================
        # Determine Training Method
        # ============================================================
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

            # Get optimizer options and hyperparameters from train_config
            optimizer_is_paged = train_config.get('optimizer_is_paged', False)
            optimizer_cautious = train_config.get('optimizer_cautious', False)
            optimizer_beta1 = train_config.get('optimizer_beta1')
            optimizer_beta2 = train_config.get('optimizer_beta2')
            optimizer_epsilon = train_config.get('optimizer_epsilon')
            optimizer_weight_decay = train_config.get('optimizer_weight_decay')

            # Schedule-Free optimizer options (RingBuffer optimizers only)
            optimizer_schedule_free = train_config.get('optimizer_schedule_free', False)
            optimizer_warmup_steps = train_config.get('optimizer_warmup_steps', 0)
            optimizer_schedule_free_r = train_config.get('optimizer_schedule_free_r', 0.0)
            optimizer_schedule_free_weight_lr_power = train_config.get('optimizer_schedule_free_weight_lr_power', 2.0)
            optimizer_use_radam = train_config.get('optimizer_use_radam', False)

            # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

            # Training scope control
            train_text_encoder = train_config.get('train_text_encoder', False)

            # Initialize trainer
            trainer = LoRATrainer(
                model_path=run.base_model_path,
                output_dir=run.output_dir,
                run_name=run.run_name,  # Pass run_name for checkpoint filename generation
                run_id=run_id,  # Pass run_id for DB metrics logging
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
                # Optimizer options and hyperparameters
                optimizer_is_paged=optimizer_is_paged,
                optimizer_cautious=optimizer_cautious,
                optimizer_beta1=optimizer_beta1,
                optimizer_beta2=optimizer_beta2,
                optimizer_epsilon=optimizer_epsilon,
                optimizer_weight_decay=optimizer_weight_decay,
                # Schedule-Free optimizer options
                optimizer_schedule_free=optimizer_schedule_free,
                optimizer_warmup_steps=optimizer_warmup_steps,
                optimizer_schedule_free_r=optimizer_schedule_free_r,
                optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
                optimizer_use_radam=optimizer_use_radam,
                # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
                # Training scope control
                train_text_encoder=train_text_encoder,
            )

            # Note: setup_optimizer() is now called inside train() method
            # This avoids double initialization and provides clearer separation of concerns

            # Get optimizer settings (passed to train() method)
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # ============================================================
            # Validate Prediction Configuration (Unified Framework)
            # ============================================================
            from core.model_loader import ModelLoader

            # Detect model's prediction configuration
            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration detected:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")
            print(f"  Detection Source: {model_pred_config['source']}")

            # Get training configuration (with "auto" support)
            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')
            strict_validation = train_config.get('strict_validation', False)

            # Auto-detect: use model's configuration
            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
                print(f"[TrainRunner] noise_process='auto' → using model's config: {training_noise_process}")

            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']
                print(f"[TrainRunner] prediction_target='auto' → using model's config: {training_prediction_target}")

            # Validate compatibility
            mismatch_warnings = []
            if training_noise_process != model_pred_config['noise_process']:
                mismatch_warnings.append(
                    f"noise_process mismatch: model={model_pred_config['noise_process']}, training={training_noise_process}"
                )
            if training_prediction_target != model_pred_config['prediction_target']:
                mismatch_warnings.append(
                    f"prediction_target mismatch: model={model_pred_config['prediction_target']}, training={training_prediction_target}"
                )

            if mismatch_warnings:
                print(f"\n{'='*60}")
                print(f"[TrainRunner] ⚠️  PREDICTION CONFIG MISMATCH DETECTED")
                print(f"{'='*60}")
                for warning in mismatch_warnings:
                    print(f"  • {warning}")
                print(f"\nThis may cause training instability or poor convergence.")
                print(f"Model was trained with: {model_pred_config['noise_process']} + {model_pred_config['prediction_target']}")
                print(f"You are training with: {training_noise_process} + {training_prediction_target}")

                if strict_validation:
                    print(f"\n❌ strict_validation=True: Aborting training due to mismatch.")
                    print(f"{'='*60}\n")
                    sys.exit(1)
                else:
                    print(f"\n⚠️  strict_validation=False: Continuing with warning.")
                    print(f"Set strict_validation=true in training config to abort on mismatch.")
                    print(f"{'='*60}\n")
            else:
                print(f"[TrainRunner] ✓ Prediction configuration validated successfully")

            # Store final training config for trainer
            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # ============================================================
            # Setup Regularization Loss (SNR or Energy)
            # ============================================================
            regularization_type = train_config.get('regularization_type', None)
            if regularization_type:
                print(f"[TrainRunner] Initializing {regularization_type.upper()} regularization...")
                trainer.config = train_config  # Pass config for factory function

                if regularization_type.lower() == 'snr':
                    from core.training.losses.snr_regularization import create_snr_regularization_loss
                    trainer.snr_regularization_loss = create_snr_regularization_loss(train_config)
                    print(f"[TrainRunner] SNR Regularization enabled:")
                    print(f"  Weight: {train_config.get('snr_regularization_weight', 0.1)}")
                    print(f"  Timestep adaptive: {train_config.get('snr_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('snr_penalty_mode', 'relu')}")
                elif regularization_type.lower() == 'energy':
                    from core.training.losses.energy_regularization import create_energy_regularization_loss
                    trainer.snr_regularization_loss = create_energy_regularization_loss(train_config)
                    print(f"[TrainRunner] Energy Regularization enabled:")
                    print(f"  Weight: {train_config.get('energy_regularization_weight', 0.05)}")
                    print(f"  Timestep adaptive: {train_config.get('energy_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('energy_penalty_mode', 'abs')}")
                    print(f"  Normalize by pixels: {train_config.get('energy_normalize_by_pixels', True)}")
                else:
                    print(f"[TrainRunner] WARNING: Unknown regularization type '{regularization_type}', skipping")
            else:
                print(f"[TrainRunner] Regularization disabled (regularization_type not set)")

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
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None):
                # Get current learning rate from optimizer (if available)
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    # Debug: Log LR retrieval
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr)

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

            # Debug: Log sample generation settings
            print(f"[TrainRunner] Sample generation settings:")
            print(f"  sample_every: {process_config['sample'].get('sample_every', 100)}")
            print(f"  sample_prompts: {len(sample_prompts) if sample_prompts else 0} prompts")
            if sample_prompts:
                for i, prompt in enumerate(sample_prompts):
                    print(f"    Prompt {i}: positive={prompt.get('positive', '')[:50]}..., negative={prompt.get('negative', '')[:50]}...")
            print(f"  sample_config: {sample_config}")

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

            # Convert save_every parameters to new interface (save_every_n_steps)
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)

            if save_every_unit == 'epochs':
                # Calculate steps per epoch (approximate, will be recalculated by trainer)
                steps_per_epoch = (len(dataset_items) + train_config.get('batch_size', 1) - 1) // train_config.get('batch_size', 1)
                save_every_n_steps = save_every * steps_per_epoch
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")
            else:
                save_every_n_steps = save_every

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep}")

            # Convert sample_prompts to single sample_prompt
            sample_prompt = "a beautiful landscape"
            if sample_prompts and len(sample_prompts) > 0:
                sample_prompt = sample_prompts[0].get('positive', 'a beautiful landscape')

            # Get sample generation settings
            sample_guidance_scale = process_config['sample'].get('guidance_scale', 3.5)
            sample_steps = process_config['sample'].get('sample_steps', 28)
            sample_width = process_config['sample'].get('width', 1024)
            sample_height = process_config['sample'].get('height', 1024)
            sample_seed = process_config['sample'].get('seed', -1)
            print(f"[TrainRunner] Sample generation config: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, sample_steps={sample_steps}, seed={sample_seed}")

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Log force_recache setting
            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Get text encoding mode
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)

            # Get latent encoding mode
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Get Multi Noise-Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Start training with new interface
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,  # Pass total_steps from YAML
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=process_config['sample'].get('sample_every', 100),
                sample_prompt=sample_prompt,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
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

            # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
            prompt_chunking_mode = train_config.get('prompt_chunking_mode', 'a1111')
            max_prompt_chunks = train_config.get('max_prompt_chunks', 0)

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
                blocks_to_swap=train_config.get('blocks_to_swap', 0),
                use_pinned_memory=train_config.get('use_pinned_memory', False),
                num_optimizer_groups=train_config.get('num_optimizer_groups', 0),
                # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
                prompt_chunking_mode=prompt_chunking_mode,
                max_prompt_chunks=max_prompt_chunks,
            )

            # Note: setup_optimizer() is now called inside train() method
            # This avoids double initialization and provides clearer separation of concerns

            # Get optimizer settings (passed to train() method)
            optimizer_type = train_config.get('optimizer', 'adamw8bit')
            lr_scheduler_type = train_config.get('lr_scheduler', 'constant')

            # ============================================================
            # Validate Prediction Configuration (Unified Framework)
            # ============================================================
            from core.model_loader import ModelLoader

            # Detect model's prediction configuration
            model_type = ModelLoader.detect_model_type(run.base_model_path)
            model_pred_config = ModelLoader.detect_prediction_config(run.base_model_path, model_type)

            print(f"[TrainRunner] Model prediction configuration detected:")
            print(f"  Noise Process: {model_pred_config['noise_process']}")
            print(f"  Prediction Target: {model_pred_config['prediction_target']}")
            print(f"  Detection Source: {model_pred_config['source']}")

            # Get training configuration (with "auto" support)
            training_noise_process = train_config.get('noise_process', 'auto')
            training_prediction_target = train_config.get('prediction_target', 'auto')
            strict_validation = train_config.get('strict_validation', False)

            # Auto-detect: use model's configuration
            if training_noise_process == 'auto':
                training_noise_process = model_pred_config['noise_process']
                print(f"[TrainRunner] noise_process='auto' → using model's config: {training_noise_process}")

            if training_prediction_target == 'auto':
                training_prediction_target = model_pred_config['prediction_target']
                print(f"[TrainRunner] prediction_target='auto' → using model's config: {training_prediction_target}")

            # Validate compatibility
            mismatch_warnings = []
            if training_noise_process != model_pred_config['noise_process']:
                mismatch_warnings.append(
                    f"noise_process mismatch: model={model_pred_config['noise_process']}, training={training_noise_process}"
                )
            if training_prediction_target != model_pred_config['prediction_target']:
                mismatch_warnings.append(
                    f"prediction_target mismatch: model={model_pred_config['prediction_target']}, training={training_prediction_target}"
                )

            if mismatch_warnings:
                print(f"\n{'='*60}")
                print(f"[TrainRunner] ⚠️  PREDICTION CONFIG MISMATCH DETECTED")
                print(f"{'='*60}")
                for warning in mismatch_warnings:
                    print(f"  • {warning}")
                print(f"\nThis may cause training instability or poor convergence.")
                print(f"Model was trained with: {model_pred_config['noise_process']} + {model_pred_config['prediction_target']}")
                print(f"You are training with: {training_noise_process} + {training_prediction_target}")

                if strict_validation:
                    print(f"\n❌ strict_validation=True: Aborting training due to mismatch.")
                    print(f"{'='*60}\n")
                    sys.exit(1)
                else:
                    print(f"\n⚠️  strict_validation=False: Continuing with warning.")
                    print(f"Set strict_validation=true in training config to abort on mismatch.")
                    print(f"{'='*60}\n")
            else:
                print(f"[TrainRunner] ✓ Prediction configuration validated successfully")

            # Store final training config for trainer
            trainer.noise_process = training_noise_process
            trainer.prediction_target = training_prediction_target

            # ============================================================
            # Setup Regularization Loss (SNR or Energy)
            # ============================================================
            regularization_type = train_config.get('regularization_type', None)
            if regularization_type:
                print(f"[TrainRunner] Initializing {regularization_type.upper()} regularization...")
                trainer.config = train_config  # Pass config for factory function

                if regularization_type.lower() == 'snr':
                    from core.training.losses.snr_regularization import create_snr_regularization_loss
                    trainer.snr_regularization_loss = create_snr_regularization_loss(train_config)
                    print(f"[TrainRunner] SNR Regularization enabled:")
                    print(f"  Weight: {train_config.get('snr_regularization_weight', 0.1)}")
                    print(f"  Timestep adaptive: {train_config.get('snr_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('snr_penalty_mode', 'relu')}")
                elif regularization_type.lower() == 'energy':
                    from core.training.losses.energy_regularization import create_energy_regularization_loss
                    trainer.snr_regularization_loss = create_energy_regularization_loss(train_config)
                    print(f"[TrainRunner] Energy Regularization enabled:")
                    print(f"  Weight: {train_config.get('energy_regularization_weight', 0.05)}")
                    print(f"  Timestep adaptive: {train_config.get('energy_timestep_adaptive', True)}")
                    print(f"  Penalty mode: {train_config.get('energy_penalty_mode', 'abs')}")
                    print(f"  Normalize by pixels: {train_config.get('energy_normalize_by_pixels', True)}")
                else:
                    print(f"[TrainRunner] WARNING: Unknown regularization type '{regularization_type}', skipping")
            else:
                print(f"[TrainRunner] Regularization disabled (regularization_type not set)")

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
            def progress_callback(phase: str, step: int, total: int, epoch: int = 0, loss: float = None):
                # Get current learning rate from optimizer (if available)
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
                    # Debug: Log LR retrieval
                    if phase == "training" and step % 100 == 0:
                        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                        print(f"[ProgressCallback] Step {step}: LR={lr:.2e}, Loss={loss_str}")
                update_training_progress(training_db, run_id, phase, step, total, epoch, loss, lr)

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

            # Debug: Log sample generation settings
            print(f"[TrainRunner] Sample generation settings:")
            print(f"  sample_every: {process_config['sample'].get('sample_every', 100)}")
            print(f"  sample_prompts: {len(sample_prompts) if sample_prompts else 0} prompts")
            if sample_prompts:
                for i, prompt in enumerate(sample_prompts):
                    print(f"    Prompt {i}: positive={prompt.get('positive', '')[:50]}..., negative={prompt.get('negative', '')[:50]}...")
            print(f"  sample_config: {sample_config}")

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

            # Convert save_every parameters to new interface (save_every_n_steps)
            save_every_unit = process_config['save'].get('save_every_unit', 'steps')
            save_every = process_config['save'].get('save_every', 100)
            max_step_saves_to_keep = process_config['save'].get('max_step_saves_to_keep', 3)

            if save_every_unit == 'epochs':
                # Calculate steps per epoch (approximate, will be recalculated by trainer)
                steps_per_epoch = (len(dataset_items) + train_config.get('batch_size', 1) - 1) // train_config.get('batch_size', 1)
                save_every_n_steps = save_every * steps_per_epoch
                print(f"[TrainRunner] Converted save_every={save_every} epochs to save_every_n_steps={save_every_n_steps}")
            else:
                save_every_n_steps = save_every

            print(f"[TrainRunner] Max step saves to keep: {max_step_saves_to_keep}")

            # Convert sample_prompts to single sample_prompt
            sample_prompt = "a beautiful landscape"
            if sample_prompts and len(sample_prompts) > 0:
                sample_prompt = sample_prompts[0].get('positive', 'a beautiful landscape')

            # Get sample generation settings
            sample_guidance_scale = process_config['sample'].get('guidance_scale', 3.5)
            sample_steps = process_config['sample'].get('sample_steps', 28)
            sample_width = process_config['sample'].get('width', 1024)
            sample_height = process_config['sample'].get('height', 1024)
            sample_seed = process_config['sample'].get('seed', -1)
            print(f"[TrainRunner] Sample generation config: width={sample_width}, height={sample_height}, guidance_scale={sample_guidance_scale}, sample_steps={sample_steps}, seed={sample_seed}")

            # Get resume from checkpoint setting
            resume_from_checkpoint = train_config.get('resume_from_checkpoint')
            if resume_from_checkpoint:
                print(f"[TrainRunner] Resume from checkpoint: {resume_from_checkpoint}")

            # Log force_recache setting
            if force_recache:
                print(f"[TrainRunner] Force recache enabled: all latent caches will be regenerated")

            # Get text encoding mode
            text_encoding_mode = train_config.get('text_encoding_mode', 'swap_onthefly')
            text_encoding_swap_interval = train_config.get('text_encoding_swap_interval', 256)

            # Get latent encoding mode
            latent_encoding_mode = train_config.get('latent_encoding_mode', 'swap_onthefly')
            latent_encoding_swap_interval = train_config.get('latent_encoding_swap_interval', 256)

            # Get Multi Noise-Timestep (MNT) settings
            multi_noise_timesteps = train_config.get('multi_noise_timesteps', 1)
            multi_noise_mode = train_config.get('multi_noise_mode', 'independent')
            trajectory_blend_alpha = train_config.get('trajectory_blend_alpha', 0.7)
            timestep_sampling_config = train_config.get('timestep_sampling', None)

            # Start training with new interface
            trainer.train(
                datasets=training_datasets,
                num_epochs=num_epochs if num_epochs else 1,
                total_steps=total_steps_config,  # Pass total_steps from YAML
                batch_size=train_config.get('batch_size', 1),
                save_every_n_steps=save_every_n_steps,
                sample_every_n_steps=process_config['sample'].get('sample_every', 100),
                sample_prompt=sample_prompt,
                sample_guidance_scale=sample_guidance_scale,
                sample_steps=sample_steps,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_seed=sample_seed,
                optimizer_type=optimizer_type,
                lr_scheduler_type=lr_scheduler_type,
                enable_bucketing=enable_bucketing,
                base_resolutions=base_resolutions,
                bucket_strategy="resize",
                multi_resolution_mode="max",
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                debug_latents=debug_latents,
                debug_latents_every=debug_latents_every,
                progress_callback=progress_callback,
                update_total_steps_callback=update_total_steps_callback,
                run_id=run_id,
                resume_from_checkpoint=resume_from_checkpoint,
                force_recache=force_recache,
                max_step_saves_to_keep=max_step_saves_to_keep,
                text_encoding_mode=text_encoding_mode,
                text_encoding_swap_interval=text_encoding_swap_interval,
                latent_encoding_mode=latent_encoding_mode,
                latent_encoding_swap_interval=latent_encoding_swap_interval,
                multi_noise_timesteps=multi_noise_timesteps,
                multi_noise_mode=multi_noise_mode,
                trajectory_blend_alpha=trajectory_blend_alpha,
                timestep_sampling_config=timestep_sampling_config,
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

        # Close log file and restore original stdout/stderr
        if log_file:
            print(f"[TrainRunner] Closing training log file...")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()


if __name__ == "__main__":
    main()
