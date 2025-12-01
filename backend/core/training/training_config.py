"""
Training configuration generator for ai-toolkit.

Generates YAML configuration files based on training parameters.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


class TrainingConfigGenerator:
    """Generate ai-toolkit YAML config from training parameters."""

    @staticmethod
    def generate_lora_config(
        run_name: str,
        dataset_path: str,
        base_model_path: str,
        output_dir: str,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "constant",
        optimizer: str = "adamw8bit",
        lora_rank: int = 16,
        lora_alpha: int = 16,
        save_every: int = 100,
        save_every_unit: str = "steps",
        sample_every: int = 100,
        sample_prompts: Optional[list] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        enable_bucketing: bool = False,
        base_resolutions: Optional[list] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        train_unet: bool = True,
        train_text_encoder: bool = False,
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        cache_latents_to_disk: bool = False,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_steps: int = 28,
        sample_cfg_scale: float = 7.0,
        sample_sampler: str = "euler",
        sample_seed: int = 42,
        # Caption processing settings
        caption_processing: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate LoRA training configuration YAML.

        Args:
            run_name: Training run identifier
            dataset_path: Path to dataset directory
            base_model_path: Path to base model
            output_dir: Output directory for checkpoints
            total_steps: Total training steps (mutually exclusive with epochs)
            epochs: Number of epochs (mutually exclusive with total_steps)
            batch_size: Batch size
            learning_rate: Learning rate
            lr_scheduler: Learning rate scheduler type
            optimizer: Optimizer type
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            save_every: Save checkpoint every N steps/epochs
            save_every_unit: Unit for save_every ("steps" or "epochs")
            sample_every: Generate sample every N steps/epochs
            sample_prompts: List of prompts for sample generation
            debug_latents: Enable debug mode to save latents
            debug_latents_every: Save debug latents every N steps
            enable_bucketing: Enable aspect ratio bucketing
            base_resolutions: List of base resolutions for bucketing (e.g., [512, 768, 1024])
            bucket_strategy: Bucketing strategy ("resize", "crop", "random_crop")
            multi_resolution_mode: Multi-resolution mode ("max" or "random")
            train_unet: Whether to train U-Net
            train_text_encoder: Whether to train text encoder
            unet_lr: U-Net learning rate (defaults to learning_rate if None)
            text_encoder_lr: Text encoder learning rate (defaults to learning_rate if None)
            text_encoder_1_lr: Text encoder 1 learning rate for SDXL (defaults to text_encoder_lr if None)
            text_encoder_2_lr: Text encoder 2 learning rate for SDXL (defaults to text_encoder_lr if None)
            cache_latents_to_disk: Whether to cache latents to disk (reduces VRAM usage during training)
            use_flash_attention: Enable Flash Attention for training (faster, lower memory)
            min_snr_gamma: Min-SNR gamma value for loss weighting (default: 5.0, set to 0 to disable)

        Returns:
            YAML configuration string
        """
        # Validate that either steps or epochs is provided
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")
        config = {
            "job": run_name,
            "config": {
                # Model settings
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "trigger_word": "",  # Can be customized
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_alpha,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": save_every,
                            "save_every_unit": save_every_unit,
                            "max_step_saves_to_keep": 10,
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_path,
                                "caption_ext": "txt",
                                # Legacy caption settings (kept for backward compatibility)
                                "caption_dropout_rate": caption_processing.get("caption_dropout_rate", 0.0) if caption_processing else 0.0,
                                "shuffle_tokens": caption_processing.get("shuffle_tokens", False) if caption_processing else False,
                                # Caption processing settings (SushiUI extended)
                                "token_dropout_rate": caption_processing.get("token_dropout_rate", 0.0) if caption_processing else 0.0,
                                "keep_tokens": caption_processing.get("keep_tokens", 0) if caption_processing else 0,
                                "shuffle_per_epoch": caption_processing.get("shuffle_per_epoch", False) if caption_processing else False,
                                "shuffle_keep_first_n": caption_processing.get("shuffle_keep_first_n", 0) if caption_processing else 0,
                                "tag_dropout_rate": caption_processing.get("tag_dropout_rate", 0.0) if caption_processing else 0.0,
                                "tag_dropout_per_epoch": caption_processing.get("tag_dropout_per_epoch", False) if caption_processing else False,
                                "tag_dropout_keep_first_n": caption_processing.get("tag_dropout_keep_first_n", 0) if caption_processing else 0,
                                "tag_dropout_exclude_person_count": caption_processing.get("tag_dropout_exclude_person_count", False) if caption_processing else False,
                                # Other settings
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "resolution": base_resolutions or [512, 768, 1024],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            **({"steps": total_steps} if total_steps else {"epochs": epochs}),
                            "gradient_accumulation_steps": 1,
                            "train_unet": train_unet,
                            "train_text_encoder": train_text_encoder,
                            "gradient_checkpointing": True,
                            "noise_scheduler": "ddpm",  # ddpm for epsilon prediction (SDXL standard)
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "unet_lr": unet_lr if unet_lr is not None else learning_rate,
                            "text_encoder_lr": text_encoder_lr if text_encoder_lr is not None else learning_rate,
                            "text_encoder_1_lr": text_encoder_1_lr if text_encoder_1_lr is not None else (text_encoder_lr if text_encoder_lr is not None else learning_rate),
                            "text_encoder_2_lr": text_encoder_2_lr if text_encoder_2_lr is not None else (text_encoder_lr if text_encoder_lr is not None else learning_rate),
                            "lr_scheduler": lr_scheduler,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": training_dtype,  # Training/activation dtype
                            "weight_dtype": weight_dtype,  # Model weight dtype
                            "output_dtype": output_dtype,  # Output latent dtype
                            "mixed_precision": mixed_precision,  # Enable autocast for mixed precision
                            "debug_latents": debug_latents,
                            "debug_latents_every": debug_latents_every,
                            "enable_bucketing": enable_bucketing,
                            "base_resolutions": base_resolutions or [1024],
                            "bucket_strategy": bucket_strategy,
                            "multi_resolution_mode": multi_resolution_mode,
                            "use_flash_attention": use_flash_attention,
                            "min_snr_gamma": min_snr_gamma,
                        },
                        "model": {
                            "name_or_path": base_model_path,
                            "is_flux": False,
                            "quantize": False,
                            "vae_dtype": vae_dtype,  # VAE-specific dtype
                        },
                        "sample": {
                            "sampler": sample_sampler,
                            "sample_every": sample_every,
                            "width": sample_width,
                            "height": sample_height,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": sample_seed,
                            "walk_seed": True,
                            "guidance_scale": sample_cfg_scale,
                            "sample_steps": sample_steps,
                        },
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_full_finetune_config(
        run_name: str,
        dataset_path: str,
        base_model_path: str,
        output_dir: str,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        lr_scheduler: str = "constant",
        optimizer: str = "adamw8bit",
        save_every: int = 100,
        save_every_unit: str = "steps",
        sample_every: int = 100,
        sample_prompts: Optional[list] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        enable_bucketing: bool = False,
        base_resolutions: Optional[List[int]] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        train_unet: bool = True,
        train_text_encoder: bool = True,
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        cache_latents_to_disk: bool = False,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_steps: int = 28,
        sample_cfg_scale: float = 7.0,
        sample_sampler: str = "euler",
        sample_seed: int = -1,
        caption_processing: Optional[dict] = None,
    ) -> str:
        """
        Generate full fine-tuning configuration YAML.

        Args:
            run_name: Training run identifier
            dataset_path: Path to dataset directory
            base_model_path: Path to base model
            output_dir: Output directory for checkpoints
            total_steps: Total training steps (mutually exclusive with epochs)
            epochs: Number of epochs (mutually exclusive with total_steps)
            batch_size: Batch size
            learning_rate: Learning rate (typically lower for full fine-tune)
            lr_scheduler: Learning rate scheduler type
            optimizer: Optimizer type
            save_every: Save checkpoint every N steps/epochs
            sample_every: Generate sample every N steps/epochs
            sample_prompts: List of prompts for sample generation

        Returns:
            YAML configuration string
        """
        # Validate that either steps or epochs is provided
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")

        # Build train config
        train_config = {
            "batch_size": batch_size,
            **({"steps": total_steps} if total_steps else {"epochs": epochs}),
            "gradient_accumulation_steps": 1,
            "train_unet": train_unet,
            "train_text_encoder": train_text_encoder,
            "gradient_checkpointing": True,
            "noise_scheduler": "ddpm",
            "optimizer": optimizer,
            "lr": learning_rate,
            "lr_scheduler": lr_scheduler,
            "weight_dtype": weight_dtype,
            "dtype": training_dtype,  # Training/activation dtype
            "output_dtype": output_dtype,
            "mixed_precision": mixed_precision,
            "use_flash_attention": use_flash_attention,
            "min_snr_gamma": min_snr_gamma,
            "debug_latents": debug_latents,
            "debug_latents_every": debug_latents_every,
        }

        # Add component-specific learning rates if specified
        if unet_lr is not None:
            train_config["unet_lr"] = unet_lr
        if text_encoder_lr is not None:
            train_config["text_encoder_lr"] = text_encoder_lr
        if text_encoder_1_lr is not None:
            train_config["text_encoder_1_lr"] = text_encoder_1_lr
        if text_encoder_2_lr is not None:
            train_config["text_encoder_2_lr"] = text_encoder_2_lr

        # Add bucketing parameters
        if enable_bucketing:
            train_config["enable_bucketing"] = True
            train_config["base_resolutions"] = base_resolutions or [1024]
            train_config["bucket_strategy"] = bucket_strategy
            train_config["multi_resolution_mode"] = multi_resolution_mode

        # Build dataset config
        dataset_config = {
            "folder_path": dataset_path,
            "caption_ext": "txt",
            "cache_latents_to_disk": cache_latents_to_disk,
        }

        # Add caption processing config if provided
        if caption_processing:
            dataset_config["caption_processing"] = caption_processing

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "trigger_word": "",
                        "network": {
                            "type": "full_finetune",
                        },
                        "save": {
                            "dtype": output_dtype,
                            "save_every": save_every,
                            "save_every_unit": save_every_unit,
                            "max_step_saves_to_keep": 3,  # Fewer saves for full models (larger size)
                        },
                        "datasets": [dataset_config],
                        "train": train_config,
                        "model": {
                            "name_or_path": base_model_path,
                            "is_flux": False,
                            "quantize": False,
                            "vae_dtype": vae_dtype,
                        },
                        "sample": {
                            "sampler": sample_sampler,
                            "sample_every": sample_every,
                            "width": sample_width,
                            "height": sample_height,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": sample_seed,
                            "walk_seed": True,
                            "guidance_scale": sample_cfg_scale,
                            "sample_steps": sample_steps,
                            "schedule_type": "sgm_uniform",
                        },
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def save_config(config_yaml: str, output_path: str) -> None:
        """
        Save YAML configuration to file.

        Args:
            config_yaml: YAML configuration string
            output_path: Path to save the config file
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path_obj, "w", encoding="utf-8") as f:
            f.write(config_yaml)
