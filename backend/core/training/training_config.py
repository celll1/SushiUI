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
        dataset_path: str,  # Deprecated - kept for backward compatibility
        base_model_path: str,
        output_dir: str,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,  # New: multiple datasets
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        optimizer: str = "adamw8bit",
        optimizer_is_paged: bool = False,
        optimizer_cautious: bool = False,
        optimizer_beta1: Optional[float] = None,
        optimizer_beta2: Optional[float] = None,
        optimizer_epsilon: Optional[float] = None,
        optimizer_weight_decay: Optional[float] = None,
        optimizer_schedule_free: bool = False,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
        optimizer_use_radam: bool = False,
        optimizer_stochastic_rounding: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dtype: str = "fp32",
        save_every: int = 100,
        save_every_unit: str = "steps",
        max_step_saves_to_keep: int = 10,
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
        train_image_encoder: bool = False,  # Image Encoder (future support)
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        image_encoder_lr: Optional[float] = None,  # Image Encoder LR (future support)
        cache_latents_to_disk: bool = False,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        reconstruction_loss_weight: float = 0.0,
        # Block Swap settings (training VRAM optimization)
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = False,
        num_optimizer_groups: int = 0,
        # Text encoding settings
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        # Latent encoding settings
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_steps: int = 28,
        sample_cfg_scale: float = 7.0,
        sample_sampler: str = "euler",
        sample_schedule_type: str = "sgm_uniform",
        sample_seed: int = 42,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",  # "a1111", "sd_scripts", "nobos"
        max_prompt_chunks: int = 0,  # 0 = unlimited
        # Resume settings
        resume_from_checkpoint: Optional[str] = None,
        # Caption processing settings
        caption_processing: Optional[Dict[str, Any]] = None,
        # Multi Noise-Timestep (MNT) settings
        multi_noise_timesteps: int = 1,
        multi_noise_mode: str = "independent",
        trajectory_blend_alpha: float = 0.7,
        timestep_sampling_config: Optional[Dict[str, Any]] = None,
        # Regularization settings (prevent overbaking)
        regularization_type: Optional[str] = None,  # "snr", "energy", or None
        snr_regularization_weight: float = 0.1,
        snr_timestep_adaptive: bool = True,
        snr_penalty_mode: str = "relu",
        energy_regularization_weight: float = 0.05,
        energy_timestep_adaptive: bool = True,
        energy_penalty_mode: str = "abs",
        energy_normalize_by_pixels: bool = True,
        # Unified training framework settings
        noise_process: str = "auto",  # "auto", "ddpm", "flow"
        prediction_target: str = "auto",  # "auto", "epsilon", "velocity", "sample"
        strict_validation: bool = False,  # If True, error on mismatch; if False, warn only
        # Reference image settings
        use_reference_images: bool = False,  # Enable reference image conditioning during training
    ) -> str:
        """
        Generate LoRA training configuration YAML.

        Args:
            run_name: Training run identifier
            dataset_path: Path to dataset directory (deprecated, use dataset_configs)
            base_model_path: Path to base model
            output_dir: Output directory for checkpoints
            dataset_configs: List of dataset configurations with path and caption_processing
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
            text_encoding_mode: Text encoding mode ("swap_onthefly", "pre_encoded_cache", "onthefly_gpu")
            text_encoding_swap_interval: Swap interval for swap_onthefly mode (default: 256)
            latent_encoding_mode: Latent encoding mode ("swap_onthefly", "pre_encoded_cache", "onthefly_gpu")
            latent_encoding_swap_interval: Swap interval for swap_onthefly mode (default: 256)

        Returns:
            YAML configuration string
        """
        # Validate that either steps or epochs is provided
        if total_steps is None and epochs is None:
            raise ValueError("Either total_steps or epochs must be provided")
        if total_steps is not None and epochs is not None:
            raise ValueError("Cannot specify both total_steps and epochs")

        # Build datasets array
        # NOTE: caption_processing settings are NOT saved to YAML
        # They are read from the database (Dataset.caption_processing) at training time
        # This ensures Dataset Management page settings are always used
        datasets_array = []
        if dataset_configs:
            # Use multiple datasets
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_caption_types = ds_config.get("caption_types", [])
                ds_dataset_id = ds_config.get("dataset_id")  # Include dataset_id for YAML editing support

                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    # Dataset ID for resolving dataset from YAML edits (required for train_runner.py)
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                    # Other settings
                    "cache_latents_to_disk": cache_latents_to_disk,
                    "resolution": base_resolutions or [512, 768, 1024],
                }

                # Add caption_types if specified
                if ds_caption_types:
                    dataset_entry["caption_types"] = ds_caption_types

                datasets_array.append(dataset_entry)
        else:
            # Fallback: use single dataset_path (backward compatibility)
            # NOTE: caption_processing is NOT saved - read from database at training time
            dataset_entry = {
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
                "resolution": base_resolutions or [512, 768, 1024],
            }
            datasets_array.append(dataset_entry)

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
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_alpha,
                            "lora_dtype": lora_dtype,
                        },
                        "dtype": {
                            "weight": weight_dtype,        # Model weight dtype (推奨: fp32, 許容: bf16, 非推奨: fp16/fp8)
                            "training": training_dtype,    # Training/activation dtype (autocast)
                            "vae": vae_dtype,              # VAE dtype (推奨: fp32, 許容: fp16 for SDXL madebyollin, 非推奨: bf16/fp8, Z-Image必須: fp32)
                            "save": output_dtype,          # Save dtype (fp32/fp16/bf16)
                        },
                        "save": {
                            "save_every": save_every,
                            "save_every_unit": save_every_unit,
                            "max_step_saves_to_keep": max_step_saves_to_keep,
                        },
                        "datasets": datasets_array,
                        "train": {
                            "batch_size": batch_size,
                            **({"steps": total_steps} if total_steps else {"epochs": epochs}),
                            "gradient_accumulation_steps": 1,
                            "train_unet": train_unet,
                            "train_text_encoder": train_text_encoder,
                            "train_image_encoder": train_image_encoder,
                            # Note: Gradient checkpointing is always enabled (hardcoded in BaseTrainer for VRAM efficiency)
                            # Unified training framework (replaces noise_scheduler)
                            "noise_process": noise_process,  # "auto", "ddpm", "flow"
                            "prediction_target": prediction_target,  # "auto", "epsilon", "velocity", "sample"
                            "strict_validation": strict_validation,
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "lr_scheduler": lr_scheduler,
                            **({"lr_warmup_steps": lr_warmup_steps} if lr_warmup_steps > 0 else {}),
                            **({"optimizer_is_paged": optimizer_is_paged} if optimizer_is_paged else {}),
                            **({"optimizer_cautious": optimizer_cautious} if optimizer_cautious else {}),
                            **({"optimizer_beta1": optimizer_beta1} if optimizer_beta1 is not None else {}),
                            **({"optimizer_beta2": optimizer_beta2} if optimizer_beta2 is not None else {}),
                            **({"optimizer_epsilon": optimizer_epsilon} if optimizer_epsilon is not None else {}),
                            **({"optimizer_weight_decay": optimizer_weight_decay} if optimizer_weight_decay is not None else {}),
                            **({"optimizer_schedule_free": optimizer_schedule_free} if optimizer_schedule_free else {}),
                            **({"optimizer_schedule_free_r": optimizer_schedule_free_r} if optimizer_schedule_free and optimizer_schedule_free_r != 0.0 else {}),
                            **({"optimizer_schedule_free_weight_lr_power": optimizer_schedule_free_weight_lr_power} if optimizer_schedule_free and optimizer_schedule_free_weight_lr_power != 2.0 else {}),
                            **({"optimizer_use_radam": optimizer_use_radam} if optimizer_schedule_free and optimizer_use_radam else {}),
                            **({"optimizer_stochastic_rounding": optimizer_stochastic_rounding} if optimizer_stochastic_rounding else {}),
                            "unet_lr": unet_lr if unet_lr is not None else learning_rate,
                            "text_encoder_lr": text_encoder_lr if text_encoder_lr is not None else learning_rate,
                            "text_encoder_1_lr": text_encoder_1_lr if text_encoder_1_lr is not None else (text_encoder_lr if text_encoder_lr is not None else learning_rate),
                            "text_encoder_2_lr": text_encoder_2_lr if text_encoder_2_lr is not None else (text_encoder_lr if text_encoder_lr is not None else learning_rate),
                            "image_encoder_lr": image_encoder_lr if image_encoder_lr is not None else learning_rate,
                            "mixed_precision": mixed_precision,  # Enable autocast for mixed precision
                            "debug_latents": debug_latents,
                            "debug_latents_every": debug_latents_every,
                            "enable_bucketing": enable_bucketing,
                            "base_resolutions": base_resolutions or [1024],
                            "bucket_strategy": bucket_strategy,
                            "multi_resolution_mode": multi_resolution_mode,
                            "use_flash_attention": use_flash_attention,
                            "min_snr_gamma": min_snr_gamma,
                            "reconstruction_loss_weight": reconstruction_loss_weight,
                            "blocks_to_swap": blocks_to_swap,
                            "use_pinned_memory": use_pinned_memory,
                            "num_optimizer_groups": num_optimizer_groups,
                            "text_encoding_mode": text_encoding_mode,
                            "text_encoding_swap_interval": text_encoding_swap_interval,
                            # Reference image settings (FLUX.2 only - uses latent concatenation for conditioning)
                            "use_reference_images": use_reference_images,
                            "latent_encoding_mode": latent_encoding_mode,
                            "latent_encoding_swap_interval": latent_encoding_swap_interval,
                            "multi_noise_timesteps": multi_noise_timesteps,
                            "multi_noise_mode": multi_noise_mode,
                            "trajectory_blend_alpha": trajectory_blend_alpha,
                            **({"timestep_sampling": timestep_sampling_config} if timestep_sampling_config else {}),
                            "resume_from_checkpoint": resume_from_checkpoint,  # Always output (None, "latest", or checkpoint filename)
                            # Regularization settings
                            **({"regularization_type": regularization_type} if regularization_type else {}),
                            "snr_regularization_weight": snr_regularization_weight,
                            "snr_timestep_adaptive": snr_timestep_adaptive,
                            "snr_penalty_mode": snr_penalty_mode,
                            "energy_regularization_weight": energy_regularization_weight,
                            "energy_timestep_adaptive": energy_timestep_adaptive,
                            "energy_penalty_mode": energy_penalty_mode,
                            "energy_normalize_by_pixels": energy_normalize_by_pixels,
                        },
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": sample_sampler,
                            "schedule_type": sample_schedule_type,
                            "sample_every": sample_every,
                            "width": sample_width,
                            "height": sample_height,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": sample_seed,
                            "guidance_scale": sample_cfg_scale,
                            "sample_steps": sample_steps,
                        },
                        # Prompt chunking settings (for long prompts >75 tokens)
                        "prompt_chunking_mode": prompt_chunking_mode,
                        "max_prompt_chunks": max_prompt_chunks,
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_full_finetune_config(
        run_name: str,
        dataset_path: str,  # Deprecated - kept for backward compatibility
        base_model_path: str,
        output_dir: str,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,  # New: multiple datasets
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        optimizer: str = "adamw8bit",
        optimizer_is_paged: bool = False,
        optimizer_cautious: bool = False,
        optimizer_beta1: Optional[float] = None,
        optimizer_beta2: Optional[float] = None,
        optimizer_epsilon: Optional[float] = None,
        optimizer_weight_decay: Optional[float] = None,
        optimizer_schedule_free: bool = False,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
        optimizer_use_radam: bool = False,
        optimizer_stochastic_rounding: bool = False,
        save_every: int = 100,
        save_every_unit: str = "steps",
        max_step_saves_to_keep: int = 3,  # Fewer for full models (larger checkpoint size)
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
        train_image_encoder: bool = False,  # Image Encoder (future support)
        unet_lr: Optional[float] = None,
        text_encoder_lr: Optional[float] = None,
        text_encoder_1_lr: Optional[float] = None,
        text_encoder_2_lr: Optional[float] = None,
        image_encoder_lr: Optional[float] = None,  # Image Encoder LR (future support)
        cache_latents_to_disk: bool = False,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        reconstruction_loss_weight: float = 0.0,
        # Block Swap settings (training VRAM optimization)
        blocks_to_swap: int = 0,
        use_pinned_memory: bool = False,
        num_optimizer_groups: int = 0,
        # Text encoding settings
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        # Latent encoding settings
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
        sample_width: int = 1024,
        sample_height: int = 1024,
        sample_steps: int = 28,
        sample_cfg_scale: float = 7.0,
        sample_sampler: str = "euler",
        sample_schedule_type: str = "sgm_uniform",
        sample_seed: int = -1,
        # Prompt chunking settings (SD/SDXL only, for long prompts >75 tokens)
        prompt_chunking_mode: str = "a1111",  # "a1111", "sd_scripts", "nobos"
        max_prompt_chunks: int = 0,  # 0 = unlimited
        resume_from_checkpoint: Optional[str] = None,
        caption_processing: Optional[dict] = None,
        # Multi Noise-Timestep (MNT) settings
        multi_noise_timesteps: int = 1,
        multi_noise_mode: str = "independent",
        trajectory_blend_alpha: float = 0.7,
        timestep_sampling_config: Optional[Dict[str, Any]] = None,
        # Regularization settings (prevent overbaking)
        regularization_type: Optional[str] = None,  # "snr", "energy", or None
        snr_regularization_weight: float = 0.1,
        snr_timestep_adaptive: bool = True,
        snr_penalty_mode: str = "relu",
        energy_regularization_weight: float = 0.05,
        energy_timestep_adaptive: bool = True,
        energy_penalty_mode: str = "abs",
        energy_normalize_by_pixels: bool = True,
        # Unified training framework settings
        noise_process: str = "add_noise",
        prediction_target: str = "auto",
        strict_validation: bool = True,
        # Reference image settings
        use_reference_images: bool = False,  # Enable reference image conditioning during training
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
            "train_image_encoder": train_image_encoder,
            # Note: Gradient checkpointing is always enabled (hardcoded in BaseTrainer for VRAM efficiency)
            "optimizer": optimizer,
            "lr": learning_rate,
            "lr_scheduler": lr_scheduler,
            **({"lr_warmup_steps": lr_warmup_steps} if lr_warmup_steps > 0 else {}),
            **({"optimizer_is_paged": optimizer_is_paged} if optimizer_is_paged else {}),
            **({"optimizer_cautious": optimizer_cautious} if optimizer_cautious else {}),
            **({"optimizer_beta1": optimizer_beta1} if optimizer_beta1 is not None else {}),
            **({"optimizer_beta2": optimizer_beta2} if optimizer_beta2 is not None else {}),
            **({"optimizer_epsilon": optimizer_epsilon} if optimizer_epsilon is not None else {}),
            **({"optimizer_weight_decay": optimizer_weight_decay} if optimizer_weight_decay is not None else {}),
            **({"optimizer_schedule_free": optimizer_schedule_free} if optimizer_schedule_free else {}),
            **({"optimizer_schedule_free_r": optimizer_schedule_free_r} if optimizer_schedule_free and optimizer_schedule_free_r != 0.0 else {}),
            **({"optimizer_schedule_free_weight_lr_power": optimizer_schedule_free_weight_lr_power} if optimizer_schedule_free and optimizer_schedule_free_weight_lr_power != 2.0 else {}),
            **({"optimizer_use_radam": optimizer_use_radam} if optimizer_schedule_free and optimizer_use_radam else {}),
            **({"optimizer_stochastic_rounding": optimizer_stochastic_rounding} if optimizer_stochastic_rounding else {}),
            "mixed_precision": mixed_precision,
            "use_flash_attention": use_flash_attention,
            "min_snr_gamma": min_snr_gamma,
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "blocks_to_swap": blocks_to_swap,
            "use_pinned_memory": use_pinned_memory,
            "num_optimizer_groups": num_optimizer_groups,
            "text_encoding_mode": text_encoding_mode,
            "text_encoding_swap_interval": text_encoding_swap_interval,
            # Reference image settings (FLUX.2 only - uses latent concatenation for conditioning)
            "use_reference_images": use_reference_images,
            "latent_encoding_mode": latent_encoding_mode,
            "latent_encoding_swap_interval": latent_encoding_swap_interval,
            "debug_latents": debug_latents,
            "debug_latents_every": debug_latents_every,
            "multi_noise_timesteps": multi_noise_timesteps,
            "multi_noise_mode": multi_noise_mode,
            "trajectory_blend_alpha": trajectory_blend_alpha,
            **({"timestep_sampling": timestep_sampling_config} if timestep_sampling_config else {}),
            "resume_from_checkpoint": resume_from_checkpoint,  # Always output (None, "latest", or checkpoint filename)
            # Regularization settings
            **({"regularization_type": regularization_type} if regularization_type else {}),
            "snr_regularization_weight": snr_regularization_weight,
            "snr_timestep_adaptive": snr_timestep_adaptive,
            "snr_penalty_mode": snr_penalty_mode,
            "energy_regularization_weight": energy_regularization_weight,
            "energy_timestep_adaptive": energy_timestep_adaptive,
            "energy_penalty_mode": energy_penalty_mode,
            "energy_normalize_by_pixels": energy_normalize_by_pixels,
            # Unified training framework settings
            "noise_process": noise_process,
            "prediction_target": prediction_target,
            "strict_validation": strict_validation,
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
        if image_encoder_lr is not None:
            train_config["image_encoder_lr"] = image_encoder_lr

        # Add bucketing parameters
        if enable_bucketing:
            train_config["enable_bucketing"] = True
            train_config["base_resolutions"] = base_resolutions or [1024]
            train_config["bucket_strategy"] = bucket_strategy
            train_config["multi_resolution_mode"] = multi_resolution_mode

        # Build datasets array
        # NOTE: caption_processing settings are NOT saved to YAML
        # They are read from the database (Dataset.caption_processing) at training time
        # This ensures Dataset Management page settings are always used
        datasets_array = []
        if dataset_configs:
            # Use multiple datasets
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_caption_types = ds_config.get("caption_types", [])
                ds_dataset_id = ds_config.get("dataset_id")  # Include dataset_id for YAML editing support

                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    # Dataset ID for resolving dataset from YAML edits (required for train_runner.py)
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }

                # Add caption_types if specified
                if ds_caption_types:
                    dataset_entry["caption_types"] = ds_caption_types

                datasets_array.append(dataset_entry)
        else:
            # Fallback: use single dataset_path (backward compatibility)
            # NOTE: caption_processing is NOT saved - read from database at training time
            dataset_config = {
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
            }

            datasets_array.append(dataset_config)

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "full_finetune",
                        },
                        "dtype": {
                            "weight": weight_dtype,        # Model weight dtype (推奨: fp32, 許容: bf16, 非推奨: fp16/fp8)
                            "training": training_dtype,    # Training/activation dtype (autocast)
                            "vae": vae_dtype,              # VAE dtype (推奨: fp32, 許容: fp16 for SDXL madebyollin, 非推奨: bf16/fp8, Z-Image必須: fp32)
                            "save": output_dtype,          # Save dtype (fp32/fp16/bf16)
                        },
                        "save": {
                            "save_every": save_every,
                            "save_every_unit": save_every_unit,
                            "max_step_saves_to_keep": max_step_saves_to_keep,
                        },
                        "datasets": datasets_array,
                        "train": train_config,
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": sample_sampler,
                            "schedule_type": sample_schedule_type,
                            "sample_every": sample_every,
                            "width": sample_width,
                            "height": sample_height,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": sample_seed,
                            "guidance_scale": sample_cfg_scale,
                            "sample_steps": sample_steps,
                        },
                        # Prompt chunking settings (for long prompts >75 tokens)
                        "prompt_chunking_mode": prompt_chunking_mode,
                        "max_prompt_chunks": max_prompt_chunks,
                    }
                ],
            },
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @staticmethod
    def generate_controlnet_config(
        run_name: str,
        dataset_path: str,  # Deprecated - kept for backward compatibility
        base_model_path: str,
        output_dir: str,
        dataset_configs: Optional[List[Dict[str, Any]]] = None,  # New: multiple datasets
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 0,
        optimizer: str = "adamw8bit",
        optimizer_is_paged: bool = False,
        optimizer_cautious: bool = False,
        optimizer_beta1: Optional[float] = None,
        optimizer_beta2: Optional[float] = None,
        optimizer_epsilon: Optional[float] = None,
        optimizer_weight_decay: Optional[float] = None,
        optimizer_schedule_free: bool = False,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
        optimizer_use_radam: bool = False,
        optimizer_stochastic_rounding: bool = False,
        save_every: int = 500,
        save_every_unit: str = "steps",
        max_step_saves_to_keep: int = 5,
        sample_every: int = 500,
        sample_prompts: Optional[list] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
        enable_bucketing: bool = False,
        base_resolutions: Optional[List[int]] = None,
        bucket_strategy: str = "resize",
        multi_resolution_mode: str = "max",
        # ControlNet does NOT train UNet/TE (hardcoded in ControlNetTrainer)
        unet_lr: Optional[float] = None,
        cache_latents_to_disk: bool = False,
        weight_dtype: str = "fp16",
        training_dtype: str = "fp16",
        output_dtype: str = "fp32",
        vae_dtype: str = "fp16",
        mixed_precision: bool = True,
        use_flash_attention: bool = False,
        min_snr_gamma: float = 5.0,
        reconstruction_loss_weight: float = 0.0,
        # Text encoding settings
        text_encoding_mode: str = "swap_onthefly",
        text_encoding_swap_interval: int = 256,
        # Latent encoding settings
        latent_encoding_mode: str = "swap_onthefly",
        latent_encoding_swap_interval: int = 256,
        sample_width: int = 512,
        sample_height: int = 512,
        sample_steps: int = 20,
        sample_cfg_scale: float = 7.0,
        sample_sampler: str = "euler",
        sample_schedule_type: str = "normal",
        sample_seed: int = 42,
        # Prompt chunking settings (SD/SDXL only)
        prompt_chunking_mode: str = "a1111",
        max_prompt_chunks: int = 0,
        resume_from_checkpoint: Optional[str] = None,
        caption_processing: Optional[dict] = None,
        # Multi Noise-Timestep (MNT) settings
        multi_noise_timesteps: int = 1,
        multi_noise_mode: str = "independent",
        trajectory_blend_alpha: float = 0.7,
        timestep_sampling_config: Optional[Dict[str, Any]] = None,
        # Regularization settings
        regularization_type: Optional[str] = None,
        snr_regularization_weight: float = 0.1,
        snr_timestep_adaptive: bool = True,
        snr_penalty_mode: str = "relu",
        energy_regularization_weight: float = 0.05,
        energy_timestep_adaptive: bool = True,
        energy_penalty_mode: str = "abs",
        energy_normalize_by_pixels: bool = True,
        # Unified training framework settings
        noise_process: str = "auto",
        prediction_target: str = "auto",
        strict_validation: bool = False,
        # ControlNet-specific parameters
        controlnet_type: str = "standard",  # "standard" or "lllite"
        controlnet_pretrained_path: Optional[str] = None,
        controlnet_init_from_unet: bool = True,
        # LLLite parameters (Phase 2)
        lllite_conditioning_channels: int = 32,
        lllite_rank: int = 64,
        # Condition generation parameters (Phase 4)
        condition_preprocessors: Optional[List[str]] = None,
        condition_cache_mode: str = "on_the_fly",
    ) -> str:
        """
        Generate ControlNet training configuration YAML.

        Args:
            run_name: Training run identifier
            dataset_path: Path to dataset directory (deprecated, use dataset_configs)
            base_model_path: Path to base model
            output_dir: Output directory for checkpoints
            total_steps: Total training steps (mutually exclusive with epochs)
            epochs: Number of epochs (mutually exclusive with total_steps)
            controlnet_type: "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
            controlnet_pretrained_path: Path to existing ControlNet checkpoint for resume
            controlnet_init_from_unet: Initialize ControlNet from base UNet weights (standard only)
            lllite_conditioning_channels: Conditioning channels for LLLite
            lllite_rank: Rank for LLLite linear layers
            condition_preprocessors: List of controlnet-aux preprocessor types
            condition_cache_mode: "pre_generate" or "on_the_fly"

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
            # ControlNet training: UNet/TE are always frozen
            "train_unet": False,
            "train_text_encoder": False,
            "train_image_encoder": False,
            "optimizer": optimizer,
            "lr": learning_rate,
            "lr_scheduler": lr_scheduler,
            **({"lr_warmup_steps": lr_warmup_steps} if lr_warmup_steps > 0 else {}),
            **({"optimizer_is_paged": optimizer_is_paged} if optimizer_is_paged else {}),
            **({"optimizer_cautious": optimizer_cautious} if optimizer_cautious else {}),
            **({"optimizer_beta1": optimizer_beta1} if optimizer_beta1 is not None else {}),
            **({"optimizer_beta2": optimizer_beta2} if optimizer_beta2 is not None else {}),
            **({"optimizer_epsilon": optimizer_epsilon} if optimizer_epsilon is not None else {}),
            **({"optimizer_weight_decay": optimizer_weight_decay} if optimizer_weight_decay is not None else {}),
            **({"optimizer_schedule_free": optimizer_schedule_free} if optimizer_schedule_free else {}),
            **({"optimizer_schedule_free_r": optimizer_schedule_free_r} if optimizer_schedule_free and optimizer_schedule_free_r != 0.0 else {}),
            **({"optimizer_schedule_free_weight_lr_power": optimizer_schedule_free_weight_lr_power} if optimizer_schedule_free and optimizer_schedule_free_weight_lr_power != 2.0 else {}),
            **({"optimizer_use_radam": optimizer_use_radam} if optimizer_schedule_free and optimizer_use_radam else {}),
            **({"optimizer_stochastic_rounding": optimizer_stochastic_rounding} if optimizer_stochastic_rounding else {}),
            "mixed_precision": mixed_precision,
            "use_flash_attention": use_flash_attention,
            "min_snr_gamma": min_snr_gamma,
            "reconstruction_loss_weight": reconstruction_loss_weight,
            "text_encoding_mode": text_encoding_mode,
            "text_encoding_swap_interval": text_encoding_swap_interval,
            "latent_encoding_mode": latent_encoding_mode,
            "latent_encoding_swap_interval": latent_encoding_swap_interval,
            "debug_latents": debug_latents,
            "debug_latents_every": debug_latents_every,
            "multi_noise_timesteps": multi_noise_timesteps,
            "multi_noise_mode": multi_noise_mode,
            "trajectory_blend_alpha": trajectory_blend_alpha,
            **({"timestep_sampling": timestep_sampling_config} if timestep_sampling_config else {}),
            "resume_from_checkpoint": resume_from_checkpoint,
            # Regularization settings
            **({"regularization_type": regularization_type} if regularization_type else {}),
            "snr_regularization_weight": snr_regularization_weight,
            "snr_timestep_adaptive": snr_timestep_adaptive,
            "snr_penalty_mode": snr_penalty_mode,
            "energy_regularization_weight": energy_regularization_weight,
            "energy_timestep_adaptive": energy_timestep_adaptive,
            "energy_penalty_mode": energy_penalty_mode,
            "energy_normalize_by_pixels": energy_normalize_by_pixels,
            # Unified training framework settings
            "noise_process": noise_process,
            "prediction_target": prediction_target,
            "strict_validation": strict_validation,
        }

        # UNet LR is used as ControlNet LR (ControlNet mirrors UNet architecture)
        if unet_lr is not None:
            train_config["unet_lr"] = unet_lr

        # Add bucketing parameters
        if enable_bucketing:
            train_config["enable_bucketing"] = True
            train_config["base_resolutions"] = base_resolutions or [512]
            train_config["bucket_strategy"] = bucket_strategy
            train_config["multi_resolution_mode"] = multi_resolution_mode

        # Build datasets array
        datasets_array = []
        if dataset_configs:
            for ds_config in dataset_configs:
                ds_path = ds_config.get("path", "")
                ds_caption_types = ds_config.get("caption_types", [])
                ds_dataset_id = ds_config.get("dataset_id")

                dataset_entry = {
                    "folder_path": ds_path,
                    "caption_ext": "txt",
                    "cache_latents_to_disk": cache_latents_to_disk,
                    **({"dataset_id": ds_dataset_id} if ds_dataset_id else {}),
                }

                if ds_caption_types:
                    dataset_entry["caption_types"] = ds_caption_types

                datasets_array.append(dataset_entry)
        else:
            dataset_entry = {
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "cache_latents_to_disk": cache_latents_to_disk,
            }
            datasets_array.append(dataset_entry)

        # Build ControlNet-specific network config
        controlnet_network_config = {
            "type": controlnet_type,
            "init_from_unet": controlnet_init_from_unet,
        }

        if controlnet_pretrained_path:
            controlnet_network_config["pretrained_path"] = controlnet_pretrained_path

        # LLLite-specific parameters
        if controlnet_type == "lllite":
            controlnet_network_config["lllite_conditioning_channels"] = lllite_conditioning_channels
            controlnet_network_config["lllite_rank"] = lllite_rank

        # Condition generation parameters
        if condition_preprocessors:
            controlnet_network_config["condition_preprocessors"] = condition_preprocessors
            controlnet_network_config["condition_cache_mode"] = condition_cache_mode

        config = {
            "job": run_name,
            "config": {
                "name": run_name,
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": output_dir,
                        "device": "cuda:0",
                        "network": {
                            "type": "controlnet",
                            "controlnet": controlnet_network_config,
                        },
                        "dtype": {
                            "weight": weight_dtype,
                            "training": training_dtype,
                            "vae": vae_dtype,
                            "save": output_dtype,
                        },
                        "save": {
                            "save_every": save_every,
                            "save_every_unit": save_every_unit,
                            "max_step_saves_to_keep": max_step_saves_to_keep,
                        },
                        "datasets": datasets_array,
                        "train": train_config,
                        "model": {
                            "name_or_path": base_model_path,
                        },
                        "sample": {
                            "sampler": sample_sampler,
                            "schedule_type": sample_schedule_type,
                            "sample_every": sample_every,
                            "width": sample_width,
                            "height": sample_height,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": sample_seed,
                            "guidance_scale": sample_cfg_scale,
                            "sample_steps": sample_steps,
                        },
                        "prompt_chunking_mode": prompt_chunking_mode,
                        "max_prompt_chunks": max_prompt_chunks,
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
