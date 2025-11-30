"""
Training configuration generator for ai-toolkit.

Generates YAML configuration files based on training parameters.
"""

from typing import Dict, Any, Optional
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
        sample_every: int = 100,
        sample_prompts: Optional[list] = None,
        debug_latents: bool = False,
        debug_latents_every: int = 50,
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
            sample_every: Generate sample every N steps/epochs
            sample_prompts: List of prompts for sample generation
            debug_latents: Enable debug mode to save latents
            debug_latents_every: Save debug latents every N steps

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
                            "max_step_saves_to_keep": 10,
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_path,
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": True,
                                "resolution": [512, 768, 1024],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            **({"steps": total_steps} if total_steps else {"epochs": epochs}),
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "lr_scheduler": lr_scheduler,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                            "debug_latents": debug_latents,
                            "debug_latents_every": debug_latents_every,
                        },
                        "model": {
                            "name_or_path": base_model_path,
                            "is_flux": False,
                            "quantize": False,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": sample_every,
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 4,
                            "sample_steps": 20,
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
        sample_every: int = 100,
        sample_prompts: Optional[list] = None,
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
                            "type": "full",  # Full fine-tuning
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": save_every,
                            "max_step_saves_to_keep": 5,  # Fewer saves for full models
                        },
                        "datasets": [
                            {
                                "folder_path": dataset_path,
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": True,
                                "resolution": [512, 768, 1024],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            **({"steps": total_steps} if total_steps else {"epochs": epochs}),
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": True,  # Train text encoder for full fine-tune
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "lr_scheduler": lr_scheduler,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                        },
                        "model": {
                            "name_or_path": base_model_path,
                            "is_flux": False,
                            "quantize": False,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": sample_every,
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts or [],
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 4,
                            "sample_steps": 20,
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
