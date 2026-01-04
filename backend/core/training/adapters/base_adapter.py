"""
Base adapter classes for model-specific training logic.

Author: Claude (2026-01-04)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn


class BaseLoRAAdapter(ABC):
    """
    Abstract base class for model-specific LoRA adapters.

    Each model architecture (SD1.5, SDXL, Z-Image) implements this interface
    to provide model-specific LoRA injection, parameter collection, and
    checkpoint saving logic.
    """

    def __init__(self, trainer, lora_rank: int, lora_alpha: int):
        """
        Initialize adapter.

        Args:
            trainer: Parent trainer instance (BaseTrainer subclass)
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha (scaling factor = alpha / rank)
        """
        self.trainer = trainer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scale = lora_alpha / lora_rank

    @abstractmethod
    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to U-Net/Transformer.

        Args:
            lora_layers: Dictionary to store LoRA layer references (key: name, value: LoRA module)

        Returns:
            Number of LoRA layers injected
        """
        pass

    @abstractmethod
    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """
        Apply LoRA to text encoder(s).

        Args:
            lora_layers: Dictionary to store LoRA layer references (key: name, value: LoRA module)

        Returns:
            Number of LoRA layers injected
        """
        pass

    @abstractmethod
    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Args:
            lora_layers: Dictionary of LoRA layers (key: name, value: LoRA module)

        Returns:
            List of parameter groups for optimizer (format: [{"params": [...], "lr": ...}, ...])
        """
        pass

    @abstractmethod
    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path
    ):
        """
        Save LoRA checkpoint in model-specific format.

        Args:
            lora_layers: Dictionary of LoRA layers (key: name, value: LoRA module)
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        pass


class BaseFullParameterAdapter(ABC):
    """
    Abstract base class for model-specific full parameter training adapters.

    Each model architecture (SD1.5, SDXL, Z-Image) implements this interface
    to provide model-specific parameter preparation, collection, and
    checkpoint saving logic.
    """

    def __init__(self, trainer):
        """
        Initialize adapter.

        Args:
            trainer: Parent trainer instance (BaseTrainer subclass)
        """
        self.trainer = trainer

    @abstractmethod
    def prepare_models_for_training(self):
        """
        Prepare models for full parameter training.

        This includes:
        - Setting requires_grad=True for trainable components
        - Freezing non-trainable components
        - Enabling gradient checkpointing
        """
        pass

    @abstractmethod
    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """
        Collect trainable parameters with per-component learning rates.

        Returns:
            List of parameter groups for optimizer (format: [{"params": [...], "lr": ...}, ...])
        """
        pass

    @abstractmethod
    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """
        Save full parameter checkpoint in model-specific format.

        Args:
            step: Current training step
            epoch: Current training epoch
            output_path: Path to save checkpoint
        """
        pass
