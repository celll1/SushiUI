"""
Aesthetic Model Trainer

Trains the aesthetic scoring model from user-scored latent records.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from tqdm import tqdm
import json
from datetime import datetime

from .aesthetic_model import create_aesthetic_model


class LatentScoreDataset(Dataset):
    """
    Dataset for loading scored latent records.

    Loads .pt files containing predicted latents and corresponding user scores.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        include_metadata: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            records: List of LatentRecord dictionaries from database
            include_metadata: Include metadata (timestep, recon_loss, caption) in return
        """
        self.records = records
        self.include_metadata = include_metadata

        print(f"[LatentScoreDataset] Loaded {len(records)} scored samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.

        Returns:
            {
                'latent': [16, H, W] Predicted latent
                'score': [1] User score (0=best, 1=worst)
                'timestep': [1] (optional, if include_metadata=True)
                'recon_loss': [1] (optional, if include_metadata=True)
                'caption': str (optional, if include_metadata=True)
            }
        """
        record = self.records[idx]

        # Load .pt file
        data = torch.load(record["filename"], map_location="cpu")

        predicted_latent = data["predicted_latent"].squeeze(0)  # [16, H, W]
        user_score = torch.tensor([record["user_score"]], dtype=torch.float32)

        result = {
            "latent": predicted_latent,
            "score": user_score,
        }

        if self.include_metadata:
            result["timestep"] = torch.tensor([record["timestep"]], dtype=torch.float32)
            result["recon_loss"] = torch.tensor([record["recon_loss"]], dtype=torch.float32)
            result["caption"] = record["caption"]

        return result


class AestheticTrainer:
    """
    Trainer for aesthetic scoring models.

    Trains a lightweight neural network to predict user scores from predicted latents.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize trainer.

        Args:
            model: Aesthetic model (LatentCNN or LatentTransformer)
            device: Device to use (cuda/cpu)
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
        """
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.MSELoss()

        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []

        print(f"[AestheticTrainer] Initialized")
        print(f"[AestheticTrainer] Model: {model.__class__.__name__}")
        print(f"[AestheticTrainer] Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"[AestheticTrainer] Learning rate: {learning_rate}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        progress_callback: Optional[Callable] = None,
    ) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            progress_callback: Callback function for progress updates

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            latents = batch["latent"].to(self.device)  # [B, 16, H, W]
            scores = batch["score"].to(self.device)  # [B, 1]

            # Forward pass
            pred_scores = self.model(latents)  # [B, 1]

            # Compute loss
            loss = self.criterion(pred_scores, scores)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Progress callback
            if progress_callback is not None:
                progress_callback({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "total_batches": num_batches,
                    "loss": loss.item(),
                })

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate model.

        Args:
            dataloader: Validation dataloader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in tqdm(dataloader, desc="Validating"):
            latents = batch["latent"].to(self.device)
            scores = batch["score"].to(self.device)

            # Forward pass
            pred_scores = self.model(latents)

            # Compute loss
            loss = self.criterion(pred_scores, scores)
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        save_dir: Path = Path("subapps/aesthetic_scorer/models"),
        model_name: str = "aesthetic",
        save_every: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints
            model_name: Model name for checkpoints
            save_every: Save checkpoint every N epochs
            progress_callback: Callback for progress updates

        Returns:
            Training summary dict
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[AestheticTrainer] Starting training for {num_epochs} epochs")

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, epoch, progress_callback)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_model_path = save_dir / f"{model_name}_best.safetensors"
                    self.save_checkpoint(best_model_path)
                    print(f"[AestheticTrainer] New best model saved (val_loss={val_loss:.6f})")
            else:
                val_loss = None
                print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {train_loss:.6f}")

            # Save periodic checkpoint
            if epoch % save_every == 0:
                checkpoint_path = save_dir / f"{model_name}_epoch{epoch}.safetensors"
                self.save_checkpoint(checkpoint_path)

        # Save final model
        final_model_path = save_dir / f"{model_name}_final.safetensors"
        self.save_checkpoint(final_model_path)

        # Training summary
        summary = {
            "num_epochs": num_epochs,
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1] if self.val_losses else None,
            "best_val_loss": best_val_loss if val_loader is not None else None,
            "best_epoch": best_epoch if val_loader is not None else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Save training history
        history_path = save_dir / f"{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[AestheticTrainer] Training completed")
        print(f"[AestheticTrainer] Final train loss: {summary['final_train_loss']:.6f}")
        if summary["final_val_loss"] is not None:
            print(f"[AestheticTrainer] Final val loss: {summary['final_val_loss']:.6f}")
            print(f"[AestheticTrainer] Best val loss: {summary['best_val_loss']:.6f} (epoch {summary['best_epoch']})")

        return summary

    def save_checkpoint(self, path: Path):
        """
        Save model checkpoint.

        Args:
            path: Output .safetensors file path
        """
        from safetensors.torch import save_file

        state_dict = self.model.state_dict()
        save_file(state_dict, str(path))

    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.

        Args:
            path: Input .safetensors file path
        """
        from safetensors.torch import load_file

        state_dict = load_file(str(path))
        self.model.load_state_dict(state_dict)
        print(f"[AestheticTrainer] Loaded checkpoint from {path}")


def create_dataloaders(
    scored_records: List[Dict[str, Any]],
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders from scored records.

    Args:
        scored_records: List of LatentRecord dictionaries (with user_score)
        batch_size: Batch size
        val_split: Validation split ratio (0.0-1.0)
        num_workers: Number of dataloader workers

    Returns:
        (train_loader, val_loader)
    """
    dataset = LatentScoreDataset(scored_records)

    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"[DataLoaders] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        return train_loader, val_loader

    else:
        # No validation split
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        print(f"[DataLoaders] Train samples: {len(dataset)} (no validation)")

        return train_loader, None
