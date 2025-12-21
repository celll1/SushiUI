"""
SQLAlchemy models for Aesthetic Scorer database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


def get_local_now():
    """Get current local time (not UTC)"""
    return datetime.now()


class LatentRecord(Base):
    """
    Record of a generated predicted latent for scoring.

    Each record corresponds to one .pt file containing:
    - latents (ground truth)
    - predicted_latent (model prediction at timestep t)
    """
    __tablename__ = "latent_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)  # latents_xxx_000000_t0.6579.pt

    # Source information
    dataset_id = Column(Integer, index=True)  # SushiUI dataset ID
    dataset_name = Column(String)
    dataset_unique_id = Column(String)  # Dataset unique_id
    image_path = Column(String)  # Original image path
    caption = Column(Text)  # Text prompt

    # Latent metadata
    timestep = Column(Float)  # 0.0-1.0
    recon_loss = Column(Float)  # MSE(predicted_latent, latents)
    latent_shape = Column(JSON)  # [B, C, H, W]
    scheduler_type = Column(String)  # "FlowMatching"

    # Scoring
    user_score = Column(Float, nullable=True)  # 0.0-1.0 (0=best, 1=worst)
    model_score = Column(Float, nullable=True)  # Predicted score from aesthetic model
    is_scored = Column(Boolean, default=False, index=True)

    # Cached decoded images (for UI display)
    true_latent_image_path = Column(String, nullable=True)  # VAE decoded ground truth
    predicted_latent_image_path = Column(String, nullable=True)  # VAE decoded prediction

    # Metadata
    created_at = Column(DateTime, default=get_local_now, index=True)
    updated_at = Column(DateTime, default=get_local_now, onupdate=get_local_now)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "filename": self.filename,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_unique_id": self.dataset_unique_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "timestep": self.timestep,
            "recon_loss": self.recon_loss,
            "latent_shape": self.latent_shape,
            "scheduler_type": self.scheduler_type,
            "user_score": self.user_score,
            "model_score": self.model_score,
            "is_scored": self.is_scored,
            "true_latent_image_path": self.true_latent_image_path,
            "predicted_latent_image_path": self.predicted_latent_image_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AestheticModel(Base):
    """
    Record of a trained aesthetic scoring model.
    """
    __tablename__ = "aesthetic_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # "aesthetic_v1", "aesthetic_v2", etc.
    version = Column(String)  # "1.0.0"

    # Model architecture
    architecture = Column(String)  # "LatentCNN", "LatentTransformer", etc.
    parameters = Column(JSON)  # Model config dict

    # Training info
    training_config = Column(JSON)  # Learning rate, batch size, etc.
    num_scored_samples = Column(Integer)  # Number of samples used for training
    num_epochs = Column(Integer)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)

    # Model file
    model_path = Column(String)  # Path to .safetensors file
    is_active = Column(Boolean, default=False, index=True)  # Currently active model

    # Metadata
    created_at = Column(DateTime, default=get_local_now)

    def to_dict(self):
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "training_config": self.training_config,
            "num_scored_samples": self.num_scored_samples,
            "num_epochs": self.num_epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "model_path": self.model_path,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
