"""
Latent Cache Management for Training

Caches VAE latents and optionally text embeddings to disk to reduce VRAM usage during training.
By default, only VAE latents are cached (text embeddings encoded on-the-fly).
Cache is stored per-dataset to allow reuse across multiple training runs.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import torch
from PIL import Image


def get_cache_base_dir() -> str:
    """
    Get the base cache directory from user settings.

    Returns:
        Base cache directory path (default: "cache/datasets")
    """
    try:
        from database.gallery_db import get_gallery_db
        from database.models import UserSettings

        db = next(get_gallery_db())
        try:
            settings = db.query(UserSettings).first()
            if settings and settings.cache_dir:
                # User configured cache directory
                return str(Path(settings.cache_dir) / "datasets")
        finally:
            db.close()
    except Exception as e:
        # Fallback to default if database query fails
        print(f"[Cache] Warning: Failed to get cache_dir from settings: {e}")

    # Default cache directory
    return "cache/datasets"


class LatentCache:
    """
    Manages disk cache for VAE latents and optionally text embeddings.

    Cache directory structure:
        cache/datasets/{dataset_unique_id}/
            ├── latents/
            │   ├── {image_hash}.pt
            │   └── ...
            ├── text_embeddings/  (optional)
            │   ├── {caption_hash}_clip1.pt
            │   ├── {caption_hash}_clip2.pt  (SDXL only)
            │   ├── {caption_hash}_pooled.pt (SDXL only)
            │   └── ...
            └── cache_info.json
    """

    def __init__(self, dataset_unique_id: str, base_cache_dir: str = None):
        """
        Initialize latent cache.

        Args:
            dataset_unique_id: Dataset unique ID (UUID)
            base_cache_dir: Base directory for cache (default: from user settings or "cache/datasets")
        """
        self.dataset_unique_id = dataset_unique_id
        if base_cache_dir is None:
            base_cache_dir = get_cache_base_dir()
        self.cache_dir = Path(base_cache_dir) / dataset_unique_id
        self.latents_dir = self.cache_dir / "latents"
        self.embeddings_dir = self.cache_dir / "text_embeddings"
        self.cache_info_path = self.cache_dir / "cache_info.json"

        # Create directories
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_image_hash(image_path: str, width: int, height: int) -> str:
        """
        Compute hash for image cache key.

        Includes image path and target dimensions to handle bucketing.

        Args:
            image_path: Path to image
            width: Target width
            height: Target height

        Returns:
            Hash string
        """
        key = f"{image_path}_{width}_{height}"
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def compute_caption_hash(caption: str) -> str:
        """
        Compute hash for caption cache key.

        Args:
            caption: Text caption

        Returns:
            Hash string
        """
        return hashlib.md5(caption.encode()).hexdigest()

    def save_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        latents: torch.Tensor
    ):
        """
        Save VAE latents to cache.

        Args:
            image_path: Source image path
            width: Target width
            height: Target height
            latents: Latent tensor [1, 4, H/8, W/8]
        """
        cache_hash = self.compute_image_hash(image_path, width, height)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        torch.save({
            'latents': latents.cpu(),
            'image_path': image_path,
            'width': width,
            'height': height,
            'created_at': datetime.utcnow().isoformat(),
        }, cache_path)

    def load_latent(
        self,
        image_path: str,
        width: int,
        height: int,
        device: str = 'cuda'
    ) -> Optional[torch.Tensor]:
        """
        Load VAE latents from cache.

        Args:
            image_path: Source image path
            width: Target width
            height: Target height
            device: Device to load tensor to

        Returns:
            Latent tensor or None if not cached
        """
        cache_hash = self.compute_image_hash(image_path, width, height)
        cache_path = self.latents_dir / f"{cache_hash}.pt"

        if not cache_path.exists():
            return None

        try:
            data = torch.load(cache_path, map_location=device)
            return data['latents']
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached latent {cache_path}: {e}")
            return None

    def save_text_embeddings(
        self,
        caption: str,
        text_embeddings: torch.Tensor,
        pooled_embeddings: Optional[torch.Tensor] = None,
        text_embeddings_2: Optional[torch.Tensor] = None
    ):
        """
        Save text embeddings to cache.

        Args:
            caption: Text caption
            text_embeddings: Text embeddings from first encoder [1, 77, 768]
            pooled_embeddings: Pooled embeddings (SDXL only) [1, 1280]
            text_embeddings_2: Text embeddings from second encoder (SDXL only) [1, 77, 1280]
        """
        caption_hash = self.compute_caption_hash(caption)

        # Save CLIP-L embeddings (or SD1.5 embeddings)
        clip1_path = self.embeddings_dir / f"{caption_hash}_clip1.pt"
        torch.save({
            'embeddings': text_embeddings.cpu(),
            'caption': caption,
            'created_at': datetime.utcnow().isoformat(),
        }, clip1_path)

        # Save SDXL-specific embeddings
        if pooled_embeddings is not None:
            pooled_path = self.embeddings_dir / f"{caption_hash}_pooled.pt"
            torch.save({
                'embeddings': pooled_embeddings.cpu(),
                'caption': caption,
                'created_at': datetime.utcnow().isoformat(),
            }, pooled_path)

        if text_embeddings_2 is not None:
            clip2_path = self.embeddings_dir / f"{caption_hash}_clip2.pt"
            torch.save({
                'embeddings': text_embeddings_2.cpu(),
                'caption': caption,
                'created_at': datetime.utcnow().isoformat(),
            }, clip2_path)

    def load_text_embeddings(
        self,
        caption: str,
        is_sdxl: bool = False,
        device: str = 'cuda'
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Load text embeddings from cache.

        Args:
            caption: Text caption
            is_sdxl: Whether to load SDXL embeddings (includes pooled and clip2)
            device: Device to load tensors to

        Returns:
            For SD1.5: (text_embeddings,)
            For SDXL: (text_embeddings, pooled_embeddings)
            Returns None if not cached
        """
        caption_hash = self.compute_caption_hash(caption)
        clip1_path = self.embeddings_dir / f"{caption_hash}_clip1.pt"

        if not clip1_path.exists():
            return None

        try:
            # Load CLIP-L embeddings
            data = torch.load(clip1_path, map_location=device)
            text_embeddings = data['embeddings']

            if is_sdxl:
                # Load pooled embeddings
                pooled_path = self.embeddings_dir / f"{caption_hash}_pooled.pt"
                if not pooled_path.exists():
                    return None

                pooled_data = torch.load(pooled_path, map_location=device)
                pooled_embeddings = pooled_data['embeddings']

                return (text_embeddings, pooled_embeddings)
            else:
                return (text_embeddings,)

        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cached embeddings for caption: {e}")
            return None

    def save_cache_info(self, model_path: str, model_type: str, item_count: int):
        """
        Save cache metadata.

        Args:
            model_path: Path to base model
            model_type: Model type ('sdxl' or 'sd15')
            item_count: Number of items in dataset
        """
        info = {
            'dataset_unique_id': self.dataset_unique_id,
            'model_path': model_path,
            'model_type': model_type,
            'created_at': datetime.utcnow().isoformat(),
            'item_count': item_count,
        }

        with open(self.cache_info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def load_cache_info(self) -> Optional[Dict]:
        """
        Load cache metadata.

        Returns:
            Cache info dict or None if not exists
        """
        if not self.cache_info_path.exists():
            return None

        try:
            with open(self.cache_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LatentCache] Warning: Failed to load cache info: {e}")
            return None

    def is_valid(self, model_path: str, model_type: str) -> bool:
        """
        Check if cache is valid for current model.

        Args:
            model_path: Current model path
            model_type: Current model type

        Returns:
            True if cache is valid
        """
        info = self.load_cache_info()
        if info is None:
            print(f"[LatentCache] Validation failed: No cache_info.json found")
            return False

        # Normalize paths for comparison (resolve to absolute, case-normalized)
        from pathlib import Path
        cached_model_path = info.get('model_path')
        if cached_model_path is None:
            print(f"[LatentCache] Validation failed: model_path not in cache_info.json")
            return False

        try:
            cached_path_normalized = Path(cached_model_path).resolve()
            current_path_normalized = Path(model_path).resolve()
        except Exception as e:
            print(f"[LatentCache] Warning: Path resolution failed ({e}), using string comparison")
            # If path resolution fails, fall back to string comparison
            cached_path_normalized = cached_model_path
            current_path_normalized = model_path

        # Check model compatibility (compare normalized paths)
        if cached_path_normalized != current_path_normalized:
            print(f"[LatentCache] Validation failed: Model path mismatch")
            print(f"[LatentCache]   Cached: {cached_path_normalized}")
            print(f"[LatentCache]   Current: {current_path_normalized}")
            return False

        if info.get('model_type') != model_type:
            print(f"[LatentCache] Validation failed: Model type mismatch")
            print(f"[LatentCache]   Cached: {info.get('model_type')}")
            print(f"[LatentCache]   Current: {model_type}")
            return False

        print(f"[LatentCache] Validation passed: Cache is valid for current model")
        return True

    def validate_cache_format(self, expected_channels: int = 4, sample_count: int = 5) -> bool:
        """
        Validate cache format by randomly sampling cached latents.

        Args:
            expected_channels: Expected number of latent channels (4 for SD/SDXL)
            sample_count: Number of random samples to check

        Returns:
            True if cache format is valid, False otherwise
        """
        import random

        # Get all cached latent files
        latent_files = list(self.latents_dir.glob("*.pt"))

        if len(latent_files) == 0:
            print(f"[LatentCache] No cached latents found in {self.latents_dir}")
            return False

        # Sample random files
        sample_size = min(sample_count, len(latent_files))
        sampled_files = random.sample(latent_files, sample_size)

        print(f"[LatentCache] Validating cache format (sampling {sample_size}/{len(latent_files)} cached latents)...")

        for latent_file in sampled_files:
            try:
                data = torch.load(latent_file, map_location='cpu')

                # Extract latent tensor from dict (cache format: {'latents': tensor, ...})
                if isinstance(data, dict):
                    latent = data.get('latents')
                    if latent is None:
                        print(f"[LatentCache] VALIDATION FAILED: 'latents' key not found in {latent_file.name}")
                        return False
                else:
                    # Legacy format: tensor directly saved
                    latent = data

                # Check shape
                if latent.dim() != 4:
                    print(f"[LatentCache] VALIDATION FAILED: Expected 4D tensor, got {latent.dim()}D")
                    return False

                # Check channels (B, C, H, W)
                if latent.shape[1] != expected_channels:
                    print(f"[LatentCache] VALIDATION FAILED: Expected {expected_channels} channels, got {latent.shape[1]}")
                    return False

                # Log sample info
                print(f"[LatentCache]   Sample: shape={latent.shape}, dtype={latent.dtype}")

            except Exception as e:
                print(f"[LatentCache] VALIDATION FAILED: Error loading {latent_file.name}: {e}")
                return False

        print(f"[LatentCache] Cache format validation PASSED")
        return True

    def clear(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.latents_dir.mkdir(parents=True, exist_ok=True)
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
