"""
Latent Generator for Aesthetic Scorer

Generates large datasets of predicted latents from SushiUI training data.
Uses minimal disk space by saving only latents + predicted_latent.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import sys
import random

# Add parent directory to path for SushiUI imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "backend"))

from sqlalchemy.orm import Session
from database.models import DatasetBase, Dataset, DatasetItem, DatasetCaption
from core.training.latent_cache import LatentCache
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


class LatentGenerator:
    """
    Generate predicted latents from SushiUI datasets for aesthetic scoring.

    This class loads a Z-Image model and generates predicted latents by:
    1. Loading cached latents from SushiUI dataset
    2. Adding noise at random timesteps
    3. Running forward pass to predict velocity
    4. Calculating predicted latent (x0 = xt - t * v)
    5. Saving minimal data (latents + predicted_latent only)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        weight_dtype: str = "fp16",
        vae_dtype: str = "fp16",
    ):
        """
        Initialize latent generator.

        Args:
            model_path: Path to Z-Image model (safetensors or diffusers folder)
            device: Device to use (cuda/cpu)
            weight_dtype: Weight dtype for transformer (fp16, bf16, fp32)
            vae_dtype: VAE dtype (fp16 recommended)
        """
        self.device = device
        self.weight_dtype = self._get_torch_dtype(weight_dtype)
        self.vae_dtype = self._get_torch_dtype(vae_dtype)

        print(f"[LatentGenerator] Loading model from {model_path}")

        # Load model components
        self._load_model(model_path)

        # Setup scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
        )

        print(f"[LatentGenerator] Model loaded successfully")
        print(f"[LatentGenerator] Transformer: {sum(p.numel() for p in self.transformer.parameters()):,} parameters")

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch.dtype."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def _load_model(self, model_path: str):
        """Load Z-Image model components."""
        from diffusers import ZImageTransformer2DModel

        # Load Transformer
        self.transformer = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.transformer.eval()

        # Load Text Encoder
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.vae_dtype,
        ).to(self.device)
        self.vae.eval()

        # Freeze all parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

    def generate_latents_from_dataset(
        self,
        sushiui_db_session: Session,
        dataset_id: int,
        num_samples: Optional[int] = None,
        timestep_range: Tuple[float, float] = (0.0, 1.0),
        output_dir: Path = Path("subapps/aesthetic_scorer/data/latents"),
        save_mode: str = "minimal",
        shuffle: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate predicted latents from SushiUI dataset.

        Args:
            sushiui_db_session: SQLAlchemy session for SushiUI datasets.db
            dataset_id: Dataset ID from SushiUI datasets.db
            num_samples: Number of samples to generate (None = all items)
            timestep_range: (min_t, max_t) for noise addition
            output_dir: Output directory for .pt files
            save_mode: "minimal" (latents + predicted_latent) or "debug" (all tensors)
            shuffle: Shuffle dataset items before generation

        Returns:
            List of generated records (for database insertion)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset from SushiUI datasets.db
        dataset = sushiui_db_session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found in SushiUI datasets.db")

        print(f"[LatentGenerator] Dataset: {dataset.name}")
        print(f"[LatentGenerator] Dataset path: {dataset.path}")
        print(f"[LatentGenerator] Dataset unique_id: {dataset.unique_id}")

        # Get dataset items
        query = sushiui_db_session.query(DatasetItem).filter(
            DatasetItem.dataset_id == dataset_id
        )

        if num_samples is not None:
            total_items = query.count()
            print(f"[LatentGenerator] Total items: {total_items}, generating {num_samples} samples")

        items = query.all()

        if shuffle:
            random.shuffle(items)

        if num_samples is not None:
            items = items[:num_samples]

        print(f"[LatentGenerator] Generating {len(items)} samples (mode={save_mode})")

        # Initialize latent cache
        cache_dir = Path(dataset.cache_dir) if dataset.cache_dir else Path("backend/cache")
        latent_cache = LatentCache(
            dataset_unique_id=dataset.unique_id,
            cache_dir=cache_dir,
            vae=self.vae,
            device=self.device,
            dtype=self.vae_dtype,
        )

        generated_records = []

        for idx, item in enumerate(tqdm(items, desc="Generating latents")):
            try:
                # Load cached latent
                latent = latent_cache.load_latent(
                    image_path=item.image_path,
                    width=item.width,
                    height=item.height,
                )  # [1, 16, H, W]

                # Random timestep
                t = random.uniform(*timestep_range)

                # Load caption
                caption = self._get_caption(sushiui_db_session, item, dataset)

                # Forward pass (no gradient)
                with torch.no_grad():
                    result = self._single_forward_pass(
                        latent=latent,
                        prompt=caption,
                        timestep=t,
                    )

                # Save to .pt file
                filename = f"latents_{dataset.unique_id}_{idx:06d}_t{t:.4f}.pt"
                output_path = output_dir / filename

                self._save_latent_data(
                    output_path=output_path,
                    data=result,
                    save_mode=save_mode,
                )

                # Create record for database
                record = {
                    "filename": str(output_path),
                    "dataset_id": dataset_id,
                    "dataset_name": dataset.name,
                    "dataset_unique_id": dataset.unique_id,
                    "image_path": item.image_path,
                    "caption": caption,
                    "timestep": t,
                    "recon_loss": result["recon_loss"],
                    "latent_shape": list(result["latents"].shape),
                    "scheduler_type": result["scheduler_type"],
                }

                generated_records.append(record)

                # Cleanup
                del latent, result
                if idx % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[LatentGenerator] Error processing item {idx}: {e}")
                continue

        total_size_gb = sum(Path(r["filename"]).stat().st_size for r in generated_records) / 1024**3
        print(f"[LatentGenerator] Generated {len(generated_records)} samples")
        print(f"[LatentGenerator] Total size: {total_size_gb:.2f} GB")
        print(f"[LatentGenerator] Average size: {total_size_gb / len(generated_records) * 1024:.2f} MB/sample")

        return generated_records

    def _single_forward_pass(
        self,
        latent: torch.Tensor,
        prompt: str,
        timestep: float,
    ) -> Dict[str, Any]:
        """
        Single forward pass without backpropagation.

        Args:
            latent: Ground truth latent [1, 16, H, W]
            prompt: Text prompt
            timestep: Noise timestep (0.0-1.0)

        Returns:
            {
                'latents': torch.Tensor [1, 16, H, W],
                'predicted_latent': torch.Tensor [1, 16, H, W],
                'timestep': float,
                'recon_loss': float,
                'caption': str,
                'scheduler_type': str,
            }
        """
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # Add noise
        noise = torch.randn_like(latent)
        t_tensor = torch.tensor([timestep], device=self.device, dtype=self.weight_dtype)
        noisy_latent = self.noise_scheduler.scale_noise(latent, t_tensor, noise)

        # Reshape for transformer (2D -> sequence)
        B, C, H, W = noisy_latent.shape
        noisy_latent_2d = noisy_latent.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        # Forward pass
        model_pred = self.transformer(
            hidden_states=noisy_latent_2d,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]  # [B, H*W, C]

        # Reshape back to 4D
        model_pred = model_pred.transpose(1, 2).view(B, C, H, W)

        # Calculate predicted latent (Flow Matching: x0 = xt - t * v)
        t_expanded = t_tensor.view(-1, 1, 1, 1)
        predicted_latent = noisy_latent - t_expanded * model_pred

        # Reconstruction loss
        recon_loss = F.mse_loss(predicted_latent, latent)

        return {
            'latents': latent.cpu(),
            'predicted_latent': predicted_latent.cpu(),
            'timestep': timestep,
            'recon_loss': recon_loss.item(),
            'caption': prompt,
            'scheduler_type': 'FlowMatching',
        }

    def _encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt using CLIP text encoder.

        Args:
            prompt: Text prompt

        Returns:
            (prompt_embeds, pooled_prompt_embeds)
        """
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(self.device)

        # Encode
        encoder_output = self.text_encoder(input_ids, output_hidden_states=True)
        prompt_embeds = encoder_output.hidden_states[-2]  # Penultimate layer
        pooled_prompt_embeds = encoder_output[0]  # Pooled output

        return prompt_embeds, pooled_prompt_embeds

    def _get_caption(
        self,
        db_session: Session,
        item: DatasetItem,
        dataset: Dataset,
    ) -> str:
        """
        Get caption for dataset item.

        Args:
            db_session: SQLAlchemy session
            item: DatasetItem
            dataset: Dataset

        Returns:
            Caption string
        """
        # Get primary caption
        caption_type = dataset.default_caption_type or "tags"

        caption_record = db_session.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
            DatasetCaption.caption_type == caption_type,
        ).first()

        if caption_record:
            return caption_record.content

        # Fallback: any caption
        any_caption = db_session.query(DatasetCaption).filter(
            DatasetCaption.item_id == item.id,
        ).first()

        if any_caption:
            return any_caption.content

        # No caption
        return ""

    def _save_latent_data(
        self,
        output_path: Path,
        data: Dict[str, Any],
        save_mode: str = "minimal",
    ):
        """
        Save latent data with configurable verbosity.

        Args:
            output_path: Output .pt file path
            data: Data dictionary
            save_mode: "minimal" or "debug"
        """
        if save_mode == "minimal":
            # Aesthetic scorer: latents + predicted_latent only
            minimal_data = {
                'latents': data['latents'],
                'predicted_latent': data['predicted_latent'],
                'timestep': data['timestep'],
                'recon_loss': data['recon_loss'],
                'caption': data['caption'],
                'scheduler_type': data['scheduler_type'],
            }
            torch.save(minimal_data, output_path)

        elif save_mode == "debug":
            # Training debug: all tensors
            torch.save(data, output_path)

        else:
            raise ValueError(f"Unknown save_mode: {save_mode}")
