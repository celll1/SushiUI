"""
Layer Pruning for Z-Image Transformer

This script implements layer pruning with greedy search to minimize
the 1-step inference loss difference from the original model.

Strategy:
1. Load original model (30 layers)
2. Load verification samples
3. Calculate baseline 1-step inference loss
4. Iteratively remove layers that minimize loss increase
5. Save pruned model to safetensors
"""

import sys
import os
import json
import argparse
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file, load_file

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from core.models.zimage_transformer import ZImageTransformer2DModel
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, AutoTokenizer


class LayerPruner:
    """
    Layer pruning with 1-step inference loss evaluation.
    """

    def __init__(
        self,
        model_path: str,
        samples: List[Dict[str, Any]],
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize pruner.

        Args:
            model_path: Path to original model (safetensors)
            samples: Verification samples (from extract_samples.py)
            device: Device for computation
            dtype: Data type for computation
        """
        self.model_path = model_path
        self.samples = samples
        self.device = device
        self.dtype = dtype

        print("[Pruner] Initializing...")
        print(f"[Pruner] Model: {model_path}")
        print(f"[Pruner] Samples: {len(samples)}")
        print(f"[Pruner] Device: {device}")
        print(f"[Pruner] dtype: {dtype}")

        # Load components
        self._load_components()

        # Prepare sample data
        self._prepare_samples()

    def _load_components(self):
        """Load VAE, Text Encoder, Tokenizer, Scheduler, Transformer."""
        print("[Pruner] Loading components...")

        # Use Tongyi-MAI/Z-Image-Turbo (same as SushiUI)
        base_model_repo = "Tongyi-MAI/Z-Image-Turbo"

        # VAE (keep on CPU during transformer evaluation to save VRAM)
        print("[Pruner] Loading VAE (CPU)...")
        self.vae = AutoencoderKL.from_pretrained(
            base_model_repo,
            subfolder="vae",
            torch_dtype=self.dtype
        ).to("cpu")  # CPU for VRAM efficiency
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Text Encoder (keep on CPU during transformer evaluation to save VRAM)
        print("[Pruner] Loading Text Encoder (CPU)...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            base_model_repo,
            subfolder="text_encoder",
            torch_dtype=self.dtype
        ).to("cpu")  # CPU for VRAM efficiency
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        # Tokenizer
        print("[Pruner] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_repo,
            subfolder="tokenizer"
        )

        # Scheduler
        print("[Pruner] Loading Scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_repo,
            subfolder="scheduler"
        )

        # Transformer (load from safetensors, keep on CPU initially)
        print(f"[Pruner] Loading Transformer from {self.model_path} (CPU)...")
        self.original_transformer = self._load_transformer_from_safetensors(self.model_path)
        self.original_transformer.to("cpu", dtype=self.dtype)  # CPU initially, move to GPU during evaluation
        self.original_transformer.eval()
        self.original_transformer.requires_grad_(False)

        print(f"[Pruner] Original transformer layers: {len(self.original_transformer.layers)}")
        print("[Pruner] Components loaded (VAE/TextEncoder: CPU, Transformer: will move to GPU during evaluation)")

    def _load_transformer_from_safetensors(self, model_path: str) -> ZImageTransformer2DModel:
        """
        Load ZImageTransformer2DModel from safetensors file.

        Args:
            model_path: Path to safetensors file

        Returns:
            ZImageTransformer2DModel instance
        """
        # Load state dict
        state_dict = load_file(model_path)

        # Determine model config from state dict
        # Count layers from state dict keys
        layer_indices = set()
        for key in state_dict.keys():
            if "layers." in key:
                # Extract layer index (e.g., "layers.0.attention.to_q.weight" -> 0)
                parts = key.split(".")
                if len(parts) > 1 and parts[0] == "layers":
                    try:
                        layer_idx = int(parts[1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        pass

        n_layers = max(layer_indices) + 1 if layer_indices else 30
        print(f"[Pruner] Detected {n_layers} layers from state_dict")

        # Get dim from first layer weight
        # layers.0.attention.to_q.weight: [dim, dim]
        first_layer_key = "layers.0.attention.to_q.weight"
        if first_layer_key in state_dict:
            dim = state_dict[first_layer_key].shape[0]
        else:
            dim = 3840  # Default

        print(f"[Pruner] Model dim: {dim}")

        # Create model with detected config
        transformer = ZImageTransformer2DModel(
            n_layers=n_layers,
            dim=dim,
            n_heads=30,  # Standard Z-Image config
            n_kv_heads=30,
            in_channels=16,
            all_patch_size=(2,),
            all_f_patch_size=(1,),
        )

        # Load state dict
        missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"[Pruner] WARNING: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[Pruner] WARNING: Unexpected keys: {len(unexpected_keys)}")

        return transformer

    def _prepare_samples(self):
        """Prepare sample data for evaluation."""
        print("[Pruner] Preparing samples...")

        self.prepared_samples = []

        # Move VAE and Text Encoder to GPU temporarily for encoding
        print("[Pruner] Moving VAE to GPU for encoding...")
        self.vae.to(self.device)

        print("[Pruner] Moving Text Encoder to GPU for encoding...")
        self.text_encoder.to(self.device)

        for i, sample in enumerate(tqdm(self.samples, desc="Encoding samples")):
            # Load image
            image = Image.open(sample["image_path"]).convert("RGB")
            width, height = image.size

            # Resize to multiples of 16 (VAE requirement)
            new_width = (width // 16) * 16
            new_height = (height // 16) * 16
            if new_width != width or new_height != height:
                image = image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to tensor
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            image_tensor = image_tensor.to(self.device, dtype=self.dtype)

            # Encode with VAE
            with torch.no_grad():
                latents = self.vae.encode(image_tensor).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # Encode caption
            caption = sample["caption"]
            with torch.no_grad():
                text_inputs = self.tokenizer(
                    caption,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                text_embeddings = self.text_encoder(
                    text_inputs.input_ids.to(self.device)
                ).last_hidden_state

            self.prepared_samples.append({
                "latents": latents,  # [1, C, H, W] on GPU
                "text_embeddings": text_embeddings,  # [1, seq_len, dim] on GPU
                "caption": caption,
                "image_path": sample["image_path"],
            })

        # Move VAE and Text Encoder back to CPU to free VRAM for Transformer evaluation
        print("[Pruner] Moving VAE back to CPU...")
        self.vae.to("cpu")

        print("[Pruner] Moving Text Encoder back to CPU...")
        self.text_encoder.to("cpu")

        # Clear CUDA cache
        torch.cuda.empty_cache()

        print(f"[Pruner] Prepared {len(self.prepared_samples)} samples (latents/embeddings on GPU, VAE/TextEncoder on CPU)")

    def evaluate_model(self, transformer: ZImageTransformer2DModel, timestep: int = 500) -> float:
        """
        Evaluate model with 1-step inference loss (GPU-based).

        Uses Flow Matching loss from training code:
        - v_loss: MSE between predicted velocity and actual velocity
        - recon_loss: MSE between reconstructed latents and original latents

        Args:
            transformer: Transformer model to evaluate
            timestep: Timestep for noise scheduling (default: 500, mid-point)

        Returns:
            Average reconstruction loss across all samples
        """
        # Move transformer to GPU for evaluation
        transformer.to(self.device, dtype=self.dtype)
        transformer.eval()

        total_recon_loss = 0.0

        with torch.no_grad():
            for sample in self.prepared_samples:
                latents = sample["latents"]  # [1, C, H, W]
                text_embeddings = sample["text_embeddings"]  # [1, seq_len, dim]

                # Add noise at timestep
                noise = torch.randn_like(latents)
                timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.long)

                # Flow Matching: x_t = (1-t) * noise + t * data
                noisy_latents = self.scheduler.add_noise(latents, noise, timestep_tensor)

                # Get timestep value (normalized 0-1)
                t = timestep / self.scheduler.config.num_train_timesteps  # 500 / 1000 = 0.5

                # Add frame dimension: [B, C, H, W] -> [B, C, 1, H, W]
                noisy_latents_4d = noisy_latents.unsqueeze(2)

                # Create attention mask (all True for single sample)
                batch_size = latents.shape[0]
                seq_len = text_embeddings.shape[1]
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)

                try:
                    # Predict velocity using Z-Image Transformer (GPU inference)
                    if self.dtype in [torch.float16, torch.bfloat16]:
                        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                            model_pred, _ = transformer(
                                x=noisy_latents_4d,
                                t=timestep_tensor,
                                context=text_embeddings,
                                attention_mask=attention_mask
                            )
                    else:
                        model_pred, _ = transformer(
                            x=noisy_latents_4d,
                            t=timestep_tensor,
                            context=text_embeddings,
                            attention_mask=attention_mask
                        )

                    # Remove frame dimension: [B, C, 1, H, W] -> [B, C, H, W]
                    model_pred = model_pred.squeeze(2)

                    # Flow Matching target: velocity = data - noise
                    target = latents - noise

                    # V-loss (velocity prediction loss)
                    v_loss = F.mse_loss(model_pred.float(), target.float())

                    # Reconstruction loss: x_0 = x_t + (1-t) * v_pred
                    predicted_latent = noisy_latents + (1.0 - t) * model_pred
                    recon_loss = F.mse_loss(predicted_latent.float(), latents.float())

                    # Use recon_loss as the evaluation metric (same as training)
                    total_recon_loss += recon_loss.item()

                except Exception as e:
                    print(f"[Pruner] WARNING: Forward pass failed: {e}")
                    import traceback
                    traceback.print_exc()
                    total_recon_loss += float('inf')

        # Move transformer back to CPU to free VRAM
        transformer.to("cpu")
        torch.cuda.empty_cache()

        avg_recon_loss = total_recon_loss / len(self.prepared_samples)
        return avg_recon_loss

    def create_pruned_model(self, layers_to_keep: List[int]) -> ZImageTransformer2DModel:
        """
        Create a pruned model by keeping only specified layers.

        Args:
            layers_to_keep: List of layer indices to keep (e.g., [0, 1, 3, 5, ...])

        Returns:
            Pruned ZImageTransformer2DModel
        """
        # Create new model with reduced layer count
        new_n_layers = len(layers_to_keep)
        pruned_transformer = ZImageTransformer2DModel(
            n_layers=new_n_layers,
            dim=self.original_transformer.dim,
            n_heads=self.original_transformer.n_heads,
            n_kv_heads=30,  # Standard
            in_channels=self.original_transformer.in_channels,
            all_patch_size=self.original_transformer.all_patch_size,
            all_f_patch_size=self.original_transformer.all_f_patch_size,
        )

        # Copy weights from original model
        with torch.no_grad():
            # Copy embedders
            pruned_transformer.all_x_embedder.load_state_dict(
                self.original_transformer.all_x_embedder.state_dict()
            )
            pruned_transformer.all_final_layer.load_state_dict(
                self.original_transformer.all_final_layer.state_dict()
            )
            pruned_transformer.t_embedder.load_state_dict(
                self.original_transformer.t_embedder.state_dict()
            )
            pruned_transformer.cap_embedder.load_state_dict(
                self.original_transformer.cap_embedder.state_dict()
            )
            pruned_transformer.rope_embedder.freqs_cis = self.original_transformer.rope_embedder.freqs_cis

            # Copy tokens
            pruned_transformer.x_pad_token.copy_(self.original_transformer.x_pad_token)
            pruned_transformer.cap_pad_token.copy_(self.original_transformer.cap_pad_token)

            # Copy refiner layers
            pruned_transformer.noise_refiner.load_state_dict(
                self.original_transformer.noise_refiner.state_dict()
            )
            pruned_transformer.context_refiner.load_state_dict(
                self.original_transformer.context_refiner.state_dict()
            )

            # Copy selected layers
            for new_idx, old_idx in enumerate(layers_to_keep):
                pruned_transformer.layers[new_idx].load_state_dict(
                    self.original_transformer.layers[old_idx].state_dict()
                )

        return pruned_transformer

    def greedy_prune(self, target_layers: int) -> Tuple[List[int], List[float]]:
        """
        Greedy layer pruning: iteratively remove the layer with minimal loss increase.

        Args:
            target_layers: Target number of layers after pruning

        Returns:
            Tuple of (layers_to_keep, loss_history)
        """
        original_num_layers = len(self.original_transformer.layers)
        layers_to_remove = original_num_layers - target_layers

        if layers_to_remove <= 0:
            print(f"[Pruner] No pruning needed: {original_num_layers} -> {target_layers}")
            return list(range(original_num_layers)), []

        print(f"[Pruner] Greedy pruning: {original_num_layers} -> {target_layers} ({layers_to_remove} layers to remove)")

        # Start with all layers
        current_layers = list(range(original_num_layers))
        loss_history = []

        # Calculate baseline loss (original model)
        print("[Pruner] Calculating baseline loss...")
        baseline_loss = self.evaluate_model(self.original_transformer)
        print(f"[Pruner] Baseline loss: {baseline_loss:.6f}")
        loss_history.append(baseline_loss)

        # Iteratively remove layers
        for iteration in range(layers_to_remove):
            print(f"\n[Pruner] Iteration {iteration + 1}/{layers_to_remove}")
            print(f"[Pruner] Current layers: {len(current_layers)}")

            best_layer_to_remove = None
            best_loss = float('inf')

            # Try removing each remaining layer
            for layer_idx in tqdm(current_layers, desc=f"Evaluating layers (iter {iteration + 1})"):
                # Create temporary layer list without this layer
                temp_layers = [l for l in current_layers if l != layer_idx]

                # Create pruned model
                pruned_model = self.create_pruned_model(temp_layers)
                pruned_model.to(self.device, dtype=self.dtype)
                pruned_model.eval()

                # Evaluate loss
                loss = self.evaluate_model(pruned_model)

                print(f"  Layer {layer_idx}: loss={loss:.6f} (delta={loss - baseline_loss:.6f})")

                if loss < best_loss:
                    best_loss = loss
                    best_layer_to_remove = layer_idx

                # Free memory
                del pruned_model
                torch.cuda.empty_cache()

            # Remove the best layer
            current_layers.remove(best_layer_to_remove)
            loss_history.append(best_loss)

            print(f"[Pruner] Removed layer {best_layer_to_remove}")
            print(f"[Pruner] New loss: {best_loss:.6f} (delta: {best_loss - baseline_loss:.6f})")
            print(f"[Pruner] Remaining layers: {current_layers}")

        print(f"\n[Pruner] Greedy pruning complete")
        print(f"[Pruner] Final layers: {current_layers}")
        print(f"[Pruner] Final loss: {loss_history[-1]:.6f} (delta: {loss_history[-1] - baseline_loss:.6f})")

        return current_layers, loss_history

    def uniform_prune(self, target_layers: int) -> List[int]:
        """
        Uniform pruning: remove layers at uniform intervals.

        Example: 30 -> 20 layers
        Remove every 3rd layer: keep [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28]

        Args:
            target_layers: Target number of layers

        Returns:
            List of layer indices to keep
        """
        original_num_layers = len(self.original_transformer.layers)
        layers_to_remove = original_num_layers - target_layers

        if layers_to_remove <= 0:
            return list(range(original_num_layers))

        # Calculate interval
        interval = original_num_layers / layers_to_remove

        layers_to_keep = []
        removed_count = 0

        for i in range(original_num_layers):
            # Check if this layer should be removed
            if removed_count < layers_to_remove and abs(i - removed_count * interval) < 0.5:
                removed_count += 1
                continue
            layers_to_keep.append(i)

        # Adjust to exactly target_layers
        if len(layers_to_keep) > target_layers:
            layers_to_keep = layers_to_keep[:target_layers]
        elif len(layers_to_keep) < target_layers:
            # Fill with remaining layers
            for i in range(original_num_layers):
                if i not in layers_to_keep:
                    layers_to_keep.append(i)
                    if len(layers_to_keep) == target_layers:
                        break

        layers_to_keep.sort()
        print(f"[Pruner] Uniform pruning: {original_num_layers} -> {target_layers}")
        print(f"[Pruner] Layers to keep: {layers_to_keep}")

        return layers_to_keep

    def save_pruned_model(self, layers_to_keep: List[int], output_path: str):
        """
        Save pruned model to safetensors.

        Args:
            layers_to_keep: List of layer indices to keep
            output_path: Output path for safetensors file
        """
        print(f"[Pruner] Creating pruned model with {len(layers_to_keep)} layers...")
        pruned_model = self.create_pruned_model(layers_to_keep)

        print(f"[Pruner] Saving to {output_path}...")
        print(f"[Pruner] Output dtype: {self.dtype}")
        state_dict = pruned_model.state_dict()

        # Convert to CPU and target dtype for saving
        state_dict_cpu = {k: v.cpu().to(dtype=self.dtype) for k, v in state_dict.items()}

        save_file(state_dict_cpu, output_path)
        print(f"[Pruner] Saved pruned model to {output_path}")

        # Add metadata
        metadata = {
            "original_layers": len(self.original_transformer.layers),
            "pruned_layers": len(layers_to_keep),
            "layers_kept": str(layers_to_keep),
            "original_model": self.model_path,
        }
        print(f"[Pruner] Metadata: {metadata}")


def main():
    parser = argparse.ArgumentParser(description="Layer Pruning for Z-Image Transformer")
    parser.add_argument("--model-path", type=str, required=True, help="Path to original model (safetensors)")
    parser.add_argument("--samples", type=str, required=True, help="Path to samples JSON (from extract_samples.py)")
    parser.add_argument("--target-layers", type=int, required=True, help="Target number of layers after pruning")
    parser.add_argument("--strategy", type=str, default="greedy", choices=["greedy", "uniform", "skip_middle"], help="Pruning strategy")
    parser.add_argument("--output", type=str, required=True, help="Output path for pruned model (safetensors)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Computation dtype")
    parser.add_argument("--save-dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"], help="Output dtype (default: same as --dtype)")

    args = parser.parse_args()

    print("=" * 60)
    print("Layer Pruning for Z-Image Transformer")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.samples}")
    print(f"Target layers: {args.target_layers}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Computation dtype: {args.dtype}")
    print(f"Save dtype: {args.save_dtype if args.save_dtype else args.dtype} (default)")
    print("=" * 60)

    # Check paths
    if not Path(args.model_path).exists():
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    if not Path(args.samples).exists():
        print(f"ERROR: Samples file not found: {args.samples}")
        sys.exit(1)

    # Load samples
    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"[Main] Loaded {len(samples)} samples")

    # Setup device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Save dtype (default to same as computation dtype)
    save_dtype = dtype_map[args.save_dtype] if args.save_dtype else dtype

    # Initialize pruner
    pruner = LayerPruner(
        model_path=args.model_path,
        samples=samples,
        device=device,
        dtype=save_dtype  # Use save_dtype for storage
    )

    # Execute pruning strategy
    if args.strategy == "greedy":
        layers_to_keep, loss_history = pruner.greedy_prune(args.target_layers)
    elif args.strategy == "uniform":
        layers_to_keep = pruner.uniform_prune(args.target_layers)
    elif args.strategy == "skip_middle":
        # Skip middle layers: keep first N/2 and last N/2
        half = args.target_layers // 2
        original_layers = len(pruner.original_transformer.layers)
        layers_to_keep = list(range(half)) + list(range(original_layers - half, original_layers))
        print(f"[Main] Skip middle strategy: {layers_to_keep}")
    else:
        print(f"ERROR: Unknown strategy '{args.strategy}'")
        sys.exit(1)

    # Save pruned model
    pruner.save_pruned_model(layers_to_keep, args.output)

    print("=" * 60)
    print("Layer Pruning Complete")
    print("=" * 60)
    print(f"Pruned model saved to: {args.output}")
    print(f"Original layers: {len(pruner.original_transformer.layers)}")
    print(f"Pruned layers: {len(layers_to_keep)}")
    print(f"Layers kept: {layers_to_keep}")


if __name__ == "__main__":
    main()
