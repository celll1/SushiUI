"""
SigLIP2 Vision Encoder Wrapper for SDXL/SD1.5 integration.

Loads a Siglip2VisionModel from safetensors, encodes reference images,
and zero-pads the output to match the combined text embedding dimension
(2048 for SDXL, 768 for SD1.5).

Usage in generation:
    ve = SigLIP2VisionEncoderWrapper("path/to/siglip2_base_vision_encoder.safetensors")
    ve.to("cuda")
    ve_pos, ve_neg = ve.encode([pil_image1, pil_image2], target_dim=2048)
    # ve_pos: [B, 1+256*N, 2048]  (header + vision patches, zero-padded)
    # ve_neg: zeros of same shape  (used for CFG negative)
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image


# Known configs keyed by hidden_size
_KNOWN_CONFIGS = {
    768: {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 16,
        "num_channels": 3,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "model_type": "siglip2_vision_model",
        # processor repo (used to create Siglip2ImageProcessor)
        "processor_repo": "google/siglip2-base-patch16-naflex",
    },
    1152: {
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "patch_size": 16,
        "num_channels": 3,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "model_type": "siglip2_vision_model",
        "processor_repo": "google/siglip2-so400m-patch16-naflex",
    },
}


def _infer_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Infer Siglip2VisionConfig parameters from state_dict tensor shapes."""
    # patch_embedding.weight: [hidden_size, patch_size^2 * num_channels]
    pe_weight = state_dict.get("embeddings.patch_embedding.weight")
    if pe_weight is None:
        # Try with vision_model prefix
        pe_weight = state_dict.get("vision_model.embeddings.patch_embedding.weight")
    if pe_weight is None:
        raise ValueError("Cannot find 'embeddings.patch_embedding.weight' in state_dict")

    hidden_size = pe_weight.shape[0]

    if hidden_size in _KNOWN_CONFIGS:
        cfg = dict(_KNOWN_CONFIGS[hidden_size])
        # Verify num_hidden_layers from actual keys
        layer_indices = set()
        for k in state_dict:
            parts = k.split(".")
            if "encoder" in parts and "layers" in parts:
                idx = parts[parts.index("layers") + 1]
                if idx.isdigit():
                    layer_indices.add(int(idx))
        if layer_indices:
            cfg["num_hidden_layers"] = max(layer_indices) + 1
        return cfg

    # Fallback: infer from shapes
    layer_indices = set()
    intermediate_size = None
    for k, v in state_dict.items():
        parts = k.split(".")
        if "encoder" in parts and "layers" in parts:
            try:
                idx = int(parts[parts.index("layers") + 1])
                layer_indices.add(idx)
            except (ValueError, IndexError):
                pass
        if "mlp.fc1.weight" in k and intermediate_size is None:
            intermediate_size = v.shape[0]

    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 12
    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    # Infer num_attention_heads: head_dim is typically 64 or 72
    # Try common values
    for head_dim in [64, 72, 96, 128]:
        if hidden_size % head_dim == 0:
            num_attention_heads = hidden_size // head_dim
            break
    else:
        num_attention_heads = 12

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "patch_size": 16,
        "num_channels": 3,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "model_type": "siglip2_vision_model",
        "processor_repo": "google/siglip2-base-patch16-naflex",
    }


class SigLIP2VisionEncoderWrapper:
    """
    Wraps Siglip2VisionModel loaded from a safetensors checkpoint.

    The safetensors file may optionally contain a 'header_token' tensor
    (shape [1, 1, MAX_DIM]) for use as a learnable boundary marker between
    text and vision embeddings. If absent, it is zero-initialized.
    """

    HEADER_KEY = "header_token"

    def __init__(self, safetensors_path: str, device: str = "cpu"):
        from safetensors.torch import load_file
        from transformers import Siglip2VisionModel, Siglip2VisionConfig, AutoProcessor

        self.safetensors_path = safetensors_path
        self.device = device

        print(f"[VisionEncoder] Loading from: {safetensors_path}")

        # Load raw state dict
        raw_sd = load_file(safetensors_path, device="cpu")

        # Separate header_token from model weights
        header_tensor = raw_sd.pop(self.HEADER_KEY, None)

        # Infer config
        cfg_dict = _infer_config_from_state_dict(raw_sd)
        self.hidden_size = cfg_dict["hidden_size"]
        processor_repo = cfg_dict.pop("processor_repo")
        cfg_dict.pop("model_type", None)

        print(f"[VisionEncoder] Config: hidden_size={self.hidden_size}, "
              f"layers={cfg_dict['num_hidden_layers']}, heads={cfg_dict['num_attention_heads']}")

        # Build Siglip2VisionConfig and model
        config = Siglip2VisionConfig(**cfg_dict)
        self.model = Siglip2VisionModel(config)

        # Load weights (strict=False to tolerate missing header_token)
        missing, unexpected = self.model.load_state_dict(raw_sd, strict=False)
        if missing:
            print(f"[VisionEncoder] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[VisionEncoder] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        self.model.eval()
        self.model.to(device)

        # Header token: learnable scalar zero-init tensor (dim set at encode time)
        # Store as a raw tensor; will be sliced/expanded at encode time.
        # Shape: [1, 1, hidden_size] — projected to target_dim via zero-pad on first use
        if header_tensor is not None:
            self.header_token = nn.Parameter(header_tensor.to(device))
        else:
            self.header_token = nn.Parameter(
                torch.zeros(1, 1, self.hidden_size, device=device)
            )

        # Load processor (uses HuggingFace cache)
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(processor_repo)
            print(f"[VisionEncoder] Processor loaded from '{processor_repo}'")
        except Exception as e:
            print(f"[VisionEncoder] Warning: Could not load processor from '{processor_repo}': {e}")
            print(f"[VisionEncoder] Falling back to google/siglip2-base-patch16-naflex")
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[VisionEncoder] Loaded successfully. Params: {total_params/1e6:.1f}M")

    def to(self, device: str) -> "SigLIP2VisionEncoderWrapper":
        self.device = device
        self.model.to(device)
        self.header_token.data = self.header_token.data.to(device)
        return self

    def encode(
        self,
        images: List[Image.Image],
        target_dim: int = 2048,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of reference images into vision embeddings.

        Each image produces 256 patch tokens (NaFlex default).
        The header_token is prepended once per batch (not per image).
        Multiple images are concatenated along the sequence dimension.

        Args:
            images: List of PIL Images (1 to N)
            target_dim: Target embedding dimension to pad to (2048 for SDXL, 768 for SD1.5)
            dtype: Output dtype (match the U-Net dtype)

        Returns:
            pos_embeds: [1, 1 + 256*N, target_dim]  (header + vision tokens, zero-padded)
            neg_embeds: [1, 1 + 256*N, target_dim]  (all zeros, used for CFG negative)
        """
        if not images:
            raise ValueError("At least one image is required for vision encoder")

        self.model.eval()
        all_patch_embeds = []

        with torch.no_grad():
            for img in images:
                if not isinstance(img, Image.Image):
                    raise TypeError(f"Expected PIL Image, got {type(img)}")
                rgb_img = img.convert("RGB")

                # Process image → NaFlex inputs
                inputs = self.processor(images=[rgb_img], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward through vision model
                out = self.model(**inputs)
                # last_hidden_state: [1, 256, hidden_size]
                patch_embeds = out.last_hidden_state.to(dtype)

                all_patch_embeds.append(patch_embeds)  # list of [1, 256, D]

        # Concatenate multiple images: [1, 256*N, hidden_size]
        if len(all_patch_embeds) == 1:
            combined_patches = all_patch_embeds[0]
        else:
            combined_patches = torch.cat(all_patch_embeds, dim=1)

        N = combined_patches.shape[1]  # 256 * num_images

        # Zero-pad hidden_size → target_dim
        if self.hidden_size < target_dim:
            pad_size = target_dim - self.hidden_size
            combined_patches = F.pad(combined_patches, (0, pad_size))
        elif self.hidden_size > target_dim:
            # Unlikely (hidden > target), but truncate if needed
            combined_patches = combined_patches[..., :target_dim]
        # If equal, no change needed (e.g. SD1.5 + base VE)

        # Prepare header token [1, 1, hidden_size] → zero-pad → [1, 1, target_dim]
        header = self.header_token.to(dtype).to(self.device)  # [1, 1, hidden_size]
        if self.hidden_size < target_dim:
            header = F.pad(header, (0, target_dim - self.hidden_size))
        elif self.hidden_size > target_dim:
            header = header[..., :target_dim]

        # Concatenate: [1, 1+256*N, target_dim]
        pos_embeds = torch.cat([header, combined_patches], dim=1)

        # Negative: all zeros (model should learn to ignore absent vision info)
        neg_embeds = torch.zeros_like(pos_embeds)

        print(f"[VisionEncoder] Encoded {len(images)} image(s) → "
              f"pos_embeds={list(pos_embeds.shape)}, neg_embeds={list(neg_embeds.shape)}")

        return pos_embeds, neg_embeds

    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        """
        Returns a state dict suitable for saving with safetensors.
        Includes both the vision model weights and the (optionally trained)
        header_token as separate tensors.
        """
        sd = {k: v.cpu().contiguous() for k, v in self.model.state_dict().items()}
        sd[self.HEADER_KEY] = self.header_token.data.cpu().contiguous()
        return sd

    def parameters(self):
        """Yield all trainable parameters (vision model + header_token)."""
        yield from self.model.parameters()
        yield self.header_token

    def named_parameters(self, prefix="vision_encoder"):
        for name, param in self.model.named_parameters():
            yield f"{prefix}.{name}", param
        yield f"{prefix}.{self.HEADER_KEY}", self.header_token

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __repr__(self):
        return (
            f"SigLIP2VisionEncoderWrapper("
            f"hidden_size={self.hidden_size}, "
            f"layers={self.model.config.num_hidden_layers}, "
            f"device={self.device})"
        )
