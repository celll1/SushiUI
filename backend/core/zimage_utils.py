"""
Z-Image Utility Functions

This module contains utility functions extracted from the Z-Image project.
These functions are used for inference and training without external dependencies.

Original source: https://github.com/ExponentialML/Z-Image
License: Apache License 2.0

Copyright 2024 Z-Image Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Modifications made by SushiUI:
- Extracted calculate_shift function and constants for standalone use
- Extracted dispatch_attention function with multi-backend support (NATIVE, SageAttention, FlashAttention)
- Added type hints for clarity
"""

from typing import Optional
import torch
import torch.nn.functional as F

# Constants from Z-Image config/model.py
BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    """
    Calculate the dynamic shift parameter for Z-Image flow matching scheduler.

    This function computes a linear interpolation of the shift parameter based on
    the image sequence length. The shift is used to adjust the noise schedule
    for different resolution images.

    Args:
        image_seq_len: The sequence length of the image (typically (H//2) * (W//2) for Z-Image latents)
        base_seq_len: Minimum sequence length (default: 256, corresponding to small images)
        max_seq_len: Maximum sequence length (default: 4096, corresponding to large images)
        base_shift: Shift value for minimum sequence length (default: 0.5)
        max_shift: Shift value for maximum sequence length (default: 1.15)

    Returns:
        The calculated shift parameter (mu) for the scheduler

    Example:
        >>> # For a 1024x1024 image with Z-Image latent dimensions
        >>> latent_h, latent_w = 128, 128  # After VAE encoding
        >>> image_seq_len = (latent_h // 2) * (latent_w // 2)  # 64 * 64 = 4096
        >>> mu = calculate_shift(image_seq_len)
        >>> print(f"Shift parameter: {mu}")  # Will be close to 1.15 (max_shift)
    """
    # Linear interpolation: y = mx + b
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def _process_mask(attn_mask: Optional[torch.Tensor], dtype: torch.dtype):
    """
    Process attention mask for PyTorch SDPA.

    Converts bool masks to float additive masks (-inf for masked positions).
    Extracted from Z-Image utils/attention.py
    """
    if attn_mask is None:
        return None

    if attn_mask.ndim == 2:
        attn_mask = attn_mask[:, None, None, :]

    # Convert bool mask to float additive mask
    if attn_mask.dtype == torch.bool:
        new_mask = torch.zeros_like(attn_mask, dtype=dtype)
        new_mask.masked_fill_(~attn_mask, float("-inf"))
        return new_mask

    return attn_mask


def dispatch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """
    Dispatch attention computation to appropriate backend.

    Supports three backends:
    - "native" or None: PyTorch SDPA (auto Flash Attention in PyTorch 2.0+)
    - "sage": SageAttention (INT8 quantized attention for 2-5x speedup)
    - "flash": Explicit Flash Attention 2

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor [batch, seq_len_v, num_heads, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to use causal masking (default: False)
        scale: Optional scale factor for attention scores
        backend: Attention backend ("native", "sage", "flash")

    Returns:
        Attention output tensor [batch, seq_len_q, num_heads, head_dim]
    """
    backend = backend or "native"

    if backend == "sage":
        # SageAttention: INT8 quantized attention
        try:
            from sageattention import sageattn

            # SageAttention expects [batch, seq_len, num_heads, head_dim] layout
            # Z-Image already uses this layout - no transpose needed

            # Process mask if provided
            processed_mask = _process_mask(attn_mask, query.dtype) if attn_mask is not None else None

            # Call SageAttention
            # Note: SageAttention uses "HND" layout notation but expects
            # [batch, seq_len, num_heads, head_dim] tensor order
            out = sageattn(
                query, key, value,
                tensor_layout="HND",
                is_causal=is_causal,
                attn_mask=processed_mask
            )

            return out.contiguous()

        except ImportError:
            print("[Z-Image Attention] WARNING: SageAttention not available, falling back to NATIVE")
            backend = "native"
        except Exception as e:
            print(f"[Z-Image Attention] WARNING: SageAttention error: {e}, falling back to NATIVE")
            backend = "native"

    elif backend == "flash":
        # Explicit Flash Attention 2
        try:
            from flash_attn import flash_attn_func

            # Flash Attention expects [batch, seq_len, num_heads, head_dim]
            # Z-Image already uses this layout - no transpose needed

            # Flash Attention doesn't support attention mask directly
            # Only supports causal masking via is_causal parameter
            if attn_mask is not None:
                print("[Z-Image Attention] WARNING: Flash Attention does not support custom masks, ignoring mask")

            # Call Flash Attention
            out = flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                causal=is_causal,
                softmax_scale=scale
            )

            return out.contiguous()

        except ImportError:
            print("[Z-Image Attention] WARNING: Flash Attention not available, falling back to NATIVE")
            backend = "native"
        except Exception as e:
            print(f"[Z-Image Attention] WARNING: Flash Attention error: {e}, falling back to NATIVE")
            backend = "native"

    # NATIVE backend (PyTorch SDPA)
    # Transpose to [batch, num_heads, seq_len, head_dim] for PyTorch SDPA
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Process attention mask
    attn_mask = _process_mask(attn_mask, query.dtype)

    # Use PyTorch's scaled_dot_product_attention (NATIVE backend)
    # PyTorch 2.0+ automatically uses Flash Attention when available
    out = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale
    )

    # Transpose back to [batch, seq_len, num_heads, head_dim]
    return out.transpose(1, 2).contiguous()
