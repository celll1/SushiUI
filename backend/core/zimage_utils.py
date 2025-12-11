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
- Added type hints for clarity
"""

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
