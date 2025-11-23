"""Normalized Attention Guidance (NAG) for diffusion models

NAG improves negative prompting by applying guidance in attention space rather than noise prediction space.
This is particularly effective for few-step sampling scenarios.

Based on:
- Paper: https://chendaryen.github.io/NAG.github.io/
- ComfyUI Implementation: https://github.com/ChenDarYen/ComfyUI-NAG (MIT License)

Key Concepts:
- Extrapolation: Z_guide = Z_pos * scale - Z_neg * (scale - 1)
- L1 Normalization: ||Z_guide||_L1 ≤ tau * ||Z_pos||_L1
- Alpha Blending: Z_final = alpha * Z_guide + (1 - alpha) * Z_pos
"""

import torch
from typing import Optional


def nag(
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    scale: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """Apply Normalized Attention Guidance to attention outputs

    Args:
        z_positive: Attention output from positive prompt [B, N, C]
        z_negative: Attention output from negative prompt [B, N, C]
        scale: Extrapolation scale (similar to CFG scale, typical: 3-7)
        tau: Normalization threshold (typical: 2.5-3.5)
        alpha: Blending factor between guided and original (typical: 0.25-0.5)

    Returns:
        Guided attention output [B, N, C]
    """
    m = min(z_positive.shape[0], z_negative.shape[0])
    if m == 0:
        return z_positive

    z_positive = z_positive[-m:]
    z_negative = z_negative[-m:]

    # Step 1: Extrapolation (similar to CFG)
    # z_guide = z_pos + scale * (z_pos - z_neg)
    #         = z_pos * scale - z_neg * (scale - 1)
    z_guidance = z_positive * scale - z_negative * (scale - 1)

    # Step 2: L1 Normalization to prevent out-of-manifold drift
    eps = 1e-6
    norm_positive = (
        torch.norm(z_positive, p=1, dim=-1, keepdim=True)
        .clamp_min(eps)
        .expand_as(z_positive)
    )
    norm_guidance = (
        torch.norm(z_guidance, p=1, dim=-1, keepdim=True)
        .clamp_min(eps)
        .expand_as(z_guidance)
    )

    # Compute scaling factor: s = ||z_guide|| / ||z_pos||
    s = norm_guidance / norm_positive

    # Clamp to tau: if s > tau, rescale z_guide to have norm = tau * ||z_pos||
    # z_guide_normalized = z_guide * min(s, tau) / s
    z_guidance = z_guidance * torch.minimum(s, s.new_full((1,), tau)) / s

    # Step 3: Alpha blending between guided and original
    # This provides smooth interpolation and stability
    z_guidance = z_guidance * alpha + z_positive * (1 - alpha)

    return z_guidance


def check_nag_activation(sigma: float, nag_sigma_end: float) -> bool:
    """Check if NAG should be active at the current noise level

    NAG is typically only applied during early/mid denoising steps where
    it has the most impact. This saves computation in later steps.

    Args:
        sigma: Current noise level
        nag_sigma_end: Sigma threshold below which NAG is disabled

    Returns:
        True if NAG should be active
    """
    return sigma >= nag_sigma_end


def prepare_nag_context(
    encoder_hidden_states: torch.Tensor,
    nag_negative_embeds: Optional[torch.Tensor],
    apply_nag: bool = True,
) -> torch.Tensor:
    """Prepare context for NAG by concatenating negative embeddings

    Args:
        encoder_hidden_states: Original context [B, seq_len, dim] or [B, dim] for pooled
        nag_negative_embeds: NAG negative prompt embeddings [1, seq_len_nag, dim] or [1, dim] for pooled
        apply_nag: Whether NAG is active this step

    Returns:
        Extended context: [positive, nag_negative] if apply_nag, else original
    """
    if not apply_nag or nag_negative_embeds is None:
        return encoder_hidden_states

    batch_size = encoder_hidden_states.shape[0]

    # Check if this is pooled embeddings (2D) or sequence embeddings (3D)
    if encoder_hidden_states.dim() == 2:
        # Pooled embeddings: [B, dim]
        # Simply expand nag_negative_embeds to match batch size and concatenate
        if nag_negative_embeds.shape[0] != batch_size:
            nag_negative_embeds = nag_negative_embeds.expand(batch_size, -1)

        # Concatenate: [positive, nag_negative]
        return torch.cat([encoder_hidden_states, nag_negative_embeds], dim=0)

    else:
        # Sequence embeddings: [B, seq_len, dim]
        target_seq_len = encoder_hidden_states.shape[1]
        nag_seq_len = nag_negative_embeds.shape[1]

        # Expand batch dimension if needed
        if nag_negative_embeds.shape[0] != batch_size:
            nag_negative_embeds = nag_negative_embeds.expand(batch_size, -1, -1)

        # Pad or truncate NAG embeddings to match sequence length
        if nag_seq_len != target_seq_len:
            if nag_seq_len < target_seq_len:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size,
                    target_seq_len - nag_seq_len,
                    nag_negative_embeds.shape[2],
                    device=nag_negative_embeds.device,
                    dtype=nag_negative_embeds.dtype
                )
                nag_negative_embeds = torch.cat([nag_negative_embeds, padding], dim=1)
            else:
                # Truncate
                nag_negative_embeds = nag_negative_embeds[:, :target_seq_len, :]

        # Concatenate: [positive_context, nag_negative_context]
        return torch.cat([encoder_hidden_states, nag_negative_embeds], dim=0)


def split_nag_context_output(
    hidden_states: torch.Tensor,
    apply_nag: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Split attention output back into positive and NAG negative parts

    Args:
        hidden_states: Combined output [positive + nag_negative, ...]
        apply_nag: Whether NAG was active

    Returns:
        (positive_output, nag_negative_output) or (output, None) if not apply_nag
    """
    if not apply_nag:
        return hidden_states, None

    # Split in half: first half is positive, second half is nag_negative
    batch_size = hidden_states.shape[0]
    half = batch_size // 2

    positive_output = hidden_states[:half]
    nag_negative_output = hidden_states[half:]

    return positive_output, nag_negative_output
