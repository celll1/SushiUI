"""
Batched Z-Image Transformer Wrapper

This wrapper converts batched tensor input to List[Tensor] format for compatibility
with the original Z-Image Transformer, while optimizing memory usage and reducing
overhead from List operations.

Key optimizations:
1. Batched patchify: Process all images at once without loops
2. Minimal List conversions: Only at input/output boundaries
3. Memory-efficient padding: Pre-allocate padded tensors
4. Preserve memory contiguity where possible

This significantly reduces VRAM usage during training with batch_size > 1.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class BatchedZImageWrapper(nn.Module):
    """
    Wrapper around Z-Image Transformer to accept batched tensor input.

    This wrapper handles the conversion between batched tensors and List[Tensor]
    format required by the Z-Image Transformer, while minimizing memory overhead.

    Input:
        x: [B, C, F, H, W] - Batched latents
        t: [B] - Timesteps
        cap_feats: [B, max_seq_len, 2560] - Batched caption embeddings (pre-padded)
        cap_mask: [B, max_seq_len] - Caption attention mask (True = valid token)

    Output:
        [B, C, F, H, W] - Batched latents
    """

    def __init__(self, transformer):
        """
        Args:
            transformer: ZImageTransformer2DModel instance
        """
        super().__init__()
        # Store wrapped transformer - use direct assignment, not add_module()
        # nn.Module will automatically register it in _modules
        self.transformer = transformer

        # Copy attributes from transformer for easy access
        self.in_channels = transformer.in_channels
        self.out_channels = transformer.out_channels
        self.dim = transformer.dim
        self.all_patch_size = transformer.all_patch_size
        self.all_f_patch_size = transformer.all_f_patch_size

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self.transformer.enable_gradient_checkpointing()

    @property
    def gradient_checkpointing(self):
        """Expose gradient_checkpointing attribute from wrapped transformer."""
        return self.transformer.gradient_checkpointing

    def train(self, mode=True):
        """Set training mode for both wrapper and wrapped transformer."""
        super().train(mode)
        self.transformer.train(mode)
        return self

    def eval(self):
        """Set evaluation mode for both wrapper and wrapped transformer."""
        super().eval()
        self.transformer.eval()
        return self

    def batched_to_list(
        self,
        x: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert batched tensors to List[Tensor] format.

        This is a lightweight conversion that preserves memory where possible.

        Args:
            x: [B, C, F, H, W] - Batched latents
            cap_feats: [B, max_seq_len, 2560] - Batched caption embeddings
            cap_mask: [B, max_seq_len] - Caption mask (True = valid)

        Returns:
            x_list: List of [C, F, H, W] tensors
            cap_list: List of [seq_len, 2560] tensors (without padding)
        """
        B = x.shape[0]

        # Convert x to list (simple unbind, no copy)
        x_list = list(x.unbind(dim=0))  # List of [C, F, H, W]

        # Convert cap_feats to list, removing padding
        cap_list = []
        for i in range(B):
            # Extract valid tokens only (remove padding)
            valid_len = cap_mask[i].sum().item()
            cap_list.append(cap_feats[i, :valid_len])  # [seq_len, 2560]

        return x_list, cap_list

    def list_to_batched(
        self,
        x_list: List[torch.Tensor],
        original_sizes: List[Tuple[int, int, int, int]]
    ) -> torch.Tensor:
        """
        Convert List[Tensor] output back to batched tensor.

        Args:
            x_list: List of [C, F, H, W] tensors
            original_sizes: List of (C, F, H, W) tuples (original sizes before padding)

        Returns:
            [B, C, F, H, W] - Batched tensor
        """
        B = len(x_list)

        # Stack into batched tensor
        # All tensors in x_list should have same shape after unpatchify
        x_batched = torch.stack(x_list, dim=0)  # [B, C, F, H, W]

        return x_batched

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with batched tensor input.

        Args:
            x: [B, C, F, H, W] - Batched latents
            t: [B] - Timesteps
            cap_feats: [B, max_seq_len, 2560] - Batched caption embeddings (pre-padded)
            cap_mask: [B, max_seq_len] - Caption mask (True = valid token)
            patch_size: Spatial patch size (default: 2)
            f_patch_size: Temporal/frame patch size (default: 1)

        Returns:
            output: [B, C, F, H, W] - Batched output latents
            info: dict - Additional information
        """
        B, C, F, H, W = x.shape

        # Store original sizes for reconstruction
        original_sizes = [(C, F, H, W) for _ in range(B)]

        # Convert batched tensors to List format (minimal conversion)
        x_list, cap_list = self.batched_to_list(x, cap_feats, cap_mask)

        # Call original Z-Image Transformer
        output_list, info = self.transformer(
            x=x_list,
            t=t,
            cap_feats=cap_list,
            patch_size=patch_size,
            f_patch_size=f_patch_size
        )

        # Convert List output back to batched tensor
        output_batched = self.list_to_batched(output_list, original_sizes)

        return output_batched, info


class BatchedZImageWrapperOptimized(BatchedZImageWrapper):
    """
    Fully optimized BatchedZImageWrapper with complete batched processing.

    This version eliminates ALL List[Tensor] loops by implementing:
    1. Batched patchify (no loops)
    2. Batched unpatchify (no loops)
    3. Direct batched tensor processing throughout

    This achieves significant VRAM reduction by:
    - Eliminating memory copies from List conversions
    - Enabling better PyTorch kernel fusion
    - Improving gradient checkpointing efficiency
    """

    def __init__(self, transformer):
        super().__init__(transformer)
        # SEQ_MULTI_OF constant from Z-Image (sequences must be multiple of this)
        self.SEQ_MULTI_OF = 64  # From config.py

    def batched_patchify(
        self,
        x: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
    ):
        """
        Batched patchify without any loops.

        Args:
            x: [B, C, F, H, W] - Batched latents
            cap_feats: [B, max_cap_len, 2560] - Batched caption features (pre-padded)
            cap_mask: [B, max_cap_len] - Caption mask (True = valid)
            patch_size: Spatial patch size (pH, pW)
            f_patch_size: Temporal patch size (pF)

        Returns:
            x_patches: [B, num_patches_padded, patch_dim] - Batched image patches
            cap_feats_padded: [B, cap_padded_len, 2560] - Caption features with SEQ_MULTI_OF padding
            sizes: [B, 3] - Original (F, H, W) for each batch item
            x_seq_lens: [B] - Sequence length for each image (after padding)
            cap_seq_lens: [B] - Sequence length for each caption (after padding)
        """
        B, C, F, H, W = x.shape
        pH = pW = patch_size
        pF = f_patch_size
        device = x.device

        # Calculate tokens
        F_tokens = F // pF
        H_tokens = H // pH
        W_tokens = W // pW
        num_tokens = F_tokens * H_tokens * W_tokens

        # Batched patchify: [B, C, F, H, W] -> [B, num_tokens, patch_dim]
        # Reshape: [B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        x = x.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        # Permute: [B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C]
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
        # Reshape: [B, num_tokens, pF * pH * pW * C]
        patch_dim = pF * pH * pW * C
        x_patches = x.reshape(B, num_tokens, patch_dim)

        # Calculate padding for SEQ_MULTI_OF alignment
        x_padding_lens = (-num_tokens) % self.SEQ_MULTI_OF  # scalar (same for all batch items)
        x_padded_len = num_tokens + x_padding_lens

        # Pad image patches if needed
        if x_padding_lens > 0:
            # Repeat last patch for padding: [B, x_padding_lens, patch_dim]
            pad_patches = x_patches[:, -1:, :].expand(B, x_padding_lens, patch_dim)
            x_patches = torch.cat([x_patches, pad_patches], dim=1)

        # Caption padding for SEQ_MULTI_OF alignment
        cap_ori_lens = cap_mask.sum(dim=1)  # [B] - number of valid tokens per batch
        max_cap_ori_len = cap_ori_lens.max().item()

        # Trim cap_feats to max valid length
        cap_feats_trimmed = cap_feats[:, :max_cap_ori_len, :]  # [B, max_cap_ori_len, 2560]

        # Pad to next multiple of SEQ_MULTI_OF
        cap_padded_len = ((max_cap_ori_len + self.SEQ_MULTI_OF - 1) // self.SEQ_MULTI_OF) * self.SEQ_MULTI_OF
        cap_padding_len = cap_padded_len - max_cap_ori_len

        if cap_padding_len > 0:
            # For each batch item, repeat last valid token
            cap_last_valid_indices = (cap_ori_lens - 1).clamp(min=0).long()  # [B]
            cap_last_tokens = cap_feats_trimmed[torch.arange(B, device=device), cap_last_valid_indices]  # [B, 2560]
            cap_pad_tokens = cap_last_tokens.unsqueeze(1).expand(B, cap_padding_len, 2560)
            cap_feats_padded = torch.cat([cap_feats_trimmed, cap_pad_tokens], dim=1)
        else:
            cap_feats_padded = cap_feats_trimmed

        # Store metadata
        sizes = torch.tensor([[F, H, W] for _ in range(B)], dtype=torch.long, device=device)  # [B, 3]
        x_seq_lens = torch.full((B,), x_padded_len, dtype=torch.long, device=device)
        cap_seq_lens = torch.full((B,), cap_padded_len, dtype=torch.long, device=device)

        return x_patches, cap_feats_padded, sizes, x_seq_lens, cap_seq_lens

    def batched_unpatchify(
        self,
        x_patches: torch.Tensor,
        sizes: torch.Tensor,
        patch_size: int,
        f_patch_size: int,
    ) -> torch.Tensor:
        """
        Batched unpatchify without any loops.

        Args:
            x_patches: [B, num_patches_padded, patch_dim] - Batched patches (may include padding)
            sizes: [B, 3] - Original (F, H, W) for each batch item
            patch_size: Spatial patch size (pH, pW)
            f_patch_size: Temporal patch size (pF)

        Returns:
            x: [B, C, F, H, W] - Reconstructed latents
        """
        B = x_patches.shape[0]
        pH = pW = patch_size
        pF = f_patch_size
        C = self.out_channels

        # Extract size (assume all batch items have same size for training)
        F, H, W = sizes[0].tolist()
        F_tokens = F // pF
        H_tokens = H // pH
        W_tokens = W // pW
        num_tokens = F_tokens * H_tokens * W_tokens

        # Remove padding: [B, num_patches_padded, patch_dim] -> [B, num_tokens, patch_dim]
        x_patches = x_patches[:, :num_tokens, :]

        # Reshape: [B, num_tokens, pF * pH * pW * C] -> [B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C]
        x = x_patches.view(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C)

        # Permute: [B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)

        # Reshape: [B, C, F, H, W]
        x = x.reshape(B, C, F, H, W)

        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Fully batched forward pass - bypasses Transformer's List[Tensor] processing entirely.

        This method reimplements the entire Transformer forward pass using only batched tensors,
        eliminating ALL List operations and loops for maximum memory efficiency.

        Args:
            x: [B, C, F, H, W] - Batched latents
            t: [B] - Timesteps
            cap_feats: [B, max_seq_len, 2560] - Batched caption embeddings (pre-padded)
            cap_mask: [B, max_seq_len] - Caption mask (True = valid token)
            patch_size: Spatial patch size (default: 2)
            f_patch_size: Temporal/frame patch size (default: 1)

        Returns:
            output: [B, C, F, H, W] - Batched output latents
            info: dict - Additional information
        """
        B, C, F, H, W = x.shape
        device = x.device

        # Step 1: Batched patchify (NO LOOPS)
        x_patches, cap_feats_padded, sizes, x_seq_lens, cap_seq_lens = self.batched_patchify(
            x, cap_feats, cap_mask, patch_size, f_patch_size
        )
        # x_patches: [B, x_padded_len, patch_dim]
        # cap_feats_padded: [B, cap_padded_len, 2560]

        # Step 2: Timestep embedding
        t_scaled = t * self.transformer.t_scale
        t_emb = self.transformer.t_embedder(t_scaled)  # [B, dim]
        adaln_input = t_emb.type_as(x_patches)

        # Step 3: Image patch embedding
        x_embedded = self.transformer.all_x_embedder[f"{patch_size}-{f_patch_size}"](x_patches)
        # x_embedded: [B, x_padded_len, dim]

        # Step 4: Position IDs and RoPE for image patches (BATCHED)
        x_padded_len = x_seq_lens[0].item()  # Same for all batch items
        cap_padded_len = cap_seq_lens[0].item()

        # Create position IDs: [B, x_padded_len, 3]
        F_tokens, H_tokens, W_tokens = F // f_patch_size, H // patch_size, W // patch_size
        x_pos_ids = self.transformer.create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens),
            start=(cap_padded_len + 1, 0, 0),
            device=device
        ).flatten(0, 2)  # [F_tokens*H_tokens*W_tokens, 3]

        # Pad to x_padded_len
        num_tokens = F_tokens * H_tokens * W_tokens
        if x_padded_len > num_tokens:
            padding_pos_ids = self.transformer.create_coordinate_grid(
                size=(1, 1, 1), start=(0, 0, 0), device=device
            ).flatten(0, 2).repeat(x_padded_len - num_tokens, 1)
            x_pos_ids = torch.cat([x_pos_ids, padding_pos_ids], dim=0)

        # Expand to batch: [B, x_padded_len, 3]
        x_pos_ids = x_pos_ids.unsqueeze(0).expand(B, -1, -1)

        # Apply RoPE
        x_freqs_cis = self.transformer.rope_embedder(x_pos_ids.reshape(B * x_padded_len, 3))
        x_freqs_cis = x_freqs_cis.reshape(B, x_padded_len, -1)  # [B, x_padded_len, rope_dim]

        # Create attention mask for image patches
        x_attn_mask = torch.ones((B, x_padded_len), dtype=torch.bool, device=device)

        # Step 5: Apply noise_refiner layers (image processing)
        for layer in self.transformer.noise_refiner:
            if self.transformer.gradient_checkpointing and self.training:
                x_embedded = torch.utils.checkpoint.checkpoint(
                    layer,
                    x_embedded,
                    x_attn_mask,
                    x_freqs_cis,
                    adaln_input,
                    use_reentrant=False
                )
            else:
                x_embedded = layer(x_embedded, x_attn_mask, x_freqs_cis, adaln_input)

        # Step 6: Caption embedding
        cap_embedded = self.transformer.cap_embedder(cap_feats_padded)
        # cap_embedded: [B, cap_padded_len, dim]

        # Step 7: Position IDs and RoPE for captions (BATCHED)
        cap_pos_ids = self.transformer.create_coordinate_grid(
            size=(cap_padded_len, 1, 1),
            start=(1, 0, 0),
            device=device
        ).flatten(0, 2)  # [cap_padded_len, 3]
        cap_pos_ids = cap_pos_ids.unsqueeze(0).expand(B, -1, -1)  # [B, cap_padded_len, 3]

        cap_freqs_cis = self.transformer.rope_embedder(cap_pos_ids.reshape(B * cap_padded_len, 3))
        cap_freqs_cis = cap_freqs_cis.reshape(B, cap_padded_len, -1)

        # Create attention mask for captions (based on cap_mask)
        cap_attn_mask = cap_mask[:, :cap_padded_len]  # [B, cap_padded_len]

        # Step 8: Apply context_refiner layers (caption processing)
        for layer in self.transformer.context_refiner:
            if self.transformer.gradient_checkpointing and self.training:
                cap_embedded = torch.utils.checkpoint.checkpoint(
                    layer,
                    cap_embedded,
                    cap_attn_mask,
                    cap_freqs_cis,
                    use_reentrant=False
                )
            else:
                cap_embedded = layer(cap_embedded, cap_attn_mask, cap_freqs_cis)

        # Step 9: Unify image and caption sequences (BATCHED, NO LOOPS)
        # Concatenate along sequence dimension
        unified = torch.cat([x_embedded, cap_embedded], dim=1)  # [B, x_padded_len + cap_padded_len, dim]
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)  # [B, x_padded_len + cap_padded_len, rope_dim]
        unified_attn_mask = torch.cat([x_attn_mask, cap_attn_mask], dim=1)  # [B, x_padded_len + cap_padded_len]

        # Step 10: Apply main transformer layers
        for layer_idx, layer in enumerate(self.transformer.layers):
            # Block Swap integration
            if hasattr(self.transformer, '_block_offloader') and self.transformer._block_offloader is not None:
                self.transformer._block_offloader.wait_for_block(layer_idx)

            if self.transformer.gradient_checkpointing and self.training:
                unified = torch.utils.checkpoint.checkpoint(
                    layer,
                    unified,
                    unified_attn_mask,
                    unified_freqs_cis,
                    adaln_input,
                    use_reentrant=False
                )
            else:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

            # Block Swap integration
            if hasattr(self.transformer, '_block_offloader') and self.transformer._block_offloader is not None:
                self.transformer._block_offloader.submit_move_blocks(layer_idx)

        # Step 11: Final layer
        unified = self.transformer.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)

        # Step 12: Extract image patches (discard caption part)
        x_output = unified[:, :x_padded_len, :]  # [B, x_padded_len, patch_dim]

        # Step 13: Batched unpatchify (NO LOOPS)
        output = self.batched_unpatchify(x_output, sizes, patch_size, f_patch_size)

        return output, {}
