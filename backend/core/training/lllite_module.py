"""
LLLite Module for ControlNet-LLLite training.

Implements the trainable LLLite module container that applies conditioning
to UNet attention layers via LoRA-style residual additions.

Compatible with kohya-ss sd-scripts checkpoint format for inference.

Architecture per attention projection (to_q, to_k, to_v):
  conditioning1: Sequential Conv2d layers (3 → cond_channels)
    - conv0: (3, cond_ch, kernel=4, stride=4)  + ReLU
    - conv2: (cond_ch, cond_ch, kernel=4, stride=4) + ReLU
    - conv4: (cond_ch, cond_ch, kernel=2, stride=2)  (no activation)
  down: Linear(hidden_dim → rank)
  mid:  Linear(cond_ch + rank → rank)
  up:   Linear(rank → hidden_dim)  (zero-initialized)

References:
- kohya-ss/sd-scripts LLLite implementation (Apache-2 license)
- SushiUI controlnet_manager.py inference code

Author: Claude (2026-01-26)
"""

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# kohya-ss cumulative index mapping for SD1.5
# Maps (block_type, block_idx, attn_idx) to kohya-ss input_blocks index
SD15_BLOCK_MAPPING = {
    ("down", 1, 0): 4,   # down_blocks[1].attentions[0] → input_blocks_4 (640ch)
    ("down", 2, 0): 7,   # down_blocks[2].attentions[0] → input_blocks_7 (1280ch)
    ("down", 2, 1): 8,   # down_blocks[2].attentions[1] → input_blocks_8 (1280ch)
}


class LLLiteConditioningEncoder(nn.Module):
    """
    Conditioning encoder for a single LLLite module.

    Processes control images through 3 convolutional layers to produce
    conditioning embeddings matching the attention layer's spatial dimensions.

    Output spatial size: H/(4*4*2) = H/32, W/32
    """

    def __init__(self, in_channels: int = 3, conditioning_channels: int = 32):
        super().__init__()
        # 3 conv layers matching kohya-ss naming: conditioning1.{0,2,4}
        # Indices skip 1,3 because kohya-ss uses Sequential with ReLU in between
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, conditioning_channels, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(conditioning_channels, conditioning_channels, kernel_size=2, stride=2, padding=0),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conditioning encoder.

        Args:
            x: Control image tensor [B, 3, H, W] normalized to [-1, 1]

        Returns:
            Conditioning embedding [B, cond_ch, H/32, W/32]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU on all layers except the last
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class LLLiteLinearLayers(nn.Module):
    """
    LoRA-style linear layers for a single LLLite module.

    Applies conditioning to hidden states via:
    1. down: Project hidden_states to lower rank
    2. mid: Process concatenated [conditioning, down(hidden_states)]
    3. up: Project back to hidden_dim (zero-initialized for stable training start)
    """

    def __init__(self, hidden_dim: int, conditioning_channels: int, rank: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank)
        self.mid = nn.Linear(conditioning_channels + rank, rank)
        self.up = nn.Linear(rank, hidden_dim)

        # Zero-initialize up projection for stable training start
        # (LLLite effect starts at zero, gradually learned)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through LoRA-style linear layers.

        Args:
            hidden_states: [B, seq_len, hidden_dim] from attention layer
            cond_embedding: [B, cond_ch, h, w] from conditioning encoder

        Returns:
            Residual to add to hidden_states [B, seq_len, hidden_dim]
        """
        # Reshape conditioning: (B, C, h, w) → (B, h*w, C)
        n, c, h, w = cond_embedding.shape
        cx = cond_embedding.view(n, c, h * w).permute(0, 2, 1)

        # Down projection on hidden states
        down_x = self.down(hidden_states)  # (B, seq_len, rank)

        # Match spatial dimensions if needed
        seq_len = hidden_states.shape[1]
        if cx.shape[1] != seq_len:
            # Interpolate conditioning to match sequence length
            cx = F.interpolate(
                cx.permute(0, 2, 1),  # (B, C, h*w)
                size=seq_len,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)  # (B, seq_len, C)

        # Match batch dimension (for CFG during sampling, not training)
        batch_size = hidden_states.shape[0]
        if cx.shape[0] != batch_size:
            cx = cx.repeat(batch_size // cx.shape[0], 1, 1)

        # Concatenate conditioning and down-projected hidden states
        cx = torch.cat([cx, down_x], dim=2)  # (B, seq_len, C + rank)

        # Mid and up projections
        cx = self.mid(cx)  # (B, seq_len, rank)
        cx = self.up(cx)   # (B, seq_len, hidden_dim)

        return cx


class LLLiteAttentionModule(nn.Module):
    """
    Complete LLLite module for a single attention projection (to_q, to_k, or to_v).

    Contains:
    - Conditioning encoder (shared Conv2d layers)
    - LoRA-style linear layers (down, mid, up)
    """

    def __init__(self, hidden_dim: int, conditioning_channels: int = 32, rank: int = 64):
        super().__init__()
        self.conditioning_encoder = LLLiteConditioningEncoder(
            in_channels=3,
            conditioning_channels=conditioning_channels,
        )
        self.linear_layers = LLLiteLinearLayers(
            hidden_dim=hidden_dim,
            conditioning_channels=conditioning_channels,
            rank=rank,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LLLite residual for an attention projection.

        Args:
            hidden_states: [B, seq_len, hidden_dim]
            condition_images: [B, 3, H, W] normalized to [-1, 1]

        Returns:
            Residual to add to hidden_states [B, seq_len, hidden_dim]
        """
        # Process control image through conditioning encoder
        cond_embedding = self.conditioning_encoder(condition_images)

        # Apply LoRA-style transformation
        residual = self.linear_layers(hidden_states, cond_embedding)

        return residual


class LLLiteModule(nn.Module):
    """
    Container for all LLLite attention modules across the UNet.

    Creates and manages LLLite modules for each compatible attention layer
    in the UNet. Handles forward patching/unpatching and checkpoint I/O
    in kohya-ss sd-scripts compatible format.

    Usage:
        # Create
        lllite = LLLiteModule.from_unet(unet, conditioning_channels=32, rank=64)

        # Training: Apply patches, run UNet, remove patches
        lllite.apply_patches(unet, condition_images)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings)
        lllite.remove_patches(unet)

        # Save kohya-ss compatible checkpoint
        state_dict = lllite.to_kohya_state_dict()
        safetensors.torch.save_file(state_dict, "lllite.safetensors")
    """

    def __init__(self):
        super().__init__()
        # Module storage: keyed by kohya-ss base name
        # e.g., "lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"
        self.lllite_modules = nn.ModuleDict()

        # Patching state
        self._patched_layers: List[Tuple[nn.Module, Any]] = []
        self._is_patched = False

    @classmethod
    def from_unet(
        cls,
        unet: nn.Module,
        conditioning_channels: int = 32,
        rank: int = 64,
        is_sdxl: bool = False,
    ) -> "LLLiteModule":
        """
        Create LLLite modules for all compatible attention layers in UNet.

        Args:
            unet: The UNet model to create LLLite modules for
            conditioning_channels: Number of channels in conditioning encoder
            rank: Rank for LoRA-style linear layers
            is_sdxl: Whether this is an SDXL UNet

        Returns:
            Initialized LLLiteModule with modules for all compatible layers
        """
        module = cls()
        block_mapping = SD15_BLOCK_MAPPING  # SDXL mapping added in Phase 3

        # Create modules for down blocks
        for (block_type, block_idx, attn_idx), kohya_idx in block_mapping.items():
            if block_type != "down":
                continue

            # Access the diffusers attention block
            try:
                attn_block = unet.down_blocks[block_idx].attentions[attn_idx]
            except (IndexError, AttributeError):
                print(f"[LLLite] Warning: down_blocks[{block_idx}].attentions[{attn_idx}] not found, skipping")
                continue

            for trans_idx, transformer_block in enumerate(attn_block.transformer_blocks):
                module._create_modules_for_transformer_block(
                    transformer_block=transformer_block,
                    kohya_prefix=f"lllite_unet_input_blocks_{kohya_idx}_1_transformer_blocks_{trans_idx}",
                    conditioning_channels=conditioning_channels,
                    rank=rank,
                )

        # Create modules for mid block
        if hasattr(unet, "mid_block") and hasattr(unet.mid_block, "attentions"):
            for attn_idx, attn_block in enumerate(unet.mid_block.attentions):
                for trans_idx, transformer_block in enumerate(attn_block.transformer_blocks):
                    module._create_modules_for_transformer_block(
                        transformer_block=transformer_block,
                        kohya_prefix=f"lllite_unet_middle_block_1_transformer_blocks_{trans_idx}",
                        conditioning_channels=conditioning_channels,
                        rank=rank,
                    )

        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[LLLite] Created {len(module.lllite_modules)} modules")
        print(f"[LLLite] Parameters: {total_params:,} total, {trainable_params:,} trainable")

        return module

    def _create_modules_for_transformer_block(
        self,
        transformer_block: nn.Module,
        kohya_prefix: str,
        conditioning_channels: int,
        rank: int,
    ):
        """Create LLLite modules for attn1 projections (to_q, to_k, to_v) in a transformer block."""
        # LLLite only patches attn1 (self-attention), not attn2 (cross-attention)
        if not hasattr(transformer_block, "attn1"):
            return

        attn = transformer_block.attn1

        for proj_name in ["to_q", "to_k", "to_v"]:
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue

            # Get hidden dimension from projection weight
            hidden_dim = proj.in_features

            # Create LLLite attention module
            base_name = f"{kohya_prefix}_attn1_{proj_name}"
            lllite_attn = LLLiteAttentionModule(
                hidden_dim=hidden_dim,
                conditioning_channels=conditioning_channels,
                rank=rank,
            )

            # Use sanitized key for ModuleDict (replace dots)
            module_key = base_name.replace(".", "_")
            self.lllite_modules[module_key] = lllite_attn

    def apply_patches(self, unet: nn.Module, condition_images: torch.Tensor):
        """
        Apply LLLite patches to UNet attention layers.

        Replaces the forward method of to_q/to_k/to_v projections
        with patched versions that add LLLite conditioning residuals.

        Args:
            unet: UNet model to patch
            condition_images: [B, 3, H, W] in [0, 1] range (will be normalized to [-1, 1])
        """
        if self._is_patched:
            raise RuntimeError("[LLLite] UNet is already patched. Call remove_patches() first.")

        # Normalize to [-1, 1]
        cond_normalized = condition_images * 2.0 - 1.0

        # Build mapping from kohya base_name to (attn_module, proj_layer)
        layer_map = self._build_unet_layer_map(unet)

        for module_key, lllite_attn in self.lllite_modules.items():
            # Convert sanitized key back to base_name
            base_name = module_key

            if base_name not in layer_map:
                continue

            proj_layer = layer_map[base_name]

            # Save original forward
            orig_forward = proj_layer.forward

            # Create patched forward closure
            # Capture lllite_attn and cond_normalized by reference
            _lllite = lllite_attn
            _cond = cond_normalized

            def make_patched_forward(original_fwd, lllite_module, cond_images):
                def patched_forward(hidden_states):
                    # Compute LLLite residual
                    residual = lllite_module(hidden_states, cond_images)
                    # Add residual to hidden states before projection
                    modified = hidden_states + residual
                    # Apply original projection
                    return original_fwd(modified)
                return patched_forward

            proj_layer.forward = make_patched_forward(orig_forward, _lllite, _cond)
            self._patched_layers.append((proj_layer, orig_forward))

        self._is_patched = True

    def remove_patches(self, unet: nn.Module):
        """
        Remove LLLite patches from UNet, restoring original forward methods.

        Args:
            unet: UNet model to unpatch (same instance used in apply_patches)
        """
        for proj_layer, orig_forward in self._patched_layers:
            proj_layer.forward = orig_forward

        self._patched_layers.clear()
        self._is_patched = False

    def _build_unet_layer_map(self, unet: nn.Module) -> Dict[str, nn.Module]:
        """
        Build a mapping from kohya-ss base names to UNet projection layers.

        Returns:
            Dict mapping base_name (e.g., "lllite_unet_input_blocks_4_1_...to_q")
            to the actual nn.Linear projection layer in UNet
        """
        layer_map = {}

        # Down blocks
        for (block_type, block_idx, attn_idx), kohya_idx in SD15_BLOCK_MAPPING.items():
            if block_type != "down":
                continue
            try:
                attn_block = unet.down_blocks[block_idx].attentions[attn_idx]
            except (IndexError, AttributeError):
                continue

            for trans_idx, transformer_block in enumerate(attn_block.transformer_blocks):
                if not hasattr(transformer_block, "attn1"):
                    continue
                attn = transformer_block.attn1
                for proj_name in ["to_q", "to_k", "to_v"]:
                    proj = getattr(attn, proj_name, None)
                    if proj is not None:
                        base_name = f"lllite_unet_input_blocks_{kohya_idx}_1_transformer_blocks_{trans_idx}_attn1_{proj_name}"
                        layer_map[base_name] = proj

        # Mid block
        if hasattr(unet, "mid_block") and hasattr(unet.mid_block, "attentions"):
            for attn_idx, attn_block in enumerate(unet.mid_block.attentions):
                for trans_idx, transformer_block in enumerate(attn_block.transformer_blocks):
                    if not hasattr(transformer_block, "attn1"):
                        continue
                    attn = transformer_block.attn1
                    for proj_name in ["to_q", "to_k", "to_v"]:
                        proj = getattr(attn, proj_name, None)
                        if proj is not None:
                            base_name = f"lllite_unet_middle_block_1_transformer_blocks_{trans_idx}_attn1_{proj_name}"
                            layer_map[base_name] = proj

        return layer_map

    def to_kohya_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Export LLLite modules as kohya-ss sd-scripts compatible state dict.

        Key format:
        {base_name}.conditioning1.{0,2,4}.{weight,bias}
        {base_name}.down.0.{weight,bias}
        {base_name}.mid.0.{weight,bias}
        {base_name}.up.0.{weight,bias}

        Returns:
            State dict compatible with kohya-ss LLLite checkpoint format
        """
        state_dict = OrderedDict()

        for module_key, lllite_attn in self.lllite_modules.items():
            base_name = module_key

            # Conditioning encoder: 3 conv layers → conditioning1.{0,2,4}
            cond_encoder = lllite_attn.conditioning_encoder
            kohya_cond_indices = [0, 2, 4]  # kohya-ss uses Sequential with ReLU interleaved
            for layer_idx, conv_layer in enumerate(cond_encoder.layers):
                kohya_idx = kohya_cond_indices[layer_idx]
                state_dict[f"{base_name}.conditioning1.{kohya_idx}.weight"] = conv_layer.weight.data
                state_dict[f"{base_name}.conditioning1.{kohya_idx}.bias"] = conv_layer.bias.data

            # Linear layers: down.0, mid.0, up.0
            linear_layers = lllite_attn.linear_layers
            state_dict[f"{base_name}.down.0.weight"] = linear_layers.down.weight.data
            state_dict[f"{base_name}.down.0.bias"] = linear_layers.down.bias.data
            state_dict[f"{base_name}.mid.0.weight"] = linear_layers.mid.weight.data
            state_dict[f"{base_name}.mid.0.bias"] = linear_layers.mid.bias.data
            state_dict[f"{base_name}.up.0.weight"] = linear_layers.up.weight.data
            state_dict[f"{base_name}.up.0.bias"] = linear_layers.up.bias.data

        return state_dict

    @classmethod
    def from_kohya_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        unet: nn.Module,
    ) -> "LLLiteModule":
        """
        Load LLLite module from kohya-ss sd-scripts compatible state dict.

        Infers conditioning_channels and rank from weight shapes.

        Args:
            state_dict: kohya-ss compatible state dict
            unet: UNet model for layer dimension reference

        Returns:
            LLLiteModule with loaded weights
        """
        module = cls()

        # Parse state dict to find unique base module names
        base_names = set()
        for key in state_dict.keys():
            # Extract base name before .conditioning1 / .down / .mid / .up
            match = re.match(r"^(.+?)\.(conditioning1|down|mid|up)\.", key)
            if match:
                base_names.add(match.group(1))

        if not base_names:
            raise ValueError("[LLLite] No valid LLLite keys found in state dict")

        # For each unique module, infer dimensions and create
        for base_name in sorted(base_names):
            # Infer conditioning channels from first conv layer
            cond_weight_key = f"{base_name}.conditioning1.0.weight"
            if cond_weight_key not in state_dict:
                print(f"[LLLite] Warning: Missing conditioning weight for {base_name}, skipping")
                continue

            conditioning_channels = state_dict[cond_weight_key].shape[0]

            # Infer rank from down layer
            down_weight_key = f"{base_name}.down.0.weight"
            if down_weight_key not in state_dict:
                print(f"[LLLite] Warning: Missing down weight for {base_name}, skipping")
                continue

            rank = state_dict[down_weight_key].shape[0]
            hidden_dim = state_dict[down_weight_key].shape[1]

            # Create module
            lllite_attn = LLLiteAttentionModule(
                hidden_dim=hidden_dim,
                conditioning_channels=conditioning_channels,
                rank=rank,
            )

            # Load conditioning encoder weights
            kohya_cond_indices = [0, 2, 4]
            for layer_idx, kohya_idx in enumerate(kohya_cond_indices):
                weight_key = f"{base_name}.conditioning1.{kohya_idx}.weight"
                bias_key = f"{base_name}.conditioning1.{kohya_idx}.bias"
                if weight_key in state_dict:
                    lllite_attn.conditioning_encoder.layers[layer_idx].weight.data.copy_(state_dict[weight_key])
                if bias_key in state_dict:
                    lllite_attn.conditioning_encoder.layers[layer_idx].bias.data.copy_(state_dict[bias_key])

            # Load linear layer weights
            for layer_name in ["down", "mid", "up"]:
                weight_key = f"{base_name}.{layer_name}.0.weight"
                bias_key = f"{base_name}.{layer_name}.0.bias"
                linear = getattr(lllite_attn.linear_layers, layer_name)
                if weight_key in state_dict:
                    linear.weight.data.copy_(state_dict[weight_key])
                if bias_key in state_dict:
                    linear.bias.data.copy_(state_dict[bias_key])

            module_key = base_name.replace(".", "_")
            module.lllite_modules[module_key] = lllite_attn

        total_params = sum(p.numel() for p in module.parameters())
        print(f"[LLLite] Loaded {len(module.lllite_modules)} modules from checkpoint")
        print(f"[LLLite] Parameters: {total_params:,}")

        return module
