"""
FLUX.2 Transformer Wrapper with Block Swap Support

This wrapper intercepts the forward pass of FluxTransformer2DModel and
integrates block swapping for VRAM optimization.

Architecture:
- transformer_blocks (dual stream): FluxTransformerBlock
- single_transformer_blocks (single stream): FluxSingleTransformerBlock

The wrapper maintains compatibility with the original forward signature
while adding block swap support for inference and training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union

from diffusers.models.modeling_outputs import Transformer2DModelOutput


class Flux2BlockSwapWrapper(nn.Module):
    """
    Wrapper for FluxTransformer2DModel with Block Swap support

    This wrapper replaces the forward method with a custom implementation
    that integrates block swapping for low VRAM environments.

    Usage:
        # Create offloader
        offloader = create_flux_block_offloader(transformer, blocks_to_swap, device)

        # Wrap transformer
        wrapper = Flux2BlockSwapWrapper(transformer, offloader)

        # Use wrapper instead of transformer
        output = wrapper(hidden_states, encoder_hidden_states, ...)
    """

    def __init__(
        self,
        transformer: nn.Module,
        block_offloader: Optional["FluxBlockOffloader"] = None,
    ):
        """
        Initialize wrapper

        Args:
            transformer: FluxTransformer2DModel instance
            block_offloader: FluxBlockOffloader for block swapping (optional)
        """
        super().__init__()
        self.transformer = transformer
        self._block_offloader = block_offloader

        # Copy config for compatibility
        self.config = transformer.config

        # Copy attributes for compatibility
        self.dtype = transformer.dtype
        self.device = next(transformer.parameters()).device

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass with block swap integration

        Same signature as FluxTransformer2DModel.forward()
        """
        # If no block offloader, use original forward
        if self._block_offloader is None or self._block_offloader.blocks_to_swap == 0:
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                return_dict=return_dict,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )

        # === Custom forward with block swap ===
        # Based on diffusers Flux2Transformer2DModel.forward()
        transformer = self.transformer
        offloader = self._block_offloader

        # Handle joint_attention_kwargs
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        # FLUX.2 uses time_guidance_embed (not time_text_embed)
        temb = transformer.time_guidance_embed(timestep, guidance)

        # Get modulation parameters
        double_stream_mod_img = transformer.double_stream_modulation_img(temb)
        double_stream_mod_txt = transformer.double_stream_modulation_txt(temb)
        single_stream_mod = transformer.single_stream_modulation(temb)[0]

        # 2. Input projection for image and text
        hidden_states = transformer.x_embedder(hidden_states)
        encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

        # 3. Calculate RoPE embeddings
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        image_rotary_emb = transformer.pos_embed(img_ids)
        text_rotary_emb = transformer.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # === 4. Dual stream blocks (transformer_blocks) with block swap ===
        num_dual_blocks = offloader.num_dual_blocks
        for index_block, block in enumerate(transformer.transformer_blocks):
            # Wait for block transfer before execution
            offloader.wait_for_block(index_block)

            if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                encoder_hidden_states, hidden_states = transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_params_img=double_stream_mod_img,
                    temb_mod_params_txt=double_stream_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # Submit next block transfer after execution
            offloader.submit_move_blocks_forward(index_block)

            # ControlNet residual
            if controlnet_block_samples is not None:
                interval_control = len(transformer.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # === 5. Single stream blocks (single_transformer_blocks) with block swap ===
        for index_block, block in enumerate(transformer.single_transformer_blocks):
            # Unified index for block swap
            unified_idx = num_dual_blocks + index_block

            # Wait for block transfer before execution
            offloader.wait_for_block(unified_idx)

            if torch.is_grad_enabled() and transformer.gradient_checkpointing:
                hidden_states = transformer._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    concat_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod_params=single_stream_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # Submit next block transfer after execution
            offloader.submit_move_blocks_forward(unified_idx)

            # ControlNet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(transformer.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = transformer.norm_out(hidden_states, temb)
        output = transformer.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def to(self, *args, **kwargs):
        """Forward .to() to transformer"""
        self.transformer.to(*args, **kwargs)
        # Update device reference
        self.device = next(self.transformer.parameters()).device
        return self

    def __getattr__(self, name: str):
        """Forward attribute access to transformer"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.transformer, name)

    def state_dict(self, *args, **kwargs):
        """Forward state_dict to transformer"""
        return self.transformer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward load_state_dict to transformer"""
        return self.transformer.load_state_dict(*args, **kwargs)


def create_flux2_block_swap_wrapper(
    transformer: nn.Module,
    blocks_to_swap: int,
    device: torch.device,
    target_dtype: Optional[torch.dtype] = None,
    use_pinned_memory: bool = False,
) -> Flux2BlockSwapWrapper:
    """
    Create FLUX.2 wrapper with Block Swap support

    Args:
        transformer: FluxTransformer2DModel instance
        blocks_to_swap: Number of blocks to keep on CPU
        device: Target device
        target_dtype: Target dtype for computation
        use_pinned_memory: Use pinned memory for faster transfer

    Returns:
        Flux2BlockSwapWrapper instance with block offloader initialized
    """
    from core.memory_management import create_flux_block_offloader

    print(f"[Flux2BlockSwapWrapper] Creating wrapper with {blocks_to_swap} blocks to swap")

    # Create block offloader
    offloader = create_flux_block_offloader(
        transformer=transformer,
        blocks_to_swap=blocks_to_swap,
        device=device,
        target_dtype=target_dtype,
        use_pinned_memory=use_pinned_memory,
        supports_backward=False  # Inference mode
    )

    # Prepare block devices
    offloader.prepare_block_devices_before_forward()

    # Create wrapper
    wrapper = Flux2BlockSwapWrapper(transformer, offloader)

    print(f"[Flux2BlockSwapWrapper] Wrapper created successfully")

    return wrapper
