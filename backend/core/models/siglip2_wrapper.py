"""
SigLIP-2 Text/Image Encoder Wrapper

Wraps google/siglip2-so400m-patch16-naflex for text and image encoding.
SigLIP-2 has no token limit and supports variable-length inputs.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from typing import Optional, List, Union, Any
from PIL import Image
import time


class SigLIP2TextEncoder(nn.Module):
    """
    SigLIP-2 Text Encoder wrapper.

    Features:
    - No token limit (NaViT/NAFlex architecture)
    - 1152 hidden dimension
    - 27 transformer layers
    - Supports variable-length text inputs
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-naflex",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_from_checkpoint: bool = False,
        shared_config: Optional[Any] = None,
        max_position_embeddings: Optional[int] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device_name = device

        if load_from_checkpoint:
            # Create empty model structure (weights will be loaded via load_state_dict)
            print(f"[SigLIP2] Creating text encoder structure (loading from checkpoint)...")

            # Load config only (no weights) - reuse shared config if provided
            from transformers import AutoConfig
            if shared_config is not None:
                print(f"[SigLIP2] Reusing shared config (skipping download)...")
                config = shared_config
            else:
                start_time = time.time()
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                config_time = time.time() - start_time
                print(f"[SigLIP2] Config loaded in {config_time:.2f}s")

            # Update max_position_embeddings BEFORE creating model
            # This ensures the Embedding layer is created with the correct size
            if max_position_embeddings is not None and hasattr(config, 'text_config'):
                original_max_pos = config.text_config.max_position_embeddings
                config.text_config.max_position_embeddings = max_position_embeddings
                print(f"[SigLIP2] Config: max_position_embeddings {original_max_pos} -> {max_position_embeddings} (BEFORE model creation)")

            # Create model with config but no weights
            # Optimized: Create on CPU first without dtype (faster), dtype will be set after weight loading
            start_time = time.time()
            self.model = AutoModel.from_config(
                config,
                trust_remote_code=True
                # Don't set torch_dtype here - it's slow for large models on CPU
                # We'll convert to dtype after weight loading
            )
            # Keep on CPU for now (will move to device after weight loading)
            model_time = time.time() - start_time
            print(f"[SigLIP2] Model structure created in {model_time:.2f}s")

            # Get text model component
            self.text_model = self.model.text_model

            # Load tokenizer
            start_time = time.time()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            tokenizer_time = time.time() - start_time
            print(f"[SigLIP2] Tokenizer loaded in {tokenizer_time:.2f}s")

            # Note: Device move will happen after weight loading (in checkpoint_utils.py)
            # This avoids moving uninitialized weights to GPU, which is slow

            # Get config
            self.config = self.text_model.config
            self.hidden_size = self.config.hidden_size  # 1152

            print(f"[SigLIP2] Text encoder structure created (weights pending):")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.config.num_hidden_layers}")
            print(f"  Vocab size: {self.config.vocab_size}")
            print(f"  Max position embeddings: {self.config.max_position_embeddings}")
        else:
            # Load from HuggingFace (with pretrained weights)
            print(f"[SigLIP2] Loading text encoder from {model_name}...")

            # Load SigLIP-2 model
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype
            )

            # Get text model component
            self.text_model = self.model.text_model

            # Fix max_position_embeddings for variable-length support (SigLIP-2 NaViT/NAFlex)
            # Set to a large value to support long prompts
            if hasattr(self.text_model.config, 'max_position_embeddings'):
                original_max_pos = self.text_model.config.max_position_embeddings
                self.text_model.config.max_position_embeddings = 4096  # Large enough for any prompt
                print(f"[SigLIP2] Updated max_position_embeddings: {original_max_pos} -> {self.text_model.config.max_position_embeddings}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Move to device
            self.text_model = self.text_model.to(device)

            # Get config
            self.config = self.text_model.config
            self.hidden_size = self.config.hidden_size  # 1152

            print(f"[SigLIP2] Text encoder loaded:")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.config.num_hidden_layers}")
            print(f"  Vocab size: {self.config.vocab_size}")
            print(f"  Max position embeddings: {self.config.max_position_embeddings}")

    def encode(
        self,
        prompts: Union[str, List[str]],
        max_length: Optional[int] = None,
        return_pooled: bool = False,
        clip_skip: int = 0,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Encode text prompts.

        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum token length (None = no limit)
            return_pooled: Return pooled output (CLS token) instead of sequence
            clip_skip: Number of layers to skip from the end (0=last layer, 1=penultimate, etc.)
            requires_grad: Enable gradients for training (default: False)

        Returns:
            Text embeddings [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        tokenizer_kwargs = {
            "padding": True,
            "truncation": False,  # No truncation by default (no token limit)
            "return_tensors": "pt"
        }

        if max_length is not None:
            tokenizer_kwargs["max_length"] = max_length
            tokenizer_kwargs["truncation"] = True

        inputs = self.tokenizer(prompts, **tokenizer_kwargs)

        # Check if token count exceeds max_position_embeddings
        input_ids = inputs['input_ids']
        seq_length = input_ids.shape[1]
        max_pos_embeddings = self.config.max_position_embeddings  # 512 for SigLIP-2

        if seq_length > max_pos_embeddings:
            print(f"[SigLIP2] WARNING: Token count {seq_length} exceeds max_position_embeddings {max_pos_embeddings}")
            print(f"[SigLIP2] Auto-truncating to {max_pos_embeddings} tokens")
            # Truncate to max_position_embeddings
            inputs['input_ids'] = input_ids[:, :max_pos_embeddings]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :max_pos_embeddings]

        # Get actual device from text_model (may differ from self.device_name if moved)
        actual_device = next(self.text_model.parameters()).device
        inputs = {k: v.to(actual_device) for k, v in inputs.items()}

        # Encode with or without gradients
        if requires_grad:
            # Training mode: enable gradients
            if clip_skip > 0:
                # Manual layer iteration (output_hidden_states not supported in SigLIP2)
                # Get embeddings
                hidden_state = self.text_model.embeddings(inputs['input_ids'])

                # Prepare attention mask
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                    attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype)

                # Pass through layers
                num_layers = len(self.text_model.encoder.layers)  # Total: 27 layers (0-26)
                # clip_skip=0: use all 27 layers (0-26)
                # clip_skip=1: use 26 layers (0-25, penultimate)
                layers_to_use = num_layers - clip_skip

                for i, layer in enumerate(self.text_model.encoder.layers):
                    if i >= layers_to_use:
                        break
                    hidden_state = layer(hidden_state, attention_mask)

                # Apply final layer norm
                hidden_state = self.text_model.final_layer_norm(hidden_state)
            else:
                # Use last layer (default)
                outputs = self.text_model(**inputs)
                hidden_state = outputs.last_hidden_state
        else:
            # Inference mode: disable gradients
            with torch.no_grad():
                if clip_skip > 0:
                    # Manual layer iteration (output_hidden_states not supported in SigLIP2)
                    # Get embeddings
                    hidden_state = self.text_model.embeddings(inputs['input_ids'])

                    # Prepare attention mask
                    attention_mask = inputs.get('attention_mask')
                    if attention_mask is not None:
                        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_state.dtype)

                    # Pass through layers
                    num_layers = len(self.text_model.encoder.layers)  # Total: 27 layers (0-26)
                    # clip_skip=0: use all 27 layers (0-26)
                    # clip_skip=1: use 26 layers (0-25, penultimate)
                    layers_to_use = num_layers - clip_skip

                    for i, layer in enumerate(self.text_model.encoder.layers):
                        if i >= layers_to_use:
                            break
                        hidden_state = layer(hidden_state, attention_mask)

                    # Apply final layer norm
                    hidden_state = self.text_model.final_layer_norm(hidden_state)
                else:
                    # Use last layer (default)
                    outputs = self.text_model(**inputs)
                    hidden_state = outputs.last_hidden_state

        if return_pooled:
            # Return pooled output (last token)
            # SigLIP2 uses the last token's hidden state
            pooled_output = hidden_state[:, -1, :]  # [batch_size, hidden_size]
            # Apply projection head (only when using full last layer, not for clip_skip)
            if clip_skip == 0:
                pooled_output = self.text_model.head(pooled_output)
            return pooled_output
        else:
            # Return sequence embeddings
            return hidden_state  # [batch_size, seq_len, hidden_size]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with pre-tokenized inputs.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_pooled: Return pooled output instead of sequence

        Returns:
            Text embeddings
        """
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if return_pooled:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory-efficient training.

        SigLIP2's Siglip2TextTransformer doesn't have built-in gradient_checkpointing_enable(),
        so we implement it manually by wrapping the encoder's forward pass.
        """
        if hasattr(self.text_model, 'gradient_checkpointing_enable'):
            # Use built-in if available (future HuggingFace updates may add it)
            self.text_model.gradient_checkpointing_enable()
            print(f"[SigLIP2] Gradient checkpointing enabled for text encoder (built-in)")
        elif hasattr(self.text_model, 'encoder') and hasattr(self.text_model.encoder, 'layers'):
            # Manual implementation: wrap encoder layers with checkpoint
            self._enable_manual_gradient_checkpointing(self.text_model.encoder)
            print(f"[SigLIP2] Gradient checkpointing enabled for text encoder (manual, {len(self.text_model.encoder.layers)} layers)")
        else:
            print(f"[SigLIP2] WARNING: text_model does not support gradient checkpointing")

    def _enable_manual_gradient_checkpointing(self, encoder):
        """
        Manually enable gradient checkpointing by monkey-patching encoder forward.

        This wraps each encoder layer with torch.utils.checkpoint.checkpoint()
        to trade computation for memory during backward pass.
        """
        import torch.utils.checkpoint as checkpoint_utils

        # Store original forward method
        original_forward = encoder.forward

        # Get reference to layers (need to capture in closure)
        layers = encoder.layers

        def checkpointed_forward(inputs_embeds, attention_mask=None, **kwargs):
            """
            Forward pass with gradient checkpointing on each layer.

            Note: SigLIP2's Siglip2Encoder.forward() uses 'inputs_embeds' (not 'hidden_states').
            We replicate this but wrap each layer call with checkpoint().

            Args:
                inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
                attention_mask: Attention mask [batch_size, seq_len] or None
                **kwargs: Additional arguments (output_attentions, output_hidden_states, return_dict)
            """
            output_attentions = kwargs.get('output_attentions', False)
            output_hidden_states = kwargs.get('output_hidden_states', False)
            return_dict = kwargs.get('return_dict', True)

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for layer in layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Use checkpoint to save memory (recomputes activations during backward)
                # use_reentrant=False is recommended for modern PyTorch
                hidden_states = checkpoint_utils.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )

                if output_attentions:
                    # Note: When using checkpoint, attention weights are not available
                    # This is a limitation of gradient checkpointing
                    all_attentions = all_attentions + (None,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Return in expected format
            if return_dict:
                from transformers.modeling_outputs import BaseModelOutput
                return BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=all_hidden_states,
                    attentions=all_attentions
                )
            else:
                return (hidden_states, all_hidden_states, all_attentions)

        # Replace forward method
        encoder.forward = checkpointed_forward
        encoder._gradient_checkpointing_enabled = True


class SigLIP2ImageEncoder(nn.Module):
    """
    SigLIP-2 Image Encoder wrapper.

    Features:
    - Variable resolution support (NaViT/NAFlex)
    - Same architecture as text encoder (shared vision-language model)
    - Supports multiple images
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-naflex",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        load_from_checkpoint: bool = False,
        shared_config: Optional[Any] = None,
        max_position_embeddings: Optional[int] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device_name = device

        if load_from_checkpoint:
            # Create empty model structure (weights will be loaded via load_state_dict)
            print(f"[SigLIP2] Creating image encoder structure (loading from checkpoint)...")

            # Load config only (no weights) - reuse shared config if provided
            from transformers import AutoConfig
            if shared_config is not None:
                print(f"[SigLIP2] Reusing shared config (skipping download)...")
                config = shared_config
            else:
                start_time = time.time()
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                config_time = time.time() - start_time
                print(f"[SigLIP2] Config loaded in {config_time:.2f}s")

            # Create model with config but no weights
            # Optimized: Create on CPU first without dtype (faster), dtype will be set after weight loading
            start_time = time.time()
            self.model = AutoModel.from_config(
                config,
                trust_remote_code=True
                # Don't set torch_dtype here - it's slow for large models on CPU
                # We'll convert to dtype after weight loading
            )
            # Keep on CPU for now (will move to device after weight loading)
            model_time = time.time() - start_time
            print(f"[SigLIP2] Model structure created in {model_time:.2f}s")

            # Get vision model component
            self.vision_model = self.model.vision_model

            # Fix max_position_embeddings for variable-resolution support
            # Use value from checkpoint metadata if available, otherwise default to 4096
            if hasattr(self.vision_model.config, 'max_position_embeddings'):
                original_max_pos = self.vision_model.config.max_position_embeddings
                if max_position_embeddings is not None:
                    self.vision_model.config.max_position_embeddings = max_position_embeddings
                    print(f"[SigLIP2] Updated max_position_embeddings (from metadata): {original_max_pos} -> {max_position_embeddings}")
                else:
                    self.vision_model.config.max_position_embeddings = 4096  # Default fallback
                    print(f"[SigLIP2] Updated max_position_embeddings (default): {original_max_pos} -> 4096")

            # Load processor
            start_time = time.time()
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            processor_time = time.time() - start_time
            print(f"[SigLIP2] Processor loaded in {processor_time:.2f}s")

            # Note: Device move will happen after weight loading (in checkpoint_utils.py)
            # This avoids moving uninitialized weights to GPU, which is slow

            # Get config
            self.config = self.vision_model.config
            self.hidden_size = self.config.hidden_size  # 1152

            print(f"[SigLIP2] Image encoder structure created (weights pending):")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.config.num_hidden_layers}")
            print(f"  Patch size: {self.config.patch_size}")
        else:
            # Load from HuggingFace (with pretrained weights)
            print(f"[SigLIP2] Loading image encoder from {model_name}...")

            # Load SigLIP-2 model
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype
            )

            # Get vision model component
            self.vision_model = self.model.vision_model

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Move to device
            self.vision_model = self.vision_model.to(device)

            # Get config
            self.config = self.vision_model.config
            self.hidden_size = self.config.hidden_size  # 1152

            print(f"[SigLIP2] Image encoder loaded:")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Num layers: {self.config.num_hidden_layers}")
            print(f"  Patch size: {self.config.patch_size}")

    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_pooled: bool = False
    ) -> torch.Tensor:
        """
        Encode images.

        Args:
            images: Single image or list of images (PIL Images)
            return_pooled: Return pooled output instead of sequence

        Returns:
            Image embeddings [batch_size, num_patches, hidden_size] or [batch_size, hidden_size]
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Process images
        inputs = self.processor(images=images, return_tensors="pt")

        # Get actual device from vision_model (may differ from self.device_name if moved)
        actual_device = next(self.vision_model.parameters()).device
        inputs = {k: v.to(actual_device) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            # SigLIP2 vision model requires pixel_values, attention_mask, and spatial_shapes
            outputs = self.vision_model(
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask', None),
                spatial_shapes=inputs.get('spatial_shapes', None)
            )

        if return_pooled:
            # Return pooled output (CLS token)
            return outputs.pooler_output  # [batch_size, hidden_size]
        else:
            # Return sequence embeddings (all patches)
            return outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_pooled: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with pre-processed pixel values.

        Args:
            pixel_values: Preprocessed images [batch_size, channels, height, width]
            return_pooled: Return pooled output instead of sequence

        Returns:
            Image embeddings
        """
        outputs = self.vision_model(pixel_values=pixel_values)

        if return_pooled:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state

    def gradient_checkpointing_enable(self):
        """
        Enable gradient checkpointing for memory-efficient training.

        SigLIP2's Siglip2VisionTransformer doesn't have built-in gradient_checkpointing_enable(),
        so we implement it manually by wrapping the encoder's forward pass.
        """
        if hasattr(self.vision_model, 'gradient_checkpointing_enable'):
            # Use built-in if available (future HuggingFace updates may add it)
            self.vision_model.gradient_checkpointing_enable()
            print(f"[SigLIP2] Gradient checkpointing enabled for image encoder (built-in)")
        elif hasattr(self.vision_model, 'encoder') and hasattr(self.vision_model.encoder, 'layers'):
            # Manual implementation: wrap encoder layers with checkpoint
            self._enable_manual_gradient_checkpointing(self.vision_model.encoder)
            print(f"[SigLIP2] Gradient checkpointing enabled for image encoder (manual, {len(self.vision_model.encoder.layers)} layers)")
        else:
            print(f"[SigLIP2] WARNING: vision_model does not support gradient checkpointing")

    def _enable_manual_gradient_checkpointing(self, encoder):
        """
        Manually enable gradient checkpointing by monkey-patching encoder forward.

        This wraps each encoder layer with torch.utils.checkpoint.checkpoint()
        to trade computation for memory during backward pass.
        """
        import torch.utils.checkpoint as checkpoint_utils

        # Get reference to layers (need to capture in closure)
        layers = encoder.layers

        def checkpointed_forward(inputs_embeds, attention_mask=None, **kwargs):
            """
            Forward pass with gradient checkpointing on each layer.

            Note: SigLIP2's Siglip2Encoder.forward() uses 'inputs_embeds' (not 'hidden_states').
            We replicate this but wrap each layer call with checkpoint().

            Args:
                inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
                attention_mask: Attention mask [batch_size, seq_len] or None
                **kwargs: Additional arguments (output_attentions, output_hidden_states, return_dict)
            """
            output_attentions = kwargs.get('output_attentions', False)
            output_hidden_states = kwargs.get('output_hidden_states', False)
            return_dict = kwargs.get('return_dict', True)

            hidden_states = inputs_embeds
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            for layer in layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Use checkpoint to save memory (recomputes activations during backward)
                # use_reentrant=False is recommended for modern PyTorch
                hidden_states = checkpoint_utils.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )

                if output_attentions:
                    all_attentions = all_attentions + (None,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Return in expected format
            if return_dict:
                from transformers.modeling_outputs import BaseModelOutput
                return BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=all_hidden_states,
                    attentions=all_attentions
                )
            else:
                return (hidden_states, all_hidden_states, all_attentions)

        # Replace forward method
        encoder.forward = checkpointed_forward
        encoder._gradient_checkpointing_enabled = True


class SigLIP2MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder combining text and optional images.

    Supports:
    - T2I: Text only → <text> [END]
    - I2I: Single image + text → <text> [IMG0] <image0> [END]
    - TI2I: Text + Image instruction → <txt1> [IMG0] <img0> <txt2> [END]
    - Multi-image: Multiple images + text → <txt> [IMG0] <img0> [IMG1] <img1> [END]

    Special tokens (learned parameters, each [1, 1, hidden_size]):
    - [END]: Sequence end token (always appended)
    - [IMG0], [IMG1], ...: Image start tokens (prepended before each image)

    All inputs are concatenated along sequence dimension.
    """

    # Maximum number of images supported
    MAX_IMAGES = 4

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-naflex",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        text_encoder: Optional['SigLIP2TextEncoder'] = None,
        image_encoder: Optional['SigLIP2ImageEncoder'] = None,
        max_position_embeddings: Optional[int] = None
    ):
        super().__init__()

        # Use provided encoders if available (from checkpoint), otherwise create new ones
        if text_encoder is not None:
            print(f"[SigLIP2] Using provided text encoder (from checkpoint)")
            self.text_encoder = text_encoder

            # If max_position_embeddings not provided, get from text encoder
            if max_position_embeddings is None and hasattr(text_encoder, 'config'):
                max_position_embeddings = text_encoder.config.max_position_embeddings
        else:
            print(f"[SigLIP2] Creating new text encoder from HuggingFace")
            self.text_encoder = SigLIP2TextEncoder(
                model_name,
                dtype,
                device,
                max_position_embeddings=max_position_embeddings
            )

        if image_encoder is not None:
            print(f"[SigLIP2] Using provided image encoder (from checkpoint)")
            self.image_encoder = image_encoder
        else:
            print(f"[SigLIP2] Creating new image encoder from HuggingFace")
            # Pass max_position_embeddings to preserve text encoder's config
            self.image_encoder = SigLIP2ImageEncoder(
                model_name,
                dtype,
                device,
                max_position_embeddings=max_position_embeddings
            )

        self.hidden_size = self.text_encoder.hidden_size

        # Special tokens (learned parameters)
        # [END] token: sequence terminator
        self.end_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, dtype=dtype, device=device) * 0.02
        )

        # [IMG0], [IMG1], ... tokens: image start markers
        self.img_tokens = nn.ParameterList([
            nn.Parameter(
                torch.randn(1, 1, self.hidden_size, dtype=dtype, device=device) * 0.02
            )
            for _ in range(self.MAX_IMAGES)
        ])

        # Legacy: Keep null_image_embedding as alias to end_token for backward compatibility
        # (old checkpoints may have this parameter)
        # Note: New models should use end_token directly
        self.null_image_embedding = self.end_token  # Alias (not a separate parameter)

        print(f"[SigLIP2] Multi-modal encoder initialized:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Special tokens: [END] + [IMG0..IMG{self.MAX_IMAGES-1}]")
        print(f"  Supports: T2I, I2I, TI2I, Multi-image")

    def encode(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image], List[List[Image.Image]]]] = None,
        use_end_token: bool = True,
        clip_skip: int = 0,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Encode text and optional images with special tokens.

        Output format:
        - T2I (no images):     <text> [END]
        - Single image:        <text> [IMG0] <image0> [END]
        - Multi-image:         <text> [IMG0] <img0> [IMG1] <img1> ... [END]

        Args:
            prompts: Text prompts (str or List[str])
            images: Optional images:
                - None: T2I mode
                - Single Image: I2I mode
                - List[Image]: Multi-image mode (same images for all batches)
                - List[List[Image]]: Per-batch images (images[b] = list of images for batch b)
            use_end_token: Append [END] token (default: True, should always be True for DEUS)
            clip_skip: Number of layers to skip from the end for text encoder (0=last layer)
            requires_grad: Enable gradients for training (default: False)

        Returns:
            Concatenated embeddings [batch_size, total_seq_len, hidden_size]
        """
        # Encode text with clip_skip
        text_embeddings = self.text_encoder.encode(prompts, clip_skip=clip_skip, requires_grad=requires_grad)  # [B, text_seq, hidden]
        batch_size = text_embeddings.shape[0]
        device = text_embeddings.device

        # Start building sequence with text embeddings
        sequence_parts = [text_embeddings]

        # Process images
        if images is not None:
            # Normalize images to List[List[Image]] format
            if isinstance(images, Image.Image):
                # Single image for all batches
                images_per_batch = [[images]] * batch_size
            elif isinstance(images, list):
                if len(images) == 0:
                    images_per_batch = [[] for _ in range(batch_size)]
                elif isinstance(images[0], Image.Image):
                    # List of images (same for all batches)
                    images_per_batch = [images] * batch_size
                else:
                    # List[List[Image]] - per-batch images
                    images_per_batch = images
            else:
                raise ValueError(f"Unsupported images type: {type(images)}")

            # Validate image count
            max_images = max(len(imgs) for imgs in images_per_batch)
            if max_images > self.MAX_IMAGES:
                raise ValueError(f"Too many images: {max_images} > {self.MAX_IMAGES}")

            # For simplicity, assume all batches have same number of images
            # (batching with different image counts requires padding/masking)
            if len(set(len(imgs) for imgs in images_per_batch)) > 1:
                raise ValueError("All batches must have the same number of images for now")

            num_images = len(images_per_batch[0])

            # Encode each image with its [IMGn] token
            for img_idx in range(num_images):
                # Get [IMGn] token
                img_token = self.img_tokens[img_idx].expand(batch_size, -1, -1)  # [B, 1, hidden]
                img_token = img_token.to(device=device)
                sequence_parts.append(img_token)

                # Encode image
                batch_images = [imgs[img_idx] for imgs in images_per_batch]
                image_embeddings = self.image_encoder.encode(batch_images)  # [B, num_patches, hidden]
                sequence_parts.append(image_embeddings)

        # Append [END] token
        if use_end_token:
            end_token = self.end_token.expand(batch_size, -1, -1)  # [B, 1, hidden]
            end_token = end_token.to(device=device)
            sequence_parts.append(end_token)

        # Concatenate all parts
        combined_embeddings = torch.cat(sequence_parts, dim=1)

        return combined_embeddings

    def encode_with_interleaved_images(
        self,
        text_segments: List[str],
        images: List[Image.Image],
        image_positions: List[int],
        clip_skip: int = 0,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Encode text with images interleaved at specified positions.

        Example:
            text_segments = ["A photo of", "next to", "in a garden"]
            images = [img1, img2]
            image_positions = [1, 2]  # Insert img1 after segment 0, img2 after segment 1
            → <txt0> [IMG0] <img0> <txt1> [IMG1] <img1> <txt2> [END]

        Args:
            text_segments: List of text segments
            images: List of images to insert
            image_positions: Position indices where each image should be inserted
                (image[i] is inserted after text_segments[image_positions[i]-1])
            clip_skip: Number of layers to skip from the end
            requires_grad: Enable gradients for training

        Returns:
            Concatenated embeddings [1, total_seq_len, hidden_size]
        """
        if len(images) != len(image_positions):
            raise ValueError("Number of images must match number of image_positions")
        if len(images) > self.MAX_IMAGES:
            raise ValueError(f"Too many images: {len(images)} > {self.MAX_IMAGES}")

        device = self.end_token.device
        sequence_parts = []

        # Create a mapping of position -> image index
        pos_to_img = {pos: idx for idx, pos in enumerate(image_positions)}

        # Process text segments and insert images
        for seg_idx, text_seg in enumerate(text_segments):
            # Encode text segment
            if text_seg:  # Skip empty segments
                text_emb = self.text_encoder.encode(text_seg, clip_skip=clip_skip, requires_grad=requires_grad)
                sequence_parts.append(text_emb)

            # Check if an image should be inserted after this segment
            if seg_idx + 1 in pos_to_img:
                img_idx = pos_to_img[seg_idx + 1]

                # Add [IMGn] token
                img_token = self.img_tokens[img_idx].to(device=device)
                sequence_parts.append(img_token)

                # Encode and add image
                image_emb = self.image_encoder.encode(images[img_idx])
                sequence_parts.append(image_emb)

        # Add [END] token
        end_token = self.end_token.to(device=device)
        sequence_parts.append(end_token)

        # Concatenate all parts
        combined_embeddings = torch.cat(sequence_parts, dim=1)

        return combined_embeddings
