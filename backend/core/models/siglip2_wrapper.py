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
        shared_config: Optional[Any] = None
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
        clip_skip: int = 0
    ) -> torch.Tensor:
        """
        Encode text prompts.

        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum token length (None = no limit)
            return_pooled: Return pooled output (CLS token) instead of sequence
            clip_skip: Number of layers to skip from the end (0=last layer, 1=penultimate, etc.)

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
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        # Encode
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
        shared_config: Optional[Any] = None
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
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}

        # Encode
        with torch.no_grad():
            outputs = self.vision_model(**inputs)

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


class SigLIP2MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder combining text and optional images.

    Supports:
    - T2I: Text only (image_embeddings = learned null embeddings)
    - I2I: Single image + optional text
    - TI2I: Text + Image instruction
    - Multi-image: Multiple images + text

    All inputs are concatenated along sequence dimension.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-naflex",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        text_encoder: Optional['SigLIP2TextEncoder'] = None,
        image_encoder: Optional['SigLIP2ImageEncoder'] = None
    ):
        super().__init__()

        # Use provided encoders if available (from checkpoint), otherwise create new ones
        if text_encoder is not None and image_encoder is not None:
            print(f"[SigLIP2] Using provided text/image encoders (from checkpoint)")
            self.text_encoder = text_encoder
            self.image_encoder = image_encoder
        else:
            print(f"[SigLIP2] Creating new text/image encoders from HuggingFace")
            self.text_encoder = SigLIP2TextEncoder(model_name, dtype, device)
            self.image_encoder = SigLIP2ImageEncoder(model_name, dtype, device)

        self.hidden_size = self.text_encoder.hidden_size

        # Learned null image embedding (for T2I mode when no images provided)
        self.null_image_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, dtype=dtype, device=device) * 0.02
        )

        print(f"[SigLIP2] Multi-modal encoder initialized:")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Supports: T2I, I2I, TI2I, Multi-image")

    def encode(
        self,
        prompts: Union[str, List[str]],
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        use_null_image: bool = True,
        clip_skip: int = 0
    ) -> torch.Tensor:
        """
        Encode text and optional images, concatenating along sequence dimension.

        Args:
            prompts: Text prompts
            images: Optional images (None for T2I, single for I2I/TI2I, list for multi)
            use_null_image: Add null image embedding when no images provided
            clip_skip: Number of layers to skip from the end for text encoder (0=last layer, 1=penultimate)

        Returns:
            Concatenated embeddings [batch_size, total_seq_len, hidden_size]
            where total_seq_len = text_seq_len + image_seq_len (or +1 for null)
        """
        # Encode text with clip_skip
        text_embeddings = self.text_encoder.encode(prompts, clip_skip=clip_skip)  # [B, text_seq, hidden]
        batch_size = text_embeddings.shape[0]

        # Encode images (or use null)
        if images is None:
            if use_null_image:
                # T2I mode: use learned null image embedding
                image_embeddings = self.null_image_embedding.expand(batch_size, -1, -1)  # [B, 1, hidden]
            else:
                # No image embeddings
                return text_embeddings
        else:
            # Encode images
            image_embeddings = self.image_encoder.encode(images)  # [B, num_patches, hidden]

        # Concatenate: [text_tokens, image_patches]
        combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)

        return combined_embeddings
