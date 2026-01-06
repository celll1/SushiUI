"""
SigLIP-2 Text/Image Encoder Wrapper

Wraps google/siglip2-so400m-patch16-naflex for text and image encoding.
SigLIP-2 has no token limit and supports variable-length inputs.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from typing import Optional, List, Union
from PIL import Image


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
        load_from_checkpoint: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device_name = device

        if load_from_checkpoint:
            # Create empty model structure (weights will be loaded via load_state_dict)
            print(f"[SigLIP2] Creating text encoder structure (loading from checkpoint)...")

            # Load config only (no weights)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Create model with config but no weights
            self.model = AutoModel.from_config(
                config,
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
        return_pooled: bool = False
    ) -> torch.Tensor:
        """
        Encode text prompts.

        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum token length (None = no limit)
            return_pooled: Return pooled output (CLS token) instead of sequence

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
            outputs = self.text_model(**inputs)

        if return_pooled:
            # Return pooled output (CLS token, first token)
            return outputs.pooler_output  # [batch_size, hidden_size]
        else:
            # Return sequence embeddings
            return outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

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
        load_from_checkpoint: bool = False
    ):
        super().__init__()

        self.model_name = model_name
        self.dtype = dtype
        self.device_name = device

        if load_from_checkpoint:
            # Create empty model structure (weights will be loaded via load_state_dict)
            print(f"[SigLIP2] Creating image encoder structure (loading from checkpoint)...")

            # Load config only (no weights)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Create model with config but no weights
            self.model = AutoModel.from_config(
                config,
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
        device: str = "cuda"
    ):
        super().__init__()

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
        use_null_image: bool = True
    ) -> torch.Tensor:
        """
        Encode text and optional images, concatenating along sequence dimension.

        Args:
            prompts: Text prompts
            images: Optional images (None for T2I, single for I2I/TI2I, list for multi)
            use_null_image: Add null image embedding when no images provided

        Returns:
            Concatenated embeddings [batch_size, total_seq_len, hidden_size]
            where total_seq_len = text_seq_len + image_seq_len (or +1 for null)
        """
        # Encode text
        text_embeddings = self.text_encoder.encode(prompts)  # [B, text_seq, hidden]
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
