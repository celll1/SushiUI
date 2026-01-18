# Z-Image LoRA Training Implementation Plan

**Status**: Planning Phase
**Author**: Claude (SushiUI Development AI)
**Date**: 2025-12-08
**Target Release**: v1.0 (Z-Image Training Support)

---

## Executive Summary

This document outlines the implementation plan for adding LoRA training support for Z-Image models to SushiUI. Z-Image represents a significant architectural departure from SD1.5/SDXL, using a Transformer backbone instead of U-Net, Qwen3 text encoder instead of CLIP, and flow matching scheduler instead of DDPM. This requires substantial modifications to the existing training infrastructure while maintaining compatibility with SD/SDXL training.

**Key Implementation Challenges**:
1. **Architecture**: Transformer-based (30 layers, dim=3840) vs U-Net
2. **Text Encoding**: Qwen3 with chat templates and variable-length embeddings (2560-dim)
3. **Scheduler**: Flow matching (velocity prediction) vs DDPM (noise prediction)
4. **Latent Space**: 16 channels vs SD/SDXL's 4 channels
5. **Memory**: Higher VRAM requirements (~15-20GB for weights alone)

**Design Decisions**:
- **Text Encoder (Qwen3) remains frozen**: LoRA is applied only to Transformer blocks, not Text Encoder
- **Caption pre-encoding is mandatory**: All captions must be pre-encoded before training
- **Compatibility with distributed LoRA**: Matches community standard format (Transformer-only weights)

**Estimated Implementation Effort**: 20-30 hours

---

## Table of Contents

1. [Architecture Comparison](#1-architecture-comparison)
2. [Implementation Phases](#2-implementation-phases)
3. [Phase 1: Model Detection and Loading](#3-phase-1-model-detection-and-loading)
4. [Phase 2: Text Encoding Infrastructure](#4-phase-2-text-encoding-infrastructure)
5. [Phase 3: Flow Matching Training Loop](#5-phase-3-flow-matching-training-loop)
6. [Phase 4: LoRA Integration](#6-phase-4-lora-integration)
7. [Phase 5: Testing and Validation](#7-phase-5-testing-and-validation)
8. [Phase 6: Memory Optimization](#8-phase-6-memory-optimization)
9. [Risk Analysis and Mitigation](#9-risk-analysis-and-mitigation)
10. [Code Organization](#10-code-organization)

---

## 1. Architecture Comparison

### 1.1 SD/SDXL vs Z-Image

| Feature | SD1.5/SDXL | Z-Image | Impact on Training |
|---------|------------|---------|-------------------|
| **Backbone** | U-Net | Transformer (30 layers) | Different forward pass, LoRA target modules |
| **Text Encoder** | CLIP (ViT-L/bigG) | Qwen3 | Chat templates, variable-length embeddings |
| **Text Embedding Dim** | 768/1280 (concatenated 2048) | 2560 | Different conditioning dimensions |
| **Text Sequence Length** | Fixed 77 tokens | Variable (up to 512/1024) | Requires dynamic padding/masking |
| **Latent Channels** | 4 | 16 | 4x memory for latents, different VAE |
| **Attention Type** | Cross-attention | Self-attention (unified tokens) | Different conditioning mechanism |
| **Normalization** | LayerNorm | RMSNorm | No impact (handled internally) |
| **Position Encoding** | None (implicit) | RoPE (Rotary Position Embeddings) | Calculated per forward pass |
| **Scheduler** | DDPM | Flow Matching (Euler) | Different loss calculation |
| **Training Objective** | Noise prediction | Velocity prediction | Different loss target |
| **VAE Shift Factor** | No | Yes (shift_factor in config) | Different latent preprocessing |

### 1.2 LoRA Target Modules

**SD/SDXL**:
```python
# U-Net attention layers
target_modules = [
    "to_q",      # Query projection
    "to_k",      # Key projection
    "to_v",      # Value projection
    "to_out.0"   # Output projection (Linear)
]

# Text Encoder MLP layers (optional in SushiUI, always enabled)
text_encoder_target_modules = [
    "mlp.fc1",   # Text Encoder 1/2 MLP layer 1
    "mlp.fc2"    # Text Encoder 1/2 MLP layer 2
]
```

**Z-Image**:
```python
# Transformer attention layers (ONLY - Text Encoder is frozen)
target_modules = [
    "to_q",      # Query projection
    "to_k",      # Key projection
    "to_v",      # Value projection
    "to_out.0"   # Output projection (ModuleList[0] - special handling required)
]

# Text Encoder (Qwen3) - NOT TARGETED in initial implementation
# Qwen3 remains frozen to match community standard and reduce VRAM usage
```

**Key Differences**:
1. `to_out` is a `ModuleList` in Z-Image, not a single `Linear` layer
2. **Text Encoder is NOT trained** in Z-Image (frozen), unlike SD/SDXL where it's optional

---

## 2. Implementation Phases

### Phase Breakdown

| Phase | Tasks | Estimated Time | Complexity |
|-------|-------|----------------|------------|
| **Phase 1** | Model detection, loading, component access | 3-4 hours | Medium |
| **Phase 2** | Text encoding (Qwen3), caption pre-encoding | 5-7 hours | High |
| **Phase 3** | Flow matching training loop | 6-8 hours | High |
| **Phase 4** | LoRA injection (ModuleList handling) | 3-4 hours | Medium |
| **Phase 5** | Testing, validation, debugging | 4-6 hours | Medium |
| **Phase 6** | Memory optimization, gradient checkpointing | 3-4 hours | Medium |

**Total**: 24-33 hours

---

## 3. Phase 1: Model Detection and Loading

### 3.1 Goals

- Detect Z-Image models from file path or model config
- Load Z-Image components (Transformer, VAE, Text Encoder, Tokenizer, Scheduler)
- Integrate into existing `LoRATrainer.__init__()` flow

### 3.2 Implementation Tasks

#### Task 1.1: Add Z-Image Detection

**File**: `backend/core/training/lora_trainer.py`

**Location**: `__init__()` method, after model path validation

```python
# Detect model type
from core.model_loader import ModelLoader

model_type = ModelLoader.detect_model_type(model_path)
self.is_zimage = (model_type == "zimage")
self.is_sdxl = False  # Will be set later for SD/SDXL

if self.is_zimage:
    print(f"[LoRATrainer] Detected Z-Image model")
```

#### Task 1.2: Load Z-Image Components

**File**: `backend/core/training/lora_trainer.py`

**New section**: After existing model loading

```python
if self.is_zimage:
    print(f"[LoRATrainer] Loading Z-Image components from {model_path}")

    # Load components using ModelLoader
    components = ModelLoader.load_zimage_from_diffusers(
        model_path=model_path,
        device="cpu",  # Load on CPU first, move to GPU later
        torch_dtype=self.weight_dtype
    )

    self.transformer = components["transformer"]
    self.vae = components["vae"]
    self.text_encoder = components["text_encoder"]
    self.tokenizer = components["tokenizer"]
    self.scheduler = components["scheduler"]

    # Z-Image specific: no text_encoder_2
    self.text_encoder_2 = None
    self.tokenizer_2 = None

    # Note: Z-Image uses Flow Matching scheduler, not DDPM
    print(f"[LoRATrainer] Using Flow Matching scheduler for Z-Image")
else:
    # Existing SD/SDXL loading logic
    # ...
```

#### Task 1.3: Update Model Type Detection

**File**: `backend/core/model_loader.py`

**New method**: `detect_model_type()`

```python
@staticmethod
def detect_model_type(model_path: str) -> str:
    """
    Detect model type from path or config.

    Returns: "sd15", "sdxl", "zimage", "unknown"
    """
    if not os.path.exists(model_path):
        return "unknown"

    # Check if it's a diffusers directory
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Z-Image: has "transformer" instead of "unet"
            if "transformer" in config:
                return "zimage"

            # SDXL: has "text_encoder_2"
            if "text_encoder_2" in config:
                return "sdxl"

            # SD1.5: has "text_encoder" but no "text_encoder_2"
            if "text_encoder" in config:
                return "sd15"

    # Safetensors file: try loading config keys
    # (Fallback: load pipeline and check components)

    return "unknown"
```

### 3.3 Success Criteria

- [ ] Z-Image models are correctly detected
- [ ] All Z-Image components load without errors
- [ ] Transformer, VAE, Text Encoder accessible
- [ ] No breaking changes to SD/SDXL training

---

## 4. Phase 2: Text Encoding Infrastructure

### 4.1 Goals

- Implement Qwen3 text encoding with chat templates
- Create caption pre-encoding and caching system
- Handle variable-length embeddings in dataset

**⚠️ MANDATORY for Z-Image Training**: Since Text Encoder (Qwen3) remains frozen, caption pre-encoding is **required** before training begins. All captions must be encoded to embeddings and cached, similar to latent caching.

### 4.2 Implementation Tasks

#### Task 2.1: Qwen3 Text Encoding

**File**: `backend/core/training/lora_trainer.py`

**New method**: `encode_prompt_zimage()`

```python
def encode_prompt_zimage(
    self,
    prompt: str,
    max_sequence_length: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode prompt using Qwen3 text encoder with chat template.

    Args:
        prompt: Text prompt
        max_sequence_length: Maximum sequence length (default: 512)

    Returns:
        prompt_embeds: [valid_seq_len, 2560]
        attention_mask: [max_sequence_length] (bool)
    """
    # Format with Qwen chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Qwen-specific feature
    )

    # Tokenize
    text_inputs = self.tokenizer(
        formatted_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(self.device)
    attention_mask = text_inputs.attention_mask.to(self.device).bool()

    # Encode with penultimate layer (similar to SDXL text_encoder_2)
    with torch.no_grad():
        encoder_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = encoder_output.hidden_states[-2]  # [1, seq_len, 2560]

    # Extract valid embeddings (masked by attention_mask)
    valid_embeds = prompt_embeds[0][attention_mask[0]]  # [valid_seq_len, 2560]

    return valid_embeds, attention_mask[0]
```

#### Task 2.2: Caption Pre-Encoding System

**File**: `backend/core/training/lora_trainer.py`

**New method**: `prepare_caption_cache()`

```python
def prepare_caption_cache(self, dataset_items: list) -> dict:
    """
    Pre-encode all captions to save VRAM during training.

    This is similar to latent caching but for text embeddings.

    Args:
        dataset_items: List of dataset items with 'caption' field

    Returns:
        caption_cache: {caption_text: {"embeddings": Tensor, "mask": Tensor}}
    """
    print(f"[LoRATrainer] Pre-encoding {len(dataset_items)} captions for Z-Image...")

    caption_cache = {}
    unique_captions = set(item.caption for item in dataset_items)

    # Move text encoder to GPU for encoding
    self.text_encoder.to(self.device)
    self.text_encoder.eval()

    for caption in tqdm(unique_captions, desc="Encoding captions"):
        if caption not in caption_cache:
            embeds, mask = self.encode_prompt_zimage(caption)
            caption_cache[caption] = {
                "embeddings": embeds.cpu(),  # [seq_len, 2560]
                "mask": mask.cpu(),           # [max_seq_len]
            }

    # Move text encoder back to CPU to save VRAM
    self.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    print(f"[LoRATrainer] Encoded {len(caption_cache)} unique captions")

    return caption_cache
```

#### Task 2.3: Dataset Item Caption Embedding Attachment

**File**: `backend/core/training/lora_trainer.py`

**Modify**: `train()` method, before training loop

```python
def train(self, dataset_items, ...):
    # Existing latent caching
    # ...

    # Z-Image: Pre-encode captions
    if self.is_zimage:
        caption_cache = self.prepare_caption_cache(dataset_items)

        # Attach pre-encoded captions to each item
        for item in dataset_items:
            item.caption_embeds = caption_cache[item.caption]["embeddings"]
            item.caption_mask = caption_cache[item.caption]["mask"]

    # Continue with training loop
    # ...
```

### 4.3 Success Criteria

- [ ] Qwen3 text encoding works with chat templates
- [ ] Caption pre-encoding completes without errors
- [ ] Variable-length embeddings stored correctly
- [ ] Text encoder moves to CPU after encoding (VRAM freed)

---

## 5. Phase 3: Flow Matching Training Loop

### 5.1 Goals

- Implement flow matching training step
- Replace noise prediction with velocity prediction
- Handle Z-Image transformer forward pass

### 5.2 Flow Matching Theory

**DDPM (SD/SDXL)**:
```python
# Sample noise and timestep
noise = torch.randn_like(latents)
timesteps = torch.randint(0, 1000, (batch_size,))

# Add noise to latents
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# Predict noise
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)

# Loss: MSE between predicted and true noise
loss = F.mse_loss(noise_pred, noise)
```

**Flow Matching (Z-Image)**:
```python
# Sample random t ∈ [0, 1]
t = torch.rand(batch_size, device=device)

# Linear interpolation: noisy = t * data + (1-t) * noise
noise = torch.randn_like(latents)
noisy_latents = t.view(-1, 1, 1, 1) * latents + (1 - t).view(-1, 1, 1, 1) * noise

# Predict velocity (direction of flow from noise to data)
velocity_pred = transformer(noisy_latents, t, cap_feats)

# True velocity: data - noise
true_velocity = latents - noise

# Loss: MSE between predicted and true velocity
loss = F.mse_loss(velocity_pred, true_velocity)
```

**Key Differences**:
- DDPM: Discrete timesteps (0-1000)
- Flow Matching: Continuous t ∈ [0, 1]
- DDPM: Predict noise
- Flow Matching: Predict velocity (velocity = data - noise)

### 5.3 Implementation Tasks

#### Task 3.1: Flow Matching Training Step

**File**: `backend/core/training/lora_trainer.py`

**New method**: `train_step_zimage()`

```python
def train_step_zimage(self, batch: list[dict]) -> torch.Tensor:
    """
    Training step for Z-Image models using flow matching.

    Args:
        batch: List of dicts with keys:
            - "latents": [16, H, W]
            - "caption_embeds": [seq_len, 2560]
            - "caption_mask": [max_seq_len]

    Returns:
        loss: Scalar tensor
    """
    # Stack latents (convert list to tensor)
    latents_list = [item["latents"] for item in batch]
    latents = torch.stack(latents_list).to(self.device, dtype=torch.float32)

    # Get caption embeddings (list of variable-length tensors)
    cap_feats_list = [
        item["caption_embeds"].to(self.device, dtype=torch.bfloat16)
        for item in batch
    ]

    batch_size = latents.shape[0]

    # Sample random timesteps t ∈ [0, 1]
    timesteps = torch.rand(batch_size, device=self.device)

    # Sample noise
    noise = torch.randn_like(latents)

    # Linear interpolation: noisy = t * data + (1-t) * noise
    t_expanded = timesteps.view(-1, 1, 1, 1)
    noisy_latents = t_expanded * latents + (1 - t_expanded) * noise

    # Add frame dimension (Z-Image requires [B, C, F, H, W])
    noisy_latents = noisy_latents.unsqueeze(2)  # [B, 16, 1, H, W]

    # Convert to list (Z-Image requirement)
    noisy_latents_list = list(noisy_latents.unbind(dim=0))

    # Scale timesteps to [0, 1000] for t_embedder
    timesteps_scaled = timesteps * 1000.0

    # Forward pass with LoRA
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=self.mixed_precision):
        model_output_list = self.transformer(
            x=noisy_latents_list,
            t=timesteps_scaled,
            cap_feats=cap_feats_list,
            patch_size=2,
            f_patch_size=1,
        )[0]  # Returns (output_list, {})

    # Stack outputs and remove frame dimension
    velocity_pred = torch.stack(model_output_list, dim=0).squeeze(2)  # [B, 16, H, W]

    # Compute true velocity (data - noise)
    true_velocity = latents - noise

    # Loss: MSE between predicted and true velocity (in FP32 for stability)
    loss = F.mse_loss(
        velocity_pred.float(),
        true_velocity.float(),
        reduction="mean"
    )

    return loss
```

#### Task 3.2: Integrate into Main Training Loop

**File**: `backend/core/training/lora_trainer.py`

**Modify**: `train()` method, main loop

```python
def train(self, dataset_items, ...):
    # ... (existing setup)

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            self.optimizer.zero_grad()

            # Choose training step based on model type
            if self.is_zimage:
                loss = self.train_step_zimage(batch)
            elif self.is_sdxl:
                loss = self.train_step_sdxl(batch)
            else:
                loss = self.train_step_sd15(batch)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # ... (logging, etc.)
```

### 5.4 Success Criteria

- [ ] Flow matching loss calculation is correct
- [ ] Timestep sampling in [0, 1] range works
- [ ] Velocity prediction loss converges
- [ ] No gradient NaN/Inf issues

---

## 6. Phase 4: LoRA Integration

### 6.1 Goals

- Apply LoRA to Z-Image Transformer attention layers **only** (Text Encoder remains frozen)
- Handle ModuleList (`to_out[0]`) injection
- Ensure trainable parameters are correct

**⚠️ Design Decision: Text Encoder (Qwen3) LoRA is NOT included in initial implementation**

**Rationale**:
1. **Community standard**: Most distributed Z-Image LoRA weights contain only Transformer parameters
2. **musubi-tuner reference**: Official Z-Image LoRA implementation (`lora_zimage.py`) targets only `ZImageTransformerBlock`, not Text Encoder
3. **Memory efficiency**: Keeping Qwen3 frozen (7B parameters) saves ~2-3GB VRAM and reduces training time
4. **Simplicity**: Qwen3 is a large LLM requiring significant resources to train
5. **Compatibility**: Matches expected format of distributed Z-Image LoRA files

**Caption Pre-encoding is MANDATORY**: Since Text Encoder is frozen, all captions must be pre-encoded before training (implemented in Phase 2).

**Future extension**: Text Encoder LoRA can be added as an optional feature (`--train_text_encoder` flag) in a future phase, similar to ai-toolkit's implementation.

### 6.2 Implementation Tasks

#### Task 4.1: Z-Image LoRA Target Module Detection

**File**: `backend/core/training/lora_trainer.py`

**New method**: `_apply_lora_zimage()`

```python
def _apply_lora_zimage(self):
    """Apply LoRA layers to Z-Image Transformer."""
    print(f"[LoRATrainer] Applying LoRA to Z-Image Transformer (rank={self.lora_rank}, alpha={self.lora_alpha})")

    lora_count = 0

    # Target all ZImageAttention modules
    for module_name, module in self.transformer.named_modules():
        if module.__class__.__name__ == "ZImageAttention":
            # Apply LoRA to to_q, to_k, to_v (standard Linear layers)
            for linear_name in ["to_q", "to_k", "to_v"]:
                if hasattr(module, linear_name):
                    original_linear = getattr(module, linear_name)
                    lora_linear = inject_lora_into_linear(
                        original_linear,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha
                    )
                    setattr(module, linear_name, lora_linear)

                    # Store reference for state_dict saving
                    storage_key = f"transformer.{module_name}.{linear_name}"
                    self.lora_layers[storage_key] = lora_linear
                    lora_count += 1

            # Special handling for to_out (ModuleList)
            if hasattr(module, "to_out") and isinstance(module.to_out, torch.nn.ModuleList):
                original_linear = module.to_out[0]
                lora_linear = inject_lora_into_linear(
                    original_linear,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha
                )
                module.to_out[0] = lora_linear

                # Storage key: "to_out.0"
                storage_key = f"transformer.{module_name}.to_out.0"
                self.lora_layers[storage_key] = lora_linear
                lora_count += 1

    print(f"[LoRATrainer] Injected {lora_count} LoRA layers into Z-Image Transformer")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
    print(f"[LoRATrainer] Trainable parameters: {trainable_params:,}")
```

#### Task 4.2: Update LoRA Application Logic

**File**: `backend/core/training/lora_trainer.py`

**Modify**: `_apply_lora()` method

```python
def _apply_lora(self):
    """Apply LoRA layers to model (model-type aware)."""
    if self.is_zimage:
        self._apply_lora_zimage()
    else:
        # Existing SD/SDXL LoRA application
        self._apply_lora_to_unet_transformers()
        # ... (text encoder LoRA)
```

#### Task 4.3: LoRA State Dict Saving

**File**: `backend/core/training/lora_trainer.py`

**Modify**: `save_checkpoint()` method

```python
def save_checkpoint(self, output_path: str):
    """Save LoRA weights to safetensors."""

    lora_state_dict = {}

    if self.is_zimage:
        # Z-Image: Extract LoRA weights from transformer
        for key, lora_module in self.lora_layers.items():
            # Extract lora_down and lora_up weights
            lora_state_dict[f"{key}.lora_down.weight"] = lora_module.lora_down.weight.to(self.output_dtype).cpu()
            lora_state_dict[f"{key}.lora_up.weight"] = lora_module.lora_up.weight.to(self.output_dtype).cpu()
    else:
        # Existing SD/SDXL saving logic
        # ...

    # Save with metadata
    metadata = {
        "model_type": "zimage" if self.is_zimage else ("sdxl" if self.is_sdxl else "sd15"),
        "lora_rank": str(self.lora_rank),
        "lora_alpha": str(self.lora_alpha),
        # ... (other metadata)
    }

    save_file(lora_state_dict, output_path, metadata=metadata)
```

### 6.3 Success Criteria

- [ ] LoRA layers correctly injected into Z-Image Transformer
- [ ] ModuleList (`to_out[0]`) handled correctly
- [ ] Trainable parameter count is reasonable
- [ ] LoRA weights save to safetensors without errors
- [ ] Saved LoRA can be loaded for inference

---

## 7. Phase 5: Testing and Validation

### 7.1 Testing Strategy

#### 7.1.1 Unit Tests

**Test 1: Model Detection**
```python
def test_zimage_detection():
    model_type = ModelLoader.detect_model_type("/path/to/zimage/model")
    assert model_type == "zimage"
```

**Test 2: Text Encoding**
```python
def test_qwen3_encoding():
    embeds, mask = trainer.encode_prompt_zimage("1girl, anime, beautiful")
    assert embeds.shape[1] == 2560  # Qwen3 dimension
    assert mask.dtype == torch.bool
```

**Test 3: Flow Matching Loss**
```python
def test_flow_matching_loss():
    # Create dummy batch
    batch = [{"latents": torch.randn(16, 64, 64), "caption_embeds": torch.randn(50, 2560)}]
    loss = trainer.train_step_zimage(batch)
    assert loss.item() > 0 and not torch.isnan(loss)
```

#### 7.1.2 Integration Tests

**Test 4: End-to-End Training (Small Dataset)**
```python
def test_zimage_training_small():
    # Train on 10 images for 100 steps
    trainer = LoRATrainer(model_path="zimage_model", lora_rank=4, learning_rate=1e-4)
    dataset = create_small_dataset(10)
    trainer.train(dataset, num_epochs=1, batch_size=1)

    # Check loss decreases
    assert final_loss < initial_loss
```

**Test 5: LoRA Inference**
```python
def test_zimage_lora_inference():
    # Load trained LoRA
    lora_path = "output/zimage_lora.safetensors"
    pipeline = load_zimage_pipeline_with_lora("base_model", lora_path)

    # Generate image
    image = pipeline("test prompt", num_inference_steps=20)
    assert image.size == (1024, 1024)
```

### 7.2 Validation Metrics

**During Training**:
- Loss convergence (should decrease over steps)
- Gradient norms (should be stable, not exploding/vanishing)
- VRAM usage (should fit in 24GB VRAM with gradient checkpointing)

**After Training**:
- Visual quality of generated images
- LoRA influence strength (compare base vs LoRA outputs)
- Concept adherence (trained concept should appear in outputs)

### 7.3 Success Criteria

- [ ] All unit tests pass
- [ ] Integration tests complete without errors
- [ ] Loss converges on small dataset
- [ ] Generated images show LoRA influence
- [ ] VRAM usage within acceptable limits

---

## 8. Phase 6: Memory Optimization

### 8.1 Goals

- Reduce VRAM usage to enable training on 24GB GPUs
- Implement gradient checkpointing
- Optimize batch processing

### 8.2 Implementation Tasks

#### Task 6.1: Gradient Checkpointing

**File**: `backend/core/training/lora_trainer.py`

**Location**: `__init__()` method, after model loading

```python
if self.is_zimage:
    # Enable gradient checkpointing for Transformer
    if hasattr(self.transformer, 'enable_gradient_checkpointing'):
        self.transformer.enable_gradient_checkpointing()
        print(f"[LoRATrainer] Gradient checkpointing enabled for Z-Image Transformer")

    # Text Encoder gradient checkpointing
    if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
        self.text_encoder.gradient_checkpointing_enable()
        print(f"[LoRATrainer] Gradient checkpointing enabled for Text Encoder")
```

#### Task 6.2: CPU Offloading Strategy

**VRAM Usage Breakdown**:
- Transformer weights (BF16): ~15-20 GB
- LoRA parameters (FP32): ~50-200 MB (rank 4-16)
- Gradients: ~15-20 GB (with gradient checkpointing: ~5-7 GB)
- Optimizer states (AdamW): ~30-40 GB (with AdamW 8-bit: ~4-5 GB)
- Latents (batch_size=1): ~1 GB
- **Total without optimization**: 60-80 GB ❌
- **Total with optimization**: 25-30 GB ✅

**Optimization Strategy**:
1. **Gradient Checkpointing**: Save 10-13 GB (-50-65%)
2. **AdamW 8-bit**: Save 25-35 GB (-80%)
3. **VAE on CPU**: Save 1-2 GB
4. **Text Encoder on CPU** (during training): Save 2-3 GB

**Implementation**:
```python
# VAE remains on CPU (only used for latent caching)
self.vae.to("cpu")

# Text Encoder on CPU after caption pre-encoding
self.text_encoder.to("cpu")

# Only Transformer on GPU during training
self.transformer.to(self.device)
```

#### Task 6.3: Batch Size Optimization

**Recommendation**: Start with batch_size=1 for Z-Image

**Reason**:
- Latents: `[B, 16, H, W]` → 16x larger than SD/SDXL
- Variable-length captions: Padding overhead
- Transformer: 30 layers with dim=3840

**Future**: Gradient accumulation for effective larger batch sizes

### 8.3 Success Criteria

- [ ] Training fits in 24GB VRAM
- [ ] Gradient checkpointing reduces memory by 50%+
- [ ] AdamW 8-bit reduces optimizer memory by 80%+
- [ ] Training speed acceptable (>1 step/sec)

---

## 9. Risk Analysis and Mitigation

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Flow matching implementation incorrect** | Medium | High | Reference musubi-tuner implementation, validate loss convergence |
| **Qwen3 encoding fails** | Low | High | Test with reference implementation, use fallback to direct embedding |
| **VRAM overflow** | High | High | Implement gradient checkpointing, CPU offloading, reduce batch size |
| **ModuleList LoRA injection breaks** | Medium | Medium | Test thoroughly, add special handling for ModuleList |
| **Loss does not converge** | Medium | High | Verify flow matching math, check learning rate, add debugging logs |
| **Generated images poor quality** | Medium | Medium | Tune hyperparameters, increase training steps, verify LoRA loading |

### 9.2 Compatibility Risks

| Risk | Mitigation |
|------|------------|
| **Break existing SD/SDXL training** | Extensive testing, separate code paths for Z-Image |
| **LoRA format incompatibility** | Follow diffusers/PEFT conventions, add format converter if needed |
| **Inference pipeline integration** | Test LoRA loading in existing inference code, add Z-Image-specific loader |

### 9.3 Performance Risks

| Risk | Mitigation |
|------|------------|
| **Training too slow** | Profile bottlenecks, optimize data loading, use torch.compile |
| **Memory leaks** | Add garbage collection, verify tensor deallocation |
| **Gradient instability** | Use gradient clipping, monitor gradient norms, reduce learning rate |

---

## 10. Code Organization

### 10.1 Modified Files

| File | Changes | Reason |
|------|---------|--------|
| `backend/core/training/lora_trainer.py` | Add Z-Image support, flow matching training loop | Core training logic |
| `backend/core/model_loader.py` | Add `detect_model_type()`, Z-Image loading | Model detection and loading |
| `backend/core/training/MODEL_ARCHITECTURES.md` | Add Z-Image architecture documentation | Reference documentation |

### 10.2 New Files

| File | Purpose |
|------|---------|
| `backend/core/training/flow_matching.py` | Flow matching utilities (timestep sampling, velocity calculation) |
| `backend/core/training/qwen3_encoding.py` | Qwen3 text encoding utilities (chat template, variable-length handling) |
| `docs/ZIMAGE_TRAINING_GUIDE.md` | User-facing training guide for Z-Image |

### 10.3 Code Structure

```
backend/core/training/
├── lora_trainer.py           # Main LoRA trainer (SD/SDXL/Z-Image)
│   ├── __init__()            # Model detection and loading
│   ├── _apply_lora()         # LoRA application (dispatches to model-specific methods)
│   ├── _apply_lora_zimage()  # Z-Image LoRA injection
│   ├── encode_prompt_zimage() # Qwen3 text encoding
│   ├── train_step_zimage()   # Flow matching training step
│   └── save_checkpoint()     # Save LoRA weights (model-type aware)
│
├── flow_matching.py          # Flow matching utilities
│   ├── sample_timesteps()    # Sample t ∈ [0, 1]
│   ├── compute_velocity()    # velocity = data - noise
│   └── linear_interpolation() # noisy = t * data + (1-t) * noise
│
├── qwen3_encoding.py         # Qwen3 utilities
│   ├── apply_chat_template() # Format prompts with chat template
│   └── extract_valid_embeddings() # Extract embeddings with attention mask
│
└── MODEL_ARCHITECTURES.md    # Architecture reference (SD/SDXL/Z-Image)
```

---

## 11. Implementation Checklist

### Phase 1: Model Detection and Loading
- [ ] Add `detect_model_type()` to `model_loader.py`
- [ ] Add Z-Image loading logic to `LoRATrainer.__init__()`
- [ ] Test Z-Image component loading
- [ ] Verify no breaking changes to SD/SDXL

### Phase 2: Text Encoding
- [ ] Implement `encode_prompt_zimage()` with Qwen3 chat template
- [ ] Implement `prepare_caption_cache()` for pre-encoding
- [ ] Test caption encoding with sample prompts
- [ ] Verify text encoder moves to CPU after encoding

### Phase 3: Flow Matching Training Loop
- [ ] Implement `train_step_zimage()` with flow matching
- [ ] Implement velocity calculation and loss
- [ ] Integrate into main training loop
- [ ] Verify loss convergence on dummy data

### Phase 4: LoRA Integration
- [ ] Implement `_apply_lora_zimage()` with ModuleList handling
- [ ] Test LoRA injection and trainable parameter count
- [ ] Implement LoRA state dict saving
- [ ] Test LoRA loading in inference pipeline

### Phase 5: Testing and Validation
- [ ] Write and run unit tests (model detection, text encoding, loss)
- [ ] Write and run integration tests (end-to-end training, inference)
- [ ] Train on small dataset (10-100 images) and verify convergence
- [ ] Generate images with trained LoRA and verify quality

### Phase 6: Memory Optimization
- [ ] Enable gradient checkpointing for Transformer and Text Encoder
- [ ] Implement CPU offloading (VAE, Text Encoder)
- [ ] Benchmark VRAM usage with optimizations
- [ ] Test training on 24GB GPU

### Documentation
- [ ] Update `MODEL_ARCHITECTURES.md` with Z-Image section
- [ ] Create `ZIMAGE_TRAINING_GUIDE.md` for users
- [ ] Add Z-Image training example config
- [ ] Update main README with Z-Image training support

---

## 12. Timeline Estimate

### Week 1: Foundation (Phases 1-2)
- **Day 1-2**: Model detection, loading, component access (6-8 hours)
- **Day 3-4**: Text encoding, caption pre-encoding (8-10 hours)
- **Day 5**: Testing Phase 1-2 (4-5 hours)

### Week 2: Training Loop (Phases 3-4)
- **Day 1-2**: Flow matching training loop (8-10 hours)
- **Day 3**: LoRA injection and saving (4-5 hours)
- **Day 4-5**: Testing Phase 3-4 (8-10 hours)

### Week 3: Optimization and Validation (Phases 5-6)
- **Day 1-2**: Memory optimization (6-8 hours)
- **Day 3-4**: End-to-end testing and debugging (8-10 hours)
- **Day 5**: Documentation and cleanup (4-6 hours)

**Total**: ~60-75 hours (15-20 working days)

---

## 13. References

### Internal Documentation
- `backend/core/training/MODEL_ARCHITECTURES.md` - SD/SDXL architecture reference
- `backend/core/models/zimage_transformer.py` - Z-Image Transformer implementation
- `backend/core/pipeline.py` - Z-Image inference pipeline

### External References
- **Flow Matching**: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
- **Qwen3**: Qwen model documentation (Hugging Face)
- **musubi-tuner**: Z-Image training implementation reference
  - `src/musubi_tuner/zimage_train_network.py` - Z-Image training script
  - `src/musubi_tuner/networks/lora_zimage.py` - Z-Image LoRA network (Transformer-only, Text Encoder frozen)
- **ai-toolkit**: General diffusion model training patterns
  - `extensions_built_in/sd_trainer/config/train.example.yaml` - Example config with `train_text_encoder: false` default
- **PEFT**: Hugging Face Parameter-Efficient Fine-Tuning library

---

## 14. Appendix: Flow Matching Mathematical Details

### 14.1 Forward Process

**Continuous-time interpolation** (t ∈ [0, 1]):

```
x_t = t * x_1 + (1 - t) * x_0
```

Where:
- `x_0`: Noise (sampled from N(0, I))
- `x_1`: Data (clean latents)
- `x_t`: Interpolated state at time t

### 14.2 Velocity Field

**Velocity** (direction of flow):

```
v_t = dx_t/dt = x_1 - x_0
```

### 14.3 Training Objective

**Loss function**:

```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||^2 ]
```

Where:
- `v_θ`: Velocity prediction network (Transformer)
- `t ~ U[0, 1]`: Uniformly sampled timestep
- `x_0 ~ N(0, I)`: Sampled noise
- `x_1`: Ground truth data (latents)

### 14.4 Comparison to DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| **Timesteps** | Discrete (0-1000) | Continuous [0, 1] |
| **Forward process** | q(x_t \| x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I) | x_t = t·x_1 + (1-t)·x_0 |
| **Prediction target** | Noise ε | Velocity v = x_1 - x_0 |
| **Loss** | E[\|\|ε - ε_θ\|\|²] | E[\|\|v - v_θ\|\|²] |
| **Denoising** | x_{t-1} = f(x_t, ε_θ, t) | x_{t-dt} = x_t + v_θ·dt |

---

**Document Status**: ✅ Complete
**Next Action**: Begin Phase 1 implementation (Model Detection and Loading)
