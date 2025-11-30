# Training Feature - Requirements Document

**Project**: SushiUI - Stable Diffusion WebUI
**Feature**: Model Training (Fine-tuning & LoRA)
**Date**: 2025-12-01
**Version**: 2.0.0
**Status**: ✅ **Phase 1 Complete** - Core training functionality implemented

---

## 1. Overview

### 1.1 Purpose

Enable users to train/fine-tune Stable Diffusion models (SD1.5/SDXL) directly from the WebUI using the dataset management system with **custom LoRA trainer implementation**.

### 1.2 Implementation Status

**Completed Features**:
- ✅ Custom LoRA trainer (independent implementation, no ai-toolkit dependency)
- ✅ Dataset integration with existing Dataset Management
- ✅ Training run management (start, stop, resume, monitor)
- ✅ Hyperparameter configuration UI
- ✅ Training progress monitoring (WebSocket real-time updates)
- ✅ Gradient checkpointing for VRAM optimization
- ✅ Mixed precision training (FP16, BF16, FP8)
- ✅ Latent caching to reduce VRAM usage
- ✅ Aspect ratio bucketing
- ✅ Multi-resolution training
- ✅ Debug VRAM profiling

**In Progress**:
- ⏳ Full model fine-tuning (all parameters)
- ⏳ Sample generation during training
- ⏳ Advanced monitoring UI

### 1.3 Supported Training Methods

| Method | Description | Use Case | VRAM Requirement (Optimized) |
|--------|-------------|----------|------------------------------|
| **LoRA** | Low-Rank Adaptation (efficient) | Character/style learning | **8-10GB** (batch_size=4, gradient checkpointing) |
| **Full Fine-tuning** | Train all model parameters | Complete model adaptation | 24GB+ (planned) |

---

## 2. Architecture

### 2.1 Training Backend

**Technology Stack**:
- **Framework**: Custom implementation using diffusers + PyTorch
- **Base Library**: diffusers (Hugging Face)
- **Optimization**: torch, bitsandbytes (AdamW8bit)
- **Config Format**: YAML (custom format, not ai-toolkit)

**Why Custom Implementation?**:
- Direct control over VRAM optimization
- Flexible gradient checkpointing
- Custom mixed precision support
- Better integration with SushiUI architecture
- No external subprocess overhead

**Integration**:
```
SushiUI Backend (FastAPI)
    ↓
LoRATrainer (Python class, backend/core/lora_trainer.py)
    ↓
diffusers pipeline (GPU)
    ↓
Model checkpoints → training/{run_name}/
```

### 2.2 Database Schema

**Training Database** (`training.db`):

```sql
-- Training runs
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER,  -- References datasets.db
    run_name VARCHAR(255) UNIQUE,
    training_method VARCHAR(50),  -- 'lora', 'full_finetune'
    base_model_path VARCHAR(500),

    -- Config
    config_yaml TEXT,  -- Full training config YAML

    -- Status
    status VARCHAR(50),  -- 'pending', 'running', 'paused', 'completed', 'failed'
    progress FLOAT,  -- 0.0 - 1.0
    current_step INTEGER,
    total_steps INTEGER,

    -- Performance metrics
    loss FLOAT,
    learning_rate FLOAT,

    -- Output
    output_dir VARCHAR(500),

    -- Logs
    log_file VARCHAR(500),
    error_message TEXT,

    -- Timestamps
    created_at DATETIME,
    started_at DATETIME,
    completed_at DATETIME,
    updated_at DATETIME
);
```

**Datasets Database** (`datasets.db`) - Referenced by training runs.

---

## 3. VRAM Optimization

### 3.1 Achieved Results

**Before Optimizations**:
- UNet forward pass: **31.22 GB**
- Total VRAM usage: **39-44 GB**

**After Optimizations**:
- UNet forward pass: **6.92 GB** (-78%!!!)
- Total VRAM usage: **8-10 GB** (-75-80%!!!)

### 3.2 Optimization Techniques

| Technique | VRAM Reduction | Performance Cost | Status |
|-----------|----------------|------------------|--------|
| **Gradient Checkpointing** | -24 GB (-78%) | +10-20% training time | ✅ Enabled by default |
| **Latent Caching** | -4 GB | None (faster!) | ✅ Enabled by default |
| **Mixed Precision (BF16)** | -15% | None | ✅ Enabled by default |
| **VAE on CPU** | -4 GB | None (with cache) | ✅ Enabled with caching |
| **AdamW8bit Optimizer** | -2 GB | Minimal | ✅ Default optimizer |
| **Batch duplication cleanup** | -10 GB | None | ✅ Implemented |

### 3.3 Precision Settings (VRAM Control)

**Available dtypes**:
- **Weight dtype**: fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2
- **Training dtype**: fp16, bf16, fp8_e4m3fn, fp8_e5m2 (for autocast)
- **Output dtype**: fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2 (for safetensors)
- **VAE dtype**: fp16 (default, SDXL VAE works fine), fp32, bf16

**Recommended settings for SDXL**:
```yaml
train:
  weight_dtype: bf16      # Model weights
  dtype: bf16             # Training/activation dtype
  output_dtype: fp32      # Safetensors output (highest precision)
  mixed_precision: true   # Enable autocast
model:
  vae_dtype: fp16         # VAE-specific (SDXL VAE works with fp16)
```

### 3.4 Debug VRAM Profiling

Enable detailed VRAM logging for troubleshooting:
```yaml
train:
  debug_vram: true  # Default: false
```

Output example:
```
[VRAM] After loading models to GPU
  Allocated: 6.57 GB
  Reserved:  6.86 GB
  Peak:      6.57 GB

[VRAM] [train_step] Before UNet forward
  Allocated: 6.47 GB
  Reserved:  6.78 GB

[VRAM] [train_step] After UNet forward
  Allocated: 6.92 GB  ← Only 0.45 GB increase (was 24 GB before!)
  Reserved:  7.42 GB
```

---

## 4. Training Configuration

### 4.1 LoRA Training Parameters

**Basic Settings**:
```yaml
job: character_lora_v1
config:
  name: character_lora_v1
  process:
    - type: sd_trainer
      training_folder: D:\celll1\webui_cl\training\character_lora_v1
      device: cuda:0
      trigger_word: ''

      # Network (LoRA)
      network:
        type: lora
        linear: 16         # LoRA rank (default: 16)
        linear_alpha: 16   # LoRA alpha (default: same as rank)

      # Saving
      save:
        dtype: float16
        save_every: 100
        max_step_saves_to_keep: 10

      # Dataset
      datasets:
        - folder_path: M:\dataset_working\character\otogibara_era
          caption_ext: txt
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true  # Enable latent caching
          resolution:
            - 512
            - 768
            - 1024

      # Training
      train:
        batch_size: 4
        steps: 1000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true  # CRITICAL for VRAM reduction

        # Noise scheduler
        noise_scheduler: flowmatch

        # Optimizer
        optimizer: adamw8bit
        lr: 0.0001
        unet_lr: 0.0001              # U-Net specific learning rate
        text_encoder_lr: 0.0001      # Text encoder LR (if trained)
        text_encoder_1_lr: 0.0001    # CLIP-L LR (SDXL)
        text_encoder_2_lr: 0.0001    # CLIP-G LR (SDXL)
        lr_scheduler: constant

        # EMA
        ema_config:
          use_ema: true
          ema_decay: 0.99

        # Precision (VRAM optimization)
        dtype: bf16                  # Training/activation dtype
        weight_dtype: bf16           # Model weight dtype
        output_dtype: fp32           # Output latent dtype
        mixed_precision: true        # Enable autocast
        debug_vram: false            # Disable verbose VRAM logging

        # Bucketing (aspect ratio support)
        enable_bucketing: true
        base_resolutions:
          - 1024
          - 1280
        bucket_strategy: resize      # resize, crop, random_crop
        multi_resolution_mode: max   # max, random

      # Model
      model:
        name_or_path: D:\celll1\webui_cl\models\IL02_0926_000405271.safetensors
        is_flux: false
        quantize: false
        vae_dtype: fp16              # VAE-specific dtype

      # Sample (not yet implemented)
      sample:
        sampler: flowmatch
        sample_every: 100
        width: 1024
        height: 1024
        prompts: []
        neg: ''
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20
```

### 4.2 Bucketing System

**Purpose**: Support multiple aspect ratios in a single training run

**Configuration**:
- `base_resolutions`: List of base sizes (e.g., [1024, 1280])
- `bucket_strategy`:
  - `resize`: Resize to fit bucket (preserves content)
  - `crop`: Center crop to bucket
  - `random_crop`: Random crop to bucket
- `multi_resolution_mode`:
  - `max`: Assign to largest resolution that fits
  - `random`: Randomly assign to compatible resolutions

**Example buckets for base_resolutions=[1024, 1280]**:
```
1024x1024: 3 images
1088x896: 3 images
1152x832: 7 images
832x1152: 21 images
768x1280: 3 images
... (26 unique buckets total)
```

---

## 5. API Endpoints

### 5.1 Training Runs

```python
# Create new training run
POST   /training/runs
Request: {
  run_name: string,
  dataset_id: number,
  training_method: "lora" | "full_finetune",
  base_model_path: string,
  # ... training parameters
}

# List all runs
GET    /training/runs
Response: TrainingRun[]

# Get run details
GET    /training/runs/{id}
Response: TrainingRun

# Delete run
DELETE /training/runs/{id}

# Start training
POST   /training/runs/{id}/start
Response: { status: "started" }

# Stop training
POST   /training/runs/{id}/stop
Response: { status: "stopped" }
```

### 5.2 WebSocket Updates

**Endpoint**: `/ws/training/{run_id}`

**Message format**:
```json
{
  "type": "training_update",
  "run_id": 123,
  "status": "running",
  "current_step": 523,
  "total_steps": 1000,
  "progress": 0.523,
  "loss": 0.0423,
  "learning_rate": 0.0001,
  "epoch": 2
}
```

---

## 6. Training Process Flow

### 6.1 Initialization

1. **Load models**:
   ```python
   # Text Encoders (CLIP-L, CLIP-G for SDXL)
   text_encoder = CLIPTextModel.from_pretrained(...)
   text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(...)

   # U-Net
   unet = UNet2DConditionModel.from_pretrained(...)

   # VAE (with vae_dtype)
   vae = AutoencoderKL.from_pretrained(..., torch_dtype=vae_dtype)
   ```

2. **Apply LoRA layers** (560 layers for SDXL U-Net):
   ```python
   # Target: to_q, to_k, to_v, to_out.0 in attention blocks
   lora_layer = LoRALinearLayer(original_module, rank, alpha)
   ```

3. **Enable gradient checkpointing**:
   ```python
   unet.enable_gradient_checkpointing()  # CRITICAL for VRAM
   ```

4. **Setup optimizer** (AdamW8bit):
   ```python
   optimizer = bnb.optim.AdamW8bit(lora_parameters, lr=learning_rate)
   ```

### 6.2 Training Loop

1. **Bucketing**: Assign images to aspect ratio buckets
2. **Batch creation**: Group images by bucket (same resolution per batch)
3. **Latent caching**: Load pre-encoded latents from disk (VAE stays on CPU)
4. **Text encoding**: Encode captions with CLIP-L and CLIP-G
5. **Training step**:
   ```python
   # Add noise
   noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

   # Forward pass (with autocast for mixed precision)
   with torch.autocast(device_type='cuda', dtype=training_dtype):
       model_pred = unet(noisy_latents, timesteps, text_embeddings, ...)

   # Loss calculation (always FP32)
   loss = F.mse_loss(model_pred.float(), noise.float())

   # Backward pass
   loss.backward()

   # Optimizer step
   optimizer.step()
   ```
6. **Cleanup**: Explicitly delete intermediate tensors to free VRAM
7. **Checkpoint saving**: Save LoRA weights every N steps
8. **WebSocket update**: Send progress to frontend

### 6.3 Memory Management

**Critical cleanups** to prevent VRAM leaks:
```python
# After torch.cat()
batched_latents = torch.cat(batch_latents, dim=0)
del batch_latents  # ← CRITICAL

# After train_step()
del batched_latents, batched_text_embeddings, batched_pooled_embeddings

# Inside train_step()
del noise, noisy_latents, model_pred, loss, added_cond_kwargs
```

---

## 7. Implementation Files

### 7.1 Backend Structure

```
backend/
├── api/
│   ├── routes.py                    # Training API endpoints
│   └── generation_utils.py          # Shared utilities
├── core/
│   ├── lora_trainer.py              # Main LoRA trainer (1500+ lines)
│   ├── train_runner.py              # Training process manager
│   ├── training_config.py           # YAML config generator
│   ├── bucketing.py                 # Aspect ratio bucketing
│   ├── latent_cache.py              # Latent caching system
│   └── vram_optimization.py         # VRAM utilities (planned)
├── database/
│   ├── models.py                    # SQLAlchemy models
│   └── connection.py                # Multi-DB connection
└── utils/
    └── image_utils.py               # Image processing
```

### 7.2 Frontend Structure

```
frontend/src/
├── app/
│   └── training/
│       └── page.tsx                 # Training page
└── components/
    └── training/
        ├── TrainingList.tsx         # List of training runs
        ├── TrainingConfig.tsx       # Configuration form (main UI)
        ├── TrainingMonitor.tsx      # Live training monitor
        └── TrainingLogs.tsx         # Real-time logs viewer
```

---

## 8. Performance Benchmarks

### 8.1 VRAM Usage (SDXL LoRA)

| Configuration | VRAM (Before) | VRAM (After) | Reduction |
|---------------|---------------|--------------|-----------|
| Batch size 1, no optimizations | 44 GB | - | - |
| + Gradient checkpointing | - | 10 GB | **-77%** |
| + Latent caching | - | 8 GB | **-82%** |
| Batch size 4, all optimizations | - | 8-10 GB | **-75-80%** |

### 8.2 Training Speed

**Hardware**: RTX 4090 (24GB)

| Dataset Size | Steps | Batch Size | Time (Optimized) |
|--------------|-------|------------|------------------|
| 50 images | 1000 | 4 | ~15 minutes |
| 100 images | 2000 | 4 | ~30 minutes |
| 500 images | 5000 | 4 | ~2 hours |

**Note**: Gradient checkpointing adds ~10-20% training time but enables much larger batch sizes.

---

## 9. Future Enhancements

### 9.1 Planned Features

- [ ] **Sample generation during training**: Preview quality at each checkpoint
- [ ] **Full fine-tuning support**: Train entire U-Net + Text Encoders
- [ ] **Advanced monitoring**: Loss curves, LR schedules, sample galleries
- [ ] **Checkpoint management**: Delete old checkpoints, export best checkpoint
- [ ] **Resume training**: Continue from any checkpoint
- [ ] **Training presets**: One-click configs for common use cases

### 9.2 Advanced VRAM Optimizations (Future)

- [ ] **CPU offloading**: Move text encoders to CPU during U-Net forward
- [ ] **Quantization**: INT8/FP8 quantization for weights
- [ ] **Flash Attention**: Memory-efficient attention (xformers)
- [ ] **Torch compile**: JIT compilation for speed

---

## 10. Troubleshooting

### 10.1 Out of Memory (OOM)

**Solutions**:
1. Enable gradient checkpointing (should be enabled by default)
2. Reduce batch size to 1
3. Enable latent caching
4. Use FP8 precision (experimental)
5. Reduce base resolutions in bucketing

### 10.2 Slow Training

**Solutions**:
1. Enable latent caching (speeds up data loading)
2. Use larger batch size (if VRAM allows)
3. Reduce sample generation frequency
4. Disable debug_vram logging

### 10.3 VRAM Debugging

Enable VRAM profiling:
```yaml
train:
  debug_vram: true
```

Check output for memory spikes at each training step.

---

## 11. Credits and References

- **diffusers**: Hugging Face diffusers library
- **bitsandbytes**: 8-bit optimizer implementation
- **ai-toolkit** (ostris): Inspiration for training architecture
- **kohya-ss/sd-scripts**: Reference for FP8 implementation

---

**Document Version**: 2.0.0
**Last Updated**: 2025-12-01
**Implementation Status**: ✅ Phase 1 Complete (Core LoRA training functional)
