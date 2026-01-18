# Z-Image Training: Batch Size and VRAM Usage Analysis

## Problem Statement

Z-Image exhibits significantly different VRAM scaling behavior compared to SD/SDXL when increasing batch size during training:

- **SD/SDXL**: VRAM increase is limited when batch size grows
- **Z-Image**: VRAM increases dramatically with batch size
  - batch_size=1: ~18GB
  - batch_size=4: ~44GB (2.4x increase)

This document analyzes the root causes and provides recommendations.

---

## Root Cause Analysis

### 1. **List[Tensor] Input Format (Primary Cause)**

**SD/SDXL Implementation** (`train_step`, Line 1254-1265):
```python
# Batched tensor format - efficient memory layout
model_pred = self.unet(
    noisy_latents,        # [B, 4, H, W] - contiguous batch dimension
    timesteps,            # [B]
    text_embeddings,      # [B, 77, 768]
    added_cond_kwargs=... # SDXL micro-conditioning
).sample
```

**Z-Image Implementation** (`train_step_zimage`, Line 1465-1514):
```python
# List format - batch dimension is decomposed
x_list = []
for i in range(batch_size):
    latent = noisy_latents[i]  # [C, H, W] - split batch
    latent_4d = latent.unsqueeze(1)  # [C, 1, H, W] - add frame dim
    x_list.append(latent_4d)

cap_feats_list = [caption_embeds[i] for i in range(batch_size)]

model_pred = self.transformer(
    x=x_list,             # List[Tensor] - separate memory allocations
    t=timesteps,          # [B]
    cap_feats=cap_feats_list,  # List[Tensor]
)
```

**Why This Matters**:
- Z-Image Transformer requires **List[Tensor]** format (architectural constraint from official implementation)
- Each tensor in the list is allocated separately in memory
- **Memory contiguity is lost** - prevents PyTorch optimizations:
  - **Kernel fusion**: Operations on contiguous memory can be fused into single kernels
  - **Memory pooling**: Efficient reuse of memory blocks for batched operations
  - **Cache locality**: CPU/GPU caches work better with contiguous data
- Internal tensor reconstruction overhead during forward/backward passes
- Gradient checkpointing becomes less efficient with fragmented memory

**Impact Estimation**:
- Batch size increase from 1→4 results in **4 separate memory regions** instead of 1 contiguous region
- Each region requires individual memory allocation, tracking, and deallocation
- Memory fragmentation increases, leading to higher peak VRAM usage

---

### 2. **16-Channel Latents**

**SD/SDXL**:
```python
noise = torch.randn_like(latents)  # [B, 4, H, W] - 4 channels
```

**Z-Image**:
```python
noise = torch.randn_like(latents)  # [B, 16, H, W] - 16 channels
```

- Z-Image uses **16 latent channels** (4x more than SD/SDXL)
- **However**, for practical training with similar-sized images:
  - SD/SDXL: VAE scale factor = 8 → 1024×1024 image → 128×128 latent
  - Z-Image: VAE scale factor = 16 → 1024×1024 image → 64×64 latent
  - **Actual latent memory usage is similar** due to compensating spatial resolution

**Verdict**: Not a significant factor in practice when training on same-sized images.

---

### 3. **Large Text Embeddings**

**SD/SDXL**:
```python
text_embeddings: torch.Tensor  # [B, 77, 768] (SD1.5)
pooled_embeddings: torch.Tensor  # [B, 1280] (SDXL)
# SDXL also has text_encoder_2 embeddings: [B, 77, 1280]
# Total: ~2 embeddings per sample
```

**Z-Image**:
```python
caption_embeds: torch.Tensor  # [B, seq_len, 2560]
caption_mask: torch.Tensor    # [B, seq_len] (bool)
# seq_len is variable (avg 200-512 tokens)
```

- Z-Image: **2560 dimensions** (1.3x larger than SDXL's 1280+768=2048)
- **Note**: SDXL has dual text encoders + pooled embeddings, so the gap is smaller than it appears
- Variable sequence length with padding (see §5)

**Verdict**: Higher dimensionality provides better expressiveness for Z-Image. The difference vs SDXL is moderate when accounting for SDXL's dual encoders.

---

### 4. **Transformer Intermediate Activations (Major Issue)**

**Z-Image Transformer Architecture**:
- 30 Flow DiT blocks (each with Self-Attention, Cross-Attention, Feed-Forward)
- Hidden dimension: **3840**
- Each attention layer generates Q, K, V tensors: `[batch, seq_len, hidden_dim]`
- Cross-Attention references caption embeddings: `[batch, seq_len, 2560]`

**Activation Memory Estimation** (batch_size=4, 1024×1024 image):
- Spatial tokens per image: 64 × 64 = 4096 tokens
- Single attention layer (Q, K, V combined): `4 × 4096 × 3840 × 2 bytes ≈ 120MB`
- 30 layers with attention: `120MB × 3 (Q/K/V) × 30 layers = 10.8GB`

**Gradient Checkpointing Limitation**:
- Gradient checkpointing is enabled but **less effective** with List[Tensor] format
- Activations cannot be fully released due to fragmented memory layout
- Recomputation overhead is higher for Transformer attention (expensive QKV projections, softmax, matmul)

**Verdict**: **This is a major problem**. The combination of large intermediate activations and inefficient gradient checkpointing due to List format leads to high VRAM usage.

---

### 5. **Caption Padding**

**Batching Implementation** (Line 3382-3410 in `train()` method):
```python
# Find max sequence length in batch
max_seq_len = max(emb.shape[0] for emb in batch_caption_embeds)

# Pad all embeddings and masks to max_seq_len
for emb, mask in zip(batch_caption_embeds, batch_caption_masks):
    seq_len = emb.shape[0]
    if seq_len < max_seq_len:
        pad_size = max_seq_len - seq_len
        padded_emb = torch.cat([emb, torch.zeros((pad_size, 2560))], dim=0)
        padded_mask = torch.cat([mask, torch.zeros(pad_size, dtype=bool)], dim=0)
```

- All captions in a batch are padded to the **longest sequence** in that batch
- As batch size increases, average padding amount increases (longer max sequence in batch)

**Verdict**: **Not a major issue**. Training with varying padding lengths is actually beneficial for robustness, as the model learns to handle different sequence lengths without being sensitive to padding.

---

### 6. **Flow Matching vs Diffusion**

**SD/SDXL (Diffusion)**:
```python
noise = torch.randn_like(latents)
timesteps = torch.randint(0, 1000, (batch_size,))
noisy_latents = scheduler.add_noise(latents, noise, timesteps)
```

**Z-Image (Flow Matching)**:
```python
timesteps = torch.rand(batch_size, device=self.device)  # continuous [0, 1]
noise = torch.randn_like(latents)
t = timesteps[:, None, None, None]
noisy_latents = (1.0 - t) * noise + t * latents  # linear interpolation
```

**Verdict**: Flow Matching uses continuous time but has minimal direct impact on VRAM usage.

---

## Summary: Primary Causes of High VRAM Scaling

### Critical Issues (Require Attention):

1. **List[Tensor] Input Format** ⚠️ **PRIMARY CAUSE**
   - Breaks memory contiguity
   - Prevents PyTorch kernel fusion and memory pooling optimizations
   - Architectural constraint from Z-Image official implementation

2. **Transformer Intermediate Activations** ⚠️ **MAJOR ISSUE**
   - 30 layers × 3840 hidden dim × attention mechanisms
   - Gradient checkpointing less effective with List format
   - High memory overhead for activation storage and recomputation

### Minor or Non-Issues:

3. **16-Channel Latents** ✓ **NOT A PROBLEM**
   - Compensated by smaller spatial resolution (VAE scale factor 16 vs 8)
   - Actual memory usage similar to SD/SDXL for same-sized images

4. **Large Text Embeddings** ✓ **ACCEPTABLE**
   - 2560 dimensions provide better expressiveness
   - Gap vs SDXL is moderate (SDXL has dual encoders + pooled)

5. **Caption Padding** ✓ **NOT A PROBLEM**
   - Training with variable padding improves robustness
   - Memory overhead is minimal compared to other factors

---

## Recommendations

### Current Best Practice:
- **Use small batch sizes (1-2)** for Z-Image training
- Due to List[Tensor] architectural constraint, batch size scaling is inherently limited

### Potential Future Improvements:

1. **Batched Tensor Format Conversion** (Requires Transformer Modification)
   - Convert List[Tensor] → Batched Tensor `[B, C, F, H, W]`
   - Requires modification of Z-Image Transformer API (large-scale change)
   - Would enable PyTorch memory optimizations

2. **Enhanced Gradient Checkpointing**
   - More aggressive checkpointing strategy
   - Selective activation recomputation (checkpoint critical layers only)

3. **FP8 Mixed Precision**
   - Quantize activations to FP8 (current: weights only)
   - Cache text embeddings in FP8 format

4. **Dynamic Sequence Length Batching** (Low Priority)
   - Group images with similar caption lengths into same batch
   - Minimize padding overhead (though impact is minor)

---

## Conclusion

The dramatic VRAM increase when scaling batch size in Z-Image training is primarily caused by:

1. **List[Tensor] input format** breaking memory contiguity and preventing PyTorch optimizations
2. **Large intermediate activations** in 30-layer Transformer with inefficient gradient checkpointing

These are **architectural constraints** inherited from the Z-Image official implementation. The current best practice is to use **batch_size=1-2** for Z-Image training.

Significant improvements would require modifying the Transformer's input API to accept batched tensors, which is a large-scale change requiring validation against the official implementation.

---

**Document Version**: 1.0
**Last Updated**: 2025-12-11
**Related Code**: `backend/core/training/lora_trainer.py` (train_step, train_step_zimage)
