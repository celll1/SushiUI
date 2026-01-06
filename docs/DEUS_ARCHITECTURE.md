# DEUS Architecture Documentation

**DEUS**: **D**ual-**E**mbeddings **U**-Net **S**tructure

完全なアーキテクチャフロー図と技術仕様

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Complete Data Flow](#complete-data-flow)
3. [Component Details](#component-details)
4. [RoPE 2D Position Encoding](#rope-2d-position-encoding)
5. [Resolution Handling](#resolution-handling)
6. [Training and Inference](#training-and-inference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DEUS Architecture                               │
│                  (Dual-Embeddings U-Net Structure)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Text Prompt + Optional Image                                   │
│     │                                                                    │
│     ├───────────────────────────────────────────────────────────┐      │
│     │                                                            │      │
│     ▼                                                            ▼      │
│  ┌──────────────────────┐                           ┌────────────────┐ │
│  │  SigLIP-2 Text Enc   │                           │ SigLIP-2 Image │ │
│  │  (No token limit)    │                           │   Encoder      │ │
│  │  768 → 1152 dim      │                           │  224x224 → 1152│ │
│  └──────────┬───────────┘                           └───────┬────────┘ │
│             │                                               │          │
│             └───────────────────┬───────────────────────────┘          │
│                                 │                                      │
│                                 ▼                                      │
│                  ┌──────────────────────────────┐                     │
│                  │   Concatenate Embeddings     │                     │
│                  │   [batch, seq_len, 1152]     │                     │
│                  └──────────────┬───────────────┘                     │
│                                 │                                      │
│                                 │ Multi-Modal Conditioning             │
│                                 │                                      │
│     ┌───────────────────────────┼──────────────────────────┐          │
│     │                           │                           │          │
│     │                           ▼                           │          │
│     │              ┌────────────────────────┐               │          │
│     │              │    DEUS U-Net          │               │          │
│     │              │  (SDXL-based + RoPE)   │               │          │
│     │              └────────────┬───────────┘               │          │
│     │                           │                           │          │
│     │   Input Latents           │       Output Latents     │          │
│     │   [B, 16, H, W]           │       [B, 16, H, W]      │          │
│     │                           │                           │          │
│     ▼                           │                           ▼          │
│  ┌──────────┐                   │                    ┌──────────┐     │
│  │ FLUX VAE │                   │                    │ FLUX VAE │     │
│  │ Encoder  │                   │                    │ Decoder  │     │
│  │ 3→16 ch  │                   │                    │ 16→3 ch  │     │
│  └──────────┘                   │                    └──────────┘     │
│                                 │                                      │
│  Input Image                    │                  Generated Image    │
│  [B, 3, H*8, W*8]               │                  [B, 3, H*8, W*8]   │
│                                 │                                      │
│                    Diffusion Process                                   │
│                    (Euler, DPM++, etc.)                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

### 1. Text-to-Image (T2I) Generation

```
Input: Text Prompt
  │
  ├─► SigLIP-2 Text Encoder
  │     │
  │     ├─► Tokenize (no length limit)
  │     │     │
  │     │     └─► [batch, seq_len, 768]
  │     │
  │     └─► Text Transformer (12 layers)
  │           │
  │           └─► [batch, seq_len, 1152]
  │
  ├─► (Optional) SigLIP-2 Image Encoder
  │     │
  │     ├─► Resize to 224x224
  │     │     │
  │     │     └─► [batch, 3, 224, 224]
  │     │
  │     └─► Vision Transformer (27 layers)
  │           │
  │           └─► [batch, num_patches, 1152]
  │
  ├─► Concatenate Text + Image Embeddings
  │     │
  │     └─► [batch, total_seq_len, 1152]
  │
  ├─► Random Noise Latents
  │     │
  │     └─► [batch, 16, H_latent, W_latent]
  │           (H_latent = target_height / 8)
  │           (W_latent = target_width / 8)
  │
  ├─► Denoising Loop (steps = 1...T)
  │     │
  │     ├─► U-Net Forward
  │     │     │
  │     │     ├─► Input: noisy_latents, timestep, condition
  │     │     │     │
  │     │     │     ├─► Time Embedding
  │     │     │     │     └─► Sinusoidal [batch, 1280]
  │     │     │     │
  │     │     │     ├─► Input Projection
  │     │     │     │     └─► Conv2d: 16 → model_channels
  │     │     │     │
  │     │     │     ├─► RoPE 2D Position Encoding
  │     │     │     │     └─► Add positional info
  │     │     │     │
  │     │     │     ├─► Down Blocks (3 stages)
  │     │     │     │     │
  │     │     │     │     ├─► Stage 1: 384ch (×1)
  │     │     │     │     │     ├─► ResNet × 2
  │     │     │     │     │     ├─► CrossAttention × 10
  │     │     │     │     │     └─► Downsample (÷2)
  │     │     │     │     │
  │     │     │     │     ├─► Stage 2: 768ch (×2)
  │     │     │     │     │     ├─► ResNet × 2
  │     │     │     │     │     ├─► CrossAttention × 10
  │     │     │     │     │     └─► Downsample (÷2)
  │     │     │     │     │
  │     │     │     │     └─► Stage 3: 1536ch (×4)
  │     │     │     │           ├─► ResNet × 2
  │     │     │     │           ├─► CrossAttention × 10
  │     │     │     │           └─► (No downsample)
  │     │     │     │
  │     │     │     ├─► Mid Block
  │     │     │     │     ├─► ResNet
  │     │     │     │     ├─► CrossAttention
  │     │     │     │     └─► ResNet
  │     │     │     │
  │     │     │     ├─► Up Blocks (3 stages)
  │     │     │     │     │
  │     │     │     │     ├─► Stage 1: 1536ch
  │     │     │     │     │     ├─► Upsample (×2)
  │     │     │     │     │     ├─► Concat skip (sparse)
  │     │     │     │     │     ├─► ResNet × 2
  │     │     │     │     │     └─► CrossAttention × 10
  │     │     │     │     │
  │     │     │     │     ├─► Stage 2: 768ch
  │     │     │     │     │     ├─► Upsample (×2)
  │     │     │     │     │     ├─► Concat skip (sparse)
  │     │     │     │     │     ├─► ResNet × 2
  │     │     │     │     │     └─► CrossAttention × 10
  │     │     │     │     │
  │     │     │     │     └─► Stage 3: 384ch
  │     │     │     │           ├─► (No upsample)
  │     │     │     │           ├─► Concat skip (sparse)
  │     │     │     │           ├─► ResNet × 2
  │     │     │     │           └─► CrossAttention × 10
  │     │     │     │
  │     │     │     └─► Output Projection
  │     │     │           └─► Conv2d: model_channels → 16
  │     │     │
  │     │     └─► Predicted Noise [batch, 16, H, W]
  │     │
  │     └─► Scheduler Step
  │           └─► Update latents
  │
  └─► VAE Decode
        │
        └─► Output Image [batch, 3, H*8, W*8]
```

### 2. Sparse Skip Connections

```
Down Blocks                    Up Blocks
  │                              │
  ├─► Block 0 ──────────────────►│ (interval=0, connect)
  │                              │
  ├─► Block 1 ─────────X─────────│ (interval=1, skip)
  │                              │
  └─► Block 2 ──────────────────►│ (interval=2, connect)

Skip Connection Rule:
- Only connect blocks where index % interval == 0
- Reduces memory usage
- Maintains global context
- Based on DiC (Diffusion in Context) paper
```

---

## Component Details

### SigLIP-2 Multi-Modal Encoder

```python
# Text Encoder
Input: text string (no length limit)
  ├─► Tokenizer (SigLIP-2 specific)
  ├─► Embedding Layer (vocab_size → 768)
  ├─► Transformer Encoder (12 layers)
  │     ├─► Self-Attention (12 heads, 768 dim)
  │     ├─► FFN (768 → 3072 → 768)
  │     └─► LayerNorm
  └─► Output: [batch, seq_len, 1152]

# Image Encoder
Input: image [batch, 3, H, W]
  ├─► Resize to 224x224
  ├─► Patch Embedding (16x16 patches)
  │     └─► 14×14 = 196 patches
  ├─► Vision Transformer (27 layers)
  │     ├─► Self-Attention (16 heads, 1152 dim)
  │     ├─► FFN (1152 → 4304 → 1152)
  │     └─► LayerNorm
  └─► Output: [batch, 196, 1152]
```

### DEUS U-Net Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    U-Net Structure                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input: [B, 16, H, W]                                     │
│    │                                                       │
│    ├─► Conv_in (16 → model_channels)                      │
│    │                                                       │
│    ├─► RoPE 2D (add positional encoding)                  │
│    │                                                       │
│    ├─► Down Blocks                                        │
│    │     ├─► Stage 0: 384ch  (H×W)                       │
│    │     ├─► Stage 1: 768ch  (H/2×W/2)                   │
│    │     └─► Stage 2: 1536ch (H/4×W/4)                   │
│    │                                                       │
│    ├─► Mid Block (1536ch, H/4×W/4)                       │
│    │                                                       │
│    ├─► Up Blocks                                          │
│    │     ├─► Stage 0: 1536ch (H/4×W/4 → H/2×W/2)        │
│    │     ├─► Stage 1: 768ch  (H/2×W/2 → H×W)            │
│    │     └─► Stage 2: 384ch  (H×W → H×W)                │
│    │                                                       │
│    └─► Conv_out (model_channels → 16)                     │
│                                                            │
│  Output: [B, 16, H, W]                                    │
└────────────────────────────────────────────────────────────┘

Each Block:
  ├─► ResNet Block (×2)
  │     ├─► GroupNorm + SiLU
  │     ├─► Conv3×3
  │     ├─► Time Embedding injection
  │     └─► Residual connection
  │
  └─► Cross-Attention Block (×10)
        ├─► LayerNorm
        ├─► Self-Attention (spatial)
        ├─► Cross-Attention (to conditioning)
        └─► FFN (4× expansion)
```

### Model Variants

| Variant | Model Channels | Channel Mult | Res Blocks | Attn Heads | Transformer Depth | U-Net Params | Total Params |
|---------|----------------|--------------|------------|------------|-------------------|--------------|--------------|
| Small   | 320            | (1,2,4,4)    | 2          | 16         | 6                 | 1.23B        | 2.45B        |
| Medium  | 384            | (1,2,4,4)    | 2          | 24         | 10                | 2.43B        | 3.65B        |
| Large   | 448            | (1,2,4,4)    | 3          | 28         | 10                | 3.54B        | 4.76B        |

**Component Breakdown (Medium variant):**
- U-Net: 2.43B (66.6%)
- SigLIP-2 Text Encoder: 0.71B (19.4%)
- SigLIP-2 Image Encoder: 0.43B (11.7%)
- FLUX VAE: 0.08B (2.3%)
- **Total: 3.65B parameters**

---

## RoPE 2D Position Encoding

### Current Implementation

```python
class RoPE2D:
    """
    2D Rotary Position Embedding

    Applies sinusoidal positional encoding to spatial dimensions
    """

    def __init__(self, dim=320, base=10000):
        # Frequency bands: θ_i = 10000^(-2i/d)
        inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        # Result: [dim/2] frequency bands
        # Range: [1.0, 1e-4] (high to low frequency)

    def forward(self, x):  # x: [B, C, H, W]
        # Generate position indices
        pos_h = arange(H)  # [0, 1, 2, ..., H-1]
        pos_w = arange(W)  # [0, 1, 2, ..., W-1]

        # Compute angles: θ_i * pos
        freqs_h = outer(pos_h, inv_freq)  # [H, dim/2]
        freqs_w = outer(pos_w, inv_freq)  # [W, dim/2]

        # Sinusoidal encoding
        emb_h = cat([sin(freqs_h), cos(freqs_h)])  # [H, dim]
        emb_w = cat([sin(freqs_w), cos(freqs_w)])  # [W, dim]

        # Expand to 2D grid
        emb_h = emb_h[:, None, :].expand(H, W, dim)  # [H, W, dim]
        emb_w = emb_w[None, :, :].expand(H, W, dim)  # [H, W, dim]

        # Combine by addition
        emb_2d = emb_h + emb_w  # [H, W, dim]

        # Add to input
        return x + emb_2d.permute(2, 0, 1)[None]  # [B, C, H, W]
```

### Frequency Distribution

RoPE使用160個の周波数帯域（dim=320の場合、dim/2ペア）:

```
Frequency Index    Wavelength (positions)    Coverage at 128x128
─────────────────────────────────────────────────────────────────
0-10               256.0 - 102.4            0.5 - 1.25 cycles (Low)
10-50              102.4 - 10.2             1.25 - 12.5 cycles (Mid)
50-160             10.2 - 0.01              12.5 - 12800 cycles (High)
```

**問題点:**
- 低周波数（大域情報）のカバレッジが不足
- 高周波数（細部）が過剰
- 解像度変更時に周波数分布が適切にスケールしない

### Visualization Results

実行した可視化により以下が判明:

1. **周波数分布**: 低周波数帯域が少なく、高周波数が支配的
2. **解像度外挿**: 現在の実装は解像度変更時に一貫性が低い
3. **2Dパターン**: 加算による結合は対角方向にバイアスが生じる可能性

**可視化ファイル:**
- `docs/rope_analysis/rope_frequency_analysis.png`
- `docs/rope_analysis/rope_resolution_extrapolation.png`
- `docs/rope_analysis/rope_2d_patterns.png`
- `docs/rope_analysis/rope_2d_combination.png`
- `docs/rope_analysis/rope_resolution_consistency.png`

---

## Resolution Handling

### Current Behavior

```
Training Resolution: 128×128 latent (1024×1024 pixels)

Inference at Different Resolutions:
  64×64  (512×512)   → Under-sampling (fewer cycles)
  128×128 (1024×1024) → Exact match
  192×192 (1536×1536) → Over-sampling (more cycles)
  256×256 (2048×2048) → Heavy over-sampling
```

**問題:**
- 位置エンコーディングが解像度に依存
- 学習時と異なる解像度での生成時に不整合

### Proposed: Resolution-Adaptive RoPE

```python
class ResolutionAdaptiveRoPE2D:
    """
    Resolution-adaptive RoPE with frequency scaling

    Key idea: Normalize position indices by resolution ratio
    to maintain consistent positional information
    """

    def __init__(self, dim=320, base=10000, train_resolution=128):
        self.train_resolution = train_resolution
        inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):  # x: [B, C, H, W]
        # Resolution scaling factors
        scale_h = H / self.train_resolution
        scale_w = W / self.train_resolution

        # Normalized position indices
        pos_h = arange(H) / scale_h  # [0, 0.5, 1.0, ..., 127]
        pos_w = arange(W) / scale_w  #   (for 256×256 inference)

        # Rest same as before...
        freqs_h = outer(pos_h, inv_freq)
        # ...

        return x + emb_2d
```

**効果:**
- 解像度に依存せず一貫した位置情報
- 学習解像度の整数倍で最適な性能
- 任意解像度でもスムーズに補間

### Alternative: Frequency Band Adjustment

```python
class FrequencyAdaptiveRoPE2D:
    """
    Adjust frequency bands based on resolution

    Larger resolutions → Add lower frequencies
    Smaller resolutions → Remove high frequencies
    """

    def get_adaptive_inv_freq(self, H, W):
        scale = max(H, W) / self.train_resolution

        if scale > 1.0:
            # Larger resolution: extend to lower frequencies
            # Add bands with longer wavelengths
            extra_low_freq = 1.0 / (self.base ** (arange(-10, 0) / self.dim))
            inv_freq = cat([extra_low_freq, self.inv_freq])
        elif scale < 1.0:
            # Smaller resolution: remove high frequencies
            # Keep only bands with shorter wavelengths
            cutoff = int(self.dim * scale)
            inv_freq = self.inv_freq[:cutoff]
        else:
            inv_freq = self.inv_freq

        return inv_freq
```

---

## Training and Inference

### Training Pipeline

```
1. Load Dataset
     ├─► Images: [B, 3, H, W]
     └─► Captions: List[str]

2. Encode Inputs
     ├─► VAE Encode: images → latents [B, 16, H/8, W/8]
     ├─► SigLIP-2 Text: captions → embeddings [B, seq, 1152]
     └─► (Optional) SigLIP-2 Image: images → embeddings [B, 196, 1152]

3. Add Noise
     └─► latents + noise * sigma_t

4. U-Net Forward
     ├─► Input: noisy_latents, timestep, conditioning
     └─► Output: predicted_noise [B, 16, H/8, W/8]

5. Compute Loss
     └─► MSE(predicted_noise, true_noise)

6. Backprop & Update
     └─► Optimizer step
```

### Inference Pipeline

```
1. Encode Prompt
     ├─► Text: prompt → [B, seq, 1152]
     └─► (Optional) Image: ref_image → [B, 196, 1152]

2. Initialize Latents
     └─► Random noise [B, 16, H/8, W/8]

3. Denoising Loop (T steps)
     For t in [T, T-1, ..., 1]:
       ├─► U-Net: predict noise at timestep t
       ├─► Scheduler: compute x_{t-1} from x_t and noise
       └─► (Optional) Guidance: CFG scale

4. VAE Decode
     └─► latents [B, 16, H/8, W/8] → images [B, 3, H, W]

5. Post-process
     └─► Clip to [0, 1], convert to PIL
```

---

## Checkpoint Format

SDXL-style unified safetensors:

```
Checkpoint Structure:
├─ model.diffusion_model.*        # U-Net weights
│    ├─ conv_in.weight
│    ├─ down_blocks.*.*.weight
│    ├─ mid_block.*.weight
│    ├─ up_blocks.*.*.weight
│    └─ conv_out.weight
│
├─ conditioner.embedders.0.transformer.*  # Text Encoder
│    ├─ embeddings.*.weight
│    └─ encoder.layers.*.*.weight
│
├─ conditioner.embedders.1.model.*        # Image Encoder
│    ├─ vision_model.*.weight
│    └─ vision_model.post_layernorm.weight
│
└─ first_stage_model.*            # VAE
     ├─ encoder.*.weight
     └─ decoder.*.weight

Metadata:
  model_type: "deus"
  architecture: "DEUS (Dual-Embeddings U-Net Structure)"
  unet_variant: "medium"
  latent_channels: "16"
  context_dim: "1152"
```

---

## References

- **SDXL Architecture**: [Stability AI SDXL Paper](https://arxiv.org/abs/2307.01952)
- **RoPE**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **SigLIP-2**: [Sigmoid Loss for Language Image Pre-Training v2](https://arxiv.org/abs/2303.15343)
- **DiC (Diffusion in Context)**: [Sparse Skip Connections](https://arxiv.org/abs/2501.00603)
- **FLUX VAE**: Black Forest Labs FLUX.1 model

---

## Implementation Files

- `backend/core/models/unet_deus.py` - U-Net architecture
- `backend/core/models/siglip2_wrapper.py` - SigLIP-2 encoders
- `backend/core/models/flux_vae_wrapper.py` - FLUX VAE
- `backend/core/pipelines/pipeline_deus.py` - Inference pipeline
- `backend/core/models/checkpoint_utils.py` - Save/load utilities
- `backend/core/model_loader.py` - Model detection and loading
- `create_unified_checkpoint.py` - Checkpoint generation script

---

**Last Updated**: 2026-01-06
**Version**: 1.0
**Status**: Production Ready
