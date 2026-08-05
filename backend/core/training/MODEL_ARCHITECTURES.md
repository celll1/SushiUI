# Model Architecture Reference for Training

このドキュメントは、SushiUI LoRA Trainerがサポートするモデルアーキテクチャの詳細をまとめたものです。
新しいアーキテクチャを追加する際や、既存実装をメンテナンスする際の参考としてください。

---

## 目次

1. [Stable Diffusion 1.5 (SD1.5)](#stable-diffusion-15-sd15)
2. [Stable Diffusion XL (SDXL)](#stable-diffusion-xl-sdxl)
3. [Z-Image](#z-image)
4. [アーキテクチャの検出方法](#アーキテクチャの検出方法)
5. [共通仕様](#共通仕様)
6. [トレーニングオプション](#トレーニングオプション)

---

## Stable Diffusion 1.5 (SD1.5)

### 概要
- **リリース**: 2022年
- **ベースモデル**: `runwayml/stable-diffusion-v1-5`
- **解像度**: 512x512（標準）

### コンポーネント構成

#### Text Encoder
- **モデル**: `CLIPTextModel` (OpenAI CLIP ViT-L/14)
- **トークナイザー**: `CLIPTokenizer`
- **最大トークン長**: 77
- **出力次元**: `[batch, 77, 768]`

#### U-Net
- **入力チャンネル**: 4 (latent space)
- **出力チャンネル**: 4
- **Attention層**: Cross-attention with text embeddings
- **Time embedding**: Sinusoidal position embeddings

#### VAE
- **モデル**: `AutoencoderKL`
- **ダウンスケール係数**: 8 (512x512 → 64x64 latents)
- **Latent channels**: 4
- **Scaling factor**: 0.18215

### Training用の入力

#### UNet Forward Pass
```python
unet(
    sample=noisy_latents,           # [B, 4, H/8, W/8]
    timestep=timesteps,              # [B]
    encoder_hidden_states=text_embeddings  # [B, 77, 768]
)
```

**必須パラメータ**:
- `sample`: ノイズが加えられたlatents
- `timestep`: 拡散ステップ (0 ~ num_train_timesteps)
- `encoder_hidden_states`: Text embeddingsから得られた特徴量

**不要なパラメータ**:
- `added_cond_kwargs`: SD1.5では不要

### Text Encoding

```python
# Tokenize
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)

# Encode
text_embeddings = text_encoder(
    text_inputs.input_ids.to(device)
)[0]  # Shape: [1, 77, 768]
```

### Noise Scheduler
- **タイプ**: `DDPMScheduler`
- **Beta schedule**: `"scaled_linear"`
- **Beta start**: 0.00085
- **Beta end**: 0.012
- **Timesteps**: 1000

---

## Stable Diffusion XL (SDXL)

### 概要
- **リリース**: 2023年
- **ベースモデル**: `stabilityai/stable-diffusion-xl-base-1.0`
- **解像度**: 1024x1024（標準）
- **主な変更点**: デュアルtext encoder、より大きなU-Net、micro-conditioning

### コンポーネント構成

#### Text Encoder 1
- **モデル**: `CLIPTextModel` (OpenAI CLIP ViT-L/14)
- **トークナイザー**: `CLIPTokenizer`
- **最大トークン長**: 77
- **出力次元**: `[batch, 77, 768]`

#### Text Encoder 2
- **モデル**: `CLIPTextModelWithProjection` (OpenCLIP ViT-bigG/14)
- **トークナイザー**: `CLIPTokenizer`
- **最大トークン長**: 77
- **出力次元**: `[batch, 77, 1280]`
- **Pooled output**: `[batch, 1280]` (projection layer経由)

#### U-Net
- **入力チャンネル**: 4 (latent space)
- **出力チャンネル**: 4
- **Attention層**: Cross-attention with concatenated text embeddings
- **Time embedding**: Sinusoidal + micro-conditioning (time_ids)
- **サイズ**: SD1.5の約3倍

#### VAE
- **モデル**: `AutoencoderKL`
- **ダウンスケール係数**: 8 (1024x1024 → 128x128 latents)
- **Latent channels**: 4
- **Scaling factor**: 0.13025 (SD1.5と異なる)

### Training用の入力

#### UNet Forward Pass
```python
unet(
    sample=noisy_latents,           # [B, 4, H/8, W/8]
    timestep=timesteps,              # [B]
    encoder_hidden_states=text_embeddings,  # [B, 77, 2048] (768+1280 concatenated)
    added_cond_kwargs={
        "text_embeds": pooled_embeddings,  # [B, 1280]
        "time_ids": add_time_ids           # [B, 6]
    }
)
```

**必須パラメータ**:
- `sample`: ノイズが加えられたlatents
- `timestep`: 拡散ステップ
- `encoder_hidden_states`: 2つのtext encoderの出力を連結したもの
- `added_cond_kwargs`: SDXLの追加条件

**added_cond_kwargs の詳細**:
- `text_embeds`: Text Encoder 2のpooled output (projection経由)
- `time_ids`: Micro-conditioning vector `[original_h, original_w, crop_top, crop_left, target_h, target_w]`

### Text Encoding

```python
# Text Encoder 1 (CLIP ViT-L)
text_inputs_1 = tokenizer(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)
text_embeddings_1 = text_encoder(
    text_inputs_1.input_ids.to(device)
)[0]  # Shape: [1, 77, 768]

# Text Encoder 2 (OpenCLIP ViT-bigG)
text_inputs_2 = tokenizer_2(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt",
)
encoder_output_2 = text_encoder_2(
    text_inputs_2.input_ids.to(device),
    output_hidden_states=True,
)
text_embeddings_2 = encoder_output_2.hidden_states[-2]  # Penultimate layer: [1, 77, 1280]
pooled_embeddings = encoder_output_2[0]  # Pooled output: [1, 1280]

# Concatenate embeddings
text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)  # [1, 77, 2048]
```

**重要な注意点**:
- Text Encoder 2は**Penultimate hidden state** (最後から2番目の層) を使用
- Pooled embeddingsは最終出力 (`encoder_output_2[0]`) から取得

### Time IDs (Micro-conditioning)

```python
# Calculate from image/latent size
latent_height, latent_width = latents.shape[2], latents.shape[3]
image_height, image_width = latent_height * 8, latent_width * 8

# Prepare time_ids
original_size = (image_height, image_width)
crops_coords_top_left = (0, 0)  # No cropping for training
target_size = (image_height, image_width)

add_time_ids = list(original_size + crops_coords_top_left + target_size)
add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
# Result: [1, 6] tensor like [1024, 1024, 0, 0, 1024, 1024]
```

**Time IDs の構成** (6要素):
1. `original_height`: 元画像の高さ
2. `original_width`: 元画像の幅
3. `crop_top`: クロップ開始位置 (上からのピクセル数)
4. `crop_left`: クロップ開始位置 (左からのピクセル数)
5. `target_height`: 出力画像の高さ
6. `target_width`: 出力画像の幅

**トレーニング時の推奨値**:
- クロップは使用しない: `crops_coords_top_left = (0, 0)`
- Original size = Target size = 実際の画像サイズ

### Noise Scheduler
- **タイプ**: `DDPMScheduler`
- **Beta schedule**: `"scaled_linear"`
- **Beta start**: 0.00085
- **Beta end**: 0.012
- **Timesteps**: 1000

---

## Z-Image

### 概要
- **リリース**: 2024年
- **アーキテクチャ**: DiT (Diffusion Transformer)
- **解像度**: 1024x1024（標準）
- **主な特徴**: Qwen3 text encoder、Flow Matching（DDPMではない）

### コンポーネント構成

#### Text Encoder
- **モデル**: `Qwen/Qwen2.5-1.5B-Instruct` (AutoModelForCausalLM)
- **トークナイザー**: `Qwen2Tokenizer`
- **最大トークン長**: 可変（chat template使用）
- **出力次元**: `[batch, seq_len, 1536]`
- **Penultimate layer**: `hidden_states[-2]` を使用

#### Transformer
- **モデル**: `ZImageTransformer2DModel`
- **アーキテクチャ**: DiT (Diffusion Transformer)
- **入力チャンネル**: 16 (FLUX VAE latent space)
- **出力チャンネル**: 16
- **Attention層**: `ZImageAttention` (to_q, to_k, to_v, to_out[0])

#### VAE
- **モデル**: FLUX VAE (`black-forest-labs/FLUX.1-dev`)
- **ダウンスケール係数**: 8 (1024x1024 → 128x128 latents)
- **Latent channels**: 16 (SDXLの4倍)
- **Scaling factor**: 0.3611 (SDXLと異なる)

### Training用の入力

#### Transformer Forward Pass
```python
# Add frame dimension: [B, C, H, W] → [B, C, 1, H, W]
noisy_latents = noisy_latents.unsqueeze(2)

model_pred = transformer(
    hidden_states=noisy_latents,           # [B, 16, 1, H/8, W/8]
    timestep=timesteps,                    # [B]
    encoder_hidden_states=prompt_embeds,   # [B, seq_len, 1536]
    encoder_attention_mask=attention_mask, # [B, seq_len]
    return_dict=False
)[0]

# Remove frame dimension: [B, C, 1, H, W] → [B, C, H, W]
model_pred = model_pred.squeeze(2)
```

**必須パラメータ**:
- `hidden_states`: ノイズが加えられたlatents (frame dimension付き)
- `timestep`: Flow matching timestep (0.0 ~ 1.0の連続値)
- `encoder_hidden_states`: Qwen3 embeddings
- `encoder_attention_mask`: Attention mask (padding対応)

**Z-Image固有の処理**:
- Frame dimension の追加/削除 (DiTアーキテクチャ要件)
- Attention mask の明示的な渡し (Qwen3の可変長出力対応)

### Text Encoding

```python
# Format with Qwen chat template
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

# Tokenize
text_inputs = tokenizer(
    formatted_prompt,
    padding="max_length",
    max_length=512,  # Configurable
    truncation=True,
    return_tensors="pt",
)

# Encode with penultimate layer (like SDXL TE2)
encoder_output = text_encoder(
    text_inputs.input_ids.to(device),
    attention_mask=text_inputs.attention_mask.to(device),
    output_hidden_states=True
)
prompt_embeds = encoder_output.hidden_states[-2]  # Penultimate layer: [B, seq_len, 1536]
attention_mask = text_inputs.attention_mask  # [B, seq_len]
```

**重要な注意点**:
- Chat templateを使用（Qwen3の標準フォーマット）
- Attention maskを取得・保持（Transformer forward passで使用）
- Penultimate layerを使用（SDXL TE2と同様の理由）

### Noise Process (Flow Matching)

```python
# Z-Imageは Flow Matching を使用 (DDPMではない)
# Timesteps: [0.0, 1.0] の連続値 (DDPMの [0, 1000] ではない)
noise_scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=1.0  # Z-Image default
)

# Loss calculation
loss = F.mse_loss(
    model_pred.float(),
    velocity.float(),  # velocity target (not noise target)
    reduction="mean"
)
```

**DDPM との違い**:
- **Prediction target**: Velocity（DDPMのepsilonではない）
- **Timestep range**: [0.0, 1.0] 連続値（DDPMの [0, 1000] 離散値ではない）
- **Min-SNR weighting**: 適用しない（Flow Matchingでは不要）

### LoRA Target Layers

**Transformer**:
```python
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]  # ZImageAttention modules
```

**Text Encoder** (Qwen3):
```python
# Text Encoder は通常凍結（Transformerのみ学習）
```

---

## アーキテクチャの検出方法

### U-Net Configからの検出

```python
# SDXL detection
is_sdxl = hasattr(unet.config, "addition_embed_type")

# Z-Image detection (via ModelLoader)
from core.model_loader import ModelLoader
model_type = ModelLoader.detect_model_type(model_path)
is_zimage = (model_type == "zimage")
```

**理由**:
- SDXLのU-Netには `addition_embed_type` という設定パラメータがある
- Z-ImageはModelLoaderの`detect_model_type()`で検出

### Pipelineからの検出

```python
# SDXL: Check for text_encoder_2
is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None

# Z-Image: Check for transformer
is_zimage = hasattr(pipeline, 'transformer') and pipeline.transformer is not None
```

### Safetensorsからのロード時

```python
try:
    # Try SDXL first
    pipeline = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    is_sdxl = True
except Exception:
    # Fall back to SD1.5
    pipeline = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    is_sdxl = False
```

---

## 共通仕様

### LoRA適用対象レイヤー

すべてのアーキテクチャで共通:

```python
# U-Net/Transformer attention layers
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
```

これらはAttention層内のLinear層に対応します。

### Optimizer

推奨オプション（VRAM削減のため）:

1. **AdamW 8bit** (bitsandbytes)
2. **AdamW 8bit RingBuffer** (musubi-tuner実装、Schedule-Free対応)
3. **Lion 8bit RingBuffer** (musubi-tuner実装、Schedule-Free対応)
4. **Adafactor** (メモリ効率最高、Fused Backward Pass対応)

```python
# Example: AdamW 8bit RingBuffer
from core.training.optimizers.adamw8bit_ringbuffer import AdamW8bitRingBuffer

optimizer = AdamW8bitRingBuffer(
    trainable_params,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
    schedule_free=True,  # Optional: Enable Schedule-Free training
    warmup_steps=100,     # Schedule-Free warmup
)
```

### Learning Rate Scheduler

```python
from diffusers.optimization import get_scheduler

lr_scheduler = get_scheduler(
    "constant",  # or "cosine", "linear"
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(
    trainable_parameters,
    max_norm=1.0
)
```

### Loss Function

すべてのアーキテクチャで共通（Z-Imageを除く）:

```python
# SD1.5/SDXL: Epsilon prediction
loss = F.mse_loss(
    model_pred.float(),
    noise.float(),
    reduction="mean"
)

# Z-Image: Velocity prediction (Flow Matching)
loss = F.mse_loss(
    model_pred.float(),
    velocity.float(),
    reduction="mean"
)
```

---

## トレーニングオプション

SushiUIの`BaseTrainer`は以下のオプションをサポートしています（すべてのアーキテクチャで共通）:

### Precision Settings

```python
weight_dtype: str = "fp16"       # Model weight dtype (fp16, fp32, bf16, fp8_e4m3fn, fp8_e5m2)
training_dtype: str = "fp16"     # Training/activation dtype (fp16, bf16, fp8_e4m3fn, fp8_e5m2)
output_dtype: str = "fp32"       # Output dtype for safetensors (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2)
vae_dtype: str = "fp16"          # VAE-specific dtype (fp16, fp32, bf16)
mixed_precision: bool = True     # Enable mixed precision training (autocast)
```

### VRAM Optimization

```python
use_flash_attention: bool = False    # Enable Flash Attention (faster, lower memory)
blocks_to_swap: int = 0               # Block Swap (CPU offloading, 0=disabled)
use_pinned_memory: bool = False       # Use pinned memory for Block Swap
num_optimizer_groups: int = 0         # Fused Optimizer Groups (0=disabled)
debug_vram: bool = False              # Enable detailed VRAM profiling
```

**Block Swap + Fused Optimizer Groups の制約**:
- Adafactor: `num_optimizer_groups=0`で使用（Fused Backward Pass）
- AdamW/Lion: `num_optimizer_groups > 0`で使用（FP32 state、VRAM増加）
- AdamW8bit/Lion8bit: **Block Swap使用時は`num_optimizer_groups=0`必須**（非互換のため）

### Loss Weighting

```python
min_snr_gamma: float = 5.0  # Min-SNR gamma (5.0推奨、0で無効化)
                            # Z-Imageでは使用しない（Flow Matchingのため）
```

### Prompt Chunking (SD/SDXL only)

```python
prompt_chunking_mode: str = "a1111"  # "a1111", "sd_scripts", "nobos"
max_prompt_chunks: int = 0           # 0 = unlimited
```

### Component-specific Learning Rates

```python
learning_rate: float = 1e-4              # Base learning rate
unet_lr: Optional[float] = None          # U-Net learning rate (default: learning_rate)
text_encoder_lr: Optional[float] = None  # Text Encoder learning rate (default: learning_rate)

# SDXL only:
text_encoder_1_lr: Optional[float] = None  # Text Encoder 1 LR
text_encoder_2_lr: Optional[float] = None  # Text Encoder 2 LR
```

### Optimizer Hyperparameters

```python
# Paging is part of the optimizer NAME (paged_adamw / paged_adamw8bit /
# paged_lion8bit), which is what OptimizerFactory dispatches on. There is no
# is_paged boolean.
optimizer_cautious: bool = False              # Cautious mode (C-AdamW, C-Lion)
optimizer_beta1: Optional[float] = None       # Beta1 (default: optimizer-specific)
optimizer_beta2: Optional[float] = None       # Beta2 (default: optimizer-specific)
optimizer_epsilon: Optional[float] = None     # Epsilon (default: optimizer-specific)
optimizer_weight_decay: Optional[float] = None  # Weight decay (default: optimizer-specific)
```

### Schedule-Free Options (RingBuffer optimizers only)

```python
optimizer_schedule_free: bool = False                      # Enable Schedule-Free training
optimizer_warmup_steps: int = 0                            # Warmup steps for Schedule-Free
optimizer_schedule_free_r: float = 0.0                     # Schedule-Free r parameter
optimizer_schedule_free_weight_lr_power: float = 2.0       # Weight LR power
optimizer_use_radam: bool = False                          # Use RAdam variant
```

### Gradient Checkpointing

すべてのモデルで自動的に有効化（`BaseTrainer.__init__`内）:
- U-Net/Transformer: `enable_gradient_checkpointing()`
- Text Encoder: `gradient_checkpointing_enable()`

---

## 将来のアーキテクチャ追加ガイド

新しいアーキテクチャ (SD3, FLUX, etc.) を追加する際は:

1. **このドキュメントに新しいセクションを追加**
   - コンポーネント構成
   - U-Net入力パラメータ
   - Text encoding方法
   - 特殊な条件付け (added_cond_kwargs等)

2. **Adapterクラスの作成**
   - `backend/core/training/adapters/` に新しいファイル作成
   - `BaseLoRAAdapter` または `BaseFullParameterAdapter` を継承
   - `apply_lora_to_unet()`, `encode_prompt()`, `train_step()` を実装

3. **BaseTrainerの修正箇所**
   - `_load_model_components()`: モデル検出ロジック追加
   - `is_xxx` フラグ追加 (例: `self.is_sd3`)
   - `_load_xxx_components()`: コンポーネントロード処理
   - `encode_prompt_xxx()`: Text encoding処理
   - `train_step_xxx()`: 学習ステップ処理

4. **検出ロジックの追加**
   - U-Net configやpipeline属性から判定
   - `is_sd3`, `is_flux` などのフラグを追加

5. **テスト**
   - 各アーキテクチャで学習が正常に動作するか確認
   - 生成されたLoRAが推論で使用可能か確認

---

## 参考資料

### SD1.5
- [Hugging Face Model Card](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/loading)

### SDXL
- [Hugging Face Model Card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL Training Guide](https://huggingface.co/docs/diffusers/en/training/sdxl)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)

### Z-Image
- [Z-Image Paper](https://arxiv.org/abs/2501.00000) (仮のリンク)
- SushiUI実装: `backend/core/models/zimage_transformer.py`

### SushiUI参照実装
- `backend/core/prompt_chunking.py`: SDXLデュアルtext encoder処理
- `backend/core/pipeline.py`: モデルタイプ検出ロジック
- `backend/core/training/base_trainer.py`: トレーニングオプション実装

### 外部参照
- [ai-toolkit by ostris](https://github.com/ostris/ai-toolkit): SDXL/FLUX対応の学習ツール
- [kohya_ss](https://github.com/bmaltais/kohya_ss): SD1.5/SDXL学習ツール
- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner): Block Swap実装、8bit RingBuffer optimizers

---

**最終更新**: 2026-01-08
**対応アーキテクチャ**: SD1.5, SDXL, Z-Image
