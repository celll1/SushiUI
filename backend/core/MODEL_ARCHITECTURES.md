# Model Architecture Reference for Training

このドキュメントは、SushiUI LoRA Trainerがサポートするモデルアーキテクチャの詳細をまとめたものです。
新しいアーキテクチャを追加する際や、既存実装をメンテナンスする際の参考としてください。

---

## 目次

1. [Stable Diffusion 1.5 (SD1.5)](#stable-diffusion-15-sd15)
2. [Stable Diffusion XL (SDXL)](#stable-diffusion-xl-sdxl)
3. [アーキテクチャの検出方法](#アーキテクチャの検出方法)
4. [共通仕様](#共通仕様)

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

## アーキテクチャの検出方法

### U-Net Configからの検出

```python
# SDXL detection
is_sdxl = hasattr(unet.config, "addition_embed_type")
```

**理由**:
- SDXLのU-Netには `addition_embed_type` という設定パラメータがある
- この値が `"text_time"` の場合、time_idsとtext_embedsが必要

### Pipelineからの検出

```python
# Check for text_encoder_2
is_sdxl = hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None
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

両アーキテクチャで同じ:

```python
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
```

これらはU-NetのAttention層内のLinear層に対応します。

### Optimizer

推奨: **AdamW 8bit** (VRAM削減のため)

```python
import bitsandbytes as bnb

optimizer = bnb.optim.AdamW8bit(
    trainable_params,
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
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

Mean Squared Error (MSE) between predicted noise and actual noise:

```python
loss = F.mse_loss(
    model_pred.float(),
    noise.float(),
    reduction="mean"
)
```

---

## 将来のアーキテクチャ追加ガイド

新しいアーキテクチャ (SD3, FLUX, etc.) を追加する際は:

1. **このドキュメントに新しいセクションを追加**
   - コンポーネント構成
   - U-Net入力パラメータ
   - Text encoding方法
   - 特殊な条件付け (added_cond_kwargs等)

2. **`lora_trainer.py`の修正箇所**
   - `__init__`: モデルロード処理
   - `encode_prompt`: Text encoding処理
   - `train_step`: added_cond_kwargsの準備
   - `unet_forward_with_lora`: U-Net呼び出し

3. **検出ロジックの追加**
   - U-Net configやpipeline属性から判定
   - `is_sd3`, `is_flux` などのフラグを追加

4. **テスト**
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

### SushiUI参照実装
- `backend/core/custom_sampling.py`: SDXL time_ids生成ロジック
- `backend/core/prompt_chunking.py`: SDXLデュアルtext encoder処理
- `backend/core/pipeline.py`: モデルタイプ検出ロジック

### 外部参照
- [ai-toolkit by ostris](https://github.com/ostris/ai-toolkit): SDXL/FLUX対応の学習ツール
- [kohya_ss](https://github.com/bmaltais/kohya_ss): SD1.5/SDXL学習ツール

---

**最終更新**: 2025-11-30
**対応アーキテクチャ**: SD1.5, SDXL
