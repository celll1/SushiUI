# SDXL/SD regularization

**作成日**: 2026-01-02
**目的**: SNR/Energy RegularizationとTimestep SamplingのSDXL/SD対応を記録

---

## エグゼクティブサマリー

Z-Image専用だったSNR Regularization、Energy Regularization、およびTimestep Samplingを**SDXL/SD1.5にも実装**しました。

**実装内容**:
1. ✅ **Timestep Sampling**: 連続分布からのtimestep sampling（Uniform実装済み、Normal/LogNormal拡張可能）
2. ✅ **SNR Regularization**: 周波数領域の過剰デノイズ抑制（DDPM向けに適応）
3. ✅ **Energy Regularization**: 空間領域のエネルギー保存（DDPM向けに適応）

**主な変更点**:
- `BaseTrainer.train_step()`: Regularization計算を追加
- Timestep変換: DDPM離散timesteps ↔ 連続timesteps [0,1]
- 最適化: `predicted_latent_for_reg`の再利用で計算削減

---

## 1. 実装の背景

### 1.1 なぜSDXLにも必要か？

**従来の認識**:
- Flow Matching: 決定論的フロー → 過剰デノイズが発生しやすい
- DDPM: 確率的拡散 → 過剰デノイズは発生しにくい

**実際の問題**:
> 「確率論的な分布であっても過剰デノイズの問題は発生しえます」（ユーザー指摘）

**理論的根拠**:
1. **SNR問題**: DDPMでも低timestep（高SNR）で予測が不安定になる
   - Min-SNR Gammaは**損失の重み付け**（間接的対処）
   - SNR Regularizationは**SNR差の直接抑制**（直接的対処）

2. **Energy問題**: DDPMでも予測latentのエネルギーが元画像から逸脱する可能性
   - v-predictionは信号とノイズの線形結合（エネルギー保存は暗黙的）
   - Energy Regularizationは**エネルギー比を明示的に学習**（直接的対処）

3. **Timestep Sampling**: 重要なtimestep範囲を重点的に学習
   - DDPM: 低timestepが重要（詳細な特徴）
   - Normal分布で低timestep重視 → 効率的な学習

---

## 2. 実装詳細

### 2.1 Timestep Sampling

**実装箇所**: `backend/core/training/base_trainer.py:1439-1452`

**変更内容**:
```python
if timesteps is None:
    if self.timestep_sampler is not None:
        # Use timestep sampler: sample from [0, 1] then scale to discrete timesteps
        timesteps_continuous = self.timestep_sampler.sample(batch_size, self.device)
        timesteps = (timesteps_continuous * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = timesteps.clamp(0, self.noise_scheduler.config.num_train_timesteps - 1)
    else:
        # Legacy behavior: sample uniformly from [0, num_train_timesteps)
        timesteps = torch.randint(...)
```

**動作**:
1. `timestep_sampler.sample()`: 連続値 `[0, 1]` を生成
2. DDPM離散timestepsにスケーリング: `t_discrete = t_continuous * num_train_timesteps`
3. クランプ: `[0, num_train_timesteps - 1]`

**対応sampler**:
- ✅ `UniformTimestepSampler`: 均一分布（デフォルト）
- ⏳ `NormalTimestepSampler`: 正規分布（将来実装）
- ⏳ `LogNormalTimestepSampler`: 対数正規分布（将来実装）
- ⏳ `BetaTimestepSampler`: ベータ分布（将来実装）

**YAML設定**:
```yaml
timestep_sampler: "uniform"  # または null（legacy動作）
timestep_min: 0.0            # 最小timestep
timestep_max: 1.0            # 最大timestep
```

---

### 2.2 SNR Regularization for DDPM

**実装箇所**: `backend/core/training/base_trainer.py:1569-1578`

**変更内容**:
```python
# SNR regularization (周波数領域の過剰デノイズ抑制)
if self.snr_regularization_loss is not None:
    # Convert discrete timesteps to continuous [0, 1] for regularization
    timesteps_continuous = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
    snr_reg_loss = self.snr_regularization_loss(
        predicted_latent_for_reg,
        latents,
        timesteps_continuous
    )
    regularization_loss = regularization_loss + snr_reg_loss
```

**Timestep変換**:
- DDPM: 離散timesteps `[0, num_train_timesteps)` (例: 0-999)
- Regularization: 連続timesteps `[0, 1]`
- 変換式: `t_continuous = t_discrete / num_train_timesteps`

**Predicted Latent計算**:
```python
if prediction_type == "epsilon":
    predicted_latent_for_reg = (noisy_latents - sqrt_one_minus_alpha_bar * model_pred) / sqrt_alpha_bar
elif prediction_type == "v_prediction":
    predicted_latent_for_reg = sqrt_alpha_bar * noisy_latents - sqrt_one_minus_alpha_bar * model_pred
elif prediction_type == "sample":
    predicted_latent_for_reg = model_pred
```

**DDPMでの効果**:
- Low timestep (t < 0.3): SNR差が大きくなりやすい
- Regularizationでpenalty適用 → SNR差を抑制
- 結果: 過剰デノイズ防止、詳細保存

**YAML設定**:
```yaml
snr_regularization_weight: 0.1  # 0.0 = disabled
snr_penalty_mode: "relu"        # relu, squared, abs
```

---

### 2.3 Energy Regularization for DDPM

**実装箇所**: `backend/core/training/base_trainer.py:1580-1589`

**変更内容**:
```python
# Energy regularization (空間領域のエネルギー保存)
if self.energy_regularization_loss is not None:
    # Convert discrete timesteps to continuous [0, 1] for regularization
    timesteps_continuous = timesteps.float() / self.noise_scheduler.config.num_train_timesteps
    energy_reg_loss = self.energy_regularization_loss(
        predicted_latent_for_reg,
        latents,
        timesteps_continuous
    )
    regularization_loss = regularization_loss + energy_reg_loss
```

**Energy Ratio計算**:
```python
energy_pred = torch.norm(predicted_latent, p=2, dim=[1,2,3])
energy_gt = torch.norm(ground_truth, p=2, dim=[1,2,3])
energy_ratio = energy_pred / (energy_gt + 1e-8)
```

**DDPMでの効果**:
- Low timestep: エネルギーが元画像から逸脱しやすい
- Regularizationでpenalty適用 → エネルギー比を1.0に近づける
- 結果: 色彩保存、過度な明度変化防止

**YAML設定**:
```yaml
energy_regularization_weight: 0.05  # 0.0 = disabled
energy_penalty_mode: "under"        # under, over, abs
```

---

### 2.4 最適化: Predicted Latentの再利用

**問題**: Regularization用とReconstruction Loss用で2回計算

**解決策**: 1回計算して再利用

**実装**:
```python
# Compute predicted latent once (used by both regularization losses)
predicted_latent_for_reg = None
if self.snr_regularization_loss is not None or self.energy_regularization_loss is not None:
    # Compute predicted latent (keep gradients for backprop)
    alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(...)
    # ... (計算)
    predicted_latent_for_reg = ...

# SNR regularization
if self.snr_regularization_loss is not None:
    snr_reg_loss = self.snr_regularization_loss(predicted_latent_for_reg, ...)

# Energy regularization
if self.energy_regularization_loss is not None:
    energy_reg_loss = self.energy_regularization_loss(predicted_latent_for_reg, ...)

# Reconstruction loss (reuse if available)
with torch.no_grad():
    if predicted_latent_for_reg is not None:
        predicted_latent_for_recon = predicted_latent_for_reg.detach()
    else:
        # Compute only if not already computed
        predicted_latent_for_recon = ...
```

**効果**:
- Regularization有効時: 1回計算（Regularization + Recon Loss）
- Regularization無効時: 1回計算（Recon Lossのみ）
- 従来: 2回計算（無駄）

---

## 3. Min-SNR Gamma vs SNR Regularization

### 3.1 Min-SNR Gamma（既存）

**実装箇所**: `backend/core/training/base_trainer.py:1538-1542`

**計算**:
```python
if self.min_snr_gamma > 0:
    loss_per_sample_weighted = apply_snr_weight(loss_per_sample, timesteps, self.noise_scheduler, self.min_snr_gamma)
```

**効果**:
```python
SNR(t) = α²_t / σ²_t
weight = min(SNR(t), γ) / SNR(t)
weighted_loss = loss * weight
```

**特徴**:
- 損失の重み付け（間接的）
- High SNR → weight小 → 学習抑制
- Low SNR → weight大 → 学習促進

### 3.2 SNR Regularization（新規）

**実装箇所**: `backend/core/training/losses/snr_regularization.py`

**計算**:
```python
snr_pred = mean(predicted_latent²) / variance(predicted_latent)
snr_gt = mean(ground_truth²) / variance(ground_truth)
snr_diff = snr_pred - snr_gt

if penalty_mode == "relu":
    penalty = max(0, snr_diff)  # SNR増加のみペナルティ
loss_reg = penalty²
```

**特徴**:
- SNR差の直接抑制（直接的）
- Low timestep (t < 0.3)で適用
- SNR増加（過剰デノイズ）にペナルティ

### 3.3 両方を併用する場合

**推奨設定**:
```yaml
# Min-SNR Gamma: 全timestepで損失バランス
min_snr_gamma: 5.0

# SNR Regularization: 低timestepで過剰デノイズ防止
snr_regularization_weight: 0.1
snr_penalty_mode: "relu"
```

**効果**:
- Min-SNR Gamma: 学習の安定化（全体的）
- SNR Regularization: 過剰デノイズ防止（局所的）
- 相補的な効果

---

## 4. v-prediction vs Energy Regularization

### 4.1 v-prediction（既存）

**定義**:
```
v = α_t * ε - σ_t * x₀
```

**効果**:
- ノイズと信号の線形結合
- エネルギー保存は暗黙的
- CFG rescaleで安定化（`guidance_rescale=0.7`）

### 4.2 Energy Regularization（新規）

**計算**:
```python
energy_ratio = ||predicted_latent||₂ / ||ground_truth||₂

if penalty_mode == "under":
    penalty = max(0, 1.0 - energy_ratio)  # エネルギー損失のみペナルティ
elif penalty_mode == "over":
    penalty = max(0, energy_ratio - 1.0)  # エネルギー増加のみペナルティ
elif penalty_mode == "abs":
    penalty = abs(energy_ratio - 1.0)     # 両方ペナルティ
```

**効果**:
- エネルギー比を明示的に1.0に近づける
- 色彩保存、明度維持
- 過度な変化防止

### 4.3 両方を併用する場合

**推奨設定**:
```yaml
# v-prediction model
prediction_type: "v_prediction"

# Energy Regularization: エネルギー保存を強化
energy_regularization_weight: 0.05
energy_penalty_mode: "under"  # エネルギー損失を防ぐ
```

**効果**:
- v-prediction: ノイズ/信号バランス
- Energy Regularization: エネルギー保存の明示的学習
- 相補的な効果

---

## 5. 使用例

### 5.1 SDXL Full Fine-tuning

**YAML設定例**:
```yaml
model_type: "sdxl"
model_path: "/path/to/sdxl_base.safetensors"

# Timestep sampling
timestep_sampler: "uniform"
timestep_min: 0.0
timestep_max: 1.0

# Min-SNR (損失重み付け)
min_snr_gamma: 5.0

# SNR Regularization (過剰デノイズ防止)
snr_regularization_weight: 0.1
snr_penalty_mode: "relu"

# Energy Regularization (エネルギー保存)
energy_regularization_weight: 0.05
energy_penalty_mode: "under"

# 基本設定
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1e-6
num_epochs: 10
```

**期待される効果**:
1. Min-SNR Gamma: 学習の安定化
2. SNR Regularization: 詳細保存、過剰デノイズ防止
3. Energy Regularization: 色彩保存、過度な明度変化防止

### 5.2 SD1.5 LoRA Training

**YAML設定例**:
```yaml
model_type: "sd"
model_path: "/path/to/sd15_base.safetensors"

# Timestep sampling (重要なtimestep重視 - 将来実装)
# timestep_sampler: "normal"
# timestep_min: 0.0
# timestep_max: 1.0

# Min-SNR (必須)
min_snr_gamma: 5.0

# Regularization (オプション、品質向上)
snr_regularization_weight: 0.05  # LoRAは軽めに
energy_regularization_weight: 0.02

# 基本設定
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 1e-4
lora_rank: 16
```

**期待される効果**:
1. LoRAでも過剰デノイズを防ぐ
2. 軽いweightで適用（LoRAは変化が小さいため）

---

## 6. トラブルシューティング

### 6.1 Regularization lossが大きすぎる

**症状**:
```
[Training] Epoch 1/10, Step 10/1000 | Loss: 0.5234 (MSE: 0.0123, SNR Reg: 0.5000, Energy Reg: 0.0111)
```

**原因**: Regularization weightが大きすぎる

**解決策**:
```yaml
# Weightを下げる
snr_regularization_weight: 0.01  # 0.1 → 0.01
energy_regularization_weight: 0.005  # 0.05 → 0.005
```

### 6.2 Regularization効果が見られない

**症状**:
- Regularization lossが常に0に近い
- 過剰デノイズが改善されない

**原因**:
1. Weightが小さすぎる
2. Penalty modeが不適切

**解決策**:
```yaml
# Weightを上げる
snr_regularization_weight: 0.2  # 0.1 → 0.2

# Penalty modeを変更
snr_penalty_mode: "squared"  # relu → squared (より強いpenalty)
```

### 6.3 VRAM不足

**症状**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**原因**: Predicted latent計算で追加VRAM使用

**解決策**:
1. Regularizationを無効化
   ```yaml
   snr_regularization_weight: 0.0
   energy_regularization_weight: 0.0
   ```

2. Batch sizeを下げる
   ```yaml
   batch_size: 1
   gradient_accumulation_steps: 8  # 累積で補う
   ```

3. Gradient checkpointingを有効化
   ```yaml
   gradient_checkpointing: true
   ```

---

## 7. 理論的背景

### 7.1 DDPMでの過剰デノイズ

**問題**:
- DDPMは確率的拡散プロセスだが、決定論的サンプラー（DDIM等）では過剰デノイズが発生
- 特に低timestep（高SNR）で不安定

**原因**:
1. **SNR不均衡**: 低timestepでSNRが非常に高い
2. **エネルギー逸脱**: 予測latentのL2ノルムが元画像から逸脱

**従来の対策**:
- Min-SNR Gamma: 損失重み付けで間接的に対処
- v-prediction: ノイズ/信号バランスで暗黙的に対処

**新しい対策**:
- SNR Regularization: SNR差を直接抑制
- Energy Regularization: エネルギー比を直接学習

### 7.2 Flow Matching vs DDPM

| 項目 | Flow Matching | DDPM |
|------|---------------|------|
| 過剰デノイズ発生率 | **高い** | 中程度 |
| 理由 | 決定論的フロー | 確率的拡散（決定論的サンプラーでは発生） |
| Regularization必要性 | **必須** | 推奨 |
| 既存対策 | なし | Min-SNR, v-pred |

**結論**: DDPMでも過剰デノイズは発生するため、Regularizationは有効

---

## 8. まとめ

### 8.1 実装された機能

| 機能 | Z-Image | SDXL/SD | 実装箇所 |
|------|---------|---------|----------|
| **Timestep Sampling** | ✅ | ✅ | `train_step()` L1439-1452 |
| **SNR Regularization** | ✅ | ✅ | `train_step()` L1569-1578 |
| **Energy Regularization** | ✅ | ✅ | `train_step()` L1580-1589 |

### 8.2 推奨設定

**SDXL Full FT**:
```yaml
min_snr_gamma: 5.0
snr_regularization_weight: 0.1
energy_regularization_weight: 0.05
```

**SD1.5 LoRA**:
```yaml
min_snr_gamma: 5.0
snr_regularization_weight: 0.05
energy_regularization_weight: 0.02
```

**Z-Image**:
```yaml
# Min-SNRは不要（Flow Matching）
snr_regularization_weight: 1.0   # DDPMより高め
energy_regularization_weight: 0.1
```

### 8.3 期待される効果

1. **過剰デノイズ防止**: 詳細保存、自然な画質
2. **エネルギー保存**: 色彩維持、過度な明度変化防止
3. **学習効率向上**: 重要なtimestep重視（Timestep Sampling拡張時）

---

## 参考資料

### 関連ファイル

- `backend/core/training/base_trainer.py`: メイン実装
- `backend/core/training/losses/snr_regularization.py`: SNR Regularization
- `backend/core/training/losses/energy_regularization.py`: Energy Regularization
- `backend/core/training/timestep_sampler.py`: Timestep Sampler framework

### 関連ドキュメント

- Historical parity investigation is retained only in the local working area.
- Flow Matching theory notes are retained only in the local research area.
- `backend/core/training/API_REFERENCE.md`: トレーニングAPI仕様

---

**作成者**: Claude (Anthropic)
**レビュー**: 実装完了、テスト推奨
