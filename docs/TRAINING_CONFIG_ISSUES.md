# Training YAML Configuration - Issue Tracking

**Date**: 2025-12-01
**Version**: 1.0.0

---

## 問題の概要

YAMLトレーニング設定に複数の問題が発見されました。このドキュメントでは各問題の詳細と修正計画を記載します。

---

## 1. すぐに実装すべき問題

### 1.1 `max_step_saves_to_keep` の未実装 🔴 **HIGH PRIORITY**

**問題**:
- YAMLに `save.max_step_saves_to_keep: 10` が出力されている
- 実際の実装がなく、古いチェックポイントが自動削除されない
- ディスク容量を圧迫する可能性

**影響**: ディスク容量不足のリスク

**実装方法**:
1. `lora_trainer.py` の `save_checkpoint()` メソッドに古いチェックポイント削除ロジックを追加
2. ファイル名から step 番号を抽出して昇順ソート
3. 最新 N 個を残して古いファイルを削除

**参考**: 要件定義書 Line 212

---

### 1.2 `cache_latents_to_disk` のデフォルト値が誤っている 🔴 **HIGH PRIORITY**

**問題**:
- 要件定義書 Line 220: `cache_latents_to_disk: true` がデフォルト
- 実際にはデフォルトは `false` であるべき（メモリ内キャッシュが基本）

**影響**:
- ディスクI/O増加
- トレーニング速度低下の可能性

**修正箇所**:
- ✅ `training_config.py` Line 44: `cache_latents_to_disk: bool = False` (完了)
- ❌ `routes.py` Line 2616: `cache_latents_to_disk: bool = True` → `False` に変更

**要件定義書の更新**: Line 220 も `false` に修正

---

### 1.3 `noise_scheduler` のデフォルトが誤っている 🔴 **CRITICAL**

**問題**:
- 現在のデフォルト: `flowmatch`
- **SDXLには flowmatch 推論のモデルは存在しない**
- 現時点では epsilon 推論なので `ddpm` または `euler` を使うべき

**影響**: トレーニングが正しく動作しない可能性

**修正箇所**:
- `training_config.py` Line 134: `noise_scheduler: flowmatch` → `ddpm` に変更
- UIで選択可能にする（`ddpm`, `euler`, `euler_a`, etc.）

**要件定義書の参照**: Line 236

---

### 1.4 Sample生成パラメータがYAMLに反映されていない 🟡 **MEDIUM PRIORITY**

**問題**:
- UIで設定した `sample_steps` と `sample_cfg_scale` が YAML に反映されていない
- 要件定義書 Line 285: `sample_steps: 20` と `guidance_scale: 4` がハードコード

**影響**: サンプル生成のカスタマイズができない

**修正箇所**:
- `training_config.py` Line 172-173: ハードコード値をパラメータ化
- `routes.py`: `sample_steps`, `sample_cfg_scale` を YAML生成時に渡す

---

## 2. 実装の確認が必要な項目

### 2.1 `gradient_accumulation_steps` の実装確認 🟡 **MEDIUM PRIORITY**

**問題**:
- YAML に `gradient_accumulation_steps: 1` が出力されている
- 実際に実装されているか不明

**確認方法**:
1. `lora_trainer.py` でgradient accumulationのロジックを検索
2. 未実装なら実装、または YAML から削除

**要件定義書の参照**: Line 230

---

### 2.2 `ema_config` の実装確認 🟡 **MEDIUM PRIORITY**

**問題**:
- YAML に `ema_config.use_ema: true` が出力されている
- EMA (Exponential Moving Average) が実装されているか不明

**確認方法**:
1. `lora_trainer.py` で EMA のロジックを検索
2. 未実装なら将来実装予定としてYAMLに残す

**要件定義書の参照**: Line 248-250

---

## 3. 設計上の問題

### 3.1 `model.quantize` フィールドの意味が不明 🟢 **LOW PRIORITY**

**問題**:
- `model.quantize: false` が出力されている
- dtype は別途 `train.weight_dtype` で指定しているため重複

**確認事項**:
- ai-toolkit での `quantize` の意味を確認
- LoRATrainer での量子化対応を確認
- 不要なら削除、必要なら用途を明確化

**要件定義書の参照**: Line 271

---

### 3.2 `vae_dtype` の配置が統一性に欠ける 🟢 **LOW PRIORITY**

**問題**:
- 現在: `model.vae_dtype: fp16`
- 他のdtype設定: `train.weight_dtype`, `train.dtype`, `train.output_dtype`
- 統一性のため `train` セクションに移動した方が良い

**修正案**:
```yaml
train:
  weight_dtype: bf16
  dtype: bf16
  output_dtype: fp32
  vae_dtype: fp16  # ← ここに移動
```

**要件定義書の参照**: Line 272

---

### 3.3 `sample.neg` フィールドが不要 🟢 **LOW PRIORITY**

**問題**:
- `sample.neg: ""` が存在
- `sample.prompts` 内の各プロンプトに `negative` フィールドがあるため重複

**修正案**:
- `sample.neg` を削除
- 各プロンプトの `negative` フィールドのみ使用

**要件定義書の参照**: Line 281

---

## 4. 将来実装予定（現時点では保留）

### 4.1 データセット個別オプション（未実装、YAMLには残す）

**項目**:
- `caption_ext: txt`
- `caption_dropout_rate: 0.05`
- `shuffle_tokens: false`
- `datasets[].resolution: [512, 768, 1024]`

**現状**:
- YAMLに出力されているが、実装未完了
- 将来実装予定のため削除しない

**参照**: ローカルの ai-toolkit (`d:\celll1\devs-test\ai-toolkit`)

---

## 5. Min-SNR Gamma実装（完了待ち）

**Status**: ✅ 実装完了、コミット待ち

**変更内容**:
- Loss計算に Min-SNR gamma weighting を追加
- Prediction type (epsilon, v_prediction, sample) 対応
- デフォルト値: `min_snr_gamma: 5.0`

**参照**:
- `lora_trainer.py`: `compute_snr()`, `apply_snr_weight()`, `get_target_from_prediction_type()`
- `training_config.py`: Line 154
- `routes.py`: Line 2633

---

## 6. 修正優先度

| 優先度 | 項目 | 理由 |
|--------|------|------|
| 🔴 CRITICAL | `noise_scheduler: flowmatch` → `ddpm` | SDXL に flowmatch モデルは存在しない |
| 🔴 HIGH | `max_step_saves_to_keep` 実装 | ディスク容量圧迫のリスク |
| 🔴 HIGH | `cache_latents_to_disk` デフォルト修正 | パフォーマンス影響 |
| 🟡 MEDIUM | `gradient_accumulation_steps` 実装確認 | YAML出力されているが動作不明 |
| 🟡 MEDIUM | `ema_config` 実装確認 | YAML出力されているが動作不明 |
| 🟡 MEDIUM | Sample生成パラメータのYAML反映 | カスタマイズできない |
| 🟢 LOW | `model.quantize` 用途確認 | 設計の明確化 |
| 🟢 LOW | `vae_dtype` の配置変更 | 統一性向上 |
| 🟢 LOW | `sample.neg` 削除 | 重複フィールド |

---

## 7. 実装計画

### Phase 1: CRITICAL修正（今すぐ）
1. ✅ `cache_latents_to_disk` デフォルト変更（完了）
2. ❌ `routes.py` の `cache_latents_to_disk` デフォルト修正
3. ❌ `noise_scheduler: ddpm` に変更
4. ❌ `max_step_saves_to_keep` 実装

### Phase 2: 実装確認（次）
1. ❌ `gradient_accumulation_steps` 実装確認
2. ❌ `ema_config` 実装確認
3. ❌ Sample生成パラメータのYAML反映

### Phase 3: 設計改善（後で）
1. ❌ `model.quantize` 用途確認・削除検討
2. ❌ `vae_dtype` 配置変更
3. ❌ `sample.neg` 削除

### Phase 4: Min-SNR Gamma（完了待ち）
1. ✅ 実装完了
2. ❌ コミット

---

**Document Version**: 1.0.0
**Last Updated**: 2025-12-01
