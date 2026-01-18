# Layer Pruning Tool

Z-Image Transformer層削減の検証ツール

## 概要

このツールは、Z-Image Transformerの層数を削減する際に、どの層を削除すべきかを検証します。

**目的**:
- オリジナルモデルと削減モデルの1-step推論出力の差を最小化
- 削除する層の組み合わせを探索
- 削減後のモデルをsafetensors形式で保存

## 使用方法

### 1. データセットから検証サンプルを抽出

```bash
python subapps/layer_pruning/extract_samples.py \
    --dataset-db datasets.db \
    --dataset-id "dataset_unique_id" \
    --num-samples 10 \
    --output samples.json
```

### 2. 層削減の探索

```bash
python subapps/layer_pruning/prune_layers.py \
    --model-path "M:\sushiUI\training\full_dezit\full_dezit_step_47058.safetensors" \
    --samples samples.json \
    --target-layers 20 \
    --strategy greedy \
    --output pruned_model.safetensors
```

**パラメータ**:
- `--model-path`: 入力モデルパス（safetensors）
- `--samples`: 検証サンプルのJSONファイル
- `--target-layers`: 削減後の層数（例: 20）
- `--strategy`: 探索戦略
  - `greedy`: 貪欲法（1層ずつ削除、最も影響が少ない層を選択）
  - `uniform`: 均等間隔で削除（例: 30層→20層なら3層に1層削除）
  - `skip_middle`: 中間層を優先的に削除
- `--output`: 出力モデルパス

### 3. 削減モデルの評価

```bash
python subapps/layer_pruning/evaluate_pruned.py \
    --original-model "M:\sushiUI\training\full_dezit\full_dezit_step_47058.safetensors" \
    --pruned-model pruned_model.safetensors \
    --samples samples.json
```

## 実装詳細

### 1-step推論ロス計算

データセットから画像を読み込み、以下の処理を行います：

1. **VAE Encoding**: 画像 → Latents
2. **Noise Scheduling**: `timestep=500` でノイズ追加
3. **1-step Denoising**: Transformer 1回の推論
4. **Loss計算**: オリジナルモデルとの出力差（MSE Loss）

### 貪欲法探索アルゴリズム

```
current_layers = [0, 1, 2, ..., 29]  # 30層
target_layers = 20

while len(current_layers) > target_layers:
    best_layer_to_remove = None
    best_loss = inf

    for layer_idx in current_layers:
        # この層を削除した場合のロスを計算
        temp_model = create_model_without_layer(layer_idx)
        loss = evaluate_1step_loss(temp_model, samples)

        if loss < best_loss:
            best_loss = loss
            best_layer_to_remove = layer_idx

    # 最も影響が少ない層を削除
    current_layers.remove(best_layer_to_remove)
    print(f"Removed layer {best_layer_to_remove}, loss: {best_loss}")
```

## 注意事項

- VRAM使用量: モデルロード（~10GB）+ VAE推論（~2GB）
- 処理時間: 貪欲法の場合、各イテレーションで全層評価が必要（30層→20層なら約10イテレーション）
- サンプル数: 10-50枚推奨（多すぎると時間がかかる、少なすぎると精度低下）

## 実装の制限事項

**現在の実装は簡易版です**：

1. **1-step推論の簡略化**:
   - Z-Imageの完全なforwardパスは複雑（patchify、RoPE、flow matching等）
   - 現在は簡易的なMSE lossで評価（`F.mse_loss(noisy_latents, latents)`）
   - **推奨**: 実際の推論パイプラインを使用してより正確な評価を行う

2. **評価の精度**:
   - 簡易実装では層削減の影響を正確に測定できない可能性
   - **改善案**: SushiUIの推論パイプラインを使用して実画像生成品質で評価

3. **層削減後のfine-tuning**:
   - このツールは層削除のみ実施（再学習は別途必要）
   - 削減後のモデルはそのままでは品質が大幅に低下する可能性
   - **推奨**: 削減後、SushiUIのLoRAトレーニング機能で品質回復

## 改善の方向性

### 現在の実装（簡易版）
```python
# 簡易的なMSE loss（transformerのforwardを使わない）
loss = F.mse_loss(noisy_latents, latents)
```

### 推奨される改善（完全な推論パス）
```python
# SushiUIの推論パイプラインを使用
from core.pipeline import DiffusionPipelineManager

# 完全な1-step推論
pipeline_manager = DiffusionPipelineManager()
pipeline_manager.load_model(...)

# 実際の推論結果を比較
original_output = pipeline_manager.generate_txt2img(params, steps=1)
pruned_output = pruned_pipeline_manager.generate_txt2img(params, steps=1)

# 画像レベルでの差分評価（LPIPS, SSIMなど）
loss = calculate_perceptual_loss(original_output, pruned_output)
```

### より高度な評価指標

1. **Perceptual Loss**: LPIPS（人間の知覚に近い評価）
2. **Feature-level Loss**: Transformer中間層の出力差
3. **Multi-step Evaluation**: 1-stepだけでなく、full inference（28 steps）での評価

## 使用上の推奨フロー

1. **簡易評価**: このツールで削減候補を特定
2. **実画像評価**: SushiUIで実際に画像生成して品質確認
3. **Fine-tuning**: 削減後のモデルでLoRAトレーニング
4. **最終評価**: 再学習後のモデルで品質検証

## 既知の問題

- `evaluate_model()` メソッドは簡易実装のため、正確な層削減効果を測定できない
- Z-ImageのBatchedZImageWrapperを使用した完全な推論が必要
- 現在はプレースホルダー実装のため、実際の使用には改善が必須
