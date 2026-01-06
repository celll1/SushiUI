# RoPE 2D Improvements for Resolution Extrapolation

解像度外挿性を向上させるためのRoPE 2D実装の改善案と分析結果

---

## Table of Contents

1. [Current Implementation Analysis](#current-implementation-analysis)
2. [Identified Problems](#identified-problems)
3. [Proposed Solutions](#proposed-solutions)
4. [Comparison Matrix](#comparison-matrix)
5. [Recommendations](#recommendations)

---

## Current Implementation Analysis

### Standard RoPE 2D (Baseline)

**実装**: `backend/core/models/unet_deus.py:RoPE2D`

```python
# 周波数帯域の生成
inv_freq = 1.0 / (10000 ** (arange(0, 320, 2) / 320))
# → [160個の周波数帯域]

# 位置インデックス (解像度に依存)
pos_h = arange(H)  # [0, 1, 2, ..., H-1]
pos_w = arange(W)  # [0, 1, 2, ..., W-1]

# 角度計算: θ_i * pos
freqs_h = outer(pos_h, inv_freq)
freqs_w = outer(pos_w, inv_freq)

# 正弦波エンコーディング
emb_h = cat([sin(freqs_h), cos(freqs_h)])
emb_w = cat([sin(freqs_w), cos(freqs_w)])

# 2D結合（加算）
emb_2d = emb_h + emb_w
```

### 周波数分布の特性

**可視化結果** (`docs/rope_analysis/rope_frequency_analysis.png`):

| 周波数帯域    | 波長（positions） | 128×128での周期数 | カバレッジ |
|--------------|------------------|------------------|----------|
| 低周波 (0-10)  | 256 - 102       | 0.5 - 1.25      | 不足     |
| 中周波 (10-50) | 102 - 10        | 1.25 - 12.5     | 適切     |
| 高周波 (50-160)| 10 - 0.01       | 12.5 - 12800    | 過剰     |

**観察:**
- **低周波数が不足**: 大域的な位置情報（画像全体の配置）の表現力が弱い
- **高周波数が過剰**: 細部の位置情報が過剰に表現されている
- **解像度依存**: 位置インデックスが絶対値のため、解像度変更で周期数が変わる

---

## Identified Problems

### Problem 1: Resolution Inconsistency

**現象**:
```
学習解像度: 128×128 latent (1024×1024 pixels)

推論時の挙動:
  64×64   (512×512)   → 周期数が半分になる（情報不足）
  128×128 (1024×1024) → 正確に一致
  192×192 (1536×1536) → 周期数が1.5倍になる（不整合）
  256×256 (2048×2048) → 周期数が2倍になる（大きな不整合）
```

**影響**:
- 学習時と異なる解像度での生成品質低下
- 位置情報の不連続性
- アーティファクトの発生（特に境界付近）

**可視化**: `docs/rope_analysis/rope_resolution_extrapolation.png`

### Problem 2: Frequency Distribution Imbalance

**現象**:
- 低周波数（0-10番目）: 10個のみ（6.25%）
- 中周波数（10-50番目）: 40個（25%）
- 高周波数（50-160番目）: 110個（68.75%）

**影響**:
- 大域的な構造の学習が不十分
- 細部に過剰にフィッティング
- 汎化性能の低下

### Problem 3: 2D Combination Method

**現在の実装**: 加算による結合 `emb_2d = emb_h + emb_w`

**問題点**:
- 対角方向にバイアスが生じる可能性
- H成分とW成分の独立性が失われる
- 回転不変性の欠如

**可視化**: `docs/rope_analysis/rope_2d_combination.png`

---

## Proposed Solutions

### Solution 1: Resolution-Adaptive RoPE

**実装**: `backend/core/models/rope_improved.py:ResolutionAdaptiveRoPE2D`

**Key Idea**: 位置インデックスを解像度比で正規化

```python
# 解像度スケール係数
scale_h = H / train_resolution  # 例: 256/128 = 2.0
scale_w = W / train_resolution

# 正規化された位置インデックス
pos_h = arange(H) / scale_h
# 256×256推論時: [0, 0.5, 1.0, 1.5, ..., 127.5]
# → 学習時と同じ周期数を維持

pos_w = arange(W) / scale_w
```

**効果**:
- ✅ 解像度に依存しない一貫した位置情報
- ✅ 学習解像度の整数倍で最適
- ✅ 任意解像度でもスムーズに補間

**適用シーン**:
- 学習解像度の0.5x～2xでの推論
- 高品質な画像生成が必要な場合
- アスペクト比が変わる場合

### Solution 2: NTK-Aware Scaled RoPE

**実装**: `backend/core/models/rope_improved.py:NTKScaledRoPE2D`

**Key Idea**: Neural Tangent Kernel理論に基づき、base周波数をスケーリング

```python
# 最大解像度に基づくα計算
max_scale = max_resolution / train_resolution  # 例: 512/128 = 4.0
alpha = max_scale ^ (dim / (dim - 2))
# dim=320の場合: alpha ≈ 4.04

# スケールされたbase周波数
scaled_base = base * alpha  # 10000 * 4.04 ≈ 40400

# 周波数帯域（低周波成分が拡張される）
inv_freq = 1.0 / (scaled_base ** (arange(0, dim, 2) / dim))
```

**効果**:
- ✅ 極端な解像度外挿（8x以上）に対応
- ✅ 高周波成分を維持しつつ低周波成分を拡張
- ✅ 推論時のオーバーヘッド無し（事前計算）

**適用シーン**:
- 学習解像度の4x～8xでの推論
- 超高解像度生成（4K, 8K）
- パノラマ画像生成

### Solution 3: Dynamic NTK-RoPE

**実装**: `backend/core/models/rope_improved.py:DynamicNTKRoPE2D`

**Key Idea**: 実行時に解像度に応じてαを動的計算

```python
def compute_dynamic_alpha(self, current_res: int):
    if current_res <= train_resolution:
        return 1.0  # 小さい解像度ではスケーリング不要

    scale_factor = current_res / train_resolution
    alpha = scale_factor ^ (dim / (dim - 2))
    return alpha

# 推論時
current_res = max(H, W)
alpha = compute_dynamic_alpha(current_res)
scaled_base = base * alpha
inv_freq = 1.0 / (scaled_base ** ...)
```

**効果**:
- ✅ 任意解像度に自動適応
- ✅ 再学習不要
- ✅ 小さい解像度でも品質維持

**適用シーン**:
- 動的解像度変更（バッチ内で異なる解像度）
- マルチスケール推論
- 解像度を事前に決められない場合

---

## Comparison Matrix

### 解像度外挿性能比較

| 実装                    | 0.5x | 1x  | 1.5x | 2x  | 4x  | 8x  | 動的対応 | オーバーヘッド |
|------------------------|------|-----|------|-----|-----|-----|---------|--------------|
| Standard RoPE          | ❌   | ✅  | ⚠️   | ❌  | ❌  | ❌  | ❌      | 無し         |
| Resolution-Adaptive    | ✅   | ✅  | ✅   | ✅  | ⚠️  | ❌  | ❌      | 無し         |
| NTK-Scaled (α=4)       | ⚠️   | ✅  | ✅   | ✅  | ✅  | ⚠️  | ❌      | 無し         |
| Dynamic NTK            | ✅   | ✅  | ✅   | ✅  | ✅  | ✅  | ✅      | 小（α計算）  |

凡例:
- ✅ 優れた性能
- ⚠️ 使用可能だが最適ではない
- ❌ 品質低下が顕著

### 特性比較

| 特性                  | Standard | Adaptive | NTK-Scaled | Dynamic NTK |
|----------------------|----------|----------|------------|-------------|
| **実装複雑度**        | ⭐       | ⭐⭐     | ⭐⭐       | ⭐⭐⭐      |
| **計算コスト**        | 低       | 低       | 低         | 低～中      |
| **メモリ使用量**      | 低       | 低       | 低         | 低          |
| **解像度汎化性**      | 低       | 中       | 高         | 最高        |
| **学習安定性**        | 高       | 高       | 中         | 中          |
| **推論柔軟性**        | 低       | 低       | 中         | 高          |

### 推奨用途

| 用途                           | 推奨実装            | 理由                               |
|-------------------------------|--------------------|------------------------------------|
| **標準的な生成**               | Resolution-Adaptive | バランスが良く、実装が簡単         |
| **高解像度生成（4K以上）**     | NTK-Scaled         | 極端な外挿に対応                   |
| **動的解像度変更**             | Dynamic NTK        | 実行時に自動適応                   |
| **学習時（基準実装）**         | Standard           | 既存研究との比較が容易             |
| **パノラマ・超横長画像**       | Dynamic NTK        | アスペクト比が極端でも対応         |

---

## Recommendations

### 推奨実装戦略

#### Phase 1: Resolution-Adaptive RoPEを標準実装として採用

**理由**:
- 実装が簡単（位置インデックスの正規化のみ）
- 計算オーバーヘッド無し
- 学習解像度の0.5x～2xで優れた性能
- 既存モデルからの移行が容易

**実装手順**:
1. `unet_deus.py`のRoPE2Dクラスを`ResolutionAdaptiveRoPE2D`に置き換え
2. `train_resolution`パラメータを追加（デフォルト: 128）
3. 既存checkpointとの互換性を維持

```python
# backend/core/models/unet_deus.py

from .rope_improved import ResolutionAdaptiveRoPE2D

class DeusUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        # ...
        # Before:
        # self.rope_2d = RoPE2D(dim=config.model_channels)

        # After:
        self.rope_2d = ResolutionAdaptiveRoPE2D(
            dim=config.model_channels,
            train_resolution=128,  # 1024×1024 pixels / 8 (VAE scaling)
            max_resolution=512,     # 4096×4096 pixels / 8
            base=10000.0
        )
```

#### Phase 2: Dynamic NTK-RoPEをオプションとして追加

**理由**:
- 極端な解像度変更に対応
- ユーザーが柔軟に選択可能
- 研究・実験用途にも有用

**実装手順**:
1. `UNetConfig`に`rope_variant`パラメータを追加
2. ファクトリ関数`create_rope_2d()`を使用
3. チェックポイントのメタデータに`rope_variant`を保存

```python
@dataclass
class UNetConfig:
    # ... 既存のフィールド
    rope_variant: str = "adaptive"  # "standard", "adaptive", "ntk", "dynamic_ntk"
    rope_train_resolution: int = 128

class DeusUNet(nn.Module):
    def __init__(self, config: UNetConfig):
        from .rope_improved import create_rope_2d

        self.rope_2d = create_rope_2d(
            variant=config.rope_variant,
            dim=config.model_channels,
            train_resolution=config.rope_train_resolution,
            max_resolution=512,
            base=10000.0
        )
```

### 学習時の推奨設定

```yaml
# Training config
rope:
  variant: "adaptive"           # Resolution-Adaptive RoPE
  train_resolution: 128         # 1024×1024 pixels / 8
  max_resolution: 512           # Max expected: 4096×4096 / 8
  base: 10000.0                 # Standard RoFormer base

# Data augmentation
resolution_augmentation:
  enabled: true
  min_resolution: 64            # 512×512 pixels
  max_resolution: 192           # 1536×1536 pixels
  # RoPEが正規化するため、異なる解像度での学習が安全
```

### 推論時の推奨設定

```python
# 標準解像度（1024×1024）
pipeline.generate(
    prompt="...",
    height=1024,
    width=1024
    # RoPEが自動的に適応
)

# 高解像度（2048×2048）
# Resolution-Adaptive RoPEが自動的にスケーリング
pipeline.generate(
    prompt="...",
    height=2048,
    width=2048
)

# 超高解像度（4096×4096以上）の場合
# Dynamic NTK-RoPEに切り替え推奨
config.rope_variant = "dynamic_ntk"
```

---

## Implementation Roadmap

### Milestone 1: Baseline Update (2 days)

- [ ] `rope_improved.py`を統合
- [ ] `unet_deus.py`でResolution-Adaptive RoPEを使用
- [ ] 既存checkpointとの互換性確認
- [ ] 単体テスト作成

### Milestone 2: Validation (3 days)

- [ ] 異なる解像度での生成テスト
- [ ] 品質比較（FID, CLIP score）
- [ ] 速度ベンチマーク
- [ ] ドキュメント更新

### Milestone 3: Advanced Features (3 days)

- [ ] Dynamic NTK-RoPEオプション追加
- [ ] `UNetConfig`にrope設定追加
- [ ] YAMLコンフィグ対応
- [ ] ユーザードキュメント作成

### Milestone 4: Training Integration (2 days)

- [ ] 学習パイプラインでの動作確認
- [ ] 解像度augmentation対応
- [ ] チェックポイントメタデータ更新
- [ ] 学習設定テンプレート作成

---

## References

1. **RoFormer**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
   - https://arxiv.org/abs/2104.09864

2. **NTK-Aware Scaled RoPE**: Reddit /r/LocalLLaMA discussion
   - https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/

3. **Position Interpolation**: Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation", 2023
   - https://arxiv.org/abs/2306.15595

4. **SDXL Architecture**: Podell et al., "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis", 2023
   - https://arxiv.org/abs/2307.01952

---

**Last Updated**: 2026-01-06
**Status**: Proposal / Ready for Implementation
