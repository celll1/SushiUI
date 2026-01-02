# CFG実装の比較分析: Z-Image (Flow Matching) vs SDXL (DDPM)

**作成日**: 2026-01-02
**目的**: Z-ImageとSDXLでCFG計算式が同一であることの理論的妥当性を検証

---

## エグゼクティブサマリー

**結論**: Z-ImageとSDXLで同じCFG計算式を使用することは**理論的に正当化されるが、最適ではない可能性がある**

**主な発見**:
1. ✅ **数学的には同一の線形補間**: 両方とも `pred = uncond + w * (cond - uncond)` の形式
2. ⚠️ **物理的意味が異なる**: DDPMは「ノイズ予測」、Flow Matchingは「速度場予測」
3. ❌ **Flow Matchingでの既知の問題**: 標準CFGは"off-manifold drift"を引き起こす（最新研究で指摘）
4. 💡 **改善の余地**: CFG-Zero*やRectified-CFG++などの専用手法が提案されている

---

## 1. 現在の実装状況

### 1.1 Z-Image (Flow Matching)

**ファイル**: `backend/core/pipeline.py:1794`

```python
# Standard CFG formula (consistent with SD/SDXL)
# pred = uncond + guidance_scale * (cond - uncond)
pred = neg + current_guidance_scale * (pos - neg)
```

**モデル出力**: 速度場 `v_θ(x_t, t)` - サンプルが移動する方向と速度
**スケジューラ**: `FlowMatchEulerDiscreteScheduler`
**更新則**: `x_{t+dt} = x_t + dt * v_θ(x_t, t)`

### 1.2 SDXL (DDPM)

**ファイル**: `backend/core/inference/custom_sampling.py:678`

```python
# Apply CFG
noise_pred = noise_pred_uncond + current_guidance_scale * (noise_pred_text - noise_pred_uncond)
```

**モデル出力**: ノイズ予測 `ε_θ(x_t, t)` - 除去すべきノイズ
**スケジューラ**: `DDPMScheduler`, `DDIMScheduler`, etc.
**更新則**: `x_{t-1} = 1/sqrt(α_t) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z`

### 1.3 共通点

**CFG公式の数学的形式**:
```
guided_output = unconditional + guidance_scale * (conditional - unconditional)
            = (1 - w) * unconditional + w * conditional    (w = guidance_scale)
```

両方とも**線形補間（Linear Interpolation）**の形式を使用している。

---

## 2. 理論的背景の違い

### 2.1 DDPM（Denoising Diffusion Probabilistic Models）

**数理的基礎**:
- **確率的拡散過程**: `q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)`
- **逆過程**: `p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))`
- **モデル予測**: ノイズ `ε_θ(x_t, t)` または x₀ (prediction_type="sample")
- **CFGの意味**: ノイズの方向を条件付きガイダンスに沿って外挿

**v-predictionの場合**:
```
v = α_t * ε - σ_t * x₀
```
これも本質的にはノイズ/信号の線形結合。

### 2.2 Flow Matching（Rectified Flow）

**数理的基礎**:
- **常微分方程式（ODE）**: `dx/dt = v_θ(x_t, t)`
- **学習目標**: `min E[||v_θ(x_t, t) - (x₁ - x₀)||²]` (velocity matching)
- **サンプリング**: Euler法で積分 `x_{t+dt} = x_t + dt * v_θ(x_t, t)`
- **CFGの意味**: 速度場の方向を条件付きガイダンスに沿って外挿

**学習されたマニフォールド**:
- Flow Matchingは**決定論的な輸送経路**を学習
- 各点 `(x_t, t)` での速度 `v_θ` は特定の方向を指す
- CFGによる外挿は、この経路から**逸脱**させる可能性

---

## 3. 最新研究による問題点の指摘

### 3.1 CFG-Zero* (2025年3月)

**論文**: "CFG-Zero⋆: Improved Classifier-Free Guidance for Flow Matching Models"
**arXiv**: https://arxiv.org/abs/2503.18886

**主な指摘**:
> "In the early stages of training, when the flow estimation is inaccurate, CFG directs samples toward incorrect trajectories."

**問題の詳細**:
1. **Off-manifold drift**: 標準CFGの外挿により、サンプルが学習されたデータマニフォールドから外れる
2. **Visual artifacts**: 過飽和、テキストミスアライメント、脆弱な動作
3. **Trajectory deviation**: 初期ステップでの速度推定誤差が累積

**提案された解決策**:
- **Optimized scale**: 速度推定の不正確さを補正するスカラーパラメータ
- **Zero-init**: ODEソルバーの最初の数ステップをゼロにする

**検証モデル**:
- Lumina-Next (Z-Imageベース)
- Stable Diffusion 3
- FLUX
- Wan-2.1 (text-to-video)

### 3.2 Rectified-CFG++ (2024年10月)

**論文**: "Rectified-CFG++ for Flow Based Models"
**arXiv**: https://arxiv.org/abs/2510.07631

**主な指摘**:
> "Naively plugging [standard CFG] into the ODE solver inherits the same off-manifold drift observed in diffusion models."

**問題の詳細**:
- **Geometric instability**: 決定論的フローは軌道の摂動に敏感
- **Oversaturation**: 線形外挿が幾何学的制約を破る
- **Structural distortion**: マニフォールド外への強制的な引き出し

**提案された解決策**:
- **Predictor-corrector guidance**:
  1. 条件付きRF更新でまず学習経路に留まる
  2. その後、条件付き/無条件速度場間でスケジュール化された補間を適用
- **Geometry-aware conditioning**: マニフォールドの境界内に留まるように設計

---

## 4. SushiUIでの実装上の考慮事項

### 4.1 現在の実装の妥当性

**✅ 正当化できる点**:
1. **diffusersライブラリの標準実装を踏襲**: Hugging Faceの公式実装も同じCFG公式を使用
2. **数学的一貫性**: 線形補間は両フレームワークで共通の操作
3. **実用的な動作**: 現在の実装で視覚的に許容可能な結果が得られている可能性

**⚠️ 潜在的な問題**:
1. **理論的最適性の欠如**: Flow Matchingに特化したCFGではない
2. **Off-manifold drift**: 特に高いguidance_scale（7.0以上）で顕在化する可能性
3. **改善の余地**: 最新研究が示す通り、より良い手法が存在

### 4.2 検証すべき症状

以下の現象が発生している場合、CFG実装の問題を示唆:

1. **過飽和（Oversaturation）**: 色が不自然に強調される
2. **構造的歪み（Structural distortion）**: 形状が崩れる
3. **テクスチャの異常**: 不自然なパターンやノイズ
4. **高CFG値での劣化**: guidance_scale > 10で顕著な品質低下

### 4.3 推奨される検証実験

**実験1**: CFG値のスイープ
```python
guidance_scales = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
# 同じプロンプト・シードで比較
# 期待: Flow Matchingで高CFG値での劣化が早い
```

**実験2**: Dynamic CFG scheduleの効果
```python
cfg_schedule_type = "linear"  # 高σで高CFG、低σで低CFG
cfg_schedule_min = 1.0
cfg_schedule_max = 7.0
# 期待: 初期ステップでのoff-manifold driftを軽減
```

**実験3**: Guidance rescaleの効果
```python
guidance_rescale = 0.7  # v-predictionモデルで推奨
# 期待: 標準偏差の補正により過飽和を軽減
```

---

## 5. 推奨される改善アプローチ

### 5.1 短期的対応（既存フレームワーク内）

#### 1. Dynamic CFG schedulingの活用

**実装箇所**: `backend/core/inference/custom_sampling.py:calculate_dynamic_cfg()`

```python
# Z-Image専用のデフォルト値を推奨
cfg_schedule_type = "linear"      # または "cosine"
cfg_schedule_min = 1.0            # 終了時はCFGを弱める
cfg_schedule_max = 5.0            # 開始時も控えめに（SDXLの7.0より低く）
```

**理論的根拠**:
- 高σ（初期ステップ）でのCFGを抑制してoff-manifold driftを軽減
- 低σ（後期ステップ）でもCFGを弱めて詳細保存

#### 2. Guidance rescaleの自動適用

**現在の実装**: v-predictionモデルでのみ `guidance_rescale=0.7`

**提案**: Flow MatchingモデルでもデフォルトでRescaleを有効化

```python
# backend/core/pipeline.py
if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
    guidance_rescale = 0.5  # Flow Matching専用の値（要調整）
    print(f"[Z-Image] Flow Matching detected, applying guidance_rescale={guidance_rescale}")
```

**理論的根拠**: `rescale_noise_cfg()` は標準偏差を補正し、過飽和を防ぐ

#### 3. Dynamic thresholdingの活用

**現在の実装**: `dynamic_threshold_percentile` パラメータで対応可能

```python
dynamic_threshold_percentile = 0.95  # Z-Imageで推奨
dynamic_threshold_mimic_scale = 7.0
```

**効果**: 極端な値を抑制し、CFGによる発散を防ぐ

### 5.2 中期的対応（専用実装）

#### CFG-Zero*風の実装

**Phase 1**: Optimized scaleの導入

```python
def calculate_cfg_scale_zimage(
    base_scale: float,
    timestep: float,
    velocity_uncond: torch.Tensor,
    velocity_cond: torch.Tensor
) -> float:
    """
    Z-Image専用の動的CFGスケール計算

    velocity_uncond, velocity_condの大きさの比を考慮して
    スケールを自動調整
    """
    # 速度場のノルムを計算
    norm_uncond = torch.norm(velocity_uncond, p=2, dim=[1,2,3]).mean()
    norm_cond = torch.norm(velocity_cond, p=2, dim=[1,2,3]).mean()

    # 条件付き/無条件の速度差が大きいほど、CFGスケールを抑制
    velocity_ratio = norm_cond / (norm_uncond + 1e-8)

    # 初期timestep（高σ）ではさらに抑制
    sigma_normalized = timestep / 1000.0
    time_factor = 1.0 - 0.5 * sigma_normalized  # 初期で0.5x, 終了で1.0x

    adjusted_scale = base_scale * time_factor / (1.0 + 0.1 * abs(velocity_ratio - 1.0))
    return max(1.0, min(adjusted_scale, base_scale))
```

**Phase 2**: Zero-initの実装

```python
# 最初のN stepではCFGを無効化
zero_init_steps = 2  # 調整可能
if i < zero_init_steps:
    current_guidance_scale = 1.0  # CFGなし
else:
    current_guidance_scale = calculate_cfg_scale_zimage(...)
```

#### Rectified-CFG++風の実装

```python
# Predictor step: 条件付き速度で更新
latents_pred = scheduler.step(velocity_cond, t, latents).prev_sample

# Corrector step: 補間適用（境界内に留まる）
if apply_cfg:
    interpolation_weight = min(current_guidance_scale - 1.0, 1.0)  # 上限1.0
    velocity_guided = velocity_uncond + interpolation_weight * (velocity_cond - velocity_uncond)
    latents = scheduler.step(velocity_guided, t, latents).prev_sample
```

### 5.3 長期的対応（研究ベース）

1. **CFG-Zero*の完全実装**: 最適化可能なscaleパラメータの学習
2. **Flow-specific metrics**: 速度場の幾何学的特性に基づくガイダンス
3. **Adaptive guidance**: サンプルごとの軌道に基づく動的調整

---

## 6. コード比較表

| 項目 | Z-Image (Flow Matching) | SDXL (DDPM) | 差異 |
|------|------------------------|-------------|------|
| **CFG公式** | `pred = neg + w * (pos - neg)` | `pred = uncond + w * (cond - uncond)` | ✅ 同一 |
| **モデル出力** | 速度場 `v_θ(x_t, t)` | ノイズ `ε_θ(x_t, t)` | ❌ 異なる |
| **更新則** | `x_{t+dt} = x_t + dt * v` | `x_{t-1} = f(x_t, ε, α, β)` | ❌ 異なる |
| **スケジューラ** | `FlowMatchEulerDiscreteScheduler` | `DDPMScheduler`, `DDIMScheduler` | ❌ 異なる |
| **Guidance rescale** | なし（v-predのみ） | v-predで0.7 | ⚠️ Flow用の値未設定 |
| **Dynamic CFG** | サポート（共通関数） | サポート（共通関数） | ✅ 同一 |
| **Dynamic threshold** | サポート（共通関数） | サポート（共通関数） | ✅ 同一 |

---

## 7. 結論と推奨事項

### 7.1 現在の実装の評価

**総合評価**: ⭐⭐⭐☆☆ (3/5)

**長所**:
- ✅ 数学的に一貫性がある（線形補間）
- ✅ diffusers標準実装に準拠
- ✅ 実装がシンプルで保守しやすい

**短所**:
- ❌ Flow Matching特有の問題（off-manifold drift）への対策なし
- ❌ 最新研究（CFG-Zero*, Rectified-CFG++）の成果を反映していない
- ⚠️ 高guidance_scaleでの品質劣化リスク

### 7.2 即座に実施すべき対策

#### 優先度: 高（即座に実施）

1. **Z-Image専用のguidance_rescale導入**
   ```python
   # backend/core/pipeline.py
   if isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
       guidance_rescale = 0.5  # 要実験調整
   ```

2. **デフォルトCFG値の見直し**
   ```python
   # Z-Imageの場合、SDXLより低めを推奨
   default_cfg_zimage = 5.0  # 現在7.0の場合
   ```

#### 優先度: 中（1-2週間以内）

3. **Dynamic CFG schedulingのデフォルト有効化**
   ```python
   cfg_schedule_type = "linear"
   cfg_schedule_min = 1.0
   cfg_schedule_max = 5.0  # Z-Image専用
   ```

4. **実験的検証の実施**
   - guidance_scale: 1.0, 3.0, 5.0, 7.0, 10.0での品質比較
   - guidance_rescale: 0.0, 0.3, 0.5, 0.7での効果測定
   - Dynamic thresholdの効果確認

#### 優先度: 低（将来的検討）

5. **CFG-Zero*実装の検討**
   - Optimized scaleの導入
   - Zero-init stepの実装

6. **Rectified-CFG++実装の検討**
   - Predictor-corrector guidanceの実装

### 7.3 技術的推奨事項

#### モデルタイプ別のデフォルト値

**Z-Image (Flow Matching)**:
```yaml
guidance_scale: 5.0              # SDXLより低め
cfg_schedule_type: "linear"      # 動的調整を推奨
cfg_schedule_min: 1.0            # 終了時はCFG弱める
cfg_schedule_max: 5.0            # 開始時も控えめ
guidance_rescale: 0.5            # Flow専用rescale
dynamic_threshold_percentile: 0.95
```

**SDXL (DDPM, v-prediction)**:
```yaml
guidance_scale: 7.0              # 標準値
cfg_schedule_type: "constant"    # または動的
guidance_rescale: 0.7            # v-pred推奨値
dynamic_threshold_percentile: 0.0  # 必要に応じて
```

---

## 8. 参考文献

1. **CFG-Zero***: Fan et al., "CFG-Zero⋆: Improved Classifier-Free Guidance for Flow Matching Models", arXiv:2503.18886, 2025
   - https://arxiv.org/abs/2503.18886
   - https://weichenfan.github.io/webpage-cfg-zero-star/

2. **Rectified-CFG++**: "Rectified-CFG++ for Flow Based Models", arXiv:2510.07631, 2024
   - https://arxiv.org/abs/2510.07631
   - https://rectified-cfgpp.github.io/

3. **Classifier-Free Guidance**: Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop

4. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023

5. **Rectified Flow**: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", ICLR 2023

---

## 9. 更新履歴

- **2026-01-02**: 初版作成
  - Z-ImageとSDXLのCFG実装比較
  - 最新研究（CFG-Zero*, Rectified-CFG++）の調査
  - 推奨事項と改善アプローチの提案

---

**作成者**: Claude (Anthropic)
**レビュー**: 推奨（実装前に実験的検証を実施）
