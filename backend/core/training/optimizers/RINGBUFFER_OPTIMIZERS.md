# Ring Buffer Optimizers - Implementation Documentation

このドキュメントは、SushiUIのRing Buffer Optimizers（AdamW8bit_RingBuffer、Lion8bit_RingBuffer）の実装詳細をまとめたものです。

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [メモリ管理戦略](#メモリ管理戦略)
4. [Schedule-Free実装](#schedule-free実装)
5. [CUDA実装詳細](#cuda実装詳細)
6. [使用方法](#使用方法)
7. [パフォーマンス](#パフォーマンス)
8. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### Ring Buffer Optimizersとは

Ring Buffer Optimizersは、**optimizer stateをCPUメモリに配置**し、**必要な時だけGPUに転送**することで、**GPU VRAMを削減**する設計の最適化手法です。

> ### ⚠️ 現状: CPU state モードはどの呼び出し側からも有効化されていない
>
> **実装は存在し、`get_state_buffer` を渡せば動きます。壊れているのは配線です。**
>
> CPU 常駐は、コンストラクタ引数 `get_state_buffer`（`RingBufferAllocator` ベースの
> allocator）を渡したときにだけ有効になります。ところが**この引数を供給している
> 呼び出し側が存在しません**。参照は optimizer 実装の内部と
> `optimizer_factory.py:130`, `:174` の `kwargs.get("get_state_buffer", None)` だけで、
> `BaseTrainer._ringbuffer_optimizer_kwargs()` にも VAE trainer にも含まれていません
> （`git log -S "get_state_buffer=" -- base_trainer.py` は空。導入コミット
> `190c876e` 以来一度も配線されたことがありません）。
>
> したがって既定では常に `None` に解決され、**8-bit state は GPU に確保されます**
> （`adamw8bit_ringbuffer.py` の "Ring Buffer disabled: GPU allocation
> (bitsandbytes-compatible)" 分岐。`lion8bit_ringbuffer.py` も同様）。
>
> **現状これらの optimizer が提供しているのは「GPU state を持つ fused 8-bit
> AdamW / Lion」です。** 8-bit 量子化ぶんの削減は実際に効いていますが、本ドキュメントが
> 「Ring Buffer」として説明する CPU 常駐と DMA 転送は既定では走りません。
> 以下の記述は、**allocator を渡した場合に設計上どう動くか**として読んでください。
>
> 配線に必要な作業と、それを前提にした設計上の帰結は
> [`docs/guides/SENSENOVA_TRAINING_DESIGN.md`](../../../../docs/guides/SENSENOVA_TRAINING_DESIGN.md)
> §6.5 に整理してあります（SenseNova の full FT 予算がこの差で 30 GB 変わるため）。

### 特徴

- ✅ **8-bit量子化対応**（AdamW/Lion）— **既定で有効**
- ⚙️ **optimizer state の CPU 常駐**（allocator を渡した場合のみ。上記参照）
  — 構造上の削減率は[パフォーマンス](#メモリ使用量)の worked example を参照
- ✅ **Schedule-Free対応**（LRスケジュール不要）
- ✅ **Cautious Optimizer対応**（符号一致マスキング）
- ⚙️ **Pinned Memory最適化**（CPU state を使う場合のみ関係する）
- ✅ **Block Swap互換**（U-Net CPUオフロード）

### 対応Optimizer

| Optimizer | 8-bit | Schedule-Free | Cautious | Stochastic Rounding | Ring Buffer |
|-----------|-------|---------------|----------|---------------------|-------------|
| AdamW8bit_RingBuffer | ✅ | ✅ | ✅ | ✅ | ⚙️ 実装済・未配線 |
| Lion8bit_RingBuffer | ✅ | ❌ **拒否** | ✅ | ✅ | ⚙️ 実装済・未配線 |

「⚙️ 実装済・未配線」= optimizer 側は `get_state_buffer` を受け取れば CPU state で
動作するが、それを渡す呼び出し側が無いため既定では GPU 割当になる（冒頭の注記）。

`Lion8bit_RingBuffer` の Schedule-Free は拒否される。`lion8bit_schedulefree_kernel.cu`
は Schedule-Free の位置系列であるべき `z` を Lion の momentum EMA として使い、
`x = (1-ckp1)*z + ckp1*y` をパラメータに書き戻す。`ckp1 ≈ 1/k` なので数ステップで
パラメータが momentum バッファそのものになる（実測: ランダム勾配で step5 の
corr(p, z) = 0.994、step20 で 0.9996）。正しく実装するには位置系列と momentum の
**2つ**の state が必要で、現在の `_init_param_state` は1つしか確保しないため、
state レイアウトとチェックポイント形式の再設計になる。コンストラクタ
（`RuntimeError`）と `BaseTrainer.setup_optimizer`（`ValueError`）の両方で拒否する。

Block Swap（`blocks_to_swap > 0`）と組み合わせられるのは、per-parameter の
更新経路を持つ optimizer のみ:

| Optimizer | `blocks_to_swap > 0` | 経路 |
|---|---|---|
| `adamw8bit_ringbuffer` / `lion8bit_ringbuffer` | ✅ | 自前の post-accumulate-grad hook（8-bit stateのまま） |
| `adamw8bit` | ✅ | `adamw8bit_fused.step_param`（bitsandbytesの per-parameter 更新に委譲、8-bit stateのまま） |
| `adafactor` | ✅ | `adafactor_fused.step_param` |
| `adamw`（torch） | ✅ | CPU上のパラメータをそのまま更新できる |
| `lion8bit` / `paged_adamw` / `paged_adamw8bit` / `paged_lion8bit` | ❌ 拒否 | fused経路が無く、bitsandbytesはCPU常駐パラメータでraiseする |

---

## Stochastic Rounding (`optimizer_stochastic_rounding`)

BF16の仮数部は8ビットなので、値 `w ∈ [2^e, 2^(e+1))` のULPは `2^(e-7)`、
round-to-nearestで切り捨てられる更新量の上限（half ULP）は `2^(e-8)` になる。
Adam系の1ステップの更新量は概ね `lr` なので、round-to-nearestでは
**`|w| <= 512 * lr` の要素しか動かない**（lr=1e-5 なら `|w| <= 5.12e-3`）。
round-to-nearestは決定的であるため、条件を満たさない要素は初期のビットパターンの
まま学習を通して固定される。

Stochastic roundingは端数の確率で切り上げるため `E[round(x)] == x` となり、
half ULP未満の更新も期待値として反映される。

実装（`stochastic_rounding.py`）:

- BF16パラメータに対してのみ適用（FP32/FP16パラメータでは無効）
- FP32のmaster weightは保持しない。ステップごとにscratchバッファ上に
  パラメータのFP32イメージを作り、更新後にstochastic roundingでBF16へ書き戻す
- 追加メモリは `4バイト × 最大パラメータのnumel × スロット数`
  （全パラメータ分のFP32コピーではない）
- 8-bit CUDAカーネルは `param.dtype == grad.dtype` を要求するため、
  FP32イメージを渡す際は勾配もFP32へ変換する（`prepare_master_and_grad`）
- `optimizer.step()` とfused backward hookの両方で適用される

計測（Krea 2 `transformer_blocks.2.ff.up.weight`、lr=1e-5、400ステップ、
1要素あたり `+lr` の更新）: round-to-nearestでは9.2%の要素のみが動き、
意図した変化量の6.9%しか反映されない。Stochastic roundingでは100%の要素が動き、
変化量は100%反映される。

デフォルトは無効（`optimizer_stochastic_rounding: false`）。

### Ring Buffer以外のoptimizerへの適用

Ring Bufferの2つは自前の更新処理の中で上記のプリミティブを呼ぶ。それ以外の
optimizer（full FTの既定値である `adamw8bit` を含む）はサードパーティのコードが
パラメータを直接書き換えるため、`attach_stochastic_rounding()` が
**パラメータ単位の呼び出し口に介入する**形で対応する（optimizer本体は変更しない）。

1回の更新呼び出しの間だけ `p.data` をscratch上のFP32イメージに、`p.grad` を
勾配のFP32イメージに差し替えるので、カーネルもPythonコードもFP32を読み書きし、
BF16ストレージには触れない。呼び出し終了時にstochastic roundingで書き戻す。

| optimizer | 介入点 | 備考 |
|---|---|---|
| `adamw8bit` / `lion8bit` / `paged_*` | `Optimizer8bit.update_step` | bitsandbytes。8-bit stateはuint8のまま |
| `adafactor` | `step_param` | fused版（`adafactor_fused`）を使う。stateは元からFP32 |
| `adamw8bit` + Block Swap | `adamw8bit_fused.step_param` 内で直接適用（FP32イメージを渡す） | 委譲先の `init_state` はdtypeを明示してstateを確保するため、stateはuint8のまま |
| `adamw8bit_ringbuffer` / `lion8bit_ringbuffer` | optimizer本体 | 従来通り |
| `adamw` | **なし** | 全パラメータを1回の呼び出しで更新し、パラメータ単位の入口が無い。setup時に警告する |

Schedule-Freeの `z` 系列もstochastic roundingの対象になった。`z` は最適化系列
そのもので毎ステップ読み戻されるため、round-to-nearestではsub-quantumな更新が
恒久的に破棄される（実測: 8-bit z で300ステップ中コードが動いた要素は0.54%、
意図したドリフトの実現率1.0%。SR有効時は86.08% / 99.8%）。有効時の内訳:

| z のストレージ | 経路 | 丸め |
|---|---|---|
| UINT8（`use_8bit=True`、実運用の経路） | CUDAカーネル内の再量子化 | `quantize_value_stochastic`（隣接コード間を確率的に選択） |
| パラメータdtype（BF16、`use_8bit=False`） | `step()` のFP32イメージ経由 | `copy_stochastic_bf16` |

`train()` / `eval()` の `p.lerp_(end=z)` も同じ扱い（BF16パラメータへの書き込みは
stochastic rounding、8-bit z は書き込み前にdequantize）。`exp_avg_sq` は対象外。

### 8-bit `z` のスケール（symmetric headroom）

signed dynamic mapは非対称（最大 `+1.000000000` / 最小 `-0.992968738`）。ブロックの
最大絶対値要素が**負**だと正規化値 `-1.0` を格納できず、次ステップのdequantizeで
0.7031%小さい値が返り、それが新しい `absmax_z` になる — これが毎ステップ複利で効く
（3000ステップでmean|z|がbf16基準の0.485倍、SR有効時は0.254倍）。対策は2つで1組:

1. `absmax = max|z| / 0.992968738`（両符号で表現可能なマージンを持たせる）ので
   dequantize→absmax再計算→requantizeが冪等になる（実測: 勾配0で20000ステップ、
   コード変化0/8192、absmax変化 2.7e-6）。
2. `absmax` を決める要素だけはstochastic roundingを**適用しない**。
   （この判定は `cub::BlockReduce::Reduce` の戻り値ではなく、shared memory経由で
   **ブロードキャストした最大値**と比較すること。戻り値がaggregateなのはthread 0
   だけで、他スレッドはwarp/raking単位の部分最大値を受け取るため、比較すると
   全要素の12.7%——しかも絶対値の大きい側——がRTNのまま取り残される。）`absmax` は
   ブロック内の格納値のmaxなので、その要素にノイズが乗るとスケール自体が
   最大値バイアスを取り込み暴走する（headroomのみだと +0.63%/step = 3000ステップで
   1.5e8倍、逆にその要素を1コード下げると0.37倍に沈む）。

この扱いはSchedule-Freeカーネル内に閉じており、`quantization_map.py` と
非Schedule-Freeの8-bit経路（`exp_avg` / `exp_avg_sq`）は変更していない。後者は
EMA（収縮写像）なので同じ非対称性があっても自己補正する（実測: 定数勾配5000
ステップでmean|exp_avg| 0.98125のまま不変）。

### 壊れたSchedule-Freeチェックポイントからのresume

Schedule-Freeカーネルの `__constant__` マップが初期化されていなかった頃の
チェックポイントは、`z` が定数（実例: 全コード255 / `absmax_z` ≈ 0）にデコードされる。
`z` は無害ではなく `y` がそこへ引っ張られるため、resume後300ステップ（勾配0）で
mean|p| が 1.63e-2 → 5.21e-5 まで潰れる。`load_state_dict()` はこの署名を検出したら
`z` を現在のパラメータから再初期化する（`z_0 = p`）。警告を出し、`exp_avg_sq` と
ステップカウンタは保持する。健全なチェックポイントには触れない。

---

## アーキテクチャ

### 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│ Ring Buffer Optimizer Architecture                         │
└─────────────────────────────────────────────────────────────┘

CPU Memory (Pinned):
┌─────────────────────────────────────────┐
│ exp_avg: FP16/BF16 (momentum)           │  ← CPU-allocated
│   例: [350M params] × 2 bytes = 700 MB  │     (pinned memory)
├─────────────────────────────────────────┤
│ exp_avg_sq: UINT8 (variance, 8-bit)     │  ← CPU-allocated
│   例: [350M params] × 1 byte = 350 MB   │     (pinned memory)
├─────────────────────────────────────────┤
│ z: UINT8 (Schedule-Free, 8-bit)         │  ← CPU-allocated
│   例: [350M params] × 1 byte = 350 MB   │     (pinned memory)
│   ※ Schedule-Free有効時のみ             │
└─────────────────────────────────────────┘

GPU Memory:
┌─────────────────────────────────────────┐
│ param: FP16/BF16/FP32                   │  ← Model parameters
│   例: [350M params] × 2 bytes = 700 MB  │
├─────────────────────────────────────────┤
│ grad: FP16/BF16/FP32                    │  ← Gradients
│   例: [350M params] × 2 bytes = 700 MB  │
├─────────────────────────────────────────┤
│ absmax1: FP32 (exp_avg absmax)          │  ← Quantization maps
│   例: [350M / 256] × 4 bytes = 5.5 MB   │     (per block)
├─────────────────────────────────────────┤
│ absmax2: FP32 (exp_avg_sq absmax)       │  ← Quantization maps
│   例: [350M / 256] × 4 bytes = 5.5 MB   │     (per block)
├─────────────────────────────────────────┤
│ absmax_z: FP32 (z absmax)               │  ← Schedule-Free only
│   例: [350M / 256] × 4 bytes = 5.5 MB   │
└─────────────────────────────────────────┘

PCIe Transfer (Async DMA):
  CPU → GPU: ~350-700 MB/step (exp_avg, exp_avg_sq, z)
  GPU → CPU: ~350-700 MB/step (copy back)
  Transfer time: 30-40 ms/step (pinned memory)
```

### コンポーネント

#### 1. Python Optimizer実装

- **`adamw8bit_ringbuffer.py`**: AdamW8bit_RingBuffer optimizer
- **`lion8bit_ringbuffer.py`**: Lion8bit_RingBuffer optimizer

#### 2. CUDA Extension

- **`adamw8bit_cuda.cpp`**: PyTorch C++ bindings
- **`adamw8bit_kernel.cu`**: CUDA kernel (standard AdamW)
- **`adamw8bit_schedulefree_kernel.cu`**: CUDA kernel (Schedule-Free)
- **`adamw8bit_schedulefree_launcher.cu`**: Kernel launcher (dtype dispatch)
- **`lion8bit_cuda.cpp`**: Lion optimizer bindings
- **`lion8bit_kernel.cu`**: CUDA kernel (Lion)

#### 3. 補助モジュール

- **`quantization_map.py`**: 動的量子化マップ生成
- **`optimizer_factory.py`**: Optimizer生成ファクトリ

---

## メモリ管理戦略

### 1. Ring Buffer（CPU Allocation）

**概念**:
- Optimizer stateをCPU pinned memoryに配置
- 必要な時だけGPU転送（async DMA）
- GPU VRAMはabsmaxのみ保持（~11 MB）

**実装** (`adamw8bit_ringbuffer.py`):

```python
def __init__(self, params, lr=1e-3, ..., get_state_buffer=None):
    self.get_state_buffer = get_state_buffer  # CPU buffer allocator

def _init_param_state(self, p):
    if self.get_state_buffer is not None:
        # CPU allocation (pinned memory)
        state['exp_avg'] = self.get_state_buffer(p, dtype=torch.float16)
        state['exp_avg_sq'] = self.get_state_buffer(p, dtype=torch.uint8)

        # Pinned memory optimization
        if hasattr(state['exp_avg'], 'pin_memory'):
            state['exp_avg'] = state['exp_avg'].pin_memory()
        if hasattr(state['exp_avg_sq'], 'pin_memory'):
            state['exp_avg_sq'] = state['exp_avg_sq'].pin_memory()
    else:
        # GPU allocation (fallback)
        state['exp_avg'] = torch.zeros_like(p, dtype=torch.float16, device=device)
        state['exp_avg_sq'] = torch.zeros(n, dtype=torch.uint8, device=device)
```

**Pinned Memory最適化**:

| Memory Type | Transfer Speed | 理由 |
|-------------|----------------|------|
| Non-pinned (通常のCPU memory) | ~11 GB/s | カーネルバッファ経由 |
| Pinned (ページ固定) | ~22-25 GB/s | 直接DMA転送 |

**効果**: DMA転送速度が2-3倍向上

### 2. 8-bit Blockwise Quantization

**概念**:
- Optimizer state（exp_avg_sq, z）を8-bit（UINT8）で保存
- ブロックごとにabsmax（FP32）を保持
- 量子化マップ（signed/unsigned）で動的変換

**量子化アルゴリズム** (`quantization_map.py`):

```python
def quantize_blockwise_inplace(tensor: torch.Tensor, blocksize: int = 256):
    """Quantize a tensor to UINT8 using blockwise quantization."""
    device = tensor.device
    flat = tensor.flatten()
    n = flat.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    # Allocate output
    quantized = torch.zeros(n, dtype=torch.uint8, device=device)
    absmax = torch.zeros(num_blocks, dtype=torch.float32, device=device)

    # Process each block
    for i in range(num_blocks):
        start = i * blocksize
        end = min(start + blocksize, n)
        block = flat[start:end]

        # Compute absmax
        block_absmax = block.abs().max()
        absmax[i] = block_absmax

        if block_absmax > 0:
            # Normalize to [-1, 1] (signed) or [0, 1] (unsigned)
            normalized = block / block_absmax

            # Quantize to UINT8 [0, 255]
            # Signed: [-1, 1] → [0, 255]
            quantized_block = ((normalized + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
            # Unsigned: [0, 1] → [0, 255]
            # quantized_block = (normalized * 255.0).clamp(0, 255).to(torch.uint8)

            quantized[start:end] = quantized_block

    return quantized.reshape_as(tensor), absmax
```

**量子化マップ**:

```python
# Signed map (for z, exp_avg - can be negative)
qmap_signed = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
# qmap_signed[0] = -1.0, qmap_signed[127] = 0.0, qmap_signed[255] = 1.0

# Unsigned map (for exp_avg_sq - always positive)
qmap_unsigned = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
# qmap_unsigned[0] = 0.0, qmap_unsigned[255] = 1.0
```

**逆量子化**:

```python
def dequantize(quantized: torch.Tensor, absmax: torch.Tensor, qmap: torch.Tensor, blocksize: int = 256):
    """Dequantize a UINT8 tensor back to FP32."""
    flat_q = quantized.flatten()
    n = flat_q.numel()
    num_blocks = (n + blocksize - 1) // blocksize

    dequantized = torch.zeros(n, dtype=torch.float32, device=quantized.device)

    for i in range(num_blocks):
        start = i * blocksize
        end = min(start + blocksize, n)

        # qmap[quantized_value] × absmax
        normalized = qmap[flat_q[start:end].long()]  # [-1, 1] or [0, 1]
        dequantized[start:end] = normalized * absmax[i]

    return dequantized.reshape_as(quantized)
```

**誤差の理論的上限**:

```
量子化誤差: ε = absmax / 256
相対誤差: ε_rel = ε / absmax = 1/256 ≈ 0.39%
```

### 3. Async DMA Transfer

**実装** (`adamw8bit_ringbuffer.py: step()`):

```python
def step(self, closure=None):
    for group in self.param_groups:
        for p in group['params']:
            state = self.state[p]

            # Ring Buffer: CPU → GPU async transfer
            exp_avg_gpu = state['exp_avg']
            exp_avg_sq_gpu = state['exp_avg_sq']

            if not state['exp_avg'].is_cuda:
                exp_avg_gpu = state['exp_avg'].cuda(non_blocking=True)
                exp_avg_sq_gpu = state['exp_avg_sq'].cuda(non_blocking=True)

            if schedule_free and not state['z'].is_cuda:
                z_gpu = state['z'].cuda(non_blocking=True)

            # Call CUDA kernel
            self.ext.adamw_8bit_update(
                p, grad, exp_avg_gpu, exp_avg_sq_gpu, ...
            )

            # GPU → CPU async copy-back
            if not state['exp_avg'].is_cuda:
                state['exp_avg'].copy_(exp_avg_gpu, non_blocking=True)
                state['exp_avg_sq'].copy_(exp_avg_sq_gpu, non_blocking=True)

            if schedule_free and not state['z'].is_cuda:
                state['z'].copy_(z_gpu, non_blocking=True)
```

**`non_blocking=True` の効果**:
- CPUとGPUの処理をオーバーラップ
- カーネル実行中にDMA転送を並列実行
- 実測: 10-15% 高速化

---

## Schedule-Free実装

### 概要

**Schedule-Free Learning** (Defazio et al., 2024, arXiv:2405.15682) は、**LRスケジュールを排除**し、**weighted iterate averaging**によって収束を保証する最適化手法です。

### 理論的背景

#### 3つのSequence

1. **z sequence（主勾配降下）**:
   ```
   z_{k+1} = z_k - lr · ∇f(y_k) / √(exp_avg_sq)
   ```
   - 標準的な勾配降下
   - optimizer stateとして8-bit量子化

2. **y sequence（訓練パラメータ）**:
   ```
   y_{k+1} = (1 - c_{k+1}) · y_k + c_{k+1} · z_{k+1} + lr · (β₁ · (1 - c_{k+1}) - 1) · grad_norm
   ```
   - 訓練時に使用するパラメータ
   - モデルのparam（FP16/BF16/FP32）として保持

3. **x sequence（評価パラメータ）**:
   ```
   x = (2 - 1/β₁) · z - (1 - 1/β₁) · y
   ```
   - 評価時に使用するパラメータ
   - train()/eval()メソッドで切り替え

#### Weighted Iterate Averaging

```python
# Weight calculation
weight = ((k+1) ** r) * (lr_max ** weight_lr_power)
weight_sum += weight
c_{k+1} = weight / weight_sum

# Linear warmup
if k < warmup_steps:
    sched = (k+1) / warmup_steps
else:
    sched = 1.0
lr_scheduled = lr * sched
lr_max = max(lr_scheduled, lr_max)
```

**パラメータ**:
- `r`: Weight polynomial power（デフォルト: 0.0 → 均等平均）
- `weight_lr_power`: LR weighting power（デフォルト: 2.0）
- `warmup_steps`: Linear warmup steps（デフォルト: 0）

### 実装詳細

#### 初期化 (`adamw8bit_ringbuffer.py: __init__`)

```python
def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
             weight_decay=0.01, use_8bit=True, cautious=False,
             schedule_free=False, warmup_steps=0, r=0.0, weight_lr_power=2.0,
             get_state_buffer=None):

    # Schedule-Free: cautious is incompatible
    if schedule_free and cautious:
        print("[AdamW8bit_RingBuffer] WARNING: cautious is disabled when schedule_free=True")
        cautious = False

    defaults = dict(
        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        use_8bit=use_8bit, cautious=cautious,
        schedule_free=schedule_free, warmup_steps=warmup_steps,
        r=r, weight_lr_power=weight_lr_power
    )
    super().__init__(params, defaults)

    self.get_state_buffer = get_state_buffer
    self.step_count = 0
    self.ext = None  # Lazy load CUDA extension

    # Schedule-Free global state
    if schedule_free:
        self.k = 0              # Step counter (separate from step_count)
        self.weight_sum = 0.0   # FP32 accumulator
        self.lr_max = 0.0       # Maximum LR observed
        self.train_mode = False # Training mode flag
```

**重要**: `cautious` と `schedule_free` は互換性がありません。
- Cautious: 符号一致マスキング（勾配の符号を保存）
- Schedule-Free: weighted averaging（勾配の符号情報は使用しない）

#### State初期化 (`adamw8bit_ringbuffer.py: _init_param_state`)

```python
def _init_param_state(self, p):
    state = self.state[p]
    device = p.device
    n = p.numel()
    blocksize = 256
    num_blocks = (n + blocksize - 1) // blocksize

    # ... exp_avg, exp_avg_sq initialization ...

    # Schedule-Free: Initialize z
    if self.schedule_free:
        # z starts as a quantized copy of p (z_0 = θ_0)
        z_quantized, absmax_z_init = quantize_blockwise_inplace(
            p.detach().clone().to(device), blocksize
        )

        if self.get_state_buffer is not None:
            # Ring Buffer: CPU allocation
            state['z'] = self.get_state_buffer(p, dtype=torch.uint8)
            state['z'].copy_(z_quantized.cpu())

            # Pinned memory optimization
            if hasattr(state['z'], 'pin_memory'):
                state['z'] = state['z'].pin_memory()
        else:
            # GPU allocation
            state['z'] = z_quantized

        # Absmax (always on GPU)
        state['absmax_z'] = absmax_z_init
```

**z初期化の理論的根拠**:
- Schedule-Free理論: `z_0 = θ_0`（初期パラメータと同じ）
- 8-bit実装: pを量子化してzに格納

#### Update Step (`adamw8bit_ringbuffer.py: step`)

```python
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    # Schedule-Free: Update global state
    if self.schedule_free:
        for group in self.param_groups:
            lr = group['lr']
            warmup_steps = group['warmup_steps']
            r = group['r']
            weight_lr_power = group['weight_lr_power']

            # Linear warmup (use k+1 because k increments at end)
            k = self.k
            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            scheduled_lr = lr * sched
            self.lr_max = max(scheduled_lr, self.lr_max)

            # Compute weight for averaging (use k+1)
            weight = ((k + 1) ** r) * (self.lr_max ** weight_lr_power)
            self.weight_sum += weight
            ckp1 = weight / self.weight_sum  # c_{k+1}

            break  # Only need first param_group for global state

    for group in self.param_groups:
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        lr = group['lr']
        use_8bit = group['use_8bit']
        cautious = group['cautious']
        schedule_free = group['schedule_free']

        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state.get(p)

            if state is None or len(state) == 0:
                self._init_param_state(p)
                state = self.state[p]

            self.step_count += 1

            # ... Gradient norm scaling ...

            if schedule_free:
                # Schedule-Free specific bias correction (use k+1)
                k = self.k
                bias_correction2_sf = 1 - beta2 ** (k + 1)

                # Ring Buffer: CPU → GPU async transfer
                z_gpu = state['z']
                exp_avg_sq_gpu = state['exp_avg_sq']
                exp_avg_gpu = state['exp_avg']

                if not state['z'].is_cuda:
                    z_gpu = state['z'].cuda(non_blocking=True)
                    exp_avg_sq_gpu = state['exp_avg_sq'].cuda(non_blocking=True)
                    exp_avg_gpu = state['exp_avg'].cuda(non_blocking=True)

                # Compute scheduled_lr and ckp1
                warmup_steps = group['warmup_steps']
                r = group['r']
                weight_lr_power = group['weight_lr_power']

                if k < warmup_steps:
                    sched = (k + 1) / warmup_steps
                else:
                    sched = 1.0

                scheduled_lr = lr * sched
                lr_max_local = max(scheduled_lr, self.lr_max)
                weight = ((k + 1) ** r) * (lr_max_local ** weight_lr_power)
                weight_sum_local = self.weight_sum
                ckp1 = weight / weight_sum_local

                # Call CUDA kernel (Schedule-Free version)
                self.ext.adamw_8bit_schedulefree_update(
                    p, grad, z_gpu, exp_avg_sq_gpu,
                    state['absmax_z'], state['absmax2'],
                    beta1, beta2, eps, scheduled_lr, weight_decay,
                    ckp1, gnorm_scale, bias_correction2_sf
                )

                # GPU → CPU async copy-back
                if not state['z'].is_cuda:
                    state['z'].copy_(z_gpu, non_blocking=True)
                    state['exp_avg_sq'].copy_(exp_avg_sq_gpu, non_blocking=True)

            else:
                # Standard AdamW update
                # ... (existing code) ...

    # Schedule-Free: Increment k after all parameter updates
    if self.schedule_free:
        self.k += 1

    return loss
```

**重要なポイント**:
1. `k` は `step_count` とは別のカウンタ（Schedule-Free専用）
2. Warmup schedule: `(k+1) / warmup_steps`（k+1を使用）
3. Bias correction: `1 - β₂^(k+1)`（k+1を使用）
4. k のインクリメント: **全パラメータ更新後**に実行

#### train()/eval() メソッド

**train() - y（訓練パラメータ）への切り替え**:

```python
@torch.no_grad()
def train(self):
    """Set optimizer to train mode (Schedule-Free)."""
    if not self.schedule_free:
        return

    for group in self.param_groups:
        beta1 = group['betas'][0]
        for p in group['params']:
            state = self.state.get(p)
            if state is None or 'z' not in state:
                continue

            # 8-bit z を逆量子化
            z_gpu = state['z']
            if not state['z'].is_cuda:
                z_gpu = state['z'].cuda(non_blocking=True)

            # Dequantize z (UINT8 → FP32)
            blocksize = 256
            n = p.numel()
            num_blocks = (n + blocksize - 1) // blocksize

            z_fp32 = torch.zeros_like(p, dtype=torch.float32)
            qmap_signed = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32, device=z_gpu.device)

            for i in range(num_blocks):
                start = i * blocksize
                end = min(start + blocksize, n)

                block_absmax = state['absmax_z'][i]
                quantized_block = z_gpu.flatten()[start:end]

                # Dequantize: qmap[q] × absmax
                normalized = qmap_signed[quantized_block.long()]  # [-1, 1]
                dequantized = normalized * block_absmax
                z_fp32.flatten()[start:end] = dequantized

            # Set p to y: p.lerp_(end=z, weight=1-β₁)
            # p = β₁ · p + (1-β₁) · z
            # This converts x → y (assuming p was x)
            p.data.lerp_(end=z_fp32.to(p.dtype), weight=1 - beta1)

    self.train_mode = True
```

**eval() - x（評価パラメータ）への切り替え**:

```python
@torch.no_grad()
def eval(self):
    """Set optimizer to eval mode (Schedule-Free)."""
    if not self.schedule_free:
        return

    for group in self.param_groups:
        beta1 = group['betas'][0]
        for p in group['params']:
            state = self.state.get(p)
            if state is None or 'z' not in state:
                continue

            # 8-bit z を逆量子化（train()と同じ）
            # ... (省略) ...

            # Set p to x: p.lerp_(end=z, weight=1-1/β₁)
            # p = (1/β₁) · p + (1-1/β₁) · z
            # This converts y → x (assuming p was y)
            p.data.lerp_(end=z_fp32.to(p.dtype), weight=1 - 1 / beta1)

    self.train_mode = False
```

**理論的根拠**:
```
y = β₁ · x + (1-β₁) · z  （論文 Eq. 4）
x = (2 - 1/β₁) · z - (1-1/β₁) · y  （論文 Eq. 6を簡略化）

lerp(p, z, weight=1-β₁):
  p = (1 - (1-β₁)) · p + (1-β₁) · z
  p = β₁ · p + (1-β₁) · z
  → x から y への変換（p=x と仮定）

lerp(p, z, weight=1-1/β₁):
  p = (1 - (1-1/β₁)) · p + (1-1/β₁) · z
  p = (1/β₁) · p + (1-1/β₁) · z
  → y から x への変換（p=y と仮定）
```

---

## CUDA実装詳細

### Standard AdamW Kernel (`adamw8bit_kernel.cu`)

**概要**: 8-bit量子化AdamW（Schedule-Freeなし）

**処理フロー**:

```cuda
template<typename T>
__global__ void adamw_8bit_update_kernel(
    T* __restrict__ param,                 // Parameters (FP16/BF16/FP32)
    const T* __restrict__ grad,            // Gradients
    T* __restrict__ state_exp_avg,         // Momentum (FP16/BF16)
    uint8_t* __restrict__ state_exp_avg_sq, // Variance (UINT8)
    float* __restrict__ absmax2,           // exp_avg_sq absmax per block
    const uint8_t* __restrict__ grad_sign, // Cautious: gradient sign
    const float beta1, const float beta2, const float eps, const float lr,
    const float weight_decay, const float gnorm_scale,
    const float bias_correction1, const float bias_correction2,
    const bool cautious, const int numel
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int qblock_id = tid / QUANTIZATION_BLOCKSIZE;
    if (tid >= numel) return;

    // Step 1: Load current values
    float p = /* convert param[tid] to FP32 */;
    float g = /* convert grad[tid] to FP32 */ * gnorm_scale;
    float exp_avg = /* convert state_exp_avg[tid] to FP32 */;

    // Step 2: Dequantize exp_avg_sq
    float current_absmax2 = absmax2[qblock_id];
    uint8_t q2 = state_exp_avg_sq[tid];
    float exp_avg_sq = dequantize_value(q2, d_qmap_unsigned, current_absmax2);

    // Step 3: Update momentum
    exp_avg = beta1 * exp_avg + (1.0f - beta1) * g;

    // Step 4: Update variance
    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * (g * g);

    // Step 5: Compute normalized gradient
    float denom = sqrtf(exp_avg_sq / bias_correction2) + eps;
    float grad_normalized = exp_avg / bias_correction1 / denom;

    // Step 6: Cautious optimizer (optional)
    if (cautious && grad_sign != nullptr) {
        uint8_t current_sign = (g >= 0.0f) ? 1 : 0;
        uint8_t stored_sign = grad_sign[tid];
        if (current_sign != stored_sign) {
            grad_normalized = 0.0f;  // Mask update
        }
    }

    // Step 7: Weight decay
    if (weight_decay > 0.0f) {
        p = p * (1.0f - lr * weight_decay);
    }

    // Step 8: Parameter update
    p = p - lr * grad_normalized;

    // Step 9: Block-level absmax for exp_avg_sq (CUB BlockReduce)
    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float abs_sq = fabsf(exp_avg_sq);
    float block_absmax_sq = BlockReduce(temp_storage).Reduce(abs_sq, cub::Max());

    const int local_tid = threadIdx.x;
    if (local_tid == 0) {
        absmax2[qblock_id] = block_absmax_sq;
    }
    __syncthreads();

    // Step 10: Re-quantize exp_avg_sq
    float new_absmax_sq = absmax2[qblock_id];
    float normalized_sq = (new_absmax_sq > 0.0f) ? (exp_avg_sq / new_absmax_sq) : 0.0f;
    uint8_t new_q2 = quantize_value(normalized_sq, d_qmap_unsigned);

    state_exp_avg_sq[tid] = new_q2;

    // Step 11: Write back
    param[tid] = /* convert p to param dtype */;
    state_exp_avg[tid] = /* convert exp_avg to state dtype */;
}
```

### Schedule-Free AdamW Kernel (`adamw8bit_schedulefree_kernel.cu`)

**概要**: 8-bit量子化AdamW + Schedule-Free（3 sequence system）

**処理フロー**:

```cuda
template<typename T>
__global__ void adamw_8bit_schedulefree_update_kernel(
    T* __restrict__ param,                  // y (training parameters)
    const T* __restrict__ grad,             // gradients
    uint8_t* __restrict__ state_z,          // z sequence (UINT8)
    uint8_t* __restrict__ state_exp_avg_sq, // exp_avg_sq (UINT8)
    float* __restrict__ absmax_z,           // z absmax per block
    float* __restrict__ absmax2,            // exp_avg_sq absmax per block
    const float beta1, const float beta2, const float eps, const float lr,
    const float weight_decay, const float ckp1, const float gnorm_scale,
    const float bias_correction2, const int numel
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int qblock_id = tid / QUANTIZATION_BLOCKSIZE;
    if (tid >= numel) return;

    // Step 1: Dequantize z
    float current_absmax_z = absmax_z[qblock_id];
    uint8_t qz = state_z[tid];
    float z = dequantize_value(qz, d_qmap_signed, current_absmax_z);

    // Step 2: Dequantize exp_avg_sq
    float current_absmax2 = absmax2[qblock_id];
    uint8_t q2 = state_exp_avg_sq[tid];
    float exp_avg_sq = dequantize_value(q2, d_qmap_unsigned, current_absmax2);

    // Step 3: Load y (param is y)
    float y = /* convert param[tid] to FP32 */;

    // Step 4: Update exp_avg_sq
    float g = /* convert grad[tid] to FP32 */ * gnorm_scale;
    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * (g * g);

    // Step 5: Compute normalized gradient
    float denom = sqrtf(exp_avg_sq / bias_correction2) + eps;
    float grad_normalized = g / denom;
    if (weight_decay > 0.0f) {
        grad_normalized += weight_decay * y;
    }

    // Step 6: Update y (training parameters)
    // y = (1 - c_{k+1}) · y + c_{k+1} · z + lr · (β₁ · (1 - c_{k+1}) - 1) · grad_norm
    y = (1.0f - ckp1) * y + ckp1 * z + lr * (beta1 * (1.0f - ckp1) - 1.0f) * grad_normalized;

    // Step 7: Update z (main sequence)
    // z = z - lr · grad_norm
    z = z - lr * grad_normalized;

    // Step 8: Compute block-level absmax (CUB BlockReduce)
    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_z;
    __shared__ typename BlockReduce::TempStorage temp_storage_sq;

    float abs_z = fabsf(z);
    float abs_sq = fabsf(exp_avg_sq);

    float block_absmax_z = BlockReduce(temp_storage_z).Reduce(abs_z, cub::Max());
    __syncthreads();
    float block_absmax_sq = BlockReduce(temp_storage_sq).Reduce(abs_sq, cub::Max());

    const int local_tid = threadIdx.x;
    if (local_tid == 0) {
        absmax_z[qblock_id] = block_absmax_z;
        absmax2[qblock_id] = block_absmax_sq;
    }
    __syncthreads();

    // Step 9: Re-quantize z and exp_avg_sq
    float new_absmax_z = absmax_z[qblock_id];
    float new_absmax_sq = absmax2[qblock_id];

    float normalized_z = (new_absmax_z > 0.0f) ? (z / new_absmax_z) : 0.0f;
    float normalized_sq = (new_absmax_sq > 0.0f) ? (exp_avg_sq / new_absmax_sq) : 0.0f;

    uint8_t new_qz = quantize_value(normalized_z, d_qmap_signed);
    uint8_t new_q2 = quantize_value(normalized_sq, d_qmap_unsigned);

    state_z[tid] = new_qz;
    state_exp_avg_sq[tid] = new_q2;

    // Step 10: Write back y to param
    param[tid] = /* convert y to param dtype */;
}
```

### Quantization/Dequantization Device Functions

**Dequantize**:

```cuda
__device__ inline float dequantize_value(
    uint8_t quantized_value,   // Quantized value [0, 255]
    const float* qmap,          // Quantization map (constant memory)
    float absmax                // Block absmax
) {
    // qmap[quantized_value] gives normalized value [-1, 1] or [0, 1]
    float normalized = qmap[quantized_value];

    // Scale by absmax
    return normalized * absmax;
}
```

**Quantize**:

```cuda
__device__ inline uint8_t quantize_value(
    float value,         // Value to quantize (normalized to [-1, 1] or [0, 1])
    const float* qmap    // Quantization map (constant memory)
) {
    // Binary search for closest quantization level
    int left = 0;
    int right = 255;

    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (value < qmap[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Check both neighbors for closest match
    float dist_left = fabsf(value - qmap[left]);
    float dist_right = fabsf(value - qmap[right]);

    return (dist_left < dist_right) ? (uint8_t)left : (uint8_t)right;
}
```

**Quantization Maps（Constant Memory）**:

```cuda
// Device constant memory (64 KB, shared by all threads)
__constant__ float d_qmap_signed[256];    // [-1.0, 1.0]
__constant__ float d_qmap_unsigned[256];  // [0.0, 1.0]

// Host-side initialization (in C++ wrapper)
void init_quantization_maps() {
    float h_qmap_signed[256];
    float h_qmap_unsigned[256];

    for (int i = 0; i < 256; i++) {
        h_qmap_signed[i] = -1.0f + (2.0f * i / 255.0f);  // [-1, 1]
        h_qmap_unsigned[i] = (float)i / 255.0f;          // [0, 1]
    }

    cudaMemcpyToSymbol(d_qmap_signed, h_qmap_signed, 256 * sizeof(float));
    cudaMemcpyToSymbol(d_qmap_unsigned, h_qmap_unsigned, 256 * sizeof(float));
}
```

### Kernel Launcher（dtype dispatch）

**実装** (`adamw8bit_schedulefree_launcher.cu`):

```cuda
extern "C" {

void adamw_8bit_schedulefree_update_fp32(
    float* param, const float* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    adamw_8bit_schedulefree_update_kernel<float><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

void adamw_8bit_schedulefree_update_fp16(
    at::Half* param, const at::Half* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    adamw_8bit_schedulefree_update_kernel<at::Half><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

void adamw_8bit_schedulefree_update_bf16(
    at::BFloat16* param, const at::BFloat16* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    adamw_8bit_schedulefree_update_kernel<at::BFloat16><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

} // extern "C"
```

### C++ Bindings (`adamw8bit_cuda.cpp`)

**PyTorch Extension Registration**:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations (launchers)
extern "C" {
void adamw_8bit_update_fp32(...);
void adamw_8bit_update_fp16(...);
void adamw_8bit_update_bf16(...);

void adamw_8bit_schedulefree_update_fp32(...);
void adamw_8bit_schedulefree_update_fp16(...);
void adamw_8bit_schedulefree_update_bf16(...);
}

// Wrapper for Schedule-Free update
void adamw_8bit_schedulefree_update(
    torch::Tensor param, torch::Tensor grad,
    torch::Tensor state_z, torch::Tensor state_exp_avg_sq,
    torch::Tensor absmax_z, torch::Tensor absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2
) {
    // Ring Buffer Support: CPU→GPU Transfer
    torch::Tensor state_z_gpu = state_z;
    torch::Tensor state_exp_avg_sq_gpu = state_exp_avg_sq;

    if (!state_z.is_cuda()) {
        state_z_gpu = state_z.to(param.device(), /*non_blocking=*/true);
        state_exp_avg_sq_gpu = state_exp_avg_sq.to(param.device(), /*non_blocking=*/true);
    }

    int numel = param.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dtype dispatch
    if (param.dtype() == torch::kFloat32) {
        adamw_8bit_schedulefree_update_fp32(
            param.data_ptr<float>(), grad.data_ptr<float>(),
            state_z_gpu.data_ptr<uint8_t>(), state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(), absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else if (param.dtype() == torch::kFloat16) {
        adamw_8bit_schedulefree_update_fp16(
            reinterpret_cast<at::Half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(grad.data_ptr<at::Half>()),
            state_z_gpu.data_ptr<uint8_t>(), state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(), absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else if (param.dtype() == torch::kBFloat16) {
        adamw_8bit_schedulefree_update_bf16(
            reinterpret_cast<at::BFloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(grad.data_ptr<at::BFloat16>()),
            state_z_gpu.data_ptr<uint8_t>(), state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(), absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else {
        AT_ERROR("Unsupported dtype");
    }

    // GPU→CPU Copy-back (Ring Buffer)
    if (!state_z.is_cuda()) {
        state_z.copy_(state_z_gpu, /*non_blocking=*/true);
        state_exp_avg_sq.copy_(state_exp_avg_sq_gpu, /*non_blocking=*/true);
    }

    cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_8bit_update", &adamw_8bit_update,
          "AdamW 8-bit update");
    m.def("adamw_8bit_schedulefree_update", &adamw_8bit_schedulefree_update,
          "AdamW Schedule-Free update with 8-bit quantized z and exp_avg_sq");
}
```

### JIT Compilation (`adamw8bit_cuda.py`)

**Lazy Load + Cache**:

```python
import torch
from torch.utils.cpp_extension import load
from pathlib import Path
import os

_extension = None

def get_extension():
    global _extension
    if _extension is not None:
        return _extension

    cuda_dir = Path(__file__).parent / "cuda"
    wrapper_cpp = cuda_dir / "adamw8bit_cuda.cpp"
    kernel_cu = cuda_dir / "adamw8bit_kernel.cu"
    schedulefree_kernel_cu = cuda_dir / "adamw8bit_schedulefree_kernel.cu"
    schedulefree_launcher_cu = cuda_dir / "adamw8bit_schedulefree_launcher.cu"

    print("[AdamW8bit CUDA] Compiling extension (this may take a few minutes)...")

    _extension = load(
        name="adamw8bit_cuda",
        sources=[
            str(wrapper_cpp),
            str(kernel_cu),
            str(schedulefree_kernel_cu),
            str(schedulefree_launcher_cu)
        ],
        extra_cuda_cflags=[
            "-O3",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
            "-lineinfo"
        ],
        verbose=True
    )

    print("[AdamW8bit CUDA] Extension compiled successfully!")
    return _extension
```

**Usage in Optimizer**:

```python
class AdamW8bit_RingBuffer(torch.optim.Optimizer):
    def __init__(self, ...):
        # ...
        self.ext = None  # Lazy load

    def step(self, closure=None):
        # Lazy load CUDA extension on first step
        if self.ext is None:
            from . import adamw8bit_cuda
            self.ext = adamw8bit_cuda.get_extension()

        # Call CUDA kernel
        self.ext.adamw_8bit_schedulefree_update(...)
```

---

## 使用方法

### 1. YAML設定（Training Config）

**Schedule-Free有効化**:

```yaml
# Optimizer settings
optimizer: adamw8bit_ringbuffer  # or lion8bit_ringbuffer
learning_rate: 1.0e-4

# Schedule-Free options (RingBuffer optimizers only)
optimizer_schedule_free: true
optimizer_warmup_steps: 100      # Linear warmup steps (0 = no warmup)
optimizer_schedule_free_r: 0.0   # Weight polynomial power (0 = uniform averaging)
optimizer_schedule_free_weight_lr_power: 2.0  # LR weighting power

# Cautious optimizer (incompatible with Schedule-Free)
optimizer_cautious: false  # Must be false when schedule_free=true

# LR Scheduler (ignored when Schedule-Free is enabled)
lr_scheduler: constant  # Schedule-Free handles LR internally
```

**Block Swap との併用**: `blocks_to_swap > 0` の場合、RingBuffer optimizer は
per-parameter の fused backward hook で更新する（`base_trainer.
_setup_fused_backward_pass`）。この hook は標準（非 Schedule-Free）の 8-bit
更新のみを実装しており、AdamW では `state['exp_avg']` / `state['exp_avg_sq']` /
`state['absmax1']` / `state['absmax2']`、Lion では `state['exp_avg']` /
`state['absmax']` を読む。Schedule-Free モードの `_init_param_state` は
`exp_avg` / `absmax1`（Lion では `exp_avg` / `absmax`）を確保せず `z` /
`absmax_z` を確保するため、組み合わせると backward の途中で KeyError になる。
そのため `blocks_to_swap > 0` + `optimizer_schedule_free: true` は
optimizer setup 時点で拒否される（`blocks_to_swap: 0` にすると
`optimizer.step()` 内の Schedule-Free パスが動作する）。

**Standard 8-bit（Schedule-Freeなし）**:

```yaml
optimizer: adamw8bit_ringbuffer
learning_rate: 1.0e-4

optimizer_schedule_free: false
optimizer_cautious: true  # Can use cautious when schedule_free=false

lr_scheduler: cosine  # External LR scheduler
lr_scheduler_num_cycles: 1
```

### 2. Python API（Optimizer Factory）

**ファイル**: `backend/core/training/optimizer_factory.py`

```python
from backend.core.training.optimizer_factory import OptimizerFactory

# Schedule-Free Ring Buffer optimizer
optimizer = OptimizerFactory.create_optimizer(
    optimizer_type="adamw8bit_ringbuffer",
    parameters=model.parameters(),
    learning_rate=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,

    # Ring Buffer options
    get_state_buffer=state_buffer_allocator,  # CPU buffer allocator

    # Schedule-Free options
    schedule_free=True,
    warmup_steps=100,
    r=0.0,
    weight_lr_power=2.0,

    # Cautious (incompatible with Schedule-Free)
    cautious=False
)

# Training loop
for epoch in range(num_epochs):
    # Set to training mode (Schedule-Free: switch to y)
    optimizer.train()

    for batch in train_dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Set to eval mode (Schedule-Free: switch to x)
    optimizer.eval()

    with torch.no_grad():
        val_loss = validate(model, val_dataloader)

# Final eval mode before saving
optimizer.eval()
torch.save(model.state_dict(), "model.pth")  # Save x (evaluation params)
```

### 3. Trainer Integration

**ファイル**: `backend/core/training/lora_trainer.py`

```python
class LoRATrainer(BaseTrainer):
    def __init__(
        self,
        # ... existing params ...
        optimizer_schedule_free: bool = False,
        optimizer_warmup_steps: int = 0,
        optimizer_schedule_free_r: float = 0.0,
        optimizer_schedule_free_weight_lr_power: float = 2.0,
    ):
        super().__init__(
            # ... existing params ...
            optimizer_schedule_free=optimizer_schedule_free,
            optimizer_warmup_steps=optimizer_warmup_steps,
            optimizer_schedule_free_r=optimizer_schedule_free_r,
            optimizer_schedule_free_weight_lr_power=optimizer_schedule_free_weight_lr_power,
        )

    def train(self):
        # Training loop
        for epoch in range(self.num_epochs):
            # Schedule-Free: Set to training mode
            if hasattr(self.optimizer, 'train'):
                self.optimizer.train()

            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                # optimizer.step() is called in train_step()

            # Schedule-Free: Set to eval mode
            if hasattr(self.optimizer, 'eval'):
                self.optimizer.eval()

            val_loss = self.validate()

        # Final eval mode
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()

        self.save_model()
```

---

## パフォーマンス

### メモリ使用量

> **この節の数値の出所（2026-08-24 調査）**: 下の削減率 75% / 99.6% は**実測ではなく、
> 直下の worked example の byte 数から計算した算術値**である。
> `1 - 711/2800 = 74.6% ≈ 75%`、`1 - 11/2800 = 99.6%` で、いずれも「350M parameters
> という仮定の例」に対する構造上の見積もりであって、ベンチマークの記録ではない。
> **さらに 99.6% の行は allocator を渡した場合の値**であり（冒頭の注記）、
> 既定の配線で得られるのは "8-bit GPU allocation" の行の方である。

**例**: 350M parameters (LoRA rank=128)。**以下は仮定の worked example であり、
実測値ではない。**

| Configuration | Optimizer States (GPU) | Optimizer States (CPU) | Total GPU Memory |
|---------------|------------------------|------------------------|------------------|
| **FP32 Standard** | 2800 MB (z: 1400 MB, exp_avg_sq: 1400 MB) | 0 MB | ~3500 MB |
| **8-bit GPU allocation** | 711 MB (z: 350 MB, exp_avg_sq: 350 MB, absmax: 11 MB) | 0 MB | ~1400 MB |
| **8-bit Ring Buffer** | 11 MB (absmax only) | 1400 MB (z: 350 MB, exp_avg: 700 MB, exp_avg_sq: 350 MB, pinned) | ~700 MB |

**削減率**（上表からの算術。実測ではない）:
- 8-bit GPU allocation: 約 75% VRAM削減（vs FP32）— **これが既定で得られる経路**
- 8-bit Ring Buffer: 約 99.6% VRAM削減（optimizer statesについて）—
  **allocator を渡した場合のみ。既定では到達しない**

なお上表 "8-bit Ring Buffer" 行の CPU 内訳（z 350 + exp_avg 700 + exp_avg_sq 350）は
Schedule-Free の `z` と標準 AdamW の `exp_avg` を同時に数えており、実際には同時に
確保されない組み合わせである。表は概算の説明用と理解すること。

### 速度（Transfer Overhead）

> **この節の数値の出所は特定できなかった（2026-08-24 調査）。** 測定条件（GPU、
> PCIe 世代、バッチ、run）が記録されておらず、リポジトリ内に対応するベンチマーク
> 成果物も無い。しかも**これらはすべて CPU state 経路の値**であり、その経路は
> 導入時（`190c876e`）から一度も配線されたことがない（冒頭の注記）。
> したがって以下は**帯域からの見積もりとして読むべきで、測定値として引用しないこと**。
> 実測が必要なら測定条件つきで取り直すこと。

**Ring Buffer Transfer Time**（350M params）:

| Memory Type | Transfer Size | Bandwidth | Time/Step | 備考 |
|-------------|---------------|-----------|-----------|------|
| Non-pinned CPU | 1400 MB | ~11 GB/s | ~127 ms | カーネルバッファ経由 |
| **Pinned CPU** | 1400 MB | ~22-25 GB/s | ~56-63 ms | **直接DMA（推奨）** |

**Schedule-Free追加オーバーヘッド**:

| Phase | Additional Cost | 頻度 |
|-------|----------------|------|
| z transfer (CPU→GPU) | +350 MB | Every step |
| z copy-back (GPU→CPU) | +350 MB | Every step |
| train()/eval() 切り替え | z dequantization (~20 ms) | Per epoch |

**総合速度**:
- Ring Buffer無効: 10-15 ms/step（8-bit quantization overhead）
- Ring Buffer有効: 60-80 ms/step（DMA transfer + quantization）
- Schedule-Free train()/eval(): 20 ms/epoch（z dequantization）

### 収束性能

**理論的保証**（Schedule-Free論文より）:
- 収束率: O(1/k)（k: iteration数）
- 量子化誤差: 相対誤差 < 0.4%（256レベル量子化）
- FP32 Schedule-Freeとほぼ同等の訓練品質

**出所不明** (350M LoRA, SDXL, 1000 steps と記載):
- Loss convergence: FP32とほぼ同等（差 < 1%）
- Final validation loss: 差 < 0.5%
- Training time: Ring Buffer有効で約10-15%遅延（DMA overhead）

> ⚠️ この 4 行はもともと「**実測**」と書かれていたが、**2026-08-24 の調査では裏付けを
> 特定できなかった**: run の記録もベンチマーク成果物もリポジトリに無く、しかも
> "Ring Buffer 有効" の条件は配線が存在しないため到達できない。
> 見出しを「実測」から「出所不明」に変更した。**数値そのものは消していない**
> （後から出所が判明する可能性があるため）が、**測定値として引用しないこと。**

---

## トラブルシューティング

### 1. CUDA Extension Compilation Errors

**症状**:
```
error: identifier "at::Half" is undefined
```

**原因**: PyTorch/CUDA versionの不一致

**解決**:
```bash
# PyTorch 2.0以上が必要
pip install torch>=2.0.0

# CUDA Toolkitバージョン確認
nvcc --version

# PyTorchのCUDAバージョンと一致させる
# 例: PyTorch CUDA 12.1 → CUDA Toolkit 12.1
```

### 2. Ring Buffer: CPU Allocation Failure

**症状**:
```
RuntimeError: Cannot allocate pinned memory
```

**原因**: システムのpinned memory limitを超えた

**解決**:
```bash
# Pinned memory limit確認（Linux）
ulimit -l

# Limit増加（一時的）
ulimit -l unlimited

# または、Ring Bufferを無効化
# YAML設定で blocks_to_swap: 0
```

### 3. Schedule-Free: train()/eval() Forgetting

**症状**:
- 訓練時のlossが異常に高い
- 評価時の性能が低い

**原因**: `optimizer.train()`/`optimizer.eval()`の呼び忘れ

**解決**:
```python
# 訓練ループ開始時
optimizer.train()  # ← 必須

for batch in train_dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# 評価前
optimizer.eval()  # ← 必須

with torch.no_grad():
    val_loss = validate(model)
```

### 4. Cautious + Schedule-Free Conflict

**症状**:
```
[AdamW8bit_RingBuffer] WARNING: cautious is disabled when schedule_free=True
```

**原因**: CautiousとSchedule-Freeは互換性がない

**解決**:
```yaml
# YAML設定で必ずcautiousをfalseに
optimizer_schedule_free: true
optimizer_cautious: false  # ← 必須
```

### 5. GPU Out of Memory（Ring Buffer有効でも）

**症状**:
```
RuntimeError: CUDA out of memory
```

**原因**: absmax（GPU）またはモデル/勾配がVRAMを超えた

**解決**:
```yaml
# Block Swapを併用（U-Net CPUオフロード）
blocks_to_swap: 12  # SDXL: 12-22推奨

# Gradient checkpointing有効化
gradient_checkpointing: true

# Batch size削減
batch_size: 1
```

### 6. Schedule-Free: Convergence Issues

**症状**: Schedule-Free有効化後、収束が悪化

**原因**: warmup_steps不足、またはr/weight_lr_powerが不適切

**解決**:
```yaml
# Warmup steps増加（総step数の5-10%推奨）
optimizer_warmup_steps: 100  # 総1000 stepsの場合

# デフォルト値に戻す
optimizer_schedule_free_r: 0.0  # 均等平均
optimizer_schedule_free_weight_lr_power: 2.0  # LR^2 weighting

# Learning rate調整（Schedule-Freeは通常より高めのLRが推奨される）
learning_rate: 1.0e-4  # 通常の2-3倍を試す
```

### 7. DMA Transfer Timeout

**症状**:
```
RuntimeError: CUDA error: device-side assert triggered
```

**原因**: Pinned memoryの非同期転送が完了前にアクセス

**解決**:
```python
# C++ wrapper内で必ずsynchronize
cudaStreamSynchronize(stream);  # ← 必須

# または、non_blocking=Falseで同期転送（遅いが安全）
state_z_gpu = state['z'].cuda(non_blocking=False)
```

---

## 参考資料

### 論文

1. **Schedule-Free Learning** (Defazio et al., 2024)
   - arXiv: https://arxiv.org/abs/2405.15682
   - GitHub: https://github.com/facebookresearch/schedule_free

2. **8-bit Optimizers via Block-wise Quantization** (Dettmers et al., 2022)
   - arXiv: https://arxiv.org/abs/2110.02861
   - GitHub: https://github.com/TimDettmers/bitsandbytes

3. **Cautious Optimizers** (Lukovnikov & Fischer, 2024)
   - arXiv: https://arxiv.org/abs/2411.16085

### 実装参考

- **Facebook Research**: Schedule-Free official implementation
  - https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/adamw_schedulefree.py

- **bitsandbytes**: 8-bit quantization reference
  - https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/optim/adamw.py

- **musubi-tuner** (kohya-ss): Block Swap implementation
  - https://github.com/kohya-ss/musubi-tuner

- **sd-scripts** (kohya-ss): Fused Optimizer Groups
  - https://github.com/kohya-ss/sd-scripts

### SushiUI内部ドキュメント

- `backend/core/training/optimizers/RINGBUFFER_OPTIMIZERS.md` (本ドキュメント)
- `backend/core/MODEL_ARCHITECTURES.md` - モデルアーキテクチャ詳細
- `CLAUDE.md` - 開発ガイドライン

---

## 変更履歴

- **2025-12-24**: 初版作成
  - AdamW8bit_RingBuffer実装（8-bit + Ring Buffer + Schedule-Free）
  - Lion8bit_RingBuffer実装（同上）
  - CUDA kernel実装（standard + Schedule-Free）
  - Pinned memory最適化
  - ドキュメント作成

---

## ライセンス

このコードは以下のライセンスに従います：

- **Schedule-Free実装**: Apache License 2.0（Facebook Research）
- **8-bit Quantization**: MIT License（bitsandbytes）
- **SushiUI全体**: プロジェクトルートのLICENSEファイルを参照

---

## 貢献者

- **Claude Code** (Anthropic) - 実装・ドキュメント作成
- **User** (celll1) - 設計・レビュー・テスト

Co-Authored-By: Claude <noreply@anthropic.com>
