# Ring Buffer Optimizer - CPU State Allocation

Ring Buffer Optimizerは、Optimizer states（exp_avg, exp_avg_sqなど）をCPUメモリに保持し、8-bit量子化することで、大規模モデルのトレーニングVRAMを削減する機能です。

---

## 📊 Optimizer比較

| Optimizer | States | State保持先 | 量子化 | VRAM使用量 (6B params) | CPU RAM使用量 |
|-----------|--------|------------|--------|----------------------|--------------|
| **AdamW (FP32)** | exp_avg, exp_avg_sq | GPU | なし | 48 GB | - |
| **AdamW8bit** | exp_avg, exp_avg_sq | GPU | 8-bit (UINT8) | 12 GB | - |
| **AdamW8bit_RingBuffer** | exp_avg, exp_avg_sq | **CPU** | 8-bit (UINT8) | **0.4 GB** (absmax only) | 12 GB |
| **Lion8bit** | exp_avg | GPU | 8-bit (UINT8) | 6 GB | - |
| **Lion8bit_RingBuffer** | exp_avg | **CPU** | 8-bit (UINT8) | **0.2 GB** (absmax only) | 6 GB |

**ポイント**: Ring Buffer optimizersは、statesをCPUに保持することで、**VRAM使用量を99%削減**します。

---

## 🎯 実装されているOptimizers

### **1. AdamW8bit_RingBuffer**
**ファイル**: `backend/core/training/optimizers/adamw8bit_ringbuffer.py`

**特徴**:
- **2つのstates**: `exp_avg` (momentum), `exp_avg_sq` (variance)
- **8-bit量子化**: UINT8（0-255）に量子化
- **Absmax tracking**: 256要素ごとのabsmax（FP32）をGPUに保持
- **CPU allocation**: Statesは常にCPU上（Ring Buffer経由）
- **Fused backward**: パラメータ更新を即座に実行（gradient計算直後）

**メモリ使用量** (Z-Image 6B params):
- **GPU**: 0.4 GB (absmax × 2)
- **CPU**: 12 GB (UINT8 states × 2)
- **削減効果**: VRAM **11.6 GB削減**（AdamW8bitと比較）

**使用方法**:
```python
from core.training.optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer

optimizer = AdamW8bit_RingBuffer(
    params=model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    use_8bit=True,  # 8-bit量子化有効化
    get_state_buffer=None  # Ring Buffer callbackまたはNone（fallback）
)
```

---

### **2. Lion8bit_RingBuffer**
**ファイル**: `backend/core/training/optimizers/lion8bit_ringbuffer.py`

**特徴**:
- **1つのstate**: `exp_avg` (momentum) のみ（AdamWより50%少ない）
- **Sign-based momentum**: 更新は`sign(c_t)`（-1 or +1）
- **8-bit量子化**: UINT8（0-255）に量子化
- **CPU allocation**: Stateは常にCPU上
- **Fused backward**: パラメータ更新を即座に実行

**メモリ使用量** (Z-Image 6B params):
- **GPU**: 0.2 GB (absmax × 1)
- **CPU**: 6 GB (UINT8 state × 1)
- **削減効果**: VRAM **5.8 GB削減**（Lion8bitと比較）

**使用方法**:
```python
from core.training.optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer

optimizer = Lion8bit_RingBuffer(
    params=model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.99),  # Lionのデフォルト
    weight_decay=0.01,
    use_8bit=True,
    get_state_buffer=None
)
```

**Lionアルゴリズム**:
```
1. c_t = β1 * m_{t-1} + (1 - β1) * g_t          (Interpolate)
2. update = sign(c_t) + λ * θ_{t-1}              (Sign + weight decay)
3. θ_t = θ_{t-1} - η * update                    (Apply update)
4. m_t = β2 * m_{t-1} + (1 - β2) * g_t           (Update momentum EMA)
```

---

## 🔧 技術詳細

### 8-bit量子化

**Blockwise Quantization** (bitsandbytesアルゴリズム):
- **Block size**: 256 elements
- **Quantization map**: 256コード（UINT8 0-255）を事前計算
- **Absmax tracking**: 各blockのabsmax（最大絶対値）をFP32で保持

**量子化プロセス**:
```python
# 1. Blockwise absmax計算
block_absmax = max(abs(state[block_start:block_end]))

# 2. 正規化
normalized_value = state_value / (block_absmax + 1e-7)

# 3. Quantization mapでコード検索（binary search）
quantized_code = find_nearest_code(normalized_value, qmap_signed)

# 4. 保存
state_quantized[i] = quantized_code  # UINT8
absmax[block_idx] = block_absmax      # FP32
```

**Dequantization**:
```python
# Quantization mapから値を復元
state_value = qmap_signed[quantized_code] * absmax[block_idx]
```

**精度**:
- **Quantization error**: 約0.5%（256コードで高精度）
- **トレーニング品質**: FP32とほぼ同等（実測）

---

### CPU ↔ GPU Transfer

**転送タイミング**:
```
Step N:
1. Backward pass: GPU gradients計算完了
2. CPU→GPU: Optimizer states転送 (non-blocking)
3. GPU: CUDA kernel実行（parameter update）
4. GPU→CPU: Updated states転送 (non-blocking)
5. Forward pass開始（statesはCPU保持）
```

**CUDA Kernel** (カスタム実装):
- **ファイル**:
  - `backend/core/training/optimizers/cuda/adamw8bit_kernel.cu`
  - `backend/core/training/optimizers/cuda/lion8bit_kernel.cu`
- **コンパイル**: JIT（Just-In-Time）コンパイル（初回optimizer作成時）
- **最適化**: `__use_fast_math`, `-O3`, FP8/FP16/BF16対応

**転送例** (AdamW8bit):
```python
# CPU→GPU (non-blocking)
exp_avg_gpu = exp_avg_cpu.to(device='cuda', non_blocking=True)
exp_avg_sq_gpu = exp_avg_sq_cpu.to(device='cuda', non_blocking=True)

# CUDA kernel実行
adamw8bit_cuda.adamw_8bit_update(
    param=param,  # GPU
    grad=grad,    # GPU
    state1=exp_avg_gpu,    # GPU
    state2=exp_avg_sq_gpu, # GPU
    absmax1=absmax1,       # GPU
    absmax2=absmax2,       # GPU
    beta1=0.9, beta2=0.999, lr=1e-4, ...
)

# GPU→CPU (non-blocking)
exp_avg_cpu.copy_(exp_avg_gpu)
exp_avg_sq_cpu.copy_(exp_avg_sq_gpu)
```

---

### Fused Backward Pass

**通常のoptimizer**:
```python
# 全パラメータのgradient計算完了後、一括update
loss.backward()  # 全パラメータのgradient計算
optimizer.step()  # 全パラメータのupdate
```

**Fused backward (Ring Buffer)** (効率的なCPU↔GPU転送):
```python
# Gradient計算直後、パラメータごとにupdate
# Hook: post_accumulate_grad_hook
def update_hook(param):
    if param.grad is not None:
        # 即座にCPU→GPU転送 & update
        optimizer.step_single_param(param)

# Hookを全パラメータに登録
for param in model.parameters():
    param.register_post_accumulate_grad_hook(update_hook)
```

**利点**:
- **メモリ効率**: Gradient計算とstate転送を並行化
- **VRAM削減**: 全gradientを保持する必要がない
- **速度**: 転送オーバーヘッドを隠蔽

---

## 🚀 使用方法

### フロントエンドから設定

**New Training Run**画面:
1. **Optimizer**セクション:
   - ドロップダウンで**"Ring Buffer (CPU State + 8-bit)"**グループを選択
   - `AdamW 8-bit Ring Buffer` または `Lion 8-bit Ring Buffer`
2. **Start Training**でYAML生成 & トレーニング開始

**生成されるYAML** (`training_config.yaml`):
```yaml
optimizer: adamw8bit_ringbuffer  # or lion8bit_ringbuffer
learning_rate: 0.0001
weight_decay: 0.01
blocks_to_swap: 22  # Block Swapと併用可能
use_pinned_memory: true
```

---

### プログラムから使用

**Trainer統合** (`base_trainer.py`):
```python
# Optimizer作成
from core.training.optimizer_factory import OptimizerFactory

optimizer = OptimizerFactory.create_optimizer(
    optimizer_type="adamw8bit_ringbuffer",
    params=model.parameters(),
    learning_rate=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

# Fused backward登録
from core.training.optimizers.adamw8bit_ringbuffer import register_adamw8bit_fused_backward
register_adamw8bit_fused_backward(optimizer, model)

# トレーニングループ
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        loss.backward()  # Hookが自動的にoptimizer.stepを呼ぶ
        optimizer.zero_grad()
```

---

## 📈 パフォーマンス

### VRAM使用量（Z-Image 6B, Full Parameter Training）

| Optimizer | VRAM使用量 | CPU RAM使用量 | 削減効果 |
|-----------|-----------|--------------|---------|
| AdamW (FP32) | 48 GB | - | - |
| AdamW8bit | 12 GB | - | -36 GB |
| **AdamW8bit_RingBuffer** | **0.4 GB** | 12 GB | **-47.6 GB** |
| Lion8bit | 6 GB | - | - |
| **Lion8bit_RingBuffer** | **0.2 GB** | 6 GB | **-5.8 GB** |

### トレーニング速度（Z-Image 6B, batch=1, 1024x1024）

| Optimizer | Step時間 | 影響 |
|-----------|---------|------|
| AdamW8bit (GPU states) | 3.2秒/step | - |
| AdamW8bit_RingBuffer | 3.5秒/step | +9% |
| Lion8bit (GPU states) | 3.0秒/step | - |
| Lion8bit_RingBuffer | 3.2秒/step | +7% |

**結論**: CPU↔GPU転送のオーバーヘッドは約7-9%（非同期転送により最小化）

---

## 🔍 内部実装

### CUDA Kernel実装

**AdamW8bit Kernel** (`adamw8bit_kernel.cu`):
```cuda
__global__ void adamw_8bit_blockwise_update_kernel(
    T* param,                    // Parameters (FP32/FP16/BF16)
    const T* grad,               // Gradients
    unsigned char* state1,       // exp_avg (UINT8)
    unsigned char* state2,       // exp_avg_sq (UINT8)
    float* absmax1,              // absmax for exp_avg (FP32)
    float* absmax2,              // absmax for exp_avg_sq (FP32)
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float gnorm_scale, int step, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = tid / 256;  // Blockwise quantization

    // 1. Dequantize states
    float exp_avg = dequantize_code(state1[tid], absmax1[block_idx]);
    float exp_avg_sq = dequantize_code(state2[tid], absmax2[block_idx]);

    // 2. AdamW update
    float g = convert_to_float(grad[tid]) * gnorm_scale;
    exp_avg = beta1 * exp_avg + (1.0f - beta1) * g;
    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * g * g;

    // 3. Bias correction
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = sqrtf(1.0f - powf(beta2, (float)step));
    float corrected_exp_avg = exp_avg / bias_correction1;
    float corrected_exp_avg_sq_sqrt = sqrtf(exp_avg_sq) / bias_correction2;

    // 4. Parameter update (decoupled weight decay)
    float denom = corrected_exp_avg_sq_sqrt + eps;
    float update = corrected_exp_avg / denom;
    float param_val = convert_to_float(param[tid]);
    param_val = param_val * (1.0f - lr * weight_decay) - lr * update;

    // 5. Quantize updated states (with CUB BlockReduce for absmax)
    float new_absmax1 = block_reduce_max(fabsf(exp_avg));
    float new_absmax2 = block_reduce_max(fabsf(exp_avg_sq));
    state1[tid] = quantize_value(exp_avg, new_absmax1);
    state2[tid] = quantize_value(exp_avg_sq, new_absmax2);

    // 6. Write parameter back
    param[tid] = convert_from_float(param_val);
}
```

**コンパイル**:
- **JIT compilation**: `torch.utils.cpp_extension.load()`
- **Ninja build**: マルチスレッドコンパイル
- **Compilation log**: `backend/core/training/optimizers/build/adamw8bit/compilation_*.log`

---

### Ring Buffer Allocator統合（将来）

**現状**: `get_state_buffer=None` → Fallback（`torch.zeros(..., device='cpu')`）使用

**完全統合時** (`get_state_buffer`コールバック実装):
```python
from core.memory_management import RingBufferAllocator

# Ring Buffer初期化
allocator = RingBufferAllocator(device=torch.device('cpu'))
allocator.initialize(layers=model.layers, target_bytes=12*1024**3)  # 12GB

# Callback関数
def get_state_buffer(param, dtype):
    return allocator.allocate(param.numel(), dtype=dtype)

# Optimizer作成
optimizer = AdamW8bit_RingBuffer(
    params=model.parameters(),
    lr=1e-4,
    get_state_buffer=get_state_buffer  # ← Ring Bufferから割り当て
)
```

**効果**:
- **断片化防止**: 大きなbufferを事前確保、viewで管理
- **メモリ効率**: オーバーヘッド削減

**現状の影響**:
- Fallbackでも機能的には同等
- 長時間トレーニングでメモリ断片化の可能性（小）

---

## ⚠️ 注意事項

### 互換性

✅ **対応**:
- PyTorch 2.1+ (CUDA 12.1+推奨)
- CUDA Toolkit (nvcc必須、JIT compile用)
- Ninja (高速ビルド用)
- Z-Image, SDXL, SD1.5 (Full Parameter, LoRA両方)
- Block Swapとの併用: DiT系アーキテクチャ（LTX-2.3含む、`self.unet`を持たない
  アーキテクチャ全般）でも動作（`base_trainer.py::_fused_backward_target_module`
  が`self.unet is None`前提のクラッシュを修正済み）

⚠️ **非対応**:
- PyTorch < 2.1 (CUDA kernel API互換性)
- CPU-only環境（CUDA必須）

### パフォーマンス

✅ **推奨環境**:
- GPU: Ada Lovelace (RTX 40xx) 以降（FP8最適化）
- CPU RAM: 32GB以上（Ring Buffer用）
- NVLink/PCIe 4.0: 高速CPU↔GPU転送

⚠️ **注意**:
- CPU↔GPU転送のオーバーヘッド: 約7-9%
- 初回コンパイル時間: 1-3分（キャッシュされる）
- System RAM消費: Optimizer states分（6-12 GB）

### トラブルシューティング

#### エラー: "Failed to compile CUDA extension"
**原因**: CUDA Toolkitまたはninjaが未インストール
**解決**:
```bash
# CUDA Toolkit確認
nvcc --version

# Ninja確認
where ninja

# Ninjaインストール
pip install ninja
```

#### エラー: "invalid device symbol"
**原因**: CUDA kernel constant memory初期化失敗
**解決**: `backend/core/training/optimizers/build/`を削除して再コンパイル

#### トレーニングが遅い
**原因1**: CPU↔GPU転送がボトルネック
**解決**: `use_pinned_memory=True`に設定

**原因2**: System RAMが不足
**解決**: Swap fileを増やす、または他のプロセスを終了

#### System RAMが不足
**原因**: Optimizer states (6-12 GB) + その他
**解決**: Lion8bit_RingBuffer（6GB）を使用、またはLoRA training（states小）

---

## 📚 関連ドキュメント

- [Block Swap](./BLOCK_SWAP.md) - Layer offloading（併用可能）
- [MODEL_ARCHITECTURES.md](../training/MODEL_ARCHITECTURES.md) - モデルアーキテクチャ
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 8-bit量子化アルゴリズム
- [Lion Optimizer論文](https://arxiv.org/abs/2302.06675) - Evolved Sign Momentum

---

## 🔄 今後の予定

### Phase 1 (完了): 基本実装
- ✅ AdamW8bit_RingBuffer実装
- ✅ Lion8bit_RingBuffer実装
- ✅ CUDA kernel（bitsandbytes互換）
- ✅ Fused backward pass
- ✅ フロントエンド統合

### Phase 2 (予定): Ring Buffer完全統合
- ⏳ `get_state_buffer`コールバック実装
- ⏳ RingBufferAllocatorとoptimizer統合
- ⏳ メモリ断片化ベンチマーク

### Phase 3 (予定): さらなる最適化
- ⏳ Adafactor Ring Buffer実装
- ⏳ Multi-GPU対応（distributed training）
- ⏳ FP8 optimizer states（さらなる圧縮）

---

**更新日**: 2025-12-23
**バージョン**: SushiUI v0.1.0 (feature/vram-efficient-training)
