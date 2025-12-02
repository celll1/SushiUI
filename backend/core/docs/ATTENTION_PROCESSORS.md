# Attention Processors

This document explains the attention acceleration options available in this Stable Diffusion WebUI.

## Overview

Three attention processor types are supported:

1. **Normal** - PyTorch 2.0+ SDPA (automatically uses Flash Attention when available)
2. **SageAttention** - Quantized attention for 2-5x speedup
3. **FlashAttention** - Explicit Flash Attention 2

## Attention Types

### Normal (Default)

Uses PyTorch 2.0+'s `scaled_dot_product_attention` (SDPA), which automatically selects the best backend:
- Flash Attention (if CUDA and hardware support)
- Memory-efficient attention
- Math implementation (fallback)

**Pros:**
- Zero setup required
- Automatically optimal on supported hardware
- No additional dependencies

**Cons:**
- May not be as fast as explicit optimizations

### SageAttention

Uses INT8 quantization for QK^T and FP16/FP8 for PV to achieve 2-5x speedup over standard attention.

**Requirements:**
```bash
pip install sageattention
```

**Supported GPUs:**
- NVIDIA Ampere (RTX 30 series, A100)
- NVIDIA Ada (RTX 40 series)
- NVIDIA Hopper (H100)

**Pros:**
- 2-5x faster than Flash Attention
- Maintains accuracy (quantization is carefully designed)
- Works with most models

**Cons:**
- Requires `sageattention` package
- CUDA 12.0+ required
- Some models may produce artifacts (rare)

**Fallback:**
If SageAttention fails to import, automatically falls back to normal SDPA.

### FlashAttention

Explicit use of Flash Attention 2 via the `flash_attn` package.

**Requirements:**
```bash
pip install flash-attn --no-build-isolation
```

**Pros:**
- Explicit control over Flash Attention usage
- Well-tested and stable

**Cons:**
- Requires compilation (can be slow to install)
- PyTorch 2.0+ already uses Flash Attention automatically in SDPA

**Fallback:**
If `flash_attn` is not installed, falls back to PyTorch SDPA (which may still use Flash Attention internally).

## Configuration

### Backend Settings

Edit `backend/config/settings.py`:

```python
attention_type: str = "normal"  # "normal", "sage", "flash"
```

Or set via environment variable:

```bash
ATTENTION_TYPE=sage
```

### Frontend Settings

1. Open Settings page
2. Scroll to "Generation Behavior" card
3. Select "Attention Type" from dropdown
4. Click "Restart Backend" for changes to take effect

## Compatibility

### NAG (Normalized Attention Guidance)

When NAG is enabled, custom attention processors (SageAttention/FlashAttention) are **not applied**.

NAG uses its own specialized attention processor that implements guidance in attention space.

**Future Enhancement:** NAG could be modified to use SageAttention/FlashAttention internally.

### ControlNet

All attention types are compatible with ControlNet.

### LoRA

All attention types are compatible with LoRA.

### SDXL

All attention types support both SD 1.5 and SDXL models.

## Performance Comparison

Typical speedups on RTX 4090 (SDXL, 1024x1024, 30 steps):

| Attention Type | Time (seconds) | Speedup |
|----------------|----------------|---------|
| Normal (SDPA)  | ~15s           | 1.0x    |
| FlashAttention | ~15s           | 1.0x    |
| SageAttention  | ~6s            | 2.5x    |

*Note: Normal already uses Flash Attention on RTX 40 series, so explicit FlashAttention provides no additional speedup.*

## Troubleshooting

### SageAttention produces black/noisy images

Try the safer variant by modifying `attention_processors.py`:

```python
# In SageAttnProcessor.__init__
from sageattention import sageattn_qk_int8_pv_fp16_cuda
self.sageattn = sageattn_qk_int8_pv_fp16_cuda
```

This uses FP16 for PV instead of FP8, reducing overflow risk.

### Installation Issues

**SageAttention:**
```bash
# Requires CUDA 12.0+
pip install sageattention
```

**FlashAttention:**
```bash
# Requires compilation
pip install flash-attn --no-build-isolation

# If compilation fails, ensure you have:
# - CUDA toolkit installed
# - Compatible compiler (gcc/g++)
# - Enough disk space for compilation
```

## Implementation Details

### Code Structure

- `backend/core/attention_processors.py` - Processor implementations
- `backend/core/pipeline.py` - Integration with generation pipeline
- `backend/config/settings.py` - Configuration
- `frontend/src/app/settings/page.tsx` - UI settings

### Processor Lifecycle

1. **Before generation:** Custom processor is set on UNet (if not using NAG)
2. **During generation:** Processor is called for each attention layer
3. **After generation:** Original processors are restored

This ensures processors don't interfere with model loading or other operations.

## References

- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [PyTorch SDPA Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
