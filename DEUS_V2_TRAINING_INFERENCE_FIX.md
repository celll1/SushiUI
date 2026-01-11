# DEUS v2 Training vs Inference Discrepancy - Root Cause and Fix

## Problem Summary

DEUS v2 model trained to loss 0.12 with good debug latents during training, but actual multi-step inference produces complete noise.

## Root Cause Identified

**Text Embedding Format Mismatch Between Training and Inference**

### Training (BEFORE FIX)
```python
# backend/core/training/base_trainer.py:encode_prompt_deus() (Line 2082-2092)
prompt_embeds = self.text_encoder.encode(prompts=prompt, clip_skip=0, requires_grad=True)
# Returns: text embeddings only [1, text_seq_len, 1152]
# ❌ Missing null_image_embedding!
```

### Inference (CORRECT)
```python
# backend/core/pipeline.py:_deus_encode_prompt() (Line 2163-2174)
all_text_embeddings = encoder.text_encoder.encode(all_prompts, clip_skip=clip_skip)
negative_text_embeddings = all_text_embeddings[0:1]
positive_text_embeddings = all_text_embeddings[1:2]

# Add null image embedding (T2I doesn't use image encoder)
null_image_embedding = encoder.null_image_embedding  # [1, 1, 1152]
negative_encoder_hidden_states = torch.cat([negative_text_embeddings, null_image_embedding], dim=1)
encoder_hidden_states = torch.cat([positive_text_embeddings, null_image_embedding], dim=1)
# Returns: text + null image [1, text_seq_len + 1, 1152]
```

**Impact**:
- Training: U-Net trained on embeddings of shape `[1, text_seq_len, 1152]`
- Inference: U-Net receives embeddings of shape `[1, text_seq_len + 1, 1152]`
- Sequence length mismatch (+1 token) causes complete prediction failure

## Why This Wasn't Caught Earlier

1. **Loss 0.12**: Training loss was calculated correctly (noise prediction MSE), but the model learned wrong positional embeddings
2. **Debug Latents**: Debug latents (saved at training step) looked reasonable because they used the same incorrect embedding format
3. **No Validation**: No inference validation during training to detect this mismatch

## Fix Applied

### Modified File: `backend/core/training/base_trainer.py`

**Line 2077-2138: `encode_prompt_deus()` method**

#### Before (Incorrect)
```python
# Enable gradients when training text encoder
if requires_grad:
    if has_fp8_weights:
        with torch.autocast(device_type='cuda', dtype=self.training_dtype):
            prompt_embeds = self.text_encoder.encode(
                prompts=prompt,
                clip_skip=0,
                requires_grad=True
            )
    else:
        prompt_embeds = self.text_encoder.encode(
            prompts=prompt,
            clip_skip=0,
            requires_grad=True
        )
    result_embeds = prompt_embeds
```

#### After (Fixed)
```python
# Encode text using SigLIP2TextEncoder (returns text embeddings only)
# Then manually add null_image_embedding (to match inference behavior)
if requires_grad:
    if has_fp8_weights:
        with torch.autocast(device_type='cuda', dtype=self.training_dtype):
            text_embeds = self.text_encoder.text_encoder.encode(
                prompts=prompt,
                clip_skip=0,
                requires_grad=True
            )
    else:
        text_embeds = self.text_encoder.text_encoder.encode(
            prompts=prompt,
            clip_skip=0,
            requires_grad=True
        )

    # Add null image embedding if requested
    if use_null_image:
        null_image_embedding = self.text_encoder.null_image_embedding  # [1, 1, 1152]
        result_embeds = torch.cat([text_embeds, null_image_embedding], dim=1)
    else:
        result_embeds = text_embeds
```

**Key Changes**:
1. Call `self.text_encoder.text_encoder.encode()` (SigLIP2TextEncoder) instead of `self.text_encoder.encode()` (SigLIP2MultiModalEncoder)
2. Manually concatenate `null_image_embedding` when `use_null_image=True`
3. Same logic for both `requires_grad=True` and `requires_grad=False` branches
4. Updated docstring: `[1, text_seq_len + 1, 1152]` (text + null image embedding)

## Verification Steps

### Before Training
1. ✅ Syntax check: `python -m py_compile backend/core/training/base_trainer.py`

### During Training
1. Verify embedding shapes in log output:
   ```
   [DEUS] Positive embeddings shape: torch.Size([1, text_seq_len + 1, 1152])
   ```

2. Compare training sample generation vs actual inference:
   - Both should now produce non-noise images
   - Both use same embedding format

### After Training
1. Load checkpoint and run inference:
   ```python
   image = pipeline(
       prompt="1girl, anime",
       num_inference_steps=28,
       guidance_scale=7.0
   )
   ```
   - Should produce coherent image (not noise)

2. Check embedding shapes in inference log:
   ```
   [DEUS] Positive embeddings shape: torch.Size([1, text_seq_len + 1, 1152])
   ```

## Expected Results

- **Loss**: Should converge similar to before (loss 0.12 was not the problem)
- **Training Samples**: Should improve significantly (currently noisy, should become coherent)
- **Actual Inference**: Should produce coherent images (currently noise, should match training samples)

## Technical Details

### SigLIP-2 Architecture in DEUS
- **Text Encoder**: SigLIP-2 text model (1152D)
- **Image Encoder**: SigLIP-2 vision model (1152D, not used in T2I)
- **Null Image Embedding**: Learned embedding `[1, 1, 1152]` representing "no image" for T2I

### Why Null Image Embedding is Required
- DEUS U-Net expects combined text+image embeddings
- T2I mode: Use null_image_embedding instead of actual image features
- Without null_image_embedding: Sequence length mismatch → position embeddings misaligned → complete prediction failure

### Comparison with SDXL
- **SDXL**: Text-only conditioning (2048D concatenated CLIP embeddings)
- **DEUS**: Text+Image conditioning (1152D SigLIP-2 embeddings + null image token)
- DEUS requires explicit null_image_embedding for T2I, SDXL does not

## Related Files

### Core Files
- ✅ `backend/core/training/base_trainer.py` - **FIXED**
- ✅ `backend/core/pipeline.py` - Inference (already correct)
- ✅ `backend/core/models/siglip2_wrapper.py` - SigLIP2 encoder (no changes needed)

### Investigation Files
- `DEUS_V2_TRAINING_INFERENCE_DIAGNOSIS.md` - Initial investigation
- `DEUS_V2_TRAINING_INFERENCE_DIAGNOSIS_UPDATED.md` - Second investigation (ruled out scheduler issue)
- `DEUS_V2_TRAINING_INFERENCE_FIX.md` - This file (root cause and fix)

## Lessons Learned

1. **Always validate inference during training**: Run actual inference (not just training sample generation) to catch these mismatches
2. **Check embedding shapes carefully**: SigLIP-2 has both text-only and text+image modes, must use correct one
3. **Docstrings matter**: Original docstring mentioned "text + null image" but implementation was wrong
4. **Loss alone isn't enough**: Low training loss doesn't guarantee correct implementation

## Next Steps

1. Restart DEUS v2 training with this fix
2. Monitor training samples (should improve significantly)
3. Run inference validation every N steps
4. Compare results with previous training (should be much better)
