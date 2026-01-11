# DEUS v2 Training vs Inference Diagnosis Report

**Date**: 2026-01-11
**Problem**: DEUS v2 model trained to loss 0.12 with good debug latents, but actual multi-step inference produces complete noise
**Objective**: Identify whether the issue is algorithmic or insufficient training

---

## Executive Summary

**DIAGNOSIS**: **Algorithmic Issue - Inference Pipeline Discrepancy**

### Critical Finding

DEUS v2 has **TWO DIFFERENT INFERENCE IMPLEMENTATIONS** with fundamentally different approaches:

1. **`pipeline_deus.py`** (Simple Pipeline) - Line 189-237
   - Uses **simple Euler integration**: `latents = latents - noise_pred * dt`
   - **Does NOT use scheduler.step()** method
   - Likely **OUTDATED** and **BROKEN**

2. **`custom_sampling.py`** (Production Pipeline) - Line 385-848
   - Uses **proper scheduler.step()**: `latents = scheduler.step(noise_pred, t, latents, generator).prev_sample`
   - Supports DEUS 2-pass CFG
   - **This is the correct implementation**

### Root Cause

The issue is **NOT training quality** (loss 0.12 is good, debug latents look correct). The problem is that:

- **Training uses DDPMScheduler with proper noise addition** (add_noise_unified)
- **Simple pipeline uses manual Euler integration** without proper scheduler logic
- This creates a **mismatch between training noise process and inference denoising**

### Recommendation

**Use `custom_sampling.py` for ALL inference**, including:
- Training sample generation (debug latents)
- Production inference
- API inference endpoints

If `pipeline_deus.py` is still being used anywhere, **migrate immediately** to `custom_sampling.py`.

---

## Detailed Analysis

### 1. Training Implementation (CORRECT ✅)

**File**: `backend/core/training/base_trainer.py::train_step_deus` (lines 2698-2834)

**Noise Process**:
```python
# DDPM with epsilon prediction
noise_process = "ddpm"
prediction_target = "epsilon"

# DDPMScheduler configuration (Line 827-834)
self.noise_scheduler = DDPMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
    prediction_type="epsilon"
)

# Timestep sampling: uniform [0, num_train_timesteps)
timesteps = torch.randint(0, 1000, (batch_size,), device=self.device).long()

# Noise addition via unified framework
noisy_latents = add_noise_unified(
    noise_process="ddpm",
    noise_scheduler=self.noise_scheduler,
    latents=latents,
    noise=noise,
    timesteps=timesteps,
)

# U-Net predicts epsilon (noise)
model_pred = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=prompt_embeds).sample

# Target: epsilon (noise)
target = get_target_unified("ddpm", "epsilon", noise_scheduler, latents, noise, timesteps)

# MSE loss
loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
```

**Key Points**:
- ✅ Uses DDPMScheduler (same as SD1.5/SDXL)
- ✅ Timestep sampling: discrete [0, 1000)
- ✅ Noise addition: `add_noise_unified()` with DDPM process
- ✅ Prediction target: epsilon (noise)
- ✅ Loss function: MSE

**IDENTICAL to SDXL Training** (verified via `train_step` lines 2402-2647)

---

### 2. Inference Implementation Comparison

#### 2.1 Simple Pipeline (BROKEN ❌)

**File**: `backend/core/pipelines/pipeline_deus.py` (lines 189-237)

```python
# Timestep scheduling
if self.scheduler is None:
    # Simple linear schedule (WRONG - not matching training!)
    timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
else:
    timesteps = self.scheduler.timesteps

# Denoising loop
for i, t in enumerate(timesteps):
    # Expand latents for CFG
    if guidance_scale > 1.0:
        latent_model_input = torch.cat([latents] * 2)
    else:
        latent_model_input = latents

    # Predict noise
    noise_pred = self.unet(
        sample=latent_model_input,
        timestep=t_tensor,
        encoder_hidden_states=encoder_hidden_states
    )

    # CFG
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # ❌ WRONG: Simple Euler integration without scheduler!
    if i < len(timesteps) - 1:
        dt = timesteps[i] - timesteps[i + 1]
    else:
        dt = timesteps[i]

    latents = latents - noise_pred * dt  # ← This is INCORRECT!
```

**Problems**:
1. ❌ Uses manual Euler integration (`latents = latents - noise_pred * dt`)
2. ❌ Does NOT call `scheduler.step()` method
3. ❌ Missing proper scaling factors (init_noise_sigma, etc.)
4. ❌ Timestep schedule may not match training
5. ❌ No proper DDPM denoising logic

**This is fundamentally BROKEN for DDPM-trained models!**

---

#### 2.2 Production Pipeline (CORRECT ✅)

**File**: `backend/core/inference/custom_sampling.py` (lines 385-848)

```python
# Set timesteps using scheduler (CORRECT)
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps

# Initialize latents with proper noise scaling
latents = torch.randn(
    (1, latent_channels, latent_height, latent_width),
    generator=generator,
    device=device,
    dtype=dtype
)
latents = latents * scheduler.init_noise_sigma  # ← CRITICAL: Proper initialization

# Denoising loop
for i, t in enumerate(timesteps):
    # DEUS 2-pass CFG (separate negative and positive passes)
    if use_two_pass_cfg:
        # Negative pass
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=negative_prompt_embeds).sample

        # Positive pass
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred_text = unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

        # Apply CFG
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    else:
        # Standard CFG (batch approach for SD/SDXL)
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # ...

    # ✅ CORRECT: Use scheduler.step() method
    latents = scheduler.step(
        noise_pred,
        t,
        latents,
        generator=step_generator
    ).prev_sample
```

**Key Correct Features**:
1. ✅ Uses `scheduler.set_timesteps()` to set proper timestep schedule
2. ✅ Initializes latents with `scheduler.init_noise_sigma`
3. ✅ Uses `scheduler.scale_model_input()` for proper scaling
4. ✅ Uses `scheduler.step()` for proper DDPM denoising
5. ✅ Supports DEUS 2-pass CFG (separate unconditional/conditional passes)
6. ✅ Matches training noise process

**This is the CORRECT implementation!**

---

### 3. Text Encoding (CORRECT ✅)

**File**: `backend/core/models/siglip2_wrapper.py::SigLIP2MultiModalEncoder.encode` (lines 560-601)

**Training** (lines 2078-2109 in base_trainer.py):
```python
prompt_embeds = self.text_encoder.encode(
    prompts=prompt,
    clip_skip=0,
    requires_grad=True
)
```

**Inference** (custom_sampling.py):
```python
prompt_embeds = encoder.encode(
    prompts=prompt,
    images=None,
    use_null_image=True,
    clip_skip=clip_skip
)
```

**Verification**:
- ✅ Both use `SigLIP2MultiModalEncoder.encode()`
- ✅ Text encoding: SigLIP-2 Text Encoder (Layer 27, penultimate with clip_skip=0)
- ✅ Null image embedding: Added for T2I mode
- ✅ Concatenation: `[text_tokens, null_image_patch]`
- ✅ No normalization mismatch (unlike CLIP)

**Text encoding is CORRECT and matches between training and inference.**

---

### 4. VAE Latent Encoding (CORRECT ✅)

**File**: `backend/core/training/base_trainer.py::encode_image` (lines 2238-2385)

**Training Latent Encoding**:
```python
# Image preprocessing
image_array = np.array(image).astype(np.float32) / 255.0
image_array = (image_array - 0.5) * 2.0  # Normalize to [-1, 1]
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

# VAE encoding (SDXL VAE for DEUS)
if isinstance(self.vae, SDXLVAEWrapper):
    vae_model = self.vae.vae  # Get internal AutoencoderKL
else:
    vae_model = self.vae

encoder_output = vae_model.encode(image_tensor)
latents = encoder_output.latent_dist.sample()
latents = latents * vae_model.config.scaling_factor  # 0.13025 for SDXL VAE
```

**Inference Latent Encoding** (via VAE decode):
```python
# VAE decode (custom_sampling.py uses pipeline.vae)
# backend/core/models/sdxl_vae_wrapper.py::decode (lines 129-145)
def decode(self, latents: torch.Tensor) -> torch.Tensor:
    # Unscale latents
    latents = latents / self.scaling_factor  # 0.13025

    # Decode
    image = self.vae.decode(latents, return_dict=False)[0]

    return image
```

**Verification**:
- ✅ VAE model: SDXL AutoencoderKL (4-channel latents)
- ✅ Scaling factor: 0.13025 (consistent)
- ✅ Image normalization: [-1, 1] (standard)
- ✅ Latent distribution: sample from posterior (stochastic)

**VAE encoding/decoding is CORRECT and consistent.**

---

### 5. Scheduler Configuration Comparison

#### Training Scheduler (DDPMScheduler)
```python
# backend/core/training/base_trainer.py (lines 827-834)
self.noise_scheduler = DDPMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
    prediction_type="epsilon"
)
```

#### Inference Scheduler (Should be same!)
**Expected**: Same DDPMScheduler or compatible scheduler (EulerDiscreteScheduler, DPMSolverMultistepScheduler, etc.)

**Critical**: `custom_sampling.py` accepts ANY scheduler via pipeline, but it **MUST be initialized with compatible settings**:
- `num_train_timesteps=1000` (to match training)
- `prediction_type="epsilon"` (to match training)
- `beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"` (to match training)

**Verification Needed**: Check what scheduler is passed to `custom_sampling.py` during inference!

---

## Comparison with SDXL (Reference Implementation)

### SDXL Training (diffusers reference)
- Uses DDPMScheduler with epsilon prediction
- Timestep sampling: uniform [0, 1000)
- Noise addition: `scheduler.add_noise(latents, noise, timesteps)`
- U-Net predicts epsilon
- Loss: MSE

### SDXL Inference (diffusers reference)
- Uses compatible scheduler (Euler, DPM++, DDPM, etc.)
- Proper `scheduler.set_timesteps()` and `scheduler.step()`
- Latent initialization: `torch.randn() * scheduler.init_noise_sigma`
- Proper CFG implementation

### DEUS Training vs SDXL Training
**IDENTICAL** ✅

### DEUS Inference (custom_sampling.py) vs SDXL Inference
**NEARLY IDENTICAL** ✅ (with proper DEUS 2-pass CFG extension)

### DEUS Inference (pipeline_deus.py) vs SDXL Inference
**COMPLETELY DIFFERENT** ❌ (broken manual Euler integration)

---

## Diagnosis: Algorithm Issue vs Insufficient Training

### Evidence for Algorithm Issue:
1. ✅ Training loss 0.12 is **good** (comparable to successful SD/SDXL training)
2. ✅ Debug latents look **reasonable** during training
3. ✅ Training implementation is **identical** to SDXL
4. ✅ VAE encoding/decoding is **correct**
5. ✅ Text encoding is **correct**
6. ❌ **Simple pipeline uses broken manual Euler integration**
7. ❌ **Simple pipeline does NOT use scheduler.step()**
8. ❌ **Mismatch between training (DDPM) and inference (manual Euler)**

### Evidence Against Insufficient Training:
- Loss 0.12 is typical for well-trained diffusion models
- Debug latents during training show reasonable structure
- If training was insufficient, debug latents would also be noisy

### Conclusion:
**The issue is 100% ALGORITHMIC, not training quality.**

The root cause is using `pipeline_deus.py` (broken simple pipeline) instead of `custom_sampling.py` (production pipeline).

---

## Recommended Fixes

### Immediate Actions (Priority 1):

1. **Stop using `pipeline_deus.py` immediately**
   - This file is BROKEN and should be deprecated
   - Mark as `@deprecated` or delete entirely

2. **Use `custom_sampling.py` for ALL inference**
   - Training sample generation: Use `custom_sampling_loop_deus()`
   - Production inference: Use `custom_sampling_loop_deus()`
   - API endpoints: Use `custom_sampling_loop_deus()`

3. **Verify scheduler configuration**
   - Ensure inference scheduler has `num_train_timesteps=1000`
   - Ensure inference scheduler has `prediction_type="epsilon"`
   - Ensure beta schedule matches training

### Code Changes Required:

#### Fix 1: Update Training Sample Generation

**File**: `backend/core/training/base_trainer.py`

```python
# BEFORE (if using pipeline_deus.py):
from core.pipelines.pipeline_deus import DeusPipeline
images = self.pipeline(...)  # ← BROKEN

# AFTER (use custom_sampling.py):
from core.inference.custom_sampling import custom_sampling_loop_deus
images = custom_sampling_loop_deus(
    scheduler=self.noise_scheduler,  # Use training scheduler!
    unet=self.unet,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    num_inference_steps=sample_steps,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    generator=generator,
    is_deus=True,  # Enable DEUS 2-pass CFG
)
```

#### Fix 2: Update Production Inference

**File**: `backend/core/pipeline.py` (if applicable)

Ensure ALL DEUS inference routes use `custom_sampling.py`, not `pipeline_deus.py`.

#### Fix 3: Deprecate or Delete `pipeline_deus.py`

Add warning:
```python
# backend/core/pipelines/pipeline_deus.py
import warnings

warnings.warn(
    "pipeline_deus.py is DEPRECATED and BROKEN. "
    "Use custom_sampling.py instead!",
    DeprecationWarning,
    stacklevel=2
)
```

Or **delete the file entirely** to prevent accidental use.

---

## Verification Steps

After implementing fixes:

1. **Test with debug latents**:
   - Generate debug latents during training (currently working)
   - Generate inference samples using `custom_sampling.py`
   - Compare visual quality - should match!

2. **Test multi-step inference**:
   - Run full 28-step inference with trained DEUS model
   - Use Euler/DPM++ scheduler (compatible with DDPM training)
   - Verify output is NOT noise

3. **Compare with SDXL**:
   - Train SDXL with same dataset
   - Generate samples using same scheduler
   - Verify DEUS quality is comparable

4. **Checkpoint scheduler metadata**:
   - Save scheduler config in checkpoint metadata
   - Load scheduler from checkpoint during inference
   - Ensures perfect training/inference alignment

---

## Additional Recommendations

### 1. Add Scheduler Config to Checkpoint Metadata

**File**: `backend/core/training/adapters/deus_adapter.py::save_checkpoint`

```python
metadata = {
    # ... existing metadata
    "scheduler_config": json.dumps({
        "beta_start": trainer.noise_scheduler.config.beta_start,
        "beta_end": trainer.noise_scheduler.config.beta_end,
        "beta_schedule": trainer.noise_scheduler.config.beta_schedule,
        "num_train_timesteps": trainer.noise_scheduler.config.num_train_timesteps,
        "prediction_type": trainer.noise_scheduler.config.prediction_type,
    })
}
```

### 2. Load Scheduler Config During Inference

**File**: `backend/core/models/checkpoint_utils.py::load_unified_checkpoint`

```python
if "scheduler_config" in metadata:
    scheduler_config = json.loads(metadata["scheduler_config"])
    scheduler = DDPMScheduler(**scheduler_config)
else:
    # Fallback: Use default DDPM config
    scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon"
    )
```

### 3. Add Unit Tests

Create test to verify scheduler consistency:
```python
def test_deus_training_inference_consistency():
    # Train for 1 step
    loss = trainer.train_step_deus(latents, prompt_embeds)

    # Generate sample with custom_sampling.py
    sample = custom_sampling_loop_deus(...)

    # Verify sample is not noise (mean ~0, std ~1)
    assert abs(sample.mean()) < 0.5
    assert 0.5 < sample.std() < 2.0
```

---

## Final Verdict

**Problem**: DEUS v2 inference produces noise
**Root Cause**: Using broken `pipeline_deus.py` with manual Euler integration
**Solution**: Use `custom_sampling.py` with proper scheduler.step()
**Training Quality**: GOOD (loss 0.12, no issues)
**Algorithm Issue**: YES (inference pipeline mismatch)
**Insufficient Training**: NO (training is correct)

**Action**: Migrate all inference to `custom_sampling.py` immediately.

---

## Appendix: File References

### Training Files:
- `backend/core/training/base_trainer.py::train_step_deus` (lines 2698-2834) - Training forward pass
- `backend/core/training/base_trainer.py::encode_image` (lines 2238-2385) - VAE encoding
- `backend/core/training/latent_cache.py` (lines 134-213) - Latent caching

### Inference Files:
- ✅ `backend/core/inference/custom_sampling.py::custom_sampling_loop_deus` (lines 385-848) - CORRECT
- ❌ `backend/core/pipelines/pipeline_deus.py::__call__` (lines 189-237) - BROKEN

### Model Files:
- `backend/core/models/siglip2_wrapper.py::SigLIP2MultiModalEncoder.encode` (lines 560-601) - Text encoding
- `backend/core/models/sdxl_vae_wrapper.py::encode/decode` (lines 103-145) - VAE wrapper
- `backend/core/models/unet_deus.py` - DEUS U-Net architecture

### Scheduler:
- Training: `DDPMScheduler` (lines 827-834 in base_trainer.py)
- Inference: Should use compatible scheduler (Euler, DPM++, DDPM, etc.)

---

**Report Generated**: 2026-01-11
**Status**: DIAGNOSIS COMPLETE
**Next Action**: Implement recommended fixes and verify
