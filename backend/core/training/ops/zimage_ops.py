"""zimage_ops.py — Z-Image loader + attention-backend free functions (plan P3a).

These are the VERBATIM bodies of ``BaseTrainer._load_zimage_components`` and
``BaseTrainer._setup_attention_backend_zimage`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3a): the arch handler is built at the END of
``BaseTrainer.__init__`` (base_trainer.py:1115), AFTER ``_load_model_components``
runs (:1104). So the load-time dispatcher CANNOT use ``trainer.arch`` here. Both
the base_trainer dispatcher AND ``arch/zimage.py`` call these free functions, so
the body is defined exactly once and stays byte-identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def load_components(trainer) -> None:
    """Load Z-Image model components."""
    print(f"{trainer.log_prefix} Detected Z-Image model")
    print(f"{trainer.log_prefix} Loading Z-Image components from {trainer.model_path}")

    from core.model_loader import ModelLoader
    components = ModelLoader.load_zimage_from_diffusers(
        model_path=trainer.model_path,
        device="cpu",
        torch_dtype=trainer.weight_dtype
    )

    # Store components
    trainer.transformer_original = components["transformer"]
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Z-Image specific: no text_encoder_2, no unet
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Convert VAE to vae_dtype
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # A training process is DEQUANT-ONLY (see ideogram4_ops.load_components for
    # the full reasoning). A Z-Image transformer CAN be weight-only quantized:
    # ModelLoader._swap_zimage_quantized_linears swaps in Int8Linear/Fp8Linear
    # whenever the on-disk state dict carries per-row scales, and an INT8
    # artifact is MIXED (high-crest layers fall back to e4m3), so a single
    # trainer can own both module types at once. Training-time sample generation
    # runs under the pipeline's no_grad denoise loop, which would otherwise make
    # the validation previews W8A8 while the trained weights are not. Applied to
    # ``transformer_original`` because that is the module the quantized Linears
    # live in; the batched wrapper below only forwards into it. Each format has
    # its own per-instance opt-out, so disabling one does not disable the other.
    # A no-op on a bf16 checkpoint.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    from core.models.ideogram4.vendor.int8_linear import disable_int8_mm
    for _label, _module in (("transformer", trainer.transformer_original),
                            ("text_encoder", trainer.text_encoder)):
        if _module is not None:
            disable_scaled_mm(_module, label=f"zimage training {_label}")
            disable_int8_mm(_module, label=f"zimage training {_label}")

    # Wrap transformer with BatchedZImageWrapperOptimized
    from core.models.batched_zimage_wrapper import BatchedZImageWrapperOptimized
    print(f"{trainer.log_prefix} Wrapping Z-Image Transformer with BatchedZImageWrapperOptimized")
    trainer.transformer = BatchedZImageWrapperOptimized(trainer.transformer_original)
    print(f"{trainer.log_prefix} Phase 2 optimization: Complete batched processing")

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_zimage(trainer.attention_backend)

    # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Z-Image)")
    elif hasattr(trainer.transformer, 'enable_gradient_checkpointing'):
        trainer.transformer.enable_gradient_checkpointing()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Z-Image Transformer")
    else:
        print(f"{trainer.log_prefix} WARNING: Gradient checkpointing not available for Z-Image Transformer")

    # Enable gradient checkpointing for Text Encoder
    if trainer.gradient_checkpointing and hasattr(trainer.text_encoder, 'gradient_checkpointing_enable'):
        trainer.text_encoder.gradient_checkpointing_enable()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Text Encoder (Qwen3)")

    # Freeze all base weights (full parameter training will unfreeze specific layers later)
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Setup Block Swap if enabled (before moving to GPU)
    trainer.layer_offload_conductor = None  # Will be initialized if blocks_to_swap > 0

    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap enabled for training: {trainer.blocks_to_swap} blocks")
        print(f"{trainer.log_prefix} Using LayerOffloadConductor (Ring Buffer implementation)")
        print(f"{trainer.log_prefix} Pinned memory: {trainer.use_pinned_memory}")

        # Import new ring buffer implementation
        from core.memory_management import LayerOffloadConductor

        # Check if transformer has layers attribute
        if not hasattr(trainer.transformer_original, 'layers'):
            raise ValueError(
                f"Transformer must have 'layers' attribute for Block Swap. "
                f"Found: {type(trainer.transformer_original)}"
            )

        # Initialize Layer Offload Conductor
        trainer.layer_offload_conductor = LayerOffloadConductor(
            layers=trainer.transformer_original.layers,
            blocks_to_swap=trainer.blocks_to_swap,
            device=trainer.device,
            use_pinned_memory=trainer.use_pinned_memory,
            cpu_buffer_size_mb=8192,  # 8GB CPU buffer for layer params
            activation_buffer_size_mb=4096,  # 4GB CPU buffer for activations
            enable_prefetch=True,  # Enable prefetching next layer
            enable_activation_offload=False  # Disable for now (experimental)
        )

        # Attach to transformer for reference
        trainer.transformer_original._layer_offload_conductor = trainer.layer_offload_conductor

        # Register hooks for automatic layer swapping
        trainer.layer_offload_conductor.register_hooks()

        print(f"{trainer.log_prefix} LayerOffloadConductor initialized successfully")
        print(f"{trainer.log_prefix} Ring buffer allocation strategy enabled")
    else:
        print(f"{trainer.log_prefix} Block Swap disabled (blocks_to_swap=0)")
        # Move Transformer to GPU normally
        print(f"{trainer.log_prefix} Moving Transformer to {trainer.device}...")
        trainer.transformer_original.to(trainer.device)
        # Note: trainer.transformer.transformer is the same object as trainer.transformer_original
        # No need to call trainer.transformer.to(device) again

    print(f"{trainer.log_prefix} Z-Image model loaded successfully")
    print(f"{trainer.log_prefix} Scheduler type: {trainer.scheduler.__class__.__name__}")
    print(f"{trainer.log_prefix} VAE latent channels: {trainer.vae.config.latent_channels}")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Z-Image models.

    Sets ``ZImageAttention._attention_backend`` on BOTH module objects (the
    importlib-loaded ``sys.modules['zimage_transformer']`` used by the loaded
    transformer AND ``core.models.zimage_transformer``) so the dual-module
    hazard cannot leave the two disagreeing (design 2.2).

    Falls back silently (keeps the default backend) if the module can't be
    resolved, so a missing dependency never aborts training.
    """
    import sys

    b = trainer._resolve_training_backend(backend)
    applied = False
    try:
        # The transformer is loaded via importlib with module name
        # "zimage_transformer"; set the attr on the ACTUAL module used.
        if 'zimage_transformer' in sys.modules:
            zimage_transformer_module = sys.modules['zimage_transformer']
            zimage_transformer_module.ZImageAttention._attention_backend = b
            applied = True
            print(f"{trainer.log_prefix} Set Z-Image attention backend '{b}' on "
                  f"module {zimage_transformer_module.__name__}")
        # Also set on core.models.zimage_transformer so both module objects agree.
        from core.models.zimage_transformer import ZImageAttention
        ZImageAttention._attention_backend = b
        applied = True
        if b == 'native':
            print(f"{trainer.log_prefix} [OK] Z-Image attention backend set to native")
        else:
            print(f"{trainer.log_prefix} [OK] Z-Image attention backend enabled: {b}")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Z-Image attention backend '{b}': {e}")
        if not applied:
            print(f"{trainer.log_prefix} Continuing with the default attention backend "
                  f"(ensure flash-attn is installed for flash: pip install flash-attn)")


def encode_prompt(trainer, prompt: str, max_sequence_length: int = 512):
    """Encode prompt using Qwen3 text encoder with chat template (Z-Image).

    VERBATIM body of ``BaseTrainer.encode_prompt_zimage`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    Returns (prompt_embeds, attention_mask).
    """
    # Format with Qwen chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = trainer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Tokenize
    text_inputs = trainer.tokenizer(
        formatted_prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(trainer.device)
    attention_mask = text_inputs.attention_mask.to(trainer.device).bool()

    # Encode with penultimate layer
    # Check if text encoder has FP8 weights (requires autocast)
    has_fp8_weights = trainer._has_fp8_text_encoder()

    with torch.no_grad():
        # For FP8 quantized text encoder, use autocast for mixed precision
        if has_fp8_weights:
            with torch.autocast(device_type='cuda', dtype=trainer.training_dtype):
                encoder_output = trainer.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                prompt_embeds = encoder_output.hidden_states[-2]
        else:
            encoder_output = trainer.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            prompt_embeds = encoder_output.hidden_states[-2]

    # Extract and detach outputs
    result_embeds = prompt_embeds[0].detach()
    result_mask = attention_mask[0].detach()

    # Free intermediate tensors to prevent VRAM accumulation
    del input_ids, encoder_output, prompt_embeds, attention_mask

    return result_embeds, result_mask


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Z-Image VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_zimage`` branch (self->trainer rename only). Runs
    inside the caller's ``with torch.no_grad()``; caller does the shared final
    dtype/CPU move.
    """
    # Z-Image VAE
    h = trainer.vae.encoder(image_tensor)
    if trainer.vae.quant_conv is not None:
        h = trainer.vae.quant_conv(h)
    mean, logvar = torch.chunk(h, 2, dim=1)
    latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    shift_factor = trainer.vae.config.shift_factor if trainer.vae.config.shift_factor is not None else 0.0
    latents = trainer.vae.config.scaling_factor * (latents - shift_factor)
    # Clean up intermediate tensors
    del h, mean, logvar
    return latents


def train_step(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Perform single training step (Z-Image).

    Args:
        latents: Image latents [B, C, H, W]
        prompt_embeds: Prompt embeddings [B, seq_len, 2560]
        attention_mask: Attention mask [B, seq_len]
        timesteps: Timesteps for this batch [B]. If None, sampled uniformly from [0, 1]
        debug_save_path: If provided, save latents for debugging
        debug_captions: Captions for debug output
        profile_vram: If True, print VRAM usage
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (unused for Z-Image, included for API consistency)

    Returns:
        Tuple of (loss tensor, reconstruction loss value)
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import add_noise_unified, print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_zimage] Start")

    # Z-Image uses Flow Matching with velocity prediction
    noise_process = getattr(trainer, 'noise_process', 'flow')  # Z-Image default: flow
    prediction_target = getattr(trainer, 'prediction_target', 'velocity')  # Z-Image default: velocity

    # Move latents to GPU with correct dtype
    # Latents come from cache (CPU, training_dtype) and must be moved to GPU before training
    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)

    # Sample random timesteps from [0, 1] if not provided
    batch_size = latents.shape[0]
    if timesteps is None:
        if trainer.timestep_sampler is not None:
            # Use timestep sampler (returns [0, 1] for flow matching)
            timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
        else:
            # Legacy behavior: uniform sampling from [0, 1]
            timesteps = torch.rand(batch_size, device=trainer.device)

    # Sample noise (standard normal distribution, now on GPU)
    noise = torch.randn_like(latents)

    # Add noise using unified framework
    noisy_latents = add_noise_unified(
        noise_process=noise_process,
        noise_scheduler=trainer.noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )

    if profile_vram:
        print_vram_usage("[train_step_zimage] Before Transformer forward")

    # Note: Gradient checkpointing automatically manages requires_grad
    # No need to manually set requires_grad_(True) - PyTorch handles this
    # prompt_embeds is always detached (from encode_prompt_zimage with no_grad)
    # attention_mask is bool type, does not need gradients

    # Add frame dimension for Z-Image: [B, C, H, W] -> [B, C, 1, H, W]
    noisy_latents_4d = noisy_latents.unsqueeze(2)

    # Predict velocity using Z-Image Transformer
    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            model_pred, _ = trainer.transformer(
                x=noisy_latents_4d,
                t=timesteps,
                cap_feats=prompt_embeds,
                cap_mask=attention_mask,
            )
    else:
        model_pred, _ = trainer.transformer(
            x=noisy_latents_4d,
            t=timesteps,
            cap_feats=prompt_embeds,
            cap_mask=attention_mask,
        )

    # Remove frame dimension: [B, C, 1, H, W] -> [B, C, H, W]
    model_pred = model_pred.squeeze(2)

    if profile_vram:
        print_vram_usage("[train_step_zimage] After Transformer forward")

    # Z-Image uses INVERTED velocity convention: v = latents - noise
    # This is opposite from standard Flow Matching (v = noise - latents)
    # diffusers Z-Image pipeline inverts the sign during inference: noise_pred = -model_output
    # So we train with target = latents - noise to match this convention
    target = latents - noise

    # Calculate MSE loss (always in fp32)
    loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss_per_sample = loss_per_element.mean([1, 2, 3])

    # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
    mse_loss = loss_per_sample.mean()

    # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
    regularization_loss = torch.tensor(0.0, device=trainer.device)

    # Compute predicted latent once (used by regularization losses and dual loss)
    # Z-Image inverse velocity: v = latents - noise, so x_0 = x_t + t * v
    predicted_latent_for_reg = None
    if trainer.snr_regularization_loss is not None or trainer.energy_regularization_loss is not None or trainer.reconstruction_loss_weight > 0:
        # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
        t = timesteps.float()
        while t.dim() < noisy_latents.dim():
            t = t.unsqueeze(-1)
        predicted_latent_for_reg = noisy_latents + t * model_pred

    # SNR regularization (周波数領域の過剰デノイズ抑制)
    if trainer.snr_regularization_loss is not None:
        # timesteps are already [0, 1] for flow matching
        snr_reg_loss = trainer.snr_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps
        )
        regularization_loss = regularization_loss + snr_reg_loss

    # Energy regularization (空間領域のエネルギー保存)
    if trainer.energy_regularization_loss is not None:
        energy_reg_loss = trainer.energy_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps
        )
        regularization_loss = regularization_loss + energy_reg_loss

    # Calculate reconstruction loss (for monitoring or dual loss training)
    # If reconstruction_loss_weight > 0, compute with gradients for backprop
    # Otherwise, compute without gradients (monitoring only)
    if trainer.reconstruction_loss_weight > 0:
        # Dual loss training: compute reconstruction loss with gradients
        # Reuse predicted_latent_for_reg if already computed (has gradients)
        if predicted_latent_for_reg is not None:
            predicted_latent_for_recon = predicted_latent_for_reg
        else:
            # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
            t = timesteps.float()
            while t.dim() < noisy_latents.dim():
                t = t.unsqueeze(-1)
            predicted_latent_for_recon = noisy_latents + t * model_pred

        recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
        recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
        recon_loss = recon_loss_per_sample.mean()

        # Normalized dual loss: alpha * pred_loss + beta * recon_loss (alpha + beta = 1.0)
        alpha = 1.0 - trainer.reconstruction_loss_weight
        beta = trainer.reconstruction_loss_weight
        combined_loss = alpha * mse_loss + beta * recon_loss

        # Total loss with regularization
        loss = combined_loss + regularization_loss
    else:
        # Standard training: prediction loss only
        # Calculate reconstruction loss for monitoring (no gradients)
        with torch.no_grad():
            # Reuse predicted_latent_for_reg if already computed, otherwise compute it
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg.detach()
            else:
                # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
                t = timesteps.float()
                while t.dim() < noisy_latents.dim():
                    t = t.unsqueeze(-1)
                predicted_latent_for_recon = noisy_latents + t * model_pred

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        # Total loss (prediction loss + regularization)
        loss = mse_loss + regularization_loss

    if profile_vram:
        print_vram_usage("[train_step_zimage] After loss calculation")

    # Debug save if requested
    if debug_save_path is not None:
        try:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                # Z-Image: x_0 = x_t + t * v (opposite sign from standard flow matching)
                t = timesteps.float()
                while t.dim() < noisy_latents.dim():
                    t = t.unsqueeze(-1)
                predicted_latent = noisy_latents + t * model_pred

            debug_data = {
                'latents': latents[0:1].detach().cpu(),
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_velocity': model_pred[0:1].detach().cpu(),
                'actual_velocity': target[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample[0].item(),
                'loss_batch_mean': loss.item(),
                'recon_loss': recon_loss_per_sample[0].item(),
                'recon_loss_batch_mean': recon_loss.item(),
                'batch_size': batch_size,
                'scheduler_type': 'FlowMatching',
            }

            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data['reference_image_path'] = first_ref

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
            del predicted_latent
        except Exception as _dbg_e:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {_dbg_e}")

    # Return loss tensor (with gradient), pred_loss value, and recon_loss value
    # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
    # The training loop will call .backward() on the loss tensor.
    pred_loss_value = mse_loss.item()
    recon_loss_value = recon_loss.item()

    # Free intermediate tensors explicitly to reduce VRAM usage
    # But keep 'loss' tensor for backward pass
    del noise, noisy_latents, noisy_latents_4d, model_pred, target
    del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

    return loss, pred_loss_value, recon_loss_value


# ============================================================
# Z-Image Sample Generation (plan P7)
# ============================================================
# Verbatim bodies of BaseTrainer._generate_sample_zimage /
# _run_zimage_denoising_loop / _decode_zimage_latents (base_trainer.py), moved
# out of the spine with the mechanical self.->trainer. receiver rename only
# (generate_sample additionally takes a sanctioned lazy base_trainer import for
# log_verbose). arch/zimage.py::sample() unpacks SampleContext into this.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = -1,
    negative_prompt: str = "",
) -> Image.Image:
    """
    Generate sample image during training (Z-Image).

    Args:
        prompt: Text prompt
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        seed: Random seed (-1 for random)

    Returns:
        PIL Image
    """
    from core.training.base_trainer import log_verbose

    print(f"{trainer.log_prefix} Generating Z-Image sample: {prompt[:50]}...")

    # Set models to eval mode for inference (same as lora_trainer.py.backup:2481-2484)
    trainer.transformer.eval()
    trainer.transformer_original.eval()
    trainer.vae.eval()
    trainer.text_encoder.eval()

    # Store original devices for restoration
    text_encoder_device = next(trainer.text_encoder.parameters()).device
    vae_device = next(trainer.vae.parameters()).device
    transformer_device = next(trainer.transformer_original.parameters()).device

    try:
        # ============================================================
        # Stage 0: Offload Transformer AND Optimizer State to CPU
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Offloading Transformer and Optimizer state to CPU")

        # Move Transformer to CPU
        trainer.transformer_original.to("cpu")

        # CRITICAL: Move Optimizer state (gradients, momentum) to CPU
        # Optimizer state (exp_avg, exp_avg_sq) stays on GPU even after model.to(cpu)
        # This can consume 2x model size in VRAM (for AdamW: exp_avg + exp_avg_sq)
        optimizer_state_dict = trainer.optimizer.state_dict()
        for param_id, state in optimizer_state_dict['state'].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                    state[key] = value.cpu()
        trainer.optimizer.load_state_dict(optimizer_state_dict)

        torch.cuda.empty_cache()
        log_verbose(f"{trainer.log_prefix} [Sample] Transformer and Optimizer state offloaded to CPU")

        # ============================================================
        # Stage 1: Text Encoding (Sequential Offloading Pattern)
        # ============================================================
        # Move Text Encoder to GPU for encoding
        if text_encoder_device != trainer.device:
            log_verbose(f"{trainer.log_prefix} [Sample] Moving Text Encoder to GPU for encoding")
            trainer.text_encoder.to(trainer.device)

        # Encode prompt
        prompt_embeds, attention_mask = trainer.encode_prompt_zimage(prompt)

        # Encode unconditional prompt only if CFG is enabled
        if guidance_scale > 1.0:
            uncond_embeds, uncond_mask = trainer.encode_prompt_zimage(negative_prompt)
        else:
            uncond_embeds, uncond_mask = None, None

        # Move Text Encoder back to CPU to free VRAM
        if text_encoder_device != trainer.device:
            log_verbose(f"{trainer.log_prefix} [Sample] Moving Text Encoder back to CPU")
            trainer.text_encoder.to(text_encoder_device)
        torch.cuda.empty_cache()

        # ============================================================
        # Stage 1.5: Move Transformer back to GPU for denoising
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Moving Transformer to GPU for denoising")
        trainer.transformer_original.to(transformer_device)
        torch.cuda.empty_cache()

        # Add batch dimension
        prompt_embeds = prompt_embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        if uncond_embeds is not None:
            uncond_embeds = uncond_embeds.unsqueeze(0)
            uncond_mask = uncond_mask.unsqueeze(0)

        # ============================================================
        # Stage 2: Denoising Loop (Transformer already on GPU from training)
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Running denoising loop (Transformer on GPU)")

        # Prepare latents with seed
        latent_height = height // 8
        latent_width = width // 8
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=trainer.device).manual_seed(seed)
        # Use FP32 for latents initialization (same as pipeline.py for numerical stability)
        latents = torch.randn(
            (1, trainer.vae.config.latent_channels, latent_height, latent_width),
            device=trainer.device,
            dtype=torch.float32,
            generator=generator,
        )

        # Setup scheduler (create new instance with same config)
        # Note: We cannot use from_config() because Z-Image scheduler.config is not a standard ConfigMixin
        inference_scheduler = type(trainer.scheduler)(
            num_train_timesteps=trainer.scheduler.config.get("num_train_timesteps", 1000),
            shift=trainer.scheduler.config.get("shift", 1.0),
            use_dynamic_shifting=trainer.scheduler.config.get("use_dynamic_shifting", False),
        )

        # Calculate dynamic shift for flow matching (same as pipeline.py:964-981)
        from core.zimage_utils import calculate_shift
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        mu = calculate_shift(
            image_seq_len,
            trainer.scheduler.config.get("base_image_seq_len", 256),
            trainer.scheduler.config.get("max_image_seq_len", 4096),
            trainer.scheduler.config.get("base_shift", 0.5),
            trainer.scheduler.config.get("max_shift", 1.15),
        )

        # Set scheduler parameters (same as pipeline.py:977-981)
        inference_scheduler.sigma_min = 0.0
        inference_scheduler.set_timesteps(num_inference_steps, device=trainer.device, mu=mu)

        # Denoising loop
        latents = _run_zimage_denoising_loop(
            trainer,
            latents=latents,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
            guidance_scale=guidance_scale,
            scheduler=inference_scheduler,
        )

        # Free prompt embeddings
        del prompt_embeds, attention_mask
        if uncond_embeds is not None:
            del uncond_embeds, uncond_mask

        # ============================================================
        # Stage 3: Offload Transformer to CPU, move VAE to GPU
        # ============================================================
        # Move Transformer to CPU to free VRAM for VAE decode
        print(f"{trainer.log_prefix} [Sample] Moving Transformer to CPU to free VRAM")
        trainer.transformer_original.to("cpu")
        torch.cuda.empty_cache()

        # Move VAE to GPU for decoding
        if vae_device != trainer.device:
            print(f"{trainer.log_prefix} [Sample] Moving VAE to GPU for decoding")
            trainer.vae.to(device=trainer.device, dtype=trainer.vae_dtype)

        # Decode latents
        image = _decode_zimage_latents(trainer, latents)

        # Move VAE back to CPU
        if vae_device != trainer.device:
            print(f"{trainer.log_prefix} [Sample] Moving VAE back to CPU")
            trainer.vae.to(device=vae_device, dtype=trainer.vae_dtype)

        # Free latents
        del latents
        torch.cuda.empty_cache()

        # ============================================================
        # Stage 4: Restore Transformer and Optimizer State to GPU
        # ============================================================
        print(f"{trainer.log_prefix} [Sample] Restoring Transformer and Optimizer state to GPU")

        # Move Transformer back to GPU
        trainer.transformer_original.to(transformer_device)

        # CRITICAL: Move Optimizer state back to GPU (skip for Ring Buffer optimizers)
        # AdamW8bit_RingBuffer and Lion8bit_RingBuffer keep states on CPU intentionally
        from ..optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
        from ..optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer
        if not isinstance(trainer.optimizer, (AdamW8bit_RingBuffer, Lion8bit_RingBuffer)):
            # Optimizer state must be on the same device as model parameters for training
            optimizer_state_dict = trainer.optimizer.state_dict()
            for param_id, state in optimizer_state_dict['state'].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                        state[key] = value.to(transformer_device)
            trainer.optimizer.load_state_dict(optimizer_state_dict)
            print(f"{trainer.log_prefix} [Sample] Optimizer state restored to GPU")
        else:
            print(f"{trainer.log_prefix} [Sample] Optimizer state kept on CPU (Ring Buffer)")

        torch.cuda.empty_cache()
        print(f"{trainer.log_prefix} [Sample] Transformer restored to GPU")

        return image

    finally:
        # Ensure all models are back to their original devices (safety fallback)
        # Text Encoder and VAE should already be on CPU from sequential offloading
        # Transformer should already be on GPU from restoration
        # But we check anyway in case of exceptions during sample generation

        # Restore models to train mode (same as lora_trainer.py.backup:2638-2639)
        trainer.transformer.train()
        trainer.transformer_original.train()


def _run_zimage_denoising_loop(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    uncond_embeds: torch.Tensor,
    uncond_mask: torch.Tensor,
    guidance_scale: float,
    scheduler,
) -> torch.Tensor:
    """Run Z-Image denoising loop for sample generation.

    Note: Uses transformer_original (not batched wrapper) for single-image inference.
    """
    # Autocast the denoise loop to the sampling compute dtype (transformer dtype,
    # typically bf16). This is unconditional (NOT gated on trainer.mixed_precision):
    # sampling always runs the DiT in its param dtype (bf16), while LoRA adapters
    # default to lora_dtype=fp32 on a bf16 base. Without autocast the bf16
    # activations hit the fp32 LoRA Linear weights and crash with a dtype mismatch
    # inside the forward — regardless of the mixed_precision flag. Mirrors the
    # anima/lens fix (a3db4a1); VAE decode stays outside in _decode_zimage_latents.
    compute_dtype = next(trainer.transformer_original.parameters()).dtype
    with torch.no_grad(), torch.autocast(device_type=trainer.device.type, dtype=compute_dtype):
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating")):
            # Check for stop flag during sample generation (allow graceful shutdown)
            stop_flag_file = trainer.output_dir / ".stop_training"
            if stop_flag_file.exists():
                print(f"\n{trainer.log_prefix} [Sample] Stop flag detected during sample generation, aborting...")
                raise KeyboardInterrupt("Training stopped by user during sample generation")

            # Skip last step if t=0 (flow matching termination, same as pipeline.py:1001-1004)
            if t == 0 and i == len(scheduler.timesteps) - 1:
                continue

            # Prepare input
            if guidance_scale > 1.0:
                latent_input = torch.cat([latents] * 2)
                embeds_input = torch.cat([uncond_embeds, prompt_embeds])
                mask_input = torch.cat([uncond_mask, attention_mask])
            else:
                latent_input = latents
                embeds_input = prompt_embeds
                mask_input = attention_mask

            # Predict noise (use original transformer for single-image inference)
            # Use inference interface: List[Tensor] format, positional args only (no cap_mask)

            # Prepare timestep (expand to batch size, same as inference pipeline)
            timestep = t.to(trainer.device).expand(latent_input.shape[0])

            # Normalize timestep to [0, 1] (Z-Image expects normalized timesteps)
            timestep = (1000 - timestep) / 1000

            # Convert latents to transformer dtype (same as inference pipeline:1037-1046)
            transformer_dtype = next(trainer.transformer_original.parameters()).dtype
            latent_input = latent_input.to(transformer_dtype)

            # Add channel dimension and convert to list (same as inference pipeline)
            latent_input_5d = latent_input.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
            latent_input_list = list(latent_input_5d.unbind(dim=0))  # List of [C, 1, H, W]

            # Convert embeddings to list (each item: [seq_len, 2560])
            embeds_input_list = list(embeds_input.unbind(dim=0))

            # Call transformer (inference interface: positional args, List format)
            model_out_list = trainer.transformer_original(
                latent_input_list,
                timestep,
                embeds_input_list,
            )[0]

            # Apply CFG if enabled (same as stable lora_trainer.py:2474-2492)
            batch_size = latents.shape[0]
            if guidance_scale > 1.0:
                # CFG output order matches input: [negative, positive]
                neg_out = model_out_list[:batch_size]  # negative (uncond)
                pos_out = model_out_list[batch_size:]  # positive (cond)
                noise_pred = []
                for j in range(batch_size):
                    neg = neg_out[j].float()
                    pos = pos_out[j].float()
                    # Standard CFG formula (consistent with stable version)
                    # pred = uncond + guidance_scale * (cond - uncond)
                    pred = neg + guidance_scale * (pos - neg)
                    noise_pred.append(pred)
                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

            # Remove frames dimension for scheduler (5D → 4D) and negate (same as stable version)
            noise_pred = -noise_pred.squeeze(2)

            # Denoise step
            latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

    return latents


def _decode_zimage_latents(trainer, latents: torch.Tensor) -> Image.Image:
    """Decode Z-Image latents to image."""
    # Unscale latents
    shift_factor = trainer.vae.config.shift_factor if trainer.vae.config.shift_factor is not None else 0.0
    latents = (latents / trainer.vae.config.scaling_factor) + shift_factor

    # Decode (convert to VAE dtype to match decoder weights)
    with torch.no_grad():
        latents = latents.to(trainer.vae.dtype)
        if trainer.vae.post_quant_conv is not None:
            latents = trainer.vae.post_quant_conv(latents)
        image = trainer.vae.decoder(latents)

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).astype(np.uint8)[0]

    return Image.fromarray(image)
