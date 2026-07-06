"""flux2_ops.py — FLUX.2 Klein loader + block-swap wiring + attention free
functions (P3c).

VERBATIM bodies of ``BaseTrainer._load_flux2_components``,
``BaseTrainer._flux2_block_swap_h2d_args``,
``BaseTrainer._wire_flux2_block_swap_driver`` and
``BaseTrainer._setup_attention_backend_flux2`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3c): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher calls ``load_components`` directly. ``_load_flux2_components``
is NOT called by ``_load_checkpoint_as_base`` (that method inlines FLUX.2 loading),
so the loader is deleted from base_trainer and dispatched directly here.

``block_swap_h2d_args`` / ``wire_block_swap_driver`` / ``setup_attention_backend``
keep 2-line delegators on the trainer because they have call sites BOTH in the
moved loader body AND in ``_load_checkpoint_as_base`` (which stays in the spine);
each body is defined exactly once here.

NOTE (behavior preservation, plan): ``block_swap_h2d_args`` calls
``enable_gradient_checkpointing()`` unconditionally as Gate 3 — the H2D
gradient-checkpointing site the GC-flag work deliberately left ungated. Moved
verbatim; behavior unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from core.attention import AttentionMode, to_diffusers_backend


def block_swap_h2d_args(trainer):
    """Policy gate + H2D args for FLUX.2 training block swap.

    FLUX.2 training block swap is supported ONLY via the H2D-only + frozen-base
    (LoRA) + gradient-checkpointing path. The standard (non-H2D) training swap has
    a pre-existing index inconsistency and is NOT functional, so anything that
    would activate it must raise a clear error instead.

    Returns a dict of kwargs to pass to create_flux_block_offloader
    ({"h2d_only": ..., "ring_size": ...}).

    Raises ValueError if the policy is violated.
    """
    # Gate 1: block swap for FLUX.2 training requires h2d_only.
    if not trainer.block_swap_h2d_only:
        raise ValueError(
            "FLUX.2 training block swap currently requires block_swap_h2d_only=True "
            "(the standard swap path is not yet functional)."
        )

    # Gate 2: requires a frozen base (LoRA training, not full-parameter FT).
    # The training mode is known at setup time via train_config['training_method'];
    # LoRA adapters are applied after this point, so we key off the mode rather than
    # inspecting requires_grad here. The offloader also has a lazy Full-FT
    # auto-detect+disable as a backstop.
    training_method = str(trainer.config.get("training_method", "lora") or "lora").strip().lower()
    if training_method != "lora":
        raise ValueError(
            "FLUX.2 training block swap (H2D-only) requires a frozen base, i.e. LoRA "
            f"training. Current training_method={training_method!r} updates the base "
            "weights (Full-FT), which needs D2H persistence and cannot use H2D-only "
            "block swap. Use training_method='lora' or disable Block Swap "
            "(blocks_to_swap=0)."
        )

    # Gate 3: requires gradient checkpointing on the transformer (H2D backward
    # re-reads base weights via recompute). Enable it if the transformer supports
    # the switch; then verify the attribute the wrapper checks becomes True.
    if hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        try:
            trainer.transformer.enable_gradient_checkpointing()
        except Exception as e:
            raise ValueError(
                "FLUX.2 training block swap (H2D-only) requires gradient checkpointing "
                f"on the transformer, but enable_gradient_checkpointing() failed: {e}"
            )
    if not getattr(trainer.transformer, "gradient_checkpointing", False):
        raise ValueError(
            "FLUX.2 training block swap (H2D-only) requires gradient checkpointing on "
            "the transformer (transformer.gradient_checkpointing must be True). The "
            "current transformer does not support enabling it, so H2D-only block swap "
            "cannot be used. Disable Block Swap (blocks_to_swap=0) instead."
        )

    print(f"{trainer.log_prefix} FLUX.2 block swap H2D-only enabled "
          f"(ring_size={trainer.block_swap_ring_size}, LoRA/frozen base, grad-ckpt on)")
    return {"h2d_only": True, "ring_size": trainer.block_swap_ring_size}


def wire_block_swap_driver(trainer):
    """Wire the offloader into the FLUX.2 forward/backward after devices are prepared.

    - Wraps self.transformer with Flux2BlockSwapWrapper (drives wait_for_block /
      submit_move_blocks_forward per block during forward). self.transformer itself
      is NOT replaced, so optimizer / LoRA / state_dict keep seeing the raw module.
    - Registers full-backward hooks so recompute-time reads pull blocks resident.
    """
    from core.models.flux2_block_swap_wrapper import Flux2BlockSwapWrapper

    trainer.flux2_transformer_wrapper = Flux2BlockSwapWrapper(
        trainer.transformer, trainer.flux2_block_offloader
    )
    trainer.flux2_block_offloader.register_backward_hooks()
    print(f"{trainer.log_prefix} FLUX.2 block swap driver wired "
          f"(wrapper + backward hooks registered)")


def load_components(trainer) -> None:
    """Load FLUX.2 Klein model components.

    FLUX.2 Klein architecture:
    - Qwen3 text encoder (Qwen3ForCausalLM)
    - Flux2Transformer2DModel (8 dual stream + 48 single stream blocks)
    - AutoencoderKLFlux2 (32ch latent with BatchNorm)
    - Flow matching with velocity prediction
    - 4D position coordinates for RoPE (T, H, W, L)

    Key differences from FLUX.1:
    - Single stream blocks use parallel attention+MLP (fused projections)
    - VAE uses BatchNorm for latent normalization
    - Text encoder extracts hidden states from layers 9, 18, 27
    """
    print(f"{trainer.log_prefix} Detected FLUX.2 Klein model")
    print(f"{trainer.log_prefix} Loading FLUX.2 components from {trainer.model_path}")

    from core.model_loader import ModelLoader

    components = ModelLoader.load_flux2_from_safetensors(
        file_path=trainer.model_path,
        device="cpu",
        torch_dtype=trainer.weight_dtype
    )

    # Store components
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer  # FLUX.2 doesn't need wrapper
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # FLUX.2 specific: no text_encoder_2, no unet
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Save base model info for checkpoint metadata
    config = components.get("config", {})
    trainer.base_model_repo = config.get("base_model_repo", None)
    trainer.is_distilled = config.get("is_distilled", False)

    # Convert VAE to vae_dtype
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Enable gradient checkpointing for Transformer (CRITICAL for VRAM reduction)
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (FLUX.2)")
    elif hasattr(trainer.transformer, 'enable_gradient_checkpointing'):
        trainer.transformer.enable_gradient_checkpointing()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for FLUX.2 Transformer")
    else:
        print(f"{trainer.log_prefix} WARNING: Gradient checkpointing not available for FLUX.2 Transformer")

    # Enable gradient checkpointing for Text Encoder (Qwen3)
    if trainer.gradient_checkpointing and hasattr(trainer.text_encoder, 'gradient_checkpointing_enable'):
        trainer.text_encoder.gradient_checkpointing_enable()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Qwen3 Text Encoder")

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_flux2(trainer.attention_backend)

    # Freeze all base weights (full parameter training will unfreeze specific layers later)
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Setup Block Swap if enabled (before moving to GPU)
    trainer.flux2_block_offloader = None  # FLUX.2 specific offloader
    trainer.flux2_transformer_wrapper = None  # Drives the offloader during forward

    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap enabled for FLUX.2 training: {trainer.blocks_to_swap} blocks")
        print(f"{trainer.log_prefix} Using FluxBlockOffloader (dual-list architecture)")
        print(f"{trainer.log_prefix} Pinned memory: {trainer.use_pinned_memory}")

        # Policy gate: FLUX.2 training block swap requires H2D-only + frozen base
        # (LoRA) + gradient checkpointing. Raises on any unsupported combination.
        _h2d_args = trainer._flux2_block_swap_h2d_args()

        # Import FLUX.2 specific block offloader
        from core.memory_management import create_flux_block_offloader

        # Check if transformer has required attributes
        if not hasattr(trainer.transformer, 'transformer_blocks') or not hasattr(trainer.transformer, 'single_transformer_blocks'):
            raise ValueError(
                f"FLUX.2 Transformer must have 'transformer_blocks' and 'single_transformer_blocks' attributes for Block Swap. "
                f"Found: {type(trainer.transformer)}"
            )

        # Initialize FLUX.2 Block Offloader
        trainer.flux2_block_offloader = create_flux_block_offloader(
            transformer=trainer.transformer,
            blocks_to_swap=trainer.blocks_to_swap,
            device=trainer.device,
            target_dtype=trainer.training_dtype,
            use_pinned_memory=trainer.use_pinned_memory,
            supports_backward=True,  # Training mode
            **_h2d_args,
        )

        # Prepare block devices (keep some on GPU, offload rest to CPU)
        trainer.flux2_block_offloader.prepare_block_devices_before_forward()

        # Wire the offloader into the forward (wrapper) and backward (hooks).
        # Without this the offloader is never driven -> device mismatch.
        trainer._wire_flux2_block_swap_driver()

        num_dual = len(trainer.transformer.transformer_blocks)
        num_single = len(trainer.transformer.single_transformer_blocks)
        print(f"{trainer.log_prefix} FLUX.2 Block Swap initialized:")
        print(f"{trainer.log_prefix}   Dual stream blocks: {num_dual}")
        print(f"{trainer.log_prefix}   Single stream blocks: {num_single}")
        print(f"{trainer.log_prefix}   Total blocks: {num_dual + num_single}")
        print(f"{trainer.log_prefix}   Blocks to swap: {trainer.blocks_to_swap}")

        # Move VAE and Text Encoder to device (Transformer managed by block offloader)
        print(f"{trainer.log_prefix} Moving VAE to {trainer.device}...")
        trainer.vae.to(trainer.device)
        print(f"{trainer.log_prefix} Moving Text Encoder to {trainer.device}...")
        trainer.text_encoder.to(trainer.device)
    else:
        # No Block Swap: move everything to GPU
        print(f"{trainer.log_prefix} Moving VAE to {trainer.device}...")
        trainer.vae.to(trainer.device)

        print(f"{trainer.log_prefix} Moving Transformer to {trainer.device}...")
        trainer.transformer.to(trainer.device)

        print(f"{trainer.log_prefix} Moving Text Encoder to {trainer.device}...")
        trainer.text_encoder.to(trainer.device)

    print(f"{trainer.log_prefix} FLUX.2 model loaded successfully")
    print(f"{trainer.log_prefix} Transformer: {trainer.transformer.__class__.__name__}")
    print(f"{trainer.log_prefix} Text Encoder: {trainer.text_encoder.__class__.__name__}")
    print(f"{trainer.log_prefix} Scheduler type: {trainer.scheduler.__class__.__name__}")

    # Debug: Check for inf/nan in Transformer parameters
    transformer_has_inf = False
    transformer_has_nan = False
    for name, param in trainer.transformer.named_parameters():
        if torch.isinf(param).any():
            print(f"{trainer.log_prefix} WARNING: Transformer param '{name}' contains inf!")
            transformer_has_inf = True
        if torch.isnan(param).any():
            print(f"{trainer.log_prefix} WARNING: Transformer param '{name}' contains nan!")
            transformer_has_nan = True
    if not transformer_has_inf and not transformer_has_nan:
        print(f"{trainer.log_prefix} Transformer parameters: No inf/nan detected")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for FLUX.2 models.

    attention_impl='conduit' (default): install ConduitFlux2* processors on the
    training transformer's NON-KV attention modules so the unified conduit runs
    the kernel (enables tq training on FLUX.2). attention_impl='diffusers': the
    legacy set_attention_backend path (byte-identical). ``resolve_backend`` refuses
    sage for training (R4) in both branches.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return

    b = trainer._resolve_training_backend(backend)

    if getattr(trainer, "attention_impl", "conduit") == "conduit":
        try:
            from core.attention import AttentionMode
            from core.pipeline_backends.flux2 import _install_flux2_conduit_processors
            try:
                trainer.transformer.set_attention_backend("native")
            except Exception:
                pass
            migrated = _install_flux2_conduit_processors(trainer.transformer, b, AttentionMode.TRAINING)
            print(f"{trainer.log_prefix} [OK] FLUX.2 attention impl=conduit backend='{b}' "
                  f"({migrated} attn modules migrated)")
        except Exception as e:
            print(f"{trainer.log_prefix} WARNING: FLUX.2 conduit install failed ({e}); "
                  f"falling back to diffusers set_attention_backend")
            try:
                trainer.transformer.set_attention_backend(to_diffusers_backend(b))
            except Exception:
                pass
        return

    # attention_impl='diffusers' (legacy, byte-identical)
    try:
        print(f"{trainer.log_prefix} Setting FLUX.2 Transformer attention backend '{b}' (impl=diffusers)...")
        trainer.transformer.set_attention_backend(to_diffusers_backend(b))
        print(f"{trainer.log_prefix} [OK] Attention backend set via set_attention_backend('{to_diffusers_backend(b)}')")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """FLUX.2 VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_flux2`` branch (self->trainer rename only). Runs
    inside the caller's ``with torch.no_grad()``; caller does the shared final
    dtype/CPU move. ``trainer._flux2_patchify_latents_for_training`` is a shared
    trainer method (also used by train_step; kept central, called via trainer).
    """
    # FLUX.2 VAE encoding with BatchNorm normalization
    latent_dist = trainer.vae.encode(image_tensor).latent_dist
    latents = latent_dist.sample()

    # DEBUG: Log raw latents
    if debug_preprocessing:
        print(f"[encode_image DEBUG] FLUX.2 raw latents:")
        print(f"  Shape: {latents.shape}")
        print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")

    # Patchify: (B, 32, H, W) -> (B, 128, H/2, W/2)
    latents = trainer._flux2_patchify_latents_for_training(latents)

    # Apply BatchNorm normalization (like pipeline.py)
    latents_bn_mean = trainer.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(trainer.vae.bn.running_var.view(1, -1, 1, 1) + trainer.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = (latents - latents_bn_mean) / latents_bn_std

    # DEBUG: Log normalized latents
    if debug_preprocessing:
        print(f"[encode_image DEBUG] FLUX.2 normalized latents:")
        print(f"  Shape: {latents.shape}")
        print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")

    del latent_dist
    return latents


def train_step(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    img_ids: torch.Tensor,
    txt_ids: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
    reference_latents_nested: Optional[List[List[torch.Tensor]]] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Perform single training step (FLUX.2 Klein).

    VERBATIM body of ``BaseTrainer.train_step_flux2`` (P6c; ``self.`` ->
    ``trainer.`` receiver rename + sanctioned lazy base_trainer import only).
    Packing helpers (``_flux2_pack_latents`` / ``_flux2_prepare_latent_ids`` /
    ``_flux2_unpack_latents_with_ids``) stay as spine methods (shared with the
    ctx-build branch, encode paths and P7 sampling) and are called via
    ``trainer.`` — mirroring the P5 ``_flux2_patchify_latents_for_training``
    delegator policy. See the original docstring for arg/shape details.
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import (
        add_noise_unified,
        get_target_unified,
        predict_original_latent_unified,
        print_vram_usage,
    )

    if profile_vram:
        print_vram_usage("[train_step_flux2] Start")

    # FLUX.2 uses Flow Matching with velocity prediction
    noise_process = getattr(trainer, 'noise_process', 'flow')  # FLUX.2 default: flow
    prediction_target = getattr(trainer, 'prediction_target', 'velocity')  # FLUX.2 default: velocity

    # Move latents to GPU with correct dtype
    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    img_ids = img_ids.to(device=trainer.device, non_blocking=True)
    txt_ids = txt_ids.to(device=trainer.device, non_blocking=True)
    prompt_embeds = prompt_embeds.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)

    # Sample random timesteps from [0, 1] if not provided
    batch_size = latents.shape[0]
    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
        else:
            timesteps = torch.rand(batch_size, device=trainer.device)

    # Set default guidance if not provided
    if guidance is None:
        guidance = torch.full((batch_size,), 3.5, device=trainer.device, dtype=trainer.training_dtype)

    # Sample noise (standard normal distribution)
    noise = torch.randn_like(latents)

    # Add noise using flow matching: noisy = (1 - t) * latents + t * noise
    noisy_latents = add_noise_unified(
        noise_process=noise_process,
        noise_scheduler=trainer.noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )

    # ============================================================
    # Reference Image Conditioning (Latent Concatenation)
    # ============================================================
    # If reference latents are provided, pack them and concatenate with noisy latents
    # This allows the model to condition on reference images during training
    #
    # Multiple reference images per batch item:
    # - reference_latents_nested is List[List[Tensor]] where each inner list contains
    #   reference latents for one batch item
    # - Each reference image gets T coordinate offset: 10, 20, 30, ...
    # - All reference latents are packed and concatenated per batch item
    #
    # Shape: noisy_latents [B, seq_len, C] + ref_latents [B, ref_seq_len, C]
    #        -> concatenated [B, seq_len + ref_seq_len, C]
    # img_ids are also extended with reference position IDs
    packed_reference_latents = None
    if reference_latents_nested is not None and len(reference_latents_nested) > 0:
        # Process each batch item's reference images
        all_packed_refs = []
        all_ref_ids = []

        for batch_idx, item_ref_latents in enumerate(reference_latents_nested):
            item_packed_refs = []
            item_ref_ids = []

            for ref_idx, ref_latent in enumerate(item_ref_latents):
                # ref_latent shape: [1, C, H, W] (single reference image)
                # Pack: (1, C, H, W) -> (1, H*W, C)
                packed_ref = trainer._flux2_pack_latents(ref_latent)
                packed_ref = packed_ref.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
                item_packed_refs.append(packed_ref)

                # Prepare position IDs for this reference image
                ref_img_id = trainer._flux2_prepare_latent_ids(ref_latent).to(trainer.device)
                # Apply T coordinate offset: T = scale + scale * ref_idx (scale=10)
                # ref_idx 0 -> T=10, ref_idx 1 -> T=20, ref_idx 2 -> T=30, etc.
                t_offset = 10 + 10 * ref_idx
                ref_img_id[..., 0] = ref_img_id[..., 0] + t_offset
                item_ref_ids.append(ref_img_id)

            # Concatenate all reference latents for this batch item
            # Shape: (1, total_ref_seq_len, C)
            item_packed_concat = torch.cat(item_packed_refs, dim=1)
            item_ids_concat = torch.cat(item_ref_ids, dim=1)

            all_packed_refs.append(item_packed_concat)
            all_ref_ids.append(item_ids_concat)

        # Stack across batch dimension
        # All batch items must have same total reference sequence length
        # (This is guaranteed if all items have same number of reference images with same dimensions)
        # If dimensions vary, we need padding - for now, assume consistent structure
        try:
            packed_reference_latents = torch.cat(all_packed_refs, dim=0)  # [B, ref_seq_len, C]
            ref_img_ids = torch.cat(all_ref_ids, dim=0)  # [B, ref_seq_len, 4]

            # Concatenate reference latents with noisy latents along sequence dimension
            noisy_latents = torch.cat([noisy_latents, packed_reference_latents], dim=1)

            # Concatenate reference position IDs with image position IDs
            img_ids = torch.cat([img_ids, ref_img_ids], dim=1)
        except RuntimeError as e:
            # Handle dimension mismatch (different reference image counts/sizes per batch item)
            print(f"{trainer.log_prefix} WARNING: Reference latent dimension mismatch in batch, skipping reference conditioning: {e}")
            packed_reference_latents = None

    if profile_vram:
        print_vram_usage("[train_step_flux2] Before Transformer forward")

    # Predict velocity using FLUX.2 Transformer.
    # When block swap is active, route the forward through the block-swap wrapper so
    # it drives the offloader (wait_for_block / submit per block). self.transformer
    # itself is NOT replaced -> optimizer / LoRA / state_dict keep seeing the raw module.
    fwd = trainer.flux2_transformer_wrapper if getattr(trainer, "flux2_transformer_wrapper", None) is not None else trainer.transformer
    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            output = fwd(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )
            model_pred = output[0]
    else:
        output = fwd(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )
        model_pred = output[0]

    if profile_vram:
        print_vram_usage("[train_step_flux2] After Transformer forward")

    # ============================================================
    # Slice output to remove reference latent predictions
    # ============================================================
    # If we concatenated reference latents, the model output contains predictions
    # for both target + reference. We only want predictions for the target latents.
    original_seq_len = latents.shape[1]  # Original target latent sequence length
    if packed_reference_latents is not None:
        # Slice to keep only predictions for target latents
        model_pred = model_pred[:, :original_seq_len, :]
        # Also slice noisy_latents for consistency in loss computation
        noisy_latents = noisy_latents[:, :original_seq_len, :]

    # Get target using unified framework
    target = get_target_unified(
        noise_process=noise_process,
        prediction_target=prediction_target,
        noise_scheduler=trainer.noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )

    # Calculate MSE loss (always in fp32)
    loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss_per_sample = loss_per_element.mean([1, 2])  # Mean over seq_len and channels

    # Flow Matching doesn't use Min-SNR weighting (uniform timestep distribution)
    mse_loss = loss_per_sample.mean()

    # Add regularization if enabled
    regularization_loss = torch.tensor(0.0, device=trainer.device)

    # Compute predicted latent once (used by regularization losses and dual loss)
    predicted_latent_for_reg = None
    if trainer.snr_regularization_loss is not None or trainer.energy_regularization_loss is not None or trainer.reconstruction_loss_weight > 0:
        predicted_latent_for_reg = predict_original_latent_unified(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noise_scheduler=trainer.noise_scheduler,
            noisy_latents=noisy_latents,
            model_pred=model_pred,
            timesteps=timesteps,
        )

    # SNR regularization
    if trainer.snr_regularization_loss is not None:
        snr_reg_loss = trainer.snr_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps
        )
        regularization_loss = regularization_loss + snr_reg_loss

    # Energy regularization
    if trainer.energy_regularization_loss is not None:
        energy_reg_loss = trainer.energy_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps
        )
        regularization_loss = regularization_loss + energy_reg_loss

    # Calculate reconstruction loss
    if trainer.reconstruction_loss_weight > 0:
        if predicted_latent_for_reg is not None:
            predicted_latent_for_recon = predicted_latent_for_reg
        else:
            predicted_latent_for_recon = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=trainer.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )

        recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
        recon_loss_per_sample = recon_loss_per_element.mean([1, 2])
        recon_loss = recon_loss_per_sample.mean()

        alpha = 1.0 - trainer.reconstruction_loss_weight
        beta = trainer.reconstruction_loss_weight
        combined_loss = alpha * mse_loss + beta * recon_loss

        loss = combined_loss + regularization_loss
    else:
        with torch.no_grad():
            if predicted_latent_for_reg is not None:
                predicted_latent_for_recon = predicted_latent_for_reg.detach()
            else:
                predicted_latent_for_recon = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=trainer.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2])
            recon_loss = recon_loss_per_sample.mean()

        loss = mse_loss + regularization_loss

    if profile_vram:
        print_vram_usage("[train_step_flux2] After loss calculation")

    # Debug save if requested
    if debug_save_path is not None:
        debug_save_path.mkdir(parents=True, exist_ok=True)
        timestep_value = timesteps[0].item()

        with torch.no_grad():
            # FLUX.2 uses standard Flow Matching: x_0 = x_t - t * v
            t = timesteps.float()
            while t.dim() < noisy_latents.dim():
                t = t.unsqueeze(-1)
            predicted_latent = noisy_latents - t * model_pred

            # Convert packed latents (B, seq_len, C) to (B, C, H, W) for visualization
            # This makes debug output consistent with other models (SD/SDXL/Z-Image)
            latents_4d = trainer._flux2_unpack_latents_with_ids(latents[0:1], img_ids[0:1])
            noisy_latents_4d = trainer._flux2_unpack_latents_with_ids(noisy_latents[0:1], img_ids[0:1])
            predicted_velocity_4d = trainer._flux2_unpack_latents_with_ids(model_pred[0:1], img_ids[0:1])
            actual_velocity_4d = trainer._flux2_unpack_latents_with_ids(target[0:1], img_ids[0:1])
            predicted_latent_4d = trainer._flux2_unpack_latents_with_ids(predicted_latent[0:1], img_ids[0:1])

        debug_data = {
            'latents': latents_4d.detach().cpu(),
            'noisy_latents': noisy_latents_4d.detach().cpu(),
            'predicted_velocity': predicted_velocity_4d.detach().cpu(),
            'actual_velocity': actual_velocity_4d.detach().cpu(),
            'predicted_latent': predicted_latent_4d.detach().cpu(),
            'timestep': timestep_value,
            'loss': loss_per_sample[0].item(),
            'loss_batch_mean': loss.item(),
            'recon_loss': recon_loss_per_sample[0].item(),
            'recon_loss_batch_mean': recon_loss.item(),
            'batch_size': batch_size,
            'scheduler_type': 'FlowMatching',
            'model_type': 'flux2',
            'img_ids_shape': list(img_ids.shape),
            'txt_ids_shape': list(txt_ids.shape),
            'latent_shape_4d': list(latents_4d.shape),  # Store 4D shape for reference
        }

        if debug_captions is not None and len(debug_captions) > 0:
            debug_data['caption'] = debug_captions[0]
            debug_data['all_captions'] = debug_captions

        if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
            first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
            if first_ref:
                debug_data['reference_image_path'] = first_ref

        torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
        del predicted_latent, latents_4d, noisy_latents_4d, predicted_velocity_4d, actual_velocity_4d, predicted_latent_4d

    # Return loss tensor and loss values
    pred_loss_value = mse_loss.item()
    recon_loss_value = recon_loss.item()

    # Free intermediate tensors
    del noise, noisy_latents, model_pred, target
    del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

    return loss, pred_loss_value, recon_loss_value
