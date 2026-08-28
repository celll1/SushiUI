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

NOTE: ``block_swap_h2d_args`` Gate 3 still calls ``enable_gradient_checkpointing()``
unconditionally -- H2D block swap hard-requires gradient checkpointing by design
and this is intentionally NOT gated on ``trainer.gradient_checkpointing``. When
that flag is False, the override is now made visible via a WARNING log instead
of being silent.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from core.attention import AttentionMode, to_diffusers_backend

from .training_method import trains_denoiser_weights


def block_swap_h2d_args(trainer):
    """Policy gate + H2D args for FLUX.2 training block swap.

    FLUX.2 training block swap is supported ONLY via the H2D-only +
    frozen-transformer + gradient-checkpointing path. The standard (non-H2D) training swap has
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

    # Gate 2: requires a frozen transformer. requires_grad cannot be inspected
    # here (the adapter unfreezes later), so the mode decides. A text-encoder-only
    # full FT keeps the transformer frozen and is allowed. The offloader's lazy
    # Full-FT detect is no substitute: it silently falls back to the standard swap
    # path, which Gate 1 above refuses as non-functional.
    if trains_denoiser_weights(trainer):
        raise ValueError(
            "FLUX.2 training block swap (H2D-only) requires a frozen transformer. "
            "This run trains the transformer weights, which needs D2H persistence "
            "and cannot use H2D-only block swap. Use training_method='lora', train "
            "only the text encoder (train_unet=False), or disable Block Swap "
            "(blocks_to_swap=0)."
        )

    # Gate 3: requires gradient checkpointing on the transformer (H2D backward
    # re-reads base weights via recompute). This is a HARD requirement of the
    # H2D block-swap path, independent of the per-run gradient_checkpointing
    # config flag (default True) -- H2D swap without grad-ckpt is unsupported
    # (OOM), so it is force-enabled here even if the flag is False. When the
    # flag is False, warn loudly so the override is visible instead of silent.
    if not getattr(trainer, "gradient_checkpointing", True):
        print(f"{trainer.log_prefix} WARNING: gradient_checkpointing=False in config, "
              f"but FLUX.2 H2D block swap (block_swap_h2d_only=True) requires gradient "
              f"checkpointing on the transformer. Force-enabling it for this run.")
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

    # A training process is DEQUANT-ONLY (see ideogram4_ops.load_components for
    # the full reasoning). FLUX.2 is in RUNTIME_INT8_ARCHS and its loader now
    # swaps Int8Linear / Fp8Linear in for a weight-only quantized checkpoint, so a
    # LoRA run over a quantized FLUX.2 base is reachable and must be fitted
    # against exactly the base function everyone else runs -- not against the
    # W8A8 fast paths, which are enabled by process-wide env flags that
    # training_process.py copies from the backend (os.environ.copy()) and which
    # grad mode cannot be used as a proxy for. Two module types, two separate
    # per-instance opt-outs: disabling one does not disable the other. train_runner
    # also disables both process-wide; that is belt-and-braces on top of the
    # per-module calls every trainer-side loader makes, which is what this is.
    # No-op on a bf16 base.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    from core.models.ideogram4.vendor.int8_linear import disable_int8_mm
    for _label, _module in (("transformer", trainer.transformer),
                            ("text_encoder", trainer.text_encoder)):
        if _module is not None:
            disable_scaled_mm(_module, label=f"flux2 training {_label}")
            disable_int8_mm(_module, label=f"flux2 training {_label}")

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
        try:
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
        except Exception as _dbg_e:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {_dbg_e}")

    # Return loss tensor and loss values
    pred_loss_value = mse_loss.item()
    recon_loss_value = recon_loss.item()

    # Free intermediate tensors
    del noise, noisy_latents, model_pred, target
    del loss_per_element, loss_per_sample, recon_loss_per_element, recon_loss_per_sample, recon_loss

    return loss, pred_loss_value, recon_loss_value


# ============================================================
# FLUX.2 Sample Generation (plan P7)
# ============================================================
# Verbatim bodies of BaseTrainer._generate_sample_flux2 and its sample-only
# helpers (_flux2_encode_prompt_for_sample / _flux2_prepare_latent_ids_for_sample
# / _flux2_pack_latents_for_sample / _flux2_compute_empirical_mu_for_sample /
# _decode_flux2_latents), moved out of the spine with the mechanical
# self.->trainer. receiver rename, a sanctioned lazy base_trainer import for
# log_verbose, and the relocated .optimizers -> ..optimizers relative import.
# The shared latent-geometry helpers _flux2_unpack_latents_with_ids and
# _flux2_unpatchify_latents stay on the trainer (also used by train_step /
# vae_encode) and are called via ``trainer.`` here. arch/flux2.py::sample()
# unpacks SampleContext into generate_sample.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    seed: int = -1,
    reference_image_path: Optional[str] = None,
    negative_prompt: str = "",
):
    """
    Generate sample image during training (FLUX.2 Klein).

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

    import random
    import numpy as np

    print(f"{trainer.log_prefix} Generating FLUX.2 sample: {prompt[:50]}...")

    # Set models to eval mode for inference
    trainer.transformer.eval()
    trainer.vae.eval()
    trainer.text_encoder.eval()

    # Store original devices for restoration
    text_encoder_device = next(trainer.text_encoder.parameters()).device
    vae_device = next(trainer.vae.parameters()).device
    transformer_device = next(trainer.transformer.parameters()).device

    try:
        # ============================================================
        # Stage 0: Offload Transformer AND Optimizer State to CPU
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Offloading Transformer and Optimizer state to CPU")

        # Move Transformer to CPU
        trainer.transformer.to("cpu")

        # CRITICAL: Move Optimizer state (gradients, momentum) to CPU
        optimizer_state_dict = trainer.optimizer.state_dict()
        for param_id, state in optimizer_state_dict['state'].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                    state[key] = value.cpu()
        trainer.optimizer.load_state_dict(optimizer_state_dict)

        torch.cuda.empty_cache()
        log_verbose(f"{trainer.log_prefix} [Sample] Transformer and Optimizer state offloaded to CPU")

        # ============================================================
        # Stage 1: Text Encoding (Qwen3)
        # ============================================================
        if text_encoder_device != trainer.device:
            log_verbose(f"{trainer.log_prefix} [Sample] Moving Text Encoder to GPU for encoding")
            trainer.text_encoder.to(trainer.device)

        # Encode prompt using FLUX.2's Qwen3 text encoder
        prompt_embeds, text_ids = _flux2_encode_prompt_for_sample(trainer, prompt)

        # Encode unconditional prompt only if CFG is enabled
        if guidance_scale > 1.0:
            negative_prompt_embeds, negative_text_ids = _flux2_encode_prompt_for_sample(trainer, negative_prompt)
        else:
            negative_prompt_embeds, negative_text_ids = None, None

        # Move Text Encoder back to CPU to free VRAM
        if text_encoder_device != trainer.device:
            log_verbose(f"{trainer.log_prefix} [Sample] Moving Text Encoder back to CPU")
            trainer.text_encoder.to(text_encoder_device)
        torch.cuda.empty_cache()

        # ============================================================
        # Stage 1.5: Move Transformer back to GPU for denoising
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Moving Transformer to GPU for denoising")
        trainer.transformer.to(transformer_device)
        torch.cuda.empty_cache()

        # ============================================================
        # Stage 1.6: Reference Image VAE encoding (FLUX.2 latent concat)
        # ============================================================
        packed_reference_latents = None
        ref_img_ids = None
        if reference_image_path:
            try:
                from PIL import Image as PILImage
                ref_img = PILImage.open(reference_image_path).convert("RGB")
                ref_img = ref_img.resize((width, height), PILImage.LANCZOS)
                print(f"{trainer.log_prefix} [Sample] Moving VAE to GPU for reference image encoding")
                trainer.vae.to(trainer.device)
                with torch.no_grad():
                    ref_tensor = torch.from_numpy(
                        np.array(ref_img).astype(np.float32) / 127.5 - 1.0
                    ).permute(2, 0, 1).unsqueeze(0).to(trainer.device, dtype=trainer.vae.dtype)
                    ref_latent = trainer.vae.encode(ref_tensor).latent_dist.sample()
                    ref_latent = ref_latent * trainer.vae.config.scaling_factor
                trainer.vae.to("cpu")
                torch.cuda.empty_cache()
                packed_reference_latents = _flux2_pack_latents_for_sample(ref_latent)
                packed_reference_latents = packed_reference_latents.to(
                    device=trainer.device, dtype=prompt_embeds.dtype)
                ref_ids = _flux2_prepare_latent_ids_for_sample(ref_latent).to(trainer.device)
                ref_ids[..., 0] = ref_ids[..., 0] + 10  # T coordinate offset
                ref_img_ids = ref_ids
                print(f"{trainer.log_prefix} [Sample] Reference image encoded: {packed_reference_latents.shape}")
            except Exception as ref_err:
                print(f"{trainer.log_prefix} [Sample] WARNING: Reference image encoding failed: {ref_err}, skipping")
                packed_reference_latents = None
                ref_img_ids = None

        # ============================================================
        # Stage 2: Prepare Latents
        # ============================================================
        vae_scale_factor = 8
        patch_size = 2

        # Ensure height/width divisible by vae_scale_factor * patch_size
        latent_height = 2 * (int(height) // (vae_scale_factor * patch_size))
        latent_width = 2 * (int(width) // (vae_scale_factor * patch_size))

        # FLUX.2 has 32 latent channels, but patchified to 128
        num_channels_latents = trainer.transformer.config.in_channels // 4  # 32

        # Create random latents with seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=trainer.device).manual_seed(seed)

        latent_shape = (1, num_channels_latents * 4, latent_height // 2, latent_width // 2)
        latents = torch.randn(latent_shape, generator=generator, device=trainer.device, dtype=prompt_embeds.dtype)

        # Prepare latent position IDs
        latent_ids = _flux2_prepare_latent_ids_for_sample(latents).to(trainer.device)

        # Pack latents: (B, C, H, W) -> (B, H*W, C)
        latents = _flux2_pack_latents_for_sample(latents)

        # Concatenate reference latents with noise latents (if provided)
        if packed_reference_latents is not None and ref_img_ids is not None:
            latents = torch.cat([latents, packed_reference_latents], dim=1)
            latent_ids = torch.cat([latent_ids, ref_img_ids], dim=1)
            print(f"{trainer.log_prefix} [Sample] Latents after reference concat: {latents.shape}")

        # ============================================================
        # Stage 3: Denoising Loop
        # ============================================================
        log_verbose(f"{trainer.log_prefix} [Sample] Running denoising loop")

        # Prepare timesteps
        image_seq_len = latents.shape[1]
        mu = _flux2_compute_empirical_mu_for_sample(image_seq_len, num_inference_steps)

        # Set timesteps with sigmas
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        trainer.scheduler.set_timesteps(num_inference_steps, device=trainer.device, mu=mu)
        timesteps = trainer.scheduler.timesteps
        trainer.scheduler.set_begin_index(0)

        # Check if distilled model (no CFG)
        is_distilled = getattr(trainer.transformer.config, "is_distilled", False)
        do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled

        # Autocast the denoise loop to the sampling compute dtype (transformer dtype,
        # bf16/fp16). This is unconditional (NOT gated on trainer.mixed_precision):
        # sampling always runs the DiT in its param dtype, while LoRA adapters default
        # to lora_dtype=fp32 on a bf16 base. Without autocast the bf16 activations hit
        # the fp32 LoRA Linear weights and crash with a dtype mismatch inside the
        # forward — regardless of the mixed_precision flag. Mirrors the anima/lens fix
        # (a3db4a1); VAE decode stays outside in _decode_flux2_latents below. Sampling
        # calls trainer.transformer directly (NOT the Flux2BlockSwapWrapper), and flux2
        # training has no fp8-quantized base path, so autocast has no wrapper/dequant
        # interactions here — trainer.transformer.dtype is always bf16/fp16.
        sample_compute_dtype = trainer.transformer.dtype
        with torch.no_grad(), torch.autocast(device_type=trainer.device.type, dtype=sample_compute_dtype):
            for i, t in enumerate(tqdm(timesteps, desc="Generating")):
                # Expand timestep
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(trainer.transformer.dtype)

                # Batch CFG: Concatenate unconditional and conditional for single forward pass
                if do_classifier_free_guidance:
                    # Double the batch: [uncond, cond]
                    latent_model_input_doubled = torch.cat([latent_model_input, latent_model_input], dim=0)
                    timestep_doubled = torch.cat([timestep, timestep], dim=0)
                    prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    text_ids_combined = torch.cat([negative_text_ids, text_ids], dim=0)
                    latent_ids_doubled = torch.cat([latent_ids, latent_ids], dim=0)

                    # Single forward pass for both unconditional and conditional
                    noise_pred_combined = trainer.transformer(
                        hidden_states=latent_model_input_doubled,
                        timestep=timestep_doubled / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds_combined,
                        txt_ids=text_ids_combined,
                        img_ids=latent_ids_doubled,
                        return_dict=False,
                    )[0]

                    # Split and apply CFG formula
                    noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Distilled model: Use guidance vector (not CFG)
                    guidance_vec = torch.full(
                        (latent_model_input.shape[0],),
                        guidance_scale,
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype
                    )
                    noise_pred = trainer.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance_vec,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]

                # Scheduler step
                latents_dtype = latents.dtype
                latents = trainer.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

        # Free prompt embeddings
        del prompt_embeds, text_ids
        if negative_prompt_embeds is not None:
            del negative_prompt_embeds, negative_text_ids

        # ============================================================
        # Stage 4: Offload Transformer to CPU, move VAE to GPU
        # ============================================================
        print(f"{trainer.log_prefix} [Sample] Moving Transformer to CPU to free VRAM")
        trainer.transformer.to("cpu")
        torch.cuda.empty_cache()

        # Move VAE to GPU for decoding
        if vae_device != trainer.device:
            print(f"{trainer.log_prefix} [Sample] Moving VAE to GPU for decoding")
            trainer.vae.to(device=trainer.device, dtype=trainer.vae_dtype)

        # Decode latents
        image = _decode_flux2_latents(trainer, latents, latent_ids, latent_height, latent_width)

        # Move VAE back to CPU
        if vae_device != trainer.device:
            print(f"{trainer.log_prefix} [Sample] Moving VAE back to CPU")
            trainer.vae.to(device=vae_device, dtype=trainer.vae_dtype)

        # Free latents
        del latents, latent_ids
        torch.cuda.empty_cache()

        # ============================================================
        # Stage 5: Restore Transformer and Optimizer State to GPU
        # ============================================================
        print(f"{trainer.log_prefix} [Sample] Restoring Transformer and Optimizer state to GPU")

        # Move Transformer back to GPU
        trainer.transformer.to(transformer_device)

        # CRITICAL: Move Optimizer state back to GPU
        from ..optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
        from ..optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer
        if not isinstance(trainer.optimizer, (AdamW8bit_RingBuffer, Lion8bit_RingBuffer)):
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
        # Restore models to train mode
        trainer.transformer.train()


def _flux2_encode_prompt_for_sample(trainer, prompt: str):
    """Encode prompt using Qwen3 text encoder for FLUX.2 sample generation."""
    max_sequence_length = 512
    hidden_states_layers = (9, 18, 27)

    device = trainer.text_encoder.device
    dtype = trainer.text_encoder.dtype

    # Apply chat template
    # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
    messages = [{"role": "user", "content": prompt}]
    text = trainer.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize
    text_inputs = trainer.tokenizer(
        text,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    # Forward pass with hidden states
    # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
    with torch.no_grad():
        outputs = trainer.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract and stack hidden states from specified layers
    # IMPORTANT: Must match pipeline.py _flux2_encode_prompt() exactly
    # Use stack + permute + reshape (NOT simple cat) for correct tensor structure
    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=dtype, device=device)

    # Reshape: (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    # Generate text IDs for RoPE
    batch_size, seq_len = prompt_embeds.shape[:2]
    text_ids = torch.zeros(batch_size, seq_len, 4, device=device, dtype=torch.long)
    text_ids[..., 0] = 0  # T dimension
    text_ids[..., 3] = torch.arange(seq_len, device=device)  # L dimension

    return prompt_embeds, text_ids


def _flux2_prepare_latent_ids_for_sample(latents: torch.Tensor) -> torch.Tensor:
    """Prepare latent position IDs for FLUX.2 sample generation."""
    batch_size, channels, height, width = latents.shape

    # Create position IDs for each latent position
    latent_ids = torch.zeros(batch_size, height * width, 4, device=latents.device)

    # T=0, H, W, L coordinates
    h_coords = torch.arange(height, device=latents.device).repeat_interleave(width)
    w_coords = torch.arange(width, device=latents.device).repeat(height)
    l_coords = torch.arange(height * width, device=latents.device)

    latent_ids[:, :, 0] = 1  # T dimension (different from text)
    latent_ids[:, :, 1] = h_coords
    latent_ids[:, :, 2] = w_coords
    latent_ids[:, :, 3] = l_coords

    return latent_ids


def _flux2_pack_latents_for_sample(latents: torch.Tensor) -> torch.Tensor:
    """Pack latents from (B, C, H, W) to (B, H*W, C) for FLUX.2."""
    batch_size, channels, height, width = latents.shape
    latents = latents.permute(0, 2, 3, 1)  # (B, H, W, C)
    latents = latents.reshape(batch_size, height * width, channels)  # (B, H*W, C)
    return latents


def _flux2_compute_empirical_mu_for_sample(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for FLUX.2 timestep scheduling."""
    # From diffusers FLUX implementation
    return 0.5 * (math.log(1 + image_seq_len) - math.log(num_steps))


def _decode_flux2_latents(
    trainer,
    latents: torch.Tensor,
    latent_ids: torch.Tensor,
    latent_height: int,
    latent_width: int
) -> Image.Image:
    """Decode FLUX.2 latents to PIL image."""
    import numpy as np

    # Step 1: Unpack latents using position IDs: (B, H*W, C) -> (B, C, H, W)
    latents = trainer._flux2_unpack_latents_with_ids(latents, latent_ids)

    # Step 2: Apply BatchNorm scaling (FLUX.2-specific)
    latents_bn_mean = trainer.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = torch.sqrt(trainer.vae.bn.running_var.view(1, -1, 1, 1) + trainer.vae.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    latents = latents * latents_bn_std + latents_bn_mean

    # Step 3: Unpatchify: (B, 128, H/2, W/2) -> (B, 32, H, W)
    latents = trainer._flux2_unpatchify_latents(latents)

    # Convert latents to VAE dtype (bfloat16 -> float32)
    latents = latents.to(dtype=trainer.vae.dtype)

    # Decode
    with torch.no_grad():
        image = trainer.vae.decode(latents, return_dict=False)[0]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype(np.uint8)

    return Image.fromarray(image)
