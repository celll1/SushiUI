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

import torch

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
