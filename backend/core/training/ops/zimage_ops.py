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
