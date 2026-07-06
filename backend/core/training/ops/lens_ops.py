"""lens_ops.py — Lens (DiT) loader + block-swap + attention free functions (P3b).

VERBATIM bodies of ``BaseTrainer._load_lens_components``,
``BaseTrainer.setup_lens_block_swap`` and
``BaseTrainer._setup_attention_backend_lens`` (base_trainer.py), moved out of the
spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher calls ``load_components`` directly. ``setup_block_swap`` and
``setup_attention_backend`` keep 2-line delegators on the trainer (late/multiple
call sites); each body is defined exactly once here.
"""
from __future__ import annotations


def load_components(trainer) -> None:
    """Load Lens model components for training.

    Lens ships as a standard diffusers directory layout. ModelLoader.load_lens_components
    returns the same component dict used by the inference path.
    """
    print(f"{trainer.log_prefix} Detected Lens model")
    print(f"{trainer.log_prefix} Loading Lens components from {trainer.model_path}")

    from core.models.lens.lens_loader import load_lens_components
    components = load_lens_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
    )

    # Store components on the trainer using the standard slots.
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer   # No wrapper for Lens.
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Lens specific: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing.
    cpu_offload_ckpt  = bool(trainer.config.get("cpu_offload_checkpointing", False))
    async_offload_ckpt = bool(trainer.config.get("async_cpu_offload_checkpointing", False))
    if cpu_offload_ckpt and async_offload_ckpt:
        print(f"{trainer.log_prefix} WARNING: both cpu_offload_checkpointing and "
              f"async_cpu_offload_checkpointing are True; using async (faster).")
        cpu_offload_ckpt = False
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Lens)")
    elif hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        trainer.transformer.enable_gradient_checkpointing(
            cpu_offload=cpu_offload_ckpt,
            async_offload=async_offload_ckpt,
        )
        ckpt_mode = ("async_cpu_offload" if async_offload_ckpt
                      else "cpu_offload" if cpu_offload_ckpt else "standard")
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Lens transformer "
              f"(mode={ckpt_mode})")

    # Freeze everything; LoRA/full-FT will unfreeze what is needed.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Optional FP8 base quantisation (LoRA-only; same helper as Anima).
    fp8_base_dtype  = trainer.config.get("fp8_base_dtype") or None
    training_method = trainer.config.get("training_method", "lora")
    if fp8_base_dtype and training_method == "lora":
        print(f"{trainer.log_prefix} Quantising frozen Lens transformer base to "
              f"{fp8_base_dtype} (LoRA-on-FP8-base)")
        from core.vram_optimization import _anima_quantize_fp8
        trainer.transformer = _anima_quantize_fp8(
            trainer.transformer, fp8_base_dtype, "Lens Transformer (training base)",
        )
        trainer.transformer_original = trainer.transformer
        trainer.transformer.requires_grad_(False)
    elif fp8_base_dtype:
        print(f"{trainer.log_prefix} WARNING: fp8_base_dtype={fp8_base_dtype} only "
              f"supported for training_method='lora' ({training_method!r}); ignoring.")

    # Block-swap deferred; conductor handle initialised to None.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving Lens transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (opt-in; SDPA fallback otherwise)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_lens(trainer.attention_backend)

    print(f"{trainer.log_prefix} Lens model loaded successfully")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor for the Lens transformer, AFTER adapter setup."""
    if not trainer.is_lens:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "transformer_blocks"):
        raise ValueError("Lens transformer must expose `.transformer_blocks` for block swap")

    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    from core.memory_management import LayerOffloadConductor
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=trainer.transformer.transformer_blocks,
        blocks_to_swap=trainer.blocks_to_swap,
        device=trainer.device,
        use_pinned_memory=trainer.use_pinned_memory,
        cpu_buffer_size_mb=8192,
        activation_buffer_size_mb=4096,
        enable_prefetch=True,
        enable_activation_offload=False,
    )
    trainer.transformer._layer_offload_conductor = trainer.layer_offload_conductor
    trainer.layer_offload_conductor.register_hooks()
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Lens")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Lens (DiT) models.

    Sets ``m._attention_backend`` on every ``LensJointAttention`` module (the
    new contract the Stage-B vendor read-site consumes, design 2.5). Also
    sets the legacy ``_use_flash_attn`` flag as a TRANSITIONAL bridge so the
    current vendor (which still reads ``_use_flash_attn`` until Phase 3e)
    keeps honoring flash during the staged rollout — the two stay consistent
    and the legacy flag is removed once the read-site migrates.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        n = 0
        for m in trainer.transformer.modules():
            if type(m).__name__ == "LensJointAttention":
                m._attention_backend = b
                # Transitional bridge for the pre-Phase-3e vendor read-site.
                m._use_flash_attn = (b == 'flash')
                n += 1
        print(f"{trainer.log_prefix} [OK] Lens attention backend '{b}' set on {n} module(s)")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Lens attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")
