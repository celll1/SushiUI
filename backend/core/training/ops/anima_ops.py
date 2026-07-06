"""anima_ops.py — Anima (Cosmos-Predict2 DiT) loader + block-swap + attention
free functions (plan P3b).

These are the VERBATIM bodies of ``BaseTrainer._load_anima_components``,
``BaseTrainer.setup_anima_block_swap`` and
``BaseTrainer._setup_attention_backend_anima`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3a/P3b): the arch handler is built at the END of
``BaseTrainer.__init__`` (base_trainer.py:1115), AFTER ``_load_model_components``
runs. So the load-time dispatcher CANNOT use ``trainer.arch`` here. The
base_trainer dispatcher calls ``load_components`` directly; ``setup_block_swap``
and ``setup_attention_backend`` keep 2-line delegators on the trainer because
they have late/multiple call sites (mode subclasses call ``setup_anima_block_swap``
via ``hasattr``; the loader body calls ``trainer._setup_attention_backend_anima``).
Each body is defined exactly once here and stays byte-identical.
"""
from __future__ import annotations


def load_components(trainer) -> None:
    """Load Anima model components for training.

    Anima ships as either a split-files HuggingFace layout or a single DiT
    safetensors plus separately-discovered Qwen3 / Qwen-Image VAE files.
    ModelLoader.load_anima_from_files handles both and returns a component
    dict identical to the one used by the inference path.
    """
    print(f"{trainer.log_prefix} Detected Anima model")
    print(f"{trainer.log_prefix} Loading Anima components from {trainer.model_path}")

    from core.model_loader import ModelLoader
    components = ModelLoader.load_anima_from_files(
        path=trainer.model_path,
        device="cpu",
        torch_dtype=trainer.weight_dtype,
    )

    # Store components on the trainer in the standard slots.
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer  # No wrapper for Anima.
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.t5_tokenizer = components["t5_tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Anima specific: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Cast VAE to the desired dtype.
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing mode for the DiT blocks. Three options:
    #   standard         (default) — activations stay on GPU
    #   cpu_offload      — blocking CPU offload (saves VRAM, slower)
    #   async_cpu_offload — non-blocking CPU offload (saves VRAM, fast)
    # When both flags are True, async wins and we warn.
    cpu_offload_ckpt = bool(trainer.config.get("cpu_offload_checkpointing", False))
    async_offload_ckpt = bool(trainer.config.get("async_cpu_offload_checkpointing", False))
    if cpu_offload_ckpt and async_offload_ckpt:
        print(f"{trainer.log_prefix} WARNING: both cpu_offload_checkpointing and "
              f"async_cpu_offload_checkpointing are True; using async (faster).")
        cpu_offload_ckpt = False
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Anima)")
    elif hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        trainer.transformer.enable_gradient_checkpointing(
            cpu_offload=cpu_offload_ckpt,
            async_offload=async_offload_ckpt,
        )
        ckpt_mode = ("async_cpu_offload" if async_offload_ckpt
                      else "cpu_offload" if cpu_offload_ckpt else "standard")
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Anima DiT "
              f"(mode={ckpt_mode})")
    if trainer.gradient_checkpointing and hasattr(trainer.text_encoder, "gradient_checkpointing_enable"):
        trainer.text_encoder.gradient_checkpointing_enable()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Qwen3 text encoder")

    # Freeze all base weights. Trainable LoRA modules are added later by the
    # adapter via apply_lora_to_unet.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Optional: FP8 the base DiT before LoRA wraps anything (LoRA-only).
    # Only safe when the base is frozen — which is true for the LoRA path
    # (Phase C.1 freezes everything before adapter injection). Full FT
    # needs trainable base weights, so silently ignore the flag with a
    # warning. We piggy-back on the Phase B.1-d inference quantiser which
    # patches each Linear's forward to dequantise on-the-fly.
    fp8_base_dtype = trainer.config.get("fp8_base_dtype") or None
    training_method = trainer.config.get("training_method", "lora")
    if fp8_base_dtype and training_method == "lora":
        print(f"{trainer.log_prefix} Quantising frozen Anima DiT base to "
              f"{fp8_base_dtype} (LoRA-on-FP8-base, ~50% VRAM reduction)")
        from core.vram_optimization import _anima_quantize_fp8
        # deepcopy + patch — replaces trainer.transformer with the quantised
        # copy so subsequent block-swap and adapter wrap target the new
        # module references.
        trainer.transformer = _anima_quantize_fp8(
            trainer.transformer, fp8_base_dtype, "DiT (training base)",
        )
        # transformer_original keeps pointing at the quantised model too,
        # so downstream move_main_model_to_* keeps working.
        trainer.transformer_original = trainer.transformer
        trainer.transformer.requires_grad_(False)
    elif fp8_base_dtype:
        print(f"{trainer.log_prefix} WARNING: fp8_base_dtype={fp8_base_dtype} is "
              f"only supported for training_method='lora' "
              f"(current: {training_method!r}); ignoring.")

    # Plain GPU move. Block-swap init is deferred to setup_anima_block_swap(),
    # which is called by the trainer subclass AFTER any structural changes
    # (LoRA wrap / full-FT requires_grad toggling). The reason: the
    # LayerOffloadConductor snapshots each layer's state_dict at hook-
    # registration time, and a later LoRA wrap inserts new submodule keys
    # (.original_module.weight) that aren't in the snapshot, breaking the
    # CPU<->GPU swap with a KeyError. Setting up after the adapter avoids
    # that. The conductor handle is initialised to None here so callers
    # can rely on attribute presence.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")
    print(f"{trainer.log_prefix} Moving Anima DiT to {trainer.device} "
          f"(block swap, if any, will redistribute after adapter setup)")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (opt-in; SDPA fallback otherwise)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_anima(trainer.attention_backend)

    print(f"{trainer.log_prefix} Anima model loaded successfully")
    print(f"{trainer.log_prefix} Scheduler: {trainer.scheduler.__class__.__name__}, "
          f"latent_channels=16")


def setup_block_swap(trainer) -> None:
    """Initialise the LayerOffloadConductor for the Anima DiT, AFTER any
    structural model changes (LoRA wrapping / full-FT param toggling).

    Idempotent: no-op when the trainer isn't on Anima, blocks_to_swap is
    0, or a conductor is already attached. The conductor snapshots each
    layer's state_dict at register_hooks() time, which is why this has to
    run after LoRALinearLayer wrappers (if any) have been inserted.
    """
    if not trainer.is_anima:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "blocks"):
        raise ValueError("Anima DiT must expose `.blocks` (nn.ModuleList) for block swap")

    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    from core.memory_management import LayerOffloadConductor
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=trainer.transformer.blocks,
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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Anima")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Anima (Cosmos DiT) models.

    Anima's vendored attention dispatches on a per-block ``attn_mode`` whose
    vocabulary is ``'torch'|'flash'`` (no 'native'/'sage'). Map native->'torch',
    flash->'flash' (R9); sage is refused upstream by ``resolve_backend`` and
    arrives here as native. Masked attention (and any failure) still falls
    back to SDPA inside the vendored kernel.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    attn_mode = 'flash' if b == 'flash' else 'torch'
    try:
        n = 0
        for m in trainer.transformer.modules():
            if hasattr(m, "attn_mode"):
                m.attn_mode = attn_mode
                n += 1
        print(f"{trainer.log_prefix} [OK] Anima attention backend '{b}' (attn_mode='{attn_mode}') "
              f"set on {n} block(s)")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Anima attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")
