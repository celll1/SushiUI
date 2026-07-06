"""krea2_ops.py — Krea 2 (single-stream MMDiT) loader + block-swap + attention
free functions (P3c).

VERBATIM bodies of ``BaseTrainer._load_krea2_components``,
``BaseTrainer.setup_krea2_block_swap`` and
``BaseTrainer._setup_attention_backend_krea2`` (base_trainer.py), moved out of the
spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b/P3c): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher (and ``_load_checkpoint_as_base``) call ``load_components``
directly. ``setup_block_swap`` keeps a 2-line delegator on the trainer (called
LATE by mode subclasses via hasattr) and ``setup_attention_backend`` keeps a
delegator (called from the moved loader body); each body is defined exactly once
here.
"""
from __future__ import annotations


def load_components(trainer) -> None:
    """Load Krea 2 components (single-stream MMDiT + Qwen3-VL TE + Qwen-Image VAE).

    The transformer is trained (LoRA wraps it, or full-FT updates it); the
    Qwen3-VL text encoder and the VAE are frozen. bf16 base (train_runner forces
    bf16). Block-swap over ``transformer_blocks`` is deferred until adapter setup.
    """
    print(f"{trainer.log_prefix} Detected Krea 2 model")
    print(f"{trainer.log_prefix} Loading Krea 2 components from {trainer.model_path}")

    from core.models.krea2.krea2_loader import load_krea2_components
    components = load_krea2_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
        load_text_encoder=True,
    )

    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Krea 2 metadata used by train_step / sample generation.
    trainer.krea2_is_distilled = bool(components.get("is_distilled", False))
    trainer.krea2_select_layers = list(
        components.get("text_encoder_select_layers")
        or [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    )
    trainer.krea2_patch_size = int(components.get("patch_size", 2))
    # Discrete flow-matching timestep shift (musubi default 2.5); config override.
    trainer.krea2_discrete_flow_shift = float(trainer.config.get("krea2_discrete_flow_shift", 2.5))

    # Single-stream DiT: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing on the vendored transformer.
    if trainer.gradient_checkpointing and hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        try:
            trainer.transformer.enable_gradient_checkpointing()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for Krea 2 transformer")
        except Exception as e:
            print(f"{trainer.log_prefix} grad checkpoint enable failed: {e}")
    elif not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Krea 2)")

    # Freeze VAE + TE; the transformer requires_grad is set by the adapter.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Block-swap deferred until after adapter setup.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving Krea 2 transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)

    if trainer.use_flash_attention:
        trainer._setup_attention_backend_krea2(trainer.attention_backend)

    print(f"{trainer.log_prefix} Krea 2 model loaded successfully (is_distilled={trainer.krea2_is_distilled})")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor over the Krea 2 ``transformer_blocks``, AFTER adapter setup."""
    if not getattr(trainer, "is_krea2", False):
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "transformer_blocks"):
        raise ValueError("Krea 2 transformer must expose `.transformer_blocks` for block swap")

    from core.memory_management import LayerOffloadConductor
    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Krea 2")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Krea 2 (training hook).

    The vendored transformer's ``forward`` calls ``_stamp_attention_backend()``
    which fans ``self._attn_backend`` out to every ``Krea2Attention`` module and
    derives the mode from the autograd state (training -> conduit refuses sage).
    This hook stamps the canonical backend string, honoring the training guard
    (``_resolve_training_backend`` maps sage -> native)."""
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        trainer.transformer._attn_backend = b
        print(f"{trainer.log_prefix} [OK] Krea 2 attention backend '{b}' stamped on transformer")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Krea 2 attention backend '{b}': {e}")


def encode_prompt(trainer, prompt: str, max_length: int = 512):
    """Encode prompt for Krea 2: 12-layer Qwen3-VL hidden-state stack.

    VERBATIM body of ``BaseTrainer.encode_prompt_krea2`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    from core.models.krea2.krea2_pipeline_ops import encode_prompt as _k_encode
    select_layers = getattr(trainer, "krea2_select_layers", None) or [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    te_device = trainer.text_encoder.device if hasattr(trainer.text_encoder, "device") else trainer.device
    embeds, mask = _k_encode(
        trainer.text_encoder, trainer.tokenizer, prompt, select_layers, max_length, te_device,
    )  # embeds [1, seq, 12, 2560], mask [1, seq]
    return embeds.detach().to("cpu"), mask[0].detach().to("cpu")
