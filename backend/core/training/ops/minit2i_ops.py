"""minit2i_ops.py — MiniT2I (pixel-space MM-JiT) loader + block-swap + attention
free functions (P3c).

VERBATIM bodies of ``BaseTrainer._load_minit2i_components``,
``BaseTrainer.setup_minit2i_block_swap`` and
``BaseTrainer._setup_attention_backend_minit2i`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b/P3c): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher (and ``_load_checkpoint_as_base``) call ``load_components``
directly. ``setup_block_swap`` keeps a 2-line delegator on the trainer (called
LATE by mode subclasses via hasattr) and ``setup_attention_backend`` keeps a
delegator (called from the moved loader body); each body is defined exactly once
here.

REPA setup stays central (``BaseTrainer._setup_repa``, plan section B "optional
cross-arch"); the moved loader body calls ``trainer._setup_repa()``.
"""
from __future__ import annotations

import torch


def load_components(trainer) -> None:
    """Load MiniT2I (pixel-space MM-JiT) components for training.

    Pixel-space: there is no VAE. The frozen FLAN-T5-Large is the text encoder.
    The MM-JiT transformer is loaded frozen here; LoRA / full-parameter adapters
    unfreeze the relevant parameters during setup.
    """
    print(f"{trainer.log_prefix} Detected MiniT2I model")
    print(f"{trainer.log_prefix} Loading MiniT2I components from {trainer.model_path}")

    from core.models.minit2i.minit2i_loader import load_minit2i_components
    flan_t5_path = trainer.config.get("minit2i_flan_t5_path") or None
    components = load_minit2i_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
        flan_t5_path=flan_t5_path,
        text_encoder_dtype=trainer.text_encoder_dtype if hasattr(trainer, "text_encoder_dtype") else torch.float32,
        vae_dtype=trainer.vae_dtype if hasattr(trainer, "vae_dtype") else torch.float16,
        # From-scratch only: inherit compatible weights from an existing model
        # instead of pure random init (ignored for non-scratch model paths).
        scratch_init_from=(trainer.config.get("minit2i_scratch_init_from") or None),
        scratch_inherit_final_layer=bool(trainer.config.get("minit2i_inherit_final_layer", False)),
    )

    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.transformer_uncond = None
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]
    trainer.minit2i_variant = components.get("variant")

    # vae_type "none" = pixel-space (vae=None, RGB-direct "latent"); "sdxl"/"flux1"
    # = latent-space (a frozen AutoencoderKL encodes images to latents for training).
    trainer.minit2i_vae_type = components.get("vae_type", "none")
    trainer.minit2i_latent = trainer.minit2i_vae_type not in (None, "none")
    trainer.minit2i_noise_scale = float(getattr(trainer.transformer.mmjit_config, "noise_scale", 2.0))
    trainer.minit2i_vae_scale_factor = int(components.get("vae_scale_factor", 8))
    trainer.vae = components.get("vae")  # None for pixel
    if trainer.vae is not None:
        trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)
        trainer.vae.requires_grad_(False)
        trainer.vae.eval()
        # High-res latent caching encodes full bucket-resolution images (up to
        # ~2048px) through the VAE. A single fp32 encode at that size peaks at
        # tens of GB (early full-res conv stages + the bottleneck spatial
        # self-attention), independent of the tiny transformer step. Tiled
        # encode/decode splits the image into overlapping tiles so VAE memory
        # is bounded by the tile size, not the image size.
        for _m in ("enable_tiling", "enable_slicing"):
            if hasattr(trainer.vae, _m):
                try:
                    getattr(trainer.vae, _m)()
                except Exception as _e:
                    print(f"{trainer.log_prefix} VAE {_m} failed: {_e}")
        print(f"{trainer.log_prefix} MiniT2I VAE tiling/slicing enabled (bounds high-res encode VRAM)")
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Gradient checkpointing on the MM-JiT transformer.
    if trainer.gradient_checkpointing and hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        try:
            trainer.transformer.enable_gradient_checkpointing()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for MiniT2I transformer")
        except Exception as e:
            print(f"{trainer.log_prefix} grad checkpoint enable failed: {e}")
    elif not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (MiniT2I)")

    # Freeze everything; adapters unfreeze what they train.
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Keep the transformer in train() mode for the whole run: MM-JiT gates
    # gradient checkpointing on `self.training`, and it has no dropout/BN so
    # train mode has no other effect. Without this, checkpointing would only
    # activate after the first sample (its finally restores train mode), and a
    # run with samples disabled would store ALL activations -> high-res OOM.
    # The frozen FLAN-T5 stays in eval() (it has dropout) — loader set it.
    trainer.transformer.train()

    # Block-swap deferred until after adapter setup.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving MiniT2I transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_minit2i(trainer.attention_backend)

    # REPA (representation alignment): load frozen encoder + build the trainable
    # projector BEFORE adapter/optimizer setup so its params join the optimizer.
    trainer._setup_repa()

    print(f"{trainer.log_prefix} MiniT2I model loaded successfully (variant={trainer.minit2i_variant})")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor over the MM-JiT double_blocks, AFTER adapter setup."""
    if not trainer.is_minit2i:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    double_blocks = trainer.transformer.model.net.double_blocks

    from core.memory_management import LayerOffloadConductor
    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=double_blocks,
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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for MiniT2I")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for MiniT2I models (training hook).

    MiniT2I routes all attention through the vendored ``mem_efficient_sdpa``
    primitive. Stage-B extends that primitive to read a transformer attr and
    delegate to the unified conduit; this hook stamps the canonical backend
    string on ``transformer._attn_backend`` for training and honors the
    training guard (sage -> native).
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        # MMJiT.forward reads the net-level attr (transformer.model.net._attn_backend)
        # and fans it out to every attention-bearing block; the outer wrapper attr
        # alone is not consulted. Stamp BOTH (mirrors the inference plumbing in
        # pipeline_backends/minit2i.py) so flash actually engages in training.
        trainer.transformer._attn_backend = b
        net = getattr(getattr(trainer.transformer, "model", None), "net", None)
        if net is not None:
            net._attn_backend = b
        print(f"{trainer.log_prefix} [OK] MiniT2I attention backend set to '{b}'")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set MiniT2I attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def encode_prompt(trainer, prompt: str, requires_grad: bool = False):
    """Encode prompt for MiniT2I: FLAN-T5-Large last_hidden_state + attention mask.

    VERBATIM body of ``BaseTrainer.encode_prompt_minit2i`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    prompt_length = int(trainer.transformer.mmjit_config.prompt_length)
    te_device = trainer.text_encoder.device if hasattr(trainer.text_encoder, "device") else trainer.device

    if not requires_grad:
        from core.models.minit2i.minit2i_pipeline_ops import encode_prompt as _encode
        embeds, mask = _encode(trainer.text_encoder, trainer.tokenizer, prompt, prompt_length, te_device)
        return embeds.detach().to("cpu"), mask[0].detach().to("cpu")

    # TE training: grad-enabled forward (no torch.no_grad), keep on GPU.
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    toks = trainer.tokenizer(
        prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_length,
    )
    input_ids = toks.input_ids.to(te_device)
    attn = toks.attention_mask.to(te_device)
    embeds = trainer.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [1, L, 1024]
    return embeds, attn[0]  # mask [L]


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """MiniT2I VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM bodies of the TWO ``is_minit2i`` early-return branches (pixel-space
    no-VAE + latent-space), self->trainer rename only. Unlike the other archs
    this is FULLY self-contained: the caller dispatches it BEFORE the shared VAE
    staging (pixel-space has no VAE, so ``next(self.vae.parameters())`` must not
    run), and it returns the final CPU/training-dtype tensor directly (no shared
    post-amble). ``minit2i_latent`` selects which sub-branch fires.
    """
    if trainer.is_minit2i and not getattr(trainer, "minit2i_latent", False):
        # Pixel-space: there is no VAE. The "latent" IS the [-1,1] RGB image
        # [1, 3, H, W]. Stored on CPU in training dtype like every other path.
        return image_tensor.to(device="cpu", dtype=trainer.training_dtype)

    if trainer.is_minit2i and getattr(trainer, "minit2i_latent", False):
        # Latent-space: VAE-encode the [-1,1] RGB image to a normalized latent
        # [1, C, H/vsf, W/vsf]. The frozen VAE is moved to GPU for caching by the
        # latent-cache pre-pass (move_vae_to_gpu).
        from core.models.minit2i.minit2i_vae import normalize_latent
        vae_device = next(trainer.vae.parameters()).device
        px = image_tensor.to(device=vae_device, dtype=trainer.vae_dtype)
        with torch.no_grad():
            sample = trainer.vae.encode(px).latent_dist.sample()
            latent = normalize_latent(sample, trainer.vae)
        return latent.to(device="cpu", dtype=trainer.training_dtype)
