"""sd_sdxl_ops.py — SD1.5/SDXL loader + attention-backend free functions (P3a).

VERBATIM bodies of ``BaseTrainer._load_sd_sdxl_components`` and
``BaseTrainer._setup_attention_backend_sd_sdxl`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

ONE loader serves BOTH archs: it SETS ``trainer.is_sdxl`` from the detected
model layout (SDXL vs SD1.5). The arch handler is constructed at the END of
``BaseTrainer.__init__`` (base_trainer.py:1115) — AFTER this runs (:1104) — so
``is_sdxl`` is only final once this function returns. The load-time dispatcher
therefore CANNOT route via ``trainer.arch``; both the base_trainer dispatcher
AND ``arch/sd15.py`` / ``arch/sdxl.py`` call these free functions so the body is
defined exactly once and stays byte-identical (plan P3a construction-order note).

Module-level names used by the moved bodies are imported here (import adjustment,
allowed by the plan); ``_vramdiag`` is imported from base_trainer at module top
(this module is only ever imported LAZILY, after base_trainer has fully loaded,
so there is no import cycle).
"""
from __future__ import annotations

from pathlib import Path

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from core.attention import AttentionMode, to_diffusers_backend
from core.training.base_trainer import _vramdiag


def load_components(trainer) -> None:
    """Load SD/SDXL model components."""
    is_safetensors = trainer.model_path.endswith('.safetensors')

    if is_safetensors:
        print(f"{trainer.log_prefix} Loading from safetensors file")
        # Try SDXL first, fall back to SD1.5
        try:
            print(f"{trainer.log_prefix} Trying SDXL pipeline...")
            temp_pipeline = StableDiffusionXLPipeline.from_single_file(
                trainer.model_path,
                torch_dtype=trainer.dtype,
                use_safetensors=True,
            )
            is_sdxl_model = True
        except Exception as e:
            print(f"{trainer.log_prefix} Not SDXL, trying SD1.5 pipeline...")
            temp_pipeline = StableDiffusionPipeline.from_single_file(
                trainer.model_path,
                torch_dtype=trainer.dtype,
                use_safetensors=True,
            )
            is_sdxl_model = False

        # Extract components
        trainer.vae = temp_pipeline.vae
        trainer.text_encoder = temp_pipeline.text_encoder
        trainer.tokenizer = temp_pipeline.tokenizer
        trainer.unet = temp_pipeline.unet

        # Save original scheduler for inference (sample generation)
        # This preserves the model's original scheduler config (prediction_type, timestep_spacing, etc.)
        trainer.original_scheduler = temp_pipeline.scheduler

        # Use DDPMScheduler for training
        trainer.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="epsilon"
        )

        # SDXL-specific components
        if is_sdxl_model:
            trainer.text_encoder_2 = temp_pipeline.text_encoder_2
            trainer.tokenizer_2 = temp_pipeline.tokenizer_2
        else:
            trainer.text_encoder_2 = None
            trainer.tokenizer_2 = None

        del temp_pipeline
        trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    else:
        print(f"{trainer.log_prefix} Loading from diffusers directory")
        trainer.vae = AutoencoderKL.from_pretrained(
            trainer.model_path,
            subfolder="vae",
            torch_dtype=trainer.vae_dtype
        )

        trainer.text_encoder = CLIPTextModel.from_pretrained(
            trainer.model_path,
            subfolder="text_encoder",
            torch_dtype=trainer.dtype
        )

        trainer.tokenizer = CLIPTokenizer.from_pretrained(
            trainer.model_path,
            subfolder="tokenizer"
        )

        trainer.unet = UNet2DConditionModel.from_pretrained(
            trainer.model_path,
            subfolder="unet",
            torch_dtype=trainer.dtype
        )

        # Save original scheduler for inference (sample generation)
        from diffusers.schedulers import EulerDiscreteScheduler
        trainer.original_scheduler = EulerDiscreteScheduler.from_pretrained(
            trainer.model_path,
            subfolder="scheduler"
        )

        # Use DDPMScheduler for training
        trainer.noise_scheduler = DDPMScheduler.from_pretrained(
            trainer.model_path,
            subfolder="scheduler"
        )

        # Check for SDXL
        if (Path(trainer.model_path) / "text_encoder_2").exists():
            trainer.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                trainer.model_path,
                subfolder="text_encoder_2",
                torch_dtype=trainer.dtype
            )
            trainer.tokenizer_2 = CLIPTokenizer.from_pretrained(
                trainer.model_path,
                subfolder="tokenizer_2"
            )
            is_sdxl_model = True
        else:
            trainer.text_encoder_2 = None
            trainer.tokenizer_2 = None
            is_sdxl_model = False

    # Store SDXL flag
    trainer.is_sdxl = is_sdxl_model

    # Custom architecture (VAE/TE swap) changes the base structure (conv channels /
    # text encoder) and is LoRA-incompatible — LoRA cannot train the resized conv
    # layers or the TE bridge adapters, and the LoRA save path does not persist them
    # (the trained pieces would be silently lost). Require full fine-tune.
    if trainer.is_sdxl:
        _tm = str(trainer.config.get("training_method", "lora") or "lora").strip().lower()
        _wants_custom = (
            str(trainer.config.get("sdxl_vae_type", "") or "").strip().lower() not in ("", "none", "sdxl")
            or str(trainer.config.get("sdxl_te_type", "") or "").strip().lower() not in ("", "none", "clip")
        )
        if _wants_custom and _tm == "lora":
            raise ValueError(
                "SDXL custom architecture (sdxl_vae_type / sdxl_te_type) requires "
                "training_method='full' — LoRA cannot train the resized conv layers or "
                "the text-encoder bridge adapters. Switch to Full Fine-tune."
            )

    # SDXL high-spec VAE migration (optional): swap the VAE and resize the U-Net
    # conv_in/conv_out to the new latent channel count (channel-partial copy; the
    # transformer body is kept and adapts during training). "none"/"sdxl" keeps the
    # standard 4ch VAE so existing SDXL runs are unchanged.
    trainer.sdxl_vae_type = "sdxl"
    if trainer.is_sdxl:
        _svt = str(trainer.config.get("sdxl_vae_type", "") or "").strip().lower()
        if _svt and _svt not in ("none", "sdxl"):
            from core.models.sdxl_custom_arch import (
                load_alt_vae, resize_unet_in_out, vae_latent_channels,
            )
            C = vae_latent_channels(_svt)
            print(f"{trainer.log_prefix} [SDXL custom] Migrating to '{_svt}' VAE ({C}ch) "
                  f"+ resizing U-Net conv_in/out")
            trainer.vae = load_alt_vae(_svt, torch_dtype=trainer.vae_dtype)
            resize_unet_in_out(trainer.unet, C)
            trainer.sdxl_vae_type = _svt
    try:
        trainer.vae_latent_channels = int(trainer.vae.config.latent_channels)
    except Exception:
        trainer.vae_latent_channels = 4

    # Custom SDXL Text Encoder (optional): swap CLIP for an alternative encoder
    # (SigLIP2 text / FLAN-T5 / Qwen3) + trainable adapters bridging to the fixed
    # U-Net interface (2048 / 1280). The CLIP TEs stay loaded but unused (encode_prompt
    # branches to the custom path); "none" keeps standard CLIP behavior unchanged.
    trainer.sdxl_te_type = "none"
    if trainer.is_sdxl:
        _tet = str(trainer.config.get("sdxl_te_type", "") or "").strip().lower()
        if _tet and _tet not in ("none", "clip"):
            from core.models.sdxl_te_registry import load_sdxl_te
            from core.models.sdxl_te_adapter import SDXLTEAdapters
            trainer.te_max_len = int(trainer.config.get("sdxl_te_max_len", 256) or 256)
            trainer.te_hidden_layer = int(trainer.config.get("sdxl_te_hidden_layer", -2))
            trainer.sdxl_te_train_encoder = bool(trainer.config.get("sdxl_te_train_encoder", False))
            _ad_dtype = getattr(trainer, "training_dtype", None) or trainer.dtype
            trainer.te_custom, trainer.te_tokenizer, trainer.te_dim = load_sdxl_te(
                _tet, dtype=trainer.dtype, device=trainer.device, max_len=trainer.te_max_len)
            if trainer.sdxl_te_train_encoder:
                trainer.te_custom.requires_grad_(True); trainer.te_custom.train()
            else:
                trainer.te_custom.requires_grad_(False); trainer.te_custom.eval()
            trainer.te_adapters = SDXLTEAdapters(trainer.te_dim).to(device=trainer.device, dtype=_ad_dtype)
            trainer.te_adapters.train()
            trainer.sdxl_te_type = _tet
            print(f"{trainer.log_prefix} [SDXL custom] Text encoder '{_tet}' "
                  f"(dim={trainer.te_dim}, max_len={trainer.te_max_len}, layer={trainer.te_hidden_layer}, "
                  f"train_encoder={trainer.sdxl_te_train_encoder}) + bridge adapters")

    # No transformer for SD/SDXL
    trainer.transformer = None
    trainer.transformer_original = None

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_sd_sdxl(trainer.attention_backend)

    # Enable gradient checkpointing for U-Net (CRITICAL for VRAM reduction)
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (SD/SDXL)")
    elif hasattr(trainer.unet, 'enable_gradient_checkpointing'):
        trainer.unet.enable_gradient_checkpointing()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for U-Net")
    else:
        print(f"{trainer.log_prefix} WARNING: Gradient checkpointing not available for U-Net")

    # Enable gradient checkpointing for Text Encoders
    if trainer.gradient_checkpointing and hasattr(trainer.text_encoder, 'gradient_checkpointing_enable'):
        trainer.text_encoder.gradient_checkpointing_enable()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Text Encoder 1")

    if trainer.gradient_checkpointing and trainer.text_encoder_2 is not None:
        if hasattr(trainer.text_encoder_2, 'gradient_checkpointing_enable'):
            trainer.text_encoder_2.gradient_checkpointing_enable()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for Text Encoder 2")

    print(f"{trainer.log_prefix} {'SDXL' if trainer.is_sdxl else 'SD1.5'} model loaded successfully")
    if trainer.debug_vram:
        _vramdiag("model_load_end")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for SD/SDXL models.

    Branches on ``self.attention_impl`` (migration flag, SCOPE LOCK 2026-07-03):

    - ``"conduit"`` (fresh-run default): install the SAME
      :class:`UnifiedAttnProcessor` that SDXL/SD1.5 INFERENCE already uses on
      the training UNet via :func:`set_attention_processor`, with
      ``mode=AttentionMode.TRAINING``. This routes attention through the
      unified conduit so ALL backends work in training — notably ``tq``, which
      the diffusers path silently collapses to native via
      ``to_diffusers_backend``. The training UNet is the same diffusers
      ``UNet2DConditionModel`` whose attention modules accept a processor
      object (the exact object SDXL inference patches), so this is a pure
      attention-only swap; ``added_cond_kwargs`` / ``time_ids`` / pooled
      embeds are computed OUTSIDE attention and are untouched.
    - ``"diffusers"``: keep the pre-migration
      ``unet.set_attention_backend(to_diffusers_backend(b))`` path
      byte-identical.

    ``_resolve_training_backend`` (R4) runs FIRST in both branches so sage is
    stripped to native regardless of impl; ``tq`` survives and now trains on
    SDXL via the conduit branch.
    """
    if trainer.unet is None:
        print(f"{trainer.log_prefix} WARNING: UNet not loaded, skipping attention backend setup")
        return

    b = trainer._resolve_training_backend(backend)

    if trainer.attention_impl == "diffusers":
        # Pre-migration path (byte-identical): diffusers registry dispatch.
        try:
            print(f"{trainer.log_prefix} Setting SD/SDXL UNet attention backend '{b}' (impl=diffusers)...")
            trainer.unet.set_attention_backend(to_diffusers_backend(b))
            print(f"{trainer.log_prefix} [OK] Attention backend set via set_attention_backend('{to_diffusers_backend(b)}')")
        except Exception as e:
            print(f"{trainer.log_prefix} WARNING: Failed to set attention backend '{b}': {e}")
            print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")
        return

    # Conduit path (default): reuse the inference UnifiedAttnProcessor in
    # TRAINING mode so tq (and every other conduit backend) engages.
    try:
        from core.inference.attention_processors import set_attention_processor
        print(f"{trainer.log_prefix} Setting SD/SDXL UNet attention via UnifiedAttnProcessor (backend='{b}', impl=conduit, mode=TRAINING)...")
        trainer._sdxl_original_attn_processors = set_attention_processor(
            trainer.unet, b, mode=AttentionMode.TRAINING
        )
        print(f"{trainer.log_prefix} [OK] Conduit attention processor installed on training UNet (backend='{b}')")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to install conduit attention processor '{b}': {e}")
        print(f"{trainer.log_prefix} Falling back to the diffusers default processor (native attention)")
