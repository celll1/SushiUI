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
allowed by the plan); base_trainer helpers are imported lazily inside functions
(this module is only ever imported LAZILY, after base_trainer has fully loaded,
so there is no import cycle).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from core.attention import AttentionMode, to_diffusers_backend


def load_components(trainer) -> None:
    """Load SD/SDXL model components."""
    # Lazy import (sibling-ops pattern): base_trainer must never be imported at
    # this module's top level - ops modules load while base_trainer may still
    # be mid-initialization in some import orders.
    from core.training.base_trainer import _vramdiag

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


def encode_prompt_custom_te(trainer, prompt: str, requires_grad: bool = False):
    """Encode a prompt with the swapped SDXL text encoder + bridge adapters.

    VERBATIM body of ``BaseTrainer._encode_prompt_custom_te`` (plan P4), moved
    out of the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    import contextlib
    from core.models.sdxl_te_registry import encode_text

    train_body = bool(getattr(trainer, "sdxl_te_train_encoder", False)) and requires_grad
    enc_ctx = contextlib.nullcontext() if train_body else torch.no_grad()
    with enc_ctx:
        hidden, pooled = encode_text(
            trainer.te_custom, trainer.te_tokenizer, [prompt],
            max_len=getattr(trainer, "te_max_len", 256),
            hidden_layer=getattr(trainer, "te_hidden_layer", -2),
            device=trainer.device,
        )
    ad_dtype = next(trainer.te_adapters.parameters()).dtype
    enc, pld = trainer.te_adapters(hidden.to(ad_dtype), pooled.to(ad_dtype))  # [1,L,2048], [1,1280]
    return enc, pld


def encode_prompt_simple(trainer, prompt: str, requires_grad: bool = False):
    """VERBATIM body of ``BaseTrainer._encode_prompt_simple`` (plan P4)."""
    if trainer.is_sdxl:
        # SDXL: Two text encoders
        text_inputs_1 = trainer.tokenizer(
            prompt,
            padding="max_length",
            max_length=trainer.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_inputs_2 = trainer.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=trainer.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

        # Check if text encoders have FP8 weights (requires autocast)
        has_fp8_weights = trainer._has_fp8_text_encoder()

        with context_manager:
            # For FP8 quantized text encoders, use autocast for mixed precision
            # This prevents "ufunc_add_CUDA not implemented for Float8_e4m3fn" errors
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=trainer.training_dtype):
                    # CRITICAL: Both text encoders must use hidden_states[-2] (penultimate layer)
                    # This matches diffusers' StableDiffusionXLPipeline.encode_prompt() implementation
                    encoder_output_1 = trainer.text_encoder(
                        text_inputs_1.input_ids.to(trainer.device),
                        output_hidden_states=True,
                    )
                    text_embeddings_1 = encoder_output_1.hidden_states[-2]

                    encoder_output_2 = trainer.text_encoder_2(
                        text_inputs_2.input_ids.to(trainer.device),
                        output_hidden_states=True,
                    )
                    text_embeddings_2 = encoder_output_2.hidden_states[-2]
                    pooled_embeddings = encoder_output_2[0]
            else:
                # CRITICAL: Both text encoders must use hidden_states[-2] (penultimate layer)
                # This matches diffusers' StableDiffusionXLPipeline.encode_prompt() implementation
                encoder_output_1 = trainer.text_encoder(
                    text_inputs_1.input_ids.to(trainer.device),
                    output_hidden_states=True,
                )
                text_embeddings_1 = encoder_output_1.hidden_states[-2]

                encoder_output_2 = trainer.text_encoder_2(
                    text_inputs_2.input_ids.to(trainer.device),
                    output_hidden_states=True,
                )
                text_embeddings_2 = encoder_output_2.hidden_states[-2]
                pooled_embeddings = encoder_output_2[0]

            text_embeddings = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)

            return text_embeddings, pooled_embeddings
    else:
        # SD1.5: Single text encoder
        text_inputs = trainer.tokenizer(
            prompt,
            padding="max_length",
            max_length=trainer.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

        # Check if text encoder has FP8 weights (requires autocast)
        has_fp8_weights = trainer._has_fp8_text_encoder()

        with context_manager:
            # For FP8 quantized text encoder, use autocast for mixed precision
            if has_fp8_weights:
                with torch.autocast(device_type='cuda', dtype=trainer.training_dtype):
                    text_embeddings = trainer.text_encoder(
                        text_inputs.input_ids.to(trainer.device),
                    )[0]
            else:
                text_embeddings = trainer.text_encoder(
                    text_inputs.input_ids.to(trainer.device),
                )[0]

            return text_embeddings


def encode_prompt_chunked(trainer, prompt: str, requires_grad: bool = False):
    """VERBATIM body of ``BaseTrainer._encode_prompt_chunked`` (plan P4)."""
    tokenizer = trainer.tokenizer_2 if trainer.is_sdxl else trainer.tokenizer
    tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]

    # Split tokens into 75-token chunks
    chunk_size = 75
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunks.append(chunk_tokens)

    # Limit chunks if max_prompt_chunks is set
    if trainer.max_prompt_chunks > 0 and len(chunks) > trainer.max_prompt_chunks:
        chunks = chunks[:trainer.max_prompt_chunks]

    # Encode each chunk
    chunk_embeds_list = []
    pooled_embeddings = None

    context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

    with context_manager:
        for idx, chunk_tokens in enumerate(chunks):
            # Decode tokens back to text
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Encode chunk
            if trainer.is_sdxl:
                # SDXL: Encode with both text encoders
                text_inputs_1 = trainer.tokenizer(
                    chunk_text,
                    padding="max_length",
                    max_length=trainer.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_inputs_2 = trainer.tokenizer_2(
                    chunk_text,
                    padding="max_length",
                    max_length=trainer.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                encoder_output_1 = trainer.text_encoder(
                    text_inputs_1.input_ids.to(trainer.device),
                    output_hidden_states=True,
                )
                text_embeddings_1 = encoder_output_1.hidden_states[-2]

                encoder_output_2 = trainer.text_encoder_2(
                    text_inputs_2.input_ids.to(trainer.device),
                    output_hidden_states=True,
                )
                text_embeddings_2 = encoder_output_2.hidden_states[-2]

                # Use pooled embeddings from first chunk only
                if idx == 0:
                    pooled_embeddings = encoder_output_2[0]

                chunk_embeds = torch.cat([text_embeddings_1, text_embeddings_2], dim=-1)
                chunk_embeds_list.append(chunk_embeds)
            else:
                # SD1.5: Single text encoder
                text_inputs = trainer.tokenizer(
                    chunk_text,
                    padding="max_length",
                    max_length=trainer.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_embeddings = trainer.text_encoder(
                    text_inputs.input_ids.to(trainer.device),
                )[0]

                chunk_embeds_list.append(text_embeddings)

    # Concatenate chunks based on chunking mode
    if trainer.prompt_chunking_mode == "a1111":
        # A1111 mode: concatenate all chunks as-is
        text_embeddings = torch.cat(chunk_embeds_list, dim=1)
    elif trainer.prompt_chunking_mode == "sd_scripts":
        # sd-scripts mode: strip BOS/EOS between chunks
        processed_chunks = []
        for idx, chunk_emb in enumerate(chunk_embeds_list):
            if len(chunk_embeds_list) == 1:
                processed_chunks.append(chunk_emb)
            elif idx == 0:
                # First chunk: remove EOS (last token before padding)
                processed_chunks.append(chunk_emb[:, :-1, :])
            elif idx == len(chunk_embeds_list) - 1:
                # Last chunk: remove BOS (first token)
                processed_chunks.append(chunk_emb[:, 1:, :])
            else:
                # Middle chunks: remove both BOS and EOS
                processed_chunks.append(chunk_emb[:, 1:-1, :])
        text_embeddings = torch.cat(processed_chunks, dim=1)
    else:  # nobos
        # NoBOS mode: strip all BOS/EOS tokens
        processed_chunks = []
        for chunk_emb in chunk_embeds_list:
            # Remove first (BOS) and last (EOS) tokens
            processed_chunks.append(chunk_emb[:, 1:-1, :])
        text_embeddings = torch.cat(processed_chunks, dim=1)

    if trainer.is_sdxl:
        return text_embeddings, pooled_embeddings
    else:
        return text_embeddings


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """SD/SDXL VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``else`` (SD/SDXL) branch, moved with the mechanical
    ``self.`` -> ``trainer.`` receiver rename only. Runs inside the caller's
    ``with torch.no_grad()``; the caller performs the shared final
    dtype/CPU move (post-amble). ``image_tensor`` arrives already moved to the
    VAE device/dtype by the shared pre-amble.
    """
    # SD/SDXL VAE - 統一された処理フロー
    from core.models.sdxl_vae_wrapper import SDXLVAEWrapper

    if isinstance(trainer.vae, SDXLVAEWrapper):
        # SDXLVAEWrapperの場合、内部のAutoencoderKLにアクセス
        vae_model = trainer.vae.vae
    else:
        # 標準のAutoencoderKL
        vae_model = trainer.vae

    # 統一されたエンコード処理
    encoder_output = vae_model.encode(image_tensor)
    latents = encoder_output.latent_dist.sample()

    # DEBUG: Log raw latents before scaling
    if debug_preprocessing:
        print(f"[encode_image DEBUG] Raw latents (before scaling):")
        print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
        print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")
        print(f"  scaling_factor: {vae_model.config.scaling_factor}")

    # Normalize via (sample - shift) * scale so a swapped high-spec VAE with
    # a shift_factor (e.g. FLUX.1 0.1159) is handled; standard SDXL has
    # shift=0 so this is identical to the previous (* scaling_factor).
    from core.models.minit2i.minit2i_vae import normalize_latent as _normalize_latent
    latents = _normalize_latent(latents, vae_model)

    # DEBUG: Log scaled latents
    if debug_preprocessing:
        print(f"[encode_image DEBUG] Scaled latents (after * scaling_factor):")
        print(f"  Mean: {latents.mean():.6f}, Std: {latents.std():.6f}")
        print(f"  Min: {latents.min():.6f}, Max: {latents.max():.6f}")

    # Clean up intermediate tensors
    del encoder_output
    return latents


def train_step(
    trainer,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    pooled_embeddings: torch.Tensor = None,
    time_ids: Optional[torch.Tensor] = None,
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Perform single training step (SD1.5/SDXL).

    Args:
        latents: Image latents [B, C, H, W]
        text_embeddings: Text prompt embeddings
        pooled_embeddings: Pooled text embeddings (SDXL only)
        timesteps: Optional timesteps tensor. If None, sample uniformly from [0, num_train_timesteps)
        debug_save_path: If provided, save latents for debugging
        debug_captions: Captions for debug output
        profile_vram: If True, print VRAM usage
        alphas_cumprod_cached: Pre-cached alphas_cumprod on GPU (for SNR weight computation)

    Returns:
        (loss_tensor, loss_value) - Loss tensor with grad and scalar value
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import (
        add_noise_unified,
        apply_snr_weight,
        get_target_unified,
        predict_original_latent_unified,
        print_vram_usage,
    )

    if profile_vram:
        print_vram_usage("[train_step] Start")

    # Move latents to GPU with correct dtype
    # Latents come from cache (CPU, training_dtype) and must be moved to GPU before training
    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)

    # Sample noise (now on GPU)
    noise = torch.randn_like(latents)

    if profile_vram:
        print_vram_usage("[train_step] After noise generation")

    # Sample random timestep (or use provided timesteps)
    batch_size = latents.shape[0]

    # Determine noise process from trainer config (set by train_runner.py)
    noise_process = getattr(trainer, 'noise_process', 'ddpm')  # Default: ddpm for backward compatibility

    if timesteps is None:
        if noise_process == "ddpm":
            # DDPM: sample discrete timesteps [0, num_train_timesteps)
            if trainer.timestep_sampler is not None:
                # Use timestep sampler: sample from [0, 1] then scale to discrete timesteps
                # IMPORTANT: DDPM convention is REVERSED from Flow Matching
                # DDPM: t=999 (noisy) → t=0 (clean)
                # Flow: t=0 (noisy) → t=1 (clean)
                # So we need to flip: YAML [0,1] → DDPM [999,0]
                # Example: YAML min=0, max=0.2 (want noisy) → DDPM [999, 800] (noisy)
                timesteps_continuous = trainer.timestep_sampler.sample(batch_size, trainer.device)
                timesteps = ((1.0 - timesteps_continuous) * trainer.noise_scheduler.config.num_train_timesteps).long()
                timesteps = timesteps.clamp(0, trainer.noise_scheduler.config.num_train_timesteps - 1)
            else:
                # Legacy behavior: sample uniformly from [0, num_train_timesteps)
                timesteps = torch.randint(
                    0,
                    trainer.noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=trainer.device,
                ).long()
        elif noise_process == "flow":
            # Flow Matching: sample continuous timesteps [0, 1]
            if trainer.timestep_sampler is not None:
                # Use timestep sampler (already returns [0, 1])
                timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
            else:
                # Uniform sampling from [0, 1]
                timesteps = torch.rand((batch_size,), device=trainer.device)
    else:
        # MNT: timesteps provided externally
        if noise_process == "ddpm":
            # Convert flow-matching timesteps [0, 1] to discrete timesteps for DDPM
            # IMPORTANT: DDPM convention is REVERSED from Flow Matching
            # DDPM: t=999 (noisy) → t=0 (clean)
            # Flow: t=0 (noisy) → t=1 (clean)
            # So we need to flip: YAML [0,1] → DDPM [999,0]
            timesteps = ((1.0 - timesteps) * trainer.noise_scheduler.config.num_train_timesteps).long()
            timesteps = timesteps.clamp(0, trainer.noise_scheduler.config.num_train_timesteps - 1)
        elif noise_process == "flow":
            # Flow matching: timesteps are already [0, 1]
            pass

    # Add noise to latents using unified framework
    noisy_latents = add_noise_unified(
        noise_process=noise_process,
        noise_scheduler=trainer.noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )

    # Prepare added_cond_kwargs for SDXL. Per-item time_ids (real original_size /
    # crop_top_left / target_size from the dataset bucketing) are passed in when
    # SDXL micro-conditioning is enabled; otherwise fall back to the legacy
    # all-equal-to-latent-size, crop=(0,0) values.
    added_cond_kwargs = None
    if trainer.is_sdxl and pooled_embeddings is not None:
        if time_ids is not None:
            add_time_ids = time_ids.to(device=trainer.device, dtype=pooled_embeddings.dtype)
        else:
            latent_height, latent_width = latents.shape[2], latents.shape[3]
            image_height, image_width = latent_height * 8, latent_width * 8
            add_time_ids = torch.tensor(
                [[image_height, image_width, 0, 0, image_height, image_width]],
                dtype=pooled_embeddings.dtype, device=trainer.device,
            ).repeat(batch_size, 1)

        added_cond_kwargs = {
            "text_embeds": pooled_embeddings,
            "time_ids": add_time_ids
        }

    if profile_vram:
        print_vram_usage("[train_step] Before UNet forward")

    # Enable gradients for gradient checkpointing
    noisy_latents.requires_grad_(True)
    text_embeddings.requires_grad_(True)
    if pooled_embeddings is not None:
        pooled_embeddings.requires_grad_(True)

    # Predict noise using UNet
    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            if trainer.is_sdxl and added_cond_kwargs is not None:
                model_pred = trainer.unet(
                    noisy_latents,
                    timesteps,
                    text_embeddings,
                    added_cond_kwargs=added_cond_kwargs
                ).sample
            else:
                model_pred = trainer.unet(
                    noisy_latents,
                    timesteps,
                    text_embeddings
                ).sample
    else:
        if trainer.is_sdxl and added_cond_kwargs is not None:
            model_pred = trainer.unet(
                noisy_latents,
                timesteps,
                text_embeddings,
                added_cond_kwargs=added_cond_kwargs
            ).sample
        else:
            model_pred = trainer.unet(
                noisy_latents,
                timesteps,
                text_embeddings
            ).sample

    if profile_vram:
        print_vram_usage("[train_step] After UNet forward")

    # DEUS debug check removed (architecture no longer maintained)

    # Get target based on unified framework
    prediction_target = getattr(trainer, 'prediction_target', 'epsilon')  # Default: epsilon for backward compatibility
    target = get_target_unified(
        noise_process=noise_process,
        prediction_target=prediction_target,
        noise_scheduler=trainer.noise_scheduler,
        latents=latents,
        noise=noise,
        timesteps=timesteps,
    )

    # Calculate loss (always in fp32)
    # TEMPORARY: .float() is redundant since everything is FP32, but kept for safety
    loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss_per_sample = loss_per_element.mean([1, 2, 3])

    # Apply Min-SNR gamma weighting (only for epsilon prediction)
    # Min-SNR was designed for epsilon prediction; applying it to v-prediction is theoretically unsound
    # When dual loss is enabled (reconstruction_loss_weight > 0), also return weights
    # to compensate for lost prediction weight by boosting reconstruction weight
    min_snr_weights = None
    if trainer.min_snr_gamma > 0 and prediction_target == "epsilon":
        if trainer.reconstruction_loss_weight > 0:
            # Return weights for dual loss compensation
            loss_per_sample_weighted, min_snr_weights = apply_snr_weight(
                loss_per_sample, timesteps, trainer.noise_scheduler, trainer.min_snr_gamma,
                return_weights=True, alphas_cumprod_cached=alphas_cumprod_cached
            )
        else:
            loss_per_sample_weighted = apply_snr_weight(
                loss_per_sample, timesteps, trainer.noise_scheduler, trainer.min_snr_gamma,
                alphas_cumprod_cached=alphas_cumprod_cached
            )
    else:
        loss_per_sample_weighted = loss_per_sample

    mse_loss = loss_per_sample_weighted.mean()

    # Add SNR and/or Energy regularization if enabled (can use both simultaneously)
    regularization_loss = torch.tensor(0.0, device=trainer.device)

    # Compute predicted latent once (used by both regularization losses and debug save)
    predicted_latent_for_reg = None
    predicted_latent_for_recon = None  # Will be set in reconstruction loss path
    if trainer.snr_regularization_loss is not None or trainer.energy_regularization_loss is not None:
        # Compute predicted latent from model_pred (keep gradients for backprop)
        predicted_latent_for_reg = predict_original_latent_unified(
            noise_process=noise_process,
            prediction_target=prediction_target,
            noise_scheduler=trainer.noise_scheduler,
            noisy_latents=noisy_latents,
            model_pred=model_pred,
            timesteps=timesteps,
        )

    # SNR regularization (周波数領域の過剰デノイズ抑制)
    if trainer.snr_regularization_loss is not None:
        # Convert timesteps to continuous [0, 1] for regularization
        if noise_process == "ddpm":
            timesteps_continuous = timesteps.float() / trainer.noise_scheduler.config.num_train_timesteps
        else:  # flow
            timesteps_continuous = timesteps.float()  # Already [0, 1]

        snr_reg_loss = trainer.snr_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps_continuous
        )
        regularization_loss = regularization_loss + snr_reg_loss

    # Energy regularization (空間領域のエネルギー保存)
    if trainer.energy_regularization_loss is not None:
        # Convert timesteps to continuous [0, 1] for regularization
        if noise_process == "ddpm":
            timesteps_continuous = timesteps.float() / trainer.noise_scheduler.config.num_train_timesteps
        else:  # flow
            timesteps_continuous = timesteps.float()  # Already [0, 1]

        energy_reg_loss = trainer.energy_regularization_loss(
            predicted_latent_for_reg,
            latents,
            timesteps_continuous
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
            predicted_latent_for_recon = predict_original_latent_unified(
                noise_process=noise_process,
                prediction_target=prediction_target,
                noise_scheduler=trainer.noise_scheduler,
                noisy_latents=noisy_latents,
                model_pred=model_pred,
                timesteps=timesteps,
            )

        recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
        recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])

        # Dual loss with min-SNR weight compensation
        # When min_snr_gamma > 0, the prediction loss is reduced by min_snr_weights for clean timesteps.
        # We compensate for this "lost" weight by boosting the reconstruction loss weight.
        #
        # Original dual loss: alpha * pred_loss + beta * recon_loss (alpha + beta = 1.0)
        # With min-SNR: pred_loss is already weighted by min_snr_weights
        #
        # Compensation formula (per-sample):
        #   lost_weight = (1 - min_snr_weight) * alpha  (weight originally for pred_loss that was reduced)
        #   effective_beta = beta + lost_weight        (boost recon_loss by lost amount)
        #   combined_loss = pred_loss_weighted + effective_beta * recon_loss
        #
        # Note: pred_loss already has min_snr_weight applied, so we use it directly without alpha multiplier

        alpha = 1.0 - trainer.reconstruction_loss_weight
        beta = trainer.reconstruction_loss_weight

        if min_snr_weights is not None:
            # Per-sample compensation: boost recon_loss weight based on how much pred_loss was reduced
            # lost_weight[i] = (1 - min_snr_weights[i]) * alpha
            # effective_beta[i] = beta + lost_weight[i]
            lost_weight = (1.0 - min_snr_weights) * alpha  # [batch_size]
            effective_beta = beta + lost_weight  # [batch_size]

            # Per-sample combined loss
            # loss_per_sample_weighted already has min_snr weighting applied
            combined_loss_per_sample = loss_per_sample_weighted + effective_beta * recon_loss_per_sample
            combined_loss = combined_loss_per_sample.mean()
        else:
            # No min-SNR: standard dual loss
            recon_loss = recon_loss_per_sample.mean()
            combined_loss = alpha * mse_loss + beta * recon_loss

        # For return value
        recon_loss = recon_loss_per_sample.mean()

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
                predicted_latent_for_recon = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=trainer.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

            recon_loss_per_element = F.mse_loss(predicted_latent_for_recon.float(), latents.float(), reduction="none")
            recon_loss_per_sample = recon_loss_per_element.mean([1, 2, 3])
            recon_loss = recon_loss_per_sample.mean()

        # Total loss (prediction loss + regularization)
        loss = mse_loss + regularization_loss

    if profile_vram:
        print_vram_usage("[train_step] After loss calculation")

    # Debug save if requested
    if debug_save_path is not None:
        debug_save_path.mkdir(parents=True, exist_ok=True)
        timestep_value = timesteps[0].item()

        # Reuse predicted_latent from reconstruction loss calculation if available
        # This avoids redundant computation (predict_original_latent_unified is expensive)
        if predicted_latent_for_recon is not None:
            predicted_latent_for_debug = predicted_latent_for_recon.detach()
        elif predicted_latent_for_reg is not None:
            predicted_latent_for_debug = predicted_latent_for_reg.detach()
        else:
            # Fallback: compute predicted_latent if not available
            with torch.no_grad():
                predicted_latent_for_debug = predict_original_latent_unified(
                    noise_process=noise_process,
                    prediction_target=prediction_target,
                    noise_scheduler=trainer.noise_scheduler,
                    noisy_latents=noisy_latents,
                    model_pred=model_pred,
                    timesteps=timesteps,
                )

        debug_data = {
            'latents': latents[0:1].detach().cpu(),
            'noisy_latents': noisy_latents[0:1].detach().cpu(),
            'predicted_noise': model_pred[0:1].detach().cpu(),
            'actual_noise': noise[0:1].detach().cpu(),
            'predicted_latent': predicted_latent_for_debug[0:1].detach().cpu(),
            'timestep': timestep_value,
            'loss': loss_per_sample_weighted[0].item(),
            'loss_batch_mean': loss.item(),
            'loss_unweighted': loss_per_sample[0].item(),
            'recon_loss': recon_loss_per_sample[0].item(),
            'recon_loss_batch_mean': recon_loss.item(),
            'batch_size': batch_size,
            'min_snr_gamma': trainer.min_snr_gamma,
        }

        if debug_captions is not None and len(debug_captions) > 0:
            debug_data['caption'] = debug_captions[0]
            debug_data['all_captions'] = debug_captions

        if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
            first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
            if first_ref:
                debug_data['reference_image_path'] = first_ref

        # SDXL micro-conditioning for this debug sample (item 0): lets the user verify
        # crop augmentation. time_ids order = [orig_h, orig_w, crop_top, crop_left,
        # target_h, target_w]. crop_top_left != (0,0) or original_size != target_size
        # indicates a random crop / scale. Per-item array included for the whole batch.
        try:
            if trainer.is_sdxl and add_time_ids is not None:
                _ti_all = add_time_ids.detach().cpu().to(torch.int64).tolist()  # [B, 6]
                _t0 = _ti_all[0]
                debug_data['sdxl_time_ids'] = _t0
                debug_data['original_size'] = [int(_t0[1]), int(_t0[0])]   # (w, h)
                debug_data['crop_top_left'] = [int(_t0[3]), int(_t0[2])]   # (left, top) = crop point
                debug_data['target_size'] = [int(_t0[5]), int(_t0[4])]     # (w, h) = bucket
                debug_data['sdxl_time_ids_all'] = _ti_all
        except Exception:
            pass

        torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:04d}.pt")
        del predicted_latent_for_debug

    # Return loss tensor (with gradient), pred_loss value, and recon_loss value
    # IMPORTANT: Do NOT call .item() on loss here - it breaks the computation graph!
    # The training loop will call .backward() on the loss tensor.
    pred_loss_value = mse_loss.item()
    recon_loss_value = recon_loss.item()

    # Free intermediate tensors explicitly to reduce VRAM usage
    # But keep 'loss' tensor for backward pass
    del noise, noisy_latents, model_pred, target, recon_loss
    if trainer.is_sdxl and added_cond_kwargs is not None:
        del added_cond_kwargs

    return loss, pred_loss_value, recon_loss_value


# ============================================================
# SD1.5 / SDXL Sample Generation (plan P7)
# ============================================================
# Verbatim body of BaseTrainer.generate_sample (base_trainer.py), moved out of
# the spine with the mechanical self.->trainer. receiver rename and a sanctioned
# lazy base_trainer import for log_verbose. BaseTrainer.generate_sample stays as
# a thin delegator (ControlNetTrainer overrides it and calls super()), and the
# sd15/sdxl handlers route sample() through trainer.generate_sample() so the
# ControlNet override is preserved.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = -1,
    current_step: int = 0,
    schedule_type: str = "uniform",
    condition_image_path: Optional[str] = None,
    reference_image_path: Optional[str] = None,
):
    """
    Generate sample image during training (SD/SDXL).
    Uses custom_sampling_loop() - EXACTLY the same method as normal txt2img generation.

    Args:
        prompt: Text prompt
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        seed: Random seed (-1 for random)
        current_step: Current training step (for logging)
        schedule_type: Timestep schedule type (uniform, karras, exponential)

    Returns:
        PIL Image
    """
    from core.training.base_trainer import log_verbose

    from PIL import Image
    import random

    print(f"{trainer.log_prefix} Generating sample: {prompt[:50]}...")

    # SD/SDXL: Use custom_sampling_loop
    from core.inference.custom_sampling import custom_sampling_loop
    from core.inference.schedulers import get_scheduler

    # Set models to eval mode
    trainer.unet.eval()
    trainer.vae.eval()
    trainer.text_encoder.eval()
    if trainer.text_encoder_2 is not None:
        trainer.text_encoder_2.eval()

    # Debug: Check if LoRA is applied to U-Net
    lora_layers_found = 0
    for name, module in trainer.unet.named_modules():
        if hasattr(module, 'lora_down') or 'LoRA' in type(module).__name__:
            lora_layers_found += 1
    log_verbose(f"{trainer.log_prefix} [Sample] U-Net has {lora_layers_found} LoRA layers")

    try:
        # ========================================
        # STEP 1: Create Temporary Pipeline Object
        # ========================================
        # custom_sampling_loop() requires a pipeline object with scheduler, unet, vae, etc.
        # Create a minimal pipeline-like object with necessary components

        if trainer.is_sdxl:
            from diffusers import StableDiffusionXLPipeline
            # Create a minimal pipeline object
            class TempPipeline:
                def __init__(self, unet, vae, text_encoder, text_encoder_2, scheduler, tokenizer, tokenizer_2):
                    self.unet = unet
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.text_encoder_2 = text_encoder_2
                    self.scheduler = scheduler
                    self.tokenizer = tokenizer
                    self.tokenizer_2 = tokenizer_2
                    # Set default config
                    self.vae_scale_factor = 8
                    self.image_processor = None  # Not needed for custom_sampling_loop

            # Map schedule_type (sgm_uniform -> uniform)
            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            # Create scheduler using get_scheduler()
            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(trainer.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            # Create temporary pipeline
            pipeline = TempPipeline(
                unet=trainer.unet,
                vae=trainer.vae,
                text_encoder=trainer.text_encoder,
                text_encoder_2=trainer.text_encoder_2,
                scheduler=scheduler,
                tokenizer=trainer.tokenizer,
                tokenizer_2=trainer.tokenizer_2
            )
        else:
            from diffusers import StableDiffusionPipeline
            # Create a minimal pipeline object for SD1.5
            class TempPipeline:
                def __init__(self, unet, vae, text_encoder, scheduler, tokenizer):
                    self.unet = unet
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.scheduler = scheduler
                    self.tokenizer = tokenizer
                    # Set default config
                    self.vae_scale_factor = 8
                    self.image_processor = None  # Not needed for custom_sampling_loop

            # Map schedule_type (sgm_uniform -> uniform)
            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            # Create scheduler using get_scheduler()
            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(trainer.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            # Create temporary pipeline
            pipeline = TempPipeline(
                unet=trainer.unet,
                vae=trainer.vae,
                text_encoder=trainer.text_encoder,
                scheduler=scheduler,
                tokenizer=trainer.tokenizer
            )

        # ========================================
        # STEP 2: Text Encoding
        # ========================================
        trainer.move_text_encoder_to_gpu()

        # Encode prompt
        if trainer.is_sdxl:
            prompt_embeds, pooled_prompt_embeds = trainer.encode_prompt(prompt, requires_grad=False)
            negative_prompt_embeds, negative_pooled_prompt_embeds = trainer.encode_prompt("", requires_grad=False)
        else:
            prompt_embeds = trainer.encode_prompt(prompt, requires_grad=False)
            negative_prompt_embeds = trainer.encode_prompt("", requires_grad=False)
            pooled_prompt_embeds = None
            negative_pooled_prompt_embeds = None

        # Pad negative embeddings to match positive embeddings sequence length (for prompt chunking)
        if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
            # Positive prompt has more tokens (chunking applied)
            # Pad negative embeddings with zeros to match
            seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
            padding = torch.zeros(
                (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                dtype=negative_prompt_embeds.dtype,
                device=negative_prompt_embeds.device
            )
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
            log_verbose(f"{trainer.log_prefix} [Sample] Padded negative embeddings: {negative_prompt_embeds.shape[1] - seq_len_diff} -> {negative_prompt_embeds.shape[1]} tokens")

        trainer.move_text_encoder_to_cpu()
        torch.cuda.empty_cache()

        # ========================================
        # STEP 2.5: Vision Encoder conditioning (if reference image + VE loaded)
        # ========================================
        ve_obj = getattr(trainer, 'vision_encoder', None)
        if reference_image_path and ve_obj is not None:
            try:
                from PIL import Image as PILImage
                ref_img = PILImage.open(reference_image_path).convert("RGB")
                target_dim = prompt_embeds.shape[-1]
                train_ve = getattr(trainer, '_train_vision_encoder', False)
                if not train_ve:
                    print(f"{trainer.log_prefix} [Sample] Moving Vision Encoder to GPU for sample conditioning")
                    ve_obj.to(trainer.device)
                ve_obj.eval()
                with torch.no_grad():
                    ve_pos, _ = ve_obj.encode([ref_img], target_dim=target_dim, dtype=prompt_embeds.dtype)
                ve_pos = ve_pos.to(trainer.device)
                ve_neg = torch.zeros_like(ve_pos)
                prompt_embeds = torch.cat([prompt_embeds, ve_pos], dim=1)
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, ve_neg], dim=1)
                if not train_ve:
                    ve_obj.to("cpu")
                    torch.cuda.empty_cache()
                    print(f"{trainer.log_prefix} [Sample] Vision Encoder moved back to CPU")
                print(f"{trainer.log_prefix} [Sample] VE conditioning applied: embeds shape {prompt_embeds.shape}")
            except Exception as ve_err:
                print(f"{trainer.log_prefix} [Sample] WARNING: VE conditioning failed: {ve_err}, skipping")

        # ========================================
        # STEP 3: Create Generator
        # ========================================
        if seed < 0:
            actual_seed = random.randint(0, 2**32 - 1)
        else:
            actual_seed = seed

        generator = torch.Generator(device=trainer.device).manual_seed(actual_seed)

        # ========================================
        # STEP 4: Call custom_sampling_loop (SAME as pipeline.generate_txt2img)
        # ========================================
        trainer.move_main_model_to_gpu()
        trainer.move_vae_to_gpu()

        # Detect v-prediction and apply guidance_rescale if needed
        is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
        guidance_rescale = 0.7 if is_v_prediction else 0.0

        log_verbose(f"{trainer.log_prefix} [Sample] Using custom_sampling_loop()")
        log_verbose(f"{trainer.log_prefix} [Sample] Scheduler: {type(pipeline.scheduler).__name__}")
        log_verbose(f"{trainer.log_prefix} [Sample] V-prediction: {is_v_prediction}, guidance_rescale: {guidance_rescale}")

        # Use autocast for sample generation (ensures LoRA dtype compatibility)
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            image = custom_sampling_loop(
                pipeline=pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                width=width,
                height=height,
                generator=generator,
                ancestral_generator=None,  # Not needed for training samples
                latents=None,
                prompt_embeds_callback=None,  # No prompt editing for training samples
                progress_callback=None,
                step_callback=None,
                developer_mode=False,
                cfg_schedule_type="constant",  # Simple constant CFG for training samples
                cfg_schedule_min=1.0,
                cfg_schedule_max=None,
                cfg_schedule_power=2.0,
                cfg_rescale_snr_alpha=0.0,
                dynamic_threshold_percentile=0.0,
                dynamic_threshold_mimic_scale=1.0,
                nag_enable=False,  # No NAG for training samples
                nag_scale=5.0,
                nag_tau=3.5,
                nag_alpha=0.25,
                nag_sigma_end=0.0,
                nag_negative_prompt_embeds=None,
                nag_negative_pooled_prompt_embeds=None,
                attention_type="normal",  # Normal attention for training samples
            )

            # Move models back to CPU
            trainer.move_main_model_to_cpu()
            trainer.move_vae_to_cpu()
            torch.cuda.empty_cache()

            log_verbose(f"{trainer.log_prefix} Sample generated successfully (seed: {actual_seed})")
            return image

    except Exception as e:
        print(f"{trainer.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
        print(f"{trainer.log_prefix} [Sample] Sample generation failed - this is expected for early training steps")
        print(f"{trainer.log_prefix} [Sample] Training will continue normally")

        # Return a placeholder image (blank white image)
        from PIL import Image
        placeholder = Image.new("RGB", (width, height), color=(255, 255, 255))
        return placeholder

    finally:
        # Restore training mode
        trainer.unet.train()
        trainer.vae.train()
        trainer.text_encoder.train()
        if trainer.text_encoder_2 is not None:
            trainer.text_encoder_2.train()

        # Ensure U-Net is back on GPU for training continuation
        trainer.move_main_model_to_gpu()
