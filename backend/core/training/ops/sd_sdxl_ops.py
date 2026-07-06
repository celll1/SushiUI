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

import torch

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
