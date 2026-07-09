"""ltx2_ops.py — LTX-2.3 (joint audio+video MM-DiT) loader + encode + train-step
+ sample free functions (plan P5).

Mirrors ``ops/anima_ops.py`` (the flow-matching, latent_ndim=5 reference). LTX-2.3
is a rectified-flow (velocity-prediction) video DiT:

  * transformer: ``LTX2VideoTransformer3DModel`` (19B, joint audio+video MM-DiT)
  * vae:         ``AutoencoderKLLTX2Video`` (128ch latents, spatial /32, temporal /8)
  * text_encoder: Gemma-3 (12B) + ``LTX2TextConnectors`` (post-connector 3840-dim)
  * scheduler:   ``FlowMatchEulerDiscreteScheduler``

The forward REQUIRES the audio branch (positional args cannot be omitted). During
video-only LoRA training we pass a no-grad dummy-noise audio tensor and set
``isolate_modalities=True`` so the a2v / v2a cross-attention is OFF and the audio
branch runs grad-free (frozen params + no-grad dummy). The audio prediction is
discarded. See ltx2_video_dataset_spec.md §"P5 — FABLE-DECIDED training design".

Construction-order note (same as anima_ops): the arch handler binds at the END of
``BaseTrainer.__init__`` (AFTER ``_load_model_components``), so the load-time
dispatcher calls ``load_components`` directly here; ``setup_block_swap`` /
``setup_attention_backend`` keep 2-line delegators on the trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# Fallback for audio_latents_per_second when it cannot be resolved from the
# loaded pipeline config (UNVERIFIED item #2). Derived from the LTX-2.3 default
# audio stack: 48000 Hz / 320 hop / 8x audio-VAE temporal compression = 18.75.
_DEFAULT_AUDIO_LATENTS_PER_SECOND = 18.75
_DEFAULT_AUDIO_IN_CHANNELS = 128
_DEFAULT_FPS = 24.0


def _resolve_audio_latents_per_second(pipeline, transformer) -> float:
    """Read audio_latents_per_second from the loaded LTX-2.3 stack.

    UNVERIFIED item #2: prefer the pipeline's own derivation
    (audio_sampling_rate / audio_hop_length / audio_vae_temporal_compression),
    then the transformer's audio-rope attribute, then a sane constant.
    """
    # 1. Pipeline attributes (populated in LTX2Pipeline.__init__).
    try:
        sr = getattr(pipeline, "audio_sampling_rate", None)
        hop = getattr(pipeline, "audio_hop_length", None)
        comp = getattr(pipeline, "audio_vae_temporal_compression_ratio", None)
        if sr and hop and comp:
            return float(sr) / float(hop) / float(comp)
    except Exception:
        pass
    # 2. Transformer audio-rope buffer (set in the LTX2 rope module).
    for attr in ("audio_rope", "audio_rope_embed"):
        rope = getattr(transformer, attr, None)
        v = getattr(rope, "audio_latents_per_second", None) if rope is not None else None
        if v:
            try:
                return float(v)
            except Exception:
                pass
    return _DEFAULT_AUDIO_LATENTS_PER_SECOND


# ----------------------------------------------------------------------
# Loading / setup
# ----------------------------------------------------------------------

def load_components(trainer) -> None:
    """Load LTX-2.3 model components for training.

    Reuses the inference loader (``ModelLoader.load_ltx2_from_path`` ->
    ``load_ltx2_from_diffusers``), which returns the assembled ``LTX2Pipeline``
    plus every component ref. We stash the pipeline (needed by encode_prompt +
    sampling), the transformer / vae / text_encoder / connectors, and the
    audio-conditioning constants (``audio_latents_per_second`` /
    ``audio_in_channels``) resolved from the loaded config.
    """
    print(f"{trainer.log_prefix} Detected LTX-2.3 model")
    print(f"{trainer.log_prefix} Loading LTX-2.3 components from {trainer.model_path}")

    # LTX-2.3 is a ~19B bf16-native DiT: fp16 (Half) OVERFLOWS to NaN in the
    # forward (NaN loss + NaN grad_norm straight out of the model, not a scaler
    # event). Enforce bf16 for this arch across the weight / training / vae dtypes
    # so the components load in bf16 AND train_step's autocast + tensor casts stay
    # consistent. Only overrides float16; leaves bf16 / fp32 configs untouched and
    # does NOT affect any other architecture's dtype handling.
    _training_dtype_overridden = False
    for _attr in ("weight_dtype", "training_dtype", "vae_dtype"):
        if getattr(trainer, _attr, None) == torch.float16:
            print(f"{trainer.log_prefix} LTX-2.3 is bf16-native; overriding "
                  f"{_attr}: float16 -> bfloat16 (fp16 overflows to NaN)")
            setattr(trainer, _attr, torch.bfloat16)
            if _attr == "training_dtype":
                _training_dtype_overridden = True
    # keep dtype-derived legacy alias consistent (self.dtype = weight_dtype).
    trainer.dtype = trainer.weight_dtype
    # The GradScaler was configured from the ORIGINAL training_dtype in __init__
    # (before this loader ran). bf16 needs no gradient scaling — and a scaler left
    # over from an fp16 config would raise "Attempting to unscale FP16 gradients"
    # / mis-scale bf16 grads. Disable it now that we are bf16.
    if _training_dtype_overridden and getattr(trainer, "use_grad_scaler", False):
        print(f"{trainer.log_prefix} Disabling GradScaler (LTX-2.3 forced to bf16; "
              f"bf16 needs no gradient scaling)")
        trainer.use_grad_scaler = False
        trainer.grad_scaler = None

    from core.model_loader import ModelLoader
    components = ModelLoader.load_ltx2_from_path(trainer.model_path, trainer.weight_dtype)

    pipeline = components["pipeline"]
    trainer.ltx2_pipeline = pipeline
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer  # No wrapper for LTX2.
    trainer.vae = components["vae"]
    trainer.audio_vae = components.get("audio_vae")
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.connectors = components["connectors"]
    trainer.vocoder = components.get("vocoder")
    trainer.scheduler = components["scheduler"]

    # LTX2 specifics: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Audio-conditioning constants (UNVERIFIED item #2 handled defensively).
    trainer.ltx2_audio_in_channels = int(
        getattr(getattr(trainer.transformer, "config", None), "audio_in_channels",
                _DEFAULT_AUDIO_IN_CHANNELS)
    )
    trainer.ltx2_audio_latents_per_second = _resolve_audio_latents_per_second(
        pipeline, trainer.transformer
    )
    # UNVERIFIED item #4: use_cross_timestep is inferred True (LTX-2.3 behavior;
    # the LTX2Pipeline denoise loop passes the model's own use_cross_timestep).
    trainer.ltx2_use_cross_timestep = bool(
        getattr(getattr(trainer.transformer, "config", None), "use_cross_timestep", True)
    )
    print(f"{trainer.log_prefix} LTX-2.3 audio conditioning: "
          f"audio_in_channels={trainer.ltx2_audio_in_channels}, "
          f"audio_latents_per_second={trainer.ltx2_audio_latents_per_second:.4f}, "
          f"use_cross_timestep={trainer.ltx2_use_cross_timestep}")

    # Cast VAE to the desired dtype.
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing (UNVERIFIED item #3: guard on presence). The
    # diffusers LTX2 transformer exposes the standard no-arg toggle and wraps the
    # transformer_blocks loop via _gradient_checkpointing_func.
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (LTX-2.3)")
    elif hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        try:
            trainer.transformer.enable_gradient_checkpointing()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for LTX-2.3 DiT")
        except Exception as e:  # noqa: BLE001
            print(f"{trainer.log_prefix} WARNING: enable_gradient_checkpointing failed "
                  f"for LTX-2.3 ({e}); continuing without it")

    # Freeze all base weights. Trainable LoRA modules are added later by the
    # adapter via apply_lora_to_unet.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.connectors.requires_grad_(False)
    if trainer.audio_vae is not None:
        trainer.audio_vae.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Optional: FP8 the frozen base DiT before LoRA wraps anything (LoRA-only),
    # mirroring anima_ops. Reuse the same anima FP8 quantiser (arch-agnostic
    # Linear-forward patch).
    fp8_base_dtype = trainer.config.get("fp8_base_dtype") or None
    training_method = trainer.config.get("training_method", "lora")
    if fp8_base_dtype and training_method == "lora":
        print(f"{trainer.log_prefix} Quantising frozen LTX-2.3 DiT base to "
              f"{fp8_base_dtype} (LoRA-on-FP8-base)")
        from core.vram_optimization import _anima_quantize_fp8
        trainer.transformer = _anima_quantize_fp8(
            trainer.transformer, fp8_base_dtype, "LTX-2.3 DiT (training base)",
        )
        trainer.transformer_original = trainer.transformer
        trainer.transformer.requires_grad_(False)
    elif fp8_base_dtype:
        print(f"{trainer.log_prefix} WARNING: fp8_base_dtype={fp8_base_dtype} is "
              f"only supported for training_method='lora' "
              f"(current: {training_method!r}); ignoring.")

    # Plain GPU move. Block-swap init deferred to setup_block_swap() (called by
    # the mode subclass AFTER LoRA wrap) — same reasoning as anima_ops.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")
    print(f"{trainer.log_prefix} Moving LTX-2.3 DiT to {trainer.device} "
          f"(block swap, if any, will redistribute after adapter setup)")
    trainer.transformer.to(trainer.device)

    print(f"{trainer.log_prefix} LTX-2.3 model loaded successfully")
    print(f"{trainer.log_prefix} Scheduler: {trainer.scheduler.__class__.__name__}, "
          f"latent_channels=128")


def setup_block_swap(trainer) -> None:
    """Initialise the LayerOffloadConductor for the LTX-2.3 DiT, AFTER any
    structural model changes (LoRA wrapping). Copy of anima_ops.setup_block_swap
    retargeted to ``transformer.transformer_blocks``.
    """
    if not getattr(trainer, "is_ltx2", False):
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "transformer_blocks"):
        raise ValueError("LTX-2.3 DiT must expose `.transformer_blocks` (nn.ModuleList) for block swap")

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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for LTX-2.3")


def setup_attention_backend(trainer, backend: str):
    """LTX-2.3 uses the diffusers attention dispatcher (SDPA by default). No
    per-block attn-mode vocabulary to set (unlike Anima's vendored kernel), so
    this is a no-op stub that keeps the arch-handler contract satisfied.
    """
    return


# ----------------------------------------------------------------------
# Text encoding (cache phase) — Gemma-3 + connectors, frozen no_grad
# ----------------------------------------------------------------------

def encode_prompt(trainer, prompt: str, max_sequence_length: int = 1024):
    """Encode a caption for LTX-2.3 (post-connector, cached detached bf16).

    Runs ``pipeline.encode_prompt(do_classifier_free_guidance=False)`` (Gemma-3)
    then the ``LTX2TextConnectors`` to obtain the video + audio text embeddings
    and the (left-padded) attention mask that the transformer forward consumes.
    All frozen / no_grad; batch dim dropped so caches accumulate per-sample.

    Returns ``(video_text_embedding, aux_dict)`` where aux_dict carries the
    audio text embedding + mask + fps for train_step (mirrors anima's payload).
    """
    pipeline = trainer.ltx2_pipeline
    device = trainer.device

    # The Gemma-3 text encoder + connectors are frozen; force eval() so the
    # transformers Gemma3 forward does not demand token_type_ids (it requires
    # them only when module.training is True). The surrounding trainer may have
    # called .train() on the model graph.
    if getattr(pipeline, "text_encoder", None) is not None:
        pipeline.text_encoder.eval()
    # Derive the encode dtype from the ACTUAL (frozen) connectors weights rather
    # than hardcoding one — the training loader may stage LTX components in fp16
    # or bf16 per the run's weight_dtype, and the embeds fed to the connectors'
    # Linear layers must match their weight dtype (and device).
    enc_dtype = torch.bfloat16
    if getattr(pipeline, "connectors", None) is not None:
        pipeline.connectors.eval()
        # encode_prompt places the Gemma-3 text encoder on `device` and returns
        # cuda embeds, but the connectors module is staged separately — move it
        # to the same device so its Linear layers match the activation device.
        pipeline.connectors.to(device)
        try:
            enc_dtype = next(pipeline.connectors.parameters()).dtype
        except StopIteration:
            pass

    with torch.no_grad():
        prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=enc_dtype,
        )
        padding_side = "left"
        tok = getattr(pipeline, "tokenizer", None)
        if tok is not None:
            padding_side = getattr(tok, "padding_side", "left")
        # Match the connectors' weight dtype/device exactly at the call site.
        prompt_embeds = prompt_embeds.to(device=device, dtype=enc_dtype)
        video_emb, audio_emb, mask = pipeline.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=padding_side
        )

    # Cache in bf16 (train_step re-casts to trainer.training_dtype anyway).
    dtype = torch.bfloat16
    return video_emb[0].detach().to(dtype), {
        "audio_text_embedding": audio_emb[0].detach().to(dtype),
        "mask": mask[0].detach(),
    }


def collate_aux(trainer, aux_list):
    """Collate a list of per-item LTX-2.3 aux dicts into ONE dict of batched
    tensors {audio_text_embedding [B, L, D], mask [B, L]}, padding the sequence
    dim (mirrors anima_ops.collate_aux).

    NOTE on fps: fps is a property of the VIDEO CLIP (per-item), NOT of the
    caption, so it is deliberately NOT sourced here (the per-caption text aux
    never carries it). The base_trainer batch-assembly injects a per-sample
    ``fps`` tensor ``[B]`` into the returned dict from the batch ITEMS
    (``_ltx2_batch_fps_tensor``); ``train_step`` reads that. If it is absent
    (e.g. stills-only fallback), train_step defaults to ``_DEFAULT_FPS``.
    """
    keys = ("audio_text_embedding", "mask")
    if not aux_list:
        raise ValueError("[LTX2 collation] empty auxiliary_data_list")
    for idx, aux in enumerate(aux_list):
        if not isinstance(aux, dict):
            raise ValueError(
                f"[LTX2 collation] item {idx} auxiliary data is "
                f"{type(aux).__name__}, expected a dict with keys {keys}"
            )
        for k in keys:
            if k not in aux or not isinstance(aux[k], torch.Tensor):
                raise ValueError(
                    f"[LTX2 collation] item {idx} is missing tensor key '{k}' "
                    f"(got keys {list(aux.keys())})"
                )

    pad_values = {"audio_text_embedding": 0.0, "mask": 0}

    batched = {}
    for k in keys:
        tensors = [aux[k] for aux in aux_list]
        max_len = max(t.shape[0] for t in tensors)
        if any(t.shape[0] != max_len for t in tensors):
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad_shape = (max_len - t.shape[0],) + tuple(t.shape[1:])
                    pad = torch.full(pad_shape, pad_values[k], dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            tensors = padded
        batched[k] = torch.stack(tensors, dim=0)

    # fps is injected per-sample by base_trainer from the batch ITEMS (see
    # docstring); not derived from the per-caption aux here.
    return batched


# ----------------------------------------------------------------------
# VAE encode — 5D video latents
# ----------------------------------------------------------------------

def _normalize_ltx_latents(trainer, latents_5d):
    """Apply LTX latents_mean/std + scaling_factor normalization (matches
    ``LTX2Pipeline._normalize_latents``). Input/return: [B, C, T, H', W']."""
    vae = trainer.vae
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents_5d.device, latents_5d.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents_5d.device, latents_5d.dtype)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return (latents_5d - mean) * scaling_factor / std


def vae_encode_clip(trainer, clip):
    """Encode a ``[T, C, H, W]`` pixel clip (RGB, [-1, 1]) to a normalized 5D LTX
    video latent ``[1, 128, T_lat, H', W']``.

    This is the callable P4b's ``video_loader.encode_and_cache_clip`` expects
    (``vae_encode_clip(clip) -> [1, C, T_lat, H', W']``). LTX video VAE consumes
    ``[B, C, T, H, W]`` — permute the clip and add the batch dim.
    """
    vae = trainer.vae
    vae_device = next(vae.parameters()).device
    # [T, C, H, W] -> [1, C, T, H, W]
    px = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device=vae_device, dtype=trainer.vae_dtype)
    with torch.no_grad():
        latent_dist = vae.encode(px).latent_dist
        latents_5d = latent_dist.sample()  # [1, 128, T_lat, H', W']
        latents_5d = _normalize_ltx_latents(trainer, latents_5d)
    del px
    return latents_5d


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """LTX-2.3 VAE-encode branch of ``BaseTrainer.encode_image`` (still path).

    A still image is a degenerate 1-frame clip. ``image_tensor`` arrives as
    ``[1, C, H, W]`` (staged on vae_device); add the temporal dim to make
    ``[1, C, 1, H, W]``, encode, normalize -> 5D ``[1, 128, 1, H', W']``. This
    routes stills through the SAME 5D train_step as video clips (T=1).
    """
    # [1, C, H, W] -> [1, C, T=1, H, W]
    image_tensor_5d = image_tensor.unsqueeze(2)
    latent_dist = trainer.vae.encode(image_tensor_5d).latent_dist
    latents_5d = latent_dist.sample()  # [1, 128, 1, H', W']
    latents_5d = _normalize_ltx_latents(trainer, latents_5d)
    del image_tensor_5d, latent_dist
    return latents_5d


# ----------------------------------------------------------------------
# Training step — rectified-flow velocity prediction (video branch)
# ----------------------------------------------------------------------

def _pack_latents(latents_5d):
    """[B, C, F, H, W] -> [B, F*H*W, C] token sequence (patch=1, patch_t=1),
    identical to ``LTX2Pipeline._pack_latents(x, 1, 1)``."""
    b, c, f, h, w = latents_5d.shape
    x = latents_5d.reshape(b, c, f, 1, h, 1, w, 1)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return x  # [B, F*H*W, C]


def train_step(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    ltx2_aux: Dict[str, torch.Tensor],
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Single LTX-2.3 training step (rectified flow / velocity prediction).

    Args:
        latents:   Normalised LTX video latents [B, 128, T_lat, H', W'] (5D;
                   T_lat=1 for stills). Cached, already latents_mean/std +
                   scaling_factor normalized at encode.
        prompt_embeds: post-connector video text embedding [B, L, 3840].
        ltx2_aux: {audio_text_embedding [B, L, D], mask [B, L],
                   fps [B] tensor (per-sample clip fps, injected by base_trainer
                   from the batch items; optional -> _DEFAULT_FPS)}.

    Returns:
        (loss tensor, prediction loss value, reconstruction loss value).
    """
    from core.training.base_trainer import print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_ltx2] Start")

    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    prompt_embeds = prompt_embeds.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    audio_emb = ltx2_aux["audio_text_embedding"].to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    mask = ltx2_aux["mask"].to(device=trainer.device, non_blocking=True)

    if latents.dim() != 5:
        raise ValueError(
            f"[train_step_ltx2] expected 5D latents [B, C, T, H, W], got "
            f"{latents.dim()}D {tuple(latents.shape)}"
        )

    batch_size = latents.shape[0]
    t_lat = latents.shape[2]
    lat_h = latents.shape[3]
    lat_w = latents.shape[4]

    # Per-sample clip fps. fps is a property of the VIDEO CLIP (per-item), and
    # VideoBucketManager groups by (spatial_bucket, clip_length) — NOT by fps —
    # so a batch CAN mix fps. base_trainer injects a per-sample fps tensor [B]
    # into the collated aux (sourced from the batch items, NOT the per-caption
    # text aux). Accept a [B] tensor (per-sample), a scalar tensor / python
    # float (broadcast), or None (default). Stills (T_lat==1) have a single
    # temporal position so fps is irrelevant to their RoPE coords; the default
    # keeps them well-formed without crashing.
    fps_raw = ltx2_aux.get("fps", None)
    if isinstance(fps_raw, torch.Tensor):
        fps_ps = fps_raw.to(device=trainer.device, dtype=torch.float32).reshape(-1)
        if fps_ps.numel() == 1:
            fps_ps = fps_ps.expand(batch_size).clone()
        elif fps_ps.numel() != batch_size:
            raise ValueError(
                f"[train_step_ltx2] fps tensor length {fps_ps.numel()} != batch "
                f"size {batch_size}"
            )
    elif fps_raw is None:
        fps_ps = torch.full((batch_size,), _DEFAULT_FPS, device=trainer.device, dtype=torch.float32)
    else:
        fps_ps = torch.full(
            (batch_size,), float(fps_raw) or _DEFAULT_FPS,
            device=trainer.device, dtype=torch.float32,
        )
    # Guard against non-positive fps (avoids div-by-zero in RoPE / L_audio).
    fps_ps = torch.clamp(fps_ps, min=1e-6)

    # Sigma sampling (flow-matching), same policy as anima.
    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
        else:
            timesteps = torch.rand(batch_size, device=trainer.device)
    sigma = timesteps.to(trainer.training_dtype)

    noise = torch.randn_like(latents)
    # Flow-matching forward: x_t = (1 - sigma) * x_0 + sigma * noise
    sigma_view = sigma.view(-1, 1, 1, 1, 1).to(latents.dtype)
    x_t = (1.0 - sigma_view) * latents + sigma_view * noise

    seq_xt = _pack_latents(x_t)               # [B, T*H'*W', 128]
    timestep = (sigma * 1000.0)               # timestep_scale_multiplier=1000

    # Dummy audio (video-only training): no-grad noise + isolate_modalities keeps
    # the audio branch grad-free. L_audio derived from clip duration.
    audio_in_channels = int(getattr(trainer, "ltx2_audio_in_channels", _DEFAULT_AUDIO_IN_CHANNELS))
    aud_lps = float(getattr(trainer, "ltx2_audio_latents_per_second", _DEFAULT_AUDIO_LATENTS_PER_SECOND))
    num_frames_pixel = (t_lat - 1) * 8 + 1
    # Per-sample audio length (clip duration * audio_latents_per_second). The
    # dummy audio tensor is a single [B, L_audio, C] block, so size it to the
    # MAX across the (possibly fps-mixed) batch; the audio branch is grad-free
    # and its prediction is discarded (video-only training), so over-length is
    # harmless. clamp >= 1 handles T=1 stills.
    l_audio_ps = torch.clamp((num_frames_pixel / fps_ps) * aud_lps, min=1.0)
    l_audio = int(l_audio_ps.max().round().item())
    with torch.no_grad():
        dummy_audio = torch.randn(
            batch_size, l_audio, audio_in_channels,
            device=trainer.device, dtype=trainer.training_dtype,
        )
    full_noise_t = torch.ones(batch_size, device=trainer.device, dtype=trainer.training_dtype) * 1000.0

    use_cross_timestep = bool(getattr(trainer, "ltx2_use_cross_timestep", True))

    # Build the video RoPE coords ourselves so the temporal axis is scaled by the
    # ACTUAL per-sample clip fps. The transformer's own path only accepts a scalar
    # `fps` (one value for the whole batch); since a batch can mix fps, we compute
    # coords with fps=1.0 (identity on the temporal division) and then divide the
    # temporal axis [:, 0] by the per-sample fps [B, 1, 1]. This exactly reproduces
    # the transformer's internal `pixel_coords[:, 0] / fps` for each sample. Passing
    # video_coords= makes the forward skip its internal (scalar-fps) computation.
    video_coords = trainer.transformer.rope.prepare_video_coords(
        batch_size, t_lat, lat_h, lat_w, trainer.device, fps=1.0,
    )  # [B, 3, N, 2]; axis 1 idx 0 == temporal (frames)
    video_coords[:, 0, :, :] = video_coords[:, 0, :, :] / fps_ps.view(-1, 1, 1)

    if profile_vram:
        print_vram_usage("[train_step_ltx2] Before DiT forward")

    def _forward():
        out = trainer.transformer(
            hidden_states=seq_xt,
            audio_hidden_states=dummy_audio,
            encoder_hidden_states=prompt_embeds,
            audio_encoder_hidden_states=audio_emb,
            timestep=timestep,
            sigma=timestep,
            audio_timestep=full_noise_t,
            audio_sigma=full_noise_t,
            encoder_attention_mask=mask,
            audio_encoder_attention_mask=mask,
            num_frames=t_lat,
            height=lat_h,
            width=lat_w,
            video_coords=video_coords,
            audio_num_frames=l_audio,
            isolate_modalities=True,
            use_cross_timestep=use_cross_timestep,
            return_dict=False,
        )
        return out

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            v_pred_video, _v_pred_audio = _forward()
    else:
        v_pred_video, _v_pred_audio = _forward()

    if profile_vram:
        print_vram_usage("[train_step_ltx2] After DiT forward")

    # Rectified-flow target in packed token space: v = noise - x0.
    target = _pack_latents(noise - latents)

    loss_per_element = F.mse_loss(v_pred_video.float(), target.float(), reduction="none")
    mse_loss = loss_per_element.mean()
    loss = mse_loss

    # Optional reconstruction loss (predicted x0 vs GT x0): x0 = x_t - sigma * v.
    recon_loss_value = 0.0
    if trainer.reconstruction_loss_weight > 0:
        with torch.no_grad():
            seq_x0 = _pack_latents(latents)
            seq_x_t = _pack_latents(x_t)
            sigma_seq = sigma.view(-1, 1, 1).to(v_pred_video.dtype)
            pred_x0 = seq_x_t - sigma_seq * v_pred_video
            recon_loss = F.mse_loss(pred_x0.float(), seq_x0.float())
            recon_loss_value = recon_loss.item()
        loss = loss + trainer.reconstruction_loss_weight * recon_loss

    pred_loss_value = mse_loss.item()

    # Discard audio prediction (video-only training).
    del noise, x_t, seq_xt, v_pred_video, _v_pred_audio, target, dummy_audio
    del loss_per_element
    return loss, pred_loss_value, recon_loss_value


# ----------------------------------------------------------------------
# Sample generation (reuse the inference pipeline, move-to-CPU staging)
# ----------------------------------------------------------------------

def generate_sample(
    trainer,
    prompt: str,
    height: int = 512,
    width: int = 768,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    seed: int = -1,
    negative_prompt: str = "",
):
    """Generate a short validation clip during LTX-2.3 training (returns the
    first frame as a PIL image, matching the arch-handler sample contract).

    Reuses the assembled ``LTX2Pipeline`` on the trainer's own components with
    move-to-CPU staging so it survives block-swap / low-VRAM layouts (mirrors
    anima_ops.generate_sample shape). Best-effort: any failure logs + returns
    None so a sampling error never aborts training.
    """
    import random as _random
    import numpy as _np
    from PIL import Image as _Image

    print(f"{trainer.log_prefix} Generating LTX-2.3 sample: {prompt[:50]}...")

    # Snap dims to the ÷32 spatial factor.
    snap = 32
    height = max(snap, (height // snap) * snap)
    width = max(snap, (width // snap) * snap)

    trainer.transformer.eval()
    trainer.vae.eval()
    if trainer.text_encoder is not None:
        trainer.text_encoder.eval()

    try:
        pipeline = trainer.ltx2_pipeline
        device = trainer.device
        # CPU generator: during training the LTX components are offload/block-swap
        # staged, so diffusers allocates the noise latent on CPU — a CUDA generator
        # cannot seed a CPU tensor ("Cannot generate a cpu tensor from a generator
        # of type cuda"). A CPU generator is universally compatible (randn_tensor
        # generates on CPU then moves) and reproducible across offload layouts.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed if (seed is not None and seed >= 0) else _random.randint(0, 2**32 - 1))

        # Keep the clip tiny for a validation sample (9 frames = 1 latent block).
        num_frames = 9

        with torch.no_grad():
            video, _audio = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
                output_type="np",
                return_dict=False,
            )
        frames_np = video[0]  # [T, H, W, C] float [0,1]
        first = (_np.clip(frames_np[0], 0.0, 1.0) * 255.0).round().astype("uint8")
        return _Image.fromarray(first)
    except Exception as e:  # noqa: BLE001
        print(f"{trainer.log_prefix} WARNING: LTX-2.3 sample generation failed ({e}); skipping")
        return None
    finally:
        trainer.transformer.train()
        trainer.vae.train()
        if trainer.text_encoder is not None:
            trainer.text_encoder.train()
