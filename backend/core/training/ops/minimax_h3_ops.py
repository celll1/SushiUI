"""minimax_h3_ops.py — MiniMax-H3 (joint video + audio, single-stream packed DiT)
loader + encode + train-step free functions (Phase 6b).

Mirrors ``ops/ltx2_ops.py`` (the other flow-matching, ``latent_ndim=5`` video
arch) but the model underneath is a different shape and every difference below
is MEASURED, not assumed:

  * transformer: ``MiniMaxH3Transformer3DModel`` — ONE stream of 50 blocks over a
    packed ``[text | audio | video]`` sequence, not a two-tower MM-DiT. There is
    no ``isolate_modalities`` switch and no per-modality block: every LoRA target
    is shared by both modalities.
  * the released base is weight-only **FP8 with dequant inside the forward**
    (19.71 GB resident, MEASURED). It is NOT dequantized once into bf16, and
    gradient checkpointing is what makes that affordable — with it, only the 50
    block-boundary activations survive the forward and each block's dequantized
    weights are transient inside its own recompute.
  * vae: ``AutoencoderKLMiniMaxH3`` — 24ch 5-D latents, spatial /16, temporal
    /4-ish (``latent_frames(T) = ceil(T/17)*5 - 3``), fp16 weights, a PINNED
    spatial tiling policy, and **ImageNet-normalised RGB over [0, 1]** pixels
    rather than the ``[-1, 1]`` every other VAE in this repo takes.
  * a SECOND autoencoder for audio (32ch, 32 kHz, 40 latents/s, mono — stereo is
    carried as two batch items and packed channel-major).
  * text encoder: Qwen3-VL-32B read at decoder layer 50, streamed one layer at a
    time from a memory-mapped 48 GiB file (``.to()`` on it detaches the mapping:
    73.08 GB RSS against 49.82 GB, MEASURED). It is never moved to the GPU.

Objective (design §10 + the Phase 0T experiment): plain flow-matching velocity
loss on BOTH modalities. H3's convention, derived from ``x0 = x_t + sigma*v``
(K0.4-verified) rather than assumed:

    x_t = (1 - sigma) * x0 + sigma * eps      =>      v = x0 - eps
    t   = 1 - sigma                            (t = 1 is clean)

Video rows ride the shift-12 sigma and audio rows the shift-3 sigma **of the same
uniform draw**, mirroring the inference dual-schedule.

Full fine-tuning is refused for this arch (design §7): see
``full_parameter_trainer._create_adapter``, the absent ``FullParameterAdapter``
class in ``adapters/minimax_h3_adapter.py``, and
``api.arch_capabilities.TRAINING_UNSUPPORTED``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# Token for the audio preprocessing chain, folded into the clip-cache key so a
# record produced by one chain is never served to another. Bump the suffix when
# ANY of the following changes: the extractor (`video_utils.extract_audio_stream`
# -> `extract_audio_window`), the sample rate, the channel count, or the latent
# normalisation. It is deliberately a version string, not a hash of this file.
MINIMAX_H3_AUDIO_PREP_VERSION = "h3-32k-stereo-v1"

# Fallback text length for the packed layout when an item carries no recorded
# token count (never expected: `encode_prompt` always returns one).
_DEFAULT_TEXT_TOKENS = 0


def minimax_h3_vae_tiling_token() -> str:
    """The clip-cache tiling token, DERIVED from the policy the loader pins.

    Not a hand-written string: it is built from
    ``core.models.minimax_h3.loader.MINIMAX_H3_VAE_TILING_POLICY``, the exact
    dict ``_build_video_vae`` passes to ``vae.enable_tiling(...)`` at load time —
    the same load the generation path uses. If that policy is ever changed, this
    token changes with it and every cached latent produced under the old policy
    becomes unaddressable, which is the required behaviour: flipping the shipped
    flags with everything else held fixed moved the latents by rel-RMS 0.355
    (384x384, K0.5) / 0.0952 (640x384, Phase 0T).
    """
    from core.models.minimax_h3.loader import MINIMAX_H3_VAE_TILING_POLICY as _P

    if not _P.get("enabled", False):
        return "off"
    return (f"tile{int(_P['tile_sample_min_height'])}x{int(_P['tile_sample_min_width'])}"
            f"_ov{int(_P['tile_sample_min_overlap_height'])}x"
            f"{int(_P['tile_sample_min_overlap_width'])}")


# ----------------------------------------------------------------------
# Loading / setup
# ----------------------------------------------------------------------

def normalize_dtypes(trainer) -> None:
    """Force ``weight_dtype`` / ``training_dtype`` to bf16, UNCONDITIONALLY.

    Not merely a correction for fp16. ``weight_dtype`` is handed straight to
    ``load_minimax_h3_from_path``, and it is the dtype the 300 weight-only FP8
    Linears DEQUANTIZE INTO inside every forward. Under fp32 the whole 50-block
    stack therefore runs in fp32 and the per-block dequantized-weight transient
    roughly doubles -- silently, with no error, producing a run that is a
    different function from the measured one (22.44 GB peak at 384x640x22; the
    larger registered cells become out-of-memory candidates).

    fp32 is not a hypothetical: it is the dtype preset a client applies to any
    architecture that is not on its bf16-native list, so this is the ORDINARY
    path for a UI-started run, not an exotic one. `train_runner`'s
    ``_is_bf16_native_base_model`` and the frontend's preset both name
    ``minimax_h3`` so the config that arrives is already right; this is the
    second line of defence, and the one that also covers a hand-written YAML.

    Every non-bf16 dtype is replaced and the original logged, so a run can never
    be quietly executed in a precision nobody measured.
    """
    training_dtype_overridden = False
    for attr in ("weight_dtype", "training_dtype"):
        was = getattr(trainer, attr, None)
        if was is not None and was != torch.bfloat16:
            print(f"{trainer.log_prefix} MiniMax-H3's block stack is bf16 (it is what the "
                  f"FP8 codes dequantize into); overriding {attr}: {was} -> bfloat16")
            setattr(trainer, attr, torch.bfloat16)
            if attr == "training_dtype":
                training_dtype_overridden = True
    trainer.dtype = trainer.weight_dtype
    if training_dtype_overridden and getattr(trainer, "use_grad_scaler", False):
        # The scaler was configured from the ORIGINAL training_dtype in
        # BaseTrainer.__init__. bf16 needs no gradient scaling, and a scaler left
        # over from an fp16 config raises "Attempting to unscale FP16 gradients".
        print(f"{trainer.log_prefix} Disabling GradScaler (MiniMax-H3 forced to bf16; "
              f"bf16 needs no gradient scaling)")
        trainer.use_grad_scaler = False
        trainer.grad_scaler = None


def load_components(trainer) -> None:
    """Load MiniMax-H3 components for training.

    Reuses the INFERENCE loader (``ModelLoader.load_minimax_h3_from_path`` ->
    ``core.models.minimax_h3.loader``), so the training base is bit-for-bit the
    model generation runs — including the FP8 Linear swap, the fp32 AdaLN
    projections, the fp16 video VAE and the pinned VAE tiling policy.
    """
    # Batch size is an ARCHITECTURAL constraint here and is knowable now, before
    # the model, the latent cache and the caption cache are built. MiniMax-H3
    # packs the caption's own rows into one attention DOCUMENT and its forward
    # takes no attention mask at all, so a batch of two captions of different
    # token counts would attend to the zero-padding of the shorter one -- and its
    # `timestep_indices` is a `(seq_len,)` tensor with no batch axis, so one
    # timestep vector has to serve every sample regardless. `train_step` keeps a
    # backstop check on the token counts; this is the one a user actually meets.
    _bs = int(trainer.config.get("batch_size", 1) or 1)
    if _bs > 1:
        raise ValueError(
            f"MiniMax-H3 training requires batch_size=1 (got {_bs}). Its packed sequence is a "
            f"single attention document that includes the caption's own rows, and its forward "
            f"accepts no attention mask, so two captions of different token counts cannot share "
            f"a batch; its per-row timestep index vector also has no batch axis, so one noise "
            f"level serves the whole batch. Use gradient_accumulation_steps to raise the "
            f"effective batch size instead -- the measured configuration space (384x640x22 "
            f"through 512x768x39) is batch 1.")

    print(f"{trainer.log_prefix} Detected MiniMax-H3 model")
    print(f"{trainer.log_prefix} Loading MiniMax-H3 components from {trainer.model_path}")

    normalize_dtypes(trainer)

    from core.model_loader import ModelLoader
    components = ModelLoader.load_minimax_h3_from_path(trainer.model_path, trainer.weight_dtype)

    trainer.minimax_h3_components = components
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.vae = components["vae"]
    trainer.audio_vae = components["audio_vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.processor = components.get("processor")
    trainer.scheduler = components["scheduler"]
    trainer.audio_scheduler = components["audio_scheduler"]

    # No U-Net, no second text encoder.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # The video VAE is fp16 BY DESIGN (loader's MINIMAX_H3_VIDEO_VAE_DTYPE;
    # measured against a full-fp32 decode at PSNR 77.74 dB with the tiling policy
    # held fixed). `move_vae_to_gpu` casts the VAE to `trainer.vae_dtype` on
    # every move, so leaving that at the run's bf16/fp32 would silently re-cast
    # the VAE away from the dtype the latents were measured under.
    trainer.vae_dtype = torch.float16
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Geometry + normalization vectors, so nothing downstream re-derives them.
    trainer.minimax_h3_latents_mean = components["latents_mean"]
    trainer.minimax_h3_latents_std = components["latents_std"]
    trainer.minimax_h3_audio_latents_mean = components["audio_latents_mean"]
    trainer.minimax_h3_audio_latents_std = components["audio_latents_std"]
    trainer.minimax_h3_pixel_mean = components["pixel_mean"]
    trainer.minimax_h3_pixel_std = components["pixel_std"]
    trainer.minimax_h3_audio_sample_rate = int(components["audio_sample_rate"])
    trainer.minimax_h3_audio_latent_rate = float(components["audio_latent_rate"])
    trainer.minimax_h3_fps = float(components["fps"])
    trainer.vae_latent_channels = int(components["latent_channels"])

    # A training process is DEQUANT-ONLY. The released DiT ships weight-only FP8
    # on 300 Linears; the W8A8 `scaled_mm` fast path is enabled by process-wide
    # env flags that `training_process.py` inherits from the backend, and grad
    # mode cannot be used as a proxy for it. Two module types, two per-instance
    # opt-outs; no-op on a checkpoint that carries neither.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    from core.models.ideogram4.vendor.int8_linear import disable_int8_mm
    n_fp8 = disable_scaled_mm(trainer.transformer, label="minimax_h3 training transformer")
    disable_int8_mm(trainer.transformer, label="minimax_h3 training transformer")
    print(f"{trainer.log_prefix} Dequant-only compute enforced on {n_fp8} FP8 Linear(s)")

    # Gradient checkpointing is LOAD-BEARING here, not an option: without it
    # autograd retains every block's DEQUANTIZED bf16 weights for the backward
    # (~750 MB x 50). With it they are transient inside each block's recompute,
    # which is what makes the 22.45 GB measured peak possible at all.
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} WARNING: gradient checkpointing is DISABLED by config. "
              f"MiniMax-H3 dequantizes its FP8 weights inside the forward, so without "
              f"checkpointing the backward retains ~750 MB of dequantized weights per block "
              f"across 50 blocks. Expect an out-of-memory failure.")
    else:
        trainer.transformer.enable_gradient_checkpointing()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for the MiniMax-H3 DiT")

    # Freeze everything; the LoRA adapter adds the only trainable parameters.
    trainer.transformer.requires_grad_(False)
    trainer.vae.requires_grad_(False)
    trainer.audio_vae.requires_grad_(False)
    if trainer.text_encoder is not None:
        trainer.text_encoder.requires_grad_(False)

    fp8_base_dtype = trainer.config.get("fp8_base_dtype") or None
    if fp8_base_dtype:
        print(f"{trainer.log_prefix} NOTE: fp8_base_dtype={fp8_base_dtype} is ignored for "
              f"MiniMax-H3 — the released base is ALREADY weight-only FP8 and is kept "
              f"resident in its e4m3 codes (dequant inside the forward).")

    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")
    print(f"{trainer.log_prefix} Moving MiniMax-H3 DiT to {trainer.device}")
    trainer.transformer.to(trainer.device)

    print(f"{trainer.log_prefix} MiniMax-H3 model loaded successfully "
          f"(latent_channels={trainer.vae_latent_channels}, fps={trainer.minimax_h3_fps}, "
          f"audio {trainer.minimax_h3_audio_sample_rate} Hz / "
          f"{trainer.minimax_h3_audio_latent_rate} latents/s)")


def setup_block_swap(trainer) -> None:
    """Initialise the ``LayerOffloadConductor`` over the 50 DiT blocks, AFTER the
    LoRA wrap (same ordering contract as ltx2/anima).

    NOT required for this arch at the registered config space: Phase 0T measured
    22.45 GB peak at 384x640x22 and 25.63 GB at the largest K6 cell
    (512x768x39), both without any swapping. It is wired because the knob exists
    and a user may want the headroom, not because a cell needs it.
    """
    if not getattr(trainer, "is_minimax_h3", False):
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "transformer_blocks"):
        raise ValueError("MiniMax-H3 DiT must expose `.transformer_blocks` for block swap")

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
    print(f"{trainer.log_prefix} [block-swap] hooks registered for MiniMax-H3")


def setup_attention_backend(trainer, backend: str):
    """Stamp the conduit backend on the transformer.

    The vendored forward calls ``_stamp_attention_backend()``, which fans
    ``self._attn_backend`` out to every ``MiniMaxH3Attention`` (blocks AND the
    token refiner) and derives the mode from the autograd state, so the conduit
    refuses the inference-only backends during a training forward by itself.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        trainer.transformer._attn_backend = b
        print(f"{trainer.log_prefix} [OK] MiniMax-H3 attention backend '{b}' stamped on transformer")
    except Exception as e:  # noqa: BLE001
        print(f"{trainer.log_prefix} WARNING: Failed to set MiniMax-H3 attention backend '{b}': {e}")


# ----------------------------------------------------------------------
# Text encoding (cache phase)
# ----------------------------------------------------------------------

def encode_prompt(trainer, prompt: str):
    """Encode a caption for MiniMax-H3.

    Returns ``(hidden [S, 5120] cpu bf16, {"num_text_tokens": tensor([S])})``.

    H3 has NO per-modality text connector and no attention mask in its forward
    (the packed sequence is one attention document), so the whole per-caption
    payload is the hidden state itself. The aux dict exists for exactly one
    reason: the packed layout's row count depends on the caption's token count,
    and after the batch assembly zero-pads embeddings to the batch maximum that
    count is no longer recoverable from the tensor. ``train_step`` uses it to
    REFUSE a mixed-length batch rather than silently attend to padding rows that
    no mask can exclude.

    The text encoder stays CPU/memory-mapped; ``h3_pipeline_ops.encode_prompt``
    streams one decoder layer at a time to the GPU.
    """
    from core.models.minimax_h3.h3_pipeline_ops import encode_prompt as _h3_encode

    hidden, num_tokens = _h3_encode(
        trainer.text_encoder, trainer.tokenizer, prompt,
        device=trainer.device, dtype=torch.bfloat16,
    )  # hidden: [1, S, 5120] on CPU
    return hidden[0].detach(), {"num_text_tokens": torch.tensor([int(num_tokens)], dtype=torch.long)}


def collate_aux(trainer, aux_list):
    """Collate the per-item MiniMax-H3 aux dicts into one dict of batched tensors.

    Only ``num_text_tokens`` travels here (see ``encode_prompt``). The window's
    AUDIO latent is deliberately NOT part of the per-caption aux — it depends on
    the sampled clip window, not on the caption — and is injected by
    ``base_trainer`` from the batch ITEMS, exactly as ltx2's per-clip ``fps`` is.
    """
    counts: List[int] = []
    for idx, aux in enumerate(aux_list or []):
        if isinstance(aux, dict) and isinstance(aux.get("num_text_tokens"), torch.Tensor):
            counts.append(int(aux["num_text_tokens"].reshape(-1)[0]))
        else:
            counts.append(_DEFAULT_TEXT_TOKENS)
    if not counts:
        return {}
    return {"num_text_tokens": torch.tensor(counts, dtype=torch.long)}


# ----------------------------------------------------------------------
# VAE encode
# ----------------------------------------------------------------------

def _normalize_video_latents(trainer, latents_5d: torch.Tensor) -> torch.Tensor:
    """``(z - mean) / std`` with the 24 fp32 per-channel vectors from the config
    (NOT the fp16 copies inside the weight file)."""
    mean = torch.tensor(list(trainer.minimax_h3_latents_mean),
                        dtype=torch.float32, device=latents_5d.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(list(trainer.minimax_h3_latents_std),
                       dtype=torch.float32, device=latents_5d.device).view(1, -1, 1, 1, 1)
    return (latents_5d.float() - mean) / std


def _normalize_audio_latents(trainer, latents_3d: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(list(trainer.minimax_h3_audio_latents_mean),
                        dtype=torch.float32, device=latents_3d.device).view(1, -1, 1)
    std = torch.tensor(list(trainer.minimax_h3_audio_latents_std),
                       dtype=torch.float32, device=latents_3d.device).view(1, -1, 1)
    return (latents_3d.float() - mean) / std


def vae_encode_clip(trainer, clip: torch.Tensor) -> torch.Tensor:
    """``[T, C, H, W]`` pixel clip (RGB, ``[-1, 1]``) -> ``[1, 24, T_lat, H/16, W/16]``.

    Two conventions, both measured (K0.5 encode was bitwise against MiniMax's own
    reference VAE):

    * the shared loader hands over ``[-1, 1]``; this VAE wants **ImageNet-
      normalised RGB over a [0, 1] base**, so the clip is remapped here rather
      than by changing the loader contract every other arch depends on;
    * the posterior is read at its MODE, not sampled. A cached training latent
      must be reproducible: rebuilding the same window has to give a bitwise
      identical record, or the cache key stops meaning what it says. (The
      generation path's keyframe conditioning samples instead, under its own
      fixed seed — a different requirement, documented there.)

    Tiling is NOT configured here: the loader pins it at load time and
    ``ArchHandler.clip_vae_tiling_policy`` carries that same policy into the
    cache key, so the cache and generation can never disagree about it.
    """
    vae = trainer.vae
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    pix_mean = torch.tensor(list(trainer.minimax_h3_pixel_mean),
                            dtype=torch.float32).view(1, -1, 1, 1)
    pix_std = torch.tensor(list(trainer.minimax_h3_pixel_std),
                           dtype=torch.float32).view(1, -1, 1, 1)
    x = ((clip.float() + 1.0) / 2.0 - pix_mean) / pix_std      # [T, 3, H, W]
    x = x.permute(1, 0, 2, 3).unsqueeze(0)                     # [1, 3, T, H, W]
    with torch.no_grad():
        z = vae.encode(x.to(device=vae_device, dtype=vae_dtype)).latent_dist.mode()
        latents = _normalize_video_latents(trainer, z)
    del x
    return latents


def vae_encode_audio_window(trainer, video_path: str, start_time: float,
                            duration: float) -> Optional[torch.Tensor]:
    """The window's AUDIO latent, cut by the SAME timestamps as its frames.

    ``[start_time, start_time + duration)`` of the source's audio track is
    trimmed with ``video_utils.extract_audio_window`` (the shipped helper the
    outpaint path already uses), resampled to 32 kHz stereo, and encoded by the
    MONO audio VAE with stereo carried as two BATCH items. The result is packed
    CHANNEL-MAJOR into ``[2 * T_aud, 32]`` — the row order the packed sequence
    uses, which set-equality alone cannot verify (K0.3's fourth mutant).

    ``src_dur == target_dur``: the frames are resampled by nearest SOURCE
    timestamp, so the window occupies the same real time on both sides and no
    time-stretch is wanted.

    Returns ``None`` when the video has no audio track (or extraction fails) —
    an explicit "silent window", which ``train_step`` handles by feeding noise
    audio rows and excluding that sample from the audio loss.
    """
    from utils.video_utils import extract_audio_window

    sr = int(getattr(trainer, "minimax_h3_audio_sample_rate", 32000))
    wav_bytes = _cached_audio_stream(trainer, video_path)
    if wav_bytes is None:
        return None
    window = extract_audio_window(
        wav_bytes, float(start_time), float(duration), float(duration),
        sample_rate=sr, channels=2,
    )
    if window is None:
        return None

    import numpy as np
    wav = torch.from_numpy(np.ascontiguousarray(window)).float()  # [2, n]
    audio_vae = trainer.audio_vae
    a_device = next(audio_vae.parameters()).device
    a_dtype = next(audio_vae.parameters()).dtype
    with torch.no_grad():
        # The autoencoder is MONO: [ch, 1, n] runs the two channels as a batch.
        lat = audio_vae.encode(wav.unsqueeze(1).to(a_device, a_dtype)).latent_dist.mode()
        lat = _normalize_audio_latents(trainer, lat)          # [ch, 32, T_aud]
    ch, c, t = lat.shape
    packed = lat.permute(0, 2, 1).reshape(ch * t, c)          # channel-major rows
    return packed.contiguous().cpu()


def _cached_audio_stream(trainer, video_path: str) -> Optional[bytes]:
    """WAV bytes of ``video_path``'s audio track, with a ONE-ENTRY cache.

    Every window of the same video needs the same track, and the extractor reads
    the whole container; a single-slot cache turns a per-window re-extract into a
    per-video one while keeping the memory bounded (one track, dropped as soon as
    the encode loop moves to another file).
    """
    from utils.video_utils import extract_audio_stream

    slot = getattr(trainer, "_minimax_h3_audio_slot", None)
    if slot is not None and slot[0] == video_path:
        return slot[1]
    try:
        with open(video_path, "rb") as fh:
            video_bytes = fh.read()
        wav_bytes = extract_audio_stream(video_bytes)
        del video_bytes
    except Exception as e:  # noqa: BLE001
        print(f"[MiniMaxH3] WARNING: audio extraction failed for "
              f"{os.path.basename(str(video_path))}: {e}")
        wav_bytes = None
    trainer._minimax_h3_audio_slot = (video_path, wav_bytes)
    return wav_bytes


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Still-image encode — PERMANENTLY out of scope for this arch (design §10).

    Not a deferral: H3's video VAE cannot DECODE fewer than 22 frames, so a
    still-trained adapter could never be validated by sampling; the T=1 latent
    path exists solely for image CONDITIONING; and there is no documented
    training formulation for stills upstream. Datasets for this arch are video
    clips of >= 22 frames.
    """
    raise NotImplementedError(
        "MiniMax-H3 does not train on still images: its video VAE cannot decode "
        "fewer than 22 frames, so a still-trained adapter cannot be validated by "
        "sampling, and the T=1 latent path is image CONDITIONING only. Use video "
        "clips of at least 22 frames (valid lengths are 17*n + 5).")


# ----------------------------------------------------------------------
# Training step
# ----------------------------------------------------------------------

def _shift_sigma(u: float, shift: float) -> float:
    """The flow schedule's sigma shift: ``shift*u / (1 + (shift-1)*u)``."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def _warn_if_shifted_sampler(trainer) -> None:
    """Announce, once per run, that a non-uniform timestep distribution composes
    with MiniMax-H3's own sigma shifts rather than replacing them."""
    if getattr(trainer, "_warned_h3_timestep_composition", False):
        return
    sampler = getattr(trainer, "timestep_sampler", None)
    if sampler is None or type(sampler).__name__ == "UniformTimestepSampler":
        return
    trainer._warned_h3_timestep_composition = True
    print(f"{trainer.log_prefix} NOTE: timestep_sampling is "
          f"{type(sampler).__name__}, and for MiniMax-H3 the sampler's output is the "
          f"PRE-SHIFT draw u, not sigma: train_step applies the model's own shift 12 "
          f"(video) / shift 3 (audio) on top of it, so the two COMPOSE and the "
          f"resulting sigma distribution is pushed further toward 1 than the "
          f"distribution you configured. Uniform (the registered per-arch default) "
          f"is what reproduces the schedule this model is sampled at.")


def _layout_for(trainer, num_text_tokens: int, t_lat: int, lat_h: int, lat_w: int,
                n_aud: int) -> Dict[str, Any]:
    """Cached ``build_packed_layout`` for one geometry.

    The layout is a pure function of the geometry, and the geometry is constant
    within a bucket, so it is built once per distinct tuple instead of once per
    step. It is the GENERATION path's own builder
    (``h3_pipeline_ops.build_packed_layout``, K0.3-verified against an
    independent ComfyUI port) — training must not grow a second assembly.
    """
    from core.models.minimax_h3.h3_pipeline_ops import build_packed_layout

    key = (int(num_text_tokens), int(t_lat), int(lat_h), int(lat_w), int(n_aud),
           str(trainer.device))
    cache = getattr(trainer, "_minimax_h3_layout_cache", None)
    if cache is None:
        cache = {}
        trainer._minimax_h3_layout_cache = cache
    if key not in cache:
        if len(cache) > 8:
            cache.clear()
        cache[key] = build_packed_layout(
            int(num_text_tokens), int(t_lat), int(lat_h), int(lat_w), int(n_aud),
            device=trainer.device,
        )
    return cache[key]


def train_step(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    h3_aux: Dict[str, torch.Tensor],
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """One MiniMax-H3 training step over the packed ``[text | audio | video]``
    sequence.

    Args:
        latents: normalized video latents ``[B, 24, T_lat, H', W']``.
        prompt_embeds: Qwen3-VL layer-50 hidden states ``[B, S, 5120]``.
        h3_aux: ``{"num_text_tokens": [B], "audio_latents": [B, 2*T_aud, 32] or
            None, "audio_present": [B] bool}``.

    Returns ``(loss, pred_loss_value, recon_loss_value)``.

    TWO batch-shape invariants, both structural rather than stylistic:

    * every sample in the batch must have the SAME caption token count. The
      packed sequence has no attention mask at all (one attention document), so
      a zero-padded text row is not ignorable — it is a real row the model
      attends to. Refused loudly instead of silently degrading;
    * one timestep vector serves the whole batch. ``timestep_indices`` is a
      ``(seq_len,)`` tensor with no batch axis (the transformer's own signature),
      so per-sample sigmas cannot be expressed. One uniform ``u`` is drawn per
      step and both modality schedules are derived from it — which is also
      exactly the inference dual-schedule.
    """
    from core.models.minimax_h3.h3_pipeline_ops import (
        AUDIO_CHANNELS, SHIFT_AUDIO, SHIFT_VIDEO, build_row_timesteps,
        patchify_video_latents,
    )
    from core.training.base_trainer import print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_minimax_h3] Start")

    if latents.dim() != 5:
        raise ValueError(
            f"[train_step_minimax_h3] expected 5D latents [B, C, T, H, W], got "
            f"{latents.dim()}D {tuple(latents.shape)}")

    device = trainer.device
    latents = latents.to(device=device, dtype=torch.float32, non_blocking=True)
    batch_size, _c, t_lat, lat_h, lat_w = latents.shape

    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    prompt_embeds = prompt_embeds.to(device=device, dtype=trainer.training_dtype,
                                     non_blocking=True)

    counts = h3_aux.get("num_text_tokens") if isinstance(h3_aux, dict) else None
    # BACKSTOP. The batch-size constraint is refused at config time in
    # `load_components`; this catches a caller that built the trainer another way
    # (a direct construction, a test) before the padding can silently become
    # conditioning.
    if isinstance(counts, torch.Tensor) and counts.numel() > 0:
        uniq = torch.unique(counts.reshape(-1))
        if uniq.numel() != 1:
            raise ValueError(
                f"[train_step_minimax_h3] this batch mixes caption lengths "
                f"{sorted(int(v) for v in uniq)}. MiniMax-H3 packs the caption's own "
                f"rows into the attended sequence and its forward takes NO attention "
                f"mask, so padded text rows cannot be excluded. Use train_batch_size=1 "
                f"for this architecture, or captions that tokenize to one length.")
        num_text_tokens = int(uniq[0])
        if num_text_tokens != prompt_embeds.shape[1]:
            # Padding happened even though the counts agree -> the assembly padded
            # to a different max. Trust the recorded count and trim.
            prompt_embeds = prompt_embeds[:, :num_text_tokens]
    else:
        num_text_tokens = int(prompt_embeds.shape[1])

    audio = h3_aux.get("audio_latents") if isinstance(h3_aux, dict) else None
    present = h3_aux.get("audio_present") if isinstance(h3_aux, dict) else None
    if isinstance(audio, torch.Tensor):
        audio = audio.to(device=device, dtype=torch.float32, non_blocking=True)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        n_aud_rows = int(audio.shape[1])
    else:
        # No audio anywhere in the batch: the audio rows still have to exist (the
        # packed sequence and both output heads are unconditional structure), so
        # they are filled with noise and excluded from the loss.
        from core.models.minimax_h3.h3_pipeline_ops import audio_latent_frames
        num_pixel_frames = _pixel_frames_for(trainer, t_lat)
        n_aud_rows = audio_latent_frames(
            num_pixel_frames, fps=float(getattr(trainer, "minimax_h3_fps", 24.0)),
            latents_per_second=float(getattr(trainer, "minimax_h3_audio_latent_rate", 40.0)),
        ) * AUDIO_CHANNELS
        audio = None
    n_aud = n_aud_rows // AUDIO_CHANNELS

    if isinstance(present, torch.Tensor):
        audio_mask = present.to(device=device).reshape(-1).float()
    else:
        audio_mask = torch.full((batch_size,), 1.0 if audio is not None else 0.0,
                                device=device)

    # --- sigma: ONE draw, BOTH schedules (K0.4's inference pair) ---
    #
    # What the sampler's output MEANS is arch-specific and the two video archs
    # read it differently on purpose. LTX-2.3 consumes it directly AS sigma;
    # MiniMax-H3 consumes it as the PRE-SHIFT uniform `u` and then applies its
    # own two shifts, because that is what its scheduler does at inference
    # (`linspace(1, 0, N)` -> shift 12 for video, shift 3 for audio). Uniform `u`
    # therefore reproduces the sigma distribution the released model is actually
    # sampled at, which is why `TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH` registers
    # uniform for this arch.
    #
    # A non-uniform configured distribution COMPOSES with those shifts rather
    # than replacing them (a logit-normal biased toward 1 becomes far more biased
    # after shift 12). That is a legitimate thing to want and is not blocked --
    # but it must not happen silently, so it is announced once per run.
    if timesteps is not None and torch.is_tensor(timesteps) and timesteps.numel() > 0:
        u = float(timesteps.reshape(-1)[0].item())
    elif getattr(trainer, "timestep_sampler", None) is not None:
        _warn_if_shifted_sampler(trainer)
        u = float(trainer.timestep_sampler.sample(1, device).reshape(-1)[0].item())
    else:
        u = float(torch.rand((), device=device).item())
    u = min(max(u, 0.0), 1.0)
    sigma_v = _shift_sigma(u, SHIFT_VIDEO)
    sigma_a = _shift_sigma(u, SHIFT_AUDIO)

    # --- forward process + velocity targets: v = x0 - eps, t = 1 - sigma ---
    eps_v = torch.randn_like(latents)
    x_t_v = (1.0 - sigma_v) * latents + sigma_v * eps_v
    target_v = patchify_video_latents(latents - eps_v)

    if audio is not None:
        x0_a = audio
        if float(audio_mask.min()) == 0.0:
            # A sample with no audio track contributes zero ROWS to the stacked
            # tensor; feeding the model a noised block of zeros would be a
            # meaningless "silence" it was never trained on. Give those samples
            # pure noise instead -- the rows are structural, and the loss mask
            # already excludes them.
            x0_a = torch.where(audio_mask.view(-1, 1, 1) > 0, x0_a,
                               torch.randn_like(x0_a))
    else:
        aud_ch = int((getattr(trainer, "minimax_h3_components", None) or {}).get(
            "audio_latent_channels", 32))
        x0_a = torch.randn(batch_size, n_aud_rows, aud_ch,
                           device=device, dtype=torch.float32)
    eps_a = torch.randn_like(x0_a)
    x_t_a = (1.0 - sigma_a) * x0_a + sigma_a * eps_a
    target_a = x0_a - eps_a

    layout = _layout_for(trainer, num_text_tokens, t_lat, lat_h, lat_w, n_aud)
    unique_timesteps, timestep_indices = build_row_timesteps(
        layout, 1.0 - sigma_v, 1.0 - sigma_a)

    video_rows = patchify_video_latents(x_t_v)

    if profile_vram:
        print_vram_usage("[train_step_minimax_h3] Before DiT forward")

    # NO autocast. The vendored transformer runs its own mixed-precision policy
    # (fp32 I/O heads and AdaLN projections, bf16 block stack, aligning each
    # activation with its projection's parameter dtype); an autocast context
    # would override those casts and change the function from the one generation
    # runs. The LoRA layers cast their fp32 masters to the activation dtype
    # themselves for the same reason (see MiniMaxH3LoRALinearLayer).
    video_velocity, audio_velocity = trainer.transformer(
        hidden_states=video_rows,
        # fp32: `audio_proj_in` is one of the model's `_keep_in_fp32_modules`, so
        # pre-casting the rows to bf16 would round the input and then immediately
        # upcast it again. The forward aligns every input with its own
        # projection's parameter dtype; let it.
        audio_hidden_states=x_t_a,
        encoder_hidden_states=prompt_embeds,
        timestep=unique_timesteps.to(device),
        timestep_indices=timestep_indices.to(device),
        token_tags=layout["token_tags"],
        position_ids=layout["position_ids"],
        video_indices=layout["video_indices"],
        audio_indices=layout["audio_indices"],
        text_indices=layout["text_indices"],
        return_dict=False,
    )

    if profile_vram:
        print_vram_usage("[train_step_minimax_h3] After DiT forward")

    # Per-modality velocity MSE, each averaged over tokens/channels/samples
    # BEFORE weighting (design §10 fixes this reduction so `audio_loss_weight`'s
    # meaning does not depend on the row counts, which differ by ~20x).
    video_loss = F.mse_loss(video_velocity.float(), target_v.float())

    audio_weight = float(getattr(trainer, "audio_loss_weight", 1.0))
    if audio_mask.sum() > 0 and audio_weight > 0.0:
        per_sample = ((audio_velocity.float() - target_a.float()) ** 2).mean(dim=(1, 2))
        audio_loss = (per_sample * audio_mask).sum() / audio_mask.sum()
    else:
        # Keeps the audio head in the graph with a zero contribution, so a
        # silent dataset (or audio_loss_weight=0) cannot change the video path.
        audio_loss = audio_velocity.float().sum() * 0.0

    loss = video_loss + audio_weight * audio_loss

    recon_loss_value = 0.0
    if getattr(trainer, "reconstruction_loss_weight", 0.0) > 0 and not getattr(
            trainer, "_warned_h3_recon_loss", False):
        print(f"{trainer.log_prefix} NOTE: reconstruction_loss_weight is not applied for "
              f"MiniMax-H3. Its training formulation is undocumented upstream and this "
              f"integration ships the plain flow-matching velocity loss rather than "
              f"inventing a weighting (design §10).")
        trainer._warned_h3_recon_loss = True

    trainer._minimax_h3_last_components = (float(video_loss.detach()),
                                           float(audio_loss.detach()),
                                           sigma_v, sigma_a)
    pred_loss_value = float(loss.detach())

    del eps_v, eps_a, x_t_v, x_t_a, video_rows, target_v, target_a
    return loss, pred_loss_value, recon_loss_value


def _pixel_frames_for(trainer, t_lat: int) -> int:
    """Invert ``latent_frames(T) = ceil(T/17)*5 - 3`` ON the grid ``T = 17n + 5``.

    On the grid, ``ceil(T/17) = n + 1`` so ``T_lat = 5n + 2`` and the inverse is
    ``n = (T_lat - 2)/5``, ``T = 17n + 5`` (7 -> 22, 12 -> 39). Only ever used
    for the audio-row count of a batch with NO audio at all, so it needs the grid
    point rather than a general inverse.
    """
    n = max(1, int(round((int(t_lat) - 2) / 5)))
    return 17 * n + 5


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def generate_sample(trainer, prompt: str, height: int = 384, width: int = 640,
                    num_inference_steps: int = 20, guidance_scale: float = 1.0,
                    seed: int = -1, negative_prompt: str = ""):
    """Validation sampling during training — NOT implemented for MiniMax-H3.

    Stated as a limitation rather than attempted: a sample needs the 48 GiB
    Qwen3-VL conditioner resident (49.82 GB peak RSS, 13.5 s per prompt,
    MEASURED) at the same time as the training stack, and the video VAE cannot
    decode fewer than 22 frames, so the cheapest possible preview is a full short
    generation. Returning ``None`` keeps the trainer's contract (a sampling
    failure never aborts a run); the LoRA is validated by saving it and
    generating through the normal generation path instead.
    """
    print(f"{trainer.log_prefix} MiniMax-H3 sample generation during training is not "
          f"implemented (the conditioner alone needs ~50 GB RAM resident and the video "
          f"VAE cannot decode under 22 frames); skipping sample.")
    return None
