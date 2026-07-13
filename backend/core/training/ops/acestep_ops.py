"""ops/acestep_ops.py — ACE-Step 1.5 (turbo) loader + encode + train-step + sample
free functions (Phase 8a).

Mirrors ``ops/ltx2_ops.py`` (the flow-matching, temporal-latent reference this
task was asked to copy EXACTLY). ACE-Step is a rectified-flow (velocity
prediction) audio DiT:

  * transformer: ``AceStepConditionGenerationModel`` (2B DiT; ``.decoder`` is
    the diffusion transformer, ``.encoder`` the frozen text/lyric/timbre
    condition encoder) -- vendored at
    ``core/models/acestep/vendor/modeling_acestep_v15_turbo.py``.
  * vae:         ``AutoencoderOobleck`` (64ch TEMPORAL-ONLY latents ``[B, T, 64]``,
    48kHz stereo, 25Hz raw latent rate).
  * text_encoder: Qwen3-Embedding-0.6B (``Qwen3Model``, 1024-dim), frozen; used
    BOTH for the "# Caption" text conditioning (full forward,
    ``last_hidden_state``) and the lyric conditioning (``embed_tokens`` lookup
    only -- no transformer forward, see ``AceStepLyricEncoder``).
  * scheduler:   none -- rectified flow with a manual sigma in [0, 1]
    (0=data, 1=noise), matching the vendored model's OWN
    ``forward()``/``training_losses()`` convention (``xt = t*x1 + (1-t)*x0``,
    ``flow = x1 - x0``) exactly; we drive ``t`` from
    ``trainer.timestep_sampler`` instead of the model's internal
    ``sample_t_r`` so ACE-Step training uses the SAME timestep-sampling
    machinery every other arch does, and call ``dit.decoder(...)`` directly
    (a single forward) rather than the model's own ``forward()`` (which also
    does its own noise sampling + a training-only CFG condition-dropout we do
    not want to duplicate/second-guess here).

SCOPE DECISIONS (Phase 8a, documented per-task; see also adapters/acestep_adapter.py
and audio_loader.py):
  * Lyrics ARE sourced per-item (follow-up to Phase 8a): a dataset item may
    carry a ``lyrics`` string (populated from a ``DatasetCaption`` row with
    ``caption_type=="lyrics"``, a SEPARATE signal from whichever caption_type
    is selected as the item's primary caption -- see
    ``train_runner.get_dataset_items_fast``). Items with no lyrics caption
    default to ``""`` and are treated as instrumental / no-lyrics, matching
    the vendored model's own well-defined empty-lyrics behavior (mirrors
    ``_generate_txt2aud_acestep``'s default ``lyrics=""``) -- fully backward
    compatible with every dataset that has never had a lyrics caption added.
    ``encode_prompt`` reuses the ONE caption-INDEPENDENT empty-lyrics asset
    precomputed at load time (``_build_empty_lyrics``) as a fast path whenever
    an item's lyrics are empty; a non-empty per-item lyrics string is run
    through the SAME Qwen3 ``embed_tokens`` lookup (no transformer forward)
    on demand. The vocal language tag baked into the lyrics text block is
    hardcoded to ``"en"`` (mirrors ``_build_empty_lyrics``) -- no per-item
    language field exists in the dataset schema yet; this is a scoped
    simplification, not a correctness issue (the language tag only affects
    the model's own language-conditioning heuristic, not tokenization).
  * The "# Metas" block (BPM/Duration/Key/Time Signature) that
    ``AceStepMixin._acestep_build_text_prompt`` renders needs per-item audio
    duration/tempo metadata this text-embedding CACHE layer doesn't have (the
    cache key is the caption STRING alone, per ``LatentCache.compute_caption_hash``)
    -- baking a duration into a cached embedding shared by
    same-caption/different-duration items would be WRONG for at least one of
    them. Training therefore uses only the Instruction+Caption sections
    (Metas omitted).
  * No audio bucket-manager: batches are grouped by encoded LATENT FRAME COUNT
    (``base_trainer.py``'s ``acestep_audio_batches``), not by a fixed clip
    duration -- dataset authors should pre-trim clips to a consistent duration
    per training run so batches are non-degenerate (batch_size=1 always works).
  * ``sample()`` declines (returns None) -- training-time audio preview is not
    wired into the image-only training UI yet (mirrors the documented
    ideogram4 "cannot sample yet" contract value).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Loading / setup
# ----------------------------------------------------------------------

def load_components(trainer) -> None:
    """Load ACE-Step 1.5 model components for training.

    Reuses the inference loader (``ModelLoader.load_acestep_from_path`` ->
    ``core.models.acestep.loader.load_acestep_from_path``), which returns the
    DiT/VAE/text_encoder/tokenizer already CPU-resident. Binds them onto the
    trainer (mirrors ltx2_ops.load_components's shape), freezes every base
    weight, and precomputes the two caption-INDEPENDENT conditioning assets
    (silence-latent timbre/context, empty-lyrics embedding) once, on CPU,
    before any GPU move -- so no lazy first-train-step device juggling is
    needed later.
    """
    print(f"{trainer.log_prefix} Detected ACE-Step 1.5 model")
    print(f"{trainer.log_prefix} Loading ACE-Step components from {trainer.model_path}")

    from core.model_loader import ModelLoader
    components = ModelLoader.load_acestep_from_path(trainer.model_path, trainer.weight_dtype)

    trainer.transformer = components["dit"]
    trainer.transformer_original = trainer.transformer  # No wrapper for ACE-Step.
    trainer.acestep_dit_config = components["dit_config"]
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.acestep_sample_rate = int(components.get("sample_rate", 48000))
    trainer.acestep_latent_frame_rate = int(components.get("latent_frame_rate", 25))
    # Read directly by _build_cache_namespace (base_trainer.py) as a fallback
    # over self.vae.config.latent_channels, which AutoencoderOobleck's config
    # does not expose under that name (it uses decoder_input_channels).
    trainer.vae_latent_channels = int(components.get("latent_channels", 64))

    # ACE-Step specifics: no dual TE / no U-Net / no diffusers scheduler
    # (flow-matching sigma sampling is driven directly by trainer.timestep_sampler,
    # exactly like ltx2_ops -- see module docstring).
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.scheduler = None
    trainer.noise_scheduler = None

    # Cast VAE to the desired dtype (defensive re-cast; the loader already
    # casts to trainer.weight_dtype, this covers vae_dtype != weight_dtype).
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing: AceStepDiTLayer/AceStepEncoderLayer subclass
    # transformers' GradientCheckpointingLayer, so the standard PreTrainedModel
    # toggle applies.
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (ACE-Step)")
    elif hasattr(trainer.transformer, "gradient_checkpointing_enable"):
        try:
            trainer.transformer.gradient_checkpointing_enable()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for ACE-Step DiT")
        except Exception as e:  # noqa: BLE001
            print(f"{trainer.log_prefix} WARNING: gradient_checkpointing_enable failed "
                  f"for ACE-Step ({e}); continuing without it")

    # Freeze all base weights. Trainable LoRA modules are added later by the
    # adapter via apply_lora_to_unet (decoder.layers.*.{self_attn,cross_attn}
    # ONLY -- encoder/tokenizer/detokenizer/VAE/text_encoder stay frozen).
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Precompute the caption-independent conditioning assets (CPU, before any
    # GPU move -- see module + function docstrings for why these are safe to
    # share across every training item/caption).
    _build_silence_latent(trainer)
    _build_empty_lyrics(trainer)

    # Plain GPU move. Block-swap init deferred to setup_block_swap() (called by
    # the mode subclass AFTER LoRA wrap) — same reasoning as anima/ltx2_ops.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")
    print(f"{trainer.log_prefix} Moving ACE-Step DiT to {trainer.device} "
          f"(block swap, if any, will redistribute after adapter setup)")
    trainer.transformer.to(trainer.device)

    print(f"{trainer.log_prefix} ACE-Step model loaded successfully")
    print(f"{trainer.log_prefix} sample_rate={trainer.acestep_sample_rate}, "
          f"latent_frame_rate={trainer.acestep_latent_frame_rate}, "
          f"latent_channels={trainer.vae_latent_channels}")


def _build_silence_latent(trainer) -> None:
    """VAE-encode literal silence (30s @ 48kHz stereo) into the ``[1, 750, 64]``
    asset used as BOTH the timbre condition and the src_latents/context for
    plain text2music training (mirrors
    ``AceStepMixin._acestep_ensure_silence_latent``, but run ONCE at load time
    on whatever device the freshly-loaded VAE is on -- CPU, per
    ``load_acestep_from_path``'s "keep everything on CPU after load" policy --
    instead of lazily via a device-move-and-restore dance)."""
    from core.models.acestep.defaults import SAMPLE_RATE, SILENCE_LATENT_FRAMES

    vae = trainer.vae
    vae_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device
    duration_sec = SILENCE_LATENT_FRAMES / 25.0  # 30s
    zeros = torch.zeros(
        1, 2, int(round(duration_sec * SAMPLE_RATE)), device=vae_device, dtype=vae_dtype
    )
    with torch.no_grad():
        # .mode() (deterministic mean): silence must encode to a fixed,
        # reproducible latent, not a stochastic draw (mirrors the inference
        # mixin's own silence-latent asset).
        silence_latent = vae.encode(zeros).latent_dist.mode()  # [1, 64, 750]
    silence_latent = silence_latent.transpose(1, 2).contiguous()  # [1, 750, 64]
    trainer.acestep_silence_latent = silence_latent.detach().to("cpu")
    print(f"{trainer.log_prefix} Built ACE-Step silence-latent asset: "
          f"shape={tuple(trainer.acestep_silence_latent.shape)}")


def _build_empty_lyrics(trainer) -> None:
    """Precompute the empty-lyrics embedding (caption-independent; see module
    docstring's scope decision). ``embed_tokens`` is a plain nn.Embedding
    lookup (no transformer forward), so this is cheap even on CPU."""
    from core.pipeline_backends.acestep import AceStepMixin

    lyrics_text = AceStepMixin._acestep_format_lyrics("", "en")
    tok = trainer.tokenizer(
        lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
    )
    te_device = next(trainer.text_encoder.parameters()).device
    lyric_ids = tok.input_ids.to(te_device)
    lyric_attention_mask = tok.attention_mask.to(te_device).bool()
    with torch.no_grad():
        lyric_hidden_states = trainer.text_encoder.embed_tokens(lyric_ids)
    trainer.acestep_empty_lyric_hidden_states = lyric_hidden_states.detach().to("cpu")
    trainer.acestep_empty_lyric_attention_mask = lyric_attention_mask.detach().to("cpu")
    print(f"{trainer.log_prefix} Built ACE-Step empty-lyrics conditioning asset: "
          f"shape={tuple(trainer.acestep_empty_lyric_hidden_states.shape)}")


def setup_block_swap(trainer) -> None:
    """Initialise the LayerOffloadConductor for the ACE-Step DiT's decoder
    layers, AFTER any structural model changes (LoRA wrapping). Copy of
    ltx2_ops.setup_block_swap retargeted to ``transformer.decoder.layers``
    (ACE-Step's DiT block list; the encoder/tokenizer/detokenizer stay
    resident since they are small and frozen)."""
    if not getattr(trainer, "is_acestep", False):
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    dit = trainer.transformer
    if not hasattr(dit, "decoder") or not hasattr(dit.decoder, "layers"):
        raise ValueError("ACE-Step DiT must expose `.decoder.layers` (nn.ModuleList) for block swap")

    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    from core.memory_management import LayerOffloadConductor
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=dit.decoder.layers,
        blocks_to_swap=trainer.blocks_to_swap,
        device=trainer.device,
        use_pinned_memory=trainer.use_pinned_memory,
        cpu_buffer_size_mb=4096,
        activation_buffer_size_mb=2048,
        enable_prefetch=True,
        enable_activation_offload=False,
    )
    dit._layer_offload_conductor = trainer.layer_offload_conductor
    trainer.layer_offload_conductor.register_hooks()
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for ACE-Step")


def setup_attention_backend(trainer, backend: str):
    """ACE-Step uses the transformers attention dispatcher (SDPA/eager by
    config, see ``AceStepAttention.forward``'s ``ALL_ATTENTION_FUNCTIONS``
    lookup). No per-block attn-mode vocabulary to set, so this is a no-op stub
    that keeps the arch-handler contract satisfied (mirrors ltx2_ops)."""
    return


# ----------------------------------------------------------------------
# Text encoding (cache phase) — Qwen3-Embedding-0.6B, frozen no_grad
# ----------------------------------------------------------------------

def encode_prompt(trainer, prompt: str, lyrics: str = ""):
    """Encode a caption (+ optional per-item lyrics) for ACE-Step (Qwen3
    "# Caption" hidden states, cached detached, dtype-native). See module
    docstring for the lyrics/Metas scope decisions.

    Args:
        prompt: The item's caption text ("# Caption" block).
        lyrics: The item's lyrics text, or ``""`` for instrumental (default;
            fully backward compatible with datasets that have no lyrics
            caption). Sourced from a ``caption_type=="lyrics"``
            ``DatasetCaption`` row -- a SEPARATE per-item signal from
            ``prompt`` (see ``train_runner.get_dataset_items_fast``).

    Returns ``(text_hidden_states, aux_dict)`` where aux_dict carries the
    caption's own attention mask PLUS the per-item lyric conditioning
    (``lyric_hidden_states``, ``lyric_attention_mask``) for train_step /
    collate_aux (mirrors ltx2_ops's ``(video_emb, aux)`` contract).
    """
    from core.models.acestep.defaults import DEFAULT_DIT_INSTRUCTION

    tokenizer = trainer.tokenizer
    text_encoder = trainer.text_encoder
    device = trainer.device

    # Frozen; force eval() so training-mode side effects never leak in (mirrors
    # ltx2_ops.encode_prompt's Gemma-3 eval() call).
    text_encoder.eval()

    text_prompt = f"# Instruction\n{DEFAULT_DIT_INSTRUCTION}\n\n# Caption\n{prompt}<|endoftext|>\n"

    with torch.no_grad():
        tok = tokenizer(
            text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt"
        )
        text_ids = tok.input_ids.to(device)
        text_attention_mask = tok.attention_mask.to(device).bool()
        text_hidden_states = text_encoder(input_ids=text_ids).last_hidden_state

    dtype = next(text_encoder.parameters()).dtype
    lyric_hidden_states, lyric_attention_mask = _encode_lyrics(trainer, lyrics, device=device, dtype=dtype)

    return text_hidden_states[0].detach().to(dtype), {
        "text_attention_mask": text_attention_mask[0].detach(),
        "lyric_hidden_states": lyric_hidden_states,
        "lyric_attention_mask": lyric_attention_mask,
    }


def _encode_lyrics(
    trainer, lyrics: str, *, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the PER-SAMPLE (batch dim dropped) lyric conditioning tensors for
    one dataset item's lyrics text.

    Empty lyrics (``""`` -- the default / instrumental case) reuse the
    precomputed caption-INDEPENDENT empty-lyrics asset built once at load time
    (``_build_empty_lyrics``) instead of re-running ``embed_tokens`` -- cheap
    either way (no transformer forward), but this keeps the common case a
    pure tensor slice/cast with no tokenizer call. A non-empty lyrics string
    is tokenized and looked up on demand (also no transformer forward --
    ``AceStepLyricEncoder`` is an ``embed_tokens`` lookup only, per the module
    docstring).

    Returns:
        ``(lyric_hidden_states [L, 1024], lyric_attention_mask [L])`` on
        ``device``/``dtype`` (mask stays bool), batch dim dropped so
        ``collate_aux`` can pad+stack per-item rows exactly like
        ``text_attention_mask``.
    """
    if not lyrics:
        return (
            trainer.acestep_empty_lyric_hidden_states[0].detach().to(device=device, dtype=dtype),
            trainer.acestep_empty_lyric_attention_mask[0].detach().to(device=device).bool(),
        )

    from core.pipeline_backends.acestep import AceStepMixin

    lyrics_text = AceStepMixin._acestep_format_lyrics(lyrics, "en")
    tokenizer = trainer.tokenizer
    text_encoder = trainer.text_encoder
    with torch.no_grad():
        tok = tokenizer(
            lyrics_text, padding="longest", truncation=True, max_length=2048, return_tensors="pt"
        )
        lyric_ids = tok.input_ids.to(device)
        lyric_attention_mask = tok.attention_mask.to(device).bool()
        lyric_hidden_states = text_encoder.embed_tokens(lyric_ids)

    return lyric_hidden_states[0].detach().to(dtype), lyric_attention_mask[0].detach()


def collate_aux(trainer, aux_list):
    """Collate a list of per-item ACE-Step aux dicts into ONE dict of batched
    tensors ``{text_attention_mask [B, L], lyric_hidden_states [B, L2, 1024],
    lyric_attention_mask [B, L2]}``, padding each field's sequence dim
    independently (text and lyric sequences have unrelated lengths) with a
    per-field pad value (mirrors ltx2_ops.collate_aux's multi-key pattern)."""
    keys = ("text_attention_mask", "lyric_hidden_states", "lyric_attention_mask")
    if not aux_list:
        raise ValueError("[ACE-Step collation] empty auxiliary_data_list")
    for idx, aux in enumerate(aux_list):
        if not isinstance(aux, dict):
            raise ValueError(
                f"[ACE-Step collation] item {idx} auxiliary data is "
                f"{type(aux).__name__}, expected a dict with keys {keys}"
            )
        for k in keys:
            if k not in aux or not isinstance(aux[k], torch.Tensor):
                raise ValueError(
                    f"[ACE-Step collation] item {idx} is missing tensor key '{k}' "
                    f"(got keys {list(aux.keys())})"
                )

    pad_values = {"text_attention_mask": 0, "lyric_hidden_states": 0.0, "lyric_attention_mask": 0}

    batched: Dict[str, torch.Tensor] = {}
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

    return batched


# ----------------------------------------------------------------------
# VAE encode — audio-only arch (no still-image path)
# ----------------------------------------------------------------------

def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """ACE-Step has no still-image concept (every training item is
    item_type=="audio"); this abstract-method slot is never called in
    practice. Audio items route through ``vae_encode_audio`` via
    ``audio_loader.encode_and_cache_audio`` instead (wired from
    ``base_trainer.py``'s ``item_type=="audio"`` branch)."""
    raise NotImplementedError(
        "acestep.vae_encode: ACE-Step is audio-only; use vae_encode_audio "
        "(audio_loader.encode_and_cache_audio) instead of the still-image path"
    )


def vae_encode_audio(trainer, waveform: torch.Tensor) -> torch.Tensor:
    """Encode-integration SEAM for ``audio_loader.encode_and_cache_audio``
    (mirrors ltx2_ops.vae_encode_clip's role for video).

    Args:
        waveform: ``[2, samples]`` stereo, 48kHz, ``[-1, 1]`` CPU tensor.

    Returns:
        ``[1, T, 64]`` ACE-Step (Oobleck) audio latent.
    """
    vae = trainer.vae
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    px = waveform.unsqueeze(0).to(device=vae_device, dtype=vae_dtype)  # [1, 2, samples]
    with torch.no_grad():
        latent_dist = vae.encode(px).latent_dist
        latents = latent_dist.sample()  # [1, 64, T]
        latents = latents.transpose(1, 2).contiguous()  # [1, T, 64]
    del px
    return latents


def vae_decode(trainer, latents, *, latent_h=None, latent_w=None):
    raise NotImplementedError(
        "acestep.vae_decode: training-time audio sampling is not wired into "
        "the image-preview UI yet; see generate_sample"
    )


# ----------------------------------------------------------------------
# Training step — rectified-flow velocity prediction
# ----------------------------------------------------------------------

def train_step(
    trainer,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Single ACE-Step training step (rectified flow / velocity prediction).

    Args:
        latents: Ground-truth audio latents ``[B, T, 64]`` (VAE-encoded, cached).
        text_embeddings: "# Caption" Qwen3 hidden states ``[B, L, 1024]``.
        aux: ``{text_attention_mask [B, L]}`` (collate_aux output).

    Returns:
        (loss tensor, prediction loss value, reconstruction loss value).
    """
    from core.training.base_trainer import print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_acestep] Start")

    device = trainer.device
    dtype = trainer.training_dtype

    latents = latents.to(device=device, dtype=dtype, non_blocking=True)  # [B, T, 64] == x0
    text_embeddings = text_embeddings.to(device=device, dtype=dtype, non_blocking=True)

    text_attention_mask = aux.get("text_attention_mask") if isinstance(aux, dict) else None
    if text_attention_mask is None:
        raise ValueError(
            "[train_step_acestep] aux is missing 'text_attention_mask' "
            "(expected collate_aux's output dict)"
        )
    text_attention_mask = text_attention_mask.to(device=device, non_blocking=True).bool()

    # Per-item lyric conditioning (collate_aux output; see encode_prompt /
    # _encode_lyrics). Items with no lyrics caption carry the shared
    # empty-lyrics embedding here (per-item, not batch-uniform expand) --
    # collate_aux already padded+stacked these to [B, L, 1024] / [B, L].
    lyric_hidden_states = aux.get("lyric_hidden_states") if isinstance(aux, dict) else None
    lyric_attention_mask = aux.get("lyric_attention_mask") if isinstance(aux, dict) else None
    if lyric_hidden_states is None or lyric_attention_mask is None:
        raise ValueError(
            "[train_step_acestep] aux is missing 'lyric_hidden_states'/"
            "'lyric_attention_mask' (expected collate_aux's output dict)"
        )
    lyric_hidden_states = lyric_hidden_states.to(device=device, dtype=dtype, non_blocking=True)
    lyric_attention_mask = lyric_attention_mask.to(device=device, non_blocking=True).bool()

    if latents.dim() != 3:
        raise ValueError(
            f"[train_step_acestep] expected 3D latents [B, T, 64], got "
            f"{latents.dim()}D {tuple(latents.shape)}"
        )
    batch_size, t_lat, c_lat = latents.shape
    if c_lat != 64:
        raise ValueError(f"[train_step_acestep] expected 64 latent channels, got {c_lat}")

    dit = trainer.transformer  # AceStepConditionGenerationModel (LoRA-wrapped decoder)

    # Caption-independent conditioning asset, precomputed at load time (see
    # _build_silence_latent). Per-item lyric conditioning (lyric_hidden_states
    # / lyric_attention_mask) is read from aux above -- collate_aux already
    # batched it per-item (empty-lyrics items carry the shared empty-lyrics
    # embedding there, not a batch-uniform expand()).
    silence_latent = trainer.acestep_silence_latent.to(device=device, dtype=dtype)  # [1, 750, 64]

    # Plain text2music conditioning (no reference audio): silence timbre +
    # silence src_latents/context, matching _generate_txt2aud_acestep's
    # no-reference-audio branch exactly (is_covers=False everywhere).
    from core.pipeline_backends.acestep import AceStepMixin
    src_latents = AceStepMixin._acestep_silence_slice(silence_latent, t_lat).to(dtype=dtype)  # [1, T, 64]
    src_latents = src_latents.expand(batch_size, -1, -1)
    chunk_masks = torch.ones(batch_size, t_lat, 64, dtype=dtype, device=device)
    is_covers = torch.zeros(batch_size, dtype=torch.bool, device=device)
    # One timbre "packed" row per batch element (order_mask=arange(B) maps each
    # packed row 1:1 to its own batch index; matches unpack_timbre_embeddings'
    # N==B, counts-all-1 case).
    timbre_packed = silence_latent.expand(batch_size, -1, -1)  # [B, 750, 64]
    refer_audio_order_mask = torch.arange(batch_size, dtype=torch.long, device=device)

    # "Diffusion inputs" attention_mask (AceStepDiTModel.forward's own
    # `attention_mask` param) — all-valid (no padding: batches are grouped by
    # uniform latent-frame count, see base_trainer.py's acestep_audio_batches).
    diffusion_attention_mask = torch.ones(batch_size, t_lat, dtype=dtype, device=device)

    if profile_vram:
        print_vram_usage("[train_step_acestep] Before prepare_condition")

    # prepare_condition is @torch.no_grad()-decorated on the vendored model
    # (frozen encoder/tokenizer/detokenizer path); calling it under no_grad
    # here as well keeps the intent explicit at the call site.
    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask, context_latents = dit.prepare_condition(
            text_hidden_states=text_embeddings,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=timbre_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            hidden_states=src_latents,
            attention_mask=diffusion_attention_mask,
            silence_latent=silence_latent,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
        )

    # Sigma sampling (flow-matching), same policy as ltx2/anima: sigma in
    # [0, 1], 0=data / 1=noise — matches the vendored model's OWN "t" convention
    # (xt = t*x1 + (1-t)*x0, x1=noise) exactly, so no remap is needed.
    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(batch_size, device)
        else:
            timesteps = torch.rand(batch_size, device=device)
    sigma = timesteps.to(dtype)

    noise = torch.randn_like(latents)
    sigma_view = sigma.view(-1, 1, 1).to(latents.dtype)
    xt = (1.0 - sigma_view) * latents + sigma_view * noise

    if profile_vram:
        print_vram_usage("[train_step_acestep] Before DiT forward")

    def _forward():
        decoder_outputs = dit.decoder(
            hidden_states=xt,
            timestep=sigma,
            timestep_r=sigma,
            attention_mask=diffusion_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=False,
        )
        return decoder_outputs[0]

    if trainer.mixed_precision:
        with torch.autocast(device_type=device.type, dtype=dtype):
            v_pred = _forward()
    else:
        v_pred = _forward()

    if profile_vram:
        print_vram_usage("[train_step_acestep] After DiT forward")

    # Rectified-flow target: v = noise - x0 (== x1 - x0 in the vendored
    # model's own naming).
    target = noise - latents

    loss_per_element = F.mse_loss(v_pred.float(), target.float(), reduction="none")
    mse_loss = loss_per_element.mean()
    loss = mse_loss

    # Optional reconstruction loss (predicted x0 vs GT x0): x0 = x_t - sigma * v.
    recon_loss_value = 0.0
    if trainer.reconstruction_loss_weight > 0:
        with torch.no_grad():
            sigma_seq = sigma.view(-1, 1, 1).to(v_pred.dtype)
            pred_x0 = xt - sigma_seq * v_pred
            recon_loss = F.mse_loss(pred_x0.float(), latents.float())
            recon_loss_value = recon_loss.item()
        loss = loss + trainer.reconstruction_loss_weight * recon_loss

    pred_loss_value = mse_loss.item()

    del noise, xt, v_pred, target, loss_per_element
    return loss, pred_loss_value, recon_loss_value


# ----------------------------------------------------------------------
# Sample generation — declines (training-time audio preview not wired yet)
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
    """ACE-Step training-time validation sampling is intentionally NOT
    implemented in Phase 8a: the image-only training-preview UI has no audio
    player surface, and ``width``/``height`` (this contract's image-shaped
    params) do not map onto anything meaningful for audio. Returns None,
    mirroring the documented "cannot sample yet" contract value (ideogram4);
    callers must skip saving when None is returned (see
    ``BaseTrainer._dispatch_sample``'s docstring)."""
    print(f"{trainer.log_prefix} ACE-Step training-time sample generation is not "
          f"wired into the image-preview UI (no audio-player surface yet); skipping.")
    return None
