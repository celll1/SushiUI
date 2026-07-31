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

from typing import Optional, Tuple

import torch


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

    # A training process is DEQUANT-ONLY (see ideogram4_ops.load_components for
    # the full reasoning). Krea 2's TE CAN be fp8: when the TE directory resolves
    # to an Ideogram-4-style "<parent>/text_encoder" layout, krea2_loader's
    # _load_qwen3vl_text_encoder() delegates to load_ideogram4_text_encoder(),
    # which swaps to Fp8Linear whenever the on-disk state dict is FP8. A
    # weight-only-FP8 transformer checkpoint is also supported, and training-time
    # sample generation runs both under the pipeline's no_grad denoise loop --
    # which would make the validation previews W8A8 while the trained weights are
    # not. Both are gated below; this is a no-op when a module is bf16.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    for _label, _module in (("transformer", trainer.transformer),
                            ("text_encoder", trainer.text_encoder)):
        if _module is not None:
            disable_scaled_mm(_module, label=f"krea2 training {_label}")

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


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Krea 2 VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_krea2`` branch (self->trainer rename only). Uses the
    PIL ``image`` + ``vae_device`` from the shared pre-amble. Runs inside the
    caller's ``with torch.no_grad()``; caller does the shared final dtype/CPU move.
    """
    # Krea 2 VAE (AutoencoderKLQwenImage): packed normalized latent
    # (1, N, 64) where N=(H//16)*(W//16); C*p*p = 16*2*2 = 64.
    from core.models.krea2.krea2_pipeline_ops import vae_encode as _krea2_vae_encode
    latents = _krea2_vae_encode(
        trainer.vae, image, height=height, width=width,
        patch_size=int(getattr(trainer, "krea2_patch_size", 2)),
        device=vae_device, dtype=trainer.vae_dtype,
    )
    return latents


def train_step(
    trainer,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Single Krea 2 training step (flow matching, velocity prediction).

    VERBATIM body of ``BaseTrainer.train_step_krea2`` (P6c; ``self.`` ->
    ``trainer.`` receiver rename only). See the original docstring for the
    flow-matching conventions (v = noise - x0, timestep = sigma, musubi shift).
    """
    from core.models.krea2.krea2_pipeline_ops import prepare_position_ids

    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_features = encoder_features.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_mask = encoder_mask.to(device=trainer.device, non_blocking=True)

    B, N, _ = latents.shape
    if latent_h is not None and latent_w is not None:
        if latent_h * latent_w != N:
            raise ValueError(
                f"[train_step_krea2] latent_h={latent_h}*latent_w={latent_w} != N={N}"
            )
    else:
        side = int(N ** 0.5)
        if side * side != N:
            raise ValueError(
                f"[train_step_krea2] non-square latent (N={N}); pass latent_h/latent_w"
            )
        latent_h = latent_w = side

    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(B, trainer.device)
        else:
            timesteps = torch.rand(B, device=trainer.device)
    sigma = timesteps.to(trainer.training_dtype)

    # Discrete flow-matching timestep shift: sigma' = s*sigma / (1 + (s-1)*sigma).
    shift = float(getattr(trainer, "krea2_discrete_flow_shift", 2.5) or 0.0)
    if shift and shift != 1.0:
        sigma = (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
    sigma_v = sigma.view(-1, 1, 1)

    noise = torch.randn_like(latents)
    noisy = (1.0 - sigma_v) * latents + sigma_v * noise   # sigma=1 -> noise
    v_target = noise - latents                            # Krea convention v = noise - x0

    text_seq_len = encoder_features.shape[1]
    position_ids = prepare_position_ids(text_seq_len, latent_h, latent_w, trainer.device)

    t_dtype = trainer.transformer.dtype

    def _fwd():
        return trainer.transformer(
            hidden_states=noisy.to(t_dtype),
            encoder_hidden_states=encoder_features.to(t_dtype),
            timestep=sigma.to(t_dtype),
            position_ids=position_ids,
            encoder_attention_mask=encoder_mask,
            return_dict=False,
        )[0]

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            out = _fwd()
    else:
        out = _fwd()

    v_pred = out.float()
    loss = torch.nn.functional.mse_loss(v_pred, v_target.float(), reduction="mean")
    pred_loss_value = loss.item()
    # Backward is performed by _execute_forward_backward; do not backward here.
    del noise, noisy, v_pred, v_target
    return loss, pred_loss_value, 0.0


# ============================================================
# Krea 2 Sample Generation (plan P7)
# ============================================================
# Verbatim body of BaseTrainer._generate_sample_krea2 (base_trainer.py), moved
# out of the spine with the mechanical self.->trainer. receiver rename and the
# relocated .optimizers -> ..optimizers relative import. arch/krea2.py::sample()
# unpacks SampleContext into this.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.5,
    seed: int = -1,
    negative_prompt: str = "",
):
    """Generate a validation sample during Krea 2 training (flow matching).

    Reuses the krea2_pipeline_ops denoise loop. UI ``guidance_scale`` maps to the
    Krea guidance convention via ``guidance = cfg_scale - 1`` (turbo/distilled
    checkpoints run with no CFG). Resolution is aligned to a multiple of 16.
    """
    from core.models.krea2.krea2_pipeline_ops import (
        encode_prompt as _k_encode, denoise_loop as _k_denoise,
        prepare_latents_txt2img as _k_prep, vae_decode as _k_decode,
    )

    print(f"{trainer.log_prefix} Generating Krea 2 sample: {prompt[:50]}...")
    patch_size = int(getattr(trainer, "krea2_patch_size", 2))
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    grid_h = height // 16
    grid_w = width // 16
    select_layers = getattr(trainer, "krea2_select_layers", None) or [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
    is_distilled = bool(getattr(trainer, "krea2_is_distilled", False))
    guidance = 0.0 if is_distilled else max(0.0, float(guidance_scale) - 1.0)

    trainer.transformer.eval()
    trainer.text_encoder.eval()
    transformer_device = next(trainer.transformer.parameters()).device
    t_dtype = trainer.transformer.dtype

    try:
        # Offload transformer + optimizer state to CPU during text encoding.
        trainer.transformer.to("cpu")
        optimizer_state_dict = trainer.optimizer.state_dict()
        for _pid, state in optimizer_state_dict["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    state[key] = value.cpu()
        trainer.optimizer.load_state_dict(optimizer_state_dict)
        torch.cuda.empty_cache()

        trainer.text_encoder.to(trainer.device)
        prompt_embeds, prompt_mask = _k_encode(
            trainer.text_encoder, trainer.tokenizer, prompt, select_layers, 512, trainer.device)
        neg_embeds = neg_mask = None
        if guidance > 0.0:
            neg_embeds, neg_mask = _k_encode(
                trainer.text_encoder, trainer.tokenizer, negative_prompt or "", select_layers, 512, trainer.device)
        trainer.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        trainer.transformer.to(transformer_device)
        torch.cuda.empty_cache()

        z_dim = int(getattr(trainer.vae.config, "z_dim", 16))
        latents = _k_prep(
            z_dim, grid_h, grid_w, patch_size, t_dtype, trainer.device,
            seed=seed if seed is not None and seed >= 0 else None,
        )

        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            out = _k_denoise(
                trainer.transformer, trainer.scheduler, latents,
                prompt_embeds.to(t_dtype), prompt_mask,
                neg_embeds.to(t_dtype) if neg_embeds is not None else None, neg_mask,
                guidance, num_inference_steps, grid_h, grid_w, patch_size, is_distilled, trainer.device,
            )

        trainer.vae.to(trainer.device)
        image = _k_decode(trainer.vae, out.float(), grid_h, grid_w, patch_size)
        trainer.vae.to("cpu")

        del prompt_embeds, prompt_mask, out, latents
        if neg_embeds is not None:
            del neg_embeds, neg_mask
        torch.cuda.empty_cache()

        # Restore optimizer state to GPU.
        from ..optimizers.adamw8bit_ringbuffer import AdamW8bit_RingBuffer
        from ..optimizers.lion8bit_ringbuffer import Lion8bit_RingBuffer
        if not isinstance(trainer.optimizer, (AdamW8bit_RingBuffer, Lion8bit_RingBuffer)):
            optimizer_state_dict = trainer.optimizer.state_dict()
            for _pid, state in optimizer_state_dict["state"].items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                        state[key] = value.to(transformer_device)
            trainer.optimizer.load_state_dict(optimizer_state_dict)
        torch.cuda.empty_cache()
        return image

    finally:
        trainer.transformer.train()
