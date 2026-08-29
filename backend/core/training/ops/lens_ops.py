"""lens_ops.py — Lens (DiT) loader + block-swap + attention free functions (P3b).

VERBATIM bodies of ``BaseTrainer._load_lens_components``,
``BaseTrainer.setup_lens_block_swap`` and
``BaseTrainer._setup_attention_backend_lens`` (base_trainer.py), moved out of the
spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher calls ``load_components`` directly. ``setup_block_swap`` and
``setup_attention_backend`` keep 2-line delegators on the trainer (late/multiple
call sites); each body is defined exactly once here.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..training_events import emit_training_warning
from .training_method import trains_denoiser_weights


def load_components(trainer) -> None:
    """Load Lens model components for training.

    Lens ships as a standard diffusers directory layout. ModelLoader.load_lens_components
    returns the same component dict used by the inference path.
    """
    print(f"{trainer.log_prefix} Detected Lens model")
    print(f"{trainer.log_prefix} Loading Lens components from {trainer.model_path}")

    from core.models.lens.lens_loader import load_lens_components
    components = load_lens_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
    )

    # Store components on the trainer using the standard slots.
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer   # No wrapper for Lens.
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Lens specific: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # Gradient checkpointing.
    cpu_offload_ckpt  = bool(trainer.config.get("cpu_offload_checkpointing", False))
    async_offload_ckpt = bool(trainer.config.get("async_cpu_offload_checkpointing", False))
    if cpu_offload_ckpt and async_offload_ckpt:
        print(f"{trainer.log_prefix} WARNING: both cpu_offload_checkpointing and "
              f"async_cpu_offload_checkpointing are True; using async (faster).")
        cpu_offload_ckpt = False
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Lens)")
    elif hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        trainer.transformer.enable_gradient_checkpointing(
            cpu_offload=cpu_offload_ckpt,
            async_offload=async_offload_ckpt,
        )
        ckpt_mode = ("async_cpu_offload" if async_offload_ckpt
                      else "cpu_offload" if cpu_offload_ckpt else "standard")
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Lens transformer "
              f"(mode={ckpt_mode})")

    # Freeze everything; LoRA/full-FT will unfreeze what is needed.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Optional FP8 base quantisation (frozen transformer only; same helper as Anima).
    fp8_base_dtype = trainer.config.get("fp8_base_dtype") or None
    if fp8_base_dtype and not trains_denoiser_weights(trainer):
        print(f"{trainer.log_prefix} Quantising frozen Lens transformer base to "
              f"{fp8_base_dtype} (LoRA-on-FP8-base)")
        from core.vram_optimization import _anima_quantize_fp8
        trainer.transformer = _anima_quantize_fp8(
            trainer.transformer, fp8_base_dtype, "Lens Transformer (training base)",
        )
        trainer.transformer_original = trainer.transformer
        trainer.transformer.requires_grad_(False)
    elif fp8_base_dtype:
        emit_training_warning(
            f"fp8_base_dtype={fp8_base_dtype} requires a "
            f"frozen transformer and is ignored when the transformer itself is trained "
            f"(full fine-tune with train_unet=True). The base stays unquantised.",
            code="fp8_base_dtype_ignored",
            prefix=trainer.log_prefix,
        )

    # Block-swap deferred; conductor handle initialised to None.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving Lens transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (opt-in; SDPA fallback otherwise)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_lens(trainer.attention_backend)

    print(f"{trainer.log_prefix} Lens model loaded successfully")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor for the Lens transformer, AFTER adapter setup."""
    if not trainer.is_lens:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "transformer_blocks"):
        raise ValueError("Lens transformer must expose `.transformer_blocks` for block swap")

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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Lens")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Lens (DiT) models.

    Sets ``m._attention_backend`` on every ``LensJointAttention`` module (the
    new contract the Stage-B vendor read-site consumes, design 2.5). Also
    sets the legacy ``_use_flash_attn`` flag as a TRANSITIONAL bridge so the
    current vendor (which still reads ``_use_flash_attn`` until Phase 3e)
    keeps honoring flash during the staged rollout — the two stay consistent
    and the legacy flag is removed once the read-site migrates.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        n = 0
        for m in trainer.transformer.modules():
            if type(m).__name__ == "LensJointAttention":
                m._attention_backend = b
                # Transitional bridge for the pre-Phase-3e vendor read-site.
                m._use_flash_attn = (b == 'flash')
                n += 1
        print(f"{trainer.log_prefix} [OK] Lens attention backend '{b}' set on {n} module(s)")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Lens attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def encode_prompt(trainer, prompt: str, max_length: int = 512):
    """Encode prompt for Lens using the inference encode_prompt function.

    VERBATIM body of ``BaseTrainer.encode_prompt_lens`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    from core.models.lens.lens_pipeline_ops import encode_prompt as _encode
    # encode_prompt returns (List[Tensor[1, L, D]], Tensor[1, L]) for a
    # single prompt. We call it with empty string as neg prompt to get
    # only the conditional side; the uncond side is discarded.
    encoder_features, encoder_mask = _encode(
        text_encoder=trainer.text_encoder,
        tokenizer=trainer.tokenizer,
        prompt=prompt,
        negative_prompt="",
        device=str(trainer.device),
        dtype=trainer.training_dtype,
        max_length=max_length,
    )
    # encoder_features: List[Tensor[2, L, D]] (batch of [cond, uncond]);
    # slice out the conditional (index 0) for each layer.
    cond_features = [f[0:1].squeeze(0).detach() for f in encoder_features]  # each [L, D]
    # encoder_mask: [2, L] — take the conditional row.
    cond_mask = encoder_mask[0].detach()  # [L]
    # Stack per-layer and add batch dim: [1, num_layers, L, D].
    # The batch dim allows torch.cat(dim=0) in the training loop to produce
    # the correct [B, num_layers, L, D] batched tensor.
    stacked = torch.stack(cond_features, dim=0).unsqueeze(0)  # [1, num_layers, L, D]
    return stacked, cond_mask


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Lens VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_lens`` branch (self->trainer rename only). Uses the
    PIL ``image`` + ``vae_device`` from the shared pre-amble. Runs inside the
    caller's ``with torch.no_grad()``; caller does the shared final dtype/CPU move.
    """
    # Lens VAE (AutoencoderKLFlux2): vae_encode handles resize, patchify,
    # BN normalise, and rearrange to flat-sequence (1, N, 128).
    from core.models.lens.lens_pipeline_ops import vae_encode as _lens_vae_encode
    latents = _lens_vae_encode(
        trainer.vae, image, height=height, width=width,
        device=vae_device, dtype=trainer.vae_dtype,
    )
    return latents


def apply_cfg_null_collated(
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    drop_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rewrite the ``drop_mask`` rows of a collated Lens batch into the condition
    its inference CFG uncond branch builds.

    ``lens_pipeline_ops.encode_prompt`` builds that branch, for a blank negative,
    as ``neg_features = [f.new_zeros(f.shape) for f in pos_features]`` and
    ``neg_mask = torch.zeros_like(pos_mask, dtype=torch.bool)`` -- zeros shaped
    like the POSITIVE encoding, at the positive's own sequence length. So the
    rewrite is ``features[drop] = 0`` / ``mask[drop] = False`` and nothing else:
    L stays the batch's own length (``_align_text_features`` is the identity when
    the two sides already match), and the per-layer axis is untouched.

    Out of place: both tensors belong to the assembled batch and are handed to
    every MNT iteration, so an in-place write would leak one iteration's null
    into the next.
    """
    if drop_mask is None:
        return encoder_features, encoder_mask
    selected = drop_mask.to(dtype=torch.bool)
    if not bool(selected.any()):
        return encoder_features, encoder_mask
    features = encoder_features.clone()
    features[selected.to(features.device)] = 0
    mask = encoder_mask.clone()
    mask[selected.to(mask.device)] = False
    return features, mask


def train_step(
    trainer,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
    cfg_drop_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Single Lens DiT training step (flow-matching, velocity prediction).

    Args:
        latents:          Flat-sequence latents [B, N, 128].
        encoder_features: Stacked multi-layer text features [B, num_layers, L, D].
        encoder_mask:     Bool mask for text tokens [B, L].
        timesteps:        Optional pre-sampled sigma values in [0, 1].
        latent_h:         Spatial height of the latent grid (height // 16).
                          Required for non-square latents; inferred from N for square.
        latent_w:         Spatial width of the latent grid (width // 16).
                          Required for non-square latents; inferred from N for square.

    Returns:
        (loss tensor, prediction loss value, reconstruction loss value)
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_lens] Start")

    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_features = encoder_features.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_mask = encoder_mask.to(device=trainer.device, non_blocking=True)

    if cfg_drop_mask is not None:
        # Aligned CFG null, applied AFTER the device/dtype moves (which may be
        # identity no-ops that would have handed back the batch's own tensors)
        # and BEFORE the per-layer conditioning list below. Routed through the
        # declared handler hook so a stage mismatch raises.
        encoder_features, encoder_mask = trainer.arch.apply_cfg_null_collated(
            trainer, encoder_features, encoder_mask, cfg_drop_mask)

    batch_size = latents.shape[0]

    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
        else:
            timesteps = torch.rand(batch_size, device=trainer.device)

    noise = torch.randn_like(latents)

    # Flow matching forward process: x_t = (1-σ)*x0 + σ*noise
    sigma = timesteps.to(trainer.training_dtype)
    sigma_view = sigma.view(-1, 1, 1)
    noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise

    # Velocity target: v = noise - x0
    v_target = noise - latents

    # Lens timestep convention: transformer receives sigma * 1000
    timestep_input = (sigma * 1000.0).to(trainer.training_dtype)

    # img_shapes for positional encoding: single 3-tuple (frame=1, H, W) required by
    # LensEmbedRope.  Lens supports arbitrary (H, W) multiples of 16; latent_h/latent_w
    # must be passed explicitly for non-square latents.
    seq_len = latents.shape[1]  # N = latent_h * latent_w
    if latent_h is not None and latent_w is not None:
        if latent_h * latent_w != seq_len:
            raise ValueError(
                f"[train_step_lens] latent_h={latent_h}, latent_w={latent_w} "
                f"inconsistent with seq_len={seq_len} (expected {latent_h * latent_w})"
            )
    else:
        # Fall back to square assumption when dims weren't supplied.
        latent_hw = int(seq_len ** 0.5)
        if latent_hw * latent_hw != seq_len:
            raise ValueError(
                f"[train_step_lens] Non-square latent (N={seq_len}): pass latent_h "
                f"and latent_w explicitly so img_shapes can be set correctly."
            )
        latent_h = latent_w = latent_hw
    img_shapes = [(1, latent_h, latent_w)]

    # encoder_features [B, num_layers, L, D] → list of num_layers tensors each [B, L, D]
    num_layers = encoder_features.shape[1]
    encoder_hidden_states_list = [encoder_features[:, i, :, :] for i in range(num_layers)]

    if profile_vram:
        print_vram_usage("[train_step_lens] Before transformer forward")

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            v_pred = trainer.transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=encoder_hidden_states_list,
                encoder_hidden_states_mask=encoder_mask,
                timestep=timestep_input,
                img_shapes=img_shapes,
            )
    else:
        v_pred = trainer.transformer(
            hidden_states=noisy_latents,
            encoder_hidden_states=encoder_hidden_states_list,
            encoder_hidden_states_mask=encoder_mask,
            timestep=timestep_input,
            img_shapes=img_shapes,
        )

    if profile_vram:
        print_vram_usage("[train_step_lens] After transformer forward")

    # MSE loss on velocity
    mse_loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float(), reduction="mean")
    loss = mse_loss

    pred_loss_value = mse_loss.item()
    recon_loss_value = 0.0

    # Backward is performed by _execute_forward_backward (single backward per
    # MNT iteration); do not call loss.backward() here.
    del noise, noisy_latents, v_pred, v_target, encoder_hidden_states_list
    return loss, pred_loss_value, recon_loss_value


# ============================================================
# Lens Sample Generation (mxfp4 TE + flow-matching DiT) (plan P7)
# ============================================================
# Verbatim body of BaseTrainer._generate_sample_lens (base_trainer.py), moved
# out of the spine with the mechanical self.->trainer. receiver rename only.
# arch/lens.py::sample() unpacks SampleContext into this.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.0,
    seed: int = -1,
    negative_prompt: str = "",
):
    """Generate a sample image during training (Lens).

    Reuses the Lens inference pipeline ops (encode_prompt / prepare_latents /
    denoise_loop / vae_decode) on the trainer's components.  move_text_encoder_to_gpu
    transparently reloads the mxfp4 text encoder if it was freed during training.
    """
    import copy as _copy
    import random as _random
    from core.models.lens.lens_pipeline_ops import (
        encode_prompt as _lens_encode_prompt,
        prepare_latents as _lens_prepare_latents,
        denoise_loop as _lens_denoise_loop,
        vae_decode as _lens_vae_decode,
    )
    from core.models.lens.lens_resolution import align_to_grid as _lens_align

    print(f"{trainer.log_prefix} Generating Lens sample: {prompt[:50]}...")
    device = trainer.device
    dtype = torch.bfloat16
    max_sequence_length = 512

    width, height = _lens_align(width, height)
    latent_h = height // 16
    latent_w = width // 16

    trainer.transformer.eval()
    trainer.vae.eval()

    try:
        # --- Offload transformer (+ optimizer state) to CPU for TE encode ---
        trainer.move_main_model_to_cpu()
        trainer._relocate_main_model_optimizer_state("cpu")
        torch.cuda.empty_cache()

        # --- Stage 1: text encoding (reloads mxfp4 TE if freed) ---
        trainer.move_text_encoder_to_gpu()
        encoder_features, encoder_mask = _lens_encode_prompt(
            trainer.text_encoder, trainer.tokenizer, prompt, negative_prompt,
            device=device, dtype=dtype, max_length=max_sequence_length,
        )
        trainer.move_text_encoder_to_cpu()
        torch.cuda.empty_cache()

        # --- Stage 2: denoising ---
        # Optimizer state stays on CPU for the whole sampling span (it is
        # never stepped here); it returns to GPU only at the final restore.
        trainer.move_main_model_to_gpu()
        seed_val = seed if (seed is not None and seed >= 0) else _random.randint(0, 2**32 - 1)
        latents = _lens_prepare_latents(height, width, dtype=dtype, device=device, seed=seed_val)
        sample_scheduler = _copy.deepcopy(trainer.scheduler)
        # Autocast the denoise loop to the sampling compute dtype (bf16). This is
        # unconditional (NOT gated on trainer.mixed_precision): sampling always
        # runs the DiT in bf16 (dtype above), while LoRA adapters default to
        # lora_dtype=fp32 on a bf16 base. Without autocast the bf16 activations
        # hit the fp32 LoRA Linear weights and crash with a dtype mismatch inside
        # the forward — regardless of the mixed_precision flag. Mirrors the anima
        # fix; the training forward avoids the same crash via its own autocast.
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            latents = _lens_denoise_loop(
                transformer=trainer.transformer, scheduler=sample_scheduler,
                latents=latents, encoder_features=encoder_features,
                encoder_mask=encoder_mask,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                latent_h=latent_h, latent_w=latent_w, tokenizer=trainer.tokenizer,
                spectrum_params={},
            )
        del encoder_features, encoder_mask

        # --- Stage 3: VAE decode ---
        trainer.move_main_model_to_cpu()
        torch.cuda.empty_cache()
        trainer.move_vae_to_gpu()
        with torch.no_grad():
            image = _lens_vae_decode(trainer.vae, latents, latent_h, latent_w)
        trainer.move_vae_to_cpu()
        del latents

        # --- Restore transformer to GPU for continued training ---
        trainer.move_main_model_to_gpu()
        trainer._relocate_main_model_optimizer_state(device)
        torch.cuda.empty_cache()
        return image

    finally:
        trainer.transformer.train()
        trainer.vae.train()
        if trainer.text_encoder is not None:
            trainer.text_encoder.train()
