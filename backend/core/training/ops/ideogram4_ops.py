"""ideogram4_ops.py — Ideogram 4 (dual-transformer DiT) loader + block-swap +
attention free functions (plan P3b).

VERBATIM bodies of ``BaseTrainer._load_ideogram4_components``,
``BaseTrainer.setup_ideogram4_block_swap`` and
``BaseTrainer._setup_attention_backend_ideogram4`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher calls ``load_components`` directly. ``setup_block_swap`` and
``setup_attention_backend`` keep 2-line delegators on the trainer (late/multiple
call sites); each body is defined exactly once here.

``to_diffusers_backend`` is a module-level name in base_trainer used by the moved
attention body; imported here (import adjustment, allowed by the plan).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from core.attention import to_diffusers_backend


def load_components(trainer) -> None:
    """Load Ideogram 4 components for LoRA training (conditional branch by default).

    The fp8 transformer (Fp8Linear) is loaded frozen; LoRA wraps it. The
    unconditional transformer is loaded only when `ideogram4_train_uncond` is set.
    """
    print(f"{trainer.log_prefix} Detected Ideogram 4 model")
    print(f"{trainer.log_prefix} Loading Ideogram 4 components from {trainer.model_path}")

    trainer.ideogram4_train_uncond = bool(trainer.config.get("ideogram4_train_uncond", False))
    trainer.ideogram4_uncond_loss_weight = float(trainer.config.get("ideogram4_uncond_loss_weight", 1.0))

    from core.models.ideogram4.ideogram4_loader import load_ideogram4_components
    components = load_ideogram4_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
        load_unconditional=trainer.ideogram4_train_uncond,
    )

    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.transformer_uncond = components.get("unconditional_transformer")
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Single-stream DiT: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # A training process is DEQUANT-ONLY. ``SUSHI_FP8_SCALED_MM`` is inherited
    # from the backend (training_process.py does os.environ.copy()), and the
    # trainer's TE encode path goes through the @torch.no_grad()-decorated
    # ``encode_text_layers``, so neither the env flag nor grad mode can be relied
    # on to keep the W8A8 fast path out of training. Switch it off explicitly on
    # every fp8 module this trainer owns, so the LoRA is fitted against exactly
    # the base function everyone else runs at inference.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    for _label, _module in (
        ("transformer", trainer.transformer),
        ("transformer_uncond", trainer.transformer_uncond),
        ("text_encoder", trainer.text_encoder),
    ):
        if _module is not None:
            disable_scaled_mm(_module, label=f"ideogram4 training {_label}")

    # Gradient checkpointing.
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Ideogram 4)")
    else:
        for t in (trainer.transformer, trainer.transformer_uncond):
            if t is not None and hasattr(t, "enable_gradient_checkpointing"):
                try:
                    t.enable_gradient_checkpointing()
                except Exception as e:
                    print(f"{trainer.log_prefix} grad checkpoint enable failed: {e}")
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Ideogram 4 transformer(s)")

    # Freeze everything; LoRA adapter wraps the fp8 base (already weight-only-fp8).
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)
    if trainer.transformer_uncond is not None:
        trainer.transformer_uncond.requires_grad_(False)

    # Block-swap deferred until after adapter setup.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving Ideogram 4 transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)
    if trainer.transformer_uncond is not None:
        trainer.transformer_uncond.to(trainer.device)

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_ideogram4(trainer.attention_backend)

    print(f"{trainer.log_prefix} Ideogram 4 model loaded successfully")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor for the Ideogram 4 transformer(s), AFTER adapter setup."""
    if not trainer.is_ideogram4:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "layers"):
        raise ValueError("Ideogram 4 transformer must expose `.layers` for block swap")

    from core.memory_management import LayerOffloadConductor
    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=trainer.transformer.layers,
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
    # Optional: a second conductor for the unconditional transformer when trained.
    if getattr(trainer, "transformer_uncond", None) is not None and getattr(trainer, "ideogram4_train_uncond", False):
        trainer.layer_offload_conductor_uncond = LayerOffloadConductor(
            layers=trainer.transformer_uncond.layers,
            blocks_to_swap=trainer.blocks_to_swap,
            device=trainer.device,
            use_pinned_memory=trainer.use_pinned_memory,
            cpu_buffer_size_mb=8192,
            activation_buffer_size_mb=4096,
            enable_prefetch=True,
            enable_activation_offload=False,
        )
        trainer.transformer_uncond._layer_offload_conductor = trainer.layer_offload_conductor_uncond
        trainer.layer_offload_conductor_uncond.register_hooks()
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Ideogram 4")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Ideogram4 models (training hook).

    The vendored ``Ideogram4AttnProcessor`` calls diffusers'
    ``dispatch_attention_fn(..., backend=self._attention_backend)``, so we set
    the per-module processor's ``_attention_backend`` to the diffusers string
    (mapped via ``to_diffusers_backend``). ``resolve_backend`` refuses sage for
    training (R4); note head_dim=256 also excludes sage at inference. Stage-B
    adds the inference-pipeline plumbing + flash_attn_varlen path; this hook
    only stamps the field for training and honors the training guard.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    diffusers_b = to_diffusers_backend(b)
    try:
        n = 0
        for t in (trainer.transformer, getattr(trainer, "transformer_uncond", None)):
            if t is None:
                continue
            for m in t.modules():
                if type(m).__name__ == "Ideogram4Attention":
                    processor = getattr(m, "processor", None)
                    if processor is not None:
                        processor._attention_backend = diffusers_b
                        n += 1
        print(f"{trainer.log_prefix} [OK] Ideogram4 attention backend '{b}' "
              f"(diffusers '{diffusers_b}') set on {n} processor(s)")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Ideogram4 attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def encode_prompt(trainer, prompt: str, max_length: int = 512):
    """Encode prompt for Ideogram 4: 13-layer Qwen3-VL hidden states.

    VERBATIM body of ``BaseTrainer.encode_prompt_ideogram4`` (plan P4), moved out
    of the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    from core.models.ideogram4.ideogram4_pipeline_ops import encode_text_layers
    stacked, mask = encode_text_layers(
        trainer.text_encoder, trainer.tokenizer, prompt, max_sequence_length=max_length,
    )  # stacked [13, L, 4096] (cpu f32), mask [L] (cpu bool)
    return stacked.unsqueeze(0).detach(), mask.detach()


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Ideogram 4 VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_ideogram4`` branch (self->trainer rename only). Uses
    the PIL ``image`` + ``vae_device`` from the shared pre-amble. Runs inside the
    caller's ``with torch.no_grad()``; caller does the shared final dtype/CPU move.
    """
    # Ideogram 4 VAE (AutoencoderKLFlux2): same flat-sequence latent
    # (1, N, 128) — BN normalise + 2x2 patchify, shared with Lens space.
    from core.models.ideogram4.ideogram4_pipeline_ops import vae_encode as _ig4_vae_encode
    latents = _ig4_vae_encode(
        trainer.vae, image, height=height, width=width,
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
    """Single Ideogram 4 training step (flow-matching, velocity prediction).

    Conventions derived from the inference path (which calls
    `scheduler.step(-v)`):
      x_sigma   = (1 - sigma) * x0 + sigma * noise   (sigma=1 -> noise)
      v_target  = x0 - noise                         (transformer output sign)
      timestep  = 1 - sigma                          (model time in [0, 1])
    These satisfy pred_x0 = x_sigma + sigma * v = x0.

    Args:
        latents:          Packed image latents [B, N, 128].
        encoder_features: 13-layer Qwen3-VL features [B, 13, L, 4096].
        encoder_mask:     Text token mask [B, L].
        latent_h/latent_w: latent grid (height//16, width//16).
    """
    from core.models.ideogram4.ideogram4_pipeline_ops import (
        concat_layer_features, build_training_conditioning,
    )

    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_features = encoder_features.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    encoder_mask = encoder_mask.to(device=trainer.device, non_blocking=True)

    B, N, _ = latents.shape
    if latent_h is not None and latent_w is not None:
        if latent_h * latent_w != N:
            raise ValueError(
                f"[train_step_ideogram4] latent_h={latent_h}*latent_w={latent_w} != N={N}"
            )
    else:
        side = int(N ** 0.5)
        if side * side != N:
            raise ValueError(
                f"[train_step_ideogram4] non-square latent (N={N}); pass latent_h/latent_w"
            )
        latent_h = latent_w = side

    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(B, trainer.device)
        else:
            timesteps = torch.rand(B, device=trainer.device)
    sigma = timesteps.to(trainer.training_dtype)
    sigma_v = sigma.view(-1, 1, 1)

    noise = torch.randn_like(latents)
    noisy = (1.0 - sigma_v) * latents + sigma_v * noise  # sigma=1 -> noise
    v_target = latents - noise                            # x0 - noise
    t_model = (1.0 - sigma).to(trainer.training_dtype)        # model time [0,1]

    # Build packed conditioning (text + image positions/indicator/segment).
    text_features = concat_layer_features(encoder_features)  # [B, L, 53248]
    cond = build_training_conditioning(text_features, encoder_mask, latent_h, latent_w)
    max_text = cond["max_text_tokens"]

    t_dtype = trainer.transformer.dtype
    text_z = torch.zeros(B, max_text, latents.shape[-1], dtype=noisy.dtype, device=trainer.device)
    pos_z = torch.cat([text_z, noisy], dim=1).to(t_dtype)
    llm_features = cond["llm_features"].to(t_dtype)

    def _cond_forward():
        return trainer.transformer(
            hidden_states=pos_z,
            timestep=t_model.to(t_dtype),
            encoder_hidden_states=llm_features,
            position_ids=cond["position_ids"],
            segment_ids=cond["segment_ids"],
            indicator=cond["indicator"],
            return_dict=False,
        )[0]

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            out = _cond_forward()
    else:
        out = _cond_forward()
    v_pred = out[:, max_text:].float()
    loss = torch.nn.functional.mse_loss(v_pred, v_target.float(), reduction="mean")

    # Optional auxiliary unconditional branch (image-only, zeroed text).
    if getattr(trainer, "ideogram4_train_uncond", False) and getattr(trainer, "transformer_uncond", None) is not None:
        uncond = trainer.transformer_uncond
        u_dtype = uncond.dtype
        neg_llm = torch.zeros(
            B, N, llm_features.shape[-1], dtype=u_dtype, device=trainer.device
        )
        neg_pos = cond["position_ids"][:, max_text:]
        neg_seg = cond["segment_ids"][:, max_text:]
        neg_ind = cond["indicator"][:, max_text:]
        neg_hidden = noisy.to(u_dtype)

        def _uncond_forward():
            return uncond(
                hidden_states=neg_hidden,
                timestep=t_model.to(u_dtype),
                encoder_hidden_states=neg_llm,
                position_ids=neg_pos,
                segment_ids=neg_seg,
                indicator=neg_ind,
                return_dict=False,
            )[0]

        if trainer.mixed_precision:
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
                neg_out = _uncond_forward()
        else:
            neg_out = _uncond_forward()
        uncond_loss = torch.nn.functional.mse_loss(neg_out.float(), v_target.float(), reduction="mean")
        loss = loss + float(getattr(trainer, "ideogram4_uncond_loss_weight", 1.0)) * uncond_loss

    pred_loss_value = loss.item()
    # Backward is performed by _execute_forward_backward; do not backward here.
    del noise, noisy, v_pred, v_target, pos_z, llm_features
    return loss, pred_loss_value, 0.0
