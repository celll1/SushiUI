"""anima_ops.py — Anima (Cosmos-Predict2 DiT) loader + block-swap + attention
free functions (plan P3b).

These are the VERBATIM bodies of ``BaseTrainer._load_anima_components``,
``BaseTrainer.setup_anima_block_swap`` and
``BaseTrainer._setup_attention_backend_anima`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3a/P3b): the arch handler is built at the END of
``BaseTrainer.__init__`` (base_trainer.py:1115), AFTER ``_load_model_components``
runs. So the load-time dispatcher CANNOT use ``trainer.arch`` here. The
base_trainer dispatcher calls ``load_components`` directly; ``setup_block_swap``
and ``setup_attention_backend`` keep 2-line delegators on the trainer because
they have late/multiple call sites (mode subclasses call ``setup_anima_block_swap``
via ``hasattr``; the loader body calls ``trainer._setup_attention_backend_anima``).
Each body is defined exactly once here and stays byte-identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..training_events import emit_training_warning
from .training_method import trains_denoiser_weights


def load_components(trainer) -> None:
    """Load Anima model components for training.

    Anima ships as either a split-files HuggingFace layout or a single DiT
    safetensors plus separately-discovered Qwen3 / Qwen-Image VAE files.
    ModelLoader.load_anima_from_files handles both and returns a component
    dict identical to the one used by the inference path.
    """
    print(f"{trainer.log_prefix} Detected Anima model")
    print(f"{trainer.log_prefix} Loading Anima components from {trainer.model_path}")

    from core.model_loader import ModelLoader
    components = ModelLoader.load_anima_from_files(
        path=trainer.model_path,
        device="cpu",
        torch_dtype=trainer.weight_dtype,
    )

    # Store components on the trainer in the standard slots.
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer  # No wrapper for Anima.
    trainer.vae = components["vae"]
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.t5_tokenizer = components["t5_tokenizer"]
    trainer.scheduler = components["scheduler"]

    # Anima specific: no dual TE / no U-Net.
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Cast VAE to the desired dtype.
    trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)

    # A training process is DEQUANT-ONLY (see ideogram4_ops.load_components for
    # the full reasoning). An Anima DiT loaded from a weight-only int8/fp8
    # checkpoint owns Int8Linear / Fp8Linear modules whose W8A8 fast paths are
    # enabled by process-wide env flags that training_process.py copies from the
    # backend (os.environ.copy()), and grad mode is not a usable proxy for "this
    # is inference". Switch both off explicitly on every module this trainer
    # owns, so a LoRA is fitted against exactly the base function everyone else
    # runs -- and so training-time sample previews, which run under the
    # pipeline's no_grad denoise loop, cannot be W8A8 while the trained weights
    # are not. Two separate module types with two separate per-instance
    # opt-outs: disabling one does not disable the other. No-op on a bf16 base,
    # which is every Anima checkpoint that ships today.
    from core.models.ideogram4.vendor.fp8_linear import disable_scaled_mm
    from core.models.ideogram4.vendor.int8_linear import disable_int8_mm
    for _label, _module in (("transformer", trainer.transformer),
                            ("text_encoder", trainer.text_encoder)):
        if _module is not None:
            disable_scaled_mm(_module, label=f"anima training {_label}")
            disable_int8_mm(_module, label=f"anima training {_label}")

    # Gradient checkpointing mode for the DiT blocks. Three options:
    #   standard         (default) — activations stay on GPU
    #   cpu_offload      — blocking CPU offload (saves VRAM, slower)
    #   async_cpu_offload — non-blocking CPU offload (saves VRAM, fast)
    # When both flags are True, async wins and we warn.
    cpu_offload_ckpt = bool(trainer.config.get("cpu_offload_checkpointing", False))
    async_offload_ckpt = bool(trainer.config.get("async_cpu_offload_checkpointing", False))
    if cpu_offload_ckpt and async_offload_ckpt:
        print(f"{trainer.log_prefix} WARNING: both cpu_offload_checkpointing and "
              f"async_cpu_offload_checkpointing are True; using async (faster).")
        cpu_offload_ckpt = False
    if not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (Anima)")
    elif hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        trainer.transformer.enable_gradient_checkpointing(
            cpu_offload=cpu_offload_ckpt,
            async_offload=async_offload_ckpt,
        )
        ckpt_mode = ("async_cpu_offload" if async_offload_ckpt
                      else "cpu_offload" if cpu_offload_ckpt else "standard")
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Anima DiT "
              f"(mode={ckpt_mode})")
    if trainer.gradient_checkpointing and hasattr(trainer.text_encoder, "gradient_checkpointing_enable"):
        trainer.text_encoder.gradient_checkpointing_enable()
        print(f"{trainer.log_prefix} Gradient checkpointing enabled for Qwen3 text encoder")

    # Freeze all base weights. Trainable LoRA modules are added later by the
    # adapter via apply_lora_to_unet.
    trainer.vae.requires_grad_(False)
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Optional: FP8 the base DiT before LoRA wraps anything. Only safe while the
    # DiT stays frozen (every adapter path, plus a text-encoder-only full FT);
    # a full FT that trains the DiT would be training quantised weights, so the
    # flag is ignored with a warning. We piggy-back on the Phase B.1-d inference
    # quantiser which patches each Linear's forward to dequantise on-the-fly.
    fp8_base_dtype = trainer.config.get("fp8_base_dtype") or None
    if fp8_base_dtype and not trains_denoiser_weights(trainer):
        print(f"{trainer.log_prefix} Quantising frozen Anima DiT base to "
              f"{fp8_base_dtype} (LoRA-on-FP8-base, ~50% VRAM reduction)")
        from core.vram_optimization import _anima_quantize_fp8
        # deepcopy + patch — replaces trainer.transformer with the quantised
        # copy so subsequent block-swap and adapter wrap target the new
        # module references.
        trainer.transformer = _anima_quantize_fp8(
            trainer.transformer, fp8_base_dtype, "DiT (training base)",
        )
        # transformer_original keeps pointing at the quantised model too,
        # so downstream move_main_model_to_* keeps working.
        trainer.transformer_original = trainer.transformer
        trainer.transformer.requires_grad_(False)
    elif fp8_base_dtype:
        emit_training_warning(
            f"fp8_base_dtype={fp8_base_dtype} requires a "
            f"frozen DiT and is ignored when the DiT itself is trained (full fine-tune "
            f"with train_unet=True). The DiT base stays unquantised.",
            code="fp8_base_dtype_ignored",
            prefix=trainer.log_prefix,
        )

    # Plain GPU move. Block-swap init is deferred to setup_anima_block_swap(),
    # which is called by the trainer subclass AFTER any structural changes
    # (LoRA wrap / full-FT requires_grad toggling). The reason: the
    # LayerOffloadConductor snapshots each layer's state_dict at hook-
    # registration time, and a later LoRA wrap inserts new submodule keys
    # (.original_module.weight) that aren't in the snapshot, breaking the
    # CPU<->GPU swap with a KeyError. Setting up after the adapter avoids
    # that. The conductor handle is initialised to None here so callers
    # can rely on attribute presence.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")
    print(f"{trainer.log_prefix} Moving Anima DiT to {trainer.device} "
          f"(block swap, if any, will redistribute after adapter setup)")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (opt-in; SDPA fallback otherwise)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_anima(trainer.attention_backend)

    print(f"{trainer.log_prefix} Anima model loaded successfully")
    print(f"{trainer.log_prefix} Scheduler: {trainer.scheduler.__class__.__name__}, "
          f"latent_channels=16")


def setup_block_swap(trainer) -> None:
    """Initialise the LayerOffloadConductor for the Anima DiT, AFTER any
    structural model changes (LoRA wrapping / full-FT param toggling).

    Idempotent: no-op when the trainer isn't on Anima, blocks_to_swap is
    0, or a conductor is already attached. The conductor snapshots each
    layer's state_dict at register_hooks() time, which is why this has to
    run after LoRALinearLayer wrappers (if any) have been inserted.
    """
    if not trainer.is_anima:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    if not hasattr(trainer.transformer, "blocks"):
        raise ValueError("Anima DiT must expose `.blocks` (nn.ModuleList) for block swap")

    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    from core.memory_management import LayerOffloadConductor
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=trainer.transformer.blocks,
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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for Anima")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for Anima (Cosmos DiT) models.

    Anima's vendored attention dispatches on a per-block ``attn_mode`` whose
    vocabulary is ``'torch'|'flash'`` (no 'native'/'sage'). Map native->'torch',
    flash->'flash' (R9); sage is refused upstream by ``resolve_backend`` and
    arrives here as native. Masked attention (and any failure) still falls
    back to SDPA inside the vendored kernel.
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    attn_mode = 'flash' if b == 'flash' else 'torch'
    try:
        n = 0
        for m in trainer.transformer.modules():
            if hasattr(m, "attn_mode"):
                m.attn_mode = attn_mode
                n += 1
        print(f"{trainer.log_prefix} [OK] Anima attention backend '{b}' (attn_mode='{attn_mode}') "
              f"set on {n} block(s)")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set Anima attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def encode_prompt(trainer, prompt: str, qwen3_max_length: int = 512,
                  t5_max_length: int = 512):
    """Encode prompt for Anima using the Phase A/B inference pipeline.

    VERBATIM body of ``BaseTrainer.encode_prompt_anima`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    from core.models.anima.anima_pipeline_ops import encode_prompt as _encode

    # Reuse the inference encode_prompt — it already handles Qwen3 hidden-state
    # extraction, T5 tokenisation for the LLM Adapter, and zero-masking.
    # Phase B.1-e added A1111-style emphasis support there which is
    # intentionally NOT applied during training (captions go through raw).
    encoded = _encode(
        text_encoder=trainer.text_encoder,
        qwen3_tokenizer=trainer.tokenizer,
        t5_tokenizer=trainer.t5_tokenizer,
        prompt=prompt,
        device=str(trainer.device),
        dtype=trainer.training_dtype,
        qwen3_max_length=qwen3_max_length,
        t5_max_length=t5_max_length,
    )
    # encode_prompt returns batched tensors of shape [1, L, ...]; drop the
    # batch dim so caches accumulate per-sample. Detach for storage.
    return {
        "prompt_embeds": encoded["prompt_embeds"][0].detach(),
        "source_mask": encoded["source_mask"][0].detach(),
        "t5_input_ids": encoded["t5_input_ids"][0].detach(),
        "t5_attn_mask": encoded["t5_attn_mask"][0].detach(),
    }


def collate_aux(trainer, aux_list):
    """Collate a list of per-item Anima auxiliary dicts into ONE dict of
    batched tensors {source_mask, t5_input_ids, t5_attn_mask}, each [B, L].

    VERBATIM body of ``BaseTrainer._collate_anima_aux`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    keys = ("source_mask", "t5_input_ids", "t5_attn_mask")
    if not aux_list:
        raise ValueError("[Anima collation] empty auxiliary_data_list")
    for idx, aux in enumerate(aux_list):
        if not isinstance(aux, dict):
            raise ValueError(
                f"[Anima collation] item {idx} auxiliary data is "
                f"{type(aux).__name__}, expected a dict with keys {keys}"
            )
        for k in keys:
            if k not in aux or not isinstance(aux[k], torch.Tensor):
                raise ValueError(
                    f"[Anima collation] item {idx} is missing tensor key '{k}' "
                    f"(got keys {list(aux.keys())})"
                )

    t5_pad_id = 0
    t5_tok = getattr(trainer, "t5_tokenizer", None)
    if t5_tok is not None and getattr(t5_tok, "pad_token_id", None) is not None:
        t5_pad_id = int(t5_tok.pad_token_id)
    pad_values = {"source_mask": 0, "t5_input_ids": t5_pad_id, "t5_attn_mask": 0}

    batched = {}
    for k in keys:
        tensors = [aux[k] for aux in aux_list]
        max_len = max(t.shape[0] for t in tensors)
        if any(t.shape[0] != max_len for t in tensors):
            padded = []
            for t in tensors:
                if t.shape[0] < max_len:
                    pad = torch.full(
                        (max_len - t.shape[0],), pad_values[k],
                        dtype=t.dtype, device=t.device,
                    )
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            tensors = padded
        batched[k] = torch.stack(tensors, dim=0)
    return batched


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """Anima VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM body of the ``is_anima`` branch (self->trainer rename only). Runs
    inside the caller's ``with torch.no_grad()``; caller does the shared final
    dtype/CPU move.
    """
    # Anima uses the Qwen-Image VAE (Wan VAE 2.1 latent space, 16ch).
    # Encode -> sample posterior -> apply latents_mean / latents_std
    # normalisation (same as anima_pipeline_ops.vae_encode_image).
    # AutoencoderKLQwenImage expects [B, C, T, H, W] (T=1 for images).
    image_tensor_5d = image_tensor.unsqueeze(2)
    latent_dist = trainer.vae.encode(image_tensor_5d).latent_dist
    latents_5d = latent_dist.sample()  # [B, 16, 1, H/8, W/8]
    from core.models.anima.anima_pipeline_ops import _get_qwen_vae_normalization
    mean_t, std_t = _get_qwen_vae_normalization(trainer.vae, latents_5d.device, latents_5d.dtype)
    latents_5d = (latents_5d - mean_t) / std_t
    # Drop the temporal dim for storage; train_step_anima re-adds it.
    latents = latents_5d.squeeze(2)
    del image_tensor_5d, latent_dist, latents_5d
    return latents


def train_step(
    trainer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    anima_aux: Dict[str, torch.Tensor],
    timesteps: Optional[torch.Tensor] = None,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[str]] = None,
    profile_vram: bool = False,
    alphas_cumprod_cached: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Single Anima training step (rectified flow / velocity prediction).

    Args:
        latents:   Normalised Qwen-Image latents [B, 16, H/8, W/8] (the
                   Anima DiT requires a singleton temporal dim, added below).
        prompt_embeds: Qwen3 hidden states [B, L_qwen, 1024], zero-masked.
        anima_aux: dict with {source_mask, t5_input_ids, t5_attn_mask}
                   as produced by encode_prompt_anima.
        timesteps: Optional pre-sampled sigma values in [0, 1]; otherwise
                   sampled via trainer.timestep_sampler or uniform random.

    Returns:
        (loss tensor, prediction loss value, reconstruction loss value)
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import print_vram_usage

    if profile_vram:
        print_vram_usage("[train_step_anima] Start")

    latents = latents.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    prompt_embeds = prompt_embeds.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    source_mask = anima_aux["source_mask"].to(device=trainer.device, non_blocking=True)
    t5_input_ids = anima_aux["t5_input_ids"].to(device=trainer.device, non_blocking=True)
    t5_attn_mask = anima_aux["t5_attn_mask"].to(device=trainer.device, non_blocking=True)

    batch_size = latents.shape[0]
    if timesteps is None:
        if trainer.timestep_sampler is not None:
            timesteps = trainer.timestep_sampler.sample(batch_size, trainer.device)
        else:
            timesteps = torch.rand(batch_size, device=trainer.device)
    timesteps = timesteps.to(trainer.training_dtype)

    noise = torch.randn_like(latents)

    # Flow-matching forward: x_t = (1 - sigma) * x_0 + sigma * noise
    sigma_view = timesteps.view(-1, *([1] * (latents.dim() - 1))).to(latents.dtype)
    noisy_latents = (1.0 - sigma_view) * latents + sigma_view * noise

    # Anima DiT requires a singleton temporal dim: [B, C, H, W] -> [B, C, 1, H, W].
    noisy_latents_5d = noisy_latents.unsqueeze(2)

    # Padding mask matches latent spatial resolution; all-valid (zeros).
    latent_h = noisy_latents.shape[-2]
    latent_w = noisy_latents.shape[-1]
    padding_mask = torch.zeros(
        (batch_size, 1, latent_h, latent_w),
        device=trainer.device, dtype=trainer.training_dtype,
    )

    if profile_vram:
        print_vram_usage("[train_step_anima] Before DiT forward")

    # TREAD token routing (arXiv 2501.04765): attach the route config to the
    # transformer for THIS training forward only, then clear it in `finally` so
    # sampling / validation always run the full network on all tokens. The
    # forward additionally gates on self.training, so this is doubly safe.
    tread_cfg = getattr(trainer, "tread_config", None)
    inner = getattr(trainer.transformer, "module", trainer.transformer)
    if tread_cfg is not None:
        inner._tread_config = tread_cfg

    # Low-rate stochastic depth (per-batch block dropout): attach for THIS training
    # forward only, clear in `finally` so sampling / validation run every block.
    # The forward additionally gates on self.training, so this is doubly safe.
    block_skip_cfg = getattr(trainer, "block_skip_config", None)
    if block_skip_cfg is not None:
        inner._block_skip_config = block_skip_cfg

    # DiT-BlockSkip (arXiv 2603.20755): attach the runtime config for THIS training
    # forward only. The forward runs a no_grad full pass to capture the skipped
    # spans' residual features for this step's exact tensors (folded precompute -
    # determinism by construction), then runs the gradient pass over the middle
    # blocks only. Residuals stay IN MEMORY: persisting one ~33MB file per step
    # (the paper's separate-phase artifact) is redundant in the folded design and
    # accumulates unbounded on disk - audit finding, removed. Cleared in `finally`
    # so sampling/validation run the full network.
    blockskip_cfg = getattr(trainer, "blockskip_config", None)
    if blockskip_cfg is not None:
        def _on_residual(df, db):
            # Detach so the skipped spans are constants in the backward graph
            # (the BlockSkip memory/compute trade); no disk round-trip.
            return df.detach(), db.detach()

        inner._blockskip_config = {
            "front": int(blockskip_cfg["front"]),
            "back": int(blockskip_cfg["back"]),
            "on_residual": _on_residual,
        }

    # The DiT forward returns velocity in 5D ([B, 16, 1, H, W]).
    try:
        if trainer.mixed_precision:
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
                model_pred = trainer.transformer(
                    x=noisy_latents_5d,
                    timesteps=timesteps,
                    context=prompt_embeds,
                    padding_mask=padding_mask,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=source_mask,
                )
        else:
            model_pred = trainer.transformer(
                x=noisy_latents_5d,
                timesteps=timesteps,
                context=prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_mask,
            )
    finally:
        if tread_cfg is not None:
            inner._tread_config = None
        if block_skip_cfg is not None:
            inner._block_skip_config = None
        if blockskip_cfg is not None:
            inner._blockskip_config = None

    # Drop the temporal dim back: [B, 16, 1, H, W] -> [B, 16, H, W].
    if model_pred.dim() == 5:
        model_pred = model_pred.squeeze(2)

    if profile_vram:
        print_vram_usage("[train_step_anima] After DiT forward")

    # Rectified flow target: v = noise - x_0  (sd-scripts anima convention,
    # matches our inference scheduler which integrates `latents + dt * v`
    # with dt = sigma_next - sigma < 0).
    target = noise - latents

    loss_per_element = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss_per_sample = loss_per_element.mean([1, 2, 3])
    mse_loss = loss_per_sample.mean()

    loss = mse_loss

    # Reconstruction loss (predicted x0 vs ground-truth x0) — optional.
    recon_loss_value = 0.0
    if trainer.reconstruction_loss_weight > 0:
        with torch.no_grad():
            pred_x0 = noisy_latents - sigma_view * model_pred  # x_0 = x_t - sigma * v
            recon_loss = F.mse_loss(pred_x0.float(), latents.float())
            recon_loss_value = recon_loss.item()
        loss = loss + trainer.reconstruction_loss_weight * recon_loss

    pred_loss_value = mse_loss.item()

    # Debug save if requested
    if debug_save_path is not None:
        try:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            timestep_value = timesteps[0].item()

            with torch.no_grad():
                # Anima noising is x_t = (1 - sigma) * x_0 + sigma * noise with the
                # target v = noise - x_0 (see above), so x_0 = x_t - sigma * v.
                # NOTE the sign: this is standard flow matching, the OPPOSITE of
                # Z-Image's x_0 = x_t + t * v (which uses v = x_0 - noise).
                predicted_latent = noisy_latents - sigma_view * model_pred
                debug_recon_per_element = F.mse_loss(
                    predicted_latent.float(), latents.float(), reduction="none")
                debug_recon_per_sample = debug_recon_per_element.mean([1, 2, 3])

            debug_data = {
                'latents': latents[0:1].detach().cpu(),
                'noisy_latents': noisy_latents[0:1].detach().cpu(),
                'predicted_velocity': model_pred[0:1].detach().cpu(),
                'actual_velocity': target[0:1].detach().cpu(),
                'predicted_latent': predicted_latent[0:1].detach().cpu(),
                'timestep': timestep_value,
                'loss': loss_per_sample[0].item(),
                'loss_batch_mean': loss.item(),
                'recon_loss': debug_recon_per_sample[0].item(),
                'recon_loss_batch_mean': debug_recon_per_sample.mean().item(),
                'batch_size': batch_size,
                'scheduler_type': 'FlowMatching',
                'model_type': 'anima',
            }

            if debug_captions is not None and len(debug_captions) > 0:
                debug_data['caption'] = debug_captions[0]
                debug_data['all_captions'] = debug_captions

            if debug_reference_image_paths is not None and len(debug_reference_image_paths) > 0:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data['reference_image_path'] = first_ref

            torch.save(debug_data, debug_save_path / f"latents_t{timestep_value:.4f}.pt")
            del predicted_latent, debug_recon_per_element, debug_recon_per_sample
        except Exception as _dbg_e:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {_dbg_e}")

    del noise, noisy_latents, noisy_latents_5d, model_pred, target
    del loss_per_element, loss_per_sample
    return loss, pred_loss_value, recon_loss_value


# ============================================================
# Anima Sample Generation (Qwen3 TE + rectified-flow DiT) (plan P7)
# ============================================================
# Verbatim body of BaseTrainer._generate_sample_anima (base_trainer.py), moved
# out of the spine with the mechanical self.->trainer. receiver rename only.
# arch/anima.py::sample() unpacks SampleContext into this.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 28,
    guidance_scale: float = 4.0,
    seed: int = -1,
    negative_prompt: str = "",
):
    """Generate a sample image during training (Anima).

    Reuses the Anima inference pipeline ops (encode_prompt / sample_txt2img /
    vae_decode_latents) directly on the trainer's own components, driven with
    the trainer's sequential-offload helpers so it survives block-swap /
    low-VRAM training layouts.  A deep-copied scheduler is used so setting the
    inference timesteps never mutates the training scheduler.
    """
    import copy as _copy
    import random as _random
    from core.models.anima.anima_pipeline_ops import (
        encode_prompt as _anima_encode_prompt,
        sample_txt2img as _anima_sample_txt2img,
        vae_decode_latents as _anima_vae_decode,
    )

    print(f"{trainer.log_prefix} Generating Anima sample: {prompt[:50]}...")
    device = trainer.device
    compute_dtype = torch.bfloat16

    # Snap to patch_spatial * vae_scale_factor (matches inference backend).
    snap = trainer.transformer.patch_spatial * 8
    height = max(snap, (height // snap) * snap)
    width = max(snap, (width // snap) * snap)

    trainer.transformer.eval()
    trainer.vae.eval()
    if trainer.text_encoder is not None:
        trainer.text_encoder.eval()

    try:
        # --- Offload transformer (+ optimizer state) to CPU for TE encode ---
        trainer.move_main_model_to_cpu()
        trainer._relocate_main_model_optimizer_state("cpu")
        torch.cuda.empty_cache()

        # --- Stage 1: text encoding ---
        trainer.move_text_encoder_to_gpu()
        cond = _anima_encode_prompt(
            trainer.text_encoder, trainer.tokenizer, trainer.t5_tokenizer,
            prompt, device=device, dtype=compute_dtype,
        )
        uncond = None
        if guidance_scale > 1.0:
            uncond = _anima_encode_prompt(
                trainer.text_encoder, trainer.tokenizer, trainer.t5_tokenizer,
                negative_prompt, device=device, dtype=compute_dtype,
            )
        trainer.move_text_encoder_to_cpu()
        torch.cuda.empty_cache()

        # --- Stage 2: denoising ---
        # Optimizer state stays on CPU for the whole sampling span (it is
        # never stepped here); it returns to GPU only at the final restore.
        trainer.move_main_model_to_gpu()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed if (seed is not None and seed >= 0)
                              else _random.randint(0, 2**32 - 1))
        sample_scheduler = _copy.deepcopy(trainer.scheduler)
        # Autocast the denoise loop to the sampling compute dtype (bf16). This is
        # unconditional (NOT gated on trainer.mixed_precision): sampling always
        # runs the DiT in bf16 (compute_dtype above), while LoRA adapters default
        # to lora_dtype=fp32 on a bf16 base. Without autocast the bf16 activations
        # hit the fp32 LoRA Linear weights and crash with a dtype mismatch inside
        # the forward — regardless of the mixed_precision flag. The training
        # forward (train_step) avoids the same crash via its own autocast branch.
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=compute_dtype):
            latents = _anima_sample_txt2img(
                transformer=trainer.transformer, scheduler=sample_scheduler,
                cond_embeds=cond, uncond_embeds=uncond,
                height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator, device=str(device), dtype=compute_dtype,
                spectrum_params={},
            )
        del cond, uncond

        # --- Stage 3: VAE decode ---
        trainer.move_main_model_to_cpu()
        torch.cuda.empty_cache()
        trainer.move_vae_to_gpu()
        with torch.no_grad():
            # VAE weights are trainer.vae_dtype (may differ from the bf16 compute
            # dtype used for denoising); match it to avoid a conv dtype mismatch.
            images = _anima_vae_decode(trainer.vae, latents.to(trainer.vae.dtype))
        trainer.move_vae_to_cpu()
        del latents

        # --- Restore transformer to GPU for continued training ---
        trainer.move_main_model_to_gpu()
        trainer._relocate_main_model_optimizer_state(device)
        torch.cuda.empty_cache()
        return images[0]

    finally:
        trainer.transformer.train()
        trainer.vae.train()
        if trainer.text_encoder is not None:
            trainer.text_encoder.train()
