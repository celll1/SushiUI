"""minit2i_ops.py — MiniT2I (pixel-space MM-JiT) loader + block-swap + attention
free functions (P3c).

VERBATIM bodies of ``BaseTrainer._load_minit2i_components``,
``BaseTrainer.setup_minit2i_block_swap`` and
``BaseTrainer._setup_attention_backend_minit2i`` (base_trainer.py), moved out of
the spine with the mechanical ``self.`` -> ``trainer.`` receiver rename only.

Construction-order note (plan P3b/P3c): the arch handler binds at the END of
``BaseTrainer.__init__`` — AFTER ``_load_model_components`` runs — so the
load-time dispatcher (and ``_load_checkpoint_as_base``) call ``load_components``
directly. ``setup_block_swap`` keeps a 2-line delegator on the trainer (called
LATE by mode subclasses via hasattr) and ``setup_attention_backend`` keeps a
delegator (called from the moved loader body); each body is defined exactly once
here.

REPA setup stays central (``BaseTrainer._setup_repa``, plan section B "optional
cross-arch"); the moved loader body calls ``trainer._setup_repa()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch


def load_components(trainer) -> None:
    """Load MiniT2I (pixel-space MM-JiT) components for training.

    Pixel-space: there is no VAE. The frozen FLAN-T5-Large is the text encoder.
    The MM-JiT transformer is loaded frozen here; LoRA / full-parameter adapters
    unfreeze the relevant parameters during setup.
    """
    print(f"{trainer.log_prefix} Detected MiniT2I model")
    print(f"{trainer.log_prefix} Loading MiniT2I components from {trainer.model_path}")

    from core.models.minit2i.minit2i_loader import load_minit2i_components
    flan_t5_path = trainer.config.get("minit2i_flan_t5_path") or None
    components = load_minit2i_components(
        model_path=trainer.model_path,
        torch_dtype=trainer.weight_dtype,
        flan_t5_path=flan_t5_path,
        text_encoder_dtype=trainer.text_encoder_dtype if hasattr(trainer, "text_encoder_dtype") else torch.float32,
        vae_dtype=trainer.vae_dtype if hasattr(trainer, "vae_dtype") else torch.float16,
        # From-scratch only: inherit compatible weights from an existing model
        # instead of pure random init (ignored for non-scratch model paths).
        scratch_init_from=(trainer.config.get("minit2i_scratch_init_from") or None),
        scratch_inherit_final_layer=bool(trainer.config.get("minit2i_inherit_final_layer", False)),
    )

    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.transformer_uncond = None
    trainer.text_encoder = components["text_encoder"]
    trainer.tokenizer = components["tokenizer"]
    trainer.scheduler = components["scheduler"]
    trainer.minit2i_variant = components.get("variant")

    # vae_type "none" = pixel-space (vae=None, RGB-direct "latent"); "sdxl"/"flux1"
    # = latent-space (a frozen AutoencoderKL encodes images to latents for training).
    trainer.minit2i_vae_type = components.get("vae_type", "none")
    trainer.minit2i_latent = trainer.minit2i_vae_type not in (None, "none")
    trainer.minit2i_noise_scale = float(getattr(trainer.transformer.mmjit_config, "noise_scale", 2.0))
    trainer.minit2i_vae_scale_factor = int(components.get("vae_scale_factor", 8))
    trainer.vae = components.get("vae")  # None for pixel
    if trainer.vae is not None:
        trainer.vae = trainer.vae.to(dtype=trainer.vae_dtype)
        trainer.vae.requires_grad_(False)
        trainer.vae.eval()
        # High-res latent caching encodes full bucket-resolution images (up to
        # ~2048px) through the VAE. A single fp32 encode at that size peaks at
        # tens of GB (early full-res conv stages + the bottleneck spatial
        # self-attention), independent of the tiny transformer step. Tiled
        # encode/decode splits the image into overlapping tiles so VAE memory
        # is bounded by the tile size, not the image size.
        for _m in ("enable_tiling", "enable_slicing"):
            if hasattr(trainer.vae, _m):
                try:
                    getattr(trainer.vae, _m)()
                except Exception as _e:
                    print(f"{trainer.log_prefix} VAE {_m} failed: {_e}")
        print(f"{trainer.log_prefix} MiniT2I VAE tiling/slicing enabled (bounds high-res encode VRAM)")
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.unet = None
    trainer.noise_scheduler = trainer.scheduler

    # Gradient checkpointing on the MM-JiT transformer.
    if trainer.gradient_checkpointing and hasattr(trainer.transformer, "enable_gradient_checkpointing"):
        try:
            trainer.transformer.enable_gradient_checkpointing()
            print(f"{trainer.log_prefix} Gradient checkpointing enabled for MiniT2I transformer")
        except Exception as e:
            print(f"{trainer.log_prefix} grad checkpoint enable failed: {e}")
    elif not trainer.gradient_checkpointing:
        print(f"{trainer.log_prefix} Gradient checkpointing disabled by config (MiniT2I)")

    # Freeze everything; adapters unfreeze what they train.
    trainer.text_encoder.requires_grad_(False)
    trainer.transformer.requires_grad_(False)

    # Keep the transformer in train() mode for the whole run: MM-JiT gates
    # gradient checkpointing on `self.training`, and it has no dropout/BN so
    # train mode has no other effect. Without this, checkpointing would only
    # activate after the first sample (its finally restores train mode), and a
    # run with samples disabled would store ALL activations -> high-res OOM.
    # The frozen FLAN-T5 stays in eval() (it has dropout) — loader set it.
    trainer.transformer.train()

    # Block-swap deferred until after adapter setup.
    trainer.layer_offload_conductor = None
    if trainer.blocks_to_swap > 0:
        print(f"{trainer.log_prefix} Block Swap requested ({trainer.blocks_to_swap} blocks); "
              f"deferred until adapter setup completes")

    print(f"{trainer.log_prefix} Moving MiniT2I transformer to {trainer.device}")
    trainer.transformer.to(trainer.device)

    # Setup attention backend if non-native (use_flash_attention is derived from it)
    if trainer.use_flash_attention:
        trainer._setup_attention_backend_minit2i(trainer.attention_backend)

    # REPA (representation alignment): load frozen encoder + build the trainable
    # projector BEFORE adapter/optimizer setup so its params join the optimizer.
    trainer._setup_repa()

    print(f"{trainer.log_prefix} MiniT2I model loaded successfully (variant={trainer.minit2i_variant})")


def setup_block_swap(trainer) -> None:
    """Initialise LayerOffloadConductor over the MM-JiT double_blocks, AFTER adapter setup."""
    if not trainer.is_minit2i:
        return
    if trainer.blocks_to_swap <= 0:
        return
    if getattr(trainer, "layer_offload_conductor", None) is not None:
        return
    double_blocks = trainer.transformer.model.net.double_blocks

    from core.memory_management import LayerOffloadConductor
    print(f"{trainer.log_prefix} [block-swap] initialising LayerOffloadConductor "
          f"(blocks_to_swap={trainer.blocks_to_swap}, pinned_memory={trainer.use_pinned_memory})")
    trainer.layer_offload_conductor = LayerOffloadConductor(
        layers=double_blocks,
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
    print(f"{trainer.log_prefix} [block-swap] LayerOffloadConductor hooks registered for MiniT2I")


def setup_attention_backend(trainer, backend: str):
    """Set the attention backend for MiniT2I models (training hook).

    MiniT2I routes all attention through the vendored ``mem_efficient_sdpa``
    primitive. Stage-B extends that primitive to read a transformer attr and
    delegate to the unified conduit; this hook stamps the canonical backend
    string on ``transformer._attn_backend`` for training and honors the
    training guard (sage -> native).
    """
    if trainer.transformer is None:
        print(f"{trainer.log_prefix} WARNING: Transformer not loaded, skipping attention backend setup")
        return
    b = trainer._resolve_training_backend(backend)
    try:
        # MMJiT.forward reads the net-level attr (transformer.model.net._attn_backend)
        # and fans it out to every attention-bearing block; the outer wrapper attr
        # alone is not consulted. Stamp BOTH (mirrors the inference plumbing in
        # pipeline_backends/minit2i.py) so flash actually engages in training.
        trainer.transformer._attn_backend = b
        net = getattr(getattr(trainer.transformer, "model", None), "net", None)
        if net is not None:
            net._attn_backend = b
        print(f"{trainer.log_prefix} [OK] MiniT2I attention backend set to '{b}'")
    except Exception as e:
        print(f"{trainer.log_prefix} WARNING: Failed to set MiniT2I attention backend '{b}': {e}")
        print(f"{trainer.log_prefix} Ensure flash-attn is installed for flash: pip install flash-attn")


def encode_prompt(trainer, prompt: str, requires_grad: bool = False):
    """Encode prompt for MiniT2I: FLAN-T5-Large last_hidden_state + attention mask.

    VERBATIM body of ``BaseTrainer.encode_prompt_minit2i`` (plan P4), moved out of
    the spine with the mechanical ``self.`` -> ``trainer.`` rename only.
    """
    prompt_length = int(trainer.transformer.mmjit_config.prompt_length)
    te_device = trainer.text_encoder.device if hasattr(trainer.text_encoder, "device") else trainer.device

    if not requires_grad:
        from core.models.minit2i.minit2i_pipeline_ops import encode_prompt as _encode
        embeds, mask = _encode(trainer.text_encoder, trainer.tokenizer, prompt, prompt_length, te_device)
        return embeds.detach().to("cpu"), mask[0].detach().to("cpu")

    # TE training: grad-enabled forward (no torch.no_grad), keep on GPU.
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    toks = trainer.tokenizer(
        prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=prompt_length,
    )
    input_ids = toks.input_ids.to(te_device)
    attn = toks.attention_mask.to(te_device)
    embeds = trainer.text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state  # [1, L, 1024]
    return embeds, attn[0]  # mask [L]


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None,
               vae_device=None, debug_preprocessing=False):
    """MiniT2I VAE-encode branch of ``BaseTrainer.encode_image`` (P5).

    VERBATIM bodies of the TWO ``is_minit2i`` early-return branches (pixel-space
    no-VAE + latent-space), self->trainer rename only. Unlike the other archs
    this is FULLY self-contained: the caller dispatches it BEFORE the shared VAE
    staging (pixel-space has no VAE, so ``next(self.vae.parameters())`` must not
    run), and it returns the final CPU/training-dtype tensor directly (no shared
    post-amble). ``minit2i_latent`` selects which sub-branch fires.
    """
    if trainer.is_minit2i and not getattr(trainer, "minit2i_latent", False):
        # Pixel-space: there is no VAE. The "latent" IS the [-1,1] RGB image
        # [1, 3, H, W]. Stored on CPU in training dtype like every other path.
        return image_tensor.to(device="cpu", dtype=trainer.training_dtype)

    if trainer.is_minit2i and getattr(trainer, "minit2i_latent", False):
        # Latent-space: VAE-encode the [-1,1] RGB image to a normalized latent
        # [1, C, H/vsf, W/vsf]. The frozen VAE is moved to GPU for caching by the
        # latent-cache pre-pass (move_vae_to_gpu).
        from core.models.minit2i.minit2i_vae import normalize_latent
        vae_device = next(trainer.vae.parameters()).device
        px = image_tensor.to(device=vae_device, dtype=trainer.vae_dtype)
        with torch.no_grad():
            sample = trainer.vae.encode(px).latent_dist.sample()
            latent = normalize_latent(sample, trainer.vae)
        return latent.to(device="cpu", dtype=trainer.training_dtype)


def apply_cfg_null_collated(
    conditioning: torch.Tensor,
    auxiliary: torch.Tensor,
    drop_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rewrite the ``drop_mask`` rows of a collated MiniT2I batch into the
    condition its inference CFG uncond branch builds.

    ``_predict_x0_cfg`` builds that branch as ``u_text=text``,
    ``u_mask=zeros_like(mask)``: the text tensor is REUSED, and
    ``MMJiT.forward`` then replaces every masked row with the learned
    ``mask_token`` before anything reads it -- including the ``context.mean``
    that feeds the pooled embedder. So zeroing the mask alone is exact, and
    zeroing the text as well would only change a tensor the forward discards.

    Out of place: ``conditioning``/``auxiliary`` belong to the assembled batch
    and are handed to every MNT iteration, so an in-place write would leak one
    iteration's null into the next.
    """
    if drop_mask is None:
        return conditioning, auxiliary
    selected = drop_mask.to(device=auxiliary.device, dtype=torch.bool)
    if not bool(selected.any()):
        return conditioning, auxiliary
    rewritten = auxiliary.clone()
    rewritten[selected] = 0
    return conditioning, rewritten


def train_step(
    trainer,
    images: torch.Tensor,
    text_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[Optional[str]]] = None,
    repa_pixels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float]:
    """Single MiniT2I training step (pixel-space flow matching, x0 prediction).

    VERBATIM body of ``BaseTrainer.train_step_minit2i`` (P6c; ``self.`` ->
    ``trainer.`` receiver rename + sanctioned lazy base_trainer import only).
    See the original docstring for the (0,1) flow convention and CFG label drop.
    """
    # Lazy import (sibling-ops pattern): keep base_trainer out of module top level.
    from core.training.base_trainer import print_vram_usage

    # noise_scale: 2.0 for pixel-space [-1,1] images, 1.0 for unit-variance VAE
    # latents (from the model config, set per vae_type).
    noise_scale = float(getattr(trainer, "minit2i_noise_scale", 2.0))

    if profile_vram:
        print_vram_usage("[train_step_minit2i] Start")

    images = images.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    text_embeds = text_embeds.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    attention_mask = attention_mask.to(device=trainer.device, non_blocking=True)

    B = images.shape[0]

    # Timesteps come from the shared, config-driven sampler (drawn once in the
    # main loop and passed in) so the UI's timestep_sampling controls MiniT2I too.
    # MiniT2I's convention is t=1 data, t=0 noise; the per-arch default
    # (param_defaults.TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["minit2i"]) is
    # logit_normal(mean=-0.8, std=0.8), reproducing the reference lognorm schedule
    # (low t = noise side). Fall back to the vendored scheduler only if no
    # timesteps were provided (e.g. a direct unit-test call).
    if timesteps is not None:
        t = timesteps.to(device=trainer.device, dtype=trainer.training_dtype)
    else:
        t = trainer.scheduler.sample_train_timesteps(B, trainer.device, dtype=trainer.training_dtype)
    t_img = t.view(-1, 1, 1, 1)

    noise = torch.randn_like(images) * noise_scale
    x_t = images * t_img + noise * (1.0 - t_img)
    denom = torch.clamp(1.0 - t_img, min=0.05)
    target = (images - x_t) / denom  # ground-truth velocity

    # CFG label drop already applied: the arch handler rewrote the dropped rows'
    # mask through apply_cfg_null_collated before this call, from the one
    # per-batch Bernoulli the trainer draws outside the MNT loop. No draw here --
    # a second one would give the same item two meanings in one pass and would
    # survive an explicit cfg_uncond_drop_rate=0.0.
    mask_eff = attention_mask

    t_dtype = trainer.transformer.dtype

    def _forward():
        return trainer.transformer(
            x_t.to(t_dtype),
            t.to(t_dtype),
            text_embeds.to(t_dtype),
            mask_eff,
        )

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            x0_pred = _forward()
    else:
        x0_pred = _forward()

    v_pred = (x0_pred.float() - x_t.float()) / denom.float()
    loss = torch.nn.functional.mse_loss(v_pred, target.float(), reduction="mean")

    pred_loss_value = loss.item()
    # Reconstruction loss (monitoring only, no gradients): unweighted MSE of the
    # predicted clean image (x0) vs the target image. This is a cleaner quality
    # signal than the (1-t)-reweighted velocity objective used for backward.
    with torch.no_grad():
        recon_loss_value = torch.nn.functional.mse_loss(
            x0_pred.float(), images.float(), reduction="mean"
        ).item()

    # REPA (representation alignment): align the DiT image hidden state captured
    # at the tap depth with frozen clean-image patch features, via the trainable
    # projector. Added to the backward loss; pred/recon above stay diffusion-only.
    # The tap (transformer.model.net._repa_tap_out) is grad-connected (it is the
    # double-block loop output, gradient-checkpoint safe).
    if getattr(trainer, "repa_enable", False) and repa_pixels is not None:
        trainer._ensure_repa_on_device()
        net = trainer.transformer.model.net
        tap = getattr(net, "_repa_tap_out", None)
        if tap is not None:
            from core.training.repa import encode_repa_targets, repa_loss as _repa_loss_fn
            patch = int(trainer.transformer.mmjit_config.patch_size)
            gh = images.shape[2] // patch
            gw = images.shape[3] // patch
            targets = encode_repa_targets(
                trainer.repa_encoder,
                repa_pixels.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True),
                gh, gw, trainer.repa_size,
            )
            rloss = _repa_loss_fn(tap, targets, trainer.repa_projector)
            loss = loss + trainer.repa_weight * rloss
            trainer.log_extra_metric("repa_loss", float(rloss.detach().item()))
            del targets
        # Release the captured graph reference (avoid retaining the activation graph).
        net._repa_tap_out = None

    # Debug save: dump the first sample's tensors (.pt) so the noising / x0
    # prediction can be inspected offline. For the latent variant ("latent" is
    # a VAE code) also decode the target and prediction back to RGB PNGs via
    # the VAE so the encode/decode round-trip is visually verifiable.
    if debug_save_path is not None:
        try:
            debug_save_path.mkdir(parents=True, exist_ok=True)
            t_val = float(t[0].item())
            is_latent = bool(getattr(trainer, "minit2i_latent", False))
            will_decode = is_latent and getattr(trainer, "vae", None) is not None
            # .pt always carries the scalar metrics (timestep/losses/caption) the
            # visualize endpoint reads. When we also write decoded webp previews
            # (latent variant), the heavy latent tensors are NOT stored — the webp
            # is the display source, so this keeps debug .pt tiny (~1KB vs ~2MB)
            # over long runs. Pixel / no-VAE runs keep the tensors so the
            # false-color latent_to_image fallback still works.
            debug_data = {
                "timestep": t_val,
                "noise_scale": noise_scale,
                "model_type": "minit2i",
                "vae_type": getattr(trainer, "minit2i_vae_type", "none"),
                "is_latent": is_latent,
                "loss": loss.item(),
                "recon_loss": recon_loss_value,
                "batch_size": B,
            }
            if not will_decode:
                # Standard tensor keys the visualize endpoint false-colors
                # (latents=Target, predicted_latent=Predicted t=0).
                debug_data["latents"] = images[0:1].detach().cpu()
                debug_data["noisy_latents"] = x_t[0:1].detach().cpu()
                debug_data["predicted_latent"] = x0_pred[0:1].detach().cpu()
                debug_data["predicted_velocity"] = v_pred[0:1].detach().cpu()
            if debug_captions:
                debug_data["caption"] = debug_captions[0]
            if debug_reference_image_paths:
                first_ref = next((p for p in debug_reference_image_paths if p is not None), None)
                if first_ref:
                    debug_data["reference_image_path"] = first_ref
            torch.save(debug_data, debug_save_path / f"latents_t{t_val:.4f}.pt")

            if will_decode:
                # Decode the noisy x_t, the predicted x0, and the target latent to
                # RGB for a visual sanity check / 3-way comparison (noisy ⇔
                # predicted ⇔ target). VAE tiling keeps this cheap at any res.
                from core.models.minit2i.minit2i_vae import denormalize_latent
                from PIL import Image as _Image
                vae_dev = next(trainer.vae.parameters()).device
                with torch.no_grad():
                    for name, lat in (("noisy", x_t[0:1]), ("target", images[0:1]), ("pred_x0", x0_pred[0:1])):
                        z = denormalize_latent(lat.to(device=vae_dev, dtype=trainer.vae_dtype), trainer.vae)
                        img = trainer.vae.decode(z).sample  # [1,3,H,W] in ~[-1,1]
                        arr = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                        arr = arr.permute(1, 2, 0).cpu().numpy()
                        # WebP (lossy q80) — debug previews accumulate (every N
                        # steps x 2 images); far smaller than PNG, fine for visual checks.
                        _Image.fromarray(arr).save(
                            debug_save_path / f"decode_t{t_val:.4f}_{name}.webp",
                            "WEBP", quality=80, method=4,
                        )
        except Exception as _dbg_e:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {_dbg_e}")

    # Backward is performed by _execute_forward_backward; do not backward here.
    del noise, x_t, target, v_pred, x0_pred, denom
    return loss, pred_loss_value, recon_loss_value


# ============================================================
# MiniT2I Sample Generation (pixel-space, no VAE) (plan P7)
# ============================================================
# Verbatim body of BaseTrainer._generate_sample_minit2i (base_trainer.py), moved
# out of the spine with the mechanical self.->trainer. receiver rename and the
# relocated .optimizers -> ..optimizers relative import. arch/minit2i.py::sample()
# unpacks SampleContext into this.


def generate_sample(
    trainer,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 100,
    guidance_scale: float = 6.0,
    seed: int = -1,
    negative_prompt: str = "",
):
    """Generate a sample during MiniT2I training (pixel-space flow matching).

    No VAE: the model output IS the [-1,1] RGB image. Aligns the requested
    resolution to a multiple of 16 (patch size) like the inference path.
    """
    from core.models.minit2i.minit2i_pipeline_ops import (
        encode_prompt as _mt_encode, denoise_loop as _mt_denoise,
        tensor_to_image as _mt_to_image, normalize_resolution as _mt_norm,
        vae_decode_latent as _mt_vae_decode,
    )

    print(f"{trainer.log_prefix} Generating MiniT2I sample: {prompt[:50]}...")
    width, height = _mt_norm(width, height)
    is_latent = getattr(trainer, "minit2i_latent", False)
    vsf = getattr(trainer, "minit2i_vae_scale_factor", 8)
    noise_scale = float(getattr(trainer, "minit2i_noise_scale", 2.0))

    trainer.transformer.eval()
    trainer.text_encoder.eval()
    transformer_device = next(trainer.transformer.parameters()).device
    text_encoder_device = next(trainer.text_encoder.parameters()).device
    t_dtype = trainer.transformer.dtype
    cfg = trainer.transformer.mmjit_config
    prompt_length = int(cfg.prompt_length)
    cfg_interval = tuple(cfg.cfg_interval)

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

        if text_encoder_device != trainer.device:
            trainer.text_encoder.to(trainer.device)
        text, mask = _mt_encode(trainer.text_encoder, trainer.tokenizer, prompt, prompt_length, trainer.device)
        if guidance_scale != 1.0 and negative_prompt:
            neg_text, neg_mask = _mt_encode(
                trainer.text_encoder, trainer.tokenizer, negative_prompt, prompt_length, trainer.device)
        else:
            neg_text, neg_mask = None, None
        if text_encoder_device != trainer.device:
            trainer.text_encoder.to(text_encoder_device)
        torch.cuda.empty_cache()

        trainer.transformer.to(transformer_device)
        torch.cuda.empty_cache()

        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            if is_latent:
                x = _mt_denoise(
                    trainer.transformer, text.to(t_dtype), mask,
                    height // vsf, width // vsf, num_inference_steps, guidance_scale, cfg_interval,
                    trainer.device, t_dtype, seed=seed if seed >= 0 else None,
                    neg_text=neg_text.to(t_dtype) if neg_text is not None else None,
                    neg_mask=neg_mask,
                    channels=int(cfg.in_channels), noise_scale=noise_scale, clamp_output=False,
                )
            else:
                x = _mt_denoise(
                    trainer.transformer, text.to(t_dtype), mask,
                    height, width, num_inference_steps, guidance_scale, cfg_interval,
                    trainer.device, t_dtype, seed=seed if seed >= 0 else None,
                    neg_text=neg_text.to(t_dtype) if neg_text is not None else None,
                    neg_mask=neg_mask,
                )
        if is_latent:
            trainer.vae.to(trainer.device)
            image = _mt_vae_decode(trainer.vae, x.float())
            trainer.vae.to("cpu")
        else:
            image = _mt_to_image(x.float())
        del text, mask, x
        if neg_text is not None:
            del neg_text, neg_mask
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
