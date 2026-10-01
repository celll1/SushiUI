"""Native Ming objective bridge to shared trainer noise, timesteps and metrics."""

from contextlib import nullcontext

import torch

from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.training import flow_loss
from core.training.mnt import training_noise_like


def train_step(trainer, batch, *, timesteps=None, cfg_drop_mask=None):
    device, dtype = trainer.device, trainer.training_dtype
    clean = batch.clean.to(device=device, dtype=dtype)
    conditioning = MingConditioning(batch.conditioning.queries.to(device=device, dtype=dtype),
                                    batch.conditioning.direct.to(device=device, dtype=dtype))
    references = None if batch.reference_latents is None else [
        None if item is None else item.to(device=device, dtype=dtype) for item in batch.reference_latents]
    if timesteps is None:
        sampler = getattr(trainer, "timestep_sampler", None)
        timesteps = sampler.sample(clean.shape[0], device) if sampler is not None else torch.rand(clean.shape[0], device=device)
    sigma = timesteps.to(device=device, dtype=dtype)
    noise = training_noise_like(trainer, clean)
    weight = float(trainer.reconstruction_loss_weight)
    if not 0 <= weight <= 1:
        raise ValueError("Ming reconstruction loss weight must be within [0, 1]")
    context = torch.autocast(device_type=torch.device(device).type, dtype=dtype) if trainer.mixed_precision else nullcontext()
    with context:
        result = flow_loss(trainer.transformer, clean=clean, noise=noise, sigma=sigma,
            conditioning=conditioning, frame_weights=batch.frame_weights, reference_latents=references,
            cfg_drop_mask=cfg_drop_mask)
    # Native velocity is clean-noise, opposite to Qwen's training prediction.
    noisy = (1 - sigma.reshape(-1, 1, 1, 1, 1)) * clean + sigma.reshape(-1, 1, 1, 1, 1) * noise
    with torch.set_grad_enabled(weight > 0):
        recovered = noisy.float() + sigma.float().reshape(-1, 1, 1, 1, 1) * result.prediction.float()
        per_frame = (recovered - clean.float()).square().mean(dim=(1, 3, 4))
        weights = batch.frame_weights.to(device=device, dtype=torch.float32)
        per_item_recon = (per_frame * weights).sum(1) / weights.sum(1)
        reconstruction = per_item_recon.mean()
    loss = (1 - weight) * result.loss + weight * reconstruction
    if getattr(trainer, "_collect_concept_sample_loss", False):
        trainer._pending_concept_sample_loss = ((1 - weight) * result.per_item_loss + weight * per_item_recon).detach()
    return loss, result.loss.detach(), reconstruction.detach()


def load_components(trainer):
    from core.models.ming_image_01.loader import load_components as native_load
    from core.models.ming_image_01.training_options import resolve_ownership
    from core.models.ming_image_01.residual_unet import configure_training_topology
    components = native_load(trainer.model_path, dtype=trainer.weight_dtype, conditioner_attention="eager")
    configure_training_topology(components.transformer, trainer.config,
                                resuming=bool(getattr(trainer, "resume_from_checkpoint", None)))
    groups, rates = resolve_ownership(trainer.config)
    trainer.ming_image_01_components = components
    trainer.ming_conditioner_trainable_groups, trainer.ming_conditioner_lrs = groups, rates
    trainer.train_text_encoder = bool(groups)
    components.conditioner.set_trainable(groups)
    trainer.transformer = trainer.transformer_original = components.transformer
    trainer.vae = components.vae.to(dtype=trainer.vae_dtype)
    trainer.text_encoder, trainer.processor = components.conditioner, components.processor
    trainer.scheduler = trainer.noise_scheduler = components.scheduler
    trainer.unet = trainer.text_encoder_2 = trainer.tokenizer_2 = trainer.t5_tokenizer = None
    trainer.tokenizer = components.processor.tokenizer


def encode_prompt(trainer, prompt, *, requires_grad=False):
    from core.models.ming_image_01.prompting import prepare_inputs, encode_inputs
    width, height, resolution = trainer.ming_prompt_geometry
    prepared = prepare_inputs(trainer.processor, [prompt], references=[[]],
                              width=width, height=height, resolution=resolution)
    return encode_inputs(trainer.text_encoder, prepared, requires_grad=requires_grad)


def vae_encode(trainer, pixels, **kwargs):
    from core.models.ming_image_01.training_data import encode_frames
    if pixels.ndim == 4:
        pixels = pixels.unsqueeze(2)
    return encode_frames(trainer.vae, pixels)


def vae_decode(trainer, latents, **kwargs):
    if latents.ndim != 5:
        raise ValueError("Ming sample latents require independent [B, C, F, H, W] frames")
    result = []
    for frame in latents.unbind(2):
        value = frame.to(device=trainer.vae.device, dtype=trainer.vae.dtype).unsqueeze(2)
        shift = torch.as_tensor(trainer.vae.config.shift_factor, device=value.device, dtype=value.dtype)
        scale = torch.as_tensor(trainer.vae.config.scaling_factor, device=value.device, dtype=value.dtype)
        result.append(trainer.vae.decode(value / scale + shift, return_dict=False)[0].squeeze(2))
    return torch.stack(result, dim=2)
