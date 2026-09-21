"""Training math and component loading for Qwen-Image 2.1."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def load_components(trainer) -> None:
    from core.models.qwen_image_21.loader import build_pipeline, load_qwen_image_21_components

    components = load_qwen_image_21_components(
        trainer.model_path, torch_dtype=trainer.weight_dtype, load_text_encoder=True
    )
    trainer.transformer = components["transformer"]
    trainer.transformer_original = trainer.transformer
    trainer.vae = components["vae"].to(dtype=trainer.vae_dtype)
    trainer.text_encoder = components["text_encoder"]
    trainer.processor = components["processor"]
    trainer.scheduler = components["scheduler"]
    trainer.noise_scheduler = trainer.scheduler
    trainer.unet = None
    trainer.text_encoder_2 = None
    trainer.tokenizer = trainer.processor.tokenizer
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.qwen_image_21_pipeline = build_pipeline(components)
    # A resumed full checkpoint may itself name another checkpoint.  Preserve
    # the loader's terminal component source so retention cannot break the next
    # save by deleting an intermediate checkpoint in that chain.
    trainer.qwen_image_21_companion_path = str(
        components.get("companion_path", trainer.model_path)
    )

    trainer.vae.requires_grad_(False).eval()
    trainer.text_encoder.requires_grad_(False).eval()
    trainer.transformer.requires_grad_(False)
    if trainer.gradient_checkpointing:
        trainer.transformer.enable_gradient_checkpointing()
    trainer.transformer.to(trainer.device)
    trainer.layer_offload_conductor = None
    setup_attention_backend(trainer, trainer.attention_backend)


def setup_attention_backend(trainer, backend: str) -> None:
    """Install Qwen 2.1's exact segmented native or packed Flash path."""
    if trainer.transformer is None:
        return
    resolved = trainer._resolve_training_backend(backend)
    if resolved not in {"native", "flash"}:
        raise ValueError(
            f"Qwen-Image 2.1 training cannot dispatch attention backend {resolved!r}; "
            "supported backends are 'native' and 'flash'"
        )
    count = 0
    for block in trainer.transformer.transformer_blocks:
        processor = block.attn.processor
        processor._attention_backend = resolved
        count += 1
    print(
        f"{trainer.log_prefix} [OK] Qwen-Image 2.1 attention backend='{resolved}' "
        f"({count} segmented processors)"
    )


def setup_block_swap(trainer) -> None:
    if trainer.blocks_to_swap <= 0 or trainer.layer_offload_conductor is not None:
        return
    if not trainer.gradient_checkpointing:
        raise ValueError("Qwen-Image 2.1 block swap requires gradient_checkpointing=True")
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
        ring_size=trainer.block_swap_ring_size,
    )
    trainer.transformer._layer_offload_conductor = trainer.layer_offload_conductor
    trainer.layer_offload_conductor.register_hooks()


def encode_prompt(trainer, prompt: str):
    pipe = trainer.qwen_image_21_pipeline
    te_device = next(trainer.text_encoder.parameters()).device
    with torch.no_grad():
        embeds, mask, _ = pipe.encode_prompt(prompt=prompt, device=te_device)
    if mask is None:
        mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
    return embeds.detach().cpu(), mask[0].detach().cpu()


def vae_encode(trainer, image_tensor, *, image=None, width=None, height=None, vae_device=None, **_kwargs):
    pipe = trainer.qwen_image_21_pipeline
    if image is not None:
        tensor = pipe.image_processor.preprocess(
            image.convert("RGBA"), width=int(width), height=int(height)
        )
    else:
        tensor = image_tensor
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] == 3:
            tensor = torch.cat([tensor, torch.ones_like(tensor[:, :1])], dim=1)
        if tensor.shape[1] != 4:
            raise ValueError(f"Qwen-Image 2.1 VAE expects RGBA input, got {tensor.shape[1]} channels")
    tensor = tensor.to(device=vae_device, dtype=trainer.vae_dtype).unsqueeze(2)
    posterior = trainer.vae.encode(tensor).latent_dist
    latents = posterior.sample()
    mean = torch.tensor(trainer.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(trainer.vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    latents = (latents - mean.view(1, -1, 1, 1, 1)) / std.view(1, -1, 1, 1, 1)
    batch, channels, _, latent_h, latent_w = latents.shape
    return latents.view(batch, channels, latent_h * latent_w).transpose(1, 2)


@torch.no_grad()
def vae_decode(trainer, latents, *, latent_h: int, latent_w: int):
    latents = latents.to(device=next(trainer.vae.parameters()).device, dtype=trainer.vae_dtype)
    batch, tokens, channels = latents.shape
    if tokens != latent_h * latent_w:
        raise ValueError(
            f"Qwen-Image 2.1 latent grid {latent_h}x{latent_w} does not match {tokens} tokens"
        )
    latents = latents.transpose(1, 2).reshape(batch, channels, 1, latent_h, latent_w)
    mean = torch.tensor(trainer.vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    std = torch.tensor(trainer.vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    latents = latents * std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
    return trainer.vae.decode(latents, return_dict=False)[0][:, :, 0]


def train_step(
    trainer,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    latent_h: Optional[int] = None,
    latent_w: Optional[int] = None,
    **_kwargs,
):
    latents = latents.to(trainer.device, trainer.training_dtype)
    encoder_features = encoder_features.to(trainer.device, trainer.training_dtype)
    encoder_mask = encoder_mask.to(trainer.device)
    batch, tokens, _ = latents.shape
    if latent_h is None or latent_w is None:
        side = int(tokens**0.5)
        if side * side != tokens:
            raise ValueError("Qwen-Image 2.1 non-square latent requires latent_h/latent_w")
        latent_h = latent_w = side
    if timesteps is None:
        timesteps = trainer.timestep_sampler.sample(batch, trainer.device) if trainer.timestep_sampler else torch.rand(batch, device=trainer.device)
    sigma = timesteps.to(device=trainer.device, dtype=trainer.training_dtype)
    sigma_view = sigma.view(-1, 1, 1)
    from core.training.mnt import training_noise_like

    noise = training_noise_like(trainer, latents)
    noisy = (1 - sigma_view) * latents + sigma_view * noise
    target = noise - latents
    prefix_mask = torch.zeros(
        encoder_features.shape[:2], dtype=torch.bool, device=trainer.device
    )
    image_mask = torch.cat(
        [prefix_mask, torch.ones(batch, tokens // 4, dtype=torch.bool, device=trainer.device)], dim=1
    )
    shapes = [[(1, int(latent_h), int(latent_w))]] * batch

    def forward():
        return trainer.transformer(
            hidden_states=noisy,
            timestep=sigma,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            img_shapes=shapes,
            img_mask=image_mask,
            return_dict=False,
        )[0]

    if trainer.mixed_precision:
        with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
            prediction = forward()
    else:
        prediction = forward()
    prediction = prediction[:, -tokens:]
    loss = F.mse_loss(prediction.float(), target.float())
    return loss, float(loss.detach()), 0.0


@torch.no_grad()
def generate_sample(
    trainer,
    *,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str = "",
    step_progress_callback=None,
):
    """Generate a validation image with the same pipeline used for inference."""
    from accelerate.hooks import remove_hook_from_module

    pipe = trainer.qwen_image_21_pipeline
    was_training = trainer.transformer.training
    transformer_device = next(trainer.transformer.parameters()).device
    generator = torch.Generator(device="cpu")
    if seed is not None and seed >= 0:
        generator.manual_seed(seed)

    def callback(_pipe, step, _timestep, kwargs):
        if step_progress_callback is not None:
            step_progress_callback(step + 1, num_inference_steps)
        return kwargs

    optimizer = getattr(trainer, "optimizer", None)

    def move_optimizer_state(device):
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    trainer.transformer.eval()
    try:
        move_optimizer_state("cpu")
        pipe.enable_model_cpu_offload(device=trainer.device)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            true_cfg_scale=float(guidance_scale),
            height=max(32, int(height) // 32 * 32),
            width=max(32, int(width) // 32 * 32),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
            callback_on_step_end=callback,
            use_kv_cache=True,
        )
        return result.images[0]
    finally:
        for module in (trainer.transformer, trainer.text_encoder, trainer.vae):
            remove_hook_from_module(module, recurse=True)
        trainer.text_encoder.to("cpu")
        trainer.vae.to("cpu")
        trainer.transformer.to(transformer_device)
        move_optimizer_state(transformer_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trainer.transformer.train(was_training)
