"""Training math and component loading for Qwen-Image 2.1."""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn.functional as F


def partition_training_enabled(trainer) -> bool:
    return bool(trainer.config.get("qwen_partition_training_enabled", False))


def _partition_plan(trainer, latent_h: int, latent_w: int):
    from core.training.qwen_partition import build_fixed_partition_plan

    mode = str(trainer.config.get("qwen_partition_mode", "fixed")).strip().lower()
    if mode != "fixed":
        raise ValueError(
            "The first Qwen partitioned-training implementation supports mode='fixed'; "
            f"got {mode!r}"
        )
    key = (
        int(getattr(trainer, "_current_epoch", 0)),
        int(getattr(trainer, "_current_batch_position", 0)),
        int(latent_h),
        int(latent_w),
        int(trainer.config.get("qwen_partition_fixed_count", 2)),
        int(trainer.config.get("qwen_partition_halo_tokens", 0)),
        int(trainer.config.get("qwen_partition_seed", 0)),
    )
    cached = getattr(trainer, "_qwen_partition_plan_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    plan = build_fixed_partition_plan(
        latent_h,
        latent_w,
        count=key[4],
        halo=key[5],
        seed=key[6],
        epoch=key[0],
        occurrence=key[1],
        split_ratio_min=float(trainer.config.get("qwen_partition_split_ratio_min", 0.35)),
        split_ratio_max=float(trainer.config.get("qwen_partition_split_ratio_max", 0.65)),
    )
    trainer._qwen_partition_plan_cache = (key, plan)
    return plan


def partition_dispatch_view(trainer, latents: torch.Tensor, *, latent_h: int, latent_w: int):
    """Shape-only view used so activation dispatch sees the largest region."""
    if not partition_training_enabled(trainer):
        return latents
    plan = _partition_plan(trainer, latent_h, latent_w)
    return latents[:, : plan.largest_input_tokens]


def _resolve_partition_checkpoint_blocks(trainer, plan, original):
    configured = trainer.config.get("qwen_partition_gradient_checkpointing_blocks")
    total_blocks = len(trainer.transformer.transformer_blocks)
    if configured is not None:
        resolved = int(configured)
    elif original is None:
        return None
    else:
        original = int(original)
        if original <= 0:
            return original
        ratio = plan.largest_input_tokens / plan.full_tokens
        # Cached ConvRot becomes slower as well as larger with checkpointing
        # fully disabled. Eight blocks was the measured floor on SM89.
        measured_floor = min(8, total_blocks)
        resolved = min(original, max(measured_floor, math.ceil(original * ratio)))
    if not 0 <= resolved <= total_blocks:
        raise ValueError(
            "qwen_partition_gradient_checkpointing_blocks must be between 0 and "
            f"{total_blocks}, got {resolved}"
        )
    return resolved


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
    trainer.qwen_image_21_transformer_variant = str(
        components.get("transformer_variant", "bf16")
    )

    trainer.vae.requires_grad_(False).eval()
    trainer.text_encoder.requires_grad_(False).eval()
    trainer.transformer.requires_grad_(False)
    # ConvRot LoRA uses the packed base for forward and a resident floating
    # weight for grad_input. Dense bases retain the ordinary autograd path.
    convrot_training_forward = str(
        trainer.config.get("qwen_convrot_training_forward", "auto")
    ).strip().lower()
    if convrot_training_forward not in {"auto", "dequant", "cached_bf16"}:
        raise ValueError(
            "qwen_convrot_training_forward must be 'auto', 'dequant', or "
            "'cached_bf16', got "
            f"{convrot_training_forward!r}"
        )
    if convrot_training_forward == "auto":
        convrot_training_forward = (
            "cached_bf16"
            if trainer.qwen_image_21_transformer_variant == "int8_convrot"
            and hasattr(trainer, "lora_rank")
            and trainer.training_dtype == torch.bfloat16
            else "dequant"
        )
    if (
        convrot_training_forward == "cached_bf16"
        and trainer.qwen_image_21_transformer_variant != "int8_convrot"
    ):
        raise ValueError(
            "Qwen-Image 2.1 cached_bf16 training requires an int8_convrot "
            f"transformer, got {trainer.qwen_image_21_transformer_variant!r}"
        )
    if convrot_training_forward == "cached_bf16" and trainer.training_dtype != torch.bfloat16:
        raise ValueError(
            "Qwen-Image 2.1 cached_bf16 training requires training_dtype=bf16, "
            f"got {trainer.training_dtype}"
        )
    trainer.qwen_convrot_training_forward = convrot_training_forward
    trainer.qwen_convrot_fused_layer_count = 0
    if trainer.gradient_checkpointing:
        trainer.transformer.enable_gradient_checkpointing()
        configured_checkpoint_blocks = trainer.config.get(
            "qwen_gradient_checkpointing_blocks"
        )
        if configured_checkpoint_blocks is None and convrot_training_forward == "cached_bf16":
            resolutions = trainer.config.get("base_resolutions") or []
            maximum_resolution = max((int(value) for value in resolutions), default=1536)
            if maximum_resolution <= 1024:
                checkpoint_blocks = 16
            elif maximum_resolution <= 1536:
                checkpoint_blocks = 24
            else:
                checkpoint_blocks = len(trainer.transformer.transformer_blocks)
        else:
            checkpoint_blocks = int(
                configured_checkpoint_blocks
                if configured_checkpoint_blocks is not None
                else len(trainer.transformer.transformer_blocks)
            )
        if not 0 <= checkpoint_blocks <= len(trainer.transformer.transformer_blocks):
            raise ValueError(
                "qwen_gradient_checkpointing_blocks must be between 0 and "
                f"{len(trainer.transformer.transformer_blocks)}, got {checkpoint_blocks}"
            )
        trainer.transformer._training_gradient_checkpointing_blocks = checkpoint_blocks
        print(
            f"{trainer.log_prefix} Qwen-Image 2.1 gradient checkpointing: "
            f"{checkpoint_blocks}/{len(trainer.transformer.transformer_blocks)} blocks"
            f" ({'explicit' if configured_checkpoint_blocks is not None else 'automatic'})"
        )
    trainer.transformer.to(trainer.device)
    if convrot_training_forward == "cached_bf16":
        if trainer.blocks_to_swap > 0:
            raise ValueError(
                "Qwen-Image 2.1 cached ConvRot backward weights cannot be combined "
                "with block swap"
            )
        from core.models.common.quantized_frozen_training import (
            enable_frozen_training_cached_backward,
        )

        (
            trainer.qwen_convrot_fused_layer_count,
            trainer.qwen_convrot_backward_cache_bytes,
        ) = enable_frozen_training_cached_backward(
            trainer.transformer,
            dtype=trainer.training_dtype,
            label="Qwen-Image 2.1 training transformer",
        )
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


def train_step_partitioned_backward(
    trainer,
    *,
    latents: torch.Tensor,
    encoder_features: torch.Tensor,
    encoder_mask: torch.Tensor,
    timesteps: torch.Tensor,
    latent_h: int,
    latent_w: int,
    backward_scale: float,
) -> tuple[float, float, float]:
    """Run every loss core sequentially and accumulate one logical gradient."""
    from core.training.mnt import training_noise_like
    from core.training.qwen_partition import (
        flatten_region,
        full_canvas_position_ids,
    )

    latents = latents.to(trainer.device, trainer.training_dtype)
    encoder_features = encoder_features.to(trainer.device, trainer.training_dtype)
    encoder_mask = encoder_mask.to(trainer.device)
    batch, tokens, channels = latents.shape
    if tokens != int(latent_h) * int(latent_w):
        raise ValueError(
            f"Qwen partition latent grid {latent_h}x{latent_w} does not match {tokens} tokens"
        )
    plan = _partition_plan(trainer, int(latent_h), int(latent_w))
    sigma = timesteps.to(device=trainer.device, dtype=trainer.training_dtype)
    sigma_view = sigma.view(-1, 1, 1)
    noise = training_noise_like(trainer, latents)
    noisy = (1 - sigma_view) * latents + sigma_view * noise
    target = noise - latents
    noisy_grid = noisy.reshape(batch, int(latent_h), int(latent_w), channels)
    target_grid = target.reshape_as(noisy_grid)
    prefix_mask = torch.zeros(
        encoder_features.shape[:2], dtype=torch.bool, device=trainer.device
    )

    original_checkpoint_blocks = getattr(
        trainer.transformer, "_training_gradient_checkpointing_blocks", None
    )
    partition_checkpoint_blocks = _resolve_partition_checkpoint_blocks(
        trainer, plan, original_checkpoint_blocks
    )
    if partition_checkpoint_blocks is not None:
        trainer.transformer._training_gradient_checkpointing_blocks = partition_checkpoint_blocks

    cuda_timing = latents.is_cuda and bool(
        trainer.config.get("qwen_partition_profile", False)
    )
    forward_ms = 0.0
    backward_ms = 0.0
    logical_loss_tensor = torch.zeros((), device=trainer.device, dtype=torch.float32)
    total_input_tokens = 0
    wall_start = time.perf_counter()
    try:
        for region in plan.regions:
            input_tokens = region.input.tokens
            total_input_tokens += input_tokens
            tile = flatten_region(noisy_grid, region.input)
            tile_target = flatten_region(target_grid, region.input)
            image_mask = torch.cat(
                [
                    prefix_mask,
                    torch.ones(
                        batch,
                        input_tokens // 4,
                        dtype=torch.bool,
                        device=trainer.device,
                    ),
                ],
                dim=1,
            )
            positions = full_canvas_position_ids(
                plan.full_height, plan.full_width, region.input
            )

            if cuda_timing:
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record(torch.cuda.current_stream(trainer.device))

            def forward():
                return trainer.transformer(
                    hidden_states=tile,
                    timestep=sigma,
                    encoder_hidden_states=encoder_features,
                    encoder_hidden_states_mask=encoder_mask,
                    img_shapes=[[(1, region.input.height, region.input.width)]] * batch,
                    img_mask=image_mask,
                    target_spatial_position_ids=positions,
                    return_dict=False,
                )[0]

            if trainer.mixed_precision:
                with torch.autocast(device_type=trainer.device.type, dtype=trainer.training_dtype):
                    prediction = forward()
            else:
                prediction = forward()
            prediction = prediction[:, -input_tokens:].reshape(
                batch, region.input.height, region.input.width, channels
            )
            target_tile_grid = tile_target.reshape_as(prediction)
            local = region.core_in_input
            core_prediction = flatten_region(prediction, local)
            core_target = flatten_region(target_tile_grid, local)
            weighted_loss = (
                region.core.tokens / plan.full_tokens
            ) * F.mse_loss(core_prediction.float(), core_target.float())

            if cuda_timing:
                forward_end.record(torch.cuda.current_stream(trainer.device))
            scaled_loss = weighted_loss * float(backward_scale)
            if trainer.use_grad_scaler:
                trainer.grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if cuda_timing:
                backward_end.record(torch.cuda.current_stream(trainer.device))
                backward_end.synchronize()
                forward_ms += forward_start.elapsed_time(forward_end)
                backward_ms += forward_end.elapsed_time(backward_end)
            logical_loss_tensor = logical_loss_tensor + weighted_loss.detach()
            del tile, tile_target, prediction, target_tile_grid, core_prediction, core_target
            del weighted_loss, scaled_loss, image_mask
    finally:
        if original_checkpoint_blocks is None:
            if hasattr(trainer.transformer, "_training_gradient_checkpointing_blocks"):
                delattr(trainer.transformer, "_training_gradient_checkpointing_blocks")
        else:
            trainer.transformer._training_gradient_checkpointing_blocks = original_checkpoint_blocks

    logical_loss = float(logical_loss_tensor.item())
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    trainer.log_extra_metric("qwen_partition_count", float(len(plan.regions)))
    trainer.log_extra_metric("qwen_partition_largest_tokens", float(plan.largest_input_tokens))
    trainer.log_extra_metric("qwen_partition_total_input_tokens", float(total_input_tokens))
    if partition_checkpoint_blocks is not None:
        trainer.log_extra_metric(
            "qwen_partition_checkpoint_blocks", float(partition_checkpoint_blocks)
        )
    if cuda_timing:
        trainer.log_extra_metric("qwen_partition_forward_ms", float(forward_ms))
        trainer.log_extra_metric("qwen_partition_backward_ms", float(backward_ms))
    trainer.log_extra_metric("qwen_partition_wall_ms", float(wall_ms))
    if latents.is_cuda:
        trainer.log_extra_metric(
            "qwen_partition_peak_allocated_gb",
            float(torch.cuda.max_memory_allocated(trainer.device) / (1024 ** 3)),
        )
    return logical_loss, logical_loss, 0.0


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
