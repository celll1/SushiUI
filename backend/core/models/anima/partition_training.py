"""Deterministic native Anima image partition plans."""
from dataclasses import replace
from contextlib import nullcontext
import math

import torch

from api.param_defaults import TRAINING_DEFAULTS
from core.models.components.latent_adapter import topology_alignment
from core.training.partition.core import PartitionBox, PartitionRegion, validate_partition_regions
from .partition import AnimaPartitionAdapter


def config_value(config, name):
    canonical, legacy = "dit_partition_" + name, "qwen_partition_" + name
    a, b = config.get(canonical), config.get(legacy)
    if a is not None and b is not None and a != b:
        raise ValueError(f"{canonical} conflicts with {legacy}")
    return a if a is not None else b if b is not None else TRAINING_DEFAULTS[canonical]


def validate_training_options(config, *, method=None):
    if config.get("blocks_to_swap", TRAINING_DEFAULTS["blocks_to_swap"]):
        if not config.get("gradient_checkpointing", TRAINING_DEFAULTS["gradient_checkpointing"]):
            raise ValueError("Anima Block Swap requires gradient checkpointing")
        if config.get("use_ema", TRAINING_DEFAULTS["use_ema"]):
            raise ValueError("Anima Block Swap fused backward does not support EMA")
        if config_value(config, "training_enabled"):
            raise ValueError("Anima partition training requires conventional backward; disable Block Swap")
    if config.get("dit_partition_global_adapter_enabled", TRAINING_DEFAULTS["dit_partition_global_adapter_enabled"]):
        if method is not None and method != "full_finetune":
            raise ValueError("Anima trainable global summaries currently require full_finetune")
        if not config_value(config, "training_enabled") or not config.get("train_unet", TRAINING_DEFAULTS["train_unet"]):
            raise ValueError("Anima global summaries require partition training and train_unet")
    if not config_value(config, "training_enabled"):
        return
    for name in ("repa_enable", "tread_enable", "blockskip_enable"):
        if config.get(name, TRAINING_DEFAULTS[name]):
            raise ValueError(f"Anima partition training does not support {name}")
    if float(config.get("block_skip_rate", TRAINING_DEFAULTS["block_skip_rate"])) > 0:
        raise ValueError("Anima partition training does not support block_skip_rate")


def region_count(trainer, clean):
    if clean.ndim != 4 or any(side % 2 for side in clean.shape[-2:]):
        raise ValueError("Anima partitions require BCHW latents with even spatial sides")
    if not config_value(trainer.config, "training_enabled"):
        return 1
    mode = config_value(trainer.config, "mode")
    if mode == "fixed":
        count = config_value(trainer.config, "fixed_count")
        if type(count) is not int or count not in (2, 4):
            raise ValueError("Anima partition count must be 2 or 4")
        return count
    if mode != "token_threshold":
        raise ValueError("Anima partition mode must be fixed or token_threshold")
    limit = config_value(trainer.config, "full_frame_max_tokens")
    if type(limit) is not int or limit < 1:
        raise ValueError("Anima partition token threshold must be a positive integer")
    tokens = (clean.shape[-2] // 2) * (clean.shape[-1] // 2)
    return 1 if tokens <= limit else 4 if tokens >= 4 * limit else 2


def training_plan(trainer, clean):
    count = region_count(trainer, clean)
    if count == 1:
        raise ValueError("Anima full-frame execution has no partition plan")
    value = lambda name: config_value(trainer.config, name)
    model = getattr(trainer.transformer, "module", trainer.transformer)
    topology = getattr(model, "_residual_unet_spec", None)
    halo = value("halo_tokens")
    from core.training.ops.crop_decode_loss import crop_decode_loss_active
    if crop_decode_loss_active(trainer):
        margin = getattr(trainer, "crop_decode_loss_margin_cells",
                         trainer.config.get("crop_decode_loss_margin_cells", TRAINING_DEFAULTS["crop_decode_loss_margin_cells"]))
        halo = max(halo, 2 * math.ceil(max(0, margin) / 4))
    if (topology or {}).get("method") == "hybrid_unet_ssdit":
        from core.rewiring.hybrid_partition import resolved_partition_halo, spatial_codec_radius
        codec_radius = spatial_codec_radius(getattr(model, "vae_latent_bridge", None), model.patch_spatial)
        halo = resolved_partition_halo(topology, "main", halo, codec_radius=codec_radius)
    plan = AnimaPartitionAdapter().build_fixed_plan(
        clean.shape[-2] // 2, clean.shape[-1] // 2, count=count,
        halo=halo, seed=value("seed"),
        epoch=int(getattr(trainer, "_current_epoch", 0)),
        occurrence=int(getattr(trainer, "_current_batch_position", 0)),
        split_ratio_min=value("split_ratio_min"), split_ratio_max=value("split_ratio_max"))
    ah, aw = topology_alignment(getattr(model, "_residual_unet_spec", None))
    if plan.full_height % ah or plan.full_width % aw:
        raise ValueError("Anima partition canvas must match cumulative U-Net alignment")
    regions = tuple(PartitionRegion(region.core, PartitionBox(
        region.input.top // ah * ah, region.input.left // aw * aw,
        min(plan.full_height, (region.input.bottom + ah - 1) // ah * ah),
        min(plan.full_width, (region.input.right + aw - 1) // aw * aw)))
        for region in plan.regions)
    validate_partition_regions(plan.full_height, plan.full_width, regions)
    return replace(plan, regions=regions)


def _backward_regions(trainer, *, clean, prompt_embeds, anima_aux, sigma, noise,
                     plan, backward_scale, debug_save_path=None, debug_captions=None,
                     debug_reference_image_paths=None):
    """Accumulate exact core-weighted velocity loss without retaining tile graphs."""
    if not math.isfinite(backward_scale) or backward_scale < 0:
        raise ValueError("Anima partition backward scale must be finite and nonnegative")
    validate_training_options(trainer.config)
    clean = clean.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True)
    noise, sigma = noise.to(clean), sigma.to(clean)
    if noise.shape != clean.shape or sigma.shape != (clean.shape[0],):
        raise ValueError("Anima partition noise/sigma geometry disagrees with targets")
    if (plan.full_height * 2, plan.full_width * 2) != clean.shape[-2:]:
        raise ValueError("Anima partition plan disagrees with target canvas")
    validate_partition_regions(plan.full_height, plan.full_width, plan.regions)
    weight = float(trainer.reconstruction_loss_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Anima reconstruction loss weight must be within [0, 1]")
    prompt_embeds = prompt_embeds.to(clean, non_blocking=True)
    aux = {name: anima_aux[name].to(device=clean.device, non_blocking=True)
           for name in ("source_mask", "t5_input_ids", "t5_attn_mask")}
    sigma_view = sigma[:, None, None, None]
    noisy, target = (1 - sigma_view) * clean + sigma_view * noise, noise - clean
    model = getattr(trainer.transformer, "module", trainer.transformer)
    from .vae_alignment import noise_kwargs
    from .checkpointing import partition_scope
    crop = lambda value, box: value[..., 2*box.top:2*box.bottom, 2*box.left:2*box.right]
    shared_graph = any(value.grad_fn is not None for value in (clean, prompt_embeds, noise))
    totals = clean.new_zeros(3, dtype=torch.float32)
    pred_items = clean.new_zeros(clean.shape[0], dtype=torch.float32)
    loss_items = torch.zeros_like(pred_items)
    capture = bool(getattr(trainer, "convergence_diagnostics_enable", False))
    stitched = (torch.empty_like(clean[:1], device="cpu", dtype=torch.float32)
                if debug_save_path is not None or capture else None)
    debug_recon = clean.new_zeros((), dtype=torch.float32)
    from .partition_timing import AnimaPartitionTiming
    timing = AnimaPartitionTiming(trainer, clean)
    from core.training.ops.crop_decode_loss import crop_decode_loss_active, compute_crop_decode_loss
    selected_region = None
    regions = list(plan.regions)
    if crop_decode_loss_active(trainer):
        probabilities = torch.tensor([r.core.tokens for r in regions], dtype=torch.float64)
        selected_region = regions.pop(int(torch.multinomial(probabilities, 1)))
        regions.append(selected_region)
    with partition_scope(model, trainer.config):
        checkpoint_blocks = sum(block.gradient_checkpoint_mode != "none" for block in model.blocks) if model.training else 0
        for index, region in enumerate(regions):
            local = crop(noisy, region.input)
            context = (torch.autocast(device_type=clean.device.type, dtype=trainer.training_dtype)
                       if trainer.mixed_precision else nullcontext())
            with timing.measure("forward"), context:
                prediction = trainer.transformer(
                    x=local.unsqueeze(2), timesteps=sigma, context=prompt_embeds,
                    padding_mask=local.new_zeros(local.shape[0], 1, *local.shape[-2:]),
                    target_input_ids=aux["t5_input_ids"], target_attention_mask=aux["t5_attn_mask"],
                    source_attention_mask=aux["source_mask"], partition_box=region.input,
                    partition_full_latents=noisy.unsqueeze(2), partition_full_noise=noise.unsqueeze(2),
                    **noise_kwargs(model, crop(noise, region.input)))
            if prediction.ndim != 5 or prediction.shape[2] != 1:
                raise ValueError("Anima partition prediction must be a singleton image frame")
            local_prediction = prediction.squeeze(2)
            prediction = crop(local_prediction, region.core_in_input)
            expected = crop(target, region.core)
            if prediction.shape != expected.shape:
                raise ValueError("Anima partition prediction disagrees with its supervised core")
            if stitched is not None:
                box = region.core
                stitched[..., 2*box.top:2*box.bottom, 2*box.left:2*box.right] = prediction[:1].detach().float().cpu()
            per_item = (prediction.float() - expected.float()).square().mean((1, 2, 3))
            with torch.set_grad_enabled(weight > 0):
                recovered = crop(noisy, region.core).float() - sigma_view.float() * prediction.float()
                recon_items = (recovered - crop(clean, region.core).float()).square().mean((1, 2, 3))
                debug_recon += region.core.tokens / plan.full_tokens * recon_items.mean().detach()
                del recovered
            combined = (1 - weight) * per_item + weight * recon_items
            fraction = region.core.tokens / plan.full_tokens
            region_loss = combined.mean() * fraction
            aux_loss = None
            if region is selected_region:
                box = region.core_in_input
                aux_loss, _ = compute_crop_decode_loss(
                    trainer, local_prediction, local, sigma, crop(clean, region.input),
                    noise_process="flow", prediction_target="velocity",
                    noise_scheduler=getattr(trainer, "noise_scheduler", None), velocity_sign="eps_minus_x0",
                    predicted_latent=local.float() - sigma_view.float() * local_prediction.float(),
                    main_loss=combined.mean(),
                    crop_bounds=(2*box.top, 2*box.bottom, 2*box.left, 2*box.right))
                if aux_loss is not None:
                    # Area-proportional selection already weights this auxiliary estimator.
                    region_loss = region_loss + aux_loss
            scaled = region_loss * backward_scale
            retain = shared_graph and index + 1 < len(plan.regions)
            with timing.measure("backward"):
                if getattr(trainer, "use_grad_scaler", False):
                    trainer.grad_scaler.scale(scaled).backward(retain_graph=retain)
                else:
                    scaled.backward(retain_graph=retain)
            totals += torch.stack((region_loss.detach(), fraction * per_item.mean().detach(),
                                  fraction * recon_items.mean().detach()))
            pred_items += fraction * per_item.detach()
            loss_items += fraction * combined.detach()
            del prediction, local_prediction, expected, per_item, combined, scaled, recon_items, region_loss, aux_loss
    timing.finish(plan, checkpoint_blocks)
    trainer._last_loss_per_sample = pred_items
    if getattr(trainer, "_collect_concept_sample_loss", False):
        trainer._pending_concept_sample_loss = loss_items
    if stitched is not None:
        first_noisy = noisy[:1].detach().float().cpu()
        recovered = first_noisy - float(sigma[0]) * stitched
        if capture:
            from core.training.ops.crop_decode_loss import _capture_predicted_latent
            _capture_predicted_latent(trainer, recovered.to(clean), model_pred=stitched,
                noisy_latents=first_noisy, timesteps=sigma[:1], noise_process="flow",
                prediction_target="velocity", noise_scheduler=trainer.noise_scheduler,
                velocity_sign="eps_minus_x0")
        if debug_save_path is not None:
            from pathlib import Path
            path = Path(debug_save_path)
            path.mkdir(parents=True, exist_ok=True)
            first_clean = clean[:1].detach().float().cpu()
            data = dict(latents=first_clean, noisy_latents=first_noisy,
                predicted_velocity=stitched, actual_velocity=target[:1].detach().float().cpu(),
                predicted_latent=recovered, timestep=float(sigma[0]), loss=float(pred_items[0]),
                loss_batch_mean=float(totals[0]), recon_loss=float((recovered-first_clean).square().mean()),
                recon_loss_batch_mean=float(debug_recon), batch_size=int(clean.shape[0]),
                scheduler_type="FlowMatching", model_type="anima", partition_count=len(plan.regions),
                partition_grid_size=[clean.shape[-1], clean.shape[-2]],
                partition_regions=[dict(top=2*r.core.top, left=2*r.core.left,
                    bottom=2*r.core.bottom, right=2*r.core.right) for r in plan.regions])
            if debug_captions:
                data.update(caption=debug_captions[0], all_captions=list(debug_captions))
            if debug_reference_image_paths:
                reference = next((p for p in debug_reference_image_paths if p), None)
                if reference:
                    data["reference_image_path"] = reference
            torch.save(data, path / f"latents_t{float(sigma[0]):.4f}.pt")
            from .training_debug import queue_debug_previews
            queue_debug_previews(trainer, path, data)
    return tuple(float(value) for value in totals)


def backward_regions(trainer, **kwargs):
    from core.training.ops.crop_decode_loss import crop_decode_loss_active, preserve_crop_vae_residency
    if crop_decode_loss_active(trainer):
        with preserve_crop_vae_residency(trainer):
            return _backward_regions(trainer, **kwargs)
    return _backward_regions(trainer, **kwargs)
