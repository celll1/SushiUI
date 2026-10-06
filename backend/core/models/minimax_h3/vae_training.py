"""H3 target-native video swap and bounded paired calibration."""

from dataclasses import replace
from math import lcm

import torch

from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair, calibration_source_side
from core.models.components.video_latent_adapter import (
    VideoCodecContract, NormalizedVideoVAE, PairedVideoCalibration, paired_calibration_frames, codec_temporal_policy,
)

PENDING = "_h3_vae_calibration_source"


def apply_swap(trainer, resolved, *, handler, module=None):
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.training.ops.training_method import resolve_training_method
    from core.training.vae_swap import check_swap_method, initialization_mode, module_latent_hash
    from core.models.components.latent_io import ResizeReport
    from .video_alignment import install_video_alignment, AlignedMiniMaxH3BlockLoopWrapper

    model = trainer.transformer
    inner = getattr(model, "transformer", model)
    pair = getattr(inner, "_video_latent_alignment", None)
    if pair is not None:
        if (resolved.latent_hash != pair.declaration["target_identity"] or not pair.ready
                or not isinstance(model, AlignedMiniMaxH3BlockLoopWrapper)
                or not isinstance(trainer.vae, NormalizedVideoVAE)
                or trainer.vae.contract != inner._video_codec_contract):
            raise ValueError("Aligned H3 training requires its calibrated saved video codec; use a fresh base to change it")
        trainer.vae_identity = replace(resolved, state_dict=None)
        trainer.vae_dtype = trainer.vae.dtype
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        channels = inner.config.in_channels
        return ResizeReport((), channels, channels, channels, 0, 0)
    if module is not None:
        raise ValueError("H3 swapped checkpoints require their saved video alignment")
    check_swap_method(resolved.source, resolve_training_method(trainer))
    if not getattr(trainer, "train_unet", TRAINING_DEFAULTS["train_unet"]):
        raise ValueError("H3 VAE swaps require train_unet=true")
    if getattr(trainer, "bundle_vae", None) is False:
        raise ValueError("H3 VAE swaps require a bundled target codec")
    if initialization_mode(trainer.config, "minimax_h3") != "paired_spatial":
        raise ValueError("H3 VAE swap requires paired_spatial initialization")
    reject_quantized_base(model, model_label="H3 VAE swap")
    image = ImageCodecContract.from_resolved(resolved)
    contract = VideoCodecContract(image, resolved.scale_temporal,
        codec_temporal_policy(image.latent_rank, resolved.vae_class))
    old = trainer.vae
    source_contract = VideoCodecContract(ImageCodecContract(inner.config.in_channels,
        handler.wiring.vae_scale_factor, 5, "per_channel", 1, 3), 4, "minimax_h3_chunks_v1")
    if (old.config.get("latent_channels") != inner.config.in_channels
            or getattr(old, "spatial_compression_ratio", None) != source_contract.image.spatial_scale
            or old.config.get("clip_length") != 17 or old.config.get("token_drop") != 3):
        raise ValueError("H3 source codec differs from the native model geometry")
    source_identity = module_latent_hash(old)
    if source_identity is None:
        raise ValueError("H3 source codec latent identity is unavailable")
    # Assign resolved tensors instead of casting away their persisted precision.
    materialized = replace(resolved, path=None) if resolved.state_dict is not None and resolved.config else resolved
    target = materialized.load_module(assign=True).requires_grad_(False).eval()
    if module_latent_hash(target) != resolved.latent_hash:
        raise ValueError("H3 target codec weights/normalization differ from their resolved identity")
    if source_identity == resolved.latent_hash:
        channels = inner.config.in_channels
        return ResizeReport((), channels, channels, channels, 0, 0)
    declaration = image.declaration(source=resolved.source, source_identity=source_identity,
        target_identity=resolved.latent_hash, old_channels=inner.config.in_channels,
        old_scale=source_contract.image.spatial_scale)
    from core.rewiring.training import prepare_topology_io_rebase
    from .residual_unet import native_config
    finish_topology = prepare_topology_io_rebase(trainer.config,"minimax_h3",native_config(inner.config))
    with torch.random.fork_rng(devices=[]):
        alignment = SpatialAlignmentPair(declaration)
    wrapper = install_video_alignment(model, alignment, contract, defer_calibration=True)
    trainer.transformer = wrapper
    trainer.transformer_original = inner
    trainer.vae = NormalizedVideoVAE(target, contract)
    trainer.vae_dtype = target.dtype
    trainer.vae_identity = replace(resolved, state_dict=None)
    trainer.vae_latent_channels = image.latent_channels
    trainer.wiring = handler.resolve_wiring(trainer)
    if getattr(inner, "_residual_unet_spec", None) is not None:
        trainer.config["dit_rewiring_spec"] = inner._residual_unet_spec
    if finish_topology is not None:
        finish_topology(native_config(inner.config))
    setattr(trainer, PENDING, (old.to("cpu").requires_grad_(False).eval(), source_contract))
    components = getattr(trainer, "minimax_h3_components", None)
    if components is not None:
        components.update(transformer=wrapper, vae=trainer.vae, vae_identity=trainer.vae_identity,
            vae_config=dict(trainer.vae.config), video_vae_dtype=str(target.dtype),
            latent_channels=image.latent_channels, vae_scale_factor_spatial=image.spatial_scale,
            vae_scale_factor_temporal=contract.temporal_scale, latents_mean=trainer.vae.config.latents_mean,
            latents_std=trainer.vae.config.latents_std, pixel_mean=(.5,)*3, pixel_std=(.5,)*3,
            vae_tiling_policy=None)
    trainer.minimax_h3_latents_mean = trainer.vae.config.latents_mean
    trainer.minimax_h3_latents_std = trainer.vae.config.latents_std
    trainer.minimax_h3_pixel_mean = trainer.minimax_h3_pixel_std = (.5,)*3
    report = inner._video_resize_report
    return replace(report, replaced=(*report.replaced, "video_alignment"),
        new_elements=report.new_elements+sum(p.numel() for p in alignment.parameters()))


def calibrate(trainer, datasets):
    pending = getattr(trainer, PENDING, None)
    if pending is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.optimizer_residency import encoder_refill_residency
    from .vae_contract import transform_pixels
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in (128, 256, 512):
        raise ValueError("Invalid H3 VAE calibration count/resolution")
    samples = {}
    for dataset in datasets:
        for item in dataset.items:
            video = item.get("item_type") == "video"
            path = item.get("video_path") if video else item.get("image_path")
            if path:
                samples.setdefault((video, path), item)
    samples = list(samples)
    if not samples:
        raise ValueError("H3 VAE calibration requires training images or clips")
    samples = [samples[i] for i in torch.linspace(0, len(samples)-1, min(count, len(samples))).long().tolist()]
    old, source = pending
    target = trainer.vae
    alignment = trainer.transformer_original._video_latent_alignment
    fit = PairedVideoCalibration(alignment, ridge=1e-2,
        source_contract=source, target_contract=target.contract)
    spatial = lcm(source.image.spatial_scale, target.contract.image.spatial_scale)
    canvas = max(spatial, (side//spatial)*spatial)
    teacher_canvas = calibration_source_side(canvas, alignment.declaration)
    frames = paired_calibration_frames(source, target.contract)

    def pixels(sample, size=canvas):
        video, path = sample
        if video:
            from core.training.video_loader import load_clip
            from core.training.components.wiring import MINIMAX_H3_TEMPORAL
            spec = replace(MINIMAX_H3_TEMPORAL, frame_multiple=frames-1, frame_offset=1,
                min_decodable_frames=frames, default_clip_lengths=(frames,))
            value = load_clip(path, clip_length=frames, target_h=size, target_w=size, spec=spec)
            return value.permute(1, 0, 2, 3)[None]
        from PIL import Image, ImageOps
        import numpy as np
        with Image.open(path) as opened:
            image = ImageOps.fit(ImageOps.exif_transpose(opened).convert("RGB"), (size, size))
            value = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()/127.5-1
        return value[None, :, None]

    # The H3 conditioner is CPU/memory-mapped: .to('cpu') would detach its mapping.
    for name in ("text_encoder", "audio_vae"):
        module = getattr(trainer, name, None)
        if module is not None and any(t.device.type != "cpu" for t in (*module.parameters(), *module.buffers())):
            raise ValueError(f"H3 calibration requires its frozen {name} on CPU")
    callback = getattr(trainer, "_vae_alignment_progress_callback", None)
    with encoder_refill_residency(trainer, text=False, vae=True):
        try:
            paired = []
            old.to(trainer.device)
            old_dtype = next(old.parameters()).dtype
            with torch.no_grad():
                for i, sample in enumerate(samples):
                    trainer._check_stop_requested()
                    value = transform_pixels(pixels(sample, teacher_canvas).to(trainer.device), encode=True).to(old_dtype)
                    z = old.encode(value).latent_dist.mode().float()
                    mean = z.new_tensor(old.config.latents_mean).view(1, -1, 1, 1, 1)
                    std = z.new_tensor(old.config.latents_std).view(1, -1, 1, 1, 1)
                    paired.append(((z-mean)/std).cpu())
                    if callback:
                        callback(phase="latent_cache", step=i+1, total=2*len(samples))
            old.to("cpu")
            if torch.device(trainer.device).type == "cuda":
                torch.cuda.empty_cache()
            target.to(trainer.device)
            with torch.no_grad():
                for i, (sample, teacher) in enumerate(zip(samples, paired)):
                    trainer._check_stop_requested()
                    value = pixels(sample).to(device=trainer.device, dtype=target.dtype)
                    fit.add(teacher, target.encode(value).latent_dist.mode().cpu())
                    if callback:
                        callback(phase="latent_cache", step=len(samples)+i+1, total=2*len(samples))
            fit.finish()
            delattr(trainer, PENDING)
        finally:
            old.to("cpu")
            target.to("cpu")
