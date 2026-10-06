"""LTX target-codec setup and bounded deterministic clip calibration."""
from dataclasses import replace
from math import lcm

import torch

from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair
from core.models.components.video_latent_adapter import VideoCodecContract, NormalizedVideoVAE, PairedVideoCalibration

PENDING = "_ltx2_vae_calibration_source"


def temporal_policy(rank, codec_class):
    if rank == 4:
        return "independent_frames_v1"
    if rank == 5 and codec_class in ("AutoencoderKLLTX2Video", "AutoencoderKLQwenImage"):
        return "causal_first_frame_v1"
    if rank == 5 and codec_class == "AutoencoderKLMiniMaxH3":
        return "minimax_h3_chunks_v1"
    raise ValueError(f"LTX calibration needs an explicit temporal policy for {codec_class}")


def apply_swap(trainer, resolved, *, handler, module=None):
    from core.models.components.latent_io import ResizeReport
    from core.models.common.module_identity import module_fingerprint
    from core.models.common.vae_source import normalization_config
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.training.vae_swap import initialization_mode
    from .video_alignment import install_video_alignment, AlignedLtx2BlockLoopWrapper, aligned_wrapper
    model = trainer.transformer
    inner = getattr(model, "transformer", model)
    pair = getattr(inner, "_video_latent_alignment", None)
    if pair is not None:
        if resolved.latent_hash != pair.declaration["target_identity"]:
            raise ValueError("Changing an aligned LTX VAE requires a fresh base checkpoint")
        if not isinstance(trainer.vae, NormalizedVideoVAE):
            raise ValueError("Aligned LTX training requires its restored normalized video codec")
        trainer.vae_identity = replace(resolved, state_dict=None)
        if not isinstance(model, AlignedLtx2BlockLoopWrapper):
            model = aligned_wrapper(inner)
            trainer.transformer = model
            trainer.ltx2_pipeline.transformer = model
        trainer.transformer_original = inner
        trainer.ltx2_block_loop_wrapper = model
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        c = inner.config.in_channels
        return ResizeReport((), c, c, c, 0, 0)
    if module is not None:
        raise ValueError("LTX swapped checkpoints require their saved video alignment")
    if initialization_mode(trainer.config, "ltx2") != "paired_spatial":
        raise ValueError("LTX VAE swap requires paired_spatial initialization")
    reject_quantized_base(model, model_label="LTX VAE swap")
    image = ImageCodecContract.from_resolved(resolved)
    contract = VideoCodecContract(image, resolved.scale_temporal,
                                 temporal_policy(image.latent_rank, resolved.vae_class))
    old = trainer.vae
    target = resolved.load_module(torch_dtype=trainer.vae_dtype).requires_grad_(False).eval()
    channels = inner.config.in_channels
    source = handler.resolve_wiring(trainer)
    identity = module_fingerprint(old)
    if identity == module_fingerprint(target) and normalization_config(old.config) == normalization_config(target.config):
        return ResizeReport((), channels, channels, channels, 0, 0)
    declaration = image.declaration(source=resolved.source, source_identity=identity,
        target_identity=resolved.latent_hash, old_channels=channels, old_scale=source.vae_scale_factor)
    with torch.random.fork_rng(devices=[]):
        alignment = SpatialAlignmentPair(declaration)
    wrapper = install_video_alignment(model, alignment, contract)
    trainer.transformer = wrapper
    trainer.transformer_original = wrapper.transformer
    trainer.ltx2_block_loop_wrapper = wrapper
    if getattr(inner, "_residual_unet_spec", None) is not None:
        trainer.config["dit_rewiring_spec"] = inner._residual_unet_spec
    pipeline = trainer.ltx2_pipeline
    pipeline.transformer = wrapper
    trainer.vae = NormalizedVideoVAE(target, contract)
    pipeline.register_modules(vae=trainer.vae)
    from .codec_pipelines.geometry import install_pipeline_clock
    install_pipeline_clock(pipeline)
    pipeline.vae_spatial_compression_ratio = contract.image.spatial_scale
    pipeline.vae_temporal_compression_ratio = contract.temporal_scale
    from diffusers.video_processor import VideoProcessor
    pipeline.video_processor = VideoProcessor(vae_scale_factor=contract.image.spatial_scale)
    setattr(trainer, PENDING, (old.to("cpu").requires_grad_(False).eval(), source))
    trainer.vae_identity = replace(resolved, state_dict=None)
    trainer.vae_latent_channels = image.latent_channels
    trainer.wiring = handler.resolve_wiring(trainer)
    report = inner._video_resize_report
    return replace(report, replaced=(*report.replaced,"video_alignment"),
                   new_elements=report.new_elements+sum(p.numel() for p in alignment.parameters()))


def calibrate(trainer, datasets):
    pending = getattr(trainer, PENDING, None)
    if pending is None:
        return
    from api.param_defaults import TRAINING_DEFAULTS
    from core.training.optimizer_residency import encoder_refill_residency, park_optimizer_state
    from core.training.base_trainer import all_optimizers
    from core.models.components.vae_registry import normalize
    count = trainer.config.get("vae_swap_calibration_samples", TRAINING_DEFAULTS["vae_swap_calibration_samples"])
    side = trainer.config.get("vae_swap_calibration_resolution", TRAINING_DEFAULTS["vae_swap_calibration_resolution"])
    if type(count) is not int or not 2 <= count <= 256 or side not in (128, 256, 512):
        raise ValueError("Invalid LTX VAE calibration count/resolution")
    samples = {}
    for dataset in datasets:
        for item in dataset.items:
            video = item.get("item_type") == "video"
            path = item.get("video_path") if video else item.get("image_path")
            if path:
                samples.setdefault((video, path), item)
    samples = list(samples)
    if not samples:
        raise ValueError("LTX VAE calibration requires training images or clips")
    samples = [samples[i] for i in torch.linspace(0, len(samples)-1, min(count,len(samples))).long().tolist()]
    old, source = pending
    target = trainer.vae
    alignment = trainer.transformer_original._video_latent_alignment
    fit = PairedVideoCalibration(alignment, ridge=1e-2)
    spatial = lcm(source.vae_scale_factor, target.spatial_compression_ratio)
    canvas = max(spatial, (side//spatial)*spatial)
    from .video_clock import calibration_clip_frames
    frames = calibration_clip_frames(source.vae_scale_temporal, target.contract)

    def pixels(sample):
        video, path = sample
        if video:
            from core.training.video_loader import load_clip
            from core.models.components.wiring import LTX2_TEMPORAL
            spec = replace(LTX2_TEMPORAL, frame_multiple=frames-1)
            value = load_clip(path, clip_length=frames, target_h=canvas, target_w=canvas, spec=spec)
            return value.permute(1,0,2,3)[None]
        from PIL import Image, ImageOps
        import numpy as np
        with Image.open(path) as opened:
            image = ImageOps.fit(ImageOps.exif_transpose(opened).convert("RGB"), (canvas, canvas))
            value = torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/127.5-1
        return value[None, :, None]

    text = [m for m in (getattr(trainer,"text_encoder",None),getattr(trainer,"connectors",None)) if m is not None]
    devices = [next(m.parameters()).device for m in text]
    parameters = [p for m in text for p in m.parameters()]
    callback = getattr(trainer,"_vae_alignment_progress_callback",None)
    with encoder_refill_residency(trainer, text=False, vae=True), park_optimizer_state(all_optimizers(trainer), parameters):
        for m in text:
            m.to("cpu")
        try:
            paired = []
            old.to(device=trainer.device,dtype=trainer.vae_dtype)
            with torch.no_grad():
                for i,sample in enumerate(samples):
                    trainer._check_stop_requested()
                    value = pixels(sample).to(device=trainer.device,dtype=trainer.vae_dtype)
                    paired.append(normalize(old.encode(value).latent_dist.mode(),old,source).cpu())
                    if callback:
                        callback(phase="latent_cache",step=i+1,total=2*len(samples))
            old.to("cpu")
            if torch.device(trainer.device).type == "cuda":
                torch.cuda.empty_cache()
            target.to(trainer.device)
            with torch.no_grad():
                for i,(sample,teacher) in enumerate(zip(samples,paired)):
                    trainer._check_stop_requested()
                    value = pixels(sample).to(device=trainer.device,dtype=trainer.vae_dtype)
                    fit.add(teacher,target.encode(value).latent_dist.mode().cpu())
                    if callback:
                        callback(phase="latent_cache",step=len(samples)+i+1,total=2*len(samples))
            fit.finish()
            delattr(trainer,PENDING)
        finally:
            old.to("cpu")
            target.to("cpu")
            for m,device in zip(text,devices):
                m.to(device)
