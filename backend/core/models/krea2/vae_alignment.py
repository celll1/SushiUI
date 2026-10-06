"""Krea packed-image boundaries for the shared target-native codec bridge."""
from dataclasses import replace

import torch

from core.models.components.spatial_swap import NAME, initialize_image_swap, calibrate_images


def apply_swap(trainer, resolved, *, handler, module=None):
    from core.models.components.latent_io import resize_latent_io, ResizeReport
    from core.models.components.wiring import KREA2_LATENT_IO
    from core.training.vae_swap import initialization_mode
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec
    model = trainer.transformer
    pair = getattr(model, NAME, None)
    if pair is not None:
        if module is None and resolved.source != pair.declaration["source"]:
            raise ValueError("Changing an aligned Krea VAE requires a fresh base checkpoint")
        trainer.vae_identity = replace(resolved, state_dict=None)
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        channels = model.img_in.in_features//4
        return ResizeReport((), channels, channels, channels, 0, 0)
    if module is not None:
        report = resize_latent_io(model, KREA2_LATENT_IO, resolved.latent_channels)
        trainer.vae = module
        trainer.vae_identity = resolved
        trainer.vae_latent_channels = resolved.latent_channels
        trainer.wiring = handler.wiring.replace(latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor, vae_norm=resolved.norm, vae_norm_pack=resolved.norm_pack)
        return report
    if initialization_mode(trainer.config,"krea2") == "zero":
        if resolved.scale_factor != 8:
            raise ValueError("Legacy zero Krea swaps require an 8x VAE; use paired_spatial")
        from core.training.arch.base_arch import ArchHandler
        return ArchHandler.apply_vae_swap(handler, trainer, resolved)
    from core.training.adapters.base_adapter import reject_quantized_base
    reject_quantized_base(model, model_label="Krea VAE swap")
    source_spec = handler.resolve_wiring(trainer)
    from .krea2_pipeline_ops import vae_pixel_ndim
    source_rank = vae_pixel_ndim(trainer.vae)
    source_spec = source_spec.replace(latent_ndim=source_rank)
    def resize(channels):
        topology = getattr(model,"_residual_unet_spec",None)
        report = resize_latent_io(model,KREA2_LATENT_IO,channels)
        model.out_channels = channels*4
        if topology is not None:
            restore_topology(model,build_spec("krea2",[{"name":"main","block_count":len(model.transformer_blocks)}],
                topology["regions"],native_config(dict(model.config))))
        return report
    from core.models.components.gaussian_alignment import POLICY
    noise_policy = (POLICY if resolved.scale_factor > source_spec.vae_scale_factor
                    and resolved.latent_channels >= model.img_in.in_features//4 else None)
    return initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.img_in.in_features//4,source_wiring=source_spec,patch_size=2,resize=resize,
        pending_key="_krea_vae_calibration_source",topology_config=lambda:native_config(dict(model.config)),
        noise_policy=noise_policy)


def noise_kwargs(model, noise):
    if getattr(getattr(model, NAME, None), "noise_alignment", None) is None:
        return {}
    if noise is None or noise.ndim != 3:
        raise ValueError("Aligned Krea requires its actual packed exposure noise")
    return {"alignment_noise": noise}


def calibrate(trainer,datasets):
    calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_krea_vae_calibration_source")


def image_grid(position_ids, prefix, token_count):
    image = position_ids[prefix:]
    if image.shape != (token_count,3) or token_count < 1:
        raise ValueError("Aligned Krea requires native text-prefix/image-tail positions")
    known = getattr(position_ids,"_krea_grid",None)
    if known is not None:
        h,w,version = known
        if h*w == token_count and position_ids._version == version:
            return h,w
    h,w = int(image[:,1].max().item())+1,int(image[:,2].max().item())+1
    from .krea2_pipeline_ops import prepare_position_ids
    if h*w != token_count or not torch.equal(
            position_ids,prepare_position_ids(prefix,h,w,position_ids.device)):
        raise ValueError("Aligned Krea requires rectangular row-major positions")
    return h,w
