"""Z-Image boundaries for target-native shared spatial VAE alignment."""
from dataclasses import replace

from core.models.components.spatial_swap import NAME, calibrate_images, initialize_image_swap


def apply_swap(trainer, resolved, module=None, *, handler):
    from core.models.components.latent_io import resize_latent_io, ResizeReport
    from core.models.components.wiring import ZIMAGE_LATENT_IO
    from core.models.zimage_residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec
    from core.training.vae_swap import initialization_mode
    model = trainer.transformer_original
    pair = getattr(model, NAME, None)
    if pair is not None:
        if resolved.source != pair.declaration["source"] and module is None:
            raise ValueError("Changing an aligned Z-Image VAE requires a fresh base checkpoint")
        trainer.vae_identity = replace(resolved, state_dict=None)
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        return ResizeReport((), model.in_channels, model.out_channels, model.in_channels, 0, 0)
    mode = initialization_mode(trainer.config, "zimage")
    if module is not None:
        report = resize_latent_io(model, ZIMAGE_LATENT_IO, resolved.latent_channels)
        model.in_channels = model.out_channels = resolved.latent_channels
        trainer.transformer.in_channels = trainer.transformer.out_channels = resolved.latent_channels
        trainer.vae = module
        trainer.vae_identity = resolved
        trainer.vae_latent_channels = resolved.latent_channels
        trainer.wiring = handler.wiring.replace(latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor, vae_norm=resolved.norm, vae_norm_pack=resolved.norm_pack)
        return report
    if mode != "paired_spatial":
        if resolved.scale_factor != 8 or resolved.ndim != 4:
            raise ValueError("Legacy zero Z-Image swaps require a 4-D 8x VAE; use paired_spatial")
        from core.training.arch.base_arch import ArchHandler
        return ArchHandler.apply_vae_swap(handler, trainer, resolved, module=module)
    if tuple(model.all_patch_size) != (2,) or tuple(model.all_f_patch_size) != (1,):
        raise ValueError("Z-Image spatial alignment requires the image 2-1 patch bank")
    from core.training.adapters.base_adapter import reject_quantized_base
    reject_quantized_base(model, model_label="Z-Image VAE swap")
    source_spec = handler.resolve_wiring(trainer)
    def resize(channels):
        topology = getattr(model, "_residual_unet_spec", None)
        report = resize_latent_io(model, ZIMAGE_LATENT_IO, channels)
        model.in_channels = model.out_channels = channels
        trainer.transformer.in_channels = trainer.transformer.out_channels = channels
        if topology is not None:
            restore_topology(model, build_spec("zimage", [{"name": "main", "block_count":len(model.layers)}],
                topology["regions"], native_config(model)))
        return report
    return initialize_image_swap(trainer, resolved, handler=handler, model=model, old_channels=model.in_channels,
        source_wiring=source_spec, patch_size=2, resize=resize, pending_key="_zimage_vae_calibration_source")


def calibrate(trainer, datasets):
    calibrate_images(trainer, datasets, model=trainer.transformer_original,
                     pending_key="_zimage_vae_calibration_source")
