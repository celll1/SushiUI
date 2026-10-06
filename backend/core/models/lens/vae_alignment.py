"""Lens target-native image codec and packed boundary alignment."""
from dataclasses import replace

from core.models.components.spatial_swap import NAME, initialize_image_swap, calibrate_images


def apply_swap(trainer, resolved, *, handler, module=None):
    from core.models.components.latent_io import resize_latent_io, ResizeReport
    from core.models.components.wiring import LENS_LATENT_IO
    from core.training.vae_swap import initialization_mode
    from .residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec
    model = trainer.transformer
    pair = getattr(model, NAME, None)
    if pair is not None:
        if module is None and resolved.source != pair.declaration["source"]:
            raise ValueError("Changing an aligned Lens VAE requires a fresh base checkpoint")
        trainer.vae_identity = replace(resolved, state_dict=None)
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        channels = model.img_in.in_features//4
        return ResizeReport((), channels, channels, channels, 0, 0)
    if module is not None:
        report = resize_latent_io(model, LENS_LATENT_IO, resolved.latent_channels)
        trainer.vae = module
        trainer.vae_identity = resolved
        trainer.vae_latent_channels = resolved.latent_channels
        trainer.wiring = handler.wiring.replace(latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor, vae_norm=resolved.norm, vae_norm_pack=resolved.norm_pack)
        return report
    if initialization_mode(trainer.config,"lens") == "zero":
        if resolved.ndim != 4 or resolved.scale_factor != 8:
            raise ValueError("Legacy zero Lens swaps require a 4D/8x VAE; use paired_spatial")
        from core.training.arch.base_arch import ArchHandler
        return ArchHandler.apply_vae_swap(handler, trainer, resolved)
    from core.training.adapters.base_adapter import reject_quantized_base
    reject_quantized_base(model, model_label="Lens VAE swap")
    def resize(channels):
        topology = getattr(model,"_residual_unet_spec",None)
        report = resize_latent_io(model,LENS_LATENT_IO,channels)
        model.in_channels, model.out_channels = channels*4, channels
        if topology is not None:
            restore_topology(model,build_spec("lens",[{"name":"main","block_count":len(model.transformer_blocks)}],
                topology["regions"],native_config(dict(model.config))))
        return report
    return initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.img_in.in_features//4,source_wiring=handler.resolve_wiring(trainer),
        patch_size=2,resize=resize,pending_key="_lens_vae_calibration_source")


def calibrate(trainer,datasets):
    calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_lens_vae_calibration_source")
