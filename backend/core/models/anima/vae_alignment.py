"""Anima image boundaries; padding-mask channels stay outside the codec bridge."""
def latent_geometry(model):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is not None:
        return pair.declaration["new_scale"],pair.declaration["new_channels"]
    return 8,int(model.in_channels)


def generation_token_stride(model):
    return latent_geometry(model)[0]*model.patch_spatial


def initialize_swap(trainer,resolved,*,handler):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import ANIMA_LATENT_IO
    from core.models.components.spatial_swap import initialize_image_swap
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.rewiring.spec import build_spec
    from .residual_unet import native_config,restore_topology
    model=trainer.transformer
    if model.patch_temporal!=1 or model.patch_spatial!=2 or not model.concat_padding_mask:
        raise ValueError("Anima VAE swap requires native image patches and padding-mask input")
    reject_quantized_base(model,model_label="Anima VAE swap")
    def resize(channels):
        topology=getattr(model,"_residual_unet_spec",None)
        report=resize_latent_io(model,ANIMA_LATENT_IO,channels)
        if topology is not None:
            restore_topology(model,build_spec("anima",[{"name":"main","block_count":len(model.blocks)}],
                topology["regions"],native_config(model)))
        return report
    return initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.in_channels,source_wiring=handler.resolve_wiring(trainer),
        patch_size=2,resize=resize,pending_key="_anima_vae_calibration_source",
        topology_config=lambda:native_config(model))


def calibrate(trainer,datasets):
    from core.models.components.spatial_swap import calibrate_images
    calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_anima_vae_calibration_source")


def map_image(model, value, *, sigma=None, output=False):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return value
    if value.ndim!=5 or value.shape[2]!=1:
        raise ValueError("Anima spatial VAE swap requires a single image frame")
    image=value.squeeze(2)
    if output:
        image=pair.output.flow_velocity(image,convention="eps_minus_x0")
    else:
        if sigma.ndim==2 and sigma.shape[1]==1:
            sigma=sigma[:,0]
        image=pair.input.flow_state(image,sigma)
    return image.unsqueeze(2)
