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
    from core.models.components.gaussian_alignment import POLICY
    source_wiring=handler.resolve_wiring(trainer)
    noise_policy=(POLICY if resolved.scale_factor>source_wiring.vae_scale_factor
                  and resolved.latent_channels>=model.in_channels else None)
    return initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.in_channels,source_wiring=source_wiring,
        patch_size=2,resize=resize,pending_key="_anima_vae_calibration_source",
        topology_config=lambda:native_config(model),noise_policy=noise_policy)


def calibrate(trainer,datasets):
    from core.models.components.spatial_swap import calibrate_images
    calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_anima_vae_calibration_source")


def noise_kwargs(model, noise):
    if getattr(getattr(model,"vae_latent_bridge",None),"noise_alignment",None) is None:
        return {}
    if noise is None or noise.ndim not in (4,5):
        raise ValueError("Aligned Anima requires its actual image exposure noise")
    return {"alignment_noise":noise.unsqueeze(2) if noise.ndim==4 else noise}


def map_image(model, value, *, sigma=None, output=False, noise=None):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return value
    if value.ndim!=5 or value.shape[2]!=1:
        raise ValueError("Anima spatial VAE swap requires a single image frame")
    image=value.squeeze(2)
    if noise is not None:
        if noise.ndim!=5 or noise.shape[2]!=1:
            raise ValueError("Aligned Anima noise requires a single image frame")
        noise=noise.squeeze(2)
    if output:
        image=pair.flow_velocity(image,convention="eps_minus_x0",noise=noise)
    else:
        if sigma.ndim==2 and sigma.shape[1]==1:
            sigma=sigma[:,0]
        image=pair.flow_state(image,sigma,noise=noise)
    return image.unsqueeze(2)
