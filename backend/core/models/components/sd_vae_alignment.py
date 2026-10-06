"""SD U-Net codec boundaries with explicit VP or flow noise coefficients."""
def resolve_wiring(handler,trainer):
    from .latent_adapter import NormalizedImageVAE
    vae=getattr(trainer,"vae",None)
    if isinstance(vae,NormalizedImageVAE):
        handler.pixel_align=vae.spatial_scale_factor
        return handler.wiring.replace(latent_channels=vae.config.latent_channels,
            vae_scale_factor=vae.spatial_scale_factor,vae_norm="identity",vae_norm_pack=1)
    return getattr(trainer,"wiring",None) or handler.wiring


def initialize_swap(trainer,resolved,*,handler):
    from .latent_io import resize_latent_io
    from .spatial_swap import initialize_image_swap
    from .vae_wrapper import SDXLVAEWrapper
    from core.training.adapters.base_adapter import reject_quantized_base
    model=trainer.unet
    if model.config.in_channels!=model.config.out_channels:
        raise ValueError("SD spatial swap requires symmetric image-only U-Net latent faces")
    reject_quantized_base(model,model_label="SD U-Net VAE swap")
    original=trainer.vae
    if isinstance(original,SDXLVAEWrapper):
        trainer.vae=original.vae
    try:
        report=initialize_image_swap(trainer,resolved,handler=handler,model=model,
            old_channels=model.config.in_channels,source_wiring=handler.resolve_wiring(trainer),
            patch_size=1,resize=lambda width:resize_latent_io(trainer,handler.wiring.latent_io,width),
            pending_key="_sd_vae_calibration_source")
        if isinstance(original,SDXLVAEWrapper) and trainer.vae is original.vae:
            trainer.vae=original
        return report
    except Exception:
        if isinstance(original,SDXLVAEWrapper) and trainer.vae is original.vae:
            trainer.vae=original
        raise


def calibrate(trainer,datasets):
    from .spatial_swap import calibrate_images
    calibrate_images(trainer,datasets,model=trainer.unet,pending_key="_sd_vae_calibration_source")


def training_coefficients(model,scheduler,timesteps,*,noise_process,reference,alphas_cumprod=None):
    if getattr(model,"vae_latent_bridge",None) is None:
        return None,None
    import torch
    if noise_process=="flow":
        sigma=timesteps.to(device=reference.device,dtype=reference.dtype)
        return 1-sigma,sigma
    if noise_process!="ddpm":
        raise ValueError("SD spatial alignment requires a declared VP or flow noise process")
    if timesteps.dtype not in (torch.int32,torch.int64):
        raise ValueError("VP training alignment requires discrete scheduler indices")
    table=scheduler.alphas_cumprod if alphas_cumprod is None else alphas_cumprod
    table=table.to(device=reference.device)
    alpha_bar=table[timesteps.to(reference.device).long()].float()
    return alpha_bar.sqrt().to(reference.dtype),(1-alpha_bar).sqrt().to(reference.dtype)


def map_input(model,latent,*,alpha):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return latent
    if latent.ndim!=4 or alpha is None:
        raise ValueError("SD spatial alignment requires BCHW latents and an explicit clean coefficient")
    return pair.input.diffusion_state(latent,alpha=alpha)


def map_prediction(model,value,*,noise_process,prediction_target,sigma):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        return value
    if prediction_target=="velocity":
        kind={"ddpm":"v","flow":"eps_minus_x0"}.get(noise_process)
    else:
        kind={"epsilon":"epsilon","sample":"x0"}.get(prediction_target)
    if kind is None or noise_process not in ("ddpm","flow"):
        raise ValueError("SD spatial alignment requires a declared prediction and noise process")
    return pair.output.prediction(value,kind=kind,sigma=sigma)
