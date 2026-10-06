"""SD U-Net codec boundaries with explicit VP or flow noise coefficients."""
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
