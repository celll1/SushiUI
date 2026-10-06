"""MiniT2I target-native image boundaries for t=1-clean, x0 prediction."""


def latent_geometry(model):
    pair = getattr(model, "vae_latent_bridge", None)
    if pair is not None:
        return pair.declaration["new_scale"], pair.declaration["new_channels"]
    from .minit2i_vae import VAE_SCALE_FACTOR
    cfg = model.mmjit_config
    return (1 if cfg.vae_type == "none" else VAE_SCALE_FACTOR), int(cfg.in_channels)


def generation_token_stride(model):
    return latent_geometry(model)[0] * int(model.mmjit_config.patch_size)


def predict_x0(model, image, timestep, context, mask):
    pair = getattr(model, "vae_latent_bridge", None)
    if pair is None:
        return model.model.net(image, timestep, context, mask)
    if image.ndim != 4 or image.shape[1] != pair.declaration["new_channels"]:
        raise ValueError("MiniT2I spatial VAE swap requires external BCHW latents")
    if timestep.shape != (image.shape[0],):
        raise ValueError("MiniT2I spatial VAE swap requires one timestep per image")
    patch = int(model.mmjit_config.patch_size)
    if any(size % patch for size in image.shape[-2:]):
        raise ValueError("MiniT2I spatial VAE swap requires complete native patches")
    # MiniT2I uses x_t=t*x0+(1-t)*noise, the reverse of SD3's timestep.
    incoming = pair.input.diffusion_state(image, alpha=timestep)
    return pair.output.prediction(model.model.net(incoming, timestep, context, mask), kind="x0")
