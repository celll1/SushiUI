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


def initialize_swap(trainer, resolved, *, handler):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.spatial_swap import initialize_image_swap
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.rewiring.spec import build_spec
    from .residual_unet import native_config, restore_topology
    model = trainer.transformer
    cfg = model.mmjit_config
    if cfg.vae_type == "none" or trainer.vae is None:
        raise ValueError("MiniT2I pixel-to-latent patch migration is not yet wired")
    reject_quantized_base(model, model_label="MiniT2I VAE swap")
    wiring = handler.resolve_wiring(trainer)

    def resize(channels):
        topology = getattr(model, "_residual_unet_spec", None)
        report = resize_latent_io(model, wiring.latent_io, channels)
        cfg.vae_type = "custom"
        model.register_to_config(vae_type="custom")
        if topology is not None:
            restore_topology(model, build_spec("minit2i",
                [{"name":"main", "block_count":cfg.depth_double}],
                topology["regions"], native_config(model)))
        return report

    return initialize_image_swap(trainer, resolved, handler=handler, model=model,
        old_channels=int(cfg.in_channels), source_wiring=wiring,
        patch_size=int(cfg.patch_size), resize=resize,
        pending_key="_minit2i_vae_calibration_source")


def calibrate(trainer, datasets):
    from core.models.components.spatial_swap import calibrate_images
    calibrate_images(trainer, datasets, model=trainer.transformer,
        pending_key="_minit2i_vae_calibration_source")


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
