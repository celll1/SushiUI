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


def pixel_latent_patch_size(model):
    from .minit2i_vae import VAE_SCALE_FACTOR
    patch = int(model.mmjit_config.patch_size)
    first = max(1, (patch + VAE_SCALE_FACTOR - 1) // VAE_SCALE_FACTOR)
    return next(size for size in range(first, patch + 1) if patch % size == 0)


def migrate_pixel_patches(model):
    """Aggregate trained pixel patches; keep constant-field responses and the body."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    net, cfg = model.model.net, model.mmjit_config
    old_patch = int(cfg.patch_size)
    patch = pixel_latent_patch_size(model)
    incoming, outgoing = net.img_embedder.proj1, net.final_layer.linear
    if (cfg.vae_type != "none" or cfg.in_channels != 3 or old_patch % patch or
            type(incoming) is not nn.Conv2d or type(outgoing) is not nn.Linear or
            incoming.kernel_size != (old_patch, old_patch) or
            incoming.stride != (old_patch, old_patch) or incoming.groups != 1 or
            incoming.padding != (0, 0) or incoming.dilation != (1, 1) or
            outgoing.out_features != old_patch * old_patch * 3):
        raise ValueError("MiniT2I pixel migration requires native RGB patch projections")
    device, dtype = incoming.weight.device, incoming.weight.dtype
    conv = nn.Conv2d(3, incoming.out_channels, patch, stride=patch,
        bias=incoming.bias is not None).to(device=device, dtype=dtype)
    linear = nn.Linear(outgoing.in_features, patch * patch * 3,
        bias=outgoing.bias is not None).to(device=outgoing.weight.device, dtype=outgoing.weight.dtype)
    with torch.no_grad():
        weight = F.adaptive_avg_pool2d(incoming.weight.float(), patch) * (old_patch / patch) ** 2
        conv.weight.copy_(weight)
        if incoming.bias is not None:
            conv.bias.copy_(incoming.bias)
        weight = outgoing.weight.reshape(old_patch,old_patch,3,-1).permute(2,3,0,1)
        weight = F.adaptive_avg_pool2d(weight.float(), patch).permute(2,3,0,1).reshape_as(linear.weight)
        linear.weight.copy_(weight)
        if outgoing.bias is not None:
            bias = outgoing.bias.reshape(old_patch,old_patch,3).permute(2,0,1)[None]
            linear.bias.copy_(F.adaptive_avg_pool2d(bias.float(), patch)[0].permute(1,2,0).flatten())
    conv.weight.requires_grad_(incoming.weight.requires_grad)
    linear.weight.requires_grad_(outgoing.weight.requires_grad)
    if conv.bias is not None:
        conv.bias.requires_grad_(incoming.bias.requires_grad)
    if linear.bias is not None:
        linear.bias.requires_grad_(outgoing.bias.requires_grad)
    net.img_embedder.proj1, net.final_layer.linear = conv, linear
    net.img_embedder.patch_size = net.final_layer.patch_size = cfg.patch_size = patch
    model.register_to_config(patch_size=patch)
    return patch


def initialize_swap(trainer, resolved, *, handler):
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.spatial_swap import initialize_image_swap
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.rewiring.spec import build_spec
    from .residual_unet import native_config, restore_topology
    model = trainer.transformer
    cfg = model.mmjit_config
    pixel = cfg.vae_type == "none"
    if pixel and int(cfg.in_channels) != 3:
        raise ValueError("MiniT2I pixel migration requires an RGB base checkpoint")
    reject_quantized_base(model, model_label="MiniT2I VAE swap")
    wiring = handler.resolve_wiring(trainer)
    patch = pixel_latent_patch_size(model) if pixel else int(cfg.patch_size)

    def resize(channels):
        topology = getattr(model, "_residual_unet_spec", None)
        if pixel:
            migrate_pixel_patches(model)
        report = resize_latent_io(model, wiring.latent_io.replace(pack_elems=patch * patch), channels)
        cfg.vae_type = "custom"
        model.register_to_config(vae_type="custom")
        if topology is not None:
            restore_topology(model, build_spec("minit2i",
                [{"name":"main", "block_count":cfg.depth_double}],
                topology["regions"], native_config(model)))
        return report

    old_vae = trainer.vae
    native_weight = model.model.net.img_embedder.proj1.weight.detach().cpu().clone() if pixel else None
    if pixel:
        from core.models.components.pixel_codec import RGBPixelCodec
        trainer.vae = RGBPixelCodec()
    try:
        report = initialize_image_swap(trainer, resolved, handler=handler, model=model,
            old_channels=int(cfg.in_channels), source_wiring=wiring,
            patch_size=patch, resize=resize,
            pending_key="_minit2i_vae_calibration_source",topology_config=lambda:native_config(model))
        if pixel:
            from core.models.components.patch_noise_alignment import POLICY, PatchNoiseAlignment
            pair = model.vae_latent_bridge
            declaration = dict(policy=POLICY, channels=pair.declaration["new_channels"],
                bridge_channels=pair.input.projection.out_channels,
                hidden_channels=model.model.net.img_embedder.proj1.out_channels, patch_size=patch)
            pair.declaration["patch_noise"] = declaration
            pair.add_module("patch_noise", PatchNoiseAlignment(declaration))
            trainer._minit2i_patch_noise_source = native_weight
        return report
    except Exception:
        trainer.vae = old_vae
        raise


def calibrate(trainer, datasets):
    from core.models.components.spatial_swap import calibrate_images
    def capture():
        native = getattr(trainer, "_minit2i_patch_noise_source", None)
        if native is not None:
            pair = trainer.transformer.vae_latent_bridge
            pair.patch_noise.capture(native, pair.input.projection,
                trainer.transformer.model.net.img_embedder.proj1)
            delattr(trainer, "_minit2i_patch_noise_source")
    calibrate_images(trainer, datasets, model=trainer.transformer,
        pending_key="_minit2i_vae_calibration_source", on_complete=capture)


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
    correction = getattr(pair, "patch_noise", None)
    if correction is None:
        return pair.output.prediction(model.model.net(incoming, timestep, context, mask), kind="x0")
    from core.models.components.image_noise_context import exposure_noise
    noise = exposure_noise(image)
    projection = model.model.net.img_embedder.proj1
    hook = projection.register_forward_hook(
        lambda module, args, output: correction.correct(output, noise, 1-timestep))
    try:
        return pair.output.prediction(model.model.net(incoming, timestep, context, mask), kind="x0")
    finally:
        hook.remove()
