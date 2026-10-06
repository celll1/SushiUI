"""Target-grid boundaries and trained IO migration for SenseNova generation."""
import torch
from torch import nn
from torch.nn import functional as F


def input_state(model, image, t):
    pair = getattr(model, "vae_latent_bridge", None)
    return image if pair is None else pair.input.diffusion_state(image, alpha=t)


def output_x0(model, image):
    pair = getattr(model, "vae_latent_bridge", None)
    return image if pair is None else pair.output.prediction(image, kind="x0")


def initialize_swap(trainer, resolved, *, handler):
    from core.models.components.spatial_swap import initialize_image_swap
    from core.models.components.pixel_codec import RGBPixelCodec
    from .latent_space import gen_geometry, latent_config_dict
    from core.training.arch.sensenova import _fm_modules_trained
    from core.training.ops.training_method import is_full_finetune
    if not is_full_finetune(trainer) or not _fm_modules_trained(trainer):
        raise ValueError("SenseNova paired calibration requires full_finetune with generation fm_modules enabled")
    model = trainer.transformer
    geometry = gen_geometry(model)
    wiring = handler.resolve_wiring(trainer)
    patch = handler._resolve_gen_patch(trainer.config,
        base_patch=geometry.patch if geometry.is_latent else None)
    old_vae = trainer.vae
    if not geometry.is_latent:
        trainer.vae = RGBPixelCodec()

    def resize(channels):
        report = migrate_generation_io(model, channels=channels, patch=patch, scale=resolved.scale_factor)
        trainer.sensenova_config_dict = latent_config_dict(
            getattr(trainer, "sensenova_config_dict", None), channels=channels, patch=patch)
        return report

    try:
        return initialize_image_swap(trainer, resolved, handler=handler, model=model,
            old_channels=geometry.channels, source_wiring=wiring, patch_size=patch,
            resize=resize, pending_key="_sensenova_vae_calibration_source")
    except Exception:
        trainer.vae = old_vae
        raise


def calibrate(trainer, datasets):
    from core.models.components.spatial_swap import calibrate_images
    return calibrate_images(trainer, datasets, model=trainer.transformer,
                           pending_key="_sensenova_vae_calibration_source")


def _spatial_weights(value, size, *, sum_preserving=False):
    old = value.shape[-1]
    if value.shape[-2] != old or (old % size and size % old):
        raise ValueError("SenseNova trained IO migration requires integer patch ratios")
    if size <= old:
        result = F.adaptive_avg_pool2d(value.float(), size)
    else:
        result = F.interpolate(value.float(), size=(size, size), mode="nearest")
    return result * (old / size) ** 2 if sum_preserving else result


def migrate_generation_io(model, *, channels, patch, scale):
    """Retain constant-field input responses and block-averaged trained outputs."""
    from .latent_space import gen_geometry, validate_gen_patch
    from core.models.components.latent_io import ResizeReport
    if getattr(model, "vae_latent_bridge", None) is not None:
        raise ValueError("Restore an existing SenseNova alignment instead of rebuilding its IO")
    geometry = gen_geometry(model)
    patch = validate_gen_patch(patch)
    merge = int(1 / model.downsample_ratio)
    if (type(channels) is not int or channels < geometry.channels or
            type(scale) is not int or scale < 1 or patch % merge):
        raise ValueError("SenseNova migration must retain every native channel and complete merged patches")
    if not model.use_pixel_head or model.use_deep_fm_head:
        raise ValueError("SenseNova spatial alignment requires the native ConvDecoder head")
    refiner = model.fm_modules["fm_refiner"] if "fm_refiner" in model.fm_modules else None
    if refiner is not None and refiner.in_channels != channels:
        raise ValueError("SenseNova spatial migration must preserve the attached refiner's internal width")
    topology = getattr(model, "_residual_unet_spec", None)
    if topology is not None:
        from .residual_unet import validate_owner_topology
        validate_owner_topology(topology, model)
    embeddings = model.fm_modules["vision_model_mot_gen"].embeddings
    head = model.fm_modules["fm_head"]
    incoming, outgoing = embeddings.patch_embedding, head.conv2
    old_patch, new_patch = geometry.vit_patch, patch // merge
    old_shuffle, shuffle = geometry.head_shuffle, patch // 4
    if (type(incoming) is not nn.Conv2d or type(outgoing) is not nn.Conv2d or
            incoming.in_channels != geometry.channels or incoming.groups != 1 or
            incoming.kernel_size != (old_patch, old_patch) or
            incoming.stride != (old_patch, old_patch) or incoming.padding != (0, 0) or
            incoming.dilation != (1, 1) or outgoing.groups != 1 or
            outgoing.kernel_size != (3, 3) or outgoing.stride != (1, 1) or
            outgoing.padding != (1, 1) or outgoing.dilation != (1, 1) or
            outgoing.out_channels != geometry.channels * old_shuffle ** 2 or
            head.ps3.upscale_factor != old_shuffle):
        raise ValueError("SenseNova spatial migration requires unquantized native generation IO")
    # Validate both ratios before replacing either module.
    weight_in = _spatial_weights(incoming.weight, new_patch, sum_preserving=True)
    packed = outgoing.weight.reshape(geometry.channels, old_shuffle, old_shuffle, -1).permute(0, 3, 1, 2)
    weight_out = _spatial_weights(packed, shuffle).permute(0, 2, 3, 1)
    new_in = nn.Conv2d(channels, incoming.out_channels, new_patch, stride=new_patch,
        bias=incoming.bias is not None).to(device=incoming.weight.device, dtype=incoming.weight.dtype)
    new_out = nn.Conv2d(outgoing.in_channels, channels * shuffle ** 2, 3, padding=1,
        bias=outgoing.bias is not None).to(device=outgoing.weight.device, dtype=outgoing.weight.dtype)
    with torch.no_grad():
        new_in.weight.zero_()
        new_in.weight[:, :geometry.channels].copy_(weight_in)
        if new_in.bias is not None:
            new_in.bias.copy_(incoming.bias)
        new_out.weight.zero_()
        new_out.weight[:geometry.channels * shuffle ** 2].copy_(
            weight_out.reshape(geometry.channels * shuffle ** 2, *outgoing.weight.shape[1:]))
        if new_out.bias is not None:
            bias = outgoing.bias.reshape(geometry.channels, old_shuffle, old_shuffle)[None]
            bias = _spatial_weights(bias, shuffle).flatten()
            new_out.bias.zero_()
            new_out.bias[:bias.numel()].copy_(bias)
    for new, old in ((new_in, incoming), (new_out, outgoing)):
        new.weight.requires_grad_(old.weight.requires_grad)
        if new.bias is not None:
            new.bias.requires_grad_(old.bias.requires_grad)
    embeddings.patch_embedding = new_in
    embeddings.patch_size = embeddings.config.patch_size = new_patch
    embeddings.config.num_channels = channels
    head.conv2, head.ps3 = new_out, nn.PixelShuffle(shuffle)
    model.gen_in_channels = model.config.gen_in_channels = channels
    model.gen_patch_size = model.config.gen_patch_size = patch
    model.gen_vit_patch_size, model.gen_vae_scale_factor = new_patch, scale
    if topology is not None:
        from core.rewiring.spec import build_spec
        from .residual_unet import native_config, restore_topology
        config = native_config(model.config)
        restore_topology(model, build_spec("sensenova",
            [{"name": "main", "block_count": config["llm_config"]["num_hidden_layers"]}],
            topology["regions"], config))
    return ResizeReport(
        replaced=("fm_modules.vision_model_mot_gen.embeddings.patch_embedding", "fm_modules.fm_head.conv2"),
        old_in_channels=geometry.channels, old_out_channels=geometry.channels,
        new_channels=channels, copied_elements=weight_in.numel() + weight_out.numel(),
        new_elements=new_in.weight.numel() + new_out.weight.numel() - weight_in.numel() - weight_out.numel())
