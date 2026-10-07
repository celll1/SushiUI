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


def validate_patch_alignment(model):
    pair = getattr(model, "vae_latent_bridge", None)
    correction = getattr(pair, "patch_noise", None)
    if correction is None:
        return
    stride = pair.declaration.get("patch_noise_source_stride")
    if type(stride) is not int or stride < 1:
        raise ValueError("SenseNova patch noise checkpoint is missing its source token stride")
    projection = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding
    declaration = correction.declaration
    if (declaration["hidden_channels"] != projection.out_channels
            or declaration["bridge_channels"] != projection.in_channels
            or projection.kernel_size != (declaration["patch_size"],) * 2):
        raise ValueError("SenseNova patch correction disagrees with its generation projection")


def patch_projection_transform(model, image, t, noise_scale, noise):
    pair = getattr(model, "vae_latent_bridge", None)
    correction = getattr(pair, "patch_noise", None)
    if correction is None:
        return None
    from .sensenova_pipeline_ops import compute_noise_scale
    validate_patch_alignment(model)
    stride = pair.declaration["patch_noise_source_stride"]
    if noise is None or noise.shape != image.shape or image.ndim != 4:
        raise ValueError("SenseNova patch correction requires this exposure's scaled BCHW noise")
    batch, _, height, width = image.shape
    patch = correction.declaration["patch_size"]
    if height % patch or width % patch:
        raise ValueError("SenseNova patch correction requires complete latent patches")
    scale = pair.declaration["new_scale"]
    source_scale = compute_noise_scale(model, height * scale / stride, width * scale / stride, 1)
    if not isinstance(noise_scale, (int, float)) or not 0 < noise_scale < float("inf"):
        raise ValueError("SenseNova patch correction requires a finite positive noise scale")
    gain = torch.full((batch,), source_scale / noise_scale, device=image.device, dtype=torch.float32)
    sigma = 1 - torch.as_tensor(t, device=image.device, dtype=torch.float32).reshape(-1)
    if sigma.numel() == 1:
        sigma = sigma.expand(batch)
    if sigma.shape != (batch,):
        raise ValueError("SenseNova patch correction requires one timestep per image")

    def transform(output):
        gh, gw = height // patch, width // patch
        if output.shape != (batch * gh * gw, correction.declaration["hidden_channels"], 1, 1):
            raise ValueError("SenseNova patch projection disagrees with the exposure grid")
        value = output.reshape(batch, gh, gw, -1).permute(0, 3, 1, 2)
        value = correction.correct(value, noise, sigma, desired_noise_gain=gain)
        return value.permute(0, 2, 3, 1).reshape_as(output)
    return transform


def initialize_swap(trainer, resolved, *, handler):
    from core.models.components.spatial_swap import initialize_image_swap
    from core.models.components.pixel_codec import RGBPixelCodec
    from .latent_space import gen_geometry, latent_config_dict
    from .residual_unet import native_config
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
    native_weight = None
    if not geometry.is_latent:
        native_weight = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding.weight.detach().cpu().clone()
        trainer.vae = RGBPixelCodec()

    def resize(channels):
        report = migrate_generation_io(model, channels=channels, patch=patch, scale=resolved.scale_factor)
        trainer.sensenova_config_dict = latent_config_dict(
            getattr(trainer, "sensenova_config_dict", None), channels=channels, patch=patch)
        return report

    try:
        report = initialize_image_swap(trainer, resolved, handler=handler, model=model,
            old_channels=geometry.channels, source_wiring=wiring, patch_size=patch,
            resize=resize, pending_key="_sensenova_vae_calibration_source",
            topology_config=lambda:native_config(model.config))
        if native_weight is not None:
            from core.models.components.patch_noise_alignment import POLICY, PatchNoiseAlignment
            pair = model.vae_latent_bridge
            projection = model.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding
            declaration = dict(policy=POLICY, channels=pair.declaration["new_channels"],
                bridge_channels=pair.input.projection.out_channels,
                hidden_channels=projection.out_channels, patch_size=projection.kernel_size[0])
            pair.declaration["patch_noise"] = declaration
            pair.declaration["patch_noise_source_stride"] = geometry.token_pixel_width
            pair.add_module("patch_noise", PatchNoiseAlignment(declaration))
            validate_patch_alignment(model)
            trainer._sensenova_patch_noise_source = native_weight
        return report
    except Exception:
        trainer.vae = old_vae
        raise


def calibrate(trainer, datasets):
    from core.models.components.spatial_swap import calibrate_images
    def capture():
        native = getattr(trainer, "_sensenova_patch_noise_source", None)
        if native is not None:
            pair = trainer.transformer.vae_latent_bridge
            pair.patch_noise.capture(native, pair.input.projection,
                trainer.transformer.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding)
            delattr(trainer, "_sensenova_patch_noise_source")
    return calibrate_images(trainer, datasets, model=trainer.transformer,
                           pending_key="_sensenova_vae_calibration_source", on_complete=capture)


def _spatial_weights(value, size, *, sum_preserving=False):
    old = value.shape[-1]
    if value.shape[-2] != old or type(size) is not int or size < 1:
        raise ValueError("SenseNova trained IO migration requires square positive patches")
    if old % size and size % old:
        # Fractional bin overlaps avoid adaptive pooling's duplicated boundary cells.
        edges = torch.arange(size + 1, device=value.device, dtype=torch.float64) * old / size
        cells = torch.arange(old, device=value.device, dtype=torch.float64)
        overlap = (torch.minimum(edges[1:, None], cells[None] + 1) -
                   torch.maximum(edges[:-1, None], cells[None])).clamp_min(0)
        average = (overlap * size / old).float()
        result = torch.matmul(torch.matmul(average, value.float()), average.T)
    elif size <= old:
        result = F.adaptive_avg_pool2d(value.float(), size)
    else:
        result = F.interpolate(value.float(), size=(size, size), mode="nearest")
    return result * (old / size) ** 2 if sum_preserving else result


def _expanded_refiner_io(refiner, channels):
    """Head and noisy-state halves must move independently in the stem."""
    if channels == refiner.in_channels:
        return None
    old = refiner.in_channels
    stem, output = refiner.stem, refiner.out
    if (channels < old or type(stem) is not nn.Conv2d or type(output) is not nn.Conv2d or
            stem.in_channels != 2 * old or output.out_channels != old or
            any(layer.groups != 1 or layer.kernel_size != (3, 3) or
                layer.stride != (1, 1) or layer.padding != (1, 1) or layer.dilation != (1, 1)
                for layer in (stem, output))):
        raise ValueError("SenseNova refiner expansion requires native channel-wise IO")
    new_stem = nn.Conv2d(2 * channels, stem.out_channels, 3, padding=1,
        bias=stem.bias is not None).to(device=stem.weight.device, dtype=stem.weight.dtype)
    new_output = nn.Conv2d(output.in_channels, channels, 3, padding=1,
        bias=output.bias is not None).to(device=output.weight.device, dtype=output.weight.dtype)
    with torch.no_grad():
        new_stem.weight.zero_()
        new_stem.weight[:, :old].copy_(stem.weight[:, :old])
        new_stem.weight[:, channels:channels + old].copy_(stem.weight[:, old:])
        if new_stem.bias is not None:
            new_stem.bias.copy_(stem.bias)
        new_output.weight.zero_()
        new_output.weight[:old].copy_(output.weight)
        if new_output.bias is not None:
            new_output.bias.zero_()
            new_output.bias[:old].copy_(output.bias)
    for new, source in ((new_stem, stem), (new_output, output)):
        new.weight.requires_grad_(source.weight.requires_grad)
        if new.bias is not None:
            new.bias.requires_grad_(source.bias.requires_grad)
    return new_stem, new_output


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
    if refiner is not None and refiner.in_channels != geometry.channels:
        raise ValueError("SenseNova refiner width disagrees with the generation head")
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
    expanded_refiner = _expanded_refiner_io(refiner, channels) if refiner is not None else None
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
    copied = weight_in.numel() + weight_out.numel()
    extra = new_in.weight.numel() + new_out.weight.numel() - copied
    replaced = ["fm_modules.vision_model_mot_gen.embeddings.patch_embedding", "fm_modules.fm_head.conv2"]
    if expanded_refiner is not None:
        source_elements = refiner.stem.weight.numel() + refiner.out.weight.numel()
        refiner.stem, refiner.out = expanded_refiner
        refiner.in_channels = channels
        copied += source_elements
        extra += refiner.stem.weight.numel() + refiner.out.weight.numel() - source_elements
        replaced.extend(("fm_modules.fm_refiner.stem", "fm_modules.fm_refiner.out"))
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
        replaced=tuple(replaced),
        old_in_channels=geometry.channels, old_out_channels=geometry.channels,
        new_channels=channels, copied_elements=copied, new_elements=extra)
