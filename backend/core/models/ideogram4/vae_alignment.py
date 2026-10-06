"""Inner-channel packed image boundaries for the dual Ideogram DiTs."""
import torch


def pair_metadata(transformer, unconditional_transformer):
    from core.models.components.spatial_swap import artifact_metadata
    first = artifact_metadata(transformer)
    second = artifact_metadata(unconditional_transformer) if unconditional_transformer is not None else {}
    if first != second:
        raise ValueError("Ideogram conditional and unconditional alignment declarations disagree")
    return first


def apply_swap(trainer,resolved,*,handler,module=None):
    from dataclasses import replace
    from core.models.components.latent_io import ResizeReport
    model,twin = trainer.transformer,trainer.transformer_uncond
    pair = getattr(model,"vae_latent_bridge",None)
    if pair is not None:
        pair_metadata(model,twin)
        if module is None and resolved.source != pair.declaration["source"]:
            raise ValueError("Changing an aligned Ideogram VAE requires a fresh base checkpoint")
        trainer.vae_identity = replace(resolved,state_dict=None)
        trainer.vae_latent_channels = pair.declaration["new_channels"]
        trainer.wiring = handler.resolve_wiring(trainer)
        width = model.in_channels//4
        return ResizeReport((),width,width,width,0,0)
    if module is not None:
        raise ValueError("Restoring an Ideogram swapped codec requires its saved paired alignment")
    from api.param_defaults import TRAINING_DEFAULTS
    mode = trainer.config.get("vae_swap_new_channel_init",TRAINING_DEFAULTS["vae_swap_new_channel_init"])
    if mode not in ("auto","paired_spatial"):
        raise ValueError("Ideogram spatial swaps require paired_spatial initialization")
    return initialize_swap(trainer,resolved,handler=handler)


def initialize_swap(trainer,resolved,*,handler):
    from copy import deepcopy
    from core.models.components.spatial_swap import initialize_image_swap
    from core.training.ops.training_method import is_full_finetune
    if not is_full_finetune(trainer) or not getattr(trainer,"ideogram4_train_uncond",False):
        raise ValueError("Ideogram spatial swaps require full_finetune with both DiTs trained")
    model, twin = trainer.transformer, trainer.transformer_uncond
    report = initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.in_channels//4,source_wiring=handler.resolve_wiring(trainer),patch_size=2,
        resize=lambda channels:resize_pair(model,twin,channels),pending_key="_ideogram_vae_calibration_source")
    if hasattr(model,"vae_latent_bridge"):
        twin.add_module("vae_latent_bridge",deepcopy(model.vae_latent_bridge))
    return report


def calibrate(trainer,datasets):
    from core.models.components.spatial_swap import calibrate_images
    if not hasattr(trainer,"_ideogram_vae_calibration_source"):
        return
    from core.training.optimizer_residency import park_optimizer_state
    from core.training.base_trainer import all_optimizers
    twin = trainer.transformer_uncond
    devices = {name:tensor.device for name,tensor in (*twin.named_parameters(),*twin.named_buffers())}
    with park_optimizer_state(all_optimizers(trainer),tuple(twin.parameters())):
        try:
            twin.to("cpu")
            calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_ideogram_vae_calibration_source")
            twin.vae_latent_bridge.load_state_dict(trainer.transformer.vae_latent_bridge.state_dict(),strict=True)
        finally:
            current = dict((*twin.named_parameters(),*twin.named_buffers()))
            for name,device in devices.items():
                tensor = current[name]
                if tensor.device != device:
                    tensor.data = tensor.data.to(device)


def restore_model_alignment(model,declaration):
    from core.models.components.spatial_swap import install
    if declaration is None:
        return
    channels = max(declaration["old_channels"],declaration["new_channels"])
    if model.in_channels != channels*4 or model.final_layer.linear.out_features != channels*4:
        raise ValueError("Ideogram native IO disagrees with the saved internal latent width")
    install(model,declaration)


def resize_pair(transformer, unconditional_transformer, channels):
    from torch import nn
    from core.models.components.latent_io import resize_latent_io, ResizeReport
    from core.models.components.wiring import IDEOGRAM4_LATENT_IO
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.rewiring.spec import build_spec
    from .residual_unet import native_config, restore_topology, pair_topology_metadata
    if unconditional_transformer is None:
        raise ValueError("Ideogram spatial migration requires both DiTs")
    models = (transformer,unconditional_transformer)
    old = transformer.in_channels // 4
    if type(channels) is not int or channels < old:
        raise ValueError("Ideogram spatial migration must preserve every internal channel")
    pair_topology_metadata(*models)
    for model in models:
        reject_quantized_base(model,model_label="Ideogram spatial migration")
        if (type(model.input_proj) is not nn.Linear or type(model.final_layer.linear) is not nn.Linear or
                model.in_channels != old*4 or model.input_proj.in_features != old*4 or
                model.final_layer.linear.out_features != old*4):
            raise ValueError("Ideogram spatial migration requires matching native packed DiT IO")
    reports = []
    for model in models:
        topology = getattr(model,"_residual_unet_spec",None)
        reports.append(resize_latent_io(model,IDEOGRAM4_LATENT_IO,channels))
        model.in_channels = model.out_channels = channels*4
        model.register_to_config(in_channels=channels*4)
        if topology is not None:
            restore_topology(model,build_spec("ideogram4",[{"name":"main","block_count":len(model.layers)}],
                topology["regions"],native_config(model)))
    return ResizeReport(tuple(prefix+path for prefix,report in zip(("transformer.","unconditional_transformer."),reports)
        for path in report.replaced),old,old,channels,
        sum(report.copied_elements for report in reports),sum(report.new_elements for report in reports))


def image_grid(position_ids, indicator):
    from .vendor.transformer import IMAGE_POSITION_OFFSET, OUTPUT_IMAGE_INDICATOR
    if position_ids.ndim != 3 or position_ids.shape[-1] != 3 or indicator.shape != position_ids.shape[:2]:
        raise ValueError("Aligned Ideogram requires batched native image positions")
    cached = getattr(position_ids, "_ideogram_alignment_grid", None)
    if (cached is not None and cached[0] == position_ids._version and
            cached[1] is indicator and cached[2] == indicator._version):
        return cached[3]
    mask = indicator == OUTPUT_IMAGE_INDICATOR
    indexes = mask[0].nonzero().flatten()
    if indexes.numel() == 0:
        raise ValueError("Aligned Ideogram requires a nonempty image suffix")
    prefix = int(indexes[0])
    if not bool(mask[:,prefix:].all()) or bool(mask[:,:prefix].any()):
        raise ValueError("Aligned Ideogram requires a common contiguous image suffix")
    image = position_ids[:,prefix:] - IMAGE_POSITION_OFFSET
    h,w = int(image[0,-1,1])+1,int(image[0,-1,2])+1
    if h < 1 or w < 1 or h*w != image.shape[1]:
        raise ValueError("Aligned Ideogram requires a rectangular image grid")
    expected = torch.stack((torch.zeros(h*w,device=image.device,dtype=image.dtype),
        torch.arange(h,device=image.device).repeat_interleave(w),
        torch.arange(w,device=image.device).repeat(h)),dim=-1)
    if not torch.equal(image,expected[None].expand_as(image)):
        raise ValueError("Aligned Ideogram requires native row-major image positions")
    result = prefix,(h,w)
    position_ids._ideogram_alignment_grid = position_ids._version,indicator,indicator._version,result
    return result


def map_image_suffix(pair, states, grid, *, prefix, timestep=None, output=False):
    from core.models.components.spatial_swap import map_packed_image
    image = map_packed_image(pair, states[:,prefix:], grid,
        sigma=None if output else 1-timestep, output=output,
        channel_order="inner", convention="x0_minus_eps")
    if prefix:
        image = torch.cat((image.new_zeros(image.shape[0],prefix,image.shape[-1]),image),dim=1)
    return image
