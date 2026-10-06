"""Inner-channel packed image boundaries for the dual Ideogram DiTs."""
import torch


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
