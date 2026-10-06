"""Validate the target codec before generation allocates conditioning or noise."""

from dataclasses import asdict

from core.models.components.video_latent_adapter import NormalizedVideoVAE


def generation_contract(components):
    model = components.get("transformer")
    inner = getattr(model, "transformer", model)
    contract = getattr(inner, "_video_codec_contract", None)
    vae = components.get("vae")
    if contract is None:
        if isinstance(vae, NormalizedVideoVAE):
            raise ValueError("H3 normalized codec requires an aligned backbone")
        return None
    if not isinstance(vae, NormalizedVideoVAE) or vae.contract != contract:
        raise ValueError("H3 generation codec disagrees with the aligned backbone")
    from .residual_unet import VIDEO_CODEC_KEY
    from .video_alignment import VIDEO_POSITION_POLICY_KEY, VIDEO_POSITION_POLICY
    if (inner.config.get(VIDEO_CODEC_KEY) != asdict(contract)
            or inner.config.get(VIDEO_POSITION_POLICY_KEY) != VIDEO_POSITION_POLICY):
        raise ValueError("H3 generation codec/clock configuration disagrees")
    pair = getattr(inner, "_video_latent_alignment", None)
    if pair is None or not pair.ready:
        raise ValueError("H3 generation requires a calibrated video bridge")
    if model.config.in_channels != contract.image.latent_channels:
        raise ValueError("H3 generation external channels disagree with the codec")
    for key, expected in (("latent_channels", contract.image.latent_channels),
                          ("vae_scale_factor_spatial", contract.image.spatial_scale),
                          ("vae_scale_factor_temporal", contract.temporal_scale)):
        if components.get(key) != expected:
            raise ValueError(f"H3 generation target geometry disagrees: {key}")
    return contract


def target_grid(contract, frames, height, width, *, topology=None):
    scale = contract.image.spatial_scale
    if any(type(value) is not int or value < 1 for value in (frames, height, width)):
        raise ValueError("H3 target frame count and dimensions must be positive integers")
    if height % (2*scale) or width % (2*scale):
        raise ValueError("H3 target dimensions must be divisible by codec scale times its 2x2 patch")
    grid = contract.latent_frames(frames), height//scale, width//scale
    if topology is not None:
        from core.rewiring.spec import preview_plan
        preview_plan(topology,grid[1]//2,grid[2]//2)
    return grid


def aligned_execution_wrapper(inner, **kwargs):
    from core.models.components.latent_adapter import _Config
    from .video_alignment import AlignedMiniMaxH3BlockLoopWrapper
    wrapper = AlignedMiniMaxH3BlockLoopWrapper(inner, **kwargs)
    wrapper.config = _Config(dict(inner.config),in_channels=inner._video_codec_contract.image.latent_channels)
    return wrapper


def validate_spatial_pin_groups(model, rows, grid):
    inner = getattr(model,"transformer",model)
    graph = getattr(inner,"_residual_unet_spec",None)
    if not rows or graph is None:
        return
    import torch
    from core.rewiring.spec import preview_plan
    frames,height,width = grid
    height,width = height//2,width//2
    if any(type(row) is not int or row < 0 or row >= frames*height*width for row in rows):
        raise ValueError("H3 spatial pins are outside the target patch grid")
    pins = torch.zeros(frames*height*width,dtype=torch.bool)
    pins[list(rows)] = True
    for level in preview_plan(graph,height,width)["levels"]:
        h,w = level["height"],level["width"]
        cells = pins.reshape(frames,h,height//h,w,width//w).permute(0,1,3,2,4)
        first = cells[...,0,0]
        if not torch.equal(cells,first[...,None,None].expand_as(cells)):
            raise ValueError("H3 spatial pins must cover whole U-net pooling cells; mixed clean/noisy labels are unsupported")
