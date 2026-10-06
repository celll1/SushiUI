"""Native H3 codec facts, including the nonuniform chunk frame policy."""

from math import isfinite, prod

FRAME_POLICY = "minimax_h3_chunks_v1"


def transform_pixels(pixels, *, encode):
    from .loader import MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD
    if pixels.ndim not in (4, 5) or pixels.shape[1] != 3:
        raise ValueError("H3 pixel normalization requires BCHW or BCTHW RGB")
    shape = (1,3)+(1,)*(pixels.ndim-2)
    mean = pixels.new_tensor(MINIMAX_H3_PIXEL_MEAN).view(shape)
    std = pixels.new_tensor(MINIMAX_H3_PIXEL_STD).view(shape)
    return ((pixels+1)*.5-mean)/std if encode else (pixels*std+mean)*2-1


def resolve_facts(config, shapes):
    blocks = config.get("block_out_channels")
    spatial = config.get("spatial_downsample_factors")
    temporal = config.get("temporal_downsample_factors")
    channels = config.get("latent_channels")
    if (not isinstance(blocks, (list, tuple)) or not blocks
            or any(type(n) is not int or n < 1 for n in blocks)
            or type(channels) is not int or channels < 1):
        raise ValueError("H3 codec channel geometry is missing")
    for factors in (spatial, temporal):
        if (not isinstance(factors, (list, tuple)) or len(factors) != len(blocks)
                or any(type(n) is not int or n not in (1, 2) for n in factors)):
            raise ValueError("H3 codec downsampling geometry is missing")
    if (prod(temporal) != 4 or config.get("clip_length") != 17 or config.get("token_drop") != 3
            or any(t > 1 and s != 2 for s, t in zip(spatial, temporal))):
        raise ValueError("H3 codec requires its declared 17-frame chunk and temporal stride 4")
    if config.get("in_channels") != 3 or config.get("out_channels") != 3:
        raise ValueError("H3 codec requires RGB encoding and decoding")
    heads, head_dim = config.get("decoder_num_attention_heads"), config.get("decoder_attention_head_dim")
    if any(type(n) is not int or n < 1 for n in (heads, head_dim)):
        raise ValueError("H3 codec decoder geometry is missing")
    width, scale = heads*head_dim, prod(spatial)
    expected = {
        "encoder.conv_in.weight": (blocks[0], 3, 3, 3, 3),
        "encoder.conv_out.weight": (2*channels, blocks[-1], 3, 3, 3),
        "quant_conv.weight": (2*channels, 2*channels, 1, 1, 1),
        "post_quant_conv.weight": (channels, channels, 1, 1, 1),
        "decoder.proj_in.weight": (width, channels),
        "decoder.proj_out.weight": (3*scale*scale*4, width),
    }
    if any(shapes.get(key) != shape for key, shape in expected.items()):
        raise ValueError("H3 codec headers disagree with its config")
    for key in ("latents_mean", "latents_std"):
        values = config.get(key)
        if (not isinstance(values, (list, tuple)) or len(values) != channels
                or any(type(n) not in (int, float) or not isfinite(n)
                       or (key == "latents_std" and n <= 0) for n in values)):
            raise ValueError("H3 codec per-channel normalization is missing or invalid")
    if config.get("scaling_factor") not in (None, 1, 1.0) or config.get("shift_factor") not in (None, 0, 0.0):
        raise ValueError("H3 codec requires per-channel normalization without scalar scaling")
    return dict(latent_channels=channels, ndim=5, scale_factor=scale, scale_temporal=4,
                vae_class="AutoencoderKLMiniMaxH3", norm="per_channel", norm_pack=1,
                scaling_factor=None, shift_factor=None)
