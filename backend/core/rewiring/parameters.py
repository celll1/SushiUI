"""Logical native parameter census, independent of quantized storage tensors."""


def parameter_counts(architecture, config):
    if architecture == "ming_image_01":
        return _ming_parameter_counts(config)
    if architecture != "qwen_image_21":
        return None
    from core.models.qwen_image_21.residual_unet import native_config
    config = native_config(config)
    keys = ("num_attention_heads", "attention_head_dim", "num_layers", "context_in_dim",
            "in_channels", "patch_size", "mlp_ratio")
    if any(type(config[key]) is not int or config[key] < 1 for key in keys):
        raise ValueError("Qwen parameter census requires positive integer native dimensions")
    out = config["out_channels"] or config["in_channels"]
    if type(out) is not int or out < 1:
        raise ValueError("Qwen output channels must be a positive integer")
    dim = config["num_attention_heads"] * config["attention_head_dim"]
    context = config["context_in_dim"]
    per_block = (4 + 3 * config["mlp_ratio"]) * dim**2 + 2 * config["attention_head_dim"]
    # Shared timestep/text projections, text norm, image projections and modulation.
    shared = (7 * dim**2 + 256 * dim + context + context * dim
              + (config["in_channels"] + out) * config["patch_size"]**2 * dim)
    blocks = [dict(stream="main", index=index, count=per_block)
              for index in range(config["num_layers"])]
    return dict(scope="native_dit", source="native_config_census",
                total=shared + len(blocks) * per_block, shared=shared, added=0, blocks=blocks)



def _ming_parameter_counts(config):
    keys = ("dim", "n_heads", "n_layers", "n_refiner_layers", "cap_feat_dim", "in_channels")
    if any(type(config[key]) is not int or config[key] < 1 for key in keys):
        raise ValueError("Ming parameter census requires positive integer native dimensions")
    d, heads, cap, channels = (config[key] for key in ("dim", "n_heads", "cap_feat_dim", "in_channels"))
    if d % heads:
        raise ValueError("Ming attention head geometry is inconsistent")
    modulation = min(d, 256)
    base = 4*d*d + 3*d*int(d/3*8) + 4*d + (2*(d//heads) if config["qk_norm"] else 0)
    main = base + 4*d*modulation + 4*d
    shared = 256*1024 + 1024 + 1024*modulation + modulation + cap + cap*d + d
    for patch, frame_patch in zip(config["all_patch_size"], config["all_f_patch_size"]):
        packed = patch*patch*frame_patch*channels
        shared += packed*d + d + d*packed + packed + modulation*d + d
    refiners = config["n_refiner_layers"]
    shared += refiners*(main + base)
    shared += 2*d if config["alignment_padding_mode"] == "learned" else 0
    blocks = [dict(stream="main", index=index, count=main) for index in range(config["n_layers"])]
    return dict(scope="native_dit", source="native_config_census", total=shared+len(blocks)*main,
                shared=shared, added=0, blocks=blocks)
