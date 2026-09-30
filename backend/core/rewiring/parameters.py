"""Logical native parameter census, independent of quantized storage tensors."""


def parameter_counts(architecture, config):
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
