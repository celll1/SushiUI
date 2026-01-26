"""
State dict key converter: diffusers format -> CompVis/LDM original format.
Based on: diffusers/scripts/convert_diffusers_to_original_sdxl.py
"""

import re
import torch

# =================#
# UNet Conversion #
# =================#

_unet_conversion_map = [
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
]

_unet_conversion_map_resnet = [
    ("in_layers.0", "norm1"),
    ("in_layers.2", "conv1"),
    ("out_layers.0", "norm2"),
    ("out_layers.3", "conv2"),
    ("emb_layers.1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
]

_unet_conversion_map_layer = []
for _i in range(3):
    for _j in range(2):
        _unet_conversion_map_layer.append((f"input_blocks.{3*_i+_j+1}.0.", f"down_blocks.{_i}.resnets.{_j}."))
        if _i > 0:
            _unet_conversion_map_layer.append((f"input_blocks.{3*_i+_j+1}.1.", f"down_blocks.{_i}.attentions.{_j}."))
    for _j in range(4):
        _unet_conversion_map_layer.append((f"output_blocks.{3*_i+_j}.0.", f"up_blocks.{_i}.resnets.{_j}."))
        if _i < 2:
            _unet_conversion_map_layer.append((f"output_blocks.{3*_i+_j}.1.", f"up_blocks.{_i}.attentions.{_j}."))
    if _i < 3:
        _unet_conversion_map_layer.append((f"input_blocks.{3*(_i+1)}.0.op.", f"down_blocks.{_i}.downsamplers.0.conv."))
        _unet_conversion_map_layer.append((f"output_blocks.{3*_i+2}.{1 if _i==0 else 2}.", f"up_blocks.{_i}.upsamplers.0."))
_unet_conversion_map_layer.append(("output_blocks.2.2.conv.", "output_blocks.2.1.conv."))
_unet_conversion_map_layer.append(("middle_block.1.", "mid_block.attentions.0."))
for _j in range(2):
    _unet_conversion_map_layer.append((f"middle_block.{2*_j}.", f"mid_block.resnets.{_j}."))


def convert_unet_state_dict_to_original(unet_state_dict: dict) -> dict:
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in _unet_conversion_map:
        if hf_name in mapping:
            mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in _unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in _unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    return {sd_name: unet_state_dict[hf_name] for hf_name, sd_name in mapping.items()}


# ================#
# VAE Conversion #
# ================#

_vae_conversion_map = [
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]
for _i in range(4):
    for _j in range(2):
        _vae_conversion_map.append((f"encoder.down.{_i}.block.{_j}.", f"encoder.down_blocks.{_i}.resnets.{_j}."))
    if _i < 3:
        _vae_conversion_map.append((f"down.{_i}.downsample.", f"down_blocks.{_i}.downsamplers.0."))
        _vae_conversion_map.append((f"up.{3-_i}.upsample.", f"up_blocks.{_i}.upsamplers.0."))
    for _j in range(3):
        _vae_conversion_map.append((f"decoder.up.{3-_i}.block.{_j}.", f"decoder.up_blocks.{_i}.resnets.{_j}."))
for _i in range(2):
    _vae_conversion_map.append((f"mid.block_{_i+1}.", f"mid_block.resnets.{_i}."))

_vae_conversion_map_attn = [
    ("norm.", "group_norm."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
]


def _reshape_weight_for_sd(w):
    if not w.ndim == 1:
        return w.reshape(*w.shape, 1, 1)
    return w


def convert_vae_state_dict_to_original(vae_state_dict: dict) -> dict:
    mapping = {k: k for k in vae_state_dict.keys()}
    for k, v in mapping.items():
        for sd_part, hf_part in _vae_conversion_map:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in _vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    for k, v in new_state_dict.items():
        for wn in ["q", "k", "v", "proj_out"]:
            if f"mid.attn_1.{wn}.weight" in k:
                new_state_dict[k] = _reshape_weight_for_sd(v)
    return new_state_dict


# =========================#
# Text Encoder Conversion #
# =========================#

_textenc_conversion_lst = [
    ("transformer.resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "text_model.final_layer_norm."),
    ("token_embedding.weight", "text_model.embeddings.token_embedding.weight"),
    ("positional_embedding", "text_model.embeddings.position_embedding.weight"),
]
_protected = {re.escape(x[1]): x[0] for x in _textenc_conversion_lst}
_textenc_pattern = re.compile("|".join(_protected.keys()))
_code2idx = {"q": 0, "k": 1, "v": 2}


def convert_openclip_text_enc_to_original(text_enc_dict: dict) -> dict:
    """Convert HF CLIPTextModelWithProjection state_dict to OpenCLIP format.
    Merges separate q_proj/k_proj/v_proj into concatenated in_proj_weight/bias.
    """
    new_state_dict = {}
    capture_qkv_weight = {}
    capture_qkv_bias = {}

    for k, v in text_enc_dict.items():
        if (k.endswith(".self_attn.q_proj.weight")
                or k.endswith(".self_attn.k_proj.weight")
                or k.endswith(".self_attn.v_proj.weight")):
            k_pre = k[: -len(".q_proj.weight")]
            k_code = k[-len("q_proj.weight")]
            if k_pre not in capture_qkv_weight:
                capture_qkv_weight[k_pre] = [None, None, None]
            capture_qkv_weight[k_pre][_code2idx[k_code]] = v
            continue

        if (k.endswith(".self_attn.q_proj.bias")
                or k.endswith(".self_attn.k_proj.bias")
                or k.endswith(".self_attn.v_proj.bias")):
            k_pre = k[: -len(".q_proj.bias")]
            k_code = k[-len("q_proj.bias")]
            if k_pre not in capture_qkv_bias:
                capture_qkv_bias[k_pre] = [None, None, None]
            capture_qkv_bias[k_pre][_code2idx[k_code]] = v
            continue

        relabelled_key = _textenc_pattern.sub(lambda m: _protected[re.escape(m.group(0))], k)
        new_state_dict[relabelled_key] = v

    for k_pre, tensors in capture_qkv_weight.items():
        if None in tensors:
            raise ValueError("Corrupted model: missing q-k-v weight in text encoder")
        relabelled_key = _textenc_pattern.sub(lambda m: _protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_weight"] = torch.cat(tensors)

    for k_pre, tensors in capture_qkv_bias.items():
        if None in tensors:
            raise ValueError("Corrupted model: missing q-k-v bias in text encoder")
        relabelled_key = _textenc_pattern.sub(lambda m: _protected[re.escape(m.group(0))], k_pre)
        new_state_dict[relabelled_key + ".in_proj_bias"] = torch.cat(tensors)

    return new_state_dict


def convert_openai_text_enc_to_original(text_enc_dict: dict) -> dict:
    """For CLIP ViT-L (SD1.5 TE, SDXL TE1), no key conversion needed."""
    return text_enc_dict
