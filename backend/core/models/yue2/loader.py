"""Strict, header-first adapter for complete Comfy-Org YuE2 checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import struct

import torch
from safetensors import safe_open

TOKENIZER_KEY = "text_encoders.yue2_tokenizer_json"
UPSTREAM_REVISION = "88da114a67df892af0329472073b96a5ef700b93"


def read_header(path):
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError("Invalid YuE2 safetensors header length")
        return json.loads(stream.read(size))


def keys_look_yue2(keys, metadata=None):
    keys = set(keys)
    return {"model.diffusion_model.vae2llm.weight",
            "model.diffusion_model.latent_pos_embed.pe"} <= keys


def is_yue2_checkpoint(path):
    if not str(path).lower().endswith(".safetensors") or not Path(path).is_file():
        return False
    try:
        header = read_header(path)
        metadata = header.get("__metadata__", {})
        return (keys_look_yue2(header) or metadata.get("yue2_format") is not None
                or metadata.get("model_type") == "yue2")
    except (ValueError, OSError, struct.error):
        return False


@dataclass(frozen=True)
class Mapping:
    source: str
    target: str
    start: int | None = None
    end: int | None = None


def key_mapping(key):
    if key == TOKENIZER_KEY:
        return [Mapping(key, "tokenizer")]
    if key.startswith("vae."):
        return [Mapping(key, key)]
    nar = key.startswith("model.diffusion_model.")
    if nar:
        target = key.removeprefix("model.diffusion_model.")
        if target == "model.norm.weight":
            target = "nar_norm.weight"
        target = re.sub(r"(model.layers.\d+.)input_layernorm", r"\1nar_input_layernorm", target)
        target = re.sub(r"(model.layers.\d+.)post_attention_layernorm", r"\1nar_pre_mlp_layernorm", target)
        target = re.sub(r"(model.layers.\d+.)self_attn", r"\1nar_self_attn", target)
        target = re.sub(r"(model.layers.\d+.)mlp", r"\1nar_mlp", target)
    elif key.startswith("text_encoders."):
        target = key.removeprefix("text_encoders.")
        target = target.replace("model.lm_head.", "lm_head.")
    else:
        raise ValueError(f"Unrecognized YuE2 tensor: {key}")
    for fused, names, widths in (("qkv_proj", ("q_proj", "k_proj", "v_proj"), (2048, 1024, 1024)),
                                 ("gate_up_proj", ("gate_proj", "up_proj"), (6144, 6144))):
        if f".{fused}." in target:
            result, offset = [], 0
            for name, width in zip(names, widths):
                marker = target.endswith(".comfy_quant")
                result.append(Mapping(key, target.replace(f".{fused}.", f".{name}."),
                                      None if marker else offset, None if marker else offset + width))
                offset += width
            return result
    return [Mapping(key, target)]


def build_empty_models():
    from .vendor.modeling_yue2 import YuE2Config, YuE2ForCausalLM, RMSNorm
    from .vendor.modeling_vae import YuE2VAEConfig, YuE2VAE
    with torch.device("meta"):
        model = YuE2ForCausalLM(YuE2Config())
        model.nar_norm = RMSNorm(model.config.hidden_size, model.config.rms_norm_eps)
        vae = YuE2VAE(YuE2VAEConfig(), decoder_only=False)
    return model, vae


def preflight_yue2(path, *, require_runtime=False):
    """Read header, tiny markers, and tokenizer; never materialize model weights."""
    from core.models.common.convrot_marker import supported_int8_convrot_marker
    header = read_header(path)
    if not keys_look_yue2(header):
        raise ValueError("Not a complete YuE2 checkpoint")
    if TOKENIZER_KEY not in header:
        raise ValueError("YuE2 checkpoint is missing its tokenizer")
    metadata = header.get("__metadata__", {})
    if metadata.get("yue2_format") != "1":
        raise ValueError("Unsupported YuE2 checkpoint format version")
    model, vae = build_empty_models()
    expected = {**dict(model.state_dict()), **{"vae." + k: v for k, v in vae.state_dict().items()}}
    plan, targets, quantized = [], {}, {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key, info in header.items():
            if key == "__metadata__":
                continue
            mappings = key_mapping(key)
            if key.endswith(".comfy_quant"):
                if info.get("dtype") != "U8" or len(info.get("shape", [])) != 1 or info["shape"][0] > 4096:
                    raise ValueError(f"Invalid YuE2 quantization marker: {key}")
                config = supported_int8_convrot_marker(key, handle.get_tensor(key), header, path=str(path))
                if config is None:
                    raise ValueError(f"Unsupported YuE2 quantization marker: {key}")
                for item in mappings:
                    if item.target.startswith("vae."):
                        raise ValueError("Quantized YuE2 VAE tensors are unsupported")
                    quantized[item.target.removesuffix(".comfy_quant")] = config
            elif info.get("dtype") == "I8" or key.endswith(".weight_scale"):
                base = key.rsplit(".", 1)[0]
                if base + ".comfy_quant" not in header:
                    raise ValueError(f"Missing YuE2 quantization declaration: {key}")
            if key != TOKENIZER_KEY and not key.endswith((".comfy_quant", ".weight_scale")):
                for item in mappings:
                    shape = list(info["shape"])
                    if item.start is not None:
                        if shape[0] != (4096 if ".qkv_proj." in key else 12288):
                            raise ValueError(f"Partial fused YuE2 projection: {key}")
                        shape[0] = item.end - item.start
                    if item.target not in expected or shape != list(expected[item.target].shape):
                        raise ValueError(f"Unknown or wrong-shape YuE2 tensor: {key} -> {item.target} {shape}")
                    if info["dtype"] not in {"I8", "F16", "BF16", "F32"}:
                        raise ValueError(f"Unsupported YuE2 tensor dtype: {key}")
            for item in mappings:
                if item.target in targets:
                    raise ValueError(f"Duplicate YuE2 destination: {item.target}")
                targets[item.target] = item
                plan.append(item)
        missing = set(expected) - set(targets)
        if missing:
            raise ValueError(f"Incomplete YuE2 checkpoint: {sorted(missing)[:8]}")
        for key in targets:
            if key.endswith((".comfy_quant", ".weight_scale")):
                base = key.rsplit(".", 1)[0]
                if base + ".weight" not in expected or base not in quantized:
                    raise ValueError(f"Unexpected YuE2 quantization sibling: {key}")
        tok_info = header[TOKENIZER_KEY]
        if tok_info["dtype"] != "U8" or len(tok_info["shape"]) != 1 or not 0 < tok_info["shape"][0] <= 32 * 1024 * 1024:
            raise ValueError("Invalid YuE2 tokenizer payload")
        tokenizer = EmbeddedTokenizer(bytes(handle.get_tensor(TOKENIZER_KEY).tolist()).decode("utf-8"))
    if require_runtime and quantized:
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime
        require_convrot_int8_runtime()
    return dict(plan=plan, quantized=quantized, metadata=metadata, tokenizer=tokenizer,
                source_tensor_count=len(header) - int("__metadata__" in header))


class EmbeddedTokenizer:
    def __init__(self, payload):
        from tokenizers import Tokenizer
        self.payload = payload
        self.tokenizer = Tokenizer.from_str(payload)
        if self.tokenizer.get_vocab_size(with_added_tokens=False) != 151643:
            raise ValueError("YuE2 ordinary vocabulary must contain exactly 151643 tokens")

    def encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if any(not 0 <= token < 151643 for token in ids):
            raise ValueError("YuE2 text encoded outside the ordinary vocabulary")
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


class ConvRotEmbedding(torch.nn.Module):
    """Gather quantized rows before undoing rotation; no full-table allocation."""
    def __init__(self, rows, width, marker_numel, dtype):
        super().__init__()
        self.register_buffer("weight", torch.empty((rows, width), dtype=torch.int8, device="meta"))
        self.register_buffer("weight_scale", torch.empty(rows, dtype=torch.float32, device="meta"))
        self.register_buffer("comfy_quant", torch.empty(marker_numel, dtype=torch.uint8, device="meta"))
        self.compute_dtype = dtype

    def forward(self, ids):
        import comfy_kitchen  # noqa: F401
        rows = self.weight[ids.reshape(-1)].contiguous()
        scales = self.weight_scale[ids.reshape(-1)].reshape(-1, 1).contiguous()
        dtype_code = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[self.compute_dtype]
        value = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(rows, scales, 256, dtype_code)
        return value.reshape(*ids.shape, rows.shape[-1])


def load_yue2_from_path(path, torch_dtype=torch.bfloat16):
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    preflight = preflight_yue2(path, require_runtime=True)
    model, vae = build_empty_models()
    for name, config in preflight["quantized"].items():
        module = model.get_submodule(name)
        if isinstance(module, torch.nn.Embedding):
            replacement = ConvRotEmbedding(module.num_embeddings, module.embedding_dim, config["marker_numel"], torch_dtype)
        elif isinstance(module, torch.nn.Linear):
            replacement = ConvRotInt8Linear(module.in_features, module.out_features,
                                            module.bias is not None, torch_dtype, device="meta", **config)
        else:
            raise ValueError(f"Unsupported quantized YuE2 module: {name}")
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, replacement)
    model_state, vae_state = {}, {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for item in preflight["plan"]:
            if item.target == "tokenizer":
                continue
            value = handle.get_tensor(item.source)
            if item.start is not None:
                value = value[item.start:item.end]
            if item.target.startswith("vae."):
                vae_state[item.target[4:]] = value.float()
            else:
                if item.target.endswith(".weight_scale"):
                    value = value.reshape(-1)
                if value.is_floating_point() and not item.target.endswith(".weight_scale"):
                    value = value.to(torch_dtype)
                model_state[item.target] = value
    model.load_state_dict(model_state, strict=True, assign=True)
    vae.load_state_dict(vae_state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    assert not any(t.is_meta for t in list(model.parameters()) + list(model.buffers()))
    return dict(type="yue2", transformer=model, vae=vae, tokenizer=preflight["tokenizer"],
                sample_rate=48000, frame_rate=25, latent_channels=64,
                model_identity={**preflight["metadata"], "upstream_revision": UPSTREAM_REVISION,
                                "checkpoint": Path(path).name, "vae_execution_dtype": "float32",
                                "weight_storage": ("convrot_int8" if preflight["quantized"]
                                                   else "dense_bf16")})
