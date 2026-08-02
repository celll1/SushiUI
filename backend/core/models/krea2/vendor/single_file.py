# The raw-format key names and layout handled here derive from krea-ai/krea-2
# (https://github.com/krea-ai/krea-2, mmdit.py), licensed under Apache-2.0.
# See vendor/transformer.py for the full license header.
"""Weight-format handling for Krea 2 (load + save single file).

Supports the five checkpoint layouts described in the integration plan:

  1. Diffusers folder / transformer-only (diffusers key names) -- handled by the
     loader, which feeds the assembled state dict through ``normalize_state_dict``.
  2. Transformer-only safetensors (diffusers key format) -- pass-through remap.
  3. Official raw format (krea-ai/krea-2 mmdit.py keys: ``first``, ``blocks``,
     ``tmlp``, ``tproj``, ``txtfusion``, ``txtmlp``, ``last``) -- key remap to the
     vendored diffusers-format model.
  4. ComfyUI single checkpoints (keys optionally prefixed ``model.diffusion_model.``;
     raw key names underneath). ``bf16`` and ``fp8_scaled`` (weight-only FP8 with
     per-tensor ``.scale_weight`` on the 28 main blocks) are supported by
     converting the per-tensor scale to the per-row ``.weight_scale`` layout and
     reusing the ideogram4 FP8 weight-only loader. ``int8_convrot`` / ``mxfp8`` /
     ``nvfp4`` are rejected with a clear error.
  5. sushiUI-trained single file (TE+DiT combined): keys ``transformer.*`` (+
     optional ``text_encoder.*``), diffusers key names, metadata carries the config
     and variant. Mirrors minit2i vendor/single_file.py.

Raw -> diffusers key remap summary (examples):
    first.weight                          -> img_in.weight
    tmlp.0.weight / tmlp.2.weight         -> time_embed.linear_1 / linear_2.weight
    tproj.1.weight                        -> time_mod_proj.weight
    txtmlp.0.scale / .1 / .3              -> txt_in.norm.weight / linear_1 / linear_2
    txtfusion.projector.weight            -> text_fusion.projector.weight
    txtfusion.layerwise_blocks.0.prenorm.scale -> text_fusion.layerwise_blocks.0.norm1.weight
    blocks.0.mod.lin (6*dim,)             -> transformer_blocks.0.scale_shift_table (6, dim)
    blocks.0.prenorm.scale / postnorm     -> transformer_blocks.0.norm1.weight / norm2.weight
    blocks.0.attn.wq/wk/wv/gate/wo        -> transformer_blocks.0.attn.to_q/to_k/to_v/to_gate/to_out.0
    blocks.0.attn.qknorm.qnorm.scale      -> transformer_blocks.0.attn.norm_q.weight
    blocks.0.mlp.gate/up/down             -> transformer_blocks.0.ff.gate/up/down
    last.norm.scale / last.linear         -> final_layer.norm.weight / final_layer.linear
    last.modulation.lin (2, dim)          -> final_layer.scale_shift_table
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

import torch

from core.models.common.single_file_format import (
    DEFAULT_MAX_SHARD_BYTES,
    VAE_PREFIX,
    dedup_tensors,
    read_state_dict,
    save_single_file_state,
)

from .transformer import Krea2Transformer2DModel

# Reuse the ideogram4 weight-only FP8 machinery (per-row e4m3 + float32 scale).
from core.models.ideogram4.vendor.fp8_linear import (
    FP8_WEIGHT_DTYPE,
    is_fp8_state_dict,
    load_fp8_state_dict,
    swap_linears_to_fp8,
)
# ... and the INT8 sibling (per-row int8 + float32 scale, same suffix).
from core.models.ideogram4.vendor.int8_linear import (
    is_int8_state_dict,
    swap_linears_to_int8,
)


TRANSFORMER_PREFIX = "transformer."
TEXT_ENCODER_PREFIX = "text_encoder."
COMFY_PREFIX = "model.diffusion_model."

# Verified Krea/Krea-2-Raw transformer config (~12.9B params, hidden 6144).
KREA2_DEFAULT_CONFIG: Dict = {
    "in_channels": 64,
    "num_layers": 28,
    "attention_head_dim": 128,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "intermediate_size": 16384,
    "timestep_embed_dim": 256,
    "text_hidden_dim": 2560,
    "num_text_layers": 12,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "text_intermediate_size": 6912,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": [32, 48, 48],
    "rope_theta": 1000.0,
    "norm_eps": 1e-05,
}

# Quantization layouts that are out of scope (comfy variants beyond bf16/fp8_scaled).
_REJECTED_QUANT_TOKENS = ("int8_convrot", "mxfp8", "nvfp4")


# ---------------------------------------------------------------------------
# Raw (mmdit.py) -> diffusers key remap
# ---------------------------------------------------------------------------

def _remap_attn_leaf(rest: str) -> str:
    """Map a raw Attention submodule path (after ``attn.``) to the diffusers name."""
    m = re.match(r"(wq|wk|wv|gate|wo)\.(.+)$", rest)
    if m:
        proj = {"wq": "to_q", "wk": "to_k", "wv": "to_v", "gate": "to_gate", "wo": "to_out.0"}[m.group(1)]
        return f"{proj}.{m.group(2)}"
    if rest == "qknorm.qnorm.scale":
        return "norm_q.weight"
    if rest == "qknorm.knorm.scale":
        return "norm_k.weight"
    return rest


def _remap_fusion_block_leaf(rest: str) -> str:
    """Map a raw TextFusionBlock submodule path to the diffusers name."""
    if rest == "prenorm.scale":
        return "norm1.weight"
    if rest == "postnorm.scale":
        return "norm2.weight"
    m = re.match(r"attn\.(.+)$", rest)
    if m:
        return f"attn.{_remap_attn_leaf(m.group(1))}"
    m = re.match(r"mlp\.(.+)$", rest)
    if m:
        return f"ff.{m.group(1)}"
    return rest


def is_raw_state_dict(sd: Dict[str, torch.Tensor]) -> bool:
    """True for the krea-ai/krea-2 mmdit.py raw key layout."""
    return (
        any(k == "first.weight" or k.startswith("first.") for k in sd)
        and any(k.startswith("txtfusion.") for k in sd)
        and any(k.startswith("blocks.") for k in sd)
    )


def remap_raw_to_diffusers(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap a raw-format state dict to the vendored diffusers key names.

    Reshapes the shared-modulation tables (``blocks.N.mod.lin`` 6*dim -> (6, dim)).
    ``.scale_weight`` (comfy per-tensor FP8 scale) is carried through and converted
    to ``.weight_scale`` by ``_convert_scale_weight`` afterwards.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk: Optional[str] = None

        m = re.match(r"blocks\.(\d+)\.mod\.lin$", k)
        if m:
            hidden = v.shape[0] // 6
            out[f"transformer_blocks.{m.group(1)}.scale_shift_table"] = v.reshape(6, hidden).contiguous()
            continue
        m = re.match(r"blocks\.(\d+)\.prenorm\.scale$", k)
        if m:
            out[f"transformer_blocks.{m.group(1)}.norm1.weight"] = v
            continue
        m = re.match(r"blocks\.(\d+)\.postnorm\.scale$", k)
        if m:
            out[f"transformer_blocks.{m.group(1)}.norm2.weight"] = v
            continue
        m = re.match(r"blocks\.(\d+)\.attn\.(.+)$", k)
        if m:
            out[f"transformer_blocks.{m.group(1)}.attn.{_remap_attn_leaf(m.group(2))}"] = v
            continue
        m = re.match(r"blocks\.(\d+)\.mlp\.(.+)$", k)
        if m:
            out[f"transformer_blocks.{m.group(1)}.ff.{m.group(2)}"] = v
            continue

        if k == "txtfusion.projector.weight":
            out["text_fusion.projector.weight"] = v
            continue
        m = re.match(r"txtfusion\.(layerwise_blocks|refiner_blocks)\.(\d+)\.(.+)$", k)
        if m:
            out[f"text_fusion.{m.group(1)}.{m.group(2)}.{_remap_fusion_block_leaf(m.group(3))}"] = v
            continue

        m = re.match(r"first\.(weight|bias)$", k)
        if m:
            out[f"img_in.{m.group(1)}"] = v
            continue
        m = re.match(r"tmlp\.0\.(weight|bias)$", k)
        if m:
            out[f"time_embed.linear_1.{m.group(1)}"] = v
            continue
        m = re.match(r"tmlp\.2\.(weight|bias)$", k)
        if m:
            out[f"time_embed.linear_2.{m.group(1)}"] = v
            continue
        m = re.match(r"tproj\.1\.(weight|bias)$", k)
        if m:
            out[f"time_mod_proj.{m.group(1)}"] = v
            continue
        if k == "txtmlp.0.scale":
            out["txt_in.norm.weight"] = v
            continue
        m = re.match(r"txtmlp\.1\.(weight|bias)$", k)
        if m:
            out[f"txt_in.linear_1.{m.group(1)}"] = v
            continue
        m = re.match(r"txtmlp\.3\.(weight|bias)$", k)
        if m:
            out[f"txt_in.linear_2.{m.group(1)}"] = v
            continue
        if k == "last.norm.scale":
            out["final_layer.norm.weight"] = v
            continue
        m = re.match(r"last\.linear\.(weight|bias)$", k)
        if m:
            out[f"final_layer.linear.{m.group(1)}"] = v
            continue
        if k == "last.modulation.lin":
            out["final_layer.scale_shift_table"] = v.contiguous()
            continue

        # Unknown raw key -> keep verbatim; surfaced as an unexpected key on load.
        out[k] = v
    return out


def _convert_scale_weight(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert comfy per-tensor FP8 ``<name>.scale_weight`` to the per-row
    ``<name>.weight_scale`` layout expected by the ideogram4 FP8 loader.

    A per-tensor scalar is broadcast to ``(out_features,)`` using the matching
    ``<name>.weight`` shape so a single Fp8Linear code path handles both.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.endswith(".scale_weight"):
            base = k[: -len(".scale_weight")]
            weight = sd.get(f"{base}.weight")
            scale = v.detach().to(torch.float32).flatten()
            if weight is not None and scale.numel() == 1:
                scale = scale.expand(weight.shape[0]).contiguous()
            out[f"{base}.weight_scale"] = scale
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Normalization entry point
# ---------------------------------------------------------------------------

def normalize_state_dict(
    raw: Dict[str, torch.Tensor], metadata: Optional[dict] = None
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    """Return (diffusers_transformer_sd, text_encoder_sd_or_None) from any layout.

    Handles the sushiUI ``transformer.``/``text_encoder.`` split, the comfy
    ``model.diffusion_model.`` prefix, and the raw -> diffusers key remap. FP8
    ``.scale_weight`` scales are converted to ``.weight_scale``.
    """
    metadata = metadata or {}

    te_sd: Optional[Dict[str, torch.Tensor]] = None

    # sushiUI split format: transformer.* (+ optional text_encoder.*).
    if any(k.startswith(TRANSFORMER_PREFIX) for k in raw):
        te_sd = {k[len(TEXT_ENCODER_PREFIX):]: v for k, v in raw.items() if k.startswith(TEXT_ENCODER_PREFIX)} or None
        sd = {k[len(TRANSFORMER_PREFIX):]: v for k, v in raw.items() if k.startswith(TRANSFORMER_PREFIX)}
    else:
        sd = dict(raw)

    # Comfy prefix strip.
    if any(k.startswith(COMFY_PREFIX) for k in sd):
        sd = {(k[len(COMFY_PREFIX):] if k.startswith(COMFY_PREFIX) else k): v for k, v in sd.items()}

    # Raw -> diffusers key remap (no-op for diffusers-format keys).
    if is_raw_state_dict(sd):
        sd = remap_raw_to_diffusers(sd)

    # Per-tensor -> per-row FP8 scale conversion.
    if any(k.endswith(".scale_weight") for k in sd):
        sd = _convert_scale_weight(sd)

    return sd, te_sd


def detect_config_and_variant(
    metadata: dict, sd: Dict[str, torch.Tensor]
) -> Tuple[Dict, bool]:
    """Return (config_dict, is_distilled). Prefers the metadata config; falls back
    to the verified default config. is_distilled from metadata variant/flag."""
    config = dict(KREA2_DEFAULT_CONFIG)
    cfg_json = metadata.get("krea2_config")
    if cfg_json:
        try:
            config.update(json.loads(cfg_json))
        except Exception:
            pass

    # Infer num_layers from the checkpoint if the metadata config is absent.
    if not cfg_json:
        depths = {
            int(m.group(1))
            for k in sd
            if (m := re.match(r"transformer_blocks\.(\d+)\.", k))
        }
        if depths:
            config["num_layers"] = max(depths) + 1

    is_distilled = _detect_is_distilled(metadata)
    return config, is_distilled


def _detect_is_distilled(metadata: dict) -> bool:
    variant = str(metadata.get("variant", "")).lower()
    if variant in ("turbo", "distilled", "tdm"):
        return True
    flag = metadata.get("is_distilled")
    if isinstance(flag, str):
        return flag.strip().lower() in ("1", "true", "yes")
    return bool(flag)


def reject_unsupported_quant(path: str, metadata: dict) -> None:
    """Raise for comfy quant layouts that are out of scope (int8_convrot/mxfp8/nvfp4)."""
    haystack = (path or "").lower() + " " + str(metadata.get("quantization", "")).lower()
    for token in _REJECTED_QUANT_TOKENS:
        if token in haystack:
            raise ValueError(
                f"[Krea2] quantization layout '{token}' is not supported. "
                f"Use a bf16 or fp8_scaled checkpoint."
            )


# ---------------------------------------------------------------------------
# Build / load / save
# ---------------------------------------------------------------------------

def build_krea2_transformer(
    diffusers_sd: Dict[str, torch.Tensor],
    config: Dict,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Krea2Transformer2DModel:
    """Instantiate a Krea2Transformer2DModel from config and load a diffusers-format
    state dict. Weight-only quantized checkpoints (per-row ``.weight_scale``) keep
    their float8/int8 Linear weights; everything else loads as ``torch_dtype``.

    INT8 and FP8 are detected INDEPENDENTLY and both swaps run, because the int8
    conversion tool emits a MIXED checkpoint on purpose: layers whose per-row
    crest factor makes int8 worse than e4m3 fall back to e4m3, in the same file.
    Each detector AND each swap helper gates on the weight DTYPE as well as the
    shared ``.weight_scale`` suffix -- ``is_int8_state_dict`` /
    ``swap_linears_to_int8`` on ``int8``, ``is_fp8_state_dict`` /
    ``swap_linears_to_fp8`` on ``float8_e4m3fn`` -- so neither can claim the
    other's layers and the order between the two calls below does not matter (a
    layer already replaced is no longer an ``nn.Linear``). The suffix test alone
    would NOT give that: it would let the fp8 swap take int8 layers and ``copy_``
    integer codes into an e4m3 buffer without raising. ``load_fp8_state_dict`` then serves both: it keys on
    "is this floating point?", so an int8 weight is moved to the device with its
    dtype intact exactly as a float8 one is."""
    model = Krea2Transformer2DModel.from_config(config)

    has_int8 = is_int8_state_dict(diffusers_sd)
    has_fp8 = is_fp8_state_dict(diffusers_sd)
    if has_int8 or has_fp8:
        model.to(torch_dtype)
        n_int8 = swap_linears_to_int8(model, diffusers_sd, compute_dtype=torch_dtype) if has_int8 else 0
        n_fp8 = swap_linears_to_fp8(model, diffusers_sd, compute_dtype=torch_dtype) if has_fp8 else 0
        parts = []
        if n_int8:
            parts.append(f"{n_int8} Int8Linear")
        if n_fp8:
            parts.append(f"{n_fp8} Fp8Linear")
        print(f"[Krea2] weight-only quantized: swapped {' + '.join(parts) or 'no'} Linear(s)")
        load_fp8_state_dict(
            model, diffusers_sd, device=torch.device("cpu"), dtype=torch_dtype,
            assign=False, strict=False,
        )
    else:
        model.to(torch_dtype)
        missing, unexpected = model.load_state_dict(diffusers_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"[Krea2] unexpected transformer keys: {unexpected[:8]}")
        # Only genuinely-missing (non-buffer) keys are fatal; RoPE has no params.
        if missing:
            raise RuntimeError(f"[Krea2] missing transformer keys: {missing[:8]}")

    model.eval()
    model.to("cpu")
    return model


def _read_safetensors(path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Read a Krea 2 single file or shard index into (state_dict, metadata)."""
    return read_state_dict(path)


def load_single_file(path: str, torch_dtype: torch.dtype = torch.bfloat16) -> dict:
    """Load a Krea 2 single-file safetensors (any of the supported layouts).

    Accepts a ``<stem>.safetensors`` file or a ``<stem>.safetensors.index.json``
    shard index (routed through the shared reader).

    Returns: {"transformer": Krea2Transformer2DModel(cpu, eval),
              "text_encoder_state_dict": dict|None, "is_distilled": bool,
              "config": dict}.
    """
    raw, metadata = _read_safetensors(path)
    reject_unsupported_quant(path, metadata)

    # Split off an embedded VAE section (``vae.*``) before normalisation, so it does
    # not pollute the transformer load. Absent -> None (loader resolves default VAE).
    vae_sd = {k[len(VAE_PREFIX):]: v for k, v in raw.items() if k.startswith(VAE_PREFIX)} or None

    diffusers_sd, te_sd = normalize_state_dict(raw, metadata)
    config, is_distilled = detect_config_and_variant(metadata, diffusers_sd)

    # Filename hint for the turbo/distilled variant when metadata is silent.
    if not is_distilled and ("turbo" in path.lower() or "distill" in path.lower()):
        is_distilled = True

    model = build_krea2_transformer(diffusers_sd, config, torch_dtype)
    return {
        "transformer": model,
        "text_encoder_state_dict": te_sd,
        "vae_state_dict": vae_sd,
        "is_distilled": is_distilled,
        "config": config,
    }


def save_single_file(
    path: str,
    transformer: Krea2Transformer2DModel,
    is_distilled: bool,
    text_encoder: Optional[torch.nn.Module] = None,
    vae: Optional[torch.nn.Module] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> None:
    """Write a sushiUI Krea 2 single-file (transformer [+ optional Qwen3-VL TE] +
    metadata). Fixed format for Phase B training checkpoints:

      keys:     transformer.<Krea2Transformer2DModel state_dict>
                text_encoder.<Qwen3VLModel state_dict>   (optional)
      metadata: model_type="krea2", krea2_config=<json>, variant (raw|turbo),
                is_distilled, has_text_encoder, format="pt"

    Tied tensors are de-duplicated by storage pointer (safetensors rejects shared
    memory) and re-tied on load. Saves as a single file within ``max_shard_bytes``
    (default 10 GB); above that, diffusers-convention shards plus a
    ``<stem>.safetensors.index.json`` are written via the shared writer (the bf16
    transformer, ~26 GB, shards).
    """
    def _named():
        for k, v in transformer.state_dict().items():
            yield f"{TRANSFORMER_PREFIX}{k}", v
        if text_encoder is not None:
            for k, v in text_encoder.state_dict().items():
                yield f"{TEXT_ENCODER_PREFIX}{k}", v
        if vae is not None:
            for k, v in vae.state_dict().items():
                yield f"{VAE_PREFIX}{k}", v

    state, dropped_tied = dedup_tensors(_named())

    config = dict(getattr(transformer, "config", {}) or {})
    metadata = {
        "model_type": "krea2",
        "variant": "turbo" if is_distilled else "raw",
        "is_distilled": "1" if is_distilled else "0",
        "krea2_config": json.dumps({k: config[k] for k in KREA2_DEFAULT_CONFIG if k in config}),
        "has_text_encoder": "1" if text_encoder is not None else "0",
        "component.vae.embedded": "1" if vae is not None else "0",
        "format": "pt",
    }
    if dropped_tied:
        metadata["tied_weights_dropped"] = json.dumps(dropped_tied)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})
    save_single_file_state(state, metadata, path, max_shard_bytes=max_shard_bytes)
