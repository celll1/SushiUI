"""Single-file format for MiniT2I (FLAN-T5 + one variant in one .safetensors).

Custom format (no diffusers dependency beyond the vendored model):
  keys:  transformer.<MiniT2IMMJiTModel state_dict>   (i.e. transformer.model.net.*)
         text_encoder.<T5EncoderModel state_dict>      (optional; else loaded from hub)
  metadata (safetensors __metadata__, all str):
         model_type="minit2i", variant="b16"|"l16", mmjit_config=<json>  (optional)

`from_single_file` prefers the metadata config; otherwise it auto-detects the
variant from transformer weight shapes (mirroring diffusers'
`infer_diffusers_model_type`, which branches on key presence + tensor shape).
"""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .transformer import MiniT2IMMJiTModel


# Known published variants (config deltas; other fields use MMJiTConfig defaults).
KNOWN_VARIANTS: Dict[str, dict] = {
    "b16": dict(hidden_size=768, txt_hidden_size=768, cond_vec_size=768,
                depth_double=17, num_heads=12, head_dim=64,
                mlp_ratio=2.6666666666666665),
    "l16": dict(hidden_size=1248, txt_hidden_size=1248, cond_vec_size=1248,
                depth_double=23, num_heads=24, head_dim=52,
                mlp_ratio=2.7051282051282053),
}
TRANSFORMER_PREFIX = "transformer."
TEXT_ENCODER_PREFIX = "text_encoder."


def _transformer_subdict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def detect_variant_from_state_dict(sd: Dict[str, torch.Tensor], prefix: str = "") -> Tuple[str, dict]:
    """Auto-detect (variant_name, config_dict) from a transformer state dict.

    Uses the same key-presence + shape idiom as diffusers.infer_diffusers_model_type:
      - hidden_size  = model.net.double_blocks.0.img_qkv.weight.shape[1]
      - depth_double = number of distinct double_blocks.N
    Matches against KNOWN_VARIANTS; raises if no known variant fits.
    """
    import re

    def key(name):
        return f"{prefix}{name}"

    qkv_key = key("model.net.double_blocks.0.img_qkv.weight")
    if qkv_key not in sd:
        raise ValueError(f"[MiniT2I single-file] missing key {qkv_key!r}; not a MiniT2I transformer")
    hidden_size = int(sd[qkv_key].shape[1])
    depths = {int(m.group(1)) for k in sd if (m := re.search(rf"{re.escape(prefix)}model\.net\.double_blocks\.(\d+)\.", k))}
    depth_double = (max(depths) + 1) if depths else 0

    for name, cfg in KNOWN_VARIANTS.items():
        if cfg["hidden_size"] == hidden_size and cfg["depth_double"] == depth_double:
            return name, dict(cfg)
    raise ValueError(
        f"[MiniT2I single-file] unknown variant (hidden_size={hidden_size}, depth_double={depth_double}); "
        f"known: {list(KNOWN_VARIANTS)}"
    )


def load_single_file(path: str, torch_dtype: torch.dtype = torch.bfloat16):
    """Load a MiniT2I single-file.

    Returns dict: {"transformer": MiniT2IMMJiTModel(cpu, eval),
                   "text_encoder_state_dict": dict|None, "variant": str}.
    """
    raw: Dict[str, torch.Tensor] = {}
    metadata: dict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        metadata = dict(md)
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    tf_sd = _transformer_subdict(raw, TRANSFORMER_PREFIX)
    if not tf_sd:
        raise ValueError(f"[MiniT2I single-file] no '{TRANSFORMER_PREFIX}*' keys in {path}")

    # Prefer metadata config; else detect from shapes.
    variant = metadata.get("variant")
    config = None
    if metadata.get("mmjit_config"):
        try:
            config = json.loads(metadata["mmjit_config"])
        except Exception:
            config = None
    if config is None:
        det_variant, config = detect_variant_from_state_dict(tf_sd, prefix="")
        variant = variant or det_variant

    model = MiniT2IMMJiTModel(**config)
    model.to(torch_dtype)
    missing, unexpected = model.load_state_dict(tf_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"[MiniT2I single-file] unexpected transformer keys: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"[MiniT2I single-file] missing transformer keys: {missing[:8]}")
    model.eval()
    model.to("cpu")

    te_sd = _transformer_subdict(raw, TEXT_ENCODER_PREFIX) or None
    return {"transformer": model, "text_encoder_state_dict": te_sd, "variant": variant}


def save_single_file(
    path: str,
    transformer: MiniT2IMMJiTModel,
    variant: str,
    text_encoder: Optional[torch.nn.Module] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Write a MiniT2I single-file (transformer [+ optional FLAN-T5] + metadata).

    Tied tensors (e.g. FLAN-T5 shares `shared.weight` with `encoder.embed_tokens.weight`)
    are de-duplicated by storage pointer — safetensors rejects shared memory, and load
    re-ties them. The dropped key is recorded in metadata for transparency.
    """
    state: Dict[str, torch.Tensor] = {}
    seen_ptrs: Dict[int, str] = {}
    dropped_tied: list = []

    def _add(key: str, v: torch.Tensor):
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            dropped_tied.append(key)  # tied to seen_ptrs[ptr]; re-tied on load
            return
        seen_ptrs[ptr] = key
        state[key] = v.detach().to("cpu").contiguous()

    for k, v in transformer.state_dict().items():
        _add(f"{TRANSFORMER_PREFIX}{k}", v)
    if text_encoder is not None:
        for k, v in text_encoder.state_dict().items():
            _add(f"{TEXT_ENCODER_PREFIX}{k}", v)

    cfg = transformer.mmjit_config
    metadata = {
        "model_type": "minit2i",
        "variant": str(variant),
        "mmjit_config": json.dumps({k: getattr(cfg, k) for k in KNOWN_VARIANTS[variant].keys()}),
        "has_text_encoder": "1" if text_encoder is not None else "0",
        "format": "pt",
    }
    if dropped_tied:
        metadata["tied_weights_dropped"] = json.dumps(dropped_tied)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})
    save_file(state, path, metadata=metadata)
