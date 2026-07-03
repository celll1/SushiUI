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

from core.models.common.single_file_format import (
    DEFAULT_MAX_SHARD_BYTES,
    VAE_PREFIX,
    dedup_tensors,
    read_state_dict,
    save_single_file_state,
)

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

    # I/O config from the patch embed conv: proj1.weight is [pca, in_channels, patch, patch].
    # vae_type maps from in_channels (3=pixel, 4=SDXL VAE, 16=FLUX.1 VAE).
    proj1_key = key("model.net.img_embedder.proj1.weight")
    in_channels, patch_size, vae_type = 3, 16, "none"
    if proj1_key in sd:
        w = sd[proj1_key]
        in_channels = int(w.shape[1])
        patch_size = int(w.shape[2])
        vae_type = {3: "none", 4: "sdxl", 16: "flux1"}.get(in_channels, "none")
    io_cfg = dict(in_channels=in_channels, patch_size=patch_size, vae_type=vae_type,
                  noise_scale=2.0 if vae_type == "none" else 1.0)

    for name, cfg in KNOWN_VARIANTS.items():
        if cfg["hidden_size"] == hidden_size and cfg["depth_double"] == depth_double:
            return name, {**dict(cfg), **io_cfg}
    raise ValueError(
        f"[MiniT2I single-file] unknown variant (hidden_size={hidden_size}, depth_double={depth_double}); "
        f"known: {list(KNOWN_VARIANTS)}"
    )


def load_single_file(path: str, torch_dtype: torch.dtype = torch.bfloat16):
    """Load a MiniT2I single-file.

    Returns dict: {"transformer": MiniT2IMMJiTModel(cpu, eval),
                   "text_encoder_state_dict": dict|None, "variant": str}.
    """
    raw, metadata = read_state_dict(path)

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
    vae_sd = _transformer_subdict(raw, VAE_PREFIX) or None
    return {"transformer": model, "text_encoder_state_dict": te_sd,
            "vae_state_dict": vae_sd, "variant": variant}


def save_single_file(
    path: str,
    transformer: MiniT2IMMJiTModel,
    variant: str,
    text_encoder: Optional[torch.nn.Module] = None,
    vae: Optional[torch.nn.Module] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> None:
    """Write a MiniT2I single-file (transformer [+ optional FLAN-T5] + metadata).

    Tied tensors (e.g. FLAN-T5 shares `shared.weight` with `encoder.embed_tokens.weight`)
    are de-duplicated by storage pointer — safetensors rejects shared memory, and load
    re-ties them. The dropped key is recorded in metadata for transparency.

    Saves as a single file when the total tensor byte size is within
    ``max_shard_bytes`` (default 10 GB); otherwise writes diffusers-convention
    shards plus a ``<stem>.safetensors.index.json`` via the shared writer.
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

    cfg = transformer.mmjit_config
    # Persist the variant-delta keys + the I/O config (in_channels/patch_size/vae_type/
    # noise_scale) so latent variants reload with the right channels/patch and VAE.
    cfg_keys = list(KNOWN_VARIANTS[variant].keys()) + ["in_channels", "patch_size", "vae_type", "noise_scale"]
    metadata = {
        "model_type": "minit2i",
        "variant": str(variant),
        "vae_type": str(getattr(cfg, "vae_type", "none")),
        "mmjit_config": json.dumps({k: getattr(cfg, k) for k in cfg_keys}),
        "has_text_encoder": "1" if text_encoder is not None else "0",
        "component.vae.embedded": "1" if vae is not None else "0",
        "format": "pt",
    }
    if dropped_tied:
        metadata["tied_weights_dropped"] = json.dumps(dropped_tied)
    if extra_metadata:
        metadata.update({k: str(v) for k, v in extra_metadata.items()})
    save_single_file_state(state, metadata, path, max_shard_bytes=max_shard_bytes)
