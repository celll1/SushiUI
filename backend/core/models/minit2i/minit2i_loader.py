"""Component loader for MiniT2I (pixel-space MM-JiT + FLAN-T5).

Supports two layouts:
  1. diffusers directory  (a single variant dir): <model>/transformer/ + <model>/scheduler/
  2. single-file safetensors: bundles the transformer (+ optionally FLAN-T5),
     variant auto-detected (see vendor/single_file.py).

FLAN-T5-Large (frozen text encoder) is loaded from, in order: an explicit path,
a sibling/`flan-t5-large` directory next to the model, or the HF hub.
No VAE (pixel space).
"""

from __future__ import annotations

import os

import torch

from .vendor import MiniT2IMMJiTModel, MiniT2IFlowMatchScheduler
from .vendor.single_file import load_single_file, detect_variant_from_state_dict
from .minit2i_vae import is_latent_vae, load_minit2i_vae, VAE_SCALE_FACTOR


def _looks_like_flan_t5(d: str) -> bool:
    """A directory is a FLAN-T5 checkpoint if it has a config.json + tokenizer."""
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    return any(os.path.isfile(os.path.join(d, t))
               for t in ("spiece.model", "tokenizer.json", "tokenizer_config.json"))


def _resolve_flan_t5(model_path: str, flan_t5_path: str | None) -> str:
    """Resolve the FLAN-T5-Large location (local dir preferred, else hub id).

    Walks up several ancestors of the model and probes common sibling names so a
    variant dir (e.g. <root>/MiniT2I/minit2i-b-16) finds <root>/flan-t5-large two
    levels up. Falls back to the HF hub id if nothing local is found.
    """
    if flan_t5_path and os.path.isdir(flan_t5_path):
        return flan_t5_path
    names = ("flan-t5-large", "flan-t5", "flan_t5_large", "text_encoder")
    base = os.path.dirname(model_path.rstrip("/\\")) if os.path.isfile(model_path) else model_path
    base = base.rstrip("/\\")
    # Probe base and up to 4 ancestor levels for any of the sibling names.
    cur = base
    for _ in range(5):
        for nm in names:
            cand = os.path.join(cur, nm)
            if _looks_like_flan_t5(cand):
                return cand
            # one extra nesting level (e.g. flan-t5-large/flan-t5-large)
            cand2 = os.path.join(cand, nm)
            if _looks_like_flan_t5(cand2):
                return cand2
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return "google/flan-t5-large"  # hub fallback


def _is_minit2i_variant_dir(d: str) -> bool:
    """True if d is a MiniT2I variant diffusers dir (has transformer/config.json marker)."""
    cfg_path = os.path.join(d, "transformer", "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        import json
        with open(cfg_path, "r", encoding="utf-8") as f:
            tcfg = json.load(f)
    except Exception:
        return False
    return (tcfg.get("_class_name") == "MiniT2IMMJiTModel"
            or ("depth_double" in tcfg and "pca_channels" in tcfg))


def find_minit2i_variant_dirs(path: str, max_depth: int = 2) -> list:
    """Find MiniT2I variant dirs at `path` or within `max_depth` levels below it.

    Handles: a variant dir itself, a repo root (<root>/MiniT2I containing
    minit2i-b-16 / minit2i-l-16), and a container (<root> with MiniT2I/ inside).
    Returns absolute paths, de-duplicated, sorted.
    """
    found = set()
    if not os.path.isdir(path):
        return []

    def _walk(d: str, depth: int):
        if _is_minit2i_variant_dir(d):
            found.add(os.path.abspath(d))
            return  # a variant dir has no nested variants
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            return
        for name in entries:
            sub = os.path.join(d, name)
            if os.path.isdir(sub):
                _walk(sub, depth + 1)

    _walk(path, 0)
    return sorted(found)


def resolve_minit2i_model_dir(path: str) -> str:
    """Resolve a user-supplied directory to a single MiniT2I variant dir.

    If `path` is already a variant dir, return it. Otherwise search inside; with
    exactly one variant return it, with several raise a clear error listing them.
    """
    if _is_minit2i_variant_dir(path):
        return path
    variants = find_minit2i_variant_dirs(path)
    if len(variants) == 1:
        print(f"[MiniT2ILoader] Resolved '{path}' -> variant dir '{variants[0]}'")
        return variants[0]
    if len(variants) > 1:
        listing = "\n  ".join(variants)
        raise ValueError(
            f"Multiple MiniT2I variants found under '{path}'. Select a specific "
            f"variant directory (B/16 and L/16 are separate models):\n  {listing}"
        )
    raise ValueError(
        f"No MiniT2I variant found under '{path}'. Point at the variant directory "
        f"that contains 'transformer/' and 'scheduler/' (e.g. .../minit2i-b-16)."
    )


def _load_flan_t5(location: str, torch_dtype: torch.dtype):
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(location)
    text_encoder = T5EncoderModel.from_pretrained(location, torch_dtype=torch_dtype)
    text_encoder.eval()
    return tokenizer, text_encoder


def _detect_variant_name(transformer: MiniT2IMMJiTModel) -> str:
    cfg = transformer.mmjit_config
    if cfg.hidden_size == 1248 or cfg.depth_double == 23:
        return "l16"
    return "b16"


def load_minit2i_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    flan_t5_path: str | None = None,
    text_encoder_dtype: torch.dtype = torch.float32,
    vae_dtype: torch.dtype = torch.float16,
    vae_local_dir: str | None = None,
) -> dict:
    """Load MiniT2I components from a diffusers dir or a single-file safetensors.

    Returns a component dict consumed by PipelineManager.load_model():
        {type:"minit2i", transformer, scheduler, text_encoder, tokenizer, variant,
         vae, vae_type, vae_scale_factor}
    vae is None for pixel-space models (vae_type="none").
    """
    is_single_file = os.path.isfile(model_path) and model_path.endswith(".safetensors")

    if is_single_file:
        print(f"[MiniT2ILoader] Loading single-file: {model_path}")
        bundle = load_single_file(model_path, torch_dtype=torch_dtype)
        transformer = bundle["transformer"]
        variant = bundle["variant"] or _detect_variant_name(transformer)
        scheduler = MiniT2IFlowMatchScheduler()  # defaults (lognorm, n_T 100)

        te_sd = bundle.get("text_encoder_state_dict")
        flan_loc = _resolve_flan_t5(model_path, flan_t5_path)
        if te_sd is not None:
            # FLAN-T5 weights are embedded; build arch from config and load them.
            from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
            cfg = AutoConfig.from_pretrained(flan_loc)
            text_encoder = T5EncoderModel(cfg).to(text_encoder_dtype)
            text_encoder.load_state_dict(te_sd, strict=False)
            text_encoder.eval()
            tokenizer = AutoTokenizer.from_pretrained(flan_loc)
        else:
            tokenizer, text_encoder = _load_flan_t5(flan_loc, text_encoder_dtype)
    else:
        print(f"[MiniT2ILoader] Loading diffusers directory: {model_path}")
        # Accept a variant dir, a repo root (.../MiniT2I) or a container
        # (.../minit2i with MiniT2I/ inside); resolve to one variant dir.
        flan_search_root = model_path
        if os.path.isdir(model_path) and not os.path.isdir(os.path.join(model_path, "transformer")):
            resolved_dir = resolve_minit2i_model_dir(model_path)
            model_path = resolved_dir
        transformer_dir = os.path.join(model_path, "transformer")
        if not os.path.isdir(transformer_dir):
            transformer_dir = model_path  # allow pointing directly at the transformer dir
        transformer = MiniT2IMMJiTModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
        variant = _detect_variant_name(transformer)

        scheduler_dir = os.path.join(model_path, "scheduler")
        if os.path.isdir(scheduler_dir):
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()

        # Resolve FLAN-T5 from the originally-supplied root too (the local
        # flan-t5-large often sits a couple of levels above the variant dir).
        flan_loc = _resolve_flan_t5(model_path, flan_t5_path)
        if flan_loc == "google/flan-t5-large" and flan_search_root != model_path:
            flan_loc = _resolve_flan_t5(flan_search_root, flan_t5_path)
        tokenizer, text_encoder = _load_flan_t5(flan_loc, text_encoder_dtype)

    transformer.eval()
    transformer.to("cpu")
    text_encoder.to("cpu")

    # Latent-space variants (vae_type != "none") also load their VAE. Pixel-space
    # (vae_type="none") keeps vae=None and decodes RGB directly.
    vae = None
    vae_type = getattr(transformer.mmjit_config, "vae_type", "none")
    if is_latent_vae(vae_type):
        vae = load_minit2i_vae(vae_type, torch_dtype=vae_dtype, local_dir=vae_local_dir)
        vae.to("cpu")
    print(f"[MiniT2ILoader] Loaded MiniT2I variant={variant} vae_type={vae_type} (FLAN-T5 from {flan_loc})")

    return {
        "type": "minit2i",
        "transformer": transformer,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "variant": variant,
        "vae": vae,
        "vae_type": vae_type,
        "vae_scale_factor": VAE_SCALE_FACTOR,
    }
