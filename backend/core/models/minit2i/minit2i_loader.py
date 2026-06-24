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
    scratch_init_from: str | None = None,
    scratch_inherit_final_layer: bool = False,
) -> dict:
    """Load MiniT2I components from a diffusers dir or a single-file safetensors.

    Returns a component dict consumed by PipelineManager.load_model():
        {type:"minit2i", transformer, scheduler, text_encoder, tokenizer, variant,
         vae, vae_type, vae_scale_factor}
    vae is None for pixel-space models (vae_type="none").
    """
    if is_scratch_spec(model_path):
        # From-scratch Full-FT: build a random-initialized model in memory (no disk
        # init model). VAE/FLAN-T5 are resolved by variant/vae_type as usual.
        scratch_variant, scratch_vae = parse_scratch_spec(model_path)
        print(f"[MiniT2ILoader] From-scratch spec: variant={scratch_variant} vae_type={scratch_vae}"
              + (f" (inherit weights from {scratch_init_from})" if scratch_init_from else ""))
        transformer = build_scratch_minit2i(scratch_variant, scratch_vae, dtype=torch_dtype,
                                            init_from=scratch_init_from or None,
                                            inherit_final_layer=scratch_inherit_final_layer)
        variant = scratch_variant
        scheduler = MiniT2IFlowMatchScheduler()
        # No path info in the sentinel: probe the local minit2i model tree for FLAN-T5
        # (sibling of the VAE local dir), then fall back to the hub.
        flan_loc = _resolve_flan_t5(model_path, flan_t5_path)
        if flan_loc == "google/flan-t5-large":
            vae_root = os.environ.get("MINIT2I_VAE_DIR") or r"M:\model\minit2i\vae"
            flan_loc = _resolve_flan_t5(os.path.dirname(vae_root.rstrip("/\\")), flan_t5_path)
        tokenizer, text_encoder = _load_flan_t5(flan_loc, text_encoder_dtype)

        transformer.eval()
        transformer.to("cpu")
        text_encoder.to("cpu")
        vae = None
        vae_type = getattr(transformer.mmjit_config, "vae_type", "none")
        if is_latent_vae(vae_type):
            vae = load_minit2i_vae(vae_type, torch_dtype=vae_dtype, local_dir=vae_local_dir)
            vae.to("cpu")
        print(f"[MiniT2ILoader] Built scratch MiniT2I variant={variant} vae_type={vae_type} "
              f"(FLAN-T5 from {flan_loc})")
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


# Sentinel "base model" for from-scratch training without writing an init model to
# disk: "scratch:minit2i:<variant>:<vae_type>" (e.g. scratch:minit2i:b16:sdxl).
SCRATCH_PREFIX = "scratch:minit2i:"


def is_scratch_spec(model_path: str) -> bool:
    return isinstance(model_path, str) and model_path.startswith(SCRATCH_PREFIX)


def parse_scratch_spec(model_path: str):
    """'scratch:minit2i:b16:sdxl' -> ('b16', 'sdxl'). Defaults vae_type to 'none'."""
    rest = model_path[len(SCRATCH_PREFIX):]
    parts = rest.split(":")
    variant = parts[0] if parts and parts[0] else "b16"
    vae_type = parts[1] if len(parts) > 1 and parts[1] else "none"
    return variant, vae_type


def _load_source_minit2i_state_dict(path: str) -> dict:
    """Load the transformer state_dict of an existing MiniT2I model (single-file or
    diffusers dir) for weight inheritance into a from-scratch build. Keys are the
    model's canonical state_dict keys (model.net.*)."""
    if os.path.isfile(path) and path.endswith(".safetensors"):
        from .vendor.single_file import load_single_file
        return load_single_file(path, torch_dtype=torch.float32)["transformer"].state_dict()
    # diffusers dir: resolve to the transformer dir and load via from_pretrained
    src = path
    if os.path.isdir(src) and not os.path.isdir(os.path.join(src, "transformer")):
        src = resolve_minit2i_model_dir(src)
    tdir = os.path.join(src, "transformer")
    if not os.path.isdir(tdir):
        tdir = src
    return MiniT2IMMJiTModel.from_pretrained(tdir, torch_dtype=torch.float32).state_dict()


def _inherit_minit2i_weights(model, source_sd: dict, inherit_final_layer: bool = False) -> None:
    """In-place: copy compatible weights from a source MiniT2I state_dict into the
    (random-initialized) target model. Same variant required for the body to match.

    - final_layer.linear (output projection): by default NEVER inherited — kept at
      the target's fresh init even when the source shape matches. It carries the
      trained output mapping (the monochrome mean-regression collapse); a warm start
      usually transfers the body but relearns this head from scratch. Set
      inherit_final_layer=True to also copy it (full copy when the shape matches;
      left at fresh init when shapes differ — patch/channel changes). norm_final
      always inherits regardless of this flag.
    - name+shape match (body, proj2, embedders, norms): full copy.
    - img_embedder.proj1 [pca,in,k,k]: when patch (kernel k) is unchanged but channel
      count differs, copy the overlapping channels and keep the rest random (the
      "carry 3ch, init the new ch" case). When patch differs (pixel<->latent), shapes
      are incompatible → left random.
    - everything else with no match: left random.
    """
    # Determine source & target patch_size so in/out-layer channel surgery is only
    # attempted when the patch (token geometry) is unchanged. src patch = proj1 conv
    # kernel; tgt patch from the model config.
    tgt_patch = int(model.mmjit_config.patch_size)
    src_proj1 = source_sd.get("model.net.img_embedder.proj1.weight")
    src_patch = int(src_proj1.shape[2]) if src_proj1 is not None else -1

    tgt_sd = model.state_dict()
    new_sd = {}
    full, partial, init = [], [], []
    for name, tparam in tgt_sd.items():
        # Output projection: by default kept at fresh init so the head relearns from
        # scratch on a warm start (see docstring). When inherit_final_layer is set,
        # fall through and treat it like any other tensor (full copy if shapes match,
        # else left at fresh init since _channel_partial_copy does not handle it).
        if "final_layer.linear" in name and not inherit_final_layer:
            new_sd[name] = tparam; init.append(name); continue
        sparam = source_sd.get(name)
        if sparam is None:
            new_sd[name] = tparam; init.append(name); continue
        if sparam.shape == tparam.shape:
            new_sd[name] = sparam.to(dtype=tparam.dtype); full.append(name); continue
        merged = _channel_partial_copy(name, sparam, tparam, src_patch, tgt_patch)
        if merged is not None:
            new_sd[name] = merged.to(dtype=tparam.dtype); partial.append(name)
        else:
            new_sd[name] = tparam; init.append(name)
    model.load_state_dict(new_sd, strict=True)
    print(f"[MiniT2ILoader] Weight inheritance: {len(full)} tensors copied, "
          f"{len(partial)} channel-partial {partial}, {len(init)} re-initialized "
          f"(src_patch={src_patch}, tgt_patch={tgt_patch})")


def _channel_partial_copy(name: str, src: "torch.Tensor", tgt: "torch.Tensor",
                          src_patch: int, tgt_patch: int):
    """Return a tensor shaped like tgt with src's overlapping channels copied in,
    for the input projection (img_embedder.proj1) when ONLY the channel count
    differs. Requires src_patch == tgt_patch (same token geometry); otherwise the
    layouts are incompatible (e.g. pixel patch16 <-> latent patch2) and None is
    returned.

    NOTE: the output projection (final_layer.linear) is handled upstream in
    _inherit_minit2i_weights — it is always left at fresh init, never partial-copied.
    """
    if src_patch != tgt_patch or src_patch <= 0:
        return None

    # proj1: Conv2d weight [pca, in_ch, k, k] — differ only in in_ch (dim 1).
    if name.endswith("img_embedder.proj1.weight"):
        if src.shape[0] != tgt.shape[0] or src.shape[2:] != tgt.shape[2:]:
            return None
        out = tgt.clone()
        n = min(src.shape[1], tgt.shape[1])
        out[:, :n] = src[:, :n].to(out.dtype)
        return out

    return None


def build_scratch_minit2i(variant: str, vae_type: str, dtype: torch.dtype = torch.bfloat16,
                          init_from: str | None = None, inherit_final_layer: bool = False):
    """Build a random-initialized MiniT2IMMJiTModel in memory (no disk write).

    variant in {b16, l16}; vae_type in {none, sdxl, flux1}. Pixel: 3ch/patch16/noise2;
    latent: VAE channels/patch2/noise1.

    init_from: optional path to an existing MiniT2I model (vanilla pixel or a local
    model) whose compatible weights are inherited into this build instead of random
    init (same variant required for the body). See _inherit_minit2i_weights.
    """
    from .vendor.single_file import KNOWN_VARIANTS
    from .minit2i_vae import vae_latent_channels, VAE_REGISTRY

    if variant not in KNOWN_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}' (expected {list(KNOWN_VARIANTS)})")
    if vae_type != "none" and vae_type not in VAE_REGISTRY:
        raise ValueError(f"Unknown vae_type '{vae_type}' (expected none/{list(VAE_REGISTRY)})")

    base = dict(KNOWN_VARIANTS[variant])
    if vae_type == "none":
        in_ch, patch, noise = 3, 16, 2.0
    else:
        in_ch, patch, noise = vae_latent_channels(vae_type), 2, 1.0

    print(f"[MiniT2ILoader] Building scratch MiniT2I variant={variant} vae_type={vae_type} "
          f"(in_channels={in_ch}, patch_size={patch})")
    model = MiniT2IMMJiTModel(
        image_size=512, patch_size=patch, in_channels=in_ch, txt_input_size=1024,
        hidden_size=base["hidden_size"], txt_hidden_size=base["txt_hidden_size"],
        cond_vec_size=base["cond_vec_size"], depth_double=base["depth_double"],
        txt_preamble_depth=2, num_heads=base["num_heads"], head_dim=base["head_dim"],
        mlp_ratio=base["mlp_ratio"], pca_channels=128, prompt_length=256,
        vae_type=vae_type, noise_scale=noise,
    ).to(dtype)

    if init_from:
        try:
            print(f"[MiniT2ILoader] Inheriting weights from: {init_from}")
            source_sd = _load_source_minit2i_state_dict(init_from)
            _inherit_minit2i_weights(model, source_sd, inherit_final_layer=inherit_final_layer)
        except Exception as e:
            print(f"[MiniT2ILoader] WARNING: weight inheritance failed ({e}); "
                  f"continuing from random init")
    return model


def create_scratch_minit2i(variant: str, vae_type: str, out_dir: str,
                           dtype: torch.dtype = torch.bfloat16) -> str:
    """Persist a random-initialized MiniT2I diffusers dir (transformer + scheduler).

    Optional helper; from-scratch training normally uses the in-memory build path
    (SCRATCH_PREFIX base model) and does not need a saved init model.
    """
    model = build_scratch_minit2i(variant, vae_type, dtype)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(os.path.join(out_dir, "transformer"))
    MiniT2IFlowMatchScheduler().save_pretrained(os.path.join(out_dir, "scheduler"))
    return out_dir
