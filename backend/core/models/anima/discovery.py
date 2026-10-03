"""Header and path discovery for Anima artifacts without importing models."""
import os
from pathlib import Path
from typing import Optional, Dict, List
from safetensors import safe_open

# Common filename patterns used to auto-discover companion components.
QWEN3_TE_PATTERNS = [
    "qwen_3_06b_base.safetensors",
    "qwen_3_06b.safetensors",
    "qwen_3_0.6b.safetensors",
    "qwen3-0.6b.safetensors",
    "qwen3_0.6b.safetensors",
]
QWEN_VAE_PATTERNS = [
    "qwen_image_vae.safetensors",
    "qwen-image-vae.safetensors",
    "qwen_image_vae_fp16.safetensors",
    "qwen_image_vae_bf16.safetensors",
]

# -------- Detection ----------

ANIMA_KEY_SIGNATURES = (
    "blocks.0.self_attn.q_proj.weight",
    "blocks.0.cross_attn.q_proj.weight",
    "blocks.0.adaln_modulation_self_attn",  # prefix match
    "final_layer.linear.weight",
    "x_embedder.proj.1.weight",
)


def _strip_net_prefix(keys):
    """Return keys with the optional 'net.' prefix stripped (used by some
    third-party tooling that re-wraps Anima DiT state dicts)."""
    return [k[len("net."):] if k.startswith("net.") else k for k in keys]


def is_anima_state_dict_keys(keys) -> bool:
    """Check whether the given safetensors keys look like an Anima DiT."""
    stripped = set(_strip_net_prefix(list(keys)))
    must_have = {
        "blocks.0.self_attn.q_proj.weight",
        "blocks.0.cross_attn.q_proj.weight",
        "final_layer.linear.weight",
    }
    if not must_have.issubset(stripped):
        return False
    # AdaLN-LoRA signature
    has_adaln = any(k.startswith("blocks.0.adaln_modulation_self_attn") for k in stripped)
    return has_adaln


def is_anima_safetensors(path: str) -> bool:
    if not path.endswith(".safetensors") or not os.path.isfile(path):
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            keys = list(f.keys())
    except Exception:
        return False

    arch = (metadata or {}).get("modelspec.architecture", "")
    if arch and "anima" in arch.lower():
        return True
    return is_anima_state_dict_keys(keys)


def detect_anima_split_layout(path: str) -> Optional[Dict[str, str]]:
    """If `path` is a directory containing a split-files Anima layout, return
    a dict {dit, text_encoder, vae} of absolute paths. Otherwise None.

    Also accepts:
      - a DiT .safetensors inside <root>/split_files/diffusion_models/ ->
        treat <root> as the layout root.
      - a directory directly containing diffusion_models/ (without split_files prefix).
    """
    if not path:
        return None

    p = Path(path)
    # If user pointed at a DiT file inside split_files, walk up to the root.
    if p.is_file() and p.suffix == ".safetensors":
        parents = [pp for pp in p.parents]
        for parent in parents:
            if (parent / "split_files" / "diffusion_models").is_dir():
                root = parent
                dit = p
                te = _find_first(root / "split_files" / "text_encoders", QWEN3_TE_PATTERNS)
                vae = _find_first(root / "split_files" / "vae", QWEN_VAE_PATTERNS)
                return {"dit": str(dit), "text_encoder": str(te) if te else None,
                        "vae": str(vae) if vae else None, "root": str(root)}
        return None

    if not p.is_dir():
        return None

    # Look for split_files/ inside this dir, or diffusion_models/ directly.
    candidate_roots: List[Path] = []
    if (p / "split_files" / "diffusion_models").is_dir():
        candidate_roots.append(p / "split_files")
    if (p / "diffusion_models").is_dir():
        candidate_roots.append(p)
    if not candidate_roots:
        return None

    base = candidate_roots[0]
    dit_dir = base / "diffusion_models"
    dits = sorted(dit_dir.glob("*.safetensors"))
    if not dits:
        return None
    dit = dits[0]  # pick the first; user can specify a more specific subpath
    te = _find_first(base / "text_encoders", QWEN3_TE_PATTERNS)
    vae = _find_first(base / "vae", QWEN_VAE_PATTERNS)
    return {
        "dit": str(dit),
        "text_encoder": str(te) if te else None,
        "vae": str(vae) if vae else None,
        "root": str(p),
    }


def _find_first(directory: Path, patterns: List[str]) -> Optional[Path]:
    if not directory.is_dir():
        return None
    # Exact-name match first
    for pat in patterns:
        candidate = directory / pat
        if candidate.is_file():
            return candidate
    # Fall back to glob with the first wildcard token (e.g. "qwen_3_*")
    for pat in patterns:
        stem = pat.split(".")[0].split("_")[0]
        for f in directory.glob(f"{stem}*.safetensors"):
            return f
    # Last resort: any .safetensors in the directory
    sf = sorted(directory.glob("*.safetensors"))
    return sf[0] if sf else None
