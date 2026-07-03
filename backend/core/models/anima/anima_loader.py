"""Loader for Anima models (DiT + Qwen3 text encoder + Qwen-Image VAE).

Supports two distribution formats:
  1. Split-files layout (HuggingFace official):
       <root>/split_files/diffusion_models/*.safetensors  -> DiT
       <root>/split_files/text_encoders/qwen_3_*.safetensors -> Qwen3 weights
       <root>/split_files/vae/qwen_image_vae*.safetensors -> Qwen-Image VAE
  2. Single DiT safetensors (derivatives like OOO_Anima):
       text encoder and VAE must be auto-discovered or specified.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors import safe_open

from .anima_models import Anima, ANIMA_DIT_CONFIG


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

# Known-good Qwen-Image VAE config (matches the official Qwen-Image repo).
# dim_mult MUST be a list, not a tuple — the encoder's __init__ does `[1] + dim_mult`.
QWEN_IMAGE_VAE_CONFIG = dict(
    base_dim=96,
    z_dim=16,
    dim_mult=[1, 2, 4, 4],
    num_res_blocks=2,
    attn_scales=[],
    temperal_downsample=[False, True, True],
    dropout=0.0,
    input_channels=3,
)


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


def discover_anima_components(dit_path: str, models_root: Optional[str] = None,
                              text_encoder_override: Optional[str] = None,
                              vae_override: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Find Qwen3 text encoder + Qwen-Image VAE for a given DiT path.

    Search order:
      1. explicit overrides
      2. split-files layout next to the DiT
      3. <models_root>/text_encoders/, <models_root>/vae/
      4. sibling directories of the DiT file
    """
    out = {"dit": dit_path, "text_encoder": text_encoder_override, "vae": vae_override}

    split = detect_anima_split_layout(dit_path)
    if split:
        if not out["text_encoder"]:
            out["text_encoder"] = split.get("text_encoder")
        if not out["vae"]:
            out["vae"] = split.get("vae")

    search_dirs: List[Path] = []
    if models_root:
        search_dirs.append(Path(models_root) / "text_encoders")
        search_dirs.append(Path(models_root) / "vae")
        search_dirs.append(Path(models_root) / "anima_components")
    # Sibling of the dit file as a last resort
    dit_p = Path(dit_path)
    if dit_p.is_file():
        search_dirs.append(dit_p.parent)

    if not out["text_encoder"]:
        for d in search_dirs:
            f = _find_first(d, QWEN3_TE_PATTERNS)
            if f:
                out["text_encoder"] = str(f)
                break
    if not out["vae"]:
        for d in search_dirs:
            f = _find_first(d, QWEN_VAE_PATTERNS)
            if f:
                out["vae"] = str(f)
                break

    return out


# -------- Loading ----------

def load_anima_dit(dit_path: str, device: str = "cpu",
                   dtype: torch.dtype = torch.bfloat16,
                   state_dict: Optional[dict] = None) -> Anima:
    """Instantiate the Anima DiT and load weights from a single safetensors file.

    Handles the optional `net.` prefix that some third-party Anima DiT
    checkpoints carry. An embedded ``first_stage_model.*`` VAE section (bundle_vae)
    is ignored here (loaded strict=False); the caller extracts it separately.

    ``state_dict`` may be supplied to reuse an already-read state dict (avoids a
    second file read when the caller has already split off the embedded VAE).
    """
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = Anima(**ANIMA_DIT_CONFIG)
        model.to(dtype)

    if state_dict is not None:
        sd = state_dict
    else:
        from core.models.common.single_file_format import read_state_dict
        sd, _md = read_state_dict(dit_path)
    # Strip net. prefix if present
    if any(k.startswith("net.") for k in sd.keys()):
        sd = {(k[len("net."):] if k.startswith("net.") else k): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    # Filter out buffers that are re-initialized in __init__ (not saved in checkpoint).
    expected_missing_substrings = ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq")
    real_missing = [k for k in missing if not any(s in k for s in expected_missing_substrings)]
    if real_missing:
        # Don't hard-fail; some derivative checkpoints may legitimately omit keys.
        # Surface a warning so the user can investigate.
        print(f"[AnimaLoader] WARNING: {len(real_missing)} missing key(s); first 5: {real_missing[:5]}")
    if unexpected:
        print(f"[AnimaLoader] WARNING: {len(unexpected)} unexpected key(s); first 5: {unexpected[:5]}")

    # Move to device
    if device != "cpu":
        model = model.to(device)
    model = model.eval().requires_grad_(False)
    return model


def load_qwen3_text_encoder(qwen3_path: str,
                             config_dir: Optional[str] = None,
                             device: str = "cpu",
                             dtype: torch.dtype = torch.bfloat16):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Either a HuggingFace-style directory or a single .safetensors file.
        config_dir: Directory containing Qwen3 config.json/tokenizer.json (required for
                    safetensors-only loading). If None, uses the bundled
                    backend/core/models/anima/configs/qwen3_06b/.
    Returns:
        (model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "qwen3_06b")

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True,
        ).model
    else:
        if not os.path.isdir(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected config.json, tokenizer.json, etc. "
                "Download from the Qwen/Qwen3-0.6B-Base HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        if qwen3_path.endswith(".safetensors"):
            state_dict = safetensors_load_file(qwen3_path, device="cpu")
        else:
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model."):]] = v
            else:
                new_sd[k] = v
        info = model.load_state_dict(new_sd, strict=False)
        print(f"[AnimaLoader] Qwen3 load: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)
    return model, tokenizer


def load_t5_tokenizer(config_dir: Optional[str] = None):
    """Load the T5 tokenizer used by the LLM Adapter to produce target_input_ids."""
    from transformers import T5TokenizerFast

    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(__file__), "configs", "t5_old")

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(
            f"T5 tokenizer config directory not found at {config_dir}. "
            "Expected spiece.model and tokenizer.json. "
            "Download from the google/t5-v1_1-xxl HuggingFace repository."
        )

    spiece_path = os.path.join(config_dir, "spiece.model")
    tokenizer_path = os.path.join(config_dir, "tokenizer.json")
    tokenizer = T5TokenizerFast(
        vocab_file=spiece_path if os.path.isfile(spiece_path) else None,
        tokenizer_file=tokenizer_path if os.path.isfile(tokenizer_path) else None,
    )
    # Sanity: the constructor can succeed on a corrupted spiece.model file
    # and then explode much later from encode_prompt's tokenize_for_anima.
    # Run a trivial encode round-trip so the error surfaces here, where
    # the message points at the actual file rather than the generation path.
    try:
        _ = tokenizer.encode("validation", add_special_tokens=False)
    except Exception as e:
        raise RuntimeError(
            f"T5 tokenizer at {config_dir} loaded but encode() failed: {e}. "
            "Check spiece.model and tokenizer.json against the "
            "google/t5-v1_1-xxl repository."
        ) from e
    return tokenizer


def load_qwen_image_vae(vae_path: str, device: str = "cpu",
                        dtype: torch.dtype = torch.bfloat16):
    """Load the Qwen-Image VAE (16ch latents, 8x spatial downscale).

    Builds the diffusers AutoencoderKLQwenImage with the known Qwen-Image config
    (passing dim_mult as a list to sidestep a tuple/list bug in the diffusers
    encoder constructor) and loads weights from a safetensors file directly.

    We don't use from_single_file because the Qwen-Image VAE single-files ship
    without diffusers config metadata, so SingleFileMixin can't pick the right
    class automatically; direct construction with the known config is simpler
    and more robust across diffusers versions.
    """
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage(**QWEN_IMAGE_VAE_CONFIG)

    sd = safetensors_load_file(vae_path, device="cpu") if vae_path.endswith(".safetensors") \
         else torch.load(vae_path, map_location="cpu", weights_only=True)

    # The Qwen-Image VAE safetensors ships in the native Wan-VAE key layout
    # (encoder.middle.0.residual.*, encoder.downsamples.*, conv1/conv2 at root,
    # etc.). Convert to the diffusers AutoencoderKLQwenImage layout
    # (encoder.mid_block.resnets.*, encoder.down_blocks.*, quant_conv, ...)
    # using diffusers' own Wan-VAE converter — Qwen-Image VAE is derived from
    # Wan VAE 2.1 and uses the same key structure.
    native_signature_keys = ("decoder.middle.0.residual.0.gamma",
                              "encoder.middle.0.residual.0.gamma",
                              "conv1.weight")
    if any(k in sd for k in native_signature_keys):
        try:
            from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
            sd = convert_wan_vae_to_diffusers(sd)
            print(f"[AnimaLoader] Converted native Wan/Qwen-Image VAE keys to diffusers layout "
                  f"({len(sd)} keys after conversion).")
        except Exception as e:
            print(f"[AnimaLoader] WARNING: convert_wan_vae_to_diffusers failed: {e}")

    info = vae.load_state_dict(sd, strict=False)
    print(f"[AnimaLoader] Qwen-Image VAE load: missing={len(info.missing_keys)}, "
          f"unexpected={len(info.unexpected_keys)}")
    if info.unexpected_keys:
        print(f"[AnimaLoader]   unexpected (first 5): {list(info.unexpected_keys)[:5]}")
    if info.missing_keys:
        print(f"[AnimaLoader]   missing    (first 5): {list(info.missing_keys)[:5]}")

    vae = vae.to(dtype).to(device).eval().requires_grad_(False)
    return vae


def build_qwen_image_vae_from_embedded(vae_state_dict, device: str = "cpu",
                                       dtype: torch.dtype = torch.bfloat16):
    """Build the Qwen-Image VAE from the known config and reattach embedded (trained)
    weights — no companion file / download. Zero-match raises. The embedded weights
    are already in diffusers AutoencoderKLQwenImage layout (saved from trainer.vae)."""
    from diffusers import AutoencoderKLQwenImage
    from core.models.common.single_file_format import reattach_embedded_weights

    vae = AutoencoderKLQwenImage(**QWEN_IMAGE_VAE_CONFIG)
    reattach_embedded_weights(vae, vae_state_dict, "VAE")
    vae = vae.to(dtype).to(device).eval().requires_grad_(False)
    return vae


def resolve_qwen_image_vae_store_dir() -> Optional[str]:
    """Resolve a shared-store Qwen-Image VAE directory (downloads once), or None."""
    try:
        from core.models.common.vae_store import resolve_vae_dir
        return resolve_vae_dir("qwen_image")
    except Exception as e:
        print(f"[AnimaLoader] Qwen-Image VAE store resolution failed: {e}")
        return None


def load_anima_components(
    dit_path: str,
    text_encoder_path: Optional[str] = None,
    vae_path: Optional[str] = None,
    models_root: Optional[str] = None,
    device: str = "cpu",
    dit_dtype: torch.dtype = torch.bfloat16,
    te_dtype: torch.dtype = torch.bfloat16,
    vae_dtype: torch.dtype = torch.bfloat16,
    qwen3_config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level entry point: discover companion files (if needed), load all components,
    return a dict ready for the pipeline manager.
    """
    discovered = discover_anima_components(
        dit_path, models_root=models_root,
        text_encoder_override=text_encoder_path,
        vae_override=vae_path,
    )

    # Read the DiT single-file once and split off any embedded VAE section
    # (bundle_vae saves under ``first_stage_model.*``). Absent -> companion/store VAE.
    from core.models.common.single_file_format import read_state_dict
    raw_dit_sd, _dit_md = read_state_dict(discovered["dit"])
    embedded_vae_sd = {
        k[len("first_stage_model."):]: v
        for k, v in raw_dit_sd.items() if k.startswith("first_stage_model.")
    } or None
    dit_only_sd = {k: v for k, v in raw_dit_sd.items() if not k.startswith("first_stage_model.")}

    # TE is always a companion; VAE may be embedded OR (new) resolved from the store.
    if not discovered.get("text_encoder"):
        raise FileNotFoundError(
            "Anima requires a companion Qwen3 text encoder (not embedded in the DiT "
            f"save) but could not locate one.\n"
            f"  DiT: {dit_path}\n"
            "  - Qwen3 text encoder: " + ", ".join(QWEN3_TE_PATTERNS) + "\n"
            "Search order (first hit wins):\n"
            f"  1. explicit overrides (text_encoder_path={text_encoder_path})\n"
            "  2. split_files/ layout next to the DiT (split_files/text_encoders/)\n"
            f"  3. models_root subdirs: {models_root}/text_encoders/, {models_root}/anima_components/\n"
            "  4. the DiT file's sibling directory\n"
        )

    print(f"[AnimaLoader] DiT          : {discovered['dit']}")
    print(f"[AnimaLoader] Text encoder : {discovered['text_encoder']}")
    print(f"[AnimaLoader] VAE          : "
          + ("embedded (bundle_vae)" if embedded_vae_sd else str(discovered['vae'])))

    dit = load_anima_dit(discovered["dit"], device="cpu", dtype=dit_dtype,
                         state_dict=dit_only_sd)
    text_encoder, qwen3_tokenizer = load_qwen3_text_encoder(
        discovered["text_encoder"], config_dir=qwen3_config_dir,
        device="cpu", dtype=te_dtype,
    )
    t5_tokenizer = load_t5_tokenizer()

    vae_source = None
    vae_path = None
    if embedded_vae_sd is not None:
        vae = build_qwen_image_vae_from_embedded(embedded_vae_sd, device="cpu", dtype=vae_dtype)
        vae_source = "embedded (checkpoint)"
    elif discovered.get("vae"):
        vae = load_qwen_image_vae(discovered["vae"], device="cpu", dtype=vae_dtype)
        vae_source = str(discovered["vae"])
        vae_path = str(discovered["vae"]) if os.path.isfile(str(discovered["vae"])) else None
    else:
        # New: shared-store hub fallback, AFTER the existing local search order.
        store_dir = resolve_qwen_image_vae_store_dir()
        if store_dir and os.path.isdir(store_dir):
            from diffusers import AutoencoderKLQwenImage
            print(f"[AnimaLoader] Loading Qwen-Image VAE from shared store: {store_dir}")
            vae = AutoencoderKLQwenImage.from_pretrained(
                store_dir, torch_dtype=vae_dtype, low_cpu_mem_usage=True
            ).to("cpu").eval().requires_grad_(False)
            vae_source = str(store_dir)
            vae_path = str(store_dir) if os.path.isdir(str(store_dir)) else None
        else:
            raise FileNotFoundError(
                "Anima could not locate a Qwen-Image VAE. Not embedded in the DiT save, "
                "no companion file found (see filenames: " + ", ".join(QWEN_VAE_PATTERNS) + "), "
                "and the shared VAE store (<models_dir>/vae/qwen_image, "
                "Qwen/Qwen-Image subfolder vae) could not be resolved/downloaded.\n"
                f"  DiT: {dit_path}\n"
                "Place a Qwen-Image VAE under split_files/vae/, models_root/vae/, or the DiT's "
                "sibling directory, or ensure network access for the store download."
            )

    from .anima_scheduler import AnimaFlowMatchScheduler
    scheduler = AnimaFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)

    return {
        "type": "anima",
        "transformer": dit,
        "text_encoder": text_encoder,
        "tokenizer": qwen3_tokenizer,
        "t5_tokenizer": t5_tokenizer,
        "vae": vae,
        "vae_source": vae_source,
        "vae_path": vae_path,
        "scheduler": scheduler,
        "vae_scale_factor": 8,
        "latent_channels": 16,
        "paths": discovered,
    }
