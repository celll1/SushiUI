"""Component loader for ACE-Step 1.5 (2B DiT + Oobleck VAE + Qwen3-Embedding
text encoder).

Phase 0+1: make the model INSTANTIATE and its weights LOAD into SushiUI's
component-dict convention (mirrors Anima / Krea2 / LTX-2.3). No sampler /
generation pipeline yet — that is Phase 2.

Distribution format (ComfyUI-style flat tree, confirmed locally at
``M:\\model\\ace-step\\``, no diffusers-style subfolder ``config.json`` files
anywhere):

    <root>/diffusion_models/acestep_v1.5_{turbo,sft,base}.safetensors  -> DiT (677 tensors)
    <root>/vae/ace_1.5_vae.safetensors                                -> Oobleck VAE (365 tensors,
                                                                          stable-audio-tools Sequential naming)
    <root>/text_encoders/qwen_0.6b_ace15.safetensors                  -> Qwen3-Embedding-0.6B (310 tensors,
                                                                          "model."-prefixed Qwen3Model)

Key-mapping verdicts (verified 2026-07-13, strict `load_state_dict` on every
component):
  * DiT:          checkpoint keys match the vendored `AceStepConditionGenerationModel`
                  state_dict names AND shapes exactly (677/677, zero diff) — no
                  remap needed. All three DiT variants (base/sft/turbo) share
                  this identical key set (`is_turbo`/`model_version` are metadata
                  only, never read by the modeling code), so this loader works
                  for any of them.
  * VAE:          checkpoint uses the original stable-audio-tools Sequential
                  layout, NOT diffusers' `AutoencoderOobleck` naming -> remapped
                  via `core.models.acestep.vae_convert`.
  * text_encoder: checkpoint is `Qwen3Model` with an extra "model." prefix
                  (as if saved from a `Qwen3ForCausalLM` wrapper, though no
                  `lm_head` is present) -> the prefix is stripped before loading
                  into a bare `transformers.Qwen3Model`.

The DiT's `text_projector` / `lyric_encoder.embed_tokens` weights are fixed at
`text_hidden_dim=1024` (verified against the checkpoint), which only the 0.6B
Qwen3-Embedding tier matches — the co-shipped 1.7B/4B `text_encoders/*.safetensors`
are NOT drop-in compatible with this DiT and are out of scope here.

No local tokenizer files exist anywhere under the model root. The tokenizer
is resolved via sibling-probe (mirrors MiniT2I's FLAN-T5 resolution) with a
fallback to the public `Qwen/Qwen3-Embedding-0.6B` hub id, whose vocab_size
(151669) matches the local checkpoint's `embed_tokens` row count exactly.

`lyric_hidden_states` (the DiT's lyric conditioning input) are pre-computed
1024-dim embeddings, not token ids — `AceStepLyricEncoder.forward` asserts
`input_ids is None` and requires `inputs_embeds` directly. `AceStepConfig.vocab_size`
(64003) is never read anywhere in the vendored modeling code (no `nn.Embedding`
of that size, no `lm_head`); it is a vestigial field inherited from the Qwen3
config template. There is therefore no separate "lyric tokenizer" component to
vendor for Phase 0/1 — how lyrics text becomes those 1024-dim embeddings
(most likely: the same Qwen3 text encoder, applied to the raw lyrics string)
is a Phase 2 sampler-design question, not a missing asset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


# Priority order: turbo first (Phase 0/1 target); sft/base share the identical
# architecture (confirmed) and work through this same loader if selected.
ACESTEP_DIT_PATTERNS: List[str] = [
    "acestep_v1.5_turbo.safetensors",
    "acestep_v1.5_sft.safetensors",
    "acestep_v1.5_base.safetensors",
]
ACESTEP_VAE_PATTERNS: List[str] = [
    "ace_1.5_vae.safetensors",
]
# Only the 0.6B tier is shape-compatible with this DiT's text_projector (see
# module docstring); qwen_1.7b_ace15 / qwen_4b_ace15 are listed for detection
# completeness only and will raise a clear shape-mismatch error if forced.
ACESTEP_TE_PATTERNS: List[str] = [
    "qwen_0.6b_ace15.safetensors",
]

# Sibling directory names probed for a local Qwen3-Embedding-0.6B tokenizer
# before falling back to the HF hub id.
_QWEN3_TOKENIZER_SIBLING_NAMES = (
    "Qwen3-Embedding-0.6B",
    "qwen3-embedding-0.6b",
    "qwen_0.6b_ace15_tokenizer",
    "qwen3_embedding_0.6b_tokenizer",
)


def _find_first(directory: Path, patterns: List[str]) -> Optional[Path]:
    if not directory.is_dir():
        return None
    for pat in patterns:
        candidate = directory / pat
        if candidate.is_file():
            return candidate
    sf = sorted(directory.glob("*.safetensors"))
    return sf[0] if sf else None


def detect_acestep_layout(path: str) -> Optional[Dict[str, Optional[str]]]:
    """If `path` is a directory containing the ACE-Step flat ComfyUI-style
    layout (``diffusion_models/`` + ``vae/`` + ``text_encoders/``), return a
    dict `{dit, vae, text_encoder, root}` of absolute paths. Otherwise None.

    Also accepts a DiT `.safetensors` file living inside
    `<root>/diffusion_models/` directly (walks up to find `<root>`).
    """
    if not path:
        return None

    p = Path(path)
    if p.is_file() and p.suffix == ".safetensors":
        for parent in p.parents:
            if (parent / "diffusion_models").is_dir():
                root = parent
                vae = _find_first(root / "vae", ACESTEP_VAE_PATTERNS)
                te = _find_first(root / "text_encoders", ACESTEP_TE_PATTERNS)
                return {
                    "dit": str(p),
                    "vae": str(vae) if vae else None,
                    "text_encoder": str(te) if te else None,
                    "root": str(root),
                }
        return None

    if not p.is_dir():
        return None
    if not (p / "diffusion_models").is_dir():
        return None

    dit_dir = p / "diffusion_models"
    dit = _find_first(dit_dir, ACESTEP_DIT_PATTERNS)
    if dit is None:
        return None
    vae = _find_first(p / "vae", ACESTEP_VAE_PATTERNS)
    te = _find_first(p / "text_encoders", ACESTEP_TE_PATTERNS)
    return {
        "dit": str(dit),
        "vae": str(vae) if vae else None,
        "text_encoder": str(te) if te else None,
        "root": str(p),
    }


def _looks_like_qwen3_tokenizer_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    return (d / "tokenizer_config.json").is_file() and (
        (d / "tokenizer.json").is_file() or (d / "vocab.json").is_file()
    )


def _resolve_qwen3_tokenizer_source(root: Optional[str]) -> str:
    """Sibling-probe for a local Qwen3-Embedding-0.6B tokenizer directory next
    to the ACE-Step model root; else fall back to the public hub id (mirrors
    MiniT2I's FLAN-T5 resolution in `core.models.minit2i.minit2i_loader`)."""
    from .defaults import QWEN3_EMBEDDING_TOKENIZER_HUB_ID

    if root:
        root_path = Path(root)
        candidates = [root_path / name for name in _QWEN3_TOKENIZER_SIBLING_NAMES]
        candidates += [root_path / "text_encoders" / name for name in _QWEN3_TOKENIZER_SIBLING_NAMES]
        for cand in candidates:
            if _looks_like_qwen3_tokenizer_dir(cand):
                return str(cand)
    return QWEN3_EMBEDDING_TOKENIZER_HUB_ID


def _build_dit(dit_path: str, torch_dtype: torch.dtype):
    from .defaults import ACESTEP_DIT_CONFIG
    from .vendor import AceStepConditionGenerationModel, AceStepConfig
    from safetensors import safe_open

    config = AceStepConfig(**ACESTEP_DIT_CONFIG)
    model = AceStepConditionGenerationModel(config)

    with safe_open(dit_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        # load_state_dict(strict=True) already raises on mismatch; this is
        # unreachable in practice but kept for a clear failure mode if a
        # future non-strict caller changes this to strict=False.
        raise RuntimeError(
            f"ACE-Step DiT state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )

    model = model.to(dtype=torch_dtype)
    model.eval()
    return model, config


def _build_vae(vae_path: str, torch_dtype: torch.dtype):
    from diffusers import AutoencoderOobleck
    from .defaults import ACESTEP_VAE_CONFIG
    from .vae_convert import convert_oobleck_state_dict
    from safetensors import safe_open

    vae = AutoencoderOobleck(**ACESTEP_VAE_CONFIG)

    with safe_open(vae_path, framework="pt") as f:
        raw_state_dict = {k: f.get_tensor(k) for k in f.keys()}
    converted = convert_oobleck_state_dict(raw_state_dict)

    missing, unexpected = vae.load_state_dict(converted, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"ACE-Step VAE state_dict mismatch after conversion: missing={missing}, unexpected={unexpected}"
        )

    vae = vae.to(dtype=torch_dtype)
    vae.eval()
    return vae


def _build_text_encoder(te_path: str, torch_dtype: torch.dtype):
    from transformers import Qwen3Config, Qwen3Model
    from .defaults import ACESTEP_TEXT_ENCODER_CONFIG
    from safetensors import safe_open

    config = Qwen3Config(**ACESTEP_TEXT_ENCODER_CONFIG)
    text_encoder = Qwen3Model(config)

    with safe_open(te_path, framework="pt") as f:
        raw_state_dict = {k: f.get_tensor(k) for k in f.keys()}

    stripped = {}
    for k, v in raw_state_dict.items():
        if not k.startswith("model."):
            raise RuntimeError(
                f"ACE-Step text encoder checkpoint key missing expected 'model.' prefix: {k!r}"
            )
        stripped[k[len("model."):]] = v

    missing, unexpected = text_encoder.load_state_dict(stripped, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"ACE-Step text encoder state_dict mismatch: missing={missing}, unexpected={unexpected}"
        )

    text_encoder = text_encoder.to(dtype=torch_dtype)
    text_encoder.eval()
    return text_encoder, config


def load_acestep_from_path(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Load ACE-Step 1.5 (2B DiT + Oobleck VAE + Qwen3-Embedding-0.6B text
    encoder) from the flat ComfyUI-style model tree.

    Returns a component dict consumed by `PipelineManager.load_model()`:
        {
          "type": "acestep",
          "is_audio": True,
          "dit": <AceStepConditionGenerationModel>,
          "dit_config": <AceStepConfig>,
          "vae": <AutoencoderOobleck>,
          "text_encoder": <Qwen3Model>,
          "text_encoder_config": <Qwen3Config>,
          "tokenizer": <PreTrainedTokenizerFast> | None,
          "tokenizer_source": str,   # local dir or hub id actually used
          "sample_rate": 48000,
          "latent_frame_rate": 25,
          "latent_channels": 64,
          "dit_path": str, "vae_path": str, "text_encoder_path": str,
        }

    No sampler / scheduler / generation entry point yet (Phase 2).
    """
    from .defaults import SAMPLE_RATE, LATENT_FRAME_RATE, LATENT_CHANNELS

    layout = detect_acestep_layout(model_path)
    if layout is None:
        raise ValueError(
            f"ACE-Step model layout not found at {model_path!r}. "
            f"Expected a directory containing diffusion_models/ + vae/ + text_encoders/, "
            f"or a DiT .safetensors file inside a diffusion_models/ directory."
        )

    dit_path = layout["dit"]
    vae_path = layout["vae"]
    te_path = layout["text_encoder"]
    root = layout["root"]

    if vae_path is None:
        raise ValueError(f"ACE-Step VAE not found under {root!r}/vae/ (expected {ACESTEP_VAE_PATTERNS})")
    if te_path is None:
        raise ValueError(f"ACE-Step text encoder not found under {root!r}/text_encoders/ (expected {ACESTEP_TE_PATTERNS})")

    print(f"[AceStepLoader] DiT:          {dit_path}")
    print(f"[AceStepLoader] VAE:          {vae_path}")
    print(f"[AceStepLoader] Text encoder: {te_path}")

    dit, dit_config = _build_dit(dit_path, torch_dtype)
    vae = _build_vae(vae_path, torch_dtype)
    text_encoder, te_config = _build_text_encoder(te_path, torch_dtype)

    tokenizer_source = _resolve_qwen3_tokenizer_source(root)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        print(f"[AceStepLoader] Tokenizer:    {tokenizer_source}")
    except Exception as e:
        print(f"[AceStepLoader] WARNING: tokenizer load failed from {tokenizer_source!r}: {e}")

    # Keep everything on CPU after load (VRAM discipline; GPU staging is Phase 2).
    for comp in (dit, vae, text_encoder):
        if comp is not None and hasattr(comp, "to"):
            try:
                comp.to("cpu")
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[AceStepLoader] Loaded ACE-Step 1.5 components (CPU-resident; no sampler wired yet).")

    return {
        "type": "acestep",
        "is_audio": True,
        "dit": dit,
        "dit_config": dit_config,
        "vae": vae,
        "text_encoder": text_encoder,
        "text_encoder_config": te_config,
        "tokenizer": tokenizer,
        "tokenizer_source": tokenizer_source,
        "sample_rate": SAMPLE_RATE,
        "latent_frame_rate": LATENT_FRAME_RATE,
        "latent_channels": LATENT_CHANNELS,
        "dit_path": dit_path,
        "vae_path": vae_path,
        "text_encoder_path": te_path,
    }
