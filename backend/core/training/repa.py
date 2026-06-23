"""REPA (REPresentation Alignment) for diffusion-transformer training.

Aligns an intermediate hidden state of the MiniT2I DiT (the image tokens at a
chosen block depth) with clean-image per-patch features from a frozen pretrained
vision encoder, via a small trainable MLP projector and a cosine-similarity
regularization. The aligned encoder representation accelerates convergence of the
generator (Yu et al., "Representation Alignment for Generation: Training Diffusion
Transformers Is Easier Than You Think", ICLR 2025, arXiv:2410.06940).

Two encoder sources are supported, both SigLIP2 so400m (1152-dim, no CLS token):
  - "tagger" : our Danbooru/anime fine-tuned SigLIP2 (domain-matched; default).
  - "siglip2": an off-the-shelf google/siglip2 checkpoint.

The encoder runs on the CLEAN image, squished to its native square resolution
(SigLIP2 normalization is mean=std=0.5, i.e. the [-1,1] range MiniT2I already uses).
Patch features are bilinearly interpolated from the encoder's g x g grid to the DiT
token grid (gh x gw); both use row-major (h*gw + w) ordering, so tokens correspond.
The projector is training-only and is not part of the exported inference model.
"""

import os
import json
import math
import glob
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_SIGLIP2_REPO = "google/siglip2-so400m-patch14-384"


# ------------------------------------------------------------------
# Encoder loading
# ------------------------------------------------------------------

def _resolve_tagger_checkpoint(model_dir: str) -> Tuple[str, str]:
    """Resolve a tagger model directory to (checkpoint_path, base_repo_id).

    Prefers best_f1 / latest, else the newest top-level *.safetensors. The base
    repo id (for module structure) is read from base_model_metadata.json.
    """
    model_dir = (model_dir or "").strip().strip('"').strip("'")
    if not model_dir or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"REPA tagger model dir not found: {model_dir!r}")

    repo = _DEFAULT_SIGLIP2_REPO
    base_meta = os.path.join(model_dir, "base_model_metadata.json")
    if os.path.isfile(base_meta):
        try:
            with open(base_meta, "r", encoding="utf-8") as f:
                repo = json.load(f).get("vision_encoder_repo", repo) or repo
        except Exception:
            pass

    # Preferred named checkpoints (top-level only).
    for name in ("best_f1.safetensors", "latest.safetensors"):
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p, repo

    # Newest top-level step_*.safetensors (highest step number), else any.
    cands = [p for p in glob.glob(os.path.join(model_dir, "*.safetensors"))
             if not os.path.basename(p).startswith("base_model")]

    def _step_num(p: str) -> int:
        base = os.path.basename(p)
        digits = "".join(ch for ch in base if ch.isdigit())
        return int(digits) if digits else -1

    step_cands = [p for p in cands if os.path.basename(p).startswith("step_")]
    if step_cands:
        return max(step_cands, key=_step_num), repo
    if cands:
        return cands[0], repo
    raise FileNotFoundError(f"No .safetensors checkpoint found in REPA tagger dir: {model_dir}")


def load_repa_encoder(
    source: str,
    *,
    tagger_model_dir: str = "",
    siglip2_repo: str = "",
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    attn_implementation: str = "sdpa",
) -> Tuple[nn.Module, int, Optional[int]]:
    """Load a frozen vision encoder for REPA.

    Returns (encoder, enc_dim, native_size). native_size is the encoder's square
    input resolution (vision_config.image_size); None if it cannot be detected
    (e.g. naflex), in which case the caller must supply an explicit resolution.
    """
    source = (source or "tagger").strip().lower()

    if source == "tagger":
        ckpt, repo = _resolve_tagger_checkpoint(tagger_model_dir)
        # Reuse the tagger's encoder loader (handles merged + LoRA checkpoints).
        from core.tagger.siglip2_tagger_model import _load_vision_encoder
        encoder = _load_vision_encoder(ckpt, repo_id=repo, attn_implementation=attn_implementation)
        print(f"[REPA] Loaded tagger vision encoder: {os.path.basename(ckpt)} (base={repo})")
    elif source == "siglip2":
        repo = (siglip2_repo or _DEFAULT_SIGLIP2_REPO).strip().strip('"').strip("'")
        from transformers import AutoModel
        _ai = {"attn_implementation": attn_implementation} if attn_implementation else {}
        try:
            full = AutoModel.from_pretrained(repo, dtype=torch.float32, local_files_only=True, **_ai)
        except Exception:
            full = AutoModel.from_pretrained(repo, dtype=torch.float32, **_ai)
        encoder = full.vision_model
        print(f"[REPA] Loaded off-the-shelf SigLIP2 vision encoder: {repo}")
    else:
        raise ValueError(f"Unknown REPA encoder source: {source!r} (expected 'tagger' or 'siglip2')")

    encoder = encoder.to(device=device, dtype=dtype).eval()
    encoder.requires_grad_(False)

    cfg = getattr(encoder, "config", None)
    enc_dim = int(getattr(cfg, "hidden_size", 0)) if cfg is not None else 0
    if enc_dim <= 0:
        # Fallback: probe a tiny forward to read the feature dim.
        with torch.no_grad():
            size = int(getattr(cfg, "image_size", 384)) if cfg is not None else 384
            probe = torch.zeros(1, 3, size, size, device=device, dtype=dtype)
            enc_dim = int(encoder(pixel_values=probe).last_hidden_state.shape[-1])
    native_size = int(getattr(cfg, "image_size", 0)) if cfg is not None else 0
    native_size = native_size if native_size and native_size > 0 else None

    return encoder, enc_dim, native_size


# ------------------------------------------------------------------
# Preprocessing + target extraction
# ------------------------------------------------------------------

def preprocess_for_repa(images_m1p1: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a [-1,1] image batch [B,3,H,W] to a square [B,3,size,size].

    SigLIP2 normalization is mean=std=0.5, i.e. exactly the [-1,1] range MiniT2I
    uses, so no channel re-normalization is required — only a spatial resize.
    """
    x = images_m1p1
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def encode_repa_targets(
    encoder: nn.Module,
    images_m1p1: torch.Tensor,
    gh: int,
    gw: int,
    size: int,
) -> torch.Tensor:
    """Clean-image patch features aligned to the DiT token grid.

    Returns [B, gh*gw, enc_dim] in row-major (h*gw + w) order, matching the DiT
    image-token ordering.
    """
    enc_dtype = next(encoder.parameters()).dtype
    x = preprocess_for_repa(images_m1p1, size).to(dtype=enc_dtype)
    feat = encoder(pixel_values=x).last_hidden_state  # [B, N, D]
    B, N, D = feat.shape
    g = int(round(math.sqrt(N)))
    if g * g != N:
        raise ValueError(
            f"REPA encoder produced {N} tokens (non-square grid); fixed-square REPA "
            f"requires a square encoder grid. Use a fixed-resolution encoder."
        )
    feat = feat.reshape(B, g, g, D).permute(0, 3, 1, 2)  # [B, D, g, g]
    if (g, g) != (gh, gw):
        feat = F.interpolate(feat, size=(gh, gw), mode="bilinear", align_corners=False)
    feat = feat.permute(0, 2, 3, 1).reshape(B, gh * gw, D)  # [B, gh*gw, D]
    return feat


# ------------------------------------------------------------------
# Projector + loss
# ------------------------------------------------------------------

class RepaProjector(nn.Module):
    """Trainable 3-layer MLP head mapping DiT hidden -> encoder feature space.

    Training-only; discarded for inference (not saved into the single-file).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        width = hidden or max(2048, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def repa_loss(
    h_dit: torch.Tensor,
    targets: torch.Tensor,
    projector: RepaProjector,
) -> torch.Tensor:
    """1 - mean patch-wise cosine similarity between projected DiT hidden and targets.

    h_dit:   [B, N, hidden]  (DiT image tokens at the aligned block; grad-bearing)
    targets: [B, N, enc_dim] (frozen clean-image features; no grad)
    """
    proj = projector(h_dit)
    proj = F.normalize(proj.float(), dim=-1)
    tgt = F.normalize(targets.float(), dim=-1)
    cos = (proj * tgt).sum(dim=-1)  # [B, N]
    return 1.0 - cos.mean()
