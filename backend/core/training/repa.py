"""REPA (REPresentation Alignment) for diffusion-transformer training.

Aligns an intermediate hidden state of a diffusion transformer (the image tokens
at a chosen block depth) with clean-image per-patch features from a frozen
pretrained vision encoder, via a small trainable MLP projector and a
cosine-similarity regularization. The aligned encoder representation accelerates
convergence of the generator (Yu et al., "Representation Alignment for Generation:
Training Diffusion Transformers Is Easier Than You Think", ICLR 2025,
arXiv:2410.06940).

Two encoder sources are supported, both SigLIP2 so400m (1152-dim, no CLS token):
  - "tagger" : our Danbooru/anime fine-tuned SigLIP2 (domain-matched; default).
  - "siglip2": an off-the-shelf google/siglip2 checkpoint.

The encoder runs on the CLEAN image, squished to its native square resolution
(SigLIP2 normalization is mean=std=0.5, i.e. the [-1,1] range training images are
already in). Patch features are bilinearly interpolated from the encoder's g x g
grid to the DiT token grid (gh x gw); both use row-major (h*gw + w) ordering, so
tokens correspond. The projector is training-only and is not part of the exported
inference model.

Everything here is architecture-neutral. What an architecture supplies is a
``RepaTapPoint`` (from its arch handler) and the token grid its own geometry
defines; an architecture with neither is refused by ``refuse_repa`` rather than
silently ignored.
"""

import os
import json
import math
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    # Match the projector's dtype: the tap follows the transformer/autocast dtype,
    # which can differ from the projector when mixed_precision is off and
    # weight_dtype != training_dtype. Cast keeps the bf16 Linear from erroring.
    proj_dtype = next(projector.parameters()).dtype
    proj = projector(h_dit.to(proj_dtype))
    proj = F.normalize(proj.float(), dim=-1)
    tgt = F.normalize(targets.float(), dim=-1)
    cos = (proj * tgt).sum(dim=-1)  # [B, N]
    return 1.0 - cos.mean()


# ------------------------------------------------------------------
# Architecture support: tap point, refusals, depth compatibility
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RepaTapPoint:
    """Where one architecture exposes its REPA tap.

    ``module`` carries ``_repa_tap_depth`` (written once at setup) and
    ``_repa_tap_out`` (written by the forward at that depth) — which is NOT
    ``trainer.transformer`` for every architecture: MiniT2I's is
    ``transformer.model.net``, and several architectures wrap the transformer in
    a training wrapper. ``depth`` is the number of blocks the tap index
    addresses; ``hidden_size`` is the width of the tapped state and so fixes the
    projector's input dim.
    """

    module: nn.Module
    hidden_size: int
    depth: int


#: Why REPA is refused for an architecture where it cannot work, or where it
#: could but is deliberately held back. Anything absent from here (and without an
#: arch-handler ``repa_tap``) is refused as merely unwired.
REPA_REFUSALS: Dict[str, str] = {
    "acestep": (
        "ACE-Step is audio: its sequence is a 1-D time axis with no spatial token "
        "grid, so there is no grid to align to an image encoder's per-patch "
        "features. The alignment target is undefined for it, not merely expensive."
    ),
    "ltx2": (
        "LTX-2.3 is video and REPA is held back for it: the frozen image teacher "
        "would have to encode EVERY frame of every clip on every step, and the "
        "transformer trunk is shared with the audio stream, so the alignment term "
        "would also steer weights that produce audio. A deliberate hold, not a "
        "missing line of code."
    ),
    "minimax_h3": (
        "MiniMax-H3 is video and REPA is held back for it: per-frame teacher cost "
        "as for LTX-2.3, a trunk shared with audio, and its packed sequence "
        "interleaves CONDITION frames with the frames being generated, so a tap "
        "returns rows the image teacher has no matching target for. A deliberate hold."
    ),
}

#: Phrases that must survive any rewording of the refusals above. api/
#: arch_capabilities.py carries its own copy rather than importing this module:
#: base_trainer imports THAT module (lazily), so a module-level edge back would
#: close the cycle. repa_lens_tap_test.py pins the two dicts equal.
REPA_REFUSAL_MARKERS: Dict[str, str] = {
    "acestep": "1-D time axis",
    "ltx2": "shared with the audio stream",
    "minimax_h3": "packed sequence",
}

for _arch, _marker in REPA_REFUSAL_MARKERS.items():
    if _marker not in REPA_REFUSALS[_arch]:
        raise RuntimeError(
            f"REPA_REFUSALS['{_arch}'] no longer states {_marker!r}; the API "
            f"capability table's reason for it is checked against that phrase "
            f"(api/arch_capabilities._REPA_REFUSAL_MARKERS), so the two would "
            f"now be free to describe different facts."
        )

_REPA_UNWIRED = (
    "{arch} has no REPA tap at this stage: no arch-handler repa_tap() and no "
    "forward that stashes the aligned hidden state. REPA is architecture-neutral "
    "by design, but each architecture is wired one at a time; 'minit2i', 'anima', "
    "'lens', 'krea2', 'ideogram4' and 'sensenova' are wired today. Either set "
    "repa_enable=false or wire {arch}'s tap first."
)


def refuse_repa(arch_name: str):
    """Refuse REPA for ``arch_name``, saying why. Never returns."""
    reason = REPA_REFUSALS.get(arch_name) or _REPA_UNWIRED.format(arch=arch_name)
    raise ValueError(f"repa_enable is not supported for architecture '{arch_name}'. {reason}")


def assert_repa_depth_compatible(trainer, align_depth: int, num_blocks: int) -> None:
    """Refuse a tap depth that another training-time depth feature would void.

    Reads the configs the FORWARD will receive (``trainer.tread_config`` etc.,
    built earlier in ``__init__``) rather than the raw config keys: those carry
    per-key fallbacks of their own, so re-reading them here would check a span
    the run does not use.

    All three combinations below leave REPA running and reporting a loss while
    its gradient is wrong, zero, or absent on part of the steps, which is why
    they are refused rather than warned.
    """
    tread = getattr(trainer, "tread_config", None)
    blockskip = getattr(trainer, "blockskip_config", None)
    stochastic = getattr(trainer, "block_skip_config", None)

    if tread is not None:
        start = int(tread.get("start_block", 0) or 0)
        end = int(tread.get("end_block", 0) or 0)
        if start <= align_depth < end:
            raise ValueError(
                f"repa_align_depth={align_depth} is inside the TREAD routed span "
                f"[{start}, {end}). Inside that span a block sees only the kept subset "
                f"of tokens (routing reshapes them to [B,1,1,keep,D]), so the tapped "
                f"rows are neither the whole token grid nor in grid order and the "
                f"alignment loss would be taken against mismatched targets. Options: "
                f"(1) set repa_align_depth outside [{start}, {end}), (2) narrow the "
                f"TREAD span, (3) disable one of the two."
            )

    if blockskip is not None:
        front = int(blockskip.get("front", 0) or 0)
        back = int(blockskip.get("back", 0) or 0)
        last = num_blocks - back
        if not (front <= align_depth < last):
            raise ValueError(
                f"repa_align_depth={align_depth} is inside a DiT-BlockSkip skipped span. "
                f"BlockSkip runs blocks [0, {front}) and [{last}, {num_blocks}) only under "
                f"no_grad (their contribution re-enters as a detached residual), so a tap "
                f"there fires on the no-grad pass and the REPA gradient is exactly zero. "
                f"The span that trains is [{front}, {last}). Options: (1) set "
                f"repa_align_depth inside it, (2) reduce blockskip_front/blockskip_back, "
                f"(3) disable one of the two."
            )

    if stochastic is not None:
        from core.training.block_dropout import eligible_blocks
        protect_start = int(stochastic.get("protect_start", 0) or 0)
        protect_end = int(stochastic.get("protect_end", 0) or 0)
        rate = float(stochastic.get("skip_rate", 0.0) or 0.0)
        if align_depth in eligible_blocks(num_blocks, protect_start, protect_end):
            raise ValueError(
                f"repa_align_depth={align_depth} is a block stochastic depth may drop "
                f"(block_skip_rate={rate}; the protected span is "
                f"[{protect_start}, {protect_end}) and every block outside it is "
                f"eligible). A dropped block is replaced by identity and never writes "
                f"the tap, so on roughly {rate:.0%} of the steps REPA would contribute "
                f"nothing at all — silently, since the run still trains. Options: "
                f"(1) set repa_align_depth inside [{protect_start}, {protect_end}), "
                f"(2) widen block_skip_protect_start/end to cover it, (3) disable one "
                f"of the two."
            )


# ------------------------------------------------------------------
# Projector plumbing shared by every training adapter
# ------------------------------------------------------------------

def repa_enabled(trainer) -> bool:
    """True when this run has a live REPA projector to train and save."""
    return (bool(getattr(trainer, "repa_enable", False))
            and getattr(trainer, "repa_projector", None) is not None)


#: Longest first: an index path also ends in the shorter suffix.
CHECKPOINT_SUFFIXES = (".safetensors.index.json", ".safetensors")


def repa_sidecar_path(checkpoint_path) -> str:
    """REPA projector sidecar path next to a checkpoint (suffix-precise).

    Replaces only a trailing checkpoint suffix so a directory component
    containing ``.safetensors`` cannot corrupt the path. Both suffixes matter:
    a sharded save writes an index and returns THAT path, and the resume loader
    in ``BaseTrainer._setup_repa`` reads the same names back.
    """
    checkpoint_path = str(checkpoint_path)
    for suffix in CHECKPOINT_SUFFIXES:
        if checkpoint_path.endswith(suffix):
            return checkpoint_path[: -len(suffix)] + ".repa.safetensors"
    return checkpoint_path + ".repa.safetensors"


def projector_param_groups(trainer, *, label: str) -> List[Dict[str, Any]]:
    """The optimizer group for the REPA projector, or none when REPA is off.

    Without this group the projector never updates: the alignment target stays a
    random frozen head and the loss term is noise added to the diffusion loss,
    with no error anywhere. The base adapters append it last so param-group order
    is stable across resume.
    """
    if not repa_enabled(trainer):
        return []
    params = [p for p in trainer.repa_projector.parameters() if p.requires_grad]
    if not params:
        return []
    from core.training.adapters.base_adapter import resolve_component_lr
    base_lr = resolve_component_lr(trainer, "unet_lr", label=f"{label} REPA projector")
    lr = base_lr * float(getattr(trainer, "repa_proj_lr_factor", 1.0))
    print(f"[{label}] {sum(p.numel() for p in params):,} trainable params (REPA projector), lr={lr}")
    return [{"params": params, "lr": lr, "name": "repa_projector", "component": "repa_projector"}]


def save_projector_sidecar(trainer, checkpoint_path, *, label: str) -> None:
    """Write the projector beside ``checkpoint_path`` so a resume can pick it up.

    Training-only state: never embedded in the inference checkpoint.
    """
    if not repa_enabled(trainer):
        return
    try:
        from safetensors.torch import save_file as _save_file
        sib = repa_sidecar_path(checkpoint_path)
        sd = {k: v.detach().cpu().contiguous().float()
              for k, v in trainer.repa_projector.state_dict().items()}
        _save_file(sd, sib)
        print(f"[{label}] Saved REPA projector -> {sib}")
    except Exception as _e:
        print(f"[{label}] WARNING: REPA projector save failed: {_e}")


# ------------------------------------------------------------------
# Per-step use
# ------------------------------------------------------------------

def take_repa_tap(trainer) -> Optional[torch.Tensor]:
    """Read and clear the hidden state the forward stashed at the tap depth.

    Clearing releases this reference into the activation graph; the caller holds
    the only remaining one, for the length of the loss computation.
    """
    module = getattr(trainer, "_repa_tap_module", None)
    if module is None:
        return None
    out = getattr(module, "_repa_tap_out", None)
    module._repa_tap_out = None
    return out


def apply_repa_loss(trainer, loss, image_tokens, repa_pixels, gh: int, gw: int):
    """Add the alignment term for ``image_tokens`` [B, gh*gw, hidden] to ``loss``.

    The token grid (gh, gw) is the ARCHITECTURE's to compute — pixel-space,
    latent-space and packed-sequence architectures each derive it differently —
    as is any slicing that reduces the tap to image tokens.
    """
    targets = encode_repa_targets(
        trainer.repa_encoder,
        repa_pixels.to(device=trainer.device, dtype=trainer.training_dtype, non_blocking=True),
        gh, gw, trainer.repa_size,
    )
    rloss = repa_loss(image_tokens, targets, trainer.repa_projector)
    loss = loss + trainer.repa_weight * rloss
    trainer.log_extra_metric("repa_loss", float(rloss.detach().item()))
    del targets
    return loss
