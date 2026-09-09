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
silently ignored. Conv U-Nets tap a feature map instead of a token sequence and
use the ``*_spatial`` entry points below, which take the grid from the map.
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


def _read_teacher_config(repo: str, filename: str) -> Optional[dict]:
    """One of a teacher repo's config JSONs, read from disk only -- never the network."""
    repo = (repo or "").strip().strip('"').strip("'")
    if not repo:
        return None
    if os.path.isdir(repo):
        path = os.path.join(repo, filename)
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
            hit = try_to_load_from_cache(repo_id=repo, filename=filename)
        except Exception:
            hit = None
        path = hit if isinstance(hit, str) else ""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def teacher_fixed_input_size(repo: str) -> Tuple[Optional[int], Optional[str]]:
    """(declared square input size, evidence there is none) for a teacher repo.

    Read off what the checkpoint's own configs DECLARE -- the image processor's
    fixed ``size``, the vision config's ``image_size`` -- rather than off the
    repo id, which the next checkpoint renames. A NaFlex-style SigLIP2 declares
    neither: its processor states ``max_num_patches``/``patch_size`` for an
    aspect-preserving grid, and Siglip2VisionConfig carries no ``image_size``
    field at all. ``(None, None)`` means neither config was readable offline,
    which is not evidence of anything.
    """
    cfg = _read_teacher_config(repo, "config.json")
    proc = _read_teacher_config(repo, "preprocessor_config.json")
    if cfg is None and proc is None:
        return None, None

    vision = cfg or {}
    if isinstance(vision.get("vision_config"), dict):
        vision = vision["vision_config"]

    size = (proc or {}).get("size")
    if isinstance(size, dict):
        size = size.get("height") or size.get("shortest_edge") or size.get("width")
    for declared in (size, vision.get("image_size")):
        if isinstance(declared, int) and declared > 0:
            return int(declared), None

    parts = []
    if proc is not None:
        patches = proc.get("max_num_patches")
        parts.append(
            f"{proc.get('image_processor_type') or 'its image processor'} declares "
            f"no fixed size"
            + (f", max_num_patches={patches}, patch_size={proc.get('patch_size')}"
               if patches else ""))
    if cfg is not None:
        parts.append(f"vision config ({vision.get('model_type') or 'unknown model_type'}) "
                     f"declares no image_size")
    return None, "; ".join(parts)


def _teacher_candidate_repos(source: str, tagger_model_dir: str, siglip2_repo: str) -> List[str]:
    """The repos whose configs describe the teacher this run would load.

    Two for the tagger source: the base repo the encoder's structure comes from
    (base_model_metadata.json), and the one the chosen checkpoint's own sidecar
    metadata names -- they differ when a dir has no base_model_metadata.json, and
    the second is then the only record of what the checkpoint was trained as.
    """
    if (source or "tagger").strip().lower() != "tagger":
        return [(siglip2_repo or _DEFAULT_SIGLIP2_REPO).strip().strip('"').strip("'")]
    ckpt, repo = _resolve_tagger_checkpoint(tagger_model_dir)
    repos = [repo]
    meta = ckpt[: -len(".safetensors")] + "_metadata.json" if ckpt.endswith(".safetensors") else ""
    if meta and os.path.isfile(meta):
        try:
            with open(meta, "r", encoding="utf-8") as f:
                named = json.load(f).get("vision_encoder_repo")
            if isinstance(named, str) and named.strip() and named.strip() not in repos:
                repos.append(named.strip())
        except Exception:
            pass
    return repos


def assert_repa_teacher_fixed_resolution(
    source: str, *, tagger_model_dir: str = "", siglip2_repo: str = "",
) -> Optional[int]:
    """Refuse a teacher whose own configs declare no fixed square input.

    REPA crops the region an item's latent encoded, squishes it to one square and
    calls ``encoder(pixel_values=...)``. A NaFlex-style SigLIP2 was trained on
    aspect-preserving patch grids and its forward takes flattened patches plus
    ``spatial_shapes`` and ``pixel_attention_mask``, so that call raises TypeError
    at the first REPA step -- after the model, the dataset and the latent cache
    are up. Refused here, where the reason reaches the run's log.

    Returns the size the configs declare, when they declare one.
    """
    try:
        repos = _teacher_candidate_repos(source, tagger_model_dir, siglip2_repo)
    except Exception:
        return None  # an unresolvable tagger dir is load_repa_encoder's error to raise
    declared = None
    for repo in repos:
        size, evidence = teacher_fixed_input_size(repo)
        if declared is None:
            declared = size
        if not evidence:
            continue
        raise ValueError(
            f"repa_enable is not supported with the teacher {repo}: its configs "
            f"declare no fixed square input ({evidence}). REPA crops the region the "
            f"latent encoded, squishes it to one square and calls the encoder with "
            f"pixel_values alone; a variable-resolution SigLIP2 takes flattened "
            f"patches plus spatial_shapes and pixel_attention_mask, and was trained "
            f"on aspect-preserving grids of at most max_num_patches patches. "
            f"repa_encoder_resolution changes neither fact. Options: (1) point "
            f"repa_tagger_model_dir at a tagger checkpoint whose base is a "
            f"fixed-resolution SigLIP2 (google/siglip2-so400m-patch14-384 declares "
            f"image_size 384 and a 384x384 processor size), (2) "
            f"repa_encoder_source='siglip2' with repa_siglip2_repo set to such a "
            f"repo, (3) repa_enable=false."
        )
    return declared


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
    ``_repa_tap_out`` (written by the forward at that depth, or by a forward
    hook where the block loop is not ours) — which is NOT
    ``trainer.transformer`` for every architecture: MiniT2I's is
    ``transformer.model.net``, and several architectures wrap the transformer in
    a training wrapper. ``depth`` is the number of blocks the tap index
    addresses; ``hidden_size`` is the width of the tapped state and so fixes the
    projector's input dim.
    """

    module: nn.Module
    hidden_size: int
    depth: int
    #: Human names for indices 0..depth-1, when the index is not a block number.
    #: The conv U-Nets set it: their "depth" is a three-site menu, so a bare
    #: number in the log would read as a transformer block depth it is not.
    site_labels: Tuple[str, ...] = ()


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
    "zimage": (
        "Z-Image's tap is DEFERRED, not impossible: its block loop is vendored and "
        "its sequence would need the same text-prefix slice Krea 2 and Ideogram 4 "
        "took, and no owner has a base checkpoint here to measure the row order "
        "against. Held until there is demand for it rather than wired untested."
    ),
    "flux2": (
        "FLUX.2's tap is DEFERRED, not impossible: 8 dual-stream plus 48 "
        "single-stream blocks are two different tap shapes in one depth axis, on "
        "top of the packed text+image sequence, and no owner has a base checkpoint "
        "here to measure the row order against. Held until there is demand for it "
        "rather than wired untested."
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
    "zimage": "DEFERRED, not impossible",
    "flux2": "two different tap shapes",
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
    "'lens', 'krea2', 'ideogram4', 'sensenova', 'sd15' and 'sdxl' are wired today. "
    "Either set repa_enable=false or wire {arch}'s tap first."
)


def refuse_repa(arch_name: str):
    """Refuse REPA for ``arch_name``, saying why. Never returns."""
    reason = REPA_REFUSALS.get(arch_name) or _REPA_UNWIRED.format(arch=arch_name)
    raise ValueError(f"repa_enable is not supported for architecture '{arch_name}'. {reason}")


def latent_source_strategy(latent_encoding_mode: str, bucket_strategy: str) -> str:
    """The bucket strategy the latent a run consumes was ACTUALLY encoded with.

    The disk latent cache is written by ``encode_image`` calls that pass no
    ``bucket_strategy`` (``_validate_and_generate_latent_caches``,
    ``_regenerate_single_latent``), i.e. by its default center crop, whatever the
    config asks for. ``controlnet_trainer`` refuses outpaint on
    ``pre_encoded_cache`` over this same fact. The two on-the-fly modes pass the
    configured strategy through.
    """
    mode = str(latent_encoding_mode or "swap_onthefly")
    if mode == "pre_encoded_cache":
        return "crop"
    return str(bucket_strategy or "resize")


def assert_repa_region_reconstructible(latent_encoding_mode, bucket_strategy) -> None:
    """Refuse a preprocessing configuration whose latent crop the teacher cannot follow.

    REPA aligns student tokens to teacher patches position by position, so the
    teacher has to encode the same pixels the latent did. A ``random_crop``
    window is drawn inside ``encode_image`` and recorded nowhere, so it can only
    be followed when the encode runs in the same batch-loop iteration that reads
    it -- which is ``onthefly_gpu`` and nothing else.

    Takes the values ``train()`` was handed rather than the config: train_runner
    passes ``bucket_strategy="resize"`` whatever the config holds, so a
    config-based check would refuse runs that align perfectly. Still called
    before the first encode.
    """
    mode = str(latent_encoding_mode or "swap_onthefly")
    strategy = latent_source_strategy(mode, str(bucket_strategy or "resize"))

    if strategy in ("resize", "crop"):
        return
    if strategy == "random_crop":
        if mode == "onthefly_gpu":
            return
        raise ValueError(
            f"repa_enable cannot be combined with bucket_strategy='random_crop' "
            f"under latent_encoding_mode={mode!r}. The random window is drawn "
            f"inside encode_image and stored nowhere, so a latent taken from the "
            f"swap buffer cannot tell the REPA teacher which region it holds, and "
            f"the alignment would be taken against a different part of the image. "
            f"Options: (1) latent_encoding_mode='onthefly_gpu', which encodes each "
            f"item in the iteration that reads it), "
            f"(2) bucket_strategy='resize' or 'crop', (3) repa_enable=false."
        )
    raise ValueError(
        f"repa_enable cannot be combined with bucket_strategy={strategy!r}: REPA "
        f"reconstructs the encoded region per item and knows only 'resize', "
        f"'crop' and 'random_crop'."
    )


def resolve_align_depth(configured: int, depth: int) -> int:
    """The tap index a run arms: ``-1`` = auto (a third of the way in), clamped.

    One rule, because an architecture that has to know the resolved index before
    ``_setup_repa`` runs (the U-Nets read the tapped block's width from it) must
    resolve it identically.
    """
    align = int(configured)
    if align < 0:
        align = max(0, depth // 3)
    return max(0, min(align, depth - 1))


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
# Spatial taps (conv U-Nets)
# ------------------------------------------------------------------
# A U-Net has no token sequence and no total depth order: down, mid and up are
# joined by skip connections. What it does have is a feature MAP, [B, C, h, w],
# whose cells already lie on a grid -- so the teacher is interpolated to that
# (h, w) and the two correspond cell for cell, with no packing order to derive.
#
# REPA was published for DiTs (arXiv:2410.06940); aligning a conv U-Net's mid
# block is an extrapolation from it, not something that paper measured.


def spatial_tap_sites(unet: nn.Module) -> List[Tuple[str, nn.Module]]:
    """The blocks REPA can read on a diffusers U-Net, in forward order.

    Three, not every block: the shallow down blocks run at full latent
    resolution, where a 27x27 teacher grid upsampled to 128x128 carries no
    information the deep sites do not, at many times the projector cost. These
    are the deepest down block, the mid block, and the first up block.
    """
    down = getattr(unet, "down_blocks", None)
    mid = getattr(unet, "mid_block", None)
    up = getattr(unet, "up_blocks", None)
    if not down or mid is None or not up:
        raise ValueError(
            "REPA's spatial tap needs a U-Net with down_blocks, a mid_block and "
            "up_blocks; this module exposes "
            f"down={down is not None}, mid={mid is not None}, up={up is not None}.")
    return [
        (f"down_blocks[{len(down) - 1}]", down[len(down) - 1]),
        ("mid_block", mid),
        ("up_blocks[0]", up[0]),
    ]


def spatial_site_width(label: str, block: nn.Module) -> int:
    """Channel count of ``block``'s output map, read off the live module.

    The block's last resnet decides it; the optional down/up sampler that
    follows keeps the channel count. Read here rather than from
    ``config.block_out_channels`` so a config that no longer describes the
    loaded tree cannot size the projector.
    """
    resnets = getattr(block, "resnets", None)
    conv = getattr(resnets[-1], "conv2", None) if resnets else None
    width = int(getattr(conv, "out_channels", 0) or 0)
    if width <= 0:
        raise ValueError(
            f"REPA cannot read the output width of the U-Net site {label!r} "
            f"({type(block).__name__}): it has no resnets[-1].conv2.out_channels.")
    return width


def arm_spatial_tap(container: nn.Module, block: nn.Module):
    """Register the forward hook that stashes ``block``'s output on ``container``.

    A hook rather than an assignment because the loop that calls the block is
    diffusers' ``UNet2DConditionModel.forward``, which we do not own. It fires
    once per forward and outside every checkpoint segment: diffusers checkpoints
    each resnet/attention INSIDE a block, so the block's own output is an
    ordinary graph tensor either way (measured on both archs, checkpointing on
    and off). The caller must ``remove()`` the handle in a ``finally`` -- a hook
    left installed would also fire during sampling.
    """
    container._repa_tap_out = None

    def _stash(_module, _args, output):
        # Down blocks return (sample, res_samples); mid and up return the sample.
        container._repa_tap_out = output[0] if isinstance(output, tuple) else output

    return block.register_forward_hook(_stash)


def apply_repa_loss_spatial(trainer, loss, feature_map, repa_pixels):
    """Add the alignment term for a conv feature map [B, C, h, w] to ``loss``.

    The grid is the map's OWN (h, w), so the teacher is interpolated to exactly
    what was tapped and student and target cannot disagree about it. Flattening
    row-major (h*gw + w) is the order ``encode_repa_targets`` builds its grid in,
    and it makes the projector's channel-axis Linear a 1x1 convolution over the
    map.
    """
    if feature_map.dim() != 4:
        raise RuntimeError(
            f"REPA's spatial tap read a tensor of shape {tuple(feature_map.shape)}; "
            f"it expects a conv feature map [B, C, h, w]. A token sequence belongs "
            f"in apply_repa_loss, which takes the grid from the architecture.")
    if torch.is_grad_enabled() and not feature_map.requires_grad:
        raise RuntimeError(
            "REPA's spatial tap read a detached tensor, so the alignment term "
            "would be added to the loss and back-propagate into nothing. A "
            "reentrant gradient checkpoint around the tapped block detaches its "
            "hook output this way (diffusers checkpoints inside the block with "
            "use_reentrant=False, which does not).")
    batch, channels, gh, gw = feature_map.shape
    if repa_pixels.shape[0] != batch:
        raise RuntimeError(
            f"REPA has {repa_pixels.shape[0]} clean image(s) for a tap of batch "
            f"{batch}; the teacher targets would be broadcast across items rather "
            f"than paired with them.")
    tokens = feature_map.permute(0, 2, 3, 1).reshape(batch, gh * gw, channels)
    return apply_repa_loss(trainer, loss, tokens, repa_pixels, gh, gw)


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
