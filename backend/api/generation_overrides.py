"""User-selectable VAE-override + TE-override for generation (RP2b).

This module carries the arch-independent plumbing for two per-generation
overrides that swap a single component of the LOADED model without a full
reload:

  * ``vae_path``          -> swap the decode/encode VAE
  * ``text_encoder_path`` -> swap the text encoder (SD1.5 / SDXL only)

The authoritative rules are the "RP2b — FABLE-DECIDED override rules" section of
``model_registry_spec.md``. This module implements the compatibility gate:

  * Candidate dims are read CHEAPLY through the component registry
    (``get_or_scan`` — header/config reads only, no weight load) plus a direct
    ``config.json`` probe for a standalone VAE/TE directory.
  * HARD mismatches raise ``ValidationError`` (HTTP 400) with both concrete dims.
  * Softer differences emit a ``add_warning`` (surfaced via ``warnings[]``).

The apply/restore lifecycle lives on ``PipelineManager`` (``load_override_vae`` /
``load_override_te``); this module only decides WHETHER to apply (arch gating +
compatibility) and drives the calls.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from api.error_handlers import ValidationError
from utils.path_redaction import display_name_for_path

# ---------------------------------------------------------------------------
# Per-arch default VAE class family (used only when a candidate/model provides
# no vae/config.json _class_name — e.g. a single-file checkpoint). The class
# name subsumes the normalization convention (shift_scale vs batchnorm vs
# identity), per the FABLE decision.
# ---------------------------------------------------------------------------
VAE_CLASS_BY_ARCH: Dict[str, Optional[str]] = {
    "sd15": "AutoencoderKL",
    "sdxl": "AutoencoderKL",
    "zimage": "AutoencoderKL",
    "flux2": "AutoencoderKLFlux2",
    "lens": "AutoencoderKLFlux2",
    "ideogram4": "AutoencoderKLFlux2",
    "anima": "AutoencoderKLQwenImage",
    "krea2": "AutoencoderKLQwenImage",
    "ltx2": "AutoencoderKLLTXVideo",
    "minit2i": None,  # pixel-space, no VAE
}

# latent_channels fallback when a VAE class omits the key in config.json.
_VAE_CLASS_DEFAULT_LC: Dict[str, int] = {
    "AutoencoderKLQwenImage": 16,
    "AutoencoderKLFlux2": 32,
}


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _vae_config_dir(path: str) -> Optional[str]:
    """Return the directory holding a VAE ``config.json`` for ``path``.

    Handles both a standalone VAE directory (``<path>/config.json``) and a
    diffusers model directory (``<path>/vae/config.json``).
    """
    if isinstance(path, str) and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "config.json")):
            cfg = _read_json(os.path.join(path, "config.json")) or {}
            # only treat as a VAE dir when the class looks like an autoencoder
            if str(cfg.get("_class_name", "")).startswith("Autoencoder"):
                return path
        if os.path.isfile(os.path.join(path, "vae", "config.json")):
            return os.path.join(path, "vae")
    return None


def _is_pid_checkpoint(path: str) -> Optional[str]:
    """Return the concrete PiD (Pixel Diffusion Decoder) ``.pth`` path when
    ``path`` is (or directly contains) one, else None.

    Recognizes a ``.pth`` file whose basename matches ``PiD_*`` (the NVIDIA PiD
    release naming, e.g. ``PiD_res2kto4k_sr4x_official_sdxl_distill_4step.pth``),
    or a directory containing exactly such a file at its top level.
    """
    if not isinstance(path, str) or not path:
        return None
    if os.path.isfile(path):
        base = os.path.basename(path)
        return path if (base.startswith("PiD_") and base.endswith(".pth")) else None
    if os.path.isdir(path):
        try:
            for name in os.listdir(path):
                if name.startswith("PiD_") and name.endswith(".pth"):
                    return os.path.join(path, name)
        except OSError:
            pass
    return None


def _te_config_dir(path: str) -> Optional[str]:
    if isinstance(path, str) and os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        if os.path.isfile(os.path.join(path, "text_encoder", "config.json")):
            return os.path.join(path, "text_encoder")
    return None


def _get_or_scan(path: str, source_type: Optional[str]) -> Dict[str, Any]:
    try:
        from core.models.component_registry import get_or_scan
        return get_or_scan(path, source_type) or {}
    except Exception:
        return {}


def describe_vae(path: str, source_type: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort observed VAE dims for a model/VAE at ``path``.

    Returns a dict with keys: ``latent_channels``, ``vae_class``,
    ``scale_spatial``, ``scale_temporal``, ``latent_ndim``, ``is_video``,
    ``present``, ``has_backbone``, ``arch``, ``kind``. Any unknown value is
    ``None``. ``kind`` is ``"pid_decoder"`` for a recognized PiD checkpoint
    (see ``_is_pid_checkpoint``), else ``"autoencoder"``.
    """
    pid_path = _is_pid_checkpoint(path)
    if pid_path is not None:
        # A .pth checkpoint has no config.json for `_vae_config_dir` to find, and
        # no component-registry entry — handle it as its own kind rather than
        # falling through to the (empty) registry-scan path below. PiD's SDXL
        # distilled variant is hardcoded 4-ch/sdxl-only for v1 (see design doc §6).
        return {
            "arch": "sdxl",
            "latent_channels": 4,
            "vae_class": None,
            "scale_spatial": None,
            "scale_temporal": None,
            "latent_ndim": 4,
            "is_video": False,
            "present": True,
            "has_backbone": False,
            "kind": "pid_decoder",
        }

    rec = _get_or_scan(path, source_type)
    comps = rec.get("components") or {}
    vcomp = comps.get("vae") or {}
    bb = comps.get("backbone") or {}
    arch = rec.get("arch")

    lc = vcomp.get("latent_channels")
    ss = vcomp.get("scale_spatial")
    ts = vcomp.get("scale_temporal")
    latent_ndim = rec.get("latent_ndim", 4)
    is_video = bool(rec.get("is_video", False))
    vae_class = None

    cfg_dir = _vae_config_dir(path)
    if cfg_dir:
        cfg = _read_json(os.path.join(cfg_dir, "config.json")) or {}
        vae_class = cfg.get("_class_name")
        if lc is None:
            lc = cfg.get("latent_channels")
        if ss is None:
            ss = cfg.get("spatial_compression_ratio")
        if ts is None:
            ts = cfg.get("temporal_compression_ratio")

    if vae_class is None:
        vae_class = VAE_CLASS_BY_ARCH.get(arch)
    if lc is None and vae_class in _VAE_CLASS_DEFAULT_LC:
        lc = _VAE_CLASS_DEFAULT_LC[vae_class]
    if ss is None:
        ss = (rec.get("expected") or {}).get("vae_scale_factor")

    present = bool(vcomp.get("present")) or cfg_dir is not None
    return {
        "arch": arch,
        "latent_channels": lc,
        "vae_class": vae_class,
        "scale_spatial": ss,
        "scale_temporal": ts,
        "latent_ndim": latent_ndim,
        "is_video": is_video,
        "present": present,
        "has_backbone": bool(bb.get("kind")),
        "kind": "autoencoder",
    }


def describe_te(path: str, source_type: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort observed text-encoder dims for a model/TE at ``path``."""
    rec = _get_or_scan(path, source_type)
    comps = rec.get("components") or {}
    tcomp = comps.get("text_encoder") or {}
    bb = comps.get("backbone") or {}
    arch = rec.get("arch")

    out_dim = tcomp.get("out_dim")
    te_type = tcomp.get("te_type")

    cfg_dir = _te_config_dir(path)
    if cfg_dir and not _vae_config_dir(path):
        cfg = _read_json(os.path.join(cfg_dir, "config.json")) or {}
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        if out_dim is None:
            hs = tc.get("hidden_size") or cfg.get("hidden_size") or cfg.get("d_model")
            if hs is not None:
                out_dim = int(hs)
        if te_type is None:
            mt = cfg.get("model_type") or (cfg.get("architectures") or [None])[0]
            if mt:
                te_type = str(mt)

    present = bool(tcomp.get("present")) or (cfg_dir is not None and not _vae_config_dir(path))
    return {
        "arch": arch,
        "out_dim": out_dim,
        "te_type": te_type,
        "present": present,
        "has_backbone": bool(bb.get("kind")),
    }


# ---------------------------------------------------------------------------
# Candidate classification (endpoints)
# ---------------------------------------------------------------------------

def _friendly_component_name(path: str) -> str:
    """Derive a human-readable candidate name from ``path``.

    If the basename is a real filename (e.g. ``my_custom_vae.safetensors``),
    it is used as-is (minus the ``.safetensors`` extension). If the basename is
    a generic diffusers subfolder name (e.g. ``vae``, ``text_encoder``), the
    parent directory (the model folder) is prepended to disambiguate, e.g.
    ``.../krea2/vae`` -> ``krea2/vae``.

    Delegates to ``utils.path_redaction`` so that this name — which is shown in
    the selector UI, recorded in ``params["vae_name"]`` and therefore written
    into shareable PNGs — is produced by the same single implementation that
    the PNG writer's redaction backstop uses. It never contains a drive letter
    or any directory above the one parent shown, and is never empty.
    """
    return display_name_for_path(path, strip_safetensors=True)


#: Provenance sidecar written next to every SushiUI VAE fine-tune export
#: (``core/training/vae/vae_trainer.py``: ``_write``).
VAE_TRAINING_SIDECAR = "sushi_vae_training.json"


def read_vae_training_sidecar(path: str) -> Optional[Dict[str, Any]]:
    """Return the trimmed provenance of a SushiUI VAE fine-tune export at
    ``path``, or None when ``path`` carries no such sidecar.

    Only the fields a candidate list needs are kept; the full sidecar (loss
    weights, base-VAE identity, EMA statistics) stays on disk.
    ``encoder_trained`` is carried because such a VAE encodes to a DIFFERENT
    latent distribution than its base model's VAE — it is not a drop-in
    replacement, and nothing about its config.json can reveal that.

    Every field is TRI-STATE: a key missing from (or unreadable in) a partial
    sidecar comes back as ``None``, never as a fabricated ``False``. Defaulting
    ``encoder_trained`` to False would present an unknown-provenance VAE as a
    drop-in replacement when it may not be one, and defaulting ``ema_applied``
    would be a positive factual claim about which weights the file holds. A
    malformed sidecar therefore degrades to "annotated, some fields unknown"
    rather than to a confident wrong answer — and never raises, so the entry
    cannot vanish from the candidate list.
    """
    if not isinstance(path, str) or not os.path.isdir(path):
        return None
    data = _read_json(os.path.join(path, VAE_TRAINING_SIDECAR))
    if not isinstance(data, dict):
        return None

    def _tri_bool(key: str) -> Optional[bool]:
        v = data.get(key)
        return bool(v) if isinstance(v, bool) else None

    base_vae = data.get("base_vae")
    base_vae_path = base_vae.get("path") if isinstance(base_vae, dict) else None

    return {
        "produced_by": data.get("produced_by"),
        "run_id": data.get("run_id"),
        "run_name": data.get("run_name"),
        "step": data.get("step"),
        "encoder_trained": _tri_bool("encoder_trained"),
        "ema_applied": _tri_bool("ema_applied"),
        "base_vae_path": base_vae_path,
    }


def classify_vae_candidate(path: str, source_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return a VAE candidate descriptor, or None when ``path`` is not a
    standalone VAE (a diffusers vae/ dir OR a model whose registry record has a
    VAE present and no backbone) and not a recognized PiD checkpoint.

    The descriptor includes a ``"kind"`` field: ``"pid_decoder"`` for a PiD
    (Pixel Diffusion Decoder) ``.pth`` checkpoint (see ``_is_pid_checkpoint``),
    else ``"autoencoder"`` for a normal VAE.

    A diffusers VAE directory that carries a ``sushi_vae_training.json`` sidecar
    (i.e. one exported by a SushiUI VAE fine-tune) additionally gets a
    ``"training"`` sub-dict — see ``read_vae_training_sidecar``. This is a plain
    annotation on the SAME classification path, not a second one: such an export
    is an ordinary diffusers VAE dir and is recognized as one either way.
    """
    pid_pth = _is_pid_checkpoint(path)
    if pid_pth is not None:
        # Always report the CONCRETE .pth path (not the containing dir when `path`
        # is a directory that holds it), so a checkpoint reachable both as its
        # dir and as its file collapses to one candidate after path-dedup.
        return {
            "name": _friendly_component_name(pid_pth),
            "path": pid_pth,
            "arch": "sdxl",
            "latent_channels": 4,
            "vae_class": None,
            "scale_spatial": None,
            "scale_temporal": None,
            "kind": "pid_decoder",
        }

    d = describe_vae(path, source_type)
    is_standalone = _vae_config_dir(path) is not None
    if not (is_standalone or (d["present"] and not d["has_backbone"])):
        return None
    name = _friendly_component_name(path)
    out = {
        "name": name,
        "path": path,
        "arch": d["arch"],
        "latent_channels": d["latent_channels"],
        "vae_class": d["vae_class"],
        "scale_spatial": d["scale_spatial"],
        "scale_temporal": d["scale_temporal"],
        "kind": "autoencoder",
    }
    training = read_vae_training_sidecar(path)
    if training is not None:
        out["training"] = training
    return out


def classify_te_candidate(path: str, source_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return a TE candidate descriptor, or None when ``path`` is not a
    standalone text encoder."""
    d = describe_te(path, source_type)
    is_standalone = _te_config_dir(path) is not None and _vae_config_dir(path) is None
    if not (is_standalone or (d["present"] and not d["has_backbone"])):
        return None
    name = _friendly_component_name(path)
    return {
        "name": name,
        "path": path,
        "arch": d["arch"],
        "out_dim": d["out_dim"],
        "te_type": d["te_type"],
    }


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------

# The compatibility gate deliberately runs BEFORE ``start_generation()`` (a
# HARD mismatch must raise HTTP 400 without opening a run), which meant every
# soft warning it emitted -- ``vae_override_warning``, ``te_override_warning``
# -- was dropped on the floor by ``add_warning()``: no generation was running
# yet, so nothing could be attached to. Those codes were therefore NEVER
# persisted on any image row from an image route. ``_warn`` now also records
# into a thread-local capture buffer that ``plan_overrides()`` returns in the
# plan and ``apply_overrides()`` (which always runs INSIDE the generation)
# replays. ``add_warning`` dedups identical entries per generation, so routes
# that already call ``plan_overrides`` after ``start_generation`` (the video
# routes) are unaffected by the replay.
_capture = threading.local()


def _warn(message: str, code: str) -> None:
    buffer = getattr(_capture, "buffer", None)
    if buffer is not None:
        entry = {"code": code, "message": message}
        if entry not in buffer:
            buffer.append(entry)
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


@contextmanager
def _capture_warnings(sink: list):
    """Collect ``_warn`` calls made on this thread into ``sink``.

    Nesting is not expected (``plan_overrides`` is a leaf), but the previous
    buffer is restored regardless so a nested call cannot orphan it.
    """
    previous = getattr(_capture, "buffer", None)
    _capture.buffer = sink
    try:
        yield
    finally:
        _capture.buffer = previous


def _check_vae_compat(loaded: Dict[str, Any], cand: Dict[str, Any]) -> None:
    """HARD -> ValidationError with both concrete dims; softer -> add_warning."""
    if cand.get("kind") == "pid_decoder":
        # PiD decoder override (F6/§6): SDXL-only for v1. Check ONLY the LOADED
        # model's arch/latent_channels against what this checkpoint variant
        # requires — vae_class/scale_spatial/scale_temporal comparisons are
        # skipped entirely (PiD is not an AutoencoderKL-family swap, so those
        # fields are meaningless for it; `cand`'s own arch/latent_channels are
        # hardcoded "sdxl"/4 by `classify_vae_candidate` and always compatible
        # with themselves).
        hard = []
        loaded_arch = loaded.get("arch")
        if loaded_arch and loaded_arch != "sdxl":
            hard.append(f"arch: PiD decoder is SDXL-only, loaded model arch={loaded_arch}")
        lc_l = loaded.get("latent_channels")
        if lc_l is not None and lc_l != 4:
            hard.append(f"latent_channels: loaded model VAE={lc_l}, PiD decoder requires 4 (SDXL)")
        if hard:
            raise ValidationError(
                "PiD decoder override is incompatible with the loaded model",
                detail="; ".join(hard),
            )
        if loaded_arch is None:
            _warn("PiD decoder override applied with an unverified property: loaded model arch is unknown",
                  code="vae_override_warning")
        return

    hard = []
    warns = []

    lc_l, lc_c = loaded["latent_channels"], cand["latent_channels"]
    if lc_l is not None and lc_c is not None:
        if lc_l != lc_c:
            hard.append(f"latent_channels: model VAE={lc_l}, candidate VAE={lc_c}")
    else:
        warns.append("latent_channels unknown for the model or candidate VAE")

    if loaded["latent_ndim"] != cand["latent_ndim"]:
        hard.append(
            f"latent_ndim: model={loaded['latent_ndim']}D "
            f"(is_video={loaded['is_video']}), candidate={cand['latent_ndim']}D "
            f"(is_video={cand['is_video']})"
        )

    vc_l, vc_c = loaded["vae_class"], cand["vae_class"]
    if vc_l and vc_c:
        if vc_l != vc_c:
            hard.append(f"vae_class family: model={vc_l}, candidate={vc_c}")
    else:
        warns.append("vae_class unknown for the model or candidate VAE")

    ss_l, ss_c = loaded["scale_spatial"], cand["scale_spatial"]
    if ss_l is not None and ss_c is not None:
        if ss_l != ss_c:
            hard.append(f"scale_spatial: model={ss_l}, candidate={ss_c}")
    else:
        warns.append("scale_spatial unknown for the model or candidate VAE")

    if loaded["is_video"]:
        ts_l, ts_c = loaded["scale_temporal"], cand["scale_temporal"]
        if ts_l is not None and ts_c is not None and ts_l != ts_c:
            hard.append(f"scale_temporal: model={ts_l}, candidate={ts_c}")

    if hard:
        raise ValidationError(
            "VAE override is incompatible with the loaded model",
            detail="; ".join(hard),
        )
    for w in warns:
        _warn(f"VAE override applied with an unverified property: {w}", code="vae_override_warning")


def _check_te_compat(loaded: Dict[str, Any], cand: Dict[str, Any]) -> None:
    hard = []
    warns = []

    od_l, od_c = loaded["out_dim"], cand["out_dim"]
    if od_l is not None and od_c is not None:
        if od_l != od_c:
            hard.append(f"text-encoder hidden dim: model={od_l}, candidate={od_c}")
    else:
        warns.append("text-encoder hidden dim unknown for the model or candidate")

    tt_l, tt_c = loaded["te_type"], cand["te_type"]
    if tt_l and tt_c and str(tt_l).lower() != str(tt_c).lower():
        warns.append(f"text-encoder type differs (model={tt_l}, candidate={tt_c})")

    if hard:
        raise ValidationError(
            "Text-encoder override is incompatible with the loaded model",
            detail="; ".join(hard),
        )
    for w in warns:
        _warn(f"Text-encoder override applied with an unverified property: {w}", code="te_override_warning")


def _warn_vae_training_provenance(apply_vae_path: str) -> None:
    """Surface what the STRUCTURAL gate cannot see about a fine-tuned VAE.

    ``_check_vae_compat`` compares latent_channels / latent_ndim / vae_class /
    scale_spatial — every one of which an encoder-trained VAE passes unchanged
    (``vae_trainer.py``'s ``_save_vae`` says as much: the latent SHAPE survives,
    the latent DISTRIBUTION does not). The only other signal is the export's
    directory name and a suffix on a ``<select>`` option label, which is exactly
    where a long string gets truncated. So the fact is also pushed onto the
    ``warnings[]`` channel of the generation response, where it cannot be
    clipped away.

    Decoder-only fine-tunes (the default, and the case the whole design is built
    around) stay silent: they leave the latent contract intact.
    """
    try:
        prov = read_vae_training_sidecar(apply_vae_path)
    except Exception:
        return
    if not prov:
        return

    run = prov.get("run_name") or "a SushiUI VAE fine-tune"
    enc = prov.get("encoder_trained")
    if enc is True:
        _warn(
            f"VAE override '{run}' was fine-tuned WITH ITS ENCODER: it encodes "
            f"to a different latent distribution than the base VAE, so cached "
            f"latents, LoRAs and diffusion checkpoints built against the base "
            f"VAE do not match it. The structural compatibility check cannot "
            f"detect this (latent channels, class family and spatial scale are "
            f"unchanged).",
            code="vae_override_warning",
        )
    elif enc is None:
        _warn(
            f"VAE override '{run}' carries an incomplete provenance sidecar: "
            f"whether its encoder was fine-tuned is unknown. If it was, cached "
            f"latents, LoRAs and diffusion checkpoints built against the base "
            f"VAE do not match it.",
            code="vae_override_warning",
        )


def check_override_compat(pipeline_manager, apply_vae_path: Optional[str],
                          apply_te_path: Optional[str]) -> None:
    """Validate candidate VAE/TE against the loaded model. Raises ValidationError
    (HTTP 400) on a HARD mismatch; emits warnings for softer differences."""
    info = getattr(pipeline_manager, "current_model_info", None) or {}
    loaded_path = info.get("source")
    loaded_st = info.get("source_type")

    if apply_vae_path:
        loaded_vae = describe_vae(loaded_path, loaded_st) if loaded_path else describe_vae("")
        cand_vae = describe_vae(apply_vae_path)
        _check_vae_compat(loaded_vae, cand_vae)
        _warn_vae_training_provenance(apply_vae_path)

    if apply_te_path:
        loaded_te = describe_te(loaded_path, loaded_st) if loaded_path else describe_te("")
        cand_te = describe_te(apply_te_path)
        _check_te_compat(loaded_te, cand_te)


# ---------------------------------------------------------------------------
# Plan + apply (drives the pipeline_manager lifecycle)
# ---------------------------------------------------------------------------

def plan_overrides(pipeline_manager, vae_path: Optional[str],
                   text_encoder_path: Optional[str]) -> Dict[str, Any]:
    """Decide which overrides to apply (arch gating) and run the compat gate.

    Returns a plan ``{"vae": path|None, "te": path|None, "vae_kind":
    "pid_decoder"|"autoencoder"|None, "warnings": [...]}``. ``warnings`` holds
    the soft compat notices raised here, which ``apply_overrides()`` replays
    once the generation is open — this function runs BEFORE
    ``start_generation()`` on the image routes, where ``add_warning()`` alone
    is a no-op. A requested override on an arch that
    does not support it is dropped from the plan (the caller's
    ``check_arch_capabilities`` emits the accepted-but-ignored warning). Raises
    ValidationError on a HARD incompatibility (including a PiD decoder override
    requested against a non-SDXL model). Call BEFORE ``start_generation``.
    """
    from api.arch_capabilities import arch_supports_feature
    arch = (getattr(pipeline_manager, "current_model_info", None) or {}).get("type")

    av = vae_path if (vae_path and arch_supports_feature(arch, "vae_override")) else None
    at = text_encoder_path if (text_encoder_path and arch_supports_feature(arch, "te_override")) else None

    vae_kind = describe_vae(av).get("kind") if av else None

    pending: List[Dict[str, Any]] = []
    if av or at:
        with _capture_warnings(pending):
            check_override_compat(pipeline_manager, av, at)
    return {"vae": av, "te": at, "vae_kind": vae_kind, "warnings": pending}


def apply_overrides(
    pipeline_manager,
    plan: Dict[str, Any],
    pid_sr_output: str = "4x",
    pid_use_gemma: bool = False,
    pid_low_vram: bool = False,
    pid_tile_native: int = 512,
    pid_tile_overlap_ratio: float = 0.25,
    pid_fast_large_decode: bool = False,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply (or restore, when a slot is None) the planned overrides.

    ``pid_sr_output``/``pid_use_gemma``/``pid_low_vram``/``pid_tile_native``/
    ``pid_tile_overlap_ratio``/``pid_fast_large_decode`` only matter when
    ``plan["vae_kind"] == "pid_decoder"`` (ignored otherwise). ``prompt`` is
    forwarded to the active ``PidVaeWrapper`` (via
    ``pipeline_manager.set_pid_prompt``) only when ``pid_use_gemma`` is set —
    the wrapper's opt-in runtime Gemma captioner needs the raw text prompt,
    which is not otherwise available at PiD's Stage-3 decode call site (those
    sites only see pre-computed embeddings). ``pid_low_vram`` opts into the
    row-chunked PiTBlock/FinalLayer decode path (default False = the exact
    original, unchunked forward — see ``PidVaeWrapper.low_vram_decode``).
    ``pid_tile_native``/``pid_tile_overlap_ratio`` configure the F9 tiled
    large-output decode (the default path once native > native_cap);
    ``pid_fast_large_decode=True`` opts back into the original whole-latent
    cap+bicubic path (see ``PidVaeWrapper``'s module docstring, F7/F9).

    Returns a metadata dict to fold into the generation params for the DB.
    Never raises: an apply failure degrades to a warning so generation can
    continue on the original component.
    """
    meta: Dict[str, Any] = {}

    # Replay the compat gate's soft warnings: `plan_overrides()` ran before
    # `start_generation()` on the image routes, so they had nowhere to land.
    # `add_warning` dedups per generation, so replaying a warning the caller
    # already recorded (video routes plan AFTER start_generation) is a no-op.
    for _pending in (plan.get("warnings") or []):
        _warn(_pending.get("message", ""), _pending.get("code") or "")

    try:
        pipeline_manager.load_override_vae(
            plan.get("vae"),
            override_kind=plan.get("vae_kind"),
            pid_sr_output=pid_sr_output,
            pid_use_gemma=pid_use_gemma,
            pid_low_vram=pid_low_vram,
            pid_tile_native=pid_tile_native,
            pid_tile_overlap_ratio=pid_tile_overlap_ratio,
            pid_fast_large_decode=pid_fast_large_decode,
        )
    except Exception as e:
        _warn(f"VAE override could not be applied: {e}", code="vae_override_error")

    try:
        pipeline_manager.load_override_te(plan.get("te"))
    except Exception as e:
        _warn(f"Text-encoder override could not be applied: {e}", code="te_override_error")

    if plan.get("vae"):
        try:
            src, _ = pipeline_manager.override_vae_identity()
        except Exception:
            src = plan["vae"]
        meta["vae_override_source"] = src
        meta["vae_override_path"] = plan["vae"]
        if plan.get("vae_kind") == "pid_decoder":
            # Always refresh the wrapper's prompt state for a PiD generation so a
            # prior generation's prompt can't leak: set it when runtime Gemma is on,
            # else clear it (None). Gated only on truthiness before, which left a
            # stale prompt when the next gen had an empty prompt.
            try:
                pipeline_manager.set_pid_prompt(prompt if pid_use_gemma else None)
            except Exception:
                pass
    if plan.get("te"):
        meta["text_encoder_override_path"] = plan["te"]
    return meta
