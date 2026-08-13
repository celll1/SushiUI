"""Background job that writes the LOADED, weight-only quantized transformer to disk.

Backs ``GET`` / ``POST /api/v1/models/export-quantized``.

WHY A JOB AND NOT A PLAIN REQUEST
---------------------------------
The write is a multi-GB, multi-minute, CPU-bound operation (an int8 Krea 2
transformer is ~12 GB). Doing it inside the request would hold the event loop's
executor for minutes and give the user no progress. So: a single worker thread,
a polled status document, and a hard refusal to start a second one.

SERIALIZATION
-------------
The worker holds ``pipeline_manager._load_model_lock`` for the whole write. That
is the same lock ``load_model`` takes, so a model (re)load cannot swap the
transformer out from under a running export -- the one interleaving that would
produce a file mixing two models. Generations do NOT take that lock, which is
why the START path additionally refuses while a generation or a training run is
in flight (``_fp8_scaled_mm_busy_reason``): a generation mutates component
device placement and (for a not-yet-converted model) can convert layers in
place, and an export must describe one settled state.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

# Architectures whose live module has a single-file export layout.
from core.models.common.quantized_export import (
    EXPORT_LAYOUTS,
    combined_linear_inventory,
    export_quantized_transformer,
    layout_module_specs,
)


_lock = threading.Lock()
_job: Optional[Dict] = None

# How long an export waits for the model-lifecycle gate before giving up. Long
# enough to outlast a generation queue the user forgot about, short enough that
# a job wedged behind a stuck gate still reports failure the same day.
_EXPORT_GATE_WAIT_SECONDS = 3600.0


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

def _loaded_transformer(pipeline_manager) -> Tuple[Optional[str], Optional[list], Optional[str]]:
    """``(arch, [(component name, module), ...], reason_it_is_unavailable)``.

    Only the architectures with an ``EXPORT_LAYOUTS`` entry are resolvable; any
    other loaded model returns a reason rather than modules.

    The component NAMES come from the layout, not from a hardcoded
    ``"transformer"``: an architecture that exports two transformers into one
    file (Ideogram 4) declares both there, and every one of them must be present
    or the export is refused -- half a model in a file that claims to be whole is
    worse than no file.
    """
    info = getattr(pipeline_manager, "current_model_info", None) or {}
    arch = str(info.get("type") or "") or None
    if not arch:
        return None, None, "no model is loaded"
    if arch not in EXPORT_LAYOUTS:
        return arch, None, (
            f"single-file export is implemented for "
            f"{', '.join(sorted(EXPORT_LAYOUTS))}; the loaded model is '{arch}'")
    components = getattr(pipeline_manager, f"{arch}_components", None) or {}
    modules = []
    for name, _prefix in layout_module_specs(arch):
        module = components.get(name)
        if module is None:
            return arch, None, f"the loaded {arch} model exposes no {name} component"
        modules.append((name, module))
    return arch, modules, None


def _model_source(pipeline_manager) -> Optional[str]:
    info = getattr(pipeline_manager, "current_model_info", None) or {}
    source = info.get("source")
    return str(source) if source else None


def _source_root(arch: str, source: Optional[str]) -> Optional[str]:
    """The directory a sibling junction set should be taken from.

    For the architectures whose loader walks UP from the weight file to a model
    ROOT, that root is what the siblings live next to -- not the file's own
    directory:

    * Anima: ``<root>/split_files/diffusion_models/<dit>.safetensors``, with
      ``split_files/text_encoders`` and ``split_files/vae`` beside it;
    * ACE-Step: ``<root>/diffusion_models/<dit>.safetensors``, with ``vae/`` and
      ``text_encoders/`` beside it. ``detect_acestep_layout`` accepts that file
      path directly, so it is a real ``current_model_info["source"]``, and
      falling back to ``os.path.dirname`` would take the siblings from
      ``diffusion_models/`` -- which holds neither -- and produce an exported
      tree the loader cannot complete.
    * MiniMax-H3: the same flat shape as ACE-Step
      (``<root>/diffusion_models/<dit>.safetensors`` with ``vae/``,
      ``text_encoders/`` and MiniMax's config-only ``official/`` beside it), and
      the same consequence -- ``official/`` in particular is where the loader
      reads every config and both tokenizers from, so an export that junctioned
      from ``diffusion_models/`` would produce a tree with no geometry at all.

    Each is asked THEIR OWN detector rather than having the walk re-implemented
    here, so a layout change moves one file.
    """
    if not source or not os.path.exists(source):
        return None
    detectors = {
        "anima": ("core.models.anima.anima_loader", "detect_anima_split_layout"),
        "acestep": ("core.models.acestep.loader", "detect_acestep_layout"),
        "minimax_h3": ("core.models.minimax_h3.loader", "detect_minimax_h3_layout"),
    }
    if arch in detectors:
        module_name, func_name = detectors[arch]
        try:
            import importlib

            layout = getattr(importlib.import_module(module_name), func_name)(source)
            if layout and layout.get("root"):
                return str(layout["root"])
        except Exception:
            pass
    return source if os.path.isdir(source) else os.path.dirname(source)


def _default_export_root() -> str:
    from config.settings import settings
    return os.path.join(settings.models_dir, "quantized_exports")


def suggested_output_path(arch: str, source: Optional[str], inventory: Dict) -> str:
    """A pre-fillable destination for ``arch``.

    Under the repo's own ``models/quantized_exports/`` (never inside the loaded
    model's own directory tree — those roots hold vanilla checkpoints and their
    siblings are loader completion sources), in the SUBDIRECTORY the arch's
    loader needs (Anima: ``split_files/diffusion_models``).
    """
    stem = "model"
    if source:
        base = os.path.basename(source.rstrip("\\/"))
        if base.endswith(".safetensors"):
            base = base[: -len(".safetensors")]
        stem = base or stem
    suffix = "int8" if inventory.get("int8") else "fp8"
    stem = f"{stem}_{suffix}"
    subdir = str(EXPORT_LAYOUTS[arch].get("output_subdir", "") or "")
    directory = os.path.join(_default_export_root(), stem, subdir) if subdir \
        else os.path.join(_default_export_root(), stem)
    return os.path.normpath(os.path.join(directory, f"{stem}.safetensors"))


def _is_inside(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(child), os.path.abspath(parent)]) \
            == os.path.abspath(parent)
    except ValueError:  # different drives
        return False


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _job_snapshot() -> Dict:
    with _lock:
        if _job is None:
            return {"state": "idle"}
        return dict(_job)


def job_is_running() -> bool:
    with _lock:
        return _job is not None and _job.get("state") == "running"


def export_status(pipeline_manager) -> Dict:
    """The document ``GET /models/export-quantized`` returns."""
    arch, modules, reason = _loaded_transformer(pipeline_manager)
    inventory = {"int8": 0, "e4m3": 0, "plain": 0, "total": 0}
    source = _model_source(pipeline_manager)
    suggested = None
    exportable = False
    if modules is not None:
        try:
            full = combined_linear_inventory(modules)
        except RuntimeError as exc:
            # The module tree changed under the walk -- an in-place runtime
            # conversion is replacing Linears right now. Transient: this is a
            # polled status, so report it and let the next poll settle it,
            # rather than 500-ing the whole endpoint.
            return {
                "exportable": False,
                "reason": f"the transformer is being modified right now ({exc})",
                "arch": arch,
                "source": source,
                "inventory": inventory,
                "has_runtime_audit": bool(getattr(pipeline_manager, "_runtime_int8_audit", None)),
                "suggested_path": None,
                "job": _job_snapshot(),
            }
        inventory = {k: full[k] for k in ("int8", "e4m3", "plain", "total")}
        if inventory["int8"] or inventory["e4m3"]:
            exportable = True
            suggested = suggested_output_path(arch, source, inventory)
        else:
            reason = (
                "the loaded transformer owns no weight-only quantized Linear layers "
                "(load a checkpoint that is already quantized, or generate once with "
                "INT8 quantization requested to convert it in place)")
    audit = getattr(pipeline_manager, "_runtime_int8_audit", None)
    return {
        "exportable": exportable,
        "reason": reason,
        "arch": arch,
        "source": source,
        "inventory": inventory,
        "has_runtime_audit": bool(audit),
        "suggested_path": suggested,
        "job": _job_snapshot(),
    }


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

class ExportBusyError(RuntimeError):
    """Something else is using the model; the export must not start."""


class ExportUnavailableError(ValueError):
    """The loaded model cannot be exported (nothing quantized, wrong arch, ...)."""


def start_export(
    pipeline_manager,
    output_path: str,
    *,
    link_siblings: bool = True,
    overwrite: bool = False,
) -> Dict:
    """Validate, then start the worker thread. Returns the initial job document."""
    global _job

    arch, modules, reason = _loaded_transformer(pipeline_manager)
    if modules is None:
        raise ExportUnavailableError(reason or "no exportable model is loaded")
    inventory = combined_linear_inventory(modules)
    if not (inventory["int8"] or inventory["e4m3"]):
        raise ExportUnavailableError(
            "the loaded transformer owns no weight-only quantized Linear layers")

    output_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path.endswith(".safetensors"):
        raise ExportUnavailableError("the destination must end in '.safetensors'")

    source = _model_source(pipeline_manager)
    source_root = _source_root(arch, source)
    if source_root and _is_inside(output_path, source_root):
        raise ExportUnavailableError(
            f"refusing to write the export inside the loaded model's own directory "
            f"({source_root}): that tree holds the source checkpoint and the sibling "
            f"directories the loader completes from. Choose a separate destination.")

    with _lock:
        if _job is not None and _job.get("state") == "running":
            raise ExportBusyError(
                f"an export to {_job.get('output_path')} is already running")
        job_id = uuid.uuid4().hex[:12]
        _job = {
            "job_id": job_id,
            "state": "running",
            "arch": arch,
            "output_path": output_path,
            "written_path": None,
            "processed": 0,
            "total": 0,
            "message": "starting",
            "error": None,
            "result": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        snapshot = dict(_job)

    thread = threading.Thread(
        target=_run_export,
        args=(pipeline_manager, job_id, arch, modules, output_path),
        kwargs={"link_siblings": link_siblings, "overwrite": overwrite,
                "source": source, "source_root": source_root},
        name=f"quant-export-{job_id}",
        daemon=True,
    )
    thread.start()
    return snapshot


def _update(job_id: str, **fields) -> None:
    with _lock:
        if _job is None or _job.get("job_id") != job_id:
            return
        _job.update(fields)


def _run_export(pipeline_manager, job_id, arch, modules, output_path, *,
                link_siblings, overwrite, source, source_root):
    def progress(done: int, total: int, key: str) -> None:
        _update(job_id, processed=done, total=total, message=f"writing {key}")

    try:
        # The PRIMARY component's config, for the arch metadata builder that
        # wants one (krea2's ``krea2_config``, flux2's ``flux2_config``). Read
        # generically rather than per-arch: a diffusers ``ConfigMixin`` exposes
        # ``.config`` and a module that has none yields {}, which every builder
        # already tolerates -- so an arch added later gets its config without
        # another branch here.
        #
        # The LOADED components' own ``config`` entry is then layered on top,
        # because the module's ``.config`` is geometry only. The loader's dict is
        # geometry PLUS the provenance it resolved for this model, and for FLUX.2
        # that provenance (``base_model_repo``, ``is_distilled``) is what decides
        # whether the reloaded export runs CFG at all. Dropping it would export a
        # full-FT save as something the loader then mis-detects. Generic for the
        # same reason as above: an arch whose components carry no ``config`` dict
        # is unaffected.
        try:
            config = dict(getattr(modules[0][1], "config", {}) or {})
        except Exception:
            config = None
        try:
            loaded = getattr(pipeline_manager, f"{arch}_components", None) or {}
            provenance = loaded.get("config")
            if isinstance(provenance, dict) and provenance:
                config = {**(config or {}), **provenance}
        except Exception:
            pass

        audit = getattr(pipeline_manager, "_runtime_int8_audit", None)
        audit_note = None
        if not audit:
            audit_note = (
                "no per-layer audit exists for this module: the quantized layers came "
                "from the loaded checkpoint, not from an in-place conversion in this "
                "session")

        from core.model_state_coordinator import model_state_coordinator
        load_lock = getattr(pipeline_manager, "_load_model_lock", None)
        _update(job_id, message="waiting for the model lifecycle gate")

        def _report_wait(reasons):
            _update(job_id, message="waiting for " + ", ".join(reasons))

        # Serializing weights while generation moves them between CPU and GPU
        # would write torn tensors, so this needs the exclusive gate -- but as a
        # background job it queues for it rather than dying on whatever happened
        # to be running when the user pressed the button. Subprocess training is
        # not a blocker: it never touches these in-process modules.
        with model_state_coordinator.mutation(
            "quantized export",
            wait_timeout=_EXPORT_GATE_WAIT_SECONDS,
            wait_for_activities=False,
            on_wait=_report_wait,
        ):
            if load_lock is not None:
                load_lock.acquire()
            try:
                _update(job_id, message="writing")
                result = export_quantized_transformer(
                    modules,
                    arch,
                    output_path,
                    config=config,
                    audit=audit,
                    audit_note=audit_note,
                    source=source,
                    link_siblings_from=source_root if link_siblings else None,
                    progress_cb=progress,
                    overwrite=overwrite,
                )
            finally:
                if load_lock is not None:
                    load_lock.release()

        _update(job_id, state="completed", result=result,
                written_path=result.get("output_path"),
                message="completed", finished_at=time.time())
    except BaseException as exc:  # noqa: BLE001 - reported verbatim to the user
        import traceback
        traceback.print_exc()
        _update(job_id, state="failed", error=f"{type(exc).__name__}: {exc}",
                message="failed", finished_at=time.time())
