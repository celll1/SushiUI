"""Unload-first adapters for live component switching."""

from __future__ import annotations

import gc
import threading
import time
import uuid
from typing import Any, Dict, Optional

from core.model_state_coordinator import model_state_coordinator


class ComponentSwitchError(RuntimeError):
    status_code = 422


class StaleComponentRevision(ComponentSwitchError):
    status_code = 409


class ComponentSwitchFailed(ComponentSwitchError):
    status_code = 500


_operation_lock = threading.Lock()
_operation: Optional[Dict[str, Any]] = None


def current_operation() -> Optional[Dict[str, Any]]:
    with _operation_lock:
        return dict(_operation) if _operation is not None else None


def _set_operation(**changes: Any) -> Dict[str, Any]:
    global _operation
    with _operation_lock:
        if _operation is None:
            _operation = {}
        _operation.update(changes)
        return dict(_operation)


def _release_device_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _switch_vision_encoder(manager: Any, candidate: Dict[str, Any]) -> None:
    manager.unload_vision_encoder()
    _release_device_cache()
    path = candidate.get("_path")
    if path:
        manager.load_vision_encoder(path)


def _switch_minimax_h3_text_encoder(manager: Any, candidate: Dict[str, Any]) -> None:
    """Detach every H3 TE owner before mapping the replacement.

    There is deliberately no ``.to()`` call in this function. Destruction of
    the assign=True storages closes the old file mapping; the loader weakref
    assertion proves no Python owner survived before another file is mapped.
    """
    components = getattr(manager, "minimax_h3_components", None)
    if not isinstance(components, dict):
        raise ComponentSwitchError("MiniMax-H3 components are not loaded.")
    old_path = components.get("text_encoder_path")
    old_origin = components.get("text_encoder_origin")
    new_path = candidate.get("_path")
    official = components.get("official_dir")
    if not isinstance(old_path, str) or not isinstance(new_path, str):
        raise ComponentSwitchError("H3 text-encoder path provenance is unavailable.")

    from core.models.minimax_h3.loader import (
        assert_no_live_text_encoder,
        build_minimax_h3_text_encoder,
    )

    # The component dict is the production owner. Keep-hot is cleared by the
    # caller before this function; no local variable ever captures the module.
    #
    # Health goes to degraded before the slot is emptied and only comes back
    # once something valid is in it again. The caller's failure handler can
    # only tell "the adapter never started" from "the adapter left a hole" by
    # this marker, and calling the hole ready re-enables generation against a
    # model whose text encoder is None.
    manager.component_health = "degraded"
    components["text_encoder"] = None
    components["text_encoder_config"] = None
    components["text_encoder_path"] = None
    components["text_encoder_origin"] = "unavailable"

    try:
        # The detachment assertion belongs inside the try: it is the check most
        # likely to fire here, and it fires precisely when a stale owner still
        # holds the old mapping -- exactly the state the restore path exists to
        # recover from.
        _release_device_cache()
        assert_no_live_text_encoder()
        replacement, config = build_minimax_h3_text_encoder(new_path, official)
    except Exception as switch_error:
        try:
            _release_device_cache()
            assert_no_live_text_encoder()
            restored, restored_config = build_minimax_h3_text_encoder(old_path, official)
            components["text_encoder"] = restored
            components["text_encoder_config"] = restored_config
            components["text_encoder_path"] = old_path
            components["text_encoder_origin"] = old_origin or "architecture_default"
            manager.component_health = "ready"
        except Exception as restore_error:
            manager.component_health = "degraded"
            raise ComponentSwitchFailed(
                f"H3 TE switch failed ({switch_error}); serial reload of the previous TE also "
                f"failed ({restore_error}). Generation is disabled until the model is reloaded."
            ) from restore_error
        raise ComponentSwitchFailed(
            f"H3 TE switch failed and the previous TE was reloaded serially: {switch_error}"
        ) from switch_error

    components["text_encoder"] = replacement
    components["text_encoder_config"] = config
    components["text_encoder_path"] = new_path
    components["text_encoder_origin"] = "selected_external"


def _switch_anima_component(manager: Any, slot: str, candidate: Dict[str, Any]) -> None:
    """Full unload-first Anima reload using its explicit companion inputs."""
    components = getattr(manager, "anima_components", None)
    info = getattr(manager, "current_model_info", None) or {}
    if not isinstance(components, dict) or info.get("type") != "anima":
        raise ComponentSwitchError("Anima components are not loaded.")
    source = info.get("source")
    source_type = info.get("source_type")
    if not isinstance(source, str) or not isinstance(source_type, str):
        raise ComponentSwitchError("Anima model source provenance is unavailable.")
    paths = components.get("paths") if isinstance(components.get("paths"), dict) else {}
    old_te = paths.get("text_encoder")
    old_vae = paths.get("vae") if components.get("vae_source") != "embedded (checkpoint)" else None
    selected = candidate.get("_path")
    if not isinstance(selected, str):
        raise ComponentSwitchError("Anima component candidate has no resolved path.")
    new_te = selected if slot == "text_encoder" else old_te
    new_vae = selected if slot == "vae" else old_vae

    def reload(text_encoder_path: Optional[str], vae_path: Optional[str]) -> None:
        manager._load_model_locked(
            source_type,
            source,
            "txt2img",
            force_reload=True,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
        )

    try:
        reload(new_te, new_vae)
    except Exception as switch_error:
        _release_device_cache()
        try:
            reload(old_te, old_vae)
            manager.component_health = "ready"
        except Exception as restore_error:
            manager.component_health = "degraded"
            raise ComponentSwitchFailed(
                f"Anima component switch failed ({switch_error}); serial reload of the previous "
                f"companions also failed ({restore_error}). Generation is disabled until reload."
            ) from restore_error
        raise ComponentSwitchFailed(
            f"Anima component switch failed and the prior companions were reloaded: {switch_error}"
        ) from switch_error


def switch_component(
    manager: Any,
    slot: str,
    candidate: Dict[str, Any],
    expected_model_revision: int,
    expected_component_revision: int,
) -> Dict[str, Any]:
    global _operation
    arch = str((getattr(manager, "current_model_info", None) or {}).get("type") or "").lower()
    adapter = None
    if slot == "vision_encoder":
        adapter = lambda: _switch_vision_encoder(manager, candidate)
    elif arch == "minimax_h3" and slot == "text_encoder":
        adapter = lambda: _switch_minimax_h3_text_encoder(manager, candidate)
    elif arch == "anima" and slot in ("text_encoder", "vae"):
        adapter = lambda: _switch_anima_component(manager, slot, candidate)
    if adapter is None:
        raise ComponentSwitchError("This component slot has no verified unload-first adapter.")
    if candidate.get("compatibility") != "compatible" or not candidate.get("switchable"):
        raise ComponentSwitchError(candidate.get("switch_reason") or "Candidate is not switchable.")
    if getattr(manager, "model_revision", 0) != expected_model_revision or getattr(manager, "component_revision", 0) != expected_component_revision:
        raise StaleComponentRevision("The loaded model or component selection changed; refresh and retry.")

    operation_id = str(uuid.uuid4())
    with _operation_lock:
        if _operation and _operation.get("state") in ("running", "waiting"):
            raise StaleComponentRevision("Another component switch is already running.")
        _operation = {
            "id": operation_id,
            "slot": slot,
            "state": "waiting",
            "phase": "waiting_for_lifecycle_gate",
            "started_at": time.time(),
        }

    try:
        with model_state_coordinator.mutation("component switch"):
            with manager._load_model_lock:
                if getattr(manager, "model_revision", 0) != expected_model_revision or getattr(manager, "component_revision", 0) != expected_component_revision:
                    raise StaleComponentRevision("The loaded model or component selection changed; refresh and retry.")
                manager.component_health = "mutating"
                _set_operation(state="running", phase="releasing_old")
                from core.keep_hot import clear_resident
                clear_resident(manager)
                _set_operation(phase="loading_new")
                adapter()
                manager.component_revision += 1
                manager.component_health = "ready"
                return _set_operation(state="succeeded", phase="complete", finished_at=time.time())
    except Exception as exc:
        if getattr(manager, "component_health", None) == "mutating":
            manager.component_health = "ready" if getattr(manager, "current_model_info", None) else "degraded"
        _set_operation(state="failed", phase="failed", error=str(exc), finished_at=time.time())
        raise
