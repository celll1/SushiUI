"""Fail-closed catalog of effective model components and scanned candidates."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Iterable, List, Optional

from core.models.component_registry import _WIRING_BY_ARCH
from utils.path_redaction import display_name_for_path


SLOTS = ("text_encoder", "vision_encoder", "backbone", "vae", "audio_vae")
_UNET_ARCHS = {"sd15", "sdxl"}
# Anima switches by reloading the whole model, so the candidate is read while
# the outgoing components are still resident. Past this size that peak is what
# decides whether the switch survives, and a candidate this large is not an
# Anima companion anyway -- its text encoder and VAE are single-digit GiB.
# Fail-closed rather than discover the ceiling by hitting it.
_ANIMA_SWITCH_MAX_BYTES = 32 * 1024 ** 3
_COMPONENT_DICTS = {
    "zimage": "zimage_components",
    "flux2": "flux2_components",
    "anima": "anima_components",
    "lens": "lens_components",
    "ideogram4": "ideogram4_components",
    "minit2i": "minit2i_components",
    "krea2": "krea2_components",
    "ltx2": "ltx2_components",
    "acestep": "acestep_components",
    "minimax_h3": "minimax_h3_components",
}


def _candidate_id(slot: str, identity: str) -> str:
    digest = hashlib.sha256(f"{slot}\0{identity}".encode("utf-8", "replace")).hexdigest()
    return f"{slot}:{digest[:24]}"


def _display(path: Optional[str], fallback: str) -> str:
    if not path:
        return fallback
    try:
        return display_name_for_path(path, strip_safetensors=True)
    except Exception:
        return os.path.basename(path) or fallback


def _kind(slot: str, arch: Optional[str]) -> str:
    if slot == "backbone":
        return "unet" if arch in _UNET_ARCHS else "transformer"
    return slot


def _component_dict(manager: Any, arch: Optional[str]) -> Optional[Dict[str, Any]]:
    attr = _COMPONENT_DICTS.get(arch or "")
    value = getattr(manager, attr, None) if attr else None
    return value if isinstance(value, dict) else None


def _component_object(manager: Any, arch: Optional[str], slot: str) -> Any:
    if slot == "vision_encoder":
        return getattr(manager, "vision_encoder", None)
    components = _component_dict(manager, arch)
    if components is not None:
        key = "dit" if arch == "acestep" and slot == "backbone" else (
            "transformer" if slot == "backbone" else slot
        )
        return components.get(key)
    pipeline = getattr(manager, "txt2img_pipeline", None)
    if pipeline is None:
        return None
    key = "unet" if slot == "backbone" else slot
    return getattr(pipeline, key, None)


def _configured_path(
    manager: Any,
    arch: Optional[str],
    slot: str,
    source: Optional[str],
) -> Optional[str]:
    if slot == "vision_encoder":
        return getattr(manager, "_vision_encoder_path", None)
    components = _component_dict(manager, arch) or {}
    direct_keys = {
        "text_encoder": "text_encoder_path",
        "backbone": "dit_path",
        "vae": "vae_path",
        "audio_vae": "audio_vae_path",
    }
    direct = components.get(direct_keys.get(slot, ""))
    if isinstance(direct, str) and direct:
        return direct
    paths = components.get("paths")
    if isinstance(paths, dict):
        path_key = "dit" if slot == "backbone" else slot
        path = paths.get(path_key)
        if isinstance(path, str) and path:
            return path
    base_dir = components.get("base_dir")
    if isinstance(base_dir, str) and slot == "text_encoder":
        return os.path.join(base_dir, "text_encoder")
    if isinstance(source, str) and os.path.isdir(source):
        folder = "transformer" if slot == "backbone" else slot
        path = os.path.join(source, folder)
        if os.path.exists(path):
            return path
    return source if slot == "backbone" else None


def _standard_pipeline_hint(
    manager: Any,
    arch: Optional[str],
    slot: str,
    source: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Identity recorded by the SD/SDXL single-file loader."""
    if arch not in _UNET_ARCHS or slot not in ("text_encoder", "vae"):
        return None
    pipeline = getattr(manager, "txt2img_pipeline", None)
    if pipeline is None:
        return None

    if slot == "vae":
        marker = getattr(pipeline, "_sushi_vae_source", None)
        if marker == "embedded (checkpoint)" and source:
            return {
                "identity": source,
                "display_name": _display(source, "Embedded checkpoint VAE"),
                "origin": "embedded_checkpoint",
                "path_display": _display(source, ""),
                "size_path": source,
            }
        if isinstance(marker, str) and marker:
            return {
                "identity": f"architecture-default:{marker}",
                "display_name": marker,
                "origin": "architecture_default",
                "path_display": _display(marker, marker) if os.path.exists(marker) else marker,
                "size_path": marker if os.path.exists(marker) else None,
            }
        return None

    embedded = getattr(pipeline, "_sushi_te_embedded", None)
    if embedded is True and source:
        return {
            "identity": source,
            "display_name": _display(source, "Embedded checkpoint text encoder"),
            "origin": "embedded_checkpoint",
            "path_display": _display(source, ""),
            "size_path": source,
        }
    if embedded is False:
        architecture = getattr(pipeline, "_sushi_arch", None)
        te_type = architecture.get("te_type") if isinstance(architecture, dict) else None
        if isinstance(te_type, str) and te_type:
            return {
                "identity": f"architecture-default:{te_type}",
                "display_name": te_type,
                "origin": "architecture_default",
                "path_display": te_type,
                "size_path": None,
            }
        return None

    info = getattr(manager, "current_model_info", None) or {}
    if info.get("source_type") == "safetensors" and source:
        return {
            "identity": source,
            "display_name": _display(source, "Embedded checkpoint text encoder"),
            "origin": "embedded_checkpoint",
            "path_display": _display(source, ""),
            "size_path": source,
        }
    return None


def _current_origin(
    slot: str,
    source: Optional[str],
    path: Optional[str],
    resident: bool,
    components: Dict[str, Any],
) -> str:
    if slot == "vision_encoder":
        return "selected_external" if path else "unused"
    if not resident and not path:
        return "unused"
    declared = components.get(f"{slot}_origin")
    if declared in {
        "embedded_checkpoint", "model_tree", "architecture_default",
        "selected_external", "unused", "unavailable",
    }:
        return declared
    if slot == "vae" and components.get("vae_source") == "embedded (checkpoint)":
        return "embedded_checkpoint"
    if source and path and os.path.isdir(source):
        try:
            if os.path.commonpath((os.path.abspath(source), os.path.abspath(path))) == os.path.abspath(source):
                return "model_tree"
        except (OSError, ValueError):
            pass
    if path and source and os.path.normcase(path) != os.path.normcase(source):
        return "architecture_default"
    if slot == "backbone" and source:
        return "embedded_checkpoint" if os.path.isfile(source) else "model_tree"
    # A resident TE/VAE without loader-reported identity is not proof that it
    # came from a single-file checkpoint. Keep provenance explicitly unknown.
    return "unavailable"


def _effective_component(manager: Any, arch: Optional[str], slot: str) -> Dict[str, Any]:
    info = getattr(manager, "current_model_info", None) or {}
    source = info.get("source") if isinstance(info.get("source"), str) else None
    obj = _component_object(manager, arch, slot)
    components = _component_dict(manager, arch) or {}
    path = _configured_path(manager, arch, slot, source)
    resident = obj is not None
    hint = _standard_pipeline_hint(manager, arch, slot, source) if path is None else None
    origin = hint["origin"] if hint is not None else _current_origin(
        slot, source, path, resident, components
    )
    fallback = (
        "Not used" if origin == "unused" else
        "Loaded component (source unavailable)" if origin == "unavailable" else
        f"Architecture {slot.replace('_', ' ')}"
    )
    identity = path or (hint["identity"] if hint is not None else f"{origin}:{arch}:{slot}:{source or ''}")
    embedded = origin == "embedded_checkpoint"
    size_path = hint.get("size_path") if hint is not None else (source if embedded and not path else path)
    return {
        "candidate_id": _candidate_id(slot, identity),
        "slot": slot,
        "kind": _kind(slot, arch),
        "display_name": hint["display_name"] if hint is not None else _display(path, fallback),
        "origin": origin,
        "residency": "resident" if resident else ("configured" if path else "unloaded"),
        "embedded": embedded,
        "path_display": hint.get("path_display") if hint is not None else (_display(path, "") if path else None),
        "container_size_bytes": _path_size(size_path),
    }


def _runtime_override(manager: Any, arch: Optional[str], slot: str) -> Optional[Dict[str, Any]]:
    attr = {"text_encoder": "_override_te_path", "vae": "_override_vae_path"}.get(slot)
    path = getattr(manager, attr, None) if attr else None
    if not isinstance(path, str) or not path:
        return None
    return {
        "candidate_id": _candidate_id(slot, f"runtime:{path}"),
        "slot": slot,
        "kind": _kind(slot, arch),
        "display_name": _display(path, "Generation override"),
        "origin": "selected_external",
        "residency": "resident",
        "embedded": False,
        "path_display": _display(path, ""),
        "container_size_bytes": _path_size(path),
    }


def _path_size(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if not name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                        continue
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            return total or None
    except OSError:
        pass
    return None


def _public_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in candidate.items() if not key.startswith("_")}


def _candidate(
    slot: str,
    arch: Optional[str],
    identity: str,
    name: str,
    *,
    origin: str,
    compatibility: str,
    compatibility_reason: Optional[str],
    switchable: bool,
    switch_reason: Optional[str],
    is_current: bool = False,
    path: Optional[str] = None,
    size_bytes: Optional[int] = None,
    load_strategy: str = "unsupported",
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "candidate_id": _candidate_id(slot, identity),
        "slot": slot,
        "kind": _kind(slot, arch),
        "display_name": name,
        "origin": origin,
        "path_display": _display(path, "") if path else None,
        "container_size_bytes": size_bytes if size_bytes is not None else _path_size(path),
        "estimated_component_size_bytes": size_bytes if size_bytes is not None else _path_size(path),
        "compatibility": compatibility,
        "compatibility_reason": compatibility_reason,
        "switchable": switchable,
        "switch_reason": switch_reason,
        "is_current": is_current,
        "load_strategy": load_strategy,
        "variant": variant,
        "_path": path,
    }


def _current_candidate(current: Dict[str, Any], arch: Optional[str]) -> Dict[str, Any]:
    candidate = _candidate(
        current["slot"], arch, current["candidate_id"], current["display_name"],
        origin=current["origin"], compatibility="compatible",
        compatibility_reason="Currently configured component", switchable=False,
        switch_reason="Current selection", is_current=True,
        size_bytes=current.get("container_size_bytes"),
        load_strategy="none",
    )
    candidate["candidate_id"] = current["candidate_id"]
    candidate["path_display"] = current.get("path_display")
    return candidate


def _expected(arch: Optional[str], field: str) -> Optional[int]:
    spec = _WIRING_BY_ARCH.get(arch or "")
    return getattr(spec, field, None) if spec is not None else None


def _standalone_compat(slot: str, arch: Optional[str], entry: Dict[str, Any]) -> tuple[str, str]:
    if arch == "anima":
        reason = str(entry.get("anima_compatibility_reason") or "Anima wiring dimensions were not verified.")
        return ("compatible", reason) if entry.get("anima_compatible") is True else ("incompatible", reason)
    candidate_arch = entry.get("arch") or entry.get("architecture")
    if candidate_arch not in (None, "unknown", arch):
        return "incompatible", f"Candidate architecture is {candidate_arch}, loaded architecture is {arch}."
    if slot == "text_encoder":
        expected, observed = _expected(arch, "te_out_dim"), entry.get("out_dim")
        label = "text-encoder output dimension"
    else:
        expected, observed = _expected(arch, "latent_channels"), entry.get("latent_channels")
        label = "VAE latent channels"
    if expected is None or observed is None:
        return "unknown", f"{label} was not observed; fail-closed."
    if int(expected) != int(observed):
        return "incompatible", f"{label} {observed} does not match required {expected}."
    if candidate_arch in (None, "unknown"):
        return "unknown", "Dimensions match, but architecture provenance is unknown; fail-closed."
    return "compatible", "Architecture and observed wiring dimensions match."


def build_catalog(
    manager: Any,
    *,
    models: Iterable[Dict[str, Any]] = (),
    text_encoders: Iterable[Dict[str, Any]] = (),
    vision_encoders: Iterable[Dict[str, Any]] = (),
    vaes: Iterable[Dict[str, Any]] = (),
    h3_text_encoders: Iterable[Dict[str, Any]] = (),
) -> Dict[str, List[Dict[str, Any]]]:
    info = getattr(manager, "current_model_info", None) or {}
    arch = str(info.get("type") or "").lower() or None
    current = {slot: _effective_component(manager, arch, slot) for slot in SLOTS}
    catalog: Dict[str, List[Dict[str, Any]]] = {
        slot: [_current_candidate(current[slot], arch)] for slot in SLOTS
    }

    if current["vision_encoder"]["origin"] != "unused":
        clear = _candidate(
            "vision_encoder", arch, "none", "None", origin="unused",
            compatibility="compatible", compatibility_reason="Vision conditioning is optional.",
            switchable=arch in _UNET_ARCHS, switch_reason=None if arch in _UNET_ARCHS else "Vision encoder is supported only by SD1.5/SDXL.",
            path=None, load_strategy="none",
        )
        catalog["vision_encoder"].append(clear)
    for entry in vision_encoders:
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        verified = entry.get("compatibility_verified") is True
        compatible = arch in _UNET_ARCHS and verified
        compatibility_reason = str(
            entry.get("compatibility_reason")
            or "Vision encoder dimensions were not verified; fail-closed."
        )
        catalog["vision_encoder"].append(_candidate(
            "vision_encoder", arch, path, str(entry.get("name") or _display(path, "Vision Encoder")),
            origin="selected_external",
            compatibility="compatible" if compatible else ("incompatible" if arch not in _UNET_ARCHS else "unknown"),
            compatibility_reason=(compatibility_reason if arch in _UNET_ARCHS else "Loaded architecture has no vision-encoder adapter."),
            switchable=compatible,
            switch_reason=(
                None if compatible else
                "Vision encoder is supported only by SD1.5/SDXL."
                if arch not in _UNET_ARCHS else compatibility_reason
            ),
            path=path, size_bytes=_gb_bytes(entry.get("size_gb")), load_strategy="standalone",
        ))

    disabled_reason = "No unload-first persistent adapter is implemented for this slot."
    for slot, entries in (("text_encoder", text_encoders), ("vae", vaes)):
        for entry in entries:
            path = entry.get("path")
            if not isinstance(path, str) or not path:
                continue
            compatibility, reason = _standalone_compat(slot, arch, entry)
            size_bytes = _gb_bytes(entry.get("size_gb")) or _path_size(path)
            anima_switchable = (
                arch == "anima"
                and compatibility == "compatible"
                and size_bytes is not None
                and size_bytes <= _ANIMA_SWITCH_MAX_BYTES
            )
            size_reason = (
                f"Candidate exceeds the Anima unload-first limit of {_ANIMA_SWITCH_MAX_BYTES / 1024 ** 3:.0f} GiB."
                if size_bytes is not None and size_bytes > _ANIMA_SWITCH_MAX_BYTES else
                "Candidate size is unknown; fail-closed."
                if arch == "anima" and size_bytes is None else None
            )
            if anima_switchable:
                switch_reason = None
            elif size_reason:
                switch_reason = size_reason
            elif arch == "anima":
                # Anima does have an adapter, so this candidate is refused on
                # its own merits -- report those. The UI shows switch_reason in
                # preference to compatibility_reason, so the generic line here
                # would replace the real reason with a false one.
                switch_reason = reason
            else:
                switch_reason = disabled_reason
            catalog[slot].append(_candidate(
                slot, arch, path, str(entry.get("name") or _display(path, slot)),
                origin="selected_external", compatibility=compatibility,
                compatibility_reason=reason, switchable=anima_switchable,
                switch_reason=switch_reason,
                path=path, size_bytes=size_bytes,
                load_strategy="architecture_resolved" if anima_switchable else "standalone",
            ))

    if arch == "minimax_h3":
        # Generic TE scans cannot prove the H3 mmap/quantization contract. Only
        # the H3 loader's header-only inspector can enable one.
        catalog["text_encoder"] = [catalog["text_encoder"][0]]
        for entry in h3_text_encoders:
            path = entry.get("path")
            if not isinstance(path, str) or not path:
                continue
            candidate_id = _candidate_id("text_encoder", path)
            if candidate_id == current["text_encoder"]["candidate_id"]:
                # The current row reports the LOADED pairing, so a measurement is
                # attached only when it was taken on the projection in use.
                loaded = os.path.basename(str(
                    ((_component_dict(manager, arch) or {}).get("te_projection") or {}).get("path")
                    or "")) or None
                agreement = entry.get("agreement")
                row = catalog["text_encoder"][0]
                row["variant"] = entry.get("variant")
                row["requires_projection"] = bool(entry.get("requires_projection"))
                row["projection"] = loaded
                row["projection_candidates"] = entry.get("projection_candidates") or []
                row["agreement"] = (
                    agreement if isinstance(agreement, dict) and loaded is not None
                    and str(agreement.get("projection") or "").lower() == loaded.lower()
                    else None)
                continue
            compatible = entry.get("compatible") is True
            reason = str(entry.get("reason") or "H3 loader compatibility is unknown.")
            switch_reason = None if compatible else "The H3-specific loader rejected this candidate."
            projection_candidates = entry.get("projection_candidates") or []
            usable_projections = [c for c in projection_candidates if c.get("usable")]
            # A converted encoder is switchable only together with the trained
            # projection its hidden state is valid through. With several usable
            # ones the switch stays open but the caller must name which, since
            # auto-resolution refuses; with none it is offering a refusal.
            if compatible and entry.get("requires_projection") and not entry.get("projection"):
                if usable_projections:
                    switch_reason = (
                        f"{len(usable_projections)} projections declare this encoder's width; "
                        f"send projection_path naming one.")
                else:
                    compatible = False
                    reason = switch_reason = str(
                        entry.get("projection_reason")
                        or "This encoder needs a trained projection and none resolves in "
                           "clip_projections/.")
            candidate = _candidate(
                "text_encoder", arch, path,
                str(entry.get("name") or _display(path, "H3 Text Encoder")),
                origin="selected_external",
                compatibility="compatible" if compatible else "incompatible",
                compatibility_reason=reason,
                switchable=compatible,
                switch_reason=switch_reason,
                path=path,
                size_bytes=entry.get("size_bytes"),
                load_strategy="architecture_resolved" if compatible else "unsupported",
                variant=entry.get("variant"),
            )
            candidate["requires_projection"] = bool(entry.get("requires_projection"))
            candidate["projection"] = (
                os.path.basename(str(entry["projection"])) if entry.get("projection") else None)
            candidate["projection_candidates"] = projection_candidates
            candidate["agreement"] = entry.get("agreement")
            catalog["text_encoder"].append(candidate)

    for entry in models:
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            continue
        candidate_arch = entry.get("architecture")
        observed = ((entry.get("observed_components") or entry.get("components") or {}).get("backbone") or {})
        kind_present = observed.get("kind")
        if candidate_arch == arch and kind_present:
            compatibility, reason = "compatible", "Architecture and backbone kind match."
        elif candidate_arch != arch:
            compatibility, reason = "incompatible", f"Candidate architecture is {candidate_arch or 'unknown'}."
        else:
            compatibility, reason = "unknown", "Backbone presence was not observed; fail-closed."
        catalog["backbone"].append(_candidate(
            "backbone", arch, path, str(entry.get("name") or _display(path, "Model")),
            origin="embedded_checkpoint" if os.path.isfile(path) else "model_tree",
            compatibility=compatibility, compatibility_reason=reason,
            switchable=False,
            switch_reason=("MiniMax-H3 DiT reload constructs the replacement before release and is unsafe for this API." if arch == "minimax_h3" else "Full checkpoints must be loaded through the model loader, not as standalone components."),
            path=path, size_bytes=_gb_bytes(entry.get("size_gb")),
            load_strategy="unsupported",
        ))

    # candidate_id is the option's identity in the UI: the value the <select>
    # submits and the key find_candidate resolves. The scans legitimately turn
    # up the file that is already loaded -- more so since they descend into the
    # text_encoders/ and vae/ subdirectories a model tree keeps its components
    # in -- and that row's id equals the current row's, since both hash the
    # same path. Two rows sharing one id makes the second unselectable (the
    # change handler compares against the current id and sees no change) and,
    # if it is reached another way, resolves to the current selection, which
    # refuses the switch. The first row wins: it is the current one, which the
    # UI positions and labels as such.
    for slot, entries in catalog.items():
        if len(entries) < 2:
            continue
        seen = set()
        unique = []
        for entry in entries:
            candidate_id = entry.get("candidate_id")
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            unique.append(entry)
        catalog[slot] = unique
    return catalog


def _gb_bytes(value: Any) -> Optional[int]:
    try:
        return int(float(value) * 1024 ** 3) if value is not None else None
    except (TypeError, ValueError):
        return None


def build_response(manager: Any, catalog: Dict[str, List[Dict[str, Any]]], operation: Any = None) -> Dict[str, Any]:
    info = getattr(manager, "current_model_info", None)
    arch = str((info or {}).get("type") or "").lower() or None
    slots = []
    for slot in SLOTS:
        current = _effective_component(manager, arch, slot) if info else None
        slot_candidates = catalog.get(slot, [])
        switchable = any(c.get("switchable") for c in slot_candidates)
        if arch == "minimax_h3" and slot in ("backbone", "vae", "audio_vae"):
            reason = "MiniMax-H3 components are disabled: the 50 GB-class mapping and unload order require dedicated hardware validation."
        elif not switchable:
            reason = "No verified unload-first adapter is available."
        else:
            reason = None
        if slot == "audio_vae":
            # Only the architectures that have one.
            visible = arch in ("ltx2", "minimax_h3")
        elif slot == "vision_encoder":
            # Reference conditioning is a UNet-only feature. Elsewhere the slot
            # is a dropdown whose every entry is incompatible and which cannot
            # even be set back to None, so it is noise rather than information.
            visible = arch in _UNET_ARCHS
        else:
            visible = True
        slots.append({
            "slot": slot,
            "visible": visible,
            "current": current,
            "runtime_override": _runtime_override(manager, arch, slot) if info else None,
            "switchable": switchable,
            "reason": reason,
            "candidates": [_public_candidate(c) for c in slot_candidates],
        })
    return {
        "loaded": info is not None,
        "model_revision": int(getattr(manager, "model_revision", 0)),
        "component_revision": int(getattr(manager, "component_revision", 0)),
        "health": getattr(manager, "component_health", "unloaded"),
        "architecture": arch,
        "operation": operation,
        "slots": slots,
    }


def find_candidate(catalog: Dict[str, List[Dict[str, Any]]], slot: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    return next((item for item in catalog.get(slot, []) if item.get("candidate_id") == candidate_id), None)
