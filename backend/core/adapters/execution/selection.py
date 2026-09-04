"""Which adapter execution backend a process uses. Explicit, and off by default.

The counterpart of ``core/attention/config.py``, which absorbs an unknown
backend string into ``native`` because a downgraded attention kernel computes
the same function. An adapter backend does not, so an unrecognised name is
REFUSED here rather than absorbed, and nothing is ever auto-selected.
``SUSHI_ADAPTER_BACKEND`` is the only entry point and there is deliberately no
API parameter; ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` phase 4 says why.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

from .dispatch import (BACKEND_UNAVAILABLE_CODE, active_backend_name,
                       is_latched, latched_reason, set_active_backend)
from .registry import BACKENDS, REFERENCE, AdapterBackend

#: Environment variable naming the backend for this process. Unset or empty
#: means the reference path.
BACKEND_ENV_VAR = "SUSHI_ADAPTER_BACKEND"


def known_adapter_backends() -> Tuple[str, ...]:
    """Every backend name this build recognises, sorted. Derived from the
    registry, so a registered backend is selectable the moment it exists."""
    return tuple(sorted(BACKENDS))


def validate_adapter_backend(name: Optional[str], *,
                             default: Optional[str] = None,
                             param: str = "adapter_backend") -> Optional[str]:
    """The recognised backend name, or ``ValueError``.

    ``None``/empty -> ``default`` (the "not set" tier; this module never invents
    one). Unlike the attention resolver this does not fall back to the built-in
    on an unknown string: a caller that asked for a fused execution path and
    silently got the reference one has been told nothing.
    """
    if name is None:
        return default
    if not isinstance(name, str):
        raise ValueError(
            f"{param} must be a string, one of {', '.join(known_adapter_backends())}; "
            f"got {name!r}")
    key = name.strip().lower()
    if not key:
        return default
    if key not in BACKENDS:
        raise ValueError(
            f"{param} must be one of {', '.join(known_adapter_backends())}; "
            f"got {name!r}")
    return key


def backend_refusal(message: str) -> Exception:
    """An ``AdapterIncompatible`` carrying ``lora_backend_unavailable``.

    Imported inside the function on purpose: ``session`` imports ``layers``,
    which imports this package, so a module-scope import would be a cycle.
    """
    from ..session import AdapterIncompatible

    return AdapterIncompatible(message, code=BACKEND_UNAVAILABLE_CODE)


def selected_adapter_backend() -> str:
    """The backend in force, ``"reference"`` when nothing is selected."""
    return active_backend_name()


def select_adapter_backend(name: Optional[str], *,
                           warn: Optional[Callable[[str, str], None]] = None,
                           log: Optional[Callable[[str], None]] = None,
                           strict: bool = True) -> str:
    """Select ``name`` for this process and return what is now in force.

    ``strict`` decides how an unusable selection is reported: raised as an
    ``AdapterIncompatible`` (a run must not start believing it got a backend it
    did not), or warned with the same code and left on the reference path.
    Unusable means unrecognised, unavailable in this process, or already
    latched off.

    Selection alone admits nothing: every region still has to pass
    ``probe.probe_region`` before ``dispatch`` will run the backend for it.
    """
    try:
        key = validate_adapter_backend(name, default=REFERENCE)
    except ValueError as error:
        return _unusable(name, str(error), warn, log, strict)

    if key == REFERENCE:
        set_active_backend(None, warn=warn, log=log)
        return REFERENCE

    backend: AdapterBackend = BACKENDS[key]
    if is_latched(key):
        return _unusable(
            key, f"adapter backend '{key}' is latched off for this process: "
                 f"{latched_reason(key)}", warn, log, strict)
    unavailable = backend.availability()
    if unavailable:
        return _unusable(
            key, f"adapter backend '{key}' cannot run in this process: "
                 f"{unavailable}", warn, log, strict)

    set_active_backend(backend, warn=warn, log=log)
    (log or print)(f"[Adapter] execution backend '{key}' selected; each region is "
                   f"admitted only after it passes the fp32 oracle probe")
    return key


def configured_adapter_backend() -> Optional[str]:
    """The backend named by the environment, or ``None`` if unset."""
    raw = os.environ.get(BACKEND_ENV_VAR, "").strip()
    return raw or None


def apply_configured_backend(*,
                             warn: Optional[Callable[[str, str], None]] = None,
                             log: Optional[Callable[[str], None]] = None,
                             strict: bool = True) -> str:
    """Apply ``SUSHI_ADAPTER_BACKEND`` if it is set. No-op otherwise."""
    configured = configured_adapter_backend()
    if configured is None:
        return selected_adapter_backend()
    return select_adapter_backend(configured, warn=warn, log=log, strict=strict)


def _unusable(name, reason: str, warn, log, strict: bool) -> str:
    message = (reason if reason.startswith("adapter backend")
               else f"adapter backend {name!r} cannot be used: {reason}")
    (log or print)(f"[Adapter] ERROR: {message}")
    if strict:
        raise backend_refusal(message)
    if warn is not None:
        try:
            warn(message, BACKEND_UNAVAILABLE_CODE)
        except Exception as error:
            (log or print)(f"[Adapter] warning channel failed ({type(error).__name__})")
    return selected_adapter_backend()


__all__ = [
    "BACKEND_ENV_VAR",
    "apply_configured_backend",
    "backend_refusal",
    "configured_adapter_backend",
    "known_adapter_backends",
    "select_adapter_backend",
    "selected_adapter_backend",
    "validate_adapter_backend",
]
