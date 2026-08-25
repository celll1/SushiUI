"""Structured notices from the training subprocess to the product surface.

The trainer runs in a child process whose only channel to the backend is
stdout, and the backend's ``log_callback`` prints that stdout to its own
console. A notice worth acting on -- a setting that was overridden, ignored, or
is not implementable on the chosen path -- therefore never reached the user.

A call through this module writes the ordinary human line AND a
sentinel-prefixed JSON line, so ``TrainingProcess`` can lift the notice off the
stream, the backend can broadcast it (``training_log``) and persist it on the
run row, and the console output stays exactly what it was.

Only calls made through this module reach the user. Ordinary stdout stays on
the console: an unstructured line has no stable identity to dedup or persist
by, and the WebSocket it would land on is a single global broadcast shared with
image-generation previews.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Chosen to be implausible in ordinary trainer output and to survive a
# .strip() at both ends.
TRAINING_EVENT_SENTINEL = "##SUSHI_TRAINING_EVENT##"

TRAINING_EVENT_LEVELS = ("info", "warning", "error")

# One notice cannot be allowed to be a payload. Longest shipped warning today is
# ~1.2 KB (the fused gradient-accumulation one).
MAX_EVENT_MESSAGE_CHARS = 2000
MAX_EVENT_CODE_CHARS = 64

# Per-run cap on what survives on the run row. Notices are "say once" by
# construction, so this is a guard against a pathological emitter, not a
# working limit.
MAX_PERSISTED_WARNINGS_PER_RUN = 50

_CONSOLE_TAG = {"info": "", "warning": "WARNING: ", "error": "ERROR: "}


def _clamp(text: Any, limit: int) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def emit_training_event(
    level: str,
    message: str,
    code: Optional[str] = None,
    prefix: str = "",
    console: bool = True,
) -> Dict[str, Any]:
    """Print the human line and the machine line. Returns the event emitted.

    ``console=False`` is for callers that already printed their own text and
    only want the notice lifted onto the channel.
    """
    if level not in TRAINING_EVENT_LEVELS:
        level = "info"
    message = _clamp(message, MAX_EVENT_MESSAGE_CHARS)
    code = _clamp(code, MAX_EVENT_CODE_CHARS) if code else None

    if console:
        head = f"{prefix} " if prefix else ""
        print(f"{head}{_CONSOLE_TAG[level]}{message}", flush=True)

    event = {"level": level, "code": code, "message": message}
    # json.dumps escapes newlines, so this is always exactly one line.
    print(f"{TRAINING_EVENT_SENTINEL} {json.dumps(event, ensure_ascii=False)}", flush=True)
    return event


def emit_training_warning(
    message: str, code: Optional[str] = None, prefix: str = "", console: bool = True
) -> Dict[str, Any]:
    return emit_training_event("warning", message, code=code, prefix=prefix, console=console)


def split_training_event(line: str) -> tuple:
    """Split a stdout line into (leading text, event). Event is None if absent.

    Parent-side. The sentinel is located anywhere in the line, not just at its
    start: ``StreamReader.readline`` splits on ``\\n`` only, and the child's
    stderr is merged into the same pipe, so a tqdm bar's carriage-return-only
    write sits unterminated in the buffer and is delivered PREPENDED to the next
    newline-terminated line — including ours. Anchoring on the start lost the
    notice and dumped raw JSON on the console instead.

    False positives are held off by requiring the remainder to be a JSON object
    with a non-empty string ``message``, which is what the emitter writes and
    what a line merely quoting the sentinel will not be. Trailing junk after the
    JSON makes the line ordinary output, which is the safe direction.

    Never raises: a malformed sentinel line is treated as ordinary output rather
    than killing the log monitor.
    """
    if not line:
        return "", None
    idx = line.find(TRAINING_EVENT_SENTINEL)
    if idx < 0:
        return line, None
    prefix = line[:idx]
    try:
        payload = json.loads(line[idx + len(TRAINING_EVENT_SENTINEL):].strip())
    except (ValueError, TypeError):
        return line, None
    if not isinstance(payload, dict):
        return line, None
    message = payload.get("message")
    if not isinstance(message, str) or not message:
        return line, None
    level = payload.get("level")
    if level not in TRAINING_EVENT_LEVELS:
        level = "info"
    code = payload.get("code")
    code = _clamp(code, MAX_EVENT_CODE_CHARS) if isinstance(code, str) and code else None
    return prefix, {
        "level": level,
        "code": code,
        "message": _clamp(message, MAX_EVENT_MESSAGE_CHARS),
    }


def parse_training_event(line: str) -> Optional[Dict[str, Any]]:
    """The event carried by a stdout line, or None. See ``split_training_event``."""
    return split_training_event(line)[1]


def merge_run_warnings(
    existing: Optional[List[Dict[str, Any]]],
    event: Dict[str, Any],
    limit: int = MAX_PERSISTED_WARNINGS_PER_RUN,
) -> Optional[List[Dict[str, Any]]]:
    """Fold an event into a run's persisted warnings.

    Returns the new list, or None when the row does not need writing (an
    ``info`` event, a duplicate, or a run already at the cap). Dropping the
    oldest would lose the notice that explains what the run has been doing
    since step 0, so the cap keeps the earliest and refuses the newest -- but
    the last slot is reserved for a marker, so a truncated list says it is
    truncated instead of looking complete.
    """
    if event.get("level") not in ("warning", "error"):
        return None
    current = list(existing or [])
    entry = {
        "level": event["level"],
        "code": event.get("code"),
        "message": event["message"],
    }
    if entry in current:
        return None
    if len(current) >= limit:
        return None
    if len(current) == limit - 1:
        entry = {
            "level": "warning",
            "code": "warnings_truncated",
            "message": (
                f"This run emitted more notices than the {limit} kept on the run "
                f"record. The rest were broadcast live but are not retained here; "
                f"they are on the backend console."
            ),
        }
    current.append(entry)
    return current
