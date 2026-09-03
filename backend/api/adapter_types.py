"""`loras[]` item parsing, and the `adapter_type` assertion.

One reader for both transports: the JSON routes hand it a list of objects, the
multipart routes hand it the JSON string of the same objects. `adapter_type` is
an assertion, checked at the route -- before the model is touched and before
the architecture's loader reads the file a second time. See
`param_defaults.LORA_ITEM_DEFAULTS` for what the field means, and
`LoRAManager.adapter_report()` for the detector all three paths share.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from api.error_handlers import ValidationError
from api.param_defaults import LORA_ITEM_DEFAULTS
from core.adapters.spec import FAMILY_NAMES

ADAPTER_TYPE_AUTO: str = LORA_ITEM_DEFAULTS["adapter_type"]

#: What a request may assert. The families come from the engine's own table, so
#: enabling a seventh pair does not need an edit here.
ADAPTER_TYPE_CHOICES = (ADAPTER_TYPE_AUTO,) + tuple(FAMILY_NAMES.values())

#: Refusal code (400 via `error_handlers.is_lora_refusal_code`).
ADAPTER_TYPE_MISMATCH_CODE = "lora_adapter_type_mismatch"

__all__ = [
    "ADAPTER_TYPE_AUTO",
    "ADAPTER_TYPE_CHOICES",
    "ADAPTER_TYPE_MISMATCH_CODE",
    "normalize_adapter_type",
    "parse_lora_items",
]


def normalize_adapter_type(value: Any) -> str:
    """The item's asserted family, or `"auto"` when it asserts nothing."""
    if value is None:
        return ADAPTER_TYPE_AUTO
    text = str(value).strip().lower()
    if not text:
        return ADAPTER_TYPE_AUTO
    if text not in ADAPTER_TYPE_CHOICES:
        raise ValidationError(
            f"Unknown adapter_type {value!r} on a loras[] item",
            detail=(f"adapter_type must be one of "
                    f"{', '.join(ADAPTER_TYPE_CHOICES)}"))
    return text


def _as_dict(item: Any, index: int) -> Dict[str, Any]:
    for attr in ("model_dump", "dict"):
        method = getattr(item, attr, None)
        if callable(method):
            item = method()
            break
    if not isinstance(item, dict):
        raise ValidationError(
            f"loras[{index}] is not an object",
            detail=f"got {type(item).__name__}")
    return dict(item)


def _assert_matches_file(item: Dict[str, Any], index: int) -> None:
    asserted = item.get("adapter_type", ADAPTER_TYPE_AUTO)
    if asserted == ADAPTER_TYPE_AUTO:
        return

    from core.extensions.lora_manager import lora_manager

    path = str(item.get("path") or "")
    report = lora_manager.adapter_report(path)
    if report is None:
        # Unresolvable or unreadable: the generation path owns that refusal
        # (`lora_not_found` / `lora_load_failed`), and reporting it as a type
        # mismatch here would name the wrong cause.
        return

    detected = report.get("adapter_type", "unknown")
    if detected == asserted:
        return

    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or path
    if detected == "unknown":
        detail = (f"'{name}': no metadata key and no tensor-key signature names "
                  f"this file's adapter algebra, so the assertion "
                  f"adapter_type={asserted!r} cannot be checked. Use "
                  f"adapter_type={ADAPTER_TYPE_AUTO!r} to apply it anyway.")
    else:
        detail = (f"'{name}' is a {detected} checkpoint, not {asserted}. "
                  f"adapter_type asserts what the file IS; it does not convert "
                  f"it. Use adapter_type={ADAPTER_TYPE_AUTO!r}, or select a "
                  f"{asserted} file.")
    raise ValidationError(
        f"loras[{index}] adapter_type does not match the file",
        detail=detail, code=ADAPTER_TYPE_MISMATCH_CODE)


async def parse_lora_items(raw: Any) -> List[Dict[str, Any]]:
    """Normalize a `loras` payload into the plain dicts every backend consumes.

    Accepts the multipart JSON string, a JSON list, or a list of Pydantic
    items. Refuses (400) an item whose `adapter_type` contradicts its file.

    Async because checking an assertion reads the file's safetensors header and
    walks the LoRA search directories, the same blocking work `GET /loras`
    already hands to a thread. An all-`auto` request -- every request that does
    not assert -- touches no disk and never leaves the loop.
    """
    items = _normalize_items(raw)
    if any(item["adapter_type"] != ADAPTER_TYPE_AUTO for item in items):
        await asyncio.to_thread(_assert_all_match_files, items)
    return items


def _assert_all_match_files(items: List[Dict[str, Any]]) -> None:
    for index, item in enumerate(items):
        _assert_matches_file(item, index)


def _normalize_items(raw: Any) -> List[Dict[str, Any]]:
    """The pure half: JSON, item shape, and the `adapter_type` vocabulary."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except ValueError as e:
            raise ValidationError("Invalid loras JSON", detail=str(e))
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError("loras must be an array",
                              detail=f"got {type(raw).__name__}")

    items: List[Dict[str, Any]] = []
    for index, entry in enumerate(raw):
        item = _as_dict(entry, index)
        item["adapter_type"] = normalize_adapter_type(item.get("adapter_type"))
        items.append(item)
    return items
