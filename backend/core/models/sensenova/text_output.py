"""Pure request/response contracts for SenseNova image-to-text generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple


IMG2TXT_TASKS = frozenset({"caption", "caption_tags", "tags", "custom"})
IMG2TXT_PROMPT_TEMPLATE_VERSION = 1

_TASK_INSTRUCTIONS = {
    "caption": (
        "Write a faithful natural-language caption for this image. Describe the "
        "visible subjects, actions, setting, composition, and important details. "
        "Do not add facts that cannot be seen. Return only the caption."
    ),
    "caption_tags": (
        "Describe this image and identify its useful visual tags. Return exactly "
        "one JSON object with keys \"caption\" (a natural-language string) and "
        "\"tags\" (an array of strings). Do not add facts that cannot be seen."
    ),
    "tags": (
        "Identify useful visual tags for this image. Return exactly one JSON "
        "object with key \"tags\" whose value is an array of strings."
    ),
}


def normalize_hint_tags(values: Iterable[Any]) -> list[str]:
    """Validate hint tags without rewriting their semantic content or order."""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("hint_tags must contain strings only")
        tag = value.strip()
        if not tag:
            raise ValueError("hint_tags must not contain empty strings")
        if len(tag) > 256:
            raise ValueError("each hint tag must be at most 256 characters")
        result.append(tag)
    if len(result) > 512:
        raise ValueError("hint_tags accepts at most 512 entries")
    return result


def build_effective_instruction(
    task: str,
    instruction: Optional[str],
    hint_tags: Iterable[str],
    template_version: int = IMG2TXT_PROMPT_TEMPLATE_VERSION,
) -> str:
    """Resolve a preset/custom instruction and append explicit input hints."""
    if task not in IMG2TXT_TASKS:
        raise ValueError(f"unsupported img2txt task: {task!r}")
    if template_version != IMG2TXT_PROMPT_TEMPLATE_VERSION:
        raise ValueError(
            f"unsupported prompt_template_version {template_version}; "
            f"supported version is {IMG2TXT_PROMPT_TEMPLATE_VERSION}"
        )

    supplied = (instruction or "").strip()
    if task == "custom" and not supplied:
        raise ValueError("instruction is required when task is 'custom'")
    resolved = supplied or _TASK_INSTRUCTIONS[task]
    if len(resolved) > 16_384:
        raise ValueError("instruction must be at most 16384 characters")

    hints = normalize_hint_tags(hint_tags)
    if hints:
        resolved += "\n\nInput hint tags (context only; verify them against the image): "
        resolved += json.dumps(hints, ensure_ascii=False)
    return resolved


def _unwrap_json_fence(text: str) -> str:
    stripped = text.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3:
        return stripped
    if lines[0].strip().lower() not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def parse_structured_output(
    task: str, raw_text: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse only syntactically and structurally valid preset output.

    Values and tag order are returned exactly as generated. Invalid structured
    output remains a successful inference and is represented by a warning.
    """
    if task == "caption":
        return {"caption": raw_text}, None
    if task == "custom":
        return None, None

    try:
        parsed = json.loads(_unwrap_json_fence(raw_text))
    except (TypeError, json.JSONDecodeError):
        return None, "The model did not return valid JSON; raw_text is preserved."
    if not isinstance(parsed, dict):
        return None, "The model returned JSON, but the top-level value is not an object."

    tags = parsed.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        return None, "The model returned JSON, but tags is not an array of strings."
    if task == "tags":
        if set(parsed) != {"tags"}:
            return None, "The model returned JSON with fields outside the requested tags schema."
        return {"tags": tags}, None

    caption = parsed.get("caption")
    if not isinstance(caption, str):
        return None, "The model returned JSON, but caption is not a string."
    if set(parsed) != {"caption", "tags"}:
        return None, "The model returned JSON with fields outside the requested caption/tags schema."
    return {"caption": caption, "tags": tags}, None
