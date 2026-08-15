"""MiniMax-H3 prompt templates and local-LLM rewriting."""

from __future__ import annotations

import difflib
import hashlib
import ipaddress
import json
import logging
import os
import re
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

from config.settings import settings


logger = logging.getLogger(__name__)
# Bumped from v3 for the revise-mode system prompt and cache-key material
# below (instruction/revise) -- a v3 cache entry was produced before either
# existed and must not be served for a revise-mode request.
GUIDE_VERSION = "minimax-h3-official-2026-08-16-v4"
BASE_MODES = {"t2va", "i2va", "fl2va", "l2va"}
ALL_MODES = BASE_MODES | {"ref2va"}
SECTION_NAMES = {
    "base": [
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    ],
    "ref": [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ],
}


class PromptAssistError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptAssistOptions:
    prompt: str
    mode: str
    duration_seconds: float
    references: List[Dict[str, str]]
    provider: str
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    context_length: int
    timeout_seconds: int
    # `instruction` is deliberately a SEPARATE field from `prompt`, not text
    # appended into it: an instruction folded into the prompt text reads to
    # the LLM as more content to describe ("make the drop harder" becomes a
    # phrase to narrate), not a directive to apply. See `revise` below and
    # `_system_prompt`'s REVISE MODE block, which is the actual fix for that
    # failure mode.
    instruction: str = ""
    # False (the default): `prompt` is freeform user intent and this call
    # behaves exactly as it always has -- expand it into a new, structured
    # MiniMax-H3 prompt, with `instruction` unused. True: `prompt` is
    # instead the CURRENT, already-structured MiniMax-H3 prompt, treated as
    # the base text to preserve, and `instruction` is the set of edits to
    # apply to it. See `_system_prompt`'s REVISE MODE block for the exact
    # wording and the reasoning behind it.
    revise: bool = False
    force_refresh: bool = False


def _normalise_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise PromptAssistError("Provider URL must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PromptAssistError("Provider URL cannot contain credentials, query, or fragment")
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise PromptAssistError("Prompt-assist providers are restricted to this computer")
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise PromptAssistError("The LLM did not return the requested JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PromptAssistError(f"The LLM returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
        raise PromptAssistError("The LLM JSON response is missing a string prompt field")
    return value


def _alignment_instruction(mode: str, duration: float, last_shot: int = 1) -> str:
    seconds = f"{duration:.2f}"
    if mode == "i2va":
        return (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
    if mode == "fl2va":
        return (
            "How the reference pictures align with the target video — Picture 1 "
            "(from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {last_shot}) aligns with the {seconds}-second mark "
            "of the target video."
        )
    if mode == "l2va":
        return (
            "How the reference pictures align with the target video — <Picture 1> "
            f"(from [Shot {last_shot}]) aligns with the {seconds}-second mark of the target video."
        )
    return ""


def build_template(prompt: str, mode: str, duration_seconds: float) -> str:
    """Build an editable scaffold without inventing visual or audio details."""
    mode = mode.lower()
    if mode not in ALL_MODES:
        raise PromptAssistError(f"Unsupported MiniMax-H3 mode: {mode}")
    source = prompt.strip()
    if mode == "ref2va":
        return (
            "subject_definitions:\n[Define only the supplied reference assets and subjects.]\n\n"
            "summary:\n[reference generation] " + source + "\n\n"
            "retention_analysis:\n[Describe how each supplied reference is retained.]\n\n"
            "detailed_description:\n[Shot 1] " + source + "\n\n"
            "overall_soundscape:\n[Describe only requested or clearly implied physical sounds.]\n\n"
            "non_diegetic_music:\n[Describe requested audience-only music, or N/A.]"
        )
    instruction = _alignment_instruction(mode, duration_seconds)
    body = (
        "integrated_multimodal_description: [Shot 1] " + source + "\n\n"
        "overall_soundscape: [Describe only requested or clearly implied physical sounds.]\n\n"
        "non_diegetic_music: [Describe requested audience-only music, or N/A.]"
    )
    return f"{instruction}\n\n{body}" if instruction else body


def normalize_prompt(prompt: str) -> str:
    normalized = prompt.strip().replace("\r\n", "\n")
    return re.sub(
        r"(?im)^(non_diegetic_music:\s*)(?:none\.?|no music\.?)\s*$",
        r"\1N/A",
        normalized,
    )


def validate_prompt(
    prompt: str,
    mode: str,
    duration_seconds: float,
    references: Optional[Iterable[Dict[str, str]]] = None,
) -> List[str]:
    warnings: List[str] = []
    family = "ref" if mode == "ref2va" else "base"
    positions: List[int] = []
    for section in SECTION_NAMES[family]:
        matches = list(re.finditer(rf"(?m)^{re.escape(section)}:\s*", prompt))
        if len(matches) != 1:
            warnings.append(f"Expected exactly one '{section}:' section")
        positions.append(matches[0].start() if matches else -1)
    present = [position for position in positions if position >= 0]
    if present != sorted(present):
        warnings.append("Prompt sections are not in the official order")
    for section in SECTION_NAMES[family][1:]:
        if not re.search(rf"\n\n{re.escape(section)}:\s*", prompt):
            warnings.append(f"Expected one blank line before '{section}:'")
    if "[" in prompt and re.search(r"\[(?:Define|Describe|reference generation)", prompt):
        warnings.append("The editable scaffold still contains placeholders")
    shots = [int(value) for value in re.findall(r"\[Shot\s+(\d+)\]", prompt)]
    unique_shots = list(dict.fromkeys(shots))
    if mode in {"i2va", "fl2va", "l2va"}:
        expected_instruction = _alignment_instruction(
            mode, duration_seconds, max(unique_shots, default=1)
        )
        actual_instruction = prompt.split("\n\n", 1)[0].strip()
        if actual_instruction != expected_instruction:
            warnings.append("The mode-specific reference alignment instruction is not exact")
        first_section = SECTION_NAMES[family][0]
        if not re.search(rf"\.\n\n{re.escape(first_section)}:", prompt):
            warnings.append("The alignment instruction must be followed by one blank line")
    if 1 not in unique_shots:
        warnings.append("The main description must contain [Shot 1]")
    if unique_shots and unique_shots != list(range(1, max(unique_shots) + 1)):
        warnings.append("Shot numbers must start at 1 and remain sequential")
    if re.search(r"\[Shot 1\]\s+At\s+\d{2}:\d{2}\.\d{3}", prompt):
        warnings.append("Shot 1 must not have a timestamp")
    times = re.findall(r"\[Shot\s+(\d+)\]\s+At\s+(\d{2}):(\d{2})\.(\d{3})", prompt)
    timed_shots = {int(shot) for shot, _, _, _ in times}
    for shot in unique_shots:
        if shot > 1 and shot not in timed_shots:
            warnings.append(f"Shot {shot} must start with an MM:SS.mmm timestamp")
    seconds_values = [int(mm) * 60 + int(ss) + int(ms) / 1000 for _, mm, ss, ms in times]
    if seconds_values != sorted(seconds_values) or len(seconds_values) != len(set(seconds_values)):
        warnings.append("Shot timestamps must be strictly increasing")
    if any(value >= duration_seconds for value in seconds_values):
        warnings.append("A shot timestamp is outside the target duration")
    if references is not None:
        allowed = {item.get("token", "") for item in references}
        used = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", prompt))
        unknown = sorted(used - allowed)
        if unknown:
            warnings.append("Unknown reference labels: " + ", ".join(unknown))
    if mode == "ref2va":
        subjects = [int(value) for value in re.findall(r"<Subject\s+(\d+)>", prompt)]
        unique_subjects = sorted(set(subjects))
        if unique_subjects and unique_subjects != list(range(1, max(unique_subjects) + 1)):
            warnings.append("Subject labels must start at 1 and remain sequential")
        definitions_match = re.search(
            r"(?s)^subject_definitions:\s*(.*?)\n\nsummary:", prompt
        )
        definitions = definitions_match.group(1) if definitions_match else ""
        undefined = [
            number for number in unique_subjects
            if f"<Subject {number}>" not in definitions
        ]
        if undefined:
            warnings.append(
                "Subject labels must be defined before use: "
                + ", ".join(f"<Subject {number}>" for number in undefined)
            )
    return warnings


def _revise_mode_block(instruction: str) -> str:
    """The REVISE MODE addendum appended to the system prompt when
    `PromptAssistOptions.revise` is True. Never called, and never appended,
    when it is False -- see `_system_prompt`'s own guard -- so the expand
    path's system prompt is untouched by this function's existence.

    Wording rationale (the actual reported failure this feature exists to
    fix): a naive "Instruction: <text>" line reads to a local LLM as one
    more piece of context describing the scene, so "make the drop harder"
    gets summarized back into prose about a harder drop instead of being
    used to edit the existing non_diegetic_music/overall_soundscape text.
    Three things are stated explicitly to head that off:
    (1) the supplied prompt is relabelled as the BASE TEXT, not intent to
    expand, so the model does not try to re-derive a whole new prompt from
    it; (2) the instruction is named as a directive to APPLY, with an
    explicit "never restate ... as new subject matter" contrast, because
    "apply this" alone still leaves room for a model to just paraphrase the
    instruction into the output; (3) the parts the instruction does not
    name must come through UNCHANGED, because an instruction-following LLM
    given permission to touch anything a novel edit implies will happily
    rewrite the whole passage "in the spirit of" the request instead of
    making the one change asked for.
    """
    return (
        "\n\nREVISE MODE: the prompt supplied below is not raw user intent to expand -- "
        "it is the CURRENT, already-structured MiniMax-H3 prompt, and it is the BASE TEXT "
        "to preserve. A separate revision instruction is supplied as part of the user "
        "message; APPLY it as a set of edits to the base text. Never restate, describe, or "
        "narrate the instruction as if it were new subject matter -- \"make the drop "
        "harder\" is a directive about the existing description, not a sentence to add to "
        "it. Anything in the base text the instruction does not address must reach your "
        "output unchanged, in its original section and its original position.\n\n"
        f"Revision instruction: {instruction.strip()}"
    )


def _user_message(options: "PromptAssistOptions") -> str:
    """The literal user-turn text sent to the LLM. Identical to
    `options.prompt` whenever `options.revise` is False -- this is the
    exact input H3 has always sent, kept byte-identical so a caller that
    never supplies an instruction sees no behaviour change at all. In
    revise mode the base text and the revision instruction are sent as two
    clearly labelled sections rather than concatenated into one paragraph,
    for the same reason `_revise_mode_block` states in the system prompt:
    blending the instruction into the prompt text lets the LLM read it as
    more content to describe instead of a directive to apply."""
    if not options.revise:
        return options.prompt
    return (
        f"Current prompt (base text to preserve):\n{options.prompt.strip()}\n\n"
        "Revision instruction (apply this as an edit; do not describe it): "
        f"{options.instruction.strip()}"
    )


def _summarize_diff(base: str, revised: str) -> str:
    """Unified line diff between a revise-mode base text and the model's
    revision. A claim of "only the parts the instruction named changed" is
    not machine-checkable, but a diff against the base is -- this is what
    lets a caller see, at a glance, whether a revise targeted the edit or
    quietly rewrote the whole piece. Shared by all three prompt-assist
    modules (imported from here, matching how they already share
    `PromptAssistCache`/`PromptAssistError`/`_extract_json` etc.)."""
    diff = difflib.unified_diff(
        base.splitlines(),
        revised.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(diff)


def _system_prompt(
    mode: str,
    duration: float,
    references: Iterable[Dict[str, str]],
    revise: bool = False,
    instruction: str = "",
) -> str:
    family = "full-reference" if mode == "ref2va" else "base"
    inventory = json.dumps(list(references), ensure_ascii=False)
    mode_rule = {
        "t2va": "Begin directly with the three base fields; there is no alignment instruction.",
        "i2va": _alignment_instruction("i2va", duration),
        "fl2va": _alignment_instruction("fl2va", duration),
        "l2va": _alignment_instruction("l2va", duration),
        "ref2va": "Use exactly the six full-reference sections in the documented order.",
    }[mode]
    revise_block = _revise_mode_block(instruction) if revise else ""
    return f"""You transform a user's intent into one MiniMax-H3 {family} video prompt.
Return exactly one JSON object: {{"prompt":"...","warnings":["..."]}}.
The JSON prompt value must be the complete formatted MiniMax-H3 prompt, not a summary or a plain sentence.

Fidelity rules:
- Preserve every user-stated identity, action, style, composition, name, number, constraint, dialogue line, lyric, and visible text.
- Convert Danbooru-style tags into natural English without silently omitting or contradicting them.
- Never invent dialogue, lyrics, quotations, visible text, reference-asset details, or speaker identity.
- Dialogue, lyrics, and visible text remain verbatim and untranslated; all other output is English.
- Use only Picture, Video, and Audio labels in this inventory: {inventory}.
- In full-reference mode you may define sequential Subject labels from the user's stated intent, but never invent unsupported visual details.
- If pixels or descriptions are unavailable, do not claim visual details that were not supplied.
- Shot 1 has no timestamp. Later shots use [Shot N] At MM:SS.mmm with strictly increasing times below {duration:.3f} seconds.
- The main description must explicitly begin its timeline with [Shot 1]. Separate every required section with one blank line.
- Put dialogue, singing, and shot-synchronised sound in the main description; ambience in overall_soundscape; audience-only music in non_diegetic_music.
- If there is no audience-only music, write exactly non_diegetic_music: N/A, never None.
- Output no Markdown or alternatives.
- Never collapse the required sections into a one-sentence prompt.

Mode requirement: {mode_rule}
Base output order: integrated_multimodal_description, overall_soundscape, non_diegetic_music.
Full-reference output order: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
For full-reference generation, make detailed_description explicit and useful, but do not pad it with invented facts.{revise_block}"""


class PromptAssistCache:
    def __init__(self, path: Path, max_entries: int):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS prompt_cache ("
            "cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at REAL NOT NULL, last_used_at REAL NOT NULL)"
        )
        return connection

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response_json FROM prompt_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if not row:
                return None
            connection.execute(
                "UPDATE prompt_cache SET last_used_at = unixepoch('now') WHERE cache_key = ?", (cache_key,)
            )
            connection.commit()
            return json.loads(row[0])

    def put(self, cache_key: str, response: Dict[str, Any]) -> None:
        with self._lock, closing(self._connect()) as connection:
            payload = json.dumps(response, ensure_ascii=False)
            connection.execute(
                "INSERT OR REPLACE INTO prompt_cache(cache_key, response_json, created_at, last_used_at) "
                "VALUES (?, ?, unixepoch('now'), unixepoch('now'))",
                (cache_key, payload),
            )
            connection.execute(
                "DELETE FROM prompt_cache WHERE cache_key IN (SELECT cache_key FROM prompt_cache "
                "ORDER BY last_used_at DESC LIMIT -1 OFFSET ?)",
                (self.max_entries,),
            )
            connection.commit()

    def clear(self) -> int:
        with self._lock, closing(self._connect()) as connection:
            count = connection.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()[0]
            connection.execute("DELETE FROM prompt_cache")
            connection.commit()
            return int(count)


class MiniMaxH3PromptAssistant:
    def __init__(self, max_cache_entries: int, cache_path: Optional[Path] = None) -> None:
        resolved_cache_path = cache_path or Path(settings.cache_dir) / "minimax_h3_prompt_assist.sqlite3"
        self.cache = PromptAssistCache(resolved_cache_path, max_cache_entries)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, provider: str, base_url: str, model: str) -> threading.Lock:
        key = f"{provider}:{base_url}:{model}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def list_models(self, provider: str, base_url: str, api_key: str = "") -> List[Dict[str, Any]]:
        base_url = _normalise_url(base_url)
        try:
            if provider == "lm_studio":
                response = requests.get(
                    f"{base_url}/api/v1/models", headers=_headers(api_key), timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return [
                    {
                        "id": item["key"],
                        "name": item.get("display_name") or item["key"],
                        "loaded": bool(item.get("loaded_instances")),
                        "size_bytes": item.get("size_bytes"),
                    }
                    for item in data.get("models", [])
                    if item.get("type") == "llm"
                ]
            if provider == "ollama":
                response = requests.get(f"{base_url}/api/tags", timeout=10)
                response.raise_for_status()
                data = response.json()
                return [
                    {
                        "id": item["name"],
                        "name": item.get("name") or item["model"],
                        "loaded": False,
                        "size_bytes": item.get("size"),
                    }
                    for item in data.get("models", [])
                ]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise PromptAssistError(f"Could not list {provider} models: {exc}") from exc
        raise PromptAssistError(f"Unsupported provider: {provider}")

    def _cache_key(self, options: PromptAssistOptions) -> str:
        material = {
            "guide": GUIDE_VERSION,
            "prompt": options.prompt.strip().replace("\r\n", "\n"),
            "mode": options.mode,
            "duration_seconds": round(options.duration_seconds, 3),
            "references": options.references,
            "provider": options.provider,
            "base_url": options.base_url,
            "model": options.model,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_output_tokens": options.max_output_tokens,
            "context_length": options.context_length,
            # Both required in the cache key material, not just GUIDE_VERSION:
            # a revise=True request with a given instruction must never be
            # served the expand-mode (or a different instruction's) cached
            # answer for the same base prompt/mode/duration.
            "instruction": options.instruction.strip().replace("\r\n", "\n"),
            "revise": options.revise,
        }
        encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def transform(self, options: PromptAssistOptions, api_key: str = "") -> Dict[str, Any]:
        if options.mode not in ALL_MODES:
            raise PromptAssistError(f"Unsupported MiniMax-H3 mode: {options.mode}")
        if not options.prompt.strip():
            raise PromptAssistError("Prompt cannot be empty")
        if options.revise and not options.instruction.strip():
            raise PromptAssistError("Revise mode requires a revision instruction")
        if options.duration_seconds <= 0:
            raise PromptAssistError("Duration must be greater than zero")
        if not options.model:
            raise PromptAssistError("Select a local LLM model first")
        base_url = _normalise_url(options.base_url)
        options = PromptAssistOptions(**{**options.__dict__, "base_url": base_url})
        cache_key = self._cache_key(options)
        if not options.force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return {**cached, "cached": True, "cache_key": cache_key}

        provider_lock = self._lock_for(options.provider, base_url, options.model)
        with provider_lock:
            if not options.force_refresh:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return {**cached, "cached": True, "cache_key": cache_key}
            system_prompt = _system_prompt(
                options.mode, options.duration_seconds, options.references,
                options.revise, options.instruction,
            )
            user_message = _user_message(options)
            if options.provider == "lm_studio":
                result, lifecycle_warnings = self._lm_studio(options, system_prompt, user_message, api_key)
            elif options.provider == "ollama":
                result, lifecycle_warnings = self._ollama(options, system_prompt, user_message)
            else:
                raise PromptAssistError(f"Unsupported provider: {options.provider}")
            parsed = _extract_json(result)
            prompt = normalize_prompt(parsed["prompt"])
            warnings = [
                str(item) for item in parsed.get("warnings", [])
                if str(item).strip().lower() not in {"", "none", "n/a"}
            ]
            structural_warnings = validate_prompt(
                prompt, options.mode, options.duration_seconds, options.references
            )
            warnings.extend(structural_warnings)
            warnings.extend(lifecycle_warnings)
            response = {
                "prompt": prompt,
                "warnings": list(dict.fromkeys(warnings)),
                "valid": not structural_warnings,
                "provider": options.provider,
                "model": options.model,
                "cached": False,
                "revise": options.revise,
                # None in expand mode -- there is no meaningful "base" to
                # diff a freeform-intent input against a structured output.
                "diff_summary": (
                    _summarize_diff(normalize_prompt(options.prompt), prompt)
                    if options.revise else None
                ),
            }
            if response["valid"]:
                self.cache.put(cache_key, response)
            return {**response, "cache_key": cache_key}

    def _lm_studio(
        self, options: PromptAssistOptions, system_prompt: str, user_message: str, api_key: str
    ) -> tuple[str, List[str]]:
        instance_id: Optional[str] = None
        warnings: List[str] = []
        try:
            load = requests.post(
                f"{options.base_url}/api/v1/models/load",
                headers=_headers(api_key),
                json={"model": options.model, "context_length": options.context_length},
                timeout=options.timeout_seconds,
            )
            load.raise_for_status()
            load_data = load.json()
            instance_id = load_data.get("instance_id") or load_data.get("model_instance_id")
            chat_payload = {
                "model": instance_id or options.model,
                "input": user_message,
                "system_prompt": system_prompt,
                "temperature": options.temperature,
                "top_p": options.top_p,
                "max_output_tokens": options.max_output_tokens,
                "context_length": options.context_length,
                "reasoning": "off",
                "stream": False,
                "store": False,
            }
            chat = requests.post(
                f"{options.base_url}/api/v1/chat",
                headers=_headers(api_key),
                json=chat_payload,
                timeout=options.timeout_seconds,
            )
            if chat.status_code == 400 and "reasoning" in chat.text.lower():
                chat_payload.pop("reasoning")
                chat = requests.post(
                    f"{options.base_url}/api/v1/chat",
                    headers=_headers(api_key),
                    json=chat_payload,
                    timeout=options.timeout_seconds,
                )
            chat.raise_for_status()
            messages = [
                item.get("content", "")
                for item in chat.json().get("output", [])
                if item.get("type") == "message"
            ]
            if not messages:
                raise PromptAssistError("LM Studio returned no message output")
            output_text = "\n".join(messages)
            try:
                first_prompt = _extract_json(output_text)["prompt"]
                structural_warnings = validate_prompt(
                    first_prompt, options.mode, options.duration_seconds, options.references
                )
            except PromptAssistError as exc:
                structural_warnings = [str(exc)]
            if structural_warnings:
                chat_payload["input"] = (
                    "Your previous answer failed validation. Correct it and return only the JSON object.\n"
                    "The JSON prompt string itself must contain every required section and the full rewrite.\n"
                    f"Validation errors: {json.dumps(structural_warnings, ensure_ascii=False)}\n"
                    f"Original user prompt: {user_message}\n"
                    f"Previous answer: {output_text}"
                )
                repair = requests.post(
                    f"{options.base_url}/api/v1/chat",
                    headers=_headers(api_key),
                    json=chat_payload,
                    timeout=options.timeout_seconds,
                )
                repair.raise_for_status()
                repaired_messages = [
                    item.get("content", "")
                    for item in repair.json().get("output", [])
                    if item.get("type") == "message"
                ]
                if repaired_messages:
                    output_text = "\n".join(repaired_messages)
            return output_text, warnings
        except requests.RequestException as exc:
            raise PromptAssistError(f"LM Studio request failed: {exc}") from exc
        finally:
            if instance_id:
                try:
                    unload = requests.post(
                        f"{options.base_url}/api/v1/models/unload",
                        headers=_headers(api_key),
                        json={"instance_id": instance_id},
                        timeout=30,
                    )
                    unload.raise_for_status()
                except requests.RequestException as exc:
                    message = f"LM Studio could not unload the prompt model: {exc}"
                    warnings.append(message)
                    logger.warning(message)

    def _ollama(
        self, options: PromptAssistOptions, system_prompt: str, user_message: str
    ) -> tuple[str, List[str]]:
        warnings: List[str] = []
        try:
            chat_payload = {
                "model": options.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "format": "json",
                "stream": False,
                "keep_alive": "1m",
                "options": {
                    "temperature": options.temperature,
                    "top_p": options.top_p,
                    "num_ctx": options.context_length,
                    "num_predict": options.max_output_tokens,
                },
            }
            response = requests.post(
                f"{options.base_url}/api/chat",
                json=chat_payload,
                timeout=options.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content")
            if not content:
                raise PromptAssistError("Ollama returned no message output")
            try:
                first_prompt = _extract_json(content)["prompt"]
                structural_warnings = validate_prompt(
                    first_prompt, options.mode, options.duration_seconds, options.references
                )
            except PromptAssistError as exc:
                structural_warnings = [str(exc)]
            if structural_warnings:
                chat_payload["messages"].append({"role": "assistant", "content": content})
                chat_payload["messages"].append({
                    "role": "user",
                    "content": (
                        "Correct the previous JSON. The prompt string itself must contain every required "
                        "section. Errors: " + json.dumps(structural_warnings, ensure_ascii=False)
                    ),
                })
                repair = requests.post(
                    f"{options.base_url}/api/chat",
                    json=chat_payload,
                    timeout=options.timeout_seconds,
                )
                repair.raise_for_status()
                content = repair.json().get("message", {}).get("content") or content
            return content, warnings
        except requests.RequestException as exc:
            raise PromptAssistError(f"Ollama request failed: {exc}") from exc
        finally:
            try:
                unload = requests.post(
                    f"{options.base_url}/api/generate",
                    json={"model": options.model, "keep_alive": 0},
                    timeout=30,
                )
                unload.raise_for_status()
            except requests.RequestException as exc:
                message = f"Ollama could not unload the prompt model: {exc}"
                warnings.append(message)
                logger.warning(message)
