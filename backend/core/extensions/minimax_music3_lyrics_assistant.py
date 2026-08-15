"""MiniMax Music 3 lyrics assistant.

Motivating defect (docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Lyrics assistant"
and the "Generation parameter contract" table's `lyrics` row): the checkpoint's
own lyric normalizer, `_normalize_lyrics` in
`backend/core/models/minimax_music3/pipeline.py`, keeps only the LEADING tag
run on a line and silently DROPS the rest of that line:

    "[Verse] The morning light"    ->  "[start]\\n[verse]"
    "[Verse]\\nThe morning light"  ->  "[start]\\n[verse]\\nThe morning light"

There is no error anywhere for the first case; the words are simply gone.
The most natural way to type lyrics destroys them. This module exists to
fix that, in three modes the user picks explicitly (design doc,
"Lyrics assistant"):

- `format`  — deterministic, NO LLM CALL. Fixes only the layout (moves text
  off a leading-tag line, one tag per line, lowercases tag case, drops blank
  noise) and is REQUIRED to preserve the user's words exactly: the ordered
  sequence of non-tag word tokens before and after must be identical, and
  `format_lyrics` raises rather than returning a result that fails that
  check. See `format_lyrics`.
- `structure` — the LLM emits ONLY tags, one per line, no prose: the
  structural control surface for a purely instrumental piece, since `lyrics`
  is required non-empty by the checkpoint contract even with no words.
- `complete` — the user supplies a theme and/or partial lyrics; the LLM
  writes or finishes the words. Any lyric lines the user actually supplied
  MUST survive verbatim in the output — validated, not trusted.

`format_lyrics` is run as the final pass over the LLM's raw output for
`structure` and `complete` too, so whatever `transform()` returns is
contract-clean by construction, before validation even runs.

This is a SIBLING of `minimax_music3_caption_rewriter.py` (the caption "AI
rewrite" assistant) and, like it, a sibling of `minimax_h3_prompt_assistant.py`
rather than an extension of either: it shares the provider/cache/transport
layer (loopback URL enforcement, strict-JSON extraction, the SQLite result
cache, the error type) imported from the H3 module, but owns its own domain
logic, its own `GUIDE_VERSION`, and its own cache file — so a caption-rewrite
cache entry and a lyrics-assist cache entry can never collide even for an
identical-looking request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings
from core.extensions.minimax_h3_prompt_assistant import (
    PromptAssistCache,
    PromptAssistError,
    _extract_json,
    _headers,
    _normalise_url,
)

logger = logging.getLogger(__name__)

# Own version line, own cache file (below) — see the module docstring's last
# paragraph for why this must never share either with the caption rewriter.
GUIDE_VERSION = "minimax-music3-lyrics-assistant-2026-08-15-v1"

MODES = ("structure", "complete")

# The model card's documented vocabulary (design doc, "Lyrics assistant").
# MiniMax's own example script also uses `[interlude]` and a descriptive
# freeform tag outside this list — freeform tags are legitimate, so a tag
# outside this set is always a WARNING, never a refusal.
DOCUMENTED_TAGS = frozenset({
    "intro", "verse", "pre-chorus", "chorus", "post-chorus",
    "bridge", "instrumental", "solo", "outro",
})

_TAG_RE = re.compile(r"\[[^\]]+\]")
_TAG_ONLY_RE = re.compile(r"^\[[^\]]+\]$")


def normalize_lyrics_text(text: str) -> str:
    """CRLF/CR -> LF. Same treatment as
    `minimax_music3_caption_rewriter.normalize_caption`, applied before the
    text is formatted, validated, cached, or returned."""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub(" ", text)


def _word_tokens(text: str) -> List[str]:
    """The ordered sequence of non-tag word tokens in `text` — the unit
    `format_lyrics`'s invariant is checked against, and the unit
    `validate_complete_lyrics` uses to look for a supplied line's words,
    preserved verbatim, inside the assistant's output."""
    return _strip_tags(text).split()


def _lowercase_tag_content(raw: str) -> str:
    """Case-normalize a tag's content for OUTPUT — used by `format_lyrics`.
    Only lowercases; a freeform tag's own spacing/hyphenation is otherwise
    left untouched (e.g. "[bass-quartet-rumbles-in]" passes through as
    "[bass-quartet-rumbles-in]", not "[bass quartet rumbles in]" — the
    hyphens are the tag author's own formatting, not layout noise)."""
    return raw.strip().lower()


def _normalize_tag_content(raw: str) -> str:
    """Case- AND separator-normalized form for COMPARISON only (matching a
    tag against `DOCUMENTED_TAGS`, e.g. "pre-chorus" vs "pre chorus") — never
    used to rewrite a tag's displayed content."""
    return re.sub(r"[\s-]+", " ", raw.strip().lower())


class LyricsFormatInvariantError(RuntimeError):
    """`format_lyrics` found its own output does not preserve the input's
    words. This is a bug in `format_lyrics` itself, never a user-input
    problem — the function is documented to preserve every word for ANY
    input, so this is raised rather than silently returning a lossy
    result."""


def format_lyrics(text: str) -> str:
    """Deterministic layout fix. NO NETWORK ACCESS — this is a pure
    function, callable with no LLM configured at all.

    Rules (design doc, mode 1 "format"):
    - move any text sharing a line with a tag onto its own line (this is
      the exact defect `_normalize_lyrics` has: it drops that text instead);
    - one tag per line;
    - tag contents are lowercased, matching the checkpoint's own
      normalization so the preview reflects what the model will see
      structurally;
    - blank lines are dropped as noise.

    Invariant, ENFORCED (not merely claimed): the ordered sequence of
    non-tag word tokens in the output equals the ordered sequence in the
    input, exactly. If it does not, `LyricsFormatInvariantError` is raised
    rather than returning the mismatched result.

    Tag emission is TOTAL, including the degenerate `[ ]` case (an empty
    or whitespace-only tag body): `_TAG_RE` matches it on the INPUT side
    (`[^\\]]+` accepts a bare space), so it must also be emitted as
    something `_TAG_RE` matches on the OUTPUT side, or it silently becomes
    an extra "word" (`"[]"`) on one side of the invariant only. Lowercasing
    an all-whitespace body strips it to the empty string, which is exactly
    that failure — so the ORIGINAL (unlowercased) body is kept whenever the
    lowercased form would be empty, which for a real, non-empty tag body
    never differs from the lowercased form anyway.
    """
    normalized = normalize_lyrics_text(text)
    original_words = _word_tokens(normalized)

    out_lines: List[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue  # drop blank noise
        pos = 0
        for match in _TAG_RE.finditer(line):
            # At most one run of text can ever sit here (between the
            # previous tag, or line start, and this one) -- flushed
            # immediately below, so this holds a scalar, not a list.
            before = line[pos:match.start()].strip()
            if before:
                out_lines.append(before)
            tag_body = match.group(0)[1:-1]
            lowered = _lowercase_tag_content(tag_body)
            out_lines.append(f"[{lowered if lowered else tag_body}]")
            pos = match.end()
        tail = line[pos:].strip()
        if tail:
            out_lines.append(tail)

    formatted = "\n".join(out_lines)
    formatted_words = _word_tokens(formatted)
    if formatted_words != original_words:
        raise LyricsFormatInvariantError(
            "format_lyrics changed the supplied words, which it must never do: "
            f"expected word sequence {original_words!r}, got {formatted_words!r}"
        )
    return formatted


_NORMALIZED_DOCUMENTED_TAGS = {_normalize_tag_content(tag) for tag in DOCUMENTED_TAGS}


def unknown_tag_warnings(text: str) -> List[str]:
    """Warn (never refuse) on any tag outside `DOCUMENTED_TAGS` — freeform
    tags are legitimate (design doc, "Lyrics assistant"). Compared against
    the SEPARATOR-normalized form of the documented set (`post-chorus` ==
    `post chorus`), not the raw hyphenated spelling — `_normalize_tag_content`
    collapses both the input tag's and the documented tag's separators the
    same way, so a hyphen-vs-space difference alone is never flagged.

    Public (no leading underscore): used across the module boundary by
    `routes.py` as part of the `/prompt-assist/music/lyrics/format`
    response payload, not only internally."""
    warnings: List[str] = []
    seen = set()
    for match in _TAG_RE.finditer(text):
        content = _normalize_tag_content(match.group(0)[1:-1])
        if content not in _NORMALIZED_DOCUMENTED_TAGS and content not in seen:
            seen.add(content)
            warnings.append(
                f"'[{content}]' is not one of the documented structure tags "
                f"({', '.join(sorted(DOCUMENTED_TAGS))}); freeform tags are "
                "allowed but the checkpoint may not treat them as a section marker"
            )
    return warnings


def _layout_warnings(text: str) -> List[str]:
    """Generic layout checks shared by both LLM-driven modes: non-empty,
    every tag alone on its own line, no text sharing a line with a tag.
    `transform()` always runs `format_lyrics` before this, so these should
    never fire in practice — kept as belt-and-suspenders against a future
    caller that validates un-formatted text."""
    warnings: List[str] = []
    if not text.strip():
        warnings.append("Lyrics must not be empty")
        return warnings
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        tags = list(_TAG_RE.finditer(stripped))
        if not tags:
            continue
        remainder = _TAG_RE.sub("", stripped).strip()
        if remainder:
            warnings.append(f"Text shares a line with a tag: {stripped!r}")
    return warnings


def validate_structure_lyrics(text: str) -> List[str]:
    """Mode 2 ("structure"): the output must be ONLY tags, one per line, no
    prose — the structural control surface for an instrumental piece."""
    warnings = _layout_warnings(text)
    if not text.strip():
        return warnings
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _TAG_ONLY_RE.fullmatch(stripped):
            warnings.append(f"Structure mode must emit only tags, one per line; got: {stripped!r}")
    warnings.extend(unknown_tag_warnings(text))
    return warnings


def _contains_contiguous_sublist(haystack: List[str], needle: List[str]) -> bool:
    """Whether `needle` appears as a contiguous, in-order run inside
    `haystack`. A plain `" ".join(...)` substring check on the SAME
    tokenizer is unsound: joining tokens with spaces manufactures word
    boundaries that were never real word boundaries, so e.g. needle
    ["in", "the", "rain"] would falsely match inside haystack
    ["spin", "the", "rainbow", "drifts"] purely because the joined strings
    "in the rain" and "spin the rainbow drifts" share that substring across
    token boundaries ("sp[in the rain]bow drifts"). Comparing token LISTS,
    not joined strings, has no such hole."""
    if not needle:
        return True
    n = len(needle)
    return any(haystack[start:start + n] == needle for start in range(len(haystack) - n + 1))


def validate_complete_lyrics(supplied_lyrics: str, output_text: str) -> List[str]:
    """Mode 3 ("complete"): any lyric line the user actually supplied (in
    the partial lyrics they sent, not a bare tag line) must survive
    VERBATIM in the output — compared as an exact, in-order, contiguous
    word-token subsequence (the same tokenizer `format_lyrics`'s own
    invariant uses), so incidental whitespace/line-wrap differences from the
    LLM do not cause a false reject, but a paraphrase or a dropped word does
    cause a genuine one."""
    warnings = _layout_warnings(output_text)
    supplied_lines = [
        line.strip() for line in supplied_lyrics.splitlines()
        if line.strip() and not _TAG_ONLY_RE.fullmatch(line.strip())
    ]
    if supplied_lines:
        output_words = _word_tokens(output_text)
        for line in supplied_lines:
            line_words = _word_tokens(line)
            if line_words and not _contains_contiguous_sublist(output_words, line_words):
                warnings.append(f"Supplied lyric line was not preserved verbatim: {line!r}")
    warnings.extend(unknown_tag_warnings(output_text))
    return warnings


@dataclass(frozen=True)
class MusicLyricsAssistOptions:
    mode: str  # "structure" | "complete"
    theme: str
    lyrics: str
    constraints: str
    provider: str
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    context_length: int
    timeout_seconds: int
    force_refresh: bool = False


def _system_prompt(mode: str, theme_supplied: bool, lyrics_supplied: bool) -> str:
    tag_list = ", ".join(f"[{tag}]" for tag in sorted(DOCUMENTED_TAGS))
    # The JSON key is "prompt", not "lyrics": `_extract_json` is shared,
    # unmodified transport code from the H3 module (see this module's own
    # docstring) and is hardcoded to require a `prompt` string field
    # regardless of domain. The outer API response still calls this field
    # `lyrics` — only the wire shape the LLM itself must produce uses
    # "prompt", to stay on the one shared, unmodified extractor.
    common_rules = f"""Return exactly one JSON object: {{"prompt":"...","warnings":["..."]}}.
The JSON "prompt" value must be the complete lyrics text, not a summary.

Structure-tag rules (apply to every mode):
- A structure tag looks like [verse] or [chorus], on its own line, nothing else on that line.
- Prefer the documented tags where they fit: {tag_list}. A freeform, descriptive tag is allowed when nothing documented fits, but never put ordinary words on the same line as a tag.
- Every tag is lowercase."""

    if mode == "structure":
        return f"""You write the section/structure map for a MiniMax Music 3 instrumental track. There are no words — lyrics is the only control surface for the arrangement, so this output IS the arrangement plan.
{common_rules}

Output rules for this mode:
- Output ONLY tags, one per line. No prose, no descriptions, no words of any kind outside the brackets.
- The section description the user gives you (mood, arrangement, instrumentation) should be reflected in WHICH tags you choose and their order and repetition, not written out as text.
"""

    lyrics_note = (
        "The user supplied partial lyrics below. Every line of those partial lyrics that is not "
        "itself a bare tag line MUST appear again in your output, VERBATIM, word for word, in the "
        "same order relative to the rest of your output — do not paraphrase, trim, or reorder it. "
        "Write new sections around it to complete the song."
        if lyrics_supplied
        else "No partial lyrics were supplied; write the full lyrics from the theme alone."
    )
    theme_note = (
        "Follow the user's theme for subject matter, mood and tone."
        if theme_supplied
        else "No theme was supplied; take the tone from the partial lyrics alone."
    )
    return f"""You write or complete lyrics for a MiniMax Music 3 track.
{common_rules}

Output rules for this mode:
- {theme_note}
- {lyrics_note}
- Between tag lines, write the actual lyric words as plain text lines — never on the same line as a tag.
"""


class MiniMaxMusic3LyricsAssistant:
    def __init__(self, max_cache_entries: int, cache_path: Optional[Path] = None) -> None:
        resolved_cache_path = cache_path or Path(settings.cache_dir) / "minimax_music3_lyrics_assist.sqlite3"
        self.cache = PromptAssistCache(resolved_cache_path, max_cache_entries)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, provider: str, base_url: str, model: str) -> threading.Lock:
        key = f"{provider}:{base_url}:{model}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _cache_key(self, options: MusicLyricsAssistOptions) -> str:
        material = {
            "guide": GUIDE_VERSION,
            "mode": options.mode,
            "theme": options.theme.strip().replace("\r\n", "\n"),
            "lyrics": options.lyrics.strip().replace("\r\n", "\n"),
            "constraints": options.constraints.strip().replace("\r\n", "\n"),
            "provider": options.provider,
            "base_url": options.base_url,
            "model": options.model,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "max_output_tokens": options.max_output_tokens,
            "context_length": options.context_length,
        }
        encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _validate(self, options: MusicLyricsAssistOptions, text: str) -> List[str]:
        if options.mode == "structure":
            return validate_structure_lyrics(text)
        return validate_complete_lyrics(options.lyrics, text)

    def transform(self, options: MusicLyricsAssistOptions, api_key: str = "") -> Dict[str, Any]:
        if options.mode not in MODES:
            raise PromptAssistError(f"Unsupported mode: {options.mode}")
        if options.mode == "structure" and not options.theme.strip():
            raise PromptAssistError("Describe the arrangement/section structure first")
        if options.mode == "complete" and not options.theme.strip() and not options.lyrics.strip():
            raise PromptAssistError("Supply a theme or some partial lyrics first")
        if not options.model:
            raise PromptAssistError("Select a local LLM model first")
        base_url = _normalise_url(options.base_url)
        options = MusicLyricsAssistOptions(**{**options.__dict__, "base_url": base_url})
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
                options.mode, bool(options.theme.strip()), bool(options.lyrics.strip())
            )
            user_message = self._user_message(options)
            if options.provider == "lm_studio":
                result, lifecycle_warnings = self._lm_studio(options, system_prompt, user_message, api_key)
            elif options.provider == "ollama":
                result, lifecycle_warnings = self._ollama(options, system_prompt, user_message)
            else:
                raise PromptAssistError(f"Unsupported provider: {options.provider}")
            parsed = _extract_json(result)
            # format_lyrics is the final pass over the LLM's output for BOTH
            # LLM-driven modes, so whatever is returned is contract-clean by
            # construction before validate() below ever runs. Guarded even
            # though format_lyrics is now provably total (every tag it
            # emits is guaranteed non-empty, so it always matches _TAG_RE on
            # the output side too -- see its own docstring): a caller of
            # `transform()` should only ever see PromptAssistError, exactly
            # like every other failure path in this method, not a bare
            # LyricsFormatInvariantError escaping past the route's
            # PromptAssistError-only handler as an unhandled 500.
            try:
                lyrics_out = format_lyrics(normalize_lyrics_text(parsed["prompt"]))
            except LyricsFormatInvariantError as exc:
                raise PromptAssistError(
                    f"The LLM's lyrics could not be safely reformatted: {exc}"
                ) from exc
            warnings = [
                str(item) for item in parsed.get("warnings", [])
                if str(item).strip().lower() not in {"", "none", "n/a"}
            ]
            structural_warnings = self._validate(options, lyrics_out)
            warnings.extend(structural_warnings)
            warnings.extend(lifecycle_warnings)
            response = {
                "lyrics": lyrics_out,
                "warnings": list(dict.fromkeys(warnings)),
                "valid": not structural_warnings,
                "provider": options.provider,
                "model": options.model,
                "mode": options.mode,
                "cached": False,
            }
            if response["valid"]:
                self.cache.put(cache_key, response)
            return {**response, "cache_key": cache_key}

    @staticmethod
    def _user_message(options: MusicLyricsAssistOptions) -> str:
        parts = [f"Mode: {options.mode}"]
        if options.theme.strip():
            label = "Section/arrangement description" if options.mode == "structure" else "Theme"
            parts.append(f"{label}: {options.theme.strip()}")
        if options.lyrics.strip():
            parts.append(f"Partial lyrics (preserve verbatim, complete around them):\n{options.lyrics.strip()}")
        if options.constraints.strip():
            parts.append(f"Additional constraints: {options.constraints.strip()}")
        return "\n\n".join(parts)

    def _repair_message(self, warnings: List[str], user_message: str, previous_answer: str) -> str:
        return (
            "Your previous answer failed validation. Correct it and return only the JSON object.\n"
            "The JSON \"prompt\" string (the lyrics text) itself must fix every issue listed below.\n"
            f"Validation errors: {json.dumps(warnings, ensure_ascii=False)}\n"
            f"Original request: {user_message}\n"
            f"Previous answer: {previous_answer}"
        )

    def _validate_first_pass(
        self, output_text: str, options: MusicLyricsAssistOptions
    ) -> List[str]:
        try:
            first_lyrics = format_lyrics(normalize_lyrics_text(_extract_json(output_text)["prompt"]))
            return self._validate(options, first_lyrics)
        except (PromptAssistError, LyricsFormatInvariantError) as exc:
            return [str(exc)]

    def _lm_studio(
        self,
        options: MusicLyricsAssistOptions,
        system_prompt: str,
        user_message: str,
        api_key: str,
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
            structural_warnings = self._validate_first_pass(output_text, options)
            if structural_warnings:
                chat_payload["input"] = self._repair_message(structural_warnings, user_message, output_text)
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
        self,
        options: MusicLyricsAssistOptions,
        system_prompt: str,
        user_message: str,
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
            structural_warnings = self._validate_first_pass(content, options)
            if structural_warnings:
                chat_payload["messages"].append({"role": "assistant", "content": content})
                chat_payload["messages"].append({
                    "role": "user",
                    "content": (
                        "Correct the previous JSON. The \"prompt\" string (the lyrics text) itself must fix every issue. "
                        "Errors: " + json.dumps(structural_warnings, ensure_ascii=False)
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


def find_lyrics_drop_warnings(lyrics: str) -> List[str]:
    """Surface the checkpoint's silent-drop defect even when the assistant
    is not used (design doc, "Lyrics assistant"): one warning per line where
    a leading tag is followed by text on the same line — that text will not
    reach the model, per `_normalize_lyrics` in
    `backend/core/models/minimax_music3/pipeline.py`.

    Matches the RAW line, exactly as `_normalize_lyrics` does — NOT a
    `.strip()`-ed copy of it. `_normalize_lyrics` only tolerates `[ \\t]*`
    before the first tag; a line like `"\\xa0[verse] words"` (a leading
    non-breaking space, or any other character `str.strip()` treats as
    whitespace but the checkpoint's own `[ \\t]*` does not) fails the
    checkpoint's own match entirely, so the WHOLE raw line — including the
    leading character and the words — passes through `_normalize_lyrics`
    unchanged, dropping nothing. Stripping first would make this detector
    disagree with the checkpoint precisely on that input, warning about a
    drop that never happens."""
    warnings: List[str] = []
    if not lyrics:
        return warnings
    for index, raw_line in enumerate(lyrics.split("\n"), start=1):
        if not raw_line.strip():
            continue
        match = _LEADING_TAGS_FOR_WARNING_RE.match(raw_line)
        if not match:
            continue
        dropped = raw_line[match.end():].strip()
        if dropped:
            warnings.append(
                f"Lyrics line {index} has text after a leading tag ('{dropped}') that will "
                "not reach the model: the checkpoint keeps only the leading tag(s) on a line "
                "and drops the rest. Put that text on its own line, or use the lyrics format assist."
            )
    return warnings


# Mirrors `_LEADING_TAGS_RE` in `core.models.minimax_music3.pipeline` exactly
# (not imported from it: that module is the checkpoint-contract source of
# truth and this warning must detect precisely what it will drop, but a
# generation-time warning helper has no reason to import a full model
# pipeline module just for one compiled regex).
_LEADING_TAGS_FOR_WARNING_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")
