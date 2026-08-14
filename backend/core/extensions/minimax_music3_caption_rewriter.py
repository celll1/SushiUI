"""MiniMax Music 3 caption rewriter ("AI rewrite").

Upstream's own "prompt refiner" is not runtime code: it is an agent skill
(`skills/music-caption-rewriter/` in MiniMax-AI/MiniMax-Music3), a SKILL.md
plus a genre router and ~1,000 caption templates, installed with
`npx skills add`, and it calls no model of its own. SushiUI does not install
it — the npm step only copies markdown, and there is nothing to execute.
Instead this module re-implements the CONTRACT that skill encodes, against
the user's own already-configured local LLM.

This is a SIBLING of `minimax_h3_prompt_assistant.py`, not an extension of
it. The two modules share the provider/cache/transport layer: the loopback
URL enforcement (`_normalise_url`), the strict-JSON extraction
(`_extract_json`), the HTTP header helper (`_headers`), the SQLite result
cache (`PromptAssistCache`), and the error type (`PromptAssistError`) are
all imported from the H3 module rather than re-implemented here, exactly as
that module already lets other code import its private
`_alignment_instruction` (see routes.py's `minimax_h3_alignment_instruction`
alias). None of that code is video-specific.

What is NOT shared is the video-shaped domain logic: H3's `ALL_MODES`,
`SECTION_NAMES`, `build_template`, `validate_prompt` and `_system_prompt`
are about shots, timestamps and reference tokens, and none of that applies
to a music caption. So this module has its own options dataclass, its own
system prompt, its own validator, and its own LM Studio / Ollama transport
methods that drive the one self-repair retry — structurally mirroring
`MiniMaxH3PromptAssistant.transform/_lm_studio/_ollama`, but built against
the music contract below.

The music contract (docs/guides/MINIMAX_MUSIC3_DESIGN.md, "Caption
rewriter (AI rewrite)"):

- Input: a short caption (required), optional lyrics (context only — never
  rewritten, never quoted), and optional freeform user constraints.
- Output: a Structured Caption with exactly three headings, in this order:
  Global Metadata, Vocal Details, Arrangement.
- 250-450 English words.
- Never quotes or reproduces a lyric line.
- Never invents a BPM or musical key the user did not supply.
- Preserves explicit exclusions the user stated (e.g. "no drums").

Music 3 strips Markdown from the caption at generation time
(`_clean_caption` in the vendored pipeline), so the headings above are
structural markers for the validator, not formatting the model will see —
the system prompt tells the LLM not to emit Markdown decoration at all.
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

# Bumped from v1 after the fidelity-rule audit below fixed false accepts and
# false rejects in the key/BPM/exclusion checks: a cached v1 result could
# have been produced by a rule that no longer exists.
GUIDE_VERSION = "minimax-music3-caption-rewriter-2026-08-14-v2"

SECTION_NAMES = ["Global Metadata", "Vocal Details", "Arrangement"]

_MIN_WORDS = 250
_MAX_WORDS = 450

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")

# --- BPM detection -----------------------------------------------------
#
# Digit forms: "128 bpm", "128 beats per minute", "tempo of 92". Matched
# separately from spelled-out forms ("ninety-two beats per minute") because
# an LLM told not to state a BPM will readily reach for prose instead of a
# bare number. Both forms feed into `_extract_bpms`, which returns a set of
# integers so comparison against the source is exact-value, not substring:
# a raw "in" check would let source "a 1992 rave revival" legitimise an
# invented "92 bpm", and would let source "at 128 bpm" legitimise an
# invented "12 bpm" (both are substrings of "128").
_BPM_DIGIT_RE = re.compile(r"\b(\d{2,3})\s*(?:bpm|beats\s+per\s+minute)\b", re.IGNORECASE)
_TEMPO_OF_RE = re.compile(r"\btempo\s+of\s+(\d{2,3})\b", re.IGNORECASE)

_ONES_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_NUMBER_WORD_ALT = "|".join(
    sorted([*_ONES_WORDS, *_TENS_WORDS, "hundred", "and"], key=len, reverse=True)
)
_SPELLED_BPM_RE = re.compile(
    rf"\b((?:(?:{_NUMBER_WORD_ALT})[\s-]+)*(?:{_NUMBER_WORD_ALT}))"
    r"\s+(?:bpm|beats\s+per\s+minute)\b",
    re.IGNORECASE,
)


def _word_number_to_int(phrase: str) -> Optional[int]:
    """Parse a spelled-out number phrase like "ninety-two" or "one hundred
    twenty". Returns None if any token is not a recognised number word."""
    tokens = [token for token in re.split(r"[\s-]+", phrase.strip().lower()) if token]
    if not tokens:
        return None
    total = 0
    found = False
    for token in tokens:
        if token == "and":
            continue
        if token in _ONES_WORDS:
            total += _ONES_WORDS[token]
            found = True
        elif token in _TENS_WORDS:
            total += _TENS_WORDS[token]
            found = True
        elif token == "hundred":
            total = (total or 1) * 100
            found = True
        else:
            return None
    return total if found else None


def _extract_bpms(text: str) -> set:
    """All BPM values stated in `text`, as integers, digit or spelled-out."""
    values = {int(match) for match in _BPM_DIGIT_RE.findall(text)}
    values |= {int(match) for match in _TEMPO_OF_RE.findall(text)}
    for phrase in _SPELLED_BPM_RE.findall(text):
        parsed = _word_number_to_int(phrase)
        if parsed is not None:
            values.add(parsed)
    return values


# --- Musical key detection ----------------------------------------------
#
# The note letter is matched case-SENSITIVELY on purpose: `[A-G]` under
# IGNORECASE also matches the lowercase article "a" and the pronoun/word
# fragments containing other note letters, so "a minor lift in the bridge"
# and "a major shift at the drop" were being rejected as invented keys.
# The mode word after it stays case-insensitive (users write "Major" too),
# and the separator allows a hyphen ("E-minor") as well as a space, and no
# separator at all for the shorthand form ("Em", "Bbm"). Modal names are
# included because "in the key of E dorian" is a key statement the BPM/key
# rule is meant to police just as much as "E minor".
_KEY_MODE_ALT = (
    "major|minor|dorian|phrygian|lydian|mixolydian|locrian|aeolian|ionian"
)
_KEY_RE = re.compile(
    rf"\b[A-G](?:#|b)?[\s-]?(?:(?i:{_KEY_MODE_ALT})|m)\b"
)

# The note letter "A" is still ambiguous even case-sensitively: it is also
# the English indefinite article, and a caption of 250-450 words of prose
# routinely opens a sentence with it ("A major lift carries it.", "A minor
# swell arrives late."). B-G have no such English-word collision, so they
# are always counted. A bare "A major"/"A minor"/modal-name form is only
# counted as a key statement when a key-context cue immediately precedes
# it (in / in the key of / key of / key:); the glued shorthand form ("Am",
# "Bbm") is exempt from this check because a capitalised bare "Am" is not
# ordinary mid-sentence English regardless of which letter it glues to.
_KEY_SHORTHAND_RE = re.compile(r"^[A-G](?:#|b)?m$", re.IGNORECASE)
_KEY_CONTEXT_CUE_RE = re.compile(
    r"(?:\bin(?:\s+the\s+key\s+of)?|\bkey\s+of|\bkey:)\s*$", re.IGNORECASE
)
_KEY_CONTEXT_WINDOW = 30


def _find_key_mentions(text: str) -> List[str]:
    """All key statements in `text`, applying the "A" disambiguation above."""
    mentions = []
    for match in _KEY_RE.finditer(text):
        matched = match.group(0)
        if matched[0].upper() == "A" and not _KEY_SHORTHAND_RE.fullmatch(matched):
            window = text[max(0, match.start() - _KEY_CONTEXT_WINDOW) : match.start()]
            if not _KEY_CONTEXT_CUE_RE.search(window):
                continue
        mentions.append(matched)
    return mentions


def _normalize_music_token(text: str) -> str:
    return re.sub(r"[\s-]+", " ", text.strip().lower())


# --- Exclusion preservation ----------------------------------------------
#
# Extracts the LAST word of the excluded phrase ("no heavy distortion" ->
# "distortion", not "heavy"), stemmed by stripping a trailing plural "s" so
# "no drums" also matches "drum kit" and "drumming". Negation is checked
# per SENTENCE rather than in a fixed lookback window, and or­der-independent,
# so "drums are absent from the mix" and "the mix omits drums entirely" are
# recognised as compliant even though the negation word follows the noun.
# The negation vocabulary covers more than "no/without/exclude": "absent",
# "omit(s)", "lack(s)", "avoid(s)", "devoid", "free of", "minus", "not".
_EXCLUSION_PHRASE_RE = re.compile(
    r"\b(?:no|without|exclud(?:e|es|ed|ing))\s+([a-zA-Z][a-zA-Z\s]*?)(?=[,.;:!?]|$)",
    re.IGNORECASE,
)
_NEGATION_WORDS_RE = re.compile(
    r"\b(?:no|without|exclud\w*|absent\w*|omit\w*|lack\w*|avoid\w*|devoid\w*|free of|minus|not)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _stem_noun(word: str) -> str:
    word = word.strip().lower()
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _sentences(text: str) -> List[str]:
    return [sentence for sentence in _SENTENCE_SPLIT_RE.split(text) if sentence.strip()]


def normalize_caption(text: str) -> str:
    """CRLF/CR -> LF, matching `minimax_h3_prompt_assistant.normalize_prompt`'s
    line-ending handling. Applied to the LLM's raw output before it is
    returned, cached, or validated -- the heading regexes anchor on `$` per
    line, and a lone `\\r` before that anchor is neither whitespace nor the
    anchor itself, so an unnormalised CRLF response fails every heading
    check and burns a repair round-trip on nothing."""
    return text.strip().replace("\r\n", "\n").replace("\r", "\n")


def _normalize_for_lyric_compare(text: str) -> str:
    """Lowercase, collapse whitespace/newlines, and drop punctuation, so a
    dropped comma or a lyric line reflowed across two lines still matches."""
    collapsed = re.sub(r"\s+", " ", text)
    stripped = re.sub(r"[^\w\s]", "", collapsed)
    return stripped.lower().strip()


@dataclass(frozen=True)
class MusicCaptionAssistOptions:
    caption: str
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


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _heading_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?m)^{re.escape(name)}:?[ \t]*$")


def validate_caption(
    prompt: str,
    lyrics: str = "",
    constraints: str = "",
    source_caption: str = "",
) -> List[str]:
    """Check the five contract properties. Returns warnings; empty = valid.

    Mirrors `minimax_h3_prompt_assistant.validate_prompt` in style
    (structural regex checks that drive the one self-repair retry) but
    against the music contract instead of the video one.
    """
    warnings: List[str] = []
    text = normalize_caption(prompt)

    # 1. Exactly one of each heading, in the required order, each preceded
    #    by a blank line unless it opens the document.
    positions: List[int] = []
    for name in SECTION_NAMES:
        matches = list(_heading_pattern(name).finditer(text))
        if len(matches) != 1:
            warnings.append(f"Expected exactly one '{name}' heading")
        positions.append(matches[0].start() if matches else -1)
    present = [position for position in positions if position >= 0]
    if present != sorted(present):
        warnings.append(
            "Sections are not in the required order: "
            + ", ".join(SECTION_NAMES)
        )
    for name, position in zip(SECTION_NAMES, positions):
        if position > 0 and not re.search(r"\n\s*\n\Z", text[:position]):
            warnings.append(f"Expected a blank line before '{name}'")

    # 2. Word count.
    word_count = _word_count(text)
    if word_count < _MIN_WORDS or word_count > _MAX_WORDS:
        warnings.append(
            f"Structured Caption must be {_MIN_WORDS}-{_MAX_WORDS} English "
            f"words (got {word_count})"
        )

    # 3. Never quote or reproduce a lyric line. Compared on a
    #    punctuation/whitespace-normalised form of both sides, so a dropped
    #    comma or a lyric line the model reflowed across two lines is still
    #    caught -- see the system prompt's "or closely enough paraphrased"
    #    wording, which an exact-substring check does not live up to.
    if lyrics.strip():
        normalized_text = _normalize_for_lyric_compare(text)
        for line in lyrics.splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r"\[[^\]]*\]", stripped):
                continue
            normalized_line = _normalize_for_lyric_compare(stripped)
            if len(normalized_line) >= 8 and normalized_line in normalized_text:
                warnings.append("The Structured Caption must not quote a lyric line")
                break

    # 4. Never invent a BPM or musical key not supplied by the user.
    # Compared as parsed values (BPM: int; key: normalised note+mode text),
    # not raw substrings -- see `_extract_bpms` for why a substring check on
    # digits is unsound (a source mentioning "1992" would legitimise an
    # invented "92 bpm").
    known_text = f"{source_caption}\n{constraints}"
    target_bpms = _extract_bpms(text)
    known_bpms = _extract_bpms(known_text)
    if target_bpms - known_bpms:
        warnings.append("The Structured Caption must not invent a BPM value")
    known_keys = {_normalize_music_token(match) for match in _find_key_mentions(known_text)}
    for key in _find_key_mentions(text):
        if _normalize_music_token(key) not in known_keys:
            warnings.append("The Structured Caption must not invent a musical key")
            break

    # 5. Preserve explicit exclusions the user stated. The excluded term is
    #    the LAST word of the captured phrase ("no heavy distortion" ->
    #    "distortion", not "heavy"), stemmed so "no drums" also matches
    #    "drum kit" or "drumming", and negation is checked per sentence
    #    (order-independent) so "drums are absent from the mix" is
    #    recognised as compliant even though "absent" follows "drums".
    exclusion_stems = set()
    for phrase in _EXCLUSION_PHRASE_RE.findall(known_text):
        words = phrase.strip().split()
        if words:
            exclusion_stems.add(_stem_noun(words[-1]))
    exclusion_stems.discard("")
    if exclusion_stems:
        sentences = _sentences(text)
        for stem in exclusion_stems:
            stem_pattern = re.compile(rf"\b{re.escape(stem)}\w*\b", re.IGNORECASE)
            violated = False
            for sentence in sentences:
                if stem_pattern.search(sentence) and not _NEGATION_WORDS_RE.search(sentence):
                    violated = True
                    break
            if violated:
                warnings.append(
                    f"The Structured Caption must preserve the exclusion of '{stem}'"
                )

    return warnings


def _system_prompt(lyrics_supplied: bool) -> str:
    lyrics_note = (
        "Lyrics are supplied only as context for mood, structure and vocal "
        "style. Never quote or reproduce any lyric line, verbatim or closely "
        "enough paraphrased to be recognizable."
        if lyrics_supplied
        else "No lyrics were supplied; do not invent any."
    )
    headings = ", ".join(SECTION_NAMES)
    return f"""You expand a short music caption into a Structured Caption for the MiniMax Music 3 model.
Return exactly one JSON object: {{"prompt":"...","warnings":["..."]}}.
The JSON prompt value must be the complete Structured Caption text, not a summary or a plain sentence.

Structure rules:
- The Structured Caption has exactly three sections, in this exact order: {headings}.
- Each heading appears alone on its own line, with one blank line before it (except the first) and one blank line before the next heading.
- Write no Markdown (no #, no **, no bullet markers, no numbered lists); the caption is stripped of Markdown before use, so the headings carry the structure, not the formatting.
- The whole Structured Caption is {_MIN_WORDS} to {_MAX_WORDS} English words.

Fidelity rules:
- {lyrics_note}
- Never invent a BPM number or a musical key name (e.g. "128 bpm", "A minor"). State one only if the user's caption or constraints already state it; otherwise describe tempo and tonal character in words.
- Preserve every explicit exclusion the user stated (for example "no drums" or "without vocals"): the Structured Caption must not describe the excluded element as present.
- Preserve every other user-stated genre, instrument, mood, and vocal characteristic; do not silently drop or contradict them.

Section content:
- Global Metadata: genre, mood, tempo character, and production style.
- Vocal Details: voice type, delivery, and vocal presence — write "instrumental, no vocals" if the user asked for no vocals.
- Arrangement: structure, instrumentation, and dynamics across the track."""


class MiniMaxMusic3CaptionRewriter:
    def __init__(self, max_cache_entries: int, cache_path: Optional[Path] = None) -> None:
        resolved_cache_path = cache_path or Path(settings.cache_dir) / "minimax_music3_prompt_assist.sqlite3"
        self.cache = PromptAssistCache(resolved_cache_path, max_cache_entries)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, provider: str, base_url: str, model: str) -> threading.Lock:
        key = f"{provider}:{base_url}:{model}"
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _cache_key(self, options: MusicCaptionAssistOptions) -> str:
        material = {
            "guide": GUIDE_VERSION,
            "caption": options.caption.strip().replace("\r\n", "\n"),
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

    def transform(self, options: MusicCaptionAssistOptions, api_key: str = "") -> Dict[str, Any]:
        if not options.caption.strip():
            raise PromptAssistError("Caption cannot be empty")
        if not options.model:
            raise PromptAssistError("Select a local LLM model first")
        base_url = _normalise_url(options.base_url)
        options = MusicCaptionAssistOptions(**{**options.__dict__, "base_url": base_url})
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
            system_prompt = _system_prompt(bool(options.lyrics.strip()))
            user_message = self._user_message(options)
            if options.provider == "lm_studio":
                result, lifecycle_warnings = self._lm_studio(options, system_prompt, user_message, api_key)
            elif options.provider == "ollama":
                result, lifecycle_warnings = self._ollama(options, system_prompt, user_message)
            else:
                raise PromptAssistError(f"Unsupported provider: {options.provider}")
            parsed = _extract_json(result)
            prompt = normalize_caption(parsed["prompt"])
            warnings = [
                str(item) for item in parsed.get("warnings", [])
                if str(item).strip().lower() not in {"", "none", "n/a"}
            ]
            structural_warnings = validate_caption(
                prompt, options.lyrics, options.constraints, options.caption
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
            }
            if response["valid"]:
                self.cache.put(cache_key, response)
            return {**response, "cache_key": cache_key}

    @staticmethod
    def _user_message(options: MusicCaptionAssistOptions) -> str:
        parts = [f"Caption: {options.caption.strip()}"]
        if options.lyrics.strip():
            parts.append(f"Lyrics (context only, never quote):\n{options.lyrics.strip()}")
        if options.constraints.strip():
            parts.append(f"Additional constraints: {options.constraints.strip()}")
        return "\n\n".join(parts)

    def _repair_message(self, warnings: List[str], user_message: str, previous_answer: str) -> str:
        return (
            "Your previous answer failed validation. Correct it and return only the JSON object.\n"
            "The JSON prompt string itself must contain every required section and the full rewrite.\n"
            f"Validation errors: {json.dumps(warnings, ensure_ascii=False)}\n"
            f"Original request: {user_message}\n"
            f"Previous answer: {previous_answer}"
        )

    def _validate_first_pass(
        self, output_text: str, options: MusicCaptionAssistOptions
    ) -> List[str]:
        try:
            first_prompt = normalize_caption(_extract_json(output_text)["prompt"])
            return validate_caption(first_prompt, options.lyrics, options.constraints, options.caption)
        except PromptAssistError as exc:
            return [str(exc)]

    def _lm_studio(
        self,
        options: MusicCaptionAssistOptions,
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
        options: MusicCaptionAssistOptions,
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
