"""Filesystem-path redaction for artifacts that leave this machine.

A PNG produced by SushiUI travels: it is uploaded, shared, and re-imported
elsewhere. Its text chunks therefore may carry names the user chose (a model
file's basename, a training run's name) but must never carry the structure the
filesystem generated around them — drive letters, home directories, project
layout. That is personal environment information about the machine, not about
the image.

This module is the shared implementation behind three call sites:

* ``api/generation_overrides._friendly_component_name`` — candidate names in
  ``GET /models/vaes`` (and the TE list), which the selector UI shows.
* ``api/generation_utils.describe_vae_override`` — the producer of
  ``params["vae_name"]``, the value written to both the PNG and the DB row.
* ``utils/image_utils.save_image_with_metadata`` — the only PNG writer, which
  applies :func:`redact_paths` as a backstop so a future call site cannot
  reintroduce a path into a shareable file by writing one into ``params``.

``database/models.py`` uses it as well, to redact at READ time the labels of
rows written before the producer was fixed (rows are never rewritten).
"""

import os
import re
from typing import Optional

#: Names the DIFFUSERS LAYOUT generates, not the user: component subfolders and
#: the conventional weight filename inside them. They repeat verbatim under
#: every model folder, so on their own they identify nothing — measured on this
#: machine, ``diffusion_pytorch_model`` occurs 20 times and ``transformer`` 5
#: times across the real model/training trees. For these, and only these, the
#: enclosing directory names are prepended until a name a HUMAN chose is
#: reached, which is what makes the label resolve to one file locally. A model
#: folder name is the same class of user-chosen text as a model file's
#: basename; the drive, the user directory and everything above the first
#: human-chosen segment are never emitted.
GENERIC_COMPONENT_NAMES = {
    "",
    "vae",
    "vae_encoder",
    "vae_decoder",
    "audio_vae",
    "transformer",
    "transformer_2",
    "unet",
    "controlnet",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "image_encoder",
    "vision_encoder",
    "encoder",
    "decoder",
    "vocoder",
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "scheduler",
    "feature_extractor",
    "safety_checker",
    # conventional weight filenames (extension stripped before the lookup)
    "diffusion_pytorch_model",
    "pytorch_model",
    "adapter_model",
    "model",
    "model.fp16",
    "diffusion_pytorch_model.fp16",
    # conventional training-checkpoint filenames
    "ema",
    "model_ema",
    "optimizer",
    "checkpoint",
}

#: A sharded weight filename (``diffusion_pytorch_model-00001-of-00002``) is as
#: generated as the unsharded one; strip the shard suffix before the lookup.
_SHARD_SUFFIX = re.compile(r"-\d+-of-\d+$")

#: Hard cap on how many path segments a display name may contain. Bounds how
#: much directory structure a pathological layout could expose, even if every
#: segment is generic.
MAX_NAME_SEGMENTS = 3

# Measured on this machine (all 7 candidate endpoints, 109 selectable paths;
# and a filesystem sweep of the model/lora/controlnet/training roots, 172
# model-like candidates): under this rule 0 API candidates and 1 filesystem
# candidate share a display name -- the latter being the same checkpoint
# present in two packagings, so the MODEL is still uniquely determined.
#
# CAVEAT for whoever extends this: some fields ship a content hash next to the
# name (``vae_name``+``vae_hash``, ``model_name``+``model_hash``,
# ``loras[].name``+``hash``, ``upscaler_model``+``upscaler_model_hash``), so a
# name collision there is recoverable. ``outpaint_controlnet_model``,
# ``controlnets[].model_path``, ``text_encoder_path`` and
# ``vae_override_source`` have NO hash anywhere -- for those the display name
# is the only identifier the artifact carries, and a collision is
# unrecoverable. Re-run the collision measurement before widening the generic
# set or lowering MAX_NAME_SEGMENTS.

#: Last-resort label. Only reachable for a path with no usable segment at all
#: (e.g. a bare drive root), which no override can realistically be. It states
#: that the NAME could not be derived — it never claims the file is unknown.
UNNAMED = "unnamed file"

_DRIVE_ROOT = re.compile(r"[A-Za-z]:")

#: Directories whose immediate children are ACCOUNT NAMES, not model names.
#: A segment found directly under one of these is a username and is never
#: emitted, even when that costs disambiguation — the resulting label is then
#: simply less specific, which is the safe direction.
_ACCOUNT_CONTAINERS = {"users", "home", "documents and settings"}


def _is_account_name(path_of_segment: str) -> bool:
    """True when ``path_of_segment`` sits directly inside a user-account
    container (``C:\\Users\\<name>``, ``/home/<name>``)."""
    parent = os.path.basename(os.path.dirname(path_of_segment.rstrip("/\\")))
    return parent.lower() in _ACCOUNT_CONTAINERS


def _is_generic(segment: str) -> bool:
    stem = segment
    for ext in (".safetensors", ".bin", ".pth", ".ckpt", ".pt", ".gguf"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = _SHARD_SUFFIX.sub("", stem)
    return stem.lower() in GENERIC_COMPONENT_NAMES


def display_name_for_path(path: str, strip_safetensors: bool = False) -> str:
    """Return a share-safe display name for ``path``.

    The name is the basename; while that basename is one the diffusers layout
    generated rather than one a human chose (``vae``, ``transformer``,
    ``diffusion_pytorch_model``), the enclosing directory is prepended, up to
    :data:`MAX_NAME_SEGMENTS` segments — ``krea2/vae``,
    ``krea2/vae/diffusion_pytorch_model``.

    A directory is prepended ONLY if it is a name in the same sense as the
    leaf: never a drive/POSIX root, and never a user-account directory
    (``C:\\Users\\<name>``, ``/home/<name>``), which would put a username in
    every shared PNG. When the only available parent is one of those the walk
    stops and the label stays less specific — ``C:\\Users\\<name>\\text_encoder``
    yields ``text_encoder``, not ``<name>/text_encoder``.

    Never returns a drive letter, an absolute path, or an empty string.
    """
    if not isinstance(path, str) or not path.strip():
        return UNNAMED
    norm = path.rstrip("/\\")

    # The leaf is always the name. Enclosing directories are added only while
    # the name so far is one the layout generated, and each candidate is
    # checked BEFORE it is added (so the segment that exhausts the budget is
    # vetted like every other one).
    base = os.path.basename(norm)
    if not base or _DRIVE_ROOT.fullmatch(base):
        return UNNAMED
    if _is_account_name(norm):
        # The path IS a user-account directory; there is no name to show.
        return UNNAMED
    segments = [base]
    current = norm
    while _is_generic(os.path.basename(current)) and len(segments) < MAX_NAME_SEGMENTS:
        parent = os.path.dirname(current)
        if not parent or parent == current:
            break
        parent_base = os.path.basename(parent)
        if not parent_base or _DRIVE_ROOT.fullmatch(parent_base):
            break
        if _is_account_name(parent):
            break
        segments.insert(0, parent_base)
        current = parent

    if not segments:
        return UNNAMED
    if strip_safetensors:
        segments[-1] = segments[-1].replace(".safetensors", "")
    return "/".join(s for s in segments if s) or UNNAMED


#: Delimiters that end a path run: the punctuation this repo's labels use
#: around a path (``override: <path> (run ..., step ...)``). Everything else,
#: including spaces, is treated as part of the path, so a path containing
#: spaces is still redacted whole.
_DELIM = r"[^,;()\[\]\"'\r\n]"

#: One path segment for the separator-anchored patterns below: no separator, no
#: whitespace, no sentence punctuation.
_SEG = r"[^\\/\s,;()\[\]\"'\r\n]"

#: Matches an absolute path in three unambiguous forms:
#:
#: * drive-qualified (``Z:\...``) — the drive letter is its own anchor, so the
#:   run may contain spaces (``Z:\my models\vae``). Not the ``s:/`` inside
#:   ``https://`` (the letter must stand alone).
#: * UNC (``\\host\share\...``) — the host segment is REQUIRED: a bare ``\\``
#:   also occurs as a doubled A1111 escape (a real prompt in this repo's
#:   gallery contains ``color pencil \(medium\\)``), and matching it alone
#:   replaced the escape with the no-name fallback text.
#: * POSIX absolute (``/home/bob/vae``) — REQUIRES at least two
#:   slash-separated, whitespace-free segments. Without that requirement the
#:   branch matched ordinary prose: measured on this repo's own warning
#:   strings, ``NAG / NegPip / DEUS / style transfer / Spectrum / FBCache``
#:   lost four feature names and ``row(s)/column(s)`` lost its slash, i.e. the
#:   redactor rewrote a degradation notice into a false statement. Prose slashes
#:   are surrounded by spaces or followed by one segment only; a path is not.
#:   The cost is that a POSIX path containing spaces is only partly matched —
#:   an acceptable trade against corrupting text, and the Windows branch (the
#:   one this repo actually produces) is unaffected.
_ABSOLUTE_PATH = re.compile(
    r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]" + _DELIM + r"*"
    r"|\\\\" + _SEG + r"+" + _DELIM + r"*"
    r"|(?<![\w.:/])/" + _SEG + r"+(?:/" + _SEG + r"+)+)"
)

#: Second pass: the TAIL of a Windows path whose middle contained one of the
#: delimiters above (``C:\Program Files (x86)\models\vae`` -> the first pass
#: removes the drive, this removes ``\models\vae``). Anchored on a separator
#: FOLLOWED BY a real segment, and requiring either a segment BEFORE the
#: separator or a second separator after it — i.e. something with the shape of
#: a path, never a lone escape. The previous form matched any backslash run and
#: silently destroyed prompt syntax: ``azarin \(exs-tia\)`` lost its escapes
#: (a literal parenthesis became an emphasis group) and the danbooru tag
#: ``\m/`` became ``m/`` — measured on 18 distinct real prompts in this
#: repo's gallery.
_WINDOWS_FRAGMENT = re.compile(
    r"(?:" + _SEG + r"+\\" + _SEG + r"+|\\" + _SEG + r"+\\" + _SEG + r"+)"
    r"(?:\\" + _SEG + r"+)*"
)


def _redact_match(match) -> str:
    raw = match.group(0)
    body = raw.rstrip()
    trailing = raw[len(body):]
    if not body:
        return raw
    return display_name_for_path(body) + trailing


def redact_paths(value: Optional[str]) -> Optional[str]:
    """Reduce every absolute path inside ``value`` to its display name.

    Redaction, not validation: a value that contains no path is returned
    unchanged, and this never raises or blocks a save. Used as the PNG
    writer's backstop.
    """
    if not isinstance(value, str) or not value:
        return value
    try:
        out = _ABSOLUTE_PATH.sub(_redact_match, value)
        if "\\" in out:
            out = _WINDOWS_FRAGMENT.sub(_redact_match, out)
        return out
    except Exception:
        return value


#: Keys whose value is text the USER typed. These are reproduced verbatim: a
#: prompt is the content of the image, and silently rewriting part of it would
#: corrupt the record of what was generated — an A1111 escape (``\(exs-tia\)``)
#: losing its backslashes turns a literal parenthesis into an emphasis group,
#: so the recorded prompt no longer reproduces the image. (A path a user typed
#: into a prompt is their own choice to share, the same as any other prompt
#: text.)
#:
#: Derived from data, not memory: every string-valued key in all 10,749
#: gallery.db rows was enumerated and any key with a hand-written-text marker
#: (A1111 escapes, ``:weight`` syntax, ", " separators, newlines) was included.
#: Re-run that enumeration after adding a free-text parameter.
USER_TEXT_KEYS = {
    "prompt",
    "negative_prompt",
    "region_prompt",           # measured: 39 rows, carries A1111 escapes
    "region_negative_prompt",  # measured: 4 rows
    "nag_prompt",
    "nag_negative_prompt",
    "lyrics",                  # ACE-Step audio; user-authored
    "caption",                 # audio prompt alias (audio_utils sidecar)
    "tipo_config",
    "tipo_prompt",
    "original_prompt",
    "processed_prompt",
}

#: Keys holding a machine token that other code MATCHES ON (``to_dict``'s
#: legacy-label gate and the frontend both compare ``warnings[].code``).
#: Rewriting one would silently break those consumers; they can never contain
#: a path, so there is nothing to redact.
IDENTIFIER_KEYS = {
    "code",
}


def redact_params_for_sharing(value, _key: Optional[str] = None):
    """Return ``value`` with every absolute path inside it reduced to a name.

    Recurses through dicts/lists and rebuilds containers (the caller's objects
    are never mutated — the same ``params`` dict is also handed to the DB
    writer, which keeps the full paths). Values under :data:`USER_TEXT_KEYS`
    and :data:`IDENTIFIER_KEYS` are passed through untouched.

    Used by the PNG writer on the full-parameter JSON blob, which — unlike the
    hand-maintained per-key chunks — carries every parameter present, including
    the local-only ones (``vae_path``, ``vae_override_path``,
    ``vae_override_source``, ``text_encoder_path``, component/LoRA paths).
    """
    if isinstance(value, dict):
        return {k: redact_params_for_sharing(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_params_for_sharing(v, _key) for v in value]
    if isinstance(value, str):
        if _key in USER_TEXT_KEYS or _key in IDENTIFIER_KEYS:
            return value
        return redact_paths(value)
    return value
