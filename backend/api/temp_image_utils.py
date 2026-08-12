"""Shared, traversal-safe resolution of `temp_img://<id>` references.

A ref is trusted only after two checks: its id is reduced to its basename
via `os.path.basename` (a path separator or `..` component in the id is
silently stripped down to whatever comes after the last separator, not
rejected outright -- see `resolve_temp_image_path`), and the resulting path's
`os.path.realpath` must still sit inside the temp directory's own realpath
(rejects a symlink planted inside it pointing elsewhere). Every caller that
resolves a `temp_img://` ref anywhere in this backend goes through this
module -- routes.py used to inline this resolution at three call sites with
no traversal check at all.
"""
from __future__ import annotations

import os
from typing import Optional

TEMP_IMG_PREFIX = "temp_img://"


class TempImageRefError(Exception):
    """A temp_img:// ref did not resolve to a real, in-bounds file."""


class TempImageRefTooLargeError(TempImageRefError):
    """A temp_img:// ref DID resolve, but its file exceeds `max_bytes`.

    Distinct from the base `TempImageRefError` (missing/out-of-bounds/not a
    ref) so a caller can tell "this ref is fine, the file is just too big for
    THIS request's cap" apart from "this ref cannot be resolved at all" --
    the two need different client responses (see `/video-mask/preview`'s
    409-vs-400 handling in routes.py).
    """


def resolve_temp_image_path(ref: str, temp_dir: str) -> str:
    """Resolve a `temp_img://<id>` ref to an absolute path inside `temp_dir`.

    Raises `TempImageRefError` for anything that is not a syntactically valid
    ref pointing at an existing, in-bounds file. Never returns a path outside
    `temp_dir`.
    """
    if not ref or not ref.startswith(TEMP_IMG_PREFIX):
        raise TempImageRefError(f"Not a {TEMP_IMG_PREFIX} reference: {ref!r}")
    image_id = os.path.basename(ref[len(TEMP_IMG_PREFIX):])
    if not image_id:
        raise TempImageRefError(f"Empty temp image id in ref: {ref!r}")
    real_temp_dir = os.path.realpath(temp_dir)
    candidate = os.path.realpath(os.path.join(real_temp_dir, image_id))
    if candidate != real_temp_dir and not candidate.startswith(real_temp_dir + os.sep):
        raise TempImageRefError(f"temp_img reference escapes its temp directory: {ref!r}")
    if not os.path.isfile(candidate):
        raise TempImageRefError(f"temp_img reference not found: {ref!r}")
    return candidate


def resolve_temp_image_bytes(ref: str, temp_dir: str, max_bytes: Optional[int] = None) -> bytes:
    """Resolve a `temp_img://` ref straight to its file bytes.

    `max_bytes`, if given, rejects an oversized file via a cheap
    `os.path.getsize` check before reading it -- e.g. the video-mask-preview
    route's manifest-canvas-derived byte cap, the same cap it already applies
    to a directly-uploaded mask PNG.
    """
    path = resolve_temp_image_path(ref, temp_dir)
    if max_bytes is not None:
        size = os.path.getsize(path)
        if size > max_bytes:
            raise TempImageRefTooLargeError(
                f"temp_img reference {ref!r} is {size} bytes, exceeding the {max_bytes} byte cap"
            )
    with open(path, "rb") as f:
        return f.read()
