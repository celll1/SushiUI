"""Recover an uploaded filename mis-decoded as latin-1 when it was UTF-8.

Starlette's multipart parser decodes a part's ``Content-Disposition``
``filename`` using the request's declared charset (``utf-8`` unless the
request itself says otherwise), and falls back to ``latin-1`` on a decode
failure -- see ``starlette.formparsers._user_safe_decode``. That fallback
never raises (latin-1 maps every byte 0-255 to one code point 1:1), so it
never DISCARDS bytes; it just reinterprets them at the wrong code points,
which is losslessly reversible: re-encoding the result as latin-1 recovers
the original bytes exactly, and decoding those as UTF-8 recovers the
original text.

This is the ONE upload-filename failure mode this repo can fix after the
fact, because it never destroys information. A genuinely invalid byte
sequence (e.g. a decoder that used ``errors="replace"``) already lost data
before this code runs -- U+FFFD is unrecoverable and is left alone; treating
that as "fixed" would misreport a lossy failure as lossless.
"""

from typing import Optional


def recover_upload_filename(name: Optional[str]) -> Optional[str]:
    """Repair ``name`` if it looks like UTF-8 bytes decoded as latin-1.

    A no-op for ASCII names (round-trips to themselves) and for names that
    are already correctly decoded (non-latin-1 code points make the
    latin-1 re-encode raise, so the original is returned unchanged).
    """
    if not name:
        return name
    try:
        return name.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name
