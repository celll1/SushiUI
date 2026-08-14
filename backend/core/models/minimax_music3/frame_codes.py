"""MiniMax Music 3 per-generation frame-code sidecar.

Implements the design doc's "Per-generation state contract"
(``docs/guides/MINIMAX_MUSIC3_DESIGN.md``): extend and repaint (design doc
phase plan items 7 and 8) both need the autoregressive-stage state of the
ORIGINAL generation, and storing ``frame_hiddens`` directly is not viable
(``[1, 9000, 32768]`` in bf16 is ~590 MB per song). The frame CODES are:
8 codes per frame, each fitting comfortably in ``int16`` (semantic vocab
16,384, residual vocab 1,024 per codebook -- both well under the ``int16``
ceiling of 32,767), so a full six-minute (9,000-frame) song is
``9000 * 8 * 2 bytes = 144,000 bytes`` (~140.6 KiB) on disk -- the "~144 KB"
figure the design doc cites. The hidden states are exactly recoverable from
the codes by a teacher-forced replay
(``MiniMaxMusic3Pipeline.generate_ar``'s ``resume_frame_codes`` path).

Written next to the audio file with a distinct extension
(``<base_name>.mm3frames.json``), separate from the generic audio metadata
sidecar ``utils.audio_utils.save_audio_with_metadata`` already writes
(``<base_name>.json``): that file is common to every audio architecture and
carries display-oriented fields (prompt/lyrics/seed/inference params) a
generic viewer can read; this one is MiniMax-Music3-specific binary STATE a
future request replays through the model, and keeping it a separate file
means an architecture that never needs it (ACE-Step) never grows one.

**dtype contract** (this is the part a previous round of this work got wrong
and crashed on): the sidecar stores codes as ``int16`` ON DISK, but
``MiniMaxMusic3Pipeline.generate_ar``'s ``resume_frame_codes``/
``resume_prefix_codes`` arguments -- and, more fundamentally, every
``nn.Embedding`` lookup the AR loop performs on them
(``language_model.model.embed_tokens``, ``rvq_depth_decoder.audio_embeddings``)
-- require an integer INDEX dtype (``long``/``int``); PyTorch's embedding
lookup raises on a ``ShortTensor`` with no context tying the error back to
"the sidecar wasn't upcast" ("Expected tensor for argument #1 'indices' to
have ... Long, Int; but got torch.ShortTensor"). ``generate_ar`` itself
upcasts on the way in ("a caller storing a compact sidecar (e.g. int16, as
the design doc's per-generation state contract does) does not need to cast
before passing it in" -- see its docstring), but this module's OWN
:func:`read_frame_codes_sidecar` also returns ``torch.int64`` unconditionally
(never the on-disk ``int16``), so every caller of this module -- not only
``generate_ar`` -- gets a use-ready tensor and the crash class cannot
recur here even if a future caller bypasses ``generate_ar``.

**Format version.** ``FRAME_CODES_FORMAT_VERSION`` is written into every
sidecar and checked on read: an unrecognized version is refused rather than
guessed at, since a silently misread frame-code array feeds directly into
embedding lookups on the language model and would not error until deep
inside a forward pass, or (worse) would silently degrade AR-resume quality
with no error at all.

The route that writes this sidecar alongside the generated audio file lands
in a later commit (design doc phase plan item 3 says the sidecar SHIP with
txt2aud; item 4 is the route layer that calls into this module). This module
is the sidecar itself: write / read / locate-next-to-an-audio-file, and the
identity fields (:meth:`MiniMaxMusic3FrameCodes.matches`) a future extend/
repaint request checks before trusting a sidecar found on disk.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from core.models.minimax_music3.defaults import FALLBACK_AUDIO_VOCAB_SIZE, SEMANTIC_VOCAB_SIZE

# Bumped whenever the on-disk JSON SHAPE changes (new/removed/renamed field,
# different encoding) -- NOT for a value-only change (e.g. a different
# sample_rate on a different checkpoint variant is a normal field value, not
# a format change).
FRAME_CODES_FORMAT_VERSION = 1

# Deliberately distinct from the generic `<base_name>.json` audio-metadata
# sidecar `utils.audio_utils.save_audio_with_metadata` writes -- see module
# docstring for why these are two separate files.
SIDECAR_SUFFIX = ".mm3frames.json"

# Checkpoint contract (design doc, "Architecture, as verified" /
# component census): one semantic code (vocab 16,384) plus seven residual
# codes (vocab 1,024 each) per frame. Not read from a component config here
# because this module must round-trip a sidecar even when no model is
# loaded (e.g. a gallery browsing session with no checkpoint resident) --
# `MiniMaxMusic3FrameCodes.matches` is what checks it against a LOADED
# pipeline's actual `num_codebooks`, not this module's own read/write path.
EXPECTED_NUM_CODEBOOKS = 8

# Per-column legal ranges (design doc, "Architecture, as verified"): column 0
# is the semantic code, columns 1..num_codebooks-1 are residual codes. A
# value outside its column's range is not merely out-of-spec -- `AUDIO_CODE_
# OFFSET + value` is STILL a valid index into the language model's
# 200,000-entry embedding table for values well past 16,383, so an
# out-of-range code does not error anywhere; it silently embeds as an
# ordinary text token during a teacher-forced AR-resume replay. This is why
# both write AND read validate every column, not just the flat int16 range
# `_tensor_to_int16_b64` also checks (that check alone would accept, and this
# module previously DID accept, values well inside int16 but outside a real
# code's legal range).
SEMANTIC_CODE_MAX = SEMANTIC_VOCAB_SIZE - 1
RESIDUAL_CODE_MAX = FALLBACK_AUDIO_VOCAB_SIZE - 1

# int16 range check on write: kept as a second, independent layer under the
# per-column check above -- it is what would catch a future checkpoint
# variant whose vocabulary genuinely exceeds int16 (in which case the
# per-column constants above would also need updating, but this still fails
# loudly instead of wrapping around).
_INT16_MAX = 32767


def sidecar_path_for_audio(audio_path: str) -> str:
    """The sidecar path this module writes/reads for a given audio file path.

    Strips the audio file's own extension (whatever it is -- FLAC today,
    per `utils.audio_utils.save_audio_with_metadata`) and appends
    `SIDECAR_SUFFIX`, mirroring how the generic `<base_name>.json` sidecar
    is already located relative to the audio file elsewhere in this repo.
    """
    base, _ext = os.path.splitext(audio_path)
    return base + SIDECAR_SUFFIX


@dataclass
class MiniMaxMusic3FrameCodes:
    """In-memory form of the sidecar. Codes are ALWAYS `torch.int64` here --
    see the module docstring's dtype contract; the on-disk representation
    (int16) is an implementation detail of `write_frame_codes_sidecar`/
    `read_frame_codes_sidecar` and never escapes to a caller.
    """

    frame_codes: torch.Tensor  # [num_frames, num_codebooks], torch.int64
    prefix_codes: torch.Tensor  # [1, num_codebooks], torch.int64
    sample_rate: int
    frame_rate: float
    prompt: str
    lyrics: str
    seed: int
    # `num_samples` is the AUDIO-FILE identity check: the decoded waveform's
    # own sample count at write time. Unlike sample_rate/frame_rate/
    # model_hash (all checkpoint-level constants that are identical for
    # every song from the same model), num_samples is per-song -- it is what
    # lets `matches()` notice "the file currently at this path is not the
    # file this sidecar was written for" (a later generation overwriting the
    # same path, a copy that swapped only one of the pair, etc). Required
    # (not optional) because it costs nothing beyond `waveform.shape[-1]`,
    # which every caller already has.
    num_samples: int = 0
    # `content_hash` is a STRONGER, optional identity check for a caller
    # that already has one available cheaply -- e.g. the same saved-file
    # bytes hash `routes.py` already computes for the gallery record.
    # Optional and skipped when falsy, mirroring `model_hash`'s convenience
    # default for a caller with no hash on hand.
    content_hash: str = ""
    # Identity fields, for `matches()` -- deliberately optional (a caller
    # without a loaded model, or an older sidecar, may not have them).
    model_hash: str = ""
    model_type: str = "minimax_music3"
    format_version: int = FRAME_CODES_FORMAT_VERSION

    @property
    def num_frames(self) -> int:
        return int(self.frame_codes.shape[0])

    @property
    def num_codebooks(self) -> int:
        return int(self.frame_codes.shape[-1])

    def matches(
        self,
        *,
        sample_rate: Optional[int] = None,
        frame_rate: Optional[float] = None,
        num_codebooks: Optional[int] = None,
        model_hash: Optional[str] = None,
        num_samples: Optional[int] = None,
        content_hash: Optional[str] = None,
        frame_rate_tolerance: float = 1e-3,
    ) -> bool:
        """Whether this sidecar is safe to resume against a currently-loaded
        model AND the audio file it claims to belong to. Every argument is
        optional so a caller can check only what it knows (e.g. `model_hash`
        is only available once a model is actually loaded); an argument that
        is `None` is not checked. `num_samples`/`content_hash` are not
        recomputed here (this module does no audio I/O) -- the CALLER reads
        or decodes the current audio file, computes them fresh, and passes
        them in for comparison against what was stored at write time. Returns
        `False` (never raises) on any mismatch -- a caller decides what
        refusing a resume means for its own request; this is a predicate,
        not a validator.
        """
        if sample_rate is not None and int(sample_rate) != self.sample_rate:
            return False
        if frame_rate is not None and abs(float(frame_rate) - self.frame_rate) > frame_rate_tolerance:
            return False
        if num_codebooks is not None and int(num_codebooks) != self.num_codebooks:
            return False
        if model_hash is not None and self.model_hash and model_hash != self.model_hash:
            return False
        if num_samples is not None and int(num_samples) != self.num_samples:
            return False
        if content_hash is not None and self.content_hash and content_hash != self.content_hash:
            return False
        return True


def _validate_code_columns(tensor: torch.Tensor, *, label: str) -> None:
    """Per-column range check -- see the module-level `SEMANTIC_CODE_MAX`/
    `RESIDUAL_CODE_MAX` comment for why this exists. Called on BOTH the
    write path (reject a bad tensor before it is ever written) and the read
    path (catch a hand-edited or otherwise corrupted sidecar). Raises
    `ValueError` naming the offending frame/column/value; never silently
    clamps or drops a value.
    """
    if tensor.numel() == 0:
        return
    array = tensor.detach().to("cpu")

    semantic = array[..., 0]
    bad_semantic = (semantic < 0) | (semantic > SEMANTIC_CODE_MAX)
    if bool(bad_semantic.any()):
        row = int(bad_semantic.nonzero()[0, 0])
        value = int(semantic[row])
        raise ValueError(
            f"{label}: semantic code (column 0) at frame {row} = {value}, outside the legal range "
            f"[0, {SEMANTIC_CODE_MAX}]."
        )

    if array.shape[-1] > 1:
        residual = array[..., 1:]
        bad_residual = (residual < 0) | (residual > RESIDUAL_CODE_MAX)
        if bool(bad_residual.any()):
            row, rel_col = (int(x) for x in bad_residual.nonzero()[0])
            col = rel_col + 1
            value = int(residual[row, rel_col])
            raise ValueError(
                f"{label}: residual code (column {col}) at frame {row} = {value}, outside the legal "
                f"range [0, {RESIDUAL_CODE_MAX}]."
            )


def _tensor_to_int16_b64(tensor: torch.Tensor, *, label: str) -> str:
    if torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError(
            f"{label} must be an integer-dtype tensor, got {tensor.dtype}. MiniMax Music 3 codes are "
            f"discrete indices; a float dtype would be silently truncated by the int16 narrowing this "
            f"function performs."
        )
    array = tensor.detach().to("cpu").to(torch.int64).numpy()
    if array.size > 0:
        max_value = int(array.max())
        min_value = int(array.min())
        if max_value > _INT16_MAX or min_value < 0:
            raise ValueError(
                f"{label}: value out of the sidecar's int16 range [0, {_INT16_MAX}]: found values in "
                f"[{min_value}, {max_value}]. This would silently corrupt the sidecar; refusing to "
                f"write it. (The released checkpoint's largest code vocabulary is 16,384, well inside "
                f"this range -- seeing this error means either a different checkpoint variant or a bug "
                f"upstream of this call.)"
            )
    int16_array = array.astype(np.int16)
    return base64.b64encode(int16_array.tobytes()).decode("ascii")


def _int16_b64_to_tensor(encoded: str, shape) -> torch.Tensor:
    raw = base64.b64decode(encoded.encode("ascii"))
    int16_array = np.frombuffer(raw, dtype=np.int16).reshape(shape)
    # Upcast to int64 HERE, unconditionally -- see module docstring's dtype
    # contract. Every reader of this module gets a use-ready tensor.
    return torch.from_numpy(int16_array.astype(np.int64)).clone()


def write_frame_codes_sidecar(
    audio_path: str,
    frame_codes: torch.Tensor,
    prefix_codes: torch.Tensor,
    *,
    sample_rate: int,
    frame_rate: float,
    prompt: str,
    lyrics: str,
    seed: int,
    num_samples: int,
    content_hash: str = "",
    model_hash: str = "",
) -> str:
    """Write the sidecar for `audio_path` and return the path written.

    `frame_codes`/`prefix_codes` are accepted as any integer-dtype tensor
    (the pipeline's own `generate`/`generate_ar` return `torch.int64`; this
    function does the int64 -> int16 narrowing for disk). Validated, in
    order, before anything is written: shape (`frame_codes` must be 2-D,
    `prefix_codes` must be `[1, num_codebooks]` matching `frame_codes`'s
    last dim), dtype (integer only), and per-column value range
    (`_validate_code_columns`) -- a malformed or out-of-range caller fails
    fast rather than producing a sidecar that reads back with silently wrong
    codes. `num_samples` is required (see `MiniMaxMusic3FrameCodes`'s field
    docstring for why); `content_hash`/`model_hash` are optional identity
    strengtheners.
    """
    if frame_codes.dim() != 2:
        raise ValueError(f"frame_codes must be 2-D [num_frames, num_codebooks], got shape {list(frame_codes.shape)}")
    if prefix_codes.dim() != 2 or prefix_codes.shape[0] != 1:
        raise ValueError(f"prefix_codes must be shape [1, num_codebooks], got shape {list(prefix_codes.shape)}")
    if prefix_codes.shape[-1] != frame_codes.shape[-1]:
        raise ValueError(
            f"prefix_codes' num_codebooks ({prefix_codes.shape[-1]}) does not match frame_codes' "
            f"({frame_codes.shape[-1]})"
        )
    _validate_code_columns(frame_codes, label="frame_codes")
    _validate_code_columns(prefix_codes, label="prefix_codes")

    payload = {
        "format_version": FRAME_CODES_FORMAT_VERSION,
        "model_type": "minimax_music3",
        "model_hash": model_hash,
        "sample_rate": int(sample_rate),
        "frame_rate": float(frame_rate),
        "prompt": prompt,
        "lyrics": lyrics,
        "seed": int(seed),
        "num_frames": int(frame_codes.shape[0]),
        "num_codebooks": int(frame_codes.shape[1]),
        "num_samples": int(num_samples),
        "content_hash": content_hash,
        "frame_codes_dtype": "int16",
        "frame_codes_b64": _tensor_to_int16_b64(frame_codes, label="frame_codes"),
        "prefix_codes_b64": _tensor_to_int16_b64(prefix_codes, label="prefix_codes"),
    }

    path = sidecar_path_for_audio(audio_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def read_frame_codes_sidecar(sidecar_path: str) -> MiniMaxMusic3FrameCodes:
    """Read a sidecar written by `write_frame_codes_sidecar`.

    Returns codes as `torch.int64` unconditionally (module docstring's
    dtype contract). Raises `ValueError` -- and ONLY `ValueError` -- for
    every recognized failure mode: an unrecognized `format_version`, a
    missing/malformed field, or an out-of-range code (`_validate_code_
    columns`, run on the codes AFTER they are decoded, so a hand-edited or
    otherwise corrupted sidecar is caught here too, not only at write time).
    Field access and base64/JSON decoding are wrapped so a missing key
    (`KeyError`) or a truncated/corrupt payload (`json.JSONDecodeError`,
    itself a `ValueError` subclass, or a decode-time `TypeError`/
    `binascii.Error`) all surface the same way, matching this docstring
    exactly rather than leaking whichever stdlib exception happened to fire.
    """
    try:
        with open(sidecar_path, encoding="utf-8") as fh:
            payload = json.load(fh)

        format_version = payload.get("format_version")
        if format_version != FRAME_CODES_FORMAT_VERSION:
            raise ValueError(
                f"MiniMax Music 3 frame-code sidecar at {sidecar_path!r} has format_version="
                f"{format_version!r}, expected {FRAME_CODES_FORMAT_VERSION!r}. Refusing to read it "
                f"rather than guess at a schema this reader was not written against."
            )
        if payload.get("frame_codes_dtype") != "int16":
            raise ValueError(
                f"MiniMax Music 3 frame-code sidecar at {sidecar_path!r} declares "
                f"frame_codes_dtype={payload.get('frame_codes_dtype')!r}, expected 'int16'."
            )

        num_frames = int(payload["num_frames"])
        num_codebooks = int(payload["num_codebooks"])

        frame_codes = _int16_b64_to_tensor(payload["frame_codes_b64"], (num_frames, num_codebooks))
        prefix_codes = _int16_b64_to_tensor(payload["prefix_codes_b64"], (1, num_codebooks))

        result = MiniMaxMusic3FrameCodes(
            frame_codes=frame_codes,
            prefix_codes=prefix_codes,
            sample_rate=int(payload["sample_rate"]),
            frame_rate=float(payload["frame_rate"]),
            prompt=payload.get("prompt", ""),
            lyrics=payload.get("lyrics", ""),
            seed=int(payload.get("seed", 0)),
            num_samples=int(payload.get("num_samples", 0)),
            content_hash=payload.get("content_hash", ""),
            model_hash=payload.get("model_hash", ""),
            model_type=payload.get("model_type", "minimax_music3"),
            format_version=format_version,
        )
    except ValueError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MiniMax Music 3 frame-code sidecar at {sidecar_path!r} is malformed ({type(exc).__name__}: "
            f"{exc}). Refusing to guess at missing or corrupted fields."
        ) from exc

    _validate_code_columns(result.frame_codes, label="frame_codes")
    _validate_code_columns(result.prefix_codes, label="prefix_codes")
    return result


def read_frame_codes_sidecar_for_audio(audio_path: str) -> Optional[MiniMaxMusic3FrameCodes]:
    """Locate + read the sidecar next to `audio_path`. `None` (not an
    exception) if no sidecar exists there -- a song generated before this
    feature shipped, or by a different architecture, simply has none; that
    is an ordinary state for a caller (e.g. the gallery's "can this be
    extended?" check) to handle, not an error.
    """
    path = sidecar_path_for_audio(audio_path)
    if not os.path.isfile(path):
        return None
    return read_frame_codes_sidecar(path)
