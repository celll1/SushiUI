"""Records what the training loop was doing when a gradient spiked.

A gradient spike is unrepeatable: by the time it shows up as a step in the loss
chart, the batch that produced it is gone and nothing on disk says which images
or captions were in it. run127 hit a global gradient norm of 17,334 against a
running median of 3.8 at step 12,355 and never recovered, and there is no way to
answer "which sample was that" after the fact.

So this watches the norm the trainer already computes each optimizer step,
against a MEDIAN of a trailing window rather than a mean -- the spike itself
would drag a mean up and hide its own successors -- and writes one JSON line
per spike carrying the batch: image paths, bucket size, captions, the timesteps
drawn, the loss and the learning rate at that moment.

It never changes what the run does. Clipping is `optimizers/fused_grad_clip.py`.
"""

from __future__ import annotations

import json
import statistics
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Trailing window the baseline median is taken over. Long enough that a burst
#: of consecutive spikes cannot become the baseline, short enough to follow the
#: real decline in gradient scale over training.
_WINDOW = 200

#: Steps of history required before anything can be called a spike. Below this
#: the median is a guess, and the first steps of a run are legitimately noisy
#: (run127's own step 509 and 1,059 carried norms of 1,537 and 1,158).
_MIN_HISTORY = 50

#: Captions are the useful half of the record and can be thousands of
#: characters; enough to identify the sample, not enough to fill the file.
_CAPTION_CHARS = 400

#: Records written before the file stops growing. A run that spikes on every
#: step is broken in a way the first few hundred records already document, and
#: an unbounded JSONL beside a 33 GB checkpoint set helps nobody.
_MAX_RECORDS = 500


class GradSpikeLog:
    """Detects outlier gradient norms and records their batch context."""

    def __init__(self, output_dir: Any, factor: float, log_prefix: str = "[Trainer]",
                 window: int = _WINDOW, min_history: int = _MIN_HISTORY,
                 max_records: int = _MAX_RECORDS):
        self.factor = float(factor)
        self.window = int(window)
        self.min_history = int(min_history)
        self.log_prefix = log_prefix
        self.path = Path(output_dir) / "grad_spikes.jsonl"
        self._history: deque = deque(maxlen=self.window)
        self.spikes_recorded = 0
        self.max_records = int(max_records)
        self._write_failed = False
        self._capped = False

    @property
    def enabled(self) -> bool:
        return self.factor > 0

    def baseline(self) -> Optional[float]:
        """The trailing median, or None while the window is still filling."""
        if len(self._history) < self.min_history:
            return None
        return statistics.median(self._history)

    def observe(
        self,
        grad_norm: Optional[float],
        *,
        step: int,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        batch: Optional[Sequence] = None,
        timesteps: Any = None,
        clip_summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Take this step's norm. Returns the record written, or None.

        A clipped step is recorded whatever its norm: the clip is what kept the
        norm down, so judging it by the resulting number would hide exactly the
        events this exists to catch.
        """
        if not self.enabled:
            return None
        norm = _finite(grad_norm)
        base = self.baseline()
        if norm is not None:
            self._history.append(norm)
        clipped = bool(clip_summary and clip_summary.get("clipped_parameters"))
        spiked = (norm is not None and base is not None
                  and base > 0 and norm > base * self.factor)
        if not (spiked or clipped):
            return None

        record: Dict[str, Any] = {
            "step": int(step),
            "epoch": (int(epoch) if epoch is not None else None),
            "grad_norm": norm,
            "baseline_median": base,
            "ratio": (norm / base if norm is not None and base else None),
            "loss": _finite(loss),
            "learning_rate": _finite(learning_rate),
            "timesteps": _timesteps(timesteps),
            "batch": _batch_context(batch),
        }
        if clip_summary:
            record["clip"] = dict(clip_summary)
        self.spikes_recorded += 1
        self._write(record)
        return record

    def _write(self, record: Dict[str, Any]) -> None:
        if self.spikes_recorded > self.max_records:
            if not self._capped:
                self._capped = True
                print(f"{self.log_prefix} gradient spike log reached "
                      f"{self.max_records} records; the file stops here. Every "
                      f"spike is still counted and reported on this channel.")
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            if not self._write_failed:
                self._write_failed = True
                print(f"{self.log_prefix} gradient spike log could not be written "
                      f"to {self.path} ({type(exc).__name__}: {exc}); spikes will "
                      f"still be reported on this channel (logged once)")


def _finite(value: Any) -> Optional[float]:
    """``float(value)``, or None for None/NaN/inf -- a spike record must not
    carry a number that json.dumps writes as bare ``NaN``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _timesteps(timesteps: Any) -> Optional[List[float]]:
    if timesteps is None:
        return None
    try:
        values = timesteps.detach().flatten().tolist()
    except AttributeError:
        try:
            values = list(timesteps)
        except TypeError:
            return None
    out = [_finite(v) for v in values[:16]]
    return [v for v in out if v is not None] or None


def _batch_context(batch: Optional[Sequence]) -> Optional[List[Dict[str, Any]]]:
    """The training loop's ``[(item, dataset), ...]`` as identifying fields.

    Tolerates a bare list of item dicts too, so a caller that has already
    unzipped the batch is not forced to rebuild it.
    """
    if not batch:
        return None
    out: List[Dict[str, Any]] = []
    for entry in batch:
        item = entry[0] if isinstance(entry, (tuple, list)) and entry else entry
        if not isinstance(item, dict):
            continue
        caption = item.get("caption") or ""
        out.append({
            "path": item.get("image_path") or item.get("video_path"),
            "width": item.get("width") or item.get("bucket_width"),
            "height": item.get("height") or item.get("bucket_height"),
            "caption": caption[:_CAPTION_CHARS],
            "caption_truncated": len(caption) > _CAPTION_CHARS,
        })
    return out or None
