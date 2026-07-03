"""
Generation phase timing collection.

A single generation runs one-at-a-time behind the ``gpu_coordinator`` generation
slot, so a process-wide singleton timer is safe: there is never more than one
generation populating it concurrently. The pipeline layer records the phases it
can cleanly bound (text encoding, denoising, VAE decode); ``routes.py`` measures
the total wall time around the whole generation call and merges everything into a
timing dict for PNG chunks / DB parameters / gallery display.

Contract:
    * ``generation_timer.reset()``  -- called by routes before a generation.
    * ``with generation_timer.phase("text_encode"): ...`` -- called by pipelines
      to accumulate a phase's elapsed seconds (repeated entries accumulate, e.g.
      per-step or CFG-doubled encode calls).
    * ``generation_timer.phases_dict()`` -- returns the recorded phases as
      ``{"time_<phase>": seconds}`` with 3-decimal rounding.

The phase keys the pipelines use map to metadata keys as:
    "text_encode" -> time_text_encode
    "denoise"     -> time_denoise
    "vae_decode"  -> time_vae_decode
"""

import time
from contextlib import contextmanager
from typing import Dict


# Canonical phase name -> metadata key. Pipelines pass the canonical name to
# ``phase()``; only these are surfaced (an unknown name is still timed but keyed
# verbatim as ``time_<name>`` so nothing is silently dropped).
_PHASE_KEYS = {
    "text_encode": "time_text_encode",
    "denoise": "time_denoise",
    "vae_decode": "time_vae_decode",
}


class GenerationTimer:
    """Process-wide accumulator for generation phase durations."""

    def __init__(self) -> None:
        self._phases: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear all recorded phases. Call once before a generation begins."""
        self._phases = {}

    @contextmanager
    def phase(self, name: str):
        """Time a code block and accumulate it under ``name``.

        Accumulates (``+=``) so multiple calls for the same phase within one
        generation (e.g. conditional + unconditional text encoding) sum up.
        Never raises out of the timing machinery itself.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._phases[name] = self._phases.get(name, 0.0) + elapsed

    def add(self, name: str, seconds: float) -> None:
        """Manually accumulate ``seconds`` under ``name`` (non-context use)."""
        self._phases[name] = self._phases.get(name, 0.0) + float(seconds)

    def phases_dict(self) -> Dict[str, float]:
        """Return recorded phases as ``{"time_<phase>": rounded_seconds}``."""
        out: Dict[str, float] = {}
        for name, seconds in self._phases.items():
            key = _PHASE_KEYS.get(name, f"time_{name}")
            out[key] = round(float(seconds), 3)
        return out


# Process-wide singleton (see module docstring for the concurrency argument).
generation_timer = GenerationTimer()


def time_phase(name: str):
    """Decorator: accumulate a function's wall time under phase ``name``.

    Used by the functional pipeline-ops modules (anima/lens/ideogram4/krea2/
    minit2i) whose encode/denoise/decode stages are free functions — decorating
    the definition covers every endpoint (txt2img/img2img/inpaint) that calls it.
    """
    def _wrap(fn):
        import functools

        @functools.wraps(fn)
        def _inner(*args, **kwargs):
            with generation_timer.phase(name):
                return fn(*args, **kwargs)
        return _inner
    return _wrap
