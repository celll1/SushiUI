"""What a generation COST, on the row it produced (design §15 Gate 2).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/generation_cost_metadata_test.py -v

Wall time was already recorded (`generation_time`, plus the phase breakdown);
peak VRAM was not, so measuring it meant sampling `nvidia-smi` from outside.
Both now travel on the SAME channel -- the per-generation
`generation_timer` singleton -> `apply_generation_timings` -> `params` -> PNG
chunks / video sidecar / the DB row's `parameters` -- rather than on a second
measurement mechanism.

The trap this file exists for: MiniMax-H3 resets the CUDA peak counter between
its own phases, so a peak read at the end of the generation would only cover
the last one. The reset helper folds the previous peak in first.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.generation_utils import apply_generation_timings  # noqa: E402
from core.inference.generation_timing import generation_timer  # noqa: E402


def test_wall_time_and_phases_still_land_on_params():
    generation_timer.reset()
    with generation_timer.phase("denoise"):
        pass
    params = {}
    apply_generation_timings(params, 12.3456)
    assert params["generation_time"] == 12.346
    assert "time_denoise" in params


def test_peak_vram_is_reported_only_when_the_endpoint_armed_it():
    """No arming, no number -- never the PREVIOUS generation's peak."""
    generation_timer.reset()
    armed = {}
    apply_generation_timings(armed, 1.0)

    # An endpoint that never called `reset()` (the upscale/audio routes today).
    generation_timer._peak_armed = False
    unarmed = {}
    apply_generation_timings(unarmed, 1.0)
    assert "peak_vram_gb" not in unarmed

    import torch

    if torch.cuda.is_available():
        assert isinstance(armed["peak_vram_gb"], float)
    else:
        assert "peak_vram_gb" not in armed


def test_a_per_phase_peak_reset_folds_into_the_generation_peak():
    generation_timer.reset()
    generation_timer._peak_armed = True          # holds with or without a GPU
    generation_timer._peak_vram_bytes = 7 * 1024 ** 3
    generation_timer.note_peak_vram()            # a phase reset would call this
    assert generation_timer.peak_vram_dict()["peak_vram_gb"] >= 7.0


def test_the_minimax_h3_phase_reset_calls_the_fold():
    """Source-anchored: the fold and the reset must stay together.

    Split them and the number silently becomes "peak since the audio decode",
    which is smaller than the truth and indistinguishable from it.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "core", "pipeline_backends", "minimax_h3.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    body = source[source.index("def _minimax_h3_reset_peak_vram"):]
    body = body[:body.index("\n    @") if "\n    @" in body else 2000]
    assert "generation_timer.note_peak_vram()" in body
    assert body.index("note_peak_vram") < body.index("reset_peak_memory_stats")


def test_the_metadata_writer_carries_the_cost_keys():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "utils", "image_utils.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for key in ("generation_time", "peak_vram_gb"):
        assert f'"{key}"' in source
