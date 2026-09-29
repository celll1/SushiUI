"""CPU contracts for the SigLIP2 patch-budget benchmark."""

import pytest

from core.tagger.foveatag.siglip2_patch_benchmark import timing_summary


def test_timing_summary_reports_tail_latency():
    summary = timing_summary([1.0, 2.0, 3.0, 9.0])
    assert summary == {"count": 4, "mean": 3.75, "p50": 2.5, "p95": 9.0, "max": 9.0}


def test_timing_summary_rejects_empty_samples():
    with pytest.raises(ValueError, match="must not be empty"):
        timing_summary([])
