"""The VAE/TE override gate must keep seeing dims the scan cannot observe.

``observed_components`` holds only what was read out of the file. Single-file
checkpoints carry no text-encoder width there, and several trees carry neither
that nor the VAE's latent channel count. The gate's HARD verdicts need a dim on
both sides, so sourcing it from observations alone turns a real mismatch into a
warning and lets the override run until it faults mid-generation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import generation_overrides as go


def _rec(observed, declared, **extra):
    rec = {"arch": "sdxl", "components": declared, "observed_components": observed}
    rec.update(extra)
    return rec


def test_declared_dim_fills_a_gap_the_observation_left():
    rec = _rec(
        observed={"text_encoder": {"te_type": "clip"}},
        declared={"text_encoder": {"te_type": "clip", "out_dim": 2048}},
    )
    assert go._component(rec, "text_encoder")["out_dim"] == 2048


def test_observation_wins_over_the_declared_value():
    rec = _rec(
        observed={"vae": {"latent_channels": 16}},
        declared={"vae": {"latent_channels": 4}},
    )
    assert go._component(rec, "vae")["latent_channels"] == 16


def test_an_explicit_none_observation_does_not_erase_the_declared_value():
    rec = _rec(
        observed={"vae": {"latent_channels": None, "present": True}},
        declared={"vae": {"latent_channels": 16}},
    )
    assert go._component(rec, "vae")["latent_channels"] == 16


def test_missing_observations_fall_back_wholesale():
    rec = _rec(observed={}, declared={"vae": {"latent_channels": 16}})
    assert go._component(rec, "vae")["latent_channels"] == 16


def test_absent_component_is_empty_not_an_error():
    assert go._component({"components": {}, "observed_components": {}}, "vae") == {}


@pytest.mark.parametrize("loaded_dim,candidate_dim,expect_hard", [
    (2048, 768, True),    # a real mismatch must still be refused outright
    (2048, 2048, False),
])
def test_te_gate_reaches_a_hard_verdict_on_fold_only_dims(
        monkeypatch, loaded_dim, candidate_dim, expect_hard):
    """End to end through the real checker, with the width in `components`
    only -- the shape every single-file checkpoint presents."""
    from api.error_handlers import ValidationError

    def fake_scan(path, source_type=None):
        dim = loaded_dim if path == "loaded" else candidate_dim
        return {
            "arch": "sdxl",
            "observed_components": {"text_encoder": {"te_type": "clip"}},
            "components": {"text_encoder": {"te_type": "clip", "out_dim": dim}},
        }

    monkeypatch.setattr(go, "_get_or_scan", fake_scan)
    monkeypatch.setattr(go, "_te_config_dir", lambda _path: None)
    monkeypatch.setattr(go, "_vae_config_dir", lambda _path: None)
    monkeypatch.setattr(go, "_warn", lambda *a, **k: None)

    loaded = go.describe_te("loaded")
    candidate = go.describe_te("candidate")

    if expect_hard:
        with pytest.raises(ValidationError) as excinfo:
            go._check_te_compat(loaded, candidate)
        assert str(loaded_dim) in str(excinfo.value.detail)
    else:
        go._check_te_compat(loaded, candidate)
