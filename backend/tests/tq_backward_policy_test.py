"""CPU contract tests for SushiUI's per-run TQ backward policy."""

import os
import sys
from types import SimpleNamespace

import pytest
import yaml

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.attention import (
    get_tq_backward_policy,
    resolve_tq_backward_mode,
    set_tq_backward_policy,
)
from core.attention.backends import _tq_attn


@pytest.fixture(autouse=True)
def _restore_policy():
    previous = get_tq_backward_policy()
    yield
    value = "fa2_deterministic" if previous.fa2_deterministic else previous.mode
    set_tq_backward_policy(value)


@pytest.mark.parametrize(
    ("value", "mode", "deterministic"),
    [
        ("auto", "auto", False),
        ("triton", "triton", False),
        ("fa2", "fa2", False),
        ("fa2_deterministic", "fa2", True),
    ],
)
def test_policy_resolves_tq_arguments(value, mode, deterministic):
    policy = set_tq_backward_policy(value)
    assert (policy.mode, policy.fa2_deterministic) == (mode, deterministic)


def test_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="tq_backward_mode"):
        set_tq_backward_policy("fastish")


def test_missing_policy_uses_triton_for_all_configs():
    assert resolve_tq_backward_mode(None, resuming=False) == "triton"
    assert resolve_tq_backward_mode(None, resuming=True) == "triton"


def test_adapter_forwards_resolved_policy(monkeypatch):
    calls = {}

    class Tensor:
        def contiguous(self):
            return self

    def tq_spy(*args, **kwargs):
        calls.update(kwargs)
        return Tensor()

    monkeypatch.setitem(sys.modules, "tq_attention", SimpleNamespace(tq_attention=tq_spy))
    set_tq_backward_policy("fa2_deterministic")
    assert _tq_attn(Tensor(), Tensor(), Tensor()) is not None
    assert calls["backward_mode"] == "fa2"
    assert calls["fa2_deterministic"] is True


def test_api_and_openapi_share_the_new_run_default():
    from api.param_defaults import TRAINING_DEFAULTS
    from api.routes import TrainingRunCreateRequest

    assert TRAINING_DEFAULTS["tq_backward_mode"] == "triton"
    assert TrainingRunCreateRequest.model_fields["tq_backward_mode"].default == "triton"
    repo = os.path.abspath(os.path.join(_BACKEND, ".."))
    with open(os.path.join(repo, "openapi.yaml"), encoding="utf-8") as handle:
        properties = yaml.safe_load(handle)["components"]["schemas"][
            "TrainingRunCreateRequest"
        ]["properties"]
    assert properties["tq_backward_mode"]["default"] == "triton"
