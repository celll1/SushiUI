"""CPU-only activation-dispatch configuration behavior."""

import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.base_trainer import (
    _resolve_activation_dispatch_settings,
    activation_dispatch_batch_reduction_enabled,
    activation_dispatch_can_microbatch,
)


def _fallbacks():
    return {
        "enable": False,
        "allow_batch_reduction": True,
        "margin_gb": 1.0,
        "seed_coef": 24.0e-6,
        "residual_frac": 0.85,
        "threshold_mb": 4,
    }


def test_run_config_overrides_constructor_fallbacks_for_adapter_trainers():
    resolved = _resolve_activation_dispatch_settings({
        "activation_dispatch_enable": True,
        "activation_dispatch_allow_batch_reduction": False,
        "activation_dispatch_margin_gb": 2.5,
        "activation_dispatch_seed_coef": 0.125,
        "activation_dispatch_residual_frac": 0.6,
        "activation_dispatch_threshold_mb": 9,
    }, **_fallbacks())

    assert resolved == {
        "enable": True,
        "allow_batch_reduction": False,
        "margin_gb": 2.5,
        "seed_coef": 0.125,
        "residual_frac": 0.6,
        "threshold_mb": 9,
    }


def test_direct_callers_retain_explicit_constructor_values():
    assert _resolve_activation_dispatch_settings(None, **{
        "enable": True,
        "allow_batch_reduction": False,
        "margin_gb": 3.0,
        "seed_coef": 0.25,
        "residual_frac": 0.5,
        "threshold_mb": 7,
    }) == {
        "enable": True,
        "allow_batch_reduction": False,
        "margin_gb": 3.0,
        "seed_coef": 0.25,
        "residual_frac": 0.5,
        "threshold_mb": 7,
    }


def test_batch_reduction_policy_is_independent_from_dispatch_enable():
    trainer = SimpleNamespace(
        activation_dispatch_allow_batch_reduction=False,
        use_fused_backward=False,
        fused_optimizer_groups=None,
    )
    assert not activation_dispatch_batch_reduction_enabled(trainer)
    assert not activation_dispatch_can_microbatch(trainer)


def test_fused_backward_never_microbatches_even_when_policy_allows_it():
    trainer = SimpleNamespace(
        activation_dispatch_allow_batch_reduction=True,
        use_fused_backward=True,
        fused_optimizer_groups=None,
    )
    assert activation_dispatch_batch_reduction_enabled(trainer)
    assert not activation_dispatch_can_microbatch(trainer)
