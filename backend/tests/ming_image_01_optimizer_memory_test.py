"""Ming Full uses bounded AdamW temporaries without changing component rates."""
import pytest
import torch
from optimizer_option_threading_test import _StubTrainer, _run_setup_optimizer
from core.training.optimizer_factory import OptimizerFactory


@pytest.mark.parametrize("full,ming,expected", [(True, True, False), (False, True, None), (True, False, None)])
def test_ming_full_only_bounds_adamw_temporaries(full, ming, expected, monkeypatch):
    monkeypatch.setattr("core.training.base_trainer.setup_fused_grad_clip", lambda *a, **k: None)
    trainer = _StubTrainer(optimizer_beta1=.85, optimizer_beta2=.95,
        optimizer_epsilon=1e-6, optimizer_weight_decay=.05)
    trainer.use_fused_backward = False
    trainer.fused_optimizer_groups = None
    trainer.trains_base_weights = full
    trainer.is_ming_image_01 = ming
    _run_setup_optimizer(trainer, optimizer_type="adamw", lr_scheduler_type="constant", total_steps=10)
    group = trainer.optimizer.param_groups[0]
    assert group["foreach"] is expected
    assert group["betas"] == (.85, .95)
    assert group["eps"] == 1e-6
    assert group["weight_decay"] == .05


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bounded_adamw_matches_scalar_reference_updates(dtype):
    generator = torch.Generator().manual_seed(172)
    actual = torch.nn.Parameter(torch.randn(32, generator=generator, dtype=dtype))
    reference = torch.nn.Parameter(actual.detach().clone())
    optimizer = OptimizerFactory.create_optimizer("adamw", [actual], 5e-5,
        betas=(.85, .95), eps=1e-6, weight_decay=.05, foreach=False)
    oracle = torch.optim.AdamW([reference], lr=5e-5, betas=(.85, .95),
        eps=1e-6, weight_decay=.05, foreach=False)
    for _ in range(3):
        actual.grad = torch.randn(actual.shape, generator=generator, dtype=dtype)
        reference.grad = actual.grad.clone()
        optimizer.step()
        oracle.step()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(optimizer.state[actual][key], oracle.state[reference][key], rtol=0, atol=0)
