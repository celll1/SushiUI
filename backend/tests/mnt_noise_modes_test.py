import pytest
import torch

from core.training.mnt import (
    MNTNoiseWindow,
    normalize_mnt_noise_mode,
    training_noise_like,
)


CPU = torch.device("cpu")


def test_mode_validation_and_legacy_alias():
    assert normalize_mnt_noise_mode("trajectory_blend") == "trajectory"
    with pytest.raises(ValueError, match="multi_noise_mode"):
        normalize_mnt_noise_mode("decorative-only")
    with pytest.raises(ValueError, match="trajectory_blend_alpha"):
        MNTNoiseWindow("trajectory", 4, 1.01)


def test_independent_keeps_the_architecture_local_legacy_draw():
    window = MNTNoiseWindow("independent", 4)
    assert window.noise_for(0, torch.zeros(2, 3, device=CPU)) is None


def test_shared_uses_one_exact_noise_trajectory():
    window = MNTNoiseWindow("shared", 4)
    reference = torch.zeros(2, 3, device=CPU)
    draws = [window.noise_for(i, reference) for i in range(4)]
    assert all(draw is draws[0] for draw in draws)


def test_trajectory_preserves_variance_and_requested_anchor_correlation():
    torch.manual_seed(1234)
    reference = torch.zeros(200_000, device=CPU)
    alpha = 0.7
    window = MNTNoiseWindow("trajectory", 2, alpha)
    first = window.noise_for(0, reference)
    second = window.noise_for(1, reference)
    assert first.std().item() == pytest.approx(1.0, abs=0.01)
    assert second.std().item() == pytest.approx(1.0, abs=0.01)
    corr = torch.corrcoef(torch.stack((first, second)))[0, 1].item()
    # Both draws share alpha * anchor, hence pairwise correlation alpha^2.
    assert corr == pytest.approx(alpha * alpha, abs=0.015)


def test_trajectory_endpoints_match_independent_and_shared_semantics():
    reference = torch.zeros(8, device=CPU)
    independent = MNTNoiseWindow("trajectory", 2, 0.0)
    assert independent.noise_for(0, reference) is not independent.noise_for(1, reference)
    shared = MNTNoiseWindow("trajectory", 2, 1.0)
    assert shared.noise_for(0, reference) is shared.noise_for(1, reference)


def test_antithetic_mode_returns_exact_sign_opposite_pairs():
    window = MNTNoiseWindow("antithetic", 4)
    reference = torch.zeros(7, device=CPU)
    draws = [window.noise_for(i, reference) for i in range(4)]
    assert torch.equal(draws[1], -draws[0])
    assert torch.equal(draws[3], -draws[2])
    assert not torch.equal(draws[0], draws[2])


def test_train_step_noise_moves_override_and_rejects_wrong_shape():
    class Trainer:
        pass

    trainer = Trainer()
    trainer._active_mnt_noise = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    reference = torch.zeros(2, 3, dtype=torch.float64)
    actual = training_noise_like(trainer, reference)
    assert actual.dtype == torch.float64
    assert torch.equal(actual, trainer._active_mnt_noise.double())

    trainer._active_mnt_noise = torch.zeros(3)
    with pytest.raises(ValueError, match="MNT noise shape"):
        training_noise_like(trainer, reference)
