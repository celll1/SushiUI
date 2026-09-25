"""Weighted multi-positive SupCon (design §8.1)."""

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_loss import positive_weights, weighted_supcon_loss


def _reference(emb, weights, tau):
    z = [[x / math.sqrt(sum(v * v for v in row)) for x in row] for row in emb]
    m = len(z)
    s = [[sum(a * b for a, b in zip(z[i], z[j])) / tau for j in range(m)] for i in range(m)]
    losses = []
    for i in range(m):
        total = sum(weights[i][j] for j in range(m) if j != i)
        if total == 0:
            continue
        log_den = math.log(sum(math.exp(s[i][a]) for a in range(m) if a != i))
        losses.append(-sum(weights[i][j] / total * (s[i][j] - log_den) for j in range(m) if j != i))
    return sum(losses) / len(losses), len(losses)


EMB = [[1.0, 0.0, 0.5], [0.8, 0.3, 0.1], [-0.2, 1.0, 0.4], [0.1, -0.7, 2.0], [0.4, 0.4, -0.3]]
# Anchor 3 has no positive; the diagonal (9.0) must be ignored.
W = [[9.0, 1.0, 0.3, 0.0, 0.0],
     [1.0, 0.0, 0.7, 0.0, 0.0],
     [0.3, 0.7, 0.0, 0.0, 0.5],
     [0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.5, 0.0, 0.0]]


def test_matches_hand_computed_loss_and_excludes_zero_positive_anchors():
    expected, used = _reference(EMB, [[0.0 if i == j else w for j, w in enumerate(r)] for i, r in enumerate(W)], 0.1)
    result = weighted_supcon_loss(torch.tensor(EMB), torch.tensor(W), temperature=0.1)
    assert result.loss.dtype == torch.float32
    assert result.anchors_used.item() == used == 4 and result.anchors_excluded.item() == 1
    assert result.loss.item() == pytest.approx(expected, rel=1e-5)


def _reference_torch(emb, weights, tau):
    z = emb / emb.norm(dim=1, keepdim=True)
    m = z.shape[0]
    losses = []
    for i in range(m):
        others = [a for a in range(m) if a != i]
        total = sum(float(weights[i][j]) for j in others)
        if total == 0:
            continue
        log_den = torch.log(sum(torch.exp(z[i] @ z[a] / tau) for a in others))
        losses.append(-sum(float(weights[i][j]) / total * (z[i] @ z[j] / tau - log_den) for j in others))
    return sum(losses) / len(losses)


def test_gradient_matches_a_float64_loop_reference():
    emb = torch.tensor(EMB, requires_grad=True)
    weighted_supcon_loss(emb, torch.tensor(W), temperature=0.05).loss.backward()
    ref = torch.tensor(EMB, dtype=torch.float64, requires_grad=True)
    _reference_torch(ref, W, 0.05).backward()
    assert torch.isfinite(emb.grad).all() and emb.grad.abs().sum() > 0
    torch.testing.assert_close(emb.grad.double(), ref.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_input_is_computed_in_fp32(dtype):
    torch.manual_seed(0)
    emb = torch.randn(12, 16)
    weights = positive_weights([0, 0, 1, 1, 2, 2], [[1], [1, 2], [], [3], [3], []], 1.0, 0.3)
    low = weighted_supcon_loss(emb.to(dtype), weights, temperature=0.1)
    upcast = weighted_supcon_loss(emb.to(dtype).float(), weights, temperature=0.1)
    full = weighted_supcon_loss(emb, weights, temperature=0.1)
    assert low.loss.dtype == torch.float32 and torch.equal(low.loss, upcast.loss)
    assert low.anchors_excluded.item() == 0
    assert low.loss.item() == pytest.approx(full.loss.item(), rel=2e-2)
