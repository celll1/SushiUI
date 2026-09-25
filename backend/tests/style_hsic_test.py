"""Phase 3 HSIC contracts (design T6 and T7)."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_hsic import category_columns
from core.style_encoder.style_loss import normalized_hsic


def test_hsic_detects_dependence_and_only_differentiates_the_style_side():
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(512, 4, generator=generator)
    tag = (source + 0.05 * torch.randn(512, 4, generator=generator)).requires_grad_()
    independent = torch.randn(512, 4, generator=generator)
    style = source.clone().requires_grad_()
    dependent_score = normalized_hsic(style, tag)
    independent_score = normalized_hsic(style, independent)
    assert dependent_score > 0.8
    assert independent_score < 0.05
    dependent_score.backward()
    assert style.grad is not None and torch.isfinite(style.grad).all()
    assert tag.grad is None


def test_hsic_refuses_missing_positive_category_and_artist_vocabulary():
    vocab = {"tag_to_idx": {"hero": 0}, "tag_to_category": {"hero": "Character"}}
    with pytest.raises(ValueError, match="no Copyright columns"):
        category_columns(vocab, {"Character": 0.1, "Copyright": 0.1})
    assert category_columns(vocab, {"Character": 0.1, "Copyright": 0.0})["Character"].tolist() == [0]
    with pytest.raises(ValueError, match="Artist tags"):
        category_columns({"tag_to_idx": {"x": 0}, "tag_to_category": {"x": "Artist"}},
                         {"Character": 0.1})
