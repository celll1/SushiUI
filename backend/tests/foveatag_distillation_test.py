"""CPU tests for the optional local SigLIP2 distillation ablation."""

import torch

from core.tagger.foveatag.distillation import (
    LocalSigLIP2DistillationHead,
    pool_to_source_anchors,
)


def _coordinates(size):
    axis = (torch.arange(size, dtype=torch.float32) + 0.5) / size
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x, y), dim=-1).unsqueeze(0)


def test_exact_16_by_16_map_preserves_anchor_order():
    values = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16)
    anchors, populated = pool_to_source_anchors(
        values,
        _coordinates(16),
        torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0, 0.0]]),
    )

    torch.testing.assert_close(anchors[0, :, 0], torch.arange(256, dtype=torch.float32))
    assert populated.all()


def test_finer_map_is_averaged_within_source_anchors():
    values = torch.ones(1, 2, 32, 32)
    anchors, populated = pool_to_source_anchors(
        values,
        _coordinates(32),
        torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0, 0.0]]),
    )

    torch.testing.assert_close(anchors, torch.ones_like(anchors))
    assert populated.all()


def test_projection_head_receives_gradients_but_teacher_does_not():
    head = LocalSigLIP2DistillationHead(student_dim=4, teacher_dim=6)
    student = torch.randn(1, 4, 16, 16, requires_grad=True)
    teacher = torch.randn(1, 256, 6, requires_grad=True)
    output = head(
        student,
        _coordinates(16),
        torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0, 0.0]]),
        teacher,
    )

    output.loss.backward()
    assert student.grad is not None
    assert head.projection.weight.grad is not None
    assert teacher.grad is None


def test_stride_coarser_than_teacher_grid_is_rejected():
    head = LocalSigLIP2DistillationHead(student_dim=4, teacher_dim=6)
    try:
        head(
            torch.randn(1, 4, 8, 8),
            _coordinates(8),
            torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0, 0.0]]),
            torch.randn(1, 256, 6),
        )
    except ValueError as error:
        assert "at least 16x16" in str(error)
    else:
        raise AssertionError("expected an undersampled student map to fail")
