"""Normalized image-grid feature alignment for structural initialization probes."""
import torch
from torch.nn import functional as F


def teacher_grid(teacher, student):
    if teacher.ndim != 5 or student.ndim != 5:
        raise ValueError("Feature alignment requires B,T,H,W,C grids")
    if teacher.shape[1] != 1 or student.shape[1] != 1:
        raise ValueError("Feature alignment currently supports single image frames")
    if teacher.shape[:2] != student.shape[:2] or teacher.shape[-1] != student.shape[-1]:
        raise ValueError("Feature batch/frame/channel dimensions must match")
    th, tw = teacher.shape[2:4]
    sh, sw = student.shape[2:4]
    if min(teacher.shape) <= 0 or min(student.shape) <= 0 or th % sh or tw % sw:
        raise ValueError("Teacher grid must divide exactly into the student grid")
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError("Feature alignment requires finite tensors")
    target = teacher.detach().float()
    if (th, tw) != (sh, sw):
        target = F.avg_pool2d(target[:, 0].permute(0, 3, 1, 2), (th // sh, tw // sw))
        target = target.permute(0, 2, 3, 1).unsqueeze(1)
    return target.to(student.device)


def normalized_feature_mse(student, teacher):
    target = teacher_grid(teacher, student)
    axes = tuple(range(1, target.ndim))
    energy = target.square().mean(dim=axes).clamp_min(torch.finfo(torch.float32).eps)
    error = (student.float() - target).square().mean(dim=axes)
    return (error / energy).mean()


def feature_metrics(student, teacher):
    with torch.no_grad():
        target = teacher_grid(teacher, student)
        value = student.detach().float()
        return dict(relative_mse=float(normalized_feature_mse(value, target)),
                    cosine=float(F.cosine_similarity(value, target, dim=-1).mean()),
                    norm_ratio=float(value.square().mean().sqrt() /
                        target.square().mean().clamp_min(torch.finfo(torch.float32).eps).sqrt()),
                    teacher_shape=list(teacher.shape), student_shape=list(student.shape))
