"""Request-local CFG role for Qwen branch-separated LoRA."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from core.adapters import LoRALinearLayer


BRANCH_MODE_SHARED = "shared"
BRANCH_MODE_COND_BASE = "cond_base_v1"
_role: ContextVar[str] = ContextVar("qwen_image_21_lora_role", default="cond")


@contextmanager
def qwen_lora_role(role: str):
    if role not in {"cond", "base"}:
        raise ValueError(f"Unknown Qwen LoRA role {role!r}")
    token = _role.set(role)
    try:
        yield
    finally:
        _role.reset(token)


class QwenCondLoRALinearLayer(LoRALinearLayer):
    """The conditional delta is absent for an empty or negative CFG branch."""

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        if _role.get() == "base":
            return x.new_zeros((*x.shape[:-1], self.original_module.out_features))
        return super().forward_delta(x)


def validate_cond_base_config(*, mode, arch, method, algorithm, weight_decompose,
                              drop_rate,
                              caption_dropout_sources=()):
    if mode == BRANCH_MODE_SHARED:
        return
    if mode != BRANCH_MODE_COND_BASE:
        raise ValueError(f"Unknown Qwen LoRA branch mode {mode!r}")
    if (arch != "qwen_image_21" or method != "lora" or algorithm != "lora"
            or weight_decompose):
        raise ValueError("Qwen cond/base branch mode requires Qwen-Image 2.1 ordinary LoRA training")
    if drop_rate is None or float(drop_rate) != 0.0:
        raise ValueError("Qwen cond/base branch mode requires cfg_uncond_drop_rate=0")
    if caption_dropout_sources:
        raise ValueError(
            "Qwen cond/base branch mode requires whole-caption dropout disabled: "
            + ", ".join(caption_dropout_sources)
        )
