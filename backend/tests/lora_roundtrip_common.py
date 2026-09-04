"""Shared plumbing for the per-architecture cheap LoRA round-trip gates.

Only the parts that carry no architecture-specific meaning live here: the
sys.path bootstrap, the warning interception, and three tiny numeric helpers.
Stub trees, target sets and every assertion stay in the per-architecture file
so a failure names its architecture in the file name alone.

The gates are named ``<arch>_lora_roundtrip_cheap_test.py`` and each runs in
about a second on CPU with no real weights. Their sibling
``minimax_h3_lora_conversion_test.py`` is the counterexample: it builds a
50-block stub at the real widths (tens of GB of host RAM) and must not be run
casually or imitated.
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from core.adapters import LoRALinearLayer  # noqa: E402

__all__ = [
    "LoRALinearLayer",  # re-exported: every architecture's wrapper class
    "lora_delta",
    "module_ids",
    "randomise_branch_tensors",
    "randomise_lora_layers",
    "warning_codes",
    "warning_probe",
]


def randomise_lora_layers(layers, seed=1234, std=0.3):
    """Give both halves real values before saving.

    ``lora_up`` initialises to zeros, so a round trip over an untouched
    adapter passes even when the two halves are transposed or swapped, and
    every "the forward matched" claim is vacuous.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in layers.values():
            for weight in (layer.lora_down.weight, layer.lora_up.weight):
                weight.copy_(torch.randn(weight.shape, generator=generator) * std)
    return layers


def randomise_branch_tensors(layers, seed=1234, std=0.3):
    """``randomise_lora_layers`` for any algebra, through ``branch_tensors()``.

    LoHa's ``hada_w2_a`` and LoKr's zeroed operand start at zero for the same
    reason ``lora_up`` does, so an untouched adapter's delta is identically zero
    and every "the forward matched" claim would be vacuous.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in layers.values():
            for weight in layer.branch_tensors().values():
                if isinstance(weight, torch.nn.Parameter):
                    weight.copy_(
                        torch.randn(weight.shape, generator=generator) * std)
    return layers


def lora_delta(down, up, x, alpha, rank, strength):
    """(alpha/rank) * strength * up(down(x)), straight from the file's tensors."""
    return (alpha / rank) * strength * (x @ down.T @ up.T)


def module_ids(model):
    return {id(m) for _name, m in model.named_modules()}


def warning_probe(monkeypatch):
    """Intercept the channel every backend's warn() writes to.

    Returns the list the probe appends ``(code, message)`` to; the code is what
    reaches the response's ``warnings[]`` and the PNG metadata chunk.
    """
    import api.generation_status as status

    recorded = []
    monkeypatch.setattr(
        status, "add_warning",
        lambda message, code=None, **kwargs: recorded.append((code, message)))
    return recorded


def warning_codes(recorded):
    return [code for code, _message in recorded]
