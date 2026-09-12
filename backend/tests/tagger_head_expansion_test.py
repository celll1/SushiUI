"""Contracts shared by full and LoRA tagger head growth."""

import torch
from torch import nn

from core.tagger.siglip2_tagger_model import SigLIP2TaggerLoRAModel, SigLIP2TaggerModel


def _model_with_head(model_type: type[nn.Module]) -> nn.Module:
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.head = nn.Linear(4, 3)
    with torch.no_grad():
        model.head.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        model.head.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
    return model


def test_full_and_lora_head_growth_preserve_rows_and_zero_new_tags():
    for model_type in (SigLIP2TaggerModel, SigLIP2TaggerLoRAModel):
        model = _model_with_head(model_type)
        old_weight = model.head.weight.detach().clone()
        old_bias = model.head.bias.detach().clone()

        weight, bias = model.expand_head(5)

        assert weight is model.head.weight
        assert bias is model.head.bias
        assert weight.shape == (5, 4)
        assert bias.shape == (5,)
        assert weight.dtype == old_weight.dtype
        assert weight.device == old_weight.device
        assert torch.equal(weight[:3], old_weight)
        assert torch.equal(bias[:3], old_bias)
        assert torch.count_nonzero(weight[3:]) == 0
        assert torch.count_nonzero(bias[3:]) == 0


def test_head_growth_rejects_non_growth():
    model = _model_with_head(SigLIP2TaggerModel)
    try:
        model.expand_head(3)
    except AssertionError:
        return
    raise AssertionError("expand_head accepted an unchanged output size")
