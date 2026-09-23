"""Tagger models must drive a real transformers-5.x Siglip2VisionModel on the NaFlex path."""

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from transformers import Siglip2VisionConfig, Siglip2VisionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tagger.siglip2_tagger_model import SigLIP2TaggerLoRAModel, SigLIP2TaggerModel

HIDDEN = 64
PATCH_DIM = 3 * 4 * 4


def _tiny_encoder() -> Siglip2VisionModel:
    torch.manual_seed(0)
    config = Siglip2VisionConfig(
        hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, patch_size=4, num_patches=16,
    )
    return Siglip2VisionModel(config)


def _naflex_batch():
    """Two rows holding the same 2x3-patch image; only the padding content differs."""
    torch.manual_seed(1)
    valid = torch.randn(6, PATCH_DIM)
    pv = torch.zeros(2, 10, PATCH_DIM)
    pv[:, :6] = valid
    pv[0, 6:] = 5.0
    pv[1, 6:] = -3.0
    mask = torch.zeros(2, 10, dtype=torch.int32)
    mask[:, :6] = 1
    shapes = torch.tensor([[2, 3], [2, 3]], dtype=torch.int64)
    return pv, mask, shapes


def _build(kind: str) -> nn.Module:
    enc = _tiny_encoder()
    if kind == "full_default":
        model = SigLIP2TaggerModel(num_tags=5, vision_encoder=enc, hidden_size=HIDDEN, is_naflex=True)
    elif kind == "full_custom_pooler":
        model = SigLIP2TaggerModel(num_tags=5, vision_encoder=enc, hidden_size=HIDDEN,
                                   cls_dim=32, hidden_proj_dim=48, is_naflex=True)
    else:
        model = SigLIP2TaggerLoRAModel(num_tags=5, vision_encoder=enc, lora_rank=4,
                                       lora_alpha=4.0, hidden_size=HIDDEN, is_naflex=True)
        for lm in model._lora_modules.values():
            nn.init.normal_(lm.lora_B, std=0.02)
    nn.init.normal_(model.head.weight)
    return model


@pytest.mark.parametrize("kind", ["full_default", "full_custom_pooler", "lora"])
def test_naflex_forward_backward_and_padding_invariance(kind):
    model = _build(kind)
    pv, mask, shapes = _naflex_batch()

    model.eval()
    with torch.no_grad():
        logits = model(pv, mask, shapes)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits[0], logits[1], atol=1e-5), "padding leaked into logits"

    model.train()
    model(pv, mask, shapes).square().mean().backward()
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    # vision_encoder.head is bypassed by the custom pooler, so only the layers are checked.
    encoder_grads = [p.grad for n, p in trainable if n.startswith("vision_encoder.encoder.")]
    assert encoder_grads, "no trainable encoder parameters"
    assert all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
    assert any(g.abs().sum() > 0 for g in encoder_grads)
