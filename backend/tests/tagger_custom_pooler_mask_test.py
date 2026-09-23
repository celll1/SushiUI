"""CustomAttentionPooling must ignore NaFlex padding tokens."""

from types import SimpleNamespace

import torch
from torch import nn

from core.tagger.siglip2_tagger_model import CustomAttentionPooling, SigLIP2TaggerModel


class _IdentityEncoder(nn.Module):
    def forward(self, pixel_values, attention_mask=None, spatial_shapes=None):
        return SimpleNamespace(last_hidden_state=pixel_values)


def _padded(valid: torch.Tensor, pad_len: int, fill: float) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = valid.shape
    x = torch.full((1, n + pad_len, d), fill)
    x[0, :n] = valid
    mask = torch.zeros(1, n + pad_len, dtype=torch.int32)
    mask[0, :n] = 1
    return x, mask


def test_pooler_output_is_independent_of_padding_length_and_content():
    torch.manual_seed(0)
    pooler = CustomAttentionPooling(in_dim=8, cls_dim=16, hidden_proj_dim=12).eval()
    valid = torch.randn(5, 8)

    x_a, m_a = _padded(valid, pad_len=3, fill=0.0)
    x_b, m_b = _padded(valid, pad_len=11, fill=7.0)

    with torch.no_grad():
        out_a = pooler(x_a, m_a)
        out_b = pooler(x_b, m_b)
        out_ref = pooler(valid.unsqueeze(0))

    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert torch.allclose(out_a, out_ref, atol=1e-6)


def test_naflex_tagger_forwards_mask_to_custom_pooler():
    torch.manual_seed(0)
    model = SigLIP2TaggerModel(
        num_tags=4, vision_encoder=_IdentityEncoder(), hidden_size=8, cls_dim=16, is_naflex=True,
    ).eval()
    nn.init.normal_(model.head.weight)
    valid = torch.randn(5, 8)

    x_a, m_a = _padded(valid, pad_len=3, fill=0.0)
    x_b, m_b = _padded(valid, pad_len=11, fill=7.0)

    with torch.no_grad():
        logits_a = model(x_a, m_a, spatial_shapes=None)
        logits_b = model(x_b, m_b, spatial_shapes=None)

    assert torch.allclose(logits_a, logits_b, atol=1e-6)
