"""CPU checks for the decoupled SDXL style cross-attention bridge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from types import SimpleNamespace
import numpy as np
from diffusers.models.attention_processor import Attention

from core.inference.attention_processors import UnifiedAttnProcessor, set_style_ip_context
from core.style_encoder.sdxl_ip_adapter import (
    SDXLStyleIPAdapter, load_sdxl_style_ip_adapter, save_sdxl_style_ip_adapter,
)
from core.style_encoder.sdxl_ip_training import (
    choose_style_reference_rows, sdxl_style_ip_denoising_loss, install_style_ip_training_processors,
)
from core.style_encoder.style_dataset import SplitParams, build_style_index
from core.style_encoder.style_labels import StyleLabels


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4)
        self.attn2.set_processor(UnifiedAttnProcessor())

    @property
    def attn_processors(self):
        return {"attn2.processor": self.attn2.processor}

    def set_attn_processor(self, processors):
        self.attn2.set_processor(processors["attn2.processor"])

    def forward(self, noisy, _timesteps, *, encoder_hidden_states, added_cond_kwargs):
        return SimpleNamespace(sample=self.attn2(noisy, encoder_hidden_states=encoder_hidden_states))


def test_style_bridge_is_decoupled_and_scale_zero_preserves_baseline():
    torch.manual_seed(7)
    unet = TinyUNet().eval().requires_grad_(False)
    adapter = SDXLStyleIPAdapter(unet, num_tokens=2)
    query = torch.randn(1, 3, 8)
    text = torch.randn(1, 4, 12)
    embedding = torch.nn.functional.normalize(torch.randn(1, 512), dim=-1)

    baseline = unet.attn2(query, encoder_hidden_states=text)
    set_style_ip_context(unet, adapter.context(embedding, scale=0))
    torch.testing.assert_close(unet.attn2(query, encoder_hidden_states=text), baseline, rtol=0, atol=0)

    set_style_ip_context(unet, adapter.context(embedding))
    styled = unet.attn2(query, encoder_hidden_states=text)
    assert not torch.equal(styled, baseline)
    styled.square().mean().backward()
    assert adapter.projection.weight.grad is not None
    assert adapter.to_k_ip[0].weight.grad is not None
    assert adapter.to_v_ip[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in unet.parameters())

    set_style_ip_context(unet, None)
    torch.testing.assert_close(unet.attn2(query, encoder_hidden_states=text), baseline, rtol=0, atol=0)


def test_style_bridge_only_conditions_positive_cfg_half():
    torch.manual_seed(11)
    unet = TinyUNet().eval()
    adapter = SDXLStyleIPAdapter(unet)
    query = torch.randn(2, 3, 8)
    text = torch.randn(2, 4, 12)
    baseline = unet.attn2(query, encoder_hidden_states=text)
    set_style_ip_context(unet, adapter.context(torch.randn(1, 512), cfg_batch=True))
    styled = unet.attn2(query, encoder_hidden_states=text)
    torch.testing.assert_close(styled[0], baseline[0], rtol=0, atol=0)
    assert not torch.equal(styled[1], baseline[1])

    set_style_ip_context(unet, adapter.context(torch.randn(2, 512), cfg_batch=False))
    ordinary_batch = unet.attn2(query, encoder_hidden_states=text)
    assert not torch.equal(ordinary_batch[0], baseline[0])
    assert not torch.equal(ordinary_batch[1], baseline[1])


def test_style_bridge_rejects_mismatched_architecture():
    adapter = SDXLStyleIPAdapter(TinyUNet())
    metadata = adapter.architecture()
    metadata["layer_widths"] = [16]
    try:
        adapter.check_architecture(metadata)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched U-Net should be rejected")


def test_style_bridge_artifact_checks_both_source_models(tmp_path):
    unet = TinyUNet()
    adapter = SDXLStyleIPAdapter(unet)
    path = tmp_path / "style_ip.safetensors"
    encoder_digest, sdxl_digest = "a" * 64, "b" * 64
    save_sdxl_style_ip_adapter(adapter, str(path), style_encoder_sha256=encoder_digest,
                               sdxl_model_sha256=sdxl_digest)
    restored = load_sdxl_style_ip_adapter(unet, str(path), style_encoder_sha256=encoder_digest,
                                          sdxl_model_sha256=sdxl_digest)
    for key, value in adapter.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    try:
        load_sdxl_style_ip_adapter(unet, str(path), style_encoder_sha256="c" * 64,
                                   sdxl_model_sha256=sdxl_digest)
    except ValueError as exc:
        assert "different style encoder" in str(exc)
    else:
        raise AssertionError("mismatched style encoder should be rejected")


def test_training_step_updates_only_bridge_and_clears_context():
    class Scheduler:
        config = SimpleNamespace(prediction_type="epsilon")

        @staticmethod
        def add_noise(clean, noise, _timesteps):
            return clean + noise

    unet = TinyUNet().requires_grad_(False)
    adapter = SDXLStyleIPAdapter(unet)
    original = install_style_ip_training_processors(unet)
    assert original
    loss = sdxl_style_ip_denoising_loss(
        unet, Scheduler(), adapter, clean_latents=torch.randn(1, 3, 8),
        style_embedding=torch.randn(1, 512), prompt_embeds=torch.randn(1, 4, 12),
        added_cond_kwargs={}, timesteps=torch.tensor([4]), noise=torch.randn(1, 3, 8),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert adapter.projection.weight.grad is not None
    assert all(parameter.grad is None for parameter in unet.parameters())
    assert unet.attn2.processor._style_ip_ctx is None


def test_style_pair_prefers_another_work_with_same_artist():
    records = []
    for item_id, copyright_name in enumerate(("work A", "work A", "work B")):
        labels = StyleLabels(frozenset({"artist"}), frozenset(),
                             frozenset({copyright_name}), False)
        records.append((item_id, item_id, labels))
    index = build_style_index(records, SplitParams.create(0, 0, 2, 3),
                              pinned_keys=np.zeros(0, dtype=np.int64))
    references = choose_style_reference_rows(index, [0, 1], np.random.default_rng(4))
    assert references.tolist() == [2, 2]
