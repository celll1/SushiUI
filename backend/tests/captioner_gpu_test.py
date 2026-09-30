"""Opt-in tiny-model CUDA smoke; no pretrained weights or dataset mutation."""

import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from transformers import Siglip2VisionConfig, Siglip2VisionModel

from core.captioner.concepts import ConceptSelector
from core.captioner.features import SigLIP2FeatureProvider
from core.captioner.model import CaptionConfig, CompactCaptioner
from core.gpu_coordinator import OffloadDecision
from core.tagger.siglip2_tagger_model import SigLIP2TaggerModel
from core.tagger.tagger_offload import TaggerTrainerHandle

pytestmark = pytest.mark.skipif(os.environ.get("SUSHI_CAPTIONER_GPU_TESTS") != "1",
                              reason="Set SUSHI_CAPTIONER_GPU_TESTS=1 after coordinating GPU use")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_tiny_siglip2_decoder_cuda_and_dram_restore(dtype, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    encoder = Siglip2VisionModel(Siglip2VisionConfig(hidden_size=24, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=3, patch_size=4, num_patches=16))
    tagger = SigLIP2TaggerModel(4, encoder, hidden_size=24).eval().requires_grad_(False).to(device)
    pixels = torch.randn(2, 8, 48, device=device)
    mask = torch.tensor([[1] * 6 + [0] * 2] * 2, device=device)
    shapes = torch.tensor([[2, 3]] * 2, device=device)
    features = SigLIP2FeatureProvider(tagger).extract(pixels, mask, shapes)
    selector = ConceptSelector(["blue hair", "red hair", "girl", "boy"], ["General"] * 4,
                               max_concepts=3, allowed_categories=["General"])
    concepts = selector.select(features.tag_logits, mode="predicted")
    model = CompactCaptioner(CaptionConfig(24, 24, 4, 12, 24, 3, 2, 48, 2, 8, 0.0,
                                          9, 0, 1, 2)).to(device=device, dtype=dtype)
    sequences = torch.tensor([[1, 3, 4, 2], [1, 4, 3, 2]], device=device)
    model.train()
    loss = model.teacher_loss(features, concepts, sequences, label_smoothing=0.05)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in tagger.parameters())
    if dtype == torch.float32:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        handle = TaggerTrainerHandle("gpu-smoke", str(tmp_path), threading.Event(),
                                    threading.Event(), threading.Event(), device_index=0)
        handle.attach(nn.ModuleDict({"captioner": model, "tagger": tagger}), optimizer, None)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        handle.offload(OffloadDecision("dram"))
        assert all(p.device.type == "cpu" for p in model.parameters())
        handle.restore()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(), weights[key], rtol=0, atol=0)
        assert all(p.device == device for p in model.parameters())
        resumed_loss = model.teacher_loss(features, concepts, sequences, label_smoothing=0.05)
        resumed_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        handle.detach()
    model.eval()
    result = model.generate(features, concepts, max_new_tokens=4, cancelled=lambda: False)
    tokens = torch.full((2, 1), 1, dtype=torch.long, device=device)
    memory, visible = model.encode_memory(features, concepts)
    done = torch.zeros(2, dtype=torch.bool, device=device)
    for position in range(result.token_ids.shape[1]):
        logits = model.decode(tokens, memory, visible)[:, -1]
        logits[:, :2] = -torch.inf
        predicted = logits.argmax(dim=-1).masked_fill(done, 0)
        assert torch.equal(predicted, result.token_ids[:, position])
        done |= predicted.eq(2)
        tokens = torch.cat((tokens, predicted[:, None]), dim=1)
    with pytest.raises(InterruptedError):
        model.generate(features, concepts, max_new_tokens=4, cancelled=lambda: True)
    assert torch.cuda.max_memory_allocated(device) < 128 * 1024**2
    model.cpu()
    tagger.cpu()
