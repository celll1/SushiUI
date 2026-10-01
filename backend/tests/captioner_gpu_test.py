"""Opt-in tiny-model CUDA smoke; no pretrained weights or dataset mutation."""

import os
from pathlib import Path
import sys
import threading
import asyncio

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


def test_caption_trainer_cuda_pause_offload_and_continue(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from captioner_training_test import database, manager
    from core.captioner.trainer import CaptionTrainer
    from core.captioner.storage import read_json
    from core.captioner.artifact import load_bundle
    import core.gpu_coordinator as module
    from core.gpu_coordinator import GPUCoordinator

    coordinator = GPUCoordinator(resume_grace_sec=0.01)
    monkeypatch.setattr(module, "gpu_coordinator", coordinator)
    monkeypatch.setattr(coordinator, "_decide_for_handle", lambda *args: OffloadDecision("dram"))
    handles = []
    register = coordinator.register_trainer
    def capture(handle):
        handles.append(handle)
        register(handle)
    monkeypatch.setattr(coordinator, "register_trainer", capture)
    factory, datasets = database(tmp_path)
    tagger = manager()
    first_step, proceed = threading.Event(), threading.Event()
    stop = threading.Event()
    results, errors, events = [], [], []
    def progress(event):
        events.append(event)
        if event.get("phase") == "training" and event["step"] == 1:
            first_step.set()
            assert proceed.wait(10)
    root = tmp_path / "cuda-run"
    config = {"device": "cuda:0", "epochs": 2, "batch_size": 2, "hidden_dim": 12,
              "heads": 3, "layers": 1, "ffn_dim": 24, "visual_queries": 2,
              "max_length": 40, "vocab_size": 280, "val_ratio": 0.0, "test_ratio": 0.0,
              "near_duplicate_distance": 0, "max_num_patches": 4, "max_concepts": 2}
    def train():
        try:
            results.append(CaptionTrainer(run_id="cuda-pause", output_dir=root, config=config,
                dataset_configs=datasets, tagger_manager=tagger, session_factory=factory,
                progress=progress, stop_event=stop, alias_table={}).run())
        except BaseException as exc:
            errors.append(exc)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats(0)
    worker = threading.Thread(target=train)
    worker.start()
    try:
        assert first_step.wait(10), errors
        handle = handles[0]
        async def exercise():
            async def generation():
                async with coordinator.generation_slot(0.01, timeout=5, require_pause_ack=True):
                    assert handle.resumed_event.is_set()
                    assert all(p.device.type == "cpu" for p in handle._model.parameters())
                    assert handle._cached_opt_state is not None
                    assert len([e for e in events if e.get("phase") == "training"]) == 1
            task = asyncio.create_task(generation())
            assert await asyncio.to_thread(handle.pause_event.wait, 3)
            proceed.set()
            await asyncio.wait_for(task, 8)
        asyncio.run(exercise())
        worker.join(15)
        assert not worker.is_alive() and not errors, errors
        assert results[0]["step"] == 8 and not results[0]["stopped"]
        assert handle.restored_event.is_set()
        assert not coordinator._handles and coordinator._active_generations == 0
        assert all(p.device.type == "cpu" for p in tagger.model.parameters())
        checkpoint = root / read_json(root / "latest.json")["checkpoint"]
        model, _, _, _, state = load_bundle(checkpoint, require_resume=True)
        assert state["step"] == 8 and all(torch.isfinite(p).all() for p in model.parameters())
        assert torch.cuda.max_memory_allocated(0) < 128 * 1024**2
    finally:
        stop.set(); proceed.set()
        if handles:
            handles[0].pause_event.clear()
        worker.join(15)
        torch.set_num_threads(threads)
