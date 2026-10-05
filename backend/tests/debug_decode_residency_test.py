"""Debug decode preserves ownership, gradients and next optimizer updates."""
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.debug_decode_residency import debug_decode_residency, DebugDecodeRestorationError


@pytest.mark.parametrize("fail", [False, True])
def test_cpu_decode_restores_vae_mode_and_dtype(fail):
    vae = torch.nn.Linear(2, 2).train()
    trainer = SimpleNamespace(device="cpu")
    with pytest.raises(RuntimeError, match="decode failed") if fail else nullcontext():
        with debug_decode_residency(trainer, vae, dtype=torch.float64):
            assert not vae.training and vae.weight.dtype == torch.float64
            if fail:
                raise RuntimeError("decode failed")
    assert vae.training and vae.weight.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("optimizer_kind", ["adamw", "adamw8bit"])
def test_cuda_decode_parks_all_components_and_preserves_next_update(fail, optimizer_kind):
    main = torch.nn.Linear(64, 64, device="cuda")
    main.register_buffer("cpu_buffer", torch.ones(2))
    main.register_buffer("gpu_buffer", torch.ones(2, device="cuda"))
    adapter = torch.nn.Linear(4, 4, device="cuda")
    text = torch.nn.Linear(4, 4, device="cuda")
    vae = torch.nn.Linear(4, 4).train()
    params = list(main.parameters()) + list(adapter.parameters())
    control = [torch.nn.Parameter(p.detach().clone()) for p in params]
    if optimizer_kind == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizers = [AdamW8bit(ps, lr=.01) for ps in (params, control)]
    else:
        optimizers = [torch.optim.AdamW(ps, lr=.01, foreach=False) for ps in (params, control)]
    for opt, values in zip(optimizers, (params, control)):
        for p in values:
            p.grad = torch.ones_like(p)
        opt.step()
    slot = torch.ones(16, device="cuda")
    slots = [{"weight": slot}]
    master = torch.ones(16)
    conductor = SimpleNamespace(flush=lambda: None, engine=SimpleNamespace(slots=slots))
    shadow = torch.ones(4, device="cuda")
    trainer = SimpleNamespace(device="cuda", transformer=main, network=adapter,
        text_encoder=text, optimizer=optimizers[0], ema_shadow={"weight": shadow},
        _layer_offload_conductors=lambda: [conductor])
    before = [p.detach().cpu().clone() for p in params]
    gradients = [p.grad.cpu().clone() for p in params]
    moments = [{k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in optimizers[0].state[p].items()} for p in params]
    with pytest.raises(RuntimeError, match="decode failed") if fail else nullcontext():
        with debug_decode_residency(trainer, vae):
            assert vae.weight.is_cuda
            for module in (main, adapter, text):
                assert all(p.device.type == "cpu" for p in module.parameters())
            assert slot.device.type == "cpu" and shadow.device.type == "cpu"
            assert all(p.grad.device.type == "cpu" for p in params)
            assert all(not value.is_cuda for state in optimizers[0].state.values()
                       for value in state.values() if isinstance(value, torch.Tensor))
            if fail:
                raise RuntimeError("decode failed")
    assert vae.weight.device.type == "cpu" and vae.training
    assert slot.is_cuda and slots[0]["weight"] is slot and shadow.is_cuda
    assert main.cpu_buffer.device.type == "cpu" and main.gpu_buffer.is_cuda
    assert not master.is_cuda
    for p, value, grad, state in zip(params, before, gradients, moments):
        assert p.is_cuda and p.grad.is_cuda
        torch.testing.assert_close(p.cpu(), value, rtol=0, atol=0)
        torch.testing.assert_close(p.grad.cpu(), grad, rtol=0, atol=0)
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                torch.testing.assert_close(optimizers[0].state[p][k].cpu(), v, rtol=0, atol=0)
            else:
                assert optimizers[0].state[p][k] == v
    for opt in optimizers:
        opt.step()
    for p, expected in zip(params, control):
        torch.testing.assert_close(p, expected, rtol=0, atol=0)


def test_failed_restore_is_distinct_from_optional_preview_failure(monkeypatch):
    vae = torch.nn.Linear(2, 2)
    original = vae.to
    calls = 0
    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("restore failed")
        return original(*args, **kwargs)
    monkeypatch.setattr(vae, "to", broken)
    with pytest.raises(DebugDecodeRestorationError, match="training must stop"):
        with debug_decode_residency(SimpleNamespace(device="cpu"), vae):
            pass


@pytest.mark.parametrize("fail", [False, True])
def test_minit2i_deferred_preview_restores_vae_and_keeps_gradient(tmp_path, fail):
    from core.training.ops.minit2i_ops import flush_pending_debug_previews
    from PIL import Image
    class VAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=2., shift_factor=.25)
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
        def decode(self, value):
            assert not self.training and not value.requires_grad
            if fail:
                raise RuntimeError("decode failed")
            torch.testing.assert_close(value, torch.full_like(value, .25))
            return SimpleNamespace(sample=value[:, :3])
    vae = VAE().train()
    vae.weight.grad = torch.tensor(3.)
    trainer = SimpleNamespace(device="cpu", vae=vae, vae_dtype=torch.float32,
        _pending_minit2i_debug_previews={"path": tmp_path, "timestep": .5,
            "previews": tuple((name, torch.zeros(1, 4, 8, 8)) for name in ("noisy", "target", "pred_x0"))})
    with pytest.raises(RuntimeError, match="decode failed") if fail else nullcontext():
        flush_pending_debug_previews(trainer)
    assert vae.training and vae.weight.grad == 3.
    assert not hasattr(trainer, "_pending_minit2i_debug_previews")
    if not fail:
        for name in ("noisy", "target", "pred_x0"):
            with Image.open(tmp_path/f"decode_t0.5000_{name}.webp") as image:
                assert image.size == (8, 8)
