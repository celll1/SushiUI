"""Live-weight rollback for successful and failed EMA exports."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.ema_stash import EmaParameterStash
from core.training.base_trainer import BaseTrainer


@pytest.mark.parametrize("failed", [False, True])
def test_export_restores_weights_name_and_parameter_identity(tmp_path, monkeypatch, failed):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    parameters = [torch.nn.Parameter(torch.arange(16, dtype=torch.bfloat16).reshape(4, 4).t()),
                  torch.nn.Parameter(torch.zeros(7))]
    original = [p.detach().clone() for p in parameters]
    shadow = {str(index): p.detach().float()+1 for index, p in enumerate(parameters)}
    trainer = SimpleNamespace(use_ema=True, _ema_param_order=list(zip(shadow, parameters)),
                              ema_shadow=shadow, run_name="live", log_prefix="test")
    def save(step, epoch):
        for name, p in trainer._ema_param_order:
            torch.testing.assert_close(p, shadow[name].to(p.dtype), rtol=0, atol=0)
        if failed: raise RuntimeError("injected save failure")
    trainer.save_checkpoint = save
    BaseTrainer._save_ema_checkpoint(trainer, 3, 1)
    assert trainer.run_name == "live"
    assert trainer._ema_param_order[0][1] is parameters[0]
    for p, before in zip(parameters, original):
        torch.testing.assert_close(p, before, rtol=0, atol=0)
    assert not list(tmp_path.iterdir())


def test_failed_capture_never_changes_live_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    parameter = torch.nn.Parameter(torch.ones(16))
    stash = EmaParameterStash([("weight", parameter)])
    def fail(*args, **kwargs): raise OSError("injected disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="injected disk failure"): stash.capture()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    assert not list(tmp_path.iterdir())


def test_failed_restore_preserves_rollback_files(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    parameter = torch.nn.Parameter(torch.ones(16))
    stash = EmaParameterStash([("weight", parameter)])
    stash.capture()
    stash.mutating = True
    original_load = torch.load
    def fail(*args, **kwargs): raise OSError("injected read failure")
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", fail)
        with pytest.raises(RuntimeError, match="rollback files retained"): stash.restore()
        stash.close()
        assert stash.root.is_dir()
    assert torch.load is original_load
    stash.restore()
    stash.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ema_export_does_not_duplicate_all_live_weights_on_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    parameters = [torch.nn.Parameter(torch.ones(2**22, device="cuda")) for _ in range(4)]
    trainer = SimpleNamespace(use_ema=True, _ema_param_order=[(str(i), p) for i, p in enumerate(parameters)],
        ema_shadow={str(i): torch.zeros_like(p, device="cpu") for i, p in enumerate(parameters)},
        run_name="live", log_prefix="test", save_checkpoint=lambda *args: None)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    BaseTrainer._save_ema_checkpoint(trainer, 3, 1)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated()-before
    assert extra <= 2*parameters[0].numel()*parameters[0].element_size()
    assert all(torch.equal(p, torch.ones_like(p)) for p in parameters)
    assert not list(tmp_path.iterdir())
