"""Anima's zero-weight reconstruction monitor reaches the metrics return."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
from core.training.ops.anima_ops import train_step


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(4, 4, 1)
        self.prediction = None

    def forward(self, x, **kwargs):
        self.prediction = self.projection(x.squeeze(2))
        return self.prediction.unsqueeze(2)


@pytest.mark.parametrize("weight", [0.0, .3])
@pytest.mark.parametrize("debug", [False, True])
def test_metric_matches_actual_x0_error_without_changing_loss(tmp_path, monkeypatch, weight, debug):
    monkeypatch.setattr("core.models.anima.training_debug.queue_debug_previews", lambda *args: None)
    torch.manual_seed(76)
    model = Predictor()
    clean, noise = torch.randn(2, 4, 8, 12), torch.randn(2, 4, 8, 12)
    sigma = torch.tensor([.2, .8])
    trainer = SimpleNamespace(device=torch.device('cpu'), training_dtype=torch.float32,
        transformer=model, transformer_original=model, reconstruction_loss_weight=weight,
        mixed_precision=False, _active_mnt_noise=noise, log_prefix='[test]')
    loss, pred, recon = train_step(trainer, latents=clean, prompt_embeds=torch.zeros(2, 3, 6),
        anima_aux={key: torch.ones(2, 3, dtype=torch.long) for key in
                   ('source_mask', 't5_input_ids', 't5_attn_mask')},
        timesteps=sigma, debug_save_path=tmp_path if debug else None)
    noisy = (1-sigma[:, None, None, None])*clean + sigma[:, None, None, None]*noise
    expected_recon = (noisy-sigma[:, None, None, None]*model.prediction-clean).square().mean()
    expected_pred = (model.prediction-(noise-clean)).square().mean()
    expected_loss = (1-weight)*expected_pred + weight*expected_recon
    torch.testing.assert_close(recon, expected_recon.detach())
    assert recon > 0 and not recon.requires_grad
    torch.testing.assert_close(loss, expected_loss)
    actual_grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=True)
    reference_grads = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    for actual, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual, reference)
    if debug:
        data = torch.load(next(tmp_path.glob('latents_*.pt')), weights_only=True)
        assert data['recon_loss_batch_mean'] == recon.item()
        torch.testing.assert_close(torch.tensor(data['recon_loss']),
            (data['predicted_latent']-data['latents']).square().mean())
    assert not torch.cuda.is_initialized()
