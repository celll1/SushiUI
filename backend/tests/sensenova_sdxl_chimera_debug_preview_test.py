import sys
from pathlib import Path
from types import SimpleNamespace

import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.ops.sensenova_sdxl_chimera_ops import flush_pending_debug_previews


class _TinyVAE(torch.nn.Module):
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=None)

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))

    def decode(self, latents, return_dict=False):
        return (latents[:, :3].clamp(-1, 1),)


def test_deferred_debug_previews_decode_and_restore_vae(tmp_path):
    vae = _TinyVAE().train()
    trainer = SimpleNamespace(
        vae=vae,
        device=torch.device("cpu"),
        _pending_chimera_debug_previews={
            "path": tmp_path,
            "t_val": 0.5,
            "previews": tuple(
                (name, torch.zeros(1, 4, 8, 8))
                for name in ("noisy", "target", "pred_x0")
            ),
        },
    )

    flush_pending_debug_previews(trainer)

    assert not hasattr(trainer, "_pending_chimera_debug_previews")
    assert vae.training
    assert next(vae.parameters()).device.type == "cpu"
    for name in ("noisy", "target", "pred_x0"):
        assert (tmp_path / f"decode_t0.5000_{name}.webp").is_file()
