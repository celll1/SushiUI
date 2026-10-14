"""SenseNova guidance-target teacher: per-row prediction under the cached null prefix."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.ops import sensenova_ops


class FakeTransformer:
    config = SimpleNamespace(t_eps=1e-3)

    def __init__(self):
        self.fm_modules = {"fm_head": lambda grid: grid * 2}
        self.language_model = SimpleNamespace(model="language-model")
        self.index_lengths = []

    def patchify(self, x, patch):
        return x.flatten(2).transpose(1, 2)

    def _build_t2i_image_indexes(self, token_h, token_w, text_length, device=None):
        self.index_lengths.append(text_length)
        return torch.zeros(1)


def test_teacher_runs_each_row_on_the_null_prefix(monkeypatch):
    import core.models.sensenova.sensenova_pipeline_ops as pipeline
    import core.models.sensenova.vae_alignment as alignment

    seen = []

    def step_context(transformer, shape, z_image, t, noise_scale, *, enable_grad, exposure_noise):
        assert shape.batch_size == 1 and not enable_grad and z_image.shape[0] == 1
        seen.append(("context", float(t)))
        return transformer.patchify(z_image, 1), z_image.flatten(2).transpose(1, 2), None

    def decoder(model, image_embeds, *, indexes, prefix_cache):
        seen.append(("decoder", prefix_cache))
        return image_embeds

    monkeypatch.setattr(pipeline, "_build_step_context", step_context)
    monkeypatch.setattr(sensenova_ops, "forward_gen_decoder_layers", decoder)
    monkeypatch.setattr(alignment, "output_x0", lambda transformer, decoded: decoded)

    transformer = FakeTransformer()
    shape = SimpleNamespace(batch_size=2, merge_size=1, grid_h=2, grid_w=2, token_h=2, token_w=2)
    null_prefix = SimpleNamespace(cache="null-cache", text_length=7)
    z_image, t = torch.randn(2, 3, 2, 2), torch.tensor([.25, .5])
    velocity = sensenova_ops._guidance_null_velocity(
        transformer, SimpleNamespace(patch=1), null_prefix, z_image, t, torch.zeros_like(z_image), 1.0,
        shape, 2, 2, torch.float32)

    assert transformer.index_lengths == [7, 7]
    assert seen == [("context", .25), ("decoder", "null-cache"), ("context", .5), ("decoder", "null-cache")]
    z = z_image.flatten(2).transpose(1, 2)
    torch.testing.assert_close(velocity, (2 * z - z) / (1 - t).view(-1, 1, 1))


def test_reference_runs_are_refused():
    from core.training.arch.sensenova import SenseNovaArchHandler
    handler = SenseNovaArchHandler.__new__(SenseNovaArchHandler)
    assert handler.guidance_loss_refusal(SimpleNamespace(config={"use_reference_images": True}))
    assert handler.guidance_loss_refusal(SimpleNamespace(config={})) is None
