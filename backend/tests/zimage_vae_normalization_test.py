"""CPU regressions for Z-Image's VAE normalization at training and inference boundaries."""
import ast
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models.components.vae_registry import normalize, denormalize
from core.training.ops import zimage_ops
from core.pipeline_backends.zimage import ZImageMixin


class Vae:
    def __init__(self, method, dtype):
        self.dtype = dtype
        self.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=0.1159)
        if method == "batchnorm":
            self.config = SimpleNamespace(scaling_factor=None, batch_norm_eps=1e-4)
            self.bn = SimpleNamespace(running_mean=torch.linspace(-1, 1, 16),
                                      running_var=torch.linspace(0.2, 2, 16))
        elif method == "per_channel":
            self.config = SimpleNamespace(scaling_factor=None,
                                          latents_mean=[0.1, 0.2, 0.3, 0.4],
                                          latents_std=[0.5, 1., 1.5, 2.])
        self.raw = torch.linspace(-1, 1, 64).reshape(1, 4, 4, 4).to(dtype)
        self.quant_conv = self.post_quant_conv = None

    def encoder(self, image):
        return torch.cat([self.raw, torch.full_like(self.raw, -1000)], dim=1)

    def decoder(self, latent):
        self.decoded = latent
        return torch.zeros(1, 3, 4, 4, dtype=self.dtype)

    def decode(self, latent, return_dict=False):
        return (self.decoder(latent),)

    def parameters(self):
        return iter([self.raw])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method", ["shift_scale", "batchnorm", "per_channel"])
def test_training_reference_and_decode_boundaries(method, dtype):
    vae = Vae(method, dtype)
    trainer = SimpleNamespace(vae=vae)
    expected = normalize(vae.raw, vae)
    actual = zimage_ops.vae_encode(trainer, torch.zeros(1, 3, 4, 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    host = SimpleNamespace(_apply_vae_tiling=lambda *a: None)
    ref = ZImageMixin._zimage_prepare_style_reference(
        host, vae, Image.new("RGB", (4, 4)), 4, 4, "cpu")
    torch.testing.assert_close(ref, expected.float(), rtol=0, atol=0)
    zimage_ops._decode_zimage_latents(trainer, actual)
    torch.testing.assert_close(vae.decoded, denormalize(actual, vae), rtol=0, atol=0)
    ZImageMixin._zimage_decode_latents(host, vae, actual.float())
    torch.testing.assert_close(vae.decoded, denormalize(actual.float().to(dtype), vae), rtol=0, atol=0)


@pytest.mark.parametrize("name", ["_generate_img2img_zimage", "_generate_inpaint_zimage"])
def test_init_image_normalization_block(name):
    # Execute the real latent assignment without loading a transformer or scheduler.
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ZImageMixin, name))))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "init_latents" for t in node.targets)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == "normalize"]
    assert len(assignments) == 1
    vae = Vae("batchnorm", torch.float32)
    namespace = dict(vae=vae, init_latents=vae.raw, normalize=normalize)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "<encode block>", "exec"), namespace)
    torch.testing.assert_close(namespace["init_latents"], normalize(vae.raw, vae), rtol=0, atol=0)
