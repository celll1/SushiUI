"""Native patch summaries, gradients and strict adapter restoration."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
torch.set_num_threads(1)

from core.models.anima.partition_global import AnimaPartitionGlobalAdapter, install, metadata, NAME
from core.training.partition import PartitionBox


def test_patch_features_match_native_anima_order():
    from core.models.anima.anima_models import PatchEmbed
    adapter = AnimaPartitionGlobalAdapter(3, 12, rank=4, summary_tokens=2)
    image = torch.arange(3*8*12).reshape(1, 3, 1, 8, 12).float()
    patch = PatchEmbed(2, 1, in_channels=3, out_channels=12)
    torch.testing.assert_close(adapter._features(image)[..., :12], patch.proj[0](image)[:, 0])


def test_zero_initialization_and_full_canvas_context_gradients():
    adapter = AnimaPartitionGlobalAdapter(3, 12, rank=4, summary_tokens=2)
    image = torch.randn(2, 3, 1, 8, 12, requires_grad=True)
    box = PartitionBox(0, 0, 2, 3)
    initial = adapter(image, box)
    assert initial.shape == (2, 1, 2, 3, 12) and torch.count_nonzero(initial) == 0
    initial.sum().backward()
    assert adapter.out.weight.grad.abs().sum() > 0
    adapter.zero_grad(set_to_none=True)
    with torch.no_grad():
        adapter.out.weight.normal_(0, .02)
    image.grad = None
    adapter(image, box).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in adapter.parameters())
    assert image.grad[..., 4:, 6:].abs().sum() > 0


def test_real_loader_restores_declared_global_adapter(tmp_path):
    from core.models.anima.anima_models import Anima
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.residual_unet import native_config
    model = Anima(max_img_h=32, max_img_w=32, max_frames=1, in_channels=4, out_channels=4,
        model_channels=24, num_blocks=2, num_heads=2, mlp_ratio=1., crossattn_emb_channels=12,
        use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .02)
    image = torch.randn(1, 4, 1, 16, 16)
    text, mask, times = torch.randn(1, 3, 12), torch.zeros(1, 1, 16, 16), torch.tensor([.5])
    native = model(image, times, text, padding_mask=mask)
    declaration = dict(schema_version=1, algorithm="anima_partition_global", latent_channels=4,
        hidden_dim=24, rank=4, summary_tokens=2, target_patch_size=2, coordinate_axes=["height", "width"])
    adapter = install(model, declaration)
    torch.testing.assert_close(model(image, times, text, padding_mask=mask), native, rtol=0, atol=0)
    with torch.no_grad():
        adapter.out.weight.normal_(0, .02)
    expected = model(image, times, text, padding_mask=mask)
    assert not torch.equal(expected, native)
    path = tmp_path / "model.safetensors"
    md = dict(metadata(model), transformer_config=json.dumps(native_config(model)))
    state = {"net." + key: value.detach().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(path), metadata=md)
    restored = load_anima_dit(str(path), dtype=torch.float32)
    torch.testing.assert_close(restored(image, times, text, padding_mask=mask), expected, rtol=0, atol=0)
    assert getattr(restored, NAME).metadata() == declaration
    del state["net." + NAME + ".out.weight"]
    save_file(state, str(path), metadata=md)
    with pytest.raises(ValueError, match="missing"):
        load_anima_dit(str(path), dtype=torch.float32)
