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

from core.models.anima.partition_global import (
    AnimaPartitionGlobalAdapter, install, metadata, NAME, APPLICATION_METADATA_KEY,
)
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
    box = PartitionBox(0, 0, 4, 4)
    def tile(value):
        return value(image[..., :8, :8], times, text, padding_mask=mask[..., :8, :8],
                     partition_box=box, partition_full_latents=image)
    native_tile = tile(model)
    declaration = dict(schema_version=1, algorithm="anima_partition_global", latent_channels=4,
        hidden_dim=24, rank=4, summary_tokens=2, target_patch_size=2, coordinate_axes=["height", "width"])
    adapter = install(model, declaration)
    torch.testing.assert_close(model(image, times, text, padding_mask=mask), native, rtol=0, atol=0)
    with torch.no_grad():
        adapter.out.weight.normal_(0, .02)
    torch.testing.assert_close(model(image, times, text, padding_mask=mask), native, rtol=0, atol=0)
    expected = tile(model)
    assert not torch.equal(expected, native_tile)
    path = tmp_path / "model.safetensors"
    md = dict(metadata(model), transformer_config=json.dumps(native_config(model)))
    state = {"net." + key: value.detach().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(path), metadata=md)
    restored = load_anima_dit(str(path), dtype=torch.float32)
    torch.testing.assert_close(restored(image, times, text, padding_mask=mask), native, rtol=0, atol=0)
    torch.testing.assert_close(tile(restored), expected, rtol=0, atol=0)
    assert getattr(restored, NAME).metadata() == declaration
    assert md[APPLICATION_METADATA_KEY] == "partition_only"
    legacy = {key: value for key, value in md.items() if key != APPLICATION_METADATA_KEY}
    save_file(state, str(path), metadata=legacy)
    restored = load_anima_dit(str(path), dtype=torch.float32)
    torch.testing.assert_close(tile(restored), expected, rtol=0, atol=0)
    torch.testing.assert_close(restored(image, times, text, padding_mask=mask), native, rtol=0, atol=0)
    del state["net." + NAME + ".out.weight"]
    save_file(state, str(path), metadata=md)
    with pytest.raises(ValueError, match="missing"):
        load_anima_dit(str(path), dtype=torch.float32)


@pytest.mark.parametrize('training', [False, True])
def test_full_frame_skips_adapter_and_leaves_its_gradients_unset(training):
    from core.models.anima.anima_models import Anima
    model = Anima(max_img_h=32, max_img_w=32, max_frames=1, in_channels=4, out_channels=4,
        model_channels=24, num_blocks=2, num_heads=2, mlp_ratio=1., crossattn_emb_channels=12,
        use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
    model.train(training)
    for p in model.parameters():
        if p.ndim >= 2:torch.nn.init.normal_(p, std=.02)
    image, text, time = torch.randn(1,4,1,16,16), torch.randn(1,3,12), torch.tensor([.5])
    mask = torch.zeros(1,1,16,16)
    expected = model(image,time,text,padding_mask=mask)
    adapter = AnimaPartitionGlobalAdapter(4,24,rank=4,summary_tokens=2)
    torch.nn.init.normal_(adapter.out.weight,std=.1)
    model.add_module(NAME,adapter)
    calls=[]
    adapter.register_forward_hook(lambda *args:calls.append(True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    # Existing optimizer moments must not move a bypassed adapter either.
    for p in adapter.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    before={key:p.detach().clone() for key,p in adapter.named_parameters()}
    out=model(image,time,text,padding_mask=mask,partition_full_latents=image)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    out.square().mean().backward();optimizer.step()
    assert calls==[] and all(p.grad is None for p in adapter.parameters())
    for name,p in adapter.named_parameters():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    assert model.final_layer.linear.weight.grad is not None
