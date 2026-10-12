"""Legacy index-passthrough Anima swaps convert to full passthrough without changing outputs."""
import json
from unittest.mock import patch

import pytest
import torch

from core.models.components.gaussian_alignment import POLICY
from core.models.components.spatial_swap import META_KEY, install
from core.models.components.vae_alignment import SpatialLatentBridge
from backend.tests.latent_adapter_contract_test import declaration


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
         patch.object(torch.cuda, "_lazy_init", return_value=None), \
         patch.object(torch._C, "_cuda_init", return_value=None):
        yield


def model(channels):
    from core.models.anima.anima_models import Anima
    return Anima(max_img_h=64, max_img_w=64, max_frames=8, in_channels=channels, out_channels=channels,
                 model_channels=64, num_blocks=2, num_heads=4, crossattn_emb_channels=32,
                 adaln_lora_dim=16, use_llm_adapter=False)


def test_conversion_preserves_dit_and_global_summary_outputs():
    from core.models.anima.partition_global import install as install_global, metadata as global_metadata
    from core.models.anima.residual_unet import native_config
    from core.models.anima.vae_alignment import map_image, noise_kwargs
    from core.models.anima.passthrough_conversion import convert
    torch.manual_seed(5)
    old, new = 4, 8
    legacy = model(new)
    install(legacy, dict(declaration(old, new), noise_policy=POLICY))
    pair = legacy.vae_latent_bridge
    incoming, outgoing = SpatialLatentBridge(new, old), SpatialLatentBridge(old, new)
    for bridge in (incoming, outgoing):
        bridge.initialize(torch.randn(bridge.source_channels*9, bridge.target_channels)*.2,
                          torch.randn(bridge.target_channels)*.1)
    pair.initialize_from_pair(incoming, outgoing)
    install_global(legacy, dict(schema_version=1, algorithm="anima_partition_global", latent_channels=new,
        hidden_dim=64, rank=8, summary_tokens=4, target_patch_size=2, coordinate_axes=["height", "width"]))
    with torch.no_grad():
        for parameter in legacy.parameters():
            parameter.add_(torch.randn_like(parameter)*.05)
    metadata = {META_KEY: json.dumps(pair.declaration, sort_keys=True),
                "transformer_config": json.dumps(native_config(legacy)), **global_metadata(legacy)}
    state, converted = convert({"net."+k: v for k, v in legacy.state_dict().items()}, metadata)
    assert json.loads(converted[META_KEY])["passthrough"] == "full"
    full = model(old + new)
    install(full, json.loads(converted[META_KEY]))
    install_global(full, json.loads(converted["sushi.anima.partition_global"]))
    full.load_state_dict({k[4:]: v for k, v in state.items()}, strict=True)
    assert full.vae_latent_bridge.ready
    legacy.eval(), full.eval()
    x, noise = torch.randn(2, new, 1, 8, 12), torch.randn(2, new, 1, 8, 12)
    sigma, context = torch.tensor([.3, .8]), torch.randn(2, 5, 32)
    mask = torch.zeros(2, 1, 8, 12)
    with torch.no_grad():
        expected = legacy(x, sigma, context, padding_mask=mask, **noise_kwargs(legacy, noise.squeeze(2)))
        actual = full(x, sigma, context, padding_mask=mask, **noise_kwargs(full, noise.squeeze(2)))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        summaries = [m.anima_partition_global_adapter(map_image(m, x, sigma=sigma, noise=noise))
                     for m in (legacy, full)]
        torch.testing.assert_close(summaries[1], summaries[0], atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="legacy"):
        convert(state, converted)


def test_strip_reshaped_moments_keeps_other_entries_and_requires_names():
    from core.models.anima.passthrough_conversion import RESHAPED, strip_reshaped_moments
    names = ["transformer."+name for name in RESHAPED[:5]] + ["transformer.blocks.0.mlp.weight"]
    payload = dict(param_groups=[dict(params=list(range(6)))], _sushi_param_names=[names],
                   state={i: {"step": 1} for i in range(6)})
    stripped, dropped = strip_reshaped_moments(payload)
    assert sorted(dropped) == sorted(names[:5]) and list(stripped["state"]) == [5]
    assert len(payload["state"]) == 6
    with pytest.raises(ValueError, match="names"):
        strip_reshaped_moments(dict(payload, _sushi_param_names=None))
