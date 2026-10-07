"""Guidance must retain the external codec boundary and U-net execution."""
import pytest
import torch

from backend.tests.minit2i_spatial_boundaries_test import model_for, no_cuda
from backend.tests.latent_adapter_contract_test import declaration, identity
from core.models.components.spatial_swap import install
from core.models.components.vae_alignment import SpatialLatentBridge


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("guidance", ["nag", "negpip", "both"])
@pytest.mark.parametrize("rewired", [False, True])
def test_guidance_calls_model_codec_boundary(batch, guidance, rewired):
    from core.inference.nag_minit2i import MiniT2INAGWrapper
    from core.inference.negpip_minit2i import MiniT2INegPipWrapper
    from core.models.minit2i.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS
    model = model_for(8)
    if rewired:
        restore_topology(model, build_spec("minit2i", [{"name": "main", "block_count": 3}],
            [dict(id="s", stream="main", parent=None, start_block=1, end_block=2,
                  down_factor=[2, 2], **OPERATORS)], native_config(model)))
    install(model, declaration(8, 4))
    model.vae_latent_bridge.initialize_from_pair(identity(SpatialLatentBridge(4, 8), .2),
        identity(SpatialLatentBridge(8, 4), .3))
    nag = negpip = None
    target = model
    calls = []
    hook = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    try:
        if guidance in ("nag", "both"):
            nag = MiniT2INAGWrapper(model, torch.randn(1, 5, 16), torch.ones(1, 5))
            target = nag
        if guidance in ("negpip", "both"):
            negpip = MiniT2INegPipWrapper(model, -torch.ones(5), nag_wrapper=nag)
            target = negpip
        image = torch.randn(batch, 4, 8, 12)
        args = (image, torch.full((batch,), .4), torch.randn(batch, 5, 16), torch.ones(batch, 5))
        if rewired:
            # The existing U-net guidance refusal must survive codec wrapping.
            with pytest.raises(ValueError, match="inactive guidance/style"):
                target(*args)
        else:
            result = target(*args)
            assert result.shape == image.shape and torch.isfinite(result).all()
        assert calls == [batch + 1 if nag is not None else batch]
        assert all(not getattr(block, "_nag_active", False) and not getattr(block, "_negpip_active", False)
                   for block in model.model.net.double_blocks)
    finally:
        hook.remove()
        if negpip is not None:
            negpip.restore()
        if nag is not None:
            nag.restore()
