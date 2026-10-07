"""FP32 optimizer masters retain MiniT2I's reduced-precision activation contract."""
import pytest
import torch


@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_fp32_norm_masters_allow_bf16_forward_and_backward(rewired, autocast):
    from core.models.minit2i.vendor.transformer import MiniT2IMMJiTModel
    from core.models.minit2i.vendor.mmjit import RMSNorm
    from core.models.minit2i.residual_unet import native_config, restore_topology
    from core.rewiring.spec import build_spec, OPERATORS

    torch.manual_seed(13)
    model = MiniT2IMMJiTModel(image_size=16, patch_size=2, in_channels=4,
        hidden_size=32, txt_hidden_size=32, cond_vec_size=32, depth_double=7,
        num_heads=2, head_dim=16, pca_channels=8, prompt_length=4,
        txt_input_size=16, txt_preamble_depth=1, vae_type="sdxl").to(torch.bfloat16)
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    for norm in norms:
        norm.float()
    if rewired:
        restore_topology(model, build_spec("minit2i", [dict(name="main", block_count=7)],
            [dict(id="half", stream="main", parent=None, start_block=1, end_block=6,
                  down_factor=[2, 2], **OPERATORS)], native_config(model)))
    image = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16)
    text = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = model(image, torch.tensor([.25, .75], dtype=torch.bfloat16), text, torch.ones(2, 4))
    assert output.shape == image.shape and output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(norm.weight.dtype == torch.float32 and norm.weight.grad is not None
               and torch.isfinite(norm.weight.grad).all() for norm in norms)
    assert model.model.net.double_blocks[0].img_qkv.weight.grad.abs().sum() > 0
