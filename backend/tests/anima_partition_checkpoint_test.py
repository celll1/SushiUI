"""Combined full-parameter Anima artifacts retain target-native partition execution."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
torch.set_num_threads(1)


def test_full_writer_and_real_loader_preserve_combined_partition_predictions(tmp_path):
    from core.models.anima.anima_models import Anima
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.partition_global import configure_training, METADATA_KEY, APPLICATION_METADATA_KEY
    from core.models.anima.residual_unet import native_config, restore_topology
    from core.models.anima.partition_training import training_plan
    from core.models.components.spatial_swap import install, META_KEY
    from core.training.adapters.anima_adapter import AnimaFullParameterAdapter
    from core.rewiring.spec import build_spec, OPERATORS
    from backend.tests.latent_adapter_contract_test import declaration, identity
    torch.manual_seed(918)
    model = Anima(max_img_h=64, max_img_w=64, max_frames=1,
        in_channels=64, out_channels=64, model_channels=24, num_blocks=7,
        num_heads=2, mlp_ratio=1., crossattn_emb_channels=12,
        use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .02)
    install(model, declaration(16, 64, scale=16))
    identity(model.vae_latent_bridge.input, .2)
    identity(model.vae_latent_bridge.output, .3)
    restore_topology(model, build_spec("anima", [{"name": "main", "block_count": 7}], [
        dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
             down_factor=[2, 2], **OPERATORS),
        dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5,
             down_factor=[2, 2], **OPERATORS)], native_config(model)))
    config = dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
        dit_partition_fixed_count=4, dit_partition_global_adapter_enabled=True,
        dit_partition_global_rank=4, dit_partition_global_tokens=2)
    adapter = configure_training(model, config)
    with torch.no_grad():
        adapter.out.weight.normal_(0, .02)
    owner = SimpleNamespace(transformer=model, config=config, vae=None, bundle_vae=False,
        train_unet=True, train_text_encoder=False, _current_epoch=4, _current_batch_position=9)
    named = list(model.named_parameters())
    optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=1e-4)
    for _, parameter in named:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved_state = optimizer.state_dict()
    path = tmp_path / "combined.safetensors"
    swap_metadata = (SimpleNamespace(provenance="synthetic", locator="registry:test"), False,
                     {"component.vae.channels": "64", "component.vae.type": "qwen_image_21"})
    with patch("core.training.vae_swap.swap_metadata", return_value=swap_metadata):
        AnimaFullParameterAdapter(owner).write_checkpoint(13, 4, path)
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        assert json.loads(metadata[METADATA_KEY])["rank"] == 4
        assert metadata[APPLICATION_METADATA_KEY] == "partition_only"
        assert json.loads(metadata[META_KEY])["new_scale"] == 16
        assert metadata["step"] == "13" and metadata["epoch"] == "4"
    restored = load_anima_dit(str(path), dtype=torch.float32)
    model.eval()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    from core.training.base_trainer import BaseTrainer
    restored_named = list(reversed(list(restored.named_parameters())))
    restored_optimizer = torch.optim.AdamW([parameter for _, parameter in restored_named], lr=1e-4)
    proxy = SimpleNamespace(log_prefix="[Anima resume]",
        _optimizer_state_entry_fits_param=BaseTrainer._optimizer_state_entry_fits_param)
    assert BaseTrainer._load_optimizer_state_by_parameter_name(proxy, restored_optimizer,
        saved_state, [[name for name, _ in named]], [[name for name, _ in restored_named]], "combined")
    assert not proxy._optimizer_state_partially_fresh
    old_parameters = dict(named)
    for name, parameter in restored_named:
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(restored_optimizer.state[parameter][key],
                optimizer.state[old_parameters[name]][key], rtol=0, atol=0)
    clean, noise = torch.randn(1, 64, 32, 48), torch.randn(1, 64, 32, 48)
    sigma, text = torch.tensor([.4]), torch.randn(1, 3, 12)
    noisy = .6*clean + .4*noise
    plan = training_plan(owner, clean)
    owner.transformer = restored
    assert training_plan(owner, clean) == plan
    def forward(value, box=None):
        crop = lambda image: image if box is None else image[..., 2*box.top:2*box.bottom, 2*box.left:2*box.right]
        local = crop(noisy)
        return value(local.unsqueeze(2), sigma, text,
            padding_mask=torch.zeros(1, 1, *local.shape[-2:]),
            partition_box=box, partition_full_latents=noisy.unsqueeze(2) if box else None,
            partition_full_noise=noise.unsqueeze(2) if box else None)
    with torch.no_grad():
        torch.testing.assert_close(forward(restored), forward(model), rtol=1e-6, atol=1e-6)
        for region in plan.regions:
            torch.testing.assert_close(forward(restored, region.input), forward(model, region.input), rtol=1e-6, atol=1e-6)
