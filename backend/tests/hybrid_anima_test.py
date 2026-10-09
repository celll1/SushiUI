"""Native Anima hybrid forward, optimizer and artifact restoration on CPU."""
import json
from pathlib import Path
import sys
from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from safetensors.torch import save_file

from core.rewiring.spec import OPERATORS, build_spec, read_checkpoint_metadata
from core.rewiring.hybrid_spec import build_hybrid_spec


@pytest.fixture
def native():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
            patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
        from core.models.anima.anima_models import Anima
        from core.models.anima.residual_unet import native_config, restore_topology
        torch.manual_seed(77)
        model = Anima(max_img_h=32, max_img_w=32, max_frames=8, in_channels=4, out_channels=4,
            model_channels=24, num_blocks=7, num_heads=2, mlp_ratio=1.0, crossattn_emb_channels=12,
            use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(0, 0.02)
        base = build_spec("anima", [{"name": "main", "block_count": 7}], [
            dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
                 down_factor=[2, 2], **OPERATORS)], native_config(model))
        graph = build_hybrid_spec(base, stream_widths={"main": 24}, boundaries=[
            dict(id="outer_boundary", region="outer", rank=6, kernel=3)], refiners=[
            dict(id="entry", stream="main", after_block=0, rank=6, kernel=3),
            dict(id="exit", stream="main", after_block=6, rank=6, kernel=3)], skips=[
            dict(id="outer_skip", stream="main", source_after=0, destination_before=6, rank=6)])
        restore_topology(model, graph)
        yield model, graph
        assert not torch.cuda.is_initialized()


def forward(model):
    return model(torch.ones(1, 4, 1, 8, 12), torch.tensor([0.5]), torch.ones(1, 3, 12),
                 padding_mask=torch.zeros(1, 1, 8, 12))


def test_initialization_noising_preserves_bf16_modulation(native):
    from core.rewiring.anima_initialization import noising_input
    model, _ = native
    model.to(torch.bfloat16).train().enable_gradient_checkpointing()
    latent = torch.ones(2,4,1,8,12,dtype=torch.bfloat16)
    noise = torch.zeros_like(latent)
    value, sigma = noising_input(latent,noise,torch.tensor([.25,.75]))
    assert value.dtype == sigma.dtype == torch.bfloat16
    assert torch.equal(value[0],torch.full_like(value[0],.75))
    assert torch.equal(value[1],torch.full_like(value[1],.25))
    with torch.autocast("cpu",dtype=torch.bfloat16):
        output = model(value,sigma,torch.ones(2,3,12,dtype=torch.bfloat16),
                       padding_mask=torch.zeros(2,1,8,12,dtype=torch.bfloat16))
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.rewiring_modules.parameters())


def test_native_training_and_strict_checkpoint_reload(native, tmp_path):
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.residual_unet import native_config, model_topology_metadata
    model, graph = native
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        forward(model).square().mean().backward()
        optimizer.step()
    for name, parameter in model.rewiring_modules.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(parameter in optimizer.state for parameter in model.rewiring_modules.parameters())
    metadata = dict(model_topology_metadata(model), model_type="anima",
                    transformer_config=json.dumps(native_config(model)))
    assert read_checkpoint_metadata(metadata) == graph
    path = tmp_path / "hybrid.safetensors"
    weights = {"net." + name: value.detach().contiguous() for name, value in model.state_dict().items()}
    save_file(weights, str(path), metadata=metadata)
    model.eval()
    restored = load_anima_dit(str(path), device="cpu", dtype=torch.float32)
    assert restored._residual_unet_spec == graph
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    with torch.no_grad():
        torch.testing.assert_close(forward(model), forward(restored), rtol=1e-5, atol=1e-6)
    assert all(not parameter.is_meta for parameter in restored.rewiring_modules.parameters())
    restored.requires_grad_(True).train()
    model.train()
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    saved_optimizer = deepcopy(optimizer.state_dict())
    new_optimizer.load_state_dict(saved_optimizer)
    for key, state in saved_optimizer["state"].items():
        for field, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, new_optimizer.state_dict()["state"][key][field])
    for target, opt in ((model, optimizer), (restored, new_optimizer)):
        opt.zero_grad()
        forward(target).square().mean().backward()
        opt.step()
    torch.testing.assert_close(forward(model), forward(restored), rtol=1e-5, atol=1e-6)
    for name, value in model.rewiring_modules.state_dict().items():
        torch.testing.assert_close(value, restored.rewiring_modules.state_dict()[name], rtol=1e-5, atol=1e-6)
    weights.pop(next(name for name in weights if name.startswith("net.rewiring_modules.")))
    save_file(weights, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="hybrid module weights"):
        load_anima_dit(str(path), device="cpu", dtype=torch.float32)


def test_reinstall_does_not_replace_learned_parameters(native):
    from core.models.anima.residual_unet import restore_topology
    model, graph = native
    parameters = list(model.rewiring_modules.parameters())
    from hybrid_provenance_test import record
    value = record()
    value["origin_spec"] = deepcopy(graph)
    model._hybrid_initialization = value
    restore_topology(model, graph)
    assert all(left is right for left, right in zip(parameters, model.rewiring_modules.parameters()))
    assert model._hybrid_initialization == value
    restore_topology(model, None)
    assert not hasattr(model, "rewiring_modules")
    assert not hasattr(model, "_hybrid_initialization")


def test_initialization_lineage_survives_native_reload_and_fine_tuning(native, tmp_path):
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.residual_unet import native_config, model_topology_metadata
    from hybrid_provenance_test import record
    model, graph = native
    value = record()
    value["origin_spec"] = deepcopy(graph)
    model._hybrid_initialization = value
    path = tmp_path / "initialized.safetensors"
    for _ in range(2):
        metadata = dict(model_topology_metadata(model), model_type="anima",
                        transformer_config=json.dumps(native_config(model)))
        weights = {"net." + name: tensor.detach().contiguous()
                   for name, tensor in model.state_dict().items()}
        save_file(weights, str(path), metadata=metadata)
        restored = load_anima_dit(str(path), device="cpu", dtype=torch.float32)
        assert restored._hybrid_initialization == value
        assert model_topology_metadata(restored) == model_topology_metadata(model)
        for name, tensor in weights.items():
            assert torch.equal(tensor, restored.state_dict()[name.removeprefix("net.")])
        model = restored.train().requires_grad_(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        forward(model).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert model._hybrid_initialization == value


def test_native_partition_and_global_adapter_backward(native):
    from core.models.anima.partition_training import training_plan, backward_regions
    from core.models.anima.partition_global import install
    model, graph = native
    model._preprocess_text_embeds = lambda context, *args: context
    install(model, dict(schema_version=1, algorithm="anima_partition_global", latent_channels=4,
        hidden_dim=24, rank=6, summary_tokens=4, target_patch_size=2, coordinate_axes=["height", "width"]), dtype=torch.float32)
    model.train().requires_grad_(True)
    clean, noise = torch.randn(1, 4, 16, 24), torch.randn(1, 4, 16, 24)
    sigma, text = torch.tensor([0.5]), torch.ones(1, 3, 12)
    config = dict(dit_rewiring_spec=graph, dit_partition_training_enabled=True,
        dit_partition_mode="fixed", dit_partition_fixed_count=2, dit_partition_halo_tokens=0,
        dit_partition_global_adapter_enabled=True, dit_partition_gradient_checkpointing_blocks=3)
    trainer = SimpleNamespace(transformer=model, config=config, device=torch.device("cpu"),
        training_dtype=torch.float32, mixed_precision=False, reconstruction_loss_weight=0.0)
    plan = training_plan(trainer, clean)
    assert all(region.input.tokens == plan.full_tokens for region in plan.regions)
    full_prediction = model(((1-sigma[0])*clean+sigma[0]*noise).unsqueeze(2), sigma, text,
                            padding_mask=torch.zeros(1, 1, 16, 24)).squeeze(2)
    full_loss = (full_prediction - (noise-clean)).square().mean()
    full_loss.backward()
    expected = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                if parameter.grad is not None}
    model.zero_grad()
    losses = backward_regions(trainer, clean=clean, prompt_embeds=text,
        anima_aux=dict(source_mask=torch.ones(1, 3), t5_input_ids=torch.ones(1, 3, dtype=torch.long),
                       t5_attn_mask=torch.ones(1, 3, dtype=torch.bool)),
        sigma=sigma, noise=noise, plan=plan, backward_scale=1.0)
    assert abs(losses[0] - full_loss.item()) < 1e-6
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-6)
    assert model.rewiring_modules.local["exit"].expand.weight.grad.abs().sum() > 0
    assert model.anima_partition_global_adapter.out.weight.grad.abs().sum() > 0
