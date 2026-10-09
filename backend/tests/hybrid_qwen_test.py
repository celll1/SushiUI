"""Native Qwen hybrid learning, global coordinates and artifact continuation."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from qwen_partition_full_checkpoint_test import _trainer, _forward
from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel
from core.models.qwen_image_21.residual_unet import native_config, restore_topology
from core.models.qwen_image_21.artifact import load_transformer, artifact_metadata
from core.rewiring.spec import OPERATORS, build_spec, checkpoint_metadata
from core.rewiring.hybrid_spec import build_hybrid_spec
from core.training.adapters.qwen_image_21_adapter import QwenImage21FullParameterAdapter


def model_and_spec():
    model = QwenImage21Transformer2DModel(in_channels=8, out_channels=8, num_layers=6,
        attention_head_dim=16, num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 6, 6))
    base = build_spec("qwen_image_21", [{"name": "main", "block_count": 6}], [
        dict(id="outer", stream="main", parent=None, start_block=1, end_block=5, down_factor=[2, 2], **OPERATORS),
        dict(id="inner", stream="main", parent="outer", start_block=2, end_block=4, down_factor=[2, 2], **OPERATORS)],
        native_config(dict(model.config)))
    spec = build_hybrid_spec(base, stream_widths={"main": 32}, boundaries=[
        dict(id="outer", region="outer", rank=6, kernel=3), dict(id="inner", region="inner", rank=6, kernel=3)],
        refiners=[dict(id="entry", stream="main", after_block=0, rank=6, kernel=3),
                  dict(id="exit", stream="main", after_block=5, rank=6, kernel=3)],
        skips=[dict(id="full", stream="main", source_after=0, destination_before=5, rank=6),
               dict(id="mid", stream="main", source_after=1, destination_before=4, rank=6)])
    return model, spec, base


def test_native_zero_init_then_all_branches_learn():
    torch.manual_seed(122)
    model, spec, base = model_and_spec()
    baseline = deepcopy(model)
    restore_topology(baseline, base)
    restore_topology(model, spec)
    grid = torch.randn(1, 8, 12, 8)
    torch.testing.assert_close(_forward(model, grid), _forward(baseline, grid), rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        _forward(model, grid).square().mean().backward()
        optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.rewiring_modules.parameters())


@pytest.mark.parametrize("sharded", [False, True])
def test_full_adapter_checkpoint_and_optimizer_resume(tmp_path, monkeypatch, sharded):
    if sharded:
        monkeypatch.setattr("core.training.adapters.qwen_image_21_adapter.DEFAULT_EXPORT_SHARD_BYTES", 1024)
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    trainer = _trainer(model)
    adapter = QwenImage21FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    groups = adapter.arch_param_groups()
    owned = [id(p) for group in groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert all(id(p) in owned for p in model.rewiring_modules.parameters())
    optimizer = torch.optim.AdamW(groups)
    grid = torch.randn(1, 8, 12, 8)
    _forward(model, grid).square().mean().backward()
    optimizer.step()
    saved = optimizer.state_dict()
    path = adapter.write_checkpoint(3, 0, tmp_path / "hybrid.safetensors")
    loaded, variant = load_transformer(str(path), dict(model.config), torch.float32)
    assert variant == "bf16"
    assert loaded._residual_unet_spec == spec
    for name, value in model.state_dict().items():
        if not name.startswith("qwen_partition_global_adapter."):
            assert torch.equal(value, loaded.state_dict()[name]), name
    restored_trainer = _trainer(loaded)
    restored_adapter = QwenImage21FullParameterAdapter(restored_trainer)
    restored_adapter.prepare_models_for_training()
    continued = torch.optim.AdamW(restored_adapter.arch_param_groups())
    continued.load_state_dict(saved)
    for key, state in saved["state"].items():
        for name, value in state.items():
            assert torch.equal(value, continued.state_dict()["state"][key][name])
    torch.testing.assert_close(_forward(model, grid), _forward(loaded, grid), rtol=1e-5, atol=1e-6)


def test_partition_halo_and_coarse_full_canvas_positions():
    from core.training.ops.qwen_image_21_ops import _partition_plan
    from core.training.qwen_partition import QwenImage21PartitionAdapter, PartitionBox, full_canvas_position_ids
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    trainer = _trainer(model)
    trainer.config.update(dit_partition_fixed_count=2, dit_partition_halo_tokens=0)
    trainer.arch = SimpleNamespace(dit_partition_adapter=lambda: QwenImage21PartitionAdapter())
    plan = _partition_plan(trainer, 64, 96)
    assert plan is _partition_plan(trainer, 64, 96)
    assert all(region.input.height % 4 == region.input.width % 4 == 0 for region in plan.regions)
    adapter = QwenImage21FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    grid = torch.randn(1, 16, 24, 8)
    box = PartitionBox(4, 8, 12, 20)
    positions = full_canvas_position_ids(16, 24, box)
    captured = []
    handle = model.pos_embed.register_forward_pre_hook(lambda m, args, kwargs:
        captured.append((args[0], kwargs.get("target_spatial_position_ids"))), with_kwargs=True)
    residual = model.qwen_partition_global_adapter(grid, box)
    _forward(model, grid, box, residual).square().mean().backward()
    handle.remove()
    for shapes, coords in captured[1:]:
        _, h, w = shapes[0]
        fh, fw = box.height // h, box.width // w
        expected = positions.reshape(box.height, box.width, 2)[::fh, ::fw].clone()
        expected[..., 0] = torch.div(expected[..., 0], fh, rounding_mode="floor")
        expected[..., 1] = torch.div(expected[..., 1], fw, rounding_mode="floor")
        assert torch.equal(coords, expected.reshape(h * w, 2))
    assert len(captured) == 3
    assert model.qwen_partition_global_adapter.out.weight.grad.abs().sum() > 0
    assert model.rewiring_modules.local["exit"].expand.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("damage", [None, "missing", "nan"])
def test_hybrid_directory_strict_shard_loading(tmp_path, damage):
    from core.models.qwen_image_21.loader import _load_hybrid_directory
    from core.models.common.single_file_format import save_single_file_state
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    model.save_config(tmp_path)
    state = model.state_dict()
    name = next(key for key in state if key.startswith("rewiring_modules."))
    if damage == "missing":
        state.pop(name)
    elif damage == "nan":
        state[name].fill_(float("nan"))
    save_single_file_state(state, checkpoint_metadata(spec), str(tmp_path / "diffusion_pytorch_model.safetensors"),
        max_shard_bytes=4096)
    if damage:
        with pytest.raises(ValueError, match="Qwen"):
            _load_hybrid_directory(tmp_path, QwenImage21Transformer2DModel.load_config(tmp_path), torch.float32, spec)
    else:
        restored = _load_hybrid_directory(tmp_path, QwenImage21Transformer2DModel.load_config(tmp_path), torch.float32, spec)
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in state.items())
        grid = torch.randn(1, 8, 12, 8)
        torch.testing.assert_close(_forward(model, grid), _forward(restored, grid), rtol=1e-5, atol=1e-6)
    assert not torch.cuda.is_initialized()


def test_training_bundle_installs_modules_before_assigning_weights(tmp_path, monkeypatch):
    import core.models.qwen_image_21.loader as loader
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    companion = tmp_path / "companion"
    companion.mkdir()
    monkeypatch.setattr(loader, "_source_components", lambda *args, **kwargs: dict(vae=torch.nn.Linear(1, 1)))
    state = {"transformer." + key: value for key, value in model.state_dict().items()}
    state["text_encoder.weight"] = torch.ones(1, 1)
    metadata = dict(artifact_metadata("training_bundle", dict(model.config)), **checkpoint_metadata(spec),
                    companion_path=str(companion), train_text_encoder="0")
    path = tmp_path / "bundle.safetensors"
    save_file(state, str(path), metadata=metadata)
    loaded = loader.load_qwen_image_21_components(str(path), torch_dtype=torch.float32, load_text_encoder=False)["transformer"]
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
