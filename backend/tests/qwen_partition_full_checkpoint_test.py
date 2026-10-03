"""Full FT global-adapter training, artifact resume, and ordinary-forward isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.models.qwen_image_21.artifact import load_transformer
from core.models.qwen_image_21.partition_global import (
    prepare_training_global_adapter, stash_partition_global_payload,
)
from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel
from core.training.adapters.qwen_image_21_adapter import QwenImage21FullParameterAdapter
from core.training.qwen_partition import PartitionBox, full_canvas_position_ids


def _trainer(model=None, dtype=torch.float32):
    if model is None:
        model = QwenImage21Transformer2DModel(
            in_channels=8, out_channels=8, num_layers=1, attention_head_dim=16,
            num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        ).to(dtype=dtype)
    return SimpleNamespace(
        transformer=model, text_encoder=torch.nn.Linear(1, 1),
        vae=torch.nn.Linear(1, 1), learning_rate=1e-3, unet_lr=1e-3,
        train_unet=True, train_text_encoder=False, trains_base_weights=True,
        model_path="unused", _loaded_checkpoint_path=None,
        config={"dit_partition_training_enabled": True,
                "dit_partition_global_adapter_enabled": True,
                "dit_partition_global_rank": 4, "dit_partition_global_tokens": 3},
    )


def _forward(model, grid, box=None, residual=None):
    height, width = grid.shape[1:3]
    image = grid if box is None else grid[:, box.top:box.bottom, box.left:box.right]
    h, w = image.shape[1:3]
    slots = torch.cat([torch.zeros(1, 2, dtype=torch.bool),
                       torch.ones(1, h * w // 4, dtype=torch.bool)], dim=1)
    return model(
        hidden_states=image.reshape(1, h * w, 8), timestep=torch.tensor([0.5]),
        encoder_hidden_states=torch.ones(1, 2, 32, dtype=grid.dtype),
        img_shapes=[[(1, h, w)]], img_mask=slots,
        target_spatial_position_ids=(None if box is None else
                                    full_canvas_position_ids(height, width, box)),
        target_input_residual=residual, return_dict=False,
    )[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sharded", [False, True])
def test_full_ft_global_trains_saves_and_resumes_without_affecting_ordinary_forward(tmp_path, monkeypatch, dtype, sharded):
    if sharded:
        import core.training.adapters.qwen_image_21_adapter as adapter_module
        monkeypatch.setattr(adapter_module, "DEFAULT_EXPORT_SHARD_BYTES", 1024)
    torch.manual_seed(31)
    trainer = _trainer(dtype=dtype)
    trainer.config["dit_partition_halo_tokens"] = 4
    adapter = QwenImage21FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    global_adapter = trainer.transformer.qwen_partition_global_adapter
    params = [p for group in adapter.arch_param_groups() for p in group["params"]]
    assert {id(p) for p in global_adapter.parameters()} <= {id(p) for p in params}
    optimizer = torch.optim.SGD(params, lr=1e-3)
    grid = torch.randn(1, 4, 4, 8, dtype=dtype)
    box = PartitionBox(0, 0, 2, 4)
    prediction = _forward(trainer.transformer, grid, box, global_adapter(grid, box))
    prediction.float().square().mean().backward()
    assert global_adapter.out.weight.grad is not None
    optimizer.step()
    assert bool(global_adapter.out.weight.count_nonzero())
    saved = {key: value.detach().clone() for key, value in global_adapter.branch_tensors().items()}
    path = adapter.write_checkpoint(3, 0, tmp_path / "run_step_000003.safetensors")
    assert path.name.endswith(".index.json") is sharded
    loaded, variant = load_transformer(str(path), dict(trainer.transformer.config), dtype)
    assert variant == "bf16"
    assert loaded._qwen_partition_global_payload["metadata"]["qwen_partition_global_training_halo_tokens"] == "4"
    assert not hasattr(loaded, "qwen_partition_global_adapter")
    assert not any("global_adapter" in name for name, _ in loaded.named_parameters())
    assert all(value.device.type == "cpu" for value in loaded._qwen_partition_global_payload["state"].values())
    trainer.transformer.eval()
    with torch.no_grad():
        ordinary = _forward(loaded, grid)
        torch.testing.assert_close(ordinary, _forward(trainer.transformer, grid), rtol=1e-5, atol=1e-6)
        payload = loaded._qwen_partition_global_payload
        del loaded._qwen_partition_global_payload
        torch.testing.assert_close(_forward(loaded, grid), ordinary, rtol=0, atol=0)
        loaded._qwen_partition_global_payload = payload
    resumed = _trainer(loaded)
    resumed._loaded_checkpoint_path = str(path)
    QwenImage21FullParameterAdapter(resumed).prepare_models_for_training()
    for name, tensor in loaded.qwen_partition_global_adapter.branch_tensors().items():
        assert torch.equal(tensor, saved[name])
    assert not hasattr(loaded, "_qwen_partition_global_payload")
    assert prepare_training_global_adapter(resumed, dtype=dtype) is loaded.qwen_partition_global_adapter


@pytest.mark.parametrize("change", ["disabled", "rank", "tokens", "missing_saved", "lora_base"])
def test_full_ft_resume_refuses_to_discard_or_initialize_global_weights(tmp_path, change):
    trainer = _trainer()
    QwenImage21FullParameterAdapter(trainer).prepare_models_for_training()
    path = QwenImage21FullParameterAdapter(trainer).write_checkpoint(3, 0, tmp_path / "checkpoint")
    loaded, _ = load_transformer(str(path), dict(trainer.transformer.config), torch.float32)
    resumed = _trainer(loaded)
    resumed._loaded_checkpoint_path = str(path)
    if change == "disabled":
        resumed.config["dit_partition_global_adapter_enabled"] = False
    elif change in {"rank", "tokens"}:
        resumed.config[f"dit_partition_global_{change}"] = 7
    elif change == "lora_base":
        resumed.trains_base_weights = False
    else:
        del loaded._qwen_partition_global_payload
    with pytest.raises(ValueError):
        QwenImage21FullParameterAdapter(resumed).prepare_models_for_training()


def test_full_ft_global_payload_refuses_partial_weights():
    trainer = _trainer()
    state = {"qwen_partition_global_adapter.out.weight": torch.zeros(32, 4)}
    metadata = {"qwen_partition_global_adapter": "latent_summary_v1",
                "qwen_partition_global_rank": "4", "qwen_partition_global_tokens": "3"}
    with pytest.raises(ValueError, match="incomplete"):
        stash_partition_global_payload(trainer.transformer, state, metadata)
    assert "qwen_partition_global_adapter.out.weight" in state
