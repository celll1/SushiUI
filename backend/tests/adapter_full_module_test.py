"""Complete auxiliary weights must not be treated as low-rank deltas."""
import pytest
import torch
from torch import nn

from core.adapters.full_module import FullModulePayload


def payload(dtype=torch.float32):
    module = nn.Sequential(nn.Linear(3, 2, dtype=dtype), nn.LayerNorm(2, dtype=dtype))
    module.register_buffer("scale", torch.ones(1, dtype=dtype))
    return FullModulePayload(module)


def test_full_payload_roundtrip_and_optimizer_ownership():
    original, restored = payload(), payload()
    state = original.export_tensors()
    restored.load_tensors(state)
    assert all(torch.equal(value, restored.branch_tensors()[key]) for key, value in state.items())
    assert set(map(id, original.trainable_parameters())) == set(map(id, original.module.parameters()))
    assert original.spec_constants() == () and not original.CHECKPOINT_WRITES_ALPHA
    state["0.weight"].zero_()
    assert not torch.equal(state["0.weight"], original.branch_tensors()["0.weight"])


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "integer", "nan", "overflow"])
def test_full_payload_refusal_leaves_every_tensor_unchanged(corruption):
    target = payload(torch.float16)
    before = target.export_tensors()
    values = {key: value.float() + 1 for key, value in before.items()}
    key = "1.bias"
    if corruption == "missing": del values[key]
    elif corruption == "extra": values["alpha"] = torch.ones(1)
    elif corruption == "shape": values[key] = torch.ones(3)
    elif corruption == "integer": values[key] = values[key].long()
    elif corruption == "nan": values[key][0] = float("nan")
    else: values[key][0] = 1e20
    with pytest.raises(ValueError): target.load_tensors(values)
    assert all(torch.equal(value, target.branch_tensors()[key]) for key, value in before.items())


def test_full_payload_refuses_changed_layout_and_nonfinite_export():
    target = payload()
    with torch.no_grad(): target.module[0].weight[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"): target.export_tensors()
    target.module.register_buffer("extra", torch.ones(1))
    with pytest.raises(ValueError, match="layout"): target.branch_tensors()


def test_training_export_preserves_full_payload_without_alpha():
    from types import SimpleNamespace
    from core.training.adapters.base_adapter import BaseLoRAAdapter
    full = payload()
    ordinary = SimpleNamespace(export_tensors=lambda: {"lora_down.weight": torch.ones(1, 3)})
    adapter = SimpleNamespace(CHECKPOINT_WRITES_ALPHA=True, lora_alpha=8)
    state = BaseLoRAAdapter.export_state_dict(adapter, {"rewiring_modules": full, "ordinary": ordinary})
    assert {key.removeprefix("rewiring_modules.") for key in state
            if key.startswith("rewiring_modules.")} == set(full.branch_tensors())
    assert "ordinary.alpha" in state and "rewiring_modules.alpha" not in state
