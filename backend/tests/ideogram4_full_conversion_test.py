import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

from core.models.ideogram4.full_conversion import convert_checkpoint, plan_conversion


def source(tmp_path, **changes):
    state = {"attention.qkv.weight": torch.arange(24).reshape(6, 4).float().to(torch.float8_e4m3fn),
             "attention.qkv.weight_scale": torch.linspace(0.03, 0.08, 6),
             "attention.qkv.bias": torch.arange(6).to(torch.bfloat16),
             "normal.weight": torch.ones(4), "empty": torch.empty(0),
             "counter": torch.tensor(7)}
    state.update(changes)
    path = tmp_path / "source.safetensors"
    save_file(state, str(path), metadata={"format": "pt"})
    return path, state


def test_effective_values_and_native_forward_match(tmp_path):
    path, state = source(tmp_path)
    original = path.read_bytes()
    output = tmp_path / "expanded.safetensors"
    convert_checkpoint(path, output, work_bytes=31)
    restored = load_file(str(output))
    expected = state["attention.qkv.weight"].to(torch.bfloat16) * state["attention.qkv.weight_scale"].to(torch.bfloat16)[:, None]
    assert torch.equal(restored["attention.qkv.weight"], expected)
    assert "attention.qkv.weight_scale" not in restored
    for key in ("normal.weight", "counter", "empty", "attention.qkv.bias"):
        assert restored[key].dtype == state[key].dtype
        assert torch.equal(restored[key], state[key])
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    quantized = Fp8Linear(4, 6, True, torch.bfloat16)
    quantized.load_state_dict({"weight": state["attention.qkv.weight"],
                              "weight_scale": state["attention.qkv.weight_scale"],
                              "bias": state["attention.qkv.bias"]})
    plain = torch.nn.Linear(4, 6).to(torch.bfloat16)
    plain.load_state_dict({"weight": restored["attention.qkv.weight"], "bias": restored["attention.qkv.bias"]})
    value = torch.randn(3, 4).to(torch.bfloat16)
    assert torch.equal(plain(value), quantized._dequant_forward(value))
    plain(value).float().square().mean().backward()
    assert plain.weight.grad is not None and torch.isfinite(plain.weight.grad).all()
    assert path.read_bytes() == original
    with safe_open(str(output), framework="pt", device="cpu") as handle:
        record = json.loads(handle.metadata()["sushi.ideogram_fp8_expansion"])
    assert record["converted_matrices"] == 1
    assert record["recovered_unquantized_original"] is False


@pytest.mark.parametrize("scale", [torch.ones(1), torch.ones(6, dtype=torch.float16),
                                  torch.zeros(6), torch.full((6,), float("nan"))])
def test_invalid_scales_do_not_publish_output(tmp_path, scale):
    path, _ = source(tmp_path, **{"attention.qkv.weight_scale": scale})
    output = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError):
        convert_checkpoint(path, output, work_bytes=31)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("key,value", [("foreign.weight", torch.ones(2, 2, dtype=torch.int8)),
                                     ("foreign.comfy_quant", torch.ones(2, dtype=torch.uint8)),
                                     ("orphan.weight_scale", torch.ones(2))])
def test_foreign_quantization_is_rejected(tmp_path, key, value):
    path, _ = source(tmp_path, **{key: value})
    with pytest.raises(ValueError):
        plan_conversion(path)


def test_overwrite_and_small_budget_refused(tmp_path):
    path, _ = source(tmp_path)
    with pytest.raises(ValueError, match="source"):
        convert_checkpoint(path, path)
    output = tmp_path / "result.safetensors"
    with pytest.raises(ValueError, match="one row"):
        convert_checkpoint(path, output, work_bytes=1)
    assert not output.exists()
    convert_checkpoint(path, output, work_bytes=31)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        convert_checkpoint(path, output)
    assert output.read_bytes() == before


def test_plain_loader_constructs_on_meta_and_assigns_exact_values(monkeypatch):
    from core.models.ideogram4.vendor.transformer import Ideogram4Transformer2DModel
    from core.models.ideogram4.ideogram4_loader import _build_ideogram4_transformer_from_state
    native = Ideogram4Transformer2DModel(in_channels=8, num_layers=2, attention_head_dim=12,
        num_attention_heads=2, intermediate_size=32, adaln_dim=8, llm_features_dim=12, mrope_section=(2,2,2))
    native.to(torch.bfloat16)
    state = native.state_dict()
    original = Ideogram4Transformer2DModel.from_config
    constructed = []

    def construct(*args, **kwargs):
        model = original(*args, **kwargs)
        constructed.append(all(value.is_meta for value in model.parameters()))
        return model

    monkeypatch.setattr(Ideogram4Transformer2DModel, "from_config", construct)
    restored = _build_ideogram4_transformer_from_state(dict(native.config), state, torch.bfloat16, "plain")
    assert constructed == [True]
    for key, value in restored.state_dict().items():
        assert torch.equal(value, state[key])
        assert not value.is_meta
        if value.numel():
            assert value.data_ptr() == state[key].data_ptr()
    assert all(parameter.requires_grad for parameter in restored.parameters())
    broken = dict(state)
    broken.pop(next(iter(broken)))
    with pytest.raises(RuntimeError, match="Missing key"):
        _build_ideogram4_transformer_from_state(dict(native.config), broken, torch.bfloat16, "missing")
