"""Initial dense placement validates ownership before any tensor device move."""

import pytest
import torch
from core.training.model_staging import stage_dense_block_model


def fixture():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(4)])
    model.bridge = torch.nn.Conv2d(2, 3, 3)
    model.register_buffer("clock", torch.arange(4))
    return model


def test_only_resident_blocks_and_codec_bridge_move():
    model = fixture()
    stage_dense_block_model(model, model.layers, 3, "meta")
    assert model.clock.is_meta and model.bridge.weight.is_meta
    assert model.layers[0].weight.is_meta
    assert all(p.device.type == "cpu" for layer in model.layers[1:] for p in layer.parameters())


@pytest.mark.parametrize("first,second", [(0,2), (2,3)])
def test_cross_block_shared_parameter_refused_before_move(first, second):
    model = fixture()
    model.layers[second].weight = model.layers[first].weight
    with pytest.raises(ValueError, match="shared"):
        stage_dense_block_model(model, model.layers, 3, "meta")
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_invalid_layer_owner_and_strategy_refused_before_move():
    for count, layers in ((4, None), (2, torch.nn.ModuleList([torch.nn.Linear(3,3) for _ in range(4)]))):
        model = fixture()
        with pytest.raises(ValueError):
            stage_dense_block_model(model, model.layers if layers is None else layers, count, "meta")
        assert all(p.device.type == "cpu" for p in model.parameters())
