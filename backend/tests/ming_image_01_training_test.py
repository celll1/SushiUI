"""CPU native Ming objective, frame supervision and optimizer ownership."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming training test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.conditioning import MingConditioning
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.training import flow_loss, optimizer_groups
from core.models.ming_image_01.residual_unet import restore_topology
from ming_image_01_rewiring_test import config, spec_for


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_unet_training_target_reference_and_dual_condition_gradients(padding):
    torch.manual_seed(91)
    model = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    if padding == "learned":
        torch.nn.init.constant_(model.x_pad_token, 0.02)
        torch.nn.init.constant_(model.cap_pad_token, 0.03)
    restore_topology(model, spec_for(model))
    frames = 3 if padding == "learned" else 1
    clean = torch.randn(2, 16, frames, 16, 16)
    condition = MingConditioning(torch.randn(2, 4, 8, requires_grad=True),
                                torch.randn(2, 3, 24, requires_grad=True))
    references = [torch.randn(16, 1, 16, 16), torch.randn(16, 2, 16, 16)]
    weights = torch.ones(2, frames)
    if frames > 1: weights[:, 0] = 0
    result = flow_loss(model, clean=clean, noise=torch.randn_like(clean), sigma=torch.tensor([0.2, 0.8]),
                       conditioning=condition, frame_weights=weights, reference_latents=references)
    assert result.prediction.shape == clean.shape
    result.loss.backward()
    assert all(layer.attention.to_q.weight.grad.abs().sum() > 0 for layer in model.layers)
    assert condition.queries.grad.abs().sum() > 0
    assert condition.direct.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()


class Oracle:
    in_channels = 16
    config = SimpleNamespace(multi_frame_output=True, cap_feat_dim=8, dim=24)

    def __call__(self, **kwargs):
        self.seen = kwargs
        outputs = []
        for value in kwargs["x"]:
            target = torch.zeros_like(value)
            target[:, 0] = 1000
            target[:, 1] = 2
            target[:, 2] = 4
            outputs.append(torch.cat([target, torch.ones_like(target[:, :1]) * 1e6], dim=1))
        return (outputs,)


def oracle_inputs():
    clean = torch.zeros(2, 16, 3, 4, 4)
    return dict(clean=clean, noise=clean.clone(), sigma=torch.tensor([0., 1.]),
                conditioning=MingConditioning(torch.ones(2, 4, 8), torch.ones(2, 3, 24)),
                frame_weights=torch.tensor([[0., 1., 3.], [0., 1., 1.]]))


def test_weighted_target_only_loss_native_time_and_aligned_zero_dropout():
    oracle = Oracle()
    result = flow_loss(oracle, **oracle_inputs(), cfg_drop_mask=torch.tensor([True, False]))
    torch.testing.assert_close(result.per_item_loss, torch.tensor([13., 10.]))
    assert result.loss.item() == 11.5
    assert torch.equal(result.model_time, torch.tensor([1., 0.]))
    assert oracle.seen["cap_feats"][0].count_nonzero() == 0
    assert oracle.seen["cap_feats_2"][0].count_nonzero() == 0
    assert oracle.seen["cap_feats"][1].count_nonzero() > 0


@pytest.mark.parametrize("key,value,message", [
    ("sigma", torch.tensor([float("nan"), .5]), "sigma"),
    ("sigma", torch.tensor([-.1, .5]), "sigma"),
    ("frame_weights", torch.zeros(2, 3), "supervised"),
    ("frame_weights", torch.tensor([[0., -1., 1.], [0., 1., 1.]]), "supervision"),
    ("cfg_drop_mask", torch.ones(2), "dropout"),
    ("reference_latents", [torch.zeros(16, 1, 4, 4)], "per training"),
])
def test_invalid_training_inputs_refused(key, value, message):
    inputs = oracle_inputs()
    inputs[key] = value
    with pytest.raises(ValueError, match=message):
        flow_loss(Oracle(), **inputs)


def test_optimizer_groups_have_explicit_rates_and_unique_parameter_ownership():
    dit = torch.nn.Linear(2, 2)
    language, connector = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    connector.requires_grad_(False)
    conditioner = SimpleNamespace(component_groups=lambda: {"language": (language,), "connector": (connector,)})
    groups = optimizer_groups(dit, conditioner, dit_lr=.001, conditioner_lrs={"language": .0001})
    assert [group["name"] for group in groups] == ["dit", "language"]
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == 4
    with pytest.raises(ValueError, match="language"):
        optimizer_groups(dit, conditioner, dit_lr=.001, conditioner_lrs={})
    with pytest.raises(ValueError, match="Unknown"):
        optimizer_groups(dit, conditioner, dit_lr=.001, conditioner_lrs={"foreign": .01})
