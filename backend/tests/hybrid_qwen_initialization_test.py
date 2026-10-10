"""The Qwen initializer must use native target-only packed inputs and loss."""
import pytest
import torch

from hybrid_qwen_test import model_and_spec
from core.models.qwen_image_21.residual_unet import restore_topology
from core.rewiring.qwen_initialization import qwen_inputs
from core.rewiring.anima_initialization import noising_input
from core.rewiring.fitting import FitExample, FitPhase, fit_teacher_modules


def test_packed_noising_and_native_teacher_update():
    torch.manual_seed(73)
    model,spec,_ = model_and_spec()
    source = {name:value.clone() for name,value in model.state_dict().items()}
    restore_topology(model,spec)
    added = {name:value.clone() for name,value in model.rewiring_modules.state_dict().items()}
    latent = torch.randn(1,96,8)
    value,sigma = noising_input(latent,torch.randn_like(latent),torch.tensor([.4]))
    kwargs = qwen_inputs(value,sigma,torch.ones(1,3,32),torch.ones(1,3,dtype=torch.bool),8,12)
    assert kwargs["img_mask"].shape == (1,3+24)
    assert not kwargs["img_mask"][:,:3].any()
    assert kwargs["img_mask"][:,3:].all()
    def predict(network,args,inputs):
        return network(*args,**inputs)[0][:,-96:]
    example = FitExample((),kwargs)
    record = fit_teacher_modules(model,[FitPhase("data_noising",2,.003,[example,example],.25)],predict,
        lambda parameters,lr:torch.optim.AdamW(parameters,lr=lr),source_sha256="1"*64,
        calibration_sha256="2"*64,sample_count=1,seed=73,clip_norm=1.)
    assert record["phases"][0]["updates"] == 2
    assert record["phases"][0]["skip_learning_rate_multiplier"] == .25
    assert all(torch.equal(model.state_dict()[name],value) for name,value in source.items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.isfinite(value).all() for value in model.rewiring_modules.state_dict().values())
    assert any(not torch.equal(value,model.rewiring_modules.state_dict()[name]) for name,value in added.items())


@pytest.mark.parametrize("height,width",[(7,12),(8,11),(8,10)])
def test_invalid_grid_is_refused(height,width):
    with pytest.raises(ValueError,match="even native image grid"):
        qwen_inputs(torch.ones(1,96,8),torch.tensor([.5]),torch.ones(1,3,32),
                    torch.ones(1,3,dtype=torch.bool),height,width)
