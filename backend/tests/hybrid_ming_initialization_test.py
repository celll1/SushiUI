"""Real tiny Ming encoding, native trajectories, fitting and resident state rollback."""
from copy import deepcopy

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from ming_image_01_conditioning_test import write_tiny_package
from hybrid_ming_test import hybrid_for
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.residual_unet import restore_topology
from core.rewiring.ming_initialization import initialize_loaded_ming
from core.rewiring.fitting import InitializationCancelled


@pytest.mark.parametrize("profile", ["generation_edit","layer_decompose"])
@pytest.mark.parametrize("outcome", ["complete","cancel","cancel_trajectory","failure"])
def test_real_native_owner_initializes_and_restores_every_component(tmp_path,profile,outcome,monkeypatch):
    from api.param_defaults import DIT_HYBRID_INITIALIZATION_DEFAULTS
    torch.manual_seed(99)
    source = tmp_path/"native"
    source.mkdir()
    _,model,_ = write_tiny_package(source,profile=profile)
    restore_topology(model,None)
    weights = source/"transformer"/"diffusion_pytorch_model.safetensors"
    save_file(model.state_dict(),str(weights))
    native = load_components(source,dtype=torch.float32,conditioner_attention="eager")
    roots = (native.transformer,native.conditioner,native.vae)
    states = [{name:value.clone() for name,value in root.state_dict().items()} for root in roots]
    flags = [(p,p.requires_grad) for root in roots for p in root.parameters()]
    modes = [(module,module.training) for root in roots for module in root.modules()]
    scheduler = deepcopy(dict(native.scheduler.config))
    items = []
    for index in range(2):
        path = tmp_path/f"synthetic-{index}.png"
        Image.new("RGBA",(128,128),(50+index*50,100,150,255)).save(path)
        items.append(dict(image_path=str(path),caption="word5 word6"))
    options = dict(DIT_HYBRID_INITIALIZATION_DEFAULTS,output_name="initialized",sample_count=2,
        holdout_count=1,resolution=128,teacher_steps=2,optimizer="adamw",stochastic_rounding=False,
        attention_backend="native",phases=[dict(kind="data_noising",updates=2,learning_rate=.003),
                                           dict(kind="native_trajectory",updates=1,learning_rate=.003)])
    events = []
    if outcome == "failure":
        def fail(*args,**kwargs):
            raise OSError("publication failed")
        monkeypatch.setattr("core.rewiring.ming_initialization.export_initialized_ming_package",fail)
    def execute():
        return initialize_loaded_ming(dict(ming_native=native),source,hybrid_for(native.transformer),
            items,options,events.append,
            lambda:(outcome == "cancel" and any(event["stage"] == "fitting" for event in events)
                    or outcome == "cancel_trajectory" and any(event["stage"] == "teacher_trajectory" for event in events)),
            execution_device="cpu")
    if outcome == "complete":
        result = execute()
        assert [event["completed"] for event in events if event["stage"] == "fitting"][-1] == 3
        restored = load_components(result["model_path"],dtype=torch.float32,conditioner_attention="eager")
        assert restored.transformer._hybrid_initialization == result["initialization"]
        assert restored.artifact.profile == profile
    else:
        with pytest.raises(InitializationCancelled if outcome.startswith("cancel") else OSError):
            execute()
        assert not (tmp_path/"initialized").exists()
    assert native.transformer.gradient_checkpointing is False
    assert not hasattr(native.transformer,"rewiring_modules")
    assert all(parameter.requires_grad == flag and parameter.grad is None for parameter,flag in flags)
    assert all(module.training == mode for module,mode in modes)
    assert dict(native.scheduler.config) == scheduler
    for root,state in zip(roots,states):
        assert all(torch.equal(value,root.state_dict()[name]) for name,value in state.items())
    assert not torch.cuda.is_initialized()
