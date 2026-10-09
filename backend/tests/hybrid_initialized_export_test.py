"""The new initialized artifact must preserve the entire native payload exactly."""
import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hybrid_anima_test import native, forward
from core.rewiring.fitting import FitExample, FitPhase, fit_teacher_modules, InitializationCancelled
from core.rewiring.initialized_export import export_initialized_anima
from core.rewiring.provenance import read_checkpoint_metadata as read_initialization


def prepare(model, root, prefix="net."):
    from core.models.anima.residual_unet import native_config
    source = root/"native.safetensors"
    state = {prefix+name: value.detach().contiguous() for name,value in model.state_dict().items()
             if not name.startswith("rewiring_modules.")}
    save_file(state, str(source), metadata=dict(transformer_config=json.dumps(native_config(model)),
                                              model_type="anima", unrelated="preserved"))
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    example = FitExample((torch.ones(1,4,1,8,12), torch.tensor([.5]), torch.ones(1,3,12)),
                         dict(padding_mask=torch.zeros(1,1,8,12)))
    fit_teacher_modules(model,[FitPhase("data_noising",1,.003,[example])],
                        lambda module,args,kwargs: module(*args,**kwargs),
                        lambda parameters,lr: torch.optim.AdamW(parameters,lr=lr),
                        source_sha256=checksum, calibration_sha256="2"*64, sample_count=1,
                        seed=77, clip_norm=1.)
    return source, state, checksum


@pytest.mark.parametrize("prefix", ["net.", ""])
def test_initialized_export_preserves_native_tensors_and_loads_fitted_model(native,tmp_path,prefix):
    from core.models.anima.anima_loader import load_anima_dit
    model, spec = native
    source, state, checksum = prepare(model,tmp_path,prefix)
    record = model._hybrid_initialization
    result = export_initialized_anima(model,source,"initialized.safetensors")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    path = tmp_path/"initialized.safetensors"
    assert result["model_path"] == str(path.resolve())
    assert result["initialization"] == record
    with safe_open(str(path),framework="pt",device="cpu") as handle:
        assert handle.metadata()["unrelated"] == "preserved"
        assert read_initialization(handle.metadata(),spec) == record
        for name,value in state.items():
            assert torch.equal(handle.get_tensor(name),value)
    loaded = load_anima_dit(str(path),dtype=torch.float32)
    from core.rewiring.inspect import inspect_model
    inspected = inspect_model(path)
    assert inspected["initialization"] == record
    assert inspected["topology"] == spec
    assert inspect_model(source)["initialization"] is None
    assert loaded._hybrid_initialization == record
    assert loaded._residual_unet_spec == spec
    assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in model.state_dict().items())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(forward(model),forward(loaded),rtol=1e-5,atol=1e-6)
    assert not list(tmp_path.glob(".sushi-hybrid-*"))


@pytest.mark.parametrize("reason", ["cancel", "checksum", "weights", "exists", "publish", "source_changed"])
def test_export_failure_never_publishes_partial_output(native,tmp_path,monkeypatch,reason):
    model,_ = native
    source,state,checksum = prepare(model,tmp_path)
    output = tmp_path/"initialized.safetensors"
    options = {}
    if reason == "cancel":
        calls = []
        def cancel():
            calls.append(1)
            return len(calls) == 3
        options["cancelled"] = cancel
        error = InitializationCancelled
    elif reason == "source_changed":
        calls = []
        def mutate():
            calls.append(1)
            if len(calls) == 2:
                with source.open("r+b") as stream:
                    stream.seek(-1,2)
                    last = stream.read(1)
                    stream.seek(-1,2)
                    stream.write(bytes([last[0]^1]))
            return False
        options["cancelled"] = mutate
        error = ValueError
    elif reason == "checksum":
        model._hybrid_initialization["source_sha256"] = "0"*64
        error = ValueError
    elif reason == "weights":
        with torch.no_grad():
            next(model.rewiring_modules.parameters()).add_(1.)
        error = ValueError
    elif reason == "exists":
        output.write_bytes(b"owner artifact")
        error = FileExistsError
    else:
        def fail(*args):
            raise OSError("publication failed")
        monkeypatch.setattr("core.rewiring.initialized_export.os.link",fail)
        error = OSError
    with pytest.raises(error):
        export_initialized_anima(model,source,"initialized.safetensors",**options)
    assert (hashlib.sha256(source.read_bytes()).hexdigest() == checksum) == (reason != "source_changed")
    assert not list(tmp_path.glob(".sushi-hybrid-*"))
    if reason == "exists":
        assert output.read_bytes() == b"owner artifact"
    else:
        assert not output.exists()


@pytest.mark.parametrize("name", ["../outside.safetensors", "con.safetensors", "bad:name.safetensors", "bad.png"])
def test_invalid_output_names_refused(native,tmp_path,name):
    model,_ = native
    source,_,_ = prepare(model,tmp_path)
    with pytest.raises(ValueError,match="basename"):
        export_initialized_anima(model,source,name)
