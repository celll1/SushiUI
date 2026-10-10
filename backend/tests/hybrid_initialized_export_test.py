"""The new initialized artifact must preserve the entire native payload exactly."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hybrid_anima_test import native, forward
from core.rewiring.fitting import FitExample, FitPhase, fit_teacher_modules, InitializationCancelled
from core.rewiring.initialized_export import export_initialized_anima
from core.rewiring.provenance import read_checkpoint_metadata as read_initialization


def test_qwen_initialized_component_retains_payload_and_reloads(tmp_path):
    from hybrid_qwen_test import model_and_spec
    from qwen_partition_full_checkpoint_test import _forward
    from core.models.qwen_image_21.artifact import artifact_metadata, load_transformer
    from core.models.qwen_image_21.residual_unet import restore_topology
    from core.rewiring.initialized_export import export_initialized_transformer
    from core.rewiring.inspect import inspect_model
    torch.manual_seed(97)
    model,spec,_ = model_and_spec()
    config = dict(model.config)
    source = tmp_path/"native.safetensors"
    native_state = {name:value.detach().clone().contiguous() for name,value in model.state_dict().items()}
    metadata = artifact_metadata("transformer",config)
    metadata["unrelated"] = "retained"
    save_file(native_state,str(source),metadata=metadata)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    restore_topology(model,spec)
    example = FitExample((torch.randn(1,8,12,8),),{})
    fit_teacher_modules(model,[FitPhase("data_noising",2,.003,[example,example],.25)],
        lambda module,args,kwargs:_forward(module,args[0]),
        lambda parameters,lr:torch.optim.AdamW(parameters,lr=lr),
        source_sha256=checksum,calibration_sha256="2"*64,sample_count=1,seed=97,clip_norm=1.)
    result = export_initialized_transformer(model,source,"initialized.safetensors",architecture="qwen_image_21")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == checksum
    with safe_open(result["model_path"],framework="pt",device="cpu") as handle:
        assert handle.metadata()["unrelated"] == "retained"
        for name,value in native_state.items():
            assert torch.equal(handle.get_tensor(name),value)
    loaded,variant = load_transformer(result["model_path"],config,torch.float32)
    assert variant == "bf16"
    assert loaded._hybrid_initialization == result["initialization"]
    assert loaded._residual_unet_spec == spec
    assert all(torch.equal(value,loaded.state_dict()[name]) for name,value in model.state_dict().items())
    inspected = inspect_model(result["model_path"])
    assert inspected["initialization"] == result["initialization"]
    assert inspected["topology"] == spec
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(_forward(model,example.args[0]),_forward(loaded,example.args[0]),rtol=1e-5,atol=1e-6)
    assert not list(tmp_path.glob(".sushi-hybrid-*"))
    from core.rewiring.initialized_export import export_initialized_qwen_package
    from core.models.qwen_image_21.artifact import load_manifest
    assets = tmp_path/"assets"
    assets.mkdir()
    for name in ("text_encoder","vae","text_encoder_config"):
        (assets/name).write_bytes(b"unchanged asset")
    (assets/"processor").mkdir()
    (assets/"scheduler").mkdir()
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(dict(model_type="qwen_image_21",format_version="1",
        components=dict(transformer=source.name,**{key:"assets/"+key for key in
            ("text_encoder","vae","text_encoder_config","processor","scheduler")}),
        transformer_config=config,vae_config={},unrelated="preserved")),encoding="utf-8")
    package = export_initialized_qwen_package(model,manifest_path,tmp_path/"initialized-package")
    declaration = load_manifest(package["model_path"])
    assert Path(declaration.transformer).parent == tmp_path/"initialized-package"
    assert Path(declaration.text_encoder) == assets/"text_encoder"
    assert Path(declaration.vae) == assets/"vae"
    assert inspect_model(package["model_path"])["initialization"] == result["initialization"]
    restored,_ = load_transformer(declaration.transformer,declaration.transformer_config,torch.float32)
    assert all(torch.equal(value,restored.state_dict()[name]) for name,value in model.state_dict().items())
    assert sorted(p.name for p in (tmp_path/"initialized-package").iterdir()) == ["manifest.json","transformer.safetensors"]
    with pytest.raises(FileExistsError):
        export_initialized_qwen_package(model,manifest_path,tmp_path/"initialized-package")
    with pytest.raises(InitializationCancelled):
        export_initialized_qwen_package(model,manifest_path,tmp_path/"cancelled-package",cancelled=lambda:True)
    assert not (tmp_path/"cancelled-package").exists()
    with pytest.raises(InitializationCancelled):
        export_initialized_qwen_package(model,manifest_path,tmp_path/"late-cancelled-package",
            cancelled=lambda:any(tmp_path.glob(".sushi-qwen-init-*/package/manifest.json")))
    assert not (tmp_path/"late-cancelled-package").exists()
    assert not list(tmp_path.glob(".sushi-qwen-init-*"))


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
    fit_teacher_modules(model,[FitPhase("data_noising",1,.003,[example],.25)],
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
