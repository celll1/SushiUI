"""Initialized Ming packages own their unchanged assets and restore fitted modules."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hybrid_ming_test import hybrid_for
from ming_image_01_conditioning_test import write_tiny_package
from core.models.ming_image_01.loader import load_components
from core.models.ming_image_01.residual_unet import restore_topology
from core.rewiring.fitting import FitExample, FitPhase, fit_teacher_modules, InitializationCancelled
from core.rewiring.initialized_export import export_initialized_ming_package


@pytest.mark.parametrize("profile", ["generation_edit","layer_decompose"])
@pytest.mark.parametrize("sharded", [False,True])
def test_publish_fitted_ming_preserves_native_assets_and_restores(tmp_path,profile,sharded):
    torch.manual_seed(98)
    source = tmp_path/"native"
    source.mkdir()
    _,model,_ = write_tiny_package(source,profile=profile)
    restore_topology(model,None)
    weights = source/"transformer"/"diffusion_pytorch_model.safetensors"
    native = {name:value.detach().clone().contiguous() for name,value in model.state_dict().items()}
    save_file(native,str(weights),metadata={"unrelated":"preserved"})
    if sharded:
        weights.unlink()
        names = sorted(native)
        mapping = {}
        for index,keys in enumerate((names[:len(names)//2],names[len(names)//2:])):
            filename = f"model-{index}.safetensors"
            save_file({key:native[key] for key in keys},str(weights.parent/filename),metadata={"unrelated":"preserved"})
            mapping.update({key:filename for key in keys})
        (weights.parent/"model.safetensors.index.json").write_text(json.dumps(dict(
            metadata=dict(total_size=sum(value.numel()*value.element_size() for value in native.values())),
            weight_map=mapping)),encoding="utf-8")
    (source/"quantization.json").write_text('{"note":"source provenance"}',encoding="utf-8")
    (source/"LICENSE").write_text("Synthetic license fixture",encoding="utf-8")
    assets = {path.relative_to(source):path.read_bytes() for path in source.rglob("*") if path.is_file()}
    from core.rewiring.ming_source import source_identity
    checksum,_ = source_identity(weights.parent)
    restore_topology(model,hybrid_for(model))
    frames = 3 if profile == "layer_decompose" else 1
    example = FitExample(([torch.randn(16,frames,16,24)],torch.tensor([.5]),[torch.randn(4,8)]),
                         dict(cap_feats_2=[torch.randn(5,24)]))
    def predict(module,args,kwargs):
        return module(*args,**kwargs).sample[0][:,:frames]
    fit_teacher_modules(model,[FitPhase("data_noising",2,.003,[example,example],.25)],predict,
        lambda parameters,lr:torch.optim.AdamW(parameters,lr=lr),
        source_sha256=checksum,calibration_sha256="2"*64,sample_count=1,seed=98,clip_norm=1.)
    result = export_initialized_ming_package(model,source,tmp_path/"initialized")
    destination = Path(result["model_path"])
    for relative,payload in assets.items():
        assert (source/relative).read_bytes() == payload
        if not (relative.parent.name == "transformer" and (relative.name.endswith(".safetensors") or relative.name.endswith(".index.json"))):
            assert (destination/relative).read_bytes() == payload
    seen = set()
    for path in (destination/"transformer").glob("*.safetensors"):
        with safe_open(str(path),framework="pt",device="cpu") as handle:
            assert handle.metadata()["unrelated"] == "preserved"
            for name in set(handle.keys()).intersection(native):
                assert torch.equal(handle.get_tensor(name),native[name])
                seen.add(name)
    assert seen == set(native)
    loaded = load_components(destination,dtype=torch.float32,conditioner_attention="eager")
    assert loaded.artifact.profile == profile
    assert loaded.transformer._residual_unet_spec == model._residual_unet_spec
    assert loaded.transformer._hybrid_initialization == result["initialization"]
    assert all(torch.equal(value,loaded.transformer.state_dict()[name]) for name,value in model.state_dict().items())
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(predict(model,example.args,example.kwargs),
                                   predict(loaded.transformer,example.args,example.kwargs),rtol=1e-5,atol=1e-6)
    with pytest.raises(FileExistsError):
        export_initialized_ming_package(model,source,destination)
    with pytest.raises(InitializationCancelled):
        export_initialized_ming_package(model,source,tmp_path/"cancelled",cancelled=lambda:True)
    with pytest.raises(InitializationCancelled):
        export_initialized_ming_package(model,source,tmp_path/"late-cancelled",
            cancelled=lambda:any(tmp_path.glob(".sushi-ming-init-*/package/manifest.json")))
    assert not (tmp_path/"cancelled").exists() and not (tmp_path/"late-cancelled").exists()
    assert not list(tmp_path.glob(".sushi-ming-init-*"))
    with pytest.raises(ValueError,match="outside"):
        export_initialized_ming_package(model,source,source/"nested-output")
    assert not (source/"nested-output").exists()
    extra = source/"mllm"/"changing-asset.txt"
    extra.write_bytes(b"native asset")
    def mutate_asset():
        if any(tmp_path.glob(".sushi-ming-init-*/package/mllm/changing-asset.txt")):
            extra.write_bytes(b"changed source asset length")
        return False
    with pytest.raises(ValueError,match="changed during publication"):
        export_initialized_ming_package(model,source,tmp_path/"asset-race",cancelled=mutate_asset)
    assert not (tmp_path/"asset-race").exists()
    assert not list(tmp_path.glob(".sushi-ming-init-*"))
    extra.unlink()
    with torch.no_grad():
        next(model.rewiring_modules.parameters()).add_(1)
    with pytest.raises(ValueError,match="changed after fitting"):
        export_initialized_ming_package(model,source,tmp_path/"changed")
    assert not (tmp_path/"changed").exists()
    assert not list(tmp_path.glob(".sushi-ming-init-*"))
    assert not torch.cuda.is_initialized()
