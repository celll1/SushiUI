"""Real streaming offline conversion; no companion model construction or GPU."""
import hashlib
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from ming_image_01_conditioning_test import write_tiny_package
from core.models.ming_image_01.conversion import convert_package


def make_source(root,profile):
    root.mkdir()
    conditioner,model,_=write_tiny_package(root,profile=profile)
    from core.models.ming_image_01.model_definition import create_transformer
    config=dict(model.config)
    config.update(dim=256,n_heads=8,n_kv_heads=8,axes_dims=[8,12,12])
    model=create_transformer(config,device="cpu",dtype=torch.float32).eval()
    if profile=="layer_decompose":
        with torch.no_grad():
            model.x_pad_token.fill_(.02)
            model.cap_pad_token.fill_(.03)
    model.save_config(root/"transformer")
    mlp=json.loads((root/"mlp/config.json").read_text())
    mlp["diffusion_inner_dim"]=256
    (root/"mlp/config.json").write_text(json.dumps(mlp))
    conditioner.native.diffusion_inner_dim=256
    conditioner.native.proj_directvlm[1]=torch.nn.Linear(48,256)
    save_file(conditioner.projection_state_dict(),str(root/"mlp/model.safetensors"))
    from core.models.ming_image_01.residual_unet import restore_topology,model_topology_metadata
    from ming_image_01_rewiring_test import spec_for
    topology=spec_for(model)
    restore_topology(model,topology)
    save_file(model.state_dict(),str(root/"transformer/diffusion_pytorch_model.safetensors"),
              metadata=model_topology_metadata(model))
    return model,topology


def digest_tree(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("profile",["generation_edit","layer_decompose"])
@pytest.mark.parametrize("format_",["int8","int8_convrot"])
def test_real_offline_quantization_shards_generation_and_frozen_companions(tmp_path,monkeypatch,profile,format_):
    source=tmp_path/"source"
    original,topology=make_source(source,profile)
    before=digest_tree(source)
    from core.models.ming_image_01.vendor.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    from core.models.ming_image_01.vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    def forbidden(*a,**k): raise AssertionError("Offline conversion constructed a companion model")
    progress=[]
    with monkeypatch.context() as patch:
        patch.setattr(BailingMM2NativeForConditionalGeneration,"from_pretrained",forbidden)
        patch.setattr(AutoencoderKLQwenImage,"from_pretrained",forbidden)
        result=convert_package(source,tmp_path/"converted",format_=format_,max_shard_bytes=4096,
                               progress_callback=lambda *values:progress.append(values))
    assert digest_tree(source)==before
    assert sum(result["report"]["converted"].values())>0
    assert progress[-1][0]==progress[-1][1]==len(original.state_dict())
    manifest=Path(result["manifest"])
    assert (manifest.parent/"transformer/diffusion_pytorch_model.safetensors.index.json").is_file()
    for name in ("mllm","connector","mlp","vae","scheduler"):
        assert digest_tree(source/name)==digest_tree(manifest.parent/name)
    from core.model_loader import ModelLoader
    assert ModelLoader.detect_model_type(str(manifest))=="ming_image_01"
    loaded=ModelLoader.load_ming_image_01_from_path(str(manifest),torch.float32)
    c=loaded["ming_native"]
    assert c.transformer._residual_unet_spec==topology
    assert c.artifact.profile==profile
    protected=("all_x_embedder.","all_final_layer.","t_embedder.","cap_embedder.")
    for key,value in original.state_dict().items():
        if key.startswith(protected) or ".adaLN_modulation." in key:
            assert torch.equal(value,c.transformer.state_dict()[key]),key
    from ming_image_01_pipeline_test import manager,params,cpu_only
    from PIL import Image
    value,_=manager(profile)
    value._install_ming_image_01(loaded,model_id="offline",source_type="local",source=str(manifest))
    images=[Image.new("RGBA",(64,64),(80,60,40,200))] if profile=="layer_decompose" else []
    states=[]
    output,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=2),images=images,
        step_callback=lambda i,t,latents:states.append(bool(torch.isfinite(latents).all())))
    assert states==[True]
    assert output.groups[0].values[0].mode=="RGBA"
    cpu_only(c)


@pytest.mark.parametrize("stop",["cancel","failure"])
def test_interrupted_conversion_never_publishes_partial_package(tmp_path,monkeypatch,stop):
    source=tmp_path/"source"
    make_source(source,"generation_edit")
    before=digest_tree(source)
    from core.models.common import convrot_export
    calls=[]
    def progress(*a): calls.append(a)
    if stop=="failure":
        def fail(*a,**k): raise OSError("quantization failed")
        monkeypatch.setattr(convrot_export,"rotate_and_quantize",fail)
    with pytest.raises(InterruptedError if stop=="cancel" else OSError):
        convert_package(source,tmp_path/"output",format_="int8_convrot",max_shard_bytes=4096,
            cancelled=lambda:bool(calls) if stop=="cancel" else False,progress_callback=progress)
    assert not (tmp_path/"output").exists()
    assert not list(tmp_path.glob(".output.ming-*"))
    assert digest_tree(source)==before


def test_source_destination_and_already_quantized_refusals(tmp_path):
    source=tmp_path/"source"
    make_source(source,"generation_edit")
    for destination in (source,source/"nested"):
        with pytest.raises((ValueError,FileExistsError)):
            convert_package(source,destination,format_="int8")
    with pytest.raises(ValueError,match="positive"):
        convert_package(source,tmp_path/"zero",format_="int8",max_shard_bytes=0)
    first=convert_package(source,tmp_path/"converted",format_="int8")
    with pytest.raises(ValueError,match="original floating"):
        convert_package(first["manifest"],tmp_path/"again",format_="int8_convrot")
    assert not (tmp_path/"again").exists()


def test_real_cli_entry_uses_cpu_default_and_native_manifest(tmp_path,monkeypatch):
    import runpy
    source=tmp_path/"source"
    make_source(source,"generation_edit")
    module=runpy.run_path(str(Path(__file__).resolve().parents[2]/"subapps/ming_image_01_convert.py"))
    monkeypatch.setattr("sys.argv",["ming_image_01_convert",str(source),str(tmp_path/"output"),"--format","int8"])
    result=module["main"]()
    assert Path(result["manifest"]).is_file()
    assert result["report"]["device"]=="cpu"
    assert not torch.cuda.is_initialized()


def test_zero_weight_audit_keeps_valid_json_and_exact_zero_codes(tmp_path):
    source=tmp_path/"source"
    model,_=make_source(source,"generation_edit")
    with torch.no_grad():
        model.layers[0].attention.to_q.weight.zero_()
    from core.models.ming_image_01.residual_unet import model_topology_metadata
    save_file(model.state_dict(),str(source/"transformer/diffusion_pytorch_model.safetensors"),
              metadata=model_topology_metadata(model))
    result=convert_package(source,tmp_path/"output",format_="int8")
    json.dumps(result,allow_nan=False)
    row=next(row for row in result["report"]["layer_audit"] if row["name"]=="layers.0.attention.to_q")
    assert row["undefined_metrics"] and row["int8_rel_rms"] is None
    from core.models.ming_image_01.loader import load_components
    c=load_components(result["manifest"],dtype=torch.float32,conditioner_attention="eager")
    assert not c.transformer.layers[0].attention.to_q.weight.float().count_nonzero()
    assert not torch.cuda.is_initialized()
