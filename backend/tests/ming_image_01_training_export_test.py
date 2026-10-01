"""Offline run-scoped ConvRot export through production jobs and native reload."""
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from api import training_export_job as job
from core.training.ming_full_export import export_ming_full_checkpoint
from ming_image_01_conversion_test import make_source, digest_tree


def checkpoint(root, profile="generation_edit"):
    root.mkdir()
    source=root/"run_step_000002"
    _, topology=make_source(source,profile)
    (source/"training.json").write_text(json.dumps({"schema_version":1,"kind":"full_parameter_weights",
        "profile":profile,"step":2,"epoch":0}))
    (root/"run_step_000002_state.json").write_text('{"global_step":2}')
    (root/"run_step_000002_optimizer.pt").write_bytes(b"complete optimizer sidecar")
    return source,topology


@pytest.mark.parametrize("profile",["generation_edit","layer_decompose"])
def test_public_config_manual_job_native_reload_and_inference(tmp_path,monkeypatch,profile):
    source,topology=checkpoint(tmp_path/"run",profile)
    root=source.parent
    before=digest_tree(source)
    from api import routes
    from core.training.training_config import TrainingConfigGenerator
    from core.training.train_runner import _prepare_training_process_config
    import yaml
    request=routes.TrainingRunCreateRequest(training_method="full_finetune",base_model_path=str(source),
        total_steps=2,training_export_format="int8_convrot",train_text_encoder=False,ming_conditioner_groups=[])
    routes._check_qwen_full_finetune_params(request)
    config=TrainingConfigGenerator.generate_full_finetune_config(request.model_dump(),run_name="run",
        base_model_path=str(source),output_dir=str(root))
    _,train,_,method=_prepare_training_process_config(yaml.safe_load(config),str(source))
    assert method=="full_finetune" and train["training_export_format"]=="int8_convrot"
    run=SimpleNamespace(id=812,run_name="run",output_dir=str(root),training_method="full_finetune",
        base_model_path=str(source),status="completed")
    db=SimpleNamespace(query=lambda *a:SimpleNamespace(filter=lambda *a:SimpleNamespace(first=lambda:run)))
    monkeypatch.setattr(routes.training_process_manager,"is_live",lambda _:False)
    threads=[];native_thread=threading.Thread
    def thread(*args,**kwargs):
        value=native_thread(*args,**kwargs);threads.append(value);return value
    monkeypatch.setattr(job.threading,"Thread",thread)
    queued=asyncio.run(routes.start_training_export(run.id,routes.TrainingExportRequest(),db))
    assert queued["state"]=="queued"
    threads[0].join(timeout=30)
    assert not threads[0].is_alive()
    status=job.export_status(root,run.id)
    assert status["state"]=="completed" and not job.job_is_running(run.id)
    published=Path(status["path"])
    assert digest_tree(source)==before
    for name in ("mllm","connector","mlp","vae","scheduler"):
        assert digest_tree(source/name)==digest_tree(published/name)
    from core.model_loader import ModelLoader
    loaded=ModelLoader.load_ming_image_01_from_path(str(published),torch.float32)
    c=loaded["ming_native"]
    assert c.transformer._residual_unet_spec==topology and c.artifact.profile==profile
    assert any(key.endswith(".comfy_quant") for key in c.transformer.state_dict())
    from ming_image_01_pipeline_test import manager,params
    from PIL import Image
    value,_=manager(profile)
    value._install_ming_image_01(loaded,model_id="training-export",source_type="local",source=str(published))
    images=[Image.new("RGBA",(64,64),(80,60,40,200))] if profile=="layer_decompose" else []
    states=[]
    result,_,_=value._ming_image_run(params(width=64,height=64,ming_image_01_layers=2),images=images,
        step_callback=lambda i,t,latents:states.append(bool(torch.isfinite(latents).all())))
    assert states==[True] and result.groups[0].values[0].mode=="RGBA"
    # Same run/source is idempotent unless replacement is requested.
    assert job.start_export(run.id,"run",root)==status
    with pytest.raises(FileExistsError):export_ming_full_checkpoint(source,root)
    overwritten=job.export_run(run.id,"run",root,overwrite=True)
    assert overwritten["state"]=="completed" and digest_tree(source)==before
    assert not (root/"exports/.int8_convrot_previous").exists()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure",["conversion","publication"])
def test_failed_replacement_retains_previous_export_and_float_checkpoint(tmp_path,monkeypatch,failure):
    source,_=checkpoint(tmp_path/"run")
    root=source.parent
    published=export_ming_full_checkpoint(source,root)
    original=digest_tree(published);floating=digest_tree(source)
    if failure=="conversion":
        from core.models.ming_image_01 import conversion
        def fail(*a,**k):raise OSError("injected conversion failure")
        monkeypatch.setattr(conversion,"convert_package",fail)
    else:
        from core.training import ming_full_export
        replace=ming_full_export.os.replace
        def fail(source_path,target):
            if Path(source_path).name=="package" and Path(target)==published:
                raise OSError("injected publication failure")
            return replace(source_path,target)
        monkeypatch.setattr(ming_full_export.os,"replace",fail)
    with pytest.raises(OSError,match="injected"):
        job.export_run(813,"run",root,overwrite=True)
    assert job.export_status(root,813)["state"]=="failed" and not job.job_is_running(813)
    assert digest_tree(published)==original and digest_tree(source)==floating
    assert not (root/"exports/.int8_convrot_previous").exists()
    assert not list((root/"exports").glob(".ming_int8_convrot_*"))


def test_latest_complete_native_checkpoint_skips_incomplete_or_invalid_packages(tmp_path):
    source,_=checkpoint(tmp_path/"run")
    root=source.parent
    newer=root/"run_step_000004";newer.mkdir()
    (root/"run_step_000004_state.json").write_text('{}')
    (root/"run_step_000004_optimizer.pt").write_bytes(b"optimizer")
    assert job.latest_complete_checkpoint(root,"run")==source
    (root/"run_step_000002_optimizer.pt").unlink()
    with pytest.raises(FileNotFoundError,match="complete floating"):
        job.latest_complete_checkpoint(root,"run")


@pytest.mark.parametrize("method,value",[("lora","int8_convrot"),("full_finetune","fp8"),
    ("full_finetune",[]),("full_finetune",None)])
def test_direct_config_refuses_invalid_training_export(method,value):
    from core.models.ming_image_01.training_options import preflight_training
    with pytest.raises(ValueError,match="training export"):
        preflight_training({"ming_conditioner_groups":[],"training_export_format":value},method=method)


@pytest.mark.parametrize('kind',['schema','packed'])
def test_native_selection_refuses_non_full_or_packed_candidates(tmp_path,kind):
    source,_=checkpoint(tmp_path/'run')
    root=source.parent
    if kind=='schema':
        payload=json.loads((source/'training.json').read_text());payload['schema_version']=True
        (source/'training.json').write_text(json.dumps(payload))
    else:
        from core.models.ming_image_01.conversion import convert_package
        packed=root/'run_step_000004'
        convert_package(source,packed,format_='int8_convrot')
        (root/'run_step_000004_state.json').write_text('{}')
        (root/'run_step_000004_optimizer.pt').write_bytes(b'optimizer')
        assert job.latest_complete_checkpoint(root,'run')==source
        source=packed
    with pytest.raises(ValueError):export_ming_full_checkpoint(source,root)
    assert not (root/'exports/int8_convrot').exists()


@pytest.mark.parametrize('method,status,live,code',[
    ('lora','completed',False,400),('full_finetune','running',True,409),
    ('full_finetune','completed',True,409),
])
def test_manual_export_refuses_non_full_and_live_runs(monkeypatch,method,status,live,code):
    from api import routes
    from core.training import training_config
    from fastapi import HTTPException
    run=SimpleNamespace(id=814,training_method=method,status=status,base_model_path='unused')
    db=SimpleNamespace(query=lambda *a:SimpleNamespace(filter=lambda *a:SimpleNamespace(first=lambda:run)))
    monkeypatch.setattr(training_config,'_detect_arch',lambda _:'ming_image_01')
    monkeypatch.setattr(routes.training_process_manager,'is_live',lambda _:live)
    def forbidden(*a,**k):raise AssertionError('Forbidden run queued an export')
    monkeypatch.setattr(job,'start_export',forbidden)
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.start_training_export(run.id,routes.TrainingExportRequest(),db))
    assert error.value.status_code==code
