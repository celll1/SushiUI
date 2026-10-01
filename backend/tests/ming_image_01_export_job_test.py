"""Offline real background export job, native package restoration and refusal."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from core.models.ming_image_01.loader import load_components
from core.models.common.int8_runtime_quantize import quantize_linears_in_place
from ming_image_01_conditioning_test import write_tiny_package
from api import quantized_export_job as job


def make_manager(root, profile="generation_edit"):
    root.mkdir()
    write_tiny_package(root, profile=profile)
    c = load_components(root, dtype=torch.float32, conditioner_attention="eager")
    audit = quantize_linears_in_place(c.transformer, arch="ming_image_01", compute_dtype=torch.float32)
    return SimpleNamespace(current_model_info={"type":"ming_image_01", "source":str(root)},
        ming_image_01_components={"ming_native":c, "transformer":c.transformer},
        _load_model_lock=threading.Lock(), _runtime_int8_audit=audit), c


@pytest.fixture(autouse=True)
def reset_job(monkeypatch):
    monkeypatch.setattr(job, "_job", None)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_real_native_background_job_and_reload(tmp_path, monkeypatch, profile):
    manager, c = make_manager(tmp_path/"source", profile)
    monkeypatch.setattr(job, "_default_export_root", lambda:str(tmp_path/"exports"))
    status = job.export_status(manager)
    assert status["exportable"] and status["output_kind"] == "native_package"
    assert not status["suggested_path"].endswith(".safetensors")
    before = {key:value.clone() for key,value in c.transformer.state_dict().items()}
    threads = []
    native_thread = threading.Thread
    def capture(*args, **kwargs):
        t = native_thread(*args, **kwargs)
        threads.append(t)
        return t
    monkeypatch.setattr(job.threading, "Thread", capture)
    destination = tmp_path/"exports"/"converted"
    started = job.start_export(manager, str(destination), link_siblings=False)
    assert started["state"] == "running" and started["output_kind"] == "native_package"
    threads[0].join(timeout=30)
    assert not threads[0].is_alive(), "CPU export did not finish"
    done = job.export_status(manager)["job"]
    assert done["state"] == "completed", done
    assert done["processed"] == done["total"] == 6
    assert Path(done["written_path"]) == destination/"manifest.json"
    note = json.loads((destination/"quantization.json").read_text())
    assert note["runtime_audit"] == manager._runtime_int8_audit
    assert note["inventory"]["int8"] + note["inventory"]["e4m3"] > 0
    loaded = load_components(done["written_path"], dtype=torch.float32, conditioner_attention="eager")
    assert loaded.artifact.profile == profile
    assert loaded.transformer._residual_unet_spec == c.transformer._residual_unet_spec
    for key,value in before.items():
        assert torch.equal(value.view(torch.uint8), loaded.transformer.state_dict()[key].view(torch.uint8)), key
        assert torch.equal(value.view(torch.uint8), c.transformer.state_dict()[key].view(torch.uint8)), key
    assert not torch.cuda.is_initialized()


def test_native_path_and_overwrite_preflight(tmp_path):
    manager,_ = make_manager(tmp_path/"source")
    for destination, overwrite in (("",False),(str(tmp_path/"bad.safetensors"),False),
        (str(tmp_path/"new"),True),(str(tmp_path/"source"/"inside"),False),
        (str(tmp_path/"source"),False)):
        with pytest.raises(job.ExportUnavailableError):
            job.start_export(manager,destination,overwrite=overwrite)
    assert job._job is None


@pytest.mark.parametrize("change", ["architecture", "transformer", "conditioner"])
def test_changed_model_refused_after_lifecycle_gate(tmp_path, change):
    manager,c = make_manager(tmp_path/"source")
    modules = [("transformer",c.transformer)]
    expected_native = (c,c.transformer,c.conditioner,c.vae)
    if change == "conditioner":
        from copy import deepcopy
        c.conditioner = deepcopy(c.conditioner)
    elif change == "architecture":
        manager.current_model_info = {"type":"sdxl"}
    else:
        from copy import deepcopy
        replacement = deepcopy(c)
        manager.ming_image_01_components = {"ming_native":replacement,"transformer":replacement.transformer}
    job._job = {"job_id":"changed", "state":"running"}
    output = tmp_path/"export"
    job._run_export(manager,"changed","ming_image_01",modules,str(output),
        link_siblings=False,overwrite=False,source=str(tmp_path/"source"),source_root=str(tmp_path/"source"),expected_native=expected_native)
    assert job._job["state"] == "failed"
    assert not output.exists()


def test_export_failure_restores_component_residency(tmp_path, monkeypatch):
    manager,c = make_manager(tmp_path/"source")
    from core.models.ming_image_01 import checkpoint, block_offload
    moves = []
    original = block_offload.move_component
    def move(module,device):
        moves.append((id(module),str(device)))
        return original(module,device)
    def fail(*a,**k):
        raise OSError("serialization failed")
    monkeypatch.setattr(block_offload,"move_component",move)
    monkeypatch.setattr(checkpoint,"save_bundle",fail)
    with pytest.raises(OSError,match="serialization failed"):
        job._export_ming_package(manager,str(tmp_path/"export"),None,"unavailable",lambda *a:None)
    assert moves[:3] == moves[3:]
    assert not (tmp_path/"export").exists()


def test_loaded_quantized_package_exports_without_inventing_audit(tmp_path):
    manager,c = make_manager(tmp_path/"source")
    manager._runtime_int8_audit = None
    output = tmp_path/"export"
    job._job = {"job_id":"loaded", "state":"running"}
    job._run_export(manager,"loaded","ming_image_01",[("transformer",c.transformer)],str(output),
        link_siblings=True,overwrite=False,source=str(tmp_path/"source"),source_root=str(tmp_path/"source"))
    assert job._job["state"] == "completed", job._job
    note = json.loads((output/"quantization.json").read_text())
    assert note["runtime_audit"] is None and note["audit_note"]
    assert job._job["result"]["linked_siblings"] == []
    assert not torch.cuda.is_initialized()
