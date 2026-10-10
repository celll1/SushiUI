"""Job ownership must survive cancellation, worker failure and dispatch failure."""
import threading
import json
from types import SimpleNamespace

import pytest

from api import hybrid_initialization_job as job
from api.rewiring_routes import HybridInitializationRequest, rewiring_schema
from core.model_state_coordinator import ModelStateCoordinator, ModelStateBusyError
from core.rewiring.fitting import InitializationCancelled
from hybrid_anima_test import native


def test_anima_source_guard_accepts_only_native_layout(native,tmp_path):
    from safetensors.torch import save_file
    from core.models.anima.residual_unet import restore_topology,native_config
    model,spec = native
    restore_topology(model,None)
    source = tmp_path/"native.safetensors"
    save_file({"net."+key:value for key,value in model.state_dict().items()},str(source),
              metadata=dict(model_type="anima",transformer_config=json.dumps(native_config(model))))
    manager = SimpleNamespace(current_model_info=dict(type="anima",source=str(source)),
        anima_components=dict(transformer=model,text_encoder=object(),vae=object(),scheduler=object(),
                              tokenizer=object(),t5_tokenizer=object()))
    options = dict(model_path=str(source),output_name="initialized.safetensors",spec=spec,resolution=512)
    checked,_,declaration = job._checked_source(manager,options)
    assert checked == source and declaration == spec
    restore_topology(model,spec)
    with pytest.raises(ValueError,match="native source"):
        job._checked_source(manager,options)


def test_qwen_source_guard_validates_package_and_added_layout(tmp_path):
    import torch
    from safetensors.torch import save_file
    from hybrid_qwen_test import model_and_spec
    from core.models.qwen_image_21.artifact import artifact_metadata
    from core.models.qwen_image_21.residual_unet import restore_topology
    model,spec,_ = model_and_spec()
    source = tmp_path/"native"
    source.mkdir()
    for name in ("te","vae","te_config"):
        (source/name).touch()
    (source/"processor").mkdir()
    config = dict(model.config)
    save_file(model.state_dict(),str(source/"transformer.safetensors"),metadata=artifact_metadata("transformer",config))
    (source/"manifest.json").write_text(json.dumps(dict(model_type="qwen_image_21",format_version="1",
        components=dict(transformer="transformer.safetensors",text_encoder="te",vae="vae",processor="processor",
                        text_encoder_config="te_config"),transformer_config=config,vae_config={})),encoding="utf-8")
    manager = SimpleNamespace(current_model_info=dict(type="qwen_image_21",source=str(source)),
        qwen_image_21_components=dict(transformer=model,text_encoder=torch.nn.Linear(1,1),
                                     vae=torch.nn.Linear(1,1),processor=object(),scheduler=object()))
    options = dict(model_path=str(source),output_name="initialized-qwen",spec=spec,resolution=512)
    checked,_,declaration = job._checked_source(manager,options)
    assert checked == source and declaration == spec
    restore_topology(model,spec)
    with pytest.raises(ValueError,match="native source"):
        job._checked_source(manager,options)


def test_request_defaults_and_schema_agree():
    request = HybridInitializationRequest(model_path="native.safetensors", output_name="new.safetensors", dataset_id=1)
    defaults = rewiring_schema()["initialization_defaults"]
    assert all(request.model_dump()[key] == value for key, value in defaults.items())
    assert "ming_image_01" in rewiring_schema()["initializer_architectures"]


@pytest.mark.parametrize("profile", ["generation_edit","layer_decompose"])
def test_ming_source_guard_validates_native_package(tmp_path,profile):
    import torch
    from safetensors.torch import save_file
    from ming_image_01_conditioning_test import write_tiny_package
    from hybrid_ming_test import hybrid_for
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.residual_unet import restore_topology
    source = tmp_path/"native"
    source.mkdir()
    _,model,_ = write_tiny_package(source,profile=profile)
    restore_topology(model,None)
    save_file(model.state_dict(),str(source/"transformer"/"diffusion_pytorch_model.safetensors"))
    native = load_components(source,dtype=torch.float32,conditioner_attention="eager")
    manager = SimpleNamespace(current_model_info=dict(type="ming_image_01",source=str(source)),
                              ming_image_01_components=dict(ming_native=native))
    spec = hybrid_for(native.transformer)
    options = dict(model_path=str(source),output_name="initialized-ming",spec=spec,resolution=128)
    checked,_,declaration = job._checked_source(manager,options)
    assert checked == source and declaration == spec
    restore_topology(native.transformer,spec)
    with pytest.raises(ValueError,match="native source"):
        job._checked_source(manager,options)


@pytest.mark.parametrize("changes", [dict(resolution=513), dict(holdout_count=32),
    dict(item_ids=[1, 1]), dict(phases=[]), dict(teacher_cfg=float("nan")), dict(seed=True)])
def test_request_refuses_invalid_recipe(changes):
    with pytest.raises(ValueError):
        HybridInitializationRequest(model_path="native", output_name="new", dataset_id=1, **changes)


@pytest.mark.parametrize("multiplier", [0, -1, 2, True, float("nan"), float("inf")])
def test_phase_refuses_invalid_skip_multiplier(multiplier):
    from api.rewiring_routes import HybridInitializationPhaseRequest
    with pytest.raises(ValueError):
        HybridInitializationPhaseRequest(kind="data_noising", updates=1, learning_rate=.001,
                                         skip_learning_rate_multiplier=multiplier)


def test_skip_multiplier_defaults_and_explicit_request_serialization():
    from api.param_defaults import DIT_HYBRID_INITIALIZATION_PHASE_DEFAULTS
    from api.rewiring_routes import HybridInitializationPhaseRequest
    phase = HybridInitializationPhaseRequest(kind="data_noising", updates=1, learning_rate=.001)
    assert phase.skip_learning_rate_multiplier == DIT_HYBRID_INITIALIZATION_PHASE_DEFAULTS["skip_learning_rate_multiplier"]
    request = HybridInitializationRequest(model_path="native", output_name="new", dataset_id=1,
        phases=[dict(kind="data_noising", updates=1, learning_rate=.001, skip_learning_rate_multiplier=.25)])
    assert request.model_dump()["phases"][0]["skip_learning_rate_multiplier"] == .25


def test_skip_multiplier_refuses_underflow_and_absent_skip_modules():
    from hybrid_rewiring_test import declaration
    from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
    spec = declaration()
    no_skips = build_hybrid_spec(legacy_topology(spec), stream_widths=spec["stream_widths"],
        boundaries=spec["boundaries"], refiners=spec["refiners"], skips=[])
    for changes in (dict(learning_rate=5e-324, skip_learning_rate_multiplier=5e-324),
                    dict(learning_rate=.001, skip_learning_rate_multiplier=.25)):
        with pytest.raises(ValueError):
            HybridInitializationRequest(model_path="native", output_name="new", dataset_id=1,
                spec=no_skips, phases=[dict(kind="data_noising", updates=1, **changes)])


@pytest.fixture
def owner(monkeypatch):
    from api import routes
    monkeypatch.setattr(routes, "_fp8_scaled_mm_busy_reason", lambda: None)
    coordinator = ModelStateCoordinator()
    monkeypatch.setattr(job, "model_state_coordinator", coordinator)
    monkeypatch.setattr(job, "_job", None)
    monkeypatch.setattr(job, "_thread", None)
    monkeypatch.setattr(job, "_cancel", threading.Event())
    monkeypatch.setattr(job, "_checked_source", lambda manager, options: ("native", {}, {}))
    return SimpleNamespace(_load_model_lock=threading.Lock()), coordinator


@pytest.mark.parametrize("outcome", ["cancel", "failure", "complete"])
def test_worker_releases_exclusive_owner(monkeypatch, owner, outcome):
    manager, coordinator = owner
    entered, release = threading.Event(), threading.Event()

    def execute(*args):
        entered.set()
        assert release.wait(5)
        if outcome == "cancel":
            assert args[-1]()
            raise InitializationCancelled("cancelled")
        if outcome == "failure":
            raise ValueError("broken")
        return {"model_path": "new"}

    monkeypatch.setattr(job, "execute", execute)
    snapshot = job.start(manager, {"output_name": "new"}, [])
    try:
        assert entered.wait(5)
        assert job.job_is_running()
        with pytest.raises(ModelStateBusyError):
            coordinator.begin_generation()
        with pytest.raises(ModelStateBusyError):
            job.cancel("stale")
        if outcome == "cancel":
            assert job.cancel(snapshot["job_id"])["state"] == "cancelling"
    finally:
        release.set()
        job._thread.join(5)
    assert not job.job_is_running()
    assert coordinator.snapshot()["mutation"] is None
    assert job.status()["state"] == {"cancel": "cancelled", "failure": "failed", "complete": "completed"}[outcome]
    if outcome == "complete":
        assert job.status()["completed"] == job.status()["total"] == 1
        assert job.status()["stage"] == "completed"
    coordinator.begin_generation()
    coordinator.end_generation()


def test_dispatch_failure_releases_owner(monkeypatch, owner):
    manager, coordinator = owner
    def fail_start(self):
        raise RuntimeError("cannot dispatch")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot dispatch"):
        job.start(manager, {"output_name": "new"}, [])
    assert coordinator.snapshot()["mutation"] is None
    assert not job.job_is_running()
    assert job.status()["state"] == "failed"
