"""Exclusive, cancellable initialization of a new artifact from the loaded base."""
from copy import deepcopy
from pathlib import Path
import threading
import time
import uuid

from core.model_state_coordinator import model_state_coordinator, ModelStateBusyError
from core.rewiring.resolution_schedule import initialization_resolutions

INITIALIZER_ARCHITECTURES = ("anima", "qwen_image_21", "ming_image_01")
_lock = threading.Lock()
_job = None
_thread = None
_cancel = threading.Event()


def status():
    with _lock:
        return deepcopy(_job) if _job is not None else {"state": "idle"}


def job_is_running():
    with _lock:
        return _thread is not None and _thread.is_alive()


def cancel(job_id):
    with _lock:
        if _job is None or _job["job_id"] != job_id:
            raise ModelStateBusyError("Initialization job ID does not match the current job")
        if _job["state"] in ("queued", "running", "cancelling"):
            _cancel.set()
            _job["state"] = "cancelling"
        return deepcopy(_job)


def _checked_source(manager, options):
    import torch
    from safetensors import safe_open
    from core.rewiring.inspect import inspect_model
    from core.models.anima.residual_unet import validate_topology_identity
    from core.rewiring.spec import preview_plan

    info = manager.current_model_info or {}
    if info.get("type") == "qwen_image_21":
        return _checked_qwen_source(manager,options)
    if info.get("type") == "ming_image_01":
        return _checked_ming_source(manager,options)
    if info.get("type") not in INITIALIZER_ARCHITECTURES:
        raise ValueError("Hybrid initialization requires a loaded native Anima, Qwen or Ming model")
    source = Path(options["model_path"]).resolve(strict=True)
    if source != Path(info.get("source") or "").resolve():
        raise ValueError("Initialization source must match the currently loaded model")
    if not source.is_file() or source.suffix.lower() != ".safetensors":
        raise ValueError("Initialization requires a standalone native Anima source")
    name = options["output_name"]
    from core.rewiring.initialized_export import validate_output_name
    validate_output_name(name)
    if (source.parent/name).exists() or (source.parent/name).is_symlink():
        raise FileExistsError("Initialization output already exists")
    inspected = inspect_model(source)
    components = manager.anima_components
    if not components or any(components.get(key) is None for key in
            ("transformer", "text_encoder", "vae", "scheduler", "tokenizer", "t5_tokenizer")):
        raise ValueError("Loaded Anima components are incomplete")
    model = components["transformer"]
    if (inspected["topology"] is not None or getattr(model,"_residual_unet_spec",None) is not None
            or hasattr(model,"rewiring_modules") or hasattr(model,"vae_latent_bridge")
            or info.get("vae_alignment") is not None):
        raise ValueError("Initialization requires a native source without existing U-Net or VAE swap")
    if getattr(model,"_block_offloader",None) is not None or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Initialization requires resident native blocks without existing gradients")
    with safe_open(str(source),framework="pt",device="cpu") as handle:
        keys = set(handle.keys())
    prefix = "net." if "net.x_embedder.proj.1.weight" in keys else ""
    if any(prefix+name not in keys for name,_ in model.named_parameters()):
        raise ValueError("Loaded native parameter layout differs from source; remove adapters and reload the base")
    if any(name.endswith(".weight") and value.dtype not in (torch.float32,torch.float16,torch.bfloat16)
           for name,value in model.state_dict().items()):
        raise ValueError("Anima initialization currently requires floating native weights")
    spec = options["spec"] or inspected["hybrid_default_spec"]
    if spec is None or spec.get("method") != "hybrid_unet_ssdit":
        raise ValueError("Initialization requires a hybrid U-Net declaration")
    spec = validate_topology_identity(spec,model)
    stride = inspected["latent_scale_factor"]
    for side in initialization_resolutions(options):
        preview_plan(spec,side//stride,side//stride)
    return source,components,spec


def _checked_qwen_source(manager,options):
    import torch
    from safetensors import safe_open
    from core.models.qwen_image_21.artifact import load_manifest
    from core.models.qwen_image_21.residual_unet import validate_topology_identity
    from core.rewiring.inspect import inspect_model
    from core.rewiring.initialized_export import validate_package_name
    from core.rewiring.spec import preview_plan
    source = Path(options["model_path"]).resolve(strict=True)
    if source != Path(manager.current_model_info.get("source") or "").resolve():
        raise ValueError("Initialization source must match the loaded Qwen package")
    if not source.is_dir() or not (source/"manifest.json").is_file():
        raise ValueError("Qwen initialization requires a native split package directory")
    validate_package_name(options["output_name"])
    output = source.parent/options["output_name"]
    if output.exists() or output.is_symlink():
        raise FileExistsError("Qwen initialization output already exists")
    manifest = load_manifest(str(source))
    for key in ("text_encoder","vae","processor","text_encoder_config","scheduler"):
        asset = getattr(manifest,key)
        if asset is not None and not Path(asset).exists():
            raise FileNotFoundError(f"Qwen source asset is missing: {key}")
    if Path(manifest.transformer).suffix != ".safetensors":
        raise ValueError("Qwen initialization currently requires an unsharded transformer component")
    inspected = inspect_model(source)
    components = manager.qwen_image_21_components
    if not components or any(components.get(key) is None for key in
                            ("transformer","text_encoder","vae","processor","scheduler")):
        raise ValueError("Loaded Qwen components are incomplete")
    model = components["transformer"]
    if (inspected["topology"] is not None or getattr(model,"_residual_unet_spec",None) is not None
            or hasattr(model,"rewiring_modules") or hasattr(model,"vae_latent_bridge")
            or manifest.transformer_config.get("vae_alignment") is not None):
        raise ValueError("Qwen initialization requires a native source without U-Net or VAE swap")
    if getattr(model,"_block_offloader",None) is not None or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Qwen initialization requires resident blocks without existing gradients")
    with safe_open(manifest.transformer,framework="pt",device="cpu") as handle:
        keys,metadata = set(handle.keys()),handle.metadata() or {}
    if metadata.get("model_type") != "qwen_image_21" or metadata.get("component") != "transformer":
        raise ValueError("Qwen initialization requires a native transformer component")
    if metadata.get("variant","bf16") != "bf16" or any(
            p.dtype not in (torch.float32,torch.float16,torch.bfloat16) for p in model.parameters()):
        raise ValueError("Qwen initialization currently requires floating native weights")
    if any(name not in keys for name,_ in model.named_parameters()):
        raise ValueError("Loaded Qwen parameter layout differs from source; remove adapters and reload")
    spec = validate_topology_identity(options["spec"] or inspected["hybrid_default_spec"],dict(model.config))
    if spec["method"] != "hybrid_unet_ssdit":
        raise ValueError("Qwen initialization requires a hybrid declaration")
    validate_topology_identity(spec,manifest.transformer_config)
    import json
    validate_topology_identity(spec,json.loads(metadata.get("config") or "{}"))
    for side in initialization_resolutions(options):
        preview_plan(spec,side//16,side//16)
    return source,components,spec


def execute(components, source, spec, items, options, progress, cancelled):
    if spec["architecture"] == "ming_image_01":
        from core.rewiring.ming_initialization import initialize_loaded_ming
        return initialize_loaded_ming(components,source,spec,items,options,progress,cancelled)
    if spec["architecture"] == "qwen_image_21":
        from core.rewiring.qwen_initialization import initialize_loaded_qwen
        return initialize_loaded_qwen(components,source,spec,items,options,progress,cancelled)
    from core.rewiring.anima_initialization import initialize_loaded_anima
    return initialize_loaded_anima(components,source,spec,items,options,progress,cancelled)


def _checked_ming_source(manager,options):
    import torch
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.models.ming_image_01.residual_unet import validate_topology_identity
    from core.rewiring.inspect import inspect_model
    from core.rewiring.initialized_export import validate_package_name
    from core.rewiring.spec import preview_plan
    source = Path(options["model_path"]).resolve(strict=True)
    if source != Path(manager.current_model_info.get("source") or "").resolve():
        raise ValueError("Initialization source must match the loaded Ming package")
    artifact = inspect_artifact(source)
    validate_package_name(options["output_name"])
    output = artifact.root.parent/options["output_name"]
    if output.exists() or output.is_symlink():
        raise FileExistsError("Ming initialization output already exists")
    directory = artifact.components["transformer"]
    from core.rewiring.ming_source import transformer_files
    transformer_files(directory)
    components = manager.ming_image_01_components
    native = (components or {}).get("ming_native")
    if native is None or any(getattr(native,key,None) is None for key in
                             ("transformer","conditioner","vae","processor","scheduler")):
        raise ValueError("Loaded Ming components are incomplete")
    model = native.transformer
    inspected = inspect_model(source)
    if (inspected["topology"] is not None or getattr(model,"_residual_unet_spec",None) is not None
            or hasattr(model,"rewiring_modules") or hasattr(model,"vae_latent_bridge")
            or artifact.configs["transformer"].get("vae_alignment") is not None
            or artifact.global_summary is not None or getattr(model,"_ming_partition_global_payload",None) is not None):
        raise ValueError("Ming initialization requires a native source without U-Net, VAE swap or global adapter")
    if getattr(model,"_block_offloader",None) is not None or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Ming initialization requires resident blocks without existing gradients")
    headers = artifact.tensor_headers["transformer"]
    if any(value.dtype not in ("F32","F16","BF16") for name,value in headers.items() if name.endswith(".weight")):
        raise ValueError("Ming initialization requires floating native DiT weights; packed conditioner weights are supported")
    if any(name not in headers or tuple(value.shape) != headers[name].shape for name,value in model.named_parameters()):
        raise ValueError("Loaded Ming parameter layout differs from source; remove adapters and reload")
    if any(p.dtype not in (torch.float32,torch.float16,torch.bfloat16) for p in model.parameters()):
        raise ValueError("Ming initialization requires floating native DiT weights")
    spec = validate_topology_identity(options["spec"] or inspected["hybrid_default_spec"],model)
    if spec["method"] != "hybrid_unet_ssdit":
        raise ValueError("Ming initialization requires a hybrid declaration")
    validate_topology_identity(spec,artifact.configs["transformer"])
    for side in initialization_resolutions(options):
        preview_plan(spec,side//16,side//16)
    return source,components,spec


def start(manager, options, items):
    global _job,_thread
    from api.routes import _fp8_scaled_mm_busy_reason
    if job_is_running():
        raise ModelStateBusyError("Hybrid initialization is already running")
    busy = _fp8_scaled_mm_busy_reason()
    if busy:
        raise ModelStateBusyError(busy)
    gate = model_state_coordinator.mutation("hybrid U-Net initialization")
    gate.__enter__()
    try:
        source,components,spec = _checked_source(manager,options)
        with _lock:
            if _thread is not None and _thread.is_alive():
                raise ModelStateBusyError("Hybrid initialization is already running")
            _cancel.clear()
            job_id = uuid.uuid4().hex
            _job = dict(job_id=job_id,state="queued",model_path=str(source),output_name=options["output_name"],
                        stage="queued",completed=0,total=0,units="items",error=None,result=None,
                        started_at=time.time(),finished_at=None)
            snapshot = deepcopy(_job)
            _thread = threading.Thread(target=_run,args=(gate,manager,job_id,source,components,spec,
                                                        deepcopy(items),deepcopy(options)),
                                       name=f"hybrid-init-{job_id}",daemon=True)
            try:
                _thread.start()
            except BaseException as exc:
                _thread = None
                _job.update(state="failed", error=f"{type(exc).__name__}: {exc}", finished_at=time.time())
                raise
        return snapshot
    except BaseException:
        gate.__exit__(None,None,None)
        raise


def _run(gate,manager,job_id,source,components,spec,items,options):
    from core.rewiring.fitting import InitializationCancelled
    result,error,terminal = None,None,"failed"

    def progress(event):
        with _lock:
            if _job is not None and _job["job_id"] == job_id:
                _job.update(event)
                if not _cancel.is_set():
                    _job["state"] = "running"

    try:
        with manager._load_model_lock:
            _checked_source(manager,options)
            if _cancel.is_set():
                raise InitializationCancelled("Hybrid initialization cancelled before preparation")
            result = execute(components,source,spec,items,options,progress,_cancel.is_set)
        terminal = "completed"
        from api import routes
        routes._models_cache = None
    except InitializationCancelled:
        terminal = "cancelled"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        gate.__exit__(None,None,None)
        with _lock:
            if _job is not None and _job["job_id"] == job_id:
                _job.update(state=terminal,result=result,error=error,finished_at=time.time())
                if terminal == "completed":
                    _job.update(stage="completed",completed=1,total=1,units="models")
