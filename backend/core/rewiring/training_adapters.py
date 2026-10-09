"""Complete hybrid module ownership in low-rank training artifacts."""
from core.adapters.full_module import FullModulePayload
from .adapters import FULL_MODULE_METADATA, FULL_MODULE_PREFIX
from .spec import checkpoint_metadata, read_checkpoint_metadata
from .provenance import model_metadata, read_checkpoint_metadata as read_initialization


def register_hybrid_payload(adapter, layers):
    model = getattr(adapter.trainer, "transformer", None)
    spec = getattr(model, "_residual_unet_spec", None)
    if spec is None or spec["method"] != "hybrid_unet_ssdit":
        return
    modules = getattr(model, "rewiring_modules", None)
    if modules is None or modules.declaration != spec:
        raise ValueError("Hybrid adapter training requires the resolved structural modules")
    if "rewiring_modules" in layers:
        raise ValueError("Hybrid structural payload was registered twice")
    from core.training.adapters.base_adapter import LORA_COMPONENT_UNET
    modules.requires_grad_(bool(getattr(adapter.trainer, "train_unet", True))).train()
    adapter.register_lora_layer(layers, "rewiring_modules", FullModulePayload(modules), LORA_COMPONENT_UNET)


def hybrid_payload_metadata(trainer, layers):
    model = getattr(trainer, "transformer", None)
    spec = getattr(model, "_residual_unet_spec", None)
    payload = layers.get("rewiring_modules")
    if spec is None or spec["method"] != "hybrid_unet_ssdit":
        if isinstance(payload, FullModulePayload):
            raise ValueError("Structural payload has no hybrid topology")
        return {}
    if not isinstance(payload, FullModulePayload) or payload.module is not getattr(model, "rewiring_modules", None):
        raise ValueError("Hybrid adapter save would omit its structural weights")
    if payload.module.declaration != spec:
        raise ValueError("Hybrid structural module declaration changed")
    return {FULL_MODULE_METADATA: "1", **checkpoint_metadata(spec), **model_metadata(model)}


def validate_hybrid_payload_resume(trainer, layers, metadata, tensors):
    payload = layers.get("rewiring_modules")
    names = {key.removeprefix(FULL_MODULE_PREFIX) for key in tensors if key.startswith(FULL_MODULE_PREFIX)}
    declaration = metadata.get(FULL_MODULE_METADATA)
    if not isinstance(payload, FullModulePayload):
        if names or declaration is not None:
            raise ValueError("Resume structural payload is not registered by this run")
        return None
    spec = getattr(trainer.transformer, "_residual_unet_spec", None)
    if declaration != "1" or read_checkpoint_metadata(metadata) != spec:
        raise ValueError("Resume hybrid structural declaration does not match this run")
    if names != set(payload.branch_tensors()):
        raise ValueError("Resume requires exactly the full structural tensor scope")
    return read_initialization(metadata, spec)


def prepare_hybrid_resume(trainer, path):
    from importlib import import_module
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        owns_payload = any(key.startswith(FULL_MODULE_PREFIX) for key in handle.keys())
    if metadata.get(FULL_MODULE_METADATA) is None and not owns_payload:
        return
    if metadata.get(FULL_MODULE_METADATA) != "1" or not owns_payload:
        raise ValueError("Resume hybrid payload declaration is incomplete")
    spec = read_checkpoint_metadata(metadata)
    if spec is None or spec["method"] != "hybrid_unet_ssdit":
        raise ValueError("Resume full-module payload requires hybrid topology")
    architecture = getattr(getattr(trainer, "arch", None), "name", None)
    if architecture != spec["architecture"]:
        raise ValueError("Resume structural payload architecture mismatch")
    owner = import_module(f"core.models.{architecture}.residual_unet")
    owner.configure_training_topology(trainer.transformer, trainer.config,
                                     resuming=True, resume_spec=spec)
