"""Compatibility checks for metadata-driven adapter topology scopes."""
from .spec import read_checkpoint_metadata

FULL_MODULE_METADATA = "sushi.rewiring.full_module"
FULL_MODULE_PREFIX = "rewiring_modules."


class HybridPayloadTransaction:
    """Prepare replacement structural weights without changing the live DiT."""
    def __init__(self, model, files, required, restore_topology):
        import weakref
        from copy import deepcopy
        from core.adapters.full_module import FullModulePayload
        from .hybrid_modules import HybridModules
        from .provenance import read_checkpoint_metadata as read_initialization
        self.model = weakref.ref(model)
        self.restore_topology = restore_topology
        self.base = deepcopy(getattr(model, "_residual_unet_spec", None))
        self.original = getattr(model, "rewiring_modules", None)
        self.record = deepcopy(getattr(model, "_hybrid_initialization", None))
        self.required = required
        self.prepared = None
        self.prepared_record = None
        self.installed = False
        owners = []
        for file in files:
            tensors = {key.removeprefix(FULL_MODULE_PREFIX): value
                       for key, value in file.tensors.items() if key.startswith(FULL_MODULE_PREFIX)}
            declaration = file.metadata.get(FULL_MODULE_METADATA)
            if declaration is None and not tensors:
                continue
            if declaration != "1" or not tensors:
                raise ValueError("Incomplete hybrid full-module adapter payload")
            if required is None or required["method"] != "hybrid_unet_ssdit":
                raise ValueError("Full-module adapter requires a hybrid topology")
            if (not file.apply_to_unet or file.strength != 1
                    or file.step_range is not None and tuple(file.step_range) != (0, 1000)):
                raise ValueError("Hybrid structural payload requires strength one, U-Net application and no step modulation")
            owners.append((file, tensors))
        if len(owners) > 1:
            raise ValueError("Only one hybrid structural payload owner can be selected")
        if owners:
            file, tensors = owners[0]
            reference = next(model.parameters())
            self.prepared_record = read_initialization(file.metadata, required)
            import torch
            with torch.device("cpu"):
                modules = HybridModules(required).to(dtype=reference.dtype)
            FullModulePayload(modules).load_tensors(tensors)
            modules.to(reference.device).requires_grad_(False).eval()
            self.prepared = modules
            file.tensors = {key: value for key, value in file.tensors.items()
                            if not key.startswith(FULL_MODULE_PREFIX)}
        elif required is not None and required["method"] == "hybrid_unet_ssdit" and self.original is None:
            raise ValueError("Hybrid adapter is missing its structural weights on a native base")

    def install(self):
        from .provenance import restore_record
        model = self.model()
        if model is None:
            raise ValueError("Adapter base was released before installation")
        self.installed = True
        try:
            if self.prepared is not None:
                model.rewiring_modules = self.prepared
            self.restore_topology(model, self.required)
            if self.prepared is not None:
                restore_record(model, self.prepared_record)
        except Exception:
            self.restore()
            raise

    def restore(self):
        from .provenance import restore_record
        if not self.installed:
            return
        model = self.model()
        if model is not None:
            if self.original is not None:
                model.rewiring_modules = self.original
            elif hasattr(model, "rewiring_modules"):
                del model.rewiring_modules
            self.restore_topology(model, self.base)
            restore_record(model, self.record)
        self.installed = False
        self.prepared = None
        self.original = None


def resolve_adapter_topology(model, files, validate_identity):
    base = getattr(model, "_residual_unet_spec", None)
    if base is not None:
        base = validate_identity(base, dict(model.config))
    declarations = []
    for file in files:
        topology = read_checkpoint_metadata(file.metadata)
        if topology is not None:
            topology = validate_identity(topology, dict(model.config))
        declarations.append(topology)
    required = declarations[0] if declarations else base
    if any(value != required for value in declarations):
        raise ValueError("Selected LoRAs require different residual topologies")
    if base is not None and required != base:
        raise ValueError("LoRA topology does not match the loaded checkpoint")
    return required
