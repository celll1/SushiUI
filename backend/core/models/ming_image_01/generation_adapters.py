"""Request-scoped native DiT adapters with restored base topology."""
from contextlib import contextmanager
from copy import deepcopy
from core.adapters import AdapterComponent, AdapterSession, AdapterIncompatible, PreparedBranch, SHAPE_MISMATCH
from core.rewiring.adapters import resolve_adapter_topology
from core.adapters.base_identity import BaseLatentIdentity
from .lora import iter_lora_slots, build_lora_branch, normalise_lora_state_dict, declared_branch_count
from .residual_unet import restore_topology, validate_topology_identity


@contextmanager
def generation_adapters(components, configs, *, resolve_path, warn=None):
    model = components.transformer
    base = deepcopy(getattr(model, "_residual_unet_spec", None))

    def prepare(file):
        if file.metadata.get("model_type") != "ming_image_01":
            raise AdapterIncompatible("Ming adapters require model_type=ming_image_01 metadata", code="ming_adapter_architecture_mismatch")
        if file.metadata.get("ming_image_01_profile") != components.artifact.profile:
            raise AdapterIncompatible("Ming adapter profile does not match the loaded checkpoint", code="ming_adapter_profile_mismatch")
        return normalise_lora_state_dict(file.tensors)

    def build(request):
        group = request.prepared.get(request.module_path)
        if group is None:
            return None
        branch = build_lora_branch(request.base, group, request.module_path)
        return branch if branch is SHAPE_MISMATCH else PreparedBranch(branch, request.file.strength)

    session = AdapterSession(resolve_path=resolve_path, warn=warn, architecture="ming_image_01",
        label="Ming LoRA", prepare_file=prepare,
        base_latent=lambda: BaseLatentIdentity(latent_channels=16),
        count_declared_branches=lambda tensors, _: declared_branch_count(tensors))
    slots = [AdapterComponent(name="transformer", module=model, iter_targets=iter_lora_slots, build_branch=build)]
    files = session.parse(configs)
    required = resolve_adapter_topology(model, files, validate_topology_identity)
    try:
        session.load(files, slots)
        restore_topology(model, required)
        yield session
    finally:
        try:
            session.unload(slots)
        finally:
            restore_topology(model, base)
