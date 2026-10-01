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
    missing=object();previous_global=getattr(model,"_ming_partition_global_generation",missing)
    global_entries=[]

    def prepare(file):
        if file.metadata.get("model_type") != "ming_image_01":
            raise AdapterIncompatible("Ming adapters require model_type=ming_image_01 metadata", code="ming_adapter_architecture_mismatch")
        if file.metadata.get("ming_image_01_profile") != components.artifact.profile:
            raise AdapterIncompatible("Ming adapter profile does not match the loaded checkpoint", code="ming_adapter_profile_mismatch")
        from .convrot_training import validate_generation_metadata
        try:validate_generation_metadata(model,file.metadata)
        except ValueError as exc:
            raise AdapterIncompatible(str(exc),code="ming_base_forward_mismatch") from exc
        from .partition_global import GLOBAL_NAME,lora_generation_branch
        try:branch=lora_generation_branch(model,file.tensors,file.metadata)
        except (ValueError,TypeError,KeyError) as exc:
            raise AdapterIncompatible(str(exc),code="ming_global_adapter_invalid") from exc
        if branch is not None:
            global_entries.append({"module":branch,"strength":file.strength if file.apply_to_unet else 0.,"step_range":file.step_range,"active":True})
        ordinary={key:value for key,value in file.tensors.items() if not key.startswith(GLOBAL_NAME+".")}
        return normalise_lora_state_dict(ordinary)

    def build(request):
        group = request.prepared.get(request.module_path)
        if group is None:
            return None
        branch = build_lora_branch(request.base, group, request.module_path)
        return branch if branch is SHAPE_MISMATCH else PreparedBranch(branch, request.file.strength)

    class MingAdapterSession(AdapterSession):
        @property
        def has_step_range(self):
            return super().has_step_range or any(entry["step_range"] is not None and tuple(entry["step_range"])!=(0,1000) for entry in global_entries)

        def set_step(self,index,total):
            super().set_step(index,total)
            if total<=0:return
            for entry in global_entries:
                window=entry["step_range"]
                entry["active"]=window is None or int(window[0]/1000*total)<=index<=int(window[1]/1000*total)

    session = MingAdapterSession(resolve_path=resolve_path, warn=warn, architecture="ming_image_01",
        label="Ming LoRA", prepare_file=prepare,
        base_latent=lambda: BaseLatentIdentity(latent_channels=16),
        count_declared_branches=lambda tensors, _: declared_branch_count(tensors))
    slots = [AdapterComponent(name="transformer", module=model, iter_targets=iter_lora_slots, build_branch=build)]
    files = session.parse(configs)
    required = resolve_adapter_topology(model, files, validate_topology_identity)
    try:
        session.load(files, slots)
        restore_topology(model, required)
        model._ming_partition_global_generation=([] if previous_global is missing else list(previous_global))+global_entries
        yield session
    finally:
        try:
            session.unload(slots)
        finally:
            try:
                restore_topology(model, base)
            finally:
                if previous_global is missing:
                    if hasattr(model,"_ming_partition_global_generation"):del model._ming_partition_global_generation
                else:model._ming_partition_global_generation=previous_global
