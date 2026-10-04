"""Native model-only transformations preserve vocabulary and inference sidecars."""
from pathlib import Path
import json
import shutil

from ..siglip2_tagger_model import _save_file_safe
from .trainer_model import _write_metadata
from .inference_bundle import publish_inference_bundle


def merge_lora(manager,output_path):
    if manager.model_type != "lora" or not hasattr(manager.model,"export_state"):
        raise ValueError("load a native LoRA PyTorch model before merging")
    source=Path(manager.checkpoint_path)
    root=Path(output_path.strip().strip('"').strip("'")) if output_path.strip() else source.parent/"merged"
    root.mkdir(parents=True,exist_ok=True)
    path=root/"model.safetensors"
    _save_file_safe(manager.model.export_state(merged=True),str(path))
    metadata=json.loads(source.with_name(f"{source.stem}_metadata.json").read_text(encoding="utf-8"))
    metadata.pop("inference_export",None)
    metadata.update(kind="foveatag_shared_trainer_weights",training_method="full",contains_frozen_base=True,
        resume_weight_mode="full_unmerged",vocabulary_file="model_vocabulary.json",step=metadata.get("step",0),
        merged_from_native_lora=True)
    _write_metadata(root/"model_metadata.json",metadata)
    vocab={**manager.vocabulary,"idx_to_tag":{str(key):value for key,value in manager.vocabulary["idx_to_tag"].items()}}
    _write_metadata(root/"model_vocabulary.json",vocab)
    optional=[]
    for suffix in ("_tag_metrics.npz","_ood_ref.npz"):
        original=Path(manager.native_sidecar_base+suffix)
        if original.is_file():
            shutil.copyfile(original,root/("model"+suffix))
            optional.append(suffix)
    publish_inference_bundle(root,"model",optional_suffixes=optional)
    return str(path)


def extract_encoder(source_path,output_path):
    from copy import deepcopy
    from dataclasses import asdict
    from ..siglip2_tagger_model import LoRALinear
    from .trainer_model import load_native_checkpoint
    from .pretraining_checkpoint import save_pretrained_encoder
    from .shared_bundle import _sha256
    source=Path(source_path)
    model,metadata=load_native_checkpoint(source)
    encoder=deepcopy(model.vision_encoder).cpu()
    def merge(module):
        for name,child in list(module.named_children()):
            if isinstance(child,LoRALinear):
                child.merge_into_base()
                setattr(module,name,child.base)
            else:
                merge(child)
    merge(encoder)
    save_pretrained_encoder(Path(output_path),encoder,run_contract={"encoder":asdict(encoder.config),
        "source_stage":"tagger_encoder_extract", "source_sha256":_sha256(source), "external_weights":[]},
        epoch=metadata.get("epoch",0),global_step=metadata.get("step",metadata.get("provenance",{}).get("global_step",0)),validation={})
    return {"output_path":str(Path(output_path).resolve()),"num_params":sum(p.numel() for p in encoder.parameters()),
        "hidden_size":encoder.config.widths[-1],"num_layers":sum(encoder.config.depths)}
