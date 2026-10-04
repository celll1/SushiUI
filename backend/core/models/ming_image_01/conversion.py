"""Atomic offline native Ming component quantization without loading model weights."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .artifact import inspect_artifact, write_manifest
from .quantization import _component_files


def convert_package(source, destination, *, format_, device="cpu", max_shard_bytes=None,
                    cancelled=None, progress_callback=None, components=None):
    import torch
    from safetensors import safe_open
    from core.models.common.quantized_export import ShardWriter, DEFAULT_EXPORT_SHARD_BYTES
    from core.models.common.int8_runtime_quantize import (
        arch_policy, linear_paths, select_targets, audit_and_quantize_int8, DEFAULT_CREST_THRESHOLD)
    from core.rewiring.artifacts import read_directory_topology
    from core.rewiring.spec import checkpoint_metadata
    from .model_definition import create_transformer
    from .conditioner_quantization import create_conditioner_meta, prepare_conditioner_components
    from .quantization import prepare_component_slots, prepare_quantized_transformer
    from .residual_unet import validate_topology_identity

    if format_ not in {"int8", "int8_convrot"}:
        raise ValueError("Ming offline format must be int8 or int8_convrot")
    selected_components = ("transformer",) if components is None else components
    supported = {"transformer", "mllm", "connector", "mlp"}
    if (not isinstance(selected_components,(list,tuple)) or not selected_components
            or any(not isinstance(name,str) or name not in supported for name in selected_components)
            or len(selected_components) != len(set(selected_components))):
        raise ValueError("Ming conversion components must be unique supported component names")
    selected_components = tuple(selected_components)
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Ming offline conversion supports CPU or explicitly selected CUDA")
    max_shard_bytes = DEFAULT_EXPORT_SHARD_BYTES if max_shard_bytes is None else max_shard_bytes
    if type(max_shard_bytes) is not int or max_shard_bytes <= 0:
        raise ValueError("Ming output shard limit must be a positive integer")
    if not str(destination).strip():
        raise ValueError("Ming destination must be nonempty")
    artifact = inspect_artifact(source)
    destination = Path(destination).resolve()
    if destination.is_relative_to(artifact.root.resolve()):
        raise ValueError("Ming conversion destination must be outside the source package")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    for component, headers in artifact.tensor_headers.items():
        if component not in selected_components and component in supported:
            continue
        if any(h.dtype not in {"F16", "BF16", "F32"} or key.endswith((".weight_scale", ".comfy_quant"))
               for key,h in headers.items()):
            raise ValueError(f"Ming conversion requires original floating weights: {component}")
    from .partition_global import payload_from_directory, write_payload
    global_payload = payload_from_directory(artifact.root,
        latent_channels=artifact.configs["transformer"]["in_channels"],
        hidden_dim=artifact.configs["transformer"]["dim"])
    prepare_conditioner_components(artifact,torch.float32,"eager")
    prepare_quantized_transformer(artifact,torch.float32)
    topology = read_directory_topology(artifact.components["transformer"])
    if topology is not None:
        topology = validate_topology_identity(topology,artifact.configs["transformer"])
    if format_ == "int8_convrot":
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime
        from core.models.common.convrot_export import rotate_and_quantize,convrot_marker_fields
        from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
        require_convrot_int8_runtime()
        marker = encode_comfy_quant_marker(convrot_marker_fields())
    plans = {}
    for component in selected_components:
        headers = artifact.tensor_headers[component]
        model = (create_transformer(artifact.configs[component],device="meta",dtype=torch.float32)
                 if component == "transformer" else create_conditioner_meta(artifact,component,torch.float32))
        prepare_component_slots(model,headers,{},torch.float32)
        policy = arch_policy("ming_image_01") if component == "transformer" else {
            "min_align":8,"excludes":(r"(^|\.)(lm_head|embed_tokens|query_tokens_dict)(\.|$)",),
            "skip_below_work_gate":False}
        targets, skipped = select_targets(linear_paths(model),set(headers),int(policy["min_align"]),
            [re.compile(pattern) for pattern in policy["excludes"]],bool(policy["skip_below_work_gate"]))
        tied = {key.removesuffix(".weight") for group in model._ming_quantized_alias_groups for key in group}
        eligible = []
        for name in targets:
            if name in tied:
                skipped.append((name,"shared embedding/output weight remains floating"))
            elif format_ == "int8_convrot" and headers[name+".weight"].shape[1] % 256:
                skipped.append((name,"input width is not divisible by ConvRot group 256"))
            else:
                eligible.append(name)
        if not eligible:
            raise ValueError(f"Ming source has no eligible native Linear weights: {component}")
        plans[component] = {"selected":set(eligible),"skipped":skipped,"rows":[],
                            "counts":{"int8":0,"e4m3":0,"convrot_int8":0}}
        del model
    total = sum(len(artifact.tensor_headers[name]) for name in selected_components)
    rows = []
    counts = {"int8":0,"e4m3":0,"convrot_int8":0}
    metadata = {} if topology is None else checkpoint_metadata(topology)
    metadata.update(model_type="ming_image_01",ming_image_01_profile=artifact.profile,quant_format=format_)

    def check_cancel():
        if cancelled is not None and cancelled():
            raise InterruptedError("Ming offline conversion cancelled")
    check_cancel()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.ming-",dir=destination.parent) as temporary:
        staging = Path(temporary)/"package"
        staging.mkdir()
        for pattern in ("LICENSE*", "NOTICE*"):
            for notice in artifact.root.glob(pattern):
                if notice.is_file():
                    shutil.copy2(notice, staging / notice.name)
        processed = 0
        for component,directory in artifact.components.items():
            check_cancel()
            output_dir = staging/component
            if component not in plans:
                shutil.copytree(directory,output_dir)
                continue
            shutil.copytree(directory,output_dir,ignore=lambda _directory,names:
                [name for name in names if name.endswith((".safetensors",".safetensors.index.json"))])
            plan = plans[component]
            basename = "diffusion_pytorch_model.safetensors" if component == "transformer" else "model.safetensors"
            writer = ShardWriter(str(output_dir/basename),metadata,max_shard_bytes)
            try:
                for path in _component_files(directory):
                    with safe_open(str(path),framework="pt",device="cpu") as handle:
                        for key in handle.keys():
                            check_cancel()
                            tensor = handle.get_tensor(key)
                            stem = key[:-len(".weight")] if key.endswith(".weight") else ""
                            audit_name = stem if selected_components == ("transformer",) else f"{component}.{stem}"
                            if stem in plan["selected"]:
                                if not torch.isfinite(tensor).all():
                                    raise ValueError(f"Ming quantization source contains nonfinite weights: {component}.{stem}")
                                if format_ == "int8_convrot":
                                    q,scale = rotate_and_quantize(tensor.to(device))
                                    chosen = "convrot_int8"
                                    row = {"name":audit_name,"format":chosen,"shape":list(tensor.shape)}
                                    writer.add(stem+".comfy_quant",marker.clone())
                                else:
                                    chosen,q,scale,row = audit_and_quantize_int8(audit_name,tensor.to(device),DEFAULT_CREST_THRESHOLD,"e4m3")
                                    undefined = [name for name,value in row.items() if isinstance(value,float) and not math.isfinite(value)]
                                    row = {**{name:None if name in undefined else value for name,value in row.items()},"undefined_metrics":undefined}
                                rows.append(row);plan["rows"].append(row)
                                if scale is None or not torch.isfinite(scale).all() or (scale <= 0).any():
                                    raise ValueError(f"Ming conversion produced invalid scales: {component}.{stem}")
                                writer.add(key,q.cpu().contiguous())
                                writer.add(stem+".weight_scale",scale.cpu().reshape(-1).contiguous())
                                counts[chosen]+=1;plan["counts"][chosen]+=1
                            else:
                                writer.add(key,tensor.contiguous())
                            processed+=1
                            if progress_callback is not None:
                                progress_callback(processed,total,key if selected_components == ("transformer",) else f"{component}.{key}")
                writer.close()
            except BaseException:
                writer.abort()
                raise
            if sum(plan["counts"].values()) != len(plan["selected"]):
                raise ValueError(f"Ming source changed during conversion: {component}")
        transformer_dir = staging/"transformer"
        if processed != total:
            raise ValueError("Ming source changed during conversion")
        skipped = [(name if selected_components == ("transformer",) else f"{component}.{name}",reason)
                   for component,plan in plans.items() for name,reason in plan["skipped"]]
        report = {"schema_version":1,"kind":"offline_weight_only_dit" if selected_components == ("transformer",) else "offline_weight_only_components","format":format_,
                  "device":str(device),"source_config_fingerprint":artifact.config_fingerprint,
                  "converted":counts,"skipped":[{"name":name,"reason":reason} for name,reason in skipped],
                  "layer_audit":rows,"runtime_audit":None,
                  "audit_note":"offline conversion; no session runtime audit"}
        report["components"] = {name:{"converted":plan["counts"],"selected_layers":len(plan["selected"])} for name,plan in plans.items()}
        (staging/"quantization.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
        if (artifact.root/"training.json").is_file():
            shutil.copy2(artifact.root/"training.json",staging/"training.json")
        if global_payload is not None:
            from types import SimpleNamespace
            write_payload(SimpleNamespace(_ming_partition_global_payload=global_payload),staging)
        write_manifest(staging,source=artifact.source)
        exported = inspect_artifact(staging)
        if exported.config_fingerprint != artifact.config_fingerprint or read_directory_topology(transformer_dir) != topology:
            raise ValueError("Ming conversion changed native config or U-Net topology")
        prepare_quantized_transformer(exported,torch.float32)
        prepare_conditioner_components(exported,torch.float32,"eager")
        check_cancel()
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(staging,destination)
    return {"manifest":str(destination/"manifest.json"),"report":report}
