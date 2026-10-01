"""Atomic offline native Ming DiT quantization without loading companion models."""
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
                    cancelled=None, progress_callback=None):
    import torch
    from safetensors import safe_open
    from core.models.common.quantized_export import ShardWriter, DEFAULT_EXPORT_SHARD_BYTES
    from core.models.common.int8_runtime_quantize import (
        arch_policy, linear_paths, select_targets, audit_and_quantize_int8, DEFAULT_CREST_THRESHOLD)
    from core.rewiring.artifacts import read_directory_topology
    from core.rewiring.spec import checkpoint_metadata
    from .model_definition import create_transformer
    from .residual_unet import validate_topology_identity

    if format_ not in {"int8", "int8_convrot"}:
        raise ValueError("Ming offline format must be int8 or int8_convrot")
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
        if component in {"mllm","connector","mlp"}:
            continue
        if any(h.dtype not in {"F16", "BF16", "F32"} or key.endswith((".weight_scale", ".comfy_quant"))
               for key,h in headers.items()):
            raise ValueError(f"Ming conversion requires original floating weights: {component}")
    from .partition_global import payload_from_directory, write_payload
    global_payload = payload_from_directory(artifact.root,
        latent_channels=artifact.configs["transformer"]["in_channels"],
        hidden_dim=artifact.configs["transformer"]["dim"])
    from .conditioner_quantization import prepare_conditioner_components
    prepare_conditioner_components(artifact,torch.float32,"eager")
    topology = read_directory_topology(artifact.components["transformer"])
    if topology is not None:
        topology = validate_topology_identity(topology,artifact.configs["transformer"])
    model = create_transformer(artifact.configs["transformer"],device="meta",dtype=torch.float32)
    expected = model.state_dict()
    headers = artifact.tensor_headers["transformer"]
    if set(headers) != set(expected) or any(headers[key].shape != tuple(value.shape) for key,value in expected.items()):
        raise ValueError("Ming source DiT state differs from its native config")
    policy = arch_policy("ming_image_01")
    targets, skipped = select_targets(linear_paths(model),set(headers),int(policy["min_align"]),
        [re.compile(pattern) for pattern in policy["excludes"]],bool(policy["skip_below_work_gate"]))
    if format_ == "int8_convrot":
        eligible = []
        for name in targets:
            if headers[name+".weight"].shape[1] % 256:
                skipped.append((name,"input width is not divisible by ConvRot group 256"))
            else:
                eligible.append(name)
        targets = eligible
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime
        from core.models.common.convrot_export import rotate_and_quantize,convrot_marker_fields
        from core.models.common.quantized_checkpoint_guard import encode_comfy_quant_marker
        require_convrot_int8_runtime()
        marker = encode_comfy_quant_marker(convrot_marker_fields())
    if not targets:
        raise ValueError("Ming source has no eligible native DiT Linear weights")
    selected = set(targets)
    del expected, model
    total = len(headers)
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
        for name,directory in artifact.components.items():
            check_cancel()
            if name != "transformer":
                shutil.copytree(directory,staging/name)
        transformer_dir = staging/"transformer"
        transformer_dir.mkdir()
        shutil.copy2(artifact.components["transformer"]/"config.json",transformer_dir/"config.json")
        writer = ShardWriter(str(transformer_dir/"diffusion_pytorch_model.safetensors"),metadata,max_shard_bytes)
        processed = 0
        try:
            for path in _component_files(artifact.components["transformer"]):
                with safe_open(str(path),framework="pt",device="cpu") as handle:
                    for key in handle.keys():
                        check_cancel()
                        tensor = handle.get_tensor(key)
                        stem = key[:-len(".weight")] if key.endswith(".weight") else ""
                        if stem in selected:
                            if not torch.isfinite(tensor).all():
                                raise ValueError(f"Ming quantization source contains nonfinite weights: {stem}")
                            if format_ == "int8_convrot":
                                q,scale = rotate_and_quantize(tensor.to(device))
                                chosen = "convrot_int8"
                                rows.append({"name":stem,"format":chosen,"shape":list(tensor.shape)})
                                writer.add(stem+".comfy_quant",marker.clone())
                            else:
                                chosen,q,scale,row = audit_and_quantize_int8(stem,tensor.to(device),DEFAULT_CREST_THRESHOLD,"e4m3")
                                undefined = [key for key,value in row.items() if isinstance(value,float) and not math.isfinite(value)]
                                rows.append({**{key:None if key in undefined else value for key,value in row.items()},
                                             "undefined_metrics":undefined})
                            if scale is None or not torch.isfinite(scale).all() or (scale <= 0).any():
                                raise ValueError(f"Ming conversion produced invalid scales: {stem}")
                            writer.add(key,q.cpu().contiguous())
                            writer.add(stem+".weight_scale",scale.cpu().reshape(-1).contiguous())
                            counts[chosen]+=1
                        else:
                            writer.add(key,tensor.contiguous())
                        processed+=1
                        if progress_callback is not None:
                            progress_callback(processed,total,key)
            writer.close()
        except BaseException:
            writer.abort()
            raise
        if processed != total or sum(counts.values()) != len(targets):
            raise ValueError("Ming source changed during conversion")
        report = {"schema_version":1,"kind":"offline_weight_only_dit","format":format_,
                  "device":str(device),"source_config_fingerprint":artifact.config_fingerprint,
                  "converted":counts,"skipped":[{"name":name,"reason":reason} for name,reason in skipped],
                  "layer_audit":rows,"runtime_audit":None,
                  "audit_note":"offline conversion; no session runtime audit"}
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
        from .quantization import prepare_quantized_transformer
        prepare_quantized_transformer(exported,torch.float32)
        check_cancel()
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(staging,destination)
    return {"manifest":str(destination/"manifest.json"),"report":report}
