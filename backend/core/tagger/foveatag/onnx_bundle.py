"""Fixed-budget ONNX stages with source cropping and routing selection on the host."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil

import numpy as np
import torch
from torch import nn

from .encoder import FoveaEncoderOutput
from .batch import collate_view_batches
from .views import build_views, ViewBatch, sample_local_specs, fit_local_spec, render_source_view
from .shared_bundle import _sha256

FEATURES = tuple(field.name for field in fields(FoveaEncoderOutput))


class OverviewGraph(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.encoder = core.encoder

    def forward(self, images, geometry):
        output = self.encoder(images, geometry)
        return tuple(getattr(output, name) for name in FEATURES)


class RouterGraph(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.router = router

    def forward(self, local_map, coordinates, boxes, mask):
        return self.router(local_map, coordinates, boxes, mask).scores


class ClassifierGraph(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, local_map, local_coordinates, global_tokens, global_coordinates, global_summary,
                local_images, local_geometry, local_mask):
        overview = FoveaEncoderOutput(local_map, local_coordinates, global_tokens, global_coordinates, global_summary)
        details = self.core.classify_encoded_overview(overview, local_images, local_geometry,
            local_view_mask=local_mask, return_details=True)
        valid = local_mask.repeat_interleave(self.core.evidence_slots_per_view, dim=1)
        evidence = details.head_output.mixed_evidence
        mean = (evidence * valid[...,None]).sum(1) / valid.sum(1).clamp_min(1)[:,None]
        return details.logits, torch.cat((details.global_features,mean),dim=1)


def export_bundle(manager, output_path, *, strip_unknown_tags=False, use_model_stem=False,
                  also_split=False, split_max_bytes=None):
    from PIL import Image
    from ..siglip2_tagger_model import LoRALinear
    from ..siglip2_inference_manager import _read_metadata, _filter_tag_metrics_npz, _split_onnx_for_webgpu
    source = Path(manager.checkpoint_path)
    target = Path(output_path.strip().strip('"').strip("'")) if output_path.strip() else source.parent / "onnx" / f"{source.stem}.onnx"
    if not target.is_absolute():
        target = source.parent / target
    if target.suffix.lower() != ".onnx":
        target = target / f"{source.stem}.onnx"
    if use_model_stem:
        target = target.with_name("model.onnx")
    target.parent.mkdir(parents=True,exist_ok=True)
    if manager.model is None or getattr(manager.model,"is_native_onnx",False):
        raise ValueError("load native PyTorch weights before ONNX export")
    model = deepcopy(manager.model).eval().cpu()
    model._ood_tap._forward_pre_hooks.clear()
    def merge(module):
        for name, child in list(module.named_children()):
            if isinstance(child,LoRALinear):
                child.merge_into_base()
                setattr(module,name,child.base)
            else:
                merge(child)
    merge(model.core)
    model.core.encoder.set_sdpa(False,allow_flash=False,external_flash=False)
    vocab = {"tag_to_idx":manager.vocabulary["tag_to_idx"],
        "idx_to_tag":{str(key):value for key,value in manager.vocabulary["idx_to_tag"].items()},
        "tag_to_category":manager.vocabulary["tag_to_category"]}
    old_count = model.head.out_features
    keep = list(range(old_count))
    if strip_unknown_tags:
        keep = [index for index in keep if vocab["tag_to_category"].get(vocab["idx_to_tag"][str(index)],"General") != "Unknown"]
        if not keep:
            raise ValueError("cannot export an empty vocabulary")
        head = nn.Linear(model.head.in_features,len(keep))
        with torch.no_grad():
            head.weight.copy_(model.head.weight[keep]); head.bias.copy_(model.head.bias[keep])
        model.core.evidence_head.head = head
        model.core.evidence_head.num_tags = len(keep)
        tags = [vocab["idx_to_tag"][str(index)] for index in keep]
        vocab.update(tag_to_idx={tag:index for index,tag in enumerate(tags)},
            idx_to_tag={str(index):tag for index,tag in enumerate(tags)},
            tag_to_category={tag:vocab["tag_to_category"].get(tag,"General") for tag in tags})
    cfg = manager.processor.config
    routed = getattr(model,"router",None) is not None
    direct = hasattr(model,"config")
    budget = (model.config.inference_modes.get("balanced",next(iter(model.config.inference_modes.values()))).local_views
        if direct else model.routing["selection_budget"] if routed else cfg.local_views)
    image = Image.new("RGB",(max(64,cfg.local_size*4),max(64,cfg.local_size*2)))
    views = build_views(image,cfg,seed=manager.processor.image_seed(image),training=False)
    batch = collate_view_batches([views])
    with torch.no_grad():
        overview = model.core.encoder(batch.overview_images,batch.overview_geometry)
    feature_values = tuple(getattr(overview,name) for name in FEATURES)
    local_images = batch.local_images[:,:budget]
    local_geometry = batch.local_geometry[:,:budget]
    local_mask = batch.local_view_mask[:,:budget]
    graphs = {target.name:(ClassifierGraph(model.core),(*feature_values,local_images,local_geometry,local_mask),
        (*FEATURES,"local_images","local_geometry","local_mask"),("logits","ood_features")),
        f"{target.stem}_overview.onnx":(OverviewGraph(model.core),(batch.overview_images,batch.overview_geometry),
            ("images","geometry"),FEATURES)}
    if routed:
        graphs[f"{target.stem}_router.onnx"]=(RouterGraph(model.router),
            (overview.local_map,overview.local_coordinates,batch.local_geometry[...,:4],batch.local_view_mask),
            ("local_map","coordinates","boxes","mask"),("scores",))
    with TemporaryDirectory(prefix="native-onnx-",dir=target.parent) as temp:
        temp = Path(temp)
        for name,(module,args,inputs,outputs) in graphs.items():
            axes = {}
            if name.endswith("_router.onnx"):
                axes.update(boxes={1:"candidates"},mask={1:"candidates"},scores={1:"candidates"})
            torch.onnx.export(module,args,str(temp/name),input_names=list(inputs),output_names=list(outputs),
                dynamic_axes=axes,opset_version=18,dynamo=False,do_constant_folding=False)
        for path in temp.iterdir():
            os.replace(path,target.parent/path.name)
    external_files = {}
    import onnx
    for name in graphs:
        graph = onnx.load(str(target.parent/name), load_external_data=False)
        for initializer in graph.graph.initializer:
            for entry in initializer.external_data:
                if entry.key == "location":
                    if Path(entry.value).name != entry.value:
                        raise ValueError("native ONNX external tensor path is not portable")
                    external_files[entry.value] = _sha256(target.parent/entry.value)
    metadata = _read_metadata(manager.checkpoint_path)
    metadata.update(architecture="foveatag",kind="foveatag_onnx_stages",format_version=1,
        fovea_view_config=cfg.to_dict(), fovea_seed=manager.processor.seed,
        max_source_pixels=manager.max_source_pixels,mixed_precision="fp32",
        vocabulary_file=f"{target.stem}_vocabulary.json",selection_budget=budget,batch_size=1,
        routing_host="dense_scale_space" if direct else "sampled_candidates" if routed else "sampled_views",
        vocabulary=vocab, graphs={name:_sha256(target.parent/name) for name in graphs},
        external_files=external_files)
    metadata.pop("inference_export", None)
    vocab_path = target.with_name(f"{target.stem}_vocabulary.json")
    vocab_path.write_text(json.dumps(vocab,ensure_ascii=False),encoding="utf-8")
    metadata_path = target.with_name(f"{target.stem}_metadata.json")
    sidecar_source=Path(manager.native_sidecar_base)
    for suffix in ("_tag_metrics.npz","_ood_ref.npz"):
        original=Path(str(sidecar_source)+suffix)
        if not original.is_file():
            continue
        destination=target.with_name(target.stem+suffix)
        if suffix == "_tag_metrics.npz" and len(keep) != old_count:
            if not _filter_tag_metrics_npz(str(original),str(destination),keep,old_count):
                raise ValueError("native exported metrics cannot be aligned to filtered vocabulary")
        else:
            shutil.copyfile(original,destination)
    if also_split:
        for name in graphs:
            _split_onnx_for_webgpu(str(target.parent/name),max_bytes=split_max_bytes)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata),encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return str(target),str(vocab_path)


class NativeOnnxAdapter(nn.Module):
    architecture="foveatag"
    training_method="onnx"
    is_native_onnx=True

    def __init__(self,path,metadata):
        super().__init__()
        import onnxruntime as ort
        self.register_buffer("device_marker",torch.zeros(1))
        self.metadata=metadata
        policy = metadata.get("routing_host")
        if policy not in {"dense_scale_space", "sampled_candidates", "sampled_views"} or metadata.get("batch_size") != 1:
            raise ValueError("unsupported native ONNX routing contract")
        if not isinstance(metadata.get("selection_budget"), int) or metadata["selection_budget"] < 0:
            raise ValueError("invalid native ONNX local budget")
        self._ood_tap=nn.Identity()
        self.router=nn.Identity() if metadata["routing_host"] != "sampled_views" else None
        root=Path(path).parent
        expected = {Path(path).name, f"{Path(path).stem}_overview.onnx"}
        if self.router is not None:
            expected.add(f"{Path(path).stem}_router.onnx")
        if set(metadata.get("graphs", {})) != expected:
            raise ValueError("native ONNX stage set differs")
        for filename, digest in metadata.get("external_files", {}).items():
            if Path(filename).name != filename or _sha256(root/filename) != digest:
                raise ValueError("native ONNX external tensor integrity differs")
        self.sessions={}
        for filename,digest in metadata["graphs"].items():
            if Path(filename).name != filename or not filename.endswith(".onnx") or _sha256(root/filename) != digest:
                raise ValueError("native ONNX graph integrity differs")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.sessions[filename]=ort.InferenceSession(str(root/filename),sess_options=options,providers=["CPUExecutionProvider"])
        self.path=Path(path)
        self.head=SimpleHead(len(metadata.get("vocabulary",{}).get("tag_to_idx",{})))

    def register_ood_feature_hook(self,hook):
        return self._ood_tap.register_forward_pre_hook(hook)

    def predict_native(self,image,processor):
        from .router import build_scale_space_candidates,source_candidate_mask,fit_candidate_boxes,select_proposals
        from .inference import source_field_sizes,_candidate_to_spec
        cfg=processor.config
        overview_views=build_views(image,cfg,seed=0,training=False,specs=())
        ov=collate_view_batches([overview_views])
        session=self.sessions[f"{self.path.stem}_overview.onnx"]
        values=session.run(None,{"images":ov.overview_images.numpy(),"geometry":ov.overview_geometry.numpy()})
        overview=dict(zip(FEATURES,values))
        if self.metadata["routing_host"] == "dense_scale_space":
            boxes=build_scale_space_candidates(torch.from_numpy(overview["local_coordinates"]),
                source_field_sizes(image.width,image.height,cfg.local_size,cfg.field_levels))
            if cfg.local_border_mode == "shift_inside":
                boxes=fit_candidate_boxes(boxes,image.width,image.height)
            valid=source_candidate_mask(boxes)
            specs=tuple(_candidate_to_spec(box,image.width,image.height,cfg.local_size) for box in boxes[0])
        else:
            specs=sample_local_specs(image.width,image.height,cfg,seed=processor.image_seed(image),training=False)
            if cfg.local_border_mode == "shift_inside":
                specs=tuple(fit_local_spec(spec,image.width,image.height) for spec in specs)
            boxes=torch.tensor([[spec.normalized_geometry(image.width,image.height)[:4] for spec in specs]],dtype=torch.float32)
            valid=torch.ones((1,len(specs)),dtype=torch.bool)
        if self.router is not None:
            session=self.sessions[f"{self.path.stem}_router.onnx"]
            scores=session.run(None,{"local_map":overview["local_map"],"coordinates":overview["local_coordinates"],
                "boxes":boxes.numpy(),"mask":valid.numpy()})[0]
            selected=select_proposals(torch.from_numpy(scores),boxes,budget=self.metadata["selection_budget"],candidate_mask=valid)[0]
            # Fixed graph budget retains invalid slots as masked crops.
            chosen=[specs[max(0,index)] for index in selected.tolist()]
        else:
            chosen=list(specs)
            selected=torch.arange(len(chosen))
        rendered=ViewBatch(overview_views.overview,overview_views.overview_spec,
            tuple(render_source_view(image,spec,cfg.pad_color) for spec in chosen),tuple(chosen),cfg.local_size)
        batch=collate_view_batches([rendered])
        local_mask=batch.local_view_mask & (selected >= 0)[None]
        inputs={**overview,"local_images":batch.local_images.numpy(),"local_geometry":batch.local_geometry.numpy(),
            "local_mask":local_mask.numpy()}
        session=self.sessions[self.path.name]
        inputs={value.name:inputs[value.name] for value in session.get_inputs()}
        logits,features=session.run(None,inputs)
        self._ood_tap(torch.from_numpy(features))
        return torch.from_numpy(logits)


class SimpleHead:
    def __init__(self,num_tags):
        self.out_features=num_tags
