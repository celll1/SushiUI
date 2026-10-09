"""Sequential native region backward under one logical Ming noise draw."""
from contextlib import nullcontext
import math
from types import SimpleNamespace
import torch
from .flow import flow_training_pair
from .partition import MingPartitionAdapter,core_result,image_guidance_scale
from .conditioning import MingConditioning
from .guidance import prepare_guidance,guidance_loss,frame_error
from .vae_alignment import forward_with_noise


def _backward_regions(trainer,batch,*,sigma,noise,plan,backward_scale,drop_mask=None,
        debug_save_path=None,debug_captions=None,debug_reference_image_paths=None):
    if not math.isfinite(backward_scale) or backward_scale < 0:
        raise ValueError("Ming partition backward scale must be finite and nonnegative")
    clean=batch.clean.to(device=trainer.device,dtype=trainer.training_dtype,non_blocking=True)
    noise=noise.to(clean)
    sigma=sigma.to(clean)
    if noise.shape != clean.shape or sigma.shape != (clean.shape[0],):
        raise ValueError("Ming partition noise/sigma geometry disagrees with targets")
    if (plan.full_height*2,plan.full_width*2) != clean.shape[-2:]:
        raise ValueError("Ming partition plan disagrees with native target canvas")
    from core.training.partition.core import validate_partition_regions
    validate_partition_regions(plan.full_height,plan.full_width,plan.regions)
    topology=getattr(trainer.transformer,"_residual_unet_spec",None)
    if topology is not None:
        from .residual_unet import validate_forward
        for region in plan.regions:
            box=region.input
            tiles=clean[...,box.top*2:box.bottom*2,box.left*2:box.right*2]
            validate_forward(trainer.transformer,list(tiles.unbind()),2,1,topology)
    from .partition_timing import MingPartitionTiming
    timing=MingPartitionTiming(trainer,clean)
    weights=batch.frame_weights.to(device=clean.device,dtype=torch.float32)
    conditioning=MingConditioning(batch.conditioning.queries.to(clean,non_blocking=True),batch.conditioning.direct.to(clean,non_blocking=True))
    references=None if batch.reference_latents is None else [None if v is None else v.to(clean) for v in batch.reference_latents]
    noisy,target,time=flow_training_pair(clean,noise,sigma)
    condition,selection,selected=prepare_guidance(trainer,sigma,drop_mask)
    recon_weight=float(trainer.reconstruction_loss_weight)
    if not 0 <= recon_weight <= 1:raise ValueError("Ming reconstruction loss weight must be within [0, 1]")
    adapter=MingPartitionAdapter()
    def crop(value,box):return value[...,2*box.top:2*box.bottom,2*box.left:2*box.right]
    def forward(region,indices=None):
        x=crop(noisy,region.input)
        queries,direct=conditioning.queries,conditioning.direct
        refs=references;step_time=time
        exposure=crop(noise,region.input)
        if indices is not None:
            x=x[indices];queries=torch.zeros_like(queries[indices]);direct=torch.zeros_like(direct[indices])
            refs=None if references is None else [references[i] for i in indices.tolist()]
            step_time=time[indices]
            exposure=exposure[indices]
        context=torch.autocast(device_type=clean.device.type,dtype=trainer.training_dtype) if trainer.mixed_precision else nullcontext()
        with timing.measure("forward"),context:
            global_adapter=getattr(trainer.transformer,"ming_partition_global_adapter",None)
            global_source=noisy if indices is None else noisy[indices]
            alignment=getattr(trainer.transformer,"vae_latent_bridge",None)
            if global_adapter is not None and alignment is not None:
                global_source=alignment.flow_state(global_source,1-step_time,noise=noise if indices is None else noise[indices])
            residual=None if global_adapter is None else global_adapter(global_source,region.input)
            output=forward_with_noise(trainer.transformer,exposure,list(x.unbind()),step_time,list(queries.unbind()),
                cap_feats_2=list(direct.unbind()),ref_x=refs,return_dict=False,
                target_input_residual=residual,
                target_spatial_position_ids=adapter.position_ids(plan.full_height,plan.full_width,region.input,device=clean.device))[0]
        return torch.stack(output)
    null_cores=[];effective=None
    if condition is not None and bool(selected.any()):
        indices=selected.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            for region in plan.regions:
                output=forward(region,indices)
                box=region.core_in_input
                value=crop(output,box)
                core=value.new_zeros(clean.shape[0],*value.shape[1:]);core[indices]=value
                null_cores.append(core)
            effective=image_guidance_scale(trainer,sigma,condition,weights,
                [(crop(target,r.core),u) for r,u in zip(plan.regions,null_cores)],plan)
    debug_prediction=torch.empty_like(clean[:1],device="cpu",dtype=torch.float32) if debug_save_path is not None else None
    debug_null=torch.empty_like(debug_prediction) if debug_prediction is not None and null_cores and bool(selected[0]) else None
    totals=clean.new_zeros(3,dtype=torch.float32)
    pred_items=torch.zeros(clean.shape[0],device=clean.device,dtype=torch.float32)
    loss_items=torch.zeros_like(pred_items);metrics={}
    shared_graph=any(value is not None and value.grad_fn is not None for value in
        [clean,conditioning.queries,conditioning.direct,*(references or [])])
    for index,region in enumerate(plan.regions):
        fraction=region.core.tokens/plan.full_tokens
        prediction=forward(region)
        result=core_result(prediction,crop(target,region.input),region,time,weights)
        if debug_prediction is not None:
            core=region.core
            debug_prediction[...,2*core.top:2*core.bottom,2*core.left:2*core.right]=result.prediction[:1].detach().float().cpu()
            if debug_null is not None:
                debug_null[...,2*core.top:2*core.bottom,2*core.left:2*core.right]=null_cores[index][:1].detach().float().cpu()
        proxy=SimpleNamespace(config=trainer.config,log_extra_metric=lambda k,v:metrics.__setitem__(k,metrics.get(k,0.)+fraction*v))
        pred_loss,per_item=guidance_loss(proxy,result,null_cores[index] if null_cores else None,
            sigma,weights,condition,selection,selected,effective_scale_override=effective)
        recovered=crop(noisy,region.core).float()+sigma.reshape(-1,1,1,1,1)*result.prediction.float()
        recon_items=frame_error(recovered,crop(clean,region.core),weights)
        recon_loss=recon_items.mean()
        loss=(1-recon_weight)*pred_loss+recon_weight*recon_loss
        scaled=fraction*loss*backward_scale
        retain=shared_graph and index+1<len(plan.regions)
        with timing.measure("backward"):
            if getattr(trainer,"use_grad_scaler",False):trainer.grad_scaler.scale(scaled).backward(retain_graph=retain)
            else:scaled.backward(retain_graph=retain)
        totals+=fraction*torch.stack((loss.detach(),pred_loss.detach(),recon_loss.detach()))
        pred_items+=fraction*per_item.detach()
        loss_items+=fraction*((1-recon_weight)*per_item+recon_weight*recon_items).detach()
        del prediction,result,loss,scaled,recovered,recon_items,pred_loss,per_item,recon_loss
    trainer._last_loss_per_sample=pred_items
    if getattr(trainer,"_collect_concept_sample_loss",False):trainer._pending_concept_sample_loss=loss_items
    for name,value in metrics.items():trainer.log_extra_metric(name,value)
    timing.finish(plan,trainer.transformer)
    from .training_debug import save_debug
    save_debug(trainer,debug_save_path,clean=clean,noisy=noisy,target=target,prediction=debug_prediction,
        sigma=sigma,frame_weights=weights,loss=totals[0],recon_loss=totals[2],unconditional=debug_null,
        captions=debug_captions,reference_paths=debug_reference_image_paths,plan=plan)
    return tuple(float(value) for value in totals)


def region_count(trainer,clean):
    from api.param_defaults import TRAINING_DEFAULTS
    config=trainer.config
    def value(name):
        canonical="dit_partition_"+name
        return config.get(canonical,config.get("qwen_partition_"+name)) if config.get(canonical,config.get("qwen_partition_"+name)) is not None else TRAINING_DEFAULTS[canonical]
    if not value("training_enabled"):return 1
    mode=value("mode")
    if mode == "fixed":
        count=value("fixed_count")
        if count not in (2,4):raise ValueError("Ming partition count must be 2 or 4")
        return count
    if mode != "token_threshold":raise ValueError("Ming partition mode must be fixed or token_threshold")
    limit=value("full_frame_max_tokens")
    if type(limit) is not int or limit < 1:raise ValueError("Ming partition token threshold must be a positive integer")
    tokens=clean.shape[2]*(clean.shape[-2]//2)*(clean.shape[-1]//2)
    return 1 if tokens<=limit else 4 if tokens>=4*limit else 2


def training_plan(trainer,clean):
    from api.param_defaults import TRAINING_DEFAULTS
    config=trainer.config
    def value(name):
        canonical="dit_partition_"+name
        candidate=config.get(canonical,config.get("qwen_partition_"+name))
        return TRAINING_DEFAULTS[canonical] if candidate is None else candidate
    count=region_count(trainer,clean)
    if count == 1:raise ValueError("Ming full-frame execution has no partition plan")
    model = getattr(trainer.transformer, "module", trainer.transformer)
    topology = getattr(model, "_residual_unet_spec", None)
    halo = value("halo_tokens")
    if (topology or {}).get("method") == "hybrid_unet_ssdit":
        from core.rewiring.hybrid_partition import resolved_partition_halo, spatial_codec_radius
        codec_radius = spatial_codec_radius(getattr(model, "vae_latent_bridge", None), 2)
        halo = resolved_partition_halo(topology, "main", halo, codec_radius=codec_radius)
    plan = MingPartitionAdapter().build_fixed_plan(clean.shape[-2]//2,clean.shape[-1]//2,count=count,
        halo=halo,seed=value("seed"),epoch=int(getattr(trainer,"_current_epoch",0)),
        occurrence=int(getattr(trainer,"_current_batch_position",0)),
        split_ratio_min=value("split_ratio_min"),split_ratio_max=value("split_ratio_max"))
    from .residual_unet import spatial_alignment
    ah, aw = spatial_alignment(trainer.transformer)
    if ah == aw == 1:
        return plan
    if plan.full_height % ah or plan.full_width % aw:
        raise ValueError("Ming U-net partition canvas must match cumulative downsample alignment")
    from dataclasses import replace
    from core.training.partition.core import PartitionBox, PartitionRegion, validate_partition_regions
    regions = tuple(PartitionRegion(region.core, PartitionBox(
        region.input.top//ah*ah, region.input.left//aw*aw,
        min(plan.full_height, (region.input.bottom+ah-1)//ah*ah),
        min(plan.full_width, (region.input.right+aw-1)//aw*aw))) for region in plan.regions)
    validate_partition_regions(plan.full_height, plan.full_width, regions)
    return replace(plan, regions=regions)


def backward_regions(trainer,batch,**kwargs):
    from .checkpointing import partition_scope
    with partition_scope(trainer.transformer,trainer.config):
        return _backward_regions(trainer,batch,**kwargs)
