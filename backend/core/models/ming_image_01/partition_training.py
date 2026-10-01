"""Sequential native region backward under one logical Ming noise draw."""
from contextlib import nullcontext
import math
from types import SimpleNamespace
import torch
from .flow import flow_training_pair
from .partition import MingPartitionAdapter,core_result,image_guidance_scale
from .conditioning import MingConditioning
from .guidance import prepare_guidance,guidance_loss,frame_error


def backward_regions(trainer,batch,*,sigma,noise,plan,backward_scale,drop_mask=None):
    if not math.isfinite(backward_scale) or backward_scale < 0:
        raise ValueError("Ming partition backward scale must be finite and nonnegative")
    clean=batch.clean.to(device=trainer.device,dtype=trainer.training_dtype)
    noise=noise.to(clean)
    sigma=sigma.to(clean)
    if noise.shape != clean.shape or sigma.shape != (clean.shape[0],):
        raise ValueError("Ming partition noise/sigma geometry disagrees with targets")
    if (plan.full_height*2,plan.full_width*2) != clean.shape[-2:]:
        raise ValueError("Ming partition plan disagrees with native target canvas")
    from core.training.partition.core import validate_partition_regions
    validate_partition_regions(plan.full_height,plan.full_width,plan.regions)
    weights=batch.frame_weights.to(device=clean.device,dtype=torch.float32)
    conditioning=MingConditioning(batch.conditioning.queries.to(clean),batch.conditioning.direct.to(clean))
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
        if indices is not None:
            x=x[indices];queries=torch.zeros_like(queries[indices]);direct=torch.zeros_like(direct[indices])
            refs=None if references is None else [references[i] for i in indices.tolist()]
            step_time=time[indices]
        context=torch.autocast(device_type=clean.device.type,dtype=trainer.training_dtype) if trainer.mixed_precision else nullcontext()
        with context:
            output=trainer.transformer(list(x.unbind()),step_time,list(queries.unbind()),
                cap_feats_2=list(direct.unbind()),ref_x=refs,return_dict=False,
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
    totals=clean.new_zeros(3,dtype=torch.float32)
    pred_items=torch.zeros(clean.shape[0],device=clean.device,dtype=torch.float32)
    loss_items=torch.zeros_like(pred_items);metrics={}
    shared_graph=any(value is not None and value.grad_fn is not None for value in
        [clean,conditioning.queries,conditioning.direct,*(references or [])])
    for index,region in enumerate(plan.regions):
        fraction=region.core.tokens/plan.full_tokens
        prediction=forward(region)
        result=core_result(prediction,crop(target,region.input),region,time,weights)
        proxy=SimpleNamespace(config=trainer.config,log_extra_metric=lambda k,v:metrics.__setitem__(k,metrics.get(k,0.)+fraction*v))
        pred_loss,per_item=guidance_loss(proxy,result,null_cores[index] if null_cores else None,
            sigma,weights,condition,selection,selected,effective_scale_override=effective)
        recovered=crop(noisy,region.core).float()+sigma.reshape(-1,1,1,1,1)*result.prediction.float()
        recon_items=frame_error(recovered,crop(clean,region.core),weights)
        recon_loss=recon_items.mean()
        loss=(1-recon_weight)*pred_loss+recon_weight*recon_loss
        scaled=fraction*loss*backward_scale
        retain=shared_graph and index+1<len(plan.regions)
        if getattr(trainer,"use_grad_scaler",False):trainer.grad_scaler.scale(scaled).backward(retain_graph=retain)
        else:scaled.backward(retain_graph=retain)
        totals+=fraction*torch.stack((loss.detach(),pred_loss.detach(),recon_loss.detach()))
        pred_items+=fraction*per_item.detach()
        loss_items+=fraction*((1-recon_weight)*per_item+recon_weight*recon_items).detach()
        del prediction,result,loss,scaled,recovered,recon_items,pred_loss,per_item,recon_loss
    trainer._last_loss_per_sample=pred_items
    if getattr(trainer,"_collect_concept_sample_loss",False):trainer._pending_concept_sample_loss=loss_items
    for name,value in metrics.items():trainer.log_extra_metric(name,value)
    trainer.log_extra_metric("ming_partition_count",float(len(plan.regions)))
    trainer.log_extra_metric("ming_partition_largest_tokens",float(plan.largest_input_tokens))
    return tuple(float(value) for value in totals)
