"""Qwen-compatible guidance schedules with native Ming frame supervision."""
import torch
from core.training import guidance_loss as _guidance


def prepare_guidance(trainer,sigma,drop_mask):
    config=trainer.config
    weight=_guidance.weight(config)
    eligible=torch.ones_like(sigma,dtype=torch.bool)
    if drop_mask is not None:
        if drop_mask.dtype!=torch.bool or drop_mask.shape!=sigma.shape:
            raise ValueError('Ming guidance null labels require a boolean [B] mask')
        eligible &= ~drop_mask.to(device=sigma.device)
    ramp=_guidance.resolve_weight_schedule(config,weight)
    if (not weight and (ramp[0]!='high_noise_smoothstep' or not ramp[1])) or not bool(eligible.any()):return None,None,None
    condition=_guidance.condition(config,eligible)
    selection,selected=_guidance.selection(trainer,sigma,condition)
    return condition,selection,selected


def frame_error(prediction,target,weights):
    per_frame=(prediction.float()-target.float()).square().mean(dim=(1,3,4))
    weights=weights.to(device=per_frame.device,dtype=torch.float32)
    return (per_frame*weights).sum(1)/weights.sum(1)


def guidance_loss(trainer,result,uncond,sigma,weights,condition,selection,selected,*,effective_scale_override=None):
    if condition is None:return result.loss,result.per_item_loss
    scale,schedule=condition[4:6]
    effective=(torch.full_like(sigma,scale,dtype=torch.float32) if schedule=='constant'
               else 1+(scale-1)*sigma.float())
    requested=effective
    distance=None
    cap=_guidance.extrapolation_cap(trainer.config)
    if effective_scale_override is not None:
        if (effective_scale_override.shape != sigma.shape or not torch.isfinite(effective_scale_override).all()
                or (effective_scale_override < 1).any()):
            raise ValueError("Ming partition guidance scale requires finite [B] values >= 1")
        effective = effective_scale_override.detach().to(device=sigma.device,dtype=torch.float32)
    elif cap>0 and uncond is not None:
        distance=frame_error(sigma.reshape(-1,1,1,1,1)*(result.target.float()-uncond.float()),
                             torch.zeros_like(result.target),weights)
        effective=_guidance.capped_scale(sigma,condition,distance,cap)
    if bool(selected.any()) and uncond is None:
        raise ValueError('Ming selected guidance requires a native null prediction')
    normal=result.per_item_loss
    if uncond is not None:
        target=uncond.float()+effective.reshape(-1,1,1,1,1)*(result.target.float()-uncond.float())
        guided=frame_error(result.prediction,target,weights)
        per_item=(1-selection)*normal+selection*guided
    else:
        guided=torch.zeros_like(normal);per_item=normal
    trainer.log_extra_metric('ming_guidance_loss_normal',float(normal.detach().mean()))
    trainer.log_extra_metric('ming_guidance_loss_guided',float(guided[selected].detach().mean()) if bool(selected.any()) else 0.)
    trainer.log_extra_metric('ming_guidance_loss_mix_weight',float(_guidance.mix_for_condition(sigma,condition).mean()))
    trainer.log_extra_metric('ming_guidance_loss_selected_fraction',float(selected.float().mean()))
    if bool(selected.any()):
        trainer.log_extra_metric('ming_guidance_effective_scale_mean',float(effective[selected].mean()))
        if distance is not None:
            trainer.log_extra_metric('ming_guidance_effective_scale_min',float(effective[selected].min()))
            trainer.log_extra_metric('ming_guidance_capped_fraction',float((effective[selected]<requested[selected]).float().mean()))
            trainer.log_extra_metric('ming_guidance_null_clean_rms',float(distance[selected].sqrt().mean()))
    return per_item.mean(),per_item
