"""Temporary SD codec boundaries using the sampler's actual noise state."""
from contextlib import contextmanager
from functools import wraps

import torch

_SCALED = frozenset({"EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler", "KDPM2AncestralDiscreteScheduler", "HeunDiscreteScheduler",
    "LMSDiscreteScheduler"})
_VP_SIGMAS = frozenset({"DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler",
    "UniPCMultistepScheduler"})
_DISCRETE = frozenset({"DDIMScheduler", "DDPMScheduler", "PNDMScheduler"})


def coefficients(scheduler, timestep, reference):
    name=type(scheduler).__name__
    if name in _SCALED or name in _VP_SIGMAS or name=="FlowMatchEulerDiscreteScheduler":
        index=getattr(scheduler,"step_index",None)
        if index is None:
            index=getattr(scheduler,"begin_index",None)
        if index is None:
            t=torch.as_tensor(timestep,device=scheduler.timesteps.device)
            index=scheduler.index_for_timestep(t)
        if name=="KDPM2DiscreteScheduler" and not scheduler.state_in_first_order:
            sigma=scheduler.sigmas_interpol[index]
        elif name=="KDPM2AncestralDiscreteScheduler" and not scheduler.state_in_first_order:
            sigma=scheduler.sigmas_interpol[index-1]
        else:
            sigma=scheduler.sigmas[index]
        sigma=sigma.to(device=reference.device,dtype=reference.dtype)
        if name=="FlowMatchEulerDiscreteScheduler" or getattr(scheduler.config,"use_flow_sigmas",False):
            return 1-sigma,sigma,"flow"
        alpha=torch.rsqrt(1+sigma.square())
        return alpha,sigma*alpha,"ddpm"
    if name in _DISCRETE:
        t=torch.as_tensor(timestep,device=reference.device).long()
        alpha_bar=scheduler.alphas_cumprod.to(reference.device)[t].to(reference.dtype)
        return alpha_bar.sqrt(),(1-alpha_bar).sqrt(),"ddpm"
    raise ValueError(f"SD spatial alignment does not support scheduler {name}")


@contextmanager
def sampling_boundary(model,scheduler):
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is None:
        yield
        return
    from core.models.components.spatial_swap import validate_loaded
    from core.models.components.sd_vae_alignment import map_input,map_prediction
    validate_loaded(model)
    targets={"epsilon":"epsilon","v_prediction":"velocity","sample":"sample",
        "original_sample":"sample","flow_prediction":"velocity"}
    prediction=("flow_prediction" if type(scheduler).__name__=="FlowMatchEulerDiscreteScheduler"
        else getattr(scheduler.config,"prediction_type","epsilon"))
    if prediction not in targets:
        raise ValueError(f"Unsupported SD spatial prediction type: {prediction}")
    context=[]

    def incoming(module,args,kwargs):
        sample=args[0] if args else kwargs["sample"]
        timestep=args[1] if len(args)>1 else kwargs["timestep"]
        alpha,sigma,process=coefficients(scheduler,timestep,sample)
        if prediction=="flow_prediction" and process!="flow":
            raise ValueError("Aligned flow prediction requires a flow sigma schedule")
        mapped=map_input(module,sample,alpha=alpha)
        context.append((sigma,process))
        if args:
            return (mapped,*args[1:]),kwargs
        return args,{**kwargs,"sample":mapped}

    def outgoing(module,args,kwargs,output):
        if not context:
            return output
        sigma,process=context.pop()
        if output is None:
            return output
        value=output[0] if isinstance(output,tuple) else output.sample
        mapped=map_prediction(module,value,noise_process=process,
            prediction_target=targets[prediction],sigma=sigma)
        if isinstance(output,tuple):
            return (mapped,*output[1:])
        return type(output)(sample=mapped)

    pre=model.register_forward_pre_hook(incoming,with_kwargs=True)
    post=None
    try:
        post=model.register_forward_hook(outgoing,with_kwargs=True,always_call=True)
        yield
    finally:
        pre.remove()
        if post is not None:
            post.remove()
        context.clear()


def aligned_sampling(function):
    @wraps(function)
    def run(*args,**kwargs):
        pipeline=args[0] if args else kwargs["pipeline"]
        with sampling_boundary(pipeline.unet,pipeline.scheduler):
            return function(*args,**kwargs)
    return run
