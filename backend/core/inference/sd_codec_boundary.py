"""Temporary SD codec boundaries using the sampler's actual noise state."""
from contextlib import contextmanager
from functools import wraps
import copy
import inspect

import torch

_SCALED = frozenset({"EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler",
    "KDPM2DiscreteScheduler", "KDPM2AncestralDiscreteScheduler", "HeunDiscreteScheduler",
    "LMSDiscreteScheduler"})
_VP_SIGMAS = frozenset({"DPMSolverMultistepScheduler", "DPMSolverSinglestepScheduler",
    "UniPCMultistepScheduler"})
_DISCRETE = frozenset({"DDIMScheduler", "DDPMScheduler", "PNDMScheduler"})


class SDNoiseTrajectory:
    """Noise-only sampler history; callers supply the same explicit step RNG."""

    def __init__(self,scheduler,noise_state):
        name=type(scheduler).__name__
        if name not in _SCALED|_VP_SIGMAS|_DISCRETE|{"FlowMatchEulerDiscreteScheduler"}:
            raise ValueError(f"Gaussian noise tracking does not support scheduler {name}")
        config=scheduler.config
        self.prediction=("flow_prediction" if name=="FlowMatchEulerDiscreteScheduler"
            else getattr(config,"prediction_type","epsilon"))
        if self.prediction not in ("epsilon","v_prediction","sample","original_sample","flow_prediction"):
            raise ValueError("Gaussian noise tracking requires a declared prediction type")
        if name=="PNDMScheduler" and self.prediction=="v_prediction":
            raise ValueError("PNDM v-prediction history does not preserve the Gaussian clean/noise split")
        if getattr(config,"clip_sample",False) or getattr(config,"thresholding",False):
            raise ValueError("Gaussian noise tracking requires an unclipped scheduler")
        if getattr(config,"variance_type",None) in ("learned","learned_range"):
            raise ValueError("Gaussian noise tracking does not support learned variance")
        if getattr(config,"stochastic_sampling",False):
            raise ValueError("Flow stochastic_sampling ignores the supplied step generator")
        if noise_state.ndim!=4 or not torch.isfinite(noise_state).all():
            raise ValueError("Gaussian noise tracking requires finite BCHW noise state")
        self.scheduler=copy.deepcopy(scheduler)
        self.state=noise_state.detach().clone()

    def noise(self,timestep):
        scaled=(self.state if type(self.scheduler).__name__=="FlowMatchEulerDiscreteScheduler"
            else self.scheduler.scale_model_input(self.state,timestep))
        _,sigma,_=coefficients(self.scheduler,timestep,scaled)
        if torch.any(sigma<=0):
            raise ValueError("Gaussian noise tracking requires a positive sampling noise coefficient")
        return scaled/sigma

    @staticmethod
    def clone_generator(generator):
        if isinstance(generator,list):
            return [SDNoiseTrajectory.clone_generator(value) for value in generator]
        if not isinstance(generator,torch.Generator):
            raise ValueError("Gaussian stochastic tracking requires an explicit step generator")
        cloned=torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state())
        return cloned

    @torch.no_grad()
    def step(self,timestep,*,generator=None,**kwargs):
        if kwargs.get("s_churn",0)!=0 or kwargs.get("per_token_timesteps") is not None:
            raise ValueError("Gaussian noise tracking requires the validated scalar timestep path")
        noise=self.noise(timestep)
        alpha,_,_=coefficients(self.scheduler,timestep,noise)
        output=(alpha*noise if self.prediction=="v_prediction" else
            torch.zeros_like(noise) if self.prediction in ("sample","original_sample") else noise)
        parameters=inspect.signature(self.scheduler.step).parameters
        if "generator" in parameters:
            kwargs["generator"]=self.clone_generator(generator)
        result=self.scheduler.step(output,timestep,self.state,**kwargs)
        self.state=result[0] if isinstance(result,tuple) else result.prev_sample
        return self.state


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
