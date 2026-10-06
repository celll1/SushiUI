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


class SDNoiseSession:
    def __init__(self):
        self.trajectory=None
        self.generator=None

    def initialize(self,scheduler,state,generator):
        self.trajectory=SDNoiseTrajectory(scheduler,state)
        if generator is None:
            self.generator=torch.Generator(device=state.device)
            self.generator.seed()
        else:
            self.generator=SDNoiseTrajectory.clone_generator(generator)

    def exposure(self,timestep,sample,override=None):
        if self.trajectory is None:
            raise ValueError("Gaussian SD generation has no initialized noise trajectory")
        noise=self.trajectory.noise(timestep) if override is None else override
        if sample.shape[0]%noise.shape[0] or sample.shape[2:]!=noise.shape[2:]:
            raise ValueError("Gaussian SD exposure noise does not match the model batch/grid")
        return noise.repeat(sample.shape[0]//noise.shape[0],1,1,1).to(sample)


def initialize_noise_tracking(model,scheduler,state,generator):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is not None:
        session.initialize(scheduler,state,generator)


def aligned_scheduler_step(model,scheduler,prediction,timestep,latents,*,generator):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is not None:
        if session.trajectory is None:
            raise ValueError("Gaussian SD scheduler has no initialized noise trajectory")
        generator=generator if generator is not None else session.generator
        session.trajectory.step(timestep,generator=generator)
    return scheduler.step(prediction,timestep,latents,generator=generator)


def reference_noise_kwargs(model,kwargs,noise):
    if getattr(model,"_sushi_sd_noise_session",None) is not None:
        return {**kwargs,"_sushi_alignment_noise":noise}
    return kwargs


def blend_tracked_noise(model,noise_state,weight):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is not None:
        if session.trajectory is None:
            raise ValueError("Gaussian SD blend has no initialized noise trajectory")
        session.trajectory.state=(1-weight)*session.trajectory.state+weight*noise_state


def aligned_noise_jump(model,scheduler,latents,timesteps,*,hi_index,lo_index,
                       generator,fallback_generator,jump):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is None:
        generator=generator if generator is not None else fallback_generator
    else:
        generator=generator if generator is not None else session.generator
        session.trajectory.state=jump(scheduler,session.trajectory.state,timesteps,
            hi_index=hi_index,lo_index=lo_index,generator=SDNoiseTrajectory.clone_generator(generator))
    return jump(scheduler,latents,timesteps,hi_index=hi_index,lo_index=lo_index,generator=generator)


def rewind_tracked_noise(model,index):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is not None:
        session.trajectory.scheduler._step_index=index


def composite_reference_noise(model,timestep,initial_noise,mask):
    session=getattr(model,"_sushi_sd_noise_session",None)
    if session is None:
        return None
    current=session.exposure(timestep,initial_noise)
    return (1-mask)*initial_noise+mask*current


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
    session=SDNoiseSession() if getattr(pair,"noise_alignment",None) is not None else None
    previous_session=getattr(model,"_sushi_sd_noise_session",None)
    model._sushi_sd_noise_session=session

    def incoming(module,args,kwargs):
        kwargs=dict(kwargs)
        override=kwargs.pop("_sushi_alignment_noise",None)
        sample=args[0] if args else kwargs["sample"]
        timestep=args[1] if len(args)>1 else kwargs["timestep"]
        alpha,sigma,process=coefficients(scheduler,timestep,sample)
        if prediction=="flow_prediction" and process!="flow":
            raise ValueError("Aligned flow prediction requires a flow sigma schedule")
        noise=session.exposure(timestep,sample,override) if session is not None else None
        mapped=map_input(module,sample,alpha=alpha,sigma=sigma,noise=noise)
        context.append((alpha,sigma,process,noise))
        if args:
            return (mapped,*args[1:]),kwargs
        return args,{**kwargs,"sample":mapped}

    def outgoing(module,args,kwargs,output):
        if not context:
            return output
        alpha,sigma,process,noise=context.pop()
        if output is None:
            return output
        value=output[0] if isinstance(output,tuple) else output.sample
        mapped=map_prediction(module,value,noise_process=process,
            prediction_target=targets[prediction],sigma=sigma,alpha=alpha,noise=noise)
        if isinstance(output,tuple):
            return (mapped,*output[1:])
        return type(output)(sample=mapped)

    pre=post=None
    try:
        pre=model.register_forward_pre_hook(incoming,with_kwargs=True)
        post=model.register_forward_hook(outgoing,with_kwargs=True,always_call=True)
        yield
    finally:
        if pre is not None:
            pre.remove()
        if post is not None:
            post.remove()
        context.clear()
        if previous_session is None:
            del model._sushi_sd_noise_session
        else:
            model._sushi_sd_noise_session=previous_session


def aligned_sampling(function):
    @wraps(function)
    def run(*args,**kwargs):
        pipeline=args[0] if args else kwargs["pipeline"]
        with sampling_boundary(pipeline.unet,pipeline.scheduler):
            return function(*args,**kwargs)
    return run
