"""Flux2 packed image/reference boundaries shared by native and custom forwards."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from core.models.components.spatial_swap import map_packed_image

_contexts = ContextVar("flux2_codec_context",default=())


@contextmanager
def alignment_context(model,hidden_states,img_ids,timestep,*,num_ref_tokens=0,ref_fixed_timestep=0.):
    if getattr(model,"vae_latent_bridge",None) is None:
        yield
        return
    bind_boundaries(model)
    token=_contexts.set(_contexts.get()+((model,img_ids,timestep,num_ref_tokens,ref_fixed_timestep),))
    try: yield
    finally: _contexts.reset(token)


def _layout(ids,batch,count):
    known=getattr(ids,"_spatial_codec_layout",None)
    try: version=ids._version
    except RuntimeError: version=None
    if known is not None and version is not None and known[:3]==(version,batch,count):
        return known[3]
    rows=ids.detach().cpu()
    if rows.ndim==3:
        if rows.shape[0] not in (1,batch) or not torch.equal(rows,rows[:1].expand_as(rows)):
            raise ValueError("Flux2 spatial alignment requires shared image grids across the batch")
        rows=rows[0]
    if rows.shape!=(count,4) or count<1 or not torch.isfinite(rows).all():
        raise ValueError("Flux2 spatial alignment requires native T/H/W/L image IDs")
    boundaries=[0,*((rows[1:,0]!=rows[:-1,0]).nonzero().flatten()+1).tolist(),count]
    segments=[]
    for start,end in zip(boundaries,boundaries[1:]):
        time=rows[start,0].item()
        region=rows[start:end]
        h,w=int(region[:,1].max())+1,int(region[:,2].max())+1
        expected=torch.stack((torch.arange(h).repeat_interleave(w),torch.arange(w).repeat(h)),dim=1)
        if h*w!=end-start or not torch.equal(region[:,1:3],expected):
            raise ValueError("Flux2 spatial alignment requires rectangular row-major image grids")
        segments.append((start,end,h,w,time>=10))
    if version is not None:ids._spatial_codec_layout=(version,batch,count,segments)
    return segments


def _map(model,states,*,output):
    for context in reversed(_contexts.get()):
        if context[0] is model:break
    else:raise ValueError("Flux2 spatial bridge has no forward geometry context")
    _,ids,sigma,num_ref_tokens,ref_sigma=context
    if output and num_ref_tokens:
        ids=ids[...,num_ref_tokens:,:]
    segments=_layout(ids,states.shape[0],states.shape[1])
    pair=model.vae_latent_bridge
    return torch.cat([map_packed_image(pair,states[:,start:end],(h,w),
        sigma=(ref_sigma if reference else sigma),output=output)
        for start,end,h,w,reference in segments],dim=1)


def _enter(module,args,kwargs):
    _bind_projections(module)
    names=("hidden_states","encoder_hidden_states","timestep","img_ids")
    values=dict(zip(names,args)); values.update(kwargs)
    context=(module,values.get("img_ids"),values.get("timestep"),
        values.get("num_ref_tokens",0) if values.get("kv_cache_mode")=="extract" else 0,
        values.get("ref_fixed_timestep",0.))
    _contexts.set(_contexts.get()+(context,))


def _leave(module,args,kwargs,output):
    contexts=_contexts.get()
    if contexts and contexts[-1][0] is module:_contexts.set(contexts[:-1])


def _input_hook(module,args):
    return (_map(_contexts.get()[-1][0],args[0],output=False),)


def _output_hook(module,args,output):
    return _map(_contexts.get()[-1][0],output,output=True)


def _bind_projections(model):
    # Quantization can replace projection modules after the checkpoint loads.
    if _input_hook not in model.x_embedder._forward_pre_hooks.values():
        model.x_embedder.register_forward_pre_hook(_input_hook)
    if _output_hook not in model.proj_out._forward_hooks.values():
        model.proj_out.register_forward_hook(_output_hook)


def bind_boundaries(model):
    """Closure-free hooks also follow EMA copies and replaced projection modules."""
    if _enter not in model._forward_pre_hooks.values():
        model.register_forward_pre_hook(_enter,with_kwargs=True)
        model.register_forward_hook(_leave,with_kwargs=True,always_call=True)
    _bind_projections(model)
