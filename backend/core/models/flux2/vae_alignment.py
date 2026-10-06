"""Flux2 packed image/reference boundaries shared by native and custom forwards."""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from core.models.components.spatial_swap import map_packed_image

_contexts = ContextVar("flux2_codec_context",default=())


def apply_swap(trainer,resolved,*,handler,module=None):
    from dataclasses import replace
    from core.models.components.latent_io import resize_latent_io,ResizeReport
    from core.models.components.wiring import FLUX2_LATENT_IO
    from core.models.components.spatial_swap import initialize_image_swap
    from core.training.vae_swap import initialization_mode
    from .residual_unet import native_config,restore_topology
    from core.rewiring.spec import build_spec
    model=trainer.transformer
    pair=getattr(model,"vae_latent_bridge",None)
    if pair is not None:
        if module is None and resolved.source!=pair.declaration["source"]:
            raise ValueError("Changing an aligned Flux2 VAE requires a fresh base checkpoint")
        trainer.vae_identity=replace(resolved,state_dict=None)
        trainer.vae_latent_channels=pair.declaration["new_channels"]
        trainer.wiring=handler.resolve_wiring(trainer)
        bind_boundaries(model)
        channels=model.x_embedder.in_features//4
        return ResizeReport((),channels,channels,channels,0,0)
    if module is not None:
        report=resize_latent_io(model,FLUX2_LATENT_IO,resolved.latent_channels)
        trainer.vae=module
        trainer.vae_identity=resolved
        trainer.vae_latent_channels=resolved.latent_channels
        trainer.wiring=handler.wiring.replace(latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor,vae_norm=resolved.norm,vae_norm_pack=resolved.norm_pack)
        return report
    if initialization_mode(trainer.config,"flux2")=="zero":
        if resolved.ndim!=4 or resolved.scale_factor!=8:
            raise ValueError("Legacy zero Flux2 swaps require a 4D/8x VAE; use paired_spatial")
        from core.training.arch.base_arch import ArchHandler
        return ArchHandler.apply_vae_swap(handler,trainer,resolved)
    from core.training.adapters.base_adapter import reject_quantized_base
    reject_quantized_base(model,model_label="Flux2 VAE swap")
    def resize(channels):
        topology=getattr(model,"_residual_unet_spec",None)
        report=resize_latent_io(model,FLUX2_LATENT_IO,channels)
        if topology is not None:
            streams=[{"name":"double","block_count":len(model.transformer_blocks)},
                {"name":"single","block_count":len(model.single_transformer_blocks)}]
            restore_topology(model,build_spec("flux2",streams,topology["regions"],native_config(model)))
        return report
    report=initialize_image_swap(trainer,resolved,handler=handler,model=model,
        old_channels=model.x_embedder.in_features//4,source_wiring=handler.resolve_wiring(trainer),
        patch_size=2,resize=resize,pending_key="_flux2_vae_calibration_source")
    if getattr(model,"vae_latent_bridge",None) is not None:bind_boundaries(model)
    return report


def calibrate(trainer,datasets):
    from core.models.components.spatial_swap import calibrate_images
    calibrate_images(trainer,datasets,model=trainer.transformer,pending_key="_flux2_vae_calibration_source")


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
