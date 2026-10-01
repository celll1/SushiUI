"""Scheduler-owned predicted-clean previews and bounded native RGBA display."""
from contextlib import contextmanager
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@contextmanager
def scheduler_preview(scheduler,state,requested=None):
    missing=object();previous=scheduler.__dict__.get('step',missing)
    original=scheduler.step
    def step(model_output,timestep,sample,*args,**kwargs):
        sigma=scheduler.sigmas[state['index']].to(sample.device,dtype=torch.float32)
        # Scheduler sees noise-clean velocity, opposite to the native DiT output.
        state['clean']=(sample.float()-sigma*model_output.float()) if requested is None or requested(state['index']) else None
        return original(model_output,timestep,sample,*args,**kwargs)
    scheduler.step=step
    try:yield
    finally:
        if previous is missing:del scheduler.step
        else:scheduler.step=previous
        state['clean']=None


def primary_latent(latent,profile):
    if latent.ndim==4 and latent.shape[1]==16 and profile=='generation_edit':return latent
    if latent.ndim!=5 or latent.shape[1]!=16:
        raise ValueError('Ming preview requires native 16-channel image/frame latents')
    index=1 if profile=='layer_decompose' else 0
    if latent.shape[2]<=index:raise ValueError('Ming preview is missing its primary target frame')
    return latent[:,:,index]


@torch.no_grad()
def decode_preview(vae,latent):
    if latent.ndim!=4 or latent.shape[0]<1 or latent.shape[1]!=16 or min(latent.shape[-2:])<1 or not torch.isfinite(latent[:1]).all():
        raise ValueError('Ming preview requires finite native [B,16,H,W] latents')
    value=latent[:1]
    # Bound native VAE preview cost to a 256-pixel longest edge.
    ratio=min(1.,32/max(value.shape[-2:]))
    height,width=(max(2,math.floor(edge*ratio)) for edge in value.shape[-2:])
    if value.shape[-2:]!=(height,width):value=F.interpolate(value.float(),size=(height,width),mode='area')
    value=value.to(device=vae.device,dtype=vae.dtype).unsqueeze(2)
    shift=torch.as_tensor(vae.config.shift_factor,device=value.device,dtype=value.dtype)
    scale=torch.as_tensor(vae.config.scaling_factor,device=value.device,dtype=value.dtype)
    decoded=vae.decode(value/scale+shift,return_dict=False)[0]
    if decoded.shape[:3]!=(1,4,1):raise ValueError('Ming preview VAE must produce independent RGBA frames')
    pixels=((decoded[0,:,0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
    rgba=Image.fromarray(pixels)
    yy,xx=np.indices((rgba.height,rgba.width));tones=np.where((yy//8+xx//8)%2,190,235).astype(np.uint8)
    backdrop=Image.fromarray(np.repeat(tones[:,:,None],3,axis=2)).convert('RGBA')
    return Image.alpha_composite(backdrop,rgba).convert('RGB')
