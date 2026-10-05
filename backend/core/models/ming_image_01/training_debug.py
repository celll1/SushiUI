"""Native sign/frame-aware debug snapshots and deferred RGBA VAE previews."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from api.param_defaults import TRAINING_DEFAULTS
from core.training.latent_debug_dump import video_filmstrip


def save_debug(trainer,path,*,clean,noisy,target,prediction,sigma,frame_weights,
               loss,recon_loss,unconditional=None,captions=None,reference_paths=None,plan=None):
    if path is None:
        return
    try:
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        frames=clean.shape[2];height,width=clean.shape[-2:]
        first_sigma=float(sigma[0])
        tensors={'latents':clean,'noisy_latents':noisy,'actual_velocity':target,
                 'predicted_velocity':prediction}
        tensors={name:value[:1].detach().float().cpu() for name,value in tensors.items()}
        tensors['predicted_latent']=tensors['noisy_latents']+first_sigma*tensors['predicted_velocity']
        data={name:video_filmstrip(value) for name,value in tensors.items()}
        roles=['image'] if frames==1 else ['auxiliary_canvas']+[f'layer_{i}' for i in range(1,frames)]
        data.update(timestep=first_sigma,native_model_time=1-first_sigma,timestep_convention='sigma',
                    velocity_sign='clean_minus_noise',loss=float(loss),recon_loss=float(recon_loss),
                    batch_size=int(clean.shape[0]),scheduler_type='FlowMatching',model_type='ming_image_01',
                    native_frame_count=frames,frame_roles=roles,frame_weights=frame_weights[0].detach().float().cpu().tolist(),
                    partition_count=1 if plan is None else len(plan.regions))
        if float(trainer.config.get('qwen_guidance_loss_weight',TRAINING_DEFAULTS['qwen_guidance_loss_weight']) or 0)>0:
            data['uncond_prediction_status']='computed' if unconditional is not None else 'not_selected'
        if unconditional is not None:
            tensors['uncond_predicted_latent']=tensors['noisy_latents']+first_sigma*unconditional[:1].detach().float().cpu()
            data['uncond_predicted_velocity']=video_filmstrip(unconditional[:1])
            data['uncond_predicted_latent']=video_filmstrip(tensors['uncond_predicted_latent'])
        if captions:data.update(caption=captions[0],all_captions=list(captions))
        if reference_paths:
            first=next((item for item in reference_paths if item),None)
            if first:data['reference_image_path']=first
        if plan is not None:
            data['partition_grid_size']=[frames*width,height]
            data['partition_regions']=[dict(top=2*r.core.top,left=2*r.core.left+frame*width,
                bottom=2*r.core.bottom,right=2*r.core.right+frame*width)
                for frame in range(frames) for r in plan.regions]
        torch.save(data,path/f'latents_t{first_sigma:.4f}.pt')
        view=trainer.config.get('qwen_debug_latent_view',TRAINING_DEFAULTS['qwen_debug_latent_view'])
        if view=='pixel':
            names={'target':'latents','noisy':'noisy_latents','pred_x0':'predicted_latent'}
            if unconditional is not None:names['uncond_pred_x0']='uncond_predicted_latent'
            trainer._pending_ming_debug_previews=dict(path=path,timestep=first_sigma,
                previews=[(name,tensors[key]) for name,key in names.items()])
    except Exception as exc:
        print(f"{getattr(trainer,'log_prefix','[Ming]')} [debug_latents] save failed: {exc}")


@torch.no_grad()
def flush_pending_debug_previews(trainer):
    pending=getattr(trainer,'_pending_ming_debug_previews',None)
    if pending is None:return
    del trainer._pending_ming_debug_previews
    from core.training.debug_decode_residency import debug_decode_residency
    with debug_decode_residency(trainer, trainer.vae, dtype=trainer.vae_dtype):
        from core.training.ops.ming_image_01_ops import vae_decode
        with torch.inference_mode():
            for name,latent in pending['previews']:
                pixels=vae_decode(trainer,latent)
                if pixels.shape[1]!=4:raise ValueError('Ming debug decode requires native RGBA')
                array=((pixels[0].float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,3,0).cpu().numpy()
                strip=np.concatenate(list(array),axis=1)
                Image.fromarray(strip).save(pending['path']/f"decode_t{pending['timestep']:.4f}_{name}.webp",'WEBP',lossless=True)
                del pixels, array, strip
