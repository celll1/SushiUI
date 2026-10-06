"""Stronger spatial compression should calibrate native teacher features."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from core.models.components.latent_adapter import SpatialAlignmentPair, calibration_source_side
from core.models.components.spatial_swap import calibrate_images


@pytest.mark.parametrize("old,new,expected", [(8,16,256),(16,32,256),(8,32,128),(16,16,512),(16,8,512),(1,16,512)])
def test_teacher_grid_matches_target_without_upsizing_or_pixel_migration(old,new,expected):
    side=calibration_source_side(512,dict(old_scale=old,new_scale=new))
    assert side==expected
    if 1<old<new:
        assert side//old==512//new


class Codec(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(()))
        self.scale=scale
        self.sides=[]

    def encode(self,pixels):
        self.sides.append(pixels.shape[-1])
        value=F.avg_pool2d(pixels.square(),self.scale)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))


@pytest.mark.parametrize("old_scale,new_scale,source_side", [(2,4,64),(1,4,128),(4,2,128)])
def test_shared_calibration_encodes_the_selected_teacher_pixels(tmp_path,monkeypatch,old_scale,new_scale,source_side):
    from core.training import optimizer_residency as residency
    import core.training.base_trainer as base
    monkeypatch.setattr(residency,"encoder_refill_residency",lambda *a,**kw:nullcontext())
    monkeypatch.setattr(residency,"park_optimizer_state",lambda *a,**kw:nullcontext())
    monkeypatch.setattr(base,"all_optimizers",lambda trainer:[])
    rng=np.random.default_rng(63)
    items=[]
    for index in range(2):
        path=tmp_path/f'{index}.png'
        Image.fromarray(rng.integers(0,256,(128,128,3),dtype=np.uint8)).save(path)
        items.append(dict(image_path=str(path)))
    source,target=Codec(old_scale),Codec(new_scale)
    declaration=dict(version=1,method="paired_spatial",token_policy="target_native",source="synthetic",
        old_channels=3,new_channels=3,old_scale=old_scale,new_scale=new_scale,input_channels=3,
        ndim=4,norm="identity",norm_pack=1)
    model=nn.Module();model.vae_latent_bridge=SpatialAlignmentPair(declaration)
    wiring=SimpleNamespace(latent_ndim=4,vae_norm="identity",vae_norm_pack=1)
    trainer=SimpleNamespace(config=dict(vae_swap_calibration_samples=2,vae_swap_calibration_resolution=128),
        device=torch.device('cpu'),vae_dtype=torch.float32,vae=target,_check_stop_requested=lambda:None,
        pending=(source,wiring))
    progress=[]
    trainer._vae_alignment_progress_callback=lambda **kw:progress.append(kw)
    calibrate_images(trainer,[SimpleNamespace(items=items)],model=model,pending_key="pending")
    assert source.sides==[source_side]*2 and target.sides==[128]*2
    assert model.vae_latent_bridge.ready and not hasattr(trainer,"pending")
    assert [entry['step'] for entry in progress]==[1,2,3,4]
    assert all(torch.isfinite(parameter).all() for parameter in model.vae_latent_bridge.parameters())
