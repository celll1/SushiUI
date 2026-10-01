"""Post-step native flow guidance; target frames exclude Layer's auxiliary canvas."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch

from api.param_defaults import REFERENCE_GUIDE_DEFAULTS
from .training_data import encode_frames


@dataclass(frozen=True)
class GuideConfig:
    image: Image.Image
    strength: float
    start: int
    end: int


def normalize_guides(entries):
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Ming Reference Guides require an ordered entry list")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("is_reference_guide"):
            raise ValueError("Ming accepts Reference Guide entries, not ControlNet image entries")
        image = entry.get("image")
        if not isinstance(image, Image.Image):
            raise ValueError("Ming Reference Guide requires a decoded reference image")
        strength = entry.get("strength", REFERENCE_GUIDE_DEFAULTS["strength"])
        start = entry.get("start_step", REFERENCE_GUIDE_DEFAULTS["start_step"])
        end = entry.get("end_step", REFERENCE_GUIDE_DEFAULTS["end_step"])
        if (type(strength) not in (int,float) or not math.isfinite(strength) or not 0<=strength<=2
                or any(type(v) is not int for v in (start,end)) or not 0<=start<=end<=1000):
            raise ValueError("Ming Reference Guide requires strength in [0,2] and ordered integer steps in [0,1000]")
        result.append(GuideConfig(image,float(strength),start,end))
    return result


@dataclass
class PreparedGuide:
    clean: torch.Tensor
    noise: torch.Tensor
    config: GuideConfig


@torch.no_grad()
def prepare_guides(vae, configs, *, width, height, batch, seed, check_cancel):
    prepared = []
    for index, config in enumerate(configs):
        check_cancel()
        if config.strength == 0:
            continue
        pixels = np.asarray(config.image.convert("RGBA").resize((width,height),Image.Resampling.BILINEAR)).copy()
        pixels = (torch.from_numpy(pixels).permute(2,0,1).float()/127.5-1)[None,:,None].expand(batch,-1,-1,-1,-1)
        clean = encode_frames(vae,pixels).squeeze(2).float()
        if clean.shape != (batch,16,height//8,width//8) or not torch.isfinite(clean).all():
            raise ValueError("Ming Reference Guide VAE latent shape/value disagrees with the native 16-channel grid")
        generator = torch.Generator(device="cpu").manual_seed((seed+17011+index)%(2**63))
        noise = torch.randn(clean.shape,generator=generator,dtype=torch.float32).to(clean.device)
        prepared.append(PreparedGuide(clean,noise,config))
        check_cancel()
    return prepared


def blend_guides(latents, guides, *, index, total, sigma, profile):
    if not guides:
        return latents
    if profile not in {"generation_edit","layer_decompose"}:
        raise ValueError("Unsupported Ming Reference Guide profile")
    if type(index) is not int or type(total) is not int or not 0<=index<total:
        raise ValueError("Invalid Ming Reference Guide step")
    sigma = torch.as_tensor(sigma,device=latents.device,dtype=latents.dtype)
    if sigma.numel()!=1 or not torch.isfinite(sigma).all() or not 0<=sigma.item()<=1:
        raise ValueError("Ming Reference Guide next sigma must be in [0,1]")
    if not latents.is_floating_point():
        raise ValueError("Ming Reference Guide requires floating latents")
    progress = index/max(total-1,1)*1000
    result = latents
    for guide in guides:
        config = guide.config
        if not config.start<=progress<=config.end or config.strength == 0:
            continue
        if guide.clean.shape != guide.noise.shape or guide.clean.device != result.device:
            raise ValueError("Ming Reference Guide latent ownership/shape mismatch")
        if profile=="generation_edit":
            if result.shape!=guide.clean.shape:
                raise ValueError("Ming Design guide must match the target latent grid")
            reference = (1-sigma)*guide.clean+sigma*guide.noise
            result = (1-config.strength)*result+config.strength*reference
        else:
            if result.ndim!=5 or result.shape[2]<2 or result.shape[:2]+result.shape[3:]!=guide.clean.shape:
                raise ValueError("Ming Layer guide must match each independent target layer")
            reference = ((1-sigma)*guide.clean+sigma*guide.noise).unsqueeze(2)
            layers = (1-config.strength)*result[:,:,1:]+config.strength*reference
            result = torch.cat((result[:,:,:1],layers),dim=2)
    return result.to(dtype=latents.dtype)
