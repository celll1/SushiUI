"""Native style preparation and step-scoped conditional CFG ownership."""
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image
import torch
from core.inference.reference_style import style_config_from_dict, StyleContext
from .training_data import encode_frames
from .style_transfer import style_scope


@dataclass
class PreparedStyle:
    clean: torch.Tensor
    noise: torch.Tensor
    config: object


def normalize_styles(entries):
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError("Ming styles require an ordered decoded entry list")
    normalized = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("image"), Image.Image):
            raise ValueError("Ming style requires a decoded reference image")
        config = style_config_from_dict(entry)
        names = ("ref_k_strength", "adain_strength", "value_adain_strength", "ref_value_mix",
                 "high_scale_start", "high_scale_end", "low_scale_start", "low_scale_end", "beta", "late_release")
        if any(not math.isfinite(getattr(config, name)) or getattr(config, name) < 0 for name in names):
            raise ValueError("Ming style values must be finite nonnegative numbers")
        if any(getattr(config, name) > 1 for name in ("adain_strength", "value_adain_strength", "ref_value_mix", "late_release")):
            raise ValueError("Ming style blend values must be within [0, 1]")
        if not 0 <= config.start_step <= config.end_step <= 1000:
            raise ValueError("Ming style steps must be ordered within [0, 1000]")
        if config.value_mode not in {"target_adain", "ref_raw"}:
            raise ValueError("Ming style value mode is unsupported")
        if config.block_range is not None and not 0 <= config.block_range[0] <= config.block_range[1]:
            raise ValueError("Ming style block range must be ordered and nonnegative")
        normalized.append((entry["image"], config))
    return normalized


@torch.no_grad()
def prepare_styles(vae, entries, *, width, height, batch, seed, check_cancel):
    prepared = []
    for index, (image, config) in enumerate(entries):
        check_cancel()
        if config.ref_k_strength == 0 and config.adain_strength == 0:
            continue
        pixels = np.asarray(image.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)).copy()
        pixels = (torch.from_numpy(pixels).permute(2,0,1).float()/127.5-1)[None,:,None].expand(batch,-1,-1,-1,-1)
        clean = encode_frames(vae, pixels).squeeze(2).float()
        if clean.shape != (batch,16,height//8,width//8) or not torch.isfinite(clean).all():
            raise ValueError("Ming style VAE latent geometry/value disagrees with native grid")
        generator = torch.Generator(device="cpu").manual_seed((seed+29023+index) % (2**63))
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean.device)
        prepared.append(PreparedStyle(clean,noise,config))
        check_cancel()
    return prepared


class MingStyleExecutor:
    def __init__(self, model, scheduler, styles, *, batch, total, combine_mode, cancelled):
        if combine_mode not in {"stack", "common_concept"}:
            raise ValueError("Ming style combine mode must be stack or common_concept")
        if type(batch) is not int or batch < 1 or type(total) is not int or total < 1:
            raise ValueError("Ming style requires positive batch and step counts")
        self.model,self.scheduler,self.styles = model,scheduler,tuple(styles)
        self.batch,self.total,self.combine_mode,self.cancelled = batch,total,combine_mode,cancelled
        self.index,self.closed = 0,False

    def check_cancel(self):
        if self.cancelled():
            from .sampling import MingGenerationCancelled
            raise MingGenerationCancelled("Ming generation cancelled during style capture")

    @torch.no_grad()
    def predict(self, hidden_states, timestep, encoder_hidden_states, *, ref_hidden_states=None,
                encoder_hidden_states_2=None, return_dict=False):
        if self.closed or self.index >= self.total:
            raise RuntimeError("Ming style executor is closed or exhausted")
        self.check_cancel()
        count = len(hidden_states)
        if count not in {self.batch,2*self.batch}:
            raise ValueError("Ming style CFG batch disagrees with native sampler")
        kwargs = dict(t=timestep, cap_feats=encoder_hidden_states, ref_x=ref_hidden_states,
                      cap_feats_2=encoder_hidden_states_2, return_dict=False)
        active = [style for style in self.styles if style.config.is_step_active(self.index,self.total)
                  and (style.config.ref_k_strength != 0 or style.config.adain_strength > 0)]
        stores = []
        try:
            if not active:
                outputs = self.model(x=hidden_states,**kwargs)[0]
            else:
                positive = hidden_states[:self.batch]
                conditional = dict(t=timestep[:self.batch],cap_feats=encoder_hidden_states[:self.batch],
                    cap_feats_2=encoder_hidden_states_2[:self.batch],
                    ref_x=None if ref_hidden_states is None else ref_hidden_states[:self.batch],return_dict=False)
                sigma = self.scheduler.sigmas[self.index].to(device=positive[0].device,dtype=torch.float32)
                if sigma.numel() != 1 or not torch.isfinite(sigma).all() or not 0 <= sigma.item() <= 1:
                    raise ValueError("Ming style requires a finite native scheduler sigma within [0, 1]")
                for style in active:
                    self.check_cancel()
                    if style.clean.shape != style.noise.shape or style.clean.shape[:2] != (self.batch,16) or style.clean.shape[-2:] != positive[0].shape[-2:]:
                        raise ValueError("Ming prepared style disagrees with target latent geometry")
                    if style.clean.device != positive[0].device or style.noise.device != positive[0].device:
                        raise ValueError("Ming style latents must share the target execution device")
                    if not torch.isfinite(style.clean).all() or not torch.isfinite(style.noise).all():
                        raise ValueError("Ming style latents must be finite")
                    noisy = ((1-sigma)*style.clean+sigma*style.noise).to(positive[0])
                    frames = positive[0].shape[1]
                    reference = list(noisy.unsqueeze(2).expand(-1,-1,frames,-1,-1).unbind())
                    ctx = StyleContext("capture",style.config,progress=style.config.step_progress(self.index,self.total))
                    stores.append((ctx.store,style.config))
                    with style_scope(self.model,ctx,reference):
                        self.model(x=reference,**conditional)
                self.check_cancel()
                cfg = stores[0][1]
                ctx = StyleContext("inject",cfg,refs=stores,combine_mode=self.combine_mode,
                                   progress=cfg.step_progress(self.index,self.total))
                with style_scope(self.model,ctx,positive):
                    outputs = self.model(x=positive,**conditional)[0]
                if count == 2*self.batch:
                    self.check_cancel()
                    unconditional = dict(t=timestep[self.batch:],cap_feats=encoder_hidden_states[self.batch:],
                        cap_feats_2=encoder_hidden_states_2[self.batch:],
                        ref_x=None if ref_hidden_states is None else ref_hidden_states[self.batch:],return_dict=False)
                    outputs += self.model(x=hidden_states[self.batch:],**unconditional)[0]
            self.check_cancel()
            self.index += 1
            if return_dict:
                from diffusers.models.modeling_outputs import Transformer2DModelOutput
                return Transformer2DModelOutput(sample=outputs)
            return (outputs,)
        finally:
            for store,_config in stores:
                store.clear()

    def close(self):
        self.closed = True
        self.styles = ()
