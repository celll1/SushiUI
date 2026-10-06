"""Codec geometry shared by the three LTX target-clock pipeline variants."""

from core.models.ltx2.video_clock import prepare_video_coords


class CodecGeometry:
    def _codec_latent_frames(self, frames):
        return self.vae.contract.latent_frames(frames)

    def _codec_video_coords(self, batch, frames, height, width, device, *, fps):
        return prepare_video_coords(self.transformer.rope,self.vae.contract,
                                    batch,frames,height,width,device,fps=fps)

    def preprocess_conditions(self, conditions, height, width, num_frames, device=None):
        if self.vae.contract.temporal_policy != "minimax_h3_chunks_v1":
            return super().preprocess_conditions(conditions,height,width,num_frames,device=device)
        import numpy as np
        import PIL.Image
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        contract = self.vae.contract
        spans = contract.frame_spans(contract.latent_frames(num_frames))
        if isinstance(conditions,LTX2VideoCondition):
            conditions = [conditions]
        pixels, strengths, indices = [], [], []
        for condition in conditions or []:
            value = condition.frames
            if isinstance(value,PIL.Image.Image):
                value = [value]
            elif isinstance(value,np.ndarray) and value.ndim == 3:
                value = value[None]
            elif isinstance(value,torch.Tensor) and value.ndim == 3:
                value = value[None]
            value = self.video_processor.preprocess_video(value,height,width,resize_mode="crop")
            index = condition.index % len(spans) if condition.index < 0 else condition.index
            if not 0 <= index < len(spans):
                raise ValueError("Video condition index is outside the target codec timeline")
            if value.shape[2] > 1 and index % 5:
                raise ValueError("H3 video conditions must start at the same codec chunk phase")
            limit = min(value.shape[2],num_frames-spans[index][0])
            if value.shape[2] > 1 and limit < 22:
                raise ValueError("The remaining target timeline cannot hold an H3 moving condition clip")
            length = 1 if limit < 22 else ((limit-5)//17)*17+5
            pixels.append(value[:,:,:length].to(device=device,dtype=self.vae.dtype))
            strengths.append(condition.strength)
            indices.append(index)
        return pixels, strengths, indices


def pipeline_class(native, vae):
    from core.models.components.video_latent_adapter import NormalizedVideoVAE
    if not isinstance(vae,NormalizedVideoVAE):
        return native
    from . import LTX2Pipeline, LTX2ImageToVideoPipeline, LTX2ConditionPipeline
    classes = {cls.__name__:cls for cls in (LTX2Pipeline,LTX2ImageToVideoPipeline,LTX2ConditionPipeline)}
    return classes[native.__name__]


def install_pipeline_clock(pipeline):
    target = pipeline_class(type(pipeline),pipeline.vae)
    if type(pipeline) is not target:
        pipeline.__class__ = target
    return pipeline
