"""Generation-scoped native target partitions; auxiliary context is recomputed."""
import torch
from .partition import MingPartitionAdapter
from .partition_global import from_payload,generation_global_residual
from core.training.partition import PartitionBox


class MingPartitionExecutor:
    def __init__(self,model,plan,*,cancelled=lambda:False):
        if not plan.cores or len(plan.cores)!=len(plan.inputs):
            raise ValueError("Ming generation plan requires matching nonempty cores and inputs")
        height,width=plan.height,plan.width
        if type(height) is not int or type(width) is not int or min(height,width)<2:
            raise ValueError("Ming generation plan requires positive patch-grid dimensions")
        area=0
        for index,(core,box) in enumerate(zip(plan.cores,plan.inputs)):
            if any(type(v) is not int for b in (core,box) for v in (b.top,b.left,b.bottom,b.right)):
                raise ValueError("Ming generation coordinates must be integer patch positions")
            if not (0<=box.top<=core.top<core.bottom<=box.bottom<=height and
                    0<=box.left<=core.left<core.right<=box.right<=width):
                raise ValueError("Ming generation core/input lies outside its patch canvas")
            for other in plan.cores[:index]:
                if max(core.top,other.top)<min(core.bottom,other.bottom) and max(core.left,other.left)<min(core.right,other.right):
                    raise ValueError("Ming generation cores overlap")
            area+=core.height*core.width
        if area!=height*width:raise ValueError("Ming generation cores do not cover the canvas")
        topology=getattr(model,'_residual_unet_spec',None)
        if topology is not None:
            from core.rewiring.spec import preview_plan
            for box in plan.inputs:preview_plan(topology,box.height,box.width)
        payload=getattr(model,'_ming_partition_global_payload',None)
        self.base_global=None if payload is None else from_payload(payload['metadata'],payload['state'],
            latent_channels=model.in_channels,hidden_dim=model.dim,dtype=model.dtype).eval().requires_grad_(False)
        self.model,self.plan,self.cancelled=model,plan,cancelled
        self.closed=False

    @torch.no_grad()
    def predict(self,hidden_states,timestep,encoder_hidden_states,*,ref_hidden_states=None,
                encoder_hidden_states_2=None,return_dict=False):
        if self.closed:raise RuntimeError("Ming generation partition executor is closed")
        from .sampling import MingGenerationCancelled
        def check():
            if self.cancelled():raise MingGenerationCancelled("Ming generation cancelled")
        check()
        grid=torch.stack(hidden_states)
        if grid.ndim!=5 or grid.shape[-2:]!=(self.plan.height*2,self.plan.width*2):
            raise ValueError("Ming generation latents disagree with the patch canvas")
        output=torch.empty_like(grid)
        adapter=MingPartitionAdapter()
        for core,box in zip(self.plan.cores,self.plan.inputs):
            check()
            native_box=PartitionBox(box.top,box.left,box.bottom,box.right)
            residual=generation_global_residual(self.model,grid,native_box)
            if self.base_global is not None:
                base=self.base_global.to(device=grid.device)(grid,native_box)
                residual=base if residual is None else base+residual
            prediction=self.model(list(grid[...,box.top*2:box.bottom*2,box.left*2:box.right*2].unbind()),
                timestep,encoder_hidden_states,ref_x=ref_hidden_states,cap_feats_2=encoder_hidden_states_2,
                target_spatial_position_ids=adapter.position_ids(self.plan.height,self.plan.width,native_box,device=grid.device),
                target_input_residual=residual,return_dict=False)[0]
            value=torch.stack(prediction)
            expected=(*grid.shape[:-2],box.height*2,box.width*2)
            if tuple(value.shape)!=expected:raise ValueError("Ming partition prediction has unexpected native geometry")
            y,x=(core.top-box.top)*2,(core.left-box.left)*2
            output[...,core.top*2:core.bottom*2,core.left*2:core.right*2]=value[...,y:y+core.height*2,x:x+core.width*2]
        values=list(output.unbind())
        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=values)
        return (values,)

    def close(self):
        self.base_global=None
        self.closed=True


def normalize_partition_config(value):
    from api.param_defaults import GENERATION_DEFAULTS
    from core.inference.qwen_image_21_partition import normalize_partition_config as normalize
    try:return normalize(value,defaults=GENERATION_DEFAULTS['ming_image_01_partition'])
    except ValueError as exc:
        raise ValueError(str(exc).replace('qwen_image_21_partition','ming_image_01_partition').replace('Qwen','Ming')) from exc


def resolve_partition_plan(value,height,width,frames):
    from core.inference.qwen_image_21_partition import resolve_partition_plan as resolve
    settings=normalize_partition_config(value)
    if type(frames) is not int or frames<1:raise ValueError('Ming target frames must be a positive integer')
    if settings['mode']=='auto':
        settings['max_target_tokens']//=frames
        if settings['max_target_tokens']<1:raise ValueError('Ming token ceiling is too small for its target frames')
    try:return resolve(settings,height,width)
    except ValueError as exc:raise ValueError(str(exc).replace('Qwen','Ming')) from exc


def plan_metadata(plan,frames,*,global_adapter):
    from dataclasses import asdict
    return {'version':1,'mode':plan.mode,'patch_height':plan.height,'patch_width':plan.width,
        'target_frames':frames,'full_target_tokens':frames*plan.height*plan.width,
        'largest_input_tokens':frames*max(b.tokens for b in plan.inputs),'kv_cache':False,
        'global_adapter':global_adapter,
        'regions':[{'core':asdict(c),'input':asdict(b)} for c,b in zip(plan.cores,plan.inputs)]}
