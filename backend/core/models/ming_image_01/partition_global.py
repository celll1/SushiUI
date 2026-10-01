"""Native multi-frame full-canvas summaries and strict checkpoint tensors."""
from collections.abc import Mapping
import torch
from torch import nn
from core.training.partition import PartitionBox


class MingPartitionGlobalAdapter(nn.Module):
    ADAPTER_ALGORITHM="ming_partition_global"
    WEIGHT_DECOMPOSE=False
    def __init__(self,latent_channels,hidden_dim,*,rank,summary_tokens,dtype=torch.float32):
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (latent_channels,hidden_dim,rank,summary_tokens)):
            raise ValueError("Ming global adapter dimensions must be positive integers")
        if rank > 512 or summary_tokens > 256:
            raise ValueError("Ming global adapter rank/tokens exceed the public bounds")
        self.latent_channels,self.hidden_dim=latent_channels,hidden_dim
        self.rank,self.summary_tokens=rank,summary_tokens
        features=4*latent_channels+3
        self.key=nn.Linear(features,rank,bias=False,dtype=dtype)
        self.value=nn.Linear(features,rank,bias=False,dtype=dtype)
        self.query=nn.Linear(features,rank,bias=False,dtype=dtype)
        self.summary_queries=nn.Parameter(torch.randn(summary_tokens,rank,dtype=dtype)/rank**.5)
        self.out=nn.Linear(rank,hidden_dim,bias=False,dtype=dtype)
        nn.init.zeros_(self.out.weight)

    def _features(self,latents,box=None):
        if latents.ndim != 5 or latents.shape[1] != self.latent_channels:
            raise ValueError("Ming global summaries require native [B,C,F,H,W] latents")
        batch,channels,frames,height,width=latents.shape
        if min(batch,frames,height,width)<1 or height%2 or width%2:
            raise ValueError("Ming global summaries require nonempty patch-aligned grids")
        h,w=height//2,width//2
        grid=latents.reshape(batch,channels,frames,h,2,w,2).permute(0,2,3,5,4,6,1).reshape(batch,frames,h,w,4*channels)
        axes=[torch.linspace(-1,1,size,device=latents.device,dtype=latents.dtype) for size in (frames,h,w)]
        coordinates=torch.stack(torch.meshgrid(*axes,indexing="ij"),dim=-1).unsqueeze(0).expand(batch,-1,-1,-1,-1)
        grid=torch.cat((grid,coordinates),dim=-1)
        if box is not None:
            if not isinstance(box,PartitionBox) or not (0<=box.top<box.bottom<=h and 0<=box.left<box.right<=w):
                raise ValueError("Ming global adapter box lies outside the target patch grid")
            grid=grid[:,:,box.top:box.bottom,box.left:box.right]
        return grid.reshape(batch,-1,grid.shape[-1])

    def forward(self,latents,box):
        full,local=self._features(latents),self._features(latents,box)
        if not torch.is_autocast_enabled(full.device.type):
            full,local=full.to(self.key.weight.dtype),local.to(self.key.weight.dtype)
        keys,values=self.key(full),self.value(full)
        weights=torch.softmax(torch.einsum("sr,bnr->bsn",self.summary_queries,keys)*self.rank**-.5,
            dim=-1,dtype=torch.float32).to(values.dtype)
        summaries=torch.einsum("bsn,bnr->bsr",weights,values)
        queries=self.query(local)
        weights=torch.softmax(torch.einsum("bnr,bsr->bns",queries,summaries)*self.rank**-.5,
            dim=-1,dtype=torch.float32).to(summaries.dtype)
        return self.out(torch.einsum("bns,bsr->bnr",weights,summaries))

    def branch_tensors(self):
        return {"key.weight":self.key.weight,"value.weight":self.value.weight,"query.weight":self.query.weight,
            "summary_queries":self.summary_queries,"out.weight":self.out.weight}

    def trainable_parameters(self):
        return iter(self.parameters())

    def spec_constants(self):
        return ()

    def metadata(self):
        return {"schema_version":1,"algorithm":"ming_partition_global","latent_channels":self.latent_channels,
            "hidden_dim":self.hidden_dim,"rank":self.rank,"summary_tokens":self.summary_tokens,
            "target_patch_size":2,"coordinate_axes":["frame","height","width"]}

    def export_tensors(self):
        return {key:value.detach().cpu().clone() for key,value in self.state_dict().items()}

    def load_tensors(self,tensors):
        expected=self.state_dict()
        if not isinstance(tensors,Mapping) or set(tensors)!=set(expected):
            raise ValueError("Ming global adapter tensors are incomplete or unexpected")
        for key,value in tensors.items():
            if (not isinstance(value,torch.Tensor) or value.shape!=expected[key].shape
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid Ming global adapter tensor: {key}")
        self.load_state_dict(tensors,strict=True)


def from_payload(metadata,tensors,*,latent_channels,hidden_dim,dtype=torch.float32):
    if (not isinstance(metadata,dict) or type(metadata.get("schema_version")) is not int
            or metadata.get("schema_version")!=1):
        raise ValueError("Unsupported Ming global adapter metadata version")
    if any(type(metadata.get(key)) is not int for key in ("latent_channels","hidden_dim","rank","summary_tokens","target_patch_size")):
        raise ValueError("Ming global adapter metadata dimensions must be integers")
    adapter=MingPartitionGlobalAdapter(latent_channels,hidden_dim,rank=metadata.get("rank"),
        summary_tokens=metadata.get("summary_tokens"),dtype=dtype)
    if metadata != adapter.metadata():
        raise ValueError("Ming global adapter metadata conflicts with the native architecture")
    adapter.load_tensors(tensors)
    return adapter


GLOBAL_NAME="ming_partition_global_adapter"


def payload_from_directory(root,*,latent_channels,hidden_dim):
    from pathlib import Path
    import json
    from safetensors.torch import load_file
    directory=Path(root)/"partition_global"
    if not directory.exists():return None
    if not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Ming global payload escapes the artifact root")
    config_path,state_path=directory/"config.json",directory/"adapter.safetensors"
    for path in (config_path,state_path):
        if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Ming global payload is incomplete or escapes its directory")
    maximum=(3*(4*latent_channels+3)*512+256*512+hidden_dim*512)*8+65536
    if config_path.stat().st_size>16384 or state_path.stat().st_size>maximum:
        raise ValueError("Ming global payload exceeds the bounded native adapter size")
    metadata=json.loads(config_path.read_text(encoding="utf-8"))
    state=load_file(str(state_path),device="cpu")
    adapter=from_payload(metadata,state,latent_channels=latent_channels,hidden_dim=hidden_dim)
    return {"metadata":adapter.metadata(),"state":{key:value.detach().cpu().clone() for key,value in state.items()}}


def write_payload(model,directory):
    from pathlib import Path
    import json
    from safetensors.torch import save_file
    adapter=getattr(model,GLOBAL_NAME,None)
    payload=({"metadata":adapter.metadata(),"state":adapter.export_tensors()} if adapter is not None
        else getattr(model,"_ming_partition_global_payload",None))
    if payload is None:return
    directory=Path(directory)/"partition_global";directory.mkdir()
    (directory/"config.json").write_text(json.dumps(payload["metadata"],sort_keys=True,indent=2)+"\n",encoding="utf-8")
    save_file(payload["state"],str(directory/"adapter.safetensors"))


def prepare_full_training(trainer,*,adapter_dtype=None,allow_adapter_resume=False):
    from api.param_defaults import TRAINING_DEFAULTS
    config=trainer.config;model=trainer.transformer
    def value(name):
        key="dit_partition_global_"+name
        v=config.get(key,config.get("qwen_partition_global_"+name))
        return TRAINING_DEFAULTS[key] if v is None else v
    payload=getattr(model,"_ming_partition_global_payload",None)
    existing=getattr(model,GLOBAL_NAME,None)
    enabled=value("adapter_enabled")
    if type(enabled) is not bool:raise ValueError("Ming global adapter enabled must be boolean")
    if not enabled:
        if payload is not None or existing is not None:raise ValueError("Ming checkpoint contains a global adapter but training disabled it")
        return None
    if not trainer.train_unet:raise ValueError("Ming global adapter requires train_unet")
    if not config.get("dit_partition_training_enabled",config.get("qwen_partition_training_enabled",False)):
        raise ValueError("Ming global adapter requires spatial partition training")
    rank,tokens=value("rank"),value("tokens")
    dtype=trainer.weight_dtype if adapter_dtype is None else adapter_dtype
    if existing is not None:
        if (existing.rank,existing.summary_tokens)!=(rank,tokens):raise ValueError("Ming global adapter dimensions changed")
        return existing.requires_grad_(True)
    if payload is not None:
        if (payload["metadata"]["rank"],payload["metadata"]["summary_tokens"])!=(rank,tokens):
            raise ValueError("Ming global adapter dimensions changed from the checkpoint")
        adapter=from_payload(payload["metadata"],payload["state"],latent_channels=model.in_channels,
            hidden_dim=model.dim,dtype=dtype)
    else:
        if getattr(trainer,"_loaded_checkpoint_path",None) and not allow_adapter_resume:raise ValueError("Cannot add a missing Ming global adapter during resume")
        adapter=MingPartitionGlobalAdapter(model.in_channels,model.dim,rank=rank,summary_tokens=tokens,dtype=dtype)
    adapter=adapter.to(next(model.parameters()).device).train().requires_grad_(True)
    setattr(model,GLOBAL_NAME,adapter)
    if payload is not None:del model._ming_partition_global_payload
    return adapter
