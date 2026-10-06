"""GPU-only synthetic Qwen bridge/U-net backward probe; no dataset access."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
import torch
from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
from core.models.qwen_image_21.vae_alignment import install
from core.models.qwen_image_21.residual_unet import native_config,restore_topology
from core.models.components.latent_io import resize_latent_io
from core.models.components.wiring import QWEN_IMAGE_21_LATENT_IO
from core.rewiring.spec import build_spec,OPERATORS

torch.manual_seed(218)
model = QwenImage21Transformer2DModel(in_channels=3,out_channels=3,num_layers=7,
    num_attention_heads=1,attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)
resize_latent_io(model,QWEN_IMAGE_21_LATENT_IO,8)
model.out_channels = 8
install(model,dict(version=1,method="paired_spatial",token_policy="target_native",source="synthetic:probe",
    old_channels=3,new_channels=8,old_scale=16,new_scale=32,input_channels=3,ndim=4,norm="shift_scale",norm_pack=1))
for layer in (model.vae_latent_bridge.input,model.vae_latent_bridge.output):
    w = torch.zeros(8*9,8)
    for c in range(8): w[c*9+4,c] = 1
    layer.initialize(w,torch.zeros(8))
regions = [dict(id="coarse",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
           dict(id="deep",stream="main",parent="coarse",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)]
restore_topology(model,build_spec("qwen_image_21",[{"name":"main","block_count":7}],regions,native_config(dict(model.config))))
model.to(device="cuda",dtype=torch.bfloat16)
model.enable_gradient_checkpointing()
x = torch.randn(1,96,8,device="cuda",dtype=torch.bfloat16,requires_grad=True)
text = torch.randn(1,3,12,device="cuda",dtype=torch.bfloat16)
mask = torch.cat((torch.zeros(1,3,device="cuda",dtype=torch.bool),torch.ones(1,24,device="cuda",dtype=torch.bool)),dim=1)
torch.cuda.reset_peak_memory_stats()
result = model(x,text,torch.tensor([.4],device="cuda"),[[(1,8,12)]],mask).sample
loss = result[:,-96:].float().square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
assert model.vae_latent_bridge.input.projection.weight.grad.abs().sum() > 0
assert model.vae_latent_bridge.output.projection.weight.grad.abs().sum() > 0
assert model.img_in.weight.grad[:,3:].abs().sum() > 0
print(json.dumps(dict(shape=list(result.shape),loss=float(loss.detach()),
    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
    unet=True,checkpointing=True,extra_channel_grad=True)))
