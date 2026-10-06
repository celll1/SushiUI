"""Target-grid generation and periodic sampling with native CPU DiTs and VAE."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _owner(monkeypatch,dtype=torch.float32,rewired=False):
    from diffusers import AutoencoderKL,FlowMatchEulerDiscreteScheduler
    from core.models.components.latent_adapter import NormalizedImageVAE
    from core.models.components.spatial_swap import install
    from core.training.arch.ideogram4 import Ideogram4ArchHandler
    from core.models.ideogram4 import ideogram4_pipeline_ops as ops
    from latent_adapter_contract_test import declaration,identity
    from ideogram4_full_training_test import _trainer
    from core.models.ideogram4.vae_alignment import resize_pair
    from api.param_defaults import IDEOGRAM4_INFERENCE_DEFAULTS
    owner=_trainer(True)
    if rewired:
        from core.models.ideogram4.residual_unet import native_config,restore_topology
        from core.rewiring.spec import build_spec,OPERATORS
        for key in ("transformer","transformer_uncond"):
            model=getattr(owner,key)
            model=type(model).from_config(model.config,num_layers=7)
            restore_topology(model,build_spec("ideogram4",[{"name":"main","block_count":7}],[
                dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
                dict(id="inner",stream="main",parent="outer",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)
            ],native_config(model)))
            setattr(owner,key,model)
    resize_pair(owner.transformer,owner.transformer_uncond,4)
    decl=declaration(2,4)
    for model in (owner.transformer,owner.transformer_uncond):
        install(model,decl);identity(model.vae_latent_bridge.input,.2);identity(model.vae_latent_bridge.output,.3)
        model.to(dtype)
    raw=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
        block_out_channels=(4,)*5,down_block_types=("DownEncoderBlock2D",)*5,
        up_block_types=("UpDecoderBlock2D",)*5,norm_num_groups=1,sample_size=128,scaling_factor=1.)
    owner.vae=NormalizedImageVAE(raw,decl)
    class Encoder(torch.nn.Linear):
        @property
        def device(self):return self.weight.device
    owner.text_encoder=Encoder(3,3)
    class Tokenizer:
        def apply_chat_template(self,*a,**k):return "synthetic prompt"
        def __call__(self,*a,**k):return {"input_ids":torch.tensor([[2,3]])}
    owner.tokenizer=Tokenizer()
    owner.device=torch.device("cpu");owner.scheduler=FlowMatchEulerDiscreteScheduler()
    owner._check_stop_requested=lambda:None
    owner.arch=Ideogram4ArchHandler(owner)
    monkeypatch.setitem(IDEOGRAM4_INFERENCE_DEFAULTS,"max_sequence_length",4)
    monkeypatch.setattr(ops,"_get_text_encoder_hidden_states",lambda encoder,ids,mask,positions:
        [torch.randn(ids.shape[0],ids.shape[1],12,generator=torch.Generator().manual_seed(18))])
    return owner


@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("rewired", [False,True])
def test_native_periodic_sample_progress_target_grid_and_owner_restoration(monkeypatch,dtype,rewired):
    from core.training.arch.base_arch import SampleContext
    owner=_owner(monkeypatch,dtype,rewired)
    progress=[];counts=[]
    handle=owner.transformer.register_forward_pre_hook(lambda module,args,kwargs:
        counts.append(kwargs["hidden_states"].shape[1]),with_kwargs=True)
    parameter=next(owner.transformer.parameters())
    parameter.grad=torch.randn_like(parameter)
    original_gradient=parameter.grad.clone()
    owner.transformer.train();owner.transformer_uncond.eval()
    owner.optimizer=torch.optim.AdamW(owner.transformer.parameters())
    owner.optimizer.state[parameter]={"step":torch.tensor(4.),"exp_avg":torch.zeros_like(parameter),
        "exp_avg_sq":torch.ones_like(parameter)}
    ctx=SampleContext(prompt="synthetic prompt",width=256,height=256,num_inference_steps=2,
        guidance_scale=2.,seed=17,step_progress_callback=lambda done,total:progress.append((done,total)))
    try:
        image=owner.arch.sample(owner,ctx)
        second=owner.arch.sample(owner,ctx)
    finally:handle.remove()
    assert image.size == (256,256) and np.array_equal(np.asarray(image),np.asarray(second))
    assert counts == [68]*4  # four text tokens + 8x8 image tokens, not the source's 16x16 grid.
    assert progress == [(1,2),(2,2)]*2
    assert owner.transformer.training and not owner.transformer_uncond.training
    torch.testing.assert_close(parameter.grad,original_gradient)
    assert owner.optimizer.state[parameter]["step"].item() == 4


def test_sample_stop_restores_training_owner(monkeypatch):
    from core.training.arch.base_arch import SampleContext
    owner=_owner(monkeypatch)
    owner.transformer.train();owner.transformer_uncond.eval()
    parameter=next(owner.transformer.parameters())
    parameter.grad=torch.ones_like(parameter)
    calls=[]
    def stop():
        calls.append(1)
        if len(calls)==3:raise RuntimeError("synthetic stop")
    owner._check_stop_requested=stop
    with pytest.raises(RuntimeError,match="synthetic stop"):
        owner.arch.sample(owner,SampleContext(prompt="test",width=256,height=256,
            num_inference_steps=2,guidance_scale=2.,seed=17))
    assert owner.transformer.training and not owner.transformer_uncond.training
    assert torch.equal(parameter.grad,torch.ones_like(parameter))


def test_backend_resolution_and_noise_follow_external_codec(monkeypatch):
    from core.pipeline_backends.ideogram4 import Ideogram4Mixin
    from core.models.ideogram4.ideogram4_pipeline_ops import prepare_latents
    owner=_owner(monkeypatch)
    backend=SimpleNamespace(ideogram4_components={"vae":owner.vae})
    config=Ideogram4Mixin._ideogram4_common_params(backend,{"width":528,"height":784,"seed":15},512,512)
    assert (config["width"],config["height"]) == (512,768)
    assert (config["grid_h"],config["grid_w"],config["packed_channels"]) == (24,16,16)
    assert prepare_latents(24,16,torch.float32,"cpu",seed=15,packed_channels=16).shape == (1,384,16)
