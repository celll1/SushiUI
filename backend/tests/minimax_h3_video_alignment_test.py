"""Real small H3 packed AV decoder with independent 2x2 convolution oracles."""

import pytest
import torch
from torch.nn import functional as F

from core.models.components.latent_adapter import ImageCodecContract,SpatialAlignmentPair
from core.models.components.video_latent_adapter import VideoCodecContract
from core.models.minimax_h3.video_alignment import install_video_alignment,map_video_rows


@pytest.fixture(autouse=True)
def suppress_import_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *a,**k:(8,9))
    monkeypatch.setattr(torch.cuda,"_lazy_init",lambda *a,**k:None)
    monkeypatch.setattr(torch._C,"_cuda_init",lambda *a,**k:None)


def oracle(bridge,rows,grid,alpha=None):
    b = rows.shape[0]
    t,h,w = grid
    c = bridge.source_channels
    latent = rows.to(bridge.projection.weight.dtype).reshape(b,t,h//2,w//2,c,2,2)
    latent = latent.permute(0,1,4,2,5,3,6).reshape(b*t,c,h,w)
    value = F.conv2d(latent,bridge.projection.weight,padding=1)
    bias = bridge.projection.bias.view(1,-1,1,1)
    if alpha is None:
        value = value+bias
    else:
        coefficients = alpha.to(value.dtype).reshape(t,h//2,w//2).repeat_interleave(2,1).repeat_interleave(2,2)
        value = value+coefficients[None,:,None].expand(b,t,bridge.target_channels,h,w).reshape(b*t,bridge.target_channels,h,w)*bias
    value = value.reshape(b,t,bridge.target_channels,h//2,2,w//2,2).permute(0,1,3,5,2,4,6)
    return value.reshape(b,-1,4*bridge.target_channels)


def setup(new,rewired,dtype,contract=None):
    from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from core.models.minimax_h3.residual_unet import native_config,restore_topology
    from core.rewiring.spec import build_spec,OPERATORS
    from core.models.components.vae_alignment import SpatialLatentBridge
    from core.models.minimax_h3.h3_pipeline_ops import build_packed_layout
    from core.models.minimax_h3.loader import _dit_target_dtype
    torch.manual_seed(94)
    model = MiniMaxH3Transformer3DModel(num_attention_heads=2,attention_head_dim=16,hidden_size=16,
        num_layers=7,num_refiner_layers=1,ffn_dim=32,in_channels=2,audio_in_channels=6,patch_size=(1,2,2),
        text_dim=10,freq_dim=8,time_embed_hidden_dim=16,time_embed_dim=8,rope_freq_dim=2,adaln_curve_grid=33).to(dtype).eval()
    for name,value in list(model.named_parameters())+list(model.named_buffers()):
        value.data = value.data.to(_dit_target_dtype(name,dtype,True))
    with torch.no_grad(): model.adaln_t_table.copy_(torch.randn_like(model.adaln_t_table))
    before = {key:value.clone() for key,value in model.state_dict().items()}
    if rewired:
        graph = build_spec("minimax_h3",[{"name":"main","block_count":7}],[
            dict(id="outer",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS)],native_config(model.config))
        restore_topology(model,graph)
    image = ImageCodecContract(new,32,4,"shift_scale",1,3) if contract is None else contract.image
    pair = SpatialAlignmentPair(image.declaration(source="synthetic",source_identity="old",target_identity="new",old_channels=2,old_scale=16))
    incoming,outgoing = SpatialLatentBridge(new,2),SpatialLatentBridge(2,new)
    incoming.initialize(torch.randn(9*new,2)*.1,torch.randn(2)*.1)
    outgoing.initialize(torch.randn(18,new)*.1,torch.randn(new)*.1)
    pair.initialize_from_pair(incoming,outgoing)
    wrapper = install_video_alignment(model,pair,VideoCodecContract(image,1,"independent_frames_v1") if contract is None else contract)
    layout = build_packed_layout(num_text_tokens=3,num_audio_latents=2,num_latent_frames=3,
        latent_height=8,latent_width=12,patch_size=(1,2,2),keyframe_anchors=("first",))
    count = layout["video_indices"].numel()
    prefix = count-72
    times = torch.zeros(layout["sequence_length"],dtype=torch.long)
    times[layout["video_indices"][:prefix]] = 1
    args = dict(hidden_states=torch.randn(2,count,4*new,dtype=dtype,requires_grad=True),
        audio_hidden_states=torch.randn(2,4,6,dtype=dtype,requires_grad=True),
        encoder_hidden_states=torch.randn(2,3,10,dtype=dtype,requires_grad=True),
        timestep=torch.tensor([.3,.8]),timestep_indices=times,
        token_tags=layout["token_tags"],position_ids=layout["position_ids"],video_indices=layout["video_indices"],
        audio_indices=layout["audio_indices"],text_indices=layout["text_indices"],return_dict=False,
        residual_video_grid=dict(frames=3,height=4,width=6,target_indices=layout["video_indices"][-72:]),
        video_codec_grids=[dict(rows=torch.arange(prefix),frames=1,height=8,width=12),
                           dict(rows=torch.arange(prefix,count),frames=3,height=8,width=12)])
    return wrapper,pair,args,before


@pytest.mark.parametrize("new,rewired",[(1,False),(4,True)])
@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_real_packed_forward_matches_conv_oracle_and_preserves_audio_identity(new,rewired,dtype):
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    wrapper,pair,args,before = setup(new,rewired,dtype)
    internal = []
    alpha = args["timestep"][args["timestep_indices"][args["video_indices"]]]
    for group in args["video_codec_grids"]:
        rows = group["rows"]
        internal.append(oracle(pair.input,args["hidden_states"][:,rows],
            (group["frames"],group["height"],group["width"]),alpha[rows]))
    reference_args = {key:value for key,value in args.items() if key != "video_codec_grids"}
    reference_args["hidden_states"] = torch.cat(internal,1)
    baseline = MiniMaxH3BlockLoopWrapper(wrapper.transformer)(**reference_args)
    expected = torch.cat([oracle(pair.output,baseline[0][:,group["rows"]],
        (group["frames"],group["height"],group["width"])) for group in args["video_codec_grids"]],1)
    actual = wrapper(**args)
    torch.testing.assert_close(actual[0],expected,rtol=0,atol=0)
    torch.testing.assert_close(actual[1],baseline[1],rtol=0,atol=0)
    with torch.no_grad():
        named = wrapper(**dict(args,return_dict=True))
    torch.testing.assert_close(named.sample,actual[0],rtol=0,atol=0)
    torch.testing.assert_close(named.audio_sample,actual[1],rtol=0,atol=0)
    assert actual[0].shape == args["hidden_states"].shape
    for key,value in before.items():
        if not key.startswith(("proj_in.","proj_out.")):
            torch.testing.assert_close(wrapper.transformer.state_dict()[key],value,rtol=0,atol=0)
    loss = sum(value.float().square().mean() for value in actual)
    loss.backward()
    for p in (pair.input.projection.weight,pair.output.projection.weight,
              wrapper.transformer.transformer_blocks[0].attn.to_q.weight):
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
    if new > 2:
        assert wrapper.transformer.proj_out.weight.grad[8:].abs().sum() > 0
    assert args["hidden_states"].grad.isfinite().all()


def test_bad_video_segments_and_timesteps_refuse_before_native_blocks():
    wrapper,_,args,_ = setup(4,False,torch.float32)
    for groups in (None,[],args["video_codec_grids"][:1],
                   [dict(args["video_codec_grids"][0],rows=torch.arange(96))]):
        with pytest.raises(ValueError): wrapper(**dict(args,video_codec_grids=groups))
    with pytest.raises(ValueError,match="clean-fraction"):
        wrapper(**dict(args,timestep=torch.tensor([.3,1.1])))
    with pytest.raises(ValueError,match="partition"):
        wrapper(**dict(args,video_indices=args["video_indices"].roll(1).fill_(0)))


@pytest.mark.parametrize("dtype",[torch.float32,torch.bfloat16])
def test_per_patch_clean_fraction_matches_independent_oracle(dtype):
    from core.models.components.vae_alignment import SpatialLatentBridge
    bridge = SpatialLatentBridge(3,5).to(dtype)
    bridge.initialize(torch.randn(27,5),torch.randn(5))
    rows = torch.randn(2,24,12,dtype=dtype)
    fractions = torch.linspace(0,1,24)
    for velocity in (False,True):
        actual = map_video_rows(bridge,rows,grid=(2,4,12),patch=(1,2,2),
                                clean_fraction=fractions,velocity=velocity)
        expected = oracle(bridge,rows,(2,4,12),None if velocity else fractions)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_checkpointing_and_segment_order_preserve_output_and_gradients():
    reference,_,args,_ = setup(4,True,torch.float32)
    checked,_,checked_args,_ = setup(4,True,torch.float32)
    checked.transformer.enable_gradient_checkpointing()
    checked_args["video_codec_grids"].reverse()
    expected,actual = reference(**args),checked(**checked_args)
    for left,right in zip(expected,actual):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    for output in (expected,actual):
        sum(value.square().mean() for value in output).backward()
    for (name,left),(other,right) in zip(reference.named_parameters(),checked.named_parameters()):
        assert name == other
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad,right.grad,rtol=1e-5,atol=1e-7)


def test_codec_contract_participates_in_unet_identity_and_refuses_malformed_fields():
    from core.models.minimax_h3.residual_unet import native_config,validate_topology_identity,VIDEO_CODEC_KEY
    wrapper,_,_,_ = setup(4,True,torch.float32)
    inner = wrapper.transformer
    config = dict(inner.config)
    assert native_config(config)[VIDEO_CODEC_KEY]["image"]["spatial_scale"] == 32
    changed = dict(config[VIDEO_CODEC_KEY])
    changed["image"] = dict(changed["image"],spatial_scale=16)
    with pytest.raises(ValueError):
        validate_topology_identity(inner._residual_unet_spec,dict(config,**{VIDEO_CODEC_KEY:changed}))
    for bad in ({},None,dict(changed["image"],extra=1)):
        with pytest.raises(ValueError,match="malformed"):
            native_config(dict(config,**{VIDEO_CODEC_KEY:dict(changed,image=bad)}))


def test_controller_install_refuses_before_rebinding_parameters():
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper
    wrapper,pair,_,_ = setup(4,False,torch.float32)
    inner = wrapper.transformer
    del inner._video_latent_alignment
    held = tuple(inner.parameters())
    base = MiniMaxH3BlockLoopWrapper(inner)
    base._block_offloader = object()
    with pytest.raises(ValueError,match="runtime controllers"):
        install_video_alignment(base,pair,inner._video_codec_contract)
    assert all(left is right for left,right in zip(held,inner.parameters()))
