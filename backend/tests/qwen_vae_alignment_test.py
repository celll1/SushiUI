"""Synthetic Qwen swap execution and portable full-checkpoint restoration."""
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch

torch.cuda.get_device_capability=lambda *a,**k:(8,9)
torch.cuda._lazy_init=lambda *a,**k:None
torch._C._cuda_init=lambda *a,**k:None
torch.set_num_threads(1)

from core.models.qwen_image_21.vendor.transformer import QwenImage21Transformer2DModel
from core.models.qwen_image_21.vae_alignment import install,map_input
from core.models.components.latent_adapter import NormalizedFrameVAE
from core.training.adapters.qwen_image_21_adapter import QwenImage21FullParameterAdapter
from core.models.qwen_image_21.residual_unet import native_config,restore_topology
from core.rewiring.spec import build_spec,OPERATORS
from latent_adapter_contract_test import identity


def model():
    return QwenImage21Transformer2DModel(in_channels=8,out_channels=8,num_layers=7,
        num_attention_heads=1,attention_head_dim=12,axes_dims_rope=(4,4,4),context_in_dim=12,mlp_ratio=1)


def declaration():
    return dict(version=1,method="paired_spatial",token_policy="target_native",source="file:synthetic",
        source_identity="source",target_identity="target",old_channels=8,new_channels=3,
        old_scale=16,new_scale=2,input_channels=3,ndim=4,norm="shift_scale",norm_pack=1)


def test_qwen_native_codec_is_registered_for_the_vae_source_api():
    from core.models.component_registry import _WIRING_BY_ARCH
    from core.models.common.vae_source import arch_native_vae
    from api.generation_overrides import VAE_CLASS_BY_ARCH
    assert "qwen_image_21" in _WIRING_BY_ARCH
    facts = arch_native_vae("qwen_image_21")
    assert (facts["latent_channels"],facts["scale_factor"],facts["ndim"],facts["scale_temporal"]) == (64,16,5,8)
    assert VAE_CLASS_BY_ARCH["qwen_image_21"] == "AutoencoderKLQwenImage21"


def test_component_load_resolves_swapped_wiring_before_arch_handler_exists(monkeypatch):
    from diffusers import AutoencoderKL
    from core.training.ops.qwen_image_21_ops import load_components
    import core.models.qwen_image_21.loader as loader
    m = initialize(model())
    codec = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=3,
        block_out_channels=(8, 8), down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2, norm_num_groups=4)
    wrapped = NormalizedFrameVAE(codec, declaration())
    components = dict(transformer=m, vae=wrapped, text_encoder=torch.nn.Linear(2, 2),
        processor=SimpleNamespace(tokenizer=object()), scheduler=object(),
        transformer_variant="bf16", text_encoder_variant="bf16")
    monkeypatch.setattr(loader, "load_qwen_image_21_components", lambda *a, **kw: components)
    monkeypatch.setattr(loader, "build_pipeline", lambda *a, **kw: object())
    trainer = SimpleNamespace(config={}, model_path="synthetic", weight_dtype=torch.float32,
        vae_dtype=torch.float32, training_dtype=torch.bfloat16, gradient_checkpointing=False,
        device=torch.device("cpu"), log_prefix="[CPU]", attention_backend="native",
        _resolve_training_backend=lambda value: value)
    load_components(trainer)
    assert not hasattr(trainer, "arch")
    assert trainer.wiring.latent_channels == 3 and trainer.wiring.vae_scale_factor == 2
    assert trainer.vae_latent_channels == 3


def initialize(m):
    install(m,declaration())
    identity(m.vae_latent_bridge.input)
    identity(m.vae_latent_bridge.output)
    return m


def forward(m,channels=3):
    tokens = torch.randn(1,8*12,channels,requires_grad=True)
    mask = torch.cat((torch.zeros(1,3,dtype=torch.bool),torch.ones(1,8*12//4,dtype=torch.bool)),dim=1)
    result = m(tokens,torch.randn(1,3,12),torch.tensor([.4]),[[(1,8,12)]],mask).sample
    return tokens,result


@pytest.mark.parametrize("rewired",[False,True])
def test_qwen_bridge_keeps_native_target_grid_and_trains_with_unet(rewired):
    m = initialize(model())
    if rewired:
        regions = [dict(id="coarse",stream="main",parent=None,start_block=1,end_block=6,down_factor=[2,2],**OPERATORS),
                   dict(id="deep",stream="main",parent="coarse",start_block=2,end_block=5,down_factor=[2,2],**OPERATORS)]
        restore_topology(m,build_spec("qwen_image_21",[{"name":"main","block_count":7}],regions,native_config(dict(m.config))))
    tokens,result = forward(m)
    assert result.shape == (1,3+8*12,3)
    result[:,-96:].square().mean().backward()
    assert torch.isfinite(tokens.grad).all()
    assert m.vae_latent_bridge.input.projection.weight.grad.abs().sum() > 0
    assert m.vae_latent_bridge.output.projection.weight.grad.abs().sum() > 0
    assert not torch.cuda.is_initialized()


def test_reference_images_use_clean_bias_and_separate_spatial_boundaries():
    m = initialize(model())
    values = torch.randn(1,8+16,3)
    mapped = map_input(m,values,torch.tensor([.4]),[[(1,2,4),(1,4,4)]])
    torch.testing.assert_close(mapped[:,:8,:3],values[:,:8]+.3)
    torch.testing.assert_close(mapped[:,8:,:3],values[:,8:]+.18)


def test_global_adapter_uses_internal_domain_and_preserves_coordinate_columns():
    from core.models.qwen_image_21.vae_alignment import map_global_grid,resize_global_payload
    m = initialize(model())
    grid = torch.randn(2,8,12,3,requires_grad=True)
    mapped = map_global_grid(m,grid,torch.tensor([.2,.7]))
    assert mapped.shape == (2,8,12,8)
    torch.testing.assert_close(mapped[:,:,:,:3],grid+torch.tensor([.24,.09])[:,None,None,None])
    payload = dict(state={key:torch.randn(4,10) for key in ("key.weight","value.weight","query.weight")})
    before = {k:v.clone() for k,v in payload["state"].items()}
    m._qwen_partition_global_payload = payload
    resize_global_payload(m,8,16)
    for key,value in before.items():
        torch.testing.assert_close(payload["state"][key][:,:8],value[:,:8])
        torch.testing.assert_close(payload["state"][key][:,-2:],value[:,-2:])
        assert not payload["state"][key][:,8:-2].count_nonzero()


def test_calibration_is_one_time_and_restores_component_residency(tmp_path):
    from PIL import Image
    import numpy as np
    from diffusers.configuration_utils import FrozenDict
    from core.training.arch.qwen_image_21 import QwenImage21ArchHandler
    from core.models.qwen_image_21.vae_alignment import calibrate
    class Codec(torch.nn.Module):
        def __init__(self,channels,scale):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.channels,self.scale = channels,scale
            self.config = FrozenDict(scaling_factor=1.,shift_factor=0.,latents_mean=[0.]*channels,latents_std=[1.]*channels)
        def encode(self,pixels):
            value = torch.nn.functional.avg_pool2d(pixels.squeeze(2),self.scale)
            value = value.repeat(1,self.channels,1,1)[:,:self.channels].unsqueeze(2)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))
    paths = []
    for i in range(2):
        path = tmp_path/f"synthetic_{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,4),dtype=np.uint8)).save(path)
        paths.append(dict(image_path=str(path)))
    m = model()
    install(m,declaration())
    target_decl = dict(declaration(),ndim=5)
    target = NormalizedFrameVAE(Codec(3,2),target_decl)
    events = []
    trainer = SimpleNamespace(transformer=m,text_encoder=torch.nn.Linear(2,2),vae=target,
        vae_dtype=torch.float32,device=torch.device("cpu"),
        config=dict(vae_swap_calibration_samples=2,vae_swap_calibration_resolution=128),
        arch=QwenImage21ArchHandler(None),_qwen_vae_calibration_source=Codec(8,16),
        _main_model_module=lambda:m,move_main_model_to_cpu=lambda:m.to("cpu"),move_main_model_to_gpu=lambda:None,
        move_vae_to_cpu=lambda:target.to("cpu"),
        _check_stop_requested=lambda:None,_vae_alignment_progress_callback=lambda **event:events.append(event))
    calibrate(trainer,[SimpleNamespace(items=paths)])
    assert m.vae_latent_bridge.ready and not hasattr(trainer,"_qwen_vae_calibration_source")
    assert next(trainer.vae.parameters()).device.type == "cpu"
    assert next(trainer.text_encoder.parameters()).device.type == "cpu"
    assert events[-1] == dict(phase="latent_cache",step=4,total=4)
    before = len(events)
    calibrate(trainer,[SimpleNamespace(items=paths)])
    assert len(events) == before


@pytest.mark.parametrize("sharded",[False,True])
def test_full_checkpoint_bundles_and_restores_target_codec_and_bridge(tmp_path,monkeypatch,sharded):
    from diffusers import AutoencoderKL
    from core.models.qwen_image_21.loader import load_qwen_image_21_components
    import core.models.qwen_image_21.loader as loader
    import core.training.adapters.qwen_image_21_adapter as adapter
    from core.models.common.single_file_format import read_state_dict
    if sharded: monkeypatch.setattr(adapter,"DEFAULT_EXPORT_SHARD_BYTES",4096)
    m = initialize(model())
    codec = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=3,block_out_channels=(8,8),
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D"),norm_num_groups=4,sample_size=16)
    wrapped = NormalizedFrameVAE(codec,declaration())
    companion = tmp_path/"companion"
    companion.mkdir()
    (companion/"manifest.json").write_text("{}")
    trainer = SimpleNamespace(transformer=m,vae=wrapped,text_encoder=torch.nn.Linear(1,1),
        train_text_encoder=False,model_path=str(companion),qwen_image_21_companion_path=str(companion),config={})
    path = QwenImage21FullParameterAdapter(trainer).write_checkpoint(3,1,tmp_path/"test.safetensors")
    state,metadata = read_state_dict(str(path))
    assert any(key.startswith("vae.") for key in state)
    assert metadata["component"] == "training_bundle" and metadata["train_text_encoder"] == "0"
    monkeypatch.setattr(loader,"_artifact_components",lambda *args,**kwargs:dict(
        transformer=None,text_encoder=torch.nn.Linear(1,1),vae=codec,processor=None,scheduler=None,
        transformer_variant="bf16",text_encoder_variant="bf16"))
    restored = load_qwen_image_21_components(str(path),torch_dtype=torch.float32)
    assert restored["transformer"].vae_latent_bridge.ready
    assert restored["text_encoder"] is not None
    assert restored["latent_channels"] == 3 and restored["vae_scale_factor"] == 2
    for key,value in m.state_dict().items():
        torch.testing.assert_close(restored["transformer"].state_dict()[key],value)
    rgba = torch.randn(1,4,1,16,16)
    torch.testing.assert_close(restored["vae"].encode(rgba).latent_dist.mode(),wrapped.encode(rgba).latent_dist.mode())
