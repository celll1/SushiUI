"""Synthetic native-codec calibration and public Flux2 artifact restoration."""
from types import SimpleNamespace
from unittest.mock import patch
import json

import pytest
import torch


@pytest.fixture(autouse=True)
def no_cuda():
    with patch.object(torch.cuda,"get_device_capability",return_value=(8,9)), \
         patch.object(torch.cuda,"_lazy_init",return_value=None), \
         patch.object(torch._C,"_cuda_init",return_value=None):
        yield


@pytest.mark.parametrize("sharded",[False,True])
@pytest.mark.parametrize("old_c,new_c",[(4,8),(8,4)])
def test_native_calibration_public_loader_and_resume(tmp_path,sharded,old_c,new_c):
    from diffusers import AutoencoderKL,AutoencoderKLFlux2,Flux2Transformer2DModel
    from PIL import Image
    import numpy as np
    from core.models.common.vae_source import resolve_vae_source
    from core.models.components.wiring import FLUX2_WIRING
    from core.training.arch.flux2 import Flux2ArchHandler
    from core.training.vae_swap import apply_latent_space,validate_latent_io
    from core.training.adapters.flux2_adapter import FLUX2FullParameterAdapter
    from core.training.base_trainer import BaseTrainer
    from core.model_loader import ModelLoader
    from core.models.flux2.residual_unet import native_config,restore_topology,generation_token_stride
    from core.rewiring.spec import build_spec,OPERATORS
    old=AutoencoderKLFlux2(latent_channels=old_c,block_out_channels=(4,)*4,norm_num_groups=4,layers_per_block=1).eval()
    target=AutoencoderKL(latent_channels=new_c,block_out_channels=(4,)*5,norm_num_groups=4,
        down_block_types=("DownEncoderBlock2D",)*5,up_block_types=("UpDecoderBlock2D",)*5,
        scaling_factor=.7,shift_factor=.2)
    directory=tmp_path/"target"; target.save_pretrained(directory)
    resolved=resolve_vae_source(f"file:{directory}",arch="flux2")
    value=Flux2Transformer2DModel(num_layers=3,num_single_layers=3,in_channels=old_c*4,
        num_attention_heads=2,attention_head_dim=8,axes_dims_rope=(2,2,2,2),
        joint_attention_dim=12,timestep_guidance_channels=8,mlp_ratio=1,guidance_embeds=True)
    if sharded:
        restore_topology(value,build_spec("flux2",[{"name":"double","block_count":3},{"name":"single","block_count":3}],
            [dict(id="s",stream="single",parent=None,start_block=1,end_block=2,down_factor=[2,2],**OPERATORS)],native_config(value)))
    trainer=SimpleNamespace(transformer=value,transformer_original=value,vae=old,
        config={"training_method":"full_finetune","vae_swap_source":resolved.source,
            "vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        weight_dtype=torch.float32,vae_dtype=torch.float32,device="cpu",train_unet=True,
        train_text_encoder=False,text_encoder=None,text_encoder_2=None,
        wiring=FLUX2_WIRING.replace(latent_channels=old_c),bundle_vae=True,is_flux2=True,
        _check_stop_requested=lambda:None,log_prefix="[SyntheticFlux]")
    trainer.arch=Flux2ArchHandler(trainer)
    trainer._main_model_module=lambda:value
    trainer.move_main_model_to_cpu=lambda:value.to("cpu")
    trainer.move_main_model_to_gpu=lambda:value.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    trainer.arch.apply_vae_swap(trainer,resolved)
    assert trainer.wiring.vae_scale_factor==16 and trainer.wiring.vae_norm=="identity"
    assert BaseTrainer._arch_pixel_align(trainer)==(64 if sharded else 32)
    images=[]
    for i in range(2):
        pixels=np.random.default_rng(i).integers(0,256,(128,128,3),dtype=np.uint8)
        path=tmp_path/f"synthetic{i}.png"; Image.fromarray(pixels).save(path)
        images.append({"image_path":str(path)})
    trainer.arch.calibrate_before_training(trainer,[SimpleNamespace(items=images)],None)
    assert value.vae_latent_bridge.ready and validate_latent_io(trainer)==[]
    assert generation_token_stride(trainer.vae)==32
    from core.training.ops.flux2_ops import vae_encode
    trainer._flux2_patchify_latents_for_training=lambda latent:BaseTrainer._flux2_patchify_latents_for_training(trainer,latent)
    latent=vae_encode(trainer,torch.randn(1,3,128,192))
    assert latent.shape==(1,new_c*4,4,6)
    raw=BaseTrainer._flux2_unpatchify_latents(trainer,latent)
    assert trainer.vae.decode(raw).sample.shape==(1,3,128,192)
    path=tmp_path/"full.safetensors"
    if sharded:
        from core.models.common.single_file_format import save_single_file_state
        with patch("core.models.common.single_file_format.save_single_file_state",
                side_effect=lambda state,metadata,path:save_single_file_state(state,metadata,path,max_shard_bytes=32768)):
            path=FLUX2FullParameterAdapter(trainer).write_checkpoint(7,0,path)
    else:path=FLUX2FullParameterAdapter(trainer).write_checkpoint(7,0,path)
    base=tmp_path/"base"/"transformer"; base.mkdir(parents=True)
    (base/"config.json").write_text(json.dumps(native_config(value)),encoding="utf-8")
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch("huggingface_hub.snapshot_download",return_value=str(base.parent)))
        stack.enter_context(patch("transformers.Qwen3ForCausalLM.from_pretrained",return_value=torch.nn.Linear(2,2)))
        stack.enter_context(patch("transformers.Qwen2TokenizerFast.from_pretrained",return_value=object()))
        stack.enter_context(patch("diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained",return_value=object()))
        components=ModelLoader.load_flux2_from_safetensors(str(path),device="cpu",torch_dtype=torch.float32,base_model_repo="synthetic-flux")
    restored=components["transformer"]
    assert restored.vae_latent_bridge.ready and components["vae"].spatial_scale_factor==16
    assert components["vae"].inner.config.scaling_factor==.7
    trainer.transformer=trainer.transformer_original=restored
    trainer.vae=components["vae"]
    with patch("core.training.vae_swap.apply_configured_vae_swap",side_effect=AssertionError("must not recalibrate")):
        apply_latent_space(trainer,components["declared_vae"])
    torch.testing.assert_close(restored.vae_latent_bridge.input.projection.weight,value.vae_latent_bridge.input.projection.weight)
    assert restored.x_embedder.in_features==max(old_c,new_c)*4
    from core.models.common.single_file_format import read_state_dict
    from safetensors.torch import save_file
    state,metadata=read_state_dict(str(path))
    del state["model.diffusion_model.vae_latent_bridge.input.projection.weight"]
    broken=tmp_path/"missing_bridge.safetensors"
    save_file(state,str(broken),metadata={key:str(item) for key,item in metadata.items()})
    with patch("huggingface_hub.snapshot_download",return_value=str(base.parent)), \
         pytest.raises(ValueError,match="does not match"):
        ModelLoader.load_flux2_from_safetensors(str(broken),device="cpu",torch_dtype=torch.float32,base_model_repo="synthetic-flux")
    assert not torch.cuda.is_initialized()
