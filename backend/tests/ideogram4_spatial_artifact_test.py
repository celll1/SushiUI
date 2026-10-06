"""Real codec calibration and dual-DiT public artifact restoration on CPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.mark.parametrize("sharded", [False,True])
def test_shared_calibration_and_public_native_codec_round_trip(tmp_path,monkeypatch,sharded):
    from diffusers import AutoencoderKL,AutoencoderKLFlux2,FlowMatchEulerDiscreteScheduler
    from core.models.common.vae_source import resolve_vae_source
    from core.models.ideogram4.vae_alignment import initialize_swap,pair_metadata
    from core.training.arch.ideogram4 import Ideogram4ArchHandler
    from core.training.adapters.ideogram4_adapter import Ideogram4FullParameterAdapter
    from core.models.ideogram4.residual_unet import native_config,save_checkpoint
    from core.models.ideogram4.ideogram4_loader import load_ideogram4_single_file,_build_ideogram4_transformer_from_state
    from core.models.ideogram4.ideogram4_pipeline_ops import vae_encode,vae_decode,_prepare_ids
    from core.models.common.single_file_format import read_state_dict,strip_prefix
    from core.models.components.spatial_swap import read_declaration
    from core.training.vae_swap import swap_metadata,apply_latent_space
    from ideogram4_full_training_test import _trainer
    trainer=_trainer(True)
    trainer.is_ideogram4=True
    source=AutoencoderKLFlux2(in_channels=3,out_channels=3,latent_channels=2,
        block_out_channels=(4,)*4,down_block_types=("DownEncoderBlock2D",)*4,
        up_block_types=("UpDecoderBlock2D",)*4,norm_num_groups=1,sample_size=128)
    target=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
        block_out_channels=(4,)*5,down_block_types=("DownEncoderBlock2D",)*5,
        up_block_types=("UpDecoderBlock2D",)*5,norm_num_groups=1,sample_size=128,scaling_factor=1.)
    target.save_pretrained(tmp_path/"target")
    resolved=resolve_vae_source(f"file:{tmp_path/'target'}",arch="ideogram4")
    trainer.vae=source;trainer.text_encoder=None;trainer.text_encoder_2=None
    trainer.config.update(training_method="full_finetune",vae_swap_calibration_samples=2,
        vae_swap_calibration_resolution=128)
    trainer.device=torch.device("cpu");trainer.vae_dtype=trainer.weight_dtype=torch.float32
    trainer._check_stop_requested=lambda:None
    handler=Ideogram4ArchHandler(trainer);trainer.arch=handler
    trainer._main_model_module=lambda:trainer.transformer
    trainer.move_main_model_to_cpu=lambda:trainer.transformer.to("cpu")
    trainer.move_main_model_to_gpu=lambda:trainer.transformer.to("cpu")
    trainer.move_vae_to_cpu=lambda:trainer.vae.to("cpu")
    initialize_swap(trainer,resolved,handler=handler)
    assert trainer.wiring.latent_channels == 4 and trainer.wiring.vae_scale_factor == 16
    assert handler.pixel_align == 32
    assert trainer.transformer.vae_latent_bridge is not trainer.transformer_uncond.vae_latent_bridge
    with pytest.raises(ValueError,match="uncalibrated"):
        pair_metadata(trainer.transformer,trainer.transformer_uncond)
    paths=[]
    for index in range(2):
        path=tmp_path/f"synthetic{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0,256,(128,128,3),dtype=np.uint8)).save(path)
        paths.append({"image_path":str(path)})
    handler.calibrate_before_training(trainer,[SimpleNamespace(items=paths)],None)
    assert trainer.transformer.vae_latent_bridge.ready and trainer.transformer_uncond.vae_latent_bridge.ready
    assert not hasattr(trainer,"_ideogram_vae_calibration_source")
    companion=tmp_path/"companion"
    for name in ("transformer","text_encoder","tokenizer"):(companion/name).mkdir(parents=True)
    (companion/"transformer"/"config.json").write_text(json.dumps(native_config(trainer.transformer)),encoding="utf-8")
    FlowMatchEulerDiscreteScheduler().save_pretrained(companion/"scheduler")
    trainer.ideogram4_base_dir=str(companion)
    trainer.vae_identity=resolved
    adapter=Ideogram4FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    optimizer=torch.optim.AdamW(adapter.arch_param_groups())
    trainer.optimizer=optimizer
    with Image.open(paths[0]["image_path"]) as image:
        encoded=vae_encode(trainer.vae,image,128,128,"cpu",torch.float32)
    assert encoded.shape == (1,16,16)
    positions,segments,indicator=_prepare_ids([2],4,4,2,torch.device("cpu"))
    def predict(model):
        return model(hidden_states=torch.cat((encoded.new_zeros(1,2,16),encoded),dim=1),
            timestep=torch.tensor([.5]),encoder_hidden_states=torch.randn(1,18,12,generator=torch.Generator().manual_seed(19)),
            position_ids=positions,segment_ids=segments,indicator=indicator,return_dict=False)[0]
    before=predict(trainer.transformer)
    before.square().mean().backward();optimizer.step();optimizer.zero_grad()
    if sharded:
        _,bundled,metadata=swap_metadata(trainer)
        assert bundled
        metadata["component.base_dir"]=str(companion)
        path=save_checkpoint(trainer.transformer,trainer.transformer_uncond,tmp_path/"full.safetensors",
            metadata=metadata,vae=trainer.vae,max_shard_bytes=2048)
    else:path=adapter.write_checkpoint(1,0,tmp_path/"full.safetensors")
    state,metadata=read_state_dict(str(path))
    assert not any(key.startswith("vae.inner.") for key in state)
    monkeypatch.setattr("core.models.ideogram4.ideogram4_loader.load_ideogram4_text_encoder",lambda *a,**k:torch.nn.Linear(3,3))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",lambda *a,**k:object())
    restored=load_ideogram4_single_file(str(path),torch_dtype=torch.float32,load_unconditional=False)
    assert restored["unconditional_transformer"] is not None
    for name,model in (("transformer",trainer.transformer),("unconditional_transformer",trainer.transformer_uncond)):
        for key,value in model.state_dict().items():assert torch.equal(value,restored[name].state_dict()[key])
    for key,value in target.state_dict().items():assert torch.equal(value,restored["vae"].inner.state_dict()[key])
    torch.testing.assert_close(predict(trainer.transformer),predict(restored["transformer"]))
    assert vae_decode(restored["vae"],encoded,4,4).size == (128,128)
    trainer.transformer=restored["transformer"];trainer.transformer_uncond=restored["unconditional_transformer"]
    trainer.vae=restored["vae"]
    original=trainer.transformer.vae_latent_bridge
    apply_latent_space(trainer,restored["declared_vae"])
    assert trainer.transformer.vae_latent_bridge is original
    assert not hasattr(trainer,"_ideogram_vae_calibration_source")
    cond_state=strip_prefix(state,"transformer.")
    with pytest.raises(ValueError,match="alignment declaration"):
        _build_ideogram4_transformer_from_state(native_config(trainer.transformer),cond_state,torch.float32,"transformer")
    cond_state.pop("vae_latent_bridge.input.projection.weight")
    with pytest.raises(RuntimeError,match="Missing key"):
        _build_ideogram4_transformer_from_state(native_config(trainer.transformer),cond_state,torch.float32,"transformer",
            alignment=read_declaration(metadata))
    trainer.transformer_uncond.vae_latent_bridge.declaration["new_scale"]=8
    with pytest.raises(ValueError,match="declarations disagree"):
        pair_metadata(trainer.transformer,trainer.transformer_uncond)
