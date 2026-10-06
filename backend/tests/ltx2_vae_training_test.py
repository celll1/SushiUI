"""CPU LTX setup/calibration/resume with native tiny codecs and synthetic media."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class Trainer(SimpleNamespace):
    def _main_model_module(self):
        return self.transformer

    def move_main_model_to_cpu(self):
        self.events.append('park')
        self.transformer.to('cpu')

    def move_main_model_to_gpu(self):
        self.events.append('restore')
        self.transformer.to(self.device)

    def move_vae_to_cpu(self):
        self.vae.to('cpu')

    def _check_stop_requested(self):
        self.checks += 1
        if self.stop_at == self.checks:
            raise RuntimeError('stop requested')


def setup(tmp_path,new,scale=2):
    from diffusers import AutoencoderKL, AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel, LTX2Pipeline
    from core.models.ltx2.residual_unet import native_config,wrap_transformer
    from core.models.common.vae_source import resolve_vae_source
    from core.rewiring.spec import build_spec,OPERATORS
    from core.training.arch.ltx2 import Ltx2ArchHandler
    model = LTX2VideoTransformer3DModel(in_channels=8,out_channels=8,
        num_attention_heads=2,attention_head_dim=16,cross_attention_dim=32,
        audio_in_channels=4,audio_out_channels=4,audio_num_attention_heads=2,
        audio_attention_head_dim=8,audio_cross_attention_dim=16,num_layers=4,
        use_prompt_embeddings=False,cross_attn_mod=True,audio_cross_attn_mod=True,
        gated_attn=True,audio_gated_attn=True)
    graph = build_spec('ltx2',[dict(name='main',block_count=4)],[
        dict(id='region',stream='main',parent=None,start_block=1,end_block=3,down_factor=[2,2],**OPERATORS)],
        native_config(model.config))
    source = AutoencoderKLLTX2Video(latent_channels=8,block_out_channels=(16,16,16),
        down_block_types=('LTX2VideoDownBlock3D',)*3,layers_per_block=(1,1,1,1),
        decoder_block_out_channels=(16,16,16),decoder_layers_per_block=(1,1,1,1),
        spatio_temporal_scaling=(True,)*3,decoder_spatio_temporal_scaling=(True,)*3,
        downsample_type=('spatiotemporal',)*3,upsample_type=('spatiotemporal',)*3,
        upsample_factor=(1,)*3,upsample_residual=(False,)*3,decoder_inject_noise=(False,)*4,
        patch_size=4,spatial_compression_ratio=32,temporal_compression_ratio=8)
    source.eval().requires_grad_(False)
    pipeline = LTX2Pipeline(scheduler=None,vae=source,audio_vae=None,text_encoder=None,
        tokenizer=None,connectors=None,transformer=model,vocoder=None)
    trainer = Trainer(transformer=wrap_transformer(model,graph),vae=source,ltx2_pipeline=pipeline,
        config={'vae_swap_new_channel_init':'auto','vae_swap_calibration_samples':2,'vae_swap_calibration_resolution':128},
        vae_dtype=torch.float32,device=torch.device('cpu'),events=[],checks=0,stop_at=None,
        text_encoder=None,connectors=None,optimizer=None,is_ltx2=True)
    trainer.transformer_original = model
    from copy import deepcopy
    trainer.config.update(dit_rewiring_mode='configure', dit_rewiring_spec=deepcopy(graph))
    trainer.arch = Ltx2ArchHandler(trainer)
    depth = scale.bit_length()
    widths = (4,8) if scale == 2 else (4,)*depth
    codec = AutoencoderKL(in_channels=3,out_channels=3,latent_channels=new,block_out_channels=widths,
        down_block_types=('DownEncoderBlock2D',)*depth,up_block_types=('UpDecoderBlock2D',)*depth,
        norm_num_groups=1,scaling_factor=2.,shift_factor=.25)
    codec.save_pretrained(tmp_path/'codec')
    resolved = resolve_vae_source('file:'+str(tmp_path/'codec'),arch='ltx2',download=False,load_weights=True)
    return trainer,resolved


def images(tmp_path):
    from PIL import Image
    import numpy as np
    items = []
    for i in range(2):
        path = tmp_path/f'synthetic_{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(32,48,3),dtype=np.uint8)).save(path)
        items.append({'image_path':str(path)})
    return [SimpleNamespace(items=items)]


def test_stronger_compression_encodes_teacher_on_target_spatial_grid(tmp_path):
    from core.models.ltx2.vae_training import PENDING
    trainer, resolved = setup(tmp_path, 10, scale=64)
    trainer.arch.apply_vae_swap(trainer, resolved)
    teacher = getattr(trainer, PENDING)[0]
    source_shapes, target_shapes = [], []
    source_encode, target_encode = teacher.encode, trainer.vae.encode
    def source_record(value, *args, **kwargs):
        source_shapes.append(tuple(value.shape))
        return source_encode(value, *args, **kwargs)
    def target_record(value, *args, **kwargs):
        target_shapes.append(tuple(value.shape))
        return target_encode(value, *args, **kwargs)
    teacher.encode, trainer.vae.encode = source_record, target_record
    trainer.arch.calibrate_before_training(trainer, images(tmp_path), None)
    assert source_shapes == [(1,3,1,64,64)]*2
    assert target_shapes == [(1,3,1,128,128)]*2
    assert trainer.transformer_original._video_latent_alignment.ready
    assert not hasattr(trainer, PENDING)


@pytest.mark.parametrize('new',[2,10])
def test_setup_calibration_and_real_training_step(tmp_path,new):
    from core.models.ltx2.vae_training import PENDING
    from core.training.ops.ltx2_ops import vae_encode,train_step
    from core.training.arch.ltx2 import Ltx2ArchHandler
    trainer,resolved = setup(tmp_path,new)
    from copy import deepcopy
    original_config = deepcopy(trainer.config)
    report = trainer.arch.apply_vae_swap(trainer,resolved)
    from core.models.ltx2.residual_unet import configure_training_topology
    configure_training_topology(trainer.transformer, trainer.config)
    configure_training_topology(trainer.transformer, original_config, resuming=True)
    assert report.new_elements > 0
    assert trainer.wiring.latent_channels == new and trainer.wiring.vae_norm == 'identity'
    assert trainer.arch.pixel_align == 4
    assert trainer.arch.temporal.frame_multiple == 1
    assert trainer.arch.temporal.latent_frames(9) == 9
    assert Ltx2ArchHandler(trainer).temporal.frame_multiple == 1
    pair = trainer.transformer_original._video_latent_alignment
    assert not pair.ready
    progress = []
    trainer._vae_alignment_progress_callback = lambda **kw:progress.append(kw)
    rng = torch.random.get_rng_state().clone()
    trainer.arch.calibrate_before_training(trainer,images(tmp_path),None)
    assert pair.ready and not hasattr(trainer,PENDING)
    assert trainer.events == ['park','restore']
    assert torch.equal(rng,torch.random.get_rng_state())
    assert [p['step'] for p in progress] == [1,2,3,4]
    pixels = torch.randn(1,3,8,12)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        expected = trainer.vae.encode(pixels[:,:,None]).latent_dist.sample()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        actual = vae_encode(trainer,pixels)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    trainer.transformer.requires_grad_(True).train()
    trainer.training_dtype=torch.float32
    trainer.mixed_precision=False
    trainer.reconstruction_loss_weight=.25
    trainer.ltx2_audio_in_channels=4
    trainer.ltx2_audio_latents_per_second=24.
    trainer.ltx2_use_cross_timestep=True
    trainer.timestep_sampler=None
    observed=[]
    hook=trainer.transformer.register_forward_pre_hook(
        lambda module,args,kwargs:observed.append(kwargs['audio_hidden_states'].shape[1]),with_kwargs=True)
    try:
        loss,_,_=train_step(trainer,torch.randn(1,new,3,4,6),torch.randn(1,4,32),
            dict(audio_text_embedding=torch.randn(1,4,16),mask=torch.ones(1,4),fps=torch.tensor([24.])),
            timesteps=torch.tensor([.3]))
    finally:
        hook.remove()
    assert observed == [3]
    assert loss.isfinite()
    loss.backward()
    assert pair.input.projection.weight.grad.abs().sum() > 0
    assert pair.output.projection.weight.grad.abs().sum() > 0
    trainer.arch.apply_vae_swap(trainer,resolved,module=trainer.vae)
    assert not hasattr(trainer,PENDING)
    from core.training.vae_swap import validate_latent_io, latent_alignment_pair
    from core.training.ops.ltx2_ops import setup_wrapper
    assert latent_alignment_pair(trainer.transformer) is pair
    assert validate_latent_io(trainer) == []
    wrapper = trainer.transformer
    del trainer.ltx2_block_loop_wrapper
    setup_wrapper(trainer)
    assert trainer.transformer is trainer.ltx2_block_loop_wrapper is wrapper
    trainer.transformer = trainer.transformer_original
    trainer.arch.apply_vae_swap(trainer,resolved,module=trainer.vae)
    assert latent_alignment_pair(trainer.transformer) is pair
    assert trainer.ltx2_pipeline.transformer is trainer.transformer
    with pytest.raises(ValueError,match='fresh base'):
        trainer.arch.apply_vae_swap(trainer,replace(resolved,content_hash='changed'))


def test_cancellation_keeps_pending_and_retry_calibrates_clips(tmp_path):
    from core.models.ltx2.vae_training import PENDING
    trainer,resolved = setup(tmp_path,2)
    trainer.arch.apply_vae_swap(trainer,resolved)
    data = [SimpleNamespace(items=[dict(item_type='video',video_path='synthetic_clip')])]
    seen=[]
    def clip(path,**kwargs):
        seen.append(kwargs['clip_length'])
        return torch.randn(kwargs['clip_length'],3,kwargs['target_h'],kwargs['target_w'],
                           generator=torch.Generator().manual_seed(8))
    trainer.stop_at=2
    with patch('core.training.video_loader.load_clip',side_effect=clip):
        with pytest.raises(RuntimeError,match='stop requested'):
            trainer.arch.calibrate_before_training(trainer,data,None)
        assert hasattr(trainer,PENDING)
        assert not trainer.transformer_original._video_latent_alignment.ready
        trainer.stop_at=None
        trainer.arch.calibrate_before_training(trainer,data,None)
    assert seen == [9,9,9]
    assert not hasattr(trainer,PENDING)
    assert trainer.transformer_original._video_latent_alignment.ready
    assert trainer.events == ['park','restore','park','restore']


def test_unknown_video_policy_is_not_inferred():
    from core.training.arch.ltx2 import Ltx2ArchHandler
    facts = dict(ndim=5,scale_factor=32,scale_temporal=8,vae_class='UnknownVideoVAE')
    compatible,reason = Ltx2ArchHandler(None).check_vae_compatibility(facts)
    assert not compatible and 'temporal policy' in reason


def test_loader_dispatch_installs_swap_before_freeze(tmp_path):
    from core.training.ops.ltx2_ops import load_components
    from core.models.ltx2.vae_training import PENDING
    from core.training.arch.ltx2 import Ltx2ArchHandler
    trainer,resolved = setup(tmp_path,2)
    trainer.config.update(training_method='full_finetune',vae_swap_source=resolved.source,
                          dit_rewiring_spec=trainer.transformer_original._residual_unet_spec)
    trainer.weight_dtype = trainer.training_dtype = torch.float32
    trainer.log_prefix = '[CPU test]'
    trainer.model_path = str(tmp_path/'native')
    trainer.gradient_checkpointing = False
    trainer.blocks_to_swap = 0
    trainer.attention_backend = 'native'
    trainer._resolve_training_backend = lambda value:value
    components = dict(pipeline=trainer.ltx2_pipeline,transformer=trainer.transformer_original,
        vae=trainer.vae,text_encoder=torch.nn.Linear(2,2),connectors=torch.nn.Linear(2,2),
        tokenizer=None,scheduler=SimpleNamespace())
    with patch('core.model_loader.ModelLoader.load_ltx2_from_path',return_value=components):
        load_components(trainer)
    assert hasattr(trainer,PENDING)
    assert trainer.vae_identity.latent_hash == resolved.latent_hash
    assert trainer.ltx2_pipeline.vae is trainer.vae
    assert trainer.ltx2_pipeline.transformer is trainer.transformer
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert Ltx2ArchHandler(trainer).resolve_wiring(trainer).latent_channels == 2
