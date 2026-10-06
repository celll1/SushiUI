"""Target-codec geometry through real clip loading and training cache seams."""
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import pytest

from ltx2_vae_training_test import setup


def test_live_bucket_and_cached_clip_resume(tmp_path):
    from core.training.base_trainer import BaseTrainer
    from core.training.latent_cache import LatentCache, vae_cache_namespace
    from core.training.vae_swap import module_latent_hash
    from core.models.components.video_latent_adapter import NormalizedVideoVAE
    from temporal_bucketing_test import _write_video
    trainer,resolved = setup(tmp_path,2)
    trainer.arch.apply_vae_swap(trainer,resolved)
    trainer.log_prefix = '[CPU video cache]'
    trainer._temporal_spec = lambda:trainer.arch.temporal
    trainer.config['ltx2_clip_lengths'] = [2,3,4]
    path = _write_video(str(tmp_path/'synthetic.avi'),8,24.)
    item = dict(item_type='video',video_path=path,image_path=path,width=64,height=48,
                num_frames=8,fps=24.)
    datasets = [SimpleNamespace(items=[item])]
    assert BaseTrainer._annotate_video_items(trainer,datasets,[64]) == 1
    assert item['clip_length'] == 4
    assert item['bucket_width'] % trainer.arch.pixel_align == 0
    assert item['bucket_height'] % trainer.arch.pixel_align == 0
    dimensions = (item['bucket_width'],item['bucket_height'])
    BaseTrainer._annotate_video_items(trainer,datasets,[64])
    assert dimensions == (item['bucket_width'],item['bucket_height'])
    trainer._video_clip_window = MethodType(BaseTrainer._video_clip_window,trainer)
    trainer._clip_vae_tiling_policy = lambda:None
    trainer._clip_audio_prep_version = lambda:None
    trainer._clip_audio_seam = lambda _:None
    identity = module_latent_hash(trainer.vae)
    cache = LatentCache('synthetic',str(tmp_path/'cache'),namespace='ltx2',
                        vae_namespace=vae_cache_namespace(identity))
    first = BaseTrainer._load_or_encode_video_clip(trainer,item,cache)
    w,h = dimensions
    assert first.shape == (1,2,4,h//2,w//2)
    # A newly restored facade must address the same disk record.
    trainer.vae = NormalizedVideoVAE(trainer.vae.inner,trainer.vae.contract)
    assert module_latent_hash(trainer.vae) == identity
    restored_cache = LatentCache('synthetic',str(tmp_path/'cache'),namespace='ltx2',
        vae_namespace=vae_cache_namespace(module_latent_hash(trainer.vae)))
    with patch.object(trainer.arch,'vae_encode_clip',side_effect=AssertionError('cache missed')):
        second = BaseTrainer._load_or_encode_video_clip(trainer,item,restored_cache)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert item['_clip_audio_latent'] is None


def test_geometry_and_temporal_policy_separate_cache_namespaces(tmp_path):
    from core.models.components.video_latent_adapter import NormalizedVideoVAE,VideoCodecContract
    from core.training.vae_swap import module_latent_hash
    from core.models.common.vae_source import content_hash_for_state_dict,latent_space_hash
    trainer,resolved = setup(tmp_path,2)
    trainer.arch.apply_vae_swap(trainer,resolved)
    codec = trainer.vae
    native = codec.inner
    assert module_latent_hash(native) == latent_space_hash(content_hash_for_state_dict(native.state_dict()),native.config)
    hashes = {module_latent_hash(codec)}
    # Hash-only fixtures keep tensors identical while varying the declared contract.
    contracts = [replace(codec.contract,image=replace(codec.contract.image,spatial_scale=4))]
    image = replace(codec.contract.image,latent_rank=5)
    contracts += [VideoCodecContract(image,scale,'causal_first_frame_v1') for scale in (1,4,8)]
    for contract in contracts:
        hashes.add(module_latent_hash(NormalizedVideoVAE(native,contract)))
    assert len(hashes) == 5


def test_public_preflight_accepts_rgb_and_rejects_unsavable_configs(tmp_path):
    from core.training.vae_swap import preflight_vae_swap, check_inherited_bundling
    from api.arch_capabilities import training_feature_unsupported_reason, TRAINING_FEATURE_ADVISORY
    _,resolved = setup(tmp_path,2)
    config = dict(vae_swap_source=resolved.source)
    kwargs = dict(arch='ltx2',method='full_finetune',bundle_vae_explicit_false=False)
    assert preflight_vae_swap(config,**kwargs) == resolved.source
    assert preflight_vae_swap({'vae_swap_source':'registry:sdxl'},**kwargs) == 'registry:sdxl'
    assert training_feature_unsupported_reason('ltx2','vae_swap','full_finetune') is None
    assert TRAINING_FEATURE_ADVISORY['ltx2']['vae_swap']['level'] == 'experimental'
    for patch_config,message in (({'train_unet':False},'train_unet'),
                                 ({'vae_swap_new_channel_init':'zero'},'paired_spatial')):
        with pytest.raises(ValueError,match=message):
            preflight_vae_swap(dict(config,**patch_config),**kwargs)
    with pytest.raises(ValueError,match='bundled'):
        preflight_vae_swap(config,**dict(kwargs,bundle_vae_explicit_false=True))
    with pytest.raises(ValueError,match='Full Fine-tune'):
        preflight_vae_swap(config,**dict(kwargs,method='lora'))
    with patch('core.models.common.vae_source.load_declared_latent_io',return_value=replace(resolved,identity_native=False)):
        with pytest.raises(ValueError,match='bundled'):
            check_inherited_bundling('saved-ltx.safetensors','ltx2',True)
    from core.training.arch.ltx2 import Ltx2ArchHandler
    for channels in (dict(pixel_channels=4),dict(output_channels=4)):
        ok,reason = Ltx2ArchHandler(None).check_vae_compatibility(dict(resolved.facts(),**channels))
        assert not ok and 'RGB' in reason


def test_route_preflight_preserves_omitted_bundle_default():
    from api.routes import TrainingRunCreateRequest, _check_vae_swap_params
    from fastapi import HTTPException
    body = dict(training_method='full_finetune',base_model_path='native-ltx',
                vae_swap_source='registry:sdxl')
    with patch('core.training.training_config._detect_arch',return_value='ltx2'):
        request = TrainingRunCreateRequest(**body)
        assert 'bundle_vae' not in request.model_fields_set
        _check_vae_swap_params(request)
        with pytest.raises(HTTPException) as error:
            _check_vae_swap_params(TrainingRunCreateRequest(**body,bundle_vae=False))
        assert error.value.status_code == 400 and 'bundled' in error.value.detail
