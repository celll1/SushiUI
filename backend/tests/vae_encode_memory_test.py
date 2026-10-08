from types import SimpleNamespace

import pytest
import torch

from api.param_defaults import QWEN21_TRAINING_ENCODE_DEFAULTS
from core.training.vae_encode_memory import bounded_image_encode, bounded_debug_decode
from core.models.qwen_image_21.vendor.autoencoder import AutoencoderKLQwenImage21


class Codec(AutoencoderKLQwenImage21):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.use_tiling = False
        for name in QWEN21_TRAINING_ENCODE_DEFAULTS:
            setattr(self, name, 96)
        self.cleared = 0

    def enable_tiling(self, **settings):
        self.use_tiling = True
        for name, value in settings.items():
            setattr(self, name, value)

    def clear_cache(self):
        self.cleared += 1


def test_encode_tiling_restores_decoder_policy_on_success_and_error():
    codec, owner = Codec(), SimpleNamespace()
    pixels = torch.zeros(1, 4, 32, 48)
    for fail in (False, True):
        try:
            with bounded_image_encode(owner, codec, pixels):
                assert codec.use_tiling
                assert codec.tile_sample_min_width == QWEN21_TRAINING_ENCODE_DEFAULTS['tile_sample_min_width']
                if fail:
                    raise RuntimeError('encode failed')
        except RuntimeError:
            assert fail
        assert not codec.use_tiling and codec.tile_sample_min_width == 96
    assert codec.cleared == 2


def test_video_and_other_codecs_are_unchanged():
    codec, owner = Codec(), SimpleNamespace()
    with bounded_image_encode(owner, codec, torch.zeros(1, 4, 2, 8, 8)):
        assert not codec.use_tiling
    with bounded_image_encode(owner, torch.nn.Identity(), torch.zeros(1, 4, 8, 8)):
        pass
    assert codec.cleared == 0


def test_debug_decode_tiles_restore_the_original_policy():
    from api.param_defaults import QWEN21_DEBUG_DECODE_DEFAULTS
    codec = Codec()
    with pytest.raises(torch.OutOfMemoryError):
        with bounded_debug_decode(codec):
            assert codec.use_tiling
            assert codec.tile_sample_min_width == QWEN21_DEBUG_DECODE_DEFAULTS['tile_sample_min_width']
            raise torch.OutOfMemoryError('synthetic decoder OOM')
    assert not codec.use_tiling and codec.tile_sample_min_width == 96 and codec.cleared == 1


def test_decode_oom_releases_cached_and_traceback_tensors_before_restore(monkeypatch):
    import weakref
    import core.training.base_trainer as base
    from core.training.debug_decode_residency import debug_decode_residency
    monkeypatch.setattr(base, 'all_optimizers', lambda trainer: [])
    refs = []
    class FailingCodec(Codec):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.cached = None
        def clear_cache(self):
            super().clear_cache()
            self.cached = None
        def to(self, *args, **kwargs):
            if self.cleared:
                assert refs[0]() is None
            return super().to(*args, **kwargs)
        def fail(self):
            activation = torch.zeros(32)
            refs.append(weakref.ref(activation))
            self.cached = activation
            raise torch.OutOfMemoryError('synthetic decoder OOM')
    codec = FailingCodec()
    with pytest.raises(torch.OutOfMemoryError, match='synthetic decoder OOM'):
        with debug_decode_residency(SimpleNamespace(device='cpu'), codec):
            codec.fail()
    assert refs[0]() is None
