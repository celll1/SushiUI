"""``debug_latents`` dumps for LTX-2.3, MiniMax-H3 and ACE-Step.

Guards the defect where all three ``train_step``s declared ``debug_save_path`` /
``debug_captions`` / ``debug_reference_image_paths`` and never read them, so
``debug_latents`` silently produced nothing while the caller looked correct.

No real model is loaded: each transformer stand-in analytically returns that
arch's exact velocity target, which pins the SIGN of the x0 reconstruction
(MiniMax-H3's is the opposite of the other two).
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.components.wiring import LTX2_TEMPORAL, MINIMAX_H3_TEMPORAL
from core.training import latent_debug_dump as dbg
from core.training.ops import acestep_ops, ltx2_ops, minimax_h3_ops


ENDPOINT_KEYS = ("latents", "noisy_latents", "predicted_velocity",
                 "actual_velocity", "predicted_latent")


# ----------------------------------------------------------------------
# window sizing
# ----------------------------------------------------------------------

def test_window_is_the_minimum_decodable_window():
    # LTX-2.3 decodes a single latent frame; MiniMax-H3 needs 22 pixel frames,
    # which its (1, 4, 4, 4, 4) chunking covers with 7 latent frames.
    assert dbg.leading_window_frames(LTX2_TEMPORAL, 33) == 1
    assert dbg.leading_window_frames(MINIMAX_H3_TEMPORAL, 12) == 7
    # Never longer than the clip actually has.
    assert dbg.leading_window_frames(MINIMAX_H3_TEMPORAL, 3) == 3
    assert dbg.leading_window_frames(None, 5) == 1


def test_window_cost_is_independent_of_clip_length():
    for t_lat in (7, 12, 52, 302):
        assert dbg.leading_window_frames(MINIMAX_H3_TEMPORAL, t_lat) == 7


def test_filmstrip_tiles_time_along_width():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    strip = dbg.video_filmstrip(x)
    assert strip.shape == (1, 2, 4, 15)
    assert torch.equal(strip[0, :, :, 5:10], x[0, :, 1])


def test_audio_strip_is_channel_by_time():
    x = torch.randn(2, 9, 4)
    strip = dbg.audio_strip(x)
    assert strip.shape == (1, 1, 4, 9)
    assert torch.equal(strip[0, 0], x[0].t())


def test_channel_stats_flag_a_collapsed_prediction():
    collapsed = torch.full((1, 3, 4, 4), 0.5)
    stats = dbg.channel_stats(collapsed, channel_dim=1)
    assert stats["mean"] == pytest.approx([0.5, 0.5, 0.5])
    assert stats["std"] == pytest.approx([0.0, 0.0, 0.0])


# ----------------------------------------------------------------------
# LTX-2.3
# ----------------------------------------------------------------------

LTX_SHAPE = (2, 6, 3, 2, 2)  # [B, C, T_lat, H', W']


class _Rope:
    def prepare_video_coords(self, b, t, h, w, device, fps=1.0):
        return torch.zeros(b, 3, t * h * w, 2)


class _Ltx2Transformer:
    """Returns LTX-2.3's exact target ``v = noise - x_0`` in packed space."""

    def __init__(self, clean, sigma):
        self.rope = _Rope()
        self.clean = clean
        self.sigma = sigma

    def __call__(self, **kw):
        b, c, t, h, w = self.clean.shape
        x_t = ltx2_ops._unpack_leading_frames(kw["hidden_states"], t, h, w)
        noise = (x_t - (1.0 - self.sigma) * self.clean) / self.sigma
        v = ltx2_ops._pack_latents(noise - self.clean)
        return v, torch.zeros_like(kw["audio_hidden_states"])


class _Ltx2ConstantTransformer:
    def __init__(self, value=0.25):
        self.rope = _Rope()
        self.value = value

    def __call__(self, **kw):
        return (torch.full_like(kw["hidden_states"], self.value),
                torch.zeros_like(kw["audio_hidden_states"]))


def _ltx2_trainer(transformer):
    return SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        timestep_sampler=None, mixed_precision=False, transformer=transformer,
        reconstruction_loss_weight=0.0, tread_config=None, blockskip_config=None,
        ltx2_block_loop_wrapper=None, log_prefix="[test]",
        arch=SimpleNamespace(temporal=LTX2_TEMPORAL),
    )


def _run_ltx2(transformer, out, sigma=0.3, **kwargs):
    latents = torch.randn(*LTX_SHAPE)
    aux = {"audio_text_embedding": torch.zeros(LTX_SHAPE[0], 5, 8),
           "mask": torch.ones(LTX_SHAPE[0], 5, dtype=torch.long)}
    ltx2_ops.train_step(
        _ltx2_trainer(transformer),
        latents=latents,
        prompt_embeds=torch.zeros(LTX_SHAPE[0], 5, 8),
        ltx2_aux=aux,
        timesteps=torch.full((LTX_SHAPE[0],), sigma),
        debug_save_path=out,
        **kwargs,
    )
    return latents


def test_ltx2_no_debug_path_writes_nothing(tmp_path):
    _run_ltx2(_Ltx2ConstantTransformer(), None)
    assert list(tmp_path.iterdir()) == []


def test_ltx2_dump_has_endpoint_keys_and_filmstrip_shape(tmp_path):
    out = tmp_path / "step_000010"
    _run_ltx2(_Ltx2ConstantTransformer(), out,
              debug_captions=["a caption", "second"],
              debug_reference_image_paths=[None, "/tmp/ref.png"])

    files = list(out.glob("latents_t*.pt"))
    assert len(files) == 1
    assert float(files[0].stem.replace("latents_t", "")) == pytest.approx(0.3)

    data = torch.load(files[0], map_location="cpu")
    assert data["model_type"] == "ltx2"
    assert data["caption"] == "a caption"
    assert data["reference_image_path"] == "/tmp/ref.png"
    assert data["window_latent_frames"] == 1
    assert data["clip_latent_frames"] == LTX_SHAPE[2]
    for key in ENDPOINT_KEYS:
        # [1, C, H, n_win*W] — the shape latent_to_image already understands.
        assert data[key].shape == (1, LTX_SHAPE[1], LTX_SHAPE[3], LTX_SHAPE[4]), key
        assert key in data["channel_stats"]
        assert len(data["channel_stats"][key]["mean"]) == LTX_SHAPE[1]


def test_ltx2_predicted_latent_matches_noising_definition(tmp_path):
    """``v = noise - x_0`` so ``x_0 = x_t - sigma * v``; the opposite sign lands
    on ``x_0 + 2 sigma v``."""
    out = tmp_path / "step_000000"
    sigma = 0.4
    latents = torch.randn(*LTX_SHAPE)
    aux = {"audio_text_embedding": torch.zeros(LTX_SHAPE[0], 5, 8),
           "mask": torch.ones(LTX_SHAPE[0], 5, dtype=torch.long)}
    ltx2_ops.train_step(
        _ltx2_trainer(_Ltx2Transformer(latents, sigma)),
        latents=latents,
        prompt_embeds=torch.zeros(LTX_SHAPE[0], 5, 8),
        ltx2_aux=aux,
        timesteps=torch.full((LTX_SHAPE[0],), sigma),
        debug_save_path=out,
    )
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    expected = dbg.video_filmstrip(latents[0:1, :, :1])
    assert torch.allclose(data["predicted_latent"], expected, atol=1e-4)
    assert torch.allclose(data["predicted_velocity"], data["actual_velocity"], atol=1e-5)
    assert data["recon_loss"] == pytest.approx(0.0, abs=1e-8)


def test_ltx2_sign_is_xt_minus_sigma_v(tmp_path):
    out = tmp_path / "step_000001"
    sigma = 0.7
    _run_ltx2(_Ltx2ConstantTransformer(0.25), out, sigma=sigma)
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    expected = data["noisy_latents"] - sigma * data["predicted_velocity"]
    assert torch.allclose(data["predicted_latent"], expected, atol=1e-5)


# ----------------------------------------------------------------------
# MiniMax-H3
# ----------------------------------------------------------------------

H3_SHAPE = (1, 4, 12, 4, 4)  # [B, C, T_lat, H', W'] — T_lat > the 7-frame window
H3_AUDIO_CHANNELS = 32
H3_N_AUD = 4


class _H3Transformer:
    """Returns MiniMax-H3's exact target ``v = x_0 - eps`` for both streams.

    ``x_t = (1 - s) x_0 + s * eps`` gives ``v = (x_0 - x_t) / s``, i.e. exactly
    the ``x_0 = x_t + s * v`` this arch uses.
    """

    def __init__(self, clean_video, sigma_v, clean_audio, sigma_a):
        self.clean_video = clean_video
        self.sigma_v = sigma_v
        self.clean_audio = clean_audio
        self.sigma_a = sigma_a

    def __call__(self, **kw):
        from core.models.minimax_h3.h3_pipeline_ops import (
            patchify_video_latents, unpatchify_video_rows)
        b, c, t, h, w = self.clean_video.shape
        x_t = unpatchify_video_rows(kw["hidden_states"], num_latent_frames=t,
                                    latent_height=h, latent_width=w,
                                    latent_channels=c)
        v_video = patchify_video_latents((self.clean_video - x_t) / self.sigma_v)
        v_audio = (self.clean_audio - kw["audio_hidden_states"]) / self.sigma_a
        return v_video, v_audio


class _H3ConstantTransformer:
    def __init__(self, value=0.25):
        self.value = value

    def __call__(self, **kw):
        return (torch.full_like(kw["hidden_states"], self.value),
                torch.full_like(kw["audio_hidden_states"], self.value))


def _h3_trainer(transformer):
    return SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        timestep_sampler=None, transformer=transformer,
        reconstruction_loss_weight=0.0, audio_loss_weight=1.0,
        log_prefix="[test]", log_extra_metric=lambda *a, **k: None,
        arch=SimpleNamespace(temporal=MINIMAX_H3_TEMPORAL),
    )


def _h3_audio_rows():
    from core.models.minimax_h3.h3_pipeline_ops import AUDIO_CHANNELS
    return AUDIO_CHANNELS * H3_N_AUD


def _run_h3(make_transformer, out, u=0.5, **kwargs):
    latents = torch.randn(*H3_SHAPE)
    audio = torch.randn(H3_SHAPE[0], _h3_audio_rows(), H3_AUDIO_CHANNELS)
    sigma_v = minimax_h3_ops._shift_sigma(u, 12.0)
    sigma_a = minimax_h3_ops._shift_sigma(u, 3.0)
    aux = {
        "num_text_tokens": torch.tensor([4] * H3_SHAPE[0]),
        "audio_latents": audio,
        "audio_present": torch.ones(H3_SHAPE[0], dtype=torch.bool),
    }
    minimax_h3_ops.train_step(
        _h3_trainer(make_transformer(latents, sigma_v, audio, sigma_a)),
        latents=latents,
        prompt_embeds=torch.zeros(H3_SHAPE[0], 4, 8),
        h3_aux=aux,
        timesteps=torch.full((H3_SHAPE[0],), u),
        debug_save_path=out,
        **kwargs,
    )
    return latents, audio, sigma_v, sigma_a


def test_h3_no_debug_path_writes_nothing(tmp_path):
    _run_h3(lambda *a: _H3ConstantTransformer(), None)
    assert list(tmp_path.iterdir()) == []


def test_h3_dump_carries_both_streams(tmp_path):
    out = tmp_path / "step_000010"
    _latents, _audio, sigma_v, sigma_a = _run_h3(
        lambda *a: _H3ConstantTransformer(), out,
        debug_captions=["h3 caption"], debug_reference_image_paths=["/tmp/r.png"])

    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert data["model_type"] == "minimax_h3"
    assert data["timestep"] == pytest.approx(sigma_v, abs=1e-4)
    assert data["audio_sigma"] == pytest.approx(sigma_a, abs=1e-6)
    assert data["window_latent_frames"] == 7
    assert data["clip_latent_frames"] == H3_SHAPE[2]

    for key in ENDPOINT_KEYS:
        assert data[key].shape == (1, H3_SHAPE[1], H3_SHAPE[3], 7 * H3_SHAPE[4]), key
        akey = dbg.AUDIO_KEY_PREFIX + key
        assert data[akey].shape == (1, 1, H3_AUDIO_CHANNELS, _h3_audio_rows()), akey
        assert key in data["channel_stats"] and akey in data["channel_stats"]


def test_h3_predicted_latent_uses_the_opposite_sign(tmp_path):
    """MiniMax-H3 targets ``v = x_0 - eps``, so ``x_0 = x_t + sigma * v``.

    With the usual flow-matching sign (minus) this lands on ``x_0 - 2 sigma v``.
    """
    out = tmp_path / "step_000000"
    latents, audio, _sv, _sa = _run_h3(
        lambda c, sv, a, sa: _H3Transformer(c, sv, a, sa), out, u=0.6)

    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert torch.allclose(data["predicted_latent"],
                          dbg.video_filmstrip(latents[0:1, :, :7]), atol=1e-3)
    assert torch.allclose(data[dbg.AUDIO_KEY_PREFIX + "predicted_latent"],
                          dbg.audio_strip(audio), atol=1e-3)
    assert data["recon_loss"] == pytest.approx(0.0, abs=1e-6)


def test_h3_still_has_no_audio_rows_and_dumps_video_only(tmp_path):
    """A still (T_lat=1) has a zero-latent audio budget, so there is no audio
    stream to dump — the video half must still be written."""
    out = tmp_path / "step_000002"
    latents = torch.randn(1, 4, 1, 4, 4)
    minimax_h3_ops.train_step(
        _h3_trainer(_H3ConstantTransformer(0.25)),
        latents=latents,
        prompt_embeds=torch.zeros(1, 4, 8),
        h3_aux={"num_text_tokens": torch.tensor([4])},
        timesteps=torch.full((1,), 0.5),
        debug_save_path=out,
    )
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert data["window_latent_frames"] == 1
    assert data["latents"].shape == (1, 4, 4, 4)
    assert not any(k.startswith(dbg.AUDIO_KEY_PREFIX) and k != "audio_sigma"
                   and k != "audio_present" for k in data)


def test_h3_sign_is_xt_plus_sigma_v(tmp_path):
    out = tmp_path / "step_000001"
    _l, _a, sigma_v, sigma_a = _run_h3(lambda *a: _H3ConstantTransformer(0.25), out, u=0.4)
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert torch.allclose(
        data["predicted_latent"],
        data["noisy_latents"] + sigma_v * data["predicted_velocity"], atol=1e-5)
    p = dbg.AUDIO_KEY_PREFIX
    assert torch.allclose(
        data[p + "predicted_latent"],
        data[p + "noisy_latents"] + sigma_a * data[p + "predicted_velocity"], atol=1e-5)


# ----------------------------------------------------------------------
# ACE-Step
# ----------------------------------------------------------------------

ACE_SHAPE = (2, 6, 64)  # [B, T_lat, 64]


@pytest.fixture
def stub_acestep_mixin(monkeypatch):
    """Stand in for ``core.pipeline_backends.acestep``.

    Importing the real package pulls every generation mixin; ``train_step`` only
    needs the silence slice, which is pure shape work.
    """
    class _Mixin:
        @staticmethod
        def _acestep_silence_slice(silence_latent, t_lat):
            return silence_latent[:, :t_lat]

    mod = types.ModuleType("core.pipeline_backends.acestep")
    mod.AceStepMixin = _Mixin
    pkg = types.ModuleType("core.pipeline_backends")
    pkg.acestep = mod
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "core.pipeline_backends", pkg)
    monkeypatch.setitem(sys.modules, "core.pipeline_backends.acestep", mod)


class _AceDecoder:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, **kw):
        return (self.fn(kw["hidden_states"], kw["timestep"]),)


class _AceDit:
    def __init__(self, fn):
        self.decoder = _AceDecoder(fn)

    def prepare_condition(self, **kw):
        return None, None, None


def _ace_trainer(fn):
    return SimpleNamespace(
        device=torch.device("cpu"), training_dtype=torch.float32,
        timestep_sampler=None, mixed_precision=False, transformer=_AceDit(fn),
        reconstruction_loss_weight=0.0, log_prefix="[test]",
        acestep_silence_latent=torch.zeros(1, 750, 64),
    )


def _run_ace(fn, out, sigma=0.3, **kwargs):
    latents = torch.randn(*ACE_SHAPE)
    aux = {
        "text_attention_mask": torch.ones(ACE_SHAPE[0], 5, dtype=torch.long),
        "lyric_hidden_states": torch.zeros(ACE_SHAPE[0], 5, 8),
        "lyric_attention_mask": torch.ones(ACE_SHAPE[0], 5, dtype=torch.long),
    }
    acestep_ops.train_step(
        _ace_trainer(fn),
        latents=latents,
        text_embeddings=torch.zeros(ACE_SHAPE[0], 5, 8),
        aux=aux,
        timesteps=torch.full((ACE_SHAPE[0],), sigma),
        debug_save_path=out,
        **kwargs,
    )
    return latents


def test_acestep_no_debug_path_writes_nothing(tmp_path, stub_acestep_mixin):
    _run_ace(lambda xt, t: torch.full_like(xt, 0.25), None)
    assert list(tmp_path.iterdir()) == []


def test_acestep_dump_is_a_channel_by_time_map(tmp_path, stub_acestep_mixin):
    out = tmp_path / "step_000010"
    _run_ace(lambda xt, t: torch.full_like(xt, 0.25), out,
             debug_captions=["ace caption"], debug_reference_image_paths=["/tmp/a.wav"])

    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert data["model_type"] == "acestep"
    assert data["caption"] == "ace caption"
    assert data["latent_frames"] == ACE_SHAPE[1]
    for key in ENDPOINT_KEYS:
        assert data[key].shape == (1, 1, ACE_SHAPE[2], ACE_SHAPE[1]), key
        assert len(data["channel_stats"][key]["mean"]) == ACE_SHAPE[2]


def test_acestep_predicted_latent_matches_noising_definition(tmp_path, stub_acestep_mixin):
    """``v = noise - x_0`` so ``x_0 = x_t - sigma * v``."""
    out = tmp_path / "step_000000"
    sigma = 0.35
    holder = {}

    def perfect(xt, t):
        x0 = holder["x0"]
        noise = (xt - (1.0 - sigma) * x0) / sigma
        return noise - x0

    latents = torch.randn(*ACE_SHAPE)
    holder["x0"] = latents
    aux = {
        "text_attention_mask": torch.ones(ACE_SHAPE[0], 5, dtype=torch.long),
        "lyric_hidden_states": torch.zeros(ACE_SHAPE[0], 5, 8),
        "lyric_attention_mask": torch.ones(ACE_SHAPE[0], 5, dtype=torch.long),
    }
    acestep_ops.train_step(
        _ace_trainer(perfect),
        latents=latents,
        text_embeddings=torch.zeros(ACE_SHAPE[0], 5, 8),
        aux=aux,
        timesteps=torch.full((ACE_SHAPE[0],), sigma),
        debug_save_path=out,
    )
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert torch.allclose(data["predicted_latent"], dbg.audio_strip(latents), atol=1e-4)
    assert data["recon_loss"] == pytest.approx(0.0, abs=1e-8)


def test_acestep_sign_is_xt_minus_sigma_v(tmp_path, stub_acestep_mixin):
    out = tmp_path / "step_000001"
    sigma = 0.8
    _run_ace(lambda xt, t: torch.full_like(xt, 0.25), out, sigma=sigma)
    data = torch.load(next(out.glob("latents_t*.pt")), map_location="cpu")
    assert torch.allclose(
        data["predicted_latent"],
        data["noisy_latents"] - sigma * data["predicted_velocity"], atol=1e-5)
