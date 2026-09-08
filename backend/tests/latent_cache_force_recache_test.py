"""force_recache has to OVERWRITE, not merely re-encode.

Skipping the ``has_latent`` short-circuit only gets a force pass as far as the
encode: the write underneath defaults to ``skip_existing=True``, so the VAE time
was spent and the result thrown away. Covers the three pre-encode paths (image,
video clip, audio clip) and what an overwrite does to the VAE stamp
``_setup_latent_caches`` records.
"""

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from PIL import Image

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training import audio_loader, video_loader
from core.training import latent_cache as lc
from core.training.base_trainer import BaseTrainer
from core.training.latent_cache import LatentCache
from core.training.vae_swap import module_latent_hash

NAMESPACE = "sdxl__c4__dtfloat16"


# --- driving the real pre-encode pass ---------------------------------------

def _cache(tmp_path):
    return LatentCache("ds1", base_cache_dir=str(tmp_path), namespace=NAMESPACE)


def _trainer(encode_value=0.0, **attrs):
    """Enough of a trainer for ``_generate_missing_latents_with_model_offload``:
    everything it offloads is already on CPU, so it never touches CUDA."""
    stub = SimpleNamespace(
        log_prefix="[Trainer]", device=torch.device("cpu"),
        vae=torch.nn.Linear(1, 1), vae_dtype=torch.float32,
        text_encoder=None, is_acestep=False, encode_value=encode_value,
        arch=SimpleNamespace(name="sdxl", temporal=None),
    )
    stub._main_model_module = lambda: None
    stub._check_stop_requested = lambda: None
    stub._temporal_spec = MethodType(BaseTrainer._temporal_spec, stub)
    stub.encode_image = lambda image, target_width, target_height: torch.full(
        (1, 4, target_height // 8, target_width // 8), stub.encode_value)
    for name, value in attrs.items():
        setattr(stub, name, value)
    return stub


def _run(trainer, cache, items, force_recache):
    BaseTrainer._generate_missing_latents_with_model_offload(
        trainer, [SimpleNamespace(unique_id="ds1", items=items)],
        {"ds1": cache}, None, force_recache=force_recache)


def _image_item(tmp_path, name="a.png", size=64):
    path = tmp_path / name
    Image.new("RGB", (size, size), (128, 128, 128)).save(path)
    return {"image_path": str(path), "width": size, "height": size,
            "item_type": "single"}


def _latent(value, size=64):
    return torch.full((1, 4, size // 8, size // 8), value)


# --- image ------------------------------------------------------------------

def test_force_recache_replaces_an_existing_image_latent(tmp_path):
    cache = _cache(tmp_path)
    items = [_image_item(tmp_path)]
    _run(_trainer(1.0), cache, items, force_recache=False)

    _run(_trainer(2.0), cache, items, force_recache=True)

    assert torch.equal(
        cache.load_latent(items[0]["image_path"], 64, 64, device="cpu"), _latent(2.0))


def test_without_force_recache_an_existing_image_latent_is_kept(tmp_path):
    cache = _cache(tmp_path)
    items = [_image_item(tmp_path)]
    _run(_trainer(1.0), cache, items, force_recache=False)

    _run(_trainer(2.0), cache, items, force_recache=False)

    assert torch.equal(
        cache.load_latent(items[0]["image_path"], 64, 64, device="cpu"), _latent(1.0))


# --- video clip -------------------------------------------------------------

CLIP_KEY = ("v.mp4", 32, 32, 0, 4, 1)


def _encode_clip(cache, value, force_recache=False, **kwargs):
    # force_recache passed only when set: the control tests then run unchanged
    # against a build that has no such parameter.
    if force_recache:
        kwargs["force_recache"] = True
    return video_loader.encode_and_cache_clip(
        cache=cache, video_path=CLIP_KEY[0], width=CLIP_KEY[1], height=CLIP_KEY[2],
        clip_start=CLIP_KEY[3], clip_length=CLIP_KEY[4], stride=CLIP_KEY[5],
        vae_encode_clip=lambda clip: torch.full((1, 8, 2, 4, 4), value),
        fps=24.0, device="cpu", **kwargs)


@pytest.fixture
def _no_decode(monkeypatch):
    monkeypatch.setattr(video_loader, "load_clip",
                        lambda *a, **k: torch.zeros(4, 3, 32, 32))
    monkeypatch.setattr(audio_loader, "load_audio",
                        lambda *a, **k: torch.zeros(2, 4800))


def test_force_recache_replaces_an_existing_clip_latent(tmp_path, _no_decode):
    cache = _cache(tmp_path)
    _encode_clip(cache, 1.0)

    _encode_clip(cache, 2.0, force_recache=True)

    assert torch.equal(cache.load_clip_latent(*CLIP_KEY, 24.0, device="cpu"),
                       torch.full((1, 8, 2, 4, 4), 2.0))


def test_force_recache_replaces_a_clip_record_read_back_whole(tmp_path, _no_decode):
    # return_record is the read the training loop makes; it has its own
    # short-circuit and would otherwise hand back the pre-force record.
    cache = _cache(tmp_path)
    _encode_clip(cache, 1.0, return_record=True)

    result = _encode_clip(cache, 2.0, force_recache=True, return_record=True)

    assert torch.equal(result["latents"], torch.full((1, 8, 2, 4, 4), 2.0))
    assert torch.equal(cache.load_clip_record(*CLIP_KEY, 24.0, device="cpu")["latents"],
                       torch.full((1, 8, 2, 4, 4), 2.0))


def test_without_force_recache_a_cached_clip_is_returned_unencoded(tmp_path, _no_decode):
    cache = _cache(tmp_path)
    _encode_clip(cache, 1.0)

    assert torch.equal(_encode_clip(cache, 2.0), torch.full((1, 8, 2, 4, 4), 1.0))


# --- audio clip -------------------------------------------------------------

AUDIO_KEY = ("a.wav", 4.0, 48000)


def _encode_audio(cache, value, force_recache=False):
    kwargs = {"force_recache": True} if force_recache else {}
    return audio_loader.encode_and_cache_audio(
        cache=cache, audio_path=AUDIO_KEY[0], clip_seconds=AUDIO_KEY[1],
        sample_rate=AUDIO_KEY[2], device="cpu",
        vae_encode_audio=lambda wav: torch.full((1, 8, 64), value), **kwargs)


def test_force_recache_replaces_an_existing_audio_latent(tmp_path, _no_decode):
    cache = _cache(tmp_path)
    _encode_audio(cache, 1.0)

    _encode_audio(cache, 2.0, force_recache=True)

    assert torch.equal(cache.load_audio_latent(*AUDIO_KEY, device="cpu"),
                       torch.full((1, 8, 64), 2.0))


def test_without_force_recache_a_cached_audio_latent_is_returned_unencoded(
        tmp_path, _no_decode):
    cache = _cache(tmp_path)
    _encode_audio(cache, 1.0)

    assert torch.equal(_encode_audio(cache, 2.0), torch.full((1, 8, 64), 1.0))


# --- the pre-encode pass reaches both seams with the flag -------------------

def test_the_video_branch_passes_force_recache_to_the_seam(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(video_loader, "encode_and_cache_clip", lambda **kw: calls.append(kw))
    window = SimpleNamespace(start_frame=0, start_time=0.0)
    trainer = _trainer(arch=SimpleNamespace(name="ltx2", temporal=SimpleNamespace(),
                                            vae_encode_clip=lambda self, clip: clip))
    trainer._video_clip_window = lambda item, training: ("v.mp4", 32, 32, 4, 1, 24.0, window)
    trainer._clip_vae_tiling_policy = lambda: None
    trainer._clip_audio_prep_version = lambda: None
    trainer._clip_audio_seam = lambda path: None

    _run(trainer, _cache(tmp_path), [{"item_type": "video", "video_path": "v.mp4"}],
         force_recache=True)

    # The branch swallows exceptions into a WARNING, so a broken call would look
    # like a pass if only the flag were asserted.
    assert "WARNING" not in capsys.readouterr().out
    assert calls[0]["force_recache"] is True


def test_the_audio_branch_passes_force_recache_to_the_seam(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(audio_loader, "encode_and_cache_audio", lambda **kw: calls.append(kw))
    trainer = _trainer(is_acestep=True,
                       arch=SimpleNamespace(name="acestep", temporal=None,
                                            vae_encode_audio=lambda self, wav: wav))

    _run(trainer, _cache(tmp_path),
         [{"item_type": "audio", "audio_path": "a.wav", "image_path": "a.wav",
           "clip_seconds": 4.0}], force_recache=True)

    assert "WARNING" not in capsys.readouterr().out
    assert calls[0]["force_recache"] is True


# --- the stamp the overwrite leaves behind ----------------------------------

def _tiny_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1, norm_num_groups=4,
        latent_channels=4, sample_size=32,
    )


def _stamp_trainer(vae, dtype=torch.float16):
    stub = SimpleNamespace(
        log_prefix="[Trainer]", model_path="M:/model/sdxl/base_a.safetensors",
        vae=vae, training_dtype=dtype, arch=SimpleNamespace(name="sdxl"),
    )
    stub._build_cache_namespace = lambda: NAMESPACE
    stub._run_vae_identity = MethodType(BaseTrainer._run_vae_identity, stub)
    stub._log_other_vae_caches = MethodType(BaseTrainer._log_other_vae_caches, stub)
    return stub


def _setup(trainer, force_recache=False):
    kwargs = {"force_recache": True} if force_recache else {}
    return BaseTrainer._setup_latent_caches(
        trainer, [SimpleNamespace(unique_id="ds1", items=[{}])], **kwargs)["ds1"]


@pytest.fixture
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "get_cache_base_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture
def _one_namespace(monkeypatch):
    """Both VAEs address one directory, so a force pass really does overwrite
    the other one's entries (see latent_cache_vae_identity_test)."""
    monkeypatch.setattr(lc, "vae_cache_namespace", lambda _hash: "vae-collision")


def test_a_force_pass_restamps_a_cache_it_is_about_to_overwrite(_cache_root):
    # The stamp is refreshed at SETUP, before the first entry is rewritten, so an
    # interrupted overwrite still leaves a stamp true of its own VAE.
    vae = _tiny_vae()
    cache = _setup(_stamp_trainer(vae))
    cache.save_latent("a.png", 512, 512, _latent(1.0, 512))

    again = _setup(_stamp_trainer(vae, torch.bfloat16), force_recache=True)

    info = again.load_cache_info()
    assert info["training_dtype"] == "torch.bfloat16"
    assert info["vae_latent_hash"] == module_latent_hash(vae)


def test_without_force_a_validated_cache_keeps_the_stamp_that_describes_it(_cache_root):
    vae = _tiny_vae()
    _setup(_stamp_trainer(vae)).save_latent("a.png", 512, 512, _latent(1.0, 512))

    again = _setup(_stamp_trainer(vae, torch.bfloat16))

    # The entries are still the fp16 run's, and the stamp still says so.
    assert again.load_cache_info()["training_dtype"] == "torch.float16"


def test_a_force_pass_never_stamps_a_vae_that_did_not_encode(_cache_root, _one_namespace):
    first, second = _tiny_vae(), _tiny_vae()
    _setup(_stamp_trainer(first)).save_latent("a.png", 512, 512, _latent(1.0, 512))

    again = _setup(_stamp_trainer(second), force_recache=True)

    # Deleted at setup, so no entry of A's survives under B's stamp.
    assert not again.has_latent("a.png", 512, 512)
    assert again.load_cache_info()["vae_latent_hash"] == module_latent_hash(second)


def test_a_force_pass_with_an_unhashable_vae_still_leaves_no_stamp(
        _cache_root, _one_namespace):
    class _UnhashableVAE(torch.nn.Module):
        def state_dict(self, *args, **kwargs):
            raise RuntimeError("state_dict unavailable")

    _setup(_stamp_trainer(_tiny_vae())).save_latent("a.png", 512, 512, _latent(1.0, 512))

    again = _setup(_stamp_trainer(_UnhashableVAE()), force_recache=True)

    assert again.load_cache_info() is None
