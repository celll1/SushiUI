"""Which VAE wrote a latent cache, and what happens when the next run has another.

The cache namespace only grows a ``vae-`` token for a DECLARED swap, so two
checkpoints with different embedded VAEs land in the SAME namespace: nothing but
the recorded identity keeps the second run from training against the first VAE's
latent space. Drives the real ``_setup_latent_caches`` wiring; why a mismatch
deletes rather than flags is in API_REFERENCE.md (``discard_latents``).
"""

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training import latent_cache as lc
from core.training.base_trainer import BaseTrainer
from core.training.latent_cache import LatentCache
from core.training.vae_swap import module_latent_hash

NAMESPACE = "sdxl__c4__dtfloat16"


# --- fixtures ---------------------------------------------------------------

def _tiny_vae():
    """A real AutoencoderKL, small enough to build in milliseconds. Two calls
    give two DIFFERENT VAEs (random init)."""
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,), layers_per_block=1, norm_num_groups=4,
        latent_channels=4, sample_size=32,
    )


def _trainer(vae, model_path="M:/model/sdxl/base_a.safetensors"):
    stub = SimpleNamespace(
        log_prefix="[Trainer]", model_path=model_path, vae=vae,
        training_dtype=torch.float16, arch=SimpleNamespace(name="sdxl"),
    )
    stub._build_cache_namespace = lambda: NAMESPACE
    stub._run_vae_identity = MethodType(BaseTrainer._run_vae_identity, stub)
    return stub


def _dataset(unique_id="ds1", count=2):
    return SimpleNamespace(unique_id=unique_id, items=[{}] * count)


def _setup(trainer, dataset=None):
    return BaseTrainer._setup_latent_caches(trainer, [dataset or _dataset()])


def _write_latent(cache, image_path="a.png", size=512, value=0.0):
    cache.save_latent(image_path, size, size,
                      torch.full((1, 4, size // 8, size // 8), value))


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "get_cache_base_dir", lambda: str(tmp_path))
    return tmp_path


# --- the identity that gets recorded ----------------------------------------

def test_a_fresh_cache_records_this_run_s_vae():
    vae = _tiny_vae()
    cache = _setup(_trainer(vae))["ds1"]

    info = cache.load_cache_info()
    assert info["vae_latent_hash"] == module_latent_hash(vae)
    assert info["namespace"] == NAMESPACE


def test_the_recorded_identity_is_the_encoding_module_not_the_resolver():
    # A swap run records the hash of the module that encodes; the resolver's
    # latent_hash covers the SOURCE state dict and is not comparable to it.
    vae = _tiny_vae()
    trainer = _trainer(vae)
    trainer.vae_identity = SimpleNamespace(family="flux1", latent_hash="deadbeefdeadbeef")

    info = _setup(trainer)["ds1"].load_cache_info()
    assert info["vae_latent_hash"] == module_latent_hash(vae)
    assert info["vae_family"] == "flux1"


# --- (b) same VAE: the cache is reused --------------------------------------

def test_the_same_vae_reuses_the_cached_latents(capsys):
    vae = _tiny_vae()
    _write_latent(_setup(_trainer(vae))["ds1"])

    again = _setup(_trainer(vae))["ds1"]

    assert again.has_latent("a.png", 512, 512)
    assert "Validation passed" in capsys.readouterr().out


def test_a_different_base_model_with_the_same_vae_keeps_the_cache(capsys):
    # The latents depend on the VAE, not on which checkpoint carried it.
    vae = _tiny_vae()
    _write_latent(_setup(_trainer(vae, "M:/model/sdxl/base_a.safetensors"))["ds1"])

    again = _setup(_trainer(vae, "M:/model/sdxl/base_b.safetensors"))["ds1"]

    assert again.has_latent("a.png", 512, 512)
    out = capsys.readouterr().out
    assert "Model path differs" in out and "Validation passed" in out


# --- (a)+(c) different VAE: separated, and said out loud ---------------------

def test_a_different_vae_discards_the_cache_and_logs_what_differed(capsys):
    first, second = _tiny_vae(), _tiny_vae()
    _write_latent(_setup(_trainer(first))["ds1"])
    capsys.readouterr()

    again = _setup(_trainer(second))["ds1"]

    out = capsys.readouterr().out
    assert "VAE identity mismatch" in out
    assert module_latent_hash(first) in out and module_latent_hash(second) in out
    assert "deleted 1 cached latents" in out and "regenerating" in out

    assert not again.has_latent("a.png", 512, 512)
    assert again.load_latent("a.png", 512, 512, device="cpu") is None
    assert again.load_cache_info()["vae_latent_hash"] == module_latent_hash(second)


def test_an_unlabelled_legacy_cache_is_not_adopted(tmp_path, capsys):
    # Pre-existing caches carry no cache_info.json at all: their VAE is unknown,
    # which is exactly the case that must not be read back.
    cache = LatentCache("ds1", base_cache_dir=str(tmp_path), namespace=NAMESPACE)
    _write_latent(cache)
    assert not cache.cache_info_path.exists()

    again = _setup(_trainer(_tiny_vae()))["ds1"]

    assert not again.has_latent("a.png", 512, 512)
    assert "No cache_info.json found" in capsys.readouterr().out


def test_one_stale_dataset_does_not_discard_the_others():
    first, second = _tiny_vae(), _tiny_vae()
    datasets = [_dataset("ds1"), _dataset("ds2")]
    for dataset in datasets:
        _write_latent(BaseTrainer._setup_latent_caches(
            _trainer(first), [dataset])[dataset.unique_id])
    # ds2 alone is re-encoded by the other VAE.
    ds2 = BaseTrainer._setup_latent_caches(_trainer(second), [datasets[1]])["ds2"]
    _write_latent(ds2, value=1.0)

    caches = BaseTrainer._setup_latent_caches(_trainer(first), datasets)

    assert caches["ds1"].has_latent("a.png", 512, 512)
    assert not caches["ds2"].has_latent("a.png", 512, 512)


# --- the three paths a flag-based invalidation missed ------------------------

def test_the_video_and_audio_hit_paths_see_a_mismatched_cache_as_empty():
    # LTX-2.3 / MiniMax-H3 / ACE-Step decide a cache hit with load_clip_record /
    # load_clip_latent / load_audio_latent, never with has_*: an entry that is
    # merely flagged would be returned and never re-encoded.
    first, second = _tiny_vae(), _tiny_vae()
    cache = _setup(_trainer(first))["ds1"]
    cache.save_clip_latent("v.mp4", 512, 512, 0, 9, 1, torch.zeros(1, 128, 2, 16, 16),
                           fps=24.0, audio_latents=torch.zeros(1, 8, 64), has_audio=True)
    cache.save_audio_latent("a.wav", 4.0, 48000, torch.zeros(1, 8, 64))

    again = _setup(_trainer(second))["ds1"]

    assert again.load_clip_latent("v.mp4", 512, 512, 0, 9, 1, 24.0, device="cpu") is None
    assert again.load_clip_record("v.mp4", 512, 512, 0, 9, 1, 24.0, device="cpu") is None
    assert again.load_audio_latent("a.wav", 4.0, 48000, device="cpu") is None


def test_entries_this_run_never_visits_are_gone_too():
    # The encode pass only walks the current items at their current bucket size,
    # so a 1024 entry left by the previous VAE would survive a flag and be read
    # by the resolution-curriculum switch (base_trainer.py's second pass).
    first, second = _tiny_vae(), _tiny_vae()
    cache = _setup(_trainer(first))["ds1"]
    _write_latent(cache, size=512, value=1.0)
    _write_latent(cache, size=1024, value=1.0)

    again = _setup(_trainer(second))["ds1"]

    # The warm-up pass re-encodes 512; the target-resolution pass must not find
    # the previous VAE's 1024 entry still sitting there.
    assert not again.has_latent("a.png", 1024, 1024)
    assert not again.has_latent("a.png", 512, 512)


def test_an_interrupted_regeneration_leaves_no_stamp_the_old_vae_can_use(capsys):
    # The stamp is written after the delete, so what it claims is true of every
    # file on disk even if the encode pass dies halfway.
    first, second = _tiny_vae(), _tiny_vae()
    _write_latent(_setup(_trainer(first))["ds1"], value=1.0)

    # Run 2 (VAE B) purges, stamps, and is interrupted after one entry.
    partial = _setup(_trainer(second))["ds1"]
    _write_latent(partial, value=2.0)
    assert partial.load_cache_info()["vae_latent_hash"] == module_latent_hash(second)
    capsys.readouterr()

    # Run 3 goes back to VAE A: B's half-written cache must not be accepted.
    again = _setup(_trainer(first))["ds1"]

    assert "VAE identity mismatch" in capsys.readouterr().out
    assert not again.has_latent("a.png", 512, 512)
    assert again.load_cache_info()["vae_latent_hash"] == module_latent_hash(first)


# --- an unhashable VAE: keep the latents, but never leave a stamp ------------

class _UnhashableVAE(torch.nn.Module):
    """A live VAE whose state_dict() raises, which is the only way
    module_latent_hash returns None for a VAE that exists."""

    def __init__(self):
        super().__init__()
        self.config = {"scaling_factor": 0.18215}

    def state_dict(self, *args, **kwargs):
        raise RuntimeError("state_dict unavailable")


def test_an_unhashable_vae_keeps_the_latents_and_drops_the_stamp(capsys):
    cache = _setup(_trainer(_tiny_vae()))["ds1"]
    _write_latent(cache, value=1.0)
    capsys.readouterr()

    again = _setup(_trainer(_UnhashableVAE()))["ds1"]

    out = capsys.readouterr().out
    assert "used UNVERIFIED" in out
    # Kept: a hash failure is not evidence of a different VAE, and deleting is
    # irreversible.
    assert again.has_latent("a.png", 512, 512)
    # Dropped: this run writes into the directory, so nothing later may trust it.
    assert again.load_cache_info() is None


def test_an_unhashable_vae_does_not_stamp_an_empty_cache(capsys):
    cache = _setup(_trainer(_UnhashableVAE()))["ds1"]

    assert cache.load_cache_info() is None
    assert "cannot identify this run's VAE" in capsys.readouterr().out


def test_the_next_healthy_run_does_not_trust_an_unverified_cache(capsys):
    # The whole point of dropping the stamp: run 1 (VAE A) stamps, run 2 cannot
    # identify its VAE and writes into the same directory, so run 3 -- even back
    # on VAE A -- must not read the mixture.
    vae = _tiny_vae()
    _write_latent(_setup(_trainer(vae))["ds1"], value=1.0)
    _setup(_trainer(_UnhashableVAE()))
    capsys.readouterr()

    again = _setup(_trainer(vae))["ds1"]

    assert "No cache_info.json found" in capsys.readouterr().out
    assert not again.has_latent("a.png", 512, 512)


# --- only the latent space is worth deleting an encode over ------------------

def test_a_training_dtype_change_keeps_the_cache(capsys):
    vae = _tiny_vae()
    _write_latent(_setup(_trainer(vae))["ds1"], value=1.0)
    capsys.readouterr()

    trainer = _trainer(vae)
    trainer.training_dtype = torch.bfloat16
    again = _setup(trainer)["ds1"]

    out = capsys.readouterr().out
    assert "Training dtype mismatch" in out
    assert "kept despite a training_dtype mismatch" in out
    assert "deleted" not in out
    # Every arch's train_step casts a loaded latent to training_dtype.
    assert again.has_latent("a.png", 512, 512)
    assert torch.equal(again.load_latent("a.png", 512, 512, device="cpu"),
                       torch.ones(1, 4, 64, 64))


def test_validate_names_the_mismatch_it_found():
    vae = _tiny_vae()
    cache = _setup(_trainer(vae))["ds1"]
    args = ("M:/model/sdxl/base_a.safetensors", "sdxl", "torch.float16")
    hash_ = module_latent_hash(vae)

    assert cache.validate(*args, vae_latent_hash=hash_) is None
    assert cache.validate(*args, vae_latent_hash="0" * 16) == "vae_identity"
    assert cache.validate(args[0], "sd15", args[2], vae_latent_hash=hash_) == "model_type"
    assert cache.validate(args[0], args[1], "torch.bfloat16",
                          vae_latent_hash=hash_) == "training_dtype"
    cache.cache_info_path.unlink()
    assert cache.validate(*args, vae_latent_hash=hash_) == "no_cache_info"


def test_a_crash_mid_stamp_leaves_the_previous_stamp_intact(monkeypatch):
    # A torn cache_info.json reads as absent, and the next run answers that with
    # a full re-encode.
    vae = _tiny_vae()
    cache = _setup(_trainer(vae))["ds1"]

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(lc, "json", SimpleNamespace(dump=boom))
    with pytest.raises(OSError):
        cache.save_cache_info("m", "sdxl", 1, "torch.float16", vae_latent_hash="x")
    monkeypatch.undo()

    assert cache.load_cache_info()["vae_latent_hash"] == module_latent_hash(vae)
    assert list(cache.cache_dir.glob("*.tmp")) != []  # the debris is the temp file


# --- the VAE identity is not always computable ------------------------------

def test_a_run_without_a_vae_records_no_identity_and_still_matches_itself():
    # Pixel-space archs (MiniT2I) have no VAE; two such runs agree rather than
    # re-encoding forever.
    cache = _setup(_trainer(None))["ds1"]
    _write_latent(cache)
    assert cache.load_cache_info()["vae_latent_hash"] is None

    assert _setup(_trainer(None))["ds1"].has_latent("a.png", 512, 512)


def test_an_unavailable_identity_does_not_adopt_a_labelled_cache(capsys):
    _write_latent(_setup(_trainer(_tiny_vae()))["ds1"])
    capsys.readouterr()

    again = _setup(_trainer(None))["ds1"]

    assert not again.has_latent("a.png", 512, 512)
    assert "current VAE identity unavailable" in capsys.readouterr().out
