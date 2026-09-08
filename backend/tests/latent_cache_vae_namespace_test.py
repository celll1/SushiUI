"""Which directory a latent cache lives in, so switching VAEs and back is free.

The VAE keys the LATENTS only: ``{dataset}/{arch}/vae-<latent_hash>/latents``,
under an architecture namespace the text-embedding cache shares and the VAE
never touches. Two VAEs therefore keep two caches instead of taking turns
deleting each other's, and the stamp checked in
``latent_cache_vae_identity_test`` stays the defence for what the path cannot
separate. Rationale: API_REFERENCE.md, "Cache Directory Structure".
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
from core.training.vae_swap import module_latent_hash


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


def _trainer(vae, model_path="M:/model/sdxl/base_a.safetensors", **attrs):
    """Drives the REAL ``_build_cache_namespace``: the point under test is which
    of its components move when the VAE does."""
    stub = SimpleNamespace(
        log_prefix="[Trainer]", model_path=model_path, vae=vae,
        training_dtype=torch.float16, arch=SimpleNamespace(name="sdxl"),
        vae_latent_channels=4, vae_dtype=torch.float16, sdxl_te_type="none",
        is_sdxl=True, is_zimage=False, is_lens=False,
    )
    for name, value in attrs.items():
        setattr(stub, name, value)
    for method in ("_build_cache_namespace", "_run_vae_identity",
                   "_log_other_vae_caches", "_setup_latent_caches",
                   "_setup_text_encoder_caches"):
        setattr(stub, method, MethodType(getattr(BaseTrainer, method), stub))
    return stub


def _dataset(unique_id="ds1", count=2):
    return SimpleNamespace(unique_id=unique_id, items=[{}] * count)


def _setup(trainer):
    return trainer._setup_latent_caches([_dataset()])["ds1"]


def _write_latent(cache, image_path="a.png", size=512, value=0.0):
    cache.save_latent(image_path, size, size,
                      torch.full((1, 4, size // 8, size // 8), value))


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "get_cache_base_dir", lambda: str(tmp_path))
    return tmp_path


# --- (a) two VAEs, two directories ------------------------------------------

def test_two_vaes_get_two_directories_under_one_arch_namespace(capsys):
    first, second = _tiny_vae(), _tiny_vae()
    cache_a = _setup(_trainer(first))
    _write_latent(cache_a, value=1.0)
    capsys.readouterr()

    cache_b = _setup(_trainer(second))

    assert cache_a.cache_dir != cache_b.cache_dir
    assert cache_a.arch_cache_dir == cache_b.arch_cache_dir
    assert cache_b.cache_dir.name == f"vae-{module_latent_hash(second)}"
    # Nothing of A's was deleted to make room for B's.
    assert "cached latents" not in capsys.readouterr().out
    assert cache_a.has_latent("a.png", 512, 512)
    assert cache_a.load_cache_info()["vae_latent_hash"] == module_latent_hash(first)


def test_a_run_that_cannot_name_its_vae_falls_into_one_shared_bucket():
    assert _setup(_trainer(None)).cache_dir.name == "vae-unknown"


# --- (b) switching away and back is free ------------------------------------

def test_switching_back_to_the_first_vae_reuses_its_latents(capsys):
    first, second = _tiny_vae(), _tiny_vae()
    _write_latent(_setup(_trainer(first)), value=1.0)

    _setup(_trainer(second))          # the comparison run
    back = _setup(_trainer(first))    # and back

    assert back.has_latent("a.png", 512, 512)
    assert torch.equal(back.load_latent("a.png", 512, 512, device="cpu"),
                       torch.ones(1, 4, 64, 64))
    assert "Validation passed" in capsys.readouterr().out


def test_without_a_vae_token_the_round_trip_re_encodes(monkeypatch):
    # What one directory per (dataset, arch) cost, measured against the same
    # scenario: the previous layout deleted A's latents to make room for B's.
    monkeypatch.setattr(lc, "vae_cache_namespace", lambda _hash: "vae-collision")
    first, second = _tiny_vae(), _tiny_vae()
    _write_latent(_setup(_trainer(first)), value=1.0)

    _setup(_trainer(second))

    assert not _setup(_trainer(first)).has_latent("a.png", 512, 512)


def test_switching_back_survives_a_swap_declared_on_only_one_of_the_runs():
    # The directory is a function of the live module, so the same VAE addresses
    # the same latents whether or not that run declared a swap for it.
    vae = _tiny_vae()
    declared = _trainer(vae, vae_identity=SimpleNamespace(
        family="flux1", latent_hash="deadbeefdeadbeef", identity_native=False))
    _write_latent(_setup(declared), value=1.0)

    assert _setup(_trainer(vae)).has_latent("a.png", 512, 512)


# --- (c) text embeddings do not depend on the VAE ---------------------------

def test_text_embeddings_stay_put_when_the_vae_changes():
    first, second = _tiny_vae(), _tiny_vae()
    trainer_a, trainer_b = _trainer(first), _trainer(second)

    embeddings = trainer_a._setup_text_encoder_caches([_dataset()])["ds1"]
    (embeddings / "cafe_clip1.pt").write_bytes(b"x")

    assert trainer_b._setup_text_encoder_caches([_dataset()])["ds1"] == embeddings
    # A declared swap does not move them either.
    trainer_b.vae_identity = SimpleNamespace(family="flux1", identity_native=False,
                                             latent_hash="deadbeefdeadbeef")
    assert trainer_b._setup_text_encoder_caches([_dataset()])["ds1"] == embeddings
    assert (embeddings / "cafe_clip1.pt").is_file()


def test_the_latent_cache_points_at_that_same_text_embedding_directory():
    trainer = _trainer(_tiny_vae())
    cache = _setup(trainer)

    assert cache.embeddings_dir == trainer._setup_text_encoder_caches([_dataset()])["ds1"]
    assert cache.embeddings_dir.parent == cache.arch_cache_dir
    assert cache.cache_dir not in cache.embeddings_dir.parents


# --- (d) what the switch left behind ----------------------------------------

def test_the_listing_names_every_vae_cache_with_what_it_cost(tmp_path):
    first, second = _tiny_vae(), _tiny_vae()
    cache_a = _setup(_trainer(first))
    _write_latent(cache_a, "a.png")
    _write_latent(cache_a, "b.png")
    cache_b = _setup(_trainer(second, model_path="M:/model/sdxl/base_b.safetensors",
                              vae_identity=SimpleNamespace(family="flux1")))
    _write_latent(cache_b)

    found = {e["path"]: e for e in lc.list_vae_namespaces(str(tmp_path), "ds1")}

    assert set(found) == {cache_a.cache_dir, cache_b.cache_dir}
    assert found[cache_a.cache_dir]["entries"] == 2
    assert found[cache_a.cache_dir]["bytes"] > 0
    assert found[cache_a.cache_dir]["vae_latent_hash"] == module_latent_hash(first)
    assert found[cache_a.cache_dir]["vae_family"] is None
    assert found[cache_a.cache_dir]["namespace"] == cache_a.arch_cache_dir.name
    assert found[cache_b.cache_dir]["vae_family"] == "flux1"
    assert found[cache_b.cache_dir]["model_path"] == "M:/model/sdxl/base_b.safetensors"


def test_an_empty_namespace_is_not_reported(capsys):
    vae = _tiny_vae()
    _setup(_trainer(None))  # a vae-unknown directory that never got an entry
    _write_latent(_setup(_trainer(vae)))
    capsys.readouterr()

    _setup(_trainer(vae))

    assert "other VAE latent cache" not in capsys.readouterr().out


def test_the_setup_pass_points_at_the_other_caches_and_deletes_nothing(capsys):
    first, second = _tiny_vae(), _tiny_vae()
    cache_a = _setup(_trainer(first))
    _write_latent(cache_a, value=1.0)
    capsys.readouterr()

    _setup(_trainer(second))

    out = capsys.readouterr().out
    assert "1 other VAE latent cache(s) for dataset 'ds1'" in out
    assert "KEPT, not deleted" in out
    assert str(cache_a.cache_dir) in out
    assert f"VAE {module_latent_hash(first)}" in out
    assert "1 latents" in out
    assert cache_a.has_latent("a.png", 512, 512)
