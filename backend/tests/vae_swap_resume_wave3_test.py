"""Resuming an anima / lens checkpoint (follow-up to design phase P7).

Neither architecture had a ``_load_checkpoint_as_base`` branch: both fell
through to the SD/SDXL arm, which read their DiT save as a Stable Diffusion
single file. ``resume_from_checkpoint`` was broken for them independently of the
VAE swap, and a swapped base additionally lost its latent identity.

The bar is ``bf917adb``'s: SAVE -> RESUME -> SAVE leaves ``component.vae.*``
byte-identical, and a native base resumes exactly as it does today.

CPU only. The DiT construction is real (anima builds through the loader's own
``init_empty_weights`` + ``assign=True`` path against a tiny config); only the
companion Qwen3 / T5 / base-directory reads are stubbed, since those are files
this test has no business opening.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.anima import anima_loader as al
from core.models.anima.anima_models import ANIMA_DIT_CONFIG, Anima
from core.models.common import vae_source as vs
from core.models.components.latent_io import resize_latent_io
from core.models.components.wiring import LENS_WIRING
from core.models.lens import lens_loader as ll
from core.training.base_trainer import BaseTrainer

# Same architecture, 1 block wide: the loader builds from this module-level
# constant, so patching it is what keeps the real load path in the test.
TINY_ANIMA_CONFIG = dict(
    ANIMA_DIT_CONFIG,
    max_img_h=64, max_img_w=64, max_frames=8, model_channels=64, num_blocks=1,
    num_heads=4, crossattn_emb_channels=32, adaln_lora_dim=16,
    use_llm_adapter=False,
)


# --- fixtures ---------------------------------------------------------------

def _anima(channels=16):
    return Anima(**dict(TINY_ANIMA_CONFIG, in_channels=channels,
                        out_channels=channels))


def _lens(in_channels=128, out_channels=32):
    from core.models.lens.vendor.transformer import LensTransformer2DModel
    return LensTransformer2DModel(
        patch_size=2, in_channels=in_channels, out_channels=out_channels,
        num_layers=1, attention_head_dim=8, num_attention_heads=2, inner_dim=16,
        enc_hidden_dim=16, axes_dims_rope=(2, 4, 2), selected_layer_index=(0,))


def _tiny_vae(latent_channels):
    from diffusers import AutoencoderKL
    return AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4), layers_per_block=1, norm_num_groups=4,
        latent_channels=latent_channels, sample_size=32)


def _standalone_vae_dir(tmp_path, latent_channels, name="swap_vae"):
    directory = tmp_path / name
    _tiny_vae(latent_channels).save_pretrained(str(directory))
    return str(directory)


def _declaration(path):
    """Every metadata key that says which latent space this checkpoint is in."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {k: v for k, v in metadata.items()
            if k.startswith("component.vae.") or k.startswith("sushi.")}


def _resume_trainer(model_path, **overrides):
    trainer = SimpleNamespace(
        log_prefix="[test]", device="cpu", model_path=str(model_path),
        weight_dtype=torch.float32, vae_dtype=torch.float32, config={},
        gradient_checkpointing=False, blocks_to_swap=0,
        use_flash_attention=False, attention_backend="native",
        bundle_vae=None, train_unet=True, train_text_encoder=False,
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


# --- anima ------------------------------------------------------------------

@pytest.fixture
def anima_companions(monkeypatch, tmp_path):
    """Stub the companion reads and record the path discovery was aimed at."""
    monkeypatch.setattr(al, "ANIMA_DIT_CONFIG", TINY_ANIMA_CONFIG)
    monkeypatch.setattr(al, "load_qwen3_text_encoder",
                        lambda *a, **k: (nn.Linear(2, 2), object()))
    monkeypatch.setattr(al, "load_t5_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(al, "load_qwen_image_vae",
                        lambda *a, **k: _tiny_vae(16))

    seen = {}
    te_file = tmp_path / "base" / "qwen_3_te.safetensors"
    te_file.parent.mkdir(parents=True, exist_ok=True)
    te_file.write_bytes(b"")
    vae_file = tmp_path / "base" / "qwen_image_vae.safetensors"
    vae_file.write_bytes(b"")

    def _discover(path, models_root=None, text_encoder_override=None,
                  vae_override=None):
        seen["path"] = str(path)
        return {"dit": str(path), "text_encoder": str(te_file),
                "vae": str(vae_file)}

    monkeypatch.setattr(al, "discover_anima_components", _discover)
    return SimpleNamespace(seen=seen, base=str(tmp_path / "base" / "anima.safetensors"))


def _save_anima(trainer, path, step=10):
    from core.training.adapters.anima_adapter import AnimaFullParameterAdapter
    AnimaFullParameterAdapter(trainer).save_checkpoint(step, 1, path)
    return str(path)


def test_an_anima_resume_rebuilds_the_swapped_latent_space_and_redeclares_it(
        tmp_path, anima_companions):
    vae = _tiny_vae(4)
    identity = vs.resolve_vae_source(
        f"file:{_standalone_vae_dir(tmp_path, 4)}", arch="anima")
    trained = _anima(4)
    first = _save_anima(
        SimpleNamespace(transformer=trained, vae=vae, bundle_vae=None,
                        vae_identity=identity,
                        arch=SimpleNamespace(name="anima")),
        tmp_path / "anima_step10.safetensors")
    declaration = _declaration(first)
    assert declaration["component.vae.channels"] == "4"
    assert declaration["component.vae.identity_native"] == "0"

    trainer = _resume_trainer(anima_companions.base)
    BaseTrainer._load_checkpoint_as_base(trainer, first)
    trainer.arch = SimpleNamespace(name="anima")

    assert trainer.is_anima is True
    # The DiT was rebuilt at the DECLARED width, not anima's own 16.
    assert trainer.transformer.x_embedder.proj[1].in_features == \
        (4 + 1) * TINY_ANIMA_CONFIG["patch_spatial"] ** 2
    assert trainer.transformer.final_layer.linear.out_features == \
        4 * TINY_ANIMA_CONFIG["patch_spatial"] ** 2
    # The trained weights, not a fresh build.
    assert torch.equal(trainer.transformer.x_embedder.proj[1].weight.float(),
                       trained.x_embedder.proj[1].weight.float())
    assert trainer.vae_identity is not None
    assert trainer.vae_identity.identity_native is False
    assert trainer.vae_identity.content_hash == identity.content_hash
    assert trainer.vae_latent_channels == 4
    assert trainer.wiring.latent_channels == 4
    assert trainer.vae.config.latent_channels == 4
    # Facts only: the session must not hold a second copy of the VAE weights.
    assert trainer.vae_identity.state_dict is None
    # Companion discovery was aimed at the BASE model, not at the checkpoint's
    # own directory (which holds nothing but other checkpoints).
    assert anima_companions.seen["path"] == anima_companions.base

    second = _save_anima(trainer, tmp_path / "anima_step20.safetensors", step=20)
    assert _declaration(second) == declaration


def test_a_native_anima_checkpoint_resumes_in_its_own_latent_space(
        tmp_path, anima_companions):
    trained = _anima(16)
    first = _save_anima(
        SimpleNamespace(transformer=trained, vae=_tiny_vae(16), bundle_vae=False,
                        vae_identity=None, arch=SimpleNamespace(name="anima")),
        tmp_path / "anima_native10.safetensors")
    declaration = _declaration(first)
    assert vs.load_declared_latent_io(first, arch="anima") is None

    trainer = _resume_trainer(anima_companions.base, bundle_vae=False)
    BaseTrainer._load_checkpoint_as_base(trainer, first)
    trainer.arch = SimpleNamespace(name="anima")

    assert getattr(trainer, "vae_identity", None) is None
    assert getattr(trainer, "base_vae_identity", None) is None
    assert trainer.transformer.final_layer.linear.out_features == \
        16 * TINY_ANIMA_CONFIG["patch_spatial"] ** 2
    assert torch.equal(trainer.transformer.final_layer.linear.weight.float(),
                       trained.final_layer.linear.weight.float())

    second = _save_anima(trainer, tmp_path / "anima_native20.safetensors", step=20)
    assert _declaration(second) == declaration
    assert vs.load_declared_latent_io(second, arch="anima") is None


def test_the_anima_resume_reads_the_dit_from_the_checkpoint_not_the_base(
        tmp_path, anima_companions):
    """The companion path moves the TE/VAE search only. Aiming the DiT read at
    it too would resume from the base model and silently discard the run."""
    trained = _anima(16)
    first = _save_anima(
        SimpleNamespace(transformer=trained, vae=_tiny_vae(16), bundle_vae=False,
                        vae_identity=None, arch=SimpleNamespace(name="anima")),
        tmp_path / "anima_only10.safetensors")

    trainer = _resume_trainer(anima_companions.base)
    BaseTrainer._load_checkpoint_as_base(trainer, first)

    assert trainer.model_path == first
    assert trainer.anima_companion_path == anima_companions.base
    assert torch.equal(trainer.transformer.x_embedder.proj[1].weight.float(),
                       trained.x_embedder.proj[1].weight.float())

    # The corrupted-checkpoint fallback re-enters the branch with model_path
    # already pointing at the checkpoint that failed.
    BaseTrainer._load_checkpoint_as_base(trainer, first)
    assert trainer.anima_companion_path == anima_companions.base


# --- lens -------------------------------------------------------------------

@pytest.fixture
def lens_base(monkeypatch, tmp_path):
    """A base Lens diffusers directory, with its component load stubbed."""
    base_dir = tmp_path / "lens_base"
    (base_dir / "transformer").mkdir(parents=True)
    (base_dir / "transformer" / "config.json").write_text("{}", encoding="utf-8")

    def _components(model_path, torch_dtype=torch.float32, **kwargs):
        return {"transformer": _lens(128, 32), "vae": _tiny_vae(32),
                "text_encoder": nn.Linear(2, 2), "tokenizer": object(),
                "scheduler": object(), "base_dir": str(model_path),
                "vae_source": str(model_path), "vae_path": None}

    monkeypatch.setattr(ll, "load_lens_components", _components)
    return str(base_dir)


def _save_lens(trainer, path, step=10):
    from core.training.adapters.lens_adapter import LensFullParameterAdapter
    LensFullParameterAdapter(trainer).save_checkpoint(step, 1, path)
    return str(path)


def test_a_lens_resume_rebuilds_the_swapped_latent_space_and_redeclares_it(
        tmp_path, lens_base):
    vae = _tiny_vae(4)
    identity = vs.resolve_vae_source(
        f"file:{_standalone_vae_dir(tmp_path, 4)}", arch="lens")
    trained = _lens(128, 32)
    resize_latent_io(trained, LENS_WIRING.latent_io, 4)
    first = _save_lens(
        SimpleNamespace(transformer=trained, vae=vae, bundle_vae=None,
                        vae_identity=identity, lens_base_dir=lens_base,
                        arch=SimpleNamespace(name="lens")),
        tmp_path / "lens_step10.safetensors")
    declaration = _declaration(first)
    assert declaration["component.vae.channels"] == "4"
    assert declaration["component.vae.identity_native"] == "0"

    trainer = _resume_trainer(lens_base)
    BaseTrainer._load_checkpoint_as_base(trainer, first)
    trainer.arch = SimpleNamespace(name="lens")

    assert trainer.is_lens is True
    # Lens counts its input packed and its output raw (design §5.1).
    assert trainer.transformer.img_in.in_features == 4 * 4
    assert trainer.transformer.proj_out.out_features == 4 * 4
    assert torch.equal(trainer.transformer.img_in.weight.float(),
                       trained.img_in.weight.float())
    assert trainer.vae_identity is not None
    assert trainer.vae_identity.identity_native is False
    assert trainer.vae_identity.content_hash == identity.content_hash
    assert trainer.vae_latent_channels == 4
    assert trainer.wiring.latent_channels == 4
    assert trainer.vae.config.latent_channels == 4
    assert trainer.vae_identity.state_dict is None
    assert trainer.lens_base_dir == lens_base

    second = _save_lens(trainer, tmp_path / "lens_step20.safetensors", step=20)
    assert _declaration(second) == declaration


def test_a_native_lens_checkpoint_resumes_in_its_own_latent_space(
        tmp_path, lens_base):
    trained = _lens(128, 32)
    first = _save_lens(
        SimpleNamespace(transformer=trained, vae=_tiny_vae(32), bundle_vae=None,
                        vae_identity=None, lens_base_dir=lens_base,
                        arch=SimpleNamespace(name="lens")),
        tmp_path / "lens_native10.safetensors")
    declaration = _declaration(first)
    assert vs.load_declared_latent_io(first, arch="lens") is None

    trainer = _resume_trainer(lens_base)
    BaseTrainer._load_checkpoint_as_base(trainer, first)
    trainer.arch = SimpleNamespace(name="lens")

    assert getattr(trainer, "vae_identity", None) is None
    assert getattr(trainer, "base_vae_identity", None) is None
    assert trainer.transformer.img_in.in_features == 128
    assert trainer.transformer.proj_out.out_features == 128
    assert torch.equal(trainer.transformer.img_in.weight.float(),
                       trained.img_in.weight.float())
    assert trainer.vae.config.latent_channels == 32

    second = _save_lens(trainer, tmp_path / "lens_native20.safetensors", step=20)
    assert _declaration(second) == declaration
    assert vs.load_declared_latent_io(second, arch="lens") is None


def test_a_lens_checkpoint_is_not_read_as_a_stable_diffusion_single_file(
        tmp_path, lens_base):
    """What the missing branch actually did: detect_model_type calls it lens,
    and the resume arm has to agree or from_single_file gets a DiT."""
    from core.model_loader import ModelLoader

    path = _save_lens(
        SimpleNamespace(transformer=_lens(128, 32), vae=_tiny_vae(32),
                        bundle_vae=None, vae_identity=None,
                        lens_base_dir=lens_base,
                        arch=SimpleNamespace(name="lens")),
        tmp_path / "lens_detect.safetensors")
    assert ModelLoader.detect_model_type(path) == "lens"
