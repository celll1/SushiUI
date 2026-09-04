"""LoHa/LoKr/DoRA TRAINING round trip, per architecture, on CPU in seconds.

For every architecture whose ``TRAINABLE_ADAPTER_PAIRS`` row carries the pair:
the REAL training adapter builds the algebra, the REAL ``save_checkpoint``
writes it, the REAL generation loader reads it back on a freshly built stub, and
the installed branch's delta must equal the trained layer's exactly. Then the
REAL ``LoRATrainer.load_checkpoint`` resumes it into a third model.

The stub trees, target sets and generation entry points are IMPORTED from each
architecture's ordinary-LoRA gate rather than copied: a divergence between what
this file thinks the architecture looks like and what that file does would make
a pass here meaningless.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/adapter_lycoris_training_roundtrip_cheap_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from lora_roundtrip_common import randomise_branch_tensors  # noqa: E402

from core.adapters import (  # noqa: E402
    DoRALinearLayer, LoHaLinearLayer, LoKrLinearLayer, LoRALinearLayer,
    TRAINABLE_ADAPTER_PAIRS,
)
from core.adapters.layers import is_adapter_wrapper  # noqa: E402

import acestep_lora_roundtrip_cheap_test as acestep_gate  # noqa: E402
import anima_lora_roundtrip_cheap_test as anima_gate  # noqa: E402
import flux2_lora_roundtrip_cheap_test as flux2_gate  # noqa: E402
import ideogram4_lora_roundtrip_cheap_test as ideogram4_gate  # noqa: E402
import krea2_lora_roundtrip_cheap_test as krea2_gate  # noqa: E402
import lens_lora_roundtrip_cheap_test as lens_gate  # noqa: E402
import ltx2_lora_roundtrip_cheap_test as ltx2_gate  # noqa: E402
import minit2i_lora_roundtrip_cheap_test as minit2i_gate  # noqa: E402
import zimage_lora_roundtrip_cheap_test as zimage_gate  # noqa: E402

RANK = 4
ALPHA = 6  # alpha/rank = 1.5, so a dropped or doubled scale shows up

BRANCH_CLASS = {"loha": LoHaLinearLayer, "lokr": LoKrLinearLayer}


def branch_class(algorithm, weight_decompose):
    """The class the run's ``(algorithm, weight_decompose)`` pair builds.

    The decomposition is an EPILOGUE: what a decomposed run installs is a
    ``DoRALinearLayer`` over the algebra's own layer, so the outer class is the
    same for all three pairs and the inner one is what differs."""
    if weight_decompose:
        return DoRALinearLayer
    return BRANCH_CLASS[algorithm]


class Arch:
    """One architecture's three moves: train, load for generation, name it."""

    def __init__(self, name, build, adapter, load, primary=lambda models: models[0]):
        self.name = name
        self._build = build          # () -> tuple of fresh component models
        self._adapter = adapter      # (models, algorithm) -> adapter
        self._load = load            # (models, path) -> applied count
        self._primary = primary      # models -> the model the deltas live on

    def train(self, algorithm, seed=1234, weight_decompose=False):
        models = self._build()
        adapter = self._adapter(models, (algorithm, weight_decompose))
        layers = {}
        adapter.apply_lora_to_unet(layers)
        adapter.apply_lora_to_text_encoders(layers)
        assert layers, f"{self.name}: nothing was wrapped"
        randomise_branch_tensors(layers, seed=seed)
        return adapter, layers, models

    def load(self, models, path):
        return self._load(models, path)


def _stub(**kwargs):
    return SimpleNamespace(config={}, **kwargs)


def _spec(trainer, pair):
    trainer.adapter_algorithm, trainer.weight_decompose = pair
    return trainer


# --- the table. One row per architecture, built out of its own gate ---------

def _krea2(models, algorithm):
    return krea2_gate.Krea2LoRAAdapter(
        _spec(krea2_gate._StubTrainer(models[0]), algorithm),
        lora_rank=RANK, lora_alpha=ALPHA, lora_dtype=torch.float32)


def _zimage(models, algorithm):
    return zimage_gate.ZImageLoRAAdapter(
        _spec(zimage_gate._StubTrainer(models[0]), algorithm),
        lora_rank=RANK, lora_alpha=ALPHA, lora_dtype=torch.float32)


def _anima(models, algorithm):
    trainer = _stub(transformer=models[0], blockskip_config=None)
    return anima_gate.AnimaLoRAAdapter(_spec(trainer, algorithm), RANK, ALPHA,
                                       torch.float32)


def _lens(models, algorithm):
    return lens_gate.LensLoRAAdapter(
        _spec(_stub(transformer=models[0]), algorithm), RANK, ALPHA,
        torch.float32)


def _ltx2(models, algorithm):
    return ltx2_gate.Ltx2LoRAAdapter(
        _spec(_stub(transformer=models[0]), algorithm), RANK, ALPHA,
        torch.float32)


def _acestep(models, algorithm):
    return acestep_gate.AceStepLoRAAdapter(
        _spec(_stub(transformer=models[0]), algorithm), RANK, ALPHA,
        torch.float32, scope=acestep_gate.ATTN_AND_MLP)


def _ideogram4(models, algorithm):
    trainer = _stub(transformer=models[0], transformer_uncond=None,
                    ideogram4_train_uncond=False)
    return ideogram4_gate.Ideogram4LoRAAdapter(_spec(trainer, algorithm), RANK,
                                               ALPHA, torch.float32)


def _minit2i(models, algorithm):
    trainer = _stub(transformer=models[0], text_encoder=models[1],
                    minit2i_variant="b16", repa_enable=False, repa_projector=None)
    return minit2i_gate.MiniT2ILoRAAdapter(_spec(trainer, algorithm), RANK,
                                           ALPHA, torch.float32)


def _flux2(models, algorithm):
    trainer = _stub(train_text_encoder=True, transformer=models[0],
                    text_encoder=models[1], unet_lr=1e-4, text_encoder_1_lr=1e-5)
    return flux2_gate.FLUX2LoRAAdapter(_spec(trainer, algorithm),
                                       lora_rank=RANK, lora_alpha=ALPHA,
                                       lora_dtype=torch.float32)


def _minit2i_load(models, path):
    backend = minit2i_gate._Backend(models[0], models[1])
    prepared = backend._minit2i_prepare_loras([{"path": path, "strength": 1.0}])
    backend._apply_te_lora_minit2i(prepared)
    return backend._load_lora_minit2i(prepared)


ARCHES = [
    Arch("krea2", lambda: (krea2_gate.build_model(),), _krea2,
         lambda m, p: krea2_gate._Backend(m[0])._load_lora_krea2(
             [{"path": p, "strength": 1.0}])),
    Arch("zimage", lambda: (zimage_gate.build_model(),), _zimage,
         lambda m, p: zimage_gate._Backend(m[0])._load_lora_zimage(
             [{"path": p, "strength": 1.0}])),
    Arch("anima", lambda: (anima_gate.build_model(),), _anima,
         lambda m, p: anima_gate._Backend(m[0])._load_lora_anima(
             [{"path": p, "strength": 1.0}])),
    Arch("lens", lambda: (lens_gate.build_model(),), _lens,
         lambda m, p: lens_gate._Backend(m[0])._load_lora_lens(
             [{"path": p, "strength": 1.0}])),
    Arch("ltx2", lambda: (ltx2_gate.build_dit(),), _ltx2,
         lambda m, p: ltx2_gate._Backend(m[0])._load_lora_ltx2(
             [{"path": p, "strength": 1.0}])),
    Arch("acestep", lambda: (acestep_gate.build_dit(),), _acestep,
         lambda m, p: acestep_gate._Backend(m[0])._load_lora_acestep(
             [{"path": p, "strength": 1.0}])),
    Arch("ideogram4", lambda: (ideogram4_gate._Stub(),), _ideogram4,
         lambda m, p: ideogram4_gate._Backend(m[0])._load_lora_ideogram4(
             [{"path": p, "strength": 1.0}])),
    Arch("minit2i",
         lambda: (minit2i_gate._Transformer(), minit2i_gate._TextEncoder()),
         _minit2i, _minit2i_load),
    Arch("flux2",
         lambda: (flux2_gate._Transformer(), flux2_gate._TextEncoder()),
         _flux2,
         lambda m, p: flux2_gate._Backend(m[0], m[1])._load_lora_flux2(
             [{"path": p, "strength": 1.0}])),
]

ALGEBRAS = ["loha", "lokr"]
#: Phase 3's dense-DoRA training rows. A subset, and the eight LoHa/LoKr rows
#: beside them are what keep it from reading as "decomposition is on".
DENSE_DORA = {"zimage", "lens", "minit2i"}
ROWS = ([(arch, algorithm, False) for arch in ARCHES for algorithm in ALGEBRAS]
        + [(arch, "lora", True) for arch in ARCHES if arch.name in DENSE_DORA])
IDS = [f"{arch.name}-{algorithm}{'+wd' if wd else ''}"
       for arch, algorithm, wd in ROWS]


def branch_paths(model, algorithm, weight_decompose=False):
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, branch_class(algorithm, weight_decompose)))


def installed_branch(model, path):
    """The one branch a generation load put over ``path``."""
    composite = model.get_submodule(path)
    assert is_adapter_wrapper(composite), f"{path} is not wrapped"
    names = composite.branch_names
    assert len(names) == 1, names
    return composite.get_branch(names[0])


@pytest.mark.parametrize("arch,algorithm,wd", ROWS, ids=IDS)
def test_the_row_this_file_gates_is_open(arch, algorithm, wd):
    """The table says these rows train; this file is what says so honestly."""
    assert (algorithm, wd) in TRAINABLE_ADAPTER_PAIRS[arch.name]


@pytest.mark.parametrize("arch,algorithm,wd", ROWS, ids=IDS)
def test_the_trainer_builds_the_algebra_the_run_asked_for(arch, algorithm, wd):
    _adapter, layers, _models = arch.train(algorithm, weight_decompose=wd)
    expected = branch_class(algorithm, wd)
    wrong = {name: type(layer).__name__ for name, layer in layers.items()
             if not isinstance(layer, expected)}
    assert not wrong, wrong
    if wd:
        inner = {type(layer.branch).__name__ for layer in layers.values()}
        assert inner == {BRANCH_CLASS.get(algorithm, LoRALinearLayer).__name__}


@pytest.mark.parametrize("arch,algorithm,wd", ROWS, ids=IDS)
def test_a_trained_checkpoint_loads_back_with_the_same_delta(arch, algorithm, wd,
                                                             tmp_path):
    adapter, layers, models = arch.train(algorithm, weight_decompose=wd)
    path = str(tmp_path / f"{arch.name}_{algorithm}.safetensors")
    adapter.save_checkpoint(layers, 100, 1, path)

    fresh = arch._build()
    applied = arch.load(fresh, path)
    # Z-Image's loader returns nothing; the per-target loop below is what
    # actually proves the coverage, on every architecture.
    assert applied in (None, len(layers)), (applied, len(layers))

    trained_model = arch._primary(models)
    paths = branch_paths(trained_model, algorithm, wd)
    assert paths, f"{arch.name}: the trainer wrapped nothing on the primary model"
    generator = torch.Generator().manual_seed(99)
    for module_path in paths:
        trained = trained_model.get_submodule(module_path)
        branch = installed_branch(arch._primary(fresh), module_path)
        assert isinstance(branch, branch_class(algorithm, wd)), type(branch).__name__
        x = torch.randn(2, trained.original_module.in_features,
                        generator=generator)
        # torch.equal, not allclose: the file's tensors are the trained ones and
        # both sides run the same algebra at the same scale, so any difference
        # is a lost alpha, a transposed factor or a folded strength.
        assert torch.equal(trained.forward_delta(x), branch.forward_delta(x)), \
            f"{arch.name}/{algorithm}: {module_path}"


@pytest.mark.parametrize("arch,algorithm,wd", ROWS, ids=IDS)
def test_the_checkpoint_declares_its_own_algebra(arch, algorithm, wd, tmp_path):
    adapter, layers, _models = arch.train(algorithm, weight_decompose=wd)
    path = str(tmp_path / f"{arch.name}_{algorithm}.safetensors")
    adapter.save_checkpoint(layers, 100, 1, path)
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    assert metadata.get("sushi.adapter.algorithm") == algorithm
    assert metadata.get("sushi.adapter.weight_decompose") == ("true" if wd
                                                              else "false")
    saved = load_file(path)
    # The live view's training-only key must not reach the file.
    assert not [k for k in saved if k.endswith(".scalar")]
    if wd:
        # One magnitude per wrapped target, in the LyCORIS spelling. Without
        # this a run could train a magnitude it never wrote.
        assert (len([k for k in saved if k.endswith(".dora_scale")])
                == len(layers))
    else:
        assert any(k.endswith(".alpha") for k in saved), \
            "a LyCORIS reader takes the scale from the per-key alpha"


@pytest.mark.parametrize("arch,algorithm,wd", ROWS, ids=IDS)
def test_resume_restores_every_factor(arch, algorithm, wd, tmp_path):
    from core.training.lora_trainer import LoRATrainer

    adapter, layers, _models = arch.train(algorithm, seed=7, weight_decompose=wd)
    path = str(tmp_path / f"{arch.name}_{algorithm}.safetensors")
    adapter.save_checkpoint(layers, 250, 3, path)
    saved = {name: {k: v.detach().clone()
                    for k, v in layer.branch_tensors().items()}
             for name, layer in layers.items()}
    if wd:
        assert all("dora_scale" in slice_ for slice_ in saved.values())

    _resumed_adapter, resumed_layers, _m = arch.train(algorithm, seed=555,
                                                      weight_decompose=wd)
    trainer = SimpleNamespace(lora_layers=resumed_layers, log_prefix="[test]",
                              lora_rank=RANK, lora_alpha=ALPHA)
    assert LoRATrainer.load_checkpoint(trainer, path) == 250
    for name, layer in resumed_layers.items():
        for key, tensor in layer.branch_tensors().items():
            assert torch.equal(tensor, saved[name][key]), f"{name}.{key}"
