"""The training adapters' wrapper tests must recognise a CompositeAdapterLayer.

Every training adapter decides what to wrap with a test for the trainer's own
``LoRALinearLayer``. A ``CompositeAdapterLayer`` fails all of them, in one of
three opposite ways:

  SKIP-test    ``if isinstance(current, LoRALinearLayer): continue`` does not
               skip a composite, so the composite is NESTED inside a fresh
               wrapper -- silently, because the composite exposes
               ``in_features``/``out_features`` and so constructs fine.
  INCLUDE-test ``is_lora_wrappable_linear(m) or isinstance(m, LoRALinearLayer)``
               rejects a composite, so an occupied target VANISHES from
               enumeration exactly when a second adapter needs it.
  DESCENT      a walk that selects ``nn.Linear`` by class descends INTO the
               composite and offers a branch weight (and the hidden base) as a
               target.

Training never meets a composite today -- it loads its own model and generation
restores in a ``finally`` -- so every assertion here is about a latent state,
and every arch's plain-tree control asserts the sweep changed nothing for a real
run. Each stub tree is imported from that architecture's own cheap round-trip
gate, so a tree that stops matching the loader is one failure, not two.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/training_adapter_composite_awareness_test.py -v
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from core.adapters import (  # noqa: E402
    CompositeAdapterLayer, LoRALinearLayer, get_module_slot, is_adapter_covered,
    named_modules_outside_adapters,
)

RANK, ALPHA = 4, 8


# --- helpers ---------------------------------------------------------------

def cover(parent, slot, branch_name="installed"):
    """Install a composite with one branch over ``parent[slot]``.

    Exactly what a generation load leaves behind, built through the same
    ``attach``/``add_branch`` the loaders call.
    """
    base = get_module_slot(parent, slot)
    composite = CompositeAdapterLayer.attach(parent, slot)
    composite.add_branch(branch_name,
                         LoRALinearLayer(base, RANK, ALPHA, "installed"),
                         strength=1.0)
    return composite


def cover_first(targets):
    """Cover the slot of the first enumerated target: ``(path, composite)``."""
    path, parent, attr, _current = targets[0]
    return path, cover(parent, attr)


def subtree_ids(module):
    return {id(m) for _name, m in module.named_modules()}


def wrapper_roots(model):
    """Slots holding an adapter, branches excluded (the walk stops at a root)."""
    return {name for name, m in named_modules_outside_adapters(model)
            if is_adapter_covered(m)}


def nested(model):
    """Wrappers whose base is itself a wrapper -- the nesting defect."""
    return {name for name, m in model.named_modules()
            if is_adapter_covered(m)
            and is_adapter_covered(getattr(m, "original_module", None))}


def assert_left_alone(model, parent, attr, composite, before_ids):
    """The composite is still in its slot, untouched inside, and not nested."""
    assert get_module_slot(parent, attr) is composite
    assert subtree_ids(composite) == before_ids
    assert nested(model) == set()


def trainer(**kwargs):
    return types.SimpleNamespace(**kwargs)


# --- ACE-Step: include-test (_is_target) + skip-test ------------------------

def test_acestep_enumerates_and_skips_an_occupied_target():
    from acestep_lora_roundtrip_cheap_test import build_dit
    from core.training.adapters.acestep_adapter import (
        DEFAULT_ACESTEP_SCOPE, AceStepLoRAAdapter, _is_target,
        iter_acestep_lora_targets,
    )

    plain = build_dit()
    total = AceStepLoRAAdapter(trainer(transformer=plain), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0
    assert len(wrapper_roots(plain)) == total

    dit = build_dit()
    targets = list(iter_acestep_lora_targets(dit, DEFAULT_ACESTEP_SCOPE))
    assert len(targets) == total
    path, parent, attr, _ = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    # INCLUDE: the predicate and the enumerator both keep the occupied slot.
    assert _is_target(composite)
    yielded = {p: cur for p, _pa, _a, cur in
               iter_acestep_lora_targets(dit, DEFAULT_ACESTEP_SCOPE)}
    assert yielded[path] is composite

    # SKIP: everything else is wrapped, the occupied slot is left alone.
    assert AceStepLoRAAdapter(trainer(transformer=dit), RANK, ALPHA
                              ).apply_lora_to_unet({}) == total - 1
    assert_left_alone(dit, parent, attr, composite, before)


# --- pure skip-tests over an already composite-aware model enumerator -------

def _skip_case(build_model, iter_targets, scope, adapter_cls, trainer_kwargs):
    plain = build_model()
    total = adapter_cls(trainer(**trainer_kwargs(plain)), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0

    model = build_model()
    targets = list(iter_targets(model, scope))
    assert len(targets) == total
    _path, parent, attr, _cur = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    count = adapter_cls(trainer(**trainer_kwargs(model)), RANK, ALPHA).apply_lora_to_unet({})
    assert count == total - 1
    assert_left_alone(model, parent, attr, composite, before)


def test_anima_skips_an_occupied_target():
    from anima_lora_roundtrip_cheap_test import build_model
    from core.models.anima.anima_lora import (
        DEFAULT_TRAINING_SCOPE, iter_anima_lora_targets,
    )
    from core.training.adapters.anima_adapter import AnimaLoRAAdapter

    _skip_case(build_model, iter_anima_lora_targets, DEFAULT_TRAINING_SCOPE,
               AnimaLoRAAdapter, lambda m: {"transformer": m})


def test_ideogram4_skips_an_occupied_target():
    from ideogram4_lora_roundtrip_cheap_test import _Stub
    from core.models.ideogram4.ideogram4_lora import (
        DEFAULT_SCOPE, iter_ideogram4_lora_targets,
    )
    from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter

    _skip_case(_Stub, iter_ideogram4_lora_targets, DEFAULT_SCOPE,
               Ideogram4LoRAAdapter, lambda m: {"transformer": m})


def test_krea2_skips_an_occupied_target():
    from krea2_lora_roundtrip_cheap_test import build_model
    from core.models.krea2.krea2_lora import DEFAULT_SCOPE, iter_krea2_lora_targets
    from core.training.adapters.krea2_adapter import Krea2LoRAAdapter

    _skip_case(build_model, iter_krea2_lora_targets, DEFAULT_SCOPE,
               Krea2LoRAAdapter, lambda m: {"transformer": m})


def test_lens_skips_an_occupied_target():
    from lens_lora_roundtrip_cheap_test import build_model
    from core.models.lens.lens_lora import DEFAULT_SCOPE, iter_lens_lora_targets
    from core.training.adapters.lens_adapter import LensLoRAAdapter

    _skip_case(build_model, iter_lens_lora_targets, DEFAULT_SCOPE,
               LensLoRAAdapter, lambda m: {"transformer": m})


def test_minit2i_skips_an_occupied_target():
    from minit2i_lora_roundtrip_cheap_test import _Transformer
    from core.models.minit2i.minit2i_lora import DEFAULT_SCOPE, iter_minit2i_lora_targets
    from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter

    _skip_case(_Transformer, iter_minit2i_lora_targets, DEFAULT_SCOPE,
               MiniT2ILoRAAdapter, lambda m: {"transformer": m})


def test_minit2i_text_encoder_skips_an_occupied_target():
    from minit2i_lora_roundtrip_cheap_test import _TextEncoder
    from core.models.minit2i.minit2i_lora import (
        TE_DEFAULT_SCOPE, iter_minit2i_te_lora_targets,
    )
    from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter

    plain = _TextEncoder()
    total = MiniT2ILoRAAdapter(trainer(text_encoder=plain), RANK, ALPHA
                               ).apply_lora_to_text_encoders({})
    assert total > 0

    te = _TextEncoder()
    targets = list(iter_minit2i_te_lora_targets(te, TE_DEFAULT_SCOPE))
    _path, parent, attr, _cur = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    assert MiniT2ILoRAAdapter(trainer(text_encoder=te), RANK, ALPHA
                              ).apply_lora_to_text_encoders({}) == total - 1
    assert_left_alone(te, parent, attr, composite, before)


# --- LTX-2.3: include-tests, feed-forward descent guard, skip-test ----------

_LTX2_FF_SCOPE = {"attention": True, "audio": False, "av_cross": False, "ff": True}


def test_ltx2_attention_include_test_keeps_an_occupied_target():
    from ltx2_lora_roundtrip_cheap_test import build_dit
    from core.training.adapters.ltx2_adapter import (
        DEFAULT_LTX2_SCOPE, Ltx2LoRAAdapter, iter_ltx2_lora_targets,
    )

    plain = build_dit()
    total = Ltx2LoRAAdapter(trainer(transformer=plain), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0

    dit = build_dit()
    targets = list(iter_ltx2_lora_targets(dit, DEFAULT_LTX2_SCOPE))
    path, parent, attr, _cur = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    yielded = {p: cur for p, _pa, _a, cur in
               iter_ltx2_lora_targets(dit, DEFAULT_LTX2_SCOPE)}
    assert yielded[path] is composite

    assert Ltx2LoRAAdapter(trainer(transformer=dit), RANK, ALPHA
                           ).apply_lora_to_unet({}) == total - 1
    assert_left_alone(dit, parent, attr, composite, before)


def test_ltx2_feed_forward_walk_does_not_descend_into_a_composite():
    from ltx2_lora_roundtrip_cheap_test import build_dit
    from core.training.adapters.ltx2_adapter import iter_ltx2_lora_targets

    plain = build_dit()
    baseline = [p for p, _pa, _a, _c in iter_ltx2_lora_targets(plain, _LTX2_FF_SCOPE)]

    dit = build_dit()
    ff_paths = [p for p in baseline if ".ff." in p]
    assert ff_paths, "the stub must have a feed-forward target for this to bite"
    target = ff_paths[0]
    parent = dit.transformer_blocks[0].ff.net
    slot = int(target.rsplit(".", 1)[1])
    composite = cover(parent, slot)

    yielded = {p: cur for p, _pa, _a, cur in iter_ltx2_lora_targets(dit, _LTX2_FF_SCOPE)}
    # The composite root is offered, its branch weights are not.
    assert yielded[target] is composite
    assert [p for p in yielded if p.startswith(target + ".")] == []
    assert sorted(yielded) == sorted(baseline)


# --- MiniMax-H3: include-test + skip-test ----------------------------------

def test_minimax_h3_enumerates_and_skips_an_occupied_target():
    from minimax_h3_lora_roundtrip_cheap_test import _Stub
    from core.training.adapters.minimax_h3_adapter import (
        DEFAULT_MINIMAX_H3_SCOPE, MiniMaxH3LoRAAdapter, iter_minimax_h3_lora_targets,
    )

    plain = _Stub()
    total = MiniMaxH3LoRAAdapter(trainer(transformer=plain), RANK, ALPHA
                                 ).apply_lora_to_unet({})
    assert total > 0

    model = _Stub()
    targets = list(iter_minimax_h3_lora_targets(model, DEFAULT_MINIMAX_H3_SCOPE))
    path, parent, attr, _cur = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    yielded = {p: cur for p, _pa, _a, cur in
               iter_minimax_h3_lora_targets(model, DEFAULT_MINIMAX_H3_SCOPE)}
    assert yielded[path] is composite

    assert MiniMaxH3LoRAAdapter(trainer(transformer=model), RANK, ALPHA
                                ).apply_lora_to_unet({}) == total - 1
    assert_left_alone(model, parent, attr, composite, before)


# --- SD1.5 / SDXL: descent guard in the UNet walk, skip in the TE walk ------

def _first_transformer_linear(unet):
    """(parent, attr, path) of a Linear inside the first Transformer2DModel."""
    for block_name, block in unet.named_modules():
        if block.__class__.__name__ != "Transformer2DModel":
            continue
        for child_name, child in block.named_modules():
            if child.__class__.__name__ == "Linear" and "." in child_name:
                *parents, last = child_name.split(".")
                parent = block
                for part in parents:
                    parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
                attr = int(last) if last.isdigit() else last
                return parent, attr, f"{block_name}.{child_name}"
    raise AssertionError("no Transformer2DModel Linear in this UNet stub")


def _unet_descent_case(build_unet, adapter_cls, trainer_kwargs):
    plain = build_unet()
    total = adapter_cls(trainer(**trainer_kwargs(plain)), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0

    unet = build_unet()
    parent, attr, _path = _first_transformer_linear(unet)
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    count = adapter_cls(trainer(**trainer_kwargs(unet)), RANK, ALPHA).apply_lora_to_unet({})
    # One fewer, and NOT "three more": an unguarded walk would have wrapped the
    # composite's base and both branch halves as if they were fresh targets.
    assert count == total - 1
    assert_left_alone(unet, parent, attr, composite, before)


def _te_skip_case(build_te, adapter_cls, trainer_kwargs):
    plain = build_te()
    total = adapter_cls(trainer(**trainer_kwargs(plain)), RANK, ALPHA
                        ).apply_lora_to_text_encoders({})
    assert total > 0

    te = build_te()
    layer = te.text_model.encoder.layers[0]
    composite = cover(layer.mlp, "fc1")
    before = subtree_ids(composite)

    count = adapter_cls(trainer(**trainer_kwargs(te)), RANK, ALPHA
                        ).apply_lora_to_text_encoders({})
    assert count == total - 1
    assert_left_alone(te, layer.mlp, "fc1", composite, before)


def test_sd15_unet_walk_does_not_descend_into_a_composite():
    from sd15_lora_roundtrip_cheap_test import build_unet
    from core.training.adapters.sd15_adapter import SD15LoRAAdapter

    _unet_descent_case(build_unet, SD15LoRAAdapter, lambda m: {"unet": m})


def test_sd15_text_encoder_skips_an_occupied_target():
    from sd15_lora_roundtrip_cheap_test import build_text_encoder
    from core.training.adapters.sd15_adapter import SD15LoRAAdapter

    _te_skip_case(build_text_encoder, SD15LoRAAdapter, lambda m: {"text_encoder": m})


def test_sdxl_unet_walk_does_not_descend_into_a_composite():
    from sdxl_lora_roundtrip_cheap_test import build_unet
    from core.training.adapters.sdxl_adapter import SDXLLoRAAdapter

    _unet_descent_case(build_unet, SDXLLoRAAdapter, lambda m: {"unet": m})


def test_sdxl_text_encoders_skip_an_occupied_target():
    from sdxl_lora_roundtrip_cheap_test import build_text_encoders
    from core.training.adapters.sdxl_adapter import SDXLLoRAAdapter

    te1, te2 = build_text_encoders()
    total = SDXLLoRAAdapter(trainer(text_encoder=te1, text_encoder_2=te2), RANK, ALPHA
                            ).apply_lora_to_text_encoders({})
    assert total > 0

    te1, te2 = build_text_encoders()
    mlp1 = te1.text_model.encoder.layers[0].mlp
    mlp2 = te2.text_model.encoder.layers[0].mlp
    c1, c2 = cover(mlp1, "fc1"), cover(mlp2, "fc2")
    before1, before2 = subtree_ids(c1), subtree_ids(c2)

    count = SDXLLoRAAdapter(trainer(text_encoder=te1, text_encoder_2=te2), RANK, ALPHA
                            ).apply_lora_to_text_encoders({})
    assert count == total - 2
    assert_left_alone(te1, mlp1, "fc1", c1, before1)
    assert_left_alone(te2, mlp2, "fc2", c2, before2)


# --- FLUX.2 / Z-Image: a bare wrappability test already skips a composite ---
# Both spell the wrap decision as is_lora_wrappable_linear(x) alone, which is
# False for a composite, so these two need no edit. Gated anyway, because that
# is a property of the shared predicate, not of these files.

def test_flux2_wrap_decision_skips_a_composite_unchanged():
    from flux2_lora_roundtrip_cheap_test import _Transformer, _TextEncoder
    from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter

    plain = _Transformer()
    total = FLUX2LoRAAdapter(trainer(transformer=plain), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0

    model = _Transformer()
    attn = model.transformer_blocks[0].attn
    composite = cover(attn, "to_q")
    before = subtree_ids(composite)

    assert FLUX2LoRAAdapter(trainer(transformer=model), RANK, ALPHA
                            ).apply_lora_to_unet({}) == total - 1
    assert_left_alone(model, attn, "to_q", composite, before)

    te = _TextEncoder()
    layer = te.model.layers[0]
    te_total = FLUX2LoRAAdapter(trainer(text_encoder=_TextEncoder(), train_text_encoder=True),
                                RANK, ALPHA).apply_lora_to_text_encoders({})
    te_composite = cover(layer.self_attn, "q_proj")
    te_before = subtree_ids(te_composite)
    assert FLUX2LoRAAdapter(trainer(text_encoder=te, train_text_encoder=True), RANK, ALPHA
                            ).apply_lora_to_text_encoders({}) == te_total - 1
    assert_left_alone(te, layer.self_attn, "q_proj", te_composite, te_before)


def test_zimage_wrap_decision_skips_a_composite_unchanged():
    from zimage_lora_roundtrip_cheap_test import build_model
    from core.training.adapters.zimage_adapter import ZImageLoRAAdapter

    plain = build_model()
    total = ZImageLoRAAdapter(trainer(transformer=plain), RANK, ALPHA).apply_lora_to_unet({})
    assert total > 0

    model = build_model()
    attn = next(m for _n, m in model.named_modules()
                if m.__class__.__name__ == "ZImageAttention")
    composite = cover(attn, "to_q")
    before = subtree_ids(composite)

    assert ZImageLoRAAdapter(trainer(transformer=model), RANK, ALPHA
                             ).apply_lora_to_unet({}) == total - 1
    assert_left_alone(model, attn, "to_q", composite, before)


# --- SenseNova: refuse rather than nest, on both routes ---------------------

def test_sensenova_lora_refuses_a_composite_covered_target():
    from sensenova_lora_roundtrip_cheap_test import build_model
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter

    plain = build_model()
    total = SenseNovaLoRAAdapter(trainer(transformer=plain), RANK, ALPHA
                                 ).apply_lora_to_unet({})
    assert total == 294

    model = build_model()
    targets = list(iter_sensenova_lora_targets(model, branch="gen"))
    path, parent, attr, _cur = targets[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    adapter = SenseNovaLoRAAdapter(trainer(transformer=model), RANK, ALPHA)
    with pytest.raises(RuntimeError) as excinfo:
        adapter.apply_lora_to_unet({})
    assert "composite" in str(excinfo.value)
    assert path in str(excinfo.value)
    assert_left_alone(model, parent, attr, composite, before)


def test_sensenova_full_finetune_refuses_a_composite_covered_target():
    from sensenova_lora_roundtrip_cheap_test import build_model
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter

    plain = build_model()
    branch, targets = SenseNovaFullParameterAdapter(
        trainer(transformer=plain, train_unet=True))._resolve_scope()
    assert branch == "gen" and len(targets) == 294

    model = build_model()
    path, parent, attr, _cur = list(iter_sensenova_lora_targets(model, branch="gen"))[0]
    composite = cover(parent, attr)
    before = subtree_ids(composite)

    adapter = SenseNovaFullParameterAdapter(trainer(transformer=model, train_unet=True))
    with pytest.raises(RuntimeError) as excinfo:
        adapter._resolve_scope()
    assert "composite" in str(excinfo.value)
    assert_left_alone(model, parent, attr, composite, before)


# --- the sample-preview detour's site collector ----------------------------

class _DetourStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)


def test_training_lora_detour_collects_wrapper_roots_not_branches():
    from core.training.temp_pipeline import collect_training_lora_sites

    model = _DetourStub()
    model.a = LoRALinearLayer(model.a, RANK, ALPHA, "a")
    composite = cover(model, "b")

    sites = collect_training_lora_sites([model])
    assert {id(w) for _p, _a, w in sites} == {id(model.a), id(composite)}
    # The composite's branch is a LoRALinearLayer too; splicing it out would put
    # the shared base into the branch slot.
    assert all(w is not composite.get_branch("installed") for _p, _a, w in sites)


def test_training_lora_detour_is_unchanged_on_a_plain_wrapped_tree():
    from core.training.temp_pipeline import collect_training_lora_sites

    model = _DetourStub()
    model.a = LoRALinearLayer(model.a, RANK, ALPHA, "a")
    model.b = LoRALinearLayer(model.b, RANK, ALPHA, "b")

    sites = collect_training_lora_sites([model])
    assert {attr for _p, attr, _w in sites} == {"a", "b"}
    assert {id(w) for _p, _a, w in sites} == {id(model.a), id(model.b)}


# --- the shared walker's own contract --------------------------------------

def test_walker_matches_named_modules_when_no_adapter_is_installed():
    from sd15_lora_roundtrip_cheap_test import build_unet

    unet = build_unet()
    assert ([n for n, _m in named_modules_outside_adapters(unet)]
            == [n for n, _m in unet.named_modules()])


def test_walker_yields_a_wrapper_and_stops_there():
    model = _DetourStub()
    composite = cover(model, "b")
    names = [n for n, _m in named_modules_outside_adapters(model)]
    assert "b" in names
    assert [n for n in names if n.startswith("b.")] == []
    assert composite.branch_names == ("installed",)


def test_is_adapter_covered_matches_both_wrapper_classes():
    from core.adapters import MiniMaxH3LoRALinearLayer

    base = nn.Linear(4, 4)
    assert not is_adapter_covered(base)
    assert not is_adapter_covered(None)
    assert is_adapter_covered(LoRALinearLayer(base, RANK, ALPHA, "x"))
    assert is_adapter_covered(MiniMaxH3LoRALinearLayer(base, RANK, ALPHA, "x"))
    assert is_adapter_covered(CompositeAdapterLayer(base))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
