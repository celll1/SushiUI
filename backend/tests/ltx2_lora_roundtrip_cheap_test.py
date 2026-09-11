"""LTX-2.3: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``Ltx2LoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and the REAL ``LTX2Mixin._load_lora_ltx2``.

The Phase-0 defect this pins: LTX-2.3 had a training adapter and NO generation
loader at all, so a self-trained LoRA could only ever be ignored.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains LTX-2.3's target
scope, persistent offloader reconciliation, lifecycle, and refusal seams.

BLOCK SWAP is checked here rather than reasoned about, because this
architecture's offloader is the only PERSISTENT one: the module set the swap
pairs by name changes once per LoRA added, not once per request.

Complementary to ``video_lora_threading_test.py``, which anchors the request
plumbing (routes, FormData, panels, openapi) and the block-swap cache
reconciliation by source inspection. This file is the numerical half: the
targets, the scale, the restore and the reload guard, executed.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ltx2_lora_roundtrip_cheap_test.py -v
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, module_ids, randomise_lora_layers,
    warning_codes, warning_probe,
)

from api.error_handlers import ValidationError  # noqa: E402
from core.adapters import CompositeAdapterLayer  # noqa: E402
from core.memory_management.block_offloading import linear_weight_dtypes  # noqa: E402
from core.models.ltx2.ltx2_lora import (  # noqa: E402
    iter_lora_slots, swappable_block_weight_footprints,
)
from core.pipeline_backends.ltx2 import LTX2Mixin  # noqa: E402
from core.training.adapters.ltx2_adapter import (  # noqa: E402
    DEFAULT_LTX2_SCOPE, Ltx2LoRAAdapter, _flatten_to_sdscripts,
    iter_ltx2_lora_targets,
)

D = 8
RANK = 4
# alpha/rank == 1.5 and strength == 0.7, so the applied scale is neither 1.0 nor
# either factor: every plausible reassociation of the two moves the bits, which
# is what makes the bit-identity gate below bite at all.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum
ATTENTION_AND_FF = {**DEFAULT_LTX2_SCOPE, "ff": True}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_k = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(D, D, bias=False)])


class _Ff(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(D, D, bias=False), nn.GELU(),
                                  nn.Linear(D, D, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attn()
        self.attn2 = _Attn()
        self.ff = _Ff()


class _Dit(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.dtype = torch.float32


def build_dit(n_blocks=2):
    """A stub with REPRODUCIBLE base weights.

    Gates 2 and 3 compare two separately built models: with the default random
    init they would differ in the base, and every "identical" claim about the
    branches would be swamped by that.
    """
    torch.manual_seed(20260903)
    return _Dit(n_blocks)


class _Backend(LTX2Mixin):
    def __init__(self, transformer):
        self.ltx2_components = {"transformer": transformer}


def wrapped_paths(model):
    """Target paths a GENERATION load covers, i.e. the composite roots."""
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def lora_layer_paths(model):
    """Paths the TRAINER wrapped -- it still installs plain wrappers."""
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, scope=None, name="ltx2.safetensors", seed=1234):
    dit = build_dit()
    adapter = Ltx2LoRAAdapter(SimpleNamespace(transformer=dit, config={}),
                              RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Gemma-3 is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return str(out), lora_layer_paths(dit)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = "lora_unet_" + _flatten_to_sdscripts(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def assert_refusal_names_the_file(recorded, code, name, directory):
    """One warning, the SHARED code, the basename -- and never the directory.

    The message is returned raw in the response's ``warnings[]`` and written
    into the PNG metadata chunk, so a resolved path in it is a leak; an
    arch-prefixed code (``ltx2_lora_not_found``) is invisible to every client
    that switches on the cross-architecture taxonomy.
    """
    assert warning_codes(recorded) == [code]
    message = recorded[-1][1]
    assert name in message, f"the refusal does not name the file: {message!r}"
    assert str(directory) not in message, f"a resolved path leaked: {message!r}"


def test_ltx2_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    backend = _Backend(dit)
    applied = backend._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(dit) == trained_paths
    assert applied == len(trained_paths)
    assert backend._ltx2_lora_wrapped_keys == trained_paths
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_ltx2_lora_targets(_Dit(), DEFAULT_LTX2_SCOPE)}
    assert any(p.endswith(".to_out.0") for p in trained_paths)


def test_ltx2_opt_in_feed_forward_scope_reaches_generation(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF,
                                         name="ff.safetensors")
    assert any(".ff." in p for p in trained_paths)

    dit = build_dit()
    assert _Backend(dit)._load_lora_ltx2([{"path": path}]) == len(trained_paths)
    assert wrapped_paths(dit) == trained_paths


def test_ltx2_the_enumerator_never_offers_a_branch_as_a_target(tmp_path):
    """Over a composite, neither walk may offer the adapter's own
    ``lora_down``/``lora_up`` as targets on the NEXT load."""
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF,
                                         name="ff.safetensors")
    dit = build_dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    slots = {p for p, _parent, _slot in iter_lora_slots(dit)}
    assert slots == trained_paths
    assert not any(".branches." in p for p in slots)
    # The trainer's ff walk used to descend into anything that was not a
    # LoRALinearLayer, which made a composite's branches targets and is why the
    # guard above exists. It is composite-aware now, so it offers the composite
    # root and nothing under it -- the stronger claim.
    raw = {p for p, _parent, _attr, _cur in iter_ltx2_lora_targets(dit, ATTENTION_AND_FF)}
    assert not any(".branches." in p for p in raw)
    assert raw == trained_paths


def test_ltx2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF)

    dit = build_dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, D)
        base = composite.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        assert not torch.allclose(composite(x), base, atol=1e-5), f"{target}: branch is inert"


def test_ltx2_selecting_the_same_file_twice_is_two_branches(tmp_path):
    """Branch names carry the request index, so a duplicate selection is not a
    duplicate-name refusal."""
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH},
                                   {"path": path, "strength": STRENGTH}])
    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        assert len(modules[target]) == 2, target
        assert len(set(modules[target].branch_names)) == 2, target


def test_ltx2_a_second_request_does_not_stack_onto_a_leaked_wrapper(tmp_path):
    """The stacking refusal used to be the accidental backstop for a wrapper
    that outlived its request; without the leading restore a leak would SUM."""
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    backend = _Backend(dit)
    backend._load_lora_ltx2([{"path": path, "strength": STRENGTH}])
    # A restore that never happened (the finally raised, say): the wrappers are
    # still installed when the next request's load runs.
    backend._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        assert len(modules[target]) == 1, f"{target}: {modules[target].branch_names}"


def test_ltx2_missing_file_refuses_and_warns(tmp_path, warnings_seen):
    absent = tmp_path / "no_such_ltx2_lora.safetensors"
    with pytest.raises((FileNotFoundError, ValidationError, RuntimeError)):
        _Backend(build_dit())._load_lora_ltx2([{"path": str(absent)}])
    assert_refusal_names_the_file(
        warnings_seen, "lora_not_found", absent.name, tmp_path)


def test_ltx2_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    dit = build_dit()
    with pytest.raises(RuntimeError):
        _Backend(dit)._load_lora_ltx2([{"path": str(broken)}])
    assert_refusal_names_the_file(
        warnings_seen, "lora_load_failed", broken.name, tmp_path)
    assert not wrapped_paths(dit)


def test_ltx2_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = "lora_unet_transformer_blocks_9_attn1_to_q"
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "ltx2"})

    dit = build_dit()
    with pytest.raises((ValidationError, RuntimeError)):
        _Backend(dit)._load_lora_ltx2([{"path": str(ghost), "strength": 1.0}])
    assert_refusal_names_the_file(
        warnings_seen, "lora_incompatible", ghost.name, tmp_path)
    assert not wrapped_paths(dit), "a refused load left wrappers on the DiT"


def test_ltx2_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    from core.adapters import AdapterIncompatible

    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_unet_transformer_blocks_9_attn1_to_q"
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "ltx2"})

    dit = build_dit()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(dit)._load_lora_ltx2([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(dit)
    assert_refusal_names_the_file(
        warnings_seen, "lora_partial", partial.name, tmp_path)


def test_ltx2_two_loras_over_the_same_targets_stack_instead_of_refusing(
        tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    dit = build_dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH},
                                   {"path": second, "strength": STRENGTH_B}])
    assert wrapped_paths(dit) == trained_paths
    assert warning_codes(warnings_seen) == []


# ---------------------------------------------------------------------------
# Block swap. LTX-2.3's offloader is PERSISTENT across generations, and both of
# its swap paths cache a description of the block tree, so what the composite
# does to that tree is measured rather than assumed.
# ---------------------------------------------------------------------------

def swap_paths(block):
    """Exactly what the standard swap pairs BY NAME (and what the coalesced H2D
    path flattens): every module whose class name ends in ``Linear`` and which
    carries a weight."""
    return set(linear_weight_dtypes(block))


def test_ltx2_block_swap_sees_the_base_at_the_same_path_as_the_old_wrapper(tmp_path):
    """``<target>.original_module`` under a composite, exactly as under
    ``LoRALinearLayer``: an arch whose wrapper stored the base under another
    attribute would shift the swap's pairing key and silently drop it."""
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF)

    bare = build_dit()
    bare_paths = [swap_paths(b) for b in bare.transformer_blocks]

    dit = build_dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH}])
    wrapped = [swap_paths(b) for b in dit.transformer_blocks]

    for i, (before, after) in enumerate(zip(bare_paths, wrapped)):
        for leaf in before:  # block-relative, e.g. "attn1.to_q"
            assert f"{leaf}.original_module" in after, f"block {i}: base moved from {leaf}"
        # One base + two branch Linears per covered target, nothing else.
        assert len(after) == 3 * len(before), i

    # Per-block relative path sets stay uniform, which is what the paired swap
    # and the coalesced H2D path both require.
    assert len(set(map(frozenset, wrapped))) == 1
    # And nothing is enrolled twice: one entry per weight tensor.
    for block in dit.transformer_blocks:
        modules = [m for _n, m in block.named_modules()
                   if m.__class__.__name__.endswith("Linear")
                   and getattr(m, "weight", None) is not None]
        assert len({id(m.weight) for m in modules}) == len(modules)


def test_ltx2_a_second_branch_changes_the_set_the_offloader_cached(tmp_path):
    """The interaction unique to this architecture: the module set grows once
    per LoRA ADDED, not once per request, so a cache built over the one-branch
    tree mispairs against the two-branch one. ``_ltx2_sync_block_swap_after_lora``
    therefore runs after the LAST file, and unconditionally."""
    path_a, _pa = train_and_save(tmp_path, scope=ATTENTION_AND_FF)
    path_b, _pb = train_and_save(tmp_path, scope=ATTENTION_AND_FF,
                                 name="second.safetensors", seed=4321)

    one = build_dit()
    _Backend(one)._load_lora_ltx2([{"path": path_a, "strength": STRENGTH}])
    one_paths = swap_paths(one.transformer_blocks[0])

    two = build_dit()
    _Backend(two)._load_lora_ltx2([{"path": path_a, "strength": STRENGTH},
                                   {"path": path_b, "strength": STRENGTH_B}])
    two_paths = swap_paths(two.transformer_blocks[0])

    covered_in_block0 = {p for p in wrapped_paths(two)
                         if p.startswith("transformer_blocks.0.")}
    assert one_paths < two_paths, "a second branch must add its own Linears"
    assert len(two_paths) - len(one_paths) == 2 * len(covered_in_block0)
    assert all(".branches.1." in p for p in two_paths - one_paths)
    # Still uniform across blocks, so the paired swap stays valid once rebuilt.
    assert len({frozenset(swap_paths(b)) for b in two.transformer_blocks}) == 1


def test_ltx2_a_partial_second_lora_breaks_the_h2d_uniformity_check(tmp_path):
    """A second LoRA covering only SOME blocks is newly reachable (it used to be
    refused as fully shadowed), and it is exactly what the coalesced H2D path
    asserts against."""
    path_a, _pa = train_and_save(tmp_path)
    full = load_file(path_a)
    block0_only = {k: v for k, v in full.items()
                   if "transformer_blocks_0_" in k}
    assert block0_only
    partial = tmp_path / "block0.safetensors"
    save_file(block0_only, str(partial), metadata={"model_type": "ltx2"})

    dit = build_dit()
    backend = _Backend(dit)
    backend._load_lora_ltx2([{"path": path_a, "strength": STRENGTH},
                             {"path": str(partial), "strength": STRENGTH_B}])

    footprints = swappable_block_weight_footprints(dit.transformer_blocks, 2)
    assert len(set(footprints)) > 1, (
        "the uneven second LoRA is invisible to the H2D uniformity check")


def test_ltx2_evicted_model_drops_the_bookkeeping_before_the_next_load(tmp_path):
    """An eviction (components cleared) must reset the maps, not park them for
    whatever transformer is loaded next -- including on the no-LoRA request that
    takes the empty-config path."""
    path, trained_paths = train_and_save(tmp_path)

    dit_a = build_dit()
    backend = _Backend(dit_a)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._ltx2_lora_original_modules.values()}
    _keep_a = list(dit_a.modules()) + list(backend._ltx2_lora_original_modules.values())

    backend.ltx2_components = {}
    backend._unload_lora_ltx2()

    dit_b = _Dit()
    b_ids_before = module_ids(dit_b)
    backend.ltx2_components = {"transformer": dit_b}
    backend._load_lora_ltx2([])  # a generation that installs no LoRA
    backend._unload_lora_ltx2()
    assert module_ids(dit_b) == b_ids_before
    assert not (module_ids(dit_b) & a_ids)


def test_ltx2_the_bookkeeping_key_dies_with_the_transformer_it_describes(tmp_path):
    """The original-module map must not outlive the model it describes.

    ``id()`` cannot express that: a freed object's address is REUSABLE, so a
    reload landing on the dead transformer's address would read as the same
    model and the next restore would splice model A's Linears into model B.
    ``AdapterSession``'s ``_ComponentState`` keys on a weakref, which this drives
    through LTX-2.3's real loader -- an arch that kept its own id-keyed map
    beside the session would fail here.
    """
    path, trained_paths = train_and_save(tmp_path)

    dit_a = build_dit()
    backend = _Backend(dit_a)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    state = backend._ltx2_lora_session.state("transformer")
    assert isinstance(state.ref, weakref.ref) and state.ref() is dit_a
    assert len(state.originals) == len(trained_paths)

    backend.ltx2_components = {}
    del dit_a
    gc.collect()
    assert state.ref() is None, (
        "the bookkeeping holds the transformer alive, so its key outlives the "
        "model and an address reuse reads as the same model")

    dit_b = _Dit()
    b_before = dict(dit_b.named_modules())
    backend.ltx2_components = {"transformer": dit_b}
    assert backend._load_lora_ltx2([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_ltx2() == len(trained_paths)
    for target in trained_paths:
        assert dict(dit_b.named_modules())[target] is b_before[target], target


def test_ltx2_a_no_lora_request_still_restores_and_still_resets(tmp_path):
    """The empty-config exit is not an exit from the reset.

    ``_load_lora_ltx2`` unloads BEFORE it returns early on an empty request, and
    ``AdapterSession.load`` binds before ITS empty-config return. Both halves
    matter: a wrapper that outlived its request would otherwise generate on a
    request that selected no LoRA, and an evicted model's bookkeeping would be
    parked for whatever transformer is loaded next.
    """
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    backend = _Backend(dit)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    assert wrapped_paths(dit) == trained_paths
    # The restore that never happened (a finally that itself raised, say).
    assert backend._load_lora_ltx2([]) == 0
    assert not wrapped_paths(dit), (
        "a request that selected no LoRA generated with the previous request's "
        "adapters still installed")
    assert not backend._ltx2_lora_wrapped_keys

    dit_a = build_dit()
    backend = _Backend(dit_a)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._ltx2_lora_original_modules.values()}
    dit_b = _Dit()
    b_ids_before = module_ids(dit_b)

    backend.ltx2_components = {"transformer": dit_b}
    assert backend._load_lora_ltx2([]) == 0
    assert not backend._ltx2_lora_wrapped_keys, (
        "model A's map survived the no-LoRA request; the next restore would "
        "splice it into model B")
    backend._unload_lora_ltx2()
    assert module_ids(dit_b) == b_ids_before
    assert not (module_ids(dit_b) & a_ids)
