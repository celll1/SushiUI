"""``AdapterSession``: the lifetime eleven backends used to hand-write, gated.

Architecture-neutral on purpose. The per-architecture round-trip files
(``<arch>_lora_roundtrip_cheap_test.py``) prove that an arch's own codec, target
set and numerics survive; this file proves the four properties the SESSION is
responsible for, over a three-Linear stub that belongs to no architecture:

1. installation is atomic -- a failure part way leaves nothing installed;
2. ``activate()`` restores in a ``finally``, so a raising body cannot carry
   wrappers into the next request;
3. the weakref-keyed bookkeeping resets when a component's module is replaced,
   INCLUDING on the empty-config path, so an unload cannot splice the previous
   model's Linears into the new tree;
4. every refusal is taken BEFORE the model is mutated, and carries its warning
   code as data as well as through the callback.

Each of the four was checked by reverting the behaviour it guards; the reverts
are recorded next to the tests. Nothing here imports ``api``: the session
reports through a callback, and this file supplies a list.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_session_cheap_test.py -v
"""

import os
import sys

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import (  # noqa: E402
    SHAPE_MISMATCH, AdapterComponent, AdapterFileMissing, AdapterIncompatible,
    AdapterLoadFailed, AdapterSession, CompositeAdapterLayer, LoRALinearLayer,
    PreparedBranch,
)

WIDTH = 4
RANK = 2
ALPHA = 6.0  # != rank, so a scale regression is visible
STRENGTH = 0.7
TARGETS = ("a", "b", "pack.0")


class _Stub(nn.Module):
    """Two attribute slots and one INDEX slot, which is the trap Anima/Lens hit."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(WIDTH, WIDTH)
        self.b = nn.Linear(WIDTH, WIDTH)
        self.pack = nn.ModuleList([nn.Linear(WIDTH, WIDTH)])


def iter_targets(model):
    yield model, "a", "a"
    yield model, "b", "b"
    yield model.pack, 0, "pack.0"


def build_branch(request):
    """The whole architecture-specific half, for a stub codec."""
    down = request.file.tensors.get(f"{request.module_path}.lora_down.weight")
    up = request.file.tensors.get(f"{request.module_path}.lora_up.weight")
    if down is None or up is None:
        return None
    if down.shape[1] != request.base.in_features or up.shape[0] != request.base.out_features:
        return SHAPE_MISMATCH
    branch = LoRALinearLayer(request.base, rank=down.shape[0], alpha=ALPHA,
                             lora_name=request.module_path)
    with torch.no_grad():
        branch.lora_down.weight.data = down.clone()
        branch.lora_up.weight.data = up.clone()
    return PreparedBranch(branch, request.file.strength)


def write_lora(tmp_path, name="a.safetensors", targets=TARGETS, seed=1, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for target in targets:
        tensors[f"{target}.lora_down.weight"] = torch.randn(RANK, width, generator=generator)
        tensors[f"{target}.lora_up.weight"] = torch.randn(WIDTH, RANK, generator=generator)
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"lora_alpha": str(ALPHA)})
    return str(path)


def make_session(warned, **kwargs):
    kwargs.setdefault("resolve_path", lambda p: str(p) if os.path.exists(str(p)) else None)
    kwargs.setdefault("warn", lambda message, code: warned.append((code, message)))
    kwargs.setdefault("log", lambda _message: None)
    return AdapterSession(label="Stub LoRA", **kwargs)


def component(model, **kwargs):
    kwargs.setdefault("build_branch", build_branch)
    return AdapterComponent(name="transformer", module=model,
                            iter_targets=iter_targets, **kwargs)


def composites(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def slots(model):
    """Slot path -> the module object currently installed there."""
    return {"a": model.a, "b": model.b, "pack.0": model.pack[0]}


@pytest.fixture
def warned():
    return []


# -- the fixture must really work, or every "nothing was installed" is vacuous --

def test_a_whole_request_installs_and_two_files_sum_over_one_slot(tmp_path, warned):
    model = _Stub()
    before = slots(model)
    session = make_session(warned)
    result = session.load(
        [{"path": write_lora(tmp_path, "a.safetensors", seed=1), "strength": STRENGTH},
         {"path": write_lora(tmp_path, "b.safetensors", seed=2), "strength": 0.4}],
        [component(model)])

    assert composites(model) == set(TARGETS)
    assert result.applied == 6
    assert [counts.per_component["transformer"] for _file, counts in result.files] == \
        [(3, 0), (3, 0)]
    assert not warned
    assert session.state("transformer").wrapped == set(TARGETS)

    # The composite sums the deltas FIRST and adds the base once, so the
    # reference has to associate the same way for torch.equal to hold.
    x = torch.randn(2, WIDTH)
    wrapper = model.a
    delta = None
    for name in wrapper.branch_names:
        contribution = wrapper.get_branch(name).forward_delta(x)
        delta = contribution if delta is None else delta + contribution
    assert torch.equal(wrapper(x), wrapper.original_module(x) + delta)

    assert session.unload([component(model)]) == 3
    assert slots(model) == before
    assert not composites(model)


def test_a_second_load_over_an_already_wrapped_tree_adds_a_branch(tmp_path, warned):
    """Within one request nothing is installed while planning, so the composite
    is only ever seen by a SECOND ``load()``. That is the path on which
    ``BranchRequest.base`` must be the composite's base and not the composite:
    ``add_branch`` refuses a branch built against a foreign base, which is the
    stale-splice guard, and it would fire on every target here."""
    model = _Stub()
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path, "first.safetensors"), "strength": 1.0}],
                 [component(model)])
    session.load([{"path": write_lora(tmp_path, "second.safetensors", seed=7),
                   "strength": 0.5}], [component(model)])

    assert composites(model) == set(TARGETS)
    assert all(len(module) == 2 for module in slots(model).values())
    assert session.state("transformer").wrapped == set(TARGETS)


# -- 1. atomic installation ---------------------------------------------------

def test_a_failure_part_way_through_installation_leaves_nothing_installed(tmp_path,
                                                                          warned):
    """REVERT THAT PROVES THIS BITES: drop the try/except/_rollback in
    ``AdapterSession._install`` and let the exception propagate. Two slots are
    then left holding composites, and the bookkeeping still claims them."""
    model = _Stub()
    before = slots(model)

    def third_target_is_not_a_branch(request):
        if request.module_path == "pack.0":
            return nn.Linear(WIDTH, WIDTH)  # no forward_delta: add_branch refuses
        return build_branch(request)

    session = make_session(warned)
    with pytest.raises(AdapterLoadFailed) as excinfo:
        session.load([{"path": write_lora(tmp_path), "strength": STRENGTH}],
                     [component(model, build_branch=third_target_is_not_a_branch)])

    assert excinfo.value.code == "lora_load_failed"
    assert isinstance(excinfo.value, RuntimeError)
    assert not composites(model)
    assert slots(model) == before
    state = session.state("transformer")
    assert not state.wrapped and not state.originals


def test_a_rollback_leaves_an_earlier_requests_wrappers_alone(tmp_path, warned):
    """Rollback undoes THIS request, not the composite an earlier one installed."""
    model = _Stub()
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path, "first.safetensors"), "strength": 1.0}],
                 [component(model)])
    installed = slots(model)

    def second_target_is_not_a_branch(request):
        if request.module_path == "b":
            return nn.Linear(WIDTH, WIDTH)
        return build_branch(request)

    with pytest.raises(AdapterLoadFailed):
        session.load([{"path": write_lora(tmp_path, "second.safetensors", seed=9),
                       "strength": 0.5}],
                     [component(model, build_branch=second_target_is_not_a_branch)])

    assert slots(model) == installed, "the earlier request's composites were disturbed"
    assert all(len(module) == 1 for module in installed.values()), \
        "a rolled-back branch survived on an earlier request's composite"
    assert session.state("transformer").wrapped == set(TARGETS)


# -- 2. restore in a finally --------------------------------------------------

def test_activate_restores_when_the_body_raises(tmp_path, warned):
    """REVERT THAT PROVES THIS BITES: in ``AdapterSession.activate`` replace the
    try/finally with a plain ``yield`` followed by ``self.unload(...)``. The
    wrappers then survive the exception into the next request."""
    model = _Stub()
    before = slots(model)
    session = make_session(warned)
    configs = [{"path": write_lora(tmp_path), "strength": STRENGTH}]

    with pytest.raises(ZeroDivisionError):
        with session.activate(configs, [component(model)]):
            assert composites(model) == set(TARGETS)
            1 / 0

    assert not composites(model)
    assert slots(model) == before
    assert not session.state("transformer").wrapped


# -- 3. the weakref-keyed reset -----------------------------------------------

def test_replacing_the_module_resets_the_bookkeeping_before_an_unload(tmp_path,
                                                                     warned):
    """The splice found on eight architectures: wrap A, swap to B, unload.

    REVERT THAT PROVES THIS BITES: drop the ``self.bind(component)`` call from
    ``AdapterSession.unload``. B's slots then receive A's Linears."""
    model_a, model_b = _Stub(), _Stub()
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path), "strength": 1.0}], [component(model_a)])

    a_ids = {id(m) for _n, m in model_a.named_modules()}
    b_before = slots(model_b)
    assert session.state("transformer").wrapped, "the stale set must be truthy"

    session.unload([component(model_b)])

    assert slots(model_b) == b_before
    assert not {id(m) for m in slots(model_b).values()} & a_ids
    assert not session.state("transformer").wrapped
    assert not session.state("transformer").originals
    assert composites(model_a) == set(TARGETS), "model A lost its wrappers"


def test_the_reset_happens_before_the_empty_config_exit(tmp_path, warned):
    """A request that selects NO adapter is exactly when a model swap goes
    unnoticed.

    REVERT THAT PROVES THIS BITES: move the ``for component: self.bind(...)``
    loop in ``AdapterSession.load`` below the ``if not configs: return``. The
    stale map then survives the empty request and the next unload splices."""
    model_a, model_b = _Stub(), _Stub()
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path), "strength": 1.0}], [component(model_a)])

    session.load([], [component(model_b)])
    assert not session.state("transformer").wrapped
    assert not session.state("transformer").originals

    a_ids = {id(m) for _n, m in model_a.named_modules()}
    b_before = slots(model_b)
    session.unload([component(model_b)])
    assert slots(model_b) == b_before
    assert not {id(m) for m in slots(model_b).values()} & a_ids


def test_an_unloaded_component_drops_its_bookkeeping(tmp_path, warned):
    model = _Stub()
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path), "strength": 1.0}], [component(model)])
    session.load(None, [component(None)])
    assert not session.state("transformer").wrapped
    assert not session.state("transformer").originals


def test_unload_restores_from_what_is_installed_not_from_the_map(tmp_path, warned):
    """REVERT THAT PROVES THIS BITES: drive the restore loop over
    ``state.originals`` instead of over the installed composites. With the map
    emptied, nothing is restored and the model stays wrapped."""
    model = _Stub()
    before = slots(model)
    session = make_session(warned)
    session.load([{"path": write_lora(tmp_path), "strength": 1.0}], [component(model)])

    session.state("transformer").originals.clear()
    assert session.unload([component(model)]) == 3
    assert slots(model) == before
    assert not composites(model)


# -- 4. refusal before mutation, and the code as data -------------------------

def test_a_missing_second_file_leaves_the_first_uninstalled(tmp_path, warned):
    """REVERT THAT PROVES THIS BITES: install per file (call ``self._install``
    inside ``load``'s loop). The first file's 3 composites then survive the
    second file's refusal."""
    model = _Stub()
    before = slots(model)
    session = make_session(warned)

    with pytest.raises(AdapterFileMissing) as excinfo:
        session.load([{"path": write_lora(tmp_path), "strength": STRENGTH},
                      {"path": str(tmp_path / "absent.safetensors")}],
                     [component(model)])

    assert excinfo.value.code == "lora_not_found"
    assert isinstance(excinfo.value, FileNotFoundError)
    assert [code for code, _m in warned] == ["lora_not_found"]
    assert "absent.safetensors" in warned[0][1]
    assert not composites(model)
    assert slots(model) == before


def test_a_foreign_second_file_leaves_the_first_uninstalled(tmp_path, warned):
    model = _Stub()
    before = slots(model)
    foreign = tmp_path / "foreign.safetensors"
    save_file({"nowhere.lora_down.weight": torch.zeros(RANK, WIDTH),
               "nowhere.lora_up.weight": torch.zeros(WIDTH, RANK)}, str(foreign))

    session = make_session(warned)
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": write_lora(tmp_path), "strength": STRENGTH},
                      {"path": str(foreign)}],
                     [component(model)])

    assert excinfo.value.code == "lora_incompatible"
    assert isinstance(excinfo.value, RuntimeError)
    assert [code for code, _m in warned] == ["lora_incompatible"]
    assert not composites(model)
    assert slots(model) == before


def test_an_unreadable_file_reports_its_type_and_basename_never_a_path(tmp_path,
                                                                      warned):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    model = _Stub()
    session = make_session(warned)

    with pytest.raises(AdapterLoadFailed) as excinfo:
        session.load([{"path": str(broken)}], [component(model)])

    message = excinfo.value.message
    assert excinfo.value.code == "lora_load_failed"
    assert "broken.safetensors" in message
    assert str(tmp_path) not in message, "the resolved path leaked into a warning"
    assert not composites(model)


def test_a_shape_mismatch_is_skipped_and_warns_partial(tmp_path, warned):
    model = _Stub()
    tensors = {f"{t}.lora_down.weight": torch.randn(RANK, WIDTH) for t in TARGETS}
    tensors.update({f"{t}.lora_up.weight": torch.randn(WIDTH, RANK) for t in TARGETS})
    tensors["b.lora_down.weight"] = torch.randn(RANK, WIDTH + 1)
    path = tmp_path / "wide.safetensors"
    save_file(tensors, str(path))

    session = make_session(warned)
    session.load([{"path": str(path), "strength": STRENGTH}], [component(model)])

    assert composites(model) == {"a", "pack.0"}
    assert [code for code, _m in warned] == ["lora_partial"]
    assert "applied 2 of 3" in warned[0][1]


def test_a_warning_channel_that_raises_does_not_replace_the_refusal(tmp_path):
    def hostile(_message, _code):
        raise KeyError("the warning channel is broken")

    session = make_session([], warn=hostile)
    with pytest.raises(AdapterFileMissing):
        session.load([{"path": str(tmp_path / "absent.safetensors")}],
                     [component(_Stub())])


def test_the_refusal_carries_its_code_with_no_warning_channel_at_all(tmp_path):
    """The pull half of the reporting contract: the code is on the exception, so
    a caller putting it on a 400 response needs no warning channel."""
    session = make_session([], warn=None)
    with pytest.raises(AdapterFileMissing) as excinfo:
        session.load([{"path": str(tmp_path / "absent.safetensors")}],
                     [component(_Stub())])
    assert excinfo.value.code == "lora_not_found"


# -- per-component accounting (the shape FLUX.2 needs) ------------------------

def test_a_disabled_component_is_not_walked_and_is_accounted_separately(tmp_path,
                                                                       warned):
    transformer, encoder = _Stub(), _Stub()
    session = make_session(warned)
    result = session.load(
        [{"path": write_lora(tmp_path), "strength": STRENGTH}],
        [component(transformer),
         AdapterComponent(name="text_encoder", module=encoder,
                          iter_targets=iter_targets, build_branch=build_branch,
                          enabled=False)])

    _file, counts = result.files[0]
    assert counts.per_component == {"transformer": (3, 0)}
    assert not composites(encoder)
    assert composites(transformer) == set(TARGETS)
