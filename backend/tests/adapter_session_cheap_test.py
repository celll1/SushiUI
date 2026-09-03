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
   code as data as well as through the callback;
5. an adapter algebra the architecture has not enabled (LoHa/LoKr/DoRA) is
   refused at PARSE time, before a single target is even walked;
6. the five decisions an ARCHITECTURE owns are hooks and really reach the
   session: how a missing file is refused (or skipped), what one file's keys
   mean, how many branches it declares to THIS pass, what a zero-target file
   means, and which of its two names a message uses.

Each was checked by reverting the behaviour it guards; the reverts
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

import safetensors  # noqa: E402

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
    kwargs.setdefault("label", "Stub LoRA")
    return AdapterSession(**kwargs)


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


def test_unload_removes_only_the_calling_sessions_branches(tmp_path, warned):
    model = _Stub()
    first = make_session(warned)
    second = make_session(warned)
    first.load([{"path": write_lora(tmp_path, "first.safetensors")}],
               [component(model)])
    second.load([{"path": write_lora(tmp_path, "second.safetensors", seed=8)}],
                [component(model)])

    assert all(module.branch_names == ("0:first.safetensors", "0:second.safetensors")
               for module in slots(model).values())

    first.unload([component(model)])
    assert composites(model) == set(TARGETS)
    assert all(module.branch_names == ("0:second.safetensors",)
               for module in slots(model).values())

    second.unload([component(model)])
    assert not composites(model)


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
    assert not composites(model_a), "the abandoned model retained this session's branches"


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


def test_rebinding_a_to_b_to_a_restores_each_old_model_before_reset(tmp_path, warned):
    model_a, model_b = _Stub(), _Stub()
    a_before, b_before = slots(model_a), slots(model_b)
    session = make_session(warned)
    config = [{"path": write_lora(tmp_path), "strength": 1.0}]

    session.load(config, [component(model_a)])
    session.load(config, [component(model_b)])
    assert slots(model_a) == a_before
    assert composites(model_b) == set(TARGETS)

    session.load([], [component(model_a)])
    assert slots(model_a) == a_before
    assert slots(model_b) == b_before
    assert not composites(model_a)
    assert not composites(model_b)


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


def test_failed_planning_does_not_freeze_the_base(tmp_path, warned):
    model = _Stub()
    foreign = tmp_path / "foreign.safetensors"
    save_file({"nowhere.lora_down.weight": torch.zeros(RANK, WIDTH),
               "nowhere.lora_up.weight": torch.zeros(WIDTH, RANK)}, str(foreign))
    session = make_session(warned)

    with pytest.raises(AdapterIncompatible):
        session.load([{"path": write_lora(tmp_path)}, {"path": str(foreign)}],
                     [component(model)])

    assert all(parameter.requires_grad for parameter in model.parameters())


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


def test_a_shape_mismatch_refuses_before_installation(tmp_path, warned):
    model = _Stub()
    tensors = {f"{t}.lora_down.weight": torch.randn(RANK, WIDTH) for t in TARGETS}
    tensors.update({f"{t}.lora_up.weight": torch.randn(WIDTH, RANK) for t in TARGETS})
    tensors["b.lora_down.weight"] = torch.randn(RANK, WIDTH + 1)
    path = tmp_path / "wide.safetensors"
    save_file(tensors, str(path))

    session = make_session(warned)
    before = slots(model)
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": str(path), "strength": STRENGTH}], [component(model)])

    assert excinfo.value.code == "lora_partial"
    assert [code for code, _m in warned] == ["lora_partial"]
    assert "shape mismatch" in warned[0][1]
    assert slots(model) == before
    assert not composites(model)


def test_a_missing_declared_target_refuses_before_installation(tmp_path, warned):
    model = _Stub()
    before = slots(model)
    path = write_lora(tmp_path, targets=(*TARGETS, "not.in.model"))

    session = make_session(warned)
    with pytest.raises(AdapterIncompatible):
        session.load([{"path": path}], [component(model)])

    assert slots(model) == before
    assert [code for code, _message in warned] == ["lora_partial"]


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


# -- 5. the capability gate (core/adapters/capability.py) ---------------------

def write_loha(tmp_path, name="loha.safetensors", targets=TARGETS):
    """A LyCORIS LoHa file. ``hada_w1_a`` alone is what the codec detects on."""
    generator = torch.Generator().manual_seed(3)
    tensors = {}
    for target in targets:
        tensors[f"{target}.hada_w1_a"] = torch.randn(WIDTH, RANK, generator=generator)
        tensors[f"{target}.hada_w1_b"] = torch.randn(RANK, WIDTH, generator=generator)
        tensors[f"{target}.hada_w2_a"] = torch.randn(WIDTH, RANK, generator=generator)
        tensors[f"{target}.hada_w2_b"] = torch.randn(RANK, WIDTH, generator=generator)
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"lora_alpha": str(ALPHA)})
    return str(path)


def _watched_component(model, visited):
    def build(request):
        visited.append(request.module_path)
        return build_branch(request)

    return component(model, build_branch=build)


def _engine_component(model, visited):
    """A component whose builder is the ENGINE's dispatcher rather than this
    file's down/up stub, so an enabled architecture really installs a LoHa."""
    from core.adapters import build_adapter_branch, group_adapter_tensors

    def build(request):
        visited.append(request.module_path)
        group = group_adapter_tensors(request.file.tensors).groups.get(
            request.module_path)
        if group is None:
            return None
        branch = build_adapter_branch(request.base, group,
                                      lora_name=request.module_path)
        if branch is SHAPE_MISMATCH:
            return branch
        return PreparedBranch(branch, request.file.strength)

    return component(model, build_branch=build)


def test_a_loha_file_is_refused_before_any_target_is_walked(tmp_path, warned):
    """The whole point of the gate: the refusal arrives during ``_parse``, so
    no target was walked, no branch built and no slot touched.

    ``sensenova`` rather than ``zimage``: nine architectures now ENABLE the
    additive LyCORIS algebras (``core/adapters/capability.py``), so only one of
    the two still behind their own gate exercises the refusal at all.

    REVERT THAT PROVES THIS BITES: drop the ``_refuse_unsupported_algebra``
    call from ``_parse``. The file then reaches the architecture's builder and
    is refused one level along as a zero-target file -- same code, but only
    after the tree has been walked, and on architectures whose builder happens
    to accept a ``hada_*`` group it would not be refused at all."""
    model = _Stub()
    before = slots(model)
    visited = []

    session = make_session(warned, architecture="sensenova")
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": write_loha(tmp_path), "strength": STRENGTH}],
                     [_engine_component(model, visited)])

    assert excinfo.value.code == "lora_incompatible"
    assert [code for code, _m in warned] == ["lora_incompatible"]
    assert "loha" in warned[0][1] and "sensenova" in warned[0][1]
    assert visited == [], "a target was walked before the refusal"
    assert slots(model) == before
    assert not composites(model)
    state = session.state("transformer")
    assert not state.wrapped and not state.originals and not state.owned


def test_an_enabled_architecture_installs_the_same_loha_file(tmp_path, warned):
    """The sibling that gives the row above its discriminating power: the SAME
    file, the same builder, an architecture whose row carries
    ``("loha", False)`` -- walked, built and installed."""
    model = _Stub()
    visited = []

    session = make_session(warned, architecture="zimage")
    result = session.load([{"path": write_loha(tmp_path), "strength": STRENGTH}],
                          [_engine_component(model, visited)])

    assert visited == list(TARGETS)
    assert result.applied == len(TARGETS)
    assert composites(model) == set(TARGETS)
    assert not warned


def test_a_dora_file_is_refused_on_the_decomposition_axis(tmp_path, warned):
    """DoRA is ordinary LoRA plus ``dora_scale``: the same tensors the gate
    lets through become a refusal on the second axis alone."""
    model = _Stub()
    before = slots(model)
    from safetensors.torch import load_file

    tensors = load_file(write_lora(tmp_path))
    for target in TARGETS:
        tensors[f"{target}.dora_scale"] = torch.ones(WIDTH)
    path = tmp_path / "dora.safetensors"
    save_file(tensors, str(path), metadata={"lora_alpha": str(ALPHA)})

    session = make_session(warned, architecture="zimage")
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": str(path)}], [component(model)])

    assert excinfo.value.code == "lora_incompatible"
    assert "dora" in warned[0][1]
    assert slots(model) == before


def test_both_lokr_forms_name_the_capability_reason_not_a_malformed_file(tmp_path,
                                                                        warned):
    """``validate()`` runs before the capability check, and its malformed-file
    arms answer the SAME code, so only the TEXT tells the user which it is.

    A full/full LoKr has no rank by construction and carries upstream's
    ``lora_dim`` alpha; reading that pair as "an alpha with no rank" blamed the
    user for a valid file. The factored form's rank is on ``lokr_w2_a``, which
    the codec's sniff did not read at all."""
    forms = {
        "full": {"a.lokr_w1": torch.randn(2, 2),
                 "a.lokr_w2": torch.randn(WIDTH // 2, WIDTH // 2)},
        "factored": {"a.lokr_w1": torch.randn(2, 2),
                     "a.lokr_w2_a": torch.randn(WIDTH // 2, RANK),
                     "a.lokr_w2_b": torch.randn(RANK, WIDTH // 2)},
    }
    for label, tensors in forms.items():
        model = _Stub()
        before = slots(model)
        path = tmp_path / f"lokr_{label}.safetensors"
        save_file({**tensors, "a.alpha": torch.tensor(ALPHA)}, str(path))

        del warned[:]
        session = make_session(warned, architecture="sensenova")
        with pytest.raises(AdapterIncompatible) as excinfo:
            session.load([{"path": str(path)}], [component(model)])

        assert excinfo.value.code == "lora_incompatible", label
        assert "lokr adapters are not enabled" in excinfo.value.message, label
        for malformed in ("scale is undefined", "is unusable", "not a known"):
            assert malformed not in excinfo.value.message, (label, malformed)
        assert slots(model) == before and not composites(model), label


def test_a_session_with_no_declared_architecture_enables_nothing(tmp_path, warned):
    """``architecture=None`` must not read as "anything goes"."""
    model = _Stub()
    session = make_session(warned)
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": write_loha(tmp_path)}], [component(model)])
    assert excinfo.value.code == "lora_incompatible"
    assert not composites(model)


def test_ordinary_lora_is_not_validated_by_the_gate(tmp_path, warned):
    """DELIBERATE CARVE-OUT. ``AdapterSpec.validate`` refuses an architecture
    string it does not know, among other things; running it over ordinary LoRA
    today would refuse working files, so the gate skips that pair entirely."""
    model = _Stub()
    generator = torch.Generator().manual_seed(11)
    tensors = {}
    for target in TARGETS:
        tensors[f"{target}.lora_down.weight"] = torch.randn(RANK, WIDTH, generator=generator)
        tensors[f"{target}.lora_up.weight"] = torch.randn(WIDTH, RANK, generator=generator)
    path = tmp_path / "foreign_metadata.safetensors"
    save_file(tensors, str(path),
              metadata={"model_type": "not_an_architecture_at_all",
                        "lora_alpha": str(ALPHA)})

    session = make_session(warned, architecture="zimage")
    result = session.load([{"path": str(path), "strength": STRENGTH}],
                          [component(model)])

    assert result.applied == 3
    assert not warned
    assert composites(model) == set(TARGETS)


def test_an_unrecognized_algebra_is_left_to_the_architecture(tmp_path, warned):
    """The other carve-out: ``unknown`` is a detection FAILURE, not a family.
    A `lora_bias=True` PEFT export sniffs as unknown, so gating on it would
    refuse valid files -- the architecture's own zero-target verdict stands."""
    model = _Stub()
    visited = []
    path = tmp_path / "unreadable_algebra.safetensors"
    save_file({f"{t}.mystery_factor": torch.randn(WIDTH, RANK) for t in TARGETS},
              str(path))

    session = make_session(warned, architecture="zimage")
    with pytest.raises(AdapterIncompatible):
        session.load([{"path": str(path)}], [_watched_component(model, visited)])

    assert visited == list(TARGETS), "the walk never happened: the gate fired"
    assert not composites(model)


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


# -- 5. the hooks an architecture owns ----------------------------------------

TE = "te::"


def te_build_branch(request):
    """A second component's codec: the same targets under its own namespace."""
    down = request.file.tensors.get(f"{TE}{request.module_path}.lora_down.weight")
    up = request.file.tensors.get(f"{TE}{request.module_path}.lora_up.weight")
    if down is None or up is None:
        return None
    branch = LoRALinearLayer(request.base, rank=down.shape[0], alpha=ALPHA,
                             lora_name=request.module_path)
    with torch.no_grad():
        branch.lora_down.weight.data = down.clone()
        branch.lora_up.weight.data = up.clone()
    return PreparedBranch(branch, request.file.strength)


def write_two_namespace_lora(tmp_path, name="both.safetensors"):
    """One file carrying a pair for each component, as MiniT2I's mixed files do."""
    generator = torch.Generator().manual_seed(3)
    tensors = {}
    for target in TARGETS:
        for prefix in ("", TE):
            tensors[f"{prefix}{target}.lora_down.weight"] = torch.randn(
                RANK, WIDTH, generator=generator)
            tensors[f"{prefix}{target}.lora_up.weight"] = torch.randn(
                WIDTH, RANK, generator=generator)
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"lora_alpha": str(ALPHA)})
    return str(path)


def encoder_component(model):
    return AdapterComponent(name="text_encoder", module=model,
                            iter_targets=iter_targets, build_branch=te_build_branch)


def test_the_missing_file_hook_owns_the_type_and_the_wording(tmp_path, warned):
    """Five architectures refuse a missing file in their own words, and two of
    them need a type the session's ``AdapterFileMissing`` is not.

    REVERT THAT PROVES THIS BITES: drop the ``self._missing_file`` branch in
    ``AdapterSession._parse``. The refusal is then a ``FileNotFoundError``
    carrying the session's own sentence, and both assertions below fail."""
    def refuse(name, _raw_path):
        error = RuntimeError(f"stub LoRA '{name}' is not where it should be")
        error.code = "stub_not_found"
        return error

    model = _Stub()
    session = make_session(warned, missing_file=refuse)
    with pytest.raises(RuntimeError) as excinfo:
        session.load([{"path": str(tmp_path / "absent.safetensors")}], [component(model)])

    assert not isinstance(excinfo.value, FileNotFoundError)
    assert str(excinfo.value) == "stub LoRA 'absent.safetensors' is not where it should be"
    assert warned == [("stub_not_found", str(excinfo.value))]
    assert not composites(model)


def test_the_missing_file_hook_may_skip_and_leave_the_verdict_to_the_caller(
        tmp_path, warned):
    """Anima reports EVERY failure's code and names one of them, so a miss must
    not stop the request at the first bad file.

    REVERT THAT PROVES THIS BITES: return an exception from the hook instead of
    ``None``. The second file then never applies."""
    skipped = []
    session = make_session(warned,
                           missing_file=lambda name, _p: skipped.append(name))
    model = _Stub()
    result = session.load([{"path": str(tmp_path / "absent.safetensors")},
                           {"path": write_lora(tmp_path), "strength": STRENGTH}],
                          [component(model)])

    assert skipped == ["absent.safetensors"]
    assert result.applied == 3
    assert composites(model) == set(TARGETS)


def test_prepare_file_runs_once_per_file_and_before_the_refusal(tmp_path, warned):
    """Every architecture with key-level diagnostics needs a hook between parsing
    a file and accounting for it -- and a file that matches NOTHING is exactly
    the one whose dropped keys must be reported before it is refused.

    REVERT THAT PROVES THIS BITES: drop the ``self.prepare(file)`` call in
    ``AdapterSession.load`` and let the branch builder prepare lazily. A
    zero-target file reaches no builder, so its warning disappears and the order
    assertion fails."""
    foreign = tmp_path / "foreign.safetensors"
    save_file({"nowhere.lora_down.weight": torch.zeros(RANK, WIDTH),
               "nowhere.lora_up.weight": torch.zeros(WIDTH, RANK)}, str(foreign))

    prepared_for = []

    def prepare(file):
        prepared_for.append(file.name)
        session.warn(f"'{file.name}': 1 key dropped", "stub_keys_dropped")
        return {"format": "stub"}

    session = make_session(warned, prepare_file=prepare)
    with pytest.raises(AdapterIncompatible):
        session.load([{"path": str(foreign)}], [component(_Stub())])

    assert prepared_for == ["foreign.safetensors"]
    assert [code for code, _m in warned] == ["stub_keys_dropped", "lora_incompatible"]


def test_the_prepared_state_reaches_the_builder_and_survives_a_second_pass(
        tmp_path, warned):
    """One parse per file, whichever pass asks for it."""
    seen = []

    def prepare(file):
        seen.append(file.name)
        return {"format": "stub"}

    def build(request):
        assert request.prepared == {"format": "stub"}
        return build_branch(request)

    session = make_session(warned, prepare_file=prepare)
    files = session.parse([{"path": write_lora(tmp_path), "strength": STRENGTH}])
    transformer, encoder = _Stub(), _Stub()
    session.load(files, [component(transformer, build_branch=build)])
    session.load(files, [AdapterComponent(name="text_encoder", module=encoder,
                                          iter_targets=iter_targets,
                                          build_branch=build)])

    assert seen == ["a.safetensors"]
    assert composites(transformer) == set(TARGETS)
    assert composites(encoder) == set(TARGETS)


def test_the_declared_count_is_asked_per_pass_with_this_passs_components(
        tmp_path, warned):
    """A pass that can reach one component must not be told the other's pairs
    were declared to it.

    REVERT THAT PROVES THIS BITES: drop ``count_declared_branches`` below and
    take the default. Each pass then declares all six pairs, applies three, and
    warns ``lora_partial`` on a file that applied in full -- twice."""
    asked = []

    def declared(tensors, components):
        asked.append(components)
        want = TE if "text_encoder" in components else ""
        return sum(1 for key in tensors
                   if key.endswith(".lora_down.weight")
                   and key.startswith(TE) is bool(want))

    session = make_session(warned, count_declared_branches=declared)
    files = session.parse([{"path": write_two_namespace_lora(tmp_path),
                            "strength": STRENGTH}])
    transformer, encoder = _Stub(), _Stub()
    session.load(files, [component(transformer)])
    session.load(files, [encoder_component(encoder)])

    assert asked == [("transformer",), ("text_encoder",)]
    assert not warned
    assert composites(transformer) == set(TARGETS)
    assert composites(encoder) == set(TARGETS)


def test_a_zero_target_hook_may_refuse_with_its_own_code(tmp_path, warned):
    """A second zero-target code -- Ideogram 4's ``lora_uncond_unavailable`` --
    must reach both halves of the reporting contract without the architecture
    raising past the session.

    REVERT THAT PROVES THIS BITES: return ``error.message`` instead of ``error``.
    The refusal is then tagged ``lora_incompatible`` on the exception AND in the
    warning, and the dedicated code is lost."""
    foreign = tmp_path / "foreign.safetensors"
    save_file({"nowhere.lora_down.weight": torch.zeros(RANK, WIDTH),
               "nowhere.lora_up.weight": torch.zeros(WIDTH, RANK)}, str(foreign))

    def describe(file, _counts):
        return AdapterIncompatible(f"LoRA '{file.name}' is for the other branch",
                                   code="stub_other_branch")

    session = make_session(warned, describe_zero_targets=describe)
    with pytest.raises(AdapterIncompatible) as excinfo:
        session.load([{"path": str(foreign)}], [component(_Stub())])

    assert excinfo.value.code == "stub_other_branch"
    assert warned == [("stub_other_branch",
                       "LoRA 'foreign.safetensors' is for the other branch")]


def test_a_zero_target_hook_may_decline_to_judge_a_file_this_pass_cannot_bind(
        tmp_path, warned):
    """MiniT2I installs in two passes; the pass that covers the text encoder must
    not refuse a file that the transformer pass applies in full.

    REVERT THAT PROVES THIS BITES: return the message text instead of ``None``.
    The second pass then refuses, and the first pass's composites are the state
    the caller is left with."""
    def describe(file, counts):
        if "text_encoder" in counts.per_component:
            return None  # this pass covers no part of the file
        return f"LoRA '{file.name}' matched nothing"

    session = make_session(warned, describe_zero_targets=describe)
    files = session.parse([{"path": write_lora(tmp_path), "strength": STRENGTH}])
    transformer, encoder = _Stub(), _Stub()
    session.load(files, [component(transformer)])
    result = session.load(files, [encoder_component(encoder)])

    assert result.applied == 0
    assert not warned
    assert not composites(encoder)
    assert composites(transformer) == set(TARGETS)


def test_a_pass_over_a_component_that_is_not_loaded_is_still_that_pass(
        tmp_path, warned):
    """The zero-target hook reads which pass it is being asked about from
    ``counts.per_component``, and an unloaded component must still appear there:
    MiniT2I runs its text-encoder pass whether or not a text encoder is loaded.

    REVERT THAT PROVES THIS BITES: filter ``load``'s component list by
    ``module is not None`` again. ``per_component`` is then empty, the hook
    cannot tell which pass it is in, and a transformer-only file is refused by
    the text-encoder pass."""
    passes = []
    session = make_session(warned, describe_zero_targets=lambda file, counts: (
        passes.append(tuple(counts.per_component)), None)[1])
    files = session.parse([{"path": write_lora(tmp_path), "strength": STRENGTH}])
    session.load(files, [AdapterComponent(name="text_encoder", module=None,
                                          iter_targets=iter_targets,
                                          build_branch=te_build_branch)])

    assert passes == [("text_encoder",)]


def test_two_passes_over_one_parsed_file_read_it_once_and_share_a_branch_name(
        tmp_path, warned, monkeypatch):
    """An architecture whose install is split in time used to pay a full
    safetensors read per pass, and its filtered per-pass lists gave one file two
    different branch names.

    REVERT THAT PROVES THIS BITES: hand each pass the request dicts instead of
    ``parse``'s files. The read count doubles."""
    reads = []
    real_open = safetensors.safe_open
    monkeypatch.setattr(safetensors, "safe_open",
                        lambda path, **kw: (reads.append(path), real_open(path, **kw))[1])

    def declared_for_pass(tensors, components):
        want_te = "text_encoder" in components
        return sum(1 for key in tensors
                   if key.endswith(".lora_down.weight")
                   and key.startswith(TE) is want_te)

    session = make_session(warned, count_declared_branches=declared_for_pass)
    files = session.parse([{"path": write_two_namespace_lora(tmp_path),
                            "strength": STRENGTH}])
    assert len(reads) == 1

    transformer, encoder = _Stub(), _Stub()
    session.load(files, [component(transformer)])
    session.load(files, [encoder_component(encoder)])

    assert len(reads) == 1
    assert (list(transformer.a.branch_names) == list(encoder.a.branch_names)
            == ["0:both.safetensors"])


def test_the_console_label_and_the_message_label_are_separate(tmp_path, warned):
    """One architecture spells itself differently on the console and in the
    sentence a user reads, and one string cannot be both.

    REVERT THAT PROVES THIS BITES: use ``self._label`` in
    ``AdapterSession._load_failed``. The message then carries the console's
    spelling."""
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    logged = []
    session = make_session(warned, label="Stub", message_label="Stub Model LoRA",
                           log=logged.append)

    with pytest.raises(AdapterLoadFailed) as excinfo:
        session.load([{"path": str(broken)}], [component(_Stub())])

    assert excinfo.value.message.startswith(
        "Stub Model LoRA 'broken.safetensors' could not be applied")
    assert logged and all(line.startswith("[Stub]") for line in logged)


def test_component_kind_filtering_and_all_disabled_warning(tmp_path, warned):
    """apply_to_unet and apply_to_text_encoder select components, and both False warns without refusal."""
    transformer, encoder = _Stub(), _Stub()
    t_comp = component(transformer)
    e_comp = encoder_component(encoder)

    def count_branches(tensors, components):
        has_t = "transformer" in components
        has_e = "text_encoder" in components
        if has_t and has_e:
            return sum(1 for k in tensors if k.endswith(".lora_down.weight"))
        if has_t:
            return sum(1 for k in tensors if k.endswith(".lora_down.weight") and not k.startswith(TE))
        if has_e:
            return sum(1 for k in tensors if k.endswith(".lora_down.weight") and k.startswith(TE))
        return 0

    path = write_two_namespace_lora(tmp_path)
    session = make_session(warned, count_declared_branches=count_branches)

    # 1. apply_to_unet=False: only encoder gets branches
    session.load([{"path": path, "apply_to_unet": False, "apply_to_text_encoder": True}],
                 [t_comp, e_comp])
    assert not hasattr(transformer.a, "branch_names")
    assert getattr(encoder.a, "branch_names", ()) == ("0:both.safetensors",)
    session.unload([t_comp, e_comp])

    # 2. apply_to_text_encoder=False: only transformer gets branches
    session.load([{"path": path, "apply_to_unet": True, "apply_to_text_encoder": False}],
                 [t_comp, e_comp])
    assert getattr(transformer.a, "branch_names", ()) == ("0:both.safetensors",)
    assert not hasattr(encoder.a, "branch_names")
    session.unload([t_comp, e_comp])

    # 3. Both disabled: warns lora_no_targets and does not refuse
    session.load([{"path": path, "apply_to_unet": False, "apply_to_text_encoder": False}],
                 [t_comp, e_comp])
    assert not hasattr(transformer.a, "branch_names")
    assert not hasattr(encoder.a, "branch_names")
    assert any(code == "lora_no_targets" for code, _ in warned)


def test_step_range_dynamic_activation(tmp_path, warned):
    """step_range controls branch activation dynamically via session.set_step."""
    model = _Stub()
    comp = component(model)
    path = write_lora(tmp_path)
    session = make_session(warned)

    assert not session.has_step_range
    session.load([{"path": path, "step_range": [200, 800]}], [comp])
    assert session.has_step_range

    composite = model.a
    branch_name = "0:a.safetensors"
    assert composite.has_branch(branch_name)

    # Step 100 of 1000 (10%): below 20% -> inactive
    session.set_step(100, 1000)
    assert not composite.is_active(branch_name)

    # Step 500 of 1000 (50%): between 20% and 80% -> active
    session.set_step(500, 1000)
    assert composite.is_active(branch_name)

    # Step 900 of 1000 (90%): above 80% -> inactive
    session.set_step(900, 1000)
    assert not composite.is_active(branch_name)

    session.unload([comp])
    assert not session.has_step_range
