"""Phase U-2-5: the two rules the exit smoke encodes, and the invariant SS11 2b-5 names.

Three things live here.

* **The prefix forward sits OUTSIDE the generation loop's checkpointed region**
  (SENSENOVA_TRAINING_DESIGN.md SS11, Phase 2b-5). The generation backward
  recomputes its own layers; it must not recompute the understanding half's.
  With a negative control that puts the prefix build inside the checkpointed
  segment and shows the count doubling, because an invariant no experiment can
  violate is not being measured.
* **The U-2-5 update-nonzero criterion** -- which is NOT "everything moved". The
  five layer-41 understanding projections a t2i loss cannot reach are named, and
  the "all 294 moved" assertion is recorded here as the thing that is SUPPOSED
  to fail on them.
* **What the production reader must produce per (branch x format)**. The exit
  smoke's reload arm had the ``gen`` answer hardcoded, so the two formats that
  had never been read back -- ``und`` x ``mixed`` (inverted: generation half
  stays int8) and ``both`` x ``bf16`` (no int8 half at all) -- could not be
  checked by it. The rule is cross-checked against a real round trip through the
  production read sequence rather than restated.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.ideogram4.vendor.int8_linear import Int8Linear
from core.models.sensenova.sensenova_lora import (
    iter_sensenova_lora_targets,
    und_gradient_unreachable_paths,
)
from core.training.ops import sensenova_ops
from core.training.probes.sensenova_full_finetune import (
    _effective_format,
    _halves,
    expected_read_shape,
    train_arm_failures,
    u2_5_unmoved_expectation,
)

from sensenova_full_finetune_save_test import (  # noqa: E402
    _BRANCHES, _FORMATS, _load_back, _save, _trained_tree,
)

_DEPTH = 3
_WIDTH = 2


# ---------------------------------------------------------------------------
# A. The prefix forward sits outside the checkpointed region (SS11 2b-5)
# ---------------------------------------------------------------------------

class _CountingUndLayer(nn.Module):
    """An understanding layer that records every invocation against a shared log."""

    attention_type = "full_attention"

    def __init__(self, log: list, index: int):
        super().__init__()
        self.k = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.v = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self._log = log
        self._index = index

    def forward(self, hidden_states, **kwargs):
        assert kwargs["return_kv"] is True
        self._log.append(self._index)
        keys = self.k(hidden_states).unsqueeze(1)
        values = self.v(hidden_states).unsqueeze(1)
        return hidden_states + keys.squeeze(1), keys, values


class _CountingUndModel(nn.Module):
    def __init__(self, log: list, layers: int = _DEPTH):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CountingUndLayer(log, i) for i in range(layers)]
        )
        self.embed_tokens = nn.Embedding(8, _WIDTH)
        self.config = SimpleNamespace(num_hidden_layers=layers, attention_dropout=0.0)


class _GenLayer(nn.Module):
    """A generation layer that consumes the prefix K/V it is handed."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index
        self.proj = nn.Linear(_WIDTH, _WIDTH, bias=False)
        self.seen_key_ids: list[int] = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("layer.__call__ must be bypassed explicitly")

    def forward(self, hidden_states, **kwargs):
        layer = kwargs["past_key_values"].layers[self.index]
        self.seen_key_ids.append(id(layer.keys))
        return self.proj(hidden_states) + layer.keys.mean()


class _GenModel(nn.Module):
    def __init__(self, layers: int = _DEPTH):
        super().__init__()
        self.layers = nn.ModuleList([_GenLayer(i) for i in range(layers)])
        self.norm_mot_gen = nn.LayerNorm(_WIDTH)


def _und_inputs(length: int = 3):
    t_idx = torch.arange(length, dtype=torch.long)
    zeros = torch.zeros_like(t_idx)
    return (
        torch.ones(1, length, dtype=torch.long),
        torch.stack([t_idx, zeros, zeros], dim=0),
        {"full_attention": None},
    )


def _gen_pass(gen_model, cache, *, checkpoint_layers: bool):
    hidden = torch.randn(1, 3, _WIDTH, requires_grad=True)
    return sensenova_ops.forward_gen_decoder_layers(
        gen_model,
        hidden,
        indexes=torch.arange(9, dtype=torch.long).reshape(3, 3),
        prefix_cache=cache,
        checkpoint_layers=checkpoint_layers,
        trainable_prefix=True,
    )


def test_the_generation_backward_never_recomputes_the_understanding_prefix():
    """The invariant: the prefix is built once, outside the checkpointed region.

    The prefix loop is run WITHOUT its own checkpointing here, so the only thing
    that could re-run an understanding layer is the generation region. Each of
    the three understanding layers must therefore be invoked exactly once, for
    the whole forward AND backward.
    """
    log: list[int] = []
    und = _CountingUndModel(log)
    gen = _GenModel()
    input_ids, indexes, mask = _und_inputs()

    cache = sensenova_ops.forward_und_prefix_layers(
        und, input_ids, indexes, mask, checkpoint_layers=False
    )
    after_prefix = list(log)
    assert after_prefix == [0, 1, 2]

    output = _gen_pass(gen, cache, checkpoint_layers=True)
    # The generation FORWARD runs no understanding layer at all.
    assert log == after_prefix

    output.square().mean().backward()
    # ...and neither does its backward, which recomputes its own segments.
    assert log == after_prefix, (
        f"the generation backward re-ran understanding layers {log[len(after_prefix):]}: "
        "the prefix forward is inside the checkpointed region"
    )
    assert und.layers[0].k.weight.grad is not None


def test_the_negative_control_puts_the_prefix_inside_and_the_count_doubles():
    """The violation the test above would catch, demonstrated.

    Without this, "the count stayed at three" is equally consistent with a
    counter that never increments.
    """
    log: list[int] = []
    und = _CountingUndModel(log)
    gen = _GenModel()
    input_ids, indexes, mask = _und_inputs()
    hidden = torch.randn(1, 3, _WIDTH, requires_grad=True)

    def prefix_and_gen(states):
        cache = sensenova_ops.forward_und_prefix_layers(
            und, input_ids, indexes, mask, checkpoint_layers=False
        )
        return sensenova_ops.forward_gen_decoder_layers(
            gen,
            states,
            indexes=torch.arange(9, dtype=torch.long).reshape(3, 3),
            prefix_cache=cache,
            checkpoint_layers=False,
            trainable_prefix=True,
        )

    output = checkpoint(prefix_and_gen, hidden, use_reentrant=False)
    assert log == [0, 1, 2]
    output.square().mean().backward()
    assert log == [0, 1, 2, 0, 1, 2], (
        "the negative control did not reproduce the recompute it exists to model"
    )


def test_the_generation_recompute_reads_the_same_prefix_tensors():
    """Identity, not just value: the cache is closure-captured, never rebuilt.

    A recompute that produced EQUAL K/V from a re-run prefix would pass a value
    comparison while paying for the understanding forward twice.
    """
    log: list[int] = []
    und = _CountingUndModel(log)
    gen = _GenModel()
    input_ids, indexes, mask = _und_inputs()
    cache = sensenova_ops.forward_und_prefix_layers(
        und, input_ids, indexes, mask, checkpoint_layers=False
    )
    ids_before = [(id(layer.keys), id(layer.values)) for layer in cache.layers]
    values_before = [layer.keys.detach().clone() for layer in cache.layers]

    _gen_pass(gen, cache, checkpoint_layers=True).square().mean().backward()

    assert [(id(l.keys), id(l.values)) for l in cache.layers] == ids_before
    for layer, before in zip(cache.layers, values_before):
        torch.testing.assert_close(layer.keys.detach(), before)
    for layer in gen.layers:
        # Once in the forward, once in the recompute -- and the same object both
        # times, which is what "outside the region" means operationally.
        assert len(layer.seen_key_ids) == 2
        assert len(set(layer.seen_key_ids)) == 1


def test_a_no_grad_prefix_is_still_refused_under_a_trainable_generation_pass():
    """The other half of the invariant: outside the region, but differentiable.

    A prefix built under ``no_grad`` sits outside the checkpointed region too,
    and produces a perfectly healthy falling loss while training nothing.
    """
    log: list[int] = []
    und = _CountingUndModel(log)
    gen = _GenModel()
    input_ids, indexes, mask = _und_inputs()
    with torch.no_grad():
        cache = sensenova_ops.forward_und_prefix_layers(
            und, input_ids, indexes, mask, checkpoint_layers=False
        )
    with pytest.raises(ValueError, match="carry no grad_fn"):
        _gen_pass(gen, cache, checkpoint_layers=True)


# ---------------------------------------------------------------------------
# B. The U-2-5 update-nonzero criterion
# ---------------------------------------------------------------------------

def _enumerate(branch: str) -> list[str]:
    from sensenova_full_finetune_save_test import _Decoder

    return [
        path
        for path, _p, _a, _m in iter_sensenova_lora_targets(_Decoder(), branch=branch)
    ]


@pytest.mark.parametrize(
    "branch,total,expected_unmoved",
    [("gen", 294, 0), ("und", 294, 5), ("both", 588, 5)],
)
def test_the_criterion_is_289_and_583_rather_than_294_and_588(
    branch, total, expected_unmoved
):
    paths = _enumerate(branch)
    assert len(paths) == total
    predicted = u2_5_unmoved_expectation(paths, 42)
    assert len(predicted) == expected_unmoved
    assert len(paths) - len(predicted) == total - expected_unmoved


def test_the_five_are_named_rather_than_counted():
    predicted = u2_5_unmoved_expectation(_enumerate("und"), 42)
    assert predicted == sorted({
        "language_model.model.layers.41.self_attn.q_proj",
        "language_model.model.layers.41.self_attn.o_proj",
        "language_model.model.layers.41.mlp.gate_proj",
        "language_model.model.layers.41.mlp.up_proj",
        "language_model.model.layers.41.mlp.down_proj",
    })
    # k_proj and v_proj of the SAME layer ARE trained: generation layer 41
    # consumes their K/V. A criterion that exempted the layer would be wrong.
    assert "language_model.model.layers.41.self_attn.k_proj" not in predicted
    assert "language_model.model.layers.41.self_attn.v_proj" not in predicted


def test_the_generation_enumeration_is_disjoint_from_the_five():
    """Why the ``gen`` branch expects 294 of 294 and not 289.

    The generation half's names carry the ``_mot_gen`` / ``mlp_mot_gen`` suffix,
    so the five never appear in that enumeration -- no subtraction applies.
    """
    assert u2_5_unmoved_expectation(_enumerate("gen"), 42) == []
    assert not set(_enumerate("gen")) & und_gradient_unreachable_paths(42)


def _und_verdict(unmoved, steps=(1, 2, 3)):
    """Drive the probe's own verdict function over the und enumeration."""
    paths = _enumerate("und")
    return train_arm_failures(
        moved=[p for p in paths if p not in unmoved],
        unmoved=list(unmoved),
        of=len(paths),
        predicted_unmoved=u2_5_unmoved_expectation(paths, 42),
        steps=list(steps),
    )


def test_the_criterion_is_silent_on_a_correct_und_run():
    """289 moved with the five predicted ones unmoved raises nothing."""
    assert _und_verdict(sorted(und_gradient_unreachable_paths(42))) == []


def test_the_criterion_fires_when_everything_moved():
    """The negative control for the criterion, through the probe's own code.

    "All 294 moved" is what the design says must FAIL here. Driven through
    ``train_arm_failures`` rather than through a bare ``assert 289 == 294``,
    which would exercise neither the expectation nor the comparison -- the
    version this replaces did exactly that and would have passed against a
    criterion that never fired.
    """
    failures = _und_verdict([])
    assert len(failures) == 1
    assert "294 moved" in failures[0]
    assert "layers.41.self_attn.q_proj" in failures[0]


def test_the_criterion_fires_on_a_dead_hook_it_did_not_predict():
    """The failure the census exists for: a parameter no hook updated.

    A tensor outside the unreachable five staying still is a silent
    non-update, and must be reported even though the COUNT (5) is right.
    """
    unmoved = sorted(und_gradient_unreachable_paths(42))
    swapped = unmoved[1:] + ["language_model.model.layers.0.self_attn.k_proj"]
    failures = _und_verdict(sorted(swapped))
    assert len(failures) == 1
    assert "layers.0.self_attn.k_proj" in failures[0]


def test_a_short_run_is_collected_rather_than_raised():
    """The step-count check is a fact about the run, so it joins the same list.

    It used to raise after the run and before any JSON was written, discarding
    every number the run had just cost.
    """
    failures = _und_verdict(sorted(und_gradient_unreachable_paths(42)), steps=(1, 2))
    assert len(failures) == 1
    assert "expected 3 steps" in failures[0]


def test_both_verdicts_are_reported_together():
    """Neither clause short-circuits the other."""
    failures = _und_verdict([], steps=(1,))
    assert len(failures) == 2


def test_the_gen_branch_criterion_expects_no_subtraction():
    """The same function, on the enumeration the five are disjoint from."""
    paths = _enumerate("gen")
    assert train_arm_failures(
        moved=paths, unmoved=[], of=len(paths),
        predicted_unmoved=u2_5_unmoved_expectation(paths, 42),
        steps=[1, 2, 3],
    ) == []
    # ...and one unmoved generation Linear is a failure, with nothing exempt.
    assert len(train_arm_failures(
        moved=paths[1:], unmoved=paths[:1], of=len(paths),
        predicted_unmoved=u2_5_unmoved_expectation(paths, 42),
        steps=[1, 2, 3],
    )) == 1


# ---------------------------------------------------------------------------
# C. What the reader must produce, per (branch x format)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("save_format", _FORMATS)
def test_the_read_shape_rule_matches_a_real_round_trip(tmp_path, branch, save_format):
    """The rule the reload arm applies, checked against the writer and reader.

    Not a second copy of the rule: the file is written by the production writer
    and read by the production read sequence, and what comes out is compared
    against what ``expected_read_shape`` claims.
    """
    written, census = _save(_trained_tree(branch), tmp_path, branch, save_format)
    effective = _effective_format(branch, save_format)
    assert census["effective_format"] == effective

    model, metadata = _load_back(written)
    assert metadata["sensenova_trained_branch"] == branch
    assert metadata["sensenova_save_format"] == effective
    assert metadata["sensenova_save_format_requested"] == save_format

    want = expected_read_shape(branch, effective)
    for half, shape in want.items():
        modules = [
            m for _p, _pa, _a, m in iter_sensenova_lora_targets(model, branch=half)
        ]
        assert len(modules) == 294
        if shape == "int8":
            assert all(type(m) is Int8Linear for m in modules), half
        else:
            assert all(
                isinstance(getattr(m, "weight", None), nn.Parameter)
                and m.weight.dtype.is_floating_point
                for m in modules
            ), half


def test_the_gen_hardcoded_expectation_is_wrong_for_the_two_untested_formats():
    """The negative control for the reload arm's actual defect.

    Before this rule existed the arm asked one question -- "are all 294
    generation Linears floating and all 294 understanding ones Int8Linear?" --
    which is the ``gen`` x ``mixed`` answer. Applied to the two shapes that had
    never been read back it is false in both directions.
    """
    gen_mixed = {"gen": "float", "und": "int8"}
    assert expected_read_shape("gen", "mixed") == gen_mixed
    # und x mixed is the INVERSE, not a variation of it.
    assert expected_read_shape("und", "mixed") == {"gen": "int8", "und": "float"}
    assert expected_read_shape("und", "mixed") != gen_mixed
    # both x bf16 has no int8 half at all, so the old check would have reported
    # 0 of 294 understanding Linears as int8 on a correct file.
    assert expected_read_shape("both", "bf16") == {"gen": "float", "und": "float"}
    assert expected_read_shape("both", "bf16") != gen_mixed


def _reload_arm_on(tmp_path, monkeypatch, written, expect_payload):
    """Run the probe's reload arm over a synthetic checkpoint.

    ONE step is substituted: ``load_sensenova_from_path``, which builds a real
    ``NEOChatModel`` and therefore needs the 17.6 GiB checkpoint's full config.
    Everything the arm is being tested for is real -- the file came from the
    production WRITER, the tree comes from the production READ SEQUENCE
    (``install_sensenova_state_dict``: guard, census, swap, verify, assign), and
    the metadata is the file's own. What the substitution removes is model
    construction from config, which the round-trip tests above do not need
    either and which no assertion here depends on.
    """
    import json

    from core.models.sensenova import loader as loader_mod
    from core.training.probes import sensenova_full_finetune as probe

    model, metadata = _load_back(written)
    monkeypatch.setattr(
        loader_mod, "load_sensenova_from_path",
        lambda path, **kwargs: {"transformer": model, "metadata": dict(metadata)},
    )
    expect = tmp_path / "train.json"
    expect.write_text(json.dumps(expect_payload), encoding="utf-8")
    return probe._run_reload_arm(SimpleNamespace(expect=str(expect))), model


def test_the_reload_arm_reports_a_wrong_shaped_read_rather_than_passing(
    tmp_path, monkeypatch
):
    """The reload arm's verdict, over a real round trip.

    Run against a file the production writer produced, then run again against a
    DELIBERATELY WRONG expectation, so that "no failures" is known to be a result
    and not the only thing this code can say.
    """
    from core.training.probes import sensenova_full_finetune as probe

    branch, save_format = "und", "mixed"
    written, _census = _save(_trained_tree(branch), tmp_path, branch, save_format)
    model, _metadata = _load_back(written)

    digests = {
        path: probe._linear_digest(module.weight)
        for path, _p, _a, module in iter_sensenova_lora_targets(model, branch=branch)
    }
    payload = {
        "checkpoint_entry": written,
        "branch": branch,
        "save_format_requested": save_format,
        "save_format_effective": "mixed",
        "post_train_digests": digests,
    }
    result, _model = _reload_arm_on(tmp_path, monkeypatch, written, payload)
    assert result["failures"] == []
    assert result["branch_from_metadata"] == "und"
    assert result["per_half"]["gen"]["int8"] == 294
    assert result["per_half"]["und"]["float_materialized"] == 294
    assert result["digests"] == {
        "compared": True, "matches": 294, "of": 294, "mismatched": []
    }

    # Negative control: one corrupted digest must be named, not averaged away.
    corrupted = dict(digests)
    victim = sorted(corrupted)[0]
    corrupted[victim] = "0" * 64
    bad, _model = _reload_arm_on(tmp_path, monkeypatch, written, {
        **payload,
        "branch": "gen",  # and a branch the file contradicts
        "post_train_digests": corrupted,
    })
    assert bad["branch_from_metadata"] == "und"  # the FILE's word, not the JSON's
    assert any(victim in f for f in bad["failures"])
    assert any("the training arm said 'gen'" in f for f in bad["failures"])


def test_the_reload_arm_names_a_half_the_reader_produced_in_the_wrong_shape(
    tmp_path, monkeypatch
):
    """The shape clause, exercised by lying about the format rather than the file.

    Claiming ``bf16`` over a file whose generation half is genuinely int8 is the
    read the gen-hardcoded arm would have performed on ``und`` x ``mixed``.
    """
    from core.models.sensenova import loader as loader_mod
    from core.training.probes import sensenova_full_finetune as probe

    written, _census = _save(_trained_tree("und"), tmp_path, "und", "mixed")
    model, metadata = _load_back(written)
    lying = dict(metadata)
    lying["sensenova_save_format"] = "bf16"
    monkeypatch.setattr(
        loader_mod, "load_sensenova_from_path",
        lambda path, **kwargs: {"transformer": model, "metadata": lying},
    )
    import json
    expect = tmp_path / "train.json"
    expect.write_text(json.dumps({
        "checkpoint_entry": written, "branch": "und",
        "save_format_requested": "mixed", "post_train_digests": {},
    }), encoding="utf-8")

    result = probe._run_reload_arm(SimpleNamespace(expect=str(expect)))
    assert any("gen half (frozen)" in f and "floating nn.Parameter" in f
               for f in result["failures"])


def test_the_reload_arm_refuses_a_file_with_no_branch_metadata(
    tmp_path, monkeypatch
):
    """G: the fallback made the cross-check compare a value with itself."""
    import json

    from core.models.sensenova import loader as loader_mod
    from core.training.probes import sensenova_full_finetune as probe

    written, _census = _save(_trained_tree("und"), tmp_path, "und", "mixed")
    model, _metadata = _load_back(written)
    monkeypatch.setattr(
        loader_mod, "load_sensenova_from_path",
        lambda path, **kwargs: {"transformer": model, "metadata": {}},
    )
    expect = tmp_path / "train.json"
    expect.write_text(json.dumps({
        "checkpoint_entry": written, "branch": "und",
        "save_format_requested": "mixed", "post_train_digests": {},
    }), encoding="utf-8")

    with pytest.raises(AssertionError, match="carries no sensenova_trained_branch"):
        probe._run_reload_arm(SimpleNamespace(expect=str(expect)))


def test_main_writes_the_json_before_it_raises(tmp_path, monkeypatch):
    """The write-then-raise contract, which nothing covered.

    A failing criterion is the measurement; a run that cost 25 GiB of writes
    must not lose its own numbers to the assertion that reports them.
    """
    from core.training.probes import sensenova_full_finetune as probe

    out = tmp_path / "result.json"
    payload = {"arm": "train", "moved_census": {"moved": 294},
               "failures": ["update-nonzero census: 294 of 294 moved"]}

    monkeypatch.setattr(probe, "_require_repo_venv", lambda: None)
    monkeypatch.setattr(probe, "_run_train_arm", lambda args: dict(payload))
    monkeypatch.setattr(probe, "_parse_args", lambda: SimpleNamespace(
        arm="train", out=str(out)))

    with pytest.raises(AssertionError, match="294 of 294 moved"):
        probe.main()

    import json
    assert json.loads(out.read_text(encoding="utf-8")) == payload


def test_main_is_silent_when_there_is_nothing_to_report(tmp_path, monkeypatch):
    from core.training.probes import sensenova_full_finetune as probe

    out = tmp_path / "result.json"
    monkeypatch.setattr(probe, "_require_repo_venv", lambda: None)
    monkeypatch.setattr(probe, "_run_train_arm",
                        lambda args: {"arm": "train", "failures": []})
    monkeypatch.setattr(probe, "_parse_args", lambda: SimpleNamespace(
        arm="train", out=str(out)))
    assert probe.main() == 0


def test_mixed_degenerates_only_where_there_is_no_frozen_half():
    for branch in _BRANCHES:
        trained, frozen = _halves(branch)
        assert set(trained) | set(frozen) == {"gen", "und"}
        assert not (set(trained) & set(frozen))
    assert _effective_format("both", "mixed") == "bf16"
    assert _effective_format("gen", "mixed") == "mixed"
    assert _effective_format("und", "mixed") == "mixed"
    # The degeneracy is mixed-only: bf16 and int8 mean the same thing on every
    # branch.
    for branch in _BRANCHES:
        assert _effective_format(branch, "bf16") == "bf16"
        assert _effective_format(branch, "int8") == "int8"
