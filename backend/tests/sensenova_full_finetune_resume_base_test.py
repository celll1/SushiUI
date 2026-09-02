"""What a SenseNova full fine-tune may resume from, and what it refuses.

Run with:
    venv/Scripts/python.exe -m pytest \
        backend/tests/sensenova_full_finetune_resume_base_test.py -v

THE GAP THIS COVERS. ``_assert_supported_quantized_training_base`` answers
"is this a distributable training base", and for a full fine-tune its answer is
plain int8 or nothing. That left the ``both`` branch with no lossless resume:
``bf16`` (and ``mixed``, which degenerates to it there) is byte-exact through
this repo's reader but carries no quantized Linear at all, and ``int8`` is
accepted only by requantizing every trained weight on every save.

``accept_resume_shaped_base`` answers the narrower question -- is this tree the
layout the run was already training in -- and it is reachable ONLY from the
resume path. The class census on the constructed tree decides; the checkpoint's
own metadata is a required cross-check that can narrow acceptance and never
widen it.

NEGATIVE CONTROLS
-----------------
The base-substitution gate is kept exactly as it was: a bf16 tree handed over
as ``model_path`` is still refused, for both methods, and
a resume-shaped tree with a missing, contradicting or absent stamp is refused by
name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.models.ideogram4.vendor.int8_linear import Int8Linear  # noqa: E402
from core.models.common.single_file_format import TRANSFORMER_PREFIX  # noqa: E402
from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    _SENSENOVA_RESUME_FORMAT_FOR_BRANCH,
    _assert_supported_quantized_training_base,
    _half_linear_layout,
    _own_save_format_remedy,
    _resume_selected_checkpoint,
    accept_resume_shaped_base,
    load_components,
)

LAYERS = 42
PER_HALF = 294
RUN_NAME = "20260825_120000_abc123"


# ---------------------------------------------------------------------------
# A MoT-shaped tree, small enough to build 588 Linears of in the test process
# ---------------------------------------------------------------------------

def _int8() -> Int8Linear:
    return Int8Linear(8, 8, False, torch.bfloat16)


def _float() -> nn.Linear:
    return nn.Linear(8, 8, bias=False).to(torch.bfloat16)


def _convrot() -> nn.Module:
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    # K must be divisible by the 256 groupsize this class requires.
    return ConvRotInt8Linear(256, 8, False, torch.bfloat16, convrot_groupsize=256,
                             marker_numel=1)


class _MoTTree(nn.Module):
    """``language_model.model.layers`` in the shape the enumerator walks.

    ``gen_factory``/``und_factory`` decide what each half's 294 Linears are, so
    one class covers every layout under test.
    """

    use_pixel_head = True
    use_deep_fm_head = False

    def __init__(self, gen_factory=_int8, und_factory=_int8, layers=LAYERS):
        super().__init__()
        core = nn.Module()
        blocks = []
        for _ in range(layers):
            block = nn.Module()
            attn = nn.Module()
            for name in ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen",
                         "o_proj_mot_gen"):
                setattr(attn, name, gen_factory())
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(attn, name, und_factory())
            block.self_attn = attn
            gen_mlp = nn.Module()
            und_mlp = nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(gen_mlp, name, gen_factory())
                setattr(und_mlp, name, und_factory())
            block.mlp_mot_gen = gen_mlp
            block.mlp = und_mlp
            blocks.append(block)
        core.layers = nn.ModuleList(blocks)
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


def _trainer(tmp_path: Path, *, entry: str, resume: str = "latest",
             run_name: str = RUN_NAME, output_dir: Path = None,
             configured_model_path: str = None, weight_dtype=torch.bfloat16):
    directory = output_dir if output_dir is not None else tmp_path
    kwargs = dict(
        model_path=str(directory / entry),
        output_dir=directory,
        run_name=run_name,
        resume_from_checkpoint=resume,
        log_prefix="[test]",
        weight_dtype=weight_dtype,
    )
    if configured_model_path is not None:
        kwargs["configured_model_path"] = configured_model_path
    return SimpleNamespace(**kwargs)


def _write_base_safetensors(path: Path, tree: nn.Module) -> Path:
    """A base int8 model file: ``tree``'s state dict, TRANSFORMER_PREFIX-keyed.

    Mirrors what the SenseNova single-file reader expects on disk -- see
    ``core.models.sensenova.loader.restore_sensenova_frozen_half_from_base``.
    """
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {f"{TRANSFORMER_PREFIX}{k}": v.contiguous() for k, v in tree.state_dict().items()}
    save_file(sd, str(path))
    return path


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _sidecars(tmp_path: Path, step: int, run_name: str = RUN_NAME):
    for suffix in ("_optimizer.pt", "_state.json"):
        _write(tmp_path / f"{run_name}_step_{step:06d}{suffix}")


def _metadata(branch: str, save_format: str, **extra: str) -> dict:
    return {
        "sensenova_trained_branch": branch,
        "sensenova_save_format": save_format,
        "step": "100",
        **extra,
    }


def _matching_frozen_tree(base_tree: nn.Module, half: str, trained_half: str) -> nn.Module:
    """A checkpoint-shaped tree: ``half`` float, dequantized exactly from
    ``base_tree``; the other half arbitrary float (the trained half's content
    is never verified against the base)."""
    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    for (_, _, _, base_mod), (_, _, _, ck_mod) in zip(
        iter_sensenova_lora_targets(base_tree, branch=half),
        iter_sensenova_lora_targets(tree, branch=half),
    ):
        with torch.no_grad():
            ck_mod.weight.copy_(
                base_mod.weight * base_mod.weight_scale.to(torch.bfloat16).unsqueeze(1)
            )
    return tree


# ---------------------------------------------------------------------------
# The census the acceptance is decided on
# ---------------------------------------------------------------------------

def test_layout_counts_each_half_by_exact_class():
    layout = _half_linear_layout(_MoTTree(gen_factory=_float, und_factory=_int8))
    assert layout["gen"]["counts"] == {"float": PER_HALF, "int8": 0, "other": 0}
    assert layout["und"]["counts"] == {"float": 0, "int8": PER_HALF, "other": 0}
    assert layout["gen"]["total"] == layout["und"]["total"] == PER_HALF


def test_a_convrot_half_is_other_not_int8():
    """``ConvRotInt8Linear`` subclasses ``Int8Linear``; an isinstance census
    would call a rotated base resumable."""
    layout = _half_linear_layout(_MoTTree(gen_factory=_float, und_factory=_convrot))
    assert layout["und"]["counts"]["int8"] == 0
    assert layout["und"]["counts"]["other"] == PER_HALF
    assert "ConvRotInt8Linear" in layout["und"]["first_other"]


# ---------------------------------------------------------------------------
# Leg 1: is this a resume at all
# ---------------------------------------------------------------------------

def test_resume_selection_requires_all_four_conditions(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    assert _resume_selected_checkpoint(_trainer(tmp_path, entry=entry)) is not None

    # (a) the run did not ask to resume
    assert _resume_selected_checkpoint(
        _trainer(tmp_path, entry=entry, resume="")) is None
    # (b) the file is not this run's checkpoint name
    other = _write(tmp_path / "someone_elses_step_000100.safetensors")
    assert _resume_selected_checkpoint(
        SimpleNamespace(model_path=str(other), output_dir=tmp_path,
                        run_name=RUN_NAME, resume_from_checkpoint="latest")) is None
    # (c) the file is outside this run's output_dir
    elsewhere = _write(tmp_path / "elsewhere" / entry)
    assert _resume_selected_checkpoint(
        SimpleNamespace(model_path=str(elsewhere), output_dir=tmp_path,
                        run_name=RUN_NAME, resume_from_checkpoint="latest")) is None
    # (d) no step number
    nostep = _write(tmp_path / f"{RUN_NAME}_step_final.safetensors")
    assert _resume_selected_checkpoint(
        SimpleNamespace(model_path=str(nostep), output_dir=tmp_path,
                        run_name=RUN_NAME, resume_from_checkpoint="latest")) is None
    # a sharded save is an entry too
    sharded = f"{RUN_NAME}_step_000200.safetensors.index.json"
    _write(tmp_path / sharded)
    assert _resume_selected_checkpoint(
        _trainer(tmp_path, entry=sharded)) is not None


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------

def test_a_mixed_checkpoint_resumes_its_own_single_half_run(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    assert accept_resume_shaped_base(
        _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
        branch="gen") == "mixed"


def test_a_bf16_checkpoint_resumes_its_own_both_halves_run(tmp_path):
    """The case that had no lossless resume before: ``both`` + ``bf16``."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    assert accept_resume_shaped_base(
        _trainer(tmp_path, entry=entry), tree, _metadata("both", "bf16"),
        branch="both") == "bf16"
    assert _SENSENOVA_RESUME_FORMAT_FOR_BRANCH["both"] == "bf16"


def test_an_int8_tree_is_left_to_the_shipped_gate(tmp_path):
    """The distributed layout keeps its existing route: gate, then materialize."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    tree = _MoTTree()
    assert accept_resume_shaped_base(
        _trainer(tmp_path, entry=entry), tree, _metadata("gen", "int8"),
        branch="gen") is None


def test_a_float_tree_that_is_not_a_resume_is_left_to_the_shipped_gate(tmp_path):
    """Handed over as ``model_path``, a bf16 file is still not a base."""
    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    not_resuming = SimpleNamespace(
        model_path=str(tmp_path / "some_release.safetensors"),
        output_dir=tmp_path, run_name=RUN_NAME, resume_from_checkpoint=None)
    assert accept_resume_shaped_base(
        not_resuming, tree, _metadata("both", "bf16"), branch="both") is None
    with pytest.raises(RuntimeError, match="unquantized bf16 base is refused"):
        _assert_supported_quantized_training_base(tree, training_method="full")


def test_the_acceptance_is_announced_on_the_channel_not_only_stdout(tmp_path):
    """Relaxing a gate is at least as worth telling the user about as degrading."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    with patch("core.training.ops.sensenova_ops.emit_training_event") as event:
        accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
            branch="gen")
    assert event.call_count == 1
    level, message = event.call_args.args
    assert level == "info"
    assert event.call_args.kwargs["code"] == "sensenova_resume_base_accepted"
    assert "'mixed'" in message and "step 100" in message
    # The suffix is left off deliberately: BaseTrainer's resume handler
    # substring-matches "safetensor" to decide an error means corruption.
    assert ".safetensors" not in message


def test_a_missing_sidecar_warns_rather_than_refusing(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _write(tmp_path / f"{RUN_NAME}_step_000100_state.json")  # optimizer.pt absent
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    with patch("core.training.ops.sensenova_ops.emit_training_warning") as warn:
        assert accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
            branch="gen") == "mixed"
    assert warn.call_count == 1
    assert warn.call_args.kwargs["code"] == "sensenova_resume_state_incomplete"
    assert "_optimizer.pt" in warn.call_args.args[0]
    assert "_state.json" not in warn.call_args.args[0]


# ---------------------------------------------------------------------------
# Negative controls: the trust rule's refusals
# ---------------------------------------------------------------------------

def test_the_wrong_layout_for_the_branch_is_refused(tmp_path):
    """Neither the 'mixed' Int8Linear layout nor the 'bf16' float layout: a
    frozen half that is ConvRot-quantized is refused either way."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_convrot)
    with pytest.raises(RuntimeError, match="und half of its decoder is not the") as e:
        accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
            branch="gen")
    assert "save_format='mixed' or 'bf16'" in str(e.value)
    # The census itself is pinned, not just the shape of the message.
    assert "gen half: float=294" in str(e.value)
    assert "und half: float=0, int8=0, other=294" in str(e.value)
    assert "ConvRotInt8Linear" in str(e.value)


# ---------------------------------------------------------------------------
# The bf16 single-half fallback: the frozen half restored from the base model
# ---------------------------------------------------------------------------

def test_a_bf16_checkpoint_resumes_a_single_half_run_from_its_base(tmp_path):
    """``sensenova_full_finetune_save_format='bf16'`` on a 'gen' run writes
    BOTH halves float. Resuming restores the frozen 'und' half's Int8Linear
    from the run's own base model, verified tensor-for-tensor first."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)

    base_tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    base_path = _write_base_safetensors(tmp_path / "base" / "sensenova_int8.safetensors",
                                         base_tree)

    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    for (_, _, _, base_mod), (_, _, _, ck_mod) in zip(
        iter_sensenova_lora_targets(base_tree, branch="und"),
        iter_sensenova_lora_targets(tree, branch="und"),
    ):
        with torch.no_grad():
            ck_mod.weight.copy_(
                base_mod.weight * base_mod.weight_scale.to(torch.bfloat16).unsqueeze(1)
            )

    trainer = _trainer(tmp_path, entry=entry, configured_model_path=str(base_path))
    assert accept_resume_shaped_base(
        trainer, tree, _metadata("gen", "bf16"), branch="gen") == "bf16"

    # The frozen half is restored to Int8Linear with the base's own codes/scale.
    for (_, _, _, base_mod), (_, _, _, restored_mod) in zip(
        iter_sensenova_lora_targets(base_tree, branch="und"),
        iter_sensenova_lora_targets(tree, branch="und"),
    ):
        assert type(restored_mod) is Int8Linear
        assert torch.equal(restored_mod.weight, base_mod.weight)
        assert torch.equal(restored_mod.weight_scale, base_mod.weight_scale)
    # The trained half is untouched: still the float weights the checkpoint carried.
    for _, _, _, gen_mod in iter_sensenova_lora_targets(tree, branch="gen"):
        assert type(gen_mod) is nn.Linear


def test_a_bf16_checkpoint_refuses_a_mismatching_base(tmp_path):
    """A base whose frozen half does not dequantize to the checkpoint's saved
    weight is refused by name, not silently substituted."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)

    base_tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    base_path = _write_base_safetensors(tmp_path / "base" / "sensenova_int8.safetensors",
                                         base_tree)

    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    for (_, _, _, base_mod), (_, _, _, ck_mod) in zip(
        iter_sensenova_lora_targets(base_tree, branch="und"),
        iter_sensenova_lora_targets(tree, branch="und"),
    ):
        with torch.no_grad():
            ck_mod.weight.copy_(
                base_mod.weight * base_mod.weight_scale.to(torch.bfloat16).unsqueeze(1)
            )
    # Perturb one frozen-half weight so it no longer matches the base.
    first_frozen = next(iter_sensenova_lora_targets(tree, branch="und"))[3]
    with torch.no_grad():
        first_frozen.weight.add_(1.0)

    trainer = _trainer(tmp_path, entry=entry, configured_model_path=str(base_path))
    with pytest.raises(RuntimeError, match="does not dequantize"):
        accept_resume_shaped_base(trainer, tree, _metadata("gen", "bf16"), branch="gen")


def test_a_bf16_checkpoint_refuses_when_the_base_is_missing(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_float)

    missing_base = str(tmp_path / "base" / "sensenova_int8.safetensors")
    trainer = _trainer(tmp_path, entry=entry, configured_model_path=missing_base)
    with pytest.raises(RuntimeError, match="base model not found"):
        accept_resume_shaped_base(trainer, tree, _metadata("gen", "bf16"), branch="gen")

    # No configured base at all -- refused by name, not a crash on a missing attr.
    trainer_unconfigured = _trainer(tmp_path, entry=entry)
    with pytest.raises(RuntimeError, match="base model path is unknown"):
        accept_resume_shaped_base(
            trainer_unconfigured, tree, _metadata("gen", "bf16"), branch="gen")


def test_int8_format_is_still_refused_unconditionally(tmp_path):
    """The bf16 fallback exists only because the FROZEN half's int8 codes are
    exactly recoverable; the TRAINED half's are not (the int8 writer
    requantizes them, lossily), so 'int8' widens nothing here."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    # What an 'int8'-format save actually produces: no float Linear anywhere.
    tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    assert accept_resume_shaped_base(
        _trainer(tmp_path, entry=entry), tree, _metadata("gen", "int8"),
        branch="gen") is None


def test_the_identity_hint_refuses_a_size_mismatched_base(tmp_path):
    """The cheap size stamp catches a wrong base before any tensor is read."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)

    base_tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    base_path = _write_base_safetensors(tmp_path / "base" / "sensenova_int8.safetensors",
                                         base_tree)
    tree = _matching_frozen_tree(base_tree, "und", trained_half="gen")

    trainer = _trainer(tmp_path, entry=entry, configured_model_path=str(base_path))
    metadata = _metadata("gen", "bf16", sensenova_base_model_identity="99999999999")
    with pytest.raises(RuntimeError, match="byte.*byte"):
        accept_resume_shaped_base(trainer, tree, metadata, branch="gen")


def test_the_stamped_base_path_is_preferred_over_configured_model_path(tmp_path):
    """A future checkpoint's own ``sensenova_base_model_path`` stamp is used
    even when the trainer's configured path points somewhere else (or
    nowhere) -- self-describing, per the design's requirement 2."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)

    base_tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    base_path = _write_base_safetensors(tmp_path / "base" / "sensenova_int8.safetensors",
                                         base_tree)
    tree = _matching_frozen_tree(base_tree, "und", trained_half="gen")

    trainer = _trainer(tmp_path, entry=entry,
                        configured_model_path=str(tmp_path / "nonexistent.safetensors"))
    metadata = _metadata("gen", "bf16", sensenova_base_model_path=str(base_path))
    assert accept_resume_shaped_base(trainer, tree, metadata, branch="gen") == "bf16"


def test_a_sharded_base_is_read_correctly(tmp_path):
    """The base model can be a shard-index save (diffusers convention, >10 GB
    on the real 18.9 GB checkpoint); the streaming reader must follow it."""
    from core.models.common.single_file_format import TRANSFORMER_PREFIX, save_single_file_state

    base_tree = _MoTTree(gen_factory=_int8, und_factory=_int8)
    sd = {f"{TRANSFORMER_PREFIX}{k}": v.contiguous() for k, v in base_tree.state_dict().items()}
    base_dir = tmp_path / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    # Small enough that these ~300 tiny (8x8) tensors span many shards.
    written = save_single_file_state(sd, {}, str(base_dir / "sensenova_int8.safetensors"),
                                      max_shard_bytes=256)
    assert written.endswith(".safetensors.index.json")

    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _matching_frozen_tree(base_tree, "und", trained_half="gen")

    trainer = _trainer(tmp_path, entry=entry, configured_model_path=written)
    assert accept_resume_shaped_base(
        trainer, tree, _metadata("gen", "bf16"), branch="gen") == "bf16"
    for _, _, _, restored_mod in iter_sensenova_lora_targets(tree, branch="und"):
        assert type(restored_mod) is Int8Linear


def test_a_half_materialized_tree_is_refused(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    tree.language_model.model.layers[0].self_attn.q_proj_mot_gen = _int8()
    with pytest.raises(RuntimeError, match="gen half of its decoder is not the"):
        accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
            branch="gen")


def test_a_convrot_frozen_half_is_refused(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    tree = _MoTTree(gen_factory=_float, und_factory=_convrot)
    with pytest.raises(RuntimeError, match="und half of its decoder is not the") as e:
        accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), tree, _metadata("gen", "mixed"),
            branch="gen")
    assert "ConvRotInt8Linear" in str(e.value)


def test_a_resume_shaped_file_with_no_stamp_is_refused(tmp_path):
    """Structure alone does not buy acceptance."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    for metadata in (None, {}, {"sensenova_trained_branch": "gen"},
                     {"sensenova_save_format": "mixed"}):
        with pytest.raises(RuntimeError, match="not this .* own save stamp"):
            accept_resume_shaped_base(
                _trainer(tmp_path, entry=entry), tree, metadata, branch="gen")


def test_a_stamp_that_contradicts_the_tensors_is_refused(tmp_path):
    """The claim is present and the structure is right, but they disagree about
    which branch/format produced the file."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_int8)
    trainer = _trainer(tmp_path, entry=entry)

    with pytest.raises(RuntimeError, match="disagrees with what it is being resumed as") as e:
        accept_resume_shaped_base(trainer, tree, _metadata("und", "mixed"), branch="gen")
    assert "sensenova_trained_branch='und'" in str(e.value)

    with pytest.raises(RuntimeError, match="disagrees with what it is being resumed as") as e:
        accept_resume_shaped_base(trainer, tree, _metadata("gen", "int8"), branch="gen")
    assert "sensenova_save_format='int8'" in str(e.value)


def test_the_stamp_cannot_widen_acceptance_past_the_structure(tmp_path):
    """A file that says all the right things over the wrong tensors is refused
    on the tensors, before the stamp is read at all."""
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    lying = _MoTTree(gen_factory=_int8, und_factory=_float)
    with pytest.raises(RuntimeError, match="gen half of its decoder is not the"):
        accept_resume_shaped_base(
            _trainer(tmp_path, entry=entry), lying, _metadata("gen", "mixed"),
            branch="gen")


# ---------------------------------------------------------------------------
# load_components: what the two routes do to the tree
# ---------------------------------------------------------------------------

def _load_components_trainer(tmp_path, entry, *, resume, branch):
    return SimpleNamespace(
        model_path=str(tmp_path / entry),
        output_dir=tmp_path,
        run_name=RUN_NAME,
        resume_from_checkpoint=resume,
        log_prefix="[test]",
        blocks_to_swap=0,
        weight_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attention_backend="native",
        train_unet=branch in ("gen", "both"),
        train_text_encoder=branch in ("und", "both"),
        config={"training_method": "full_finetune", "optimizer": "adafactor",
                "gradient_accumulation_steps": 1, "batch_size": 1,
                "num_optimizer_groups": 0, "use_ema": False,
                "blocks_to_swap": 0},
        training_dtype=torch.bfloat16,
    )


def _run_load_components(trainer, tree, metadata):
    components = {"transformer": tree, "tokenizer": object(), "config": object(),
                  "metadata": metadata, "config_dict": {}}
    with patch("core.models.sensenova.loader.load_sensenova_from_path",
               return_value=components), \
         patch("core.training.ops.sensenova_ops.setup_attention_backend"), \
         patch("core.training.ops.sensenova_ops.assert_full_finetune_contract"), \
         patch("core.models.sensenova.loader.materialize_int8_decoder_linears") as mat:
        load_components(trainer)
    return mat


def test_load_components_skips_materialization_on_an_accepted_resume(tmp_path):
    entry = f"{RUN_NAME}_step_000100.safetensors"
    _write(tmp_path / entry)
    _sidecars(tmp_path, 100)
    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    trainer = _load_components_trainer(tmp_path, entry, resume="latest", branch="both")
    mat = _run_load_components(trainer, tree, _metadata("both", "bf16"))
    mat.assert_not_called()
    assert trainer.sensenova_resumed_save_format == "bf16"


def test_load_components_still_materializes_an_int8_base(tmp_path):
    entry = "sensenova_int8.safetensors"
    tree = _MoTTree()
    trainer = _load_components_trainer(tmp_path, entry, resume=None, branch="gen")
    mat = _run_load_components(trainer, tree, {})
    mat.assert_called_once()
    assert trainer.sensenova_resumed_save_format is None


def test_load_components_still_refuses_a_bf16_model_path(tmp_path):
    """The shipped base gate, untouched, on the route that is not a resume."""
    tree = _MoTTree(gen_factory=_float, und_factory=_float)
    trainer = _load_components_trainer(
        tmp_path, "someone_elses_bf16.safetensors", resume=None, branch="both")
    with pytest.raises(RuntimeError, match="unquantized bf16 base is refused"):
        _run_load_components(trainer, tree, {})


# ---------------------------------------------------------------------------
# The refusal message, and the run-creation warning
# ---------------------------------------------------------------------------

def test_the_base_refusal_separates_base_substitution_from_resume():
    assert _own_save_format_remedy(None) == ""
    text = _own_save_format_remedy(
        {"sensenova_save_format": "mixed", "sensenova_trained_branch": "gen"})
    assert "sensenova_full_finetune_save_format='mixed'" in text
    assert "only 'int8'" in text
    assert "accept_resume_shaped_base" in text
    assert "RESUMING the run" in text


@pytest.mark.parametrize(
    "train_unet,train_text_encoder,save_format,expected_code",
    [
        (True, False, "mixed", None),
        (True, True, "bf16", None),
        # 'bf16' on a single-half branch resumes via the base-model fallback,
        # not the branch's own lossless format -- a different warning, not a
        # refusal.
        (True, False, "bf16", "sensenova_save_format_resume_needs_base"),
        (True, False, "int8", "sensenova_save_format_not_resumable"),
        (True, True, "int8", "sensenova_save_format_not_resumable"),
        (True, True, "mixed", None),  # degenerates to bf16, which IS the format
    ],
)
def test_run_creation_says_which_resume_the_save_format_leaves(
    train_unet, train_text_encoder, save_format, expected_code
):
    from core.training.train_runner import _warn_on_unresumable_sensenova_save_format

    train_config = {"train_unet": train_unet, "train_text_encoder": train_text_encoder}
    with patch("core.training.training_events.emit_training_warning") as warn:
        _warn_on_unresumable_sensenova_save_format(train_config, save_format)
    if expected_code is None:
        warn.assert_not_called()
    else:
        assert warn.call_args.kwargs["code"] == expected_code


def test_a_run_with_nothing_to_train_is_not_warned_about_a_branch_it_lacks():
    """``train_unet=False, train_text_encoder=False`` has no branch at all; the
    trainer refuses it. Re-deriving one here named 'und' for such a run."""
    from core.training.train_runner import _warn_on_unresumable_sensenova_save_format

    with patch("core.training.training_events.emit_training_warning") as warn:
        _warn_on_unresumable_sensenova_save_format(
            {"train_unet": False, "train_text_encoder": False}, "bf16")
    warn.assert_not_called()
    # ... while an und-only run, which IS a branch, still gets the advice.
    # 'bf16' on a single-half branch resumes via the base-model fallback, so
    # this is the "needs base" advisory rather than a flat refusal.
    with patch("core.training.training_events.emit_training_warning") as warn:
        _warn_on_unresumable_sensenova_save_format(
            {"train_unet": False, "train_text_encoder": True}, "bf16")
    assert warn.call_args.kwargs["code"] == "sensenova_save_format_resume_needs_base"


# ---------------------------------------------------------------------------
# The two user-facing surfaces, kept honest and kept in step with each other
# ---------------------------------------------------------------------------

REPO_ROOT = BACKEND_ROOT.parent
_OPENAPI = (REPO_ROOT / "openapi.yaml").read_text(encoding="utf-8")
_CAPABILITIES = (
    BACKEND_ROOT / "api" / "arch_capabilities.py"
).read_text(encoding="utf-8")


def test_the_save_format_description_separates_the_two_gates():
    """It is the surface a user reads when CHOOSING the format, so it must not
    steer a restartable run at `int8` on the strength of the base-substitution
    gate alone."""
    assert "accept_resume_shaped_base" in _OPENAPI
    assert "`int8` to distribute" in _OPENAPI
    # The old sentence claimed int8 was the only re-selectable format full stop.
    assert "It is also the only one of the three that can be" not in _OPENAPI
    # ... and the resume claim is scoped to what was actually run.
    assert "resume has not been" in _OPENAPI
    assert "an inference from the same" in _OPENAPI
    # The old sentence claimed 'mixed' was the ONLY resumable format for a
    # single-half run; the bf16 base-model fallback makes that false.
    assert "resumes losslessly only from `mixed`" not in _OPENAPI
    assert "A single-half run can also resume from `bf16`" in _OPENAPI
    assert "verified tensor-for-tensor" in _OPENAPI


@pytest.mark.parametrize(
    "sentence",
    [
        # S2: und@64px IS measured (U-2-5, 26.2571 GiB); only above it is not.
        "understanding half alone above 64px",
        # S5: the int8 byte count came off a gen-branch save.
        "measured on a GENERATION-branch save",
    ],
)
def test_the_advisory_and_its_openapi_mirror_say_the_same_corrected_thing(sentence):
    assert sentence in _CAPABILITIES
    assert sentence in _OPENAPI


def test_the_two_overstatements_are_gone_from_both_surfaces():
    for text in (_CAPABILITIES, _OPENAPI):
        assert "at any resolution are unmeasured" not in text
        assert "checkpoint is 32.68 GiB in bf16 and 17.59 GiB in int8" not in text
