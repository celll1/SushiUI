"""Phase U-2-2 step 3 precondition: the SenseNova full-fine-tune checkpoint format.

Round trips every (branch x format) pair through the PRODUCTION read sequence
(``loader.install_sensenova_state_dict`` -- the guard/census/swap/verify/assign
block ``load_sensenova_from_path`` itself runs), and records the two traps the
read path cannot catch:

* a "mixed" file that kept the dequantized half's stale ``weight_scale`` keys is
  refused on read by a message describing the INVERSE defect;
* a HALF-materialized file loads clean, with no warning, as a valid wrong model.

The tree is synthetic (the shapes and naming of
``sensenova_int8_materialize_test``); no test here needs the 17.6 GiB
checkpoint.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.common.single_file_format import (
    TRANSFORMER_PREFIX, read_state_dict, strip_prefix,
)
from core.models.ideogram4.vendor.int8_linear import Int8Linear, quantize_weight_to_int8
from core.models.sensenova.loader import (
    SENSENOVA_BRANCH_LINEAR_COUNTS,
    install_sensenova_state_dict,
    materialize_int8_decoder_linears,
    save_sensenova_full_finetune_checkpoint,
)
from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
from core.training.adapters.sensenova_adapter import SenseNovaFullParameterAdapter

_LAYERS = 42
_IN, _OUT = 8, 4
_BRANCHES = ("gen", "und", "both")
_FORMATS = ("mixed", "bf16", "int8")


# ---------------------------------------------------------------------------
# Synthetic tree
# ---------------------------------------------------------------------------

def _int8_linear_from(weight: torch.Tensor) -> Int8Linear:
    codes, scale = quantize_weight_to_int8(weight)
    module = Int8Linear(weight.shape[1], weight.shape[0], False, torch.bfloat16)
    module.weight.copy_(codes)
    module.weight_scale.copy_(scale)
    return module


def _quant(seed: int) -> Int8Linear:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(_OUT, _IN, generator=generator, dtype=torch.float32)
    return _int8_linear_from(weight)


def _crested(seed: int) -> Int8Linear:
    """A high-crest row: one large element sets ``amax``, the rest sit near 2e-2.

    That separates the int8 grid step (4.0/127) from the bf16 ULP of the
    elements an update is added to (6.1e-5 at 2e-2), which is the window
    ``test_int8_requantization_discards_updates_below_half_a_grid_step`` needs.
    """
    generator = torch.Generator().manual_seed(seed)
    weight = torch.rand(_OUT, _IN, generator=generator, dtype=torch.float32) * 4e-3 + 0.018
    weight[:, 0] = 4.0
    return _int8_linear_from(weight.to(torch.bfloat16).float())


def _plain(seed: int) -> nn.Linear:
    """What ``NEOChatModel(config)`` gives the loader before any swap."""
    return nn.Linear(_IN, _OUT, bias=False)


class _Decoder(nn.Module):
    """The 42-layer MoT attribute layout ``iter_sensenova_lora_targets`` walks."""

    use_pixel_head = True
    use_deep_fm_head = False

    def __init__(self, factory=_quant, layers=_LAYERS):
        super().__init__()
        seed = 0
        blocks = []
        for _ in range(layers):
            block = nn.Module()
            attn = nn.Module()
            mlp, mlp_gen = nn.Module(), nn.Module()
            for stem in ("q_proj", "k_proj", "v_proj", "o_proj"):
                for name in (stem, f"{stem}_mot_gen"):
                    setattr(attn, name, factory(seed))
                    seed += 1
            for stem in ("gate_proj", "up_proj", "down_proj"):
                for parent in (mlp, mlp_gen):
                    setattr(parent, stem, factory(seed))
                    seed += 1
            block.self_attn = attn
            block.mlp = mlp
            block.mlp_mot_gen = mlp_gen
            blocks.append(block)
        core = nn.Module()
        core.layers = nn.ModuleList(blocks)
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model
        # One non-decoder tensor, so the writer's "everything else" path is
        # exercised rather than assumed.
        self.embed_tokens = nn.Embedding(4, _IN)


def _paths(transformer: nn.Module, branch: str) -> "dict[str, nn.Module]":
    return {
        path: module
        for path, _p, _a, module in iter_sensenova_lora_targets(transformer, branch=branch)
    }


def _halves(branch: str) -> "tuple[str, ...]":
    return ("gen", "und") if branch == "both" else (branch,)


def _trained_tree(branch: str, *, perturb: float = 0.05) -> nn.Module:
    """An int8 base with ``branch`` materialized to bf16 and then MOVED."""
    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch=branch)
    generator = torch.Generator().manual_seed(1234)
    if perturb:
        with torch.no_grad():
            for half in _halves(branch):
                for module in _paths(transformer, half).values():
                    module.weight.add_(
                        torch.randn(module.weight.shape, generator=generator,
                                    dtype=torch.float32).to(torch.bfloat16) * perturb
                    )
    return transformer


def _load_back(path: str) -> "tuple[nn.Module, dict]":
    """The production read sequence, against a freshly built plain tree."""
    raw, metadata = read_state_dict(path)
    sd = strip_prefix(raw, TRANSFORMER_PREFIX)
    model = _Decoder(factory=_plain)
    install_sensenova_state_dict(model, sd, {}, torch.bfloat16, path=path)
    return model, metadata


def _save(transformer, tmp_path, branch, save_format, **kwargs):
    return save_sensenova_full_finetune_checkpoint(
        transformer, str(tmp_path / "run_step_000100"),
        branch=branch, save_format=save_format,
        config={"downsample_ratio": 0.5}, **kwargs,
    )


# ---------------------------------------------------------------------------
# Round trip: every branch x every format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", _BRANCHES)
@pytest.mark.parametrize("save_format", _FORMATS)
def test_round_trip_through_the_production_read_path(tmp_path, branch, save_format):
    transformer = _trained_tree(branch)
    trained_before = {
        path: module.weight.detach().clone()
        for half in _halves(branch)
        for path, module in _paths(transformer, half).items()
    }
    frozen_halves = tuple(h for h in ("gen", "und") if h not in _halves(branch))
    frozen_before = {
        path: (module.weight.detach().clone(), module.weight_scale.detach().clone())
        for half in frozen_halves
        for path, module in _paths(transformer, half).items()
    }

    written, census = _save(transformer, tmp_path, branch, save_format)
    effective = "bf16" if (save_format == "mixed" and branch == "both") else save_format
    assert census["effective_format"] == effective

    model, metadata = _load_back(written)
    assert metadata["model_type"] == "sensenova"
    assert metadata["sensenova_trained_branch"] == branch
    assert metadata["sensenova_save_format"] == effective
    assert metadata["sensenova_save_format_requested"] == save_format
    assert json.loads(metadata["sensenova_config"])["downsample_ratio"] == 0.5

    loaded_trained = {p: m for half in _halves(branch) for p, m in _paths(model, half).items()}
    loaded_frozen = {p: m for half in frozen_halves for p, m in _paths(model, half).items()}
    assert len(loaded_trained) == sum(SENSENOVA_BRANCH_LINEAR_COUNTS[h] for h in _halves(branch))
    assert len(loaded_frozen) == sum(SENSENOVA_BRANCH_LINEAR_COUNTS[h] for h in frozen_halves)

    # Structure: module class and parameter/buffer status, per half.
    for path, module in loaded_trained.items():
        if effective == "int8":
            assert type(module) is Int8Linear, path
            assert not isinstance(module.weight, nn.Parameter)
            assert module.weight.dtype is torch.int8
            assert "weight" in dict(module.named_buffers())
            assert list(module.parameters()) == []
        else:
            assert type(module) is nn.Linear, path
            assert isinstance(module.weight, nn.Parameter)
            assert module.weight.dtype is torch.bfloat16
            assert not hasattr(module, "weight_scale")
    for path, module in loaded_frozen.items():
        if effective == "bf16":
            assert type(module) is nn.Linear, path
            assert isinstance(module.weight, nn.Parameter)
            assert module.weight.dtype is torch.bfloat16
        else:
            assert type(module) is Int8Linear, path
            assert module.weight.dtype is torch.int8
            assert list(module.parameters()) == []

    # Numerics.
    for path, module in loaded_trained.items():
        before = trained_before[path]
        if effective == "int8":
            codes, scale = quantize_weight_to_int8(before)
            assert torch.equal(module.weight, codes), path
            assert torch.equal(module.weight_scale, scale), path
        else:
            assert torch.equal(module.weight.detach(), before), path
    for path, module in loaded_frozen.items():
        codes, scale = frozen_before[path]
        if effective == "bf16":
            expected = codes * scale.to(torch.bfloat16).unsqueeze(1)
            assert torch.equal(module.weight.detach(), expected), path
        else:
            assert torch.equal(module.weight, codes), path
            assert torch.equal(module.weight_scale, scale), path

    # The non-decoder tensor rides along untouched in every format.
    assert torch.equal(model.embed_tokens.weight.detach(),
                       transformer.embed_tokens.weight.detach())


def test_mixed_degenerates_to_bf16_when_both_halves_are_trained(tmp_path):
    """No int8 half is left to keep, so the file IS the bf16 one -- and says so."""
    transformer = _trained_tree("both")
    mixed, mixed_census = _save(transformer, tmp_path, "both", "mixed")
    bf16, bf16_census = _save(transformer, tmp_path / "b", "both", "bf16")
    assert mixed_census["effective_format"] == bf16_census["effective_format"] == "bf16"

    mixed_sd = strip_prefix(read_state_dict(mixed)[0], TRANSFORMER_PREFIX)
    bf16_sd = strip_prefix(read_state_dict(bf16)[0], TRANSFORMER_PREFIX)
    assert set(mixed_sd) == set(bf16_sd)
    assert all(torch.equal(mixed_sd[k], bf16_sd[k]) for k in mixed_sd)
    assert not any(k.endswith(".weight_scale") for k in mixed_sd)
    # The requested format is still recorded, so the degeneracy is inspectable.
    assert read_state_dict(mixed)[1]["sensenova_save_format_requested"] == "mixed"


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL 1 -- the stale weight_scale trap (design 6.4 / loader gate)
# ---------------------------------------------------------------------------

def test_stale_weight_scale_is_refused_on_read_by_the_wrong_message(tmp_path):
    """A "mixed" file that KEPT the dequantized half's scales cannot be read.

    ``swap_linears_to_int8`` needs an int8 weight AND a scale sibling, so the
    bf16 half is not swapped; ``verify_quantized_swap`` then sees 588 scales
    against 294 int8 weights and reports a PARTIALLY SCALE-LESS file -- the
    inverse of what actually happened. Recorded verbatim because that message
    is unfindable from the symptom.
    """
    transformer = _trained_tree("gen")
    written, _ = _save(transformer, tmp_path, "gen", "mixed")
    raw, _metadata = read_state_dict(written)
    sd = strip_prefix(raw, TRANSFORMER_PREFIX)

    # What the writer must never do: keep the now-meaningless scale.
    base = _Decoder()
    stale = dict(sd)
    for path, module in _paths(base, "gen").items():
        stale[f"{path}.weight_scale"] = module.weight_scale.clone()
    assert len(stale) == len(sd) + 294

    with pytest.raises(RuntimeError) as excinfo:
        install_sensenova_state_dict(_Decoder(factory=_plain), stale, {}, torch.bfloat16,
                                     path=written)
    message = str(excinfo.value)
    assert "scales=588, quantized weights=294, swapped=294" in message
    assert (
        "the file carries 294 int8/uint8 '.weight' tensor(s) but 588 "
        "'.weight_scale' sibling(s) -- every quantized weight needs its per-row "
        "scale, so a scale-less (or partially scale-less) file cannot be read back"
    ) in message

    # And the shipped writer emits no such key, so the trap is unreachable by
    # construction rather than by remembering to drop something.
    assert sum(1 for k in sd if k.endswith(".weight_scale")) == 294
    assert all(sd[k[: -len(".weight_scale")] + ".weight"].dtype is torch.int8
               for k in sd if k.endswith(".weight_scale"))


def test_writer_refuses_a_mis_shaped_scale_on_the_frozen_half(tmp_path):
    """The squeeze trap, at the WRITE end of the pipe.

    ``materialize_int8_decoder_linears`` refuses an ``[out, 1]`` scale rather
    than reshaping it; dequantizing the frozen half for a bf16 save is the same
    broadcast and needs the same refusal, or it writes a silently wrong weight.
    """
    transformer = _trained_tree("gen")
    victim = next(iter(_paths(transformer, "und").values()))
    victim.weight_scale = victim.weight_scale.reshape(_OUT, 1)
    with pytest.raises(RuntimeError, match=r"weight_scale of shape \(4, 1\)"):
        _save(transformer, tmp_path, "gen", "bf16")
    assert not list(tmp_path.iterdir())
    # It only matters where the scale is actually consumed: mixed and int8 pass
    # the frozen half through byte for byte and never broadcast it.
    _save(transformer, tmp_path / "mixed", "gen", "mixed")


def test_writer_refuses_a_scale_beside_a_materialized_linear(tmp_path):
    """The same rule, asserted at the WRITE end where the cause is in hand."""
    transformer = _trained_tree("gen")
    victim = next(iter(_paths(transformer, "gen").values()))
    victim.register_buffer("weight_scale", torch.ones(_OUT))
    with pytest.raises(RuntimeError, match="a dequantized weight has no scale"):
        _save(transformer, tmp_path, "gen", "mixed")
    assert not list(tmp_path.glob("*.safetensors*"))


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL 2 -- a partial half loads clean; the writer refuses it
# ---------------------------------------------------------------------------

def _truncated_state_dict(tmp_path, kept: int):
    """A "mixed" gen file in which only ``kept`` of the 294 gen Linears are bf16."""
    transformer = _trained_tree("gen")
    written, _ = _save(transformer, tmp_path, "gen", "mixed")
    sd = strip_prefix(read_state_dict(written)[0], TRANSFORMER_PREFIX)
    base = _Decoder()
    reverted = sorted(_paths(base, "gen"))[kept:]
    for path in reverted:
        module = _paths(base, "gen")[path]
        sd[f"{path}.weight"] = module.weight.clone()
        sd[f"{path}.weight_scale"] = module.weight_scale.clone()
    return sd, reverted


def test_a_half_materialized_file_loads_clean_with_no_warning(tmp_path):
    """The read path has no branch awareness and no count assertion (design 6.4)."""
    sd, reverted = _truncated_state_dict(tmp_path, kept=150)
    model = _Decoder(factory=_plain)
    swapped = install_sensenova_state_dict(model, sd, {}, torch.bfloat16)

    # 294 understanding + 144 reverted generation Linears, and it is happy.
    assert swapped == 294 + len(reverted) == 438
    gen = _paths(model, "gen")
    assert sum(1 for m in gen.values() if type(m) is nn.Linear) == 150
    assert sum(1 for m in gen.values() if type(m) is Int8Linear) == 144
    # Nothing distinguishes this from a deliberate save; that is the defect.


def test_the_writer_refuses_the_tree_that_would_produce_it(tmp_path):
    transformer = _trained_tree("gen")
    base = _Decoder()
    targets = _paths(transformer, "gen")
    for path in sorted(targets)[150:]:
        parent_path, _, attr = path.rpartition(".")
        parent = transformer.get_submodule(parent_path)
        setattr(parent, attr, _paths(base, "gen")[path])

    with pytest.raises(RuntimeError) as excinfo:
        _save(transformer, tmp_path, "gen", "mixed")
    message = str(excinfo.value)
    assert "144 of 294 gen-branch decoder Linear(s) still holding an int8 buffer" in message
    assert "loads clean" in message
    assert not list(tmp_path.iterdir())


def test_an_off_count_tree_is_refused_before_anything_is_written(tmp_path):
    transformer = _trained_tree("gen", perturb=0.0)
    layers = transformer.language_model.model.layers
    transformer.language_model.model.layers = nn.ModuleList(list(layers)[:41])
    with pytest.raises(RuntimeError, match="enumerated 287 trained and 287 frozen"):
        _save(transformer, tmp_path, "gen", "mixed")
    assert not list(tmp_path.iterdir())


def test_nothing_readable_survives_a_write_that_fails_midway(tmp_path, monkeypatch):
    """Atomic commit: shards go down under provisional names, the index last."""
    from core.models.common import quantized_export

    transformer = _trained_tree("gen")
    calls = {"n": 0}
    real_add = quantized_export.ShardWriter.add

    def exploding_add(self, key, tensor):
        calls["n"] += 1
        if calls["n"] == 400:
            raise OSError("disk full")
        return real_add(self, key, tensor)

    monkeypatch.setattr(quantized_export.ShardWriter, "add", exploding_add)
    with pytest.raises(OSError, match="disk full"):
        _save(transformer, tmp_path, "gen", "mixed", max_shard_bytes=4096)
    # Provisional shards deleted, no index, nothing read_state_dict can open.
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Memory discipline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_shard_bytes", [8192, 64])
def test_the_writer_never_holds_more_than_one_shard(tmp_path, max_shard_bytes):
    """Host cost is one shard buffer OR one oversized tensor, never the model.

    Both sides of that bound are exercised: 8192 is above every tensor here, 64
    is below the largest, which is the case that matters on the real checkpoint
    (a 4 GiB threshold against a 1187 MiB ``lm_head``).
    """
    from core.models.common import quantized_export

    transformer = _trained_tree("gen")
    peak = {"bytes": 0, "tensors": 0}
    real_add = quantized_export.ShardWriter.add

    def watched_add(self, key, tensor):
        real_add(self, key, tensor)
        peak["bytes"] = max(peak["bytes"], self.buffer_bytes)
        peak["tensors"] = max(peak["tensors"], len(self.buffer))

    largest = max(t.numel() * t.element_size() for t in transformer.state_dict().values())
    quantized_export.ShardWriter.add = watched_add
    try:
        written, _ = _save(transformer, tmp_path, "gen", "mixed",
                           max_shard_bytes=max_shard_bytes)
    finally:
        quantized_export.ShardWriter.add = real_add

    # ShardWriter flushes BEFORE inserting a tensor that would overflow.
    assert peak["bytes"] <= max(max_shard_bytes, largest)
    assert peak["tensors"] >= 1
    if max_shard_bytes < largest:
        # The oversized-tensor branch: that tensor alone IS the shard.
        assert peak["bytes"] == largest
    assert written.endswith(".safetensors.index.json")
    assert len(json.loads(Path(written).read_text())["weight_map"]) > 0
    _load_back(written)  # and it is still a readable checkpoint


# ---------------------------------------------------------------------------
# Format 3: the requantization loss, against a derived bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", _BRANCHES)
def test_int8_round_trip_error_is_bounded_by_half_a_grid_step(tmp_path, branch):
    """|w - q*s| <= s/2 by construction (round-half-to-even onto the grid),
    with s = row amax / 127. Asserted from both sides so the bound is not vacuous.
    """
    transformer = _trained_tree(branch)
    before = {
        path: module.weight.detach().clone().float()
        for half in _halves(branch)
        for path, module in _paths(transformer, half).items()
    }
    written, _ = _save(transformer, tmp_path, branch, "int8")
    model, _metadata = _load_back(written)

    worst_ratio = 0.0
    for half in _halves(branch):
        for path, module in _paths(model, half).items():
            weight = before[path]
            step = weight.abs().amax(dim=1) / 127.0
            recovered = module.weight.float() * module.weight_scale.float().unsqueeze(1)
            error = (weight - recovered).abs().amax(dim=1)
            assert torch.all(error <= step / 2 + torch.finfo(torch.float32).eps), path
            worst_ratio = max(worst_ratio, float((error / (step / 2)).max()))
    # Tight, not vacuous: some row actually spends most of its half-step.
    assert worst_ratio > 0.5


def test_int8_requantization_discards_updates_below_half_a_grid_step(tmp_path):
    """The loss is a THRESHOLD in absolute update size, not a percentage.

    The update has to land in a REAL window to test anything: above the bf16 ULP
    of the elements it is added to, so the trained weight actually moves, and
    below half a grid step, so requantization throws that movement away. An
    update below the bf16 ULP is discarded by the parameter's own storage before
    the writer ever sees it, and an assertion built on one passes against a
    writer that stores fp32 exactly.

    ``_CRESTED`` gives that window room: one large element per row sets
    ``amax``, so the grid step is coarse (4.0/127 = 3.15e-2, half-step 1.57e-2)
    while the elements carrying the update sit near 2e-2, where the bf16 ULP is
    6.1e-5.
    """
    half_step = 4.0 / 127.0 / 2.0
    tiny, large = 0.35 * 2 * half_step, 3.0 * 2 * half_step
    # The window, stated rather than assumed.
    assert 6.1e-5 < tiny < half_step < large

    codes, moved = {}, {}
    for label, magnitude in (("untrained", 0.0), ("tiny", tiny), ("large", large)):
        transformer = _Decoder(factory=_crested)
        materialize_int8_decoder_linears(transformer, branch="gen")
        weights = _paths(transformer, "gen")
        before = {p: m.weight.detach().clone() for p, m in weights.items()}
        if magnitude:
            with torch.no_grad():
                for module in weights.values():
                    module.weight.add_(
                        torch.full(module.weight.shape, magnitude, dtype=torch.bfloat16)
                    )
        # The arm cannot silently degenerate into "the update was a bf16 no-op".
        moved[label] = all(
            not torch.equal(m.weight.detach(), before[p]) for p, m in weights.items()
        )
        written, _ = _save(transformer, tmp_path / label, "gen", "int8")
        model, _m = _load_back(written)
        codes[label] = {p: m.weight.clone() for p, m in _paths(model, "gen").items()}

    assert moved["tiny"] and moved["large"] and not moved["untrained"]
    # Moved in bf16, gone after requantization.
    assert all(torch.equal(codes["tiny"][p], codes["untrained"][p]) for p in codes["tiny"])
    assert all(not torch.equal(codes["large"][p], codes["untrained"][p])
               for p in codes["large"])


# ---------------------------------------------------------------------------
# Adapter: format resolution and refusals
# ---------------------------------------------------------------------------

def _adapter(transformer, branch, save_format):
    trainer = SimpleNamespace(
        transformer=transformer,
        train_unet=branch in ("gen", "both"),
        train_text_encoder=branch in ("und", "both"),
        config={},
        sensenova_full_finetune_save_format=save_format,
        sensenova_model_config=None,
        model_path=None,
        log_prefix="[SenseNova]",
    )
    return SenseNovaFullParameterAdapter(trainer)


@pytest.mark.parametrize("save_format", _FORMATS)
def test_adapter_saves_each_format(tmp_path, save_format):
    transformer = _trained_tree("gen")
    adapter = _adapter(transformer, "gen", save_format)
    adapter.save_checkpoint(100, 1, tmp_path / "run_step_000100")
    written = list(tmp_path.glob("run_step_000100*"))
    assert written, save_format


def test_adapter_refuses_an_unknown_format(tmp_path):
    adapter = _adapter(_trained_tree("gen"), "gen", "fp8")
    with pytest.raises(ValueError, match="Unknown sensenova_full_finetune_save_format"):
        adapter.save_checkpoint(100, 1, tmp_path / "run")
    assert not list(tmp_path.iterdir())


def test_adapter_falls_back_to_the_config_channel_then_the_default(tmp_path):
    adapter = _adapter(_trained_tree("gen"), "gen", None)
    adapter.trainer.config = {"sensenova_full_finetune_save_format": "int8"}
    assert adapter._resolve_save_format() == "int8"
    adapter.trainer.config = {}
    assert adapter._resolve_save_format() == "mixed"


def test_adapter_announces_the_both_branch_degeneracy(tmp_path):
    from core.training import training_events

    seen = []
    adapter = _adapter(_trained_tree("both"), "both", "mixed")
    real = training_events.emit_training_warning
    try:
        training_events.emit_training_warning = lambda message, **kw: (
            seen.append((message, kw.get("code"))) or real(message, **kw)
        )
        adapter.save_checkpoint(100, 1, tmp_path / "run_step_000100")
    finally:
        training_events.emit_training_warning = real
    assert any(code == "sensenova_save_format_degenerate" for _msg, code in seen)


def test_adapter_stamps_the_configured_base_model_identity(tmp_path):
    """The resume-fallback's other half: what ``restore_sensenova_frozen_half_from_base``
    reads back (``core.training.ops.sensenova_ops._sensenova_resume_base_model_path``)."""
    from core.models.sensenova.loader import sensenova_base_model_identity

    base_path = tmp_path / "base" / "sensenova_int8.safetensors"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_bytes(b"\x00" * 1024)

    transformer = _trained_tree("gen")
    adapter = _adapter(transformer, "gen", "bf16")
    adapter.trainer.configured_model_path = str(base_path)
    adapter.save_checkpoint(100, 1, tmp_path / "run_step_000100")
    _model, metadata = _load_back(str(tmp_path / "run_step_000100.safetensors"))
    assert metadata["sensenova_base_model_path"] == str(base_path)
    assert metadata["sensenova_base_model_identity"] == sensenova_base_model_identity(base_path)


# ---------------------------------------------------------------------------
# The setting's plumbing, end to end
# ---------------------------------------------------------------------------

def test_save_format_api_yaml_openapi_and_frontend_parity():
    import asyncio

    import yaml

    from api.param_defaults import (
        SENSENOVA_FULL_FINETUNE_SAVE_FORMATS, TRAINING_DEFAULTS,
    )
    from api.routes import TrainingRunCreateRequest, get_training_defaults
    from core.training.training_config import TrainingConfigGenerator

    assert TRAINING_DEFAULTS["sensenova_full_finetune_save_format"] == "mixed"
    assert asyncio.run(get_training_defaults())["sensenova_full_finetune_save_format"] == "mixed"
    request = TrainingRunCreateRequest(
        training_method="full_finetune", base_model_path="models/sensenova"
    )
    assert request.sensenova_full_finetune_save_format == "mixed"

    for generator, kwargs in (
        (TrainingConfigGenerator.generate_full_finetune_config, {}),
        (TrainingConfigGenerator.generate_lora_config, {}),
    ):
        config = yaml.safe_load(generator(
            {**request.model_dump(), "total_steps": 1, "epochs": None,
             "sensenova_full_finetune_save_format": "int8"},
            run_name="save-format", base_model_path=request.base_model_path,
            output_dir="output", **kwargs,
        ))
        train = config["config"]["process"][0]["train"]
        assert train["sensenova_full_finetune_save_format"] == "int8"

    root = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((root / "openapi.yaml").read_text(encoding="utf-8"))
    prop = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"][
        "sensenova_full_finetune_save_format"
    ]
    assert prop["default"] == "mixed"
    assert tuple(prop["enum"]) == SENSENOVA_FULL_FINETUNE_SAVE_FORMATS
    # The enum is ENFORCED, not documentation: an unknown format has to fail at
    # run creation, because the adapter's own refusal fires at the first
    # save_every -- hours into a run whose only artefact is that save.
    import typing

    annotation = TrainingRunCreateRequest.model_fields[
        "sensenova_full_finetune_save_format"
    ].annotation
    assert typing.get_args(annotation) == SENSENOVA_FULL_FINETUNE_SAVE_FORMATS
    with pytest.raises(Exception) as invalid:
        TrainingRunCreateRequest(
            training_method="full_finetune", base_model_path="models/sensenova",
            sensenova_full_finetune_save_format="fp8",
        )
    assert "sensenova_full_finetune_save_format" in str(invalid.value)
    api_source = (root / "frontend/src/utils/api.ts").read_text(encoding="utf-8")
    form_source = (
        root / "frontend/src/components/training/TrainingConfig.tsx"
    ).read_text(encoding="utf-8")
    assert "sensenova_full_finetune_save_format?: string" in api_source
    assert 'updateParam("sensenova_full_finetune_save_format"' in form_source


def test_trainer_reads_the_setting_off_the_train_config():
    from core.training.base_trainer import BaseTrainer

    source = Path(BaseTrainer.__module__ and
                  sys.modules["core.training.base_trainer"].__file__).read_text(encoding="utf-8")
    assert 'self.sensenova_full_finetune_save_format = str(_tc.get(' in source


# ---------------------------------------------------------------------------
# Gate G: the two step-3 refusals are OPEN (U-2-2 step 3). The precondition this
# file's own subject was -- a writable checkpoint format -- is what unlocked
# them; sensenova_full_finetune_acceptance_test.py owns the acceptance path.
# ---------------------------------------------------------------------------

def test_full_finetune_is_accepted_now_that_a_format_exists():
    from api.arch_capabilities import TRAINING_UNSUPPORTED

    assert "full_finetune" not in TRAINING_UNSUPPORTED["sensenova"]
    assert "relora" in TRAINING_UNSUPPORTED["sensenova"]
