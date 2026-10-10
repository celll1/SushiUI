"""Hybrid graph, independent zero-init oracle and learnable image-grid checks."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch

from core.rewiring.spec import OPERATORS, build_spec, digest, checkpoint_metadata, read_checkpoint_metadata, rebase_spec
from core.rewiring.runtime import run_rewired_stream
from core.rewiring.hybrid_spec import (build_hybrid_spec, validate_hybrid_spec, legacy_topology,
                                      hybrid_metadata, read_hybrid_metadata, hybrid_preview, module_layout)
from core.rewiring.hybrid_spec import hybrid_parameter_counts
from core.rewiring.hybrid_modules import HybridModules


def declaration():
    base = build_spec("anima", [{"name": "main", "block_count": 7}], [
        dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
             down_factor=[2, 2], **OPERATORS),
        dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5,
             down_factor=[2, 2], **OPERATORS)], {})
    return build_hybrid_spec(base, stream_widths={"main": 8}, boundaries=[
        dict(id="outer_boundary", region="outer", rank=3, kernel=3),
        dict(id="inner_boundary", region="inner", rank=3, kernel=3)], refiners=[
        dict(id="entry", stream="main", after_block=0, rank=3, kernel=3),
        dict(id="exit", stream="main", after_block=6, rank=3, kernel=3)], skips=[
        dict(id="outer_skip", stream="main", source_after=0, destination_before=6, rank=3),
        dict(id="middle_skip", stream="main", source_after=1, destination_before=5, rank=3)])


def test_hybrid_block_swap_rejected_before_partial_optimizer_updates():
    from core.rewiring.training import validate_training_features
    config = dict(training_method="full_finetune", dit_rewiring_spec=declaration(), blocks_to_swap=2)
    with pytest.raises(ValueError, match="Anima only"):
        validate_training_features(config, "ming_image_01")
    validate_training_features(config, "anima")
    config["blocks_to_swap"] = 0
    validate_training_features(config, "anima")


def test_metadata_roundtrip_and_tampering():
    spec = declaration()
    assert read_hybrid_metadata(hybrid_metadata(spec)) == spec
    assert read_checkpoint_metadata(checkpoint_metadata(spec)) == spec
    rebound = rebase_spec(spec, {"changed_input_channels": 64})
    assert rebound["skips"] == spec["skips"]
    assert rebound["module_layout_sha256"] == spec["module_layout_sha256"]
    damaged = hybrid_metadata(spec)
    damaged["sushi.hybrid.spec_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        read_hybrid_metadata(damaged)
    damaged = deepcopy(spec)
    damaged["boundaries"][0]["rank"] = 4
    with pytest.raises(ValueError):
        validate_hybrid_spec(damaged)


def test_opt_in_preset_preserves_base_and_equal_grid_skips():
    from core.rewiring.hybrid_spec import default_hybrid_spec
    base = legacy_topology(declaration())
    saved = deepcopy(base)
    preset = default_hybrid_spec(base, {"main": 8})
    assert base == saved
    assert preset["base_config_sha256"] == base["base_config_sha256"]
    assert {item["region"] for item in preset["boundaries"]} == {"outer", "inner"}
    assert preset["refiners"] and preset["skips"]
    assert all(item["rank"] <= 8 for item in preset["skips"])
    assert validate_hybrid_spec(preset) == preset


def test_builder_rehashes_edits_and_rejects_cross_grid_skip():
    from api.rewiring_routes import HybridBuildRequest, build_hybrid
    from fastapi import HTTPException
    spec = declaration()
    request = dict(base_spec=spec, stream_widths=spec["stream_widths"],
                   boundaries=deepcopy(spec["boundaries"]), refiners=spec["refiners"], skips=deepcopy(spec["skips"]))
    request["boundaries"][0]["rank"] = 4
    rebuilt = build_hybrid(HybridBuildRequest(**request))["spec"]
    assert rebuilt["module_layout_sha256"] != spec["module_layout_sha256"]
    assert validate_hybrid_spec(rebuilt) == rebuilt
    request["skips"][0]["destination_before"] = 4
    with pytest.raises(HTTPException) as error:
        build_hybrid(HybridBuildRequest(**request))
    assert error.value.status_code == 422


def test_measured_candidate_matches_recorded_graph_without_mutating_source():
    import json
    from collections import Counter
    from core.rewiring.hybrid_spec import measured_hybrid_candidates
    record = Path(__file__).resolve().parents[2] / "docs/audits/data/hybrid_anima_public_feature_fusion_matrix_2026-10.json"
    expected = json.loads(record.read_text())["arms"]["hybrid"]["graph"]
    base = legacy_topology(expected)
    base["regions"][0].update(start_block=4, end_block=24)
    before = deepcopy(base)
    candidates = measured_hybrid_candidates(base, expected["stream_widths"], native_patch_spatial=2)
    assert base == before and len(candidates) == 1
    spec = candidates[0]["spec"]
    assert spec == expected
    assert read_checkpoint_metadata(checkpoint_metadata(spec)) == expected
    assert Counter(block["depth"] for block in hybrid_preview(spec, 64, 64,
        batch_size=1, element_bytes=2)["blocks"]) == {0: 16, 1: 8, 2: 4}
    modules = HybridModules(spec)
    assert len(modules.state_dict()) == 67
    assert sum(parameter.numel() for parameter in modules.parameters()) == 35629056
    spec["skips"][0]["rank"] = 1
    assert measured_hybrid_candidates(base, expected["stream_widths"], native_patch_spatial=2)[0]["spec"] == expected
    for patch in (None, 4, 2.0, True):
        assert measured_hybrid_candidates(base, expected["stream_widths"], native_patch_spatial=patch) == []


@pytest.mark.parametrize("architecture,count,width", [
    ("anima", 28, 1024), ("anima", 32, 2048), ("qwen_image_21", 28, 2048),
    ("ming_image_01", 28, 2048),
])
def test_measured_candidate_does_not_claim_unmeasured_native_layouts(architecture, count, width):
    from core.rewiring.hybrid_spec import measured_hybrid_candidates
    base = build_spec(architecture, [{"name": "main", "block_count": count}], [
        dict(id="r0", stream="main", parent=None, start_block=1, end_block=count-1,
             down_factor=[2, 2], **OPERATORS)], {})
    assert measured_hybrid_candidates(base, {"main": width}, native_patch_spatial=2) == []


def test_parameter_census_matches_registered_modules():
    spec = declaration()
    modules = HybridModules(spec)
    census = hybrid_parameter_counts(spec)
    assert census["added"] == sum(parameter.numel() for parameter in modules.parameters())
    for item in census["modules"]:
        if item["kind"] == "boundary":
            expected = sum(p.numel() for part in (modules.down[item["id"]], modules.up[item["id"]]) for p in part.parameters())
        else:
            group = modules.local if item["kind"] == "refiner" else modules.skips
            expected = sum(p.numel() for p in group[item["id"]].parameters())
        assert item["count"] == expected
    from core.rewiring.spec import preview_plan
    assert preview_plan(spec, 8, 12)["hybrid_parameter_counts"] == census


def test_new_graph_initialization_preserves_exposure_rng_and_is_reproducible():
    torch.manual_seed(572)
    before = torch.get_rng_state().clone()
    first = HybridModules(declaration())
    assert torch.equal(torch.get_rng_state(), before)
    second = HybridModules(declaration())
    assert torch.equal(torch.get_rng_state(), before)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_unsupported_training_features_are_explicit():
    from core.rewiring.training import validate_training_features
    config = dict(dit_rewiring_spec=declaration(), training_method="full_finetune")
    validate_training_features(config, "anima")
    validate_training_features(dict(config, dit_partition_training_enabled=True), "anima")
    validate_training_features(dict(config, training_method="lora"), "anima")
    for fields in (dict(training_method="relora"), dict(training_method="controlnet")):
        with pytest.raises(ValueError):
            validate_training_features(dict(config, **fields), "anima")


@pytest.mark.parametrize("change", [
    lambda spec: spec.update(schema_version=2.0),
    lambda spec: spec["stream_widths"].update(main=True),
    lambda spec: spec["skips"][0].update(destination_before=2),
    lambda spec: spec["skips"][0].update(source_after=6),
    lambda spec: spec["boundaries"][0].update(kernel=4),
    lambda spec: spec["refiners"][0].update(id=spec["boundaries"][0]["id"]),
])
def test_invalid_graphs_are_rejected(change):
    spec = declaration()
    change(spec)
    spec["module_layout_sha256"] = digest(module_layout(spec))
    with pytest.raises(ValueError):
        validate_hybrid_spec(spec)


def execute(index, hidden, context):
    return hidden + torch.tanh(hidden) * (index + 1) * 0.015


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_initialization_preserves_prefix_and_nested_baseline(dtype):
    torch.manual_seed(41)
    spec = declaration()
    modules = HybridModules(spec).to(dtype=dtype)
    value = torch.randn(2, 3 + 8*12, 8, dtype=dtype)
    actual = modules.forward_stream(value, "main", 8, 12, 3, None, execute, lambda *args: None)
    expected = run_rewired_stream(value, legacy_topology(spec), "main", 8, 12, 3, None,
                                  execute, lambda *args: None)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[:, :3], expected[:, :3])
    for module in modules.up.values():
        assert torch.count_nonzero(module(torch.zeros(2, 8, 8, 12, dtype=dtype))) == 0


def test_branches_learn_and_restore_full_state():
    torch.manual_seed(7)
    spec = declaration()
    modules = HybridModules(spec)
    optimizer = torch.optim.AdamW(modules.parameters(), lr=0.01)
    value = torch.randn(1, 8*12, 8, requires_grad=True)
    for _ in range(3):
        optimizer.zero_grad()
        output = modules.forward_stream(value, "main", 8, 12, 0, None, execute, lambda *args: None)
        output.square().mean().backward()
        optimizer.step()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for name, parameter in modules.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    restored = HybridModules(read_hybrid_metadata(hybrid_metadata(spec)))
    restored.load_state_dict(modules.state_dict(), strict=True)
    actual = restored.forward_stream(value, "main", 8, 12, 0, None, execute, lambda *args: None)
    assert torch.equal(actual, modules.forward_stream(value, "main", 8, 12, 0, None, execute, lambda *args: None))
    saved = deepcopy(optimizer.state_dict())
    other_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    other_optimizer.load_state_dict(saved)
    assert len(other_optimizer.state) == len(list(restored.parameters()))
    with pytest.raises(RuntimeError):
        restored.load_state_dict({}, strict=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_checkpointed_modules_preserve_outputs_and_all_gradients(dtype, cpu_offload):
    torch.manual_seed(31)
    plain = HybridModules(declaration()).to(dtype=dtype)
    with torch.no_grad():
        for parameter in plain.parameters():
            parameter.normal_(0, 0.1)
    checked = HybridModules(declaration()).to(dtype=dtype)
    checked.load_state_dict(plain.state_dict())
    original = torch.randn(2, 3 + 8*12, 8, dtype=dtype)
    inputs, outputs = [], []
    for modules, use_checkpoint in ((plain, False), (checked, True)):
        value = original.clone().requires_grad_(True)
        inputs.append(value)
        output = modules.forward_stream(value, "main", 8, 12, 3, None, execute, lambda *args: None,
                                       checkpoint_modules=use_checkpoint, checkpoint_offload=cpu_offload)
        outputs.append(output)
        output.float().square().mean().backward()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(inputs[0].grad, inputs[1].grad)
    for (name, reference), (_, candidate) in zip(plain.named_parameters(), checked.named_parameters()):
        assert reference.grad is not None and candidate.grad is not None, name
        assert torch.isfinite(candidate.grad).all(), name
        torch.testing.assert_close(reference.grad, candidate.grad, rtol=0, atol=0)


def test_compressed_skip_storage_estimate():
    preview = hybrid_preview(declaration(), 8, 12, batch_size=2, element_bytes=4)
    sizes = {item["id"]: item["stored_bytes"] for item in preview["learned_skip_edges"]}
    assert sizes == {"outer_skip": 2*3*8*12*4, "middle_skip": 2*3*4*6*4}
    with pytest.raises(ValueError):
        HybridModules(declaration()).forward_stream(torch.randn(1, 80, 8), "main", 8, 10, 0,
                                                      None, execute, lambda *args: None)


@pytest.mark.parametrize("batch,groups", [(1, 1), (2, 1), (1, 2)])
def test_activation_lifetimes_match_runtime_ownership(batch, groups):
    from core.rewiring.hybrid_modules import HybridExecution
    from core.rewiring.spec import preview_plan
    spec = declaration()
    preview = preview_plan(spec, 8, 12)["hybrid_activation_layout"]
    sizes = {(item["id"], item["kind"]): item["elements"] for item in preview["entries"]}
    assert sizes == {("outer", "bypass"): 768, ("outer", "low_input"): 192,
                     ("inner", "bypass"): 192, ("inner", "low_input"): 48,
                     ("outer_skip", "compressed_skip"): 288, ("middle_skip", "compressed_skip"): 72}
    assert preview["streams"] == [dict(stream="main", peak_elements=1560, peak_block=2, peak_phase="before")]
    class Observed(HybridExecution):
        def __init__(self):
            super().__init__(HybridModules(spec), "main", groups)
            self.region_elements = {}
            self.peak = 0

        def record(self):
            self.peak = max(self.peak, sum(self.region_elements.values()) + sum(t.numel() for t in self.saved.values()))

        def down(self, region, grid, pooled):
            result = super().down(region, grid, pooled)
            self.region_elements[region["id"]] = grid.numel() + result.numel()
            self.record()
            return result

        def up(self, region, update):
            del self.region_elements[region["id"]]
            return super().up(region, update)

        def after(self, *args):
            result = super().after(*args)
            self.record()
            return result

    observed = Observed()
    run_rewired_stream(torch.randn(batch, 3 + 8*groups*12, 8), legacy_topology(spec),
        "main", 8*groups, 12, 3, None, execute, lambda *args: None,
        grid_groups=groups, extensions=observed)
    assert observed.peak == 1560*batch*groups
    assert not observed.saved and not observed.region_elements


def test_activation_release_at_adjacent_regions_does_not_double_count():
    from core.rewiring.spec import preview_plan
    base = build_spec("anima", [{"name": "main", "block_count": 9}], [
        dict(id="a", stream="main", parent=None, start_block=1, end_block=4, down_factor=[2, 2], **OPERATORS),
        dict(id="b", stream="main", parent=None, start_block=4, end_block=7, down_factor=[2, 2], **OPERATORS)], {})
    spec = build_hybrid_spec(base, stream_widths={"main": 8}, boundaries=[], refiners=[], skips=[])
    layout = preview_plan(spec, 8, 12)["hybrid_activation_layout"]
    assert len(layout["entries"]) == 4
    assert layout["streams"][0]["peak_elements"] == 960
