"""Compressed skip residency must preserve encoder and projection gradients."""
from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration, execute
from core.rewiring.hybrid_modules import HybridModules, HybridExecution
from core.rewiring.hybrid_spec import (build_hybrid_spec, legacy_topology, validate_hybrid_spec,
    hybrid_metadata, read_hybrid_metadata, module_layout)
from core.rewiring.spec import digest, preview_plan
from core.rewiring.runtime import run_rewired_stream


def cpu_spec(spec):
    skips = [dict(item, storage="cpu") for item in spec["skips"]]
    return build_hybrid_spec(legacy_topology(spec), stream_widths=spec["stream_widths"],
        boundaries=spec["boundaries"], refiners=spec["refiners"], skips=skips)


def test_storage_roundtrip_preserves_old_layout_and_parameters():
    old = declaration()
    changed = cpu_spec(old)
    assert changed["module_layout_sha256"] == old["module_layout_sha256"]
    assert digest(changed) != digest(old)
    assert all("storage" not in item for item in validate_hybrid_spec(old)["skips"])
    assert read_hybrid_metadata(hybrid_metadata(changed)) == changed
    HybridModules(changed).load_state_dict(HybridModules(old).state_dict(), strict=True)
    layout = preview_plan(changed, 8, 12)["hybrid_activation_layout"]
    assert all(item["training_storage"] == "cpu" for item in layout["entries"]
               if item["kind"] == "compressed_skip")


@pytest.mark.parametrize("storage", [None, "unknown", False, 1, [], {}])
def test_unsupported_storage_rejected(storage):
    value = declaration()
    value["skips"][0]["storage"] = storage
    value["module_layout_sha256"] = digest(module_layout(value))
    with pytest.raises(ValueError, match="storage"):
        validate_hybrid_spec(value)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("checkpoint_mode", ["none", "normal", "cpu"])
def test_cpu_skips_match_resident_outputs_and_all_gradients(device, checkpoint_mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA needed for transfer acceptance")
    torch.manual_seed(81)
    checkpointed = checkpoint_mode != "none"
    checkpoint_offload = checkpoint_mode == "cpu"
    spec = declaration()
    resident = HybridModules(spec).to(device)
    for module in resident.modules():
        if hasattr(module, "expand"):
            torch.nn.init.normal_(module.expand.weight, std=0.02)
        if hasattr(module, "restore"):
            torch.nn.init.normal_(module.restore.weight, std=0.02)
    offloaded = HybridModules(cpu_spec(spec)).to(device)
    offloaded.load_state_dict(deepcopy(resident.state_dict()))
    value = torch.randn(2, 3 + 8*12, 8, device=device, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    baseline = resident.forward_stream(value, "main", 8, 12, 3, None, execute,
        lambda *args: None, checkpoint_modules=checkpointed, checkpoint_offload=checkpoint_offload)
    observed = []
    class Checked(HybridExecution):
        def after(self, *args):
            result = super().after(*args)
            observed.extend((tensor.device.type, tensor.requires_grad) for tensor in self.saved.values())
            return result
    execution = Checked(offloaded, "main", checkpoint_modules=checkpointed, checkpoint_offload=checkpoint_offload)
    actual = run_rewired_stream(other, legacy_topology(spec), "main", 8, 12, 3,
        None, execute, lambda *args: None, extensions=execution)
    assert observed and all(location == "cpu" and grad for location, grad in observed)
    assert not execution.saved
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    baseline.square().mean().backward()
    actual.square().mean().backward()
    # CPU copies can change the ordering of FP32 gradient accumulation on CUDA.
    tolerance = dict(rtol=1e-5, atol=1e-8) if device == "cuda" else dict(rtol=0, atol=0)
    torch.testing.assert_close(other.grad, value.grad, **tolerance)
    for (name, left), (_, right) in zip(resident.named_parameters(), offloaded.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
        assert torch.isfinite(right.grad).all(), name
        assert right.grad.abs().sum() > 0, name


def test_api_builder_and_preview_preserve_storage():
    from api.rewiring_routes import HybridBuildRequest, build_hybrid, RewiringPreviewRequest, validate_rewiring
    spec = cpu_spec(declaration())
    result = build_hybrid(HybridBuildRequest(base_spec=spec, **{
        key: spec[key] for key in ("stream_widths", "boundaries", "refiners", "skips")}))
    assert result["spec"] == spec
    preview = validate_rewiring(RewiringPreviewRequest(spec=spec, token_height=8, token_width=12))
    assert preview["spec"] == spec
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        build_hybrid(HybridBuildRequest(base_spec=spec, stream_widths=spec["stream_widths"],
            boundaries=spec["boundaries"], refiners=spec["refiners"], skips=[{}]))
    assert error.value.status_code == 422


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("storage", ["cpu", "recompute"])
def test_inference_does_not_offload_skips(device, storage):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA needed for inference residency acceptance")
    spec = cpu_spec(declaration())
    for item in spec["skips"]:
        item["storage"] = storage
    execution = HybridExecution(HybridModules(spec).to(device), "main")
    with torch.no_grad():
        execution.after(0, torch.randn(1, 8*12, 8, device=device), 0, 8, 12)
    assert execution.saved
    assert all(value.device.type == device and not value.requires_grad for value in execution.saved.values())


def test_cuda_graph_factory_preserves_cpu_and_cuda_rng():
    if not torch.cuda.is_available():
        pytest.skip("CUDA factory verification")
    torch.manual_seed(719)
    cpu_before, cuda_before = torch.get_rng_state(), torch.cuda.get_rng_state()
    with torch.device("cuda"):
        modules = HybridModules(declaration())
    assert all(parameter.is_cuda for parameter in modules.parameters())
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
