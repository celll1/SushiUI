"""Independent local oracle for hybrid partition context; no attention equivalence claim."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import torch

from hybrid_rewiring_test import declaration, execute
from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_partition import convolution_halo, resolved_partition_halo, spatial_codec_radius
from core.training.partition import PartitionBox


def test_local_convolution_cores_match_full_grid_with_live_weights():
    torch.manual_seed(101)
    graph = declaration()
    modules = HybridModules(graph)
    with torch.no_grad():
        for parameter in modules.parameters():
            parameter.normal_(0, 0.12)
    height, width = 64, 96
    full = torch.randn(1, height * width, 8)
    expected = modules.forward_stream(full, "main", height, width, 0, None, execute, lambda *args: None)
    expected = expected.transpose(1, 2).reshape(1, 8, height, width)
    halo = resolved_partition_halo(graph, "main", 0)
    assert halo > 0 and halo % 2 == 0
    assert resolved_partition_halo(graph, "main", halo + 10) == halo + 10
    for core in (PartitionBox(0, 0, 64, 48), PartitionBox(0, 48, 64, 96)):
        box = core.expand(halo, full_height=height, full_width=width)
        box = PartitionBox(box.top // 4 * 4, box.left // 4 * 4,
                           min(height, (box.bottom + 3) // 4 * 4), min(width, (box.right + 3) // 4 * 4))
        value = full.transpose(1, 2).reshape(1, 8, height, width)[..., box.top:box.bottom, box.left:box.right]
        value = value.flatten(2).transpose(1, 2)
        actual = modules.forward_stream(value, "main", box.height, box.width, 0, None, execute, lambda *args: None)
        actual = actual.transpose(1, 2).reshape(1, 8, box.height, box.width)
        torch.testing.assert_close(actual[..., core.top-box.top:core.bottom-box.top, core.left-box.left:core.right-box.left],
            expected[..., core.top:core.bottom, core.left:core.right], rtol=1e-5, atol=1e-6)


def test_anima_planner_expands_real_context_and_covers_cores_once():
    with patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
            patch.object(torch.cuda, "_lazy_init"), patch.object(torch._C, "_cuda_init"):
        from core.models.anima.partition_training import training_plan
        graph = declaration()
        trainer = SimpleNamespace(config=dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
            dit_partition_fixed_count=2, dit_partition_halo_tokens=0), transformer=SimpleNamespace(_residual_unet_spec=graph, patch_spatial=2))
        plan = training_plan(trainer, torch.zeros(1, 64, 128, 192))
        halo = max(convolution_halo(graph, "main"))
        for region in plan.regions:
            for start, core_start in ((region.input.top, region.core.top), (region.input.left, region.core.left)):
                assert start == 0 or core_start - start >= halo
            for end, core_end, bound in ((region.input.bottom, region.core.bottom, plan.full_height),
                                          (region.input.right, region.core.right, plan.full_width)):
                assert end == bound or end - core_end >= halo
            assert region.input.height % 4 == 0 and region.input.width % 4 == 0
        assert not torch.cuda.is_initialized()


def test_codec_context_is_added_to_hybrid_context():
    pair = SimpleNamespace(input=SimpleNamespace(source_pack=1, projection=SimpleNamespace(kernel_size=(3, 3))),
        output=SimpleNamespace(target_pack=1, projection=SimpleNamespace(kernel_size=(3, 3))))
    radius = spatial_codec_radius(pair, 2)
    assert radius == (2, 2)
    assert resolved_partition_halo(declaration(), "main", 0, codec_radius=radius) == 16
    pair.input.source_pack = pair.output.target_pack = 4
    assert spatial_codec_radius(pair, 2) == (4, 4)
