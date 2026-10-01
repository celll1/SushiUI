"""CPU native PipelineManager generation and phase cleanup integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*a, **k):
    raise AssertionError("Ming manager test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.is_available = lambda: False
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.pipeline import DiffusionPipelineManager
from core.models.ming_image_01.residency import MingPhaseResidency
from core.models.ming_image_01.results import MingGenerationOutput
from api.error_handlers import ValidationError
from ming_image_01_sampling_test import make_sampler
from ming_image_01_conditioning_test import make_conditioner
from ming_image_01_prompting_test import processor


def manager(profile):
    components = make_sampler(profile).components
    components.conditioner = make_conditioner().eval()
    components.conditioner.set_trainable(set())
    components.processor = processor()
    components.artifact.summary = lambda: {"components": {}}
    result = {"ming_native": components, "transformer": components.transformer,
              "vae": components.vae, "text_encoder": components.conditioner}
    value = DiffusionPipelineManager()
    value.device = "cpu"
    value._install_ming_image_01(result, model_id="tiny-Ming", source_type="local", source="tiny-Ming")
    return value, components


def params(**kw):
    return {"prompt": "word5", "width": 32, "height": 32, "steps": 1, "cfg_scale": 1.,
            "seed": 7, "batch_size": 1, "mask_blur": 0, **kw}


def cpu_only(components):
    assert all(parameter.device.type == "cpu" for module in
               (components.conditioner, components.transformer, components.vae) for parameter in module.parameters())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_native_manager_reference_batch_roles_reproducibility_and_progress(profile):
    value, components = manager(profile)
    source = Image.new("RGBA", (32, 32), (120, 40, 50, 100))
    steps = []
    callback_steps = []
    kwargs = params(batch_size=2, ming_image_01_layers=2)
    output, seed, ancestral = value.generate_img2img(kwargs, source,
        progress_callback=lambda *a, **k: steps.append((a, k)),
        step_callback=lambda *a: callback_steps.append(a))
    assert isinstance(output, MingGenerationOutput) and seed == ancestral == 7
    assert len(output.groups) == 2
    expected = ["image"] if profile == "generation_edit" else ["auxiliary_canvas", "layer_1", "layer_2"]
    assert all(group.roles == expected for group in output.groups)
    assert len(steps) == len(callback_steps) == 1
    assert steps[0][0][2].shape == (2,16,4,4) and steps[0][0][4].shape == (2,16,4,4)
    assert not steps[0][1]
    assert output.model_info["profile"] == profile and output.model_info["source"] == "tiny-Ming"
    repeated, *_ = value.generate_img2img(kwargs, source)
    for left, right in zip(output.groups, repeated.groups):
        assert all(a.tobytes() == b.tobytes() for a, b in zip(left.values, right.values))
    cpu_only(components)


def test_design_text_and_inpaint_keep_native_alpha_and_unmasked_pixels():
    value, components = manager("generation_edit")
    text, *_ = value.generate_txt2img(params())
    assert text.primary.mode == "RGBA"
    original = Image.new("RGBA", (32, 32), (20, 40, 60, 90))
    mask = Image.new("L", original.size, 0)
    mask.paste(255, (16, 0, 32, 32))
    result, *_ = value.generate_inpaint(params(), original, mask)
    assert result.primary.mode == "RGBA"
    assert result.primary.crop((0, 0, 16, 32)).tobytes() == original.crop((0, 0, 16, 32)).tobytes()
    cpu_only(components)


def test_design_outpaint_preserves_placed_rgba_rectangle():
    value, components = manager("generation_edit")
    original = Image.new("RGBA", (16, 16), (20, 80, 40, 113))
    result, *_ = value.generate_outpaint(params(canvas_width=32, canvas_height=32,
        place_x=8, place_y=8, place_width=16, place_height=16), original)
    assert result.primary.size == (32, 32) and result.primary.mode == "RGBA"
    assert result.primary.crop((8, 8, 24, 24)).tobytes() == original.tobytes()
    cpu_only(components)


def test_layer_refuses_inpaint_and_missing_reference_before_residency(monkeypatch):
    value, components = manager("layer_decompose")
    def forbidden(*a, **k): raise AssertionError("Unexpected residency mutation")
    monkeypatch.setattr(components.conditioner, "to", forbidden)
    with pytest.raises(ValidationError, match="reference"):
        value.generate_txt2img(params())
    with pytest.raises(ValidationError, match="inpaint/outpaint"):
        value.generate_inpaint(params(), Image.new("RGBA", (32, 32)), Image.new("L", (32, 32)))


@pytest.mark.parametrize("failure", ["cancel", "callback"])
def test_manager_cancel_or_callback_failure_cleans_up_and_next_request_recovers(failure):
    value, components = manager("generation_edit")
    def callback(*a):
        if failure == "cancel": value.cancel_requested = True
        else: raise RuntimeError("callback failed")
    with pytest.raises(RuntimeError, match="cancelled|callback failed"):
        value.generate_txt2img(params(steps=2), step_callback=callback)
    cpu_only(components)
    assert len(components.transformer._forward_pre_hooks) == 0
    value.reset_cancel_flag()
    assert value.generate_txt2img(params())[0].primary.mode == "RGBA"


def test_phase_policy_never_overlaps_and_cleans_partial_staging_failure():
    events = []
    class Module:
        def __init__(self, name): self.name = name
        def to(self, *, device):
            events.append((self.name, str(device)))
            if self.name == "vae" and str(device) == "cuda": raise RuntimeError("synthetic staging failure")
    components = SimpleNamespace(conditioner=Module("conditioner"), transformer=Module("transformer"), vae=Module("vae"))
    residency = MingPhaseResidency(components, device="cuda", cpu_text_encoding=False)
    with residency("conditioner"):
        with pytest.raises(RuntimeError, match="overlap"):
            with residency("denoising"): pass
    assert events[-3:] == [("conditioner", "cpu"), ("transformer", "cpu"), ("vae", "cpu")]
    with pytest.raises(RuntimeError, match="staging failure"):
        with residency("denoising"): pass
    assert events[-3:] == [("conditioner", "cpu"), ("transformer", "cpu"), ("vae", "cpu")]
    assert residency.active is None and not torch.cuda.is_initialized()


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_manager_runs_restored_nested_unet_with_native_reference_and_layer_frames(profile):
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology
    from ming_image_01_rewiring_test import config, spec_for
    value, components = manager(profile)
    padding = "learned" if profile == "layer_decompose" else "zero_masked"
    configuration = config(padding)
    transformer = create_transformer(configuration, device="cpu", dtype=torch.float32).eval()
    if padding == "learned":
        torch.nn.init.constant_(transformer.x_pad_token, 0.02)
        torch.nn.init.constant_(transformer.cap_pad_token, 0.03)
    topology = spec_for(transformer)
    restore_topology(transformer, topology)
    components.transformer = transformer
    components.artifact.configs["transformer"] = configuration
    value.ming_image_01_components["transformer"] = transformer
    request = params(width=64, height=64, ming_image_01_layers=2)
    result, *_ = value.generate_img2img(request, Image.new("RGBA", (64, 64), (120, 80, 40, 100)))
    assert result.model_info["dit_rewiring_spec"] == topology
    assert request["dit_rewiring_actual"]["spec"] == topology
    assert request["dit_rewiring_actual"]["native_layout"] == "target_reference_padding_caption_v1"
    assert len(result.groups[0].values) == (3 if padding == "learned" else 1)
    assert result.primary.size == (64, 64)
    cpu_only(components)


def test_generation_snapshot_is_captured_before_callback_model_changes():
    value, _ = manager("generation_edit")
    def change_metadata(*a): value.current_model_info["source"] = "another-model"
    result, *_ = value.generate_txt2img(params(), step_callback=change_metadata)
    assert result.model_info["source"] == "tiny-Ming"
    assert value.current_model_info["source"] == "another-model"


@pytest.mark.parametrize("override", [{"width": 33}, {"negative_prompt": "negative"},
    {"cfg_scale": float("nan")}, {"ming_image_01_layers": True}, {"unet_quantization": "fp8"}])
def test_invalid_requests_refused_before_model_residency(monkeypatch, override):
    value, components = manager("layer_decompose")
    def forbidden(*a, **k): raise AssertionError("Unexpected residency mutation")
    monkeypatch.setattr(components.conditioner, "to", forbidden)
    with pytest.raises(ValidationError):
        value.generate_img2img(params(**override), Image.new("RGBA", (32, 32)))


@pytest.mark.parametrize("route", ["generate_txt2img", "generate_img2img", "generate_inpaint", "generate_outpaint"])
@pytest.mark.parametrize("validation", [True, False])
def test_image_route_error_boundary_preserves_validation_status(route, validation):
    import ast
    from copy import deepcopy
    from api.error_handlers import APIError, GenerationError
    tree = ast.parse((Path(__file__).resolve().parents[1] / "api/routes.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == route)
    boundary = next(node for node in function.body if isinstance(node, ast.Try)
                    and any(isinstance(h.type, ast.Name) and h.type.id == "APIError" for h in node.handlers))
    probe = ast.parse("def probe():\n    try: raise failure\n    except Exception: pass").body[0]
    probe.body[0].handlers = deepcopy(boundary.handlers)
    namespace = {"APIError": APIError, "GenerationError": GenerationError, "_gen_id": "cpu-fixture",
                 "fail_generation": lambda *a, **k: None, "attach_error_context": lambda *a: None,
                 "error_context": lambda *a: {}, "failure": ValidationError("bad input") if validation else RuntimeError("unexpected")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), "route-boundary", "exec"), namespace)
    with pytest.raises(APIError) as caught:
        namespace["probe"]()
    assert caught.value.status_code == (400 if validation else 500)
    if validation:
        assert caught.value is namespace["failure"]
    assert not torch.cuda.is_initialized()
