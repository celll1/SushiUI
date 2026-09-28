"""CPU contracts for geometry, causal cache, global adapters and generation lifetime."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference.qwen_image_21_partition import (
    Bounds, PartitionExecutor, normalize_partition_config, resolve_partition_plan,
)
from core.models.qwen_image_21.partition_global import (
    GLOBAL_ADAPTER_NAME, QwenPartitionGlobalAdapter, global_adapter_metadata, read_partition_global_payload,
)
from core.models.qwen_image_21.vendor import QwenImage21Transformer2DModel


def strips(n, axis="x"):
    return None if n == 1 else dict(axis=axis, ratio=1 / n, first=None, second=strips(n - 1, axis))


@pytest.mark.parametrize("count", [2, 3, 6])
@pytest.mark.parametrize("halo", [0, 2])
def test_manual_covers_once_and_expands_halo(count, halo):
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(count), halo_tokens=halo), 12, 24)
    coverage = torch.zeros(12, 24)
    assert len(plan.cores) == count
    for core, box in zip(plan.cores, plan.inputs):
        coverage[core.top:core.bottom, core.left:core.right] += 1
        assert box == core.expanded(halo, 12, 24)
        assert all(v % 2 == 0 for v in (core.top, core.left, core.bottom, core.right))
    assert torch.equal(coverage, torch.ones_like(coverage))


def test_auto_ceiling_counts_halo_and_full_frame_is_noop():
    assert resolve_partition_plan(dict(mode="off"), 64, 64) is None
    assert resolve_partition_plan(dict(mode="auto", max_target_tokens=4096), 64, 64) is None
    assert resolve_partition_plan(dict(mode="manual", layout=None), 64, 64) is None
    plan = resolve_partition_plan(dict(mode="auto", max_target_tokens=96, halo_tokens=2), 12, 24)
    assert max(b.tokens for b in plan.inputs) <= 96
    with pytest.raises(ValueError):
        resolve_partition_plan(dict(mode="auto", max_target_tokens=1), 12, 24)


@pytest.mark.parametrize("value", ["[1]", dict(mode=[]), dict(version=True), dict(halo_tokens=1),
    dict(max_target_tokens=0), dict(unknown=1), dict(layout=dict(axis=[], ratio=.5, first=None, second=None)),
    dict(layout=dict(axis="x", ratio=float("nan"), first=None, second=None))])
def test_malformed_config_rejected(value):
    with pytest.raises(ValueError):
        normalize_partition_config(value)


def model():
    torch.manual_seed(19)
    return QwenImage21Transformer2DModel(in_channels=8, out_channels=8, num_layers=1,
        attention_head_dim=16, num_attention_heads=2, context_in_dim=32,
        mlp_ratio=2, axes_dims_rope=(4, 6, 6)).eval()


def inputs(with_reference=True, dtype=torch.float32):
    torch.manual_seed(4)
    prefix = 16 if with_reference else 0
    mask = torch.cat([torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, (prefix + 48) // 4, dtype=torch.bool)], 1)
    return dict(hidden_states=torch.randn(1, prefix + 48, 8, dtype=dtype),
        timestep=torch.tensor([.5]), encoder_hidden_states=torch.randn(1, 2 + prefix // 4, 32, dtype=dtype),
        img_shapes=[([(1, 4, 4)] if with_reference else []) + [(1, 4, 12)]],
        img_mask=mask, return_dict=False)


@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_transformer_cache_and_traversal_agree(with_reference, dtype):
    net = model().to(dtype=dtype)
    args = inputs(with_reference, dtype)
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3), halo_tokens=2), 4, 12)
    reversed_plan = replace(plan, cores=tuple(reversed(plan.cores)), inputs=tuple(reversed(plan.inputs)))
    with torch.no_grad():
        uncached = PartitionExecutor(net, plan, use_cache=False)
        cached = PartitionExecutor(net, plan, use_cache=True)
        reverse = PartitionExecutor(net, reversed_plan, use_cache=False)
        expected = uncached.predict(branch="cond", **args)
        assert torch.equal(expected, reverse.predict(branch="cond", **args))
        atol = 0.025 if dtype == torch.bfloat16 else 2e-5
        torch.testing.assert_close(expected, cached.predict(branch="cond", **args), atol=atol, rtol=atol)
        args["timestep"] = torch.tensor([.2])
        torch.testing.assert_close(uncached.predict(branch="cond", **args), cached.predict(branch="cond", **args), atol=atol, rtol=atol)
        args["encoder_hidden_states"] = -args["encoder_hidden_states"]
        torch.testing.assert_close(uncached.predict(branch="negative", **args), cached.predict(branch="negative", **args), atol=atol, rtol=atol)
    assert set(cached.caches) == {"cond", "negative"}
    cached.close()
    assert not cached.caches and not cached.modules


def payload(net):
    adapter = QwenPartitionGlobalAdapter(8, net.inner_dim, rank=4, summary_tokens=3)
    torch.nn.init.normal_(adapter.out.weight, std=.1)
    state = {f"{GLOBAL_ADAPTER_NAME}.{key}": value for key, value in adapter.export_tensors().items()}
    return read_partition_global_payload(net, state, global_adapter_metadata(adapter))


def test_global_strength_step_gate_and_rng_isolation():
    net = model()
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3)), 4, 12)
    saved = payload(net)
    source = SimpleNamespace(apply_to_unet=True, strength=.5, step_range=(500, 1000))
    executor = PartitionExecutor(net, plan, [(saved, source)], step_sources=[source], use_cache=False)
    args = inputs(False)
    with torch.no_grad():
        executor.set_step(0, 10)
        original = executor.predict(branch="cond", **args)
        assert not executor.modules
        rng = torch.get_rng_state().clone()
        executor.set_step(5, 10)
        changed = executor.predict(branch="cond", **args)
        assert torch.equal(rng, torch.get_rng_state())
        assert executor.modules and not torch.equal(changed, original)
        source.strength = 0
        assert torch.equal(original, executor.predict(branch="cond", **args))
    executor.close()
    assert not executor.modules and not executor.payloads


def test_global_outputs_sum_and_reference_prefix_is_not_modified():
    net = model()
    saved = payload(net)
    adapter = QwenPartitionGlobalAdapter(8, 32, rank=4, summary_tokens=3)
    adapter.load_tensors(saved["state"])
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3)), 4, 12)
    args = inputs(True)
    grid = args["hidden_states"][:, -48:].reshape(1, 4, 12, 8)
    residuals = []
    class Recorder:
        config = net.config
        inner_dim = net.inner_dim
        transformer_blocks = net.transformer_blocks
        def __call__(self, hidden_states, target_input_residual, **_kwargs):
            residuals.append(target_input_residual.clone())
            assert not torch.count_nonzero(target_input_residual[:, :16])
            return (hidden_states,)
    half = SimpleNamespace(apply_to_unet=True, strength=.5, step_range=None)
    zero = SimpleNamespace(apply_to_unet=True, strength=0, step_range=None)
    executor = PartitionExecutor(Recorder(), plan, [(saved, None), (saved, half), (saved, zero)], use_cache=False)
    with torch.no_grad():
        actual = executor.predict(branch="cond", **args)
        assert torch.equal(actual, args["hidden_states"][:, -48:])
        assert len(executor.modules) == 2
        for residual, box in zip(residuals, plan.inputs):
            torch.testing.assert_close(residual[:, 16:], 1.5 * adapter(grid, box))
    executor.close()


def test_changed_prompt_invalidates_prefix_cache():
    net = model()
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3)), 4, 12)
    executor = PartitionExecutor(net, plan)
    args = inputs()
    with torch.no_grad():
        executor.predict(branch="cond", **args)
        old = executor.caches["cond"]
        args["encoder_hidden_states"].add_(1)
        executor.predict(branch="cond", **args)
        assert executor.caches["cond"] is not old
    executor.close()


def test_non_global_step_gate_invalidates_cache_and_cancel_releases():
    net = model()
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3)), 4, 12)
    source = SimpleNamespace(apply_to_unet=True, strength=1, step_range=(500, 1000))
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        return calls > 1
    executor = PartitionExecutor(net, plan, step_sources=[source], cancelled=cancel)
    with torch.no_grad(), pytest.raises(RuntimeError, match="cancelled"):
        executor.predict(branch="cond", **inputs())
    assert executor.caches
    executor.set_step(0, 10)
    assert not executor.caches
    executor.caches["cond"] = object()
    executor.set_step(5, 10)
    assert not executor.caches
    executor.close()
    assert not executor.step_sources


def test_openapi_default_and_multipart_wiring():
    import yaml
    from api.param_defaults import GENERATION_DEFAULTS
    schema = yaml.safe_load((BACKEND.parent / "openapi.yaml").read_text(encoding="utf-8"))
    prop = schema["components"]["schemas"]["GenerationParams"]["properties"]["qwen_image_21_partition"]
    assert prop["default"] == GENERATION_DEFAULTS["qwen_image_21_partition"]
    assert schema["components"]["schemas"]["QwenImage21Partition"]["properties"]["mode"]["default"] == "off"
    text = (BACKEND / "api/routes.py").read_text(encoding="utf-8")
    assert text.count("qwen_image_21_partition: str = Form(") == 4
    assert text.count('"qwen_image_21_partition": _parse_qwen_partition(qwen_image_21_partition)') == 4
    assert normalize_partition_config(json.dumps(prop["default"])) == prop["default"]


@pytest.mark.parametrize("cfg", [1.0, 3.0])
def test_vendor_pipeline_advances_full_canvas_once_per_step(monkeypatch, cfg):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from core.models.qwen_image_21.vendor.pipeline import QwenImage21Pipeline
    class TinyPipeline(QwenImage21Pipeline):
        @property
        def _execution_device(self):
            return torch.device("cpu")
    net = model()
    vae = torch.nn.Linear(1, 1)
    vae.config = SimpleNamespace(z_dim=8)
    class Processor:
        tokenizer = SimpleNamespace(encode=lambda _value: [1])
        def apply_chat_template(self, *_args, **_kwargs):
            return [[1]]
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = TinyPipeline(scheduler, vae, torch.nn.Linear(1, 1), Processor(), net)
    pipe.set_progress_bar_config(disable=True)
    args = inputs(False)
    monkeypatch.setattr(pipe, "encode_prompt", lambda **kw: (
        args["encoder_hidden_states"] if kw["prompt"] == "test" else -args["encoder_hidden_states"],
        torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool)))
    monkeypatch.setattr(pipe, "prepare_latents", lambda *_args: (args["hidden_states"].clone(), None))
    monkeypatch.setattr(pipe, "maybe_free_model_hooks", lambda: None)
    steps = []
    original = scheduler.step
    def step(pred, t, latent, **kwargs):
        assert pred.shape == latent.shape == (1, 48, 8)
        steps.append(pred.clone())
        return original(pred, t, latent, **kwargs)
    monkeypatch.setattr(scheduler, "step", step)
    plan = resolve_partition_plan(dict(mode="manual", layout=strips(3)), 4, 12)
    executor = PartitionExecutor(net, plan)
    kwargs = dict(prompt="test", negative_prompt="negative", true_cfg_scale=cfg,
        width=192, height=64, num_inference_steps=3, output_type="latent", use_kv_cache=True)
    result = pipe(**kwargs, partition_executor=executor, before_step_callback=executor.set_step).images
    assert result.shape == (1, 48, 8) and len(steps) == 3
    assert set(executor.caches) == ({"cond"} if cfg == 1 else {"cond", "uncond"})
    executor.close()
    steps.clear()
    ordinary = pipe(**kwargs).images
    assert len(steps) == 3 and ordinary.shape == result.shape


def test_backend_off_path_and_failure_cleanup(monkeypatch):
    import core.inference.qwen_image_21_partition as module
    from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
    net = model()
    saved = payload(net)
    net._qwen_partition_global_payload = saved
    received = []
    def run(**kwargs):
        received.append(kwargs)
        if "partition_executor" in kwargs:
            executor = kwargs["partition_executor"]
            executor.modules["sentinel"] = object()
            raise RuntimeError("test failure")
        return SimpleNamespace(images=["image"])
    mixin = QwenImage21Mixin()
    mixin.qwen_image_21_components = {"transformer": net}
    mixin.cancel_requested = False
    monkeypatch.setattr(mixin, "_qwen_image_21_pipe", lambda: run)
    monkeypatch.setattr(mixin, "_load_lora_qwen21", lambda _: 0)
    unloads = []
    monkeypatch.setattr(mixin, "_unload_lora_qwen21", lambda: unloads.append(1))
    result = mixin._qwen_image_21_run(dict(seed=4, width=192, height=64,
        qwen_image_21_partition=dict(mode="auto", max_target_tokens=48)))
    assert result[0] == "image" and "partition_executor" not in received[0]
    assert not any(isinstance(m, QwenPartitionGlobalAdapter) for m in net.modules())
    params = dict(seed=4, width=192, height=64, qwen_image_21_partition=dict(mode="manual", layout=strips(3)))
    with pytest.raises(RuntimeError, match="test failure"):
        mixin._qwen_image_21_run(params)
    executor = received[-1]["partition_executor"]
    assert not executor.modules and not executor.payloads and len(unloads) == 2
    assert params["qwen_image_21_partition_actual"]["global_adapters"][0]["name"] == "full model"


def test_lora_embedded_global_load_and_failed_second_file_are_atomic(tmp_path, monkeypatch):
    from safetensors.torch import save_file
    from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin
    net = model()
    saved = payload(net)
    state = {f"{GLOBAL_ADAPTER_NAME}.{key}": tensor for key, tensor in saved["state"].items()}
    state.update({"lora_unet_transformer_blocks__0__attn__to_q.lora_down.weight": torch.ones(2, 32) * .1,
                  "lora_unet_transformer_blocks__0__attn__to_q.lora_up.weight": torch.ones(32, 2) * .1})
    good = tmp_path / "good.safetensors"
    bad = tmp_path / "bad.safetensors"
    save_file(state, good, metadata=saved["metadata"])
    save_file({f"{GLOBAL_ADAPTER_NAME}.out.weight": saved["state"]["out.weight"]}, bad, metadata=saved["metadata"])
    mixin = QwenImage21Mixin()
    mixin.qwen_image_21_components = {"transformer": net}
    original = net.transformer_blocks[0].attn.to_q
    monkeypatch.setattr(mixin, "_qwen21_resolve_lora_path", lambda path: path)
    assert mixin._load_lora_qwen21([dict(path=str(good), strength=.5)]) == 1
    assert mixin._qwen21_loaded_files[0].qwen_global_payload is not None
    assert net.transformer_blocks[0].attn.to_q is not original
    mixin._unload_lora_qwen21()
    assert net.transformer_blocks[0].attn.to_q is original and not mixin._qwen21_loaded_files
    from core.adapters import AdapterLoadFailed
    with pytest.raises(AdapterLoadFailed) as error:
        mixin._load_lora_qwen21([dict(path=str(good)), dict(path=str(bad))])
    assert "incomplete" in str(error.value.__cause__)
    assert net.transformer_blocks[0].attn.to_q is original and not mixin._qwen21_loaded_files
    # A global-only artifact is valid and never counted as an ordinary LoRA branch.
    global_only = tmp_path / "global.safetensors"
    save_file({f"{GLOBAL_ADAPTER_NAME}.{key}": tensor for key, tensor in saved["state"].items()}, global_only, metadata=saved["metadata"])
    assert mixin._load_lora_qwen21([dict(path=str(global_only))]) == 0
    assert len(mixin._qwen21_loaded_files) == 1
    mixin._unload_lora_qwen21()


def test_distributed_halo_guidance_roundtrip_and_legacy_fallback(tmp_path):
    from api.param_defaults import QWEN_IMAGE_21_GENERATION_DEFAULTS
    from core.extensions.lora_manager import _partition_guidance_from_metadata, LoRAManager
    from safetensors.torch import save_file
    net = model()
    adapter = QwenPartitionGlobalAdapter(8, net.inner_dim, rank=4, summary_tokens=3)
    metadata = global_adapter_metadata(adapter, {"dit_partition_halo_tokens": 0})
    expected = QWEN_IMAGE_21_GENERATION_DEFAULTS["qwen_image_21_partition"]["halo_tokens"]
    assert metadata["qwen_partition_global_generation_halo_tokens"] == str(expected)
    state = {f"{GLOBAL_ADAPTER_NAME}.{key}": value for key, value in adapter.export_tensors().items()}
    restored = read_partition_global_payload(net, state, metadata)
    assert restored["metadata"] == metadata
    path = tmp_path / "distributed.safetensors"
    save_file(state, path, metadata=metadata)
    record = LoRAManager.__new__(LoRAManager)._read_lora_header(path)
    assert record["partition_guidance"] == {
        "halo_tokens": expected, "halo_pixels": expected * 16,
        "training_halo_tokens": 0, "source": "file_metadata",
    }
    legacy = _partition_guidance_from_metadata(global_adapter_metadata(adapter))
    assert legacy["source"] == "sushiui_default"
    assert legacy["training_halo_tokens"] is None
    assert _partition_guidance_from_metadata({}) is None
    for invalid in ("abc", "-2", "3", "16.5"):
        invalid_metadata = {**metadata, "qwen_partition_global_generation_halo_tokens": invalid}
        assert _partition_guidance_from_metadata(invalid_metadata)["source"] == "sushiui_default"
    declared = {**metadata, "qwen_partition_global_generation_halo_tokens": "8"}
    assert _partition_guidance_from_metadata(declared)["halo_tokens"] == 8
