"""Strategy section 7's SenseNova gate: is the trained null the inference null?

`local/strategy/cfg_null_alignment/IMPLEMENTATION_STRATEGY.md` section 6.3 makes
one claim -- a dropped item is encoded as the condition inference's own uncond
branch builds -- and section 7 asks for it to be shown on tokens, image indexes,
prefix K/V and one denoise step. Four arms, cheapest first:

* ``tokens``   -- real tokenizer, no weights: the training null's query string,
                  token ids and prefix length against inference's uncond arm.
* ``indexes``  -- real tokenizer, no weights: the image t/h/w indexes the
                  training step builds against the ones inference builds, at
                  several resolutions.
* ``kv``       -- WEIGHTS: prefix K/V layer by layer, training null vs the real
                  ``sensenova_pipeline_ops.encode_prompt`` uncond branch.
* ``velocity`` -- WEIGHTS: one denoise step on a fixed image and timestep,
                  driven from each prefix.
* ``weights``  -- ``kv`` then ``velocity`` in one load (one process, one model).

The two weight arms load the ~17.6 GiB int8 checkpoint; run one at a time and
let the process exit. The understanding half is frozen throughout (this probe
never builds a trainable prefix and never calls backward), which is the mode
section 7 names.

    venv/Scripts/python.exe backend/core/training/probes/sensenova_cfg_null_parity.py --arm tokens
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.models.sensenova.vendor.conversation import get_conv_template  # noqa: E402
from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402

DEFAULT_TEMPLATE = "neo1_0"
DEFAULT_CHECKPOINT = ("sensenova", "sensenova_int8.safetensors")
# Aligned to TOKEN_GRID_ALIGN (32). Two shapes, one of them non-square, so a
# token grid that happened to be symmetric cannot hide an index defect.
DEFAULT_RESOLUTIONS = ((512, 512), (384, 640))
# int8 weights plus the bf16 heads sit around 14 GiB; the gate leaves the rest
# of a shared card alone and makes an over-budget arm fail with a number.
VRAM_GATE_FRACTION = 0.72


def _repo_venv_python() -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return (REPO_ROOT / "venv" / relative).resolve()


def _require_repo_venv() -> None:
    expected = os.path.normcase(str(_repo_venv_python()))
    actual = os.path.normcase(str(Path(sys.executable).resolve()))
    if actual != expected:
        raise RuntimeError(
            "Run this probe with the repository virtualenv: "
            f"{_repo_venv_python()} {Path(__file__).resolve()}"
        )


def _model_root() -> str:
    """`backend/tests/model_root.py`'s rule, not a second copy of it."""
    spec = importlib.util.spec_from_file_location(
        "_probe_model_root", BACKEND_ROOT / "tests" / "model_root.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_root()


def default_model_path() -> str:
    return os.path.join(_model_root(), *DEFAULT_CHECKPOINT)


# ---------------------------------------------------------------------------
# Weightless: the vendor builders bound to a `self` that carries no parameters
# ---------------------------------------------------------------------------


class QueryBuilders:
    """`NEOChatModel`'s own query/text-input/index builders, no weights.

    These three methods read nothing but `self.template`, `self.system_message`
    and `self.device`, so binding them to this object runs the REAL builders --
    the arms below are not a restatement of what the model does.
    """

    _build_t2i_query = NEOChatModel._build_t2i_query
    _build_t2i_text_inputs = NEOChatModel._build_t2i_text_inputs
    _build_t2i_image_indexes = NEOChatModel._build_t2i_image_indexes

    def __init__(self, template: str = DEFAULT_TEMPLATE, device: str = "cpu"):
        self.template = template
        # modeling_neo_chat.py:261-262, the model's own two lines.
        self.system_message = get_conv_template(template).system_message
        self.device = torch.device(device)


def checkpoint_template(model_dir: str) -> str:
    """The template the checkpoint's own config names (header read, no tensors)."""
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)["template"]


def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


def inference_uncond_query(transformer, negative_prompt: Optional[str] = None) -> str:
    """`sensenova_pipeline_ops.encode_prompt`'s uncond arm, its own two lines."""
    negative_prompt = (negative_prompt or "").strip()
    return transformer._build_t2i_query(negative_prompt, append_text="<img>")


def training_null_query(trainer, transformer) -> str:
    """Whatever `_build_prefix_inputs(cfg_null=True)` actually hands the tokenizer."""
    from core.training.ops import sensenova_ops

    sensenova_ops._build_prefix_inputs(
        trainer, transformer, "a caption the null must ignore", [], True
    )
    return trainer.tokenizer.last_query


class _RecordingTokenizer:
    """Wrap a tokenizer and remember the last query it was handed."""

    def __init__(self, inner):
        self._inner = inner
        self.last_query: Optional[str] = None

    def __call__(self, query, **kwargs):
        self.last_query = query
        return self._inner(query, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class WeightlessTrainer:
    """The `sensenova_ops.encode_prompt` trainer surface, minus the model."""

    train_text_encoder = False
    sensenova_phase_evictor = None
    sensenova_four_phase = None
    gradient_checkpointing = False
    training_dtype = torch.float32
    device = "cpu"

    def __init__(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer


def token_parity(transformer, tokenizer) -> dict[str, Any]:
    """Arm 1: same query string, same token ids, same prefix length."""
    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN
    from core.training.ops import sensenova_ops

    recording = _RecordingTokenizer(tokenizer)
    trainer = WeightlessTrainer(transformer, recording)

    training = sensenova_ops._build_prefix_inputs(
        trainer, transformer, "a caption the null must ignore", [], True
    )
    training_query = recording.last_query
    inference_query = inference_uncond_query(transformer)
    inference_ids, inference_indexes, _ = transformer._build_t2i_text_inputs(
        tokenizer, inference_query
    )

    conditional_query = transformer._build_t2i_query(
        "a caption the null must ignore",
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>",
    )
    conditional_ids, _, _ = transformer._build_t2i_text_inputs(
        tokenizer, conditional_query
    )

    # encode_prompt's own derivation on the frozen route.
    training_text_length = int(training.indexes[0].max()) + 1
    return {
        "query_equal": training_query == inference_query,
        "training_query": training_query,
        "inference_query": inference_query,
        "ids_equal": torch.equal(training.tokens, inference_ids),
        "null_token_count": int(inference_ids.shape[1]),
        "conditional_token_count": int(conditional_ids.shape[1]),
        "training_text_length": training_text_length,
        # Training reads max+1, inference reads shape[1]; equal on the text-only
        # path because t is an arange there, asserted rather than assumed.
        "inference_text_length": int(inference_indexes.shape[1]),
        "text_length_equal": training_text_length == int(inference_indexes.shape[1]),
        "null_differs_from_conditional": int(inference_ids.shape[1])
        != int(conditional_ids.shape[1]),
    }


def index_parity(transformer, tokenizer, resolutions=DEFAULT_RESOLUTIONS) -> dict[str, Any]:
    """Arm 2: the image t/h/w indexes, training step vs inference uncond."""
    from core.training.ops import sensenova_ops

    recording = _RecordingTokenizer(tokenizer)
    trainer = WeightlessTrainer(transformer, recording)
    training = sensenova_ops._build_prefix_inputs(
        trainer, transformer, "a caption", [], True
    )
    training_text_length = int(training.indexes[0].max()) + 1

    inference_query = inference_uncond_query(transformer)
    inference_ids, inference_indexes, _ = transformer._build_t2i_text_inputs(
        tokenizer, inference_query
    )

    # patch_size 16 * merge_size 2; the probe states the geometry it assumes so a
    # checkpoint that changed it fails here rather than silently comparing 0x0.
    patch = 32
    results = []
    for width, height in resolutions:
        if width % patch or height % patch:
            raise ValueError(f"{width}x{height} is not aligned to the {patch}px token grid")
        token_h, token_w = height // patch, width // patch
        training_indexes = transformer._build_t2i_image_indexes(
            token_h, token_w, training_text_length, device=transformer.device
        )
        inference_image_indexes = transformer._build_t2i_image_indexes(
            token_h, token_w, inference_indexes.shape[1], device=inference_ids.device
        )
        results.append(
            {
                "resolution": [width, height],
                "token_grid": [token_h, token_w],
                "equal": torch.equal(training_indexes, inference_image_indexes),
                "t_value": int(training_indexes[0][0]),
            }
        )
    return {"resolutions": results, "all_equal": all(r["equal"] for r in results)}


# ---------------------------------------------------------------------------
# With weights
# ---------------------------------------------------------------------------


def _cuda_memory() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }


def _host_rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _apply_vram_gate(fraction: float = VRAM_GATE_FRACTION) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"applied": False}
    torch.cuda.set_per_process_memory_fraction(float(fraction), 0)
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "applied": True,
        "fraction": float(fraction),
        "budget_bytes": int(total * fraction),
    }


class LoadedTrainer:
    """The `sensenova_ops.encode_prompt` trainer surface over the real model."""

    train_text_encoder = False
    sensenova_phase_evictor = None
    sensenova_four_phase = None
    gradient_checkpointing = False

    def __init__(self, transformer, tokenizer, device: str, dtype: torch.dtype):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.device = device
        self.training_dtype = dtype


def _copy_cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (layer.keys.detach().clone(), layer.values.detach().clone())
        for layer in cache.layers
    ]


def _inference_uncond_prefix(transformer, tokenizer, prompt, width, height, cfg_scale):
    """The real `sensenova_pipeline_ops.encode_prompt`, uncond branch kept."""
    from core.models.sensenova import sensenova_pipeline_ops as pipeline_ops

    prefix = pipeline_ops.encode_prompt(
        transformer,
        tokenizer,
        prompt,
        height=height,
        width=width,
        cfg_scale=cfg_scale,
        batch_size=1,
    )
    if prefix.uncond_past_key_values is None:
        raise AssertionError("inference built no uncond branch; cfg_scale must be > 1")
    return prefix


def prefix_kv_parity(transformer, tokenizer, args) -> dict[str, Any]:
    """Arm 3: layer-by-layer prefix K/V, training null vs inference uncond."""
    from core.models.sensenova import sensenova_pipeline_ops as pipeline_ops
    from core.training.ops import sensenova_ops

    width, height = args.width, args.height
    inference_prefix = _inference_uncond_prefix(
        transformer, tokenizer, args.caption, width, height, args.cfg_scale
    )
    inference_layers = _copy_cache_layers(inference_prefix.uncond_past_key_values)
    inference_image_indexes = inference_prefix.uncond_indexes_image.detach().clone()
    inference_text_length = int(inference_image_indexes[0][0])
    pipeline_ops.clear_prefix_caches(inference_prefix)
    del inference_prefix
    torch.cuda.empty_cache()

    trainer = LoadedTrainer(transformer, tokenizer, args.device, torch.bfloat16)
    training_prefix = sensenova_ops.encode_prompt(trainer, args.caption, cfg_null=True)
    training_layers = _copy_cache_layers(training_prefix.cache)

    if len(training_layers) != len(inference_layers):
        raise AssertionError(
            f"layer count differs: training {len(training_layers)} vs inference "
            f"{len(inference_layers)}"
        )

    per_layer = []
    for index, ((t_k, t_v), (i_k, i_v)) in enumerate(
        zip(training_layers, inference_layers)
    ):
        entry = {
            "layer": index,
            "shape_equal": tuple(t_k.shape) == tuple(i_k.shape)
            and tuple(t_v.shape) == tuple(i_v.shape),
            "keys_identical": tuple(t_k.shape) == tuple(i_k.shape)
            and bool(torch.equal(t_k, i_k)),
            "values_identical": tuple(t_v.shape) == tuple(i_v.shape)
            and bool(torch.equal(t_v, i_v)),
        }
        if entry["shape_equal"] and not (entry["keys_identical"] and entry["values_identical"]):
            entry["max_abs_key_delta"] = float(
                (t_k.float() - i_k.float()).abs().max().cpu()
            )
            entry["max_abs_value_delta"] = float(
                (t_v.float() - i_v.float()).abs().max().cpu()
            )
        else:
            entry["training_shape"] = list(t_k.shape)
            entry["inference_shape"] = list(i_k.shape)
        per_layer.append(entry)

    training_image_indexes = transformer._build_t2i_image_indexes(
        height // 32, width // 32, training_prefix.text_length, device=args.device
    )
    identical = all(e["keys_identical"] and e["values_identical"] for e in per_layer)
    result = {
        "layers": len(per_layer),
        "prefix_tokens": int(training_layers[0][0].shape[-2]),
        "inference_prefix_tokens": int(inference_layers[0][0].shape[-2]),
        "training_text_length": int(training_prefix.text_length),
        "inference_text_length": inference_text_length,
        "kv_bit_identical": identical,
        "image_indexes_identical": bool(
            torch.equal(training_image_indexes, inference_image_indexes)
        ),
        "first_mismatch": next(
            (
                e
                for e in per_layer
                if not (e["keys_identical"] and e["values_identical"])
            ),
            None,
        ),
    }
    if not identical:
        result["per_layer"] = per_layer
    return result, training_prefix, training_layers, inference_layers


def _training_velocity(transformer, prefix_cache, text_length, fixed, args):
    """`sensenova_ops.train_step`'s forward, from a caller-fixed noised image.

    train_step draws its own noise; this repeats its arithmetic on a fixed
    ``z_image`` so both prefixes see byte-identical inputs. Everything after the
    noising is train_step's own call sequence.
    """
    from core.training.ops.sensenova_ops import forward_gen_decoder_layers

    z_image, t, token_h, token_w, merge_size = fixed
    from core.models.sensenova.sensenova_pipeline_ops import _build_step_context

    shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=token_h * merge_size,
        grid_w=token_w * merge_size,
        token_h=token_h,
        token_w=token_w,
    )
    noise_scale = fixed_noise_scale(transformer, shape)
    z, image_embeds, _ = _build_step_context(transformer, shape, z_image, t[0], noise_scale)
    indexes = transformer._build_t2i_image_indexes(
        token_h, token_w, text_length, device=args.device
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = forward_gen_decoder_layers(
            transformer.language_model.model,
            image_embeds,
            indexes=indexes,
            prefix_cache=prefix_cache,
            checkpoint_layers=False,
        )
        decoded = transformer.fm_modules["fm_head"](
            hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        patch = transformer.patch_size * merge_size
        x0_pred = (
            decoded.view(1, 3, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, token_h * token_w, patch * patch * 3)
        )
        denominator = (1 - t).view(1, 1, 1).clamp_min(transformer.config.t_eps)
        return ((x0_pred - z) / denominator).float()


def fixed_noise_scale(transformer, shape) -> float:
    from core.models.sensenova.sensenova_pipeline_ops import compute_noise_scale

    return compute_noise_scale(transformer, shape.grid_h, shape.grid_w, shape.merge_size)


def velocity_parity(transformer, tokenizer, args, training_layers, inference_layers):
    """Arm 4: one denoise step from each prefix, same image and timestep."""
    from core.training.ops.sensenova_ops import (
        _TrainingPrefixCache,
        _TrainingPrefixLayer,
    )

    merge_size = int(1 / transformer.downsample_ratio)
    token_h = args.height // (transformer.patch_size * merge_size)
    token_w = args.width // (transformer.patch_size * merge_size)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    x0 = (
        torch.rand(
            (1, 3, args.height, args.width),
            generator=generator,
            device=args.device,
            dtype=torch.bfloat16,
        )
        .mul_(2)
        .sub_(1)
    )
    eps = torch.randn(
        x0.shape, generator=generator, device=args.device, dtype=torch.bfloat16
    )
    t = torch.tensor([0.5], device=args.device, dtype=torch.float32)
    shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=token_h * merge_size,
        grid_w=token_w * merge_size,
        token_h=token_h,
        token_w=token_w,
    )
    noise_scale = fixed_noise_scale(transformer, shape)
    z_image = t.to(torch.bfloat16).view(1, 1, 1, 1) * x0 + (1 - t).to(
        torch.bfloat16
    ).view(1, 1, 1, 1) * (eps * noise_scale)
    fixed = (z_image, t, token_h, token_w, merge_size)

    def cache_from(layers):
        return _TrainingPrefixCache(
            [_TrainingPrefixLayer(keys, values) for keys, values in layers]
        )

    text_length = int(training_layers[0][0].shape[-2])
    inference_length = int(inference_layers[0][0].shape[-2])
    with torch.no_grad():
        v_training = _training_velocity(
            transformer, cache_from(training_layers), text_length, fixed, args
        )
        v_inference = _training_velocity(
            transformer, cache_from(inference_layers), inference_length, fixed, args
        )
    delta = (v_training - v_inference).abs()
    return {
        "shape": list(v_training.shape),
        "identical": bool(torch.equal(v_training, v_inference)),
        "max_abs_delta": float(delta.max().cpu()),
        "training_abs_mean": float(v_training.abs().mean().cpu()),
        "inference_abs_mean": float(v_inference.abs().mean().cpu()),
    }


def conditional_control(transformer, tokenizer, args, training_layers) -> dict[str, Any]:
    """Negative control: the same comparison against the CONDITIONAL prefix.

    Without it, "identical" above is only evidence that the two arms compared
    something; this shows the comparison can tell the two conditions apart.
    """
    from core.training.ops import sensenova_ops

    trainer = LoadedTrainer(transformer, tokenizer, args.device, torch.bfloat16)
    conditional = sensenova_ops.encode_prompt(trainer, args.caption)
    conditional_layers = _copy_cache_layers(conditional.cache)
    velocity = velocity_parity(
        transformer, tokenizer, args, training_layers, conditional_layers
    )
    return {
        "conditional_text_length": int(conditional.text_length),
        "null_text_length": int(training_layers[0][0].shape[-2]),
        "kv_shape_equal": tuple(conditional_layers[0][0].shape)
        == tuple(training_layers[0][0].shape),
        "velocity_identical": velocity["identical"],
        "velocity_max_abs_delta": velocity["max_abs_delta"],
    }


def run_weight_arms(args) -> dict[str, Any]:
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    if not torch.cuda.is_available():
        raise RuntimeError("the SenseNova K/V and velocity arms require CUDA")
    gate = _apply_vram_gate(args.vram_fraction)
    host_before = _host_rss()
    components = load_sensenova_from_path(args.model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    host_after_load = _host_rss()
    transformer.to(args.device)
    transformer.eval()
    # Both arms run under one backend so any drift is the prefix, not the kernel.
    layers = set_attention_backend(transformer, args.attention, AttentionMode.INFERENCE)
    resident = _cuda_memory()
    host_after_cuda = _host_rss()

    kv, training_prefix, training_layers, inference_layers = prefix_kv_parity(
        transformer, tokenizer, args
    )
    del training_prefix
    result: dict[str, Any] = {
        "arm": args.arm,
        "attention_backend": args.attention,
        "attention_layers": layers,
        "geometry": {"width": args.width, "height": args.height},
        "kv": kv,
        "vram_gate": gate,
    }
    if args.arm in ("velocity", "weights"):
        result["velocity"] = velocity_parity(
            transformer, tokenizer, args, training_layers, inference_layers
        )
        result["conditional_control"] = conditional_control(
            transformer, tokenizer, args, training_layers
        )
    result["memory"] = {
        "cuda_resident": resident,
        "cuda_peak_allocated": int(torch.cuda.max_memory_allocated()),
        "host_rss_before_load": host_before,
        "host_rss_after_load": host_after_load,
        "host_rss_after_cuda": host_after_cuda,
        "host_rss_peak_observed": max(host_before, host_after_load, host_after_cuda, _host_rss()),
    }
    return result


def run_weightless_arms(args) -> dict[str, Any]:
    model_dir = os.path.dirname(args.model_path)
    transformer = QueryBuilders(template=checkpoint_template(model_dir))
    tokenizer = load_tokenizer(model_dir)
    if args.arm == "tokens":
        return {"arm": "tokens", "tokens": token_parity(transformer, tokenizer)}
    return {"arm": "indexes", "indexes": index_parity(transformer, tokenizer)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("tokens", "indexes", "kv", "velocity", "weights"),
        required=True,
    )
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--caption", default="a photograph of a red bicycle")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--attention", default="native")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vram-fraction", type=float, default=VRAM_GATE_FRACTION)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    if args.arm in ("tokens", "indexes"):
        result = run_weightless_arms(args)
    else:
        result = run_weight_arms(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
