"""Phase U-0: does a differentiable SenseNova prefix pass actually reach und LoRA?

Three arms, each in its own process (``--arm``), all under the shared VRAM gate:

* ``parity``   -- no-grad K/V bitwise parity, training prefix loop vs vendor
                  ``_t2i_prefix_forward`` (checkpointed and not).
* ``grad``     -- 588 LoRA (gen 294 + und 294), one backward, gradient census;
                  plus the prefix-GC ON/OFF loss and gradient parity.
* ``gcoff``    -- per-layer census of the dequantized-weight materialization a
                  non-checkpointed prefix pass retains, aborting at a ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.probes.sensenova_real_checkpoint import (  # noqa: E402
    EXPECTED_LAYERS,
    EXPECTED_TARGETS,
    VRAM_GATE_FRACTION,
    _apply_vram_gate,
    _build_fixed_inputs,
    _cuda_memory,
    _gradient_stats,
    _repo_venv_python,
    _require_repo_venv,
)

# Understanding-branch LoRA targets. Unlike the generation branch these carry no
# suffix at all -- neither on the Linear (gen: ``q_proj_mot_gen``) nor on the
# parent (gen: ``mlp_mot_gen``). U-1 folds this into
# ``iter_sensenova_lora_targets(transformer, branch=...)``.
UND_ATTN_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj")
UND_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_UND_TARGETS = 294

# The last und layer's post-attention half is structurally unreachable from an
# image loss: its K/V are consumed by gen layer 41, but its attention output and
# MLP only produce hidden_42, which a t2i prefix discards (inference keeps
# ``past_key_values`` and drops ``last_hidden_state``).
def _expected_dead_und_targets(num_layers: int) -> set[str]:
    last = num_layers - 1
    prefix = f"language_model.model.layers.{last}"
    return {
        f"{prefix}.self_attn.q_proj",
        f"{prefix}.self_attn.o_proj",
        f"{prefix}.mlp.gate_proj",
        f"{prefix}.mlp.up_proj",
        f"{prefix}.mlp.down_proj",
    }


# Stop an arm well before the gate so an over-budget measurement aborts with a
# number instead of an OOM.
CEILING_FRACTION_OF_GATE = 0.60
# Headroom above the ceiling estimate for activations, gradients and allocator
# fragmentation when sizing the non-checkpointed prefix depth.
GC_OFF_SAFETY_BYTES = 1024 ** 3


def _iter_und_lora_targets(
    transformer: nn.Module,
) -> Generator[tuple[str, Any, str, nn.Module], None, None]:
    """Yield ``(module_path, parent, attr, module)`` per understanding target.

    Same 4-tuple contract as ``iter_sensenova_lora_targets`` so U-1 can adopt
    this as its ``branch="und"`` arm without changing any consumer.
    """
    from core.models.sensenova.sensenova_lora import _is_lora_target

    language_model = getattr(transformer, "language_model", None)
    llm_core = getattr(language_model, "model", None) if language_model is not None else None
    layers = getattr(llm_core, "layers", None)
    if layers is None:
        return
    for layer_idx, block in enumerate(layers):
        prefix = f"language_model.model.layers.{layer_idx}"
        attn = getattr(block, "self_attn", None)
        if attn is not None:
            for attr_name in UND_ATTN_ATTRS:
                module = getattr(attn, attr_name, None)
                if _is_lora_target(module):
                    yield f"{prefix}.self_attn.{attr_name}", attn, attr_name, module
        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            for attr_name in UND_MLP_ATTRS:
                module = getattr(mlp, attr_name, None)
                if _is_lora_target(module):
                    yield f"{prefix}.mlp.{attr_name}", mlp, attr_name, module


class _TrainingPrefixLayer:
    """A prefix cache layer built from checkpoint OUTPUTS, not cache writes."""

    flash_k_cache = None
    flash_v_cache = None
    flash_prefix_len = None

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _TrainingPrefixCache:
    _kv_cache_streamer = None
    _kv_cache_streamer_branch = None

    def __init__(self, layers: list[_TrainingPrefixLayer]):
        self.layers = layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return int(self.layers[layer_idx].keys.shape[-2])


def _assert_prefix_cache_structure(cache: Any, expected_layers: int) -> None:
    """``_assert_immutable_prefix_cache`` minus its ``requires_grad`` refusal.

    U-1 splits that function; U-0 must not touch it, so the structural half is
    restated here and the refusal is replaced by the positive assertion the
    caller makes instead (every K/V carries a ``grad_fn``).
    """
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) != expected_layers:
        raise AssertionError(f"prefix cache has {layers and len(layers)} layers")
    for name in ("_kv_cache_streamer", "_kv_cache_streamer_branch"):
        if getattr(cache, name, None) is not None:
            raise AssertionError("training prefix cache carries an inference KV streamer")
    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise AssertionError(f"prefix cache layer is missing non-empty {name}")
        for name in ("flash_k_cache", "flash_v_cache"):
            if getattr(layer, name, None) is not None:
                raise AssertionError("training prefix cache carries inference flash buffers")


def training_prefix_forward(
    model: Any,
    input_ids: torch.Tensor,
    indexes: torch.Tensor,
    attention_mask: dict,
    *,
    checkpoint_from: Optional[int] = 0,
    max_layers: Optional[int] = None,
    on_layer: Optional[Callable[[int], None]] = None,
) -> tuple[torch.Tensor, _TrainingPrefixCache]:
    """Run the understanding decoder stack and return ``(hidden, prefix_cache)``.

    ``checkpoint_from=0`` checkpoints every layer, ``None`` checkpoints none, and
    an integer N leaves layers ``[0, N)`` un-checkpointed (used to bound the
    non-checkpointed measurement).

    K/V leave each layer as explicit checkpoint OUTPUTS rather than through
    ``past_key_values.update()``: that write is a checkpoint-segment side effect,
    so a recompute would append a second time, and a side-effected tensor is not
    an output autograd can route a gradient through.
    """
    layers = list(model.layers[: model.config.num_hidden_layers])
    if max_layers is not None:
        layers = layers[:max_layers]
    for layer in layers:
        if layer.attention_type not in attention_mask:
            raise AssertionError(f"no mask for attention type {layer.attention_type!r}")
    # Vendor Qwen3Model.forward sets this on the pre-built-mask path.
    model.current_index = indexes[0].max()

    hidden_states = model.embed_tokens(input_ids)
    cache_layers: list[_TrainingPrefixLayer] = []
    for index, layer in enumerate(layers):
        mask = attention_mask[layer.attention_type]

        def layer_forward(states: torch.Tensor, _layer=layer, _mask=mask):
            # Base Module.__call__ skips Transformers' cache-dropping checkpoint
            # wrapper while keeping module hooks, as the gen loop does.
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=None,
                exist_non_image_gen_tokens=True,
                exist_image_gen_tokens=False,
                indexes=indexes,
                attention_mask=_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                return_kv=True,
            )

        if checkpoint_from is not None and index >= checkpoint_from:
            hidden_states, keys, values = checkpoint(
                layer_forward, hidden_states, use_reentrant=False
            )
        else:
            hidden_states, keys, values = layer_forward(hidden_states)
        cache_layers.append(_TrainingPrefixLayer(keys, values))
        if on_layer is not None:
            on_layer(index)
    return hidden_states, _TrainingPrefixCache(cache_layers)


def _forward_gen_layers(
    model: Any,
    hidden_states: torch.Tensor,
    *,
    indexes: torch.Tensor,
    prefix_cache: Any,
    checkpoint_layers: bool,
) -> torch.Tensor:
    """``ops.forward_gen_decoder_layers`` with the structural-only prefix check.

    A verbatim call would trip ``_assert_immutable_prefix_cache``'s
    ``requires_grad`` refusal, which U-0 is forbidden to modify.
    """
    layers = model.layers
    _assert_prefix_cache_structure(prefix_cache, len(layers))
    image_gen_indicators = torch.ones(
        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
    )
    for layer in layers:
        def layer_forward(states: torch.Tensor, _layer=layer) -> torch.Tensor:
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=indexes,
                attention_mask=None,
                past_key_values=prefix_cache,
                use_cache=False,
                update_cache=False,
            )

        if checkpoint_layers:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)
    return model.norm_mot_gen(hidden_states)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _load(model_path: str, seed: int):
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    components = load_sensenova_from_path(model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    transformer.requires_grad_(False)
    transformer.to("cuda")
    set_attention_backend(transformer, "native", AttentionMode.TRAINING)
    return transformer, components["tokenizer"]


def _text_inputs(transformer, tokenizer, caption: str):
    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN

    query = transformer._build_t2i_query(
        caption,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>",
    )
    return transformer._build_t2i_text_inputs(tokenizer, query)


def _attach_lora(transformer, *, gen: bool, und: bool):
    from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
    from core.training.adapters.sd15_adapter import LoRALinearLayer

    branches: dict[str, dict[str, LoRALinearLayer]] = {"gen": {}, "und": {}}
    sources = []
    if gen:
        sources.append(("gen", list(iter_sensenova_lora_targets(transformer))))
    if und:
        sources.append(("und", list(_iter_und_lora_targets(transformer))))
    for branch, targets in sources:
        if len(targets) != EXPECTED_TARGETS:
            raise AssertionError(f"{branch} branch yielded {len(targets)} targets")
        for module_path, parent, attr, current in targets:
            wrapper = LoRALinearLayer(
                current, rank=1, alpha=1, lora_name=module_path, lora_dtype=torch.float32
            )
            setattr(parent, attr, wrapper)
            branches[branch][module_path] = wrapper
    return branches


def _und_dequant_bytes(wrappers: dict[str, nn.Module]) -> int:
    """Bytes of bf16 weight ``Int8Linear._dequant_forward`` hands to autograd."""
    return sum(
        wrapper.weight.numel() * torch.finfo(torch.bfloat16).bits // 8
        for wrapper in wrappers.values()
    )


def _budget(vram_gate: dict[str, Any]) -> dict[str, Any]:
    budget = int(vram_gate["budget_bytes"])
    return {
        **vram_gate,
        "ceiling_bytes": int(budget * CEILING_FRACTION_OF_GATE),
        "ceiling_fraction_of_gate": CEILING_FRACTION_OF_GATE,
    }


# ---------------------------------------------------------------------------
# Arm 1 -- no-grad K/V parity
# ---------------------------------------------------------------------------


def _run_parity_arm(args: argparse.Namespace) -> dict[str, Any]:
    vram_gate = _budget(_apply_vram_gate())
    transformer, tokenizer = _load(args.model_path, args.seed)
    transformer.eval()
    model_resident = _cuda_memory()
    input_ids, indexes, mask = _text_inputs(transformer, tokenizer, args.caption)
    llm = transformer.language_model.model

    with torch.no_grad():
        vendor_cache, vendor_hidden = transformer._t2i_prefix_forward(input_ids, indexes, mask)
        vendor_kv = [
            (layer.keys.clone(), layer.values.clone()) for layer in vendor_cache.layers
        ]
    del vendor_cache
    torch.cuda.empty_cache()

    comparisons = {}
    arms = (
        ("checkpointed", 0, False),
        ("not_checkpointed", None, False),
        # U-1 must run the prefix under autocast: LoRALinearLayer keeps fp32
        # adapters and relies on autocast to meet the bf16 base activation.
        ("checkpointed_autocast", 0, True),
    )
    for label, checkpoint_from, autocast in arms:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=autocast
        ):
            hidden, cache = training_prefix_forward(
                llm, input_ids, indexes, mask, checkpoint_from=checkpoint_from
            )
            hidden = llm.norm(hidden)
        _assert_prefix_cache_structure(cache, EXPECTED_LAYERS)
        keys_equal = sum(
            int(torch.equal(cache.layers[i].keys, vendor_kv[i][0])) for i in range(EXPECTED_LAYERS)
        )
        values_equal = sum(
            int(torch.equal(cache.layers[i].values, vendor_kv[i][1])) for i in range(EXPECTED_LAYERS)
        )
        comparisons[label] = {
            "layers": EXPECTED_LAYERS,
            "keys_bitwise_equal": keys_equal,
            "values_bitwise_equal": values_equal,
            "all_bitwise_equal": keys_equal == EXPECTED_LAYERS
            and values_equal == EXPECTED_LAYERS,
            "last_hidden_state_bitwise_equal": bool(torch.equal(hidden, vendor_hidden)),
            "key_shape": list(cache.layers[0].keys.shape),
            "key_dtype": str(cache.layers[0].keys.dtype),
        }
        del hidden, cache
        torch.cuda.empty_cache()

    return {
        "arm": "parity",
        "caption_chars": len(args.caption),
        "prefix_tokens": int(input_ids.shape[1]),
        "attention_dropout": float(llm.config.attention_dropout),
        "comparisons": comparisons,
        "model_resident": model_resident,
        "peak": {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        },
        "vram_gate": vram_gate,
    }


# ---------------------------------------------------------------------------
# Arm 2 -- gradient propagation + prefix GC ON/OFF parity
# ---------------------------------------------------------------------------


def _loss_from_prefix(transformer, prefix_cache, fixed_inputs) -> torch.Tensor:
    import torch.nn.functional as F

    _x0, t, z, image_embeds, image_indexes, x0_tokens, token_h, token_w = fixed_inputs
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = _forward_gen_layers(
            transformer.language_model.model,
            image_embeds,
            indexes=image_indexes,
            prefix_cache=prefix_cache,
            checkpoint_layers=True,
        )
        decoded = transformer.fm_modules["fm_head"](
            hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2)
        )
        patch = transformer.patch_size * int(1 / transformer.downsample_ratio)
        x0_pred = (
            decoded.view(1, 3, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, token_h * token_w, patch * patch * 3)
        )
        denominator = (1 - t).clamp_min(transformer.config.t_eps)
        return F.mse_loss(
            ((x0_pred - z) / denominator).float(),
            ((x0_tokens - z) / denominator).float(),
        )


def _grad_run(
    transformer,
    branches,
    input_ids,
    indexes,
    mask,
    fixed_inputs,
    *,
    checkpoint_from: Optional[int],
) -> dict[str, Any]:
    for wrappers in branches.values():
        for wrapper in wrappers.values():
            wrapper.lora_down.weight.grad = None
            wrapper.lora_up.weight.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _hidden, prefix_cache = training_prefix_forward(
            transformer.language_model.model,
            input_ids,
            indexes,
            mask,
            checkpoint_from=checkpoint_from,
        )
    _assert_prefix_cache_structure(prefix_cache, EXPECTED_LAYERS)
    # The positive assertion §13.3 asks for: if the prefix were still built under
    # no_grad the loss would look perfectly healthy and und would never move.
    grad_fn_layers = sum(
        int(layer.keys.grad_fn is not None and layer.values.grad_fn is not None)
        for layer in prefix_cache.layers
    )
    prefix_built = _cuda_memory()

    loss = _loss_from_prefix(transformer, prefix_cache, fixed_inputs)
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    result = {
        "checkpoint_from": checkpoint_from,
        "prefix_layers_with_grad_fn": grad_fn_layers,
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "seconds": seconds,
        "prefix_built_memory": prefix_built,
        "peak": {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        },
    }
    for branch, wrappers in branches.items():
        result[f"{branch}_up_grad"] = _gradient_stats(wrappers, "lora_up")
        result[f"{branch}_down_grad"] = _gradient_stats(wrappers, "lora_down")
    result["und_no_grad_targets"] = sorted(
        name
        for name, wrapper in branches["und"].items()
        if wrapper.lora_up.weight.grad is None
        or not bool(torch.count_nonzero(wrapper.lora_up.weight.grad))
    )
    del prefix_cache, loss
    torch.cuda.empty_cache()
    return result


def _run_grad_arm(args: argparse.Namespace) -> dict[str, Any]:
    vram_gate = _budget(_apply_vram_gate())
    transformer, tokenizer = _load(args.model_path, args.seed)
    branches = _attach_lora(transformer, gen=True, und=True)
    transformer.train()
    model_resident = _cuda_memory()
    und_bytes = _und_dequant_bytes(branches["und"])

    input_ids, indexes, mask = _text_inputs(transformer, tokenizer, args.caption)
    fixed_inputs = _build_fixed_inputs(transformer, indexes, args.seed)

    per_layer = und_bytes // EXPECTED_LAYERS
    headroom = vram_gate["ceiling_bytes"] - model_resident["allocated"] - GC_OFF_SAFETY_BYTES
    gc_off_layers = max(0, min(EXPECTED_LAYERS, headroom // per_layer))

    runs = {
        "prefix_gc_on": _grad_run(
            transformer, branches, input_ids, indexes, mask, fixed_inputs,
            checkpoint_from=0,
        ),
        "prefix_gc_off_prefix_layers": _grad_run(
            transformer, branches, input_ids, indexes, mask, fixed_inputs,
            checkpoint_from=int(gc_off_layers),
        ),
    }
    left, right = runs["prefix_gc_on"], runs["prefix_gc_off_prefix_layers"]
    parity = {
        "uncheckpointed_prefix_layers": int(gc_off_layers),
        "loss_equal": left["loss"] == right["loss"],
        "loss_delta": right["loss"] - left["loss"],
    }
    for branch in ("gen", "und"):
        for direction in ("up", "down"):
            key = f"{branch}_{direction}_grad"
            parity[f"{key}_sha256_equal"] = left[key]["sha256"] == right[key]["sha256"]

    return {
        "arm": "grad",
        "targets": {
            "gen": len(branches["gen"]),
            "und": len(branches["und"]),
            "total": len(branches["gen"]) + len(branches["und"]),
        },
        "expected_dead_und_targets": sorted(_expected_dead_und_targets(EXPECTED_LAYERS)),
        "und_dequant_weight_bytes": und_bytes,
        "und_dequant_weight_bytes_per_layer": per_layer,
        "prefix_tokens": int(input_ids.shape[1]),
        "model_resident": model_resident,
        "runs": runs,
        "gc_parity": parity,
        "peak": {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        },
        "vram_gate": vram_gate,
    }


# ---------------------------------------------------------------------------
# Arm 3 -- non-checkpointed prefix materialization census
# ---------------------------------------------------------------------------


class _CeilingReached(RuntimeError):
    def __init__(self, layers: int):
        super().__init__(f"stopped after {layers} un-checkpointed prefix layers")
        self.layers = layers


def _run_gcoff_arm(args: argparse.Namespace) -> dict[str, Any]:
    vram_gate = _budget(_apply_vram_gate())
    transformer, tokenizer = _load(args.model_path, args.seed)
    branches = _attach_lora(transformer, gen=False, und=True)
    transformer.train()
    model_resident = _cuda_memory()
    und_bytes = _und_dequant_bytes(branches["und"])
    ceiling = vram_gate["ceiling_bytes"]

    input_ids, indexes, mask = _text_inputs(transformer, tokenizer, args.caption)
    samples: list[dict[str, int]] = []

    def on_layer(index: int) -> None:
        torch.cuda.synchronize()
        allocated = int(torch.cuda.memory_allocated())
        samples.append({
            "layer": index,
            "allocated": allocated,
            "delta_from_resident": allocated - model_resident["allocated"],
        })
        if allocated > ceiling:
            raise _CeilingReached(index + 1)

    stopped_at = None
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            training_prefix_forward(
                transformer.language_model.model, input_ids, indexes, mask,
                checkpoint_from=None, on_layer=on_layer,
            )
    except _CeilingReached as stop:
        stopped_at = stop.layers
    except torch.cuda.OutOfMemoryError as oom:
        return {
            "arm": "gcoff",
            "outcome": "oom",
            "detail": str(oom)[:400],
            "samples": samples,
            "vram_gate": vram_gate,
        }
    finally:
        torch.cuda.empty_cache()

    completed = len(samples)
    measured = samples[-1]["delta_from_resident"] if samples else 0
    per_layer = measured / completed if completed else 0.0
    return {
        "arm": "gcoff",
        "outcome": "ceiling" if stopped_at else "completed",
        "layers_run_uncheckpointed": completed,
        "measured_retained_bytes": measured,
        "measured_bytes_per_layer": per_layer,
        "extrapolated_full_depth_bytes": per_layer * EXPECTED_LAYERS,
        "analytic_und_dequant_bytes": und_bytes,
        "analytic_bytes_per_layer": und_bytes / EXPECTED_LAYERS,
        "prefix_tokens": int(input_ids.shape[1]),
        "model_resident": model_resident,
        "ceiling_bytes": ceiling,
        "samples": samples,
        "peak": {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        },
        "vram_gate": vram_gate,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_ARMS = {
    "parity": _run_parity_arm,
    "grad": _run_grad_arm,
    "gcoff": _run_gcoff_arm,
}


def _run_arm_subprocess(args: argparse.Namespace, arm: str, workdir: Path) -> dict[str, Any]:
    result_path = workdir / f"{arm}.json"
    cmd = [
        str(_repo_venv_python()),
        str(Path(__file__).resolve()),
        "--model-path", args.model_path,
        "--caption", args.caption,
        "--seed", str(args.seed),
        "--arm", arm,
        "--arm-json", str(result_path),
    ]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env, timeout=args.timeout_s, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"U-0 {arm} arm exited with code {completed.returncode}")
    if not result_path.is_file():
        raise RuntimeError(f"U-0 {arm} arm wrote no JSON result")
    with result_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--caption", default="a red cube on a white table")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--arm", choices=sorted(_ARMS), default=None)
    parser.add_argument("--arm-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--arms", default="parity,grad,gcoff")
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    if args.arm is not None and args.arm_json is None:
        parser.error("--arm requires --arm-json")
    return args


def main() -> int:
    _require_repo_venv()
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the SenseNova U-0 probe requires CUDA")
    if args.arm is not None:
        result = _ARMS[args.arm](args)
        path = Path(args.arm_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    requested = [name for name in args.arms.split(",") if name]
    with tempfile.TemporaryDirectory(prefix="sensenova_u0_") as raw:
        workdir = Path(raw)
        results = {arm: _run_arm_subprocess(args, arm, workdir) for arm in requested}
    payload = {
        "probe": "sensenova_phase_u0_und_prefix",
        "checkpoint": Path(args.model_path).name,
        "vram_gate_fraction": VRAM_GATE_FRACTION,
        "arms": results,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
