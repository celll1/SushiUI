"""Does the SenseNova understanding branch keep "similar but different" prompts apart?

This is a training-scope question. The generation branch never sees the caption:
it only reads the understanding branch's per-layer prefix K/V (42 layers,
`[B, H_kv, S_text, D]`, layer-matched, attended in full and non-causally by every
image token). If a one-tag edit barely moves that K/V, the und branch is the
bottleneck and has to be trained; if it moves it far, the und branch already
carries the detail and a rendering failure lives on the generation side.

Arms, each in its own process (``--arm``):

* ``kv``      per-layer K/V distance between the two captions of each pair,
              read against three references from the caption set: an identical
              pair (must be exactly 0), a reorder pair (same tag SET, different
              order -- the semantic floor) and an unrelated pair (the ceiling).
* ``readout`` the same pairs pushed through the generation decoder with one
              fixed noise sample, measuring how far the predicted x0 moves.
              This is what the image tokens actually read.
* ``qa``      the und branch answering a question about the tag list through its
              own LM head. Interpretable evidence rather than a distance.

Two confounds are recorded rather than assumed away. Image tokens sit at
``t = text_length`` (`_build_t2i_image_indexes`), so a pair whose captions
tokenize to different lengths also moves every image token's RoPE position; the
per-pair token lengths are reported and the aligned metrics are only computed
when the lengths match. And the shipped base stores ``k_proj``/``v_proj`` in
int8 (weight-only, dequantized to bf16 at matmul), so the measured K/V carry
that quantization -- which is the production condition, not a probe artifact.

Host RAM: the int8 base is an 18 GB safetensors read into a ~17.6 GiB resident
tree; expect a ~25 GiB host peak while loading. Run ONE arm at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training.probes.sensenova_real_checkpoint import (  # noqa: E402
    EXPECTED_LAYERS,
    VRAM_GATE_FRACTION,
    _apply_vram_gate,
    _build_training_prefix,
    _cuda_memory,
    _repo_venv_python,
    _require_repo_venv,
)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _seq_axis(tensor: torch.Tensor, seq_len: int) -> int:
    """Locate the sequence axis of a K/V tensor by its length.

    Asserted rather than assumed: exactly one axis may match, or the caller is
    measuring the wrong dimension.
    """
    matches = [axis for axis, size in enumerate(tensor.shape) if size == seq_len]
    if len(matches) != 1:
        raise AssertionError(
            f"cannot identify the sequence axis of a {tuple(tensor.shape)} K/V "
            f"tensor at seq_len {seq_len}"
        )
    return matches[0]


def _relative_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / a.norm().clamp_min(1e-12))


def _per_position_relative(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
    """``||a_p - b_p|| / ||a_p||`` for every sequence position p."""
    flat_a = a.movedim(axis, 0).reshape(a.shape[axis], -1)
    flat_b = b.movedim(axis, 0).reshape(b.shape[axis], -1)
    return (flat_a - flat_b).norm(dim=1) / flat_a.norm(dim=1).clamp_min(1e-12)


def _pooled_cosine_distance(a: torch.Tensor, b: torch.Tensor, axis: int) -> float:
    """``1 - cos`` between the position-averaged tensors.

    Order-free, so it stays defined when the two captions tokenize to different
    lengths (an added or removed tag), where position alignment does not exist.
    """
    pooled_a = a.mean(dim=axis).flatten()
    pooled_b = b.mean(dim=axis).flatten()
    return float(1.0 - torch.nn.functional.cosine_similarity(pooled_a, pooled_b, dim=0))


def _compare_caches(cache_a: Any, cache_b: Any, first_divergence: Optional[int]) -> dict:
    """Per-layer K/V comparison of two prefix caches."""
    layers_a, layers_b = cache_a.layers, cache_b.layers
    if len(layers_a) != len(layers_b):
        raise AssertionError("prefix caches have different layer counts")
    len_a = int(cache_a.get_seq_length())
    len_b = int(cache_b.get_seq_length())
    aligned = len_a == len_b

    per_layer: list[dict] = []
    for index, (layer_a, layer_b) in enumerate(zip(layers_a, layers_b)):
        entry: dict[str, Any] = {"layer": index}
        for name in ("keys", "values"):
            a = getattr(layer_a, name).detach().float()
            b = getattr(layer_b, name).detach().float()
            axis = _seq_axis(a, len_a)
            entry[f"{name}_pooled_cosine_distance"] = _pooled_cosine_distance(a, b, axis)
            if aligned:
                entry[f"{name}_relative_frobenius"] = _relative_frobenius(a, b)
                profile = _per_position_relative(a, b, axis)
                entry[f"{name}_max_position_relative"] = float(profile.max())
                if first_divergence is not None and 0 < first_divergence < len_a:
                    # Causality check: a decoder-only prefix cannot change a
                    # position that precedes the edit. A non-zero number here
                    # means the alignment, not the model, is wrong.
                    entry[f"{name}_pre_edit_max_relative"] = float(
                        profile[:first_divergence].max()
                    )
                    entry[f"{name}_at_edit_relative"] = float(profile[first_divergence])
                    entry[f"{name}_post_edit_mean_relative"] = float(
                        profile[first_divergence:].mean()
                    )
        per_layer.append(entry)

    summary: dict[str, Any] = {
        "seq_length_a": len_a,
        "seq_length_b": len_b,
        "position_aligned": aligned,
        "layers": len(layers_a),
    }
    for key in per_layer[0]:
        if key == "layer":
            continue
        values = [entry[key] for entry in per_layer]
        summary[f"mean_{key}"] = sum(values) / len(values)
        summary[f"last_layer_{key}"] = per_layer[-1][key]
    return {"summary": summary, "per_layer": per_layer}


def _first_divergence(ids_a: torch.Tensor, ids_b: torch.Tensor) -> Optional[int]:
    a = ids_a[0].tolist()
    b = ids_b[0].tolist()
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return None


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def _load(args: argparse.Namespace):
    from core.attention import AttentionMode
    from core.models.sensenova.loader import load_sensenova_from_path
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    components = load_sensenova_from_path(args.model_path, torch_dtype=torch.bfloat16)
    transformer = components["transformer"]
    tokenizer = components["tokenizer"]
    transformer.to("cuda")
    transformer.eval()
    layers = set_attention_backend(transformer, "native", AttentionMode.TRAINING)
    if layers != EXPECTED_LAYERS:
        raise AssertionError(f"expected {EXPECTED_LAYERS} attention layers, got {layers}")
    return transformer, tokenizer


def _load_pairs(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not items:
        raise SystemExit(f"{path} holds no caption pairs")
    return items


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def _arm_kv(args: argparse.Namespace) -> dict:
    transformer, tokenizer = _load(args)
    resident = _cuda_memory()
    pairs = _load_pairs(Path(args.caption_set))
    results = []
    for index, pair in enumerate(pairs):
        ids_a, _, cache_a = _build_training_prefix(transformer, tokenizer, pair["a"])
        ids_b, _, cache_b = _build_training_prefix(transformer, tokenizer, pair["b"])
        divergence = _first_divergence(ids_a, ids_b)
        comparison = _compare_caches(cache_a, cache_b, divergence)
        del cache_a, cache_b
        torch.cuda.empty_cache()
        entry = {
            "index": index,
            "kind": pair["kind"],
            "axis": pair.get("axis"),
            "changed": pair.get("changed"),
            "n_tags_a": pair.get("n_tags_a"),
            "token_length_a": int(ids_a.shape[1]),
            "token_length_b": int(ids_b.shape[1]),
            "first_divergent_token_index": divergence,
            **comparison,
        }
        results.append(entry)
        summary = comparison["summary"]
        relative = summary.get("mean_keys_relative_frobenius")
        print(
            f"[{index:2d}] {pair['kind']:<13} {str(pair.get('axis') or ''):<14} "
            f"len {entry['token_length_a']:>4}/{entry['token_length_b']:<4} "
            f"K relF {'   n/a ' if relative is None else format(relative, '.5f')} "
            f"K cosd {summary['mean_keys_pooled_cosine_distance']:.6f} "
            f"V cosd {summary['mean_values_pooled_cosine_distance']:.6f}",
            flush=True,
        )
    return {"arm": "kv", "model_resident": resident, "pairs": results}


def _arm_readout(args: argparse.Namespace) -> dict:
    """How far does the generation branch's x0 prediction move per pair?

    One noise sample, one timestep, one resolution, shared across every pair, so
    the only thing that varies between the two runs of a pair is the prompt.
    """
    from types import SimpleNamespace

    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context,
        compute_noise_scale,
    )
    from core.training.ops.sensenova_ops import forward_gen_decoder_layers

    transformer, tokenizer = _load(args)
    resident = _cuda_memory()
    pairs = _load_pairs(Path(args.caption_set))
    device = torch.device("cuda")
    size = int(args.readout_resolution)
    merge_size = int(1 / transformer.downsample_ratio)
    grid_h = grid_w = size // transformer.patch_size
    token_h, token_w = grid_h // merge_size, grid_w // merge_size

    generator = torch.Generator(device=device).manual_seed(args.seed)
    x0 = (
        torch.rand(
            (1, 3, size, size), generator=generator, device=device, dtype=torch.bfloat16
        )
        .mul_(2)
        .sub_(1)
    )
    eps = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.bfloat16)
    t = torch.tensor(float(args.readout_timestep), device=device, dtype=torch.bfloat16)
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    x_t = t * x0 + (1 - t) * noise_scale * eps
    prefix_shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=grid_h,
        grid_w=grid_w,
        token_h=token_h,
        token_w=token_w,
    )

    def predict(caption: str) -> torch.Tensor:
        _, prefix_indexes, cache = _build_training_prefix(transformer, tokenizer, caption)
        z, image_embeds, _ = _build_step_context(
            transformer, prefix_shape, x_t, t, noise_scale
        )
        image_indexes = transformer._build_t2i_image_indexes(
            token_h, token_w, prefix_indexes.shape[1], device=prefix_indexes.device
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = forward_gen_decoder_layers(
                transformer.language_model.model,
                image_embeds,
                indexes=image_indexes,
                prefix_cache=cache,
                attention_mask=None,
                checkpoint_layers=False,
            )
            image_2d = hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2)
            decoded = transformer.fm_modules["fm_head"](image_2d)
        del cache
        torch.cuda.empty_cache()
        return decoded.detach().float()

    results = []
    for index, pair in enumerate(pairs):
        pred_a = predict(pair["a"])
        pred_b = predict(pair["b"])
        relative = _relative_frobenius(pred_a, pred_b)
        cosine = float(
            1.0
            - torch.nn.functional.cosine_similarity(
                pred_a.flatten(), pred_b.flatten(), dim=0
            )
        )
        results.append(
            {
                "index": index,
                "kind": pair["kind"],
                "axis": pair.get("axis"),
                "changed": pair.get("changed"),
                "n_tags_a": pair.get("n_tags_a"),
                "x0_relative_frobenius": relative,
                "x0_cosine_distance": cosine,
            }
        )
        print(
            f"[{index:2d}] {pair['kind']:<13} {str(pair.get('axis') or ''):<14} "
            f"x0 relF {relative:.5f}  cosd {cosine:.6f}",
            flush=True,
        )
    return {
        "arm": "readout",
        "model_resident": resident,
        "resolution": size,
        "timestep": float(args.readout_timestep),
        "pairs": results,
    }


QA_QUESTIONS = (
    "What is the hair colour described by these tags? Answer with one word.",
    "What is the eye colour described by these tags? Answer with one word.",
    "Is the hair long or short according to these tags? Answer with one word.",
    "How many girls are in the scene according to these tags? Answer with a number.",
)


def _arm_qa(args: argparse.Namespace) -> dict:
    """Ask the und branch, through its own LM head, about the tag list.

    A distance says the representation moved; this says the move carried the
    right meaning. Text-only, so ``chat()`` (which indexes ``grid_hw``
    unconditionally) is not usable and the query is built directly.
    """
    from transformers import GenerationConfig

    from core.models.sensenova.vendor.conversation import get_conv_template

    transformer, tokenizer = _load(args)
    resident = _cuda_memory()
    pairs = _load_pairs(Path(args.caption_set))
    transformer.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    transformer.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")

    def ask(caption: str, question: str) -> str:
        template = get_conv_template(transformer.template)
        template.system_message = transformer.system_message
        template.append_message(template.roles[0], f"Tags: {caption}\n\n{question}")
        template.append_message(template.roles[1], None)
        query = template.get_prompt() + "<think>\n\n</think>\n\n"
        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(transformer.device)
        attention_mask = model_inputs["attention_mask"].to(transformer.device)
        eos = tokenizer.convert_tokens_to_ids(template.sep.strip())
        output = transformer.generate(
            pixel_values=None,
            input_ids=input_ids,
            grid_hw=None,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=int(args.qa_max_new_tokens),
                do_sample=False,
                eos_token_id=eos,
            ),
        )
        text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return text.split(template.sep.strip())[0].strip()

    results = []
    for index, pair in enumerate(pairs):
        if pair["kind"] not in ("minimal_edit", "identical"):
            continue
        answers = {
            side: {question: ask(pair[side], question) for question in QA_QUESTIONS}
            for side in ("a", "b")
        }
        results.append(
            {
                "index": index,
                "kind": pair["kind"],
                "axis": pair.get("axis"),
                "changed": pair.get("changed"),
                "answers": answers,
            }
        )
        print(f"[{index:2d}] {pair['kind']} {pair.get('axis')} changed={pair.get('changed')}")
        for question in QA_QUESTIONS:
            print(f"     Q {question}")
            print(f"       a: {answers['a'][question]!r}")
            print(f"       b: {answers['b'][question]!r}")
        sys.stdout.flush()
    return {"arm": "qa", "model_resident": resident, "pairs": results}


ARMS = {"kv": _arm_kv, "readout": _arm_readout, "qa": _arm_qa}


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="SenseNova und-branch discrimination probe")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--caption-set", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    parser.add_argument("--arms", default="kv,readout,qa")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--readout-resolution", type=int, default=512)
    parser.add_argument("--readout-timestep", type=float, default=0.5)
    parser.add_argument("--qa-max-new-tokens", type=int, default=24)
    parser.add_argument("--vram-fraction", type=float, default=VRAM_GATE_FRACTION)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    _require_repo_venv()

    if args.arm is None:
        # Parent: one child process per arm so no arm inherits another's
        # allocator state or a second copy of the tree.
        outputs: dict[str, Any] = {}
        base = Path(args.json_out) if args.json_out else Path("und_discrimination.json")
        for arm in [item.strip() for item in args.arms.split(",") if item.strip()]:
            out = base.with_name(f"{base.stem}_{arm}.json")
            command = [
                str(_repo_venv_python()),
                str(Path(__file__).resolve()),
                "--model-path", args.model_path,
                "--caption-set", args.caption_set,
                "--arm", arm,
                "--seed", str(args.seed),
                "--readout-resolution", str(args.readout_resolution),
                "--readout-timestep", str(args.readout_timestep),
                "--qa-max-new-tokens", str(args.qa_max_new_tokens),
                "--vram-fraction", str(args.vram_fraction),
                "--json-out", str(out),
            ]
            print(f"\n=== arm {arm} ===", flush=True)
            completed = subprocess.run(command, cwd=str(REPO_ROOT))
            if completed.returncode != 0:
                raise SystemExit(f"arm {arm} failed with exit code {completed.returncode}")
            outputs[arm] = str(out)
        print(json.dumps({"arms": outputs}, indent=2))
        return 0

    gate = _apply_vram_gate(args.vram_fraction)
    started = time.perf_counter()
    result = ARMS[args.arm](args)
    result["vram_gate"] = gate
    result["wall_seconds"] = time.perf_counter() - started
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
