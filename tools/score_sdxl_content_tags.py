"""Measure target-content retention and reference-content leakage with a frozen tagger."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


WORKTREE = Path(__file__).resolve().parents[1]
BASE = WORKTREE / "outputs/sdxl_style_ip_dataset39"
sys.path.insert(0, str(WORKTREE / "backend"))

from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=("primary", "confirm"), default="confirm")
    parser.add_argument("--bridge-strength", type=float, default=0.05)
    parser.add_argument("--bridge-output-dir", default=str(BASE))
    parser.add_argument("--ip-layer-start", type=float, default=0.0)
    args = parser.parse_args()
    suffix = "_confirm" if args.cohort == "confirm" else ""
    label = f"strength_{args.bridge_strength:g}".replace(".", "p") + suffix
    if args.ip_layer_start:
        label += f"_late{round(100 * (1 - args.ip_layer_start))}"
    bridge_output = Path(args.bridge_output_dir).resolve()
    pairs = json.loads((BASE / f"reference_route_24pairs{suffix}.json").read_text(encoding="utf-8"))["pairs"]
    manifest = {row["item_key"]: row for row in map(
        json.loads, (BASE / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}

    checkpoint = WORKTREE.parents[2] / "tagger_models/cca72ce1-7420-4164-9f24-c30ae77cdf2f/best_f1.safetensors"
    vocabulary = checkpoint.with_name("best_f1_vocabulary.json")
    manager = SigLIP2InferenceManager()
    manager.load_model(str(checkpoint), str(vocabulary))
    tag_to_idx = manager.vocabulary["tag_to_idx"]
    tag_categories = manager.vocabulary["tag_to_category"]

    def content_tags(item: dict) -> set[str]:
        return {tag for tag in item["caption"].split(", ")
                if tag != "original" and tag in tag_to_idx
                and tag_categories.get(tag) in ("Character", "Copyright")}

    def probabilities(path: str | Path, indices: list[int]) -> list[float]:
        if not indices:
            return []
        with Image.open(path) as source:
            kwargs = {"images": source.convert("RGB"), "return_tensors": "pt"}
            if manager.is_naflex:
                kwargs["max_num_patches"] = 256
            inputs = manager.processor(**kwargs)
        pixel_values = inputs["pixel_values"].to("cuda")
        attention = inputs["pixel_attention_mask"].to("cuda") if manager.is_naflex else None
        shapes = inputs["spatial_shapes"].to("cuda") if manager.is_naflex else None
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = manager.model(pixel_values, attention, shapes)[0]
        return torch.sigmoid(logits.float()[indices]).cpu().tolist()

    rows = []
    for pair in pairs:
        n = pair["pair"]
        target = manifest[pair["target_item_key"]]
        reference = manifest[pair["reference_item_key"]]
        target_tags = sorted(content_tags(target))
        reference_only = sorted(content_tags(reference) - set(target_tags))
        indices = [tag_to_idx[tag] for tag in target_tags + reference_only]
        paths = {
            "bridge_baseline": bridge_output / "evaluation" / label / f"pair_{n:02d}_baseline.png",
            "bridge": bridge_output / "evaluation" / label / f"pair_{n:02d}_style.png",
            "reference_baseline": pair["baseline_path"],
            "reference_route": pair["reference_route_path"],
            "target_image": target["path"],
        }
        scores = {}
        for name, path in paths.items():
            probs = probabilities(path, indices)
            split = len(target_tags)
            scores[name] = {
                "target_content_probability": float(np.mean(probs[:split])) if split else None,
                "reference_only_probability": float(np.mean(probs[split:])) if reference_only else None,
            }
        rows.append({"pair": n, "shared_copyright": bool(set(target["copyrights"]) &
                                                       set(reference["copyrights"])),
                     "target_tags": target_tags, "reference_only_tags": reference_only,
                     "scores": scores})
        print(json.dumps({"pair": n, "target_tags": target_tags,
                          "reference_only_tags": reference_only}), flush=True)

    rng = np.random.default_rng(17)
    summary = {}
    for subset, selected in (("all", rows),
                             ("same_copyright", [row for row in rows if row["shared_copyright"]]),
                             ("different_copyright", [row for row in rows if not row["shared_copyright"]])):
        summary[subset] = {}
        for method, baseline in (("bridge", "bridge_baseline"), ("reference_route", "reference_baseline")):
            summary[subset][method] = {}
            for metric in ("target_content_probability", "reference_only_probability"):
                delta = np.asarray([row["scores"][method][metric] - row["scores"][baseline][metric]
                                    for row in selected if row["scores"][method][metric] is not None])
                summary[subset][method][metric + "_count"] = len(delta)
                summary[subset][method][metric + "_delta_mean"] = float(delta.mean()) if len(delta) else None
                if len(delta):
                    resampled = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
                    summary[subset][method][metric + "_bootstrap_95pct"] = [
                        float(value) for value in np.quantile(resampled, [0.025, 0.975])]
    output = bridge_output / f"content_tags_24pairs_{label}.json"
    output.write_text(json.dumps({"pairs": rows, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
