"""Score bridge and legacy SDXL reference generations with independent CLIP and style embeddings."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, Siglip2ImageProcessor


WORKTREE = Path(__file__).resolve().parents[1]
OUTPUT = WORKTREE / "outputs/sdxl_style_ip_dataset39"
sys.path.insert(0, str(WORKTREE / "backend"))

from core.style_encoder.style_artifacts import load_style_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--bridge-strength", type=float, default=0.1)
    parser.add_argument("--cohort", choices=("primary", "confirm"), default="primary")
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--ip-layer-start", type=float, default=0.0)
    parser.add_argument("--bridge-output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    if not 0 <= args.ip_layer_start < 1:
        parser.error("ip-layer-start must be in [0, 1)")
    label = f"strength_{args.bridge_strength:g}".replace(".", "p")
    suffix = "_confirm" if args.cohort == "confirm" else ""
    label += suffix
    if args.ip_layer_start:
        label += f"_late{round(100 * (1 - args.ip_layer_start))}"
    records = json.loads((OUTPUT / f"reference_route_{args.pairs}pairs{suffix}.json").read_text(encoding="utf-8"))["pairs"]
    bridge_output = Path(args.bridge_output_dir).resolve()
    shuffled = None
    if args.shuffled:
        shuffled = {row["pair"]: row for row in json.loads((bridge_output / f"evaluation_{label}_shuffled.json").read_text(
            encoding="utf-8"))["pairs"]}
    manifest = {row["item_key"]: row for row in map(
        json.loads, (OUTPUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    encoder_path = WORKTREE / "outputs/style_hsic_ab_dataset39/baseline/best_style_encoder.safetensors"
    metadata = json.loads(encoder_path.with_name("best_metadata.json").read_text(encoding="utf-8"))
    style_processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
    style_model = load_style_encoder(str(encoder_path), "sdpa").to("cuda", dtype=torch.bfloat16).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True).to(
        "cuda", dtype=torch.bfloat16).eval()

    def style_embed(path: str | Path) -> torch.Tensor:
        with Image.open(path) as source:
            data = style_processor(images=[source.convert("RGB")], max_num_patches=256, return_tensors="pt")
        with torch.inference_mode():
            return style_model.embed(data["pixel_values"].to("cuda", dtype=torch.bfloat16),
                                     data["pixel_attention_mask"].to("cuda"),
                                     data["spatial_shapes"].to("cuda")).float()[0]

    def clip_image(path: str | Path) -> torch.Tensor:
        with Image.open(path) as source:
            data = clip_processor(images=[source.convert("RGB")], return_tensors="pt")
        with torch.inference_mode():
            value = clip_model.get_image_features(pixel_values=data["pixel_values"].to(
                "cuda", dtype=torch.bfloat16)).pooler_output.float()[0]
        return torch.nn.functional.normalize(value, dim=0)

    def clip_text(prompt: str) -> torch.Tensor:
        data = clip_processor(text=[prompt], padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            value = clip_model.get_text_features(input_ids=data["input_ids"].to("cuda"),
                                                 attention_mask=data["attention_mask"].to("cuda")).pooler_output.float()[0]
        return torch.nn.functional.normalize(value, dim=0)

    results = []
    for record in records:
        n = record["pair"]
        target = manifest[record["target_item_key"]]
        reference = manifest[record["reference_item_key"]]
        reference_style = style_embed(reference["path"])
        shuffled_style = style_embed(manifest[shuffled[n]["reference_item_key"]]["path"]) if shuffled else None
        target_clip = clip_image(target["path"])
        prompt_clip = clip_text(target["caption"])
        paths = {
            "bridge_baseline": bridge_output / "evaluation" / label / f"pair_{n:02d}_baseline.png",
            "bridge": bridge_output / "evaluation" / label / f"pair_{n:02d}_style.png",
            "reference_baseline": Path(record["baseline_path"]),
            "reference_route": Path(record["reference_route_path"]),
        }
        if shuffled:
            paths["bridge_shuffled"] = bridge_output / "evaluation" / (label + "_shuffled") / f"pair_{n:02d}_style.png"
        scores = {}
        for name, path in paths.items():
            image_style = style_embed(path)
            image_clip = clip_image(path)
            scores[name] = {"style_cosine": float(reference_style @ image_style),
                            "prompt_clip_cosine": float(prompt_clip @ image_clip),
                            "target_clip_cosine": float(target_clip @ image_clip)}
            if shuffled_style is not None:
                scores[name]["shuffled_reference_style_cosine"] = float(shuffled_style @ image_style)
        shared_copyright = bool(set(target["copyrights"]) & set(reference["copyrights"]))
        results.append({"pair": n, "shared_copyright": shared_copyright, "scores": scores})
        print(json.dumps({"pair": n, "shared_copyright": shared_copyright,
                          "bridge_style_delta": scores["bridge"]["style_cosine"] - scores["bridge_baseline"]["style_cosine"],
                          "reference_style_delta": scores["reference_route"]["style_cosine"] - scores["reference_baseline"]["style_cosine"]}), flush=True)

    summary = {}
    bootstrap = np.random.default_rng(17)
    for subset, rows in (("all", results),
                         ("same_copyright", [row for row in results if row["shared_copyright"]]),
                         ("different_copyright", [row for row in results if not row["shared_copyright"]])):
        summary[subset] = {"count": len(rows)}
        for method, baseline in (("bridge", "bridge_baseline"), ("reference_route", "reference_baseline")):
            summary[subset][method] = {}
            for metric in ("style_cosine", "prompt_clip_cosine", "target_clip_cosine"):
                delta = np.asarray([row["scores"][method][metric] - row["scores"][baseline][metric]
                                    for row in rows])
                name = metric + "_delta"
                summary[subset][method][name + "_mean"] = float(delta.mean()) if len(delta) else None
                summary[subset][method][name + "_positive"] = int((delta > 0).sum())
                if len(delta):
                    resampled = delta[bootstrap.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
                    summary[subset][method][name + "_bootstrap_95pct"] = [
                        float(value) for value in np.quantile(resampled, [0.025, 0.975])]
        summary[subset]["bridge_minus_reference_route"] = {}
        for metric in ("style_cosine", "prompt_clip_cosine", "target_clip_cosine"):
            delta = np.asarray([
                (row["scores"]["bridge"][metric] - row["scores"]["bridge_baseline"][metric]) -
                (row["scores"]["reference_route"][metric] - row["scores"]["reference_baseline"][metric])
                for row in rows])
            key = metric + "_delta"
            summary[subset]["bridge_minus_reference_route"][key + "_mean"] = float(delta.mean()) if len(delta) else None
            if len(delta):
                resampled = delta[bootstrap.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
                summary[subset]["bridge_minus_reference_route"][key + "_bootstrap_95pct"] = [
                    float(value) for value in np.quantile(resampled, [0.025, 0.975])]
        if shuffled:
            summary[subset]["reference_sensitivity"] = {}
            for metric in ("style_cosine", "shuffled_reference_style_cosine", "prompt_clip_cosine"):
                delta = np.asarray([row["scores"]["bridge"][metric] - row["scores"]["bridge_shuffled"][metric]
                                    for row in rows])
                summary[subset]["reference_sensitivity"][metric + "_correct_minus_shuffled_mean"] = (
                    float(delta.mean()) if len(delta) else None)
                if len(delta):
                    resampled = delta[bootstrap.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
                    summary[subset]["reference_sensitivity"][metric + "_bootstrap_95pct"] = [
                        float(value) for value in np.quantile(resampled, [0.025, 0.975])]
    out_label = label + ("_shuffled" if shuffled else "")
    (bridge_output / f"style_comparison_{args.pairs}pairs_{out_label}.json").write_text(json.dumps({
        "metrics": "style encoder cosine; independent CLIP ViT-L prompt and target-image cosine",
        "pairs": results, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
