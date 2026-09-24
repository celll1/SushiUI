"""Compare SDXL style-bridge and text-only generations for held-out dataset-39 artists."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Siglip2ImageProcessor

WORKTREE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE / "backend"))

from core.inference.attention_processors import set_attention_processor, set_style_ip_context
from core.model_loader import ModelLoader
from core.style_encoder.sdxl_ip_adapter import load_sdxl_style_ip_adapter
from core.style_encoder.sdxl_ip_runtime import load_sdxl_style_pipeline
from core.style_encoder.style_artifacts import load_style_encoder
from train_sdxl_style_ip import digest


def embed_image(image: Image.Image, model, processor, budget: int) -> torch.Tensor:
    inputs = processor(images=[image], max_num_patches=budget, return_tensors="pt")
    with torch.no_grad():
        return model.embed(inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
                           inputs["pixel_attention_mask"].to("cuda"),
                           inputs["spatial_shapes"].to("cuda")).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    output = WORKTREE / "outputs/sdxl_style_ip_dataset39"
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    by_artist = defaultdict(list)
    for item in manifest:
        if item["split"] == 2:
            by_artist[item["artist"]].append(item)
    artists = [items for items in by_artist.values() if len(items) >= 2]
    rng = np.random.default_rng(17)
    selected = [artists[int(i)] for i in rng.choice(len(artists), size=min(args.pairs, len(artists)), replace=False)]

    model_path = REPO / "models/NoobAI-XL-Vpred-v1.0.safetensors"
    encoder_path = WORKTREE / "outputs/style_hsic_ab_dataset39/baseline/best_style_encoder.safetensors"
    with encoder_path.with_name("best_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    pipe = load_sdxl_style_pipeline(str(model_path))
    ModelLoader._configure_v_prediction_scheduler(pipe)
    pipe.to("cuda")
    set_attention_processor(pipe.unet)
    adapter = load_sdxl_style_ip_adapter(
        pipe.unet, str(output / "latest_style_ip.safetensors"),
        style_encoder_sha256=digest(encoder_path), sdxl_model_sha256=digest(model_path),
    ).to("cuda", dtype=torch.float32).eval()
    style_model = load_style_encoder(str(encoder_path), "sdpa").to("cuda", dtype=torch.bfloat16).eval()
    processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
    label = f"strength_{args.strength:g}".replace(".", "p")
    image_dir = output / "evaluation" / label
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, items in enumerate(selected):
        target, reference = items[:2]
        with Image.open(reference["path"]) as source:
            reference_image = source.convert("RGB")
        reference_style = embed_image(reference_image, style_model, processor, 256)
        seed = 94117 + n

        def generate(ctx):
            set_style_ip_context(pipe.unet, ctx)
            try:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode():
                    return pipe(
                        prompt=target["caption"], negative_prompt="", width=512, height=512,
                        num_inference_steps=args.steps, guidance_scale=5.0, guidance_rescale=0.7,
                        generator=generator,
                    ).images[0]
            finally:
                set_style_ip_context(pipe.unet, None)

        baseline = generate(None)
        if n == 0:
            zero = generate(adapter.context(reference_style, scale=0, cfg_batch=True))
            if not np.array_equal(np.asarray(baseline), np.asarray(zero)):
                raise AssertionError("style strength 0 changed the SDXL generation")
        styled = generate(adapter.context(reference_style, scale=args.strength, cfg_batch=True))
        baseline.save(image_dir / f"pair_{n:02d}_baseline.png")
        styled.save(image_dir / f"pair_{n:02d}_style.png")
        baseline_style = embed_image(baseline, style_model, processor, 256)
        styled_style = embed_image(styled, style_model, processor, 256)
        score_base = float((baseline_style * reference_style).sum())
        score_style = float((styled_style * reference_style).sum())
        row = {"pair": n, "reference_item_key": reference["item_key"],
               "target_item_key": target["item_key"], "baseline_style_similarity": score_base,
               "bridge_style_similarity": score_style, "delta": score_style - score_base}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / f"evaluation_{label}.json").write_text(json.dumps({"pairs": rows, "mean_delta": float(np.mean(
        [row["delta"] for row in rows]))}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
