"""Plot held-out artists from the completed style encoder without displaying source images."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from transformers import Siglip2ImageProcessor


WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "backend"))

from core.style_encoder.style_artifacts import load_style_encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=str(WORKTREE / "outputs/style_artist_map_dataset39.png"))
    args = parser.parse_args()
    if not 10 <= args.artists <= 20 or args.batch_size < 1:
        parser.error("artists must be 10..20 and batch-size must be positive")

    manifest = WORKTREE / "outputs/sdxl_style_ip_dataset39/manifest.jsonl"
    groups = defaultdict(list)
    for row in map(json.loads, manifest.read_text(encoding="utf-8").splitlines()):
        if row["split"] == 2:
            groups[row["artist"]].append(row)
    selected = sorted((rows for rows in groups.values() if len(rows) >= 3),
                      key=lambda rows: (-len(rows), rows[0]["artist"]))[:args.artists]
    if len(selected) < args.artists:
        raise ValueError(f"Only {len(selected)} held-out artists have at least three images")
    rows = [row for group in selected for row in group]
    labels = np.concatenate([np.full(len(group), i) for i, group in enumerate(selected)])

    model_path = WORKTREE / "outputs/style_hsic_ab_dataset39/baseline/best_style_encoder.safetensors"
    metadata = json.loads(model_path.with_name("best_metadata.json").read_text(encoding="utf-8"))
    processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
    model = load_style_encoder(str(model_path), "sdpa").to("cuda", dtype=torch.bfloat16).eval()
    vectors = []
    for start in range(0, len(rows), args.batch_size):
        images = []
        for row in rows[start:start + args.batch_size]:
            with Image.open(row["path"]) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, max_num_patches=int(metadata["style_max_num_patches"]),
                           return_tensors="pt")
        with torch.inference_mode():
            embedding = model.embed(inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
                                    inputs["pixel_attention_mask"].to("cuda"),
                                    inputs["spatial_shapes"].to("cuda")).float()
            embedding = torch.nn.functional.normalize(embedding, dim=1)
        vectors.append(embedding.cpu().numpy())
    points = TSNE(n_components=2, perplexity=10, metric="cosine", init="random",
                  learning_rate="auto", random_state=17).fit_transform(np.concatenate(vectors))

    fig, ax = plt.subplots(figsize=(10, 7), dpi=170)
    colors = plt.get_cmap("tab20").colors
    markers = ("o", "s", "^", "D", "P")
    for i, group in enumerate(selected):
        subset = points[labels == i]
        ax.scatter(subset[:, 0], subset[:, 1], s=58, c=[colors[i]], marker=markers[i % len(markers)],
                   alpha=0.86, edgecolors="white", linewidths=0.55,
                   label=f"{chr(65 + i)}  ({len(group)} images)")
    ax.set_title(f"Held-out artists · {len(selected)} artists, {len(rows)} images", fontsize=14, pad=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(color="#d9d9d9", linewidth=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(title="Artist", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, ncol=1, fontsize=9, title_fontsize=10)
    fig.text(0.09, 0.02, "Each point is one real image. Letters anonymize authors. 2D distances are approximate.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.045, 0.82, 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"output": str(output), "authors": len(selected), "images": len(rows),
                      "images_per_author": [len(group) for group in selected]}), flush=True)


if __name__ == "__main__":
    main()
