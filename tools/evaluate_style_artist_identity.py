"""Measure same-artist retrieval and pair discrimination on held-out real images."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import Siglip2ImageProcessor


WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "backend"))

from core.style_encoder.style_artifacts import load_style_encoder


def measure(embeddings: np.ndarray, rows: list[dict]) -> dict:
    artists = np.asarray([row["artist"] for row in rows])
    copyrights = [set(row["copyrights"]) for row in rows]
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    same = artists[:, None] == artists[None, :]
    np.fill_diagonal(same, False)
    eligible = same.any(axis=1)
    ranked = np.argsort(-sims, axis=1, kind="stable")
    first = same[np.arange(len(rows)), ranked[:, 0]]
    top_five = same[np.arange(len(rows))[:, None], ranked[:, :5]].any(axis=1)
    nearest_positive = np.max(np.where(same, sims, -np.inf), axis=1)
    nearest_negative = np.max(np.where(~same, sims, -np.inf), axis=1)

    upper = np.triu_indices(len(rows), 1)
    pair_same = same[upper]
    pair_scores = sims[upper]
    shared_copyright = np.asarray([
        bool(copyrights[i] & copyrights[j]) for i, j in zip(*upper)
    ])

    def pair_subset(select: np.ndarray) -> dict:
        labels = pair_same[select]
        scores = pair_scores[select]
        positives = scores[labels]
        negatives = scores[~labels]
        return {
            "same_artist_pairs": int(len(positives)),
            "different_artist_pairs": int(len(negatives)),
            "same_artist_mean_cosine": float(positives.mean()) if len(positives) else None,
            "different_artist_mean_cosine": float(negatives.mean()) if len(negatives) else None,
            "pair_auc": float(roc_auc_score(labels, scores)) if len(positives) and len(negatives) else None,
        }

    return {
        "held_out_images": len(rows),
        "held_out_artists": int(len(np.unique(artists))),
        "queries_with_another_same_artist_image": int(eligible.sum()),
        "top1_same_artist": float(first[eligible].mean()),
        "top5_contains_same_artist": float(top_five[eligible].mean()),
        "positive_beats_nearest_negative": float((nearest_positive[eligible] > nearest_negative[eligible]).mean()),
        "pairs_all": pair_subset(np.ones(len(pair_scores), dtype=bool)),
        "pairs_same_copyright": pair_subset(shared_copyright),
        "pairs_different_copyright": pair_subset(~shared_copyright),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=str(WORKTREE / "outputs/style_artist_identity_dataset39.json"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    manifest = WORKTREE / "outputs/sdxl_style_ip_dataset39/manifest.jsonl"
    rows = [row for row in map(json.loads, manifest.read_text(encoding="utf-8").splitlines())
            if row["split"] == 2]
    if len(rows) != 490:
        raise ValueError(f"Expected 490 held-out images, found {len(rows)}")
    output = {"dataset": 39, "split": "artists excluded from training", "models": {}}
    for name in ("baseline", "hsic"):
        model_path = WORKTREE / "outputs/style_hsic_ab_dataset39" / name / "best_style_encoder.safetensors"
        metadata = json.loads(model_path.with_name("best_metadata.json").read_text(encoding="utf-8"))
        processor = Siglip2ImageProcessor.from_dict(metadata["preprocessing"])
        model = load_style_encoder(str(model_path), "sdpa").to("cuda", dtype=torch.bfloat16).eval()
        vectors = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            images = []
            for row in batch:
                with Image.open(row["path"]) as source:
                    images.append(source.convert("RGB"))
            inputs = processor(images=images, max_num_patches=int(metadata["style_max_num_patches"]),
                               return_tensors="pt")
            with torch.inference_mode():
                vector = model.embed(inputs["pixel_values"].to("cuda", dtype=torch.bfloat16),
                                     inputs["pixel_attention_mask"].to("cuda"),
                                     inputs["spatial_shapes"].to("cuda")).float()
                vector = torch.nn.functional.normalize(vector, dim=1)
            vectors.append(vector.cpu().numpy())
            if (start // args.batch_size) % 20 == 0:
                print(json.dumps({"model": name, "embedded": min(start + args.batch_size, len(rows))}), flush=True)
        output["models"][name] = measure(np.concatenate(vectors), rows)
        print(json.dumps({"model": name, "result": output["models"][name]}), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
