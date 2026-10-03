"""Render anonymous General-only panels without touching training RNG or CUDA."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def probability_groups(probs, positive, known, threshold):
    predicted = probs >= threshold
    return {
        "TP": positive & known & predicted,
        "FP": ~positive & known & predicted,
        "FN": positive & known & ~predicted,
        "TN": ~positive & known & ~predicted,
    }


def render(job):
    import torch
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    torch.cuda._lazy_init = lambda *a, **k: None
    torch._C._cuda_init = lambda *a, **k: None
    torch.set_num_threads(2)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from core.tagger.tag_parsing import resolve_caption_tags
    from core.tagger.foveatag.artifact import FoveaArtifactConfig
    from core.tagger.foveatag.model import FoveaTagger
    from core.tagger.foveatag.batch import collate_view_batches
    from core.tagger.foveatag.supervised_data import build_supervised_views
    from PIL import Image
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    request = json.loads((job / "request.json").read_text())
    comparison = Path(request["comparison"])
    args = request["arguments"]
    payload = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=False)
    contract = payload["run_contract"]
    mode = contract["input_mode"]
    shared = json.loads((comparison / "shared_initialization_metadata.json").read_text())
    cfg = FoveaArtifactConfig.from_dict(shared["initialization_contract"]["model"])
    vocab = json.loads(Path(args["vocabulary_path"]).read_text())
    mapping, categories = vocab["tag_to_idx"], vocab["tag_to_category"]
    db = sqlite3.connect(f"file:{Path(args['datasets_db']).as_posix()}?mode=ro", uri=True)
    selected = None
    try:
        # Match the earlier deterministic General sample; avoid the photo domain.
        for domain in sorted((int(key) for key in args["train_counts"]), reverse=True):
            if domain == args["photo_dataset_id"]:
                continue
            candidates = db.execute("""SELECT i.id,i.image_path FROM dataset_items i
                WHERE i.dataset_id=? AND EXISTS (SELECT 1 FROM dataset_captions c
                WHERE c.item_id=i.id AND c.is_tags_format=1
                AND (c.content LIKE '%general%' OR c.tag_data LIKE '%general%'))
                ORDER BY i.id LIMIT 200""", (domain,)).fetchall()
            for item_id, image_path in candidates:
                tags = set()
                for data, content in db.execute("SELECT tag_data,content FROM dataset_captions WHERE item_id=? AND is_tags_format=1", (item_id,)):
                    tags.update(resolve_caption_tags(data, content))
                if {tag for tag in tags if categories.get(tag) == "Rating"} != {"general"}:
                    continue
                try:
                    with Image.open(image_path) as source:
                        if source.width * source.height > args["max_source_pixels"]:
                            continue
                        image = source.convert("RGB")
                        image.load()
                except (OSError, ValueError):
                    continue
                selected = (item_id, image, tags)
                break
            if selected:
                break
    finally:
        db.close()
    if selected is None:
        raise ValueError("No bounded annotated-General sample")
    item_id, image, tags = selected
    model = FoveaTagger(cfg.num_tags, encoder_config=cfg.encoder, feature_dim=cfg.feature_dim,
        evidence_slots_per_view=cfg.evidence_slots_per_view, pool_heads=cfg.pool_heads, temperature=cfg.temperature)
    model.load_state_dict(payload["model"], strict=True)
    step, epoch = int(payload["global_step"]), int(payload["epoch"])
    threshold = float(payload["best_threshold"])
    del payload
    model.eval()
    views = build_supervised_views(image, cfg.views, seed=args["seed"], training=False, input_mode=mode)
    batch = collate_view_batches([views])
    with torch.inference_mode():
        probs = model(batch.overview_images, batch.overview_geometry, batch.local_images,
            batch.local_geometry, local_view_mask=batch.local_view_mask)[0].sigmoid().numpy()
    positive = np.zeros(cfg.num_tags, dtype=bool)
    for tag in tags:
        if tag in mapping:
            positive[int(mapping[tag])] = True
    known = np.ones(cfg.num_tags, dtype=bool)
    for category in ("Rating", "Quality", "Model"):
        indices = [int(mapping[tag]) for tag in mapping if categories.get(tag) == category]
        if indices and not positive[indices].any():
            known[indices] = False
    groups = probability_groups(probs, positive, known, threshold)
    counts = {key: int(mask.sum()) for key, mask in groups.items()}
    rows = []
    for key, color in (("TP", "green"), ("FP", "red"), ("FN", "blue")):
        indices = np.flatnonzero(groups[key])
        indices = indices[np.argsort(-probs[indices], kind="stable")][:100]
        rows.extend((f"{key} {rank + 1:03d}", float(probs[index]), color) for rank, index in enumerate(indices))
    preview = request["focus_preview"] and mode != "source_native"
    focus = build_supervised_views(image, cfg.views, seed=args["seed"], training=False,
        input_mode="source_native") if preview else views
    fig = Figure(figsize=(17, 11))
    axes = fig.subplots(1, 3)
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input: annotated Rating:General")
    y = np.arange(len(rows))
    axes[1].barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    axes[1].set_yticks(y, [r[0] for r in rows], fontsize=max(3, min(8, 650 / max(1, len(rows)))))
    axes[1].set_xlim(0, 1)
    axes[1].axvline(threshold, color="black", linestyle="--")
    axes[1].set_title(" / ".join(f"{key}={count}" for key, count in counts.items()))
    axes[1].set_xlabel("Probability; top 100 per TP / FP / FN; TN count only")
    axes[2].imshow(image)
    geometry = []
    for index, spec in enumerate(focus.local_specs):
        box = spec.source_box
        color = f"C{index % 10}"
        axes[2].add_patch(Rectangle((box.x, box.y), box.width, box.height,
            fill=False, linewidth=2, edgecolor=color, linestyle="--" if preview else "-"))
        axes[2].text(max(0, box.x), max(0, box.y), f"L{index + 1}: {spec.pixels_per_canvas_pixel:.2f} source px/canvas px", color=color, fontsize=8)
        geometry.append({"x": box.x, "y": box.y, "width": box.width, "height": box.height,
            "source_pixels_per_canvas_pixel": spec.pixels_per_canvas_pixel})
    axes[2].set_xlim(0, image.width)
    axes[2].set_ylim(image.height, 0)
    axes[2].axis("off")
    label = "C policy PREVIEW (NOT used by this inference)" if preview else (
        "Source-native views USED (not attribution)" if mode == "source_native" else
        "No source-native focus: overview only" if mode == "overview_only" else "Placebo views from overview; NOT native focus")
    axes[2].set_title(label, fontsize=10)
    fig.suptitle(f"FoveaTag {mode} / checkpoint epoch cursor {epoch}, step {step}")
    fig.text(.5, .02, "Anonymous group ranks, no tag names/IDs. Annotation-based FP may include missing labels. Boxes show input views, not causal evidence or a learned router.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .96))
    name = f"general_{mode}_{step}_{uuid.uuid4().hex[:8]}.png"
    root = job.parent
    temporary = root / (name + ".tmp")
    fig.savefig(temporary, format="png", dpi=120)
    os.replace(temporary, root / name)
    index_path = root / "index.json"
    previous = json.loads(index_path.read_text()) if index_path.is_file() else {"items": []}
    item = {"filename": name, "mode": mode, "step": step, "epoch": epoch,
        "catalog_step": request["catalog_step"], "counts": counts, "threshold": threshold,
        "sample_id": item_id, "native_focus_used": mode == "source_native",
        "focus_preview": preview, "local_boxes": geometry}
    atomic_json(index_path, {"items": [item, *previous["items"]][:12], "error": None})
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    job_path = Path(sys.argv[1]).resolve()
    try:
        render(job_path)
    except Exception as exc:
        # Never propagate sample annotations or per-tag values into API errors.
        path = job_path.parent / "index.json"
        state = json.loads(path.read_text()) if path.is_file() else {"items": []}
        atomic_json(path, {**state, "error": f"Diagnostic generation failed ({type(exc).__name__}); training is unaffected"})
        raise
    finally:
        if job_path.name.startswith("job-") and job_path.parent.name == "diagnostics":
            shutil.rmtree(job_path)
