"""Build a sparse Likelihood-Ratio (LR) matrix from training labels.

The LR matrix powers the "lr_matrix" context method in conditional inference
(see :class:`SigLIP2InferenceManager._apply_lr_correction`).  For each anchor
tag *c* and target tag *n* we precompute

    LR(n, c) = log [ P(y_n=1 | y_c=1) / P(y_n=1 | y_c=0) ]

with Laplace smoothing.  Storage is sparsified to ~10-150 MB by:

- limiting anchors to the top-K most frequent tags,
- keeping only the top-M targets per anchor whose ``|LR|`` >= threshold.

At inference time the manager looks up LR vectors by anchor index and adds
them to the raw logits, scaled by a user-chosen lambda.

This module is dual-purpose:

- **Importable** -- ``from core.tagger.lr_matrix_builder import build_lr_matrix``
  exposes the function for programmatic use (e.g. future API endpoint).
- **Executable** -- ``python -m core.tagger.lr_matrix_builder --dataset-ids 26
  --vocab-path .../vocabulary.json --output .../lr_matrix.npz`` runs it as a
  CLI tool.

CLI example
-----------

::

    python -m core.tagger.lr_matrix_builder \\
        --dataset-ids 26 \\
        --vocab-path tagger_models/<run>/vocabulary.json \\
        --output     tagger_models/<run>/lr_matrix.npz \\
        --top-anchors 10000 --top-targets 1000 --lr-threshold 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

# When invoked as a CLI (``python -m core.tagger.lr_matrix_builder``) the
# ``backend/`` package root is already on sys.path because we use the
# ``-m`` form.  When invoked as a plain script (``python
# backend/core/tagger/lr_matrix_builder.py``) we need to add ``backend/``
# manually so that ``database`` and ``core`` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))                  # backend/core/tagger
_BACKEND_DIR = os.path.dirname(os.path.dirname(_HERE))              # backend
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx_to_tag = {int(k): v for k, v in vocab["idx_to_tag"].items()}
    tag_to_idx = {tag: idx for idx, tag in idx_to_tag.items()}
    return tag_to_idx, idx_to_tag


def _collect_samples(
    dataset_ids: List[int],
    caption_types: Optional[List[str]] = None,
) -> List[List[str]]:
    """Pull (positive) tag lists from the datasets DB.

    Returns one list of normalized tags per sample (image).
    """
    # Local imports - these resolve via _BACKEND_DIR added to sys.path.
    from database import DatasetsSessionLocal
    from database.models import DatasetItem, DatasetCaption
    from core.tagger.tag_vocabulary import normalize_tag

    samples: List[List[str]] = []
    db = DatasetsSessionLocal()
    try:
        for did in dataset_ids:
            print(f"[LRBuild] Loading dataset_id={did}...", flush=True)
            items = db.query(DatasetItem).filter(DatasetItem.dataset_id == did).all()
            print(f"[LRBuild]   {len(items)} items", flush=True)
            if not items:
                continue
            valid_ids = [it.id for it in items if it.image_path and os.path.isfile(it.image_path)]
            print(f"[LRBuild]   {len(valid_ids)} items have image files on disk", flush=True)

            CHUNK = 500
            captions_by_item: Dict[int, list] = defaultdict(list)
            for i in range(0, len(valid_ids), CHUNK):
                chunk = valid_ids[i:i + CHUNK]
                q = (
                    db.query(DatasetCaption)
                    .filter(
                        DatasetCaption.item_id.in_(chunk),
                        DatasetCaption.is_tags_format == True,  # noqa: E712
                    )
                )
                if caption_types:
                    q = q.filter(DatasetCaption.caption_type.in_(caption_types))
                for cap in q.all():
                    captions_by_item[cap.item_id].append(cap)

            print(f"[LRBuild]   building tag lists...", flush=True)
            for item_id in valid_ids:
                caps = captions_by_item.get(item_id)
                if not caps:
                    continue
                tag_set = set()
                for cap in caps:
                    raw_tags: List[str] = []
                    if cap.tag_data:
                        try:
                            data = json.loads(cap.tag_data) if isinstance(cap.tag_data, str) else cap.tag_data
                            if isinstance(data, list):
                                raw_tags = [r["tag"] for r in data if isinstance(r, dict) and "tag" in r]
                        except (json.JSONDecodeError, TypeError):
                            pass
                    if not raw_tags and cap.content:
                        raw_tags = [t.strip() for t in cap.content.split(",") if t.strip()]
                    for t in raw_tags:
                        tag_set.add(normalize_tag(t))
                if tag_set:
                    samples.append(sorted(tag_set))
            print(f"[LRBuild]   {len(samples)} cumulative samples", flush=True)
    finally:
        db.close()
    return samples


def _build_label_matrix(
    samples: List[List[str]],
    tag_to_idx: Dict[str, int],
    n_tags: int,
) -> sp.csr_matrix:
    """Build a binary CSR matrix [n_samples, n_tags] from tag lists."""
    indptr_list: List[int] = [0]
    indices: List[int] = []
    skipped_tags = 0
    for tags in samples:
        for t in tags:
            idx = tag_to_idx.get(t)
            if idx is None:
                skipped_tags += 1
                continue
            indices.append(idx)
        indptr_list.append(len(indices))
    if skipped_tags:
        print(f"[LRBuild]   {skipped_tags} tag occurrences not in vocabulary (skipped)")
    indices_np = np.asarray(indices, dtype=np.int32)
    data_np = np.ones(len(indices_np), dtype=np.int8)
    indptr_np = np.asarray(indptr_list, dtype=np.int64)
    return sp.csr_matrix((data_np, indices_np, indptr_np), shape=(len(samples), n_tags))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_lr_matrix(
    dataset_ids: List[int],
    vocab_path: str,
    output_path: str,
    top_anchors: int = 10000,
    top_targets: int = 1000,
    lr_threshold: float = 1.0,
    eps: float = 0.5,
    min_anchor_count: int = 10,
    caption_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the sparse LR matrix from training labels and save as ``.npz``.

    Parameters
    ----------
    dataset_ids
        List of ``Dataset.id`` values to scan in the datasets DB.
    vocab_path
        Path to ``vocabulary.json`` from the trained model (defines the
        ``tag -> index`` mapping used by the inference manager).
    output_path
        Destination ``.npz`` path.  Convention: place alongside the
        vocabulary so that :func:`SigLIP2InferenceManager.load_model`
        picks it up automatically.
    top_anchors
        Number of most-frequent tags to use as anchors (rare tags are
        excluded because their statistics are unreliable).
    top_targets
        Maximum number of target entries kept per anchor.
    lr_threshold
        Drop entries whose ``|LR|`` is below this value.
    eps
        Laplace smoothing constant.
    min_anchor_count
        Skip anchors that appear in fewer than this many samples.
    caption_types
        Optional caption-type filter (e.g. ``["tags", "booru"]``).

    Returns
    -------
    dict
        Summary statistics: ``n_samples``, ``n_anchors``, ``n_entries``,
        ``output_size_bytes``.
    """
    print(f"[LRBuild] Vocabulary:    {vocab_path}")
    print(f"[LRBuild] Output:        {output_path}")
    print(f"[LRBuild] Anchors top-K: {top_anchors}, targets per anchor: {top_targets}")
    print(f"[LRBuild] |LR| threshold: {lr_threshold}, eps={eps}, min anchor count: {min_anchor_count}")

    tag_to_idx, idx_to_tag = _load_vocab(vocab_path)
    n_tags = len(idx_to_tag)
    print(f"[LRBuild] {n_tags} tags in vocabulary")

    samples = _collect_samples(dataset_ids, caption_types)
    n_samples = len(samples)
    if n_samples == 0:
        raise RuntimeError("No samples found for the specified dataset_ids.")
    print(f"[LRBuild] {n_samples} samples loaded")

    print("[LRBuild] Building sparse label matrix (CSR)...")
    Y_csr = _build_label_matrix(samples, tag_to_idx, n_tags)
    nnz = Y_csr.nnz
    print(f"[LRBuild]   shape={Y_csr.shape}  nnz={nnz} (avg {nnz/n_samples:.1f} tags/sample)")

    print("[LRBuild] Computing per-tag counts...")
    counts = np.asarray(Y_csr.sum(axis=0)).ravel().astype(np.int64)   # [n_tags]

    print("[LRBuild] Converting to CSC for column access...")
    Y_csc = Y_csr.tocsc()

    eligible = np.where(counts >= min_anchor_count)[0]
    eligible = eligible[np.argsort(-counts[eligible])][:top_anchors]
    n_anchors = len(eligible)
    print(f"[LRBuild] Selected {n_anchors} anchors (most-frequent tags with count >= {min_anchor_count})")

    anchor_tag_indices: List[int] = []
    anchor_to_offset:   List[int] = [0]
    target_tag_indices: List[int] = []
    lr_values:          List[float] = []

    n_total = float(n_samples)
    t0 = time.time()
    for k, c_idx in enumerate(eligible):
        anchor_col = Y_csc[:, int(c_idx)]
        n_pos = int(anchor_col.nnz)
        if n_pos < min_anchor_count:
            anchor_tag_indices.append(int(c_idx))
            anchor_to_offset.append(len(target_tag_indices))
            continue
        sample_ids = anchor_col.indices
        cooc = np.asarray(Y_csr[sample_ids].sum(axis=0)).ravel().astype(np.int64)
        n_neg = n_total - n_pos

        cooc_neg = counts - cooc
        p_pos = (cooc.astype(np.float64) + eps) / (n_pos + 2.0 * eps)
        p_neg = (cooc_neg.astype(np.float64) + eps) / (max(1.0, n_neg) + 2.0 * eps)
        lr = np.log(p_pos / p_neg)

        # Drop the anchor's self-LR (uninformative for context correction).
        lr[c_idx] = 0.0

        keep_mask = np.abs(lr) >= lr_threshold
        if not keep_mask.any():
            anchor_tag_indices.append(int(c_idx))
            anchor_to_offset.append(len(target_tag_indices))
            continue
        kept_ids = np.where(keep_mask)[0]
        if len(kept_ids) > top_targets:
            order = np.argsort(-np.abs(lr[kept_ids]))[:top_targets]
            kept_ids = kept_ids[order]
        kept_ids = np.sort(kept_ids)   # sorted for deterministic & slice-friendly storage

        anchor_tag_indices.append(int(c_idx))
        target_tag_indices.extend(int(x) for x in kept_ids)
        lr_values.extend(float(x) for x in lr[kept_ids])
        anchor_to_offset.append(len(target_tag_indices))

        if (k + 1) % 100 == 0 or (k + 1) == n_anchors:
            dt = time.time() - t0
            rate = (k + 1) / dt
            eta = (n_anchors - k - 1) / max(rate, 1e-9)
            print(
                f"[LRBuild]   anchor {k+1}/{n_anchors}  "
                f"({rate:.1f}/s, ETA {eta/60:.1f}m, "
                f"entries so far: {len(lr_values)})",
                flush=True,
            )

    print(f"[LRBuild] Total entries: {len(lr_values)}")
    print(f"[LRBuild] Saving to {output_path} ...")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(
        output_path,
        anchor_tag_indices=np.asarray(anchor_tag_indices, dtype=np.int32),
        anchor_to_offset=np.asarray(anchor_to_offset,     dtype=np.int32),
        target_tag_indices=np.asarray(target_tag_indices, dtype=np.int32),
        lr_values=np.asarray(lr_values,                   dtype=np.float16),
    )
    out_size = os.path.getsize(output_path)
    print(f"[LRBuild] Done. File size: {out_size/1e6:.1f} MB")

    return {
        "n_samples":         n_samples,
        "n_anchors":         n_anchors,
        "n_entries":         len(lr_values),
        "output_size_bytes": out_size,
        "output_path":       output_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset-ids", required=True,
                   help="Comma-separated dataset IDs to scan (e.g. 26 or 26,27).")
    p.add_argument("--vocab-path",  required=True,
                   help="Path to vocabulary.json from the trained model.")
    p.add_argument("--output",      required=True,
                   help="Destination .npz path.  Convention: place alongside vocabulary.json.")
    p.add_argument("--top-anchors", type=int, default=10000,
                   help="Number of most-frequent tags to use as anchors (default: 10000).")
    p.add_argument("--top-targets", type=int, default=1000,
                   help="Maximum number of target entries per anchor (default: 1000).")
    p.add_argument("--lr-threshold", type=float, default=1.0,
                   help="Drop entries with abs(LR) below this value (default: 1.0).")
    p.add_argument("--eps",         type=float, default=0.5,
                   help="Laplace smoothing constant (default: 0.5).")
    p.add_argument("--caption-types",
                   help="Optional comma-separated caption_type filter (e.g. 'tags,booru').")
    p.add_argument("--min-anchor-count", type=int, default=10,
                   help="Skip anchors that appear in fewer than this many samples (default: 10).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_ids   = [int(x) for x in args.dataset_ids.split(",") if x.strip()]
    caption_types = [x.strip() for x in args.caption_types.split(",")] if args.caption_types else None
    try:
        build_lr_matrix(
            dataset_ids=dataset_ids,
            vocab_path=args.vocab_path,
            output_path=args.output,
            top_anchors=args.top_anchors,
            top_targets=args.top_targets,
            lr_threshold=args.lr_threshold,
            eps=args.eps,
            min_anchor_count=args.min_anchor_count,
            caption_types=caption_types,
        )
        return 0
    except Exception as e:   # noqa: BLE001
        print(f"[LRBuild] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
