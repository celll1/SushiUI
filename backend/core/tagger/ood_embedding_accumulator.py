"""
Online OOD embedding accumulator for tagger training.

Collects CLS embeddings (the pooled feature vector fed to the classification
head) using reservoir sampling, then fits a multivariate Gaussian with
Ledoit-Wolf shrinkage at checkpoint save time.  The resulting reference
distribution is used at inference time for Mahalanobis-distance OOD detection.

File saved: ``{checkpoint_name}_ood_ref.npz``
Contents:   mu (D,), cov_inv (D, D), p50, p95  (all float32)
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


class OodEmbeddingAccumulator:
    """Reservoir-sampled CLS-embedding accumulator for in-distribution fitting.

    Algorithm: Vitter's Algorithm R.  After *max_samples* embeddings have been
    seen, each new embedding replaces a random existing one with probability
    max_samples / n_seen.  This ensures a uniform random sample over all seen
    embeddings regardless of arrival order.
    """

    def __init__(self, max_samples: int = 4000, seed: int = 42) -> None:
        self.max_samples  = max_samples
        self._rng         = random.Random(seed)
        self.reservoir: list[np.ndarray] = []  # each entry: (D,) float32
        self.n_seen: int  = 0

    # ------------------------------------------------------------------

    def update(self, embs: np.ndarray) -> None:
        """Add a batch of embeddings (B, D) to the reservoir."""
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[np.newaxis]  # handle single-vector input
        for e in embs:
            self.n_seen += 1
            if len(self.reservoir) < self.max_samples:
                self.reservoir.append(e.copy())
            else:
                j = self._rng.randint(0, self.n_seen - 1)
                if j < self.max_samples:
                    self.reservoir[j] = e.copy()

    # ------------------------------------------------------------------

    def finalize(self, save_path: str) -> dict:
        """Fit a multivariate Gaussian and save to *save_path*.

        Returns a summary dict.  Returns {} if fewer than 10 embeddings have
        been collected (not enough data to fit a meaningful distribution).
        """
        n = len(self.reservoir)
        if n < 10:
            print(f"[OodEmbeddingAccumulator] Skipping: only {n} embeddings collected.")
            return {}

        E = np.stack(self.reservoir, axis=0).astype(np.float64)  # (N, D)
        mu = E.mean(axis=0)

        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(assume_centered=False)
            lw.fit(E)
            cov_inv = np.linalg.inv(lw.covariance_)
        except Exception as _e:
            print(f"[OodEmbeddingAccumulator] LedoitWolf failed ({_e}); using diagonal regularization")
            cov = np.cov(E, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = np.linalg.inv(cov)

        # Compute per-sample Mahalanobis distances for percentile thresholds
        diffs = E - mu  # (N, D)
        dists = np.sqrt(np.maximum(0.0, np.einsum("nd,de,ne->n", diffs, cov_inv, diffs)))
        p50 = float(np.percentile(dists, 50))
        p95 = float(np.percentile(dists, 95))

        np.savez_compressed(
            save_path,
            mu      = mu.astype(np.float32),
            cov_inv = cov_inv.astype(np.float32),
            p50     = np.float32(p50),
            p95     = np.float32(p95),
        )
        print(
            f"[OodEmbeddingAccumulator] Saved OOD reference → {os.path.basename(save_path)} "
            f"| n={n} (of {self.n_seen} seen) | p50={p50:.2f} | p95={p95:.2f}"
        )
        return {"n_samples": n, "n_seen": self.n_seen, "p50": p50, "p95": p95}

    # ------------------------------------------------------------------

    def save_reservoir(self, path: str) -> None:
        """Save raw reservoir to *path* for later resume."""
        if not self.reservoir:
            return
        np.savez_compressed(
            path,
            reservoir = np.stack(self.reservoir, axis=0).astype(np.float32),
            n_seen    = np.int64(self.n_seen),
        )

    def restore_from_reservoir(self, path: str) -> bool:
        """Restore reservoir from a file saved by *save_reservoir*.

        Returns True on success, False if file not found or invalid.
        """
        if not os.path.isfile(path):
            return False
        try:
            data = np.load(path)
            arr  = data["reservoir"]
            self.reservoir = [arr[i] for i in range(len(arr))]
            self.n_seen    = int(data["n_seen"])
            print(
                f"[OodEmbeddingAccumulator] Restored reservoir: "
                f"{len(self.reservoir)} samples (n_seen={self.n_seen})"
            )
            return True
        except Exception as _e:
            print(f"[OodEmbeddingAccumulator] WARNING: could not restore reservoir: {_e}")
            return False
