"""
Per-tag prediction histogram accumulator for online threshold metric computation.

During training, each batch's sigmoid predictions and labels are accumulated into
per-tag histograms. At checkpoint time the histograms are used to derive per-tag
metrics (hard_rate, FP/FN rates, best F1 threshold) which are saved alongside the
model as ``{name}_tag_metrics.npz``.

Memory footprint (V=106 k tags, K=100 bins):
  2 × (pos_hist + total_hist) × V × K × 4 bytes ≈ 170 MB  (sample-count independent)
"""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import torch


class TagMetricsAccumulator:
    """Online per-tag histogram accumulator with two-epoch sliding window.

    Two histogram sets are maintained:
    - ``cur_*``: predictions from the *current* (in-progress) epoch.
    - ``prev_*``: predictions from the *previous completed* epoch.

    At mid-epoch checkpoint saves: use ``cur + prev`` (partial current + full previous).
    At epoch-boundary saves: use ``cur`` only (full current epoch, cleanest signal).
    After saving at epoch boundary: rotate (prev ← cur, cur ← zero).

    ``tag_count`` accumulates across *all* epochs and is never reset —
    it represents training-set tag frequency.
    """

    def __init__(self, vocab_size: int, n_bins: int = 100) -> None:
        self.vocab_size = vocab_size
        self.n_bins = n_bins

        # Current epoch
        self.pos_hist_cur   = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_hist_cur = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_images_cur: int = 0

        # Previous completed epoch
        self.pos_hist_prev   = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_hist_prev = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_images_prev: int = 0

        # All-epoch cumulative tag frequency (never reset)
        self.tag_count = np.zeros(vocab_size, dtype=np.int32)
        self.total_images_all: int = 0  # denominator for global_freq

    # ------------------------------------------------------------------
    # Update (called every training batch)
    # ------------------------------------------------------------------

    def update(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one batch of predictions.

        Args:
            preds:  ``[B, V]`` CPU float tensor — sigmoid probabilities.
            labels: ``[B, V]`` CPU bool or float tensor — ground-truth labels.
        """
        p = preds.float()         # [B, V]
        l = (labels > 0.5)        # [B, V] bool
        B, V = p.shape

        # Bin index: [B, V]
        bin_idx = (p * self.n_bins).long().clamp(0, self.n_bins - 1)

        # ── tag_count (all-epoch cumulative) ──────────────────────────
        self.tag_count += l.sum(dim=0).numpy().astype(np.int32)
        self.total_images_all += B
        self.total_images_cur  += B

        # ── total_hist (all predictions, regardless of label) ─────────
        # flat_idx[b, v] = v * n_bins + bin_idx[b, v]
        flat_idx = (
            torch.arange(V, dtype=torch.long).unsqueeze(0).expand(B, -1) * self.n_bins
            + bin_idx
        ).reshape(-1)  # [B*V]
        ones = torch.ones(B * V, dtype=torch.int32)
        total_flat = torch.zeros(V * self.n_bins, dtype=torch.int32)
        total_flat.scatter_add_(0, flat_idx, ones)
        self.total_hist_cur += total_flat.reshape(V, self.n_bins).numpy()

        # ── pos_hist (positive samples only — sparse) ─────────────────
        pos_b, pos_v = torch.where(l)
        if len(pos_b) > 0:
            pos_flat_idx = pos_v * self.n_bins + bin_idx[pos_b, pos_v]
            ones_pos = torch.ones(len(pos_b), dtype=torch.int32)
            pos_flat = torch.zeros(V * self.n_bins, dtype=torch.int32)
            pos_flat.scatter_add_(0, pos_flat_idx.long(), ones_pos)
            self.pos_hist_cur += pos_flat.reshape(V, self.n_bins).numpy()

    # ------------------------------------------------------------------
    # Epoch rotation
    # ------------------------------------------------------------------

    def rotate_epoch(self) -> None:
        """Call at epoch end (after saving epoch-boundary checkpoint).

        Moves current epoch histograms to the ``prev`` slot and zeros ``cur``.
        """
        np.copyto(self.pos_hist_prev,   self.pos_hist_cur)
        np.copyto(self.total_hist_prev, self.total_hist_cur)
        self.total_images_prev = self.total_images_cur

        self.pos_hist_cur[:] = 0
        self.total_hist_cur[:] = 0
        self.total_images_cur = 0

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _merged(self, epoch_boundary: bool):
        """Return (pos_hist, total_hist, total_images) for the requested window."""
        if epoch_boundary:
            return self.pos_hist_cur, self.total_hist_cur, self.total_images_cur
        else:
            return (
                self.pos_hist_cur  + self.pos_hist_prev,
                self.total_hist_cur + self.total_hist_prev,
                self.total_images_cur + self.total_images_prev,
            )

    def compute_metrics(
        self,
        epoch_boundary: bool = False,
        hard_lo: float = 0.25,
        hard_hi: float = 0.75,
    ) -> dict:
        """Derive per-tag metrics from accumulated histograms.

        Returns a dict of 1-D float32 arrays of length ``vocab_size``:
          ``n_pos``, ``n_neg``, ``global_freq``,
          ``hard_rate``, ``fp_rate_50``, ``fn_rate_50``,
          ``best_f1``, ``best_thr``.
        NaN is used for tags with insufficient data.
        """
        pos_h, total_h, _ = self._merged(epoch_boundary)
        neg_h = total_h - pos_h  # [V, K]

        n_pos = pos_h.sum(axis=1).astype(np.float32)   # [V]
        n_neg = neg_h.sum(axis=1).astype(np.float32)   # [V]

        # Global frequency from all-epoch tag_count
        denom = max(self.total_images_all, 1)
        global_freq = self.tag_count.astype(np.float32) / denom

        # Hard rate: P(hard_lo ≤ p ≤ hard_hi | label=0)
        lo_bin = int(hard_lo * self.n_bins)
        hi_bin = int(hard_hi * self.n_bins)
        hard_neg = neg_h[:, lo_bin : hi_bin + 1].sum(axis=1).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            hard_rate = np.where(n_neg > 0, hard_neg / n_neg, np.nan)

        # Cumulative distributions (suffix sums from high → low threshold)
        # pos_cdf[v, k] = fraction of pos samples with pred >= bin_edge[k]
        # neg_cdf[v, k] = fraction of neg samples with pred >= bin_edge[k]
        pos_rev = pos_h[:, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float32)  # [V, K]
        neg_rev = neg_h[:, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float32)  # [V, K]
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_rate = np.where(n_pos[:, None] > 0, pos_rev / n_pos[:, None], np.nan)  # recall
            fp_rate = np.where(n_neg[:, None] > 0, neg_rev / n_neg[:, None], np.nan)
            fn_rate = 1.0 - tp_rate  # miss rate

        # FP/FN at threshold=0.50 (bin index = 50)
        bin_50 = 50
        fp_rate_50 = fp_rate[:, bin_50]
        fn_rate_50 = fn_rate[:, bin_50]

        # Best F1 sweep over threshold grid (vectorised over all tags at once)
        # We evaluate at each bin boundary = k/n_bins for k=1..n_bins-1
        # Precision = TP / (TP + FP),  Recall = TP / (TP + FN)
        with np.errstate(invalid="ignore", divide="ignore"):
            tp_arr = tp_rate * n_pos[:, None]       # [V, K]
            fp_arr = fp_rate * n_neg[:, None]       # [V, K]
            fn_arr = fn_rate * n_pos[:, None]       # [V, K]
            prec = tp_arr / (tp_arr + fp_arr + 1e-8)
            rec  = tp_arr / (tp_arr + fn_arr + 1e-8)
            f1   = 2.0 * prec * rec / (prec + rec + 1e-8)

        # For tags with no positive samples, f1 is NaN → best_f1 = NaN
        has_pos = n_pos > 0
        best_f1  = np.where(has_pos, np.nanmax(f1, axis=1),  np.nan).astype(np.float32)
        best_bin = np.where(has_pos, np.nanargmax(f1, axis=1), -1).astype(np.int32)
        best_thr = np.where(
            has_pos,
            best_bin.astype(np.float32) / self.n_bins,
            np.nan,
        ).astype(np.float32)

        return {
            "n_pos":       n_pos,
            "n_neg":       n_neg,
            "global_freq": global_freq,
            "hard_rate":   hard_rate.astype(np.float32),
            "fp_rate_50":  fp_rate_50.astype(np.float32),
            "fn_rate_50":  fn_rate_50.astype(np.float32),
            "best_f1":     best_f1,
            "best_thr":    best_thr,
        }

    # ------------------------------------------------------------------
    # Scatter-plot helper (visualization only)
    # ------------------------------------------------------------------

    def compute_scatter_for_vis(self, min_npos: int = 20) -> dict:
        """FP/FN scatter data for live visualization (cur epoch only).

        Only includes tags with n_pos >= min_npos in the current epoch,
        giving statistically reliable FP/FN rate estimates (±0.22 CI at n=20).

        Returns a dict with compact float lists suitable for JSON serialization:
            fp, fn : FP/FN rate at threshold=0.5, one value per qualifying tag
            n_pos  : positive sample count per qualifying tag
            n_tags : number of qualifying tags
            total_images : images seen in current epoch
        """
        pos_h = self.pos_hist_cur                          # [V, K]
        neg_h = self.total_hist_cur - self.pos_hist_cur    # [V, K]

        n_pos = pos_h.sum(axis=1).astype(np.float32)       # [V]
        n_neg = neg_h.sum(axis=1).astype(np.float32)       # [V]

        mask = (n_pos >= min_npos) & (n_neg > 0)
        if not mask.any():
            return {
                "fp": [], "fn": [], "n_pos": [],
                "n_tags": 0, "total_images": self.total_images_cur,
            }

        bin_50 = self.n_bins // 2  # = 50 for default n_bins=100

        # Suffix cumsum: counts[v, k] = samples with pred >= bin_edge[k]
        pos_rev = pos_h[:, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float32)
        neg_rev = neg_h[:, ::-1].cumsum(axis=1)[:, ::-1].astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            fp_rate = np.where(n_neg > 0, neg_rev[:, bin_50] / n_neg, np.nan)
            fn_rate = np.where(n_pos > 0,
                               1.0 - pos_rev[:, bin_50] / n_pos, np.nan)

        idx = np.where(mask)[0]
        return {
            "fp":           np.round(fp_rate[idx], 4).tolist(),
            "fn":           np.round(fn_rate[idx], 4).tolist(),
            "n_pos":        n_pos[idx].astype(np.int32).tolist(),
            "n_tags":       int(mask.sum()),
            "total_images": self.total_images_cur,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        epoch_boundary: bool = False,
        tag_names: Optional[List[str]] = None,
        hard_lo: float = 0.25,
        hard_hi: float = 0.75,
    ) -> None:
        """Save histograms + derived metrics to a compressed .npz file."""
        pos_h, total_h, total_images = self._merged(epoch_boundary)
        metrics = self.compute_metrics(epoch_boundary=epoch_boundary,
                                       hard_lo=hard_lo, hard_hi=hard_hi)

        save_kwargs: dict = {
            "pos_hist":     pos_h,
            "total_hist":   total_h,
            "tag_count":    self.tag_count,
            "total_images": np.array([total_images], dtype=np.int64),
            "n_bins":       np.array([self.n_bins],  dtype=np.int32),
            "hard_lo":      np.array([hard_lo],      dtype=np.float32),
            "hard_hi":      np.array([hard_hi],      dtype=np.float32),
        }
        save_kwargs.update({k: v for k, v in metrics.items()})

        if tag_names is not None:
            # Store as object array so np.load can round-trip strings
            save_kwargs["tag_names"] = np.array(tag_names, dtype=object)

        np.savez_compressed(path, **save_kwargs)

    @staticmethod
    def load(path: str) -> dict:
        """Load a saved .npz and return a plain dict.

        The dict contains all saved arrays plus derived keys for inference use:
        ``best_thr``, ``best_f1``, ``global_freq``, ``n_pos``, ``tag_names``
        (the last one may be absent for older files).
        """
        data = np.load(path, allow_pickle=True)
        result = {k: data[k] for k in data.files}
        # Unwrap scalar arrays for convenience
        for scalar_key in ("n_bins", "hard_lo", "hard_hi"):
            if scalar_key in result:
                result[scalar_key] = result[scalar_key].item()
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """True if at least one batch has been accumulated."""
        return self.total_images_cur > 0 or self.total_images_prev > 0
