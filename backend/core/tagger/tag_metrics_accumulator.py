"""
Per-tag prediction histogram accumulator for online threshold metric computation.

During training, each batch's sigmoid predictions and labels are accumulated into
per-tag histograms. At checkpoint time the histograms are used to derive per-tag
metrics (hard_rate, FP/FN rates, best F1 threshold) which are saved alongside the
model as ``{name}_tag_metrics.npz``.

Memory footprint (V=106 k tags, K=100 bins):
  3 × (pos_hist + total_hist) × V × K × 4 bytes ≈ 255 MB  (sample-count independent)
"""

from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import torch


class TagMetricsAccumulator:
    """Online per-tag histogram accumulator with smooth sliding window.

    Three histogram slots are maintained:

    - ``cur_*``:  predictions from the current (in-progress) epoch (epoch N).
    - ``prev_*``: full predictions from the previous completed epoch (epoch N-1).
    - ``pp_*``:   full predictions from epoch N-2, faded out as epoch N progresses.

    Window at any point during epoch N after ``cur_images`` batches have been seen::

        pp_weight  = max(0, 1 - cur_images / pp_images)
        effective  = cur  +  prev  +  pp * pp_weight

    This keeps the effective window at ≈ 2 epochs throughout training, eliminating
    the abrupt 2-epoch → 1-epoch jump that occurred at epoch boundaries in the
    old two-slot design.

    **Edge cases:**

    *Epoch size decreases:*
        pp is consumed more slowly (pp_weight > 0 at epoch end).  On the
        next :meth:`rotate_epoch` call, the old pp is discarded and replaced by
        old prev, so data older than 2 epochs is never retained past a rotation.

    *Epoch size increases:*
        pp_weight clamps to 0 before the epoch ends.  The pp contribution
        vanishes entirely; prev is never touched mid-epoch.

    At epoch-boundary checkpoint saves the window is ``cur + prev`` (pp_weight ≈ 0
    and about to be discarded by rotate_epoch anyway).

    ``tag_count`` accumulates across *all* epochs and is never reset —
    it represents training-set tag frequency.
    """

    def __init__(self, vocab_size: int, n_bins: int = 100) -> None:
        self.vocab_size = vocab_size
        self.n_bins = n_bins

        # Current epoch (epoch N)
        self.pos_hist_cur   = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_hist_cur = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_images_cur: int = 0

        # Previous completed epoch (epoch N-1)
        self.pos_hist_prev   = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_hist_prev = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_images_prev: int = 0

        # Prev-prev epoch (epoch N-2), being faded out
        self.pos_hist_pp   = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_hist_pp = np.zeros((vocab_size, n_bins), dtype=np.int32)
        self.total_images_pp: int = 0

        # All-epoch cumulative tag frequency (never reset)
        self.tag_count = np.zeros(vocab_size, dtype=np.int32)
        self.total_images_all: int = 0  # denominator for global_freq

        # ── Train-count deficit tracking (exposure-balancing augmentation) ──
        # tag_count_epoch_start: snapshot of tag_count at the last epoch boundary.
        # last_epoch_delta:      tag_count gained over the last completed epoch
        #                        (≈ the tag's *current* per-epoch exposure rate).
        # epochs_elapsed:        number of completed epochs (cumulative, restored).
        # Deficit for a tag = max(0, last_epoch_delta * epochs_elapsed - tag_count):
        # a tag present since epoch 1 at a steady rate has deficit≈0, while a
        # late-joined / newly-grown tag has a positive deficit (under-exposed
        # relative to its current rate). Genuinely-rare-but-stable tags also have
        # deficit≈0 (handled by the low-F1 path instead).
        self.tag_count_epoch_start = np.zeros(vocab_size, dtype=np.int32)
        self.last_epoch_delta = np.zeros(vocab_size, dtype=np.int32)
        self.epochs_elapsed: int = 0

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

        Performs a three-way rotation:
          pp  ← prev   (prev-prev becomes the new fade-out slot)
          prev ← cur   (current becomes previous)
          cur  ← zero  (ready for the next epoch)

        Data older than 2 epochs (the old pp) is discarded on rotation.
        """
        # pp ← prev
        np.copyto(self.pos_hist_pp,   self.pos_hist_prev)
        np.copyto(self.total_hist_pp, self.total_hist_prev)
        self.total_images_pp = self.total_images_prev

        # prev ← cur
        np.copyto(self.pos_hist_prev,   self.pos_hist_cur)
        np.copyto(self.total_hist_prev, self.total_hist_cur)
        self.total_images_prev = self.total_images_cur

        # cur ← zero
        self.pos_hist_cur[:] = 0
        self.total_hist_cur[:] = 0
        self.total_images_cur = 0

        # ── Train-count deficit: finalize the completed epoch's exposure delta ──
        # (tag_count keeps accumulating; we snapshot the per-epoch gain here.)
        self.last_epoch_delta = (self.tag_count - self.tag_count_epoch_start).astype(np.int32)
        np.copyto(self.tag_count_epoch_start, self.tag_count)
        self.epochs_elapsed += 1

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _merged(self, epoch_boundary: bool):
        """Return (pos_hist_f, total_hist_f, total_images) for the current window.

        At ``epoch_boundary=True``:
            Returns ``cur + prev`` as float32.  pp is omitted because it is
            about to be discarded by the next :meth:`rotate_epoch` call, and
            its weight would already be near 0 at a full-epoch boundary.

        At ``epoch_boundary=False`` (mid-epoch):
            Returns ``cur + prev + pp * pp_weight`` where::

                pp_weight = max(0, 1 − total_images_cur / total_images_pp)

            This gives a smooth transition as the current epoch progresses.
        """
        cur_f   = self.pos_hist_cur.astype(np.float32)
        prev_f  = self.pos_hist_prev.astype(np.float32)
        c_tot_f = self.total_hist_cur.astype(np.float32)
        p_tot_f = self.total_hist_prev.astype(np.float32)

        if epoch_boundary or self.total_images_pp == 0:
            pos   = cur_f   + prev_f
            total = c_tot_f + p_tot_f
            n_img = self.total_images_cur + self.total_images_prev
        else:
            pp_weight = max(0.0, 1.0 - self.total_images_cur / self.total_images_pp)
            pp_f     = self.pos_hist_pp.astype(np.float32)
            pp_tot_f = self.total_hist_pp.astype(np.float32)
            pos   = cur_f   + prev_f   + pp_f     * pp_weight
            total = c_tot_f + p_tot_f  + pp_tot_f * pp_weight
            n_img = (self.total_images_cur + self.total_images_prev
                     + int(self.total_images_pp * pp_weight))

        return pos, total, n_img

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

        # FP/FN at threshold=0.50 (bin index = n_bins // 2)
        bin_50 = self.n_bins // 2
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

        # Best F1 / threshold per tag.
        # nanmax / nanargmax raise ValueError when an entire row is NaN (numpy
        # treats that as an error, not just a warning).  This occurs for tags
        # where n_neg == 0 in the current window (all observations positive),
        # making fp_rate — and therefore all f1 values — NaN.
        # Guard: temporarily replace all-NaN rows with 0 so the reduction
        # doesn't raise; those rows are excluded from the result by the mask.
        has_pos  = n_pos > 0
        _all_nan = np.all(np.isnan(f1), axis=1)          # rows where f1 is entirely NaN
        _f1_safe = f1.copy()
        _f1_safe[_all_nan, 0] = 0.0                       # dummy non-NaN to suppress error
        has_valid_f1 = has_pos & ~_all_nan
        best_f1  = np.where(has_valid_f1, np.nanmax(_f1_safe,    axis=1), np.nan).astype(np.float32)
        best_bin = np.where(has_valid_f1, np.nanargmax(_f1_safe,  axis=1), -1  ).astype(np.int32)
        best_thr = np.where(
            has_valid_f1,
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
    # Vocabulary growth (Danbooru vocab expansion during training)
    # ------------------------------------------------------------------

    def grow(self, new_vocab_size: int) -> None:
        """Resize all per-tag arrays to ``new_vocab_size`` (zero-padding new tags).

        Called when the vocabulary expands mid-training so the accumulator stays
        aligned with the grown head/labels. New tags start with empty histograms
        (best_f1 = NaN until they accumulate data), matching a fresh tag.
        """
        if new_vocab_size <= self.vocab_size:
            return
        pad = new_vocab_size - self.vocab_size

        def _pad2d(a: np.ndarray) -> np.ndarray:
            return np.concatenate(
                [a, np.zeros((pad, self.n_bins), dtype=a.dtype)], axis=0
            )

        self.pos_hist_cur    = _pad2d(self.pos_hist_cur)
        self.total_hist_cur  = _pad2d(self.total_hist_cur)
        self.pos_hist_prev   = _pad2d(self.pos_hist_prev)
        self.total_hist_prev = _pad2d(self.total_hist_prev)
        self.pos_hist_pp     = _pad2d(self.pos_hist_pp)
        self.total_hist_pp   = _pad2d(self.total_hist_pp)
        self.tag_count = np.concatenate(
            [self.tag_count, np.zeros(pad, dtype=self.tag_count.dtype)]
        )
        self.tag_count_epoch_start = np.concatenate(
            [self.tag_count_epoch_start, np.zeros(pad, dtype=self.tag_count_epoch_start.dtype)]
        )
        self.last_epoch_delta = np.concatenate(
            [self.last_epoch_delta, np.zeros(pad, dtype=self.last_epoch_delta.dtype)]
        )
        self.vocab_size = new_vocab_size

    # ------------------------------------------------------------------
    # Deficiency selection (low-F1 Danbooru augmentation feed)
    # ------------------------------------------------------------------

    def deficient_tag_indices(
        self,
        f1_threshold: float,
        top_k: int,
        epoch_boundary: bool = False,
    ) -> List[int]:
        """Return vocab indices of existing tags whose per-tag F1 is deficient.

        A tag qualifies when it has a *valid* (non-NaN) ``best_f1`` below
        ``f1_threshold``.  Tags with NaN F1 are excluded on purpose: early in
        training most tags are NaN simply because their batches have not been
        seen yet (not because they are genuinely deficient), so including them
        would flood the augmentation feed with noise.  Established vocab tags
        always have data, so NaN is transient.

        No minimum positive-count gate is applied — a tag with even a single
        positive yields a (noisy) valid F1 and is eligible, matching the
        intent of surfacing under-collected tags.

        Results are sorted by F1 ascending (worst first) and capped at
        ``top_k``.
        """
        if top_k <= 0:
            return []
        m = self.compute_metrics(epoch_boundary=epoch_boundary)
        best_f1 = m["best_f1"]  # [V] float32, NaN where insufficient data
        valid = ~np.isnan(best_f1)
        deficient = valid & (best_f1 < f1_threshold)
        idx = np.where(deficient)[0]
        if idx.size == 0:
            return []
        order = np.argsort(best_f1[idx])  # ascending: worst F1 first
        return idx[order][:top_k].astype(int).tolist()

    def deficient_train_count_indices(
        self,
        top_k: int,
        min_deficit_ratio: float = 0.3,
        min_per_epoch: int = 10,
    ) -> List[int]:
        """Return vocab indices of tags under-exposed relative to their current rate.

        For each tag, ``expected = last_epoch_delta * epochs_elapsed`` is what its
        cumulative ``tag_count`` would be had it been present (at its current
        per-epoch exposure rate) since epoch 1. The deficit is
        ``max(0, expected - tag_count)`` and the deficit *ratio* is
        ``deficit / expected`` (fraction under-exposed). A tag qualifies when:

          - it has a meaningful current rate (``last_epoch_delta >= min_per_epoch``
            — filters noise from tags with a handful of exposures), and
          - ``deficit_ratio >= min_deficit_ratio``.

        This targets late-joined / newly-grown tags (positive deficit) while
        excluding genuinely-rare-but-stable tags (deficit≈0; the low-F1 path
        covers those). Returns indices sorted by deficit descending, capped at
        ``top_k``. Empty until at least 2 epochs have completed (need a stable
        per-epoch rate and a non-trivial expected baseline).
        """
        if top_k <= 0 or self.epochs_elapsed < 2:
            return []
        delta = self.last_epoch_delta.astype(np.float64)
        tc = self.tag_count.astype(np.float64)
        expected = delta * float(self.epochs_elapsed)
        with np.errstate(invalid="ignore", divide="ignore"):
            deficit = np.maximum(0.0, expected - tc)
            deficit_ratio = np.where(expected > 0, deficit / expected, 0.0)
        eligible = (self.last_epoch_delta >= int(min_per_epoch)) & \
                   (deficit_ratio >= float(min_deficit_ratio))
        idx = np.where(eligible)[0]
        if idx.size == 0:
            return []
        order = np.argsort(deficit[idx])[::-1]  # descending: largest deficit first
        return idx[order][:top_k].astype(int).tolist()

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

    def compute_calibration_table(
        self,
        epoch_boundary: bool = False,
        method: str = "jeffreys",
        eps: float = 0.5,
        prior_strength: float = 10.0,
    ) -> np.ndarray:
        """Compute per-tag per-bin calibrated posterior probabilities.

        Supported methods:

        ``"jeffreys"`` (default):
            Jeffreys-prior / Laplace smoothing per bin:
                calib[v,b] = (pos[v,b] + eps) / (total[v,b] + 2*eps)
            Empty bins fall back to the tag's marginal frequency π[v].
            Correct for bimodal discriminative models: bins with 0 negatives
            and ≥1 positive yield a high posterior (e.g. 0.75–0.90) instead
            of the artificially low value produced by the Beta-BB prior.

        ``"beta_bb"`` (legacy):
            Beta-Binomial smoothing with tag-specific prior:
                π[v]       = n_pos[v] / n_total[v]
                α[v]       = π[v] * prior_strength
                β[v]       = (1 − π[v]) * prior_strength
                calib[v,b] = (pos[v,b] + α[v]) / (total[v,b] + α[v] + β[v])
            Empty bins naturally yield π[v].

        Returns float16 [vocab_size, n_bins].  No NaN values.
        """
        pos_h, total_h, _ = self._merged(epoch_boundary)
        pos_f   = pos_h.astype(np.float32)
        total_f = total_h.astype(np.float32)

        n_pos_tag   = pos_f.sum(axis=1, keepdims=True)    # [V, 1]
        n_total_tag = total_f.sum(axis=1, keepdims=True)  # [V, 1]
        pi = np.where(n_total_tag > 0, n_pos_tag / n_total_tag, 0.0)  # [V, 1]

        if method == "beta_bb":
            alpha = pi * prior_strength
            beta  = (1.0 - pi) * prior_strength
            calib = (pos_f + alpha) / (total_f + alpha + beta)
        else:  # "jeffreys"
            calib = (pos_f + eps) / (total_f + 2.0 * eps)
            # Empty bins: fall back to marginal base rate
            calib = np.where(total_f > 0, calib, pi)

        return calib.astype(np.float16)

    def save(
        self,
        path: str,
        epoch_boundary: bool = False,
        tag_names: Optional[List[str]] = None,
        hard_lo: float = 0.25,
        hard_hi: float = 0.75,
        calib_method: str = "jeffreys",
        calib_eps: float = 0.5,
        calib_prior_strength: float = 10.0,
    ) -> None:
        """Save histograms + derived metrics to a compressed .npz file."""
        pos_h, total_h, total_images = self._merged(epoch_boundary)
        metrics = self.compute_metrics(epoch_boundary=epoch_boundary,
                                       hard_lo=hard_lo, hard_hi=hard_hi)

        save_kwargs: dict = {
            # Merged window histograms (used by inference and restore)
            "pos_hist":            pos_h.astype(np.float32),
            "total_hist":          total_h.astype(np.float32),
            "calibration_table":   self.compute_calibration_table(
                                       epoch_boundary,
                                       method=calib_method,
                                       eps=calib_eps,
                                       prior_strength=calib_prior_strength),
            "calib_method":        np.array([calib_method],          dtype=object),
            "calib_eps":           np.array([calib_eps],             dtype=np.float32),
            "calib_prior_strength": np.array([calib_prior_strength], dtype=np.float32),
            "tag_count":           self.tag_count,
            # Train-count deficit tracking (cumulative across resumes)
            "tag_count_epoch_start": self.tag_count_epoch_start,
            "last_epoch_delta":    self.last_epoch_delta,
            "epochs_elapsed":      np.array([self.epochs_elapsed], dtype=np.int64),
            "total_images":        np.array([total_images],          dtype=np.int64),
            "total_images_all":    np.array([self.total_images_all], dtype=np.int64),
            "n_bins":              np.array([self.n_bins],  dtype=np.int32),
            "hard_lo":             np.array([hard_lo],      dtype=np.float32),
            "hard_hi":             np.array([hard_hi],      dtype=np.float32),
            # Individual slot sizes — allow restore_from_npz to rebuild the window
            "total_images_cur":    np.array([self.total_images_cur],  dtype=np.int64),
            "total_images_prev":   np.array([self.total_images_prev], dtype=np.int64),
            "total_images_pp":     np.array([self.total_images_pp],   dtype=np.int64),
        }
        save_kwargs.update({k: v for k, v in metrics.items()})

        if tag_names is not None:
            # Store as object array so np.load can round-trip strings
            save_kwargs["tag_names"] = np.array(tag_names, dtype=object)

        np.savez_compressed(path, **save_kwargs)

    def restore_from_npz(self, path: str) -> bool:
        """Restore accumulator state from a previously saved .npz file.

        The saved file stores the *merged* window histograms (cur+prev+pp*w).
        On restore these are loaded into the ``prev`` slot; ``cur`` and ``pp``
        are left zeroed.  After the next epoch rotation, pp will be populated
        with the restored data, giving a correct two-epoch sliding window.

        ``total_images_prev`` and ``total_images_pp`` are restored from the
        saved slot-size keys (added in the three-slot design) or estimated
        from ``total_images`` for older files.

        Returns True on success, False if the file is missing or incompatible.
        """
        if not os.path.isfile(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)

            saved_n_bins = int(data["n_bins"].item() if data["n_bins"].ndim == 0
                               else data["n_bins"][0])
            if saved_n_bins != self.n_bins:
                print(
                    f"[TagMetricsAccumulator] restore_from_npz: n_bins mismatch "
                    f"(saved={saved_n_bins}, current={self.n_bins}) — skipping restore"
                )
                return False

            pos_h   = data["pos_hist"].astype(np.int32)    # [V_saved, K]
            total_h = data["total_hist"].astype(np.int32)  # [V_saved, K]
            V_saved = pos_h.shape[0]

            # Vocabulary may have grown since the checkpoint was saved.
            # Restore only the rows that exist in both old and new vocab.
            V_restore = min(V_saved, self.vocab_size)

            self.pos_hist_prev[:V_restore]   = pos_h[:V_restore]
            self.total_hist_prev[:V_restore] = total_h[:V_restore]

            # Restore slot image counts
            def _load_int(key: str, fallback: int) -> int:
                if key not in data.files:
                    return fallback
                v = data[key]
                return int(v.item() if v.ndim == 0 else v[0])

            _ti = data["total_images"]
            total_images_merged = int(_ti.item() if _ti.ndim == 0 else _ti[0])

            self.total_images_prev = _load_int("total_images_prev", total_images_merged)
            self.total_images_pp   = _load_int("total_images_pp",   0)
            # cur is zeroed — total_images_cur stays 0

            # tag_count: all-epoch cumulative
            if "tag_count" in data.files:
                tc = data["tag_count"].astype(np.int32)
                self.tag_count[:min(len(tc), self.vocab_size)] = tc[:min(len(tc), self.vocab_size)]

            # Train-count deficit tracking (cumulative across resumes)
            if "tag_count_epoch_start" in data.files:
                _ts = data["tag_count_epoch_start"].astype(np.int32)
                self.tag_count_epoch_start[:min(len(_ts), self.vocab_size)] = _ts[:min(len(_ts), self.vocab_size)]
            else:
                # Older checkpoint: seed snapshot to current tag_count so the next
                # completed epoch yields a sane (non-inflated) delta.
                np.copyto(self.tag_count_epoch_start, self.tag_count)
            if "last_epoch_delta" in data.files:
                _ld = data["last_epoch_delta"].astype(np.int32)
                self.last_epoch_delta[:min(len(_ld), self.vocab_size)] = _ld[:min(len(_ld), self.vocab_size)]
            if "epochs_elapsed" in data.files:
                _ee = data["epochs_elapsed"]
                self.epochs_elapsed = int(_ee.item() if _ee.ndim == 0 else _ee[0])

            # total_images_all: all-epoch image count
            if "total_images_all" in data.files:
                _tia = data["total_images_all"]
                self.total_images_all = int(_tia.item() if _tia.ndim == 0 else _tia[0])
            else:
                self.total_images_all = self.total_images_prev

            return True
        except Exception as e:
            print(f"[TagMetricsAccumulator] restore_from_npz failed ({path}): {e}")
            return False

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
