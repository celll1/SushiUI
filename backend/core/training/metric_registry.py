"""Display metadata for bespoke, arch/method-specific training metrics.

These are the optional per-step scalars that only some trainers emit (REPA
alignment for MiniT2I, the generate-region-only MSE for outpaint ControlNet,
…). They are stored generically in ``TrainingMetrics.extra_metrics`` (a
``{name: float}`` JSON dict) rather than as dedicated DB columns, so adding a
new one requires NO schema change and NO API/param threading — the trainer just
calls ``self.log_extra_metric(name, value)`` and declares its chart appearance
here.

The ``/training/runs/{run_id}/metrics_db`` endpoint echoes the entries for the
metric names actually present in a run so the loss chart can label/colour the
series without a frontend code change. A metric with no entry here still renders
(falling back to its raw name + a dashed line), so registering is optional — it
only controls presentation.

Fields per entry:
  - label:  human-readable series name shown in the legend/tooltip.
  - color:  hex line colour (optional; the frontend hashes the name if absent).
  - dashed: draw as a dashed overlay line (default True for aux losses — they
            typically live on a different scale than the main loss and should
            not dominate the chart's Y-range pooling).
"""

EXTRA_METRIC_DEFS = {
    # REPA representation-alignment loss (MiniT2I). Formerly the dedicated
    # repa_loss column (backfilled into extra_metrics by auto_migrate).
    "repa_loss": {"label": "REPA", "color": "#f59e0b", "dashed": True},
    # Outpaint ControlNet: MSE over the generate region only (the learning
    # signal that matters for outpaint, isolated from the byte-identical known
    # region which the training masks out).
    "gen_loss": {"label": "Gen region", "color": "#a78bfa", "dashed": True},
    # Outpaint ControlNet: raw MSE over ONLY the 1-cell generate-side ring
    # adjacent to the known region (~2-3% of generate cells) -- gen_loss
    # averages this away, so this is the dedicated instrument for the seam band.
    "seam_loss": {"label": "Seam", "color": "#f43f5e", "dashed": True},
}
