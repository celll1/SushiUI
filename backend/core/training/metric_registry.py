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
  - axis:   optional. Set to "right" to render the series against a separate,
            independently-auto-scaled secondary Y-axis instead of pooling it
            into the primary (loss-scale) Y-range. Use this for metrics whose
            magnitude is orders of magnitude away from loss (e.g. learning
            rate, ~1e-4) and would otherwise be an invisible flat line.
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
    # Outpaint ControlNet: raw MSE over the known (anchored) region only --
    # the loss-vs-timestep instrumentation's live-chart counterpart to
    # gen_loss/seam_loss (full per-sample/per-timestep breakdown goes to the
    # loss_vs_t.jsonl sidecar; see scratchpad "Outpaint ControlNet:
    # loss-vs-timestep instrumentation" design doc).
    "known_loss": {"label": "Known region", "color": "#34d399", "dashed": True},
    # Actually-applied per-step learning rate (optimizer.param_groups[0]['lr']),
    # logged for every trainer (LoRA/full-FT/ControlNet share the same
    # BaseTrainer.train() loop). Schedules can now be non-constant
    # (plateau_cosine_floor), so this is a real curve, not just a flat line --
    # but at ~1e-4 it's 3+ orders of magnitude below loss (~0.03), so it needs
    # its own axis rather than the shared pooled Y-range.
    "lr": {"label": "Learning Rate", "color": "#38bdf8", "dashed": False, "axis": "right"},
    # SenseNova shared-prefix four-phase: cumulative count of generation
    # backwards whose understanding gradient a skipped batch threw away. A
    # skip mid-window drops every backward the window had run, whose generation
    # updates already landed, so it is charted rather than only logged. A count,
    # not a loss, hence the secondary axis. Emitted only on a skip.
    "sn_und_grad_dropped": {"label": "SenseNova und grad dropped", "color": "#f87171",
                            "dashed": True, "axis": "right"},
    # SenseNova MoT phase eviction: per-step time and volume of the half swaps,
    # split by direction (see sensenova_phase_eviction's TRANSFER ACCOUNTING).
    # Seconds and GiB are both orders away from the loss scale, so they share
    # the secondary axis for the same reason "lr" does.
    #
    # The SECONDS change unit with sn_swap_overlap below: 0 means host wall time
    # around a blocking copy, and the two buckets sum to the step's transfer
    # term; 1 means CUDA event time on that direction's own stream, where the
    # two directions run concurrently and their sum EXCEEDS the transition's
    # wall. Read the two series against that flag, never across it.
    "sn_d2h_s": {"label": "SenseNova D2H (s)", "color": "#38bdf8", "dashed": False,
                 "axis": "right"},
    "sn_h2d_s": {"label": "SenseNova H2D (s)", "color": "#a78bfa", "dashed": False,
                 "axis": "right"},
    "sn_d2h_gib": {"label": "SenseNova D2H (GiB)", "color": "#22d3ee", "dashed": True,
                   "axis": "right"},
    "sn_h2d_gib": {"label": "SenseNova H2D (GiB)", "color": "#c084fc", "dashed": True,
                   "axis": "right"},
    # 1 while sensenova_mot_overlap_transfer is running the two-stream path.
    # Not decorative: it is what tells a reader which unit the two seconds
    # series above are in, including mid-run if a pin failure downgrades it.
    # AND-ed over the step's transitions, so the step that straddles such a
    # downgrade -- whose seconds are part event time, part host wall -- reads 0
    # rather than claiming a unit half of its total is not in.
    "sn_swap_overlap": {"label": "SenseNova swap overlapped", "color": "#facc15",
                        "dashed": True, "axis": "right"},
    # RUN-CUMULATIVE CUDA high-water (never reset), so these are monotone step
    # curves, not per-step usage: they say when the peak last moved, not what a
    # step cost. Emitted only alongside the swap counters above.
    "sn_peak_alloc_gib": {"label": "SenseNova peak allocated (GiB)", "color": "#34d399",
                          "dashed": False, "axis": "right"},
    "sn_peak_resv_gib": {"label": "SenseNova peak reserved (GiB)", "color": "#fb923c",
                         "dashed": False, "axis": "right"},
    # Per-component actual LRs (only emitted when a run trains more than one
    # optimizer param group at potentially-different LRs, e.g. UNet+TE1/TE2 or
    # +VisionEncoder runs -- see base_trainer.py's per-step logging site next
    # to "lr"). Keys are derived from _build_component_lr_list()'s component
    # names lowercased/stripped of non-alnum (see that method's docstring for
    # the exact name strings: "U-Net", "TE1", "TE2", "VisionEncoder",
    # "ControlNet"). Share the same secondary right axis as "lr" since they're
    # the same unit and often overlapping magnitude. Any component not listed
    # here (e.g. lr_controlnet) still renders via the frontend's generic
    # lr*-prefix -> secondary-axis fallback, just without a curated label.
    "lr_unet": {"label": "LR (U-Net)", "color": "#38bdf8", "dashed": False, "axis": "right"},
    "lr_te1": {"label": "LR (TE1)", "color": "#fb923c", "dashed": False, "axis": "right"},
    "lr_te2": {"label": "LR (TE2)", "color": "#c084fc", "dashed": False, "axis": "right"},
    "lr_visionencoder": {"label": "LR (Vision Encoder)", "color": "#4ade80", "dashed": False, "axis": "right"},
    # ---- VAE decoder fine-tune (network.type == "vae_decoder") ------------
    # Per-step loss components. The chart's primary Y-range is the total loss,
    # which these sum into (times their weights), so they stay on the main axis.
    "vae_recon_loss": {"label": "VAE recon (MSE+L1)", "color": "#60a5fa", "dashed": True},
    "vae_lpips_loss": {"label": "VAE LPIPS", "color": "#f472b6", "dashed": True},
    "vae_dc_loss": {"label": "VAE YCbCr DC", "color": "#facc15", "dashed": True},
    # Only emitted when pattern_weight > 0 (opt-in; see param_defaults.py).
    "vae_pattern_loss": {"label": "VAE pattern", "color": "#c084fc", "dashed": True},
    # Only emitted when l_invented_weight > 0 (opt-in; see param_defaults.py).
    # The UNWEIGHTED term, in (8-bit levels)^2 inside flat windows. Its square
    # root is a RELATIVE TREND INDICATOR, NOT an absolute level: the value
    # carries the term's Weber photometric weight (0.16 bright .. 0.98 black)
    # and its channel weights, so it under-reads true invented luma by ~1.1x in
    # dark windows and ~2.5x in bright ones. Absolute levels against the 1/255
    # visibility bar come from the frozen g1flat harness only.
    "vae_invented_loss": {"label": "VAE invented HF", "color": "#22d3ee", "dashed": True},
    # Fraction of candidate windows that passed the flat test on that step.
    # Charted next to the term because the two are only interpretable together:
    # a falling loss with a collapsing coverage means the term stopped firing,
    # not that the decoder stopped inventing. It is a 0..1 fraction, so it goes
    # on the secondary axis rather than into the loss-scale pooled range.
    "vae_invented_cov": {"label": "VAE invented coverage", "color": "#818cf8",
                         "dashed": False, "axis": "right"},
    # Only emitted when the ENCODER is trainable: under a frozen encoder the
    # posterior KL is constant w.r.t. every trainable parameter and the term is
    # not constructed at all.
    #
    # This is the WEIGHTED contribution to the total loss, not the raw KL. The
    # raw value is 1e4-1e5, and charting it anywhere would wreck whichever axis
    # it landed on -- on the right axis in particular it would flatten
    # vae_val_psnr (~30) and vae_val_blockiness (~1.0), which are the only
    # signals that a fine-tune is going wrong. The weighted contribution shares
    # the magnitude of the other loss components, so it belongs with them on the
    # main axis; the raw KL stays in the per-step console log.
    "vae_kl_loss": {"label": "VAE KL (weighted)", "color": "#a3e635", "dashed": True},
    # Periodic held-out validation. These are the user's only signal that a
    # fine-tune is going wrong (PSNR falling = the decoder is drifting off the
    # data; blockiness rising above ~1.0 = it is manufacturing latent-cell grid
    # structure). Both live on the secondary axis: PSNR is ~25-35 dB and
    # blockiness ~1.0, orders of magnitude away from the loss scale.
    "vae_val_psnr": {"label": "Val PSNR (dB)", "color": "#34d399", "dashed": False, "axis": "right"},
    "vae_val_blockiness": {"label": "Val blockiness", "color": "#fb923c", "dashed": False, "axis": "right"},
    # ---- MiniMax-H3 (joint video + audio) ---------------------------------
    # Per-modality velocity MSE that sum (video + audio_loss_weight * audio)
    # into the total loss -- see minimax_h3_ops.train_step.
    "h3_video_loss": {"label": "H3 video loss", "color": "#60a5fa", "dashed": True},
    "h3_audio_loss": {"label": "H3 audio loss", "color": "#f472b6", "dashed": True},
    # Fraction of the batch whose item carried a real audio track this step (vs.
    # the noise-filled, zero-weighted fallback rows). A flat 0 for the whole run
    # means the audio term never actually saw data.
    "h3_audio_present": {"label": "H3 audio present", "color": "#facc15",
                         "dashed": False, "axis": "right"},
}
