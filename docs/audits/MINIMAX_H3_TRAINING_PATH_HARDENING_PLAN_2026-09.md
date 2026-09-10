# MiniMax-H3 training-path hardening plan (2026-09)

## Scope

Harden three statically confirmed failure modes without changing the supported
MiniMax-H3 objective, model weights, sampling path, or default configuration:

1. Refuse empty captions and caption processing that can erase the whole caption
   before any model component is loaded.
2. Refuse source/target/instruction reference datasets because MiniMax-H3 training
   does not yet pack reference conditioning rows.
3. Exclude zero-padded tails of short audio windows from the audio loss.

## Change units

### H1 — caption contract

Add a MiniMax-H3 dataset preflight after dataset resolution and before dataset
loading. Refuse a positive whole-caption dropout rate and any selected item with
no non-empty caption. Keep token/tag processing legal when it cannot erase the
entire caption. Pin direct YAML and database-backed entry points with CPU tests.

### H2 — instruction-reference contract

At the same pre-load boundary, detect reference-mode dataset items and refuse the
run with a message that distinguishes target-only still training from unsupported
source-conditioned Ref2VA training. Do not reinterpret or silently discard the
source image. Keep ordinary still-image datasets accepted.

### H3 — short-audio row mask

Carry a per-row validity mask beside collated MiniMax-H3 audio latents. A missing
audio sample has no valid rows; a short window marks only its original rows valid.
Apply the mask to the per-row audio MSE and denominator while preserving the
existing all-silent exactly-zero graph branch.

## Verification

- Run focused CPU-only unit tests for each change unit.
- Run the complete `backend/tests/minimax_h3_training_test.py` file.
- Compile and import every edited backend module with CUDA initialization stubbed
  as required by `AGENTS.md`.
- Do not start or restart either server and do not run a real model, VAE, or CUDA
  training step.

## Commit boundaries

The plan is committed first. H1, H2, and H3 are then committed independently so
each behavior change can be reviewed or reverted without taking the others.
