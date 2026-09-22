# Live training sample settings

The Training Monitor Samples pane edits future sample jobs without changing the
training objective. `GET/PUT /api/v1/training/runs/{id}/sample-config` exposes a
small, architecture-neutral subset: prompt list, scheduled interval, width,
height, inference steps, CFG scale, and seed. Architecture-specific sampling
keys remain in the original config and are not edited by this panel.

The PUT accepts only a live run and requires `expected_revision`; a stale edit
returns 409. It updates the run's config YAML and the on-disk config used by
resume, then atomically publishes `.sample_live_config.json` in the output
directory. The trainer reads that run-scoped revision before its next sample
decision, replaces all editable values together, and writes
`.sample_live_applied.json`. GET distinguishes desired and applied revisions.
The sample itself uses one snapshot; a change during generation affects only a
later sample. The step-0 image is never rewritten by a live edit. PNG metadata
records `sample_settings_revision` alongside the actual prompt and generation
parameters. A sample error after a live edit emits a training warning without
terminating the run; the pre-existing fail-fast rule remains for untouched
scheduled sampling.

This is intentionally separate from LR controls and timestep distribution.
LR commands must apply before forward because fused backward can advance the
optimizer inside backward. The timestep controller runs on successful optimizer
updates. Sample settings have no optimizer effect and use their own replacement
state rather than the LR command queue. The JSON atomic-write primitive is
shared; the API semantics and application seams are not.

An already-running trainer launched before this code is deployed cannot poll
the new revision; use it on a subsequently started training process. The
backend is not restarted to install this feature while an owner run is active.
