# Timestep Distribution Morphing (resume-time)

Status: design only — nothing below is implemented yet.

## Problem

`timestep_sampling` is read once per run (`BaseTrainer.train`,
`backend/core/training/base_trainer.py:14849`) and turned into a single
`TimestepSampler` that is then fixed for the whole run. When a run is resumed
with a *different* distribution — uniform → logit-normal, a shifted `mean`, a
narrowed `[min, max]` — the change takes effect on the first step after resume.

The loss a diffusion model reports is strongly t-dependent, so the step-to-step
loss series jumps discontinuously at that point even if the weights are fine.
Worse, the *gradient* distribution changes in one step: the optimizer's moment
estimates (and any EMA / plateau / spike-detection state keyed off loss scale —
`grad_spike_log.py`, `lr_triggers.py`) were accumulated under the old t-density
and are, for a few hundred steps, estimates of a distribution that no longer
exists. That is the damage mechanism this feature addresses.

The fix is to move the sampling density from the old configuration to the new
one over a user-specified number of steps instead of instantaneously.

## Scope

- Backend: a wrapper sampler + resume-time resolution of the "from" endpoint +
  persistence of the morph anchor in `state.json`.
- Frontend: configuration UI under the existing `timestep_sampling` section, and
  a live readout of the *current* distribution in the training monitor.
- Metrics: morph progress and the realised t-statistics as `extra_metrics`
  series (no DB schema change — see
  `docs/guides/TRAINING_DIAGNOSTICS_AND_AUXILIARY_LOSSES.md` and
  `core/training/metric_registry.py`).

Non-goals: changing any architecture's timestep convention; morphing anything
other than `timestep_sampling` (LR already has its own retarget machinery in
`lr_schedules.py` / `LrScheduleRetargetPanel.tsx`); scheduled *mid-run*
distribution curricula that are not tied to a resume.

## Mechanism

Two ways to interpolate between two distributions A (old) and B (new), both
parameterised by λ ∈ [0, 1]:

**1. Quantile interpolation (default).** Draw `u ~ U(0,1)` once and emit

```
t(u; λ) = (1 − λ)·icdf_A(u) + λ·icdf_B(u)
```

This is the Wasserstein-2 geodesic (displacement interpolation) between the two
laws. Concretely, equally ranked mass in A and B is coupled and moves between
the two quantiles rather than being independently selected from either endpoint.
This does not guarantee unimodality or the absence of atoms. `t(·; λ)` is monotone
in `u` because it is a convex combination of two monotone functions, which means
the wrapper has a valid `icdf` whenever both endpoints do, and
`sample_stratified` (`timestep_sampler.py:105`) remains available. The
grad-t-cosine probe's changing median is handled separately below.

**2. Mixture (fallback, and selectable).**

```
t ~ A with prob (1 − λ),  t ~ B with prob λ
```

The probability law is then linear in λ, which is the more obvious reading of
"morph". For well-separated modes this can retain the old mode while a second
one grows, although bimodality is not guaranteed for overlapping endpoints. For
a narrowing range change
(`[0,1] → [0.2,0.8]`) mixture is arguably the honest interpolation; for a mode
shift it trains two disjoint regimes at once, which is closer to the thing we
are trying to avoid. A general mixture has no closed-form quantile, so the
wrapper deliberately exposes no `icdf`; under it
`sample_stratified` falls back to independent draws exactly as it already does
for `BetaTimestepSampler`, and the grad-t-cosine probe's median split is
estimated numerically instead of analytically.

Default: `quantile`. `mixture` is required when either endpoint has no `icdf`
(today: `BetaTimestepSampler`) — the wrapper detects that and switches with a
logged warning rather than failing.

### λ schedule and its step axis

λ is a function of a dedicated count of **successful optimizer updates**:

```
p = clamp((optimizer_update_step − morph_start_update) / morph_steps, 0, 1)
λ = p                              # curve: "linear"
λ = 0.5·(1 − cos(π·p))             # curve: "cosine"  (C¹ at both ends)
```

`global_step` is deliberately not used. In this trainer it counts
forward/backward iterations (including individual MNT iterations), while an
optimizer update happens only at the gradient-accumulation boundary; fused
backward may make that boundary every iteration. Using `global_step` would make
the same `steps: 2000` finish after a different number of weight updates when
MNT, gradient accumulation, or the optimizer backend changes.

`optimizer_update_step` increments only after weights were actually updated. A
CUDA-recovery iteration that advances the LR scheduler without updating weights,
or a GradScaler step skipped for non-finite gradients, does not advance it. It is
persisted in checkpoint state; it must not be
reconstructed from `global_step`, `scheduler_step`, or their ratio because all
three can diverge after skipped work.

`cosine` is the default: the derivative of λ with respect to update position is
zero at both endpoints, so the interpolation parameter departs from A and
arrives at B without a slope discontinuity. This does not claim that every
endpoint law has a smooth density: clamped normal has boundary atoms, for
example. `linear` is kept because it is easier to reason about in a chart.

λ is fixed for one gradient-accumulation window, so every draw contributing
to one optimizer update comes from the same law. When an MNT window crosses an
optimizer boundary, its timestep block is partitioned at that boundary and
`sample_stratified` is called once per partition under the applicable λ. This
slightly changes the scope of stratification during a morph, but preserves the
stronger invariant that one optimizer update never mixes two morph positions.
With fused backward the effective accumulation window is one iteration.

## Configuration

`timestep_sampling` gains an optional nested `morph` block. The outer block
keeps meaning "the distribution this run trains at" (the morph *target*):

```yaml
timestep_sampling:
  distribution: logit_normal
  mean: 0.5
  std: 1.0
  min_timestep: 0.0
  max_timestep: 1.0
  morph:
    enabled: true
    steps: 2000               # successful optimizer updates in the transition
    curve: cosine             # cosine | linear
    interpolation: quantile   # quantile | mixture
    from: null                # optional explicit source; null => resolve from the run
```

Defaults live in `backend/api/param_defaults.py` beside the existing
`timestep_sampling` entry (`TRAINING_DEFAULTS["timestep_sampling"]`), and the
per-arch map `TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH` is untouched — `morph` is
absent from every per-arch default, i.e. off.

Validation at the API boundary rejects `steps <= 0`, unknown curve or
interpolation values, non-finite sampler parameters, `std <= 0`, and non-positive
Beta parameters. Before equality checks or persistence, endpoint configs are
canonicalised: distribution aliases are normalised, omitted sampler defaults are
materialised, irrelevant keys and `morph` are removed, and numeric values are
normalised. Thus `lognormal` and `logit_normal`, or an omitted default and its
explicit value, do not start a pointless morph.

### Resolving `from`

`from: null` (the normal case) means "whatever this run was previously training
at". Resolution order:

1. The state paired with the **checkpoint actually selected for resume** (see
   below). If it records an in-flight morph, that record is authoritative. An
   unchanged target continues it; a changed target uses its currently effective
   law as the new source.
2. An explicit `morph.from`, when there is no in-flight record. Explicit `from`
   does not override an in-flight record; abandoning that history requires
   disabling morph for one resume or starting from a checkpoint without it.
3. `{output_dir}/{run_name}_config.yaml`, the per-run config store already used
   for pinned values (`base_trainer.py:4787`), read *before* the API overwrites
   it with the new config. This requires `routes.py` to read the old file back
   before `save_config` clobbers it on resume and to stash it into the new
   config's `timestep_sampling.morph.from`. This is only a fallback source; it
   never supersedes checkpoint state. If the old config omitted
   `timestep_sampling`, resolve the same per-architecture default that
   `BaseTrainer.train` would have used.
4. Nothing found → the morph is a no-op. Log, set `enabled: false`, train at the
   target distribution. This is also what a *fresh* run with `morph.enabled`
   does.

Resolution is consequently split across two seams. The API captures the old
config before overwriting it, but the trainer resolves authoritative state only
after it knows which `latest` fallback or explicit checkpoint was actually
loaded. The trainer must load the paired state before constructing the sampler,
printing its startup diagnostics, or arming the grad-t-cosine probe. A crash
resume that performs no intervening API update therefore still continues the
same morph.

If the resolved `from` equals the target, the morph is skipped with a log line;
a user who re-resumes without changing the distribution should not get 2000
steps of pointless wrapper.

### Persistence and re-resume

`state.json` (saved with the checkpoint; the loader is around
`base_trainer.py:15359`) gains:

```json
"timestep_morph": {
  "version": 1,
  "start_update": 10300,
  "steps": 2000,
  "curve": "cosine",
  "interpolation": "quantile",
  "from": { "...": "sampler config" },
  "to":   { "...": "sampler config" }
}
```

The top-level state also records `optimizer_update_step`. Both values describe
completed successful updates. The state is saved from the same sampler instance
used for training, rather than reconstructed from the current YAML.

Behaviour on resume while a morph is in flight:

- **target unchanged** → continue the same morph: reuse `start_update`, so a
  crash 800 updates into a 2000-update morph resumes at λ≈0.4, not λ=0.
- **target changed again** → start a new morph whose `from` is the *currently
  effective* law: the wrapper frozen at the present λ. A frozen quantile
  wrapper retains an exact `icdf`; a frozen mixture retains exact `sample`
  semantics but has no `icdf`. Therefore a requested quantile morph whose
  source contains a mixture falls back to mixture with a warning, just like a
  Beta endpoint. It is not described as an exact quantile interpolation.
- **morph finished** (`optimizer_update_step >= start_update + steps`) → the record is
  dropped from `state.json` on the next save.

Sampler expressions are serialised recursively with `version` and `kind`
fields. Nesting depth is capped at four. A fifth in-flight retarget flattens the
frozen source into a deterministic 4097-point piecewise-linear quantile table.
The table is made from 262144 CPU samples inside an isolated, deterministically
seeded CPU RNG context; construction saves and restores global CPU RNG state and
never touches the training CUDA generator. The seed is stored for auditability.
Flattening is approximate and is
logged as such. The table itself, rather than the samples or seed alone, is
persisted so a later software version reproduces the same source law.

`start_update` is always the successful optimizer-update count at which the new
configuration was first seen, so a morph is anchored to the resume, not to the
run.

## Backend implementation sketch

New class in `backend/core/training/timestep_sampler.py`:

```python
class MorphingTimestepSampler(TimestepSampler):
    def __init__(self, source, target, steps, curve="cosine",
                 interpolation="quantile", start_update=0): ...
    def set_optimizer_update_step(self, step: int) -> None: ...
    @property
    def lam(self) -> float: ...
    def sample(self, batch_size, device): ...
    def icdf(self, u): ...            # quantile mode only
    def is_finished(self) -> bool: ...
    def state(self) -> dict: ...      # the state.json record above
```

`min_timestep` / `max_timestep` on the wrapper are `min(A.min, B.min)` /
`max(A.max, B.max)` — the union, since draws during the transition can land
anywhere between the two supports. Under quantile interpolation the realised
range is in fact the λ-interpolated one, but consumers treat these fields as
conservative support bounds and the union is the only choice that remains valid
for the whole transition.

Wiring:

- `TimestepSampler.from_config` recognises `config["morph"]` and builds the
  wrapper (endpoints built by recursing on the same function with `morph`
  stripped). A separate state restore path reconstructs a versioned sampler
  expression after the selected checkpoint state is loaded.
- `BaseTrainer.train` sets the sampler's optimizer-update position at the start
  of each accumulation window. The MNT stratified path is partitioned when it
  crosses an optimizer boundary, as described above; ordinary `.sample(...)`
  call sites remain unchanged.
- `save_training_state` stores both `optimizer_update_step` and the active
  sampler's `state()`. Every code path that actually changes weights increments
  the counter exactly once; scheduler-only recovery paths do not.
- The startup log block (`base_trainer.py:14878`) prints the morph: both
  endpoints, curve, update window, resolved `start_update`; and
  `log_timestep_distribution_median` runs for *both* endpoints so the
  clean/noisy side of each is on the record.

Risk note: quantile mode draws a uniform variate and applies `icdf`. Plain normal
and logit-normal samplers use `torch.randn`, so even at λ=0 the samples are equal
in law, not bit-identical, and the RNG stream differs. Mixture consumes an
additional branch draw. Resume across a distribution change therefore makes no
draw-for-draw identity guarantee; reproducibility means reproducing the same
morph state and RNG state under the same implementation version.

### Grad-t-cosine probe

The probe's split follows the active distribution rather than being captured
once at startup. At the start of each MNT partition it receives the active
sampler median. Quantile mode uses `icdf(0.5)`. Mixture mode uses the weighted
empirical CDF of deterministic endpoint quantile tables built at sampler setup;
this does not consume the training RNG. All passes are consequently classified
as the lower or upper half of the law that produced them. If updating the split
is not implemented, the probe must be disabled during a morph rather than
silently reporting buckets under a stale threshold.

## Metrics and display

New `extra_metrics` (registered in `core/training/metric_registry.py`; no DB
schema change — these four chart series use `log_extra_metric` only):

| name | family / scale_group | meaning |
|---|---|---|
| `timestep_morph_lambda` | bounded_diagnostic / unit_interval | λ for the accumulation window; absent when no morph is active |
| `timestep_batch_mean` | bounded_diagnostic / unit_interval | mean of t drawn in this forward/backward iteration |
| `timestep_batch_p10`, `timestep_batch_p90` | bounded_diagnostic / unit_interval | per-iteration drawn window; equal to the draw at batch size 1 |

`timestep_batch_*` are logged whether or not a morph is active. They use the
same per-MNT-iteration `global_step` as loss metrics, so they can be joined
without assigning a completed window retrospectively to one of its earlier
steps. Whole-MNT-window statistics, if later wanted, use separately named
metrics rather than changing these series' aggregation semantics.

Frontend:

1. **Config** (`TrainingConfig.tsx`, `trainingConfigDefinitions.tsx`,
   `trainingParams.ts`, `frontend/src/utils/api.ts` type, `openapi.yaml`): a
   "Morph on resume" subsection inside the existing timestep block — enable,
   steps, curve, interpolation. Only meaningful on a resumed run; the UI says so
   rather than hiding the controls, because the value has to be set *before* the
   resume starts.
2. **Runtime status**: the existing training-status response and live WebSocket
   payload gain `timestep_morph_status`, containing the resolved source and
   target sampler expressions, requested and effective interpolation, fallback
   reason, curve, `start_update`, `optimizer_update_step`, `steps`, and λ. This
   is the authoritative view. The run config is only user intent and cannot
   describe an in-flight nested or flattened source. `openapi.yaml` and the
   frontend API/WebSocket types change with this payload.
3. **Monitor** (`TrainingMonitor.tsx`): a compact readout of the current
   distribution — reuse `TimestepDistributionGraph.tsx`, drawing the source and
   target PDFs as faint lines and the current interpolated PDF solid, captioned
   `logit_normal(0.0, 1.0) → logit_normal(0.5, 1.0) — 43% (update 860 / 2000)`.
   The readout uses `timestep_morph_status`; the metric series is for charting,
   not state reconstruction.
   `TimestepDistributionGraph` needs one addition: an overlay curve given λ and
   two parameter sets. For quantile mode the intermediate PDF has no closed
   form — evaluate `t((i+0.5)/512;λ)` and histogram those deterministic
   midpoint quantiles, which is ample for a sparkline. Before adding the overlay,
   its endpoint evaluators
   must match the backend: logit-normal and Beta outputs are affine-scaled to
   `[min_timestep,max_timestep]`, normal clamp atoms are represented at the
   boundaries, and custom/quantile-table samplers are supported. All curves in
   an overlay share one y-scale.
4. **Chart**: the four new series appear automatically through the
   extra-metrics channel, get legend entries from the registry, and are
   toggleable in `MetricSeriesPicker`.

## Verification plan

- Unit: λ schedule endpoints and monotonicity on the successful-update axis;
  skipped/scheduler-only steps do not advance λ; all optimizer implementations
  increment exactly once per weight update; one accumulation window sees one λ;
  quantile-mode `icdf` monotone in
  `u` at λ ∈ {0, 0.5, 1}; at λ=0 and λ=1 the wrapper's draws match the endpoint
  sampler's law (KS on a large sample); `sample_stratified` marginals under the
  wrapper and across an MNT/accumulation boundary; mixture fallback when an
  endpoint or frozen source lacks `icdf`; canonical equality; `state()`
  round-trip including re-resume, nesting, version refusal, and deterministic
  flattening. Tests assert equality in law, not bit identity.
- Integration: a short run with `steps: 2` proves both endpoints, config
  threading, persisted update count, metrics, runtime status, and completion.
- Resume: a synthetic `state.json` with an in-flight morph resumes at the right
  λ rather than restarting the transition; cover `latest` fallback, an explicit
  older checkpoint, no API update between crash and resume, a changed target,
  and a scheduler-only recovery step.
- Behavioural acceptance: from one checkpoint and fixed seed, compare immediate
  switching with a representative morph. Record the resume-boundary loss and
  update-norm jump, grad-spike/plateau-trigger firings, and short post-morph loss.
  The feature is not considered validated merely because it samples the intended
  law: it must reduce the targeted transition spike without degrading the
  post-morph objective over the measured window. Exact numeric thresholds are
  recorded with the experiment because they depend on architecture and batch
  regime; the comparison protocol and raw series are retained alongside the
  result.
