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
laws. Concretely: the mass *moves* — a mode at t=0.3 slides toward t=0.7,
passing through 0.5 — and at every λ the intermediate law is a single coherent
distribution. `t(·; λ)` is monotone in `u` because it is a convex combination of
two monotone functions, which means the wrapper has a valid closed-form `icdf`
and `sample_stratified` (`timestep_sampler.py:105`) keeps working unchanged,
along with `_maybe_build_grad_t_cos_probe`'s median split
(`base_trainer.py:7200`).

**2. Mixture (fallback, and selectable).**

```
t ~ A with prob (1 − λ),  t ~ B with prob λ
```

The *density* is then linear in λ, which is the more obvious reading of "morph",
but the intermediate law is bimodal: the old mode does not move, it fades while
a second mode grows elsewhere. For a narrowing range change
(`[0,1] → [0.2,0.8]`) mixture is arguably the honest interpolation; for a mode
shift it trains two disjoint regimes at once, which is closer to the thing we
are trying to avoid. Mixture has no closed-form quantile, so under it
`sample_stratified` falls back to independent draws exactly as it already does
for `BetaTimestepSampler`, and the grad-t-cosine probe's median split is
estimated numerically instead of analytically.

Default: `quantile`. `mixture` is required when either endpoint has no `icdf`
(today: `BetaTimestepSampler`) — the wrapper detects that and switches with a
logged warning rather than failing.

### λ schedule

λ is a function of `global_step`:

```
p = clamp((global_step − morph_start_step) / morph_steps, 0, 1)
λ = p                              # curve: "linear"
λ = 0.5·(1 − cos(π·p))             # curve: "cosine"  (C¹ at both ends)
```

`cosine` is the default: the derivative of the density with respect to step is
zero at both endpoints, so neither the departure from A nor the arrival at B is
itself a discontinuity in the rate of change. `linear` is kept because it is the
easier thing to reason about when reading a chart.

λ is recomputed once per training step (not per MNT iteration, not per
micro-batch) so every draw inside one optimizer step comes from the same law.

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
    steps: 2000               # length of the transition in optimizer steps
    curve: cosine             # cosine | linear
    interpolation: quantile   # quantile | mixture
    from: null                # optional explicit source; null => resolve from the run
```

Defaults live in `backend/api/param_defaults.py` beside the existing
`timestep_sampling` entry (`TRAINING_DEFAULTS["timestep_sampling"]`), and the
per-arch map `TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH` is untouched — `morph` is
absent from every per-arch default, i.e. off.

### Resolving `from`

`from: null` (the normal case) means "whatever this run was previously training
at". Resolution order:

1. `state.json` written by the last checkpoint of this run (see below) — if it
   records an in-flight morph, that record wins entirely.
2. `{output_dir}/{run_name}_config.yaml`, the per-run config store already used
   for pinned values (`base_trainer.py:4787`), read *before* the API overwrites
   it with the new config. This requires `routes.py` to read the old file back
   before `save_config` clobbers it on resume and to stash it into the new
   config's `timestep_sampling.morph.from`. Resolution therefore happens at
   **run-update time in the API**, not in the trainer, which keeps the trainer's
   input a plain self-contained config.
3. Nothing found → the morph is a no-op. Log, set `enabled: false`, train at the
   target distribution. This is also what a *fresh* run with `morph.enabled`
   does.

If the resolved `from` equals the target, the morph is skipped with a log line;
a user who re-resumes without changing the distribution should not get 2000
steps of pointless wrapper.

### Persistence and re-resume

`state.json` (saved with the checkpoint; the loader is around
`base_trainer.py:15359`) gains:

```json
"timestep_morph": {
  "start_step": 41200,
  "steps": 2000,
  "curve": "cosine",
  "interpolation": "quantile",
  "from": { "...": "sampler config" },
  "to":   { "...": "sampler config" }
}
```

Behaviour on resume while a morph is in flight:

- **target unchanged** → continue the same morph: reuse `start_step`, so a crash
  at step 42000 of a 41200→43200 morph resumes at λ≈0.4, not λ=0.
- **target changed again** → start a new morph whose `from` is the *currently
  effective* law: the wrapper frozen at the present λ. Implementation is a
  nested wrapper, which is exact for both interpolation modes and costs one
  extra `icdf` call per draw. Nesting depth is bounded by refusing to nest more
  than 4 deep (log + flatten to a numerical quantile snapshot at that point; 4
  re-resumes inside one transition is already pathological).
- **morph finished** (`global_step ≥ start_step + steps`) → the record is
  dropped from `state.json` on the next save.

`start_step` is always the `global_step` at which the new configuration was
first seen, so a morph is anchored to the resume, not to the run.

## Backend implementation sketch

New class in `backend/core/training/timestep_sampler.py`:

```python
class MorphingTimestepSampler(TimestepSampler):
    def __init__(self, source, target, steps, curve="cosine",
                 interpolation="quantile", start_step=0): ...
    def set_global_step(self, step: int) -> None: ...   # sets self._lam
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
range is in fact the λ-interpolated one, but the base class validates against
the declared range and the union is the only choice that is never violated.

Wiring:

- `TimestepSampler.from_config` recognises `config["morph"]` and builds the
  wrapper (endpoints built by recursing on the same function with `morph`
  stripped).
- `BaseTrainer.train` calls `timestep_sampler.set_global_step(global_step)` once
  per step, next to where the other per-step schedule state is advanced. All
  existing `.sample(...)` call sites (`base_trainer.py:11740`, `:11750`,
  `:17965`, and the stratified path at `:7271`) are unchanged.
- The startup log block (`base_trainer.py:14878`) prints the morph: both
  endpoints, curve, step window, resolved `start_step`; and
  `log_timestep_distribution_median` runs for *both* endpoints so the
  clean/noisy side of each is on the record.

Risk note: in quantile mode the wrapper consumes exactly one `torch.rand` per
draw, like the plain samplers, so λ=0 is bit-identical to the unwrapped source.
Mixture mode consumes an extra draw and is therefore *not* bit-identical at λ=0.
That is acceptable (a resume across a config change already does not guarantee
draw-for-draw identity) but is recorded here so it is not later read as a bug.

## Metrics and display

New `extra_metrics` (registered in `core/training/metric_registry.py`; no schema
change, no API threading — `log_extra_metric` only):

| name | family / scale_group | meaning |
|---|---|---|
| `timestep_morph_lambda` | bounded_diagnostic / unit_interval | λ this step; absent when no morph is configured |
| `timestep_batch_mean` | bounded_diagnostic / unit_interval | mean of the t actually drawn this step |
| `timestep_batch_p10`, `timestep_batch_p90` | bounded_diagnostic / unit_interval | the drawn window, so a narrowing range is visible |

`timestep_batch_*` are logged whether or not a morph is active — they are the
cheap, always-useful answer to "what t is this run actually training at", and at
batch 1 with MNT they are computed over the whole MNT window.

Frontend:

1. **Config** (`TrainingConfig.tsx`, `trainingConfigDefinitions.tsx`,
   `trainingParams.ts`, `frontend/src/utils/api.ts` type, `openapi.yaml`): a
   "Morph on resume" subsection inside the existing timestep block — enable,
   steps, curve, interpolation. Only meaningful on a resumed run; the UI says so
   rather than hiding the controls, because the value has to be set *before* the
   resume starts.
2. **Monitor** (`TrainingMonitor.tsx`): a compact readout of the current
   distribution — reuse `TimestepDistributionGraph.tsx`, drawing the source and
   target PDFs as faint lines and the current interpolated PDF solid, captioned
   `logit_normal(0.0, 1.0) → logit_normal(0.5, 1.0) — 43% (step 860 / 2000)`.
   λ comes from the latest `timestep_morph_lambda` point and the endpoint
   configs from the run's config, so no new endpoint is needed.
   `TimestepDistributionGraph` needs one addition: an overlay curve given λ and
   two parameter sets. For quantile mode the intermediate PDF has no closed
   form — sample `t(u;λ)` on a 512-point `u`-grid and histogram it, which is
   ample for a sparkline.
3. **Chart**: the three new series appear automatically through the
   extra-metrics channel, get legend entries from the registry, and are
   toggleable in `MetricSeriesPicker`.

## Verification plan

- Unit: λ schedule endpoints and monotonicity; quantile-mode `icdf` monotone in
  `u` at λ ∈ {0, 0.5, 1}; at λ=0 and λ=1 the wrapper's draws match the endpoint
  sampler's law (KS on a large sample); `sample_stratified` marginals under the
  wrapper; mixture fallback when an endpoint lacks `icdf`; `state()` round-trip
  including the re-resume / nesting rule.
- Integration: a ~3-step smoke run with `morph.enabled` proving the config
  threads through and the metrics appear — no convergence run.
- Resume: a synthetic `state.json` with an in-flight morph resumes at the right
  λ rather than restarting the transition.
