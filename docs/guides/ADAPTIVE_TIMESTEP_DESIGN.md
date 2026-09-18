# Adaptive timestep distribution

`timestep_sampling.adaptive` is an opt-in controller for SenseNova SDXL Chimera
flow-velocity training. It is off by default and does not change existing runs.

## Modes and boundary

- `off`: no instrumentation or distribution change.
- `observe`: collect the same statistics and publish the recommendation, but
  continue sampling the configured base distribution.
- `bounded`: move toward the recommendation with a cosine quantile morph.

The first shipped version requires Chimera, `batch_size: 1`, and refuses a
simultaneous resume-time `morph`. The scalar prediction loss is then an exact
per-item observation. Other architectures are refused rather than assigned an
approximate SNR: notably, original SenseNova has a resolution-dependent noise
scale and MiniMax-H3 has coupled video/audio schedules. Chimera's clean-time
`t` gives noise fraction `sigma = 1 - t` directly.

## Coordinate and loss normalization

Bins are fixed in effective log-SNR:

```
logSNR = 2 * (log(1 - sigma) - log(sigma))
```

For straight flow interpolation `x_t = (1-sigma)x0 + sigma*noise` and velocity
target `v = noise - x0` (the sign does not affect MSE), the clean prediction is
`x0_hat = x_t - sigma*v_hat`. Therefore the metric compared across bins is
`L_x0 = sigma^2 * L_velocity`. This removes the velocity target's coordinate
scale before the controller compares progress.

## Controller

Each bin keeps fast and slow EMAs of `L_x0`. The signal is
`clamp(log(fast / slow), -1, 1)`: positive means recent learning stalled or
worsened, negative means it is improving. Absolute loss is deliberately not a
control signal, because feeding high loss directly back into sampling density
creates a positive-feedback loop in difficult/noisy regions.

The signal is exponentiated with `controller_gain`, then projected so its
expectation under the configured base law is exactly one while every density
ratio remains within `coverage_floor` and `max_density_ratio`. Thus adaptation
cannot remove the noise tail or concentrate beyond the configured cap. A new
target is represented by a persisted quantile table; `bounded` reaches it over
`morph_updates` successful optimizer updates. Another decision is not made
until the control interval, cooldown, and current morph have all elapsed.

```yaml
timestep_sampling:
  distribution: logit_normal
  mean: -0.8
  std: 0.8
  adaptive:
    mode: observe          # off | observe | bounded
    warmup_updates: 2000
    control_interval: 500
    bins: 8
    log_snr_min: -10
    log_snr_max: 10
    coverage_floor: 0.2
    max_density_ratio: 2.0
    controller_gain: 0.15
    morph_updates: 1000
    cooldown_updates: 500
    min_observations: 128
```

## Resume and observability

Checkpoint state includes bin counts, both EMAs, the current effective sampler,
an in-flight adaptive morph, density ratios, and controller position. Resume
therefore continues the same law rather than rebuilding it from YAML.

The timestep status sidecar includes the full `adaptive` status. Training
metrics expose controller count, mean x0-equivalent bin loss, and the maximum
density ratio. Use `observe` first on an established run; switch to `bounded`
only after the bin coverage and progress ratios are credible.
