# Multi-noise-timestep modes

`multi_noise_timesteps: N` reuses one assembled training batch for `N`
forward/backward passes. Each pass remains a `global_step`; MNT does not average
the `N` losses into one optimizer update. `stratified_timesteps` controls the
timestep draws independently of the noise mode described here.

MNT and adaptive timestep sampling are orthogonal and can be enabled together.
The noise mode controls correlation between the Gaussian noise tensors, while
the adaptive sampler controls the timestep marginal. Stratification draws each
MNT window from the adaptive sampler's current effective quantiles, and the
controller observes the exact per-item prediction loss from every pass (also
when OOM recovery splits a physical batch into micro-chunks).

## Noise modes

Every shipped mode preserves the marginal law `epsilon ~ N(0, I)`. Changing a
mode therefore changes correlation inside an MNT window, not the per-pass noise
scale or training target.

| Mode | Noise for iteration i | Intended use |
|---|---|---|
| `independent` | fresh `epsilon_i` | Maximum noise diversity; legacy behavior |
| `shared` | one `epsilon_anchor` for all i | Compare timesteps on one exact clean-to-noise trajectory |
| `trajectory` | `alpha * epsilon_anchor + sqrt(1-alpha^2) * epsilon_i` | Continuous interpolation between independent and shared while retaining unit variance |
| `antithetic` | pairs `epsilon, -epsilon` | Cancel finite-window noise mean and reduce Monte Carlo variance |

`trajectory_blend_alpha` is a correlation coefficient in `[0, 1]`: zero is
independent and one is shared. It is not an unnormalised linear blend; that
would shrink noise variance at intermediate values and silently change the
training distribution.

For odd `N`, `antithetic` ends with the positive member of a new pair. Even MNT
sizes are therefore recommended for that mode.

## Execution and memory contract

The coupled noise is created once for a full MNT iteration before forward/OOM
recovery. Micro-batch retries slice that same tensor, so a retry cannot change
the example's noise. Shared anchors are retained on the clean latent's device
(normally CPU) between iterations and moved with the ordinary train-step input.

MiniMax-H3 currently rejects non-independent modes: its objective has coupled
video and audio noise streams, while the common MNT contract carries one clean
latent/noise tensor. Silently coupling only video would make the selected mode
false for half of that objective.

## Acceptance checks

- `independent` takes the unchanged architecture-local random draw path.
- `shared` returns the same tensor throughout a window.
- `trajectory` measures the requested anchor correlation and unit variance.
- `antithetic` returns exact sign-opposite pairs.
- OOM micro-batching slices rather than redraws coupled noise.
- Unknown modes and alpha values outside `[0, 1]` fail before training begins.
