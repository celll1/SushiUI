# SenseNova SDXL Chimera v3: polar tangent-flow design

Status: **implemented, acceptance measurements pending**. Format-4 artifacts,
polar targets, the radial/tangent heads, training loss, one-NFE sampler,
img2img/inpaint source paths, CFG probes, initialization API, and frontend
selection are wired. Existing v1/v2 artifacts and Run 139 remain unchanged;
v3 still requires a new artifact and step-zero training run.

This proposal replaces Chimera v2's statistically radial residual split with a
pointwise polar decomposition. Its central contract is:

> CFG may extrapolate direction, but it must never extrapolate radius.

The resulting model is a new prediction type and a new stochastic interpolant.
It is not a sampler-only patch for a v2 checkpoint and is not resume-compatible
with v1 or v2 training.

## 1. Motivation

Chimera v2 made both endpoint velocities observable and separated an analytic
least-squares drift from a learned residual. That removes the worst endpoint
target variance, but it does not make the learned residual pointwise tangent to
the current latent. Its orthogonality is statistical:

```text
E[<y, r>] = 0
```

rather than per sample:

```text
<y, r> = 0.
```

Consequently, affine CFG can amplify the residual's radial component. A global
norm clamp can limit the result after the fact, but cannot express the intended
invariant. A projector-only v2 retrofit would remove the radial component, but
would also retain `r(0) = 0`; it cannot provide the conditional direction that
exists at the noise endpoint as a posterior expectation.

v3 therefore changes the path itself. It treats radius and direction as
separate state variables, trains a scalar radial velocity and a tangent angular
velocity, and applies CFG only to the tangent velocity.

## 2. Goals and non-goals

### Goals

- Make radial CFG overshoot impossible by construction for every sample and
  every guidance scale.
- Preserve an exact path from Gaussian noise to the centered SDXL latent
  distribution; do not force all samples onto one dataset-average-radius shell.
- Permit a nonzero, condition-dependent direction at the noise endpoint.
- Keep both endpoint radial targets observable.
- Use one U-Net evaluation per solver step per conditioning branch.
- Keep the SDXL-like U-Net parameter count effectively unchanged; the added
  scalar head is negligible relative to the U-Net.
- Retain SenseNova conditioning, reference-image conditioning, and existing
  timestep-distribution infrastructure where their contracts still apply.

### Non-goals

- v3 does not claim that a constant-radius sphere is the image manifold. Polar
  coordinates cover the full nonzero centered latent space; radius remains a
  learned state variable.
- v3 does not make angular extrapolation harmless. It prevents radial escape;
  an excessively large angular step can still rotate too far.
- v3 does not convert or continue a v2 optimizer state.
- v3 does not select a production angular-step limit without measurements.

## 3. Notation and numerical geometry

Let `D = C * H * W`. All inner products and reductions in this section execute
in fp32 per sample, even when U-Net activations use bf16:

```text
<a, b> = mean_D(a * b)
||a||  = sqrt(<a, a>)
```

Thus a normalized direction has RMS norm one. Let:

```text
x_c  = x0 - mu                 centered clean SDXL latent
e    = epsilon                 standard Gaussian noise
rho0 = ||e||
rho1 = ||x_c||
n0   = e / rho0
n1   = x_c / rho1
```

`mu` is measured after the selected SDXL VAE's scaling and shift have been
applied. Its value and provenance are artifact-owned. Degenerate radii below a
recorded epsilon are rejected during training rather than silently normalized.

v3 retains the endpoint-observable cubic amplitudes from v2:

```text
alpha(s) = 2s^2 - s^3
sigma(s) = 1 - s - s^2 + s^3

alpha'(s) = 4s - 3s^2
sigma'(s) = -1 - 2s + 3s^2
```

Time runs from `s = 0` at noise to `s = 1` at clean data.

## 4. Polar stochastic interpolant

### 4.1 Direction path

Let:

```text
theta = acos(clamp(<n0, n1>, -1, 1))
e1    = (n1 - cos(theta) * n0) / sin(theta)
gamma(s) = 2s - s^2
```

`gamma` is the lowest-degree schedule satisfying
`gamma(0) = 0`, `gamma(1) = 1`, and `gamma'(1) = 0`. The spherical path is:

```text
n(s) = cos(gamma(s) * theta) * n0
     + sin(gamma(s) * theta) * e1
```

and its derivative is:

```text
n'(s) = gamma'(s) * theta *
        [-sin(gamma(s) * theta) * n0
         + cos(gamma(s) * theta) * e1].
```

Near coincident directions use the continuous small-angle limit. Near the
antipodal singularity, the implementation must choose a deterministic
orthogonal direction and record a counter; it must not divide by a clamped
`sin(theta)` and pretend the result is exact. Antipodal events should be
negligible in the high-dimensional latent space, but the behavior must still be
defined and tested.

### 4.2 Radius path

The per-pair radius is the quadrature interpolation:

```text
rho(s) = sqrt(sigma(s)^2 * rho0^2 + alpha(s)^2 * rho1^2)
```

with derivative:

```text
rho'(s) = [sigma * sigma' * rho0^2
           + alpha * alpha' * rho1^2] / rho(s).
```

This preserves sample-dependent endpoint radii and the RMS contraction of the
v2 path without assuming that noise and clean directions are exactly
orthogonal for an individual pair.

### 4.3 Full path and target velocity

```text
z(s) = alpha(s) * mu + rho(s) * n(s)

u*(s) = alpha'(s) * mu
      + rho'(s) * n(s)
      + rho(s) * n'(s).
```

The last two terms are exactly orthogonal: `rho' * n` is radial and
`rho * n'` is tangent.

The endpoints are:

```text
z(0) = epsilon
u*(0) = -epsilon + 2 * rho0 * Log_n0(n1)

z(1) = x0
u*(1) = x0.
```

The noise-end tangent target is deliberately not deterministic for an
individual prompt. MSE learns its conditional expectation:

```text
E[2 * rho0 * Log_n0(n1) | epsilon, condition].
```

For a null condition this posterior is broad; for an informative condition it
can have a nonzero mean direction. This is the intended uncertainty. Unlike a
direct velocity target, that uncertainty is confined to the tangent space and
cannot be converted by CFG into radial growth.

At the clean endpoint, `gamma'(1) = 0`, so the tangent target vanishes and the
full target remains the observable clean latent.

### 4.4 Noise-end variance budget

The conditional direction at `s = 0` is not free. For a 1024-pixel SDXL latent,
`D = 4 * 128 * 128 = 65,536`; an independent Gaussian direction and clean-data
direction are overwhelmingly close to orthogonal, so `theta` concentrates near
`pi / 2`. With `rho0 = rho1 = 1`, the reference path has:

| `s` | `rho` | `||tau*||` RMS | `lambda*` |
|---:|---:|---:|---:|
| 0.00 | 1.00 | 3.14 | -1.00 |
| 0.25 | 0.71 | 1.68 | -1.17 |
| 0.50 | 0.53 | 0.83 | 0.00 |
| 1.00 | 1.00 | 0.00 | +1.00 |

At the noise endpoint, the tangent target's second moment is therefore about
`pi^2` under this isotropic unit-RMS reference. Its irreducible conditional
variance is:

```text
E[||tau* - E[tau* | epsilon, condition]||^2],
```

which is at most that second moment but can remain close to it when the
condition leaves many valid directions. This is roughly one order of magnitude
larger than the unit-RMS missing-component reference for the v1 straight path;
it is an intentional cost of restoring a condition-dependent noise-end
direction, not evidence by itself that training is broken.

The exact conditional variance is not directly observable from one paired
target. Diagnostics therefore keep three quantities separate: target second
moment, predicted conditional-mean norm, and held-out residual MSE. At the MSE
optimum the latter approaches the irreducible variance; before convergence it
is only an upper-bound proxy and must not be labeled as a measured Bayes floor.

The cause is explicit: `gamma'(0) = 2` and `theta ~= pi / 2`. The alternative
smoothstep schedule

```text
gamma_0(s) = 3s^2 - 2s^3
```

has zero derivative at both endpoints and removes the noise-end target variance,
but also restores a condition-independent first direction. More generally, the
terminal-flat cubic family

```text
gamma_a(s) = a*s + (3 - 2a)*s^2 + (a - 2)*s^3
```

has `gamma_a'(0) = a` and `gamma_a'(1) = 0`; the baseline quadratic is the
`a = 2` member and smoothstep is `a = 0`. `a` is an artifact-owned path choice,
never an inference knob. v3 initially tests `a = 2` against `a = 0`; an
intermediate value requires its own measured artifact contract. The design
prefers `a = 2` only if its conditional endpoint signal improves equal-NFE
generation enough to justify the measured variance and training allocation.

Finally, this is a genuinely new interpolation. Defining `rho` by quadrature
and `n` by a geodesic replaces the interior of `sigma*epsilon + alpha*x_c`;
it does not merely re-express that Cartesian path. Only its endpoints and the
chosen endpoint observability contract are shared.

## 5. Network prediction contract

For `y = z - alpha(s) * mu`, compute:

```text
rho = ||y||
n   = y / rho.
```

The U-Net predicts two quantities:

1. a scalar radial speed `lambda_theta(n, s, log(rho), c)` per sample;
2. a four-channel raw tangent field `h_theta(n, s, log(rho), c)`.

The actual trunk input is fully specified as follows:

```text
y = z - alpha(s) * mu
rho = ||y||
n = y / rho

spatial U-Net input: n
scalar conditioning: s and log(rho)
cross/additional conditioning: existing SenseNova bridge, pooled, crop,
                               resolution, and reference-image inputs
```

`(n, rho, s)` is information-equivalent to `z` because `mu` and `alpha` are
artifact-known, while presenting the spatial trunk with unit-RMS input at every
timestep. `log(rho)` is embedded alongside the timestep embedding rather than
left for convolutions to reconstruct through a global reduction.

The tangent head is the ordinary four-channel SDXL-shaped output convolution.
The scalar head uses the conditioned mid-block activation: global-average pool
over space in fp32, apply LayerNorm, concatenate the scalar time/radius
embedding, and use one affine output to produce one `lambda` per sample. It does
not pool the final four-channel tangent prediction. Exact feature dimensions
and initialization are recorded in the architecture config before Phase 3;
there is no analytic target leakage into the head.

The effective tangent field is projected pointwise in sample space:

```text
P_n(h) = h - n * <n, h>
tau_theta = P_n(h_theta).
```

Training targets are:

```text
lambda* = rho'(s)
tau*    = rho(s) * n'(s).
```

The geometric loss is:

```text
L_rad = mean_batch((lambda_theta - lambda*)^2)
L_tan = mean_batch(mean_D((tau_theta - tau*)^2))
L_geo = L_rad + L_tan.
```

Because `||n|| = 1` in RMS geometry and both tangent terms are orthogonal to
`n`, `L_geo` is exactly the mean squared vector-field error of
`lambda * n + tau`. The radial scalar is therefore not diluted by `D` output
elements and needs no arbitrary dimensionality multiplier.

The scalar head adds a negligible number of parameters.
It is explicit rather than recovered from a four-channel output so its loss,
conditioning behavior, checkpoints, and diagnostics remain independently
auditable. The final tangent and scalar heads start from default initialization
for scratch v3 training.

No target is normalized to constant variance. Any loss balancing introduced
after measurements must be explicit, bounded, logged, and artifact-owned.

## 6. CFG contract

Evaluate positive and null branches at the same `(z, s)`:

```text
tau_c = P_n(h_c)
tau_u = P_n(h_u)

tau_cfg = tau_u + w(s) * (tau_c - tau_u).
```

Since tangent space is linear:

```text
<n, tau_cfg> = 0
```

for every finite guidance scale, up to fp32 reduction tolerance.

The radial speed is an unguided anchor:

```text
lambda_anchor = lambda_c
v_cfg = alpha'(s) * mu + lambda_anchor * n + tau_cfg.
```

`lambda_c` means the positive-conditioning branch at ordinary condition strength
one. CFG scale, dynamic-CFG scheduling, negative prompting, and CFG norm controls
must not alter it. Unconditional generation uses an empty positive condition;
it does not emulate unconditional generation by setting a nonempty prompt's CFG
scale to zero. This distinction is part of the API contract.

An optional radial condition strength `eta` may interpolate
`lambda_u + eta * (lambda_c - lambda_u)` only for `eta` in `[0, 1]`; it is a
separate bounded conditioning control, not CFG. v3 defaults to `eta = 1` and
does not expose it until a measured use case exists.

This gives the required invariant:

```text
d rho / ds = lambda_anchor,
```

which is independent of guidance scale. CFG can rotate the trajectory but
cannot make it leave its learned radial trajectory.

Existing global or channel CFG norm clamps are compatibility features, not the
v3 safety mechanism. They default off for v3. If retained experimentally, they
may scale only `tau_cfg`; they must never reintroduce a radial component.

## 7. Polar one-NFE solver

Plain Cartesian Euler introduces radial leakage from a tangent update at order
`Delta s^2`. v3 therefore integrates the two coordinates separately while
still using one U-Net evaluation per branch and step.

Given `rho`, `n`, `lambda_anchor`, and `tau_cfg`:

```text
kappa = lambda_anchor / rho
rho_next = rho * exp(Delta s * kappa)

omega = tau_cfg / rho
phi   = Delta s * ||omega||

n_next = cos(phi) * n + sin(phi) * omega / ||omega||
z_next = alpha(s_next) * mu + rho_next * n_next.
```

For a negligible angular speed, use the continuous zero-angle limit. The
exponential radial update is first-order accurate like Euler but guarantees a
positive radius. The spherical exponential update is exact for a frozen
tangent field and preserves `||n_next|| = 1`. A final fp32 renormalization is a
numerical-drift correction, not a norm clamp.

This solver is named `polar_exp_euler_v1`. It remains a one-NFE method; scalar
reductions and trigonometric operations are small relative to the U-Net.

An optional angular CFL guard may scale `omega` so that `abs(phi)` does not
exceed an artifact-recorded limit. Its default is disabled until a CFG-scale
sweep establishes a threshold. If enabled, the UI and metrics must distinguish
an angular cap from the removed Cartesian CFG norm clamp.

The angular risk is largest around the radius minimum: the reference
`rho(0.5) ~= 0.53` makes `omega = tau_cfg / rho` about 1.9 times as sensitive
to a fixed tangent RMS as it would be at unit radius. The CFG sweep must include
this region; radial invariance is not evidence that an uncapped angular step is
accurate.

The exact noise endpoint can no longer skip the U-Net: its radial term is
analytic, but its conditional tangent posterior is the feature v3 is designed
to learn. Equal-NFE comparisons with v2 must count this first evaluation.

## 8. Training and timestep sampling

The v3 path makes exact endpoint samples legal, but the noise endpoint carries
high irreducible angular variance. It should be observed and diagnosed, not
made the dominant training mass. The initial training policy is:

- use the existing bounded log-SNR sampler and MNT batch construction;
- retain `log(q * alpha^2 / sigma^2)` as the stratification coordinate for
  continuity with v2, while labeling it `linear-proxy log-SNR` because the
  polar path is not a linear mixture;
- log radial and tangent losses separately in every bin;
- keep exact endpoint probes outside the stochastic training quota;
- do not enable adaptive redistribution until all ready bins have stable
  radial and tangent observations.

The adaptive controller must not collapse these signals into a single opaque
loss. In particular, raw tangent loss is not a cross-bin difficulty score:
near `s = 0` it contains the expected irreducible floor described in Section
4.4, so feeding it directly to density control would concentrate samples where
the target is least reducible. Initial v3 runs therefore permit `observe` mode
only. Promotion to bounded adjustment requires a per-bin learning-progress or
excess-over-recorded-floor signal that does not normalize the training target
or hide its physical scale.

The v3 observation record contains at least radial MSE, tangent MSE,
conditional/null tangent delta, angular target second moment, learned
conditional-mean norm, and angular step size. Any later policy that changes
density is bounded by the existing floor, cooldown, maximum-density-ratio, and
resume-state contracts.

MNT remains compatible because all samples in a batch are still ordinary
`s` samples. Resume must preserve the epoch's batch order and MNT trajectory as
it does for v2; a v3 artifact cannot inherit a v2 controller state.

## 9. Conditioning and supported modalities

The SenseNova understanding branch, bridge, pooled conditioning, crop and
resolution conditioning, and reference-image tokens feed both v3 heads through
the same U-Net trunk. Their output semantics differ:

- positive/null differences in the tangent field are CFG-eligible;
- the positive radial prediction is used at strength one and is not
  extrapolated;
- negative-prompt conditioning affects the null tangent branch only.

Text-to-image and reference-image text-to-image therefore remain native. For
image-to-image and inpainting, construction of a known source trajectory uses
the source latent and a fixed noise draw with the same polar interpolant.
Source reinjection must occur in polar coordinates or be followed by an exact
state decomposition; Cartesian blending after the solver step would violate
the radial CFG audit. SenseNova image-to-text and text/image-to-text paths do
not use the diffusion prediction contract and remain independently routable.

## 10. Diagnostics

Every diagnostic probe and training sample record for v3 must expose:

- `rho`, predicted `lambda_c`, `lambda_u`, and selected radial anchor;
- tangent RMS for positive, null, delta, and guided fields;
- `abs(<n, tau_c>)`, `abs(<n, tau_u>)`, and
  `abs(<n, tau_cfg>)` before and after the fp32 projector;
- proposed angular displacement `phi` and any angular-cap scale;
- radial exponential ratio and minimum radius;
- radial MSE and tangent MSE per timestep/log-SNR bin;
- noise-end conditional/null angular posterior delta;
- antipodal, small-angle, and degenerate-radius counters.

The CFG diagnostic API must sweep guidance scales at identical noise, condition,
and timesteps. At a fixed latent state its primary v3 assertion is that changing
CFG scale changes `tau_cfg` but leaves the selected radial prediction and radial
update bitwise identical before final casting. A complete rollout is not
required to have an identical radius trace: once guidance changes the direction,
the conditioned U-Net may predict a different radial speed at a later state.
The structural guarantee is the absence of direct CFG extrapolation in that
radial speed. Image metrics and latent percentiles remain secondary outcome
checks.

The frontend timestep-distribution panel continues to show the realized sample
density. It additionally labels v3's stratification axis as proxy log-SNR and
can plot radial/tangent observations independently.

## 11. Artifact and compatibility contract

v3 uses artifact format 4. The prediction block contains the equivalent of:

```yaml
prediction:
  type: polar_tangent_flow
  version: 1
  path: observable_polar_geodesic_v1
  time_direction: zero_noise_to_one_clean
  radial_schedule: cubic_quadrature_v1
  angular_schedule: terminal_flat_cubic_v1
  angular_endpoint_slope: 2.0
  cfg_mode: tangent_only_v1
  radial_anchor: positive_condition
  integrator: polar_exp_euler_v1
  latent_mean: <artifact tensor/reference>
  reduction_dtype: float32
```

The artifact also records VAE identity, latent scaling and shift, `mu`
calibration provenance, normalization epsilon, small-angle threshold, antipodal
policy, and any enabled angular-step limit.

Compatibility rules:

- v1/v2 checkpoints cannot be relabeled as v3;
- v3 cannot resume v1/v2 optimizer, scheduler, MNT, adaptive-timestep, or RNG
  state;
- the understanding branch may reference the same pinned SenseNova weights;
- the SDXL VAE may reference the same pinned donor artifact;
- a v2 U-Net trunk may be used only as an explicitly experimental
  initialization with both output heads reset and global step zero;
- scratch U-Net initialization is the default.

The initializer exposes that experimental path as `chimera_warmstart_source`.
It preserves the source bridge and all U-Net parameters except `conv_out`, adds
a freshly initialized radial head, and records the source artifact and reset
set in `chimera.json`. Optimizer, scheduler, timestep-controller, RNG, epoch,
and source global-step state are never copied.

## 12. API and UI surface

Path, target, projector, radial anchor, and integrator are artifact-owned and
must not be loose generation parameters. The generation API keeps ordinary CFG
scale and dynamic-CFG controls, but v3 routes them exclusively to the tangent
field.

Only the following may become request-level sampler controls after measured
defaults exist:

- angular-step limit (`null` means disabled);
- solver selection, initially restricted to `polar_exp_euler_v1`;
- diagnostic-only emission of the polar trace.

Any API addition follows the repository's OpenAPI-first/default-source rules.
The model inspector must show the prediction type prominently and refuse a
Cartesian or v2 sampler for a v3 artifact.

## 13. Rejected alternatives

### Projecting the existing v2 residual only

This makes CFG tangent but preserves the v2 endpoint target `r(0) = 0`. Adding
a conditional prior only at inference changes the vector field without a
matching training path.

### A fixed expected-radius shell

Retracting every sample to `R(s) = sqrt(q * alpha^2 + sigma^2)` prevents radial
CFG escape, but also forces the clean endpoint toward one average norm and
cannot reproduce the true clean-latent radius distribution. v3 instead learns
the per-sample scalar radius path.

### A condition-dependent analytic base

Putting a prompt-derived latent mean into the analytic radial term gives
positive and null branches different base trajectories, after which CFG can
again extrapolate between radii. Conditional information belongs in the learned
heads; only the tangent head is guided.

### Direct velocity CFG plus a norm clamp

This detects overshoot after radial and tangent components have already mixed.
It neither preserves an invariant nor distinguishes a useful angular update
from destructive radial growth.

### Midpoint as the mandatory default

A midpoint solve doubles U-Net evaluations. The polar split removes the Euler
radial-leak mechanism at one NFE, so higher-order solvers remain optional
quality experiments rather than a prerequisite for correctness.

## 14. Verification and acceptance gates

### Algebraic and CPU tests

- all path endpoint values and derivatives match the formulas above;
- `||n(s)|| = 1` and `<n(s), n'(s)> = 0` across random and adversarial pairs;
- radial/tangent loss decomposition equals direct vector MSE;
- projected positive, null, delta, and guided fields are tangent for negative,
  zero, ordinary, and extreme finite CFG scales;
- radial updates are CFG-scale invariant;
- the solver preserves direction norm and positive radius;
- small-angle, near-antipodal, and degenerate-radius policies are deterministic;
- bf16 model outputs with fp32 reductions remain finite near both endpoints.

### GPU integration tests

- one training step, checkpoint, strict reload, and deterministic continuation;
- one generation at each supported resolution and aspect ratio;
- equal-state CFG probes with identical selected radial predictions and radial
  updates, plus full-rollout checks for bounded non-divergent radius traces;
- equal-state dynamic-CFG changes only the tangent update; full trajectories
  may subsequently produce different conditioned radial predictions;
- reference-image, image-to-image, and inpaint routing checks;
- peak VRAM and step-time comparison with v2 at equal U-Net geometry.

### Training gates

- start from a new v3 run and new step zero;
- demonstrate finite, non-collapsed radial and tangent losses in every ready
  timestep bin;
- report the measured noise-end tangent target second moment, predicted-mean
  norm, and held-out residual-floor proxy separately; approximately `pi^2`
  target energy under the unit-RMS orthogonal reference is expected, not an
  automatic failure;
- compare the `a = 2` conditional-endpoint path with the `a = 0` smoothstep
  control at equal data, optimizer steps, and NFE before accepting the former;
- keep adaptive timestep in `observe` until its v3 signal is proven not to chase
  the noise-end variance floor;
- demonstrate a measurable positive/null tangent delta near the noise endpoint;
- demonstrate no CFG-dependent radial divergence in the diagnostic API;
- compare equal-NFE samples against v2 before enabling an angular cap or
  adaptive timestep policy;
- promote no default based solely on training loss.

## 15. Implementation sequence

Each phase is independently reviewed and committed:

1. pure fp32 polar-path math, target generation, and algebraic tests;
2. artifact prediction schema and strict compatibility refusal;
3. scalar radial head, tangent head, projector, and geometric loss;
4. `polar_exp_euler_v1` generation solver and CFG routing;
5. training checkpoint/resume plumbing with v3-only state;
6. diagnostic API, run database metrics, and frontend plots;
7. reference-image, image-to-image, and inpaint verification;
8. scratch dummy artifact and bounded smoke training;
9. measured CFG sweep, angular-limit decision, and acceptance report.

No phase stops or restarts a live training run unless the repository owner
separately requests that operation.

## 16. Research context

The path construction follows the conditional probability-path viewpoint of
[Stochastic Interpolants](https://arxiv.org/abs/2303.08797). The tangent update
and spherical exponential solver are consistent with the geometric treatment
of vector fields in
[Riemannian Flow Matching](https://arxiv.org/abs/2302.03660). Recent guidance
and norm-control work also motivates separating or projecting the guided
component rather than merely clipping its final norm:
[CFG-Zero*](https://arxiv.org/abs/2503.18886),
[Improving CFG of Flow Matching via Manifold Projection](https://arxiv.org/abs/2601.21892),
and [NormGuard](https://arxiv.org/abs/2606.27771).

Those works motivate the design direction; the particular observable polar
path, positive radial anchor, loss identity, and Chimera artifact contract in
this document are repository-specific proposals and require the acceptance
tests above. No formula, compatibility rule, or acceptance criterion depends on
reproducing CFG-Zero*, Manifold Projection, or NormGuard; those three references
are non-normative context only.
