# SenseNova SDXL Chimera v2 design

Status: **design accepted; implementation not started**.

Chimera v2 keeps the frozen SenseNova understanding branch, conditioning
bridge, SDXL-shaped U-Net, and bundled SDXL VAE. It changes the latent path and
the meaning of the U-Net output. The purpose is to stop asking the denoiser for
an unobservable paired image at the pure-noise endpoint or an unobservable
paired noise sample at the clean endpoint.

This is a checkpoint-semantic change, not a sampler option for a v1 artifact.

## 1. Problem statement

Chimera v1 uses the straight path

```text
z(t) = t*x0 + (1-t)*epsilon
u(t) = dz/dt = x0 - epsilon
```

At `t=0`, `z=epsilon`; the particular training pair's `x0` is not determined by
that latent and its conditioning. At `t=1`, `z=x0`; the particular epsilon is
not determined. The conditional expectation `E[u | z,c]` remains well-defined,
so this is not the absence of a conditional vector field. The problem is that a
direct-velocity training pair still carries maximal irreducible target variance
at both endpoints, after which CFG extrapolates the difference between learned
conditional means.

Changing only the output basis cannot remove this. Any invertible
parameterization that reconstructs the same v1 endpoint velocity must retain
the missing component somewhere. v2 therefore changes the probability path as
well as the output parameterization.

## 2. Endpoint-observable path

Let clean time be `s in [0,1]` and define

```text
z(s)     = alpha(s)*x0 + sigma(s)*epsilon
alpha(s) = 2*s^2 - s^3       = s^2*(2-s)
sigma(s) = 1 - s - s^2 + s^3 = (1-s)^2*(1+s)
```

The velocity is

```text
u(s) = alpha'(s)*x0 + sigma'(s)*epsilon
alpha'(s) = 4*s - 3*s^2
sigma'(s) = -1 - 2*s + 3*s^2
```

The boundary contract is:

| Boundary | State | Velocity | Observable from state? |
|---|---|---|---|
| `s=0` | `z=epsilon` | `u=-epsilon=-z` | yes |
| `s=1` | `z=x0` | `u=x0=z` | yes |

The coefficient of the unknown `x0` is zero at the noise endpoint and the
coefficient of the unknown epsilon is zero at the clean endpoint. The path is
symmetric under exchanging `s <-> 1-s` and `x0 <-> epsilon`. It deliberately
allows the latent RMS to contract in the middle; normalizing that contraction
away would also remove the non-zero, analytically known endpoint drift.

For centered unit-variance clean latents, `s=0.5` gives
`Var(z)=0.28125` and `RMS(z)=0.5303`. Near the noise endpoint,
`alpha(s)=O(s^2)`, hence `SNR=O(s^4)`; representative natural-log SNR values
are about `-7.7` at `s=0.1` and `-4.7` at `s=0.2`. These are deliberate path
properties, not implementation rounding. They require measured input-scale and
sampler-grid gates below.

The selected cubic is the lowest-degree symmetric polynomial satisfying

```text
alpha(0)=0, alpha'(0)=0, alpha(1)=1, alpha'(1)=1
sigma(s)=alpha(1-s)
```

Higher-order or learned paths are outside the first v2 boundary.

## 3. Analytic skip and learned residual

The U-Net does not predict `u` directly. Known state-aligned motion is removed
analytically:

```text
u_theta(z,s,c) = analytic_state_drift(z,s) + r_theta(z,s,c)
```

Let `mu` be the four-channel clean-latent mean after the bundled VAE's scaling
and shift, `x_c=x0-mu`, and `q=E[x_c^2]` be one centered scalar second moment.
Noise is independent, zero-mean, and unit-variance. Define `z_c=z-alpha*mu`.
Then

```text
c_skip(s) =
  (q*alpha(s)*alpha'(s) + sigma(s)*sigma'(s))
  / (q*alpha(s)^2 + sigma(s)^2)

u_theta = alpha'(s)*mu + c_skip(s)*z_c + r_theta
r_target = u_target - alpha'(s)*mu - c_skip(s)*z_c
```

This is the centered least-squares state-aligned coefficient. The artifact
records the four-value `mu` and scalar `q`, measured from the training dataset
after the bundled VAE's scaling and shift. Using an uncentered variance or
silently assuming zero latent mean is invalid. A per-channel `q` extension
requires a new path/preconditioning version.

At the endpoints:

```text
s=0: c_skip=-1, r_target=0
s=1: c_skip=+1, r_target=0
```

Therefore the network target, its irreducible paired-sample variance, and the
CFG residual all vanish exactly at both endpoints. No division by `alpha`,
`sigma`, `alpha'`, or `sigma'` is used. In particular, `r_target` must not be
renormalized to constant variance near an endpoint; doing so would recreate the
unobservable endpoint target that this design removes.

For `q=1`, `|c_skip|` reaches about `1.64` around `s=0.3/0.7`, while residual
variance vanishes at the endpoints and concentrates in the middle. This is the
accepted cost of declining EDM-style output normalization: endpoint uncertainty
is not restored merely to equalize loss scale. The magnitude is nevertheless a
measured acceptance item, not an assumption that every SDXL latent distribution
behaves like `q=1`.

## 4. Training contract

For each example:

1. Encode `x0` with the artifact's bundled SDXL VAE contract.
2. Draw epsilon and clean time `s` using the run's timestep sampler.
3. Construct `z`, centered `z_c`, `u_target`, `c_skip`, and the unnormalized
   `r_target` with the equations above.
4. Feed `(z,s,conditioning)` to the U-Net and regress its four-channel output
   to `r_target`.
5. Reconstruct `u_theta=alpha'*mu+c_skip*z_c+r_theta` only for previews,
   trajectory probes, and velocity-space diagnostics.

The primary loss is MSE on the unnormalized residual. A normalized residual may
be logged with a floor for diagnosis but may not drive the optimizer or the
adaptive timestep controller. Raw residual loss, reconstructed velocity error,
`c_skip`, residual RMS, and effective log-SNR are logged per timestep bin.

Effective SNR is

```text
SNR(s) = q*alpha(s)^2 / sigma(s)^2
```

MNT stratification and adaptive-timestep observations use effective log-SNR,
not the numeric value of `s`. Exact endpoint samples are legal and have a zero
residual target, but ordinary continuous sampling need not add special endpoint
mass. Any explicit endpoint quota is recorded separately from interior bins.

The timestep shift changes the sampling density over effective log-SNR. It must
not alter `alpha`, `sigma`, or their derivatives. Existing v1 controller state
is not resumable because its bins and loss semantics differ.

## 5. Generation contract

Euler remains the default sampler:

```text
z_c = z - alpha(s)*mu
z_next = z + (s_next-s) * (alpha'(s)*mu + c_skip(s)*z_c + r_cfg)
```

At the first interval, `s=0`, v2 performs

```text
u=-z
z_next=(1-delta_s)*z
```

analytically and does not call the U-Net. This saves one NFE. The last endpoint
is a destination and is not evaluated. Partial img2img beginning at `s>0` starts
with the ordinary learned-residual step. Inpaint source re-injection uses the v2
`alpha/sigma` path at the next time.

The Euler solver and its time grid are separate choices. Uniform `s` is the
baseline, not an accepted production default: because `SNR=O(s^4)` near noise,
it can spend several evaluations in a region with little conditional signal.
Before selecting the default, the implementation must compare uniform-`s` and
bounded equal-log-SNR grids at identical NFE. Exact endpoints remain explicit;
the log-SNR grid covers only finite interior bounds. The comparison reports
trajectory error against a high-NFE reference, first-interval error, and fixed-
seed sample diagnostics. No grid wins solely from image appearance.

CFG is applied only to the learned residual:

```text
r_cfg = r_uncond + cfg*(r_cond-r_uncond)
u_cfg = alpha'*mu + c_skip*z_c + r_cfg
```

The analytic skip is shared and is never amplified by CFG. At `s=0` and `s=1`,
`r_cond=r_uncond=0` by contract, so CFG has no endpoint direction to overshoot.
CFG norm limiting, if requested, measures and limits the residual guidance
delta before adding the analytic skip. Dynamic CFG remains available as a
secondary policy, not as the mechanism enforcing endpoint safety.

The CFG probe records analytic-skip RMS, conditional/unconditional residual
RMS, residual delta, reconstructed velocity RMS, Euler update size, and whether
the step bypassed the U-Net. The first record must report zero residual delta.

## 6. Artifact and resume boundary

v2 retains model type `sensenova_sdxl_chimera` but bumps the artifact format and
requires the following prediction declaration:

```json
{
  "prediction": {
    "type": "endpoint_observable_residual",
    "version": 1,
    "time_direction": "zero_noise_to_one_clean",
    "path": "symmetric_cubic_observable_v1",
    "latent_mean": [0.0, 0.0, 0.0, 0.0],
    "latent_centered_second_moment": 1.0
  }
}
```

The real measured mean and centered second moment replace the example values.
Their dataset identity, sample count, VAE identity, and accumulation precision
are recorded in provenance. The loader dispatches v1 direct velocity and v2
residual semantics explicitly. Missing prediction metadata is never guessed as
v2.

A v1 U-Net/checkpoint cannot resume as v2, even though tensor names and shapes
match. Optimizer, EMA, adaptive-timestep, MNT trajectory, and checkpoint state
are likewise incompatible. v2 may initialize non-output U-Net tensors from a
v1 or SDXL source only through a separately named experimental initializer;
the output convolution is reset and the run starts at step zero. The default
remains scratch initialization.

## 7. API and frontend

Path selection is artifact-owned, not a generation request parameter. Existing
generation and training forms require no new sampler switch. Model facts,
training status, checkpoint metadata, and sample metadata expose the prediction
type and path version. The timestep-distribution chart labels its horizontal
axis as effective log-SNR for v2 and shows the realized post-controller sample
density.

Starting a v2 run from a v1 checkpoint fails before model or optimizer loading
with both prediction contracts in the error. The frontend displays that refusal
without offering a force-resume path.

## 8. v1 compatibility startup

Before v2 implementation, v1 production and training-preview sampling use one
bounded compatibility change: when a trajectory starts at exact `t=0`, its
first Euler interval uses `u=-epsilon=-z` analytically and skips U-Net/CFG. All
later v1 steps retain direct-velocity semantics. This does not change v1
training targets or make a v1 artifact a v2 artifact.

This startup is intentionally a discrete inference policy. Its comparison on a
fixed live checkpoint establishes whether the observed first-step CFG burst is
causal before v2 commits to a new training path.

It is biased relative to the v1 vector field: dropping the first learned term
removes approximately `delta_t * E[x0 | z=epsilon,c]` from the first update
(`delta_t * E[x0|c]` when initial noise and data are independent). It is a
bounded causal probe, not a permanent correction to the v1 probability path.

## 9. Verification and acceptance

Implementation is accepted only after all of the following pass:

1. Algebra tests verify endpoint state, endpoint velocity, `c_skip`, zero
   residual, symmetry, centered-mean handling, and finite coefficients over
   dense fp32/fp16 grids.
2. Sampling tests prove that exact `s=0` performs the analytic Euler update with
   no U-Net or negative-branch call and that partial img2img does not take it.
3. Training tests prove the U-Net target is `r_target`, remains unnormalized,
   and reconstructs the path velocity.
4. CFG tests prove the analytic skip is not guided and both endpoint residual
   deltas are zero.
5. Artifact and resume tests reject every v1/v2 semantic mismatch before tensor
   loading.
6. MNT/adaptive tests bin by effective log-SNR, preserve deterministic batch
   order over resume, and do not import v1 controller state.
7. Fixed-seed v1 samples compare the old first learned step against the new
   analytic startup in production and through the live training-sample API.
8. A new scratch v2 run demonstrates finite per-bin losses and CFG probes before
   any quality or convergence claim is made.
9. At equal NFE, uniform-`s` and bounded equal-log-SNR Euler grids are compared
   against a high-NFE trajectory reference. The production grid is selected
   from measured integration error and fixed-seed diagnostics.
10. Before the full run, a calibration pass records latent RMS, `c_skip`, raw
    residual target RMS, learned residual RMS, and their maxima per log-SNR bin.
    The run is not accepted if non-finite values, unrecorded middle-bin loss
    concentration, or a material train/sample scale mismatch is observed.

## 10. Implementation sequence

1. Land and measure the v1 analytic first-step policy.
2. Add pure path/preconditioning algebra and tests.
3. Add versioned artifact/load/resume contracts.
4. Wire v2 training targets and metrics.
5. Wire Euler generation, CFG, img2img/inpaint, and probes.
6. Update MNT/adaptive log-SNR handling and frontend metrics.
7. Build a new scratch artifact and run API training/generation smoke tests.

No v2 code path is enabled merely by the presence of this document.
