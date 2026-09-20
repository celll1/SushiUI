# SenseNova SDXL Chimera v4: destruction-coordinate tangent flow

Status: **implementation contract**. This document replaces the earlier,
unimplemented resolution-calibrated difficulty-time proposal. Resolution-aware
sampling remains a possible policy on top of this contract; it is not a new
prediction target.

Chimera v3 made CFG pointwise radial-safe, but coupled three separate jobs to
one angular schedule:

1. how much clean-image direction remains in a noised training state;
2. the scale of the tangent regression target;
3. how strongly text conditioning may distinguish cond from null.

The Beta(3, 2) schedule improved the first job at the reference resolution, but
its derivative vanishes at the noise endpoint. Consequently the learned
tangent and its loss also vanish there. Run 143 demonstrated the resulting
failure: middle-time single-step reconstructions acquired structure while a
free rollout remained on essentially the same coarse trajectory; a debug item
at `t=0.0381` decoded to the latent posterior mean rather than a structured
image.

v4 keeps the measured Beta(3, 2) destruction path while separating target units
and conditioning confidence from path speed:

> The U-Net predicts tangent displacement per unit destruction coordinate.
> Conditioning is admitted only in proportion to the clean direction already
> observable in the current state.

## 1. Coordinates and orientation

As in v3, `t = 0` is noise and `t = 1` is clean. Define the artifact-owned
destruction coordinate

```text
d(t)  = 4 t^3 - 3 t^4
d'(t) = 12 t^2 (1 - t).
```

`d` is the Beta(3, 2) CDF used by Run 143's v3 angular path. It is monotone,
endpoint-exact, and terminal-flat. The polar state is unchanged from that v3
path:

```text
n(t) = Exp_n0(d(t) Log_n0(n1))
z(t) = alpha(t) mu + rho(t) n(t).
```

Therefore every clean/noise pair produces exactly the same training input at
the same `t` as the v3 Beta(3, 2) artifact. v4 does not undo the path's measured
destruction profile.

## 2. Why v3 wastes the noise endpoint

v3 regresses the tangent with respect to path time:

```text
tau_t* = rho d'(t) theta e(t).
```

At `t ~= 0`, both the state rotation `d(t)` and the target multiplier `d'(t)`
are tiny. More noise-end samples do not fix this target collapse. Conversely,
replacing `d(t)` with a linear angular path would restore target scale but
would also make the middle of the path substantially easier, reversing the
reason Beta(3, 2) was introduced.

The endpoint also has an empirical-conditioning failure. In a finite dataset,
a nearly unique caption often identifies one training image, so a conditional
noise-end regression can spend capacity memorizing `caption -> image`. The
null branch cannot do the same and instead sees the broad dataset posterior.
Their difference is precisely the component CFG extrapolates.

The relevant costs are not only parameter count. High irreducible target
variance consumes gradient budget and training steps, while empirical
caption-image collapse consumes representation capacity.

## 3. v4 tangent target

The tangent head predicts displacement per unit `d`, not velocity per unit
`t`:

```text
tau_d* = dz_tangent / dd = rho theta e(t).
```

It is pointwise orthogonal to `n(t)`. The radial head remains in `t` units:

```text
lambda_t* = d rho / dt.
```

This mixed-coordinate decomposition is exact because radial and tangent
components are orthogonal:

```text
dz = (alpha' mu + lambda_t n) dt + tau_d dd.
```

The loss remains explicit and unnormalized:

```text
L = MSE(lambda_pred, lambda_t*) + MSE(tau_pred, tau_d*) + L_REPA.
```

No division by `d'(t)` occurs in training, inference, or diagnostics. In
particular, v3 weights cannot be converted by `tau_t / d'(t)` because that
expression is singular at both endpoints.

## 4. Conditioning reliability gate

High-dimensional independent noise and data directions concentrate at angle
`theta ~= pi/2`. At destruction coordinate `d`, alignment with the clean
direction is therefore

```text
R(d) ~= cos((1 - d) pi/2) = sin(pi d/2).
```

The fraction of clean-direction energy explained by the current state is

```text
g(d) = R(d)^2 = sin^2(pi d/2).
```

This fixed, parameter-free reliability gate is part of the artifact contract.
It satisfies `g(0)=0`, `g(1)=1`, and has zero slope at both endpoints.

The U-Net field is conceptually decomposed as

```text
tau_cond(z,t,c) = tau_base(z,t) + g(d(t)) Delta_tau_cond(z,t,c).
```

At the noise endpoint, cond and null must be identical even when the empirical
caption identifies one image. The unconditional base field remains active; v4
does not suppress all generative transport as v3 Beta(3, 2) did.

The implementation gates the *conditioning contribution*, not the complete
U-Net output and not the target. It must be applied where cond differs from the
context-free base, so LayerNorm biases or nonzero empty-prompt embeddings
cannot evade the endpoint equality. The gate covers native-prefix attention,
the pooled-conditioning contribution, and future reference-image context.

CFG remains tangent-only:

```text
tau_cfg = tau_null + w (tau_cond - tau_null).
```

Because both branches share their base and their difference carries `g(d)`,
CFG vanishes analytically at `d=0`. Radial velocity remains anchored to the
positive branch and is never CFG-extrapolated.

## 5. Solver

For one interval `t_i -> t_(i+1)`, define

```text
Delta_t = t_(i+1) - t_i
Delta_d = d(t_(i+1)) - d(t_i).
```

The radial exponential update continues to use `Delta_t`. The spherical
exponential update uses `Delta_d`:

```text
rho_next = rho exp(Delta_t lambda_t / rho)
phi      = Delta_d ||tau_d|| / rho
n_next   = Exp_n(phi tau_d / ||tau_d||).
```

The solver is still one NFE per learned interval. The first interval may use
the existing analytic radial step with zero learned tangent. This is an
explicit discrete policy: it avoids asking the model at the exact empirical
endpoint, while all subsequent learned intervals use nonvanishing `tau_d`
units.

An angular cap, when configured, applies to `phi` after multiplication by
`Delta_d`. Radius positivity and tangent-only CFG invariants are unchanged.

## 6. Local clean recovery and diagnostics

The v4 clean-direction estimate uses `tau_d / rho`; it never divides by
`d'(t)`. Exact target fields recover the paired clean endpoint at every
nonsingular interior state. A learned prediction near pure noise may still
decode to a posterior mean. That is a statistical statement, not numerical
amplification by a vanishing schedule derivative.

Debug comparisons must use fixed held-out image/noise pairs at fixed
coordinates. A random first batch item at a random timestep is retained for
batch inspection but is not a convergence curve. The standard diagnostic set
is `t = 0.04, 0.15, 0.33, 0.50, 0.75` plus a free rollout with a fixed seed.

Required v4 probe fields include:

- `destruction_coordinate` and `delta_d`;
- base, cond, null, and guided tangent RMS;
- effective conditioning difference before and after `g(d)`;
- angular displacement and cap scale;
- radial update and CFG radial delta;
- fixed-coordinate local reconstruction error;
- free-rollout low-frequency structure and periodicity diagnostics.

## 7. Training and timestep policy

Run 143's noised-state distribution remains a valid starting policy because
the v4 state path is identical. Sampling policy is expressed in `t`, while
metrics also expose the induced `d` distribution. MNT and stratification remain
compatible.

Adaptive timestep control must not interpret raw noise-end `tau_d` MSE as
reducible difficulty. Its irreducible unconditional angular variance remains
large near pure noise even though conditional memorization is gated away.
Observation mode is the initial v4 default.

Resolution-aware distribution or grid shifts from the retired v4 draft may be
added later as sampling policy. They do not change the v4 target or artifact
format.

## 8. Artifact and compatibility contract

v4 uses a new prediction declaration and artifact format. It must record:

```yaml
prediction:
  type: destruction_coordinate_polar_flow
  path: observable_polar_geodesic_v1
  radial_units: dt
  tangent_units: beta_3_2_d
  destruction_schedule: confidence_gated_beta_3_2_v1
  conditioning_gate: clean_alignment_squared_v1
  cfg_mode: tangent_only_v1
  radial_anchor: positive_condition
  integrator: mixed_coordinate_polar_exp_euler_v1
```

v3 artifacts retain their existing semantics and solver. Loaders must never
infer v4 from an angular schedule name alone.

A v3 checkpoint is not resume-compatible. The supported warm start is:

- inherit the U-Net trunk and time embedding;
- inherit the condition bridge;
- inherit the radial head;
- inherit the REPA projector when stored by the training checkpoint;
- reinitialize `conv_out`, the tangent prediction head;
- start with a fresh optimizer and global step zero.

## 9. Implementation phases

1. Add pure `d(t)`, reliability-gate, target, recovery, and mixed-coordinate
   solver functions with finite-difference and endpoint tests.
2. Add a strict v4 artifact contract and loader dispatch while preserving all
   v1-v3 behavior.
3. Gate every conditioning contribution and assert cond/null endpoint equality.
4. Wire v4 training, diagnostics, generation, img2img, checkpoint save, and
   warm-start conversion.
5. Add API/default/schema/frontend exposure only where a user-facing choice is
   required; artifact-owned constants are not duplicated as settings.
6. Build a v4 warm-start artifact from Run 143, stop Run 143 cooperatively, and
   start a new step-zero run with fresh optimizer state.

## 10. Acceptance gates

- v4 and v3 Beta(3, 2) produce identical noised states at equal `t`.
- Finite differences with respect to `d` agree with `tau_d*`.
- `tau_d*` is finite and nonzero for a generic orthogonal pair at `t=0`; no
  operation divides by `d'(t)`.
- Exact v4 targets recover the paired clean latent at interior coordinates.
- Mixed-coordinate solver radius and direction agree with small-step reference
  integration and preserve unit direction.
- Cond and null outputs are equal at `d=0` for arbitrary text contexts.
- CFG changes no radial update at any guidance scale.
- A v3 artifact cannot enter the v4 solver, and a v4 artifact cannot enter the
  v3 solver.
- Warm start inherits the declared trunk, bridge, radial head, and REPA state,
  resets the tangent head, and begins with no optimizer state.
- Fixed-coordinate debug images and free rollout are both produced before a
  v4 run is judged against Run 143.
