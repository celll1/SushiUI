# SenseNova SDXL Chimera v4: resolution-calibrated difficulty time

Status: **design only**. Run 141 remains a v3 artifact and training contract.
Its noise-biased timestep distribution is an inexpensive measurement phase,
not a v4 implementation and not an artifact-format change.

v3 fixes the radial CFG failure structurally, but its path coordinate is not a
uniform measure of denoising difficulty. The same path coordinate also produces
different visible destruction at different spatial resolutions. v4 separates
those concepts:

> Sample and integrate uniformly in difficulty time; map difficulty time to the
> v3 geometric path with a monotone, resolution-conditioned calibration.

This preserves v3's pointwise polar decomposition and tangent-only CFG. It
changes only how quickly the trajectory advances through that geometry.

## 1. Problem statement

v3 uses clean-time `s`, with `s = 0` at Gaussian noise and `s = 1` at clean
data. Its angular schedule rotates toward the clean direction early. In Run 141,
the median of `logit_normal(mean=0, std=1)` is `s = 0.5`, so much of the
training mass lands where high-resolution samples already preserve most of
their recognizable structure.

This has two distinct causes:

1. Path geometry: `s` is a polynomial interpolation parameter, not a calibrated
   measure of inverse-problem difficulty.
2. Spatial redundancy: at equal per-coordinate corruption, a larger latent has
   more correlated observations from which low-frequency and semantic structure
   can be recovered.

A scalar latent cosine or nominal log-SNR cannot capture the second effect.

## 2. Measurement from Run 141

The step-1000 debug target was resized to three resolutions, encoded with the
artifact's bundled SDXL VAE, corrupted with the production v3
`polar_flow_target`, and decoded with the same VAE. The prompt was not read or
used. One fixed noise seed family was used across the comparison.

The table reports low-frequency RGB correlation between the resized target and
the decoded noised latent:

| pixels | latent | `s=0.25` | `s=0.50` | `s=0.71875` |
|---:|---:|---:|---:|---:|
| 288 x 512 | 36 x 64 | 0.522 | 0.817 | 0.906 |
| 576 x 1024 | 72 x 128 | 0.647 | 0.876 | 0.928 |
| 864 x 1536 | 108 x 192 | 0.727 | 0.898 | 0.929 |

The clean-direction cosine in latent space was nearly resolution invariant:

| `s` | clean-direction cosine |
|---:|---:|
| 0.25 | about 0.63 |
| 0.50 | about 0.923 |
| 0.71875 | about 0.992 |

The divergence between these two measurements is the important result. The
path applies almost the same global angular corruption, while decoded semantic
structure becomes easier to recover as the number of spatial observations
grows. Resolution therefore belongs in the difficulty calibration.

These three points are evidence for the effect, not enough data to fit the
production mapping. A production calibration requires multiple images, aspect
ratios, noise draws, and path coordinates.

## 3. Immediate v3 mitigation

For a run dominated by one resolution, shifting the existing sampling density
toward the noise endpoint is the cheapest response. It changes neither the v3
path nor its targets and remains resume-compatible.

Run 141 uses the following provisional policy after the measurement:

```yaml
timestep_sampling:
  distribution: logit_normal
  min_timestep: 0.02
  max_timestep: 0.99
  mean: -0.7
  std: 1.2
```

This has median `s = sigmoid(-0.7) ~= 0.33`, assigns roughly 37 percent of mass
below `s = 0.25`, and retains roughly 11 percent below `s = 0.10`. The lower
bound avoids making the high-variance noise-end tangent target the dominant
training problem. Clean and middle regions remain represented.

This policy is intentionally resolution-specific. It must not be copied as a
universal default for mixed-resolution training.

## 4. Difficulty coordinate

Introduce a normalized difficulty-time coordinate `d in [0, 1]`, with the same
orientation as v3:

```text
d = 0: maximally noisy / hardest
d = 1: clean / easiest
```

Let `R` denote resolution metadata, minimally latent height and width. A
strictly increasing warp maps difficulty time to geometric path time:

```text
s = T(d; R)
T(0; R) = 0
T(1; R) = 1
dT/dd > 0.
```

`T` is chosen so equal increments in `d` produce approximately equal changes
in measured denoising difficulty. At high resolution, an interior `d` generally
maps to a smaller `s` than at low resolution because more corruption is needed
to destroy the same amount of recoverable structure.

Uniform training in `d` then becomes meaningful. It does not require users to
hand-design a different timestep density for every resolution.

## 5. Reparameterized v3 targets

v4 retains the exact v3 polar state at the mapped path coordinate:

```text
z_v4(d; R) = z_v3(T(d; R)).
```

By the chain rule:

```text
lambda_d = lambda_s * dT/dd
tau_d    = tau_s    * dT/dd
dz/dd    = dz/ds    * dT/dd.
```

The network is conditioned on `d`, `log(rho)`, and the existing resolution and
crop conditioning. It predicts `lambda_d` and `tau_d`. Tangent projection,
radial anchoring, and tangent-only CFG are unchanged, so CFG still cannot
directly extrapolate radius.

The polar solver advances with `Delta d`. Its radial exponential and spherical
exponential updates consume the reparameterized radial and tangent velocities.
The U-Net NFE count is unchanged; the warp and its derivative are negligible
scalar work.

Because the timestep embedding and target scale change, this is a new training
contract. A v3 checkpoint may be evaluated as a trunk warm start, but its
optimizer state and prediction heads are not assumed resume-compatible.

## 6. Calibrating the warp

### 6.1 Required properties

The fitted warp must be:

- monotone in `d` for every supported resolution;
- continuous with a bounded derivative;
- exact at both endpoints;
- interpolable across latent area and aspect ratio;
- artifact-owned and identical in training and generation;
- fitted offline, never recomputed from the current minibatch target.

The last rule prevents target leakage and makes generation deterministic.

### 6.2 Difficulty measurement

No single metric is sufficient. Calibration should measure a model-independent
destruction curve over held-out latents with at least:

- multiscale decoded low-frequency correlation;
- a perceptual feature distance or correlation;
- latent local-structure correlation at several pooling scales;
- optional edge or segmentation stability as a semantic cross-check.

Metrics are normalized between their empirical noise and clean endpoints,
combined with fixed recorded weights, and monotonized with isotonic regression.
The result is an empirical recoverable-structure curve `C_R(s)`. Difficulty time
is its normalized monotone coordinate, and `T` is the inverse lookup.

VAE decoding is required only for offline calibration and validation. Ordinary
training does not decode samples to choose timesteps.

### 6.3 Compact representation

The initial artifact representation should be a small table over:

```text
(log latent area, aspect-ratio magnitude, d) -> s
```

with monotone cubic interpolation along `d` and bounded linear interpolation
across resolution anchors. A parametric log-SNR shift may be tested as a
compression of the table:

```text
ell_eff(s, R) = ell_proxy(s) + kappa * log(N_eff(R) / N_ref)
```

but it is accepted only if its error against the empirical curves is small.
The v3 polar path is not a linear Gaussian mixture, so nominal SNR alone is not
the ground truth.

## 7. Training distribution and MNT

The v4 default sampler is uniform in `d`. Distribution metrics must show both
realized `d` density and the induced `s` density; presenting only `s` would make
a correct resolution-conditioned run look nonuniform.

MNT remains compatible. Stratification is performed in calibrated `d`, because
that is the coordinate intended to represent difficulty. Samples with different
resolutions may map the same stratum to different `s` values.

Adaptive timestep control, if enabled later, operates on learning progress in
`d` bins. It must not use raw noise-end tangent MSE as difficulty because v3's
irreducible angular variance remains present. The v3 observation-only safety
rule remains in force until a progress-based controller is validated.

## 8. Generation

Generation selects a uniform or explicitly configured grid in `d`; each point
is mapped through the artifact calibration for the requested output resolution.
Training and generation therefore traverse the same resolution-conditioned
path. Image-to-image strength is specified in `d`, while diagnostics expose
the corresponding `s` to keep the geometry auditable.

Crop and aspect-ratio conditioning remain network inputs. The path warp uses
latent canvas dimensions, not the uncropped source-file dimensions.

## 9. Rejected shortcuts

- **One global noise-biased distribution:** useful for Run 141, but cannot
  equalize difficulty across resolutions.
- **Latent cosine alone:** the experiment shows it is nearly resolution
  invariant while decoded structure retention is not.
- **Per-batch online metric inversion:** leaks target information into sampling,
  destabilizes resume behavior, and has no generation-time equivalent.
- **Changing only generation timesteps:** creates a train/generation mismatch.
- **Changing only the path without the chain-rule target:** learns the wrong
  vector field for the new time coordinate.

## 10. Implementation phases

1. Build an offline calibration probe over representative held-out images,
   resolutions, aspect ratios, `s` values, and noise seeds.
2. Fit and cross-validate monotone destruction curves and a compact warp table.
3. Add an artifact-versioned calibration block and pure mapping functions for
   `T(d; R)` and `dT/dd`.
4. Add v4 target construction and unit tests for endpoints, monotonicity, chain
   rule, tangent orthogonality, and finite derivatives.
5. Wire uniform-`d` training, MNT stratification, resume state, and dual `d/s`
   metrics.
6. Wire the polar solver and img2img strength through the same mapping.
7. Start a new v4 run; do not relabel a v3 continuation as v4.

## 11. Acceptance gates

- Across calibrated resolutions, equal `d` bins have substantially closer
  destruction scores than equal `s` bins.
- The mapping is strictly monotone and endpoint-exact for all supported canvas
  sizes; its derivative is finite and bounded.
- Numerical finite differences of `z_v4(d)` agree with the transformed radial
  and tangent targets.
- Tangent-only CFG retains v3's pointwise radial-invariance assertion.
- Uniform-`d` training covers every difficulty bin without concentrating on the
  irreducible noise-end angular floor.
- Equal-NFE generation does not regress against v3 at the reference resolution
  and improves cross-resolution consistency.
- Checkpoint resume preserves the calibration identity, sampler state, epoch
  order, MNT trajectory, and realized `d/s` metrics.

Until these gates pass, the shipped architecture remains v3 and the Run 141
distribution shift remains an explicitly provisional experiment.
