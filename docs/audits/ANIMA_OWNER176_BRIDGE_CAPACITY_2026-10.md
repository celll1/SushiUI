# Learned Anima/Qwen VAE bridge capacity and update audit

Date: 2026-10-12. Owner run 176 was stopped normally and saved at step 134832.
The actual saved bridge, head, timestep embedding and final AdaLN weights were
used. No checkpoint parameter or run setting was modified by the probes.
The run was requested to resume from that checkpoint after GPU verification.

## Question and answer boundary

The question is whether a permanent codec-alignment path restricts the denoiser
to jagged outputs, rather than whether the target Qwen VAE can decode detail.
Earlier claims about a full-rank output linear layer did not test the complete
bridge or final normalized-head domain. Treat those as partial evidence.

The learned **convolutions alone** can transmit a detailed target. However,
adding the actual final LayerNorm/AdaLN constraints makes the same target much
harder to fit and leaves visible detail errors in this bounded test. Therefore
this audit does **not** establish that the complete swapped model is free from
a capacity/conditioning bottleneck. It also does not prove a structural
impossibility or identify the unique cause of the owner run's jaggies.

## Checkpoint and training evidence

`SpatialAlignmentPair` installs two 64-to-64 3x3 convolutions for this swap.
Both declare packing factors 1; neither downsamples the external latent grid.
The original codec was 16-channel/8x; the target is 64-channel/16x.
The two convolutions together have 73,856 weight/bias parameters.

The original calibration kernels are retained in frozen Gaussian-alignment
buffers. Reconstruct the initialization by placing these into the original
16-channel rows/columns, with center identities for the added 48 channels.
Compare to the checkpoint after rounding the initialization to the saved dtype.

| Learned kernel | Changed coefficient fraction | L2 change / initial L2 |
|---|---:|---:|
| Input | 78.3284% | 0.234073 / 7.188390 = 3.2563% |
| Output | 75.4584% | 0.325832 / 14.580986 = 2.2346% |

Both added-channel sections also changed; this is not merely comparing a
rounded saved tensor against an FP32 initializer. The saved AdamW8bit optimizer
has named state for input weight/bias and output weight/bias. All four record
step 134832, configured learning rate 5e-6 and nonzero moment information.
For uint8 moment tensors, the report's raw code norms are **not** moment norms;
nonzero `absmax` scales are reported separately. Bias moments are FP32 tensors.
The live gradient log independently reports nonzero input-kernel and bias
gradients, including step 134770.

This establishes optimizer inclusion, gradient activity and kernel updates.
It does not establish that optimization has learned a well-conditioned map or
that every coefficient changes at every step.

## Frequency conditioning

Evaluate the complex circular-convolution symbol at a 17x17 frequency grid,
using FP64 SVD of the actual saved kernels.

| Kernel | Minimum sampled rank at relative threshold 1e-6 | Worst sampled condition number |
|---|---:|---:|
| Input | 64 | 195,499 |
| Output | 64 | 8,749 |

No rank loss was observed on these sampled periodic symbols, but both contain
weak directions. This is not a proof of full finite zero-padded operator rank,
nor of good conditioning at untested frequencies. Finite input-map inversion
also needs care: after 1,600 conjugate-gradient iterations, general-image map
residual RMSE was 4.15e-5 while recovered-input latent RMSE was 0.09496. A tiny
map residual is not equivalent to recovering its poorly conditioned input.

## Fixed-output-path witness

Use one API-verified `rating:general` image at 1472x896 and a synthetic
512x512 chart with fine diagonals, circles and 2px-period stripes. Encode with
the checkpoint-declared target VAE in FP32, posterior mode, using the runtime
normalization facade. The direct codec roundtrip is the image reference.

Seed 73 produces Gaussian noise. Set sigma to **1**, so the input contains no
clean-image signal. Desired velocity is `noise - clean`. Keep the checkpoint's
Gaussian noise complement and flow-velocity bias convention in the output.

For a zero-padded convolution A and fixed affine/noise term c, solve

`min_u ||A u + c - (noise - clean)||^2`

with 6,000 FP32 conjugate-gradient iterations of the normal equations. The
kernel and bias remain fixed. Reconstruct `clean = noise - predicted_velocity`;
there is no low-noise identity retention in this witness.

The saved 256x2048 linear head is also held fixed. Its FP64 pseudoinverse
constructs a post-AdaLN feature for each token that emits u, respecting the
actual phase-major/channel-minor unpatch ordering. Evaluate FP32 and BF16
head/bridge boundaries, followed by the same FP32 target VAE decode.

| Target / output path | Pixel RMSE vs codec reference, RGB [0,1] |
|---|---:|
| General, free post-AdaLN + fixed head/bridge, FP32 | 0.003517 |
| General, same witness with BF16 head/bridge | 0.014450 |
| Synthetic, free post-AdaLN + fixed head/bridge, FP32 | 0.009162 |
| Synthetic, same witness with BF16 head/bridge | 0.026031 |

Visual inspection shows preserved general-image contours and detail through
the convolutions; BF16 and extreme synthetic stripes have additional errors.
This rules out "these convolutions always impose coarse jagged blocks" for
these targets. It is a target-aware inverse witness, **not sampled generation**
and **not evidence that the DiT can produce these internal features**.

## Actual normalized-head constraint

The final head uses `linear(LayerNorm(x) * (1 + scale(sigma)) + shift(sigma))`.
Include the saved time embedding, RMSNorm, AdaLN low-rank weights and time
residuals, rather than treating arbitrary post-AdaLN features as realizable.

For the particular inverse witness above, the minimum-norm, mean-zero
normalized feature needed to emit each pre-bridge patch has RMS 5.73-39.09 on
the general image (median 11.05). None meet LayerNorm's approximately unit
variance. This invalidates promoting that witness to a complete-head proof.
Other pre-bridge approximate solutions are not excluded by this result.

Then optimize **only free pre-LayerNorm token features**, through the actual
frozen FP32 final head and bridge, for 2,500 Adam iterations. Checkpoint weights
are not optimizer parameters. This respects the normalized-head domain and
removes DiT trunk learning and attention from the test.

| Target | Initial velocity MSE | Final velocity MSE | Pixel RMSE vs codec reference |
|---|---:|---:|---:|
| General | 3.08034 | 0.0178664 | 0.0448454 |
| Synthetic | 3.07888 | 0.0142352 | 0.0741964 |

The constrained fit retains roughness and edge errors. It was still improving
at the budget limit; this is not a globally optimal solution or an impossibility
proof. Unlike the free witness, it does not establish a visually equivalent
non-jaggy output through the actual head domain. The constrained case is FP32;
no BF16 constrained-optimization acceptance is claimed.

## Consequences

- Codec compression alone is not the diagnosed cause.
- Both learned bridge kernels are being updated; a frozen/unregistered bridge
  is not the observed explanation.
- Permanent convolutions do not themselves force coarse-block output, but
  their poor conditioning and interaction with normalized-head range remain
  concrete concerns supported by these measurements.
- Do not label this simply "insufficient long training", or promote a new
  adapter as the established remedy. First compare complete output paths with
  better-conditioned, target-native residual parameterizations, including
  final normalization and noise convention. Such changes are not implemented
  by this audit.

[Numeric report](data/anima_owner176_bridge_capacity_2026-10.json).
Private reproducible probes and general-only images remain in ignored
`local/verification/owner176_bridge_capacity.py` and
`local/verification/owner176_bridge_capacity/`. The `general_detail_comparison.png`
has codec reference, unrestricted FP32 witness, unrestricted BF16 witness and
the actual normalized-head fit, in that order. No owner image, caption, latent,
optimizer state or checkpoint is committed.
