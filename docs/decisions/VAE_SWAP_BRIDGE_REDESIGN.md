# Spatial VAE swap bridge redesign

Status: design decision, 2026-10-12. Not implemented. Supersedes the sampling
profile work as the primary fix for run 176's jagged detail; that work stays a
separate, lower-priority track ([sampling decision](VAE_SWAP_SAMPLING_METHOD.md)).

## Problem

Run 176 (native Anima DiT, Qwen-Image 2.1 VAE 64ch/16x, paired_spatial,
Gaussian noise alignment) shows shard/staircase texture from step 0 that does
not improve over 134k steps while concepts are learned. The step-0 texture is
the expected consequence of replacing the native Qwen-Image VAE (16ch/8x). The
defect is that training cannot remove it.

## Evidence

### Bridge capacity without the DiT

`examples/vae_swap_bridge_capacity_probe.py` loads only the two codecs and the
step-134832 bridge. One rating:general image at 512x768 (x1) and a 2x Lanczos
enlargement (x2). R² is per channel, spatial mean removed, averaged over target
channels 0-15 / 16-63:

| Path | x1 R² 0-15 / 16-63 | x2 R² 0-15 / 16-63 | Decoded |
|---|---:|---:|---|
| Qwen 2.1 codec round trip | 1 / 1 | 1 / 1 | sharp |
| Half-resolution source latent -> `A_out` (step-0 clean estimate) | -0.01 / -0.09 | -0.69 / -0.75 | run-176 shards |
| True target `y` -> `A_in` -> `A_out` (16ch bottleneck) | -0.04 / -0.05 | -0.94 / -0.72 | run-176 shards |
| Bottleneck, channels 16-63 replaced by exact `y` | -0.04 / 1 | -0.94 / 1 | much better, streaks/ghosting remain |

Even with perfect information, the path through the 16 source-like channels
reproduces the artifact. The codec is not at fault. A single image is used; the
x2 trend is indicative only.

### What 134k steps of training did

Weights read from the step-134832 checkpoint:

| Quantity | Value |
|---|---|
| `x_embedder` column norm, new channels 16-63 vs source channels 0-15 | 0.115 vs 1.219 |
| `final_layer.linear` row norm, new vs source channels | 0.080 vs 0.252 |
| Output bridge drift from its frozen calibration | 0.19% |
| Input bridge drift from its frozen calibration | 2.3% |

The DiT still reads and writes almost entirely through the 16 calibrated
channels, so its output stays near the bridge's regression.

Sample-gradient periodicity at 16/32px shows no consistent boundary phase, so
the texture is not a token seam.

## Root causes

Current construction is in `SpatialAlignmentPair.initialize_from_pair`
([latent_adapter.py](../../backend/core/models/components/latent_adapter.py)).
`old`/`new` are source/target channel counts, `width=max(old,new)`.

1. **Index passthrough.** Backbone channels `old..width` are identity copies of
   target channels with the same index. Target channels `0..old` have no direct
   path. They are reachable only through `A_out` applied to the 16 channels the
   pretrained body treats as source latents, so detail in those target
   channels conflicts with the pretrained meaning of those channels. Channel
   index has no meaning in the target codec, so the split is arbitrary.
2. **Noise coupling.** Backbone channels `0..old` carry `W·ε`, where `W` mixes
   all 64 target channels. Channels `old..width` carry `ε[old:]`. The two groups
   are correlated rather than independent white noise.
3. **Learnability.** New `x_embedder` columns and `final_layer` rows start at
   zero. They are trained as slices of pretrained tensors at the body LR
   (5e-6 in run 176) with no separate group. The bridges also use the body LR.
   A path that is zero at both ends grows slowly, which matches the measured
   norms. This is an inference, verified only by those norms.
4. **Calibration scale.** Calibration fits 256px target images against 128px
   source images (`calibration_source_side`). The parameter only accepts
   128/256/512, while training uses 1024-1536. Bridge R² falls further at x2.

Secondary: patch 2 on a 16x codec gives 32px tokens, four times the native area
per final-layer projection. This is not changed here.

## Design

### D1. Invertible target basis (declaration version 2)

Applies when `new > old`. Let `A_c` be the `old x new` center tap of the
calibrated input bridge, with SVD `A_c = U Σ Vᵀ` and `W = U Vᵀ`, the existing
whitening map. Let `V⊥` (`new x (new-old)`) be an orthonormal complement of
`V`.

Input (backbone state `b`):

```
b[:old]  = A_in · y                     (unchanged calibrated 3x3, source-like)
b[old:]  = V⊥ᵀ · y                      (1x1, frozen buffer, replaces index copy)
```

Output (target velocity):

```
v_y = G · v_b[:old] + V⊥ · v_b[old:] + R(ε)
```

- `G` is a 3x3 conv `old -> new` fitted by ridge regression on calibration data
  so that `G(A_in y) ≈ V Vᵀ y`. This is the component of `y` that lies in the
  source-like subspace. If `A_in` were 1x1, this would be exact as
  `V Σ⁻¹ Uᵀ`. `G` replaces the current `A_out`.
- `V⊥` is frozen. The target-native round trip `y -> b -> y` is then near
  identity for every target direction, including the directions current
  channels `0..old` could not reach.
- `R(ε) = V Vᵀ ε − G(W ε)` is the frozen noise complement. It replaces
  `ε − A_out W ε`.
- Backbone noise is `[W ε; V⊥ᵀ ε]`, an orthogonal transform of `ε`, so all
  backbone channels see independent unit Gaussian noise. `GaussianNoiseAlignment`
  keeps the input delta for `b[:old]` and stores `V⊥` plus the new residual.

Step-0 behaviour is the same as v1: new IO weights are zero, so output equals
`G·v_b[:old] + R`, which matches the current regression path. Version 2 does
not change the initial image. It removes the structural ceiling and the noise
coupling.

Scope: shared component, used by every arch that calls `initialize_image_swap`
or `SpatialAlignmentPair`. Version 1 declarations keep their current behaviour
and continue to load. New swaps with `new > old` default to version 2.
`new <= old` is unchanged.

### D2. Learnable latent IO

- Split the latent-facing modules into source and new parts. `x_embedder` keeps
  its pretrained columns, and a separate `Linear` holds the new-channel columns
  and is summed. `final_layer.linear` keeps its pretrained rows, and a separate
  `Linear` holds the new rows, concatenated in the declared `(p1 p2 t C)` order.
  Merge the parts back into the original tensors on save and export, so
  checkpoint keys and shapes are unchanged.
- Add optimizer groups and parameters `vae_swap_io_lr_factor` and
  `vae_swap_bridge_lr_factor`, both multiplying the body LR. Defaults go in
  `param_defaults.TRAINING_DEFAULTS`. Follow the OpenAPI-first and frontend
  threading rules, and preserve the factors on resume.
- Keep zero initialization. Zero output is the correct "no information" state,
  and D1 gives the new path a real target.
- Implement per arch through `LatentIOSpec`. Anima is first; other image
  backbones follow the same spec.

### D3. Calibration at training scale

- Allow `vae_swap_calibration_resolution` up to the largest training base
  resolution (at least 1024 and 1536). Keep half-side source encoding for 16x
  over 8x.
- Default the resolution to the run's smallest `base_resolutions` entry instead
  of 256. Raise the default sample count to 64.
- Memory is encode-only and streamed into the existing Gram accumulators. The
  input regressor for 64ch x 9 taps needs at least ~10x its 576 features in
  observations. Each 1024px image gives 4096, so 64 images is ample.

### Separate track: native Anima schedule

SushiUI's latent-area exponential shift gives 51.676 at 1472x896/40 steps with
the native codec, so every evaluation is above sigma 0.5. That is a native
generation defect independent of this redesign. Restore the audited shift 3
baseline as profile 1 of the sampling decision. For Qwen 2.1 swap runs (3.776),
schedule work is not expected to fix the jaggies.

## Migration of run 176

Version 1 checkpoints keep loading and generating. To continue run 176's body
under version 2, provide an explicit re-alignment step:

1. Keep the body, the source columns/rows of the latent IO, and the LLM adapter.
2. Recalibrate `A_in` and `G` at training scale and build `V⊥`.
3. Reset the new IO parts to zero. They are near zero now, so little is lost.

This is a declared conversion with its own metadata, not a silent reload. The
existing guard "Changing an aligned Anima VAE requires a fresh base
checkpoint" stays for any other codec change. Optimizer state for the new IO
parts starts fresh.

## Acceptance gates

| Gate | Check | Pass |
|---|---|---|
| G1 Bridge round trip | Probe arm `y -> b -> y` with D1 at x1 and x2, calibration at training scale | R² ≥ 0.99 on all 64 channels; decode visually equal to the codec round trip |
| G2 Noise | Backbone noise covariance from 10⁴ draws | Within sampling error of I; no cross-group correlation |
| G3 Step 0 | Same seed/prompt as v1 at step 0 | No worse than v1 |
| G4 Smoke | ~3 training steps, version 2 + D2 groups, FP32/BF16 production dtype | Finite loss; new IO parts receive non-zero updates at their factor |
| G5 Resume/export | Save, reload, resume; merged keys equal v1 layout | Optimizer groups restored; generation works |
| G6 Owner run | Compare frozen checkpoints under identical seed/prompt/NFE | Contours improve over run 176 at matched steps; new IO norms grow |

G1-G5 run without long training. G6 is the owner's gate. A falling loss alone
does not pass G6.

## Implementation order

1. D1 in the shared bridge, plus a probe arm for G1/G2. Commit.
2. D3 calibration scale. Commit.
3. D2 for Anima: split IO, optimizer groups, merge-on-save, API parameters.
   Commit.
4. Run 176 re-alignment conversion. Commit.
5. Extend D2 to other image arches as needed.

The comparison review in CLAUDE.md applies to steps 1 and 3.
