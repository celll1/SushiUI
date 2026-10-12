# Spatial VAE swap bridge redesign

Status: design decision, 2026-10-12. D1 and D2 are implemented for Anima; D3
and the run 176 conversion are pending. Supersedes the sampling profile work as
the primary fix for run 176's jagged detail; that work stays a separate,
lower-priority track ([sampling decision](VAE_SWAP_SAMPLING_METHOD.md)).

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

The independent [bridge capacity audit](../audits/ANIMA_OWNER176_BRIDGE_CAPACITY_2026-10.md)
agrees from another side. With all 64 backbone channels free, the learned
convolutions can transmit a detailed target. However, the input map has sampled
condition numbers up to 195,499, and fitting through the real normalized final
head leaves roughness. Both point to conditioning of the output path rather
than codec capacity.

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

### D1. Full passthrough (declaration field `passthrough: "full"`)

Implemented. Applies when `new > old`. The backbone width becomes `old + new`
instead of `max(old, new)`:

```
b[:old] = A_in · y        (calibrated 3x3, source-like, unchanged)
b[old:] = y               (every target channel, gain 1, center tap)
v_y     = A_out · v_b[:old] + v_b[old:] + (ε − A_out W ε)
```

- Every target channel has its own identity input and output path. The ceiling
  from root cause 1 disappears, and the path has condition number 1.
- The noise complement is unchanged. As in the legacy layout, the new channels
  predict the clean correction, and the frozen residual supplies the noise not
  carried through the source channels.
- When the new IO is zero, the output equals the legacy pair exactly. Step-0
  generation and the noise endpoint are therefore unchanged (unit tested).
- Root cause 2 is not removed: the source channels' `W ε` is still a mix of the
  same `ε` the new channels see. This is redundancy, not missing information.
  The source channels still see unit white noise, so the pretrained body's
  input distribution is kept.

A complement-basis variant (`b[old:] = V⊥ᵀ y`, width 64) was rejected. Measured
on run 176's frozen calibration, the 16 directions it leaves to `A_out` have
DC-gain singular values from 4.94 down to 0.114 (condition number about 43).
Full passthrough avoids that. It also lets run 176 convert exactly: legacy
channel `k >= 16` becomes channel `16 + k`.

Code:
- `backbone_width()`, validation, and the pair layout are in
  [latent_adapter.py](../../backend/core/models/components/latent_adapter.py).
- Anima opts in through `initialize_image_swap(..., passthrough="full")`.
- Other architectures keep the legacy `max(old, new)` layout. Their loaders
  still hard-code that width, so extending the field to them needs per-arch work.
- Saved legacy declarations load and behave as before.

### D2. Learnable latent IO

Implemented for Anima full fine-tuning on full-passthrough swaps.

- `SplitLinear` ([split_linear.py](../../backend/core/models/components/split_linear.py))
  replaces `x_embedder.proj.1` and `final_layer.linear` when the optimizer
  groups are built.
  - The target-channel columns (`(c r m n)` order) and rows (`(p1 p2 t C)`
    order) become `weight_fresh`. The rest is `weight_kept`.
  - `state_dict` hooks present the merged `weight`, so checkpoint keys,
    shapes and loaders are unchanged.
- Optimizer groups:
  - `vae_swap_io`: the fresh parts, at the body LR × `vae_swap_io_lr_factor`
    (default 10).
  - `vae_swap_bridge`: both bridge convolutions, at the body LR ×
    `vae_swap_bridge_lr_factor` (default 1).
  - The factors are threaded through `param_defaults`, OpenAPI, the request
    model, the YAML builder, and the frontend config.
- Zero initialization is kept. The default factor of 10 is a starting value,
  not a measured optimum.

### D3. Calibration at training scale

Pending.

- Extend `vae_swap_calibration_resolution` beyond 128/256/512 (768/1024/1536)
  for every codec path that reads it. Keep half-side source encoding for 16x
  over 8x.
- Keep the global default at 256. Video calibration shares this key, and a
  larger default would raise its memory. Anima Qwen 2.1 runs should set the
  smallest training base resolution explicitly.
- Memory is encode-only and streamed into the existing Gram accumulators.
  A 1024px image gives 4096 observations for the 576-feature input regressor.

### Separate track: native Anima schedule

SushiUI's latent-area exponential shift gives 51.676 at 1472x896/40 steps with
the native codec, so every evaluation is above sigma 0.5. That is a native
generation defect independent of this redesign. Restore the audited shift 3
baseline as profile 1 of the sampling decision. For Qwen 2.1 swap runs (3.776),
schedule work is not expected to fix the jaggies.

## Migration of run 176

Pending. Legacy checkpoints keep loading and generating. The conversion is
function-preserving and needs no recalibration:

| Tensor | Legacy (width 64) | Full passthrough (width 80) |
|---|---|---|
| Bridge input rows | `0..15` calibrated, `16..63` index identity (trained) | `0..15` unchanged, `32..79` = legacy `16..63`, `16..31` = identity of target `0..15` |
| Bridge output columns | same split | same remapping |
| `x_embedder` columns, `final_layer` rows, global-adapter key/query/value columns | channel `k` | channel `k` for `k < 16`, `16 + k` for `k >= 16`; channels `16..31` zero |

The converted model computes the same function as the legacy checkpoint at
conversion time. It writes a new checkpoint with the updated declaration.
Optimizer moments for reshaped tensors start fresh, through the named
optimizer restore. The guard "Changing an aligned Anima VAE requires a fresh
base checkpoint" stays for any other codec change.

## Acceptance gates

| Gate | Check | Status |
|---|---|---|
| G1 Reachability | Every target channel has a gain-1 input and output path | Passed: unit tests |
| G2 Step 0 | Untrained full passthrough equals the legacy pair, including the noise endpoint | Passed: unit tests |
| G3 Layout | Split IO selects exactly the zero-initialized columns/rows; merged state dict is unchanged; save/load round trip is exact | Passed: Anima swap integration test |
| G4 Smoke | 3 BF16 optimizer steps on a small Anima | Passed: finite loss; fresh parts move at their own LR |
| G5 Real model | Real Anima/Qwen 2.1 swap: calibration, a few GPU steps, save, resume, generation | Pending: GPU occupied by run 176 |
| G6 Owner run | Frozen checkpoints at identical seed/prompt/NFE vs run 176 | Owner gate; a falling loss alone does not pass |

## Implementation order

1. D1 + D2 for Anima, with tests. Done.
2. D3 calibration scale.
3. Run 176 conversion tool.
4. G5 on GPU when the owner frees it.
5. Extend D1/D2 to other image architectures as needed.
