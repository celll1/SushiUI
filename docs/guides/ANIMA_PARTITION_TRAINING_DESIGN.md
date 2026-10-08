# Anima partition training and residual U-Net

## Scope

Implement complete-coverage image partition training, a trainable full-canvas
global adapter, and composition with residual U-Net and target-native VAE swap.
After verification, create run #174 from the current #173 settings, replacing
the base with `M:\model\anima\anima-base-v1.0.safetensors` and the topology with
Anima's inspected 28-block preset. Do not start the new run automatically.

The requested global adapter settings match runs #167 and #173: rank 64,
16 summary tokens, and partition activation above 4096 image patch tokens.
Dataset, optimizer, sample, exposure and dropout settings inherit #173 at creation.

## Current state and constraints

- The native model patchifies BCHW latents through a singleton frame axis.
  Image patch size is 2; the standard codec is 16 channels at spatial scale 8.
- Image self-attention uses 3D RoPE. Text conditioning enters separate
  cross-attention through the LLM adapter. Tiles retain complete text context.
- `AnimaArchHandler` does not yet declare partition support. Runner preflight
  rejects partition settings; rewiring validation also rejects their composition.
- U-Net execution and paired-spatial VAE swap already exist. With Qwen 2.1's
  scale 16 and two nested 2x regions, full canvases require 128px alignment.
- Run #173 is active. CPU/synthetic verification must not disturb its GPU work.

## Execution contract

1. Sample timestep, conditioning-dropout decision and full-canvas noise once per
   logical exposure. Partitioning never resamples any of these per tile.
2. Generate 2/4-region plans using the shared partition subsystem. Supervised
   cores cover the full patch grid exactly once. Halo inputs may overlap.
3. Expand halo inputs to cumulative per-axis U-Net alignment. Keep cores unchanged;
   reject a full canvas that violates the topology instead of dropping pixels.
4. Keep each tile's full-canvas RoPE coordinates. At a nested downsample use coarse
   integer coordinates, with the tile origin divided by the cumulative factor.
5. Compute the global summary from the same full noisy canvas. Map through the
   VAE bridge with the actual full exposure noise before native-channel processing.
   A zero-initialized output projection adds a patch-embedding residual without
   altering the initial native prediction.
6. Recompute the small global-adapter graph per tile. Backpropagate sequentially,
   weighting each core loss by its area / full canvas area. Preserve batch means,
   gradient accumulation, GradScaler and the shared conditioning graph lifetime.
7. Stitch detached core predictions only when debug output requires a full canvas.
   Normal training must not retain full tile activation graphs or predictions.

## Wiring and persistence

- Install the adapter before optimizer construction; include its parameters in
  Full FT and preserve the existing trainable-component learning-rate ownership.
- Export versioned adapter dimensions and weights with native model config,
  U-Net topology, VAE declaration and bridge weights. Restore before optimizer
  state loading and refuse incompatible dimensions or missing declared tensors.
- Both full-frame and partitioned generation consume a checkpoint-declared global
  adapter. Sample generation uses the active VAE/U-Net canvas alignment.
- Declare handler depth and partition support only after runtime integration.
  Update API/OpenAPI capability documentation and frontend static capability
  review together; do not broadly remove unrelated feature refusal gates.

## Verification and milestones

1. Global-coordinate RoPE foundation: compare tile embeddings with a slice of the
   full grid, including nonzero origins and nested downsample factors.
2. Native small-model forward/backward: core coverage, full-frame fallback,
   deterministic plans, halo, fixed/tokens-threshold modes, gradient accumulation,
   per-item conditioning dropout and independent sample exposures.
3. Combined small-model tests: global adapter + U-Net + Qwen-shaped target-native
   VAE bridge, finite nonzero backbone/adapter gradients and stitched debug output.
4. Artifact integration: save/load prediction equality, named optimizer restoration,
   resumed batch-plan continuation, full-frame and partitioned sampling.
5. Real-model smoke when GPU is available: general-rated images only for agent
   image verification; no hidden dataset mutation or competing GPU process.
6. Create #174 through the public API and compare persisted settings with #173;
   list only intended architecture-specific changes. Owner runs frontend builds.

After backend fixes, restart through `/api/v1/system/restart-backend` and verify
recovery before handing control back. Check the active process first; preserve
the owner's running training process and never start servers directly.

## Implementation status

Native global-coordinate RoPE, nested U-Net coordinates and portable global
adapter weights are implemented. The partition planner preserves exact core
coverage while expanding inputs to cumulative U-Net alignment. Sequential
velocity/reconstruction backward is implemented as an internal primitive; CPU
tests compare its losses and gradients with a full-canvas pointwise reference,
including a shared conditioning graph and accumulation scaling. Checkpoint
prefix scopes preserve activation-offload modes and restore them on exceptions.

Trainer dispatch now performs one logical noise draw and sequential region
backward with gradient accumulation scaling and shared deferred-metric flushing.
Debug output stitches detached supervised cores; convergence diagnostics receive
the recovered full first sample. Synthetic real-Anima tests include its LLM
adapter, nested U-Net, global adapter and 64-channel/16x target-native VAE bridge.

Partition execution currently refuses REPA, TREAD, BlockSkip, stochastic block
dropout and crop-decode auxiliary loss rather than silently dropping them. Run
#173 has these options disabled. Capability/API/UI gates, combined artifact and
resume verification and run #174 creation remain pending. The implementation is
not yet advertised as an executable training capability. CPU tests do not
establish real-model convergence or performance.
