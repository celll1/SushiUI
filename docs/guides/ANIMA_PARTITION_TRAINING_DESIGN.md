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
#173 has these options disabled. The handler, config converter, runner preflight,
rewiring gate, frontend controls and OpenAPI now expose the implementation.
Combined full-parameter artifacts preserve every model tensor exactly. Restored
full-frame and region predictions agree within FP32 rounding tolerance (1e-6),
and named optimizer moments restore even with reversed parameter traversal.
Trainable Anima global summaries currently require Full FT; LoRA cannot declare
new summary weights because its artifact writer does not yet export them.
CPU tests do not establish real-model convergence or performance.

Run #174 was created through the public API as
`Anima_Full_UNet_2stage_deep4_Qwen21VAE_Global`, pending and not started. A fresh
#173 settings read supplied 20 epochs and dataset #38. Saved-parameter comparison
found only base model, native U-Net spec, run identity/name, cleared resume path
and disabled Ming quantized-base dequantization changed. The Qwen 2.1 codec,
global rank 64 / 16 tokens and checkpointed prefix counts of 24 were preserved.
The actual generated YAML passes native Anima checkpoint/partition preflight
without CUDA initialization. Real-weight synthetic GPU checks are recorded below;
dataset calibration and generation-quality acceptance remain separate. The owner
subsequently stopped #173 at step 19854 after backend
restart lost its child-process tracking. The restart endpoint now refuses live
and starting registered training children with HTTP 409, preventing that loss
through the API. Run #173 must not be resumed implicitly.

## DRAM investigation

A read-only snapshot on 2026-10-08 measured 57.31 GiB working set in the active
#173 training process, versus 4.50 GiB in the backend process. This is process
resident memory, not a tensor ownership census or proof of a leak. The current
run uses asynchronous activation offload with a 2048 MiB pinned arena budget.
That arena alone does not explain the training process's footprint.

Prioritize an ownership census at encoding, backward, optimizer-step and save
boundaries: CPU weights and mirrors, optimizer state, activation arena, text and
latent swap buffers, and checkpoint staging. Account unique storage rather than
tensor references. Inspect high-water buffers and in-flight transfers before
reducing prefetch depth; a smaller buffer may increase encoder swap frequency.
Successful shared VAE calibration deletes its retained source-VAE owner; the
failure path retains it for retry, so it must be distinguished from a leak.
Anima's frozen FP8 option copies its base during quantization, but full DiT FT
does not take that path. Avoid adding a second inference model or full-canvas
prediction graph while the owner's run is active.

## Real-weight GPU checks and topology performance

On 2026-10-08, the official Anima BF16 weights passed synthetic Full FT GPU
forward/backward and AdamW8bit steps with nested U-Net, a 64-channel/16x bridge
and global summaries. Two-region and four-region sequential backward produced
finite gradients. The initially zero output projection received gradients on
the first update; all global-summary parameters received nonzero gradients on
the next update. No dataset images were read and #174 remains pending.

The separate ablation below isolates full-frame DiT execution on RTX 6000 Ada:
batch 1, BF16, AdamW8bit, checkpoint prefix 24, identical global-summary settings,
one warmup and three measured iterations per resolution/condition. Times are
medians. Forward includes the FP32 MSE loss; backward includes checkpoint
recomputation. Iteration includes optimizer and synchronization. VRAM is peak
PyTorch allocated memory after optimizer-state warmup, not Task Manager usage.

| Square input | Topology | Image tokens | Forward ms | Backward ms | Iteration ms | Peak VRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | Native | 1024 | 87.0 | 191.8 | 349.1 | 11.881 |
| 512 | U-Net | 1024 | 93.8 | 181.8 | 350.3 | 11.885 |
| 512 | Qwen 2.1 VAE grid | 256 | 88.4 | 174.6 | 338.0 | 11.901 |
| 512 | Both | 256 | 93.1 | 180.0 | 344.0 | 11.889 |
| 1024 | Native | 4096 | 249.9 | 949.9 | 1279.5 | 12.243 |
| 1024 | U-Net | 4096 | 120.3 | 357.1 | 557.1 | 12.293 |
| 1024 | Qwen 2.1 VAE grid | 1024 | 90.5 | 179.1 | 342.6 | 11.901 |
| 1024 | Both | 1024 | 93.0 | 174.8 | 345.4 | 11.896 |

At 1024, iteration speedups are 2.30x for U-Net, 3.73x for the swapped grid,
and 3.70x for both. U-Net uses the two-stage, deepest-four-block default.
Its region-weighted image-token count is 43.75% of native (8 full-resolution,
16 half-resolution and 4 quarter-resolution blocks); attention's quadratic
term is 32.2%, before text, shared projections and other operations. The swapped
grid has one quarter as many image tokens: 16ch/8x becomes 64ch/16x, retaining
the same latent element count per pixel. Both give 10.94% of native region-weighted
image tokens and 2.01% of its image-only quadratic term. These are operation
proxies, not whole-model FLOP counts or measured throughput predictions.

U-Net retains all weights and adds skip/down/up work, so parameter count is
unchanged and VRAM reduction is not guaranteed. These tests measured almost
constant VRAM because weights, gradients and optimizer dominate with
checkpointing enabled. At small grids, fixed overhead and approximately 70–75 ms
optimizer steps dominate; combining U-Net with swap did not improve iteration
time over swap alone at either tested size. Larger grids require separate
measurement; do not extrapolate the quadratic proxy to wall-clock speed.

Native plus global summaries has 2,091,213,696 parameters. The synthetic swapped
configuration has 2,092,110,848 (+897,152, about 0.043%), including its changed
input/output projections, spatial bridge and global-summary input dimensions.
Process RSS was approximately 2.4–2.6 GiB in these isolated probes; it does not
represent the full dataset worker/swap-buffer footprint observed in #173.

The probes use synthetic latent/text inputs (five source and three target text
tokens), center-identity spatial coefficients, and no real VAE calibration.
They exclude VAE encode/decode, caption encoding, disk I/O, spatial partitioning
in the ablation and the calibrated Gaussian noise complement. Therefore these
are DiT grid/bridge measurements, not complete #174 training throughput or a
generation-quality test. Full-run captions and calibrated bridge/noise handling
can change the timings. The partition functional probe's cold first optimizer
step must not be used to compare two versus four regions.

Reproducible tools: `backend/tools/verify_anima_partition_gpu.py` and
`backend/tools/benchmark_anima_topology_gpu.py`; both require explicit model,
saved-parameter JSON and output paths. Raw local measurements are retained in
`local/verification/anima174_gpu_partition_smoke.json` and
`local/verification/anima174_topology_benchmark.json`. The processes exited and
GPU memory returned to approximately 1.8 GiB baseline; no training was started.

### Batch-size ablation

The same four conditions were measured at 1024x1024 with actual batches 1, 2
and 4 on 2026-10-08. Batch 1 was remeasured in the same session. All earlier
exclusions apply; checkpoint prefix remains 24, CPU/async activation offload is
disabled, and each condition has one warmup plus three measured samples.
Rows show medians; per-image time is iteration divided by actual batch size.

| Topology | Batch | Forward ms | Backward ms | Iteration ms | ms/image | Peak VRAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 1 | 247.4 | 945.4 | 1267.9 | 1267.9 | 12.231 |
| Native | 2 | 533.6 | 1996.3 | 2607.0 | 1303.5 | 14.502 |
| Native | 4 | 1206.6 | 4355.2 | 5637.5 | 1409.4 | 20.968 |
| U-Net | 1 | 109.0 | 365.9 | 553.3 | 553.3 | 12.275 |
| U-Net | 2 | 218.4 | 761.5 | 1057.8 | 528.9 | 14.055 |
| U-Net | 4 | 486.7 | 1654.3 | 2220.5 | 555.1 | 20.068 |
| Swap | 1 | 82.0 | 172.6 | 334.6 | 334.6 | 11.889 |
| Swap | 2 | 102.6 | 360.4 | 541.1 | 270.6 | 11.868 |
| Swap | 4 | 232.5 | 800.7 | 1114.2 | 278.6 | 12.261 |
| Both | 1 | 83.3 | 158.8 | 310.8 | 310.8 | 11.895 |
| Both | 2 | 87.3 | 173.3 | 338.7 | 169.3 | 11.907 |
| Both | 4 | 97.1 | 311.4 | 491.5 | 122.9 | 12.306 |

Combined versus swap alone is 1.08x at batch 1, 1.60x at batch 2 and 2.27x at
batch 4. The small batch-1 difference varies across sessions; earlier results
showed no combined advantage there. At batch 4, combining is 2.53x faster per
image than its own batch-1 baseline, with approximately 0.41 GiB extra allocated
memory. Native and U-Net alone do not show comparable batching gains.

This supports the hypothesis that small low-resolution block operations gain
utilization from batching while optimizer costs are amortized. It does not
identify individual kernel bottlenecks without a profiler. Non-monotonic small
VRAM differences can reflect allocator lifetimes and kernel workspace choices;
they are not evidence that doubling the batch reduces tensor storage. No OOM
or nonfinite loss occurred. Throughput is not a convergence-equivalence claim:
larger batches change gradient averaging and optimizer update frequency.

Raw results: `local/verification/anima174_batch_benchmark.json`. Reproduce using
the benchmark tool with `--samples 3 --batches 1 2 4 --resolutions 1024` plus the
explicit model, params and output arguments. The tool now requires batch sizes
and resolutions explicitly. #174 stays pending; GPU memory returned to 1.8 GiB.

### Frozen encoder reuse (2026-10-08)

Concept sample multipliers expose an item repeatedly inside nearby batch
windows. Previously the shared swap producer encoded every occurrence and
discarded its result at the next refill. The latent consumer remains keyed by
image path, so 603 encoding calls could leave only 187 entries. Ming's native
producer similarly discarded results after consuming each batch position.

The implementation plan and completed sequence are:

1. Introduce one trainer-owned CPU LRU with a tensor-payload budget and entry
   limit, without adding a disk cache or changing REST/config parameters.
2. Reuse frozen, eval-mode VAE posterior parameters for identical processed
   pixels; keep the native sampling and normalization operations outside the
   cache. VAE swaps wrap the raw codec inside the normalized facade.
3. Reuse frozen TE outputs by resolved caption, lyrics, null-condition flag,
   producer identity, precision and run configuration. Preserve occurrence
   caption processing; shuffled/dropped captions are different keys.
4. Deduplicate Anima's true CPU-prefetch TE batch misses, and wire Ming's
   native swap and CPU-prefetch conditioning producers by actual processed
   token/vision inputs and dimensions. Single/batched TE payload formats use
   separate namespaces.
5. Check RNG, invalidation, cancellation, ownership, capacity and existing
   concept-plan/resume paths with synthetic inputs. Restore original module
   modes after Anima, SD/SDXL and Lens sample generation, including mixed
   trainable/frozen children.

`backend/core/training/encoder_reuse.py` owns the policy. Internal defaults live
in `backend/api/param_defaults.py`: at most 512 MiB of stored tensor payload and
4096 entries per trainer. Metadata, transient copies and existing swap/prefetch
buffers are additional memory. Oversized entries are not retained; eviction
only affects performance. Logs expose cumulative hits, misses, evictions and
retained tensor bytes at swap refills.

VAE reuse currently recognizes the standard diffusers diagonal Gaussian
posterior only; other return types bypass it. Exact pixels include the actual
crop, flip and channel preprocessing. A batch-shaped VAE result is cached as a
whole, so different regroupings of its images miss. TE reuse in the common path
is enabled for the reviewed Anima, SD1.5, SDXL, Qwen-Image 2.1, Z-Image, FLUX.2,
Lens and Krea2 producers; other architectures bypass pending contract review.
Trainable encoders, training-mode encoders, reference-conditioned common TE
calls and producer changes bypass or invalidate reuse. Ming separately keys
its actual prepared multimodal inputs. Normal parameter updates/replacements,
configuration and small normalization buffers participate in identity;
large buffer moves conservatively cause a miss.

Caches are process-local and absent from checkpoints. Restart/resume rebuilds
them without modifying the saved occurrence plan or adding RNG draws. Each
posterior sample still executes at the original call site; the existing
per-image-path latent-buffer consumer semantics are unchanged. Caption/null
draws and training timesteps remain occurrence-specific. Anima true-batch TE
deduplication can change padding/batch dimensions and floating-point rounding;
it does not promise bitwise equality across different producer batch shapes.

The synthetic 603-exposure/187-input regression executes 187 VAE forwards with
416 hits. Posterior samples and generator state match uncached calls exactly
for both raw and normalized toy codecs. This is an operation-count check,
not a measured #174 wall-time speedup. Input loading, preprocessing, input
hashing and component residency transfers still execute; all-hit windows do
not yet skip GPU residency transitions. No owner images or sidecars are read
for these tests, and no training run is started by the verification.
