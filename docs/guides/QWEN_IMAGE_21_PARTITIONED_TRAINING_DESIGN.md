# Qwen-Image 2.1 partitioned full-frame training design

Status: **proposed; not implemented**

Architecture key: `qwen_image_21`

This document defines a non-equivalent training acceleration for Qwen-Image
2.1. It partitions the target image-token block into rectangular regions,
processes every region within the same logical training item, and accumulates
their gradients before one optimizer step. Every source-image location remains
in the loss on every presentation. Only cross-partition target attention is
removed.

This is not crop training. A crop-training step observes and supervises only a
subset of the source image. A partitioned step covers the complete source image
and assembles its objective from all non-overlapping loss cores.

The feature is opt-in until its memory, throughput, convergence, and generation
quality gates pass. Existing full-frame training remains unchanged when it is
disabled.

## 1. Goals and non-goals

The design must:

* reduce peak activation memory without dropping any image region from a
  logical training item;
* reduce dense target self-attention work by replacing one long target block
  with several shorter independent target blocks;
* let partition count, orientation, and boundaries vary by image and epoch;
* support heterogeneous image dimensions and aspect ratios without treating a
  long side as proof that an image must be split;
* preserve the full-canvas positional coordinates of every target token;
* accumulate one area-normalized full-image loss and perform one optimizer
  update only after every partition has completed;
* compose with the Qwen ConvRot frozen-base path, floating-point adapter
  backward, Flash Attention, gradient checkpointing, MNT, and activation
  dispatch;
* reproduce the same partition plan after checkpoint resume; and
* make halo context optional rather than part of the base contract.

The first implementation does not promise equivalence to dense full-frame
training. Tokens in different partitions cannot attend to each other. It also
does not implement tiled generation, change the VAE or latent cache, or permit
an optimizer step per tile.

## 2. Numerical objective

Let a target latent grid contain `N = H * W` tokens. A partition plan defines
non-overlapping rectangular loss cores `C_i` such that:

```text
union(C_i) = full latent grid
intersection(C_i, C_j) = empty for i != j
sum_i |C_i| = N
```

Each core may be expanded by an optional halo to form transformer input region
`I_i`. The model predicts all tokens in `I_i`, but only predictions whose
global coordinates lie in `C_i` contribute to loss:

```text
L = sum_i (|C_i| / N) * MSE(prediction_i[C_i], target[C_i])
```

The clean latent, sampled timestep, and Gaussian noise belong to the logical
full image. SushiUI creates the full noisy latent once and slices every input
region from it. It does not sample independent timesteps or unrelated noise
fields per region.

Every per-region backward contributes to the same adapter gradients. Gradient
clipping, scaler update, optimizer step, scheduler step, EMA update, progress
counter, and external gradient-accumulation counter occur only after all
regions in the logical item have completed. External gradient-accumulation
scaling is applied once to the assembled objective, not once per region.

For distributed training, all but the last region of a logical item use the
equivalent of `no_sync`; otherwise partition count would multiply gradient
communication.

## 3. Expected resource behavior

For `K` balanced regions with `N/K` target tokens each, target self-attention
work changes from `N^2` to:

```text
K * (N / K)^2 = N^2 / K
```

Two balanced regions therefore retain about one half of the target-attention
work, while four retain about one quarter. Mildly uneven two-way cuts remain
close: a 60:40 split has a squared-token ratio of `0.60^2 + 0.40^2 = 0.52`.

This is not an end-to-end speedup bound. Across all regions, the model still
processes approximately `N` target tokens through projections and MLPs, and it
repeats the text prefix. Kernel launch overhead also rises. Conversely, shorter
sequences can require fewer checkpointed blocks and less recomputation, so the
measured benefit can exceed the attention-only share of the original profile.

Peak trainable activation residency follows the largest region rather than the
sum of all regions because each region is forwarded, backpropagated, and
released before the next begins. Frozen weights, the ConvRot BF16 grad-input
cache, optimizer state, and other persistent allocations do not shrink.

At a 1536-class square resolution, a 96 by 96 target grid contains 9,216
tokens. Balanced halves contain about 4,608 tokens and balanced quarters about
2,304. These are workload examples, not fixed spatial limits.

## 4. Feasibility is not a fixed width or fixed resolution

The planner must not split an image merely because its width or height exceeds
a configured pixel value. For example, an elongated image may have a long side
above a square preset while containing fewer latent tokens than that square.
A fixed side-length rule would split it without a memory reason.

The authoritative decision is a runtime feasibility estimate for the actual
unpartitioned workload. Its key includes at least:

* latent height and width, and therefore target-token count;
* valid prefix-token count;
* batch size and MNT multiplicity relevant to the live step;
* training dtype and frozen-base variant;
* attention backend;
* active checkpoint-block count and activation-offload policy; and
* the dispatcher's measured or conservative cold-start profile.

The pure transformer activation cost is governed mainly by sequence length,
not aspect ratio. Aspect ratio remains in the key so shape-specific kernels,
padding, and future measurements are not incorrectly collapsed. Two shapes
with equal target-token and prefix counts are expected to be close until a
measurement shows otherwise.

The planner asks the shared activation dispatcher for predicted peak bytes and
applies a safety margin. A target-token ceiling may exist as a deterministic
fallback before sufficient measurements exist, but it is not expressed as a
fixed image width or height and does not supersede a valid measured profile.

There are two independent reasons to partition:

1. **Required partitioning.** The unpartitioned workload is predicted not to
   fit. The planner must find a safe plan or refuse the item before CUDA OOM.
2. **Elective partitioning.** The unpartitioned workload fits, but a
   deterministic epoch-scoped draw elects to partition it for lower attention
   work and broader exposure to partitioned attention neighborhoods.

An elongated image that fits is therefore kept whole unless elective
partitioning selects it. A square image that fits follows the same rule; shape
does not grant or remove the elective probability.

## 5. Adaptive partition planner

### 5.1 Modes

The design admits these modes:

| Mode | Purpose |
|---|---|
| `off` | Existing dense full-frame training. |
| `fixed` | Prototype and diagnostic mode with a requested count such as 2 or 4. |
| `adaptive` | Production candidate driven by predicted peak residency and an optional elective policy. |

Fixed mode does not require fixed boundaries. It fixes only the requested
region count; orientation and cut positions may still vary by epoch.

### 5.2 Recursive rectangular planning

Adaptive planning begins with the complete latent rectangle. If it must or has
elected to partition, it recursively splits the region with the largest
predicted expanded-input workload until every region is feasible.

Candidate cuts:

* are aligned to the Qwen 2 by 2 image-slot contract;
* normally follow the current rectangle's longer side;
* may choose the other axis when it gives a better plan;
* stay within configurable balance bounds, initially proposed as 35:65 to
  65:35; and
* obey a minimum core side and a maximum region count.

Candidate plans are ranked lexicographically:

1. every halo-expanded input is predicted to fit;
2. the maximum predicted peak is minimized;
3. `sum(input_tokens_i^2)` is minimized;
4. extreme region aspect ratios are penalized; and
5. boundaries close to the same image's recent deterministic plans are
   penalized.

The last rule distributes attention boundaries over epochs. It must not make
resume depend on an uncheckpointed history cache; any history used by the cost
is derived from the run seed, item identity, and epoch or is saved in training
state.

If no plan satisfies the safety constraint before the maximum region count or
minimum side is reached, the item is refused with its predicted workload and
planner limits. The trainer must not silently drop regions or resize the image.

### 5.3 Plan variability

A plan is a deterministic function of:

```text
run seed
stable dataset-item identity
epoch
dataset repeat/occurrence index
latent shape
planner policy version
optional sigma band
```

Consequently, the same image may be full-frame in one epoch, split into two in
another, and split into three or four in another. Partition count is an output
of planning, not the main production control.

Elective partitioning of a feasible image uses a separate probability. The
draw is deterministic from the same identity tuple, so resume reproduces it.
When elected, the planner samples a stricter temporary workload target below
the full-image prediction and derives a safe plan from that target. The
configured feasibility ceiling remains an absolute upper bound; jitter never
raises it.

This policy gives full-frame examples to the adapter while ensuring that
images which require partitioning are not the only source of partitioned
attention. That avoids coupling partition behavior exclusively to the largest
or most square dataset buckets.

## 6. Optional halo

`halo = 0` is a valid hard partition. With a positive halo, each loss core is
expanded on every internal edge and clipped at the real canvas boundary:

```text
input_i = expand(core_i, halo) intersect full_canvas
loss_i  = core_i only
```

Input regions may overlap, but loss cores never do. Halo supplies local context
across an artificial boundary without restoring long-range cross-partition
attention. The feasibility estimator and planner use the expanded input size,
not the smaller core size; increasing halo may therefore increase region count.

The first prototype must support `halo = 0`. Positive halo is independently
measured rather than being required to validate the partition engine.

## 7. Global position contract

Each independent region must retain the coordinates it had in the complete
latent canvas. Re-centering every region as a small image would make different
parts of the source positionally indistinguishable and would not reproduce the
corresponding RoPE subproblem.

The Qwen forward contract therefore gains target position metadata separate
from rectangular input shape:

```text
full_latent_shape
input_box
loss_core_box
target_position_ids
```

Height and width RoPE indices are sliced from the complete canvas coordinate
grid. The target frame-axis position remains the same as the complete target
block. For token pairs retained in one region, this preserves the full
forward's relative target-target phase and the target-text positional phase.

The existing full-frame call remains the default and constructs its current
centered grid when explicit metadata is absent.

## 8. Sequential execution and integration points

The partition plan must exist before activation dispatch. Cropping only inside
the Qwen training op would cause dispatch to predict the full sequence while
the model receives a region, preventing correct checkpoint and offload
selection.

The logical step is:

```text
load/cache full latent and conditioning
sample logical timestep and full noise
build deterministic partition plan
for each region:
    dispatch using the halo-expanded region workload
    slice noisy latent and target
    forward with global position metadata
    compute area-weighted core loss
    backward into the shared adapter gradients
    release region activations
perform the logical optimizer/scheduler/scaler update
```

The implementation must integrate with:

* **Gradient checkpointing.** Select the checkpoint count from the actual
  region workload. A partitioned step must not retain the full-frame automatic
  count merely because the cached latent is full size.
* **Activation offload.** Remain a capacity fallback per region. Partitioning
  should first try the non-offloaded fast path.
* **ConvRot.** Keep frozen base forward in INT8 ConvRot and base grad-input plus
  adapter backward in the existing floating path.
* **Flash Attention.** Continue to use the validated Qwen attention backend for
  each shorter sequence.
* **Latent and conditioning caches.** Cache the full latent and conditioning;
  do not create persistent per-plan tile caches.
* **MNT.** Every MNT objective covers every loss core. A base implementation
  reuses one plan across the logical item's MNT evaluations; a later
  sigma-adaptive policy may select a deterministic plan per sigma band.
* **Batching.** Batch-1 executes regions directly. A later optimization may
  group equal-shaped regions from different logical items, but may not retain
  all region activations merely to improve utilization.

## 9. Optional sigma-adaptive planning

Sigma-adaptive planning is recorded as an extension, not a first-prototype
requirement. Qwen flow training constructs:

```text
x_sigma = (1 - sigma) * x0 + sigma * noise
```

High sigma therefore corresponds to a noise-dominated state where global
layout and text allocation are expected to matter more; low sigma is closer to
the clean image and more dominated by local refinement. A provisional policy
can allow a larger region workload at high sigma and request smaller regions
at low sigma.

This never drops image coverage. It changes only partition granularity. At
least one non-sigma-adaptive mode must remain available so quality and speed
effects can be attributed independently. Thresholds and schedules are not
defaults until full-versus-partition prediction divergence has been measured
across sigma.

## 10. Optional position sidecar

Exact global RoPE coordinates are mandatory. A trainable position sidecar is
optional and is added only if measurements show that a region needs explicit
knowledge of the full canvas or its box beyond the retained RoPE relations.

A candidate sidecar encodes normalized per-token coordinates plus normalized
`input_box` and `full_latent_shape` metadata with Fourier features and a small
zero-initialized projection into target hidden width. It is stored in a
dedicated namespace of the adapter artifact.

The sidecar must be structurally gated:

```text
position_delta = is_partitioned * sidecar(position_metadata)
```

It is exactly zero for ordinary full-frame generation. This makes the normal
generation path independent of the sidecar computation, although the LoRA
weights themselves have still been optimized under the mixed training
distribution. Generation-quality gates must therefore cover adapters trained
with and without it. Missing sidecar tensors are an error only for an artifact
whose metadata declares them.

The first prototype does not need this module. Global position IDs, variable
boundaries, and mixed full/partitioned exposure are evaluated first.

## 11. Proposed configuration surface

Names remain provisional until implementation. Defaults will live only in
`backend/api/param_defaults.py`, and any API fields will be added OpenAPI-first.

```text
qwen_partition_training_enabled
qwen_partition_mode                 # off | fixed | adaptive
qwen_partition_fixed_count          # prototype: typically 2 or 4
qwen_partition_memory_fraction      # safety ceiling for predicted peak
qwen_partition_token_fallback       # cold-start fallback, not a side length
qwen_partition_elective_probability # feasible images may still partition
qwen_partition_min_core_side
qwen_partition_max_regions
qwen_partition_split_ratio_min
qwen_partition_split_ratio_max
qwen_partition_halo_tokens
qwen_partition_sigma_adaptive
qwen_partition_position_sidecar
qwen_partition_seed
```

The UI should expose a simple off/fixed/adaptive selector, fixed count for the
prototype, elective probability, halo, and an advanced memory safety control.
Planner internals such as measured profile coefficients are diagnostics, not
user-authored tuning fields.

The checkpoint identity records the complete policy, policy version, seed,
and any position-sidecar declaration. Resume refuses a changed policy unless a
future explicit migration contract defines how the change is applied.

## 12. Diagnostics

Logical-step metrics must distinguish image progress from region substeps:

```text
partition_reason                    # none | required | elective | fixed
partition_count
full_target_tokens
largest_input_tokens
total_input_tokens                  # includes halo overlap
sum_input_tokens_squared
predicted_peak_bytes
actual_peak_bytes
checkpoint_blocks_per_region
region_forward_ms
region_backward_ms
logical_image_ms
```

Progress, ETA, samples-per-second, and scheduler state count logical images,
not region forwards. A diagnostic response may additionally report regions per
second, but it must not label that number as samples per second.

## 13. Correctness and acceptance gates

### 13.1 CPU and deterministic gates

* Every generated plan covers the complete latent grid exactly once in its
  loss cores, for square, portrait, landscape, prime-like, and minimum shapes.
* Halo expansion stays inside the canvas and never changes core ownership.
* Every input shape satisfies Qwen image-slot alignment.
* Fixed 2- and 4-region plans work with even and uneven boundaries.
* Adaptive plans do not partition a feasible elongated image merely because a
  side exceeds a square preset.
* Elective draws make a feasible image alternate reproducibly between full and
  partitioned presentations across epochs.
* Save/resume reconstructs the same reason, count, boxes, order, and halo.
* Full-frame mode remains numerically unchanged when metadata is absent.

### 13.2 Gradient oracle

Build a small reference forward whose target attention mask is block diagonal
according to one partition plan while its prefix remains shared. Because the
block-causal prefix cannot attend to the later target, this represents the same
mathematical approximation without sequential activation release.

With stochastic layers disabled, compare the reference against sequential
per-region execution:

* stitched predictions on every loss core;
* area-weighted total loss;
* every adapter gradient; and
* optimizer result after one logical step.

This gate tests the partition engine itself. It does not compare partitioned
attention with dense full-frame attention, which is intentionally different.

### 13.3 GPU performance gates

Measure full, fixed-2, fixed-4, adaptive hard-partition, and adaptive halo
variants on at least one square, portrait, and landscape bucket. Record:

* peak allocated and reserved VRAM;
* forward, backward, and optimizer time;
* checkpoint recomputation count;
* dispatcher prediction error;
* attention backend selected; and
* ConvRot base/cache residency.

Required partitioning passes when it keeps the workload below the configured
safety ceiling without activation offload. Elective partitioning remains
available only if it produces a measured logical-image throughput benefit for
the relevant workload; lower peak VRAM alone is not described as a speedup.

### 13.4 Training and generation quality gates

Compare dense and partitioned adapters with matched source-image presentations,
optimizer steps, seeds where meaningful, and total supervised pixels. Evaluate:

* convergence by logical image and wall time;
* long-range object count, pose, and spatial relations;
* typography spanning a potential boundary;
* repeated structures and bilateral symmetry;
* boundary-position error heatmaps over changing epoch plans;
* full-frame generation with the resulting adapter; and
* checkpoint/resume continuity.

Quality runs must include datasets where some buckets require partitioning and
others fit whole, plus an elective policy that sometimes partitions the latter.
This distinguishes the intended mixed training distribution from a policy
coupled only to large images.

## 14. Known risks and staged rollout

The approximation removes all direct target-target correspondence across a
partition. It can weaken composition, object identity across distant regions,
large text, and structures crossing a boundary. Randomized boundaries and
optional halo distribute and soften the error but do not restore global
attention.

Repeated prefix work and multiple backward calls can offset attention savings
on smaller images. Variable region counts also make logical-step duration
variable. Persistent model and ConvRot cache memory set a floor below which
partitioning cannot reduce residency.

Training-time patch methods provide useful evidence that full-size exposure,
variable patch sizes, and position information matter, but they do not
establish equivalence for this design. Patch Diffusion reports poor quality
when full-size exposure is removed and uses randomized sizes plus coordinates;
its crop objective differs from the complete-coverage objective here:

* Wang et al., [Patch Diffusion: Faster and More Data-Efficient Training of
  Diffusion Models](https://arxiv.org/abs/2304.12526), 2023.

The staged implementation order is:

1. fixed 2/4 hard partition, exact global positions, and gradient oracle;
2. adaptive measured-feasibility planner and deterministic epoch variation;
3. elective partitioning of feasible images;
4. optional halo;
5. sigma-adaptive granularity, after prediction-divergence measurement; and
6. optional position sidecar, only if global-position metadata is insufficient.

No later stage is implied by the acceptance of an earlier one.
