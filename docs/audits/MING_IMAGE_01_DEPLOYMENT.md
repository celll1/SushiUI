# Ming official model deployment and live UI verification

## U-Net Full training follow-up (2026-10-04)

Run 172 copies run 171's datasets and learning settings, with Full parameter
training, explicit INT8 base materialization and the non-LoRA dequant forward.
Its fresh `latest` selection exposed a false resume topology guard: only an
actually resolved full checkpoint now enters the immutable-resume path
(`0020369c`; 61 CPU rewiring/resume cases passed).

The first real GPU backward produced finite gradient norm 1.701255, then AdamW
failed allocating its first `exp_avg_sq` state (45.27 GiB live, 1.69 GiB reserved
but unused). The emergency native model, training state and optimizer were saved
and quarantined under the run's `failed_optimizer_init_20261004` directory;
that attempted update is not a successful training acceptance result.
Ming Full AdamW now disables parameter-sized foreach temporaries and releases
unused CUDA activation pools before the optimizer seam. Learning rates, betas,
epsilon, weight decay, clipping and component ownership remain configured.
The five targeted CPU optimizer construction/update checks passed, including
exact FP32/BF16 scalar-reference updates. Modules compiled and imported with
CUDA forbidden. The broader optimizer-option/resume suite reported 67 passes
and seven existing fixture/contract failures: six lack current fused-gradient
probe attributes and one assumes every stochastic-rounding config read is a
literal-default call. The production code triggering those failures predates
this change. Subsequent retries started from the original base, avoiding the
partially initialized emergency optimizer state.

The cache-release retry also exhausted VRAM during first moment allocation
(gradient norm 1.743109); its GPU-to-CPU emergency move failed, so only its
training-state JSON was saved and quarantined. Allocate neutral AdamW moments
before the batch work starts, preserving every restored state and leaving step
counters at zero. The Ming activation dispatcher subtracts missing CUDA
gradient storage from live headroom, excluding existing/CPU/frozen gradients
and fused-backward ownership. The six targeted memory checks and materialized
INT8 Full pending-gradient resume oracle passed.

GPU verdict: run 172 failed before any successful optimizer update. Neutral
moments allocated successfully (DiT plus moments: 34.66 GiB); VAE and all 256
caption encodes passed, with conditioner peak 42.14 GiB. At the first training
batch (latent 128x184), subtracting missing full gradients made live headroom
-2.4 GiB. Same-batch offload at 256 KiB and the strongest retry still failed.
BF16 weights, full gradients and two AdamW moments require about 46 GiB before
activation/workspace and other CUDA/desktop residency. U-Net reduces token work,
not this persistent parameter/gradient/optimizer storage. This same-AdamW preset
has not passed Full GPU acceptance on the current 48 GiB environment.

The step-0 native bundle, state and neutral optimizer saved successfully;
CPU inspection confirmed the requested topology and all 521 AdamW counters at
zero, with no pending gradients (package 32,530,174,796 bytes; optimizer
24,581,787,406 bytes). The failed batch's state is not a completed-batch seam;
all three entries are quarantined under `failed_activation_budget_20261004`
and excluded from automatic `latest` lookup. The trainer exited and GPU use
returned to approximately 1 GiB. Initial failed
weight/optimizer duplicates were removed after retaining diagnostic JSON/logs;
the newest step-0 bundle remains, with approximately 97.6 GiB free on M:.
Changing to the standard Full AdamW8bit preset changes optimizer semantics;
maintaining native AdamW would require additional host-state streaming work.
The owner was asked to choose that next verification policy. A read-back draft
in ignored `local/deployment/ming_run172_adamw8bit_candidate.json` changes only
`optimizer` from `adamw` to `adamw8bit`; it has not been applied. No learning-
config change, frontend build or five-epoch completion is claimed.

Date: 2026-10-04. Owner explicitly authorized main GPU/API use and API restart.

## Generation Color Flatten wiring (2026-10-05)

Nonzero `color_flatten_strength` previously hit the Ming pending-options guard
despite being exposed by the shared frontend slider. It now passes through the
backend, generator and sampler to a native image-processor subclass. Shared
`flatten_chroma` runs on normalized RGB before PIL quantization; alpha remains
unchanged for every Design/Layer frame. Each sampling request resets the
strength from its explicit value or central default, including reuse after an
enabled request. Latent output bypasses image postprocessing.

CPU regressions passed 23 cases: shared smoothing oracle, alpha equality,
Design/Layer frame handling, reset-to-off and invalid strength refusal, plus
existing native sampling checks. Changed modules compiled/imported without
CUDA. Static frontend review confirms slider range 0..100, API transport and
queue/inheritance already carry the existing field. No API schema or default
changed; owner frontend build/type-check was not run. Full-model GPU generation
with the fix has not been measured in this follow-up.

## CPU pixel lookahead implementation (2026-10-05)

Swap-on-the-fly latent refill now prepares the next original batch's pixels in
one CPU worker while the caller prepares prompts and encodes the current batch.
The VAE and prompt processor remain single-owner; original epoch/batch crop
coordinates, Layer/reference transforms, source hashes, shape-transition CUDA
cache release and existing progress callback phases are retained. This adds no
disk tensor cache and changes neither VAE precision nor training batch size.

Admission uses a 512 MiB estimated combined working-set budget for current and
next batches, including source decode scratch, normalized outputs and compressed
file bytes. This is a conservative admission estimate, not a codec allocator
hard limit. Oversized pairs and file-stat failures use serial preparation.
Cancellation is checked while waiting; consumer failures close the generator,
cancel queued work and join any active pixel load before encoder residency exits.

Changed modules compiled and imported without initializing CUDA. Six new CPU
checks plus ten existing buffer checks passed (16 total), covering exact native
Design/Layer latent and reference equality, original indices, overlap, oversized
fallback, worker errors and cleanup. Static review confirms existing progress,
stop and error paths remain wired; no frontend/API contract changes are needed.

A warmed FP32/B1 VAE-only comparison used six actual JPEG/PNG/WebP Design sources
at fixed 1024x1024, eight Torch CPU threads and serial/prefetch/prefetch/serial
order. Mean elapsed time fell from 1.617 s to 1.387 s (14.3% less time; 16.6%
more images/sec). All four passes produced exactly equal latents. Peak allocated
CUDA memory across the comparison was 5.41 GiB. This small comparison excludes
prompt processing, mixed epoch buckets and full training/optimizer residency;
it does not establish a whole-training step/sec improvement. Reproduction and
results: ignored `local/deployment/ming_pixel_prefetch_gpu_benchmark.{py,json}`.

The code applies to newly started training children; no run was restarted or
its configuration changed. Run 172's separate Full AdamW memory limit remains
unresolved. Same-shape grouping, larger VAE batches and encoder BF16 remain
unimplemented candidates requiring separate memory/numerical acceptance.

## Latent encode performance investigation (2026-10-04)

This follow-up is source inspection, read-only API comparison and a bounded CPU
measurement; no training restart, GPU benchmark or precision change was made.
Run 167 (Qwen) reports BF16 VAE; runs 170/172 (Ming) report FP32. All three use
batch 1 and `swap_onthefly` with interval 256. Ming's loader explicitly loads
the native VAE in FP32. This is a concrete precision difference, not a measured
explanation of the whole speed gap. Earlier generation failures justify keeping
the working FP32 path; encoder-only lower precision has not been accepted.

`MingBatchBuffer._fill_latents` serializes image read/hash, paired pixel preparation,
prompt preparation, VAE encode and blocking CPU latent copies. It releases CUDA
allocator cache whenever the upcoming logical batch's shape changes. The shared
Qwen/SDXL image swap refill also encodes items serially and copies latents to CPU;
it does not establish that those models already have batched VAE/pixel prefetch.
Ming's VAE stays resident for the grouped phase, so it is not reloaded per image.
The native VAE uses posterior `mode()`, while the Qwen bridge uses `sample()`;
do not substitute Qwen's distribution/normalization to chase speed.

A one-pass CPU sample of 12 existing JPEG/PNG/WebP assets, four per extension,
center-cropped/resized to 1024x1024 RGBA, with eight Torch CPU threads, measured:

| Preparation stage | Mean per image |
|---|---:|
| File read and SHA-256 verification | 35.6 ms |
| RGBA decode, center crop and resize | 66.0 ms |
| Tensor normalization and finite/range checks | 8.1 ms |

This approximately 110 ms includes neither the native prompt processor nor VAE,
and is not the actual mixed-bucket epoch or a controlled cold/warm filesystem
benchmark. The ignored reproducible script/results are
`local/deployment/ming_latent_cpu_prepare_benchmark.{py,json}`. No disk tensor
cache was created. Source-content verification remains necessary.

Implementation priority:

1. Prefetch paired pixels on CPU in a small, byte-bounded queue while one owner
   encodes the preceding item on GPU. Preserve original epoch/batch-position crop
   seeds, source SHA checks, paired Layer/reference transforms, cancellation and
   worker exception propagation. Limit decoded-source memory as well as output
   tensors; huge source images make a count-only queue unsafe. Prompt processor
   ownership must remain serialized unless separately proven thread-safe.
2. Group identical VAE geometry within a bounded refill window, scattering
   outputs back to their original logical indices. This reduces shape-change
   cache flushes without changing concept/batch order. Start with Design; native
   Layer frames remain independent images. The VAE feature cache is mutable, so
   parallel calls into the same GPU VAE are invalid. Never globally remove the
   shape-transition cache release that fixed real VRAM retention.
3. Measure same-geometry VAE batches of two, separate from the optimizer's
   training batch size, only when live VRAM allows. Retain batch-one fallback;
   Full optimizer residency changes available encoding memory. The native VAE
   supports the B axis but optional slicing would defeat this batching. Validate
   latent differences before accepting a different convolution execution shape.
4. Add bounded pinned staging and event-owned H2D/D2H overlap if profiling shows
   transfer stalls. `non_blocking=True` alone is insufficient; a CPU buffer must
   not be consumed/reused until its transfer event completes.
5. Evaluate encoder-only BF16/autocast against FP32 with finite checks, latent
   error, RGBA/alpha reconstruction and training-loss checks. Keep decoder FP32
   and make any accepted precision choice explicit in API/defaults/cache identity.
   No precision change or numerical speedup is assumed from Qwen's success.

At the next authorized GPU verification, measure source/hash, pixel/prompt CPU
time, H2D, VAE, D2H, shape-flush counts and encoder residency transfer separately.
Use a short warmed fixed-shape sample and one mixed-bucket refill, with CUDA events
for GPU work and wall time for the complete phase. Compare identical inputs,
weights, precision, crop identities and optimizer residency; report latency,
images/sec, peak allocated/reserved VRAM, cancellation and latent agreement.
Reducing the 256-item interval only shortens the initial wait and increases
refill/swap frequency; it is not evidence of improved total encoding throughput.

## Deployed packages

| Profile | Local model directory | HF revision | Converted package bytes |
|---|---|---|---:|
| Design (`generation_edit`) | `M:/model/ming_image_01/design_int8_convrot` | `208087ada1486931692c1896f38d4cd16ff3df82` | 26,739,650,742 before license/provenance additions |
| Layer (`layer_decompose`) | `M:/model/ming_image_01/layer_int8_convrot` | `9fabca8b62a67f1f53a957d46389c00451c11e52` | 27,017,075,174 before license/provenance additions |

Official originals are in the independent worktree's ignored
`local/models/ming-image-0.1-sources/{Design,Layer}` directories. Earlier work
had not created these full models; tiny CPU fixtures were never release models.
Both packages now contain native six-component manifests, quantization reports,
MIT notices, and pinned HF repository/revision provenance. The main model-search
configuration includes `M:/model/ming_image_01`, preserving the other search roots.
Both models are discovered through `/api/v1/models` and selected in Generate.

Conversion ran on CPU with CUDA initialization forbidden. Each profile converted
15,261 Linear layers: 238 DiT, 14,824 MLLM, 196 connector, and 3 projections.
Embeddings, tied outputs, queries, normalization and other protected tensors stay
floating. Eight official weight files with identical SHA256, totaling 34.26 GB,
are shared by hardlinks between original downloads on the same volume.

## Real model findings and fixes

The first live Design load inherited generic FP16 from public model dispatch;
its generated RGBA PNG contained zero in every channel. Public Ming loads now
use BF16, and the native RGBA VAE loads in FP32. Conditioning, scheduler latents
and decoded pixels reject nonfinite values before a black PNG can be saved.
The unchanged source package was used for subsequent successful inference.

Live Layer review found the API selected auxiliary frame zero as `image` while
preview/training semantics expect Layer 1. Primary decoded output, normal gallery
responses and skip-gallery responses now select Layer 1; ordered `images`,
`groups` and canvas ownership remain unchanged. OpenAPI records this contract.
The component graphic explicitly labels floating parameter objects: packed
weights are buffers, so that census is not the full architectural parameter count.
Serialized storage is displayed separately.

## Verification scope

- Main integration: `90869f1d`; precision correction: `958e9dff`; Layer primary,
  notices and accounting wording: `cb1277e7`.
- Main API restart used its supported endpoint; no server was launched directly.
- Main integration CPU checks: 59 cases. Precision/manager/conversion checks:
  37 cases. Updated conversion-notice checks: 9 cases. Layer-primary/persistence/
  manager checks: 42 cases. These sets overlap and must not be added as unique cases.
- Changed backend files compile and import with CUDA forbidden; OpenAPI parses.
- Frontend contracts and primary-output transfer consumers were reviewed
  statically. No agent frontend build/type-check was run; owner compiler and
  full interactive acceptance remain separate.
- Design: UI selection/load and Generate, 1024x1024, 12 steps, CFG 1, seed 123456,
  RGBA PNG persisted and displayed. Both a plain caption and native-schema JSON
  were exercised after precision correction.
- Layer: UI selection/load, Design result sent to img2img, 512x512, 12 native
  steps, CFG 2, seed 123456, two requested layers. Three RGBA PNGs persisted and
  displayed with `auxiliary_canvas`, `layer_1`, `layer_2` roles; their alpha ranges
  were 234-255, 137-255, 242-255 in this sample.

These are functional inference checks. Design samples contain unintended
lettering and do not satisfy every requested object; the Layer sample separates
subject and background but does not establish clean transparency or exact
recomposition. Numerical parity against original full-precision inference,
quantization quality acceptance, official-model GPU training/resume and high-
resolution Layer memory limits have not been established by these inference runs.

## Final primary-output and preview verification

The final Layer API run completed without error and reported `_1.png` as
`last_result.filename` (gallery ID 11619); its output group retained canvas and
both layers. The follow-up Preview selection change is `4304b481`, integrated
into main. Existing results verified Layer 2 / Layer 1 switching in the browser,
with main image URLs and active indicators changing together. This UI verification
required no additional GPU inference. See the frontend audit for review scope.

## LoRA training and VRAM diagnosis (2026-10-04)

Run 168 is the pending full-data Ming Design preset copied from Qwen run 167.
The read-back parameters differ only in model/run identity, fresh-start resume
selection, and FP32 VAE precision required by the native Ming path. The original
datasets (38, 37, 30), rank/alpha 128, base LR 5e-5, 1024/1280/1536 resolutions,
partition/global adapter, guidance loss and ConvRot policies are retained.
The full preset has not been started.

Run 169 exposed and corrected shared `single` image-item compatibility,
cancellable preflight progress, duplicate primary-image decoding and redundant
preloaded-epoch validation. Its full-data initialization was stopped; no optimizer
step was completed. Run 170 uses 24 unchanged images/captions, eight from each
original dataset (registered as 44, 42, 43), with a three-step cap, save every step
and sampling disabled. This is a functional check, not a quality or throughput
benchmark of the original dataset.

The initial run 170 reached about 48 GB physical GPU usage. Phase diagnostics
showed 19.38 GiB of live CUDA allocations but 40.75 GiB reserved when its frozen
conditioner loaded: mixed-resolution VAE pools survived component offload.
Release those pools before staging the next component and after encoder offload.
Also release obsolete VAE pools between resolution buckets within a swap window;
the outer training loop's bucket cleanup does not cover that prefill loop.
All-valid key masks are removed before attention dispatch, preserving equivalent
outputs/gradients while avoiding unnecessary Flash-to-native fallback.

With phase release applied, the real run resumed step 1, restored all 277 named
optimizer tensors and completed step 3 with exit code 0. Conditioner load reserved
19.52 GiB, its forward peaked at 20.34 GiB live / 21.39 GiB reserved, and returning
to DiT left 7.55 GiB reserved. VAE prefill still peaked at 29.64 GiB reserved in
this run: the additional within-window bucket cleanup was added afterwards and
verified on CPU, not rerun on GPU. Only Flash dispatch was reported in the final
training log. These numbers are phase measurements, not an asserted maximum over
every forward/backward allocation.

Step 3 loss was 5.517778; warmup LR was 5e-7. Its checkpoint contains 414 finite
tensors, 277 changed since step 1 and all 136 ordinary LoRA up matrices nonzero.
Native profile, INT8 backward policy and global adapter metadata were retained;
LoRA, training state and optimizer artifacts were saved. CPU checks covered
attention/query chunk/residency (59 cases before the allocator additions),
updated residency (28 cases), and buffer behavior (7 cases); these sets overlap.
Changed backend files compile and import without CUDA initialization. Full-data
long-run memory behavior and quality recovery remain unverified.

## Five-epoch continuation: odd patch grids

Extending run 170 to five epochs exposed a partition-planner rejection of a
105x60 native patch grid (an aligned 210x120 latent). Ming's adapter now opts
into odd patch-grid sides; the common planner retains its even-side requirement
by default for Qwen. Core coverage, halo bounds and native pixel/patch alignment
remain validated. U-Net downsample divisibility remains a separate constraint.
CPU checks passed 36 Ming planning/forward/backward cases, including odd grids,
guided losses and aligned U-Net cases, plus four existing Qwen coverage cases.
The learned-padding test fixtures now initialize tokens that production obtains
from the checkpoint instead of relying on `torch.empty` contents.
Run 170 retains the 24-image subset and a total five-epoch/120-step target.

The next GPU continuation passed the partition planner and reached step 5, then
exposed full-frame BF16 guidance sharing one autocast scope between its no-grad
null pass and trainable pass. That cached detached FP32-to-BF16 weight casts and
removed the positive loss graph. Separate the two autocast scopes, as the tiled
path already does. The CPU cast-cache regression and existing guidance suite
passed 21 cases; saved step-5 LoRA/optimizer state is used for continuation.
The corrected GPU continuation reached step 8/120 with finite loss 7.6341295,
LR 1.75e-6 and status `running`; step 6-8 checkpoints were saved. Five-epoch
completion is not claimed. Browser reconnection timed out, so this continuation
was verified through the trainer log and REST status rather than a new UI capture.

## Dataset preflight latency

On the original 17,683-image dataset, DB/caption materialization took about three
seconds, while native Ming preflight fully decoded every original image before
bucketing. Preserve successful plain-Design size validation in a per-run JSON
cache under `.dataset_cache/ming_preflight_sizes_v1.json`. Every reuse still reads
and hashes the source bytes; only an exact SHA-256 match skips pixel decoding.
Sidecars/references use the full validation path, changed images are decoded
again, corrupt cache files fall back to validation, and failed/cancelled passes
do not replace the cache. Per-batch content checks remain in place.

Eight original dataset-38 images measured 0.233 seconds on first validation and
0.011 seconds on cached validation (about 22x for that small, OS-cached sample).
This is not a full-dataset speed guarantee; initial validation remains necessary.

2026-10-05 update: the same filename now carries schema v2 file stamps
(device/file ID, size, nanosecond modification/creation-or-change times). Unchanged
plain-Design files reuse SHA-256 and oriented geometry without reading image
payloads. Old v1 entries are upgraded after one hash read; new/changed plain
images use oriented headers, deferring full decoding until pixel loading. Layer
and reference/sidecar records retain full preflight. Content-based supervision,
crop and tensor-cache identities are unchanged. Pixel loading still checks the
actual bytes against SHA-256; caption-only/prefilled checks use file stamps.
Metadata equality is a fast change detector, not a replacement content hash.

An OS-cached 32-image sample read 25,977,857 bytes with the old cache path and
zero image payload bytes with the new warm path: 0.0279 s versus 0.00551 s
(about 5x). The one-time v1 upgrade took 0.0247 s. This does not predict whole
dataset or network-drive latency. The dataset/buffer/cache/pixel-prefetch CPU
regression suite passed 109 tests without using the training GPU.

PNG refinement: Pillow's `PNG.getexif()` itself calls `load()` when EXIF is
absent. Preflight now scans PNG metadata chunks, including EXIF after IDAT,
without decompressing pixels. The final 51-case dataset suite includes tests
that forbid `PngImageFile.load()` during new-image preflight and preserve the
old supervision identity. On the same 32 images, uncached validation took
0.727 s with full decoding versus 0.0276 s with metadata-only inspection
(about 26x); both read the same source bytes for SHA-256.
The combined dataset/progress CPU suite passed 49 cases, and edited modules
compiled/imported without initializing CUDA. Initialization progress now retains
the explicit `Validating Ming images` detail instead of a generic loading label.
The running child was not restarted; these changes apply to subsequent training
processes, and a cache must first be populated by a successful validation pass.
# Training refill and transfer follow-up (2026-10-04)

Follow-up: swap buffer progress now uses the same `latent_cache` and
`text_encoder_cache` callback phases as Qwen/SDXL, including initial refill,
subsequent refill and partial tail windows. Successful item counts are reported
at start, every ten batches and completion; cancellation never emits a false
completion. The existing frontend polling and phase bar consume these fields.
The running child retains its imported code, so this correction applies on its
next restart; the progressing run was not interrupted solely for this display fix.

Ming coincident latent/conditioning refills now defer restoring DiT until both
encoder phases finish, including failure/stop cleanup. Plain Design batches
without references or persistent conditioning caches prepare text inputs from
captions and bucket geometry without decoding target pixels. Layer/reference
and persistent-cache paths retain their paired-pixel preparation. Source hashes
remain verified, including buffered consumption; no weaker mtime-only check is
introduced. Full-frame and partition input copies use non-blocking transfers,
matching the shared Qwen/SDXL path; this alone does not promise transfer overlap.

CPU buffer/residency/guidance regression: 59 passed. Existing async activation
offloader CUDA value/gradient/lifetime regression: 3 passed. Run 170 stopped
normally and saved step 209 before changing its existing activation transfer
mode to async with the configured 2048 MiB pinned arena and synchronous overflow
fallback. End-to-end throughput improvements have not yet been measured.
