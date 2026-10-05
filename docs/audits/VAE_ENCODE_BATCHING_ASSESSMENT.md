# Independent VAE encode batches (2026-10-05)

## Conclusion and scope

Separating frozen VAE preparation batch size from optimizer batch size is
feasible. It is not an established speed improvement: native Ming FP32 encoding
was slower at B=2 and B=4 than B=1 in the measured conditions, with substantially
more VRAM. Preserve the current behavior by default. An explicit upper bound
could support other GPUs, lower resolutions and different VAEs after measurement.

This assessment covers all 17 entries currently in `ARCH_REGISTRY`, including
architectures without an image VAE. Only Ming received GPU measurements; other
entries are source assessments, not performance acceptance. Training runs,
optimizer settings and the loaded generation model were not changed. This is
an implementation proposal, not a shipped API option. VAE decoder/encoder
fine-tuning is a separate objective; this proposal concerns frozen encoding for
LoRA/Full backbone training, not its optimizer batch size.

## Current preparation boundary

`BaseTrainer.encode_image` accepts one PIL image, builds a batch-one tensor and
dispatches through `ArchHandler.vae_encode`. Shared swap refill in
`base_trainer.py` iterates flattened training items one at a time, then scatters
CPU latents into its existing buffer. Training batches are assembled later.
Ming uses its native epoch-owned `MingBatchBuffer`, with one-logical-batch CPU
lookahead and a single-owner VAE. Its clean outputs, reference latents, paired
crops and conditioning must remain separately owned.

Most underlying VAE encoders accept B>1, but that does not make the surrounding
PIL processing, latent packing, cache identity and consumption code batch-ready.
Batch slicing explicitly splits B into one-item encodes; disabling it to pursue
throughput removes a memory safeguard. Spatial tiling is a different dimension
and must keep its current policy. A model accepting B does not imply that its
GPU kernels have spare capacity: convolution work may already saturate at B=1.

## Architecture matrix

All speed statements below except Ming are hypotheses to test. No non-Ming
model was loaded onto the GPU for this assessment. Native VAE descriptions do
not override the installed/swapped VAE contract.

| Architecture | Current encode and batching changes | Throughput judgment |
|---|---|---|
| SD1.5 | `sd_sdxl_ops.vae_encode`: BCHW encoder, posterior sample, registry normalization; unwrap `SDXLVAEWrapper` if present. Stack identically sized, individually prepared images, then split B preserving each item's metadata. | Technically straightforward candidate, particularly small images; speed unknown. |
| SDXL | Same encode seam as SD1.5; preserve each image's crop and `time_ids` separately. Frozen encoder dtype remains the configured one. | Candidate; 1024px may already saturate B=1, and increased activation memory needs measurement. |
| Z-Image | `zimage_ops`: direct encoder + optional quant-conv, split moments, random posterior noise, registry normalization. Current tensor arithmetic supports B; do not replace the direct encoder with another distribution path. | Candidate; original per-item noise semantics must survive grouping. No measured gain. |
| Flux2 | `flux2_ops`: BCHW posterior sample, registry normalization in VAE-owned packed domain, then backbone patchify. Both stages must retain B. Registry uses stored BN running statistics, not statistics computed across new batches. | Candidate; eval/frozen state and packed-domain normalization must remain unchanged. Gain unknown. |
| Anima | `anima_ops`: BCHW becomes BCTHW with T=1, Qwen/Wan-family posterior sample, per-channel normalization, remove T. Stack on B, never concatenate independent images on T. | Candidate, but causal VAE kernel saturation and slicing may limit gains. No GPU evidence. |
| Qwen-Image 2.1 | `qwen_image_21_ops`: RGBA preprocessing, BCTHW/T=1, posterior sample, mean/std normalization, flatten each B to a token sequence. PIL branch currently receives one image; new tensor path must reproduce alpha/preprocessing exactly. | Candidate, not proven faster. Run 167's BF16 setting differs from Ming FP32; do not transfer Ming timing to Qwen. |
| Ming Design | `training_data.encode_frames`: RGBA BCFHW; independent F frames encoded at T=1, posterior mode, native shift/scale. Group prepared targets across logical items on B and scatter back; current native buffer also owns prompts/references. | Measured B=2/4 slower than B=1 at 512/1024 FP32. Keep B=1 default on this GPU. |
| Ming Layer | Same native VAE; F represents independent target images/layers, not a causal video. Flatten `(item, frame)` onto B with an explicit scatter map, then restore each item's F and ordered reference list. Different frame/reference counts need not force padding when flattened. | Same encoder work as Design, but Layer packing was not measured. Larger B is not accepted as a speed improvement. Preserve every role/weight/reference. |
| Lens | `lens_ops` calls a PIL-one-image helper: RGB/LANCZOS preparation, Flux2 VAE posterior mode, registry normalization, B-aware rearrange. Introduce a prepared-tensor helper; changing only the call-site batch count does nothing. | Candidate after wrapper refactor; same VAE family as Flux2 does not establish the same measured throughput. |
| Ideogram4 | PIL-one-image RGB helper + Flux2 posterior mode. Current packing explicitly uses `view(1, ...)` twice; replace with actual B and split before logical consumption. Stored BN statistics must remain unchanged. | Candidate after wrapper/packing changes; no measured gain. |
| Krea2 | PIL-one-image RGB helper, Qwen-family native mode, VAE-rank-dependent T=1, registry normalization and latent packing. Add a prepared-tensor helper and retain runtime VAE rank/patch size. | Candidate after wrapper refactor; no measured gain. |
| MiniT2I | Pixel variant has no VAE. Latent variant uses configured/swappable VAE and posterior sample. Loader enables tiling and slicing when supported; slicing can serialize a nominal B>1 call. | Pixel variant gets no VAE speed benefit. Latent variant requires a measured slicing/memory policy before true parallel batching. |
| SenseNova U1.5 | Native pixel-space path returns BCHW RGB directly, no VAE. Swapped VAE path uses `latent_space.encode` with posterior sampling and registry normalization, already B-shaped. Respect alignment and MoT/text-output contract. | No native VAE benefit. Swapped-VAE benefit depends on actual installed encoder, not architecture name. |
| SenseNova SDXL Chimera | `sensenova_sdxl_chimera_ops`: BCHW posterior sample, native shift/scale. B-aware arithmetic; preserve any conditioning and each item's metadata. | Straightforward candidate for frozen encode; no measured gain. |
| LTX-2.3 | Still path T=1; clip helper maps TCHW to BCTHW with B=1, samples posterior and normalizes. Group clips only with identical spatial shape, temporal length and causal/tiling contract; retain each clip window and its audio association. | Video already has substantial temporal parallel work and memory; B growth may worsen throughput/memory. Do not treat T as B or batch unequal clips by unvalidated padding. |
| MiniMax-H3 | Clip helper maps TCHW to BCTHW with B=1, ImageNet RGB normalization, posterior mode and native latent normalization. Preserve temporal compression, leading padding, chunk/overlap policy and still/clip equivalence. Vendor encoder accepts B and can slice it. Audio encoding remains a separate owner/window. | Technically possible for equal clips, but resource-heavy; benefit unknown and lower implementation priority. |
| ACE-Step 1.5 | No image encode. Audio helper builds `[1,2,samples]`, Oobleck posterior sample and `[B,T,64]` output. Group equal-length waveforms; do not invent silent-padding semantics for different durations. | Separate audio-encode option could be evaluated. An image VAE batch setting should not silently claim audio coverage. |
| YuE2 | Shipped Phase-A ABC-planner training consumes native token sequences, no VAE latents. `vae_encode` explicitly refuses. | Not applicable. |

Ming Design/Layer are two profiles of one registry architecture. MiniMax Music 3
is generation-only and outside this training option; its codec pipeline is not
a missing image-VAE training batch path.

## Ming measurements

Hardware: NVIDIA RTX 6000 Ada Generation, 48 GiB. Official converted Design
package's native VAE loaded in FP32, eval/no-grad, tiling/slicing off. Eight real
dataset images were prepared once per resolution, put on GPU in advance and
encoded with posterior mode. Each batch size was warmed, then measured in
B=1,2,4,4,2,1 order. Each timed pass encoded the same eight images. Table values
average two passes. The existing generation pipeline was not replaced.

Environment: Torch 2.10.0+cu130, CUDA 13.0, cuDNN 9.12; existing cuDNN TF32
permission remains enabled, matmul TF32 disabled. FP32 here names tensor/weight
dtype, not a claim that every convolution uses strict IEEE FP32 multiplication.

| Resolution | VAE B | ms/image | Change versus B=1 | Peak allocated CUDA GiB |
|---|---:|---:|---:|---:|
| 512x512 | 1 | 47.18 | baseline | 1.76 |
| 512x512 | 2 | 49.20 | 4.3% slower | 2.98 |
| 512x512 | 4 | 50.90 | 7.9% slower | 5.44 |
| 1024x1024 | 1 | 219.04 | baseline | 5.53 |
| 1024x1024 | 2 | 223.23 | 1.9% slower | 10.43 |
| 1024x1024 | 4 | 225.78 | 3.1% slower | 20.22 |

Peak allocated includes VAE weights, the eight prepared GPU inputs, output
latents and encoder allocations in this process. It excludes the owner's model
process, desktop, full trainer, gradients and optimizer state; it is not total
device occupancy nor a Full-training memory budget. Timings exclude source
read/decode, prompt processing, CPU-GPU transfer and output CPU copies. Small
timing differences are not statistical confidence intervals; this bounded
comparison establishes no batching speed benefit in these conditions.

B=1 repeated output was exact. B=2/4 relative latent RMS error versus B=1 was
approximately 1.5e-5; maximum absolute error was 0.00113 (512) / 0.00108 (1024).
All values were finite. Changed convolution execution produces small numerical
differences even with posterior mode. No decoded-image or training-loss
acceptance was performed for batching. Reproduction/results are ignored files
`local/deployment/ming_vae_batch_benchmark.{py,json}`; no disk tensor cache was
created. Do not confuse these encode-only times with the earlier CPU-lookahead
end-to-end VAE preparation comparison.

## Proposed option and integration

1. Add a centrally owned `vae_encode_batch_size` (name proposed) as a positive
   integer upper bound, default 1, independently of `batch_size`, accumulation
   and MNT. Update OpenAPI, request/config serialization, defaults, UI, copy and
   resume paths together. UI should distinguish encoding images from optimizer
   samples and explain that effective batch can shrink for compatibility/VRAM.
   Enable only on supported frozen image-encode paths; explicitly handle no-VAE,
   video and audio boundaries rather than silently ignoring an unsupported value.
2. Separate per-item image preparation from the VAE tensor encoder. Retain crop,
   SDXL time IDs, alpha, resize arithmetic, source verification, cache keys and
   reference ownership. Plan within the existing refill window, grouping exact
   `(VAE identity, pixel dtype, channels, H,W, temporal/causal contract, tiling
   policy)` matches. Cached hits do not consume GPU encode slots. Scatter results
   back to original logical positions without changing training batch order,
   number of updates or scheduler steps. Initially use contiguous compatible
   items; arbitrary regrouping adds RNG and buffer ownership work.
3. Keep preparation and sampling order stable. Several current encoders sample
   a posterior; grouping can change random-number consumption even if images
   match. Prefer grouped moment computation followed by per-item sampling in the
   original order/shape with an explicit generator policy. Preserve mode paths
   as mode. CPU crop RNG and GPU diffusion/posterior RNG must not accidentally
   advance in a different order. A new seed policy needs explicit versioning,
   not a claim of bitwise backward compatibility.
4. One owner invokes the VAE; mutable temporal feature caches make concurrent
   calls unsafe. Preserve CPU lookahead, but cap both pixel working memory and
   GPU batch memory. Respect actual free memory after encoder residency changes;
   optimizer moments/gradients and retained allocator state still matter even
   when DiT weights are offloaded. Split a compatible group down to B=1 after a
   genuine CUDA OOM, clearing failed temporary outputs and preserving prepared
   inputs/noise. Do not classify source errors as OOM or claim B=1 must fit.
5. Keep current slicing/tiling policy unless explicitly selected and validated.
   A slicing-enabled VAE should report the effective serialization rather than
   advertise parallel B>1. Record requested/effective encode policy and numerical
   provenance in run/checkpoint metadata. VAE batch changes are not changes to
   the optimizer batch ledger, but they may change recomputed latent bits;
   persisted cache identity/version must represent the accepted encode policy.
6. Progress counts successful original items, not encode calls; stopping during
   a GPU call completes that call, then suppresses consumption and joins CPU
   workers. OOM retries must not double-count progress. Preserve error diagnostics
   and partial tail groups. Never build an unbounded decoded-image queue merely
   to fill a larger VAE batch.

Implementation order: shared prepared-tensor seam for SD/SDXL/Z-Image/Flux2/
Anima/Qwen/Chimera and independent configuration contract first; PIL wrappers
next; Ming target/reference scatter with unchanged native preprocessing next.
Video/audio require their own equal-window contract and remain separate gates.
No arch should be marked faster solely because its interface accepts B.

Acceptance should compare B=1/2/4 at small and actual training buckets, warmed
in alternating order, at unchanged encoder precision, tiling and source inputs.
Measure encode-only and end-to-end refill separately, live allocated/reserved
VRAM, numerical error and source/crop/reference identity. Posterior-sample tests
must control the original per-item random noise. Exercise tail groups, mixed
buckets, cached hits, source drift, cancellation, OOM fallback and resumed runs.
For Ming Layer verify all ordered roles, frame weights and reference outputs.
No need to repeatedly run full suites or load every architecture during design.

## Source entry points

- `backend/core/training/arch/__init__.py`: current capability roster.
- `backend/core/training/base_trainer.py`: `encode_image`, cache/swap refill and
  subsequent training-batch assembly.
- `backend/core/training/ops/*_ops.py`: architecture normalization, posterior
  policy, temporal/audio seams and MiniT2I slicing setup.
- `backend/core/models/components/vae_registry.py`: VAE-owned normalization,
  packed domains and frozen BN running statistics.
- `backend/core/models/{lens,ideogram4,krea2}/*_pipeline_ops.py`: PIL-only encode
  wrappers and single-item packing.
- `backend/core/models/ming_image_01/{training_buffer,training_data,dataset}.py`:
  native epoch preparation, independent frames and target/reference ownership.
- Vendored Ming/MiniMax-H3 and installed Diffusers Qwen/LTX2 encoders: B-axis
  support, batch slicing and temporal cache behavior.
