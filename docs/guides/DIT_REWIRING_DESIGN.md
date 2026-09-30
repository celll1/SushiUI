# DiT residual rewiring contract

Status: implemented experimental option; functional verification accepted by
the owner on 2026-10-01. Qwen-Image 2.1 has actual GPU/API
inference, LoRA/full training, checkpoint restore/resume and block-swap workflow
verification. Other architectures have provisional native implementations and
CPU contracts only, as requested by the owner. Frontend verification is static;
consolidated build/type/render checks belong to the owner.

This feature changes transformer token resolution between existing blocks. It
adds no model parameters and does not turn arbitrary foreign checkpoint formats
into supported models. A native execution/artifact adapter validates each
architecture's masks, positions, stream layout and checkpoint identity. Existing
model/training/adapter capability boundaries remain authoritative. SD1.5, SDXL
and SenseNova SDXL Chimera already use U-Nets and are outside DiT rewiring.

## Default topology

A new user-selected U-Net conversion defaults to two nested regions with four
blocks at the deepest level. This does not automatically convert ordinary models
or replace the topology declared by an existing checkpoint.

For Qwen-Image 2.1's 32 main blocks, block indices are zero-based half-open:

| Region | Block interval | Spatial downsample relative to parent |
|---|---|---|
| Outer | `[4, 28)` | 2 by 2 |
| Inner | `[14, 18)` | 2 by 2 |

At 1024px, eight blocks use 4096 image tokens, twenty use 1024 tokens, and four
use 256 tokens. The latent input/output grid and text tokens retain their native
meaning. The default block allocation is:

| Zero-based blocks | Image token grid at 1024px | Tokens per block | Blocks |
|---|---|---|---|
| `[0, 4)`, `[28, 32)` | 64 by 64 | 4096 | 8 |
| `[4, 14)`, `[18, 28)` | 32 by 32 | 1024 | 20 |
| `[14, 18)` | 16 by 16 | 256 | 4 |

```text
blocks    0..3    4..13    14..17    18..27    28..31
tokens    4096    1024       256      1024      4096
outer skip        [----------------------)
inner skip                  [-----)
```

Each skip connects its region entry to its exit, restoring that region's residual
update. The outer region is `[4, 28)` and the inner region is `[14, 18)`.
The 1024 values above describe image token counts, not image pixels.

This allocation retains more blocks at 1024 tokens than the sixteen-deepest-block
probe. Its image-token-linear work is approximately 41.40625% of dense execution;
this is not a full-model FLOP or throughput estimate. Fixed pooling and upsampling
add no parameters, so checkpoint parameter size does not shrink. The measured
quality improvement is partial and still requires calibration.

Other architectures must resolve valid model-specific intervals before
registering a default; do not reuse these indices without validation.

## Execution

Each region preserves its input image hidden states and executes its blocks on
a pooled copy. Restore only the update:

```text
low_in = pool(hidden)
low_out = execute_region(low_in)
hidden = saved_hidden + upsample(low_out - low_in)
```

Use fixed average pooling, bilinear upsampling with `align_corners=False`, and
identity delta addition initially. Each nested region performs its own subtraction
and restoration. Preserve text updates, native block order, masks, positions,
conditioning, gradient checkpointing, and offload ownership. Region context is
per forward and must not depend on a mutable global skip stack.

Allow nested or disjoint regions; refuse crossing regions, cycles, reordered
blocks, unsupported stream transitions, and indivisible grids. Rebuild coarse
positions using the architecture's coordinate contract. Restore the original
grid and context before final output projection.

## Configuration and artifacts

All API defaults and presets belong to `backend/api/param_defaults.py`.
Configuration is architecture-neutral, with per-architecture generation and
training capabilities. Do not add rewired variants to ModelType/ARCH_REGISTRY.

Store a versioned canonical JSON topology, its SHA256, architecture, stream block
counts, block-layout/config identities, intervals, parent regions, operator and
position rules, and any learned topology tensors. Safetensors metadata values
are strings. Single files, shard indices, and directory artifacts must agree.
Unknown versions, incompatible layouts, missing tensors, or conflicting metadata
must fail before mutating the active model. Metadata cannot execute Python.

Generation restores the checkpoint topology automatically and shows it read-only.
Training can inherit a topology or configure a new conversion. Resume requires
agreement between run configuration and checkpoint topology. LoRA artifacts must
carry their required topology and base identity; incompatible adapter stacks must
be refused atomically. Preserve existing adapter capability tables.

## Editor and acceptance

Provide a shared graphical topology renderer: block index on the horizontal axis,
resolution levels on the vertical axis, draggable interval endpoints, and explicit
skip edges. Show image pixels, latent grid, token grid, per-level counts, and
validation errors. Backend-resolved plans are authoritative for editor previews.
Keep presets, YAML editing, clone, export, and queues consistent.

The uncalibrated Qwen probes show blur and granular artifacts. Shortening the
deepest region from sixteen to four blocks improves some shapes but does not
establish acceptable quality or universal performance gains. Full/LoRA
save-load-resume, dtype gradients, checkpointing/offload and API round trips
have functional evidence below. The owner accepts functional verification for
this delivery and defers quality-recovery training. Calibration quality remains
unproven; non-Qwen GPU acceptance and consolidated frontend checks are deferred
according to the owner's explicit verification scope.
Do not infer reduced checkpoint/optimizer size from reduced token computation.

The method was assessed from
[lodestone-rock/residual-to-unet](https://github.com/lodestone-rock/residual-to-unet/tree/4834458ea1e2dbede89e776ebccbe23233d10fd0).
Its seven-layer linear reference is not a real DiT quality or speed benchmark.
The SushiUI probe is independently written from the residual-update formulation.

## Current architecture coverage

The training configure/inherit capability list is
`backend/core/rewiring/capability.py`. The two additional acoustic generation
paths do not enter that training list. Native dimensions and presets are resolved
from `backend/api/param_defaults.py` and the architecture adapters; a nonmatching
block layout requires an explicit validated graph rather than guessed intervals.

| Architecture | Rewired stream / layout | Training boundary | Verification |
|---|---|---|---|
| Qwen-Image 2.1 | main image tail; text stays native | LoRA and full parameter | GPU/API plus CPU |
| Krea2, Anima, Z-Image | native image stream and architecture-owned positions | Supported LoRA/full paths | CPU/provisional |
| Flux2 | double and single streams; default rewires single only | Supported LoRA/full paths | CPU/provisional |
| Lens, MiniT2I | native packed image grid | Supported LoRA/full paths | CPU/provisional |
| Ideogram4 | native image backbone/refiner contract | LoRA only | CPU/provisional |
| SenseNova U1.5 | native MoT generation stream | Supported LoRA/full half; residual batch size 1 | CPU/provisional |
| ACE-Step 1.5 | temporal diffusion decoder | Supported LoRA/full paths | CPU/provisional |
| LTX-2.3 | spatial grid independently per video frame | Supported native training paths | CPU/provisional |
| MiniMax-H3 | packed target video grid per frame; text/reference/audio retained | Existing LoRA-only path | CPU/provisional |
| MiniMax Music 3 | temporal flow DiT; timestep prefix retained | Generation only | CPU/provisional |
| YuE2 | acoustic NAR half; START/END and immutable AR prefix KV retained | NAR training refused; Phase-A AR LoRA unchanged | CPU/provisional |

Image grids require native divisibility. Video transport never pools across
frames and does not shrink audio/text/reference tokens. ACE-Step and Music 3
use native zero-tail temporal transport; YuE2 pads embedded acoustic hidden
states after adding native position/time embeddings. Architecture-owned policy
identities distinguish these paths. Padding is removed before native output
heads. YuE2 preserves native chunking/context budgets and refuses padding that
would exceed the model context before AR prefill.

Qwen rewiring requires native causal modulation and one contiguous 2D target
image grid. Reference/editing, partition/style paths and image KV caching are
refused or disabled by their explicit gates. Shared training gates refuse TREAD,
REPA, partition/global-adapter modes and block skipping. These restrictions do
not imply that the corresponding dense architecture feature is unsupported.

## API and user workflow

All endpoints below are versioned under `/api/v1`; `openapi.yaml` owns their
request/response contract.

| Endpoint | Behavior |
|---|---|
| `GET /schema/dit-rewiring` | Defaults-owned presets and training runtime capability list. |
| `POST /models/rewiring/inspect` | Native layout/config hashes, restored graph, preset, editable initial graph and grid geometry. |
| `POST /models/rewiring/validate` | Canonical graph/hash and authoritative per-block grid/skip preview. |
| `POST /models/rewiring/export` | New metadata-only acoustic safetensors checkpoint; no overwrite or training capability expansion. |

Training fields are `dit_rewiring_mode` and `dit_rewiring_spec`; their defaults
come from `TRAINING_DEFAULTS`. `inherit` restores the checkpoint graph and an
explicit spec asserts exact agreement. `configure` installs an explicit graph
or the native-layout preset. Resume cannot change graph identity. Missing LoRA
resume declarations denote dense execution and cannot silently inherit a
rewired base. Failed resume/adapter validation leaves base graph state intact.
The two fields survive request/YAML, run retrieval, clone, preset and queue
configuration through the same parameter contract.

The training editor displays each native stream, validated grid counts and skip
edges. Dragging endpoints or editing intervals, parent regions and per-axis
factors changes the same canonical spec. Crossings/cycles/config mismatches
produce errors. The active-model summary displays checkpoint topology read-only.
Loaded generation automatically restores metadata; the request need not repeat
it. Generation adapter scopes validate stack topology and restore the base graph
on unload, including failed inference.

Acoustic safetensors model cards additionally provide an editor/export panel for
complete YuE2 and flat Music 3 checkpoints. Inspection mounts only when opened;
export stays disabled until the graph validates. A successful export enters the
local model list/selection and loading remains an explicit normal model action.
Source and output use different names in the same directory, preserving Music 3
companion lookup. Native component directories/shards and GGUF export refuse;
existing metadata-driven generation from supported formats is unaffected.

Export changes only canonical graph metadata and preserves tensor descriptors,
dtypes, quantization declarations, unrelated metadata and every tensor byte.
It streams with bounded memory, checks source identity/configuration, verifies
the new header and publishes atomically without overwriting a racing output.
Failures remove temporary files. A full file copy needs disk space and a
hard-link-capable filesystem. This conversion does not calibrate weights.

## Artifact and generation provenance

Canonical metadata keys are `sushi.rewire.schema_version`, `sushi.rewire.spec`
and `sushi.rewire.spec_sha256`. Absence of all three denotes dense execution;
partial declarations, bad digests, unsupported versions, incompatible native
configuration/block hashes and conflicting shards refuse. Graph JSON contains
architecture, stream counts, half-open intervals, parents, shrink factors and
fixed operators. It contains no executable code. All shards/indexes/directory
members must agree where the native format uses them.

Native writers/loaders and supported full/LoRA resume paths own persistence.
Frozen companion references retain native VAE/conditioning provenance where an
artifact does not bundle those components. The complete YuE2 format still requires
its embedded tokenizer and VAE, exact native tensor census and supported
quantization; inspection uses shape-only meta models and bounded markers while
skipping tokenizer/weight payloads. Normal loading validates the tokenizer.
Music 3 inspection retains native companion configs and supported quantization
rather than guessing flow/condition dimensions.

Image/video metadata and audio JSON sidecars retain `dit_rewiring_actual` from
successful execution. Audio records identify native/padded token counts and
chunk-local/song-local scope explicitly; overlapping Music 3 chunk lengths are
not a song duration. YuE2 records full/visible AR prefix length, preserved
boundary count, ODE steps and actual velocity evaluations. It snapshots graph
identity across native midpoint solves and publishes only after finite waveform
decode and cleanup. Failure/cancellation clears stale caller metadata. YuE2's
existing token/latent sidecar stores the same graph under `effective_config`.
These records are provenance, not GPU throughput measurements.

## Verified behavior and practical limits

The 2026-10-01 consolidated CPU run covers all 21 `dit_rewiring*test.py` modules
and both MiniMax-H3 reload/native probe regressions: 231 tests and 95 subtests
pass in 50.86 seconds with CUDA uninitialized. It covers
independent residual output/gradient oracles, native block loops, FP32/BF16/FP16
identity transport, metadata/shard/config refusals, supported training adapters,
checkpoint restore/resume, adapter lifetime, inspection/preview APIs and acoustic
export. Tiny CPU fixtures are not production quality or throughput tests.

Qwen GPU/API evidence from 2026-09-30 and 2026-10-01 is retained under
`local/measurements/` (ignored machine-local artifacts). It includes:

- Paired 1024px, 40-step, CFG 1, INT8 ConvRot inference for dense, single-stage,
  sixteen/eight/four-deepest-block graphs. Uncalibrated blur/granular artifacts
  occur; shortening the deepest region improves some shapes but not every prompt.
- Actual metadata-driven FastAPI inference pixel-identical to the four-deepest
  probe; one full request takes 19.27 seconds with 9.21 GiB peak allocated memory.
- LoRA 1024px dataset/API training, periodic samples and checkpoint/optimizer/RNG
  resume both resident and with 30 swapped blocks/two ring slots.
- Full parameter training step one and resume to step two: all 297 trainable
  tensors, 7,115,124,736 parameters, frozen TE/VAE, finite saved weights and
  restored optimizer states. The four-shard checkpoint is about 14.23 GB.
- Generation from the saved full checkpoint without explicit topology and
  matching loaded-model/PNG graph metadata. A 1024px, 40-step API check completes
  40 native forwards in 34.95 seconds with 14.00 GiB peak allocated memory.
  Its two ordinary training steps do not recover image quality; pronounced
  blur/pixelation remains, and this is not a calibration result.
- The original INT8 model with metadata-bearing zero LoRA reproduces the earlier
  four-deepest-block image exactly (pixel MAE 0), completing 40 native forwards
  in 24.90 seconds with 9.21 GiB peak allocated memory. The requested KV cache
  is disabled for rewiring; adapter cleanup restores the dense base topology.
  These whole-request timings include generation overhead and are not a paired
  warm throughput comparison. Reports are in `rewiring_qwen_final_20261001` and
  `rewiring_qwen_original_final_20261001` under the local measurements directory.

Four-deepest-block loop observations are 3.98–4.33 step/sec; one paired warm
prompt improves about 5% over the single-stage graph. Other timings vary enough
that they do not establish a universal speed multiplier. They are progress-loop
averages, not GPU-event measurements. The default image-token-linear work ratio
is 41.40625%, excluding text/joint attention, pooling, encoders and VAE. Whole-model
FLOPs and quality after calibration are unmeasured. Counting image-image token
pairs alone gives `(8 + 20/16 + 4/256)/32 = 28.955078125%` of the dense default;
this is a structural attention-pair ratio, not fused-kernel FLOPs or speed.
Fixed transport preserves
parameter/checkpoint/optimizer size. Calibration is experimental and workload
dependent; short workflow checks do not establish recovered image/audio quality.

GPU validation is restricted to Qwen-Image 2.1 by owner instruction. Non-Qwen
production GPU quality, speed, dtype/offload and full media workflows remain
unverified; those implementations are provisional. Frontend build/type/render
verification is explicitly owner-run; the agent uses static source/API review.

## Consolidated owner checks

- Create a Qwen LoRA/full run; Configure / Model preset shows 8/20/4 blocks at
  4096/1024/256 image tokens for a 1024px preview.
- Drag/edit boundaries, shrink factors and parents. Verify crossing/cyclic or
  indivisible graphs fail and restoring the preset recovers the display.
- Save/reopen/clone/preset/queue a run and confirm mode/spec survive. Older runs
  receive API defaults without overwriting saved values.
- Change base models and check identity mismatch until a compatible graph is
  selected. Load a saved rewired checkpoint and check its read-only graph.
- Generate with restored Qwen metadata and inspect saved graph/cache provenance.
- Statically review other model options; their GPU acceptance is not requested.
- For acoustic export, confirm pending/invalid graphs disable export, an existing
  filename refuses, the new file appears in selection and source state is retained.
- Run the owner's consolidated frontend type/build and browser checks. No such
  commands are run by the agent.

The owner accepts inference/training/save-restore functional verification for
this delivery. The checks above remain the owner's consolidated manual pass;
quality-recovery training is a separate deferred evaluation. This guide does
not claim calibrated quality or production acceptance for every architecture/format.
