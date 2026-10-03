# Tagger subsystem

SushiUI has two independent tagger stacks. The legacy WD-compatible stack is
inference-only. The SigLIP2 stack supports inference, dataset sidecar editing,
full-parameter or LoRA training, checkpoint analysis, and ONNX export. Do not
share checkpoint assumptions between them.

## Ownership

The owner abandoned the A/B/C prerequisite at step 43,322. The new
`/tagger-training/foveatag/joint-review` route registers a separate
`foveatag_joint_review` run using only the stopped scratch run's own weights.
It trains encoder/head/router together, uses a new optimizer and an explicit
fixed-square, unpadded overview policy, and retains square source-native locals
inside source boundaries. Shape distortion of the overview is a recorded
tradeoff. No fabricated gate or external tagger weights are used.
The legacy bounded start is limited by `review_steps`; it stops with a resumable joint
checkpoint and a model/router-only `review.safetensors` for owner visual review.
The latter is the latest review export, not an automatically accepted best model.
`POST /tagger-training/runs/{run_id}/start?continuous=true` resumes joint runs
without that segment stop. The joint monitor uses continuous starts; explicit
stop still saves the optimizer/scheduler/RNG state.
Defaults live in
`TAGGER_JOINT_REVIEW_DEFAULTS`, not the generic SigLIP2 creation request.
The monitor's loss curve uses the classification ASL term; the joint summary
records classification/ranking/diversity terms separately.
Diagnostics are enabled for this workflow but remain optional. At exact global
step multiples of the user-selected interval, they capture one image (without a rating filter)
image from that step's actual training batch, its pre-update probabilities,
supervision mask, and the native boxes actually selected for that forward.
Solid boxes are router selections; dashed boxes are random exploration.
No fresh inference or alternative image substitutes for the training forward.
If source metadata is unavailable, or the CPU renderer is busy, the
attempt is recorded as skipped. Rendering uses an isolated CPU process without
copying the full optimizer checkpoint. Neither a numeric
gate nor the agent's inspection of generated images substitutes for owner
review. Normal inference-manager integration and repeated refinement remain
separate unfinished production work.

Registered `foveatag_phase3` comparison monitors expose optional diagnostic
panels under `/tagger-training/runs/{run_id}/diagnostics`. They default to OFF;
settings can be changed while running without changing the frozen A/B/C training
contract. For these legacy comparisons only, `interval_steps` is a minimum catalog-step interval evaluated at saved
checkpoint boundaries (not a wall-clock schedule). A single isolated CPU worker
copies the atomic resume checkpoint and evaluates one deterministic,
explicitly annotated Rating:General non-photo sample. The first selected sample
ID is pinned across later panels; if it becomes unavailable, generation reports
an error instead of replacing it silently. This adds checkpoint I/O and CPU/RAM
cost, but never forwards through the live training model or changes its
optimizer, RNG or sample budget. Manual generation requires a paused run.

Panels contain anonymized TP/FP/FN group ranks (up to 100 each), TN counts and
unknown-label masking; no tag names or tag IDs are published. Counts remain
annotation-based rather than guaranteed ground truth. A has no native focus;
B uses overview-derived placebo views; C overlays the exact native view boxes
passed to that diagnostic inference. Source pixels per canvas pixel are shown,
so source-derived does not imply every view is exactly 1:1. An optional C-policy
preview for A/B is explicitly marked as NOT used by that inference. Boxes are
input footprints, not causal attribution or a trained router. The monitor supports
older/newer navigation and step selection; polling does not move a selected older
panel. All published panels remain in the history (no 12-entry truncation).
Disabling generation does not delete prior images or cancel an in-flight worker.
This diagnostic option supports registered comparisons and joint-review runs, not general
SigLIP2/native trainer runs. Frontend build/type-check remains owner-run.

| Concern | Owner |
|---|---|
| Legacy WD model load and prediction | `backend/core/extensions/tagger_manager.py` and `/api/v1/tagger/*` |
| SigLIP2 model, checkpoint inheritance, and head growth | `backend/core/tagger/siglip2_tagger_model.py` |
| SigLIP2 loaded-model inference | `backend/core/tagger/siglip2_inference_manager.py` |
| Shared loaded/live response selection | `backend/core/tagger/tag_selection.py` |
| Dataset listing and `.txt` sidecars | `/api/v1/tagger/browser/*` and `backend/core/tagger/browser_sidecars.py` |
| Training orchestration and checkpoint bundles | `backend/core/tagger/tagger_trainer.py` |
| Run persistence and REST lifecycle | `/api/v1/tagger-training/*` in `backend/api/routes.py` |
| API defaults | `TAGGER_TRAINING_DEFAULTS` in `backend/api/param_defaults.py` |
| Tagger UI | `frontend/src/components/tagger/` and `frontend/src/components/training/tagger/` |

The frontend obtains training defaults from
`/api/v1/schema/tagger-training-defaults`. A new UI default must not be added
locally; add it to `TAGGER_TRAINING_DEFAULTS`, the request schema, OpenAPI, and
the frontend request type.

## Inference paths

Both PyTorch SigLIP2 variants expose `forward_features()` for the compact
captioner foundation. It returns patch features, the actual head-input pooled
features, and logits from one encoder call. Ordinary `forward()` still returns
only logits. The caption feature provider requires evaluation mode and detaches
its outputs. The compact captioner now borrows the loaded PyTorch tagger under
a model-lifetime lock, with strict GPU pause acknowledgement for inference.
The Captioner tab on `/tagger` supports inference preview and explicit natural-caption
persistence. The Captioner Training section on `/training` owns dedicated training
runs with a private frozen encoder and checkpoint sample review. FoveaTag remains a future
plan. See [the captioner design](../plans/COMPACT_CAPTIONER_DESIGN_2026-10.md).

`POST /api/v1/tagger/siglip2/load` detects full, LoRA, or ONNX weights. The UI
may preview checkpoint metadata, but the status returned by the loaded manager
is authoritative. A vocabulary beside a checkpoint is only an initial path
convention; the manager validates the actual files.

`POST /api/v1/tagger/siglip2/predict` can prefer the active training model. If
that model is unavailable or temporarily offloaded, the route falls back to the
loaded inference model. Both paths call `select_tag_response`, which owns:

- global and per-tag threshold comparison;
- unreliable-tag exclusion and threshold floors;
- optional OOD threshold adjustment;
- Quality and Rating top-1 selection;
- response ordering and calibrated-probability fields.

The deprecated `display_calibration` request field remains accepted for API
compatibility but has no effect. Calibrated probabilities are returned whenever
the checkpoint has calibration data.

## Dataset browser safety

Browser routes resolve requested paths beneath the selected dataset root.
Sidecar writes use a sibling temporary file followed by `os.replace`. Bulk
editing is fail-closed: if any selected sidecar cannot be read, Apply remains
disabled until the read succeeds or the selection changes. Batch inference
holds the shared GPU coordination slot per image and writes tag names, including
the selected Quality and Rating tags.

The batch inference endpoint is an SSE stream. A terminal `done` or `error`
event ends the frontend operation; transport failure must also clear the busy
state.

## Training and resume contract

Training supports `full` and `lora`. Validation predictions are collected as
CPU float16 tensors to bound memory for large vocabularies. If online vocabulary
growth makes loader labels narrower than the current head, validation pads only
the new tag axis with zeros.

Each resumable checkpoint name pairs model weights with state, optimizer,
vocabulary, tag metrics, OOD reference, and metadata where enabled. Step
retention deletes the complete expired bundle. `latest` and `best_f1` are not
part of step retention.

Danbooru augmentation also persists its metrics, promoted tags, active
co-occurrence/query sets, and current-epoch collection progress atomically.
These files are required because tags already promoted into the vocabulary may
not be rediscoverable after restart.

When vocabulary indices change on resume, FP32 optimizer head state is aligned
by tag name. Bitsandbytes head state is cleared instead: moving quantized rows
would invalidate block-wise scale metadata. Encoder and adapter optimizer state
is not reset.

## Verification boundaries

The experimental native FoveaTag shared trainer maps `use_flash_attention` to
the external `flash-attn` kernels through `core/attention/`. Global attention
uses the dense kernel; shifted and padded windows pack connected regions into
independent varlen segments, preserving the existing mask and singleton padded
tokens. Selecting this option requires an importable installation. CUDA
FP16/BF16 with supported head dimensions use FlashAttention; CPU/FP32 and
unsupported head dimensions use SDPA. A selected kernel failure is fatal rather
than silently changing execution during training. Disabling the option excludes
both external FlashAttention and PyTorch's flash SDPA kernel.

The native optimized path computes RMSNorm statistics in FP32 to prevent FP16
padding-gradient NaNs. The frozen A/B/C runner retains its original eager path.
Opt-in real GPU tests in `backend/tests/foveatag_flash_attention_test.py` compare
outputs and gradients for global, shifted/padded, FP16/BF16 and checkpointed
execution. These checks do not establish end-to-end speedups or UI deployment.

On the isolated design branch, standard run creation also accepts
`architecture=foveatag` and the six `fovea_*` defaults exposed by the schema.
Full training accepts an empty encoder path for random initialization; LoRA
requires a local scratch-pretrained encoder. SigLIP2 pooling dimensions and
repository initialization are refused. The configuration form exposes the
native input settings alongside shared training options. Deployment and live
UI verification remain pending; the frozen comparison uses its import endpoint.

The native shared training-model prediction handle accepts coordinate-aware
views, uses deterministic validation sampling without consuming training RNG,
checks the source-pixel cap before decode and retains the run's CUDA precision.
It restores the prior train/eval mode even if prediction fails. Loaded SigLIP2
calibration and OOD references are not applied to the native model; native
live concurrency is serialized with native training updates.

The historical `/tagger/siglip2/load` and `/predict` endpoints also recognize
native shared-trainer FoveaTag metadata. Full unmerged and merged checkpoints
are self-contained; compact LoRA exports require their fingerprint-matched
scratch encoder. Native view configuration, seed, decode cap and precision
come from the artifact, not NaFlex patch settings or a downloaded processor.
Sibling vocabulary and calibration/OOD artifacts follow the native metadata
reference. Native OOD uses overview-plus-mixed-evidence features, not the MIL
classifier input. Model-only `best_f1` files do not require optimizer state.
Native ONNX and inference-manager merge operations are explicitly refused;
the trainer already produces native merged exports. Live API/UI deployment
remains pending.

Native shared training holds the handle's reentrant model lock from forward
through optimizer update, and for head expansion. Prediction, offload/restore
and attach/detach use the same lock. Validation acquires it per batch and
restores the prior mode, allowing prediction between batches without exposing
training to eval toggles or transient OOD hooks. Opt-in GPU concurrency tests
cover full/LoRA, FP16/BF16 FlashAttention and exact RAM/disk state restoration.
The frozen A/B/C runner's training numerics remain unchanged.

Native online resumes bind `_collection_runtime.json` to the selected model,
optimizer and runtime bundle. It retains dynamic/co-occurrence/query pools and
recency, same-epoch collection quotas, query-expansion caps, resolver timing,
unavailable targets, augmentation-added vocabulary and pending/deduplicated
head-growth proposals. Native resume selects this bundle before rebuilding
online state, ignores common legacy snapshots and starts the collector after
vocabulary restoration. Pool entries saved just before a worker proposal are
reconciled against the restored vocabulary. Epoch-boundary quotas belong to the
completed model epoch and are cleared when entering the next epoch.
Snapshot failures are fatal; missing/invalid bound collection state cannot
silently restart collection. In-flight requests, prefetched samples and network
arrival order remain outside the exact offline-replay guarantee.
Native online collection applies the source pixel cap before decoding; the
processor also rejects oversized inputs as a backstop.
For native runs, `save_best_only` suppresses periodic step/epoch archives while
retaining `best_f1`, the latest resume bundle and explicit stop checkpoints.

Native model-only publications use `_inference_commit.json` after weights,
metadata, vocabulary and selected optional artifacts finish saving. This also
covers `best_f1` without optimizer state. Training details expose completed
latest/best publications as `inference_checkpoint_paths`, distinguishing
full-unmerged, compact LoRA and merged files and showing their vocabulary path.
UI polling checks file generation stamps; inference loading verifies hashes.
Partial or overwritten publications are not displayed, and disabled stale
optional files are not included. Step retention removes the publication marker.

OOD reservoirs now retain their owned sampling RNG and capacity. Native restore
is strict; legacy reservoir files without RNG remain readable in the legacy
path.

Native checkpoints save exact current/previous/previous-previous histograms and
exposure counters in `_metrics_runtime.npz`, separately from inference metrics.
Epoch-boundary snapshots represent the state after rotation without mutating the
live accumulator. Rolling F1 buffers and learned thresholds use `_f1_runtime.npz`;
native step saves occur after periodic F1/threshold updates. Runtime sidecars are
atomic, hash-bound to resume bundles and included in retention cleanup.
Exact metrics are persisted whenever saving metrics, rolling F1 or exposure
tracking requires them. Masked labels do not count as observed negatives in
native histograms, rolling F1 or validation. Vocabulary reorders and one-source
renames align rows; new or multi-source-merged rows restart because their joint
observations cannot be reconstructed from marginal histograms.

CPU contract tests live under `backend/tests/` for response selection, head
growth, optimizer migration, validation collection, browser sidecars, and
checkpoint artifacts. Real model throughput, VRAM, FlashAttention behavior, and
GPU pause/offload behavior require an environment with the corresponding model
weights and are verified separately.
