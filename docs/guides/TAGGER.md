# Tagger subsystem

SushiUI has two independent tagger stacks. The legacy WD-compatible stack is
inference-only. The SigLIP2 stack supports inference, dataset sidecar editing,
full-parameter or LoRA training, checkpoint analysis, and ONNX export. Do not
share checkpoint assumptions between them.

## Ownership

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

CPU contract tests live under `backend/tests/` for response selection, head
growth, optimizer migration, validation collection, browser sidecars, and
checkpoint artifacts. Real model throughput, VRAM, FlashAttention behavior, and
GPU pause/offload behavior require an environment with the corresponding model
weights and are verified separately.
