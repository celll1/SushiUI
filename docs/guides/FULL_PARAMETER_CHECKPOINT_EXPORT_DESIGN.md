# Full-parameter checkpoint and quantized export design

Status: Qwen-Image 2.1 prototype implemented. The architecture-neutral
`training_export_format` control, stopped-run API, and Training Monitor action
are wired; other architectures remain capability-gated. Real-model generation,
peak-memory, and disk-headroom acceptance gates remain open.

## Why `output_dtype` is not the format selector

The training UI currently offers FP32, FP16, and BF16 for `output_dtype`.
`BaseTrainer` parses the value, but the Qwen-Image 2.1 full-parameter writer
serializes its live `state_dict()` without casting. Several other full-parameter
writers do likewise. The field therefore cannot be assumed to describe the
actual saved weight dtype on every architecture. Audit and correct that
existing behavior separately; do not add `int8_convrot` to this dtype field.

ConvRot is a *format*: eligible Linear weights become rotated INT8 tensors
with per-row scales and markers; ineligible tensors retain a floating dtype.
Its completeness, kernel availability, metadata, and loader rules are
architecture-specific. A generic scalar-to-`torch.dtype` conversion cannot
produce a valid ConvRot artifact.

## Two artifacts with different guarantees

| Artifact | Purpose | Required contract |
|---|---|---|
| Training checkpoint | Exact continuation | Save all trainable weights without lossy quantization, plus optimizer/scheduler/RNG and component identity. A supported floating save dtype may be chosen only if it preserves the run's declared resume contract; no hidden downcast. |
| Inference export | Generation/distribution | Optional architecture-supported ConvRot conversion of a completed checkpoint. It may be lossy, and has its own manifest, format metadata, and generation validation. It is not an optimizer checkpoint. |

`latest` and automatic fallback select only complete, resumable training
checkpoints. Export files live in a distinct namespace and cannot displace the
last intact optimizer-bearing checkpoint through retention or cleanup. An
export failure must not invalidate the training checkpoint. Export should
stream components/layers instead of holding dense and quantized full-model
copies in GPU memory.

### Export triggers and disk bound

There is **no periodic INT8 export**. It may start only when:

* a run reaches `completed` normally and its configured export format is
  `int8_convrot`; or
* the owner explicitly clicks Export in Training Monitor for a `stopped` run,
  after its process has exited and a complete floating checkpoint is present.

Neither a save interval, a stop request still in progress, nor a `failed` run
triggers export. A completed run may offer an explicit retry if its automatic
export failed. The export job reads the selected complete floating checkpoint,
never partially saved live weights. Recheck run state and checkpoint identity
under a lock before starting, and prevent concurrent resume, checkpoint
retention, or another export of that run until the source snapshot is secured.

Keep at most one published export per run and format by default, with a stable
path and a recorded source step/checkpoint digest. Repeating the same request
is idempotent. Replacing an older export requires an explicit overwrite choice
in the UI; publish a new file/manifest atomically before removing the old one.
Preflight the temporary disk peak, which can briefly include both exports;
failure leaves the old export and the resumable checkpoint intact. Do not
prune training checkpoints just to make room for an export.

An INT8-only artifact cannot give an **exact resume** of floating-point
training: dequantization does not restore the pre-quantized weight, and a saved
optimizer moment from that weight is no longer paired with the same parameter.
If no floating checkpoint remains, an explicit dequantization starts a **new
training lineage** with fresh optimizer and scheduler state (and the configured
warmup), not a resumed step. The UI/API must label these actions differently.
Do not silently dequantize an export when `resume_from_checkpoint=latest` is
requested, or reset the optimizer while reporting a successful resume.

## API and capability shape

Keep `output_dtype` for floating checkpoint dtype after its real behavior is
audited. Add a distinct, architecture-neutral export-format control, e.g.
`training_export_format: none | int8_convrot`. This controls automatic export
on normal completion; a stopped run can request the format through a dedicated
run-scoped export API and Training Monitor action. Do not add an export cadence
control. Accept `int8_convrot` only where the architecture
has a converter, strict component census, validated loader, and generation
round trip; otherwise reject before starting the export. The architecture
adapter, not the common API, owns tensor naming and which components are
quantized. An output may contain floating non-Linear tensors without becoming
mislabelled as wholly INT8.

Do not change any architecture's existing save-format setting implicitly.
SenseNova's branch-specific `sensenova_full_finetune_save_format` has its own
resume contract and needs a deliberate migration before the common control
could replace it. Qwen's `training_bundle` checkpoint contains the trained TE
when TE training is enabled; export quantizes that TE, never the frozen
companion TE. A mixed Original-DiT/ConvRot-TE training
manifest is an input composition, not a third distributed weight format.

## Acceptance gates

1. Verify the actual floating dtype of each full-parameter writer against
   `output_dtype`; either honor supported choices or refuse unsupported ones.
2. Resume from a retained floating checkpoint with exact weights, optimizer,
   scheduler, and RNG even after one or more exports and retention passes.
3. Validate every quantized component's key/shape/marker census and compare
   fixed-seed generation against its floating source; record quality and speed.
4. Refuse an export as `resume_from_checkpoint` with a clear new-lineage
   instruction. Verify explicit INT8-to-floating initialization starts at step
   zero with a fresh optimizer and configured warmup.
5. Measure conversion time, peak VRAM, host RAM, and disk use, including
   interruptions or failed writes. Publish an export atomically only after
   all components and metadata pass validation.
6. Prove that periodic saves, in-flight stops, and failed runs create no INT8
   export; normal completion and an explicit stopped-run request do. Repeated
   exports cannot accumulate one copy per checkpoint, and an interrupted
   replacement preserves the prior published export.
