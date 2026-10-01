# Ming-Image 0.1 training data contract

The native dataset reader is CPU-verified. Normal training remains unavailable
until shared caches, the optimization loop and exact optimizer resume are
connected. Shared dataset discovery and epoch refresh now attach these records.
This format supplies supervision; it does not invent ground-truth
layers from a flat image or promise quality recovery.

## Profiles and ordered targets

Design (`generation_edit`) accepts an ordinary captioned RGB/RGBA target. RGB
is promoted to opaque RGBA. Optional edit references must share the source
canvas geometry. A sidecar can specify them explicitly.

Layer (`layer_decompose`) requires `canvas.png.ming.json` beside its primary
image. All paths resolve relative to that sidecar and must stay within the
explicit dataset root, including through symlinks. Assets have the same
EXIF-corrected source geometry. Layer targets must have explicit RGBA channels.

```json
{
  "schema_version": 1,
  "profile": "layer_decompose",
  "targets": [
    {"path": "canvas.png", "role": "auxiliary_canvas", "weight": 0},
    {"path": "foreground.png", "role": "layer_1", "weight": 1},
    {"path": "background.png", "role": "layer_2", "weight": 1}
  ],
  "references": ["canvas.png"]
}
```

The auxiliary canvas occupies target slot zero because the native model emits
that slot. Its label and loss weight are dataset-owned; it is not assumed to be
a composited image. Every target requires an explicit finite, nonnegative
float32-representable weight, and the item must retain a positive finite total
in float32. Reference images condition the model and never enter the target
loss. The roles must be exactly `auxiliary_canvas`, `layer_1`, ... in order;
Design's sole target role is `image`. Duplicate JSON keys and unknown fields
are refused.

## Paired transforms and batch identity

`load_pixels` applies one original-canvas `(x, y, width, height)` crop, bilinear
resize and optional horizontal flip to every target and reference. The output
pixel grid is divisible by 16. All four channels are normalized to [-1, 1].
The pixel layout is `[4, F, H, W]`; each target is VAE-encoded independently so
Layer's ordered image axis cannot be temporally compressed.

Batches are keyed by profile, output width/height and target-frame count.
Reference counts may differ per item; their native ragged representation is
preserved. The same count requirement applies to target-frame loss weights.

Preflight hashes the sidecar and every referenced asset. Loading rechecks root
containment and rejects content drift, requiring records/caches to be rebuilt.
The pixel cache identity includes ordered content hashes, sidecar identity,
roles, source geometry, crop, flip and output grid. It is not a complete text or
latent cache identity: conditioning caches additionally need caption, model,
processor, topology and trainability identities before reuse.

## Evidence and remaining integration

23 dedicated CPU tests cover matching transforms and alpha, invalid supervision,
path escape, content drift, distinct frame-count buckets and both profiles
through the real tiny native VAE, MLLM conditioner, registered training handler,
backward and AdamW updates. The shared `BaseTrainer.encode_image` also preserves
Ming alpha. CUDA initialization is prohibited in these tests. This evidence
does not claim that a normal training request or optimizer resume works yet.

### Shared discovery and resume identity

The train-runner dataset wrapper preserves the catalog's explicit dataset root.
At setup and after epoch reload, the shared trainer attaches native records.
Layer target/reference assets without their own primary sidecar are excluded
from sample discovery; an unrelated loose image is refused. A primary with its
own sidecar remains a separate sample even if another sample references it.
Validation completes across all datasets before item lists are changed.

Bucket metadata carries the fresh record, frame signature and supervision
identity with each epoch's caption. Planned boundaries and repeated priority
occurrences are preserved when incompatible frame/grid groups are split.
Concept ordering groups native frame signatures before creating its plan,
allowing ragged reference counts. Shared dataset fingerprints, concept digests
and checkpoint batch ledgers additionally identify ordered source content,
roles and loss weights. The existing ledger resolver refuses changed Ming
supervision. This is data-cursor evidence, not proof of optimizer/RNG/scaler
resume or a functioning normal training request.

The discovery/epoch/batch additions pass 37 related CPU checks; after supervision
identity was wired, 15 affected concept/resume checks passed. Full backward
coverage from the native data reader remains applicable. Other architectures'
manifest/digest formats are unchanged when no Ming metadata is present.


### Shared execution transport

After native preparation, `shared_step_inputs` packages the two conditioning
streams into one tensor plus explicit `(query_length, query_width, direct_width)`
layout and per-frame weights. The shared loop's auxiliary dict is a typed
transport envelope. Reference latents stay in the shared per-item reference list,
so microchunk slicing retains ragged lists including absent references. Native
dispatch validates the envelope and restores `MingTrainingBatch` before the loss.

The transport tensor keeps both encoder output graphs reachable through the
common two-stage microbatch replay. Native streams and their gradients are
preserved while unused feature columns are padded with zero. CPU tests compare
all model/conditioner gradients with an unsplit batch under the same noise,
sigmas and CFG-drop mask. This execution bridge is connected to shared backward;
normal item assembly/cache and complete resume remain pending.


### Common MNT preparation

`prepare_shared_latents` reads the record's paired pixels at each selected bucket
and applies the crop planner's explicit box, or the shared crop/resize/random
strategy. Random crop uses the common Python RNG stream. All references and
RGBA targets use the same selected window. Native VAE work happens once before
MNT; `MingLatentBatch.for_step` owns fresh trainable conditioning or batch-local
frozen reuse. This avoids repeated VAE work and avoids retaining a consumed
trainable encoder graph across optimizer updates. Common assembly and MNT call
these seams directly. Persistent cache, full normal-run verification and exact
optimizer resume still remain separate pending work.


### Persistent record reuse

`prepare_record_latents` can consume a `snapshot_cache_context` captured while
frozen producer weights are on CPU. Paired pixel hashes identify targets and
references together. Expected latent roles/shapes derive from current records;
expected conditioning shapes derive from the native token-layout helper and
projection dimensions. A mismatch or corrupt entry is regenerated. Caption
changes select a new conditioning entry without invalidating the image latents.
Cached references retain absent per-item slots and ordered native frames.

Only a cache miss stages its encoder on the requested execution device. VAE
cleanup returns to CPU even if transfer/encoding fails. Frozen conditioner
cleanup returns to CPU; trainable conditioning remains resident for backward.
A cache context rejects producer parameter-version or precision changes. Its
execution descriptor must describe the actual attention, tiling and runtime
policy. Common mode selection/eager prefill remains pending; default training
still follows the staged on-the-fly path until that connection is complete.
