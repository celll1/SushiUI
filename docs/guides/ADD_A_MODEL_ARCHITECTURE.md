# Add a Model Architecture

SushiUI currently supports 9 architectures (SD1.5, SDXL, Z-Image, Flux2,
Anima, Lens, Krea2, Ideogram4, MiniT2I). This is the procedure for adding a
10th. Each step below points at the existing file(s) to mirror; read the
sibling implementation for the closest existing architecture before writing
new code.

## 1. Detection — `backend/core/model_loader.py`

`ModelLoader.detect_model_type` is the single entry point that inspects a
model path (directory, single safetensors file, or sushiUI shard index) and
returns a `ModelType` string. For a single-file checkpoint it dispatches to
per-architecture signature heuristics — see `_keys_look_krea2`,
`_keys_look_lens`, `_keys_look_ideogram4`, `_keys_look_anima` — which check
tensor key names/shapes and embedded metadata, not file extensions or paths.
Add a new `_keys_look_<arch>(keys, metadata)` helper and a branch in
`detect_model_type` that calls it. For a sushiUI-shard-index checkpoint,
detection is metadata-first (`model_type` field), falling back to a
weight-map key probe — extend the same fallback logic.

## 2. Pipeline backend — `backend/core/pipeline_backends/<arch>.py`

Add a new module following the shape of the existing per-architecture files
(`zimage.py`, `flux2.py`, `anima.py`, `lens.py`, `krea2.py`, `ideogram4.py`,
`minit2i.py`). This is a mixin consumed by `backend/core/pipeline.py`'s
`PipelineManager` — implement the architecture-specific generation logic
(text encoding, conditioning construction, denoising loop entry point) here
rather than growing `pipeline.py` itself.

## 3. Attention backend registration — one entry, not a conduit change

If the new architecture needs no new attention kernel, it uses the existing
conduit unmodified. If it needs a new kernel, adding it is a **one-entry
change**: one `AttentionBackend` descriptor in
`backend/core/attention/registry.py` plus one callable in
`backend/core/attention/backends.py` — see the module docstring at the top
of `registry.py`, which documents this explicitly. Do not add
architecture-specific branches to `dispatch.py`.

## 4. Training adapter — `backend/core/training/adapters/`

Add `<arch>_adapter.py` alongside the existing adapters (`sd15_adapter.py`,
`sdxl_adapter.py`, `zimage_adapter.py`, `flux2_adapter.py`, `anima_adapter.py`,
`lens_adapter.py`, `krea2_adapter.py`, `ideogram4_adapter.py`,
`minit2i_adapter.py`), following `base_adapter.py`'s interface. See
`backend/core/training/adapters/MODEL_ADAPTER_DESIGN.md` for the adapter
pattern and `backend/core/training/MODEL_ARCHITECTURES.md` for what an
adapter must supply per architecture (text encoding shape, conditioning
kwargs, time-ids/pooled-embedding handling, LoRA target modules).

## 5. Single-file format — `backend/core/models/common/single_file_format.py`

If the architecture should support save/load as a sushiUI single-file
checkpoint, use this module's writer/reader rather than a bespoke format:

- Saves are automatically sharded once total tensor size exceeds
  `DEFAULT_MAX_SHARD_BYTES` (10 GB), producing
  `<stem>-00001-of-000NN.safetensors` shards plus a
  `<stem>.safetensors.index.json` index (`weight_map` maps each tensor key
  to its shard file). This index file is the selectable entry point for
  loading — pass its path, not an individual shard, to the loader.
- For non-sharded (small) checkpoints, a single `.safetensors` file with an
  embedded `model_type` metadata field is enough for detection.

## 6. Completion-from-siblings pattern

Some architectures store components (e.g. text encoder, VAE) as separate
sibling files/directories rather than one monolithic checkpoint. The loader
supports completing a partial checkpoint by locating and merging in these
sibling components at load time rather than requiring the user to point at
one giant file — follow the existing completion logic in
`backend/core/model_loader.py` for the closest existing architecture (Krea2,
Ideogram4, and Lens all use single-file loading; check their loader paths
for the sibling-completion pattern before writing a new one).

## 7. Document the new architecture

Add a row for the new architecture to `docs/guides/MODEL_FACTS.md` (a
per-architecture facts reference maintained alongside this guide set) once
it exists, and extend `backend/core/training/MODEL_ARCHITECTURES.md` with
the training-adapter details from step 4.

## 8. Wire up generation/training parameters

If the new architecture introduces new generation or training parameters
(rather than reusing existing ones), follow
`docs/guides/ADD_A_PARAMETER.md` for generation parameters, or the
equivalent `TRAINING_DEFAULTS`/adapter-config path for training parameters —
`backend/api/param_defaults.py` is still the single source of truth for any
new default value.

## Verification

- `py_compile` and a real import
  (`venv/Scripts/python.exe -c "import backend.core.pipeline_backends.<arch>"`)
  on every new/changed backend file.
- Round-trip a saved single-file checkpoint through the reader in
  `single_file_format.py` to confirm the index/shard scheme loads back
  correctly for a large (>10 GB) model.
- Confirm `detect_model_type` returns the correct `ModelType` for a real
  checkpoint of the new architecture before wiring up the rest of the stack.
