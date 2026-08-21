# Add a Model Architecture

SushiUI currently supports 14 architectures: 10 image (SD1.5, SDXL, Z-Image,
Flux2, Anima, Lens, Krea2, Ideogram4, MiniT2I, SenseNova U1.5), 2 video that
also generate audio jointly (LTX-2.3, MiniMax-H3) and 2 audio (ACE-Step 1.5,
MiniMax Music 3). `ModelType` in `backend/core/model_loader.py` is the
authoritative *generation* list — check it rather than this sentence if the
two ever disagree. `ARCH_REGISTRY` in `backend/core/training/arch/__init__.py`
is the authoritative *training-capable* list; it has 12 entries because
MiniMax Music 3's training is out of scope (design forward-compatible, not
implemented — see `docs/guides/MINIMAX_MUSIC3_DESIGN.md`'s "Training
forward-compatibility" section) and SenseNova U1.5's training is a deliberately
separate future phase (its base checkpoint is converted unmerged from the
8-step distillation LoRA specifically to keep that lineage canonical — see the
`sensenova` row in `docs/guides/MODEL_FACTS.md`). Do not assume the two lists
are always the same size — they diverge further with each generation-only
architecture added.

This is the procedure for adding the next one. Sections 1-8 are the common
surface; **section 9 is the additional surface a video (or audio) architecture
needs on top of it**. Each step points at the existing file(s) to mirror; read
the sibling implementation for the closest existing architecture before writing
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

## 6b. Component-switch catalog is opt-in per architecture

`backend/core/models/components/component_catalog.py` and
`POST /models/current/components/switch` let a user swap one component (text
encoder, VAE, ...) of an already-loaded model without a full reload, but only
for an architecture that has a per-architecture **unload-first adapter**
registered for that slot — a new architecture is not automatically wired into
this surface just because it has a load-time component-selection mechanism
(e.g. `text_encoder_file` on `POST /models/load`). Without an adapter, the
catalog's generic `{slot}_origin` reporting still works (it reads back what
the loader recorded, e.g. `selected_external` for an override load), so
declining to build the adapter does not break anything — it just means the
architecture's components can only be chosen at load time, not hot-swapped
after. Decide this deliberately per architecture rather than assuming parity
with MiniMax-H3, which does have the adapter for its `te_override`.

## 7. Document the new architecture

Add a row for the new architecture to `docs/guides/MODEL_FACTS.md` (a
per-architecture facts reference maintained alongside this guide set) once
it exists, and extend `backend/core/training/MODEL_ARCHITECTURES.md` with
the training-adapter details from step 4. Also update the architecture count in
this file, `AGENTS.md` and `docs/guides/ARCHITECTURE_MAP.md`, and — for a video
or audio architecture — check that `docs/guides/REQUEST_LIFECYCLE.md` still
describes its request path correctly.

Measured numbers belong in the MODEL_FACTS row with the conditions they were
measured under. Unmeasured or subjective performance language does not belong
anywhere user-visible; a feature whose registered bar was missed is recorded as
missed, with the numbers.

## 8. Wire up generation/training parameters

If the new architecture introduces new generation or training parameters
(rather than reusing existing ones), follow
`docs/guides/ADD_A_PARAMETER.md` for generation parameters, or the
equivalent `TRAINING_DEFAULTS`/adapter-config path for training parameters —
`backend/api/param_defaults.py` is still the single source of truth for any
new default value.

## 9. If it is a video architecture: the additional surface

A video architecture reuses everything above and adds the following. The two
existing video archs are the references, and they differ deliberately: LTX-2.3
drives stock diffusers pipelines, MiniMax-H3 vendors its model classes and owns
its denoise loop (upstream ships a Modular pipeline only). There are now two
audio architectures, and they differ the same way: ACE-Step was built by
mirroring LTX-2.3 (stock-pipeline-shaped), while MiniMax Music 3 vendors its
model classes and owns its own multi-stage loop for the same reason MiniMax-H3
does — upstream ships no usable `DiffusionPipeline` (see
`docs/guides/MINIMAX_MUSIC3_DESIGN.md`'s "Dependency gate" section). Diffing
LTX-2.3 against MiniMax-H3, or ACE-Step against MiniMax Music 3, shows exactly
what the per-arch surface is; do not assume every architecture in a modality
looks like the first one you read.

**Detection and routing.** Video models are loaded and dispatched separately
from image-model detection: `pipeline.py` sets `is_<arch>_model` and the load
result carries `is_video: True`, `latent_channels` and the VAE scale factors.
The image endpoints must refuse the model with a reason — extend
`_reject_if_video_model` / `_reject_if_audio_model` in `backend/api/routes.py`
rather than letting an image route fail somewhere deeper.

**`ComponentWiringSpec` handles 5-D latents already.** Set `latent_ndim=5` (and
`latent_ndim=3` for a waveform latent); nothing about the spec needs extending.
`vae_scale_factor` means the **VAE's own** spatial compression — if the
transformer additionally patchifies, that belongs in `pixel_align`, not here.

**`TemporalSpec` is where every clip-length rule lives**
(`backend/core/models/components/wiring.py`). Declare `frame_multiple`,
`frame_offset`, the production `min_frames`/`max_frames`, the hard
`min_decodable_frames` VAE floor, the closed-form `latent_frames`, `fps_fixed`,
`default_clip_lengths`, `max_pixel_hw` and whether an invalid length snaps or
400s. `max_frames` is the ENFORCED ceiling and may be `None` (no enforced
top at all — MiniMax-H3, as of the 362 decision: RoPE is computed on the fly,
so nothing structural stops a longer clip, only the `frame_multiple`/
`frame_offset` grid is structural); a separate `trained_max_frames` field
carries an ADVISORY documented-range top instead, past which a request is
accepted and warned as untested rather than refused or clamped. Do not
conflate the two: a real per-arch production ceiling belongs in `max_frames`,
a merely-documented-not-measured one belongs in `trained_max_frames`. Keeping
the two floors (`min_frames` vs. `min_decodable_frames`) separate matters too:
a training clip shorter than the production generation floor is not a
violation, but nothing may go below the decodable floor. One table is then
read by route validation, bucketing, the video loader, the clip-cache key, and
the `video_constraints` block of `GET /schema/arch-capabilities` — so the
frontend builds its clip-length list from the backend's own rule and no shared
file grows an `if arch ==`.

**Per-arch generation defaults, not route special cases.** Add a
`VIDEO_GEN_ARCH_OVERLAYS[arch]` entry in `backend/api/param_defaults.py`; the
routes resolve omitted fields through `video_defaults_for_arch(loaded_arch)`
(JSON bodies use `model_fields_set`, multipart routes use `Form(None)`
sentinels). Pass those **resolved** defaults to `check_arch_capabilities`, or
every video-only key will read as user-set and warn on every request. Audio has
the identical pattern: `AUDIO_GEN_ARCH_OVERLAYS`/`audio_defaults_for_arch`, and
the outpaint/repaint twins (`OUTPAINT_AUDIO_ARCH_OVERLAYS`/
`outpaint_audio_defaults_for_arch`, and the repaint equivalent), all in
`backend/api/param_defaults.py`. Before MiniMax Music 3, audio had exactly one
architecture (ACE-Step) and no overlay mechanism at all — the per-arch overlay
had to be introduced as a **prerequisite**, not added alongside the second
architecture's own defaults, because a flat single-architecture default dict
cannot express "the same field means something different, or does not exist,
on the next audio architecture" (MiniMax Music 3's `flow_guidance_scale`,
`num_inference_steps`-per-chunk and required `lyrics` have no ACE-Step
analogue). If you are the second architecture in any modality that only ever
had one before, check for this trap before writing the new defaults.

**Capabilities, honestly.** Declare what the architecture genuinely cannot do
(`arch_capabilities.py`) with a factual reason per feature, and use
`TRAINING_UNSUPPORTED` for a training method it cannot offer. The accept-and-warn
convention applies: an unsupported generation parameter is warned about, not
400'd, and only when it is set to a non-default value.

**Conduit registration is optional.** LTX-2.3 bypasses the attention conduit
entirely and declares `attention_impl` unsupported; MiniMax-H3 routes through it.
Register only if the architecture's attention call is yours to make.

**Acceleration is per-arch, and each feature must earn its place by
measurement.** Block swap needs a block-loop wrapper
(`models/<arch>_block_loop_wrapper.py`, over the shared
`TransformerBlockOffloader`) plus an `_ensure_<arch>_swap_and_offload`; FBCache,
Spectrum, TREAD and BlockSkip each attach to that wrapper. Register the pass/fail
bar before you measure, and if a feature misses it, remove the code rather than
ship it disabled — MiniMax-H3's FBCache is the worked example.

**Training.** Beyond the adapter of step 4 you need `training/arch/<arch>.py`
(the `ArchHandler`, carrying `wiring`, `pixel_align` and `temporal`),
`training/ops/<arch>_ops.py` (the real work: component loading, prompt encode,
`vae_encode_clip`, `train_step`, sampling), registration in `ARCH_REGISTRY` +
`_EXPECTED_ARCH_KEYS` + `resolve_arch_name` **in the same priority order as
`base_trainer._build_cache_namespace`** (a module-level assert enforces the key
set; the ordering is the cache-namespace stability invariant), and the
`is_<arch>` branches in `base_trainer.py`. Video adds `vae_encode_clip` to the
handler interface: a `[T,C,H,W]` pixel clip in the shared loader's convention →
a normalized 5-D latent. If the architecture's pixel or tiling convention differs
from the shared loader's, the arch owns the conversion — and if the tiling policy
changes the *output* rather than just peak memory, it must be part of
`LatentCache.compute_clip_hash` or cached latents will silently disagree with
what generation produces.

**Reuse, do not re-derive**: `utils/video_utils.py` (encode/mux/probe/window
extraction), `utils/dataset_scanner.py` (`VIDEO_EXTS` + ffprobe →
`DatasetItem.item_type`/`video_meta`), `training/video_augment.py`
(temporally-consistent augmentation), `training/latent_cache.py`,
`training/video_loader.py` and `training/bucketing.py`'s temporal section — all
of which take an explicit `TemporalSpec` and fall back to the LTX-2.3 rule when
it is absent. The frontend needs **no new panel**: video is a mode inside the
existing txt2img/img2img panels, plus the temporal side of the Outpaint and
Inpaint tabs (temporal inpaint regenerates a latent-group-aligned span and
pastes the rest back exact; its own range control, not the keyframe timeline,
reads its granularity off `video_constraints.latent_chunk_pattern`). Gallery
send-to and the generation queue's item-type accounting need a matching
branch for each new video route, or a queued or gallery-originated request for
it silently drops.

**`backend/tests/quantized_capability_parity_test.py` fails until the
quantization registries are wired**, by design. That is the forcing function for
`ARCH_QUANT_POLICY` / `QUANTIZED_LINEAR_ARCHS` / `RUNTIME_INT8_ARCHS` /
`EXPORT_LAYOUTS` and for using `is_lora_wrappable_linear` instead of
`isinstance(x, nn.Linear)` in the LoRA target predicate — the latter silently
drops every quantized target and has been found on four architectures in this
repo already.

**Owning real quantized-Linear modules is not the same claim as being in
`RUNTIME_INT8_ARCHS`/`QUANTIZED_LINEAR_ARCHS`.** Those tables advertise a
`quantized_gemm_mode`/`unet_quantization` **runtime toggle** reachable from a
generation request. An architecture can have working packed-weight Linear
classes (MiniMax Music 3 has both `GGUFQ8_0Linear` and a reused
`ConvRotInt8Linear` for its Q8_0 and INT8 ConvRot checkpoint formats) while
still correctly staying out of both tables, if the only path that builds those
modules is a load-time checkpoint-format choice with no live toggle behind it —
adding the table entry would advertise a capability the wiring does not
actually expose. Add the entry only when a real runtime converter reaches that
architecture's modules; do not add it reflexively because the class exists.

**Detection and refusal ordering matters as much as the checks themselves,**
learned from a chain of load-path defects on MiniMax Music 3's
`text_encoder_file` selection (design doc, "Current status", the two
independent-audit paragraphs): (1) any detection/refusal that a generic
teardown-then-load flow can reach must run in a **header-only preflight**,
before the live model is torn down — a check that only runs inside the new
architecture's own loader fires too late, after a bad request has already
cost the user their loaded model (this repo's existing hybrid-model preflight
in `_load_model_locked` is the pattern to extend, not re-derive). (2) a
detector shared or adjacent to another architecture's existing detector needs
an explicit architecture (or format) gate — a check written against one
architecture's file shapes will misfire on a different architecture's model
being loaded unless it is gated to fire only for the architecture it was
written for. (3) a detector that reports "any unrecognized key" rather than
"any of these specific signature keys" must be fed the same key set its real
caller uses (e.g. excluding a container format's own non-tensor metadata
entries) — a synthetic test fixture that happens not to carry that extra key
will not catch the gap.

## Verification

- `py_compile` and a real import
  (`venv/Scripts/python.exe -c "import backend.core.pipeline_backends.<arch>"`)
  on every new/changed backend file.
- Round-trip a saved single-file checkpoint through the reader in
  `single_file_format.py` to confirm the index/shard scheme loads back
  correctly for a large (>10 GB) model.
- Confirm `detect_model_type` returns the correct `ModelType` for a real
  checkpoint of the new architecture before wiring up the rest of the stack.
