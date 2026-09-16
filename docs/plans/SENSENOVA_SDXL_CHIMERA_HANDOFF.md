# SenseNova SDXL Chimera implementation handoff

Status date: 2026-09-17

## Continuation status

The implementation handoff has now been carried through P4 and the independently
shippable text-output and source-image editing parts of P7. Subsequent commits are:

- `af93edf9 Load Chimera production donor sources`
- `e3a959be Load Chimera understanding branch selectively`
- `58397fee Register Chimera model artifacts`
- `70afa236 Add Chimera txt2img inference`
- `2931f621 Train Chimera bridge and U-Net stages`
- `5907b359 Preserve Chimera understanding text output`
- `1a7f6153 Add Chimera source-image editing`
- `75469d79 Document Chimera architecture`

The shipped surface now includes path-based scratch/transplant construction,
understanding-only loading, model/API registration, deterministic txt2img,
stage-exact `bridge_align` / `unet` / `joint` full training, production-loadable
directory checkpoints with resume/rotation support, OpenAPI/default/frontend
training controls, a Settings initializer, i2t/ti2t routed through the frozen
understanding branch, and deterministic SDEdit/RePaint for img2img, inpaint,
and the shared spatial-outpaint orchestration. Inpaint pins the preserve region
at every flow step and composites the original pixels after decode. The
combined Chimera P0-P4/P7, source-loader, capability, and CFG regression run is
90 passing tests, including zero/full-strength SDEdit and exact outpaint
placement preservation. The frontend model union, capability-driven img2txt tab and
queue gate preserve the Chimera architecture id rather than coercing it to
`sensenova`; the repository owner still owns the frontend type-check/build.

The machine-local P5 artifact is still intentionally absent. This file's source
choice guard remains in force: no SDXL donor was selected by the owner, and an
unaligned bridge must not be promoted to `bridge_state="aligned"` without the
pre-registered held-out thresholds required by the design. Reference-ti2i also
remains unadvertised until its image-conditioned bridge/joint quality gate
passes; it is independent of the now-wired source-image editing routes.

This note was the original restart point for implementing
`sensenova_sdxl_chimera`; the continuation status above supersedes its original
implementation-state statements while preserving the source-choice history.

## User goal and fixed decisions

- Keep the existing SenseNova U1.5 understanding branch as a frozen,
  hash-pinned external reference.
- Replace the SenseNova generation branch with an SDXL-shaped U-Net operating
  in the selected SDXL model's four-channel latent space.
- Keep the U-Net's trainable parameter count and tensor census exactly equal to
  the selected SDXL donor U-Net.
- Default to scratch initialization, with experimental strict SDXL U-Net
  transplant available.
- Extract and bundle the selected SDXL model's VAE, as in VAE swap.
- Use SenseNova-style three-axis `t:h:w = 2:1:1` RoPE rather than retaining
  SDXL's old positional treatment. Crop origin and size microconditioning are
  part of the contract.
- Build a full-size scratch bootstrap artifact at
  `M:/models/snu1.5_sdxl_chimera`, then exercise inference and training.
- Wire generation, training, model initialization, and status/capability
  reporting through backend and frontend.
- Preserve SenseNova multimodal functionality where structurally possible:
  i2t and ti2t/it2t use the frozen understanding-only path; reference-image
  ti2i uses the multimodal prefix and conditioning bridge. Reference-ti2i
  quality remains gated on image-conditioned bridge/joint training data.

The complete architecture contract is in
`docs/guides/SENSENOVA_SDXL_CHIMERA_DESIGN.md`. Section 7.6 records the latest
i2t/ti2t/reference-ti2i decision.

## Completed commits

1. `ff67abaa Design SenseNova SDXL Chimera`
   - Added the full design and linked it from the document map.
2. `f441e161 Build Chimera conditioning foundation`
   - Added flow algebra, donor-equal U-Net construction, the conditioning
     bridge/resampler, and parameter-free SenseNova three-axis RoPE.
3. `92b2d118 Add Chimera artifact contract`
   - Added format-v1 manifests/configs, source hashing, sharded weights,
     atomic publication, strict reload, bundled VAE verification, and P1 CPU
     round-trip tests.
   - Added the i2t/ti2t/reference-ti2i design boundary.

The worktree was clean immediately before this handoff file was added.

## Implemented modules

`backend/core/models/sensenova_sdxl_chimera/` currently contains:

- `flow.py`: `t=0` noise to `t=1` clean straight-path noising, velocity target,
  and Euler step.
- `positional.py`: parameter-free three-axis RoPE, `rotate_half`, float physical
  coordinates, one spatial unit per 32 output pixels, and crop-aware U-Net
  query positions.
- `conditioning_bridge.py`: relative 25/50/75/100% layer selection,
  layer-specific K/V projections with initially inert zero gates, learned-query
  resampling to `77 x 2048`, pooled `1280`, context-position barycenters, and
  position variance.
- `unet.py`: constructs `UNet2DConditionModel` from the real donor config;
  supports reproducible `scratch` and strict `sdxl_transplant`; verifies exact
  `(name, shape)` and parameter-count parity; zeroes only `conv_out` for scratch.
- `artifact.py`: format constants, canonical hashes, checkpoint/header reads,
  pinned SenseNova source validation, bridge-geometry inference, and manifest
  construction.
- `builder.py`: component-level builder and atomic target publication. It
  currently accepts an already-loaded donor U-Net and `ResolvedVAE`; it does
  not yet load production paths itself.
- `loader.py`: reconstructs U-Net, bridge, and VAE from an artifact, verifies
  manifest/config/tensor/VAE identity, and resolves the pinned understanding
  source before materializing artifact weights.

Important current boundary: `load_chimera_artifact(...,
load_understanding=False)` works. `load_understanding=True` intentionally
imports a not-yet-created `understanding.py` and therefore is not complete.

## Verification already run

From `backend/`, using the repository venv:

```powershell
..\venv\Scripts\python.exe -m pytest tests\sensenova_sdxl_chimera_p0_test.py tests\sensenova_sdxl_chimera_p1_test.py -q
```

Result: `11 passed`.

The tests cover:

- donor/new U-Net census and parameter-count equality;
- scratch reproducibility and strict transplant identity;
- bridge shapes, masks, gradients, and selected-layer rules;
- crop-aware physical coordinates and RoPE reference/invariance;
- flow endpoints;
- tiny sharded artifact build/reload with exact U-Net/VAE tensors;
- relocated identical SenseNova source acceptance and mutated source refusal;
- atomic build publication and non-empty-target refusal;
- damaged tensor-census refusal.

`py_compile` passed for all Chimera files and tests. A real import also passed:

```text
sensenova_sdxl_chimera 1 2.8.3
```

The final number is the venv's `flash_attn` version, confirming that the repo
venv—not global Python—was used.

## Local real-model facts observed

The configured external model root in `local/model_root.txt` is `M:/model`
(singular), while the requested Chimera output path is `M:/models/...`
(plural). Do not silently rewrite the requested output path.

SenseNova candidates:

```text
M:\model\sensenova\sensenova_int8_convrot.safetensors  18,872,370,216 bytes
M:\model\sensenova\sensenova_int8.safetensors          18,872,241,160 bytes
```

The directory contains the required tokenizer/config siblings. The ConvRot
checkpoint was the inspected primary candidate.

SDXL candidates:

```text
M:\model\sdxl\aiwv-03-beta1.fp16.safetensors
M:\model\sdxl\animagine-xl-4.0.safetensors
M:\model\sdxl\Illustrious-XL-v2.0.safetensors
```

`animagine-xl-4.0.safetensors` was inspected. It declares SDXL in ModelSpec and
uses the expected LDM prefixes:

```text
model.diffusion_model.*
first_stage_model.*
```

The donor choice among the three SDXL files has not been confirmed. Do not
build the multi-gigabyte final artifact until that choice is explicit or an
existing project convention unambiguously selects one.

The common `resolve_vae_source("model:<animagine>")` correctly finds the VAE
weights but refuses because the LDM single file carries no explicit latent
normalization config. The production source builder must load/infer the VAE
through diffusers' SDXL single-file conversion, then persist the resulting
config and actual extracted tensors. It must not substitute the generic LDM
default scaling factor.

## Permission/session issue

The user had enabled full access, but this session still reported
`workspace-write / restricted`. Consequently:

- invoking `venv\Scripts\python.exe` needed escalation because its interpreter
  target is under the user profile;
- reading `M:` also needed escalation;
- global Python must not be used: only the venv has the required FlashAttention
  build.

The user stopped the session because repeated approvals make the workflow
unusable. On resume, first confirm the new session actually exposes full
filesystem/process access. Do not attempt to install Python or rebuild the
venv. The venv is healthy.

## Remaining work, in dependency order

### P1b: production sources and understanding-only load

1. Add path-based SDXL source loading:
   - diffusers directory support;
   - LDM single-file support for the observed `model.diffusion_model.*` and
     `first_stage_model.*` layout;
   - extract only the donor U-Net/VAE where practical;
   - construct a `ResolvedVAE` from the actual converted VAE config/tensors;
   - expose a path-based atomic builder used by both API and CLI.
2. Implement `understanding.py`:
   - load only unsuffixed SenseNova language/vision/tokenizer weights;
   - do not materialize `_mot_gen`, `vision_model_mot_gen`, timestep embedder,
     flow head, or refiner;
   - handle the real plain-int8 and ConvRot-int8 source formats;
   - expose prompt/image prefix hidden states and selected per-layer K/V;
   - keep the existing img2txt generation surface usable for i2t/ti2t.
3. Add artifact preflight tests proving no live `_mot_gen` parameter and no
   generation tensor payload is read.
4. Reconsider full 18.9GB source hashing on every model load. The current code
   is correct but potentially slow. Any cache must remain content-safe rather
   than downgrading the pinned hash to filename/mtime identity.

### P1c: model/API registration

1. Add `sensenova_sdxl_chimera` to `ModelType` and metadata-first directory
   detection in `backend/core/model_loader.py`.
2. Route production loading before broad SDXL heuristics and preserve
   preflight-before-active-model-teardown behavior.
3. OpenAPI-first: add
   `POST /api/v1/models/sensenova-sdxl-chimera/initialize`, its schemas and
   scratch/transplant examples.
4. Put every endpoint default in `backend/api/param_defaults.py`.
5. Add the route implementation, strict relative `output_name` validation, and
   model-list/status reporting.
6. Add `examples/api/build_sensenova_sdxl_chimera.py`.

### P2: txt2img inference

- Implement SenseNova prefix capture and immediate bridge reduction.
- Implement a diffusers U-Net attention processor using SenseNova three-axis
  RoPE with no new parameters.
- Build generation-local post-RoPE K/V caches and clear them in `finally`.
- Implement sequential and batched CFG, flow Euler sampling, SDXL time ids,
  bundled VAE decode, deterministic seeding, and preview handling.
- Add a pipeline backend and `PipelineManager` dispatch.
- Keep reference-prefix metadata in cache keys even before reference quality is
  advertised.

### P3/P4: training

- Add `bridge_align`, `unet`, and `joint` stages.
- Require aligned bridge state before transplanted-U-Net diffusion training.
- Add architecture handler, ops, training adapter, registry/capability rows,
  save/resume, optimizer groups, and stage-specific freezing.
- Add OpenAPI/default/config/frontend fields for Chimera training.
- Preserve i2t/ti2t inference even if first-release Chimera training does not
  train text-output objectives.

### P5: real bootstrap and smoke

- Build scratch artifact at `M:/models/snu1.5_sdxl_chimera`.
- Optionally build transplanted sibling
  `snu1.5_sdxl_chimera_sdxl_init` after bridge alignment is available.
- Verify actual U-Net parameter equality, bundled VAE identity, finite
  random-weight inference, one training step, save/resume, and peak memory.

### P6/P7: frontend and additional modes

- Initializer UI, architecture unions, model status, generation/training
  panels, warnings, and queue persistence are implemented.
- The repository owner—not the agent—runs frontend build/type-check.
- Img2img, inpaint, outpaint, i2t, and ti2t are implemented. Reference-ti2i
  remains gated on image-conditioned bridge/joint training and quality evidence.

## Repository rules to retain on resume

- GPT-6 Astra must not use subagents in this repository.
- API changes are OpenAPI-first.
- API defaults live only in `backend/api/param_defaults.py`.
- Never start or stop backend/frontend directly.
- Always use `venv/Scripts/python.exe`.
- After every backend edit, run both `py_compile` and a real import; stub CUDA
  initialization when importing trainer stacks.
- Do not run frontend build/type-check.
- Continue making small verified stage commits with the required
  `Co-Authored-By: OpenAI Codex <codex@openai.com>` trailer.
