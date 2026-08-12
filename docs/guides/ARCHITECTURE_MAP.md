# Architecture Map

The backend drives **12 architectures**: 9 image (SD1.5, SDXL, Z-Image, Flux2,
Anima, Lens, Krea2, Ideogram4, MiniT2I), 2 video that also generate audio
jointly (LTX-2.3, MiniMax-H3) and 1 audio (ACE-Step 1.5). `ARCH_REGISTRY` in
`backend/core/training/arch/__init__.py` is the authoritative list (a
module-level assertion pins it against the trainer's cache-namespace keys);
`docs/guides/MODEL_FACTS.md` holds the per-architecture facts.

## Directory tree (top 2 levels + key `backend/core` modules)

```
webui_cl/
├── backend/
│   ├── api/            # FastAPI routes, WS protocol, param defaults
│   ├── core/            # pipeline orchestration, per-arch backends, training
│   │   ├── attention/       # unified attention conduit (backend-agnostic)
│   │   ├── inference/       # sampling loops, NAG, spectrum, schedulers
│   │   ├── models/common/   # single-file save/load format, VAE store
│   │   ├── pipeline_backends/ # one file per architecture (mixins)
│   │   ├── training/        # trainers, adapters, optimizers, losses
│   │   ├── model_loader.py
│   │   ├── pipeline.py
│   │   ├── keep_hot.py       # opt-in cross-generation GPU-resident component tracking
│   │   └── vram_optimization.py
│   ├── database/        # SQLAlchemy models
│   └── utils/            # image_utils.py (metadata), misc helpers
├── frontend/
│   └── src/
│       ├── app/           # Next.js App Router pages
│       ├── components/    # generation/, viewer/, training/, dataset(s)/, common/
│       ├── contexts/       # StartupContext, GenerationQueueContext, etc.
│       └── utils/          # api.ts (typed API client)
├── docs/                 # design docs, this guide set (docs/guides/)
├── examples/api/         # frontend-less API example scripts
├── models/, tagger_models/, taglist*/  # model weights and tag dictionaries
└── *.db                  # gallery.db, datasets.db, training.db (SQLite)
```

## Backend module responsibilities

| Module | Responsibility |
|---|---|
| `backend/core/pipeline.py` | `PipelineManager`: loads/holds the active pipeline, dispatches `generate_txt2img/img2img/inpaint` to the right per-architecture backend. |
| `backend/core/inference/custom_sampling.py` | The actual sampling loops (txt2img/img2img/inpaint), prompt chunking/editing, ControlNet, NAG, Advanced CFG. |
| `backend/core/inference/reference_style.py` | Arch-agnostic training-free reference-image style transfer (StyleAligned/VSP-style attention KV-injection): `StyleTransferConfig`, `inject_kv`, `cross_batch_adain_qk`, `make_ref_value`, `frequency_scale_vector`, `StyleContext`. Wired into SD1.5/SDXL (`attention_processors.py`), Krea2 (`models/krea2/vendor/transformer.py`), and FLUX.2 (`style_flux2.py`). |
| `backend/core/inference/style_flux2.py` | FLUX.2-specific style-transfer attention processors (dual-stream `transformer_blocks` + single-stream `single_transformer_blocks`); mutually exclusive with FLUX.2 Image-Edit `ref_images` and with NAG/NegPip. |
| `backend/core/model_loader.py` | Detects model type/architecture from a checkpoint (single-file signature heuristics, e.g. `_keys_look_krea2`), builds and returns the loaded pipeline. |
| `backend/core/attention/` | The attention conduit: `registry.py` (per-backend capability descriptors: native/flash/sage/tq), `dispatch.py` (routes a call to the resolved backend), `config.py` (capability-based downgrade rules), `backends.py` (kernel callables). Adding a backend is a one-entry change here — see `docs/guides/ADD_A_MODEL_ARCHITECTURE.md`. |
| `backend/core/pipeline_backends/` | One file per architecture (`zimage.py`, `flux2.py`, `anima.py`, `lens.py`, `krea2.py`, `ideogram4.py`, `minit2i.py`, plus the non-image `ltx2.py`, `minimax_h3.py`, `acestep.py`; SD1.5/SDXL are handled by the base `pipeline.py` path) — architecture-specific generation logic as mixins. |
| `backend/core/models/<arch>/` | Per-architecture component loaders and vendored model classes (e.g. `minimax_h3/loader.py` + `h3_pipeline_ops.py` + `h3_references.py` + `vendor/`, `ltx2/loader.py`, `acestep/`). A vendored architecture owns its denoise loop here when upstream ships no usable `DiffusionPipeline`. |
| `backend/core/models/<arch>_block_loop_wrapper.py` | Re-owns a transformer's block loop so block swap, gradient checkpointing and (where measured worthwhile) cache/forecast features can attach to it — `ltx2_block_loop_wrapper.py`, `minimax_h3_block_loop_wrapper.py`. |
| `backend/core/models/components/wiring.py` | `ComponentWiringSpec` (latent channels/ndim/packing, VAE scale factor and normalization, text-encoder output shape) and `TemporalSpec`/`TEMPORAL_SPECS` — the per-video-arch clip-length grid, frame-rate and canvas contract read by route validation, bucketing, the video loader, the clip-cache key and the capabilities payload. |
| `backend/core/keep_hot.py` | Arch-agnostic `keep_models_hot` state (model_key computation, VRAM guard, resident-set tracking); wired into `pipeline.py` (SD1.5/SDXL) and all 7 DiT image `pipeline_backends/*.py` files. Not wired into `ltx2.py`, `minimax_h3.py` or `acestep.py`. |
| `backend/core/training/` | `base_trainer.py` (shared loop, block-swap, optimizer wiring), `lora_trainer.py` / `full_parameter_trainer.py`, `adapters/` (per-architecture training adapters — text encoding, conditioning, time-ids), `optimizers/`, `losses/`, `bucketing.py`, `latent_cache.py`. |
| `backend/core/training/vae/` | Decoder-only VAE fine-tuning (`network.type: vae_decoder`), reached from `train_runner.py`. Standalone — does **not** subclass `BaseTrainer` (that class is a diffusion spine, and its `encode_image` wraps the VAE forward in `no_grad`). See `docs/guides/VAE_TRAINING.md`. |
| `backend/core/inference/video_mask_timeline.py` | Pure-Python spatial-mask timeline for `/generate/inpaint/video`: manifest validation, keyframe interpolation (`hold`/`affine`/`sdf`), rasterization to soft per-frame masks, max-pooling onto the latent token grid to decide pinned rows, and pixel-exact-at-mask==0.0 compositing. No model/server/GPU dependency. |
| `backend/core/inference/video_mask_preview.py` | `/video-mask/preview` support: calls `video_mask_timeline.rasterize_mask_timeline` verbatim (so a preview matches the real generation rasterization), then downscales and packs the requested frames into one sprite-strip PNG. |
| `backend/core/inference/context_tiled_decode.py` | `vae_tile_mode: "context"` — tiled decode with a discarded real-context margin instead of an overlap cross-fade. |
| `backend/core/inference/global_group_norm.py` | `vae_tile_global_norm` — opt-in two-pass whole-image GroupNorm statistics for a tiled decode. Both are installed by `PipelineManager._apply_vae_tiling`; see `docs/guides/VAE_DECODE_BEHAVIOR.md`. |
| `backend/api/routes.py` | All FastAPI endpoints. The image generation endpoints (`/generate/txt2img`, `/generate/img2img`, `/generate/inpaint`) are `multipart/form-data` (`Form(...)` params), not JSON; the video/audio ones (`/generate/txt2vid`, `/generate/txt2aud`) take a JSON body, and the ones that carry a file (`/generate/img2vid`, `/generate/ref2vid`, `/generate/outpaint/video`, `/generate/outpaint/audio`) are multipart. `_reject_if_video_model` / `_reject_if_audio_model` keep each family's endpoints from accepting another family's loaded model. |
| `backend/api/param_defaults.py` | Single source of truth for every default value (`GENERATION_DEFAULTS`, `TRAINING_DEFAULTS`, `TAGGER_TRAINING_DEFAULTS`, `VAE_TRAINING_DEFAULTS`), exposed via `/schema/*`. |
| `backend/api/websocket.py` | Progress-streaming WebSocket implementation (protocol documented in `backend/api/WS_PROTOCOL.md`). |
| `backend/utils/image_utils.py` | Saves generated images with embedded PNG metadata (generation parameters). |
| `backend/database/models.py` | SQLAlchemy models: `UserSettings`, `GeneratedImage`, `Dataset`/`DatasetItem`/`DatasetCaption`, `TrainingRun`/`TrainingMetrics`/`TrainingCheckpoint`, `TaggerTrainingRun`, etc. |

## Frontend structure

| Path | Responsibility |
|---|---|
| `frontend/src/components/generation/Txt2ImgPanel.tsx` / `Img2ImgPanel.tsx` / `InpaintPanel.tsx` | The three generation panels: params state, UI controls, loop-generation step params, FormData/apiParams construction. **There is no separate video or audio panel**: when `StartupContext` reports the loaded model as `isVideo`/`isAudio`, these panels switch mode (Txt2Img → txt2vid/txt2aud/ref2vid, Img2Img → img2vid/ref2vid, Inpaint → `/generate/inpaint/video` on a loaded MiniMax-H3 `fl2va` model) and render the video/audio controls and `<video>` output inline. |
| `frontend/src/components/generation/VideoInpaintTimeline.tsx` | The `/generate/inpaint/video` control surface: one shared ruler/playhead (`components/timeline/Timeline.tsx`) with two stacked tracks — a regenerate-range track (two-handle drag, snapped to `video_constraints.latent_chunk_pattern` latent-group boundaries so a built request needs no server-side snap) and a mask-keyframe track (add/duplicate/delete, transform, interpolation, composite feather). Replaces the former separate `VideoInpaintRangeTimeline`/`VideoInpaintMaskTimeline`. |
| `frontend/src/components/timeline/Timeline.tsx` | Shared horizontal timeline primitive (ruler, playhead, click/drag-to-seek); tracks (regenerate range, mask keyframes, outpaint placements) are plain children, not a plugin abstraction. |
| `frontend/src/components/generation/VideoMaskPreviewOverlay.tsx` | Renders a `/video-mask/preview` rasterization as a semi-transparent overlay on the video frame. |
| `frontend/src/components/generation/VideoMaskFrameEditor.tsx` | Wraps `ImageEditor` for a spatial-mask keyframe, adding frame-to-frame navigation without leaving the editor. |
| `frontend/src/utils/maskConventions.ts` | Single source of truth for mask polarity (`white_generate`), the on-screen overlay blend (`screen`/0.5 alpha), and mask-layer PNG encoding — previously duplicated across `ImageEditor`, `InpaintPanel`, and `videoMaskTimeline.ts`. |
| `frontend/src/utils/timelineScale.ts` | Pure clientX↔value and value↔CSS-position arithmetic shared by every timeline track. |
| `frontend/src/utils/canvasFit.ts` | Center-crop-cover mapping of an arbitrary-aspect source image onto a fixed-size output canvas, matching the backend's `center_crop_resize_frames` (`backend/core/inference/outpaint_utils.py`); used so a mask drawn against an on-screen frame lines up with the same pixel offset the backend produces. |
| `frontend/src/utils/videoMaskPersistence.ts` | Reload-persistence for the video-inpaint mask manifest: small JSON (keyframes, feather, canvas size, `temp_img://` refs) in localStorage, PNG bytes via the existing `tempImageStorage.ts` backend-temp mechanism. |
| `frontend/src/hooks/useSnapshotHistory.ts` | Generic undo/redo for a controlled value held by a parent component; used for `VideoInpaintTimeline`'s mask-keyframe edits (not mask drawing itself, which has ImageEditor's own canvas-scoped undo). |
| `frontend/src/hooks/useMaskPreview.ts` | Debounces a mask manifest + assets + requested frame list into one `/video-mask/preview` fetch and tracks whether the held result is stale relative to the current input. |
| `frontend/src/hooks/useVideoFrameImage.ts` | Grabs a video frame and maps it onto the output canvas via `canvasFit.ts`. |
| `frontend/src/components/common/MiniMaxH3ReferenceSelector.tsx` | The omni-reference uploads of a `/generate/ref2vid` request (images, videos with an optional positional soundtrack, standalone audio), rendered by `Txt2ImgPanel.tsx` and `Img2ImgPanel.tsx` when the loaded MiniMax-H3 partition is `ref2va`. The upload order is the packed order, so the control keeps it. `OutpaintPanel.tsx` also renders it, image-only, gated to `ref2va` + `extend_forward` — see the `/generate/outpaint/video` reference bullet in `docs/guides/MODEL_FACTS.md` for how that surface was accepted past a failed gate. |
| `frontend/src/components/generation/OutpaintPanel.tsx` + `OutpaintTimeline.tsx` / `OutpaintPlacementCanvas.tsx` | Outpaint tab, spatial (image) and temporal (video/audio extend); the timeline restricts placements to the ones the loaded architecture's conditioning can anchor. |
| `frontend/src/components/generation/LoopGenerationPanel.tsx` | Loop-generation queue configuration UI. |
| `frontend/src/components/viewer/ImageGrid.tsx` | Gallery grid; reads generation metadata off each `GeneratedImage`. |
| `frontend/src/components/training/` | Training run configuration and monitoring UI. |
| `frontend/src/contexts/StartupContext.tsx` | Fetches and holds `generationDefaults`/`trainingDefaults`/`taggerTrainingDefaults`/`vaeTrainingDefaults` from `/schema/*` at startup. |
| `frontend/src/contexts/GenerationQueueContext.tsx` | Client-side generation/loop queue state. |
| `frontend/src/utils/api.ts` | Typed API client: request/response interfaces, FormData construction for img2img/inpaint, JSON body for txt2img. |

## Databases

| File | Holds |
|---|---|
| `gallery.db` | `GeneratedImage` rows (generated images + parameters + metadata) and `UserSettings` (includes `model_dirs`: additional base-model search directories). |
| `datasets.db` | `Dataset`, `DatasetItem`, `DatasetCaption`, `tag_dictionary`, caption-processing presets. |
| `training.db` | `TrainingRun`, `TrainingMetrics`, `TrainingCheckpoint`, `TrainingSamples`, `TaggerTrainingRun`, `TaggerTrainingMetrics`. |

## Where parameters, defaults, and schemas live

- Default values: `backend/api/param_defaults.py` (only place to edit).
- API contract (request/response schemas, examples): `openapi.yaml`, kept in
  sync with `backend/api/routes.py`.
- Runtime schema fetched by the frontend: `GET /api/v1/schema/generation-defaults`,
  `/schema/training-defaults`, `/schema/tagger-training-defaults`,
  `/schema/vae-training-defaults`.
- Full parameter-addition checklist: `docs/guides/ADD_A_PARAMETER.md`.
