# Architecture Map

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
| `backend/core/pipeline_backends/` | One file per architecture (`zimage.py`, `flux2.py`, `anima.py`, `lens.py`, `krea2.py`, `ideogram4.py`, `minit2i.py`; SD1.5/SDXL are handled by the base `pipeline.py` path) — architecture-specific generation logic as mixins. |
| `backend/core/keep_hot.py` | Arch-agnostic `keep_models_hot` state (model_key computation, VRAM guard, resident-set tracking); wired into `pipeline.py` (SD1.5/SDXL) and all 7 DiT `pipeline_backends/*.py` files. Not wired into `ltx2.py`. |
| `backend/core/training/` | `base_trainer.py` (shared loop, block-swap, optimizer wiring), `lora_trainer.py` / `full_parameter_trainer.py`, `adapters/` (per-architecture training adapters — text encoding, conditioning, time-ids), `optimizers/`, `losses/`, `bucketing.py`, `latent_cache.py`. |
| `backend/core/training/vae/` | Decoder-only VAE fine-tuning (`network.type: vae_decoder`), reached from `train_runner.py`. Standalone — does **not** subclass `BaseTrainer` (that class is a diffusion spine, and its `encode_image` wraps the VAE forward in `no_grad`). See `docs/guides/VAE_TRAINING.md`. |
| `backend/core/inference/context_tiled_decode.py` | `vae_tile_mode: "context"` — tiled decode with a discarded real-context margin instead of an overlap cross-fade. |
| `backend/core/inference/global_group_norm.py` | `vae_tile_global_norm` — opt-in two-pass whole-image GroupNorm statistics for a tiled decode. Both are installed by `PipelineManager._apply_vae_tiling`; see `docs/guides/VAE_DECODE_BEHAVIOR.md`. |
| `backend/api/routes.py` | All FastAPI endpoints; generation endpoints (`/generate/txt2img|img2img|inpaint`) are `multipart/form-data` (`Form(...)` params), not JSON. |
| `backend/api/param_defaults.py` | Single source of truth for every default value (`GENERATION_DEFAULTS`, `TRAINING_DEFAULTS`, `TAGGER_TRAINING_DEFAULTS`, `VAE_TRAINING_DEFAULTS`), exposed via `/schema/*`. |
| `backend/api/websocket.py` | Progress-streaming WebSocket implementation (protocol documented in `backend/api/WS_PROTOCOL.md`). |
| `backend/utils/image_utils.py` | Saves generated images with embedded PNG metadata (generation parameters). |
| `backend/database/models.py` | SQLAlchemy models: `UserSettings`, `GeneratedImage`, `Dataset`/`DatasetItem`/`DatasetCaption`, `TrainingRun`/`TrainingMetrics`/`TrainingCheckpoint`, `TaggerTrainingRun`, etc. |

## Frontend structure

| Path | Responsibility |
|---|---|
| `frontend/src/components/generation/Txt2ImgPanel.tsx` / `Img2ImgPanel.tsx` / `InpaintPanel.tsx` | The three generation panels: params state, UI controls, loop-generation step params, FormData/apiParams construction. |
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
