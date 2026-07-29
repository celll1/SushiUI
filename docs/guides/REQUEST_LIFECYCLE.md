# Request Lifecycle

## Generation request (txt2img / img2img / inpaint)

1. **Frontend UI** — `Txt2ImgPanel.tsx` / `Img2ImgPanel.tsx` / `InpaintPanel.tsx`
   build a `params` object (see the `DEFAULT_PARAMS` constant in each panel)
   from user input, then either queue it or call the relevant generate
   function directly.
2. **`frontend/src/utils/api.ts`** — `generateTxt2Img` / `generateImg2Img` /
   `generateInpaint`. **All three** post `multipart/form-data`, not JSON,
   even for txt2img (there is a `GenerationParams` Pydantic model in
   `routes.py`, but it backs only the training-preview endpoints, not
   `/generate/txt2img`). Complex fields — `loras`, `controlnets`,
   `tipo_config` — are JSON-serialized to strings before being appended to
   the `FormData` and are parsed back to objects on the backend
   (`json.loads(...)`); a value missing from `FormData.append(...)` here
   never reaches the backend even if it's present in `params`.
3. **`backend/api/routes.py`** — `generate_txt2img` / `generate_img2img` /
   `generate_inpaint` (all `Form(...)` parameters). Each handler builds a
   plain `dict` of parameters (independently from the Form signature — a
   parameter missing from this dict never reaches the pipeline even if it
   was received as a Form field) and calls into `pipeline_manager`.
4. **`backend/core/pipeline.py`** — `PipelineManager.generate_txt2img/img2img/inpaint`
   resolves the active architecture and VRAM strategy, moves components
   between GPU/CPU (`vram_optimization.py`), and dispatches to either the
   shared sampling loop or an architecture-specific backend.
5. **`backend/core/inference/custom_sampling.py`** (SD1.5/SDXL and the
   shared path) or **`backend/core/pipeline_backends/<arch>.py`**
   (Z-Image, Flux2, Anima, Lens, Krea2, Ideogram4, MiniT2I) — runs the actual
   denoising loop, applying prompt chunking, ControlNet conditioning, NAG,
   Advanced CFG, and spectrum guidance as configured.
6. **VAE decode** and **`backend/utils/image_utils.py`** — decodes the final
   latent (skipped for pixel-space architectures) and saves a PNG with
   generation parameters embedded as metadata. Decode-side options
   (`vae_tiling`, `vae_tile_threshold`, `vae_tile_mode`,
   `vae_tile_global_norm`) are installed onto the VAE by
   `PipelineManager._apply_vae_tiling` **before** the sampling loop, not at the
   final decode — so they also apply to any in-loop `vae.decode` (SD1.5/SDXL
   `flatten_in_loop`, `vae_drift_correction`). See
   `docs/guides/VAE_DECODE_BEHAVIOR.md`.
7. **Database** — a `GeneratedImage` row is inserted into `gallery.db` with
   the same parameters (`backend/database/models.py`).
8. **Response / gallery** — the API response returns the image path/id; the
   frontend gallery (`frontend/src/components/viewer/ImageGrid.tsx`) reads
   the stored parameters back out of `GeneratedImage.to_dict()` for display.

## Progress reporting

Two parallel channels report progress during a generation or training run:

- **WebSocket** — real-time per-step progress messages. See
  `backend/api/WS_PROTOCOL.md` for message types and field tables;
  implementation in `backend/api/websocket.py`.
- **Polling** — `GET /api/v1/generation/status` returns the current
  generation state for clients that are not (or not yet) connected to the
  WebSocket; `GET /api/v1/training/active` is the training equivalent.

## Model load flow

`POST /api/v1/models/load` (`source_type`, `source`, optional `revision`,
all `Form(...)`) calls `PipelineManager.load_model`, which:

1. Resolves `source` to a concrete file or directory.
2. For a single-file checkpoint, runs architecture detection in
   `backend/core/model_loader.py` (`detect_model_type`, backed by per-arch
   signature heuristics such as `_keys_look_krea2` / `_keys_look_lens` /
   `_keys_look_ideogram4` / `_keys_look_anima`, checking tensor key shapes
   and embedded metadata).
3. Loads the detected architecture's transformer/UNet, text encoder(s), and
   VAE. For architectures using the sushiUI single-file format
   (`backend/core/models/common/single_file_format.py`), missing components
   can be completed from sibling files/directories rather than requiring one
   monolithic checkpoint — see that module's docstring and
   `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` for the completion pattern.
4. Builds the pipeline object(s) held by `PipelineManager` for subsequent
   `/generate/*` calls.

After a restart (`POST /api/v1/system/restart-backend`), the backend may
auto-load a previously active model; see `docs/guides/API_TESTING.md` for
the poll-then-recheck sequence needed to avoid racing that auto-load.
