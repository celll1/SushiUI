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

## Video generation request (txt2vid / img2vid)

Two architectures serve these endpoints — LTX-2.3 and MiniMax-H3 — and the
route is the same for both; everything architecture-specific is resolved from
per-arch tables rather than branched on in the handler. Per-architecture facts
are in `docs/guides/MODEL_FACTS.md`.

1. **Frontend UI** — there is no separate video panel. When `StartupContext`
   reports the loaded model as `isVideo`, `Txt2ImgPanel.tsx` switches to video
   mode: it renders the clip-length/frame-rate controls, builds its clip-length
   options from the backend's own `video_constraints` payload
   (`videoFrameOptions` in `frontend/src/utils/api.ts`), and hides the CFG and
   negative-prompt controls when the loaded architecture declares them
   unsupported (`archSupportsFeature`). `Img2ImgPanel.tsx` is the img2vid
   equivalent. When the loaded model is MiniMax-H3's `ref2va` transformer
   partition (`model_info.variant`), both `Txt2ImgPanel.tsx` and
   `Img2ImgPanel.tsx` additionally render `MiniMaxH3ReferenceSelector.tsx` and
   route the request to `ref2vid` as soon as it carries at least one
   reference, since `/generate/img2vid` refuses a `ref2va` checkpoint
   outright. Either panel can add keyframe anchors to that same `ref2vid`
   request, as its optional `keyframe_images`/`keyframe_frame_indices`:
   `Txt2ImgPanel.tsx` renders a dedicated `MiniMaxH3Ref2VidKeyframeSelector.tsx`
   for them, while `Img2ImgPanel.tsx` carries its own uploaded image plus any
   img2vid last-frame/keyframe anchors along as that same list. With no
   references the same partition still serves a plain txt2vid/img2vid
   request.
2. **`frontend/src/utils/api.ts`** — `generateTxt2Vid` posts a **JSON** body to
   `/generate/txt2vid` (unlike the image routes, which are all multipart).
   `generateImg2Vid` is multipart because it carries an uploaded keyframe
   `image`, plus an optional `last_frame_image` for architectures that condition
   on a last frame as well. `generateRef2Vid` is multipart for the same reason
   and appends its reference files in packed order, because on `/generate/ref2vid`
   the order the files are sent is part of the request.
3. **`backend/api/routes.py::generate_txt2vid`** — a `Txt2VidRequest` Pydantic
   body. In order:
   - `quantized_gemm_mode` and `attention_type` are normalized/validated first,
     so a bad value is a 400 rather than a 500 from inside the run;
   - a non-video loaded model is rejected;
   - **fields the client omitted are filled from the LOADED architecture's
     video defaults** (`request.model_fields_set` +
     `param_defaults.video_defaults_for_arch`, backed by
     `VIDEO_GEN_DEFAULTS` merged with `VIDEO_GEN_ARCH_OVERLAYS[arch]`). Order
     matters: an omitted `num_frames` must be filled from the loaded arch's
     overlay *before* it is validated;
   - `validate_video_geometry` / `validate_video_steps` then check the resolved
     values against that architecture's `TemporalSpec`
     (`core/models/components/wiring.py`): the canvas is a hard 400, an invalid
     clip length is snapped to the arch's grid with a `warnings[]` entry on the
     archs whose spec says so, and a step count the arch's scheduler cannot build
     a schedule from is a 400 here rather than a 500 after a paid-for text encode;
   - `check_arch_capabilities` is passed the **resolved** defaults, so
     "the user set this" is judged against the numbers the request was actually
     filled from — this is what keeps a guidance-distilled architecture from
     warning about `guidance_scale` on every UI-originated generation.
4. **`backend/core/pipeline.py::generate_txt2vid`** dispatches on
   `is_ltx2_model` / `is_minimax_h3_model` to
   `backend/core/pipeline_backends/{ltx2,minimax_h3}.py`, inside a
   `gpu_coordinator.generation_slot`. Both return the same tuple contract:
   `(frames uint8 [T,H,W,3], audio FloatTensor [channels, samples] or None,
   audio_sample_rate or None, actual_seed)`.
   - **LTX-2.3** drives stock diffusers pipelines (`LTX2Pipeline` /
     `LTX2ImageToVideoPipeline` / `LTX2ConditionPipeline`) with
     `callback_on_step_end` closures for progress and latent preview, and injects
     block swap / FBCache / Spectrum at the *transformer* level via
     `Ltx2BlockLoopWrapper`. Component residency is accelerate's
     `model_cpu_offload_seq`.
   - **MiniMax-H3** runs a repo-owned loop
     (`core/models/minimax_h3/h3_pipeline_ops.py`) over vendored model classes,
     because upstream ships a Modular pipeline only. Staging is strictly
     sequential and each phase gives the GPU back before the next starts: text
     encode (the 51.5 GB Qwen3-VL conditioner is never moved — each decoder layer
     is materialised on the GPU for one call and only the layer-50 hidden state
     survives) → optional keyframe encode → denoise with the DiT alone resident →
     DiT back to CPU → video VAE decode → audio VAE decode.
   - `keep_models_hot` is not wired for either: no video component set is worth
     leaving resident between generations.
5. **Encode and save** — `backend/utils/video_utils.py::save_video_with_metadata`
   writes an H.264 mp4 (AAC audio muxed in when the pipeline returned any; FFV1 +
   FLAC when a lossless output is requested), a poster PNG sharing the mp4's base
   name, and a sidecar JSON of the generation parameters. The poster feeds
   `create_thumbnail`.
6. **Database** — a `GeneratedImage` row, as for images, with
   `parameters["is_video"] = True` plus `num_frames`, `fps` and `duration`. The
   gallery's `steps`/`cfg_scale` **columns** are filled from the video keys
   (`num_inference_steps`, `guidance_scale`) rather than the image ones the
   shared record helper would otherwise default; `vae_name`/`vae_hash` record the
   VAE that produced the frames (the video VAE, on an architecture that owns
   two). Any `warnings[]` accumulated during the request are stored as
   `effective_warnings`.
7. **Response / gallery** — the response carries `warnings[]` alongside the row.
   The gallery renders `is_video` rows through a `<video>` element with the
   poster PNG as its thumbnail.

`POST /generate/outpaint/video` follows the same route shape for temporal
extension: multipart (it carries the input clip), same per-arch default
resolution and `TemporalSpec` validation, same output contract. The input frames
come back exact either way, but by different mechanisms — LTX-2.3 generates the
whole timeline and this repo pastes the input back over its span, while
MiniMax-H3 is asked only for the missing span and the output is a
concatenation — and **which placements are offered is an architecture
property**, decided by what that model's conditioning can anchor. See
`docs/guides/MODEL_FACTS.md`.

`POST /generate/ref2vid` is multipart for its reference files and follows the
same route shape and output contract, with two gates ahead of everything else:
a MiniMax-H3 model must be loaded, and its transformer variant must be `ref2va`
— the two refusals are separate because "no H3 loaded" and "the wrong H3
partition loaded" have different fixes. Reference counts are validated before
any upload is read, and the files are decoded before the GPU slot is taken, so
a bad request pays for neither. `PipelineManager.generate_ref2vid` has no second
architecture to dispatch to. See `openapi.yaml` for the request surface and
`docs/guides/MODEL_FACTS.md` for what the references cost.

`POST /generate/inpaint/video` is the temporal counterpart of image inpaint:
multipart, takes an uploaded clip plus a `[regenerate_start_frame,
regenerate_end_frame)` pixel-frame range, and regenerates only that span while
pasting the rest of the input back after decode. MiniMax-H3 `fl2va` only (a
`ref2va` load or a non-H3 video model is refused, mirroring `/generate/img2vid`'s
and `/generate/outpaint/video`'s partition gates). The route expands the
requested range OUTWARD to the video VAE's latent-group boundaries — the
frontend's `VideoInpaintRangeTimeline.tsx` snaps its own handles to those same
boundaries (read from `video_constraints.latent_chunk_pattern`) so a built
request is already the range the server will run. See the route's own
docstring in `backend/api/routes.py` and `openapi.yaml` for the full parameter
surface.

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
