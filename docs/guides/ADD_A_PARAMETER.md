# Add a Generation Parameter

Ordered checklist for threading a new generation parameter end-to-end. The
worked example throughout is `spectrum_w_decay`, an existing parameter —
`grep -rn spectrum_w_decay` across the paths below to see every site it
touches, and mirror that shape for your new parameter.

## Checklist (do these in order)

1. **`backend/api/param_defaults.py`** — add the default to
   `GENERATION_DEFAULTS` first. Every other default below must read from
   this dict, never a hardcoded literal.
2. **`openapi.yaml`** — add the field to the relevant request schema(s)
   under `components/schemas`, with `description` and `example`.
3. **`backend/api/routes.py`**:
   - Pydantic `GenerationParams` model field:
     `spectrum_w_decay: float = GENERATION_DEFAULTS["spectrum_w_decay"]`.
   - `Form(...)` parameter in **all three** of `generate_txt2img`,
     `generate_img2img`, `generate_inpaint`:
     `spectrum_w_decay: float = Form(GENERATION_DEFAULTS["spectrum_w_decay"])`.
   - The `params` dict literal built inside **each** of those three handlers:
     `"spectrum_w_decay": spectrum_w_decay,`.
4. **`backend/core/pipeline.py`** — thread the value through call sites if
   `PipelineManager` needs to read or forward it.
5. **Consumer** — the module that actually uses the value, e.g.
   `backend/core/inference/custom_sampling.py` or
   `backend/core/inference/spectrum_forecaster.py`.
6. **Metadata / gallery display**:
   - `backend/utils/image_utils.py` — add the key to the metadata field list
     so it's embedded in the saved PNG.
   - `frontend/src/components/viewer/ImageGrid.tsx` — add the key to its
     matching metadata-field list so the gallery detail view shows it.
7. **`frontend/src/utils/api.ts`**:
   - Add the field to the relevant TypeScript param interface(s).
   - Append it to the `FormData` in **all three** of `generateTxt2Img`,
     `generateImg2Img`, and `generateInpaint` — every generation function
     builds multipart `FormData` via explicit `append` calls; a field missing
     from any of them never reaches the backend
     (see `docs/guides/REQUEST_LIFECYCLE.md`).
8. **All three panels** (`Txt2ImgPanel.tsx`, `Img2ImgPanel.tsx`,
   `InpaintPanel.tsx`):
   - Add the field to each panel's `DEFAULT_PARAMS`.
   - Add the UI control.
   - Add the field to the loop-generation `stepParams` construction (the
     object built when queueing the next loop step — it does **not**
     inherit via spread, each field is listed explicitly).
   - In `InpaintPanel.tsx` (and `Img2ImgPanel.tsx`) specifically, also add it
     to the `apiParams` object built when dequeuing a queued item to call
     `generateImg2Img`/`generateInpaint` — this is a separate site from
     `stepParams` and is easy to miss.

## Common failure patterns

| # | Missing site | Symptom |
|---|---|---|
| 1 | `FormData.append(...)` in `api.ts` | Value visible in the browser payload inspector's JS object but never appears in the actual network request body. |
| 2 | `Form(...)` parameter in `routes.py` | Value sent by the frontend but FastAPI silently drops it (no such parameter declared). |
| 3 | `params` dict literal in the route handler | Value received as a Form parameter but never reaches `pipeline_manager` (`NameError` if referenced, or simply absent). |
| 4 | `apiParams` object (Img2Img/Inpaint dequeue) | Main generation works; queued/loop generation sends `undefined`/`null` for this field. |
| 5 | `stepParams` object (loop-generation enqueue) | First generation in a session works; the second+ loop iteration reverts to the default. |
| 6 | `DEFAULT_PARAMS` in a panel | Toggling the UI control appears to do nothing, or the value is `undefined` until the user touches the control at least once. |

## Verification recipe

1. **Grep parity against a sibling parameter** that already has full
   coverage (e.g. `spectrum_w_decay`, `spectrum_delta_cap`): for each file
   the sibling appears in, confirm your new parameter appears the same
   number of times in the same functions/objects.
   ```
   grep -rn spectrum_w_decay backend/api/routes.py backend/api/param_defaults.py \
     backend/utils/image_utils.py frontend/src/utils/api.ts \
     frontend/src/components/generation/*.tsx frontend/src/components/viewer/ImageGrid.tsx
   ```
2. **Compile-check** every changed backend file:
   `venv/Scripts/python.exe -m py_compile backend/api/routes.py ...`
3. **Real import**, not just `py_compile`, to catch module-load-time errors:
   `venv/Scripts/python.exe -c "import backend.api.routes"`
4. Frontend build/type-check is run by the repo owner — do not run it
   yourself; a careful read-through of the panel and `api.ts` diffs is the
   substitute.
