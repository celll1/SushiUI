# SushiUI Backend API Examples

Frontend-less examples showing how to drive the backend directly with HTTP requests.

## Prerequisites

- Backend running and reachable at `http://localhost:8000`. If it is not
  running, ask the repo owner to start it — agents should not launch servers
  themselves (see `AGENTS.md`).
- venv Python with `requests` installed (verified present: `requests==2.32.5`
  in the repo's `venv/`). All scripts use `requests`, not `urllib`.

Run with the project's venv interpreter, resolved relative to the repo root
(`venv/Scripts/python.exe` on Windows, `venv/bin/python` on POSIX):

```
venv/Scripts/python.exe examples/api/health_and_schema.py
```

## Base URL / route prefix

All API routes are mounted under **`/api/v1`** — confirmed directly from
`backend/main.py`:

```python
app.include_router(router, prefix="/api/v1")
app.include_router(logs_router, prefix="/api/v1")
```

So, e.g., the health check is `GET http://localhost:8000/api/v1/health`, not
`/health` and not `/api/health`.

## Scripts

- `health_and_schema.py` — GET `/api/v1/health` and GET
  `/api/v1/schema/generation-defaults`. Read-only, runs for real.
- `txt2img_minimal.py` — POST `/api/v1/generate/txt2img` with only `prompt`
  set. **Surprise**: this endpoint takes `multipart/form-data` (FastAPI
  `Form(...)` params), not JSON, even though a `GenerationParams` Pydantic
  model exists elsewhere in `routes.py` (it's only used by the
  training-preview endpoints). Defaults to dry-run (prints the exact
  method/url/headers/body); pass `--no-dry-run` to actually generate.
- `gallery_browse.py` — GET `/api/v1/images` (pagination params are `skip`
  and `limit`, not `page`/`offset`) then GET `/api/v1/images/{image_id}` for
  the first result. Read-only, runs for real by default.
- `training_run.py` — create (POST `/api/v1/training/runs`, JSON body), start
  (POST `/api/v1/training/runs/{run_id}/start`, no body), poll status (GET
  `.../status`), get metrics (GET `.../metrics`). Defaults to dry-run because
  starting a real training run spawns a GPU-resident subprocess and mutates
  `training.db` and disk state — pass `--no-dry-run` (with valid
  `--dataset-id`/`--base-model-path`) only when you actually intend to train.

## WebSocket / progress streaming

These examples cover REST only. For the WebSocket protocol (progress
streaming during generation/training), see `backend/api/WS_PROTOCOL.md`
(message types, field tables, and a minimal Python client). The
implementation lives in `backend/api/websocket.py`.

## Interactive API docs

Swagger UI is available at `http://localhost:8000/docs` while the backend is running.
