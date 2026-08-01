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
- `bench_fp8_scaled_mm.py` — measurement gate **G1** for the opt-in FP8 W8A8
  scaled-GEMM fast path in `Fp8Linear`. Two vehicles, two arms each; times
  sampler steps (via the progress WebSocket), not wall clock including model
  load; saves a 4-prompt × 2-seed quality set per arm for a human A/B.
  Defaults to dry-run. See the module docstring for the full protocol and the
  exact command lines. The pre-registered decision rule is reproduced below.

## FP8 scaled-GEMM gate (G1) — pre-registered decision rule

Written down **before** any measurement existed, and duplicated here so it
cannot be quietly edited in one place after the fact. `--report` evaluates the
recorded numbers against it and prints which branch applies.

**Vehicles.** Krea 2 carries the speed gate (`fp8_fast` vs `bf16`): it ships
bf16 locally, bf16 is its shipping production configuration today, and it is a
single transformer that fits VRAM. Ideogram 4 carries the regression + quality
arm (`fp8_fast` vs `fp8_dequant`, same shipped FP8 checkpoint) and does **not**
carry the ≥1.10× claim — a dequantized-bf16 Ideogram 4 arm is invalid, because
it keeps two transformers resident (asymmetric CFG) and would measure offload
traffic rather than the GEMM.

**Flip the default (and proceed to Phase 2) requires ALL of:**

1. Krea 2 `fp8_fast` ≥ **1.10×** the steps/s of Krea 2 `bf16` — median of ≥3
   timed runs, 1 warmup, fixed prompt/seed/shape, ≥20 steps.
2. Ideogram 4 `fp8_fast` ≥ **1.00×** Ideogram 4 `fp8_dequant`.
3. Both quality A/Bs clean.

**If Krea 2 lands in 1.00–1.10×:** keep the path, reframed as "removes the
dequantization step for models already stored in FP8". Make **no** speed claim
anywhere, do not flip the default, and do **not** generalize it to the runtime
`unet_quantization` enum — Phase 2's value proposition was speed.

**If Krea 2 is below 1.00×:** revert.

Recorded explicitly: *"beats the dequant path" is a valid reason to keep this
code for checkpoints that are already FP8 on disk, but never on its own a
reason to flip a default or widen the surface.*

Run `--probe` **first**: it records which `torch._scaled_mm` scaling mode the
GPU accepts (rowwise vs tensorwise). A gate result without that recorded is
uninterpretable. `SUSHI_FP8_SCALED_MM` is read at import time, so the backend
must be started with the value each arm needs — `POST /system/restart-backend`
cannot inject it (it passes no `env=`), so the repo owner has to launch it.
`SUSHI_FP8_FAST_ACCUM` stays at its shipping default (1) for every arm.

The matched FP8 Krea 2 checkpoint is produced by
`subapps/fp8_quantize/quantize_transformer_fp8.py`.

## WebSocket / progress streaming

These examples cover REST only. For the WebSocket protocol (progress
streaming during generation/training), see `backend/api/WS_PROTOCOL.md`
(message types, field tables, and a minimal Python client). The
implementation lives in `backend/api/websocket.py`.

## Interactive API docs

Swagger UI is available at `http://localhost:8000/docs` while the backend is running.
