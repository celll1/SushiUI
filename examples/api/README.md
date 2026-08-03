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

## int8 W8A8 gate (G2) — pre-registered decision rule (separate from G1)

Written down **before** any int8 arm, CLI flag, or GEMM code exists in
`bench_fp8_scaled_mm.py`. This is a different fast path (`torch._int_mm`
W8A8, not FP8) with its own gate; it does not replace or modify the FP8 rule
above, which stays exactly as written. **Not implemented yet** — see the
module docstring's `MEASUREMENT GATE G2` section for the full text this
summarizes; `--report` does not currently evaluate it.

**Vehicle.** Krea 2 only, quantized to int8 **from its bf16 source**, not
from the shipped e4m3 checkpoint — e4m3 has already discarded weight
information int8 cannot recover, so quantizing from it would judge int8
against a floor already lowered by a different lossy step.

**Five arms** (one process): `bf16` (anchor, both axes), `int8_weight_only`,
`int8_w8a8_eager`, `int8_w8a8_fused` (the arm this gate decides on), and
`int8_w8a8_hadamard` (built only if the outlier retry below triggers).

**Quality — all four required**, measured on `int8_w8a8_fused` vs `bf16`:

1. Flat-region residual (flattest 256×256 tile, high-pass σ=6) ≤ **1.15×
   bf16** at seeds 987654321 and 12345. Calibrated against the FP8 gate's
   actual numbers (seed 12345: bf16 0.199 / dequant 0.319 / fast 0.398; seed
   987654321: bf16 0.351 / dequant 0.358 / fast 0.532) — this bar admits a
   dequant-shaped result and rejects a fast-shaped one.
2. Residual power-spectrum ratio at the 32–128px mottle wavelength ≤ **1.3×
   bf16** (the FP8 fast arm measured 3.0–8.4× here).
3. Brightness drift vs bf16 not one-signed across all 8 quality pairs, and
   mean |dV| ≤ **1.0** (the FP8 fast arm was +2.93 mean, positive in 8/8).
4. Blind human A/B clean at the mottle seed (987654321, quality prompt index
   1, the flat-gradient prompt) — its bf16 reference is genuinely clean,
   which is what makes this a real judgement.

**Speed.** `int8_w8a8_fused` ≥ **1.10×** bf16 steps/s — the same bar the FP8
fast path cleared (1.155× on Krea 2). There is **no** requirement to beat
the FP8 fast path itself: that path failed its own quality gate and is not
the shipped default.

**Branches:**

- **Both pass** → the int8 path may default ON for int8 checkpoints. This is
  licensed where the FP8 flip was not because this gate is anchored to bf16
  on *both* axes, not to "beats dequant" (the weaker, checkpoint-relative
  comparison G1 used for Ideogram 4).
- **Quality passes, speed fails** → ships as a factual VRAM-reduction
  format only; no speed claim anywhere.
- **Quality fails in an outlier-shaped way** → one pre-authorized retry
  with the Hadamard rotation added (`int8_w8a8_hadamard`); no further
  retries after that.
- **Anything else** → the code is removed, not parked behind a flag.

**Phase 0 measurements this gate was designed against** (pre-dating the
rule, cited for context, not as its source): raw `torch._int_mm` at Krea 2
shapes measured 2.857–3.075× bf16 (layer-count-weighted 3.009×; the earlier,
smaller-scope threshold this work was originally sized against was 1.30×,
not this gate's 1.10×). Eager int8 W8A8 chain 1.515× vs the shipped fused
FP8 path's 1.550×; fused int8 2.561×. Per-row accuracy on 112 real Krea 2
layers: e4m3 error flat at 0.02628–0.02649, int8 error 0.01016–0.03117;
geomean advantage 2.06× weight-only / 2.19× W8A8 (a prior simulation's 3.3×
was ~60% optimistic). One inverted layer, `transformer_blocks.27.ff.down`
(int8 0.03117 vs e4m3 0.02628), predicted in advance by within-row crest
factor (32.6 vs a typical 4.5–6) — the concrete precedent for the outlier
retry clause above. The GPU ran at a 240W cap, 735 MHz SM under load (vs
3105 MHz max) for all of it — ratios between arms hold, absolute figures do
not generalize.

### G2 is an inference gate only

G2 above decides the int8 W8A8 path for **generation**. It says nothing about
INT8 in **training**, which is a separate pre-registered gate (**G3**) with its
own rule, vehicle and bar. G3's full text lives next to the training code it
governs, in `backend/core/training/INT8_W8A8_TRAINING_GATE.md`; it is not
duplicated here, so there is exactly one copy to edit.

## WebSocket / progress streaming

These examples cover REST only. For the WebSocket protocol (progress
streaming during generation/training), see `backend/api/WS_PROTOCOL.md`
(message types, field tables, and a minimal Python client). The
implementation lives in `backend/api/websocket.py`.

## Interactive API docs

Swagger UI is available at `http://localhost:8000/docs` while the backend is running.
