# WebSocket Progress Protocol — `/api/v1/ws/progress`

This document describes the actual behavior of the `/api/v1/ws/progress` WebSocket
endpoint, as implemented in `backend/main.py` and `backend/api/websocket.py`. It is
written for API-only clients that do not use the bundled Next.js frontend.

**Polling alternative**: a client that does not want to hold a WebSocket connection
open, or that is affected by the lack of a `complete`/`error` message documented
below, can instead poll `GET /generation/status` (see `openapi.yaml`,
`GenerationStatusResponse`). It reports the same per-step state
(`current_step`/`total_steps`/`phase`) for image generation as the `progress`
message here, plus an explicit `status` field (`idle` / `running` / `error`) and a
`last_result` / `last_error` populated when the generation ends — signals this
WebSocket channel does not provide. It is backed by an in-memory, single-process
snapshot (`backend/api/generation_status.py`) updated from the same progress
callback that feeds this WebSocket, so it does not affect WS behavior; it only
tracks image generation, not training/tagger/dataset-scan progress.

## Endpoint

```
ws://<host>:<port>/api/v1/ws/progress
```

- Registered directly on the FastAPI app (`backend/main.py:153-155`), before the CORS
  middleware, so it is reachable regardless of `Origin` header.
- No query parameters, headers, or authentication are read or required by the route
  handler. `websocket_endpoint()` (`backend/api/websocket.py:241-250`) calls
  `websocket.accept()` unconditionally.
- No subprotocol negotiation.

## Connection model: single global broadcast channel

There is **no per-request / per-task subscription model**. Every connected client is
appended to `ConnectionManager.active_connections`
(`backend/api/websocket.py:8-23`), and every message the server produces —
image-generation progress, LoRA/Full-FT training metrics, tagger-training metrics,
and dataset-scan progress — is broadcast (`ConnectionManager.broadcast`,
`backend/api/websocket.py:232-237`) to **all** currently connected clients, with no
job id, request id, or client id filtering.

Consequences for a client:
- To start a generation, POST to the REST endpoint (e.g. `/generate/txt2img`,
  `/generate/img2img`, `/generate/inpaint`) on a separate HTTP connection. The
  WebSocket connection only receives server-pushed progress; it never receives a
  request/response pairing with the POST that triggered generation.
- If more than one generation/training run is active at once, or more than one
  client is connected, a client cannot tell from the message alone which REST
  call produced it — there is no correlating id field in the `progress` message
  type (see below). Training/tagger-training messages do carry a `run_id`, which
  can be used to filter those specific streams.
- The connection is not closed by the server when a generation or training run
  finishes; it stays open indefinitely for subsequent requests until the client
  disconnects or the network drops it.

## Lifecycle

1. **Connect**: client opens the WebSocket. Server calls `accept()` and registers
   the connection — no message is sent back to the client on connect (the `type:
   "connected"` message some clients may show is produced by the frontend's own
   Next.js SSE proxy at `frontend/src/app/api/progress/route.ts`, not by the
   backend itself).
2. **Receive**: server pushes JSON text frames (see message types below) whenever
   generation/training code calls into `ConnectionManager`.
3. **Client-to-server messages**: the server loop calls
   `websocket.receive_text()` in a `while True` loop (`backend/api/websocket.py:246-248`)
   but the received text is discarded (`# Handle incoming messages if needed`,
   never implemented). Sending anything from the client has no effect. A client
   that never sends anything is fine; the `receive_text()` call is only there so
   the server can detect disconnects via `WebSocketDisconnect`.
4. **Keepalive**: there **is** a server-side heartbeat. `ConnectionManager.start_sender()`
   (`backend/api/websocket.py:206-230`) waits on an internal event with a 30-second
   timeout; if no real message was queued within that window, it broadcasts
   `{"type": "ping"}` to all active connections. There is no `pong` reply expected
   or read by the server — this is a one-way keepalive, not a WebSocket-protocol
   ping/pong frame. Clients should tolerate/ignore `{"type": "ping"}` frames.
5. **Disconnect**: on `WebSocketDisconnect`, the server removes the connection from
   `active_connections` (`ConnectionManager.disconnect`,
   `backend/api/websocket.py:25-26`) and takes no other action. `broadcast()`
   swallows send exceptions per-connection (bare `except: pass`,
   `backend/api/websocket.py:234-237`), so a dead/slow connection does not raise
   or affect other connections.

## Error behavior

There is **no dedicated `error` message type** emitted by the generation or
training pipelines over this channel. If a generation raises an exception, the
error surfaces only through the REST response of the POST request that started
it (the WebSocket stream simply stops advancing for that run). A client must treat
"no further progress messages, but total_steps threshold not reached" plus a
REST-level error/500 response as the failure signal — do not wait on this
WebSocket for an error notification.

There is likewise no `complete` / `done` message type. The `progress` message
(see below) is the only signal of process advancement; a client determines
completion either by `step == total_steps` (equivalently `progress == 100`) in a
`progress` message, or — more reliably — by the REST POST response returning.

## Message types

All messages are JSON-encoded text frames. Every message has a `type` field. The
exact set of types actually constructed and sent, with file:line evidence:

### `ping`

Heartbeat, sent when the queue has been idle for 30s.

Source: `backend/api/websocket.py:220`

| field | type | description |
|-------|------|-------------|
| `type` | string | always `"ping"` |

```json
{"type": "ping"}
```

### `progress`

Image-generation step progress, optionally carrying a base64 preview image and
CFG metrics. Sent from `ConnectionManager.send_progress_sync()`
(`backend/api/websocket.py:39-53`), called from the per-step callback built by
`create_progress_callback_factory()` (`backend/api/generation_utils.py:118-198`,
dispatch at line 192), and directly from several routes for non-generation
long-running operations such as dataset tag-statistics scans and thumbnail
regeneration (e.g. `backend/api/routes.py:806`, `:4864`, `:5182`, `:5650`,
`:5700`, `:5771`, `:9040`, `:9065`, `:9090`, `:9114`). There is also an
`async def send_progress()` variant (`backend/api/websocket.py:28-37`) with the
same payload shape, defined but not called from any current call site — only the
synchronous `send_progress_sync` path is actually exercised.

| field | type | description |
|-------|------|-------------|
| `type` | string | always `"progress"` |
| `step` | int | current step (1-indexed display step for image generation; the underlying denoise loop's step `-1`, the initial-noise state, is remapped to display step `0`) |
| `total_steps` | int | total step count for this run/operation |
| `progress` | float | `(step / total_steps) * 100`; `0` if `total_steps` is falsy |
| `message` | string | human-readable status text, e.g. `"Step 3/28"` |
| `preview_image` | string (optional) | base64-encoded JPEG (quality 75) of a TAESD-decoded latent preview; only present when a preview was generated for this step |
| `cfg_metrics` | object (optional) | passthrough of whatever `cfg_metrics` dict the sampling loop supplied for this step (shape defined by the caller, e.g. `custom_sampling.py`, not fixed by the WS layer); only present alongside a preview |

```json
{
  "type": "progress",
  "step": 15,
  "total_steps": 28,
  "progress": 53.57,
  "message": "Step 15/28",
  "preview_image": "<base64-jpeg>",
  "cfg_metrics": {}
}
```

### `training_metrics`

LoRA/Full-FT training step metrics. Sent from
`ConnectionManager.send_training_metrics()` (`backend/api/websocket.py:55-109`),
called from the training loop in `backend/core/training/base_trainer.py` (the
`_flush_metrics_to_db` broadcast site).

| field | type | description |
|-------|------|-------------|
| `type` | string | always `"training_metrics"` |
| `run_id` | int | training run id |
| `step` | int | global training step |
| `loss` | float | training loss |
| `resume_seq` | int | 0 for the initial run, incremented per resume, so a resumed run's metrics can be charted as a separate curve |
| `epoch` | int (optional) | current epoch, if known |
| `recon_loss` | float (optional) | reconstruction loss component, if computed |
| `extra_metrics` | object (optional) | `{name: float}` bespoke arch/method-specific per-step scalars (REPA `repa_loss`, outpaint ControlNet `gen_loss`, …). Display metadata comes from `core.training.metric_registry.EXTRA_METRIC_DEFS`; replaces the former dedicated `repa_loss` field |
| `learning_rate` | float (optional) | current LR |
| `grad_norm` | float (optional) | overall gradient norm |
| `grad_norm_text_encoder` | float (optional) | gradient norm, single/shared text encoder |
| `grad_norm_text_encoder_1` | float (optional) | gradient norm, text encoder 1 (dual-encoder archs) |
| `grad_norm_text_encoder_2` | float (optional) | gradient norm, text encoder 2 (dual-encoder archs) |
| `grad_norm_unet` | float (optional) | gradient norm, U-Net/DiT backbone |
| `grad_norm_vision_encoder` | float (optional) | gradient norm, vision encoder (if trained) |

```json
{
  "type": "training_metrics",
  "run_id": 42,
  "step": 500,
  "loss": 0.0231,
  "resume_seq": 0,
  "epoch": 3,
  "learning_rate": 0.0001,
  "grad_norm_unet": 1.42
}
```

### `tagger_metrics`

Tagger-training progress/metrics. Sent from
`ConnectionManager.send_tagger_metrics()` (`backend/api/websocket.py:111-165`),
called from `_make_tagger_progress_callback()`'s inner `callback()` in
`backend/api/routes.py` (e.g. the `event_type == "step"` branch at
`backend/api/routes.py:9537-9559`, plus additional branches around
`:9573` and others in the same function for other `event_type` values such as
epoch/validation events).

| field | type | description |
|-------|------|-------------|
| `type` | string | always `"tagger_metrics"` |
| `run_id` | string | tagger training run id |
| `event` | string | event kind, e.g. `"step"`; other values are emitted by other branches of the same callback (epoch/eval events) and carry the fields relevant to that event, all optional per the sender signature |
| `step` | int | current step |
| `resume_seq` | int | 0 for initial run, incremented per resume (same semantics as `training_metrics`) |
| `epoch` | int (optional) | current epoch |
| `loss` | float (optional) | training loss |
| `lr` | float (optional) | learning rate |
| `f1` | float (optional) | validation F1 |
| `train_f1` | float (optional) | training-set F1 |
| `threshold` | float (optional) | tag-decision threshold |
| `progress` | float (optional) | 0-100 progress for the current phase |
| `precision` | float (optional) | validation precision |
| `recall` | float (optional) | validation recall |
| `fp_fn_scatter` | object (optional) | false-positive/false-negative scatter data, shape defined by the caller |

```json
{
  "type": "tagger_metrics",
  "run_id": "tagger_20260101_120000",
  "event": "step",
  "step": 100,
  "resume_seq": 0,
  "epoch": 1,
  "loss": 0.0512,
  "lr": 0.0002,
  "progress": 12.5
}
```

### `dataset_scan_progress`

Progress for dataset directory-walk / drift-check operations that run before
tagger or LoRA/Full-FT training. Sent from
`ConnectionManager.send_dataset_scan_progress()`
(`backend/api/websocket.py:167-204`), called from multiple sites in
`backend/api/routes.py` (e.g. `:7707`, `:7729`, `:7749`, `:7773`, `:7789`,
`:7801`, `:7825`, `:7837`, and the equivalent block starting at `:9782`).

| field | type | description |
|-------|------|-------------|
| `type` | string | always `"dataset_scan_progress"` |
| `scope` | string | `"tagger"` or `"training"` |
| `run_id` | string or int | tagger run id (string) or LoRA/Full-FT run id (int) |
| `dataset_id` | int | dataset being scanned |
| `phase` | string | one of `"drift_walk"`, `"drift_done"`, `"rescan"`, `"cleanup"`, `"skipped"` |
| `files_walked` | int | files walked so far |
| `items_in_db` | int | items already tracked in the DB |
| `items_missing` | int | items in DB no longer found on disk |
| `items_new` | int | items found on disk not yet in DB |
| `message` | string (optional) | human-readable status, only present if non-empty |
| `dataset_name` | string (optional) | dataset display name, only present if non-empty |

```json
{
  "type": "dataset_scan_progress",
  "scope": "training",
  "run_id": 7,
  "dataset_id": 3,
  "phase": "drift_walk",
  "files_walked": 12000,
  "items_in_db": 11500,
  "items_missing": 12,
  "items_new": 500
}
```

## Frontend reference implementation (for context, not required behavior)

The bundled Next.js frontend does not connect to this WebSocket from the browser
directly. Instead, `frontend/src/app/api/progress/route.ts` runs a server-side
Next.js API route that opens the WebSocket to the backend on the server, and
re-emits every frame (plus its own synthetic `{"type": "connected"}` on open and
`{"type": "closed", code, reason}` on close) as Server-Sent Events to the browser.
Those two synthetic types (`connected`, `closed`) are **not** produced by the
backend itself — do not expect them if you connect directly to
`/api/v1/ws/progress`.

## Minimal Python client

Generation is started via a separate REST POST endpoint (e.g.
`POST /api/v1/generate/txt2img`) — this WebSocket only receives server-pushed
progress and never needs anything sent to it by the client.

```python
import asyncio
import json
import websockets

WS_URL = "ws://localhost:8000/api/v1/ws/progress"

async def watch_progress():
    async with websockets.connect(WS_URL) as ws:
        async for raw in ws:
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "ping":
                # One-way keepalive; no reply expected.
                continue

            if msg_type == "progress":
                step = msg.get("step")
                total = msg.get("total_steps")
                print(f"[progress] {msg.get('message')} ({step}/{total})")
                if total and step is not None and step >= total:
                    # Heuristic only: this WS has no dedicated "complete" message.
                    # Prefer relying on the REST POST response for authoritative
                    # completion/failure status.
                    break

            elif msg_type == "training_metrics":
                print(f"[training run={msg['run_id']}] step={msg['step']} loss={msg['loss']}")

            elif msg_type == "tagger_metrics":
                print(f"[tagger run={msg['run_id']}] event={msg['event']} step={msg['step']}")

            elif msg_type == "dataset_scan_progress":
                print(f"[scan {msg['scope']} run={msg['run_id']}] phase={msg['phase']} "
                      f"walked={msg['files_walked']}")

            else:
                print(f"[unknown message type] {msg}")

if __name__ == "__main__":
    asyncio.run(watch_progress())
```

Start a generation from a second, independent HTTP client (e.g. `requests.post(...,
"http://localhost:8000/api/v1/generate/txt2img", json={...})`) while the above
script is running; its REST response is the authoritative source for the final
result and for any error, since this WebSocket has no `complete`/`error` message
type and is a global, unfiltered broadcast channel shared by every connected
client and every concurrent operation.
