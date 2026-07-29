# Calling the API from an Agent

The full backend is drivable over HTTP at `/api/v1` (see `openapi.yaml` for
the complete surface, and `backend/api/WS_PROTOCOL.md` for the WebSocket
progress protocol). This doc is about *which* calls an agent can make on its
own judgment, and how to make them safely.

## venv discovery

Resolve the interpreter relative to the repo root, never assume a path:

- Windows: `venv/Scripts/python.exe`
- POSIX: `venv/bin/python`

```
venv/Scripts/python.exe examples/api/health_and_schema.py
```

## Base URL

Everything is mounted under `/api/v1` (confirmed in `backend/main.py`), e.g.
`http://localhost:8000/api/v1/health` — not `/health`, not `/api/health`.

## Read-only endpoints — safe to call anytime

These only read state and are safe for an agent to call without asking:

- `GET /health`
- `GET /models/current`
- `GET /images` (gallery listing, `skip`/`limit` pagination) and
  `GET /images/{image_id}`
- `GET /schema/generation-defaults`, `/schema/training-defaults`,
  `/schema/tagger-training-defaults`, `/schema/vae-training-defaults`
- `GET /generation/status`
- `GET /training/active`

## State-changing operations — need explicit owner sanction

Do not call these unless the repo owner has explicitly asked for it in the
current task:

- `POST /generate/txt2img|img2img|inpaint` — heavy, GPU-mutating, writes to
  `gallery.db` and `outputs/`.
- `POST /models/load` — swaps the loaded pipeline, GPU-mutating.
- `POST /system/restart-backend` — kills and relaunches the backend process.
- `POST /training/runs/{id}/start` — spawns a GPU-resident subprocess and
  mutates `training.db` and disk state.

## Dry-run convention

The scripts in `examples/api/` default to **dry-run**: they print the exact
method/URL/headers/body they would send instead of sending it, and require
an explicit `--no-dry-run` flag to actually perform a state-changing
operation. Follow this convention in any ad-hoc test script you write —
default to printing the request, require an explicit opt-in to fire it.

## Polling after a restart

After `POST /system/restart-backend`, the process exits and relaunches; the
old HTTP connection will drop.

1. Poll `GET /health` in a loop until it responds again.
2. **Re-check `GET /models/current`** once health is up — the backend may
   auto-load a previously active model in the background, which can race a
   manual `/models/load` call issued right after restart. Confirm the
   currently loaded model matches what you expect before proceeding, rather
   than assuming your own load call "won".

## Progress streaming

For anything beyond a one-shot status poll (e.g. watching a generation or
training run step-by-step), use the WebSocket protocol documented in
`backend/api/WS_PROTOCOL.md` rather than tight-polling `/generation/status`.
