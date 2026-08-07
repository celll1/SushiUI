# AGENTS.md

SushiUI is a Stable-Diffusion-style web UI: a FastAPI backend (`backend/`)
driving 12 diffusion architectures — 9 image (SD1.5, SDXL, Z-Image, Flux2,
Anima, Lens, Krea2, Ideogram4, MiniT2I), 2 video (LTX-2.3, MiniMax-H3, both of
which also generate audio jointly) and 1 audio (ACE-Step 1.5) — plus LoRA /
full-parameter / tagger / VAE-decoder training, and a Next.js
frontend (`frontend/`). The authoritative list is `ARCH_REGISTRY` in
`backend/core/training/arch/__init__.py`; per-architecture facts are in
`docs/guides/MODEL_FACTS.md`. Every capability is reachable through the versioned
REST API under `/api/v1` (see `openapi.yaml`), so agents can drive and verify
most changes without touching the UI. This file is the durable, checked-in
subset of repo conventions for coding agents; read it before making changes.

## Non-negotiable rules

- **`backend/api/param_defaults.py` is the single source of truth for every
  API default value.** `GENERATION_DEFAULTS`, `TRAINING_DEFAULTS`,
  `TAGGER_TRAINING_DEFAULTS`, and `VAE_TRAINING_DEFAULTS` back the
  Pydantic/`Form()` defaults in `backend/api/routes.py` and are exposed to the
  frontend via the `/schema/generation-defaults`, `/schema/training-defaults`,
  `/schema/tagger-training-defaults`, and `/schema/vae-training-defaults`
  endpoints. Never hardcode a default anywhere else.
- **API changes are openapi-first.** `openapi.yaml` is kept in full sync with
  `backend/api/routes.py`; update the spec (paths, schemas under
  `components/schemas`, examples) as part of any endpoint or parameter
  change, not as an afterthought. See `docs/guides/ADD_A_PARAMETER.md` for
  the full parameter checklist.
- **Never start, stop, or restart backend/frontend servers directly.** The
  backend restarts itself via `POST /api/v1/system/restart-backend`. If the
  backend does not appear to be running, ask the repo owner to start it —
  do not run `python main.py`, `npm run dev`, or similar yourself.
- **Always use the repo's virtualenv Python**, resolved relative to the repo
  root: `venv/Scripts/python.exe` on Windows, `venv/bin/python` on POSIX.
  Never invoke a bare `python`/`python3` or a system interpreter.
- **After any backend edit, verify with both `python -m py_compile` on the
  changed files and a real import** (e.g.
  `venv/Scripts/python.exe -c "import backend.api.routes"`) — `py_compile`
  alone misses module-load-time `NameError`s and similar failures.
- **Frontend build and type-checking are run by the repository owner, not by
  agents.** Do not run `npm run build` or `npm run type-check`; rely on
  careful reading of `frontend/src/utils/api.ts` and the generation panels
  instead.
- **Commit style:** a concise, imperative summary line, optional body
  explaining the "why", and a `Co-Authored-By:` trailer identifying the
  agent. Follow the existing history (`git log --oneline`) for tone.
- **Keep comments short.** A comment earns its place by saying something the
  code cannot: a non-obvious constraint, a measured number, a trap that a
  plausible "simplification" would walk into. It does not earn its place by
  restating the code, narrating the investigation that produced it, or
  reproducing an argument that belongs in the commit message or a scratchpad
  note. Prefer one sentence to a paragraph and a paragraph to a block; link to
  the durable note rather than inlining it. If a comment needs more than a few
  lines to justify the code beneath it, that is a signal to name things better
  or to put the reasoning where reasoning lives. **Before finishing, re-read
  what you wrote and cut it down** — the cost of a comment is paid by every
  future reader, not by the author. Older files carry comments written before
  this rule: **trim them when you are editing that file anyway**, as part of
  the change, rather than in a separate sweep.

## Where to look for a given task

| Task | Read first |
|---|---|
| Add/change an API parameter | `docs/guides/ADD_A_PARAMETER.md` |
| Understand a generation request end-to-end | `docs/guides/REQUEST_LIFECYCLE.md` |
| Find which file owns which responsibility | `docs/guides/ARCHITECTURE_MAP.md` |
| Add a new model architecture (incl. the extra surface a video arch needs) | `docs/guides/ADD_A_MODEL_ARCHITECTURE.md` |
| Per-architecture facts (CFG convention, VAE, attention, weight formats, measured performance) | `docs/guides/MODEL_FACTS.md` |
| Call the API directly (scripts, smoke tests) | `docs/guides/API_TESTING.md`, `examples/api/` |
| WebSocket progress messages | `backend/api/WS_PROTOCOL.md` |
| Training parameters / config | `backend/core/training/TRAINING_PARAMS_GUIDE.md`, `backend/core/training/API_REFERENCE.md` |
| Fine-tune a VAE (`training_method: vae_decoder`; decoder by default, encoder behind a double gate) | `docs/guides/VAE_TRAINING.md` |
| VAE decode behavior: tiling options, decoder non-locality, measured artifact facts | `docs/guides/VAE_DECODE_BEHAVIOR.md` |
| Understand what a VAE fine-tune's crop policy and `resolution` feed the decoder, and how its memory/time scale (measured; checkpointing / activation offload / tiling are analysed, not all of them config keys) | `docs/guides/VAE_TRAINING_RESOLUTION.md` |
| Model architecture internals (SD1.5/SDXL/etc.) | `backend/core/training/MODEL_ARCHITECTURES.md` |
| Attention backend selection | `backend/core/docs/ATTENTION_PROCESSORS.md`, `backend/core/attention/registry.py` |
| Block swap / CPU offload during training | `backend/core/memory_management/BLOCK_SWAP.md` |
| Find any other doc in the repo | `docs/guides/DOC_MAP.md` |
| Assess a paper/technique before integrating it | `.claude/agents/research-integrator.md` |

## Reusable subagents

Repo-specific subagent definitions live in `.claude/agents/*.md` (tracked in
git — see the negated `.gitignore` entries for `.claude/agents/`). Check
there before writing a one-off agent prompt from scratch:
`arch-maintainer`, `code-auditor`, `api-tester`, `docs-maintainer`,
`feature-worker`, `orchestrator`, and `research-integrator` (paper/technique
integration assessment) currently exist.

## About CLAUDE.md

`CLAUDE.md` at the repo root is the repo owner's personal, gitignored
development log and convention set (Japanese-language, includes local paths
and historical narrative). It is **not** checked into git and may be absent
in a fresh clone. This file (`AGENTS.md`) mirrors the durable, path-free
subset of those conventions that every agent needs regardless of clone.
