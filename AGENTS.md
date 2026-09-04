# AGENTS.md

SushiUI is a Stable-Diffusion-style web UI: a FastAPI backend (`backend/`)
driving 14 diffusion architectures — 10 image (SD1.5, SDXL, Z-Image, Flux2,
Anima, Lens, Krea2, Ideogram4, MiniT2I, SenseNova U1.5), 2 video (LTX-2.3,
MiniMax-H3, both of which also generate audio jointly) and 2 audio (ACE-Step
1.5, MiniMax Music 3) — plus LoRA / full-parameter / tagger / VAE-decoder
training, and a Next.js frontend (`frontend/`). The authoritative *generation*
list is `ModelType` in `backend/core/model_loader.py`; the authoritative
*training-capable* list is `ARCH_REGISTRY` in
`backend/core/training/arch/__init__.py` (13 entries — every generation
architecture except MiniMax Music 3). SenseNova U1.5 supports LoRA (generation
branch, plus the understanding branch when `train_text_encoder` is set),
reference-conditioned datasets, and — since U-2-2 step 3 — **full-parameter
training of either MoT half or both** (`train_unet` / `train_text_encoder`
select them; both halves is measured but expensive, and the capability table
says so on its advisory axis rather than pretending it is unsupported), under a
per-run contract that is not
negotiable (bf16, no gradient accumulation, no EMA,
`blocks_to_swap=0`, and one of three optimizers — `adafactor`, or either
ring-buffer optimizer with `optimizer_state_host_resident`) and refused before
the model loads. Physical batch is 1 **unless `enable_bucketing` is on**
(a batch is one pixel tensor at one resolution; the prompts are packed, not
padded) — that one is conditional, applies to LoRA too, and the capability
surface expresses it with an `unless` clause rather than as an absolute.
`relora` and `controlnet` are still refused for it. See
`docs/guides/SENSENOVA_TRAINING_DESIGN.md` for its
implemented and pending boundaries, and `docs/guides/MINIMAX_MUSIC3_DESIGN.md`
for the remaining training-out-of-scope architecture.
Per-architecture facts are
in `docs/guides/MODEL_FACTS.md`. **Adapters (LoRA and the LyCORIS algebras)
are an architecture-neutral subsystem of their own**, `backend/core/adapters/`:
it owns the adapter spec, target topology, tensor grouping, checkpoint codecs,
the `AdapterSession` runtime that eleven architectures install through, and an
execution-backend registry. Which `(algorithm, weight_decompose)` pairs an
architecture accepts is decided by the two tables in
`backend/core/adapters/capability.py` — one for generation, one for training —
and by nothing else; do not add a second place that decides it. See
`docs/guides/LYCORIS_ADAPTER_DESIGN.md`. Every capability is reachable through the
versioned REST API under `/api/v1` (see `openapi.yaml`), so agents can drive
and verify most changes without touching the UI. This file is the durable,
checked-in subset of repo conventions for coding agents; read it before making
changes.

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

  Importing the trainer or pipeline stack normally creates a CUDA context,
  which is unwelcome while the owner's training run holds the GPU. It is not a
  reason to skip the import: the single trigger is
  `diffusers.models.autoencoders.autoencoder_kl` calling
  `torch.cuda.get_device_capability("cuda")` at import time, so stubbing that
  one function plus no-oping `torch.cuda._lazy_init` and `torch._C._cuda_init`
  before the import lets it complete with `torch.cuda.is_initialized() == False`
  and the GPU untouched (measured: ~1.0 GB RSS for
  `core.training.base_trainer`). `CUDA_VISIBLE_DEVICES=""` does NOT work —
  diffusers raises `Invalid device id` instead.

  ```python
  import torch
  torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
  torch.cuda._lazy_init = lambda *a, **k: None
  torch._C._cuda_init = lambda *a, **k: None
  import core.training.base_trainer  # cwd backend/, or PYTHONPATH=backend
  assert not torch.cuda.is_initialized()
  ```
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
| MiniMax Music 3 integration (vendoring rationale, capability refusals, frame-code state contract, weight formats) | `docs/guides/MINIMAX_MUSIC3_DESIGN.md` |
| Call the API directly (scripts, smoke tests) | `docs/guides/API_TESTING.md`, `examples/api/` |
| WebSocket progress messages | `backend/api/WS_PROTOCOL.md` |
| Training parameters / config | `backend/core/training/TRAINING_PARAMS_GUIDE.md`, `backend/core/training/API_REFERENCE.md` |
| Anything adapter-related: LoRA variants (LoHa/LoKr/DoRA), the shared engine in `backend/core/adapters/`, `adapter_type`, adapter execution backends | `docs/guides/LYCORIS_ADAPTER_DESIGN.md` |
| Fine-tune a VAE (`training_method: vae_decoder`; decoder by default, encoder behind a double gate) | `docs/guides/VAE_TRAINING.md` |
| VAE decode behavior: tiling options, decoder non-locality, measured artifact facts | `docs/guides/VAE_DECODE_BEHAVIOR.md` |
| Understand what a VAE fine-tune's crop policy and `resolution` feed the decoder, and how its memory/time scale (measured; checkpointing / activation offload / tiling are analysed, not all of them config keys) | `docs/guides/VAE_TRAINING_RESOLUTION.md` |
| Model architecture internals (SD1.5/SDXL/etc.) | `backend/core/training/MODEL_ARCHITECTURES.md` |
| Attention backend selection | `backend/core/docs/ATTENTION_PROCESSORS.md`, `backend/core/attention/registry.py` |
| Block swap / CPU offload during training | `backend/core/memory_management/BLOCK_SWAP.md` |
| Find any other doc in the repo | `docs/README.md`, `docs/guides/DOC_MAP.md` |
| Assess a paper/technique before integrating it | `.claude/agents/research-integrator.md` |

## Reusable subagents

Repo-specific subagent definitions live in `.claude/agents/*.md` (tracked in
git — see the negated `.gitignore` entries for `.claude/agents/`). Check there
before writing a one-off agent prompt from scratch. Each definition sets both
its model and its reasoning effort in frontmatter (`model:` accepts `opus`,
`sonnet`, `haiku`, `fable`; `effort:` accepts `low`, `medium`, `high`, `xhigh`,
`max`, subject to what the chosen model supports). A dispatcher may override
`model` per call, but not `effort`.

| Agent | Model / effort | Use for |
|---|---|---|
| `orchestrator` | opus / high | Planning and supervising multi-agent work; the only agent that commits |
| `arch-maintainer` | opus / high | Changes scoped to one architecture's backend, loader, or training adapter |
| `feature-worker` | opus / high | A scoped feature, parameter, or bug fix across backend/frontend layers |
| `code-auditor` | opus / high | Independent read-only adversarial review of a diff before commit |
| `research-integrator` | opus / high | Assessing a paper or technique before any implementation starts |
| `consultant` | fable / low | A second opinion on a decision — a different model, read-only, judgment not edits |
| `docs-maintainer` | opus / low | Keeping tracked docs in sync with actual code |
| `api-tester` | opus / low | Driving the running backend over HTTP to verify real behavior |

Implementation work defaults to opus at high effort: the repo's failure mode is
a plausible-looking edit to numerically sensitive code, which a cheaper reader
does not catch. Bounded verification-and-sync work runs at low effort. When a
decision is contested rather than unknown, ask `consultant` before spending an
implementation pass on it — a second opinion from the same model as the caller
is worth little, which is why that one agent is deliberately a different model.

## About CLAUDE.md

`CLAUDE.md` at the repo root is the repo owner's personal, gitignored
development log and convention set (Japanese-language, includes local paths
and historical narrative). It is **not** checked into git and may be absent
in a fresh clone. This file (`AGENTS.md`) mirrors the durable, path-free
subset of those conventions that every agent needs regardless of clone.
