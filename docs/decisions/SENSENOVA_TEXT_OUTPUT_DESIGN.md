# SenseNova image-to-text design

Status: implemented; real-checkpoint quality evaluation pending.

This decision defines the implemented inference and training contracts for
SenseNova text output in SushiUI. The real-checkpoint quality comparison is a
separate owner-run acceptance step; it is not inferred from CPU-only tests.

In this document, `img2txt` is the product and API name. It covers both:

- **i2t**: image plus a SushiUI task preset produces text;
- **ti2t / it2t**: image plus an explicit instruction, optionally with hint
  tags, produces text.

The neutral `img2txt` name avoids making the input-order spelling part of the
API. Pure text-to-text chat is outside this design.

## Evidence and implementation boundary

The upstream SenseNova repository includes a Visual Understanding (VQA)
example which loads an image and calls `NEOChatModel.chat()`. Its FAQ also
describes multi-image visual understanding. The upstream training guide lists
visual-language tasks and a mixed-task training framework. These facts establish
capability and architectural feasibility; local support is implemented below,
but they do not establish a quality result for a locally fine-tuned checkpoint.

The SushiUI implementation now connects that machinery:

- `backend/core/models/sensenova/vendor/modeling_neo_chat.py` contains
  `chat()`, autoregressive `generate()`, the understanding vision path, and the
  language-model output head;
- `backend/core/pipeline_backends/sensenova.py` dispatches text output through
  the same loaded transformer;
- `backend/api/routes.py` exposes the versioned `img2txt` route and validates
  model capability before image decode or device allocation;
- `frontend/src/app/generate/page.tsx` gates a dedicated SenseNova text-output
  workspace from advertised loaded-model capabilities;
- the shared queue carries text results without routing them to the media
  gallery.

The training path supports task-homogeneous `t2i`, `ti2i`, `i2t_caption`,
`i2t_tags`, and `i2t_caption_tags` steps. Text steps bypass the VAE, construct
assistant-only causal-LM labels, and normalize CE by supervised target tokens.
Explicit scopes replace the legacy branch flags only when task views are
configured; older image-generation configurations retain their prior meaning.

The vendored upstream provenance is recorded in
`docs/legal/THIRD_PARTY_PROVENANCE.md`. The relevant upstream references are:

- <https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/README.md>
- <https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/vqa/inference.py>
- <https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/FAQ.md>
- <https://github.com/OpenSenseNova/SenseNova-U1/blob/main/training/README.md>

## Product decision: add one dedicated Generate tab

Add an `img2txt` tab to the existing Generate page. Do not place this workflow
inside txt2img, img2img, or a new txt2txt panel.

This is the least surprising split because the primary object and result are
different from every existing panel: the user supplies an image and consumes
editable text. Reusing img2img would leave image-output controls, gallery
semantics, denoising progress, and loop-generation concepts attached to a text
result. Reusing txt2txt would make the required image look optional and would
implicitly promise general chat, which is not in scope.

The tab is capability-gated, not merely hidden with a frontend model-name
check:

1. Extend `/api/v1/schema/arch-capabilities` with an output-mode capability.
2. Initially, only `sensenova` advertises `img2txt`.
3. Render the tab only when the currently loaded model advertises that mode.
4. If `?tab=img2txt` is opened without that capability, or the loaded model is
   replaced while the tab is active, fall back to `txt2img` and do not mount
   the img2txt panel.
5. The backend repeats the architecture check. UI hiding is not authorization.

This meets the SenseNova-only requirement while allowing a future architecture
to opt in without another hard-coded tab rule.

### Panel layout

The panel is deliberately small:

1. **Reference image** -- one required image in the first implementation. The
   multipart field is named `images` and is represented as a list, but v1
   requires exactly one. This leaves upstream multi-image support as a
   compatible capability increase after memory and token-budget measurements.
2. **Task** -- `Caption`, `Caption + tags`, `Tags`, or `Custom instruction`.
3. **Hint tags** -- optional input context for caption-oriented tasks. Hint tags
   are never presented as ground truth and are not silently copied into the
   structured result.
4. **Instruction** -- a visible, editable instruction. Presets supply a
   versioned backend template; editing it turns the request into ti2t without
   changing endpoints.
5. **Text generation controls** -- maximum new tokens, deterministic/sampling
   mode, temperature, top-p, top-k, repetition penalty, and seed where sampling
   uses one. Stable captioning defaults to deterministic decoding.
6. **Result** -- raw text is always shown and copyable. When a preset requests
   structure, a parsed caption/tag view is shown beside it. The user can
   download `.txt` or `.json` and requeue the frozen request.

`Tags` remains available for joint-training experiments and convenience, but
the UI labels it as secondary to SushiUI's dedicated tagger. This feature is
not intended to replace the tagger for high-accuracy tagging.

The first implementation has no multi-turn history and does not store uploaded
images or text results in the media gallery/database. The global queue retains
completed text results for the browser session, and explicit download provides
durability. A later batch-captioning product must make its own source-image,
overwrite, and audit policy rather than inheriting `GeneratedImage` semantics.

## Inference API contract

Add an OpenAPI-first multipart endpoint:

```text
POST /api/v1/generate/img2txt
```

The request contains:

| Field | Contract |
|---|---|
| `images` | Repeated upload field; exactly one in v1. |
| `task` | `caption`, `caption_tags`, `tags`, or `custom`. |
| `instruction` | Optional override; required for `custom`. |
| `hint_tags` | Optional JSON string array, distinct from target/output tags. |
| decoding fields | Backend-defaulted autoregressive controls listed above. |
| `prompt_template_version` | Optional known version; omission selects the current default. |

All defaults belong in `backend/api/param_defaults.py`. The Pydantic/Form
declarations, `openapi.yaml`, frontend API types, and schema/default endpoint
must remain synchronized under `docs/guides/ADD_A_PARAMETER.md`.

The success response is a discriminated text result:

```json
{
  "kind": "text",
  "task": "caption_tags",
  "raw_text": "...",
  "structured": {"caption": "...", "tags": ["..."]},
  "parse_warning": null,
  "effective_instruction": "...",
  "prompt_template_version": 1,
  "model": {"type": "sensenova", "source": "..."},
  "timing": {"preprocess_seconds": 0.0, "generation_seconds": 0.0}
}
```

`structured` is nullable. The backend may remove an expected wrapper or parse
valid JSON, but must never invent, reorder, or silently repair semantic content.
Malformed structured output is a successful model inference with `raw_text`
and `parse_warning`, not a transport failure.

### Dispatch and model use

Add a modality-level `generate_img2txt` entry to the pipeline manager and a
SenseNova implementation in its pipeline backend. It uses the already-loaded
transformer/tokenizer and the vendored understanding preprocessing and
`chat()` path; it must not load a second copy of the checkpoint.

The request path has these invariants:

- reject an absent or non-SenseNova model before image decoding or GPU work;
- validate encoded bytes, decoded pixel count, instruction length, and output
  token limit before entering the generation slot;
- preprocess the image using the upstream visual-understanding geometry and
  dtype contract, not the image-generation VAE path;
- use the shared GPU generation coordinator, cancellation state, and WebSocket
  lifecycle;
- return residency/offload state in `finally`, including cancellation and
  failures;
- do not apply image-generation LoRA adapters to the understanding path unless
  their saved scope explicitly includes compatible understanding modules.

The SDXL VAE swap affects generation image tokens only. Img2txt always consumes
pixels through the understanding vision encoder; routing it through the swapped
VAE would change the model contract and is forbidden.

Prompt construction is backend-owned and versioned. The response returns the
effective instruction and version so a saved result can be reproduced. The
frontend does not assemble hidden system prompts.

## Queue and result contract

Panels continue to enqueue; only
`frontend/src/components/generation/GenerationQueueProcessor.tsx` dispatches.
Extend the queue with panel/type `img2txt`, freeze the uploaded `File`, task,
instruction, hint tags, decoding parameters, model identity, and template
version at enqueue time, and preserve them across tab unmounts.

Make completed results a discriminated union:

- media: existing `image | video | audio`, with URL;
- text: `text`, with `rawText`, nullable structured fields, and no URL.

Text results go to the owning img2txt panel's completed-result state. They must
not enter `resultFeed` or `FloatingGallery`; creating a fake URL or overloading
`GeneratedImage` would corrupt both type and persistence contracts. The queue
and result UI must remain usable if the user changes tabs while decoding.

## Training task model

Introduce explicit SenseNova tasks. Do not infer the objective from a caption
column name or from `train_text_encoder`:

| Task | Input | Target | Loss path |
|---|---|---|---|
| `t2i` | text | image | existing generation flow loss |
| `ti2i` | text + reference image | image | existing reference-conditioned flow loss |
| `i2t_caption` | image, optional hint tags | natural caption | understanding causal LM CE |
| `i2t_tags` | image | canonical tag sequence | understanding causal LM CE |
| `i2t_caption_tags` | image, optional hint tags | structured caption + tags | understanding causal LM CE |

For text-output tasks, labels cover assistant output tokens only. System,
instruction, hint, and image-context tokens are masked with the CE ignore index.
Loss is averaged per non-masked target token. Flow loss keeps its current
per-image-token reduction. The two normalized losses are logged separately and
combined only through explicit task/loss weights; their raw magnitudes are not
comparable.

The present vendored generic `NEOChatModel.forward()` is not a usable shortcut:
it intentionally raises before its older CE body. Training must add a tested
understanding-forward entry that follows the active vendored decoder/indexing
contract rather than deleting that guard and assuming the unreachable body is
current.

### Caption and tag supervision

The recommended dataset view jointly teaches three explicit response styles:

- natural caption only;
- natural caption plus canonical tags in a structured response;
- tags only, at a lower or separately configurable task weight.

One physical dataset item may expose more than one task view. The task scheduler
selects a view when the item is drawn; it must not duplicate every item in
memory. Caption-only examples prevent the model from learning that every
caption must contain a tag block. Combined examples teach one-pass extraction.
Tags-only examples keep the format addressable but need not compete equally
with the stronger dedicated tagger.

Input hint tags and target tags are separate caption sources in dataset config.
Hint tags use deterministic, seeded dropout so the captioner works with or
without them and cannot succeed by unconditional copying. Target tag order and
serialization are canonical within a prompt-template version to avoid spending
capacity on arbitrary permutations. Tag targets may be prepared offline by the
tagger, but online tagger distillation is not part of each training step.

### Dataset and sampling contract

Extend each dataset entry with a list of task views and optional weights, for
example:

```yaml
datasets:
  - dataset_id: 25
    task_views:
      - task: i2t_caption
        target_caption_types: [natural_language]
        hint_caption_types: [tags]
        weight: 1.0
      - task: i2t_caption_tags
        target_caption_types: [natural_language, tags]
        weight: 0.5
  - dataset_id: 39
    task_views:
      - task: t2i
        target_caption_types: [natural_language]
        weight: 1.0
```

The exact field defaults will be fixed in `DATASET_LEVEL_PARAMS` when
implemented and exposed through API, YAML round-trip, edit/duplicate/resume,
and frontend dataset selection.

Dataset selection remains item-proportional by default: concatenate eligible
items from all selected dataset IDs, shuffle globally, and draw items at
random. Never interpret multiple selected IDs as “draw the same count from
each dataset”; a small single-work LoRA dataset and a multi-million-item corpus
must not receive equal mass merely because each has one ID. Task-view weights
choose the objective for an eligible item. A future explicit dataset weight may
alter corpus mass, but its neutral value preserves item-proportional sampling.

All item selection, task-view selection, hint dropout, and batch formation use
checkpointed RNG/scheduler state. Resume must continue the same sequence rather
than restarting its mixture.

## Mixed image-output and text-output training

A single run containing `t2i`/`ti2i` and `i2t_*` is realistic, but it is not the
first implementation milestone.

Upstream documents a mixed-task framework, and the local model has separate
understanding and generation paths inside the same decoder layers. In the local
MoT implementation, generation has its own attention projections, norms, and
MLP/MoE modules; shared token embeddings and the unified interface do not mean
the two objectives update identical parameters everywhere. Consequently,
positive transfer is plausible but unverified. Likely benefits are preservation
of both directions, improved visual-language grounding at shared boundaries,
and reduced one-sided forgetting. Likely costs are optimizer state for both
halves, task-switch residency traffic, unequal sequence costs, gradient-scale
competition, and generation regressions from poorly weighted shared updates.

Mixed training therefore uses **task-homogeneous microbatches**. A scheduler
selects one task, the bucket manager forms a batch valid for that task, and the
matching forward/loss path runs. Do not put pixel-VQA inputs and VAE/noise image
targets in one tensor contract. With the SDXL VAE swap, generation tasks consume
latents while understanding tasks still consume pixels.

Separate optimizer parameter groups cover:

- understanding vision/projector;
- understanding decoder branch;
- shared embeddings and LM head;
- generation decoder branch;
- generation vision/flow modules.

The selected task determines which groups may receive gradients. New explicit
scope controls replace overloaded inference from `train_unet` and
`train_text_encoder`; legacy image-generation runs retain their current meaning
when no task list is configured. A configuration that selects a task but freezes
every parameter on its path is refused before model load.

The scheduler is step-based, supports configurable task weights, and logs draw
counts, tokens/images processed, normalized `loss_flow`, normalized `loss_ce`,
and per-text-task CE. It alternates homogeneous steps rather than relying on
gradient accumulation to mix objectives; current SenseNova full fine-tuning
requires accumulation 1. Checkpoints include task scheduler position, RNG,
optimizer groups, and the prompt-template versions used by the run.

No default mixed ratio is justified before measurement. Initial experiments
compare generation-only, i2t-only, and at least two mixed ratios at matched
compute, using held-out caption judgment/metrics, tag precision/recall against
curated targets, and the existing image-generation quality suite. A mixed
default may ship only if it improves text output without breaching the agreed
generation regression gate.

## Implementation order

1. **Inference backend** -- capability, defaults, OpenAPI route, preprocessing,
   pipeline dispatch, cancellation, and direct API tests.
2. **Inference frontend** -- dedicated gated tab, queue union, text result UI,
   downloads, and deep-link/model-swap behavior.
3. **i2t-only training** -- task schema, assistant-only CE path, component
   scopes, metrics, deterministic resume, and caption/tag task views.
4. **Mixed training** -- task scheduler, task-specific bucket batches, optimizer
   groups/residency, checkpoint state, and regression experiments.

This order makes the existing checkpoint's understanding quality directly
measurable before any training implementation can change it.

## Implementation record

The implementation landed as independently reviewable commits:

- `7062a3a5` and `d18f67ad`: API, inference dispatch, gated Generate workspace,
  queue, text results, download, and stale-tab fallback;
- `19f2bbbf`, `0d6f3052`, `f3d64779`, and `a850a9ae`: task schema,
  item-proportional deterministic scheduling, understanding forward, and
  assistant-only supervision;
- `c72b5e1e` and `35dfbdc5`: explicit component scopes and checkpoint metadata,
  target-token-normalized CE, mixed-task training-loop dispatch, per-task
  metrics, task-aware update census, and deterministic resume state.
- `957cd994`: img2txt LoRA selection with understanding-only application and
  generation-only adapter refusal;
- `534eb751` and `59b6b365`: task-view/eligibility fingerprints for exact
  resume plus per-task normalized loss, cumulative draws, and throughput.
- `77d01051` and `0d44b609`: dataset-independent preflight ordering and
  explicit flow-loss weighting across full and microbatched backward paths.

CPU-only regression coverage exercises the API/config round trip, task and hint
determinism, LoRA/full-fine-tune scope collection, checkpoint layouts, resume,
assistant masking, CE weighting, and legacy SenseNova training behavior. No GPU
or VRAM claim follows from those tests. The final mixed-task quality gate remains
an explicit real-checkpoint run by the repository owner.

## Acceptance gates

Inference is complete only when:

- the tab appears for an advertised loaded SenseNova model and nowhere else;
- stale/deep-linked tab state falls back safely after model replacement;
- the route refuses the wrong model before decode/GPU allocation;
- preset i2t and explicit-instruction ti2t both reach the upstream
  understanding path;
- raw output survives structured-parse failure;
- queue progress/results survive panel unmount, and text never enters the media
  gallery;
- VAE-swap configurations still feed VQA pixels through the understanding
  vision encoder;
- OpenAPI, defaults, route types, capability types, and frontend types agree.

Training is complete only when:

- CE supervises assistant target tokens and no prompt/image-context tokens;
- selecting multiple datasets preserves global item-proportional sampling by
  default;
- task draws, hint dropout, and batches reproduce across resume;
- each task reports its own normalized loss and throughput;
- a disabled task produces no gradients or optimizer updates on its exclusive
  branch;
- old image-generation configs retain their prior task, loss, and sampling
  behavior;
- mixed VAE-swap training keeps the two visual input geometries separate;
- mixed-task quality is measured against both text and generation baselines
  before a default mixed ratio is selected (owner-run pending).
