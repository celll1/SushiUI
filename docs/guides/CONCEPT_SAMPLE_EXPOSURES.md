# Concept ordering: sample exposures

`concept_batch_order` can give each focused sample multiple exposures within its
local focus segment. It creates new sample occurrences and re-batches compatible
samples; it does not repeat an already assembled batch. Background samples remain
single-pass. MNT still runs its own consecutive iterations inside each assembled
batch, so combining the two increases training work on both axes.

## Configuration

Defaults come from `TRAINING_DEFAULTS["concept_batch_order"]` in
`backend/api/param_defaults.py` and `/api/v1/schema/training-defaults`.

```yaml
concept_batch_order:
  enabled: true
  category: character
  batch_multiplier_mode: uniform_int
  batch_multiplier: 1
  batch_multiplier_min: 1
  batch_multiplier_max: 3
```

`fixed` uses `batch_multiplier` as the total exposure count per focused sample.
`uniform_int` draws an inclusive integer between min and max once per sample and
epoch. All three count fields accept integers from 1 to 64, and min must not
exceed max. Inactive mode values are retained but do not affect the plan signature.
The default fixed count of one, including an equivalent random range 1–1, retains
the original planner, config hash and stochastic behavior.

`focus_batches` measures the original segment before multiplication. Each extra
round uses each eligible sample at most once, shuffles within compatible buckets
and creates new batches. An original sample never fills two positions in the
same batch. Partial batches are retained, including a single-image bucket where
adjacent repeats cannot be avoided. Boundary swaps keep their original base-batch
semantics; additional exposures remain in their original focus segment.

In `front` mode background intervals count actual focus batches including extra
exposures. In `spread` mode the expanded segment is placed as one unit. Replay is
separate: only original base batches enter its candidate pool, and replay does
not recursively multiply itself.

## Caption and conditioning ownership

`sample_occurrence.py` supplies shallow item views and a thread-local caption RNG
context. Captions resolve lazily from the raw source and its complete processing
configuration. Each occurrence has a reproducible independent draw for enabled
shuffle, token/tag dropout and whole-caption dropout. Epoch-fixed shuffle/dropout
becomes fixed per occurrence inside an epoch when multiplication is active.
Disabled processing stays disabled. The categorized tag path also applies
configured token dropout when resolving an occurrence.

Conditioning dropout remains a separate per-sample decision through the existing
architecture handler. It is not inferred from an empty caption or stored inside
an embedding cache. Existing `cfg_uncond_drop_per_mnt` controls sharing inside an
MNT window; another occurrence always has another window. Existing restrictions
on combining whole-caption dropout and aligned CFG-null still apply.

The shared swap buffer and CPU text prefetch use occurrence keys for exposure
views and retain legacy keys for ordinary runs. Native Ming buffers read the
same resolved caption views and retain their supervision/reference identities.
Encoder windows remain bounded; no epoch-wide embedding duplication or automatic
disk-cache expansion is introduced.

Multiplied captions currently support `swap_onthefly`, `cpu_prefetch`, and
`onthefly_gpu` text encoding. `pre_encoded_cache` is rejected explicitly rather
than substituting a stale epoch caption embedding. Architecture-specific
trainable-encoder restrictions continue to apply. Latent modes and augmentation
continue through their existing owners; this option does not promise a fresh
VAE posterior sample or a new crop for every exposure.

Batch-scoped occurrence RNG isolation makes caption prefill and resume preparation
independent of subsequent timestep/noise/null draws. The architecture's timestep
distribution, MNT stratification and sampler state remain authoritative.

## Architecture contracts

`core/training/concept_support.py` is the CPU-only authority for concept ordering.
The training registry and API capability table require an entry for every training
architecture. The same validator runs for API recipes, runner preflight, the shared
trainer and the YuE2 native loop. Unavailable checkpoint paths defer architecture
checking to runner preflight; exposure/cache incompatibility is still checked
immediately. Disabled settings retain their saved values without arming a feature.

| Architectures / path | Boundary |
| --- | --- |
| SD1.5, SDXL, Z-Image, FLUX.2, Anima, Lens, Ideogram4, MiniT2I, Krea2, Qwen-Image 2.1, Ming-Image 0.1, SenseNova SDXL Chimera | Shared image planner, occurrence captions, text buffers and resume; each architecture's existing method/CFG/cache restrictions still apply. |
| SenseNova U1.5 ordinary image generation | Uses the shared planner and its native packed-prefix encoder; explicit `task_views` are refused because their owner regroups items and replaces captions. Existing `rebase_remaining` restrictions remain. CPU routing verified; GPU acceptance pending. |
| LTX-2.3, MiniMax-H3, ACE-Step | Unsupported: their clip/audio duration and modality batch owners are not implemented by this image planner. |
| YuE2 | Unsupported: native ABC examples are tokenized before the loop and do not own occurrence-aware captions or batch ledgers. Configurations are refused instead of ignored. |
| VAE-only, tagger, style-encoder training | Outside the image-generation conditioning contract. |

Clients read `training_feature_unsupported[arch].concept_batch_order` and the
SenseNova advisory from the capability schema. VAE-only requests are rejected by
the method-specific API and YAML preflight checks.
The form prevents enabling unsupported combinations and lets an imported enabled
recipe turn the feature off without discarding its multiplier settings.

Future image architectures that use `BaseTrainer.train` inherit occurrence RNG,
caption resolution, encoder windows, pending-gradient saves and batch-ledger
continuation. Add an explicit support entry and run the registry-wide contract test.
A native loop must validate the setting and implement those owners before declaring
support; otherwise declare a factual refusal. There is no supported-by-omission
fallback for a newly registered architecture.

An architecture with additional batch compatibility constraints may annotate
`_concept_batch_signature` with a JSON-serializable identity before planning.
The shared bucket metadata carries it; the planner, boundary swaps, ledger and
resume identity include it. Annotate every affected item consistently. Existing
Ming signatures retain their original encoding and default plan digests.

The 2026-10-05 registry audit covers all 17 handlers: public API acceptance/refusal,
per-image-architecture shared caption/buffer/suffix resolution, SenseNova packed
prefix routing, native/YAML refusal before dataset work and custom batch signatures.
These are CPU contract tests, not real-model loss or GPU validation for every model.
The architecture expansion passed 221 regression tests and compilation/real imports
of all nine changed backend modules without initializing CUDA. Frontend capability
types, imported-state preservation, checkbox recovery and submit errors were reviewed
statically; build/type-check remain owner-run.

## Resume and accounting

Multiplied plans use concept plan v2 and batch ledger v2. The ledger stores
base/multiplier/replay kind and each occurrence's source, stable ID, segment,
index, seed and epoch. Resolving a suffix rebuilds occurrence views rather than
aliasing the same dataset item dictionary.

The existing `existing`, `strict` and `rebase_remaining` policies remain in place.
A remaining-data rebase preserves completed batches and retained occurrences,
removes only unprocessed occurrences from removed datasets, and admits base plus
multiplier occurrences from newly selected datasets. New replay is not imported.
Retained dataset internal changes and exposure-caption configuration changes are
checked against the ledger manifest. Changing multiplier settings, GAS or planner
version cannot silently apply the old cursor to a new plan.

Inside GAS, shared trainers save named pending gradients and GradScaler state in
an atomic `_gradients.pt` sidecar referenced by the checkpoint state. This can
increase disk usage substantially for full fine-tuning; capacity planning includes
pending gradients and checkpoint rotation removes their sidecar. Native Ming
retains its existing optimizer-payload runtime owner instead of duplicating the
same gradients in another file. Matching optimizer restoration is required before
pending shared gradients are restored. MNT checkpoint boundaries remain completed
batch boundaries. Before the first optimizer update, empty moments are accepted
only with matching optimizer class and ordered parameter names.

Actual batch counts are planned for every epoch when multiplication is active,
including random counts and partial buckets. Epoch totals are not extrapolated
from the first epoch. Resume re-plans counts with the checkpoint's effective seed.
Iterations equal actual batches × MNT; optimizer updates and scheduler advances
continue on their own existing axes. Exposure statistics retain unique target
sample counts as their denominator and record successful repeated sample passes.

The training panel exposes fixed/random controls and an explicitly illustrative
rebatching diagram. Actual counts and plan digest are logged when the epoch plan
is resolved. Edit/save, copying and YAML roundtrips preserve nested settings;
disabling the feature retains the saved multiplier options.

## Verification boundary

CPU tests cover default compatibility, random determinism, rebatching, partial
buckets, caption/worker isolation, occurrence-aware rebasing, public config
roundtrip and pending shared gradients. Native Ming shared-loop tests use batch
size 2, MNT 2 and GAS 3 and compare continuous execution with mid-window resume:
conditioning, latents, noise, timesteps and final weights match exactly on CPU for
swap/fixed and CPU-prefetch/random paths.

On 2026-10-05, real Ming Design INT8 LoRA GPU runs #173 (fixed 2) and #174
(random 2–3) used batch size 2, MNT 2, GAS 3, two-batch encoder windows and
conditioning dropout 0.5. Both completed four steps, saved pending CUDA gradients,
then resumed to step eight with the same plan digest and finite losses. With 404
source images, fixed planning produced 808 exposures / 462 batches, and random
planning produced 1008 exposures / 623 batches. #174 used one base batch per
focus segment, exercising multiplier occurrences in the initial training window.
GPU continuation was checked functionally; exact trace/weight equality was checked
on CPU. Other architectures were covered through shared-path tests, not real-model
GPU runs. Existing #172 parameters were preserved.

The regression suite passed 168 tests; focused checks also cover YAML rejection
before model loading and restoring an empty pre-update optimizer. Backend modules
passed compilation and real imports without creating a CUDA context. Frontend
types, default merging, fixed/random controls, edit/copy/save and input error paths
were reviewed statically; owner-run build/type-check and visual confirmation remain.
Concept acquisition and image quality improvements are unmeasured.

See the original [concept ordering contract](../plans/CONCEPT_BATCH_ORDER_DESIGN_2026-09.md)
for category matching and unsupported task combinations.
