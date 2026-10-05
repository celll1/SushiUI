# Training Edit Config preservation audit (2026-10-05)

## Confirmed defects and repairs

Run 172's INT8 Full initialization rejection was reachable through an unchanged
Edit Config save: `full_finetune_dequantize_int8_base` was missing from the
frontend restore list, so the saved true became the default false. The same
omission affected `training_export_format` and five Chimera fields:
`chimera_flow_version`, `chimera_v2_parameterization`, `chimera_v2_latent_mean`,
`chimera_v2_latent_centered_second_moment`, `chimera_v2_calibration`.
All seven now participate in restoration, request transport and presets.

Other confirmed preservation hazards repaired:

- Ming Full requests omitted `qwen_convrot_training_forward`, replacing an
  explicit saved value with the API default; preserve the existing field.
- Explicit empty `sample_prompts` were discarded, leaving default prompts;
  restore an empty list while retaining omission semantics for old presets.
- Delayed bundle-VAE defaults could apply after the restore timer cleared;
  mark the restored model synchronously, as for timestep/guidance defaults.
- Model/capability-dependent method, adapter, feature, reference and required
  value effects could rewrite loaded values after the short restore timer.
  Protect the restored model/method identity until the user changes that recipe.
  Preserve optimizer settings until the optimizer is explicitly changed.
  Incompatible saved values remain available for inspection/API validation.
- Save was available while an edit fetch was pending or had failed. Disable
  submission until the current edit run is successfully restored; reopening the
  editor retries a failed load. Copy-fetch failure also releases its restore flag.
- Legacy `qwen_partition_*` train keys were emitted alongside synthesized
  default `dit_partition_*` values. Populate missing canonical values from the
  legacy YAML before request validation. Explicit conflicting saved canonical
  values remain conflicts; do not conceal them.

## Verification and boundaries

Static inventory covered the defaulted form fields and every top-level frontend
`TrainingRunCreateRequest` field. Separate owners are intentional: legacy
dataset/network identity is reconstructed through dataset configs/training
method, VAE training has its own editor, and Qwen partition aliases are restored
to canonical keys by the API. Guard tests fail if a new field has neither a
restore path nor an explicitly accounted-for owner.

Six Node runtime tests execute the real restore callback, pass-through and preset
functions, including nondefault Full/Chimera data, empty prompts, late bundle
defaults and six late normalization effects. TypeScript transpilation here only
executes the tests; it is not a frontend build or type check. Initial backend
round-trip/preservation/config suite: 48 passed plus 32 subtests; new legacy
partition cases are checked separately. Routes compile and really import with
CUDA initialization stubbed. No GPU training or numerical model change is part
of this repair.

Run 172's INT8 materialization opt-in was restored via the API before this audit;
other run values were not globally rewritten. Past dropped values cannot be
reconstructed safely without a known backup. This audit does not claim every
architecture's complete browser interaction was exercised. Owner validation:
reopen Edit Config for run 172, confirm the materialization checkbox, save without
changes and reopen to confirm persistence; also verify nondefault export and
Chimera values and an intentionally empty sample-prompt list. Frontend build and
type checking remain owner-run.
