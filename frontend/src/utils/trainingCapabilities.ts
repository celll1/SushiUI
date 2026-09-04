// Reading the backend's capability matrix: what a training run may pick for an
// architecture, and the reason for anything it may not. Pure functions over the
// served payload -- no request, no client -- so they live beside api.ts rather
// than inside it. api.ts re-exports them, so callers import either.
//
// Types come through `import type`, which TypeScript erases, so there is no
// runtime cycle back into api.ts.

import type {
  AdapterFamily,
  ArchCapabilities,
  TrainingFeatureAdvisory,
  TrainingFeatureRefusal,
  TrainingRequiredValue,
} from "./api";

// The algebras a training run may pick for `arch`. Fails closed to "lora",
// which every architecture has always offered, so an unknown arch loses
// nothing; the other direction would offer one the backend refuses.
export const trainableAdapterAlgorithms = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): Array<"lora" | "loha" | "lokr"> => {
  const families = arch ? caps?.adapter_families?.[arch]?.trainable : undefined;
  if (!families) return ["lora"];
  const offered = ["lora", "loha", "lokr"] as const;
  const trainable = offered.filter((name) => families.includes(name));
  return trainable.length ? [...trainable] : ["lora"];
};

// The decomposed family name for a base algebra: DoRA is an epilogue on three
// algebras, not a fourth, so the pair is (algorithm, weight_decompose).
export const decomposedAdapterFamily = (
  algorithm: string | null | undefined
): AdapterFamily =>
  ({ lora: "dora", loha: "doha", lokr: "dokr" } as Record<string, AdapterFamily>)[
    algorithm ?? "lora"
  ] ?? "dora";

// Whether a TRAINING run on `arch` may set `weight_decompose` with this base
// algebra. Fails CLOSED, like `trainableAdapterAlgorithms`: an unknown arch or
// an unloaded matrix hides the control rather than offering a run the backend
// refuses before the model loads.
export const weightDecomposeTrainable = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  algorithm: string | null | undefined
): boolean => {
  const families = arch ? caps?.adapter_families?.[arch]?.trainable : undefined;
  return !!families?.includes(decomposedAdapterFamily(algorithm));
};

// The backend's own sentence for an algebra this architecture will not train,
// for a tooltip beside a choice that is not offered.
export const adapterTrainingRefusalReason = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  algorithm: AdapterFamily
): string | undefined =>
  arch ? caps?.adapter_families?.[arch]?.untrainable?.[algorithm] : undefined;

// One training-config feature's refusal for one architecture.
export interface TrainingFeatureRefusal {
  reason: string;
  // Training methods the refusal applies to; absent = all of them (Z-Image
  // trains no text encoder under LoRA while a full fine-tune does).
  methods?: string[];
}

// The reason `feature` cannot run for `arch` under `method`, or undefined when
// it can. Undefined for an unknown arch or an unloaded matrix: the control stays
// visible and the backend refuses the run, which is recoverable — a control that
// vanishes because the frontend has never heard of the architecture is not.
export const trainingFeatureUnsupportedReason = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  method?: string | null
): string | undefined => {
  if (!arch) return undefined;
  const entry = caps?.training_feature_unsupported?.[arch]?.[feature];
  if (!entry) return undefined;
  if (entry.methods && method && !entry.methods.includes(method)) return undefined;
  return entry.reason;
};

// Whether `arch`'s training-sample path READS `parameter`, i.e. whether its
// control is worth offering. Same direction as trainingFeatureUnsupportedReason
// above, and for the same reason: an unknown arch or an unloaded matrix answers
// TRUE, so the control stays visible and the value is simply not written for an
// architecture that turns out not to read it. The opposite direction makes every
// sample control vanish on a startup fetch that has not landed yet — including
// the sampler and schedule selects, which were unconditional before this gate
// existed — and a control that disappears because the frontend has not heard
// back from the backend is not recoverable by the user.
//
// Matches api/arch_capabilities.training_sample_key_supported, which gates the
// generated YAML and the sample PNG's metadata on the backend and fails open on
// the same input.
export const trainingSampleParameterSupported = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  parameter: string
): boolean => {
  const table = caps?.training_sample_supported_params;
  if (!arch || !table || !table[arch]) return true;
  return table[arch].includes(parameter);
};

export const trainingSampleNote = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): string | undefined => arch ? caps?.training_sample_notes?.[arch] : undefined;

// What an OMITTED cfg_uncond_drop_rate resolves to on `arch`, or undefined when
// the architecture has no default (the mechanism is not in play there, or the
// matrix has not loaded). Never a literal in a component: the number is the
// backend's, and a second copy of it is what turns an explicit 0 back into 0.1.
export const cfgUncondDropDefault = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined
): number | undefined => {
  if (!arch) return undefined;
  return caps?.cfg_uncond_drop_defaults?.[arch];
};

// What the backend says ABOUT a training feature it does implement.
export interface TrainingFeatureAdvisory {
  // "high_memory" — the reason carries measured numbers; "experimental" — the
  // path is implemented and thinly measured. Neither is a gate.
  level: "experimental" | "high_memory";
  reason: string;
  methods?: string[];
}

// The advisory for `feature` on `arch` under `method`, or undefined when there
// is none. NEVER a reason to hide or disable a control: the backend accepts and
// runs the feature, so a caller that treats this like a refusal recreates the
// contradiction the axis exists to end.
export const trainingFeatureAdvisory = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  feature: string,
  method?: string | null
): TrainingFeatureAdvisory | undefined => {
  if (!arch) return undefined;
  const entry = caps?.training_feature_advisory?.[arch]?.[feature];
  if (!entry) return undefined;
  if (entry.methods && method && !entry.methods.includes(method)) return undefined;
  return entry;
};

// One training-config parameter's required value for one architecture.
export interface TrainingRequiredValue {
  value: string | number | boolean;
  reason: string;
  // Training methods the requirement applies to; absent = all of them.
  methods?: string[];
  // The full admitted set when the contract admits more than one value; `value`
  // is then its default member. Absent = `value` is the only legal one. A
  // control offers exactly these and leaves a current member alone.
  values?: (string | number | boolean)[];
  // The config that LIFTS the requirement (all pairs must hold). Present only
  // for a CONDITIONAL requirement: SenseNova's batch_size=1 holds unless
  // `enable_bucketing` is on. Resolved by passing the run's params below.
  unless?: Record<string, string | number | boolean>;
}

// The config values `arch` requires under `method`, param -> {value, reason}.
// Empty for an unknown arch or an unloaded matrix: unconstrained, so a control
// keeps its own default and the backend refuses the run if that is wrong —
// recoverable, where a control pinned to a value invented here is not.
//
// `params` resolves the CONDITIONAL entries (`unless`). Without it they are
// omitted rather than returned: every consumer here pins a control, and a pin
// the config may already have lifted would disable a supported configuration.
export const trainingRequiredValues = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  method?: string | null,
  params?: Record<string, any> | null
): Record<string, TrainingRequiredValue> => {
  if (!arch) return {};
  const entries = caps?.training_required_values?.[arch];
  if (!entries) return {};
  const out: Record<string, TrainingRequiredValue> = {};
  for (const [param, entry] of Object.entries(entries)) {
    if (entry.methods && method && !entry.methods.includes(method)) continue;
    if (entry.unless) {
      // No config to read the lift off: omit, rather than pin a control the
      // run may already have lifted. TrainingConfig.tsx passes the lift params.
      if (!params) continue;
      if (Object.entries(entry.unless).every(([key, value]) => params[key] === value)) continue;
    }
    out[param] = entry;
  }
  return out;
};

// The reason `method` is refused for `arch`, or undefined when it is offered.
// Used to disable a training-method control AND to title it with the backend's
// own wording rather than a second copy of it in the UI.
export const trainingMethodUnsupportedReason = (
  caps: ArchCapabilities | null | undefined,
  arch: string | null | undefined,
  method: string
): string | undefined => {
  if (!arch) return undefined;
  return caps?.training_unsupported?.[arch]?.[method];
};
