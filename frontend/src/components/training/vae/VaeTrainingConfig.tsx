"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  createVaeTrainingRun,
  updateVaeTrainingRun,
  getTrainingRunParams,
  listDatasets,
  getModels,
  fetchVaes,
  Dataset,
  TrainingRun,
  VaeEntry,
  VaeTrainingConfig as VaeConfig,
  VaeTrainingRunCreateRequest,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import NumberInput from "../../common/NumberInput";

interface VaeTrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TrainingRun) => void;
  onRunUpdated?: (run: TrainingRun) => void;
  /** When set, the form edits this existing run instead of creating a new one. */
  editRunId?: number | null;
}

interface ModelListEntry {
  name: string;
  path: string;
  architecture?: string;
  is_video?: boolean;
}

/**
 * Architectures whose VAE this trainer cannot consume, so they are not offered
 * as a base model while the base VAE comes from the model itself.
 *
 * The trainer's forward is a 4-D `[B, 3, H, W]` `encode().latent_dist` ->
 * `decode().sample` on a module with a `.decoder`
 * (core/training/vae/vae_trainer.py). Per the per-arch VAE class table in
 * api/generation_overrides.py `VAE_CLASS_BY_ARCH`:
 *   - anima / krea2  -> AutoencoderKLQwenImage, whose `_encode` unpacks
 *                       `(B, C, num_frames, H, W)`, i.e. 5-D;
 *   - ltx2           -> AutoencoderKLLTXVideo, also 5-D;
 *   - minit2i        -> no VAE at all (pixel space).
 * Everything else (sd15 / sdxl / zimage -> AutoencoderKL, flux2 / lens /
 * ideogram4 -> AutoencoderKLFlux2) is 4-D and exposes the required interface.
 * Unknown/new architectures are NOT hidden: the loader is generic over any
 * diffusers `Autoencoder*` class, so only the known-incompatible ones are
 * excluded here.
 */
const NON_TRAINABLE_VAE_ARCHS = new Set(["anima", "krea2", "ltx2", "minit2i"]);

const isVaeTrainableModel = (m: ModelListEntry): boolean =>
  m.is_video !== true &&
  !NON_TRAINABLE_VAE_ARCHS.has((m.architecture || "").toLowerCase());

/**
 * Fallback defaults, used only until GET /schema/vae-training-defaults answers
 * (i.e. while the backend is still starting). Kept byte-identical to
 * backend/api/param_defaults.py VAE_TRAINING_DEFAULTS, which is the single
 * source of truth; the fetched values overwrite these on arrival.
 */
const DEFAULT_VAE_CONFIG: VaeConfig = {
  batch_size: 1,
  total_steps: 2000,
  gradient_accumulation_steps: 1,
  learning_rate: 1e-5,
  optimizer: "adamw",
  optimizer_weight_decay: 0.001,
  max_grad_norm: 0.1,
  lr_scheduler: "constant",
  lr_warmup_steps: 0,
  seed: 42,
  num_workers: 2,
  save_every: 500,
  max_step_saves_to_keep: 3,
  vae_source: "model",
  vae_path: "",
  vae_arch: "sdxl",
  train_decoder: true,
  decoder_blocks: "all",
  train_encoder: false,
  acknowledge_latent_space_break: false,
  encoder_blocks: "all",
  resolution: 512,
  crop_scale_policy: "downscale",
  crop_scale_max_downscale: 0.0,
  dtype: "bf16",
  ema_enabled: true,
  ema_decay: 0.999,
  mse_weight: 1.0,
  l1_weight: 0.0,
  lpips_weight: 0.1,
  lpips_net: "vgg",
  ycbcr_dc_weight: 0.1,
  ycbcr_dc_y_weight: 0.25,
  ycbcr_dc_chroma_weight: 1.0,
  ycbcr_dc_eps: 0.001,
  pattern_weight: 0.0,
  pattern_size: 8,
  l_invented_weight: 0.0,
  l_invented_y_weight: 1.0,
  l_invented_chroma_weight: 0.25,
  l_invented_flat_t_y: 2.0,
  l_invented_flat_t_c: 1.25,
  kl_weight: 1e-6,
  export_bare_ldm: false,
  validation_every: 100,
  validation_num_images: 8,
  validation_resolution: 1024,
};

// Loss weights the backend checks: if every one of them is 0 the run is refused
// (no training signal). Mirrored here so the form can say so before submitting.
const LOSS_WEIGHT_KEYS: (keyof VaeConfig)[] = [
  "mse_weight", "l1_weight", "lpips_weight", "ycbcr_dc_weight", "pattern_weight",
  "l_invented_weight",
];

// Optimizers resolvable by OptimizerFactory without a trainer-provided
// allocator (the *_ringbuffer variants need one, which this trainer does not
// build, so they are not offered).
const OPTIMIZERS = [
  "adamw", "adamw8bit", "adafactor", "lion8bit",
  "paged_adamw", "paged_adamw8bit", "paged_lion8bit",
];

// diffusers get_scheduler names.
const LR_SCHEDULERS = [
  "constant", "constant_with_warmup", "linear", "cosine",
  "cosine_with_restarts", "polynomial",
];

const inputClass =
  "w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500";
const numberClass =
  "w-32 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500";

export default function VaeTrainingConfig({
  onClose,
  onRunCreated,
  onRunUpdated,
  editRunId,
}: VaeTrainingConfigProps) {
  const isEditMode = !!editRunId;
  const { vaeTrainingDefaults } = useStartup();

  const [cfg, setCfg] = useState<VaeConfig>(DEFAULT_VAE_CONFIG);
  const [runName, setRunName] = useState("");
  const [baseModelPath, setBaseModelPath] = useState("");
  const [resumeFrom, setResumeFrom] = useState("");
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>([]);

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<ModelListEntry[]>([]);
  const [vaes, setVaes] = useState<VaeEntry[]>([]);

  const [loadingRun, setLoadingRun] = useState(isEditMode);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Precedence, in increasing order: DEFAULT_VAE_CONFIG (offline fallback) <
  // /schema/vae-training-defaults < this run's stored values (/params) < the
  // user's edits. The last two are kept in refs because StartupContext only
  // fetches the defaults once a model has finished loading, which can land
  // after this panel mounted and after /params answered.
  const paramsPatchRef = useRef<Partial<VaeConfig> | null>(null);
  const touchedRef = useRef(false);
  const defaultsRef = useRef<Partial<VaeConfig> | null>(null);
  defaultsRef.current = (vaeTrainingDefaults as Partial<VaeConfig> | null) ?? null;

  const setField = <K extends keyof VaeConfig>(key: K, value: VaeConfig[K]) => {
    touchedRef.current = true;
    setCfg((prev) => ({ ...prev, [key]: value }));
  };

  // Backend-fetched defaults (single source of truth). Applied in edit mode too
  // -- otherwise any key the run does not carry would keep the hardcoded
  // fallback even though the endpoint answered -- but always underneath the
  // run's own values. Skipped once the user has edited anything, so a late
  // arrival cannot overwrite what they typed.
  useEffect(() => {
    if (!vaeTrainingDefaults || touchedRef.current) return;
    setCfg((prev) => ({
      ...prev,
      ...(vaeTrainingDefaults as Partial<VaeConfig>),
      ...(paramsPatchRef.current ?? {}),
    }));
  }, [vaeTrainingDefaults]);

  useEffect(() => {
    listDatasets()
      .then((res) => setDatasets(res.datasets || []))
      .catch((err) => console.error("[VaeTrainingConfig] Failed to load datasets:", err));
    fetchVaes()
      .then((res) => setVaes((res.vaes || []).filter((v) => v.kind !== "pid_decoder")))
      .catch((err) => console.error("[VaeTrainingConfig] Failed to load VAEs:", err));
  }, []);

  useEffect(() => {
    getModels()
      .then((res) => {
        const list: ModelListEntry[] = res.models || [];
        setModels(list);
        // Auto-select the first model whose VAE this trainer can consume (the
        // default vae_source is "model", so an untrainable auto-selection would
        // only be discovered after the run was started).
        const firstTrainable = list.find(isVaeTrainableModel);
        if (!isEditMode && firstTrainable) {
          setBaseModelPath((prev) => prev || firstTrainable.path);
        }
      })
      .catch((err) => console.error("[VaeTrainingConfig] Failed to load models:", err));
  }, [isEditMode]);

  const loadRun = useCallback(async (runId: number) => {
    setLoadingRun(true);
    try {
      const params = await getTrainingRunParams(runId);
      setRunName(params.run_name || "");
      setBaseModelPath(params.base_model_path || "");
      setResumeFrom(params.resume_from_checkpoint || "");
      setSelectedDatasetIds((params.dataset_configs || []).map((d) => d.dataset_id));
      // process.vae carries every VAE-specific key (including seed and
      // num_workers), so this is the bulk of the run's config; the remaining
      // run-shape knobs live in process.train / process.save and come back as
      // flat fields.
      const patch: Partial<VaeConfig> = { ...(params.vae_config ?? {}) };
      if (params.total_steps != null) patch.total_steps = params.total_steps;
      if (params.batch_size != null) patch.batch_size = params.batch_size;
      if (params.gradient_accumulation_steps != null) patch.gradient_accumulation_steps = params.gradient_accumulation_steps;
      if (params.learning_rate != null) patch.learning_rate = params.learning_rate;
      if (params.optimizer != null) patch.optimizer = params.optimizer;
      if (params.optimizer_weight_decay != null) patch.optimizer_weight_decay = params.optimizer_weight_decay;
      if (params.max_grad_norm != null) patch.max_grad_norm = params.max_grad_norm;
      if (params.lr_scheduler != null) patch.lr_scheduler = params.lr_scheduler;
      if (params.lr_warmup_steps != null) patch.lr_warmup_steps = params.lr_warmup_steps;
      if (params.save_every != null) patch.save_every = params.save_every;
      if (params.max_step_saves_to_keep != null) patch.max_step_saves_to_keep = params.max_step_saves_to_keep;

      // Remembered so a late-arriving schema fetch re-applies the run's values
      // on top of the defaults instead of over them.
      paramsPatchRef.current = patch;
      setCfg((prev) => ({
        ...prev,
        ...(defaultsRef.current ?? {}),
        ...patch,
      }));
    } catch (err: unknown) {
      console.error("[VaeTrainingConfig] Failed to load run params:", err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingRun(false);
    }
  }, []);

  useEffect(() => {
    if (editRunId) loadRun(editRunId);
  }, [editRunId, loadRun]);

  const toggleDataset = (id: number) =>
    setSelectedDatasetIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const activeLossCount = LOSS_WEIGHT_KEYS.filter((k) => Number(cfg[k]) > 0).length;

  // With vae_source "model" the base model's own VAE is what gets trained, so
  // only models whose VAE this trainer can consume are offered. With "path" or
  // "store" the base model is just the run's (route-validated) model reference
  // and its VAE is never loaded, so the full list stays available.
  const trainableModels = models.filter(isVaeTrainableModel);
  const selectableModels = cfg.vae_source === "model" ? trainableModels : models;
  const hiddenModelCount = models.length - trainableModels.length;
  const selectedModel = models.find((m) => m.path === baseModelPath);
  const selectedModelIsUntrainable =
    cfg.vae_source === "model" && !!selectedModel && !isVaeTrainableModel(selectedModel);

  const handleSave = async () => {
    if (selectedDatasetIds.length === 0) {
      setError("At least one dataset must be selected.");
      return;
    }
    if (!baseModelPath.trim()) {
      setError("Base model is required.");
      return;
    }
    if (selectedModelIsUntrainable) {
      setError(
        `${selectedModel?.architecture?.toUpperCase() || "The selected model"} does not have a VAE ` +
        "this trainer can fine-tune (video/5-D autoencoder, or pixel-space with no VAE). Pick " +
        "another base model, or set Base VAE source to 'Explicit path' / 'Shared VAE store'."
      );
      return;
    }
    if (cfg.vae_source === "path" && !cfg.vae_path.trim()) {
      setError("Base VAE source is 'Explicit path' but no path was given.");
      return;
    }
    if (cfg.vae_source === "store" && !cfg.vae_arch.trim()) {
      setError("Base VAE source is 'Shared VAE store' but no store key was given.");
      return;
    }
    if (activeLossCount === 0) {
      setError("All loss weights are 0: there is no training signal. Set at least one above 0.");
      return;
    }
    if (cfg.train_encoder && !cfg.acknowledge_latent_space_break) {
      setError(
        "Encoder training requires the acknowledgement checkbox under \"What to train\". " +
        "The backend refuses train_encoder without it."
      );
      return;
    }
    // A "max downscale" below 1 would name an UPSCALE bound, which the knob does
    // not mean; the backend refuses it outright rather than clamping, so say so
    // here instead of round-tripping the refusal.
    if (
      cfg.crop_scale_policy === "mixed" &&
      cfg.crop_scale_max_downscale > 0 &&
      cfg.crop_scale_max_downscale < 1
    ) {
      setError(
        "Max downscale is a downscale factor, so it must be 0 (unbounded) or at least 1.0. " +
        "1.0 means \"never downscale\", which is what the 'native' crop scale policy says."
      );
      return;
    }

    setSaving(true);
    setError(null);
    try {
      const payload: VaeTrainingRunCreateRequest = {
        dataset_configs: selectedDatasetIds.map((id) => ({
          dataset_id: id,
          caption_types: [],
          filters: {},
        })),
        run_name: runName.trim() || undefined,
        training_method: "vae_decoder",
        base_model_path: baseModelPath.trim(),
        // The create route needs a top-level step count for the DB column; the
        // same value is repeated inside vae_config, which is what the trainer
        // actually reads.
        total_steps: cfg.total_steps,
        resume_from_checkpoint: resumeFrom.trim() || null,
        vae_config: {
          ...cfg,
          // Only meaningful for vae_source "path"; cleared otherwise so a stale
          // value cannot shadow the run's own base model.
          vae_path: cfg.vae_source === "path" ? cfg.vae_path.trim() : "",
          // The panel always trains the decoder: train_decoder=false is only
          // reachable through a hand-written config, and the backend refuses it
          // both on its own (nothing trainable) and together with
          // train_encoder=true (encoder-only under a frozen decoder).
          train_decoder: true,
          train_encoder: cfg.train_encoder,
          // Never send a stale acknowledgement: the backend refuses it without
          // train_encoder, and it must not be able to authorise a later run.
          acknowledge_latent_space_break:
            cfg.train_encoder && cfg.acknowledge_latent_space_break,
          export_bare_ldm: cfg.train_encoder ? false : cfg.export_bare_ldm,
          // Read only by the "mixed" per-sample draw, and HARD-REFUSED by the
          // backend under any other policy (a knob nothing reads must not be
          // silently recorded). The policy handler clears it too; this spread is
          // the defensive layer, because /params on an edited run and the
          // late-arriving schema defaults can both set the policy without going
          // through that handler.
          crop_scale_max_downscale:
            cfg.crop_scale_policy === "mixed" ? cfg.crop_scale_max_downscale : 0,
        },
      };

      const run = isEditMode
        ? await updateVaeTrainingRun(editRunId as number, payload)
        : await createVaeTrainingRun(payload);

      if (isEditMode) {
        (onRunUpdated || onRunCreated)(run);
      } else {
        onRunCreated(run);
      }
    } catch (err: unknown) {
      const anyErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(anyErr?.response?.data?.detail || anyErr?.message || "Failed to save VAE training run");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
        <h2 className="text-lg font-semibold">
          {isEditMode ? "Edit VAE Training Run" : "New VAE Training Run"}
        </h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {loadingRun && (
          <div className="text-sm text-gray-400">Loading run parameters...</div>
        )}

        {/* Scope */}
        <section className="border border-gray-700 rounded p-3 bg-gray-800/40">
          <p className="text-sm text-gray-300">
            Fine-tunes the VAE <b>decoder</b> with the <b>encoder frozen</b>, so the latent
            space that cached latents, LoRAs and diffusion models depend on is unchanged.
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Encoder training is available under &quot;What to train&quot; and requires a separate
            acknowledgement: it changes the latent distribution, so existing latent caches,
            LoRAs and diffusion checkpoints no longer match the result.
          </p>
        </section>

        {/* Run name */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">Run Name</label>
          <input
            type="text"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            disabled={isEditMode}
            placeholder="leave empty for an auto-generated name"
            className={`${inputClass} disabled:opacity-50`}
          />
          {isEditMode && (
            <p className="text-xs text-gray-500 mt-1">The run name cannot be changed after creation.</p>
          )}
        </section>

        {/* Base model */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">Base Model</label>
          <select
            value={selectableModels.some((m) => m.path === baseModelPath) ? baseModelPath : ""}
            onChange={(e) => setBaseModelPath(e.target.value)}
            className={inputClass}
          >
            <option value="">Select a model...</option>
            {selectableModels.map((m) => (
              <option key={m.path} value={m.path}>
                {m.name}
                {m.architecture ? ` (${m.architecture.toUpperCase()})` : ""}
              </option>
            ))}
          </select>
          <input
            type="text"
            value={baseModelPath}
            onChange={(e) => setBaseModelPath(e.target.value)}
            placeholder="or type a model path"
            className={`${inputClass} mt-2 font-mono text-xs`}
          />
          <p className="text-xs text-gray-500 mt-1">
            The run&apos;s base model. With Base VAE source set to &quot;Model&quot;, this model&apos;s VAE
            is the one that gets fine-tuned.
          </p>
          {cfg.vae_source === "model" && hiddenModelCount > 0 && (
            <p className="text-xs text-gray-500 mt-1">
              {hiddenModelCount} model{hiddenModelCount === 1 ? " is" : "s are"} not listed: their
              VAE is a video/5-D autoencoder or the model is pixel-space, which this trainer&apos;s
              4-D encode/decode step cannot consume. Set Base VAE source to &quot;Explicit path&quot; or
              &quot;Shared VAE store&quot; to train a different VAE with one of them as the run&apos;s base
              model.
            </p>
          )}
          {selectedModelIsUntrainable && (
            <p className="text-xs text-red-400 mt-1">
              {selectedModel?.architecture?.toUpperCase() || "This model"} does not have a VAE this
              trainer can fine-tune (video/5-D autoencoder, or pixel-space with no VAE). Pick
              another model, or set Base VAE source to &quot;Explicit path&quot; / &quot;Shared VAE store&quot;.
            </p>
          )}
        </section>

        {/* Datasets */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Datasets</label>
          {datasets.length === 0 ? (
            <p className="text-sm text-gray-500">Loading datasets...</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {datasets.map((dataset) => {
                const selected = selectedDatasetIds.includes(dataset.id);
                return (
                  <label
                    key={dataset.id}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded border cursor-pointer transition-colors text-sm ${
                      selected
                        ? "border-blue-500 bg-blue-900/30 text-white"
                        : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={() => toggleDataset(dataset.id)}
                      className="accent-blue-500"
                    />
                    <span className="font-medium">{dataset.name}</span>
                    <span className="text-xs text-gray-400">
                      {dataset.total_items.toLocaleString()} imgs
                    </span>
                  </label>
                );
              })}
            </div>
          )}
          <p className="text-xs text-gray-500 mt-1">
            Raw images only: training is a live encode/decode forward on pixels, so captions
            and cached latents are not used.
          </p>
        </section>

        {/* Base VAE source */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Base VAE</h3>
          <div className="flex flex-wrap gap-2">
            {([
              ["model", "Model (use the base model's VAE)"],
              ["path", "Explicit path"],
              ["store", "Shared VAE store"],
            ] as const).map(([value, label]) => (
              <button
                key={value}
                onClick={() => setField("vae_source", value)}
                className={`px-3 py-1.5 rounded text-sm border transition-colors ${
                  cfg.vae_source === value
                    ? "border-blue-500 bg-blue-600 text-white"
                    : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {cfg.vae_source === "path" && (
            <div>
              <select
                value={vaes.some((v) => v.path === cfg.vae_path) ? cfg.vae_path : ""}
                onChange={(e) => setField("vae_path", e.target.value)}
                className={inputClass}
              >
                <option value="">Select a VAE...</option>
                {vaes.map((v) => (
                  <option key={v.path} value={v.path}>
                    {v.name}
                    {v.arch ? ` (${v.arch})` : ""}
                  </option>
                ))}
              </select>
              <input
                type="text"
                value={cfg.vae_path}
                onChange={(e) => setField("vae_path", e.target.value)}
                placeholder="or type a diffusers VAE directory / .safetensors path"
                className={`${inputClass} mt-2 font-mono text-xs`}
              />
            </div>
          )}

          {cfg.vae_source === "store" && (
            <div>
              <input
                type="text"
                value={cfg.vae_arch}
                onChange={(e) => setField("vae_arch", e.target.value)}
                placeholder="sdxl"
                className={inputClass}
              />
              <p className="text-xs text-gray-500 mt-1">
                Shared-VAE-store key (for example <code>sdxl</code>, <code>sd15</code>,{" "}
                <code>flux1</code>, <code>flux2</code>, <code>qwen_image</code>). The store&apos;s
                <code> sdxl</code> entry is madebyollin/sdxl-vae-fp16-fix, whose fp16 safety comes
                from a weight rescaling that fine-tuning does not preserve; the trainer logs a
                warning when it detects that base.
              </p>
            </div>
          )}
        </section>

        {/* What to train */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">What to train</h3>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Decoder blocks</label>
            <select
              value={cfg.decoder_blocks}
              onChange={(e) => setField("decoder_blocks", e.target.value as VaeConfig["decoder_blocks"])}
              className={inputClass}
            >
              <option value="all">all</option>
              <option value="up_blocks">up_blocks</option>
              <option value="mid_block">mid_block</option>
              <option value="conv_out">conv_out</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Which part of the decoder to unfreeze. <code>all</code> also includes
              <code> post_quant_conv</code>, which is part of the decode path. A checkpoint can
              only be resumed by a run with the same setting.
            </p>
          </div>
          {/* Encoder training: double gate. The two checkboxes are separate,
              deliberate actions, and neither is pre-checked. Turning the first
              one off clears the acknowledgement, so it can never be left set
              from an earlier edit (the backend refuses that combination too). */}
          <div className="border-t border-gray-700 pt-3 space-y-2">
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={cfg.train_encoder}
                onChange={(e) => {
                  const on = e.target.checked;
                  touchedRef.current = true;
                  setCfg((prev) => ({
                    ...prev,
                    train_encoder: on,
                    // Both directions of the gate are enforced here, matching
                    // the backend: no acknowledgement without encoder training,
                    // and no bare-LDM export with it.
                    acknowledge_latent_space_break: on
                      ? prev.acknowledge_latent_space_break
                      : false,
                    export_bare_ldm: on ? false : prev.export_bare_ldm,
                  }));
                }}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
              />
              <span className="text-sm text-gray-300">Train the encoder as well</span>
            </label>
            <p className="text-xs text-gray-500">
              The encoder defines the latent distribution. Training it means:
            </p>
            <ul className="text-xs text-gray-500 list-disc pl-6 space-y-0.5">
              <li>every cached latent produced with the original VAE no longer matches this one and has to be re-encoded;</li>
              <li>LoRAs and diffusion checkpoints trained against the original VAE were trained on latents this VAE does not produce;</li>
              <li>the result is a new VAE, not a drop-in replacement for the base model&apos;s VAE.</li>
            </ul>
            <p className="text-xs text-gray-500">
              The run writes to <code>&lt;run_name&gt;_vae_encoder_trained</code> instead of{" "}
              <code>&lt;run_name&gt;_vae</code>, records <code>encoder_trained: true</code> in the
              provenance sidecar, and the bare LDM <code>.safetensors</code> export is refused
              (that format carries no scaling/shift values of its own).
            </p>

            {cfg.train_encoder && (
              <div className="pl-4 border-l border-gray-600 space-y-3 pt-1">
                <label className="flex items-start gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={cfg.acknowledge_latent_space_break}
                    onChange={(e) => setField("acknowledge_latent_space_break", e.target.checked)}
                    className="w-4 h-4 mt-0.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
                  />
                  <span className="text-sm text-gray-300">
                    I acknowledge that existing latent caches, LoRAs and diffusion checkpoints
                    will not match the VAE this run produces.
                  </span>
                </label>
                {!cfg.acknowledge_latent_space_break && (
                  <p className="text-xs text-yellow-400">
                    Required: the backend refuses <code>train_encoder</code> without this
                    acknowledgement.
                  </p>
                )}

                <div>
                  <label className="block text-xs text-gray-400 mb-1">Encoder blocks</label>
                  <select
                    value={cfg.encoder_blocks}
                    onChange={(e) => setField("encoder_blocks", e.target.value as VaeConfig["encoder_blocks"])}
                    className={inputClass}
                  >
                    <option value="all">all</option>
                    <option value="down_blocks">down_blocks</option>
                    <option value="mid_block">mid_block</option>
                    <option value="conv_out">conv_out</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Which part of the encoder to unfreeze. <code>all</code> also includes
                    <code> quant_conv</code>, which is part of the encode path.
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <label className="text-xs text-gray-400 w-40">KL weight</label>
                  <NumberInput
                    min={0} step={1e-6} parse="float"
                    value={cfg.kl_weight}
                    defaultValue={DEFAULT_VAE_CONFIG.kl_weight}
                    onCommit={(v) => setField("kl_weight", v)}
                    className={numberClass}
                  />
                  <span className="text-xs text-gray-500">Posterior KL term (the LDM value).</span>
                </div>
                <p className="text-xs text-gray-500 -mt-1">
                  Only applied while the encoder is trainable. With the encoder frozen the term
                  is constant with respect to every trainable parameter, so it is not
                  constructed and this value is ignored. The latent is sampled from the
                  posterior instead of taken at its mode while the encoder trains.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Losses */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Losses</h3>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">MSE weight</label>
            <NumberInput
              min={0} step={0.1} parse="float"
              value={cfg.mse_weight}
              defaultValue={DEFAULT_VAE_CONFIG.mse_weight}
              onCommit={(v) => setField("mse_weight", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">L2 reconstruction term (sd-vae-ft-mse&apos;s base term).</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">L1 weight</label>
            <NumberInput
              min={0} step={0.1} parse="float"
              value={cfg.l1_weight}
              defaultValue={DEFAULT_VAE_CONFIG.l1_weight}
              onCommit={(v) => setField("l1_weight", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">L1 reconstruction term (the LDM / ft-EMA term). 0 by default.</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">LPIPS weight</label>
            <NumberInput
              min={0} step={0.05} parse="float"
              value={cfg.lpips_weight}
              defaultValue={DEFAULT_VAE_CONFIG.lpips_weight}
              onCommit={(v) => setField("lpips_weight", v)}
              className={numberClass}
            />
            <select
              value={cfg.lpips_net}
              onChange={(e) => setField("lpips_net", e.target.value as VaeConfig["lpips_net"])}
              disabled={cfg.lpips_weight <= 0}
              className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white disabled:opacity-50"
            >
              <option value="vgg">vgg</option>
              <option value="alex">alex</option>
              <option value="squeeze">squeeze</option>
            </select>
          </div>
          <p className="text-xs text-gray-500 -mt-1">
            0.1 is the published value from sd-vae-ft-mse. This term is what creates plausible
            high frequency, so a larger weight works against the artifact this fine-tune targets.
            Requires the <code>lpips</code> package when above 0; the backend checks that before
            training starts.
          </p>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">YCbCr DC weight</label>
            <NumberInput
              min={0} step={0.05} parse="float"
              value={cfg.ycbcr_dc_weight}
              defaultValue={DEFAULT_VAE_CONFIG.ycbcr_dc_weight}
              onCommit={(v) => setField("ycbcr_dc_weight", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">Colour-drift term.</span>
          </div>
          <p className="text-xs text-gray-500 -mt-1">
            Per-pixel Charbonnier on YCbCr with the luma channel downweighted, plus a Charbonnier
            on the per-image, per-channel spatial mean (DC). Measurement on the SDXL VAEs showed
            39-51/255 of red DC drift over 8 encode/decode roundtrips, which is a spatial-mean
            defect a purely per-pixel penalty barely constrains.
          </p>
          {cfg.ycbcr_dc_weight > 0 && (
            <div className="pl-4 border-l border-gray-700 space-y-2">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Luma (Y) weight</label>
                <NumberInput
                  min={0} step={0.05} parse="float"
                  value={cfg.ycbcr_dc_y_weight}
                  defaultValue={DEFAULT_VAE_CONFIG.ycbcr_dc_y_weight}
                  onCommit={(v) => setField("ycbcr_dc_y_weight", v)}
                  className={numberClass}
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Chroma weight</label>
                <NumberInput
                  min={0} step={0.05} parse="float"
                  value={cfg.ycbcr_dc_chroma_weight}
                  defaultValue={DEFAULT_VAE_CONFIG.ycbcr_dc_chroma_weight}
                  onCommit={(v) => setField("ycbcr_dc_chroma_weight", v)}
                  className={numberClass}
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Charbonnier epsilon</label>
                <NumberInput
                  min={0} step={0.001} parse="float"
                  value={cfg.ycbcr_dc_eps}
                  defaultValue={DEFAULT_VAE_CONFIG.ycbcr_dc_eps}
                  onCommit={(v) => setField("ycbcr_dc_eps", v)}
                  className={numberClass}
                />
              </div>
            </div>
          )}

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Pattern weight</label>
            <NumberInput
              min={0} step={0.01} parse="float"
              value={cfg.pattern_weight}
              defaultValue={DEFAULT_VAE_CONFIG.pattern_weight}
              onCommit={(v) => setField("pattern_weight", v)}
              className={numberClass}
            />
            {cfg.pattern_weight > 0 && (
              <>
                <label className="text-xs text-gray-400">Cell size (px)</label>
                <NumberInput
                  min={1} step={1} parse="int"
                  value={cfg.pattern_size}
                  defaultValue={DEFAULT_VAE_CONFIG.pattern_size}
                  onCommit={(v) => setField("pattern_size", v)}
                  className="w-20 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </>
            )}
          </div>
          <p className="text-xs text-gray-500 -mt-1">
            Latent-cell grid-phase penalty. 0 by default: the 8 px grid artifact it targets was
            measured at ratio ~1.0 (i.e. absent) on four production VAEs under three independent
            metric definitions.
          </p>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Invented HF weight</label>
            <NumberInput
              min={0} max={10} step={0.1} parse="float"
              value={cfg.l_invented_weight}
              defaultValue={DEFAULT_VAE_CONFIG.l_invented_weight}
              onCommit={(v) => setField("l_invented_weight", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">Flat-region penalty.</span>
          </div>
          <p className="text-xs text-gray-500 -mt-1">
            Penalises high-frequency energy in the decode that a least-squares projection onto
            the source&apos;s own high-frequency content cannot explain, inside windows where a
            plane fit says the source is flat or a smooth gradient. The projection coefficient is
            detached from the gradient, so the term is reduced by emitting less unexplained
            energy rather than by correlating more with the source. Every other term in the bank
            is an agreement-with-source objective. The window geometry, the highpass basis and
            the projection constants are fixed in the backend. 0 disables the term.
          </p>
          <p className="text-xs text-gray-500 -mt-1">
            Not a standalone objective: the term&apos;s own minimum inside a flat window is
            &quot;emit no high frequency at all&quot;, and it charges exact reproduction of the
            source&apos;s own detail slightly more than it charges a blur of it. It is meant to
            run alongside an agreement-with-source term (MSE / LPIPS), which supplies the
            opposing pull.
          </p>
          {cfg.l_invented_weight > 0 && (
            <div className="pl-4 border-l border-gray-700 space-y-2">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Luma (Y) weight</label>
                <NumberInput
                  min={0} max={4} step={0.05} parse="float"
                  value={cfg.l_invented_y_weight}
                  defaultValue={DEFAULT_VAE_CONFIG.l_invented_y_weight}
                  onCommit={(v) => setField("l_invented_y_weight", v)}
                  className={numberClass}
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Chroma weight</label>
                <NumberInput
                  min={0} max={4} step={0.05} parse="float"
                  value={cfg.l_invented_chroma_weight}
                  defaultValue={DEFAULT_VAE_CONFIG.l_invented_chroma_weight}
                  onCommit={(v) => setField("l_invented_chroma_weight", v)}
                  className={numberClass}
                />
              </div>
              {LOSS_WEIGHT_KEYS.every(
                (k) => k === "l_invented_weight" || Number(cfg[k]) <= 0,
              ) && (
                <p className="text-xs text-amber-400">
                  This is the only active loss weight. On its own, the configuration whose loss
                  is lowest is a decoder that emits no high frequency inside flat regions. Set
                  mse_weight or another agreement-with-source term above 0.
                </p>
              )}
              {cfg.l_invented_y_weight <= 0 && cfg.l_invented_chroma_weight <= 0 && (
                <p className="text-xs text-red-400">
                  Both channel weights are 0: the term would be identically zero while still
                  being computed every step. The backend refuses this.
                </p>
              )}
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Flat threshold Y</label>
                <NumberInput
                  min={0.25} max={8} step={0.25} parse="float"
                  value={cfg.l_invented_flat_t_y}
                  defaultValue={DEFAULT_VAE_CONFIG.l_invented_flat_t_y}
                  onCommit={(v) => setField("l_invented_flat_t_y", v)}
                  className={numberClass}
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-40">Flat threshold chroma</label>
                <NumberInput
                  min={0.25} max={8} step={0.25} parse="float"
                  value={cfg.l_invented_flat_t_c}
                  defaultValue={DEFAULT_VAE_CONFIG.l_invented_flat_t_c}
                  onCommit={(v) => setField("l_invented_flat_t_c", v)}
                  className={numberClass}
                />
              </div>
              <p className="text-xs text-gray-500">
                Plane-fit residual thresholds in 8-bit levels: a window counts as flat when the
                RMS residual of a least-squares plane fit is at or below them, so smooth
                gradients qualify as well as constant regions. The fraction of candidate windows
                selected is charted as &quot;VAE invented coverage&quot;.
              </p>
            </div>
          )}

          {activeLossCount === 0 && (
            <p className="text-xs text-red-400">
              All loss weights are 0: there is no training signal. The backend refuses this
              configuration.
            </p>
          )}
        </section>

        {/* Run shape */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Run shape</h3>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Total steps</label>
            <NumberInput
              min={1} step={100} parse="int"
              value={cfg.total_steps}
              defaultValue={DEFAULT_VAE_CONFIG.total_steps}
              onCommit={(v) => setField("total_steps", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">Optimizer steps. This trainer is step-based; there is no epoch mode.</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Batch size</label>
            <NumberInput
              min={1} step={1} parse="int"
              value={cfg.batch_size}
              defaultValue={DEFAULT_VAE_CONFIG.batch_size}
              onCommit={(v) => setField("batch_size", v)}
              className={numberClass}
            />
            <label className="text-xs text-gray-400">Gradient accumulation</label>
            <NumberInput
              min={1} step={1} parse="int"
              value={cfg.gradient_accumulation_steps}
              defaultValue={DEFAULT_VAE_CONFIG.gradient_accumulation_steps}
              onCommit={(v) => setField("gradient_accumulation_steps", v)}
              className="w-20 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Resolution</label>
            <NumberInput
              min={64} step={64} snap={8} parse="int"
              value={cfg.resolution}
              defaultValue={DEFAULT_VAE_CONFIG.resolution}
              onCommit={(v) => setField("resolution", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">
              Square random crop, multiple of 8. No aspect-ratio bucketing: the decoder&apos;s
              non-local terms (one flattened attention, 30 GroupNorms) reduce over the spatial
              axes, so they see latent area and never aspect.
            </span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Crop scale policy</label>
            <select
              value={cfg.crop_scale_policy}
              onChange={(e) => {
                const next = e.target.value as VaeConfig["crop_scale_policy"];
                setField("crop_scale_policy", next);
                // The bound is only read under "mixed", and the backend REFUSES
                // a non-zero value under any other policy rather than ignoring
                // it, so leaving it set would block the run.
                if (next !== "mixed") setField("crop_scale_max_downscale", 0);
              }}
              className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="downscale">downscale (short side to resolution)</option>
              <option value="native">native (crop full-size pixels)</option>
              <option value="mixed">mixed (per-sample factor)</option>
            </select>
            {cfg.crop_scale_policy === "mixed" && (
              <>
                <label className="text-xs text-gray-400">Max downscale</label>
                <NumberInput
                  min={0} step={0.5} parse="float"
                  value={cfg.crop_scale_max_downscale}
                  defaultValue={DEFAULT_VAE_CONFIG.crop_scale_max_downscale}
                  onCommit={(v) => setField("crop_scale_max_downscale", v)}
                  className="w-20 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </>
            )}
          </div>
          <p className="text-xs text-gray-500">
            How much an image is resampled before the crop is taken. <b>downscale</b> scales the
            short side to exactly the resolution, which resamples 95.8% of the datasets in use by
            a median 2.30x; <b>native</b> crops out of the full-size pixels and upscales only
            images smaller than the crop; <b>mixed</b> draws the factor per sample over
            [1, short/resolution], log-uniformly, bounded by Max downscale (0 = unbounded).
            Native crops are cheaper in the dataloader, not more expensive. Validation is not
            affected: it is always a deterministic centre crop under the downscale policy, so the
            held-out PSNR keeps one meaning across a policy change.
          </p>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Compute dtype</label>
            <select
              value={cfg.dtype}
              onChange={(e) => setField("dtype", e.target.value as VaeConfig["dtype"])}
              className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
            >
              <option value="bf16">bf16</option>
              <option value="fp32">fp32</option>
            </select>
            <span className="text-xs text-gray-500">
              Autocast dtype; weights are always held in fp32 as the optimizer&apos;s master copy.
              fp16 is not available: SD1.5/SDXL-family VAEs overflow it in their decoder
              activations, and this trainer has no gradient scaler.
            </span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Learning rate</label>
            <NumberInput
              // Positive minimum: the backend does not refuse lr=0, and a 0-LR
              // run trains nothing while reporting success.
              min={1e-9} step={1e-6} parse="float"
              value={cfg.learning_rate}
              defaultValue={DEFAULT_VAE_CONFIG.learning_rate}
              onCommit={(v) => setField("learning_rate", v)}
              className={numberClass}
            />
            <label className="text-xs text-gray-400">Weight decay</label>
            <NumberInput
              min={0} step={0.001} parse="float"
              value={cfg.optimizer_weight_decay}
              defaultValue={DEFAULT_VAE_CONFIG.optimizer_weight_decay}
              onCommit={(v) => setField("optimizer_weight_decay", v)}
              className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Optimizer</label>
            <select
              value={cfg.optimizer}
              onChange={(e) => setField("optimizer", e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
            >
              {OPTIMIZERS.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </select>
            <label className="text-xs text-gray-400">Gradient clip</label>
            <NumberInput
              min={0} step={0.05} parse="float"
              value={cfg.max_grad_norm}
              defaultValue={DEFAULT_VAE_CONFIG.max_grad_norm}
              onCommit={(v) => setField("max_grad_norm", v)}
              className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">LR scheduler</label>
            <select
              value={cfg.lr_scheduler}
              onChange={(e) => setField("lr_scheduler", e.target.value)}
              className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
            >
              {LR_SCHEDULERS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <label className="text-xs text-gray-400">Warmup steps</label>
            <NumberInput
              min={0} step={10} parse="int"
              value={cfg.lr_warmup_steps}
              defaultValue={DEFAULT_VAE_CONFIG.lr_warmup_steps}
              onCommit={(v) => setField("lr_warmup_steps", v)}
              className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Seed</label>
            <NumberInput
              min={0} step={1} parse="int"
              value={cfg.seed}
              defaultValue={DEFAULT_VAE_CONFIG.seed}
              onCommit={(v) => setField("seed", v)}
              className={numberClass}
            />
            <label className="text-xs text-gray-400">DataLoader workers</label>
            <NumberInput
              min={0} step={1} parse="int"
              value={cfg.num_workers}
              defaultValue={DEFAULT_VAE_CONFIG.num_workers}
              onCommit={(v) => setField("num_workers", v)}
              className="w-20 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </section>

        {/* EMA */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">EMA</h3>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={cfg.ema_enabled}
              onChange={(e) => setField("ema_enabled", e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
            />
            <span className="text-sm text-gray-300">Exponential moving average over the trainable parameters</span>
          </label>
          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">EMA decay</label>
            <NumberInput
              min={0.000001} max={0.999999} step={0.001} parse="float"
              value={cfg.ema_decay}
              defaultValue={DEFAULT_VAE_CONFIG.ema_decay}
              onCommit={(v) => setField("ema_decay", v)}
              disabled={!cfg.ema_enabled}
              className={`${numberClass} disabled:opacity-50`}
            />
            <span className="text-xs text-gray-500">Must be between 0 and 1 (exclusive).</span>
          </div>
          <p className="text-xs text-gray-500">
            Both sd-vae-ft-ema and PiD use EMA. The decay is warmup-ramped. With EMA on, the run
            writes two directories: <code>{`<run_name>${cfg.train_encoder ? "_vae_encoder_trained" : "_vae"}`}</code>{" "}
            (EMA weights) and <code>{`<run_name>${cfg.train_encoder ? "_vae_encoder_trained" : "_vae"}_noema`}</code>{" "}
            (live weights). With EMA off it writes one, containing the live weights.
          </p>
        </section>

        {/* Saving and validation */}
        <section className="border border-gray-700 rounded p-3 space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Saving and validation</h3>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Save every (steps)</label>
            <NumberInput
              min={0} step={50} parse="int"
              value={cfg.save_every}
              defaultValue={DEFAULT_VAE_CONFIG.save_every}
              onCommit={(v) => setField("save_every", v)}
              className={numberClass}
            />
            <label className="text-xs text-gray-400">Checkpoints to keep</label>
            <NumberInput
              min={0} step={1} parse="int"
              value={cfg.max_step_saves_to_keep}
              defaultValue={DEFAULT_VAE_CONFIG.max_step_saves_to_keep}
              onCommit={(v) => setField("max_step_saves_to_keep", v)}
              className="w-20 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
            <span className="text-xs text-gray-500">0 keeps all.</span>
          </div>

          <label className="flex items-start gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={cfg.export_bare_ldm}
              disabled={cfg.train_encoder}
              onChange={(e) => setField("export_bare_ldm", e.target.checked)}
              className="w-4 h-4 mt-0.5 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0 disabled:opacity-50"
            />
            <span className="text-sm text-gray-300">
              Also export a bare LDM <code>.safetensors</code>
              <span className="block text-xs text-gray-500">
                AutoencoderKL only. The file carries no <code>config.json</code>, so whatever
                loads it supplies <code>scaling_factor</code> / <code>shift_factor</code>. That is
                correct only while the encoder is frozen, so this export is refused when the
                encoder is trained; the diffusers directory is always written.
              </span>
            </span>
          </label>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Resume from checkpoint</label>
            <input
              type="text"
              value={resumeFrom}
              onChange={(e) => setResumeFrom(e.target.value)}
              placeholder="empty = start from the beginning"
              className={`${inputClass} font-mono text-xs`}
            />
            <p className="text-xs text-gray-500 mt-1">
              <code>latest</code> picks the highest-numbered <code>checkpoints/step_*</code>
              {" "}directory, and starts fresh when the run has none yet. Otherwise: a checkpoint
              directory name under the run&apos;s <code>checkpoints/</code> (for example{" "}
              <code>step_00000500</code>) or an absolute path — a name that does not exist is an
              error listing the available checkpoints. The checkpoint must come from a run with
              the same Decoder blocks setting.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Validate every (steps)</label>
            <NumberInput
              min={0} step={50} parse="int"
              value={cfg.validation_every}
              defaultValue={DEFAULT_VAE_CONFIG.validation_every}
              onCommit={(v) => setField("validation_every", v)}
              className={numberClass}
            />
            <span className="text-xs text-gray-500">0 disables validation.</span>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-gray-400 w-40">Validation images</label>
            <NumberInput
              min={1} step={1} parse="int"
              value={cfg.validation_num_images}
              defaultValue={DEFAULT_VAE_CONFIG.validation_num_images}
              onCommit={(v) => setField("validation_num_images", v)}
              className={numberClass}
            />
            <label className="text-xs text-gray-400">Validation resolution</label>
            <NumberInput
              min={64} step={64} snap={8} parse="int"
              value={cfg.validation_resolution}
              defaultValue={DEFAULT_VAE_CONFIG.validation_resolution}
              onCommit={(v) => setField("validation_resolution", v)}
              className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          <p className="text-xs text-gray-500">
            Validation is a deterministic centre crop under the downscale policy, at 1024 by
            default: at 512 the held-out PSNR is measured on content ~4x richer in near-Nyquist
            energy than anything generation produces, while at 1024 the median source is
            downscaled only ~1.1x. Changing this mid-run puts a step in the PSNR chart.
            Validation images are taken from the tail of the dataset and excluded from training.
            The validation PSNR / blockiness series in the loss chart is the signal that a
            fine-tune is going wrong: PSNR falling means the decoder is drifting off the data,
            blockiness rising above ~1.0 means it is manufacturing latent-cell grid structure.
          </p>
        </section>

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded text-sm text-red-400 whitespace-pre-wrap">
            {error}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex-shrink-0 p-4 border-t border-gray-700 flex justify-end gap-3">
        <button
          onClick={onClose}
          className="px-4 py-2 text-sm text-gray-300 hover:text-white transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving || loadingRun}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:text-gray-400 rounded text-sm transition-colors"
        >
          {saving
            ? (isEditMode ? "Saving..." : "Creating...")
            : (isEditMode ? "Save Changes" : "Create Run")}
        </button>
      </div>
    </div>
  );
}
