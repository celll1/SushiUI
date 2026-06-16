"use client";

import { useState, useEffect } from "react";
import {
  createTaggerTrainingRun,
  updateTaggerTrainingRun,
  listDatasets,
  Dataset,
  TaggerTrainingRun,
  TaggerTrainingRunCreateRequest,
  TaggerDatasetConfig,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";

interface TaggerTrainingConfigProps {
  onClose: () => void;
  onRunCreated: (run: TaggerTrainingRun) => void;
  /** If provided, the form operates in edit mode for this run. */
  editRun?: TaggerTrainingRun;
}

const DEFAULT_CONFIG: Omit<TaggerTrainingRunCreateRequest, "dataset_configs"> = {
  run_name: "",
  training_method: "lora",
  vision_encoder_path: "",
  init_head_from: "",
  lora_rank: 32,
  lora_alpha: 16,
  learning_rate: 3e-4,
  head_lr_multiplier: 10.0,
  optimizer: "adamw8bit",
  warmup_steps: 100,
  epochs: 10,
  batch_size: 32,
  vocab_min_count: 10,
  vocab_use_gelbooru_categories: true,
  num_workers: 4,
  num_workers_override: null as number | null,
  save_every_n_steps: 500,
  save_every_n_epochs: 0,
  keep_last_n_checkpoints: 3,
  checkpoint_save_mode: "lora",
  mixed_precision: "bf16",
  use_flash_attention: false,
  gradient_checkpointing: true,
  loss_function: "asl" as string,
  loss_clip: 0.05,
  loss_gamma_neg: 4,
  loss_gamma_pos: 1,
  loss_gamma0: 4.0,
  loss_m0: 0.2,
  loss_rho: 0.5,
  loss_beta: 2.0,
  loss_label_weight: "fisher" as string,
  val_split_mode: "percent" as string,
  val_split: 0.05,
  val_fixed_size: undefined as number | undefined,
  validate_every: 1,
  save_best_only: false,
  excluded_categories: [] as string[],
  ban_tags: "",
  use_tag_aliases: false,
  save_base_model: false,
  quality_masking_mode: "intra_group" as const,
  cls_dim: undefined as number | undefined,
  hidden_proj_dim: undefined as number | undefined,
  // LR matrix (conditional inference) — built once at training start when enabled.
  build_lr_matrix_on_start: false,
  lr_top_anchors: 10000,
  lr_top_targets: 1000,
  lr_threshold: 1.0,
  lr_min_anchor_count: 10,
  // Pre-flight: detect dataset drift + auto-rescan.  Adds the time of
  // one directory walk per dataset (~5 min for 3M items on NVMe).
  rescan_before_training: "off",
  // Training-time F1 metrics
  train_f1_eval_every_n_steps: 100,
  train_f1_threshold_search_every_n_steps: 500,
  train_f1_initial_threshold: 0.35,
  train_f1_buffer_batches: 16,
  // Online Danbooru augmentation
  enable_danbooru_augmentation: false,
  danbooru_query_enable: true,
  danbooru_query_expand_enable: false,
  danbooru_query_new_tag_min_count: 200,
  danbooru_query_resolve_top_k: 50,
  danbooru_query_max_expanded_tags: 0,
  danbooru_query_expand_categories: [0, 3, 4],
  danbooru_query_resolve_interval: 3600,
  danbooru_query_collect_per_epoch: 0,
  danbooru_new_tag_collect_per_epoch: 0,
  danbooru_low_f1_collect_per_epoch: 0,
  danbooru_tags: "",
  danbooru_injection_interval: 4,
  danbooru_injection_batch_size_ratio: 1.0,
  danbooru_min_score: 0,
  danbooru_max_posts_per_query: 200,
  danbooru_api_interval: 1.4,
  danbooru_dl_speed_kbps: 500,
  danbooru_buffer_size: null,
  danbooru_vocab_expand: false,
  danbooru_new_tag_min_count: 200,
  danbooru_new_tag_lookback_days: 90,
  danbooru_new_tag_categories: [0, 3, 4],
  danbooru_new_tag_min_count_by_cat: {} as Record<string, number>,
  danbooru_new_tag_survey_interval: 3600,
  danbooru_max_dynamic_tags: 0,
  danbooru_query_weight_static: 1.0,
  danbooru_query_weight_new_tag: 1.0,
  danbooru_query_weight_low_f1: 1.0,
  danbooru_query_weight_train_count: 1.0,
  danbooru_low_f1_enable: false,
  danbooru_low_f1_threshold: 0.5,
  danbooru_low_f1_top_k: 500,
  danbooru_low_f1_min_posts: 50,
  danbooru_train_count_enable: false,
  danbooru_train_count_top_k: 500,
  danbooru_train_count_min_deficit_ratio: 0.3,
  danbooru_train_count_min_per_epoch: 10,
  danbooru_train_count_min_posts: 50,
  danbooru_train_count_collect_per_epoch: 0,
  danbooru_quality_tag_enable: false,
  danbooru_quality_tag_thresholds: "",
  danbooru_quality_tag_attach_negative: false,
  danbooru_cooc_expand_enable: false,
  danbooru_cooc_min_count: 50,
  danbooru_cooc_categories: [0, 3, 4],
  danbooru_query_weight_cooc: 0.1,
  danbooru_cooc_collect_per_epoch: 50,
  danbooru_cooc_order_random: true,
};

type ConfigState = Omit<TaggerTrainingRunCreateRequest, "dataset_configs">;

export default function TaggerTrainingConfig({
  onClose,
  onRunCreated,
  editRun,
}: TaggerTrainingConfigProps) {
  const isEditMode = !!editRun;

  // Derive initial config from editRun when in edit mode
  const initialConfig: ConfigState = isEditMode
    ? {
        run_name: editRun.run_name,
        training_method: (editRun.config?.training_method as ConfigState["training_method"]) ?? DEFAULT_CONFIG.training_method,
        vision_encoder_path: editRun.vision_encoder_path,
        init_head_from: (editRun.config?.init_head_from as string) ?? DEFAULT_CONFIG.init_head_from,
        lora_rank: (editRun.config?.lora_rank as number) ?? DEFAULT_CONFIG.lora_rank,
        lora_alpha: (editRun.config?.lora_alpha as number) ?? DEFAULT_CONFIG.lora_alpha,
        learning_rate: (editRun.config?.learning_rate as number) ?? DEFAULT_CONFIG.learning_rate,
        head_lr_multiplier: (editRun.config?.head_lr_multiplier as number) ?? DEFAULT_CONFIG.head_lr_multiplier,
        optimizer: (editRun.config?.optimizer as string) ?? DEFAULT_CONFIG.optimizer,
        warmup_steps: (editRun.config?.warmup_steps as number) ?? DEFAULT_CONFIG.warmup_steps,
        epochs: (editRun.config?.epochs as number) ?? DEFAULT_CONFIG.epochs,
        batch_size: (editRun.config?.batch_size as number) ?? DEFAULT_CONFIG.batch_size,
        vocab_min_count: (editRun.config?.vocab_min_count as number) ?? DEFAULT_CONFIG.vocab_min_count,
        vocab_use_gelbooru_categories: (editRun.config?.vocab_use_gelbooru_categories as boolean) ?? DEFAULT_CONFIG.vocab_use_gelbooru_categories,
        num_workers: (editRun.config?.num_workers as number) ?? DEFAULT_CONFIG.num_workers,
        num_workers_override: (editRun.config?.num_workers_override as number | null) ?? DEFAULT_CONFIG.num_workers_override,
        save_every_n_steps: (editRun.config?.save_every_n_steps as number) ?? DEFAULT_CONFIG.save_every_n_steps,
        save_every_n_epochs: (editRun.config?.save_every_n_epochs as number) ?? DEFAULT_CONFIG.save_every_n_epochs,
        keep_last_n_checkpoints: (editRun.config?.keep_last_n_checkpoints as number) ?? DEFAULT_CONFIG.keep_last_n_checkpoints,
        checkpoint_save_mode: (editRun.config?.checkpoint_save_mode as string) ?? DEFAULT_CONFIG.checkpoint_save_mode,
        mixed_precision: (editRun.config?.mixed_precision as string) ?? DEFAULT_CONFIG.mixed_precision,
        use_flash_attention: (editRun.config?.use_flash_attention as boolean) ?? DEFAULT_CONFIG.use_flash_attention,
        gradient_checkpointing: (editRun.config?.gradient_checkpointing as boolean) ?? DEFAULT_CONFIG.gradient_checkpointing,
        loss_function: (editRun.config?.loss_function as string) ?? DEFAULT_CONFIG.loss_function,
        loss_gamma_neg: (editRun.config?.loss_gamma_neg as number) ?? DEFAULT_CONFIG.loss_gamma_neg,
        loss_gamma_pos: (editRun.config?.loss_gamma_pos as number) ?? DEFAULT_CONFIG.loss_gamma_pos,
        loss_gamma0: (editRun.config?.loss_gamma0 as number) ?? DEFAULT_CONFIG.loss_gamma0,
        loss_m0: (editRun.config?.loss_m0 as number) ?? DEFAULT_CONFIG.loss_m0,
        loss_rho: (editRun.config?.loss_rho as number) ?? DEFAULT_CONFIG.loss_rho,
        loss_beta: (editRun.config?.loss_beta as number) ?? DEFAULT_CONFIG.loss_beta,
        loss_label_weight: (editRun.config?.loss_label_weight as string) ?? DEFAULT_CONFIG.loss_label_weight,
        val_split_mode: (editRun.config?.val_split_mode as string) ?? DEFAULT_CONFIG.val_split_mode,
        val_split: (editRun.config?.val_split as number) ?? DEFAULT_CONFIG.val_split,
        val_fixed_size: (editRun.config?.val_fixed_size as number) ?? DEFAULT_CONFIG.val_fixed_size,
        validate_every: (editRun.config?.validate_every as number) ?? DEFAULT_CONFIG.validate_every,
        save_best_only: (editRun.config?.save_best_only as boolean) ?? DEFAULT_CONFIG.save_best_only,
        excluded_categories: (editRun.config?.excluded_categories as string[]) ?? DEFAULT_CONFIG.excluded_categories,
        ban_tags: (editRun.config?.ban_tags as string) ?? DEFAULT_CONFIG.ban_tags,
        use_tag_aliases: (editRun.config?.use_tag_aliases as boolean) ?? DEFAULT_CONFIG.use_tag_aliases,
        save_base_model: (editRun.config?.save_base_model as boolean) ?? DEFAULT_CONFIG.save_base_model,
        quality_masking_mode: (editRun.config?.quality_masking_mode as ("intra_group" | "cross_group" | undefined)) ?? DEFAULT_CONFIG.quality_masking_mode,
        cls_dim: (editRun.config?.cls_dim as number | undefined) ?? DEFAULT_CONFIG.cls_dim,
        hidden_proj_dim: (editRun.config?.hidden_proj_dim as number | undefined) ?? DEFAULT_CONFIG.hidden_proj_dim,
        build_lr_matrix_on_start: (editRun.config?.build_lr_matrix_on_start as boolean) ?? DEFAULT_CONFIG.build_lr_matrix_on_start,
        lr_top_anchors: (editRun.config?.lr_top_anchors as number) ?? DEFAULT_CONFIG.lr_top_anchors,
        lr_top_targets: (editRun.config?.lr_top_targets as number) ?? DEFAULT_CONFIG.lr_top_targets,
        lr_threshold: (editRun.config?.lr_threshold as number) ?? DEFAULT_CONFIG.lr_threshold,
        lr_min_anchor_count: (editRun.config?.lr_min_anchor_count as number) ?? DEFAULT_CONFIG.lr_min_anchor_count,
        rescan_before_training: (editRun.config?.rescan_before_training as ("off" | "path" | "smart" | "force" | boolean | undefined)) ?? DEFAULT_CONFIG.rescan_before_training,
        train_f1_eval_every_n_steps: (editRun.config?.train_f1_eval_every_n_steps as number) ?? DEFAULT_CONFIG.train_f1_eval_every_n_steps,
        train_f1_threshold_search_every_n_steps: (editRun.config?.train_f1_threshold_search_every_n_steps as number) ?? DEFAULT_CONFIG.train_f1_threshold_search_every_n_steps,
        train_f1_initial_threshold: (editRun.config?.train_f1_initial_threshold as number) ?? DEFAULT_CONFIG.train_f1_initial_threshold,
        train_f1_buffer_batches: (editRun.config?.train_f1_buffer_batches as number) ?? DEFAULT_CONFIG.train_f1_buffer_batches,
        // Online Danbooru augmentation
        enable_danbooru_augmentation: (editRun.config?.enable_danbooru_augmentation as boolean) ?? DEFAULT_CONFIG.enable_danbooru_augmentation,
        danbooru_query_enable: (editRun.config?.danbooru_query_enable as boolean) ?? DEFAULT_CONFIG.danbooru_query_enable,
        danbooru_query_expand_enable: (editRun.config?.danbooru_query_expand_enable as boolean) ?? DEFAULT_CONFIG.danbooru_query_expand_enable,
        danbooru_query_new_tag_min_count: (editRun.config?.danbooru_query_new_tag_min_count as number) ?? DEFAULT_CONFIG.danbooru_query_new_tag_min_count,
        danbooru_query_resolve_top_k: (editRun.config?.danbooru_query_resolve_top_k as number) ?? DEFAULT_CONFIG.danbooru_query_resolve_top_k,
        danbooru_query_max_expanded_tags: (editRun.config?.danbooru_query_max_expanded_tags as number) ?? DEFAULT_CONFIG.danbooru_query_max_expanded_tags,
        danbooru_query_expand_categories: (editRun.config?.danbooru_query_expand_categories as number[]) ?? DEFAULT_CONFIG.danbooru_query_expand_categories,
        danbooru_query_resolve_interval: (editRun.config?.danbooru_query_resolve_interval as number) ?? DEFAULT_CONFIG.danbooru_query_resolve_interval,
        danbooru_query_collect_per_epoch: (editRun.config?.danbooru_query_collect_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_query_collect_per_epoch,
        danbooru_new_tag_collect_per_epoch: (editRun.config?.danbooru_new_tag_collect_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_new_tag_collect_per_epoch,
        danbooru_low_f1_collect_per_epoch: (editRun.config?.danbooru_low_f1_collect_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_low_f1_collect_per_epoch,
        danbooru_tags: (editRun.config?.danbooru_tags as string) ?? DEFAULT_CONFIG.danbooru_tags,
        danbooru_injection_interval: (editRun.config?.danbooru_injection_interval as number) ?? DEFAULT_CONFIG.danbooru_injection_interval,
        danbooru_injection_batch_size_ratio: (editRun.config?.danbooru_injection_batch_size_ratio as number) ?? DEFAULT_CONFIG.danbooru_injection_batch_size_ratio,
        danbooru_min_score: (editRun.config?.danbooru_min_score as number) ?? DEFAULT_CONFIG.danbooru_min_score,
        danbooru_max_posts_per_query: (editRun.config?.danbooru_max_posts_per_query as number) ?? DEFAULT_CONFIG.danbooru_max_posts_per_query,
        danbooru_api_interval: (editRun.config?.danbooru_api_interval as number) ?? DEFAULT_CONFIG.danbooru_api_interval,
        danbooru_dl_speed_kbps: (editRun.config?.danbooru_dl_speed_kbps as number) ?? DEFAULT_CONFIG.danbooru_dl_speed_kbps,
        danbooru_buffer_size: (editRun.config?.danbooru_buffer_size as number | null) ?? DEFAULT_CONFIG.danbooru_buffer_size,
        danbooru_vocab_expand: (editRun.config?.danbooru_vocab_expand as boolean) ?? DEFAULT_CONFIG.danbooru_vocab_expand,
        danbooru_new_tag_min_count: (editRun.config?.danbooru_new_tag_min_count as number) ?? DEFAULT_CONFIG.danbooru_new_tag_min_count,
        danbooru_new_tag_lookback_days: (editRun.config?.danbooru_new_tag_lookback_days as number) ?? DEFAULT_CONFIG.danbooru_new_tag_lookback_days,
        danbooru_new_tag_categories: (editRun.config?.danbooru_new_tag_categories as number[]) ?? DEFAULT_CONFIG.danbooru_new_tag_categories,
        danbooru_new_tag_min_count_by_cat: (editRun.config?.danbooru_new_tag_min_count_by_cat as Record<string, number>) ?? DEFAULT_CONFIG.danbooru_new_tag_min_count_by_cat,
        danbooru_new_tag_survey_interval: (editRun.config?.danbooru_new_tag_survey_interval as number) ?? DEFAULT_CONFIG.danbooru_new_tag_survey_interval,
        danbooru_max_dynamic_tags: (editRun.config?.danbooru_max_dynamic_tags as number) ?? DEFAULT_CONFIG.danbooru_max_dynamic_tags,
        danbooru_query_weight_static: (editRun.config?.danbooru_query_weight_static as number) ?? DEFAULT_CONFIG.danbooru_query_weight_static,
        danbooru_query_weight_new_tag: (editRun.config?.danbooru_query_weight_new_tag as number) ?? DEFAULT_CONFIG.danbooru_query_weight_new_tag,
        danbooru_query_weight_low_f1: (editRun.config?.danbooru_query_weight_low_f1 as number) ?? DEFAULT_CONFIG.danbooru_query_weight_low_f1,
        danbooru_query_weight_train_count: (editRun.config?.danbooru_query_weight_train_count as number) ?? DEFAULT_CONFIG.danbooru_query_weight_train_count,
        danbooru_low_f1_enable: (editRun.config?.danbooru_low_f1_enable as boolean) ?? DEFAULT_CONFIG.danbooru_low_f1_enable,
        danbooru_low_f1_threshold: (editRun.config?.danbooru_low_f1_threshold as number) ?? DEFAULT_CONFIG.danbooru_low_f1_threshold,
        danbooru_low_f1_top_k: (editRun.config?.danbooru_low_f1_top_k as number) ?? DEFAULT_CONFIG.danbooru_low_f1_top_k,
        danbooru_low_f1_min_posts: (editRun.config?.danbooru_low_f1_min_posts as number) ?? DEFAULT_CONFIG.danbooru_low_f1_min_posts,
        danbooru_train_count_enable: (editRun.config?.danbooru_train_count_enable as boolean) ?? DEFAULT_CONFIG.danbooru_train_count_enable,
        danbooru_train_count_top_k: (editRun.config?.danbooru_train_count_top_k as number) ?? DEFAULT_CONFIG.danbooru_train_count_top_k,
        danbooru_train_count_min_deficit_ratio: (editRun.config?.danbooru_train_count_min_deficit_ratio as number) ?? DEFAULT_CONFIG.danbooru_train_count_min_deficit_ratio,
        danbooru_train_count_min_per_epoch: (editRun.config?.danbooru_train_count_min_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_train_count_min_per_epoch,
        danbooru_train_count_min_posts: (editRun.config?.danbooru_train_count_min_posts as number) ?? DEFAULT_CONFIG.danbooru_train_count_min_posts,
        danbooru_train_count_collect_per_epoch: (editRun.config?.danbooru_train_count_collect_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_train_count_collect_per_epoch,
        danbooru_quality_tag_enable: (editRun.config?.danbooru_quality_tag_enable as boolean) ?? DEFAULT_CONFIG.danbooru_quality_tag_enable,
        danbooru_quality_tag_thresholds: (editRun.config?.danbooru_quality_tag_thresholds as string) ?? DEFAULT_CONFIG.danbooru_quality_tag_thresholds,
        danbooru_quality_tag_attach_negative: (editRun.config?.danbooru_quality_tag_attach_negative as boolean) ?? DEFAULT_CONFIG.danbooru_quality_tag_attach_negative,
        danbooru_cooc_expand_enable: (editRun.config?.danbooru_cooc_expand_enable as boolean) ?? DEFAULT_CONFIG.danbooru_cooc_expand_enable,
        danbooru_cooc_min_count: (editRun.config?.danbooru_cooc_min_count as number) ?? DEFAULT_CONFIG.danbooru_cooc_min_count,
        danbooru_cooc_categories: (editRun.config?.danbooru_cooc_categories as number[]) ?? DEFAULT_CONFIG.danbooru_cooc_categories,
        danbooru_query_weight_cooc: (editRun.config?.danbooru_query_weight_cooc as number) ?? DEFAULT_CONFIG.danbooru_query_weight_cooc,
        danbooru_cooc_collect_per_epoch: (editRun.config?.danbooru_cooc_collect_per_epoch as number) ?? DEFAULT_CONFIG.danbooru_cooc_collect_per_epoch,
        danbooru_cooc_order_random: (editRun.config?.danbooru_cooc_order_random as boolean) ?? DEFAULT_CONFIG.danbooru_cooc_order_random,
      }
    : DEFAULT_CONFIG;

  const initialDatasetIds: number[] = isEditMode
    ? ((editRun.dataset_configs as unknown as TaggerDatasetConfig[]) ?? []).map((dc) => dc.dataset_id)
    : [];

  const [config, setConfig] = useState<ConfigState>(initialConfig);
  const { taggerTrainingDefaults } = useStartup();

  // Apply backend-fetched defaults when they arrive (only for new runs, not edit mode)
  useEffect(() => {
    if (!taggerTrainingDefaults || isEditMode) return;
    setConfig(prev => ({ ...prev, ...(taggerTrainingDefaults as Partial<ConfigState>) }));
  }, [taggerTrainingDefaults, isEditMode]);

  // selectedDatasetIds tracks numeric dataset.id values
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<number[]>(initialDatasetIds);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fixed category list — same order as backend CATEGORY_ORDER
  const ALL_CATEGORIES = ["General", "Character", "Copyright", "Artist", "Meta", "Rating", "Quality", "Model"];

  // Only show datasets that have tags-format captions
  const tagDatasets = datasets.filter((d) => d.has_tags_captions);

  useEffect(() => {
    listDatasets()
      .then((res) => setDatasets(res.datasets || []))
      .catch(console.error);
  }, []);

  const handleDatasetToggle = (datasetId: number) => {
    setSelectedDatasetIds((prev) =>
      prev.includes(datasetId)
        ? prev.filter((id) => id !== datasetId)
        : [...prev, datasetId]
    );
  };

  const handleSave = async () => {
    if (!config.run_name.trim()) {
      setError("Run name is required.");
      return;
    }
    if (selectedDatasetIds.length === 0) {
      setError("At least one dataset must be selected.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const datasetConfigs: TaggerDatasetConfig[] = selectedDatasetIds.map((id) => ({
        dataset_id: id,
        caption_types: [],
      }));
      const payload = {
        ...config,
        dataset_configs: datasetConfigs,
        // A disabled collection path contributes weight 0 regardless of the
        // (preserved) UI value, so the backend never selects an inactive path.
        danbooru_query_weight_static: config.danbooru_query_enable ? config.danbooru_query_weight_static : 0,
        danbooru_query_weight_new_tag: config.danbooru_vocab_expand ? config.danbooru_query_weight_new_tag : 0,
        danbooru_query_weight_low_f1: config.danbooru_low_f1_enable ? config.danbooru_query_weight_low_f1 : 0,
        danbooru_query_weight_train_count: config.danbooru_train_count_enable ? config.danbooru_query_weight_train_count : 0,
        danbooru_query_weight_cooc: config.danbooru_cooc_expand_enable ? config.danbooru_query_weight_cooc : 0,
      };
      const run = isEditMode
        ? await updateTaggerTrainingRun(editRun.run_id, payload)
        : await createTaggerTrainingRun(payload);
      onRunCreated(run);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  const setField = <K extends keyof ConfigState>(
    key: K,
    value: ConfigState[K]
  ) => setConfig((prev) => ({ ...prev, [key]: value }));

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
        <h2 className="text-lg font-semibold">{isEditMode ? "Edit Training Run" : "New Tagger Training Run"}</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">

        {/* Run name */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">Run Name</label>
          <input
            type="text"
            value={config.run_name}
            onChange={(e) => setField("run_name", e.target.value)}
            placeholder="e.g. siglip2_tagger_v1_20260409"
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </section>

        {/* Training method */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Training Method</label>
          <div className="flex gap-3">
            {(["lora", "full"] as const).map((method) => (
              <button
                key={method}
                onClick={() => setField("training_method", method)}
                className={`px-4 py-2 rounded text-sm border transition-colors ${
                  config.training_method === method
                    ? "border-blue-500 bg-blue-600 text-white"
                    : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                }`}
              >
                {method === "lora" ? "LoRA" : "Full Parameter"}
              </button>
            ))}
          </div>
        </section>

        {/* Vision encoder path */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Vision Encoder Path
          </label>
          <input
            type="text"
            value={config.vision_encoder_path}
            onChange={(e) => setField("vision_encoder_path", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Local path to <code>siglip2_so400m_vision_encoder.safetensors</code>, a LoRA/merged tagger checkpoint,
            or a HuggingFace repo ID (e.g. <code>google/siglip2-so400m-patch16-naflex</code>) or URL.
          </p>
        </section>

        {/* Init head from */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Init Head From <span className="text-gray-500 font-normal">(optional)</span>
          </label>
          <input
            type="text"
            value={config.init_head_from ?? ""}
            onChange={(e) => setField("init_head_from", e.target.value)}
            placeholder="leave empty to zero-initialize head"
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Path to a tagger checkpoint to inherit head weights from. Rows are copied and expanded (zero-init) or trimmed to match the current vocabulary size.
          </p>
        </section>

        {/* Dataset selection */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Datasets <span className="text-gray-500">(tags format only)</span>
          </label>
          {datasets.length === 0 ? (
            <p className="text-sm text-gray-500">Loading datasets...</p>
          ) : tagDatasets.length === 0 ? (
            <p className="text-sm text-gray-500">
              No datasets with tags-format captions found. Scan a dataset that has tag captions first.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {tagDatasets.map((dataset) => {
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
                      onChange={() => handleDatasetToggle(dataset.id)}
                      className="accent-blue-500"
                    />
                    <span className="font-medium">{dataset.name}</span>
                    <span className="text-xs text-gray-400">
                      {dataset.total_items.toLocaleString()} imgs · {dataset.total_tags.toLocaleString()} tagged
                    </span>
                  </label>
                );
              })}
            </div>
          )}

        </section>

        {/* Tag Filtering */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-3">Tag Filtering</label>

          {/* Excluded Categories */}
          <div className="mb-4">
            <label className="block text-xs text-gray-400 mb-2">
              Excluded Categories
              <span className="text-gray-500 ml-1">(checked categories will NOT be trained)</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {ALL_CATEGORIES.map((cat) => {
                const excluded = (config.excluded_categories as string[]).includes(cat);
                return (
                  <label
                    key={cat}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded border cursor-pointer text-xs transition-colors ${
                      excluded
                        ? "border-red-500 bg-red-900/30 text-red-300"
                        : "border-gray-600 bg-gray-800 text-gray-300 hover:bg-gray-700"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={excluded}
                      onChange={() => {
                        const curr = (config.excluded_categories as string[]);
                        setField(
                          "excluded_categories" as keyof ConfigState,
                          (excluded ? curr.filter((c) => c !== cat) : [...curr, cat]) as ConfigState[keyof ConfigState]
                        );
                      }}
                      className="accent-red-500"
                    />
                    <span>{cat}</span>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Ban Tags */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">
              Ban Tags
              <span className="text-gray-500 ml-1 font-normal">— one per line. Exact tag or wildcard (* matches any string, ? matches one char). e.g. <code className="text-gray-400">some tag</code>, <code className="text-gray-400">prefix_*</code>, <code className="text-gray-400">bad*</code></span>
            </label>
            <textarea
              value={typeof config.ban_tags === "string" ? config.ban_tags : ""}
              onChange={(e) => setField("ban_tags" as keyof ConfigState, e.target.value as ConfigState[keyof ConfigState])}
              placeholder={"some tag\nprefix_*\nbad*\n"}
              rows={4}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-600 font-mono focus:outline-none focus:border-blue-500 resize-y"
            />
          </div>

          {/* Tag alias resolution */}
          <div>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.use_tag_aliases ?? false}
                onChange={(e) => setField("use_tag_aliases", e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
              />
              <span className="text-sm text-gray-300">Resolve tag aliases</span>
              <span className="text-xs text-gray-500">(maps deprecated tags to canonical form via tagother/tag_aliases.json)</span>
            </label>
          </div>

          {/* Gelbooru category supplement */}
          <div>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.vocab_use_gelbooru_categories ?? true}
                onChange={(e) => setField("vocab_use_gelbooru_categories", e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
              />
              <span className="text-sm text-gray-300">Use Gelbooru categories</span>
              <span className="text-xs text-gray-500">(resolves Unknown tag categories via taglist_gel/ when not found in the Danbooru taglist; Danbooru takes precedence)</span>
            </label>
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.save_base_model ?? true}
                onChange={(e) => setField("save_base_model", e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
              />
              <span className="text-sm text-gray-300">Save base model</span>
              <span className="text-xs text-gray-500">(copies base weights into training directory for self-contained checkpoints)</span>
            </label>
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Quality tag masking</label>
            <select
              value={config.quality_masking_mode ?? "intra_group"}
              onChange={(e) => setField("quality_masking_mode", e.target.value as "intra_group" | "cross_group")}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200 focus:ring-0"
            >
              <option value="intra_group">Intra-group (recommended) — mask siblings within the labelled group</option>
              <option value="cross_group">Cross-group (legacy) — train all non-positive quality tags as negatives</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              When a quality tag is present on a sample:
              {" "}<b>intra-group</b> masks siblings (best/high/normal/medium share gradients, low/bad/worst share gradients) so cross-group good-vs-bad is the only signal.
              {" "}<b>cross-group</b> trains every non-positive quality tag as a negative — only correct when intra-group labels are truly mutually exclusive and prevalence-balanced.
            </p>
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Rescan datasets before training</label>
            <select
              value={
                typeof config.rescan_before_training === "boolean"
                  ? (config.rescan_before_training ? "path" : "off")
                  : (config.rescan_before_training ?? "off")
              }
              onChange={(e) => setField("rescan_before_training", e.target.value as "off" | "path" | "smart" | "force")}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-sm text-gray-200 focus:ring-0"
            >
              <option value="off">Off — skip pre-flight check</option>
              <option value="path">Path drift only — detect added / missing files</option>
              <option value="smart">Smart — path drift + caption mtime (catches in-place edits)</option>
              <option value="force">Force — always rescan, no drift detection</option>
            </select>
            <p className="text-xs text-gray-500 mt-1">
              When the chosen mode detects drift (or always in &quot;force&quot;), runs a
              full rescan.  Pre-flight walk adds ~5 min per 3M items on NVMe.
            </p>
          </div>
          <div>
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={config.build_lr_matrix_on_start ?? false}
                onChange={(e) => setField("build_lr_matrix_on_start", e.target.checked)}
                className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
              />
              <span className="text-sm text-gray-300">Build LR matrix at start</span>
              <span className="text-xs text-gray-500">
                (precomputes co-occurrence statistics for conditional inference; adds ~5-30 min once at training start)
              </span>
            </label>
            {config.build_lr_matrix_on_start && (
              <div className="mt-2 ml-6 grid grid-cols-2 lg:grid-cols-4 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Top anchors</label>
                  <input
                    type="number"
                    min={100}
                    max={97000}
                    value={config.lr_top_anchors ?? 10000}
                    onChange={(e) => setField("lr_top_anchors", parseInt(e.target.value) || 10000)}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Top targets / anchor</label>
                  <input
                    type="number"
                    min={10}
                    max={10000}
                    value={config.lr_top_targets ?? 1000}
                    onChange={(e) => setField("lr_top_targets", parseInt(e.target.value) || 1000)}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">|LR| threshold</label>
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={config.lr_threshold ?? 1.0}
                    onChange={(e) => setField("lr_threshold", parseFloat(e.target.value) || 1.0)}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Min anchor count</label>
                  <input
                    type="number"
                    min={1}
                    value={config.lr_min_anchor_count ?? 10}
                    onChange={(e) => setField("lr_min_anchor_count", parseInt(e.target.value) || 10)}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            )}
          </div>
        </section>

        {/* LoRA parameters */}
        {config.training_method === "lora" && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">LoRA Parameters</label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Rank</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={config.lora_rank}
                  onChange={(e) => setField("lora_rank", parseInt(e.target.value) || 32)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Alpha</label>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={config.lora_alpha}
                  onChange={(e) => setField("lora_alpha", parseInt(e.target.value) || 16)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </section>
        )}

        {/* Custom Attention Pooling (Full FT only) */}
        {config.training_method === "full" && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Custom Attention Pooling <span className="text-xs text-gray-500">(Full FT only)</span>
            </label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  cls_dim <span className="text-gray-600">(empty = use pooler_output)</span>
                </label>
                <input
                  type="number"
                  min={64}
                  max={4096}
                  placeholder="e.g. 768"
                  value={config.cls_dim ?? ""}
                  onChange={(e) => setField("cls_dim" as keyof ConfigState, e.target.value === "" ? undefined : (parseInt(e.target.value) || undefined) as ConfigState[keyof ConfigState])}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  hidden_proj_dim <span className="text-gray-600">(empty = proj to cls_dim directly)</span>
                </label>
                <input
                  type="number"
                  min={64}
                  max={8192}
                  placeholder="e.g. 2048"
                  disabled={!config.cls_dim}
                  value={config.hidden_proj_dim ?? ""}
                  onChange={(e) => setField("hidden_proj_dim" as keyof ConfigState, e.target.value === "" ? undefined : (parseInt(e.target.value) || undefined) as ConfigState[keyof ConfigState])}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-blue-500 disabled:opacity-40 disabled:cursor-not-allowed"
                />
              </div>
            </div>
          </section>
        )}

        {/* Training hyperparameters */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-3">Training Parameters</label>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Learning Rate</label>
              <input
                type="number"
                step="1e-5"
                min="1e-6"
                max="1"
                value={config.learning_rate}
                onChange={(e) => setField("learning_rate", parseFloat(e.target.value) || 3e-4)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Head LR Multiplier</label>
              <input
                type="number"
                step="0.5"
                min="0.1"
                max="100"
                value={config.head_lr_multiplier}
                onChange={(e) => setField("head_lr_multiplier", parseFloat(e.target.value) || 10)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Epochs</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={config.epochs}
                onChange={(e) => setField("epochs", parseInt(e.target.value) || 10)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Min Tag Count
              </label>
              <input
                type="number"
                min={1}
                value={config.vocab_min_count}
                onChange={(e) => setField("vocab_min_count", parseInt(e.target.value) || 1)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Batch Size</label>
              <input
                type="number"
                min={1}
                max={512}
                value={config.batch_size}
                onChange={(e) => setField("batch_size", parseInt(e.target.value) || 32)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Num Workers</label>
              <input
                type="number"
                min={0}
                max={16}
                value={config.num_workers}
                onChange={(e) => setField("num_workers", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Num Workers Override
                <span className="text-gray-500 ml-1">(0 = force single-process)</span>
              </label>
              <input
                type="number"
                min={0}
                max={16}
                placeholder="not set"
                value={config.num_workers_override ?? ""}
                onChange={(e) => {
                  const v = e.target.value === "" ? null : parseInt(e.target.value);
                  setField("num_workers_override", v);
                }}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Save Every N Steps
                <span className="text-gray-500 ml-1">(0 = disabled)</span>
              </label>
              <input
                type="number"
                min={0}
                step={100}
                value={config.save_every_n_steps}
                onChange={(e) => setField("save_every_n_steps", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Save Every N Epochs
                <span className="text-gray-500 ml-1">(0 = disabled)</span>
              </label>
              <input
                type="number"
                min={0}
                step={1}
                value={config.save_every_n_epochs ?? 0}
                onChange={(e) => setField("save_every_n_epochs", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Keep Last N Checkpoints
                <span className="text-gray-500 ml-1">(0 = keep all)</span>
              </label>
              <input
                type="number"
                min={0}
                value={config.keep_last_n_checkpoints}
                onChange={(e) => setField("keep_last_n_checkpoints", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">
                Checkpoint Save Mode
              </label>
              <select
                value={config.checkpoint_save_mode}
                onChange={(e) => setField("checkpoint_save_mode", e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              >
                <option value="lora">LoRA only (compact, requires base encoder)</option>
                <option value="merged">Merged full model (standalone, larger file)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Warmup Steps</label>
              <input
                type="number"
                min={0}
                max={10000}
                value={config.warmup_steps}
                onChange={(e) => setField("warmup_steps", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Validate Every (epochs)</label>
              <input
                type="number"
                min={1}
                max={100}
                value={config.validate_every}
                onChange={(e) => setField("validate_every", parseInt(e.target.value) || 1)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="col-span-2">
              <label className="block text-xs text-gray-400 mb-1">Validation Split</label>
              <div className="flex gap-2">
                <select
                  value={config.val_split_mode}
                  onChange={(e) => setField("val_split_mode", e.target.value)}
                  className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                >
                  <option value="percent">Percent (%)</option>
                  <option value="fixed">Fixed (samples)</option>
                </select>
                {config.val_split_mode === "fixed" ? (
                  <input
                    type="number"
                    min={1}
                    step={100}
                    value={config.val_fixed_size}
                    onChange={(e) => setField("val_fixed_size", parseInt(e.target.value) || 500)}
                    className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                    placeholder="500"
                  />
                ) : (
                  <input
                    type="number"
                    min={1}
                    max={50}
                    step={1}
                    value={Math.round((config.val_split ?? 0.05) * 100)}
                    onChange={(e) => setField("val_split", (parseInt(e.target.value) || 5) / 100)}
                    className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                    placeholder="5"
                  />
                )}
                <span className="flex items-center text-xs text-gray-500">
                  {config.val_split_mode === "fixed" ? "samples" : "%"}
                </span>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Training F1 Eval Interval (steps)</label>
              <input
                type="number"
                min={0}
                step={10}
                value={config.train_f1_eval_every_n_steps ?? 100}
                onChange={(e) => setField("train_f1_eval_every_n_steps", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Threshold Search Interval (steps)</label>
              <input
                type="number"
                min={0}
                step={100}
                value={config.train_f1_threshold_search_every_n_steps ?? 500}
                onChange={(e) => setField("train_f1_threshold_search_every_n_steps", parseInt(e.target.value) || 0)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Initial Threshold</label>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={config.train_f1_initial_threshold ?? 0.35}
                onChange={(e) => setField("train_f1_initial_threshold", parseFloat(e.target.value) || 0.35)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Training F1 Buffer (batches)</label>
              <input
                type="number"
                min={1}
                max={256}
                step={1}
                value={config.train_f1_buffer_batches ?? 16}
                onChange={(e) => setField("train_f1_buffer_batches", parseInt(e.target.value) || 16)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </section>

        {/* Optimizer */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Optimizer</label>
          <select
            value={config.optimizer}
            onChange={(e) => setField("optimizer", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="adamw">AdamW</option>
            <option value="adamw8bit">AdamW 8-bit</option>
            <option value="lion8bit">Lion 8-bit</option>
          </select>
        </section>

        {/* Mixed precision */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-2">Mixed Precision</label>
          <select
            value={config.mixed_precision}
            onChange={(e) => setField("mixed_precision", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
            <option value="fp32">fp32 (no AMP)</option>
          </select>
        </section>

        {/* Loss function selector */}
        <section>
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Loss Function
          </label>
          <select
            value={config.loss_function}
            onChange={(e) => setField("loss_function", e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            <option value="asl">ASL (Asymmetric Loss)</option>
            <option value="cs_asl">CS-ASL (Continuous Symmetric ASL)</option>
            <option value="h_cs_asl">H-CS-ASL (Hierarchical CS-ASL)</option>
            <option value="la_s_asl">LA-S-ASL (Logit-Adjusted Symmetric ASL)</option>
            <option value="fw_bbce">FW-BBCE (Fisher-Weighted Balanced BCE)</option>
          </select>
        </section>

        {/* ASL parameters */}
        {config.loss_function === "asl" && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Asymmetric Loss Parameters
            </label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">gamma_neg</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="10"
                  value={config.loss_gamma_neg}
                  onChange={(e) => setField("loss_gamma_neg", parseFloat(e.target.value) || 4)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">gamma_pos</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="10"
                  value={config.loss_gamma_pos}
                  onChange={(e) => setField("loss_gamma_pos", parseFloat(e.target.value) || 1)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">clip</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="0.5"
                  value={config.loss_clip}
                  onChange={(e) => setField("loss_clip", parseFloat(e.target.value) || 0)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </section>
        )}

        {/* CS-ASL / H-CS-ASL / LA-S-ASL shared parameters */}
        {(["cs_asl", "h_cs_asl", "la_s_asl"] as string[]).includes(config.loss_function ?? "") && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              π-Adjusted Loss Parameters
            </label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">gamma0</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="10"
                  value={config.loss_gamma0}
                  onChange={(e) => setField("loss_gamma0", parseFloat(e.target.value) || 4)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">m0</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={config.loss_m0}
                  onChange={(e) => setField("loss_m0", parseFloat(e.target.value) || 0.2)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">beta</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  max="10"
                  value={config.loss_beta}
                  onChange={(e) => setField("loss_beta", parseFloat(e.target.value) || 2)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              {config.loss_function !== "la_s_asl" && (
                <div>
                  <label className="block text-xs text-gray-400 mb-1">rho</label>
                  <input
                    type="number"
                    step="0.05"
                    min="0"
                    max="1"
                    value={config.loss_rho}
                    onChange={(e) => setField("loss_rho", parseFloat(e.target.value) || 0.5)}
                    className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
              )}
              <div>
                <label className="block text-xs text-gray-400 mb-1">clip</label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="0.5"
                  value={config.loss_clip}
                  onChange={(e) => setField("loss_clip", parseFloat(e.target.value) || 0)}
                  className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </section>
        )}

        {/* H-CS-ASL: label weighting */}
        {config.loss_function === "h_cs_asl" && (
          <section>
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Label Weighting (H-CS-ASL)
            </label>
            <select
              value={config.loss_label_weight}
              onChange={(e) => setField("loss_label_weight", e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="fisher">Fisher information</option>
              <option value="entropy_fisher">Entropy × Fisher</option>
              <option value="effective">Effective number</option>
            </select>
          </section>
        )}

        {/* Boolean options */}
        <section className="space-y-3">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.gradient_checkpointing}
              onChange={(e) => setField("gradient_checkpointing", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">Gradient Checkpointing</span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.use_flash_attention}
              onChange={(e) => setField("use_flash_attention", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">
              FlashAttention-2 (encoder; requires bf16/fp16)
            </span>
          </label>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={config.save_best_only}
              onChange={(e) => setField("save_best_only", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">Save Best Only (by F1)</span>
          </label>
        </section>

        {/* Online Danbooru Augmentation */}
        <section className="space-y-3 border-t border-gray-700 pt-4">
          <h3 className="text-sm font-semibold text-gray-300">Online Danbooru Augmentation</h3>
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={!!config.enable_danbooru_augmentation}
              onChange={(e) => setField("enable_danbooru_augmentation", e.target.checked)}
              className="accent-blue-500"
            />
            <span className="text-sm text-gray-300">Enable Danbooru Augmentation</span>
          </label>

          {config.enable_danbooru_augmentation && (
            <div className="space-y-3 pl-2">
              <div className="p-3 bg-yellow-900/30 border border-yellow-700 rounded text-xs text-yellow-300">
                ⚠ Danbooru API rate limit: minimum 1.4 s between calls. Avoid accessing Danbooru from other
                processes while this is running.
              </div>

              {/* General augmentation pipeline settings (apply to all modes) */}
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-48">Injection interval (base steps)</label>
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={config.danbooru_injection_interval ?? 4}
                  onChange={(e) => setField("danbooru_injection_interval", parseInt(e.target.value) || 4)}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
                <span className="text-xs text-gray-500">
                  Interrupt-batch every N base steps. LR scheduler & global_step do not advance
                  on injection batches (resume-safe).
                </span>
              </div>

              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-48">Injection batch size ratio</label>
                <input
                  type="number"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={config.danbooru_injection_batch_size_ratio ?? 1.0}
                  onChange={(e) => setField("danbooru_injection_batch_size_ratio", parseFloat(e.target.value) || 1.0)}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
                <span className="text-xs text-gray-500">1.0 = full batch (B), 0.5 = B/2.</span>
              </div>

              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-48">Min post score</label>
                <input
                  type="number"
                  min={0}
                  value={config.danbooru_min_score ?? 0}
                  onChange={(e) => setField("danbooru_min_score", parseInt(e.target.value) || 0)}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-48">Max posts per query</label>
                <input
                  type="number"
                  min={1}
                  max={1000}
                  value={config.danbooru_max_posts_per_query ?? 200}
                  onChange={(e) => setField("danbooru_max_posts_per_query", parseInt(e.target.value) || 200)}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
                <span className="text-xs text-gray-500">
                  Cap on posts fetched per single Danbooru search (page 1). Per <strong>resolved tag</strong>
                  when Query Expand is on (wildcards are expanded at resolution, then each tag is searched
                  individually); per <strong>query string</strong> (whole wildcard) when off. The per-mode
                  "Posts/tag/epoch" caps below are the cumulative per-tag limit across an epoch — effective
                  per-tag ≈ min of the two.
                </span>
              </div>

              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-400 w-48">Buffer size (samples)</label>
                <input
                  type="number"
                  min={0}
                  max={512}
                  value={config.danbooru_buffer_size ?? 0}
                  onChange={(e) => {
                    const v = parseInt(e.target.value);
                    setField("danbooru_buffer_size", Number.isFinite(v) && v > 0 ? v : null);
                  }}
                  className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                />
                <span className="text-xs text-gray-500">0 = auto (2 × batch_size).</span>
              </div>

              {/* Query mode (first-class collection mode; owns the Tag Queries) */}
              <div className="pt-2 border-t border-gray-700">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={config.danbooru_query_enable ?? true}
                    onChange={(e) => setField("danbooru_query_enable", e.target.checked)}
                    className="w-4 h-4 rounded"
                  />
                  <span className="text-sm text-gray-300">
                    Query mode (collect via the Danbooru queries below; uses the Query weight at the bottom)
                  </span>
                </label>

                {config.danbooru_query_enable && (
                  <div className="pl-6 space-y-2 mt-2">
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Tag Queries (one per line)</label>
                      <textarea
                        rows={4}
                        value={config.danbooru_tags ?? ""}
                        onChange={(e) => setField("danbooru_tags", e.target.value)}
                        placeholder={"1girl score:>50\n1boy score:>30\nsolo -monochrome score:>20"}
                        className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500 resize-y"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Danbooru tag search queries. Space-separated terms are ANDed.
                        Prefix a tag with <code>!</code> or <code>-</code> to exclude
                        (e.g. <code>solo !furry</code>). Use a negative-style query line to
                        collect posts that are deliberately outside your target tag set —
                        this improves robustness against false positives.
                      </p>
                    </div>

                    <label className="flex items-center gap-2 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={!!config.danbooru_query_expand_enable}
                        onChange={(e) => setField("danbooru_query_expand_enable", e.target.checked)}
                        className="w-4 h-4 rounded border-gray-600 bg-gray-800 text-blue-500 focus:ring-0"
                      />
                      <span className="text-sm text-gray-300">Expand vocabulary from queries</span>
                      <span className="text-xs text-gray-500">(resolve query tags/wildcards → add new tags, collect per-tag)</span>
                    </label>

                    {config.danbooru_query_expand_enable && (
                      <div className="pl-6 grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Min post_count</label>
                          <input
                            type="number"
                            min={0}
                            value={config.danbooru_query_new_tag_min_count ?? 200}
                            onChange={(e) => setField("danbooru_query_new_tag_min_count", parseInt(e.target.value) || 0)}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Top-K per query (0=∞)</label>
                          <input
                            type="number"
                            min={0}
                            value={config.danbooru_query_resolve_top_k ?? 50}
                            onChange={(e) => setField("danbooru_query_resolve_top_k", parseInt(e.target.value) || 0)}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Max expanded (run, 0=∞)</label>
                          <input
                            type="number"
                            min={0}
                            value={config.danbooru_query_max_expanded_tags ?? 0}
                            onChange={(e) => setField("danbooru_query_max_expanded_tags", parseInt(e.target.value) || 0)}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Re-resolve interval (s)</label>
                          <input
                            type="number"
                            min={60}
                            value={config.danbooru_query_resolve_interval ?? 3600}
                            onChange={(e) => setField("danbooru_query_resolve_interval", parseInt(e.target.value) || 3600)}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-gray-400 mb-1">Posts/tag/epoch (0=∞)</label>
                          <input
                            type="number"
                            min={0}
                            value={config.danbooru_query_collect_per_epoch ?? 0}
                            onChange={(e) => setField("danbooru_query_collect_per_epoch", parseInt(e.target.value) || 0)}
                            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                          />
                        </div>
                        <div className="col-span-2 text-xs text-gray-500">
                          Resolved tags are collected PER-TAG within the Query pool (bounded by the Query
                          weight), so a wildcard that resolves to N tags contributes N collection units —
                          not one. Eligible categories: General / Copyright / Character.
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Vocab Expansion */}
              <div className="pt-2 border-t border-gray-700">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!config.danbooru_vocab_expand}
                    onChange={(e) => setField("danbooru_vocab_expand", e.target.checked)}
                    className="w-4 h-4 rounded"
                  />
                  <span className="text-sm text-gray-300">Vocab Expansion (auto-add new Danbooru tags during training)</span>
                </label>

                {config.danbooru_vocab_expand && (
                  <div className="mt-3 space-y-3 pl-7">
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Min post count</label>
                      <input
                        type="number"
                        min={1}
                        value={config.danbooru_new_tag_min_count ?? 200}
                        onChange={(e) => setField("danbooru_new_tag_min_count", parseInt(e.target.value) || 200)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Lookback days</label>
                      <input
                        type="number"
                        min={1}
                        max={365}
                        value={config.danbooru_new_tag_lookback_days ?? 90}
                        onChange={(e) => setField("danbooru_new_tag_lookback_days", parseInt(e.target.value) || 90)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Survey interval (sec)</label>
                      <input
                        type="number"
                        min={60}
                        value={config.danbooru_new_tag_survey_interval ?? 3600}
                        onChange={(e) => setField("danbooru_new_tag_survey_interval", parseInt(e.target.value) || 3600)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Max dynamic tags</label>
                      <input
                        type="number"
                        min={0}
                        value={config.danbooru_max_dynamic_tags ?? 0}
                        onChange={(e) => setField("danbooru_max_dynamic_tags", parseInt(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">
                        LRU cap on the persisted new-tag collection list (0 = unlimited). Evicts the
                        least-recently-collected tag; a regressed tag is re-collected by Low-F1.
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Posts/tag/epoch (0=∞)</label>
                      <input
                        type="number"
                        min={0}
                        value={config.danbooru_new_tag_collect_per_epoch ?? 0}
                        onChange={(e) => setField("danbooru_new_tag_collect_per_epoch", parseInt(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">
                        Per-tag per-epoch collection cap so a high-post_count new tag does not
                        monopolise the injected batches. Reset each epoch.
                      </span>
                    </div>

                    {/* Tag categories to discover */}
                    <div className="flex items-start gap-3">
                      <label className="text-xs text-gray-400 w-48 pt-1">Tag categories</label>
                      <div className="flex flex-wrap gap-3">
                        {[
                          { code: 0, label: "General" },
                          { code: 4, label: "Character" },
                          { code: 3, label: "Copyright" },
                          { code: 1, label: "Artist" },
                          { code: 5, label: "Meta" },
                        ].map(({ code, label }) => {
                          const cats = config.danbooru_new_tag_categories ?? [0, 3, 4];
                          const checked = cats.includes(code);
                          return (
                            <label key={code} className="flex items-center gap-1.5 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={(e) => {
                                  const cur = config.danbooru_new_tag_categories ?? [0, 3, 4];
                                  const next = e.target.checked
                                    ? [...cur, code].sort((a, b) => a - b)
                                    : cur.filter((c) => c !== code);
                                  setField("danbooru_new_tag_categories", next);
                                }}
                                className="w-3.5 h-3.5 rounded accent-blue-500"
                              />
                              <span className="text-xs text-gray-300">{label}</span>
                            </label>
                          );
                        })}
                      </div>
                    </div>

                    {/* Per-category min post count overrides */}
                    {(config.danbooru_new_tag_categories ?? [0, 3, 4]).length > 0 && (
                      <div className="flex items-start gap-3">
                        <label className="text-xs text-gray-400 w-48 pt-1">Per-category min count</label>
                        <div className="flex flex-col gap-2">
                          <p className="text-xs text-gray-500">
                            Override the shared min post count per category (blank = use the shared value).
                            Copyright post counts dwarf individual characters, so a higher copyright bar
                            keeps it from over-dominating the discovered tags.
                          </p>
                          {[
                            { code: 0, label: "General" },
                            { code: 4, label: "Character" },
                            { code: 3, label: "Copyright" },
                            { code: 1, label: "Artist" },
                            { code: 5, label: "Meta" },
                          ]
                            .filter(({ code }) => (config.danbooru_new_tag_categories ?? [0, 3, 4]).includes(code))
                            .map(({ code, label }) => {
                              const map = config.danbooru_new_tag_min_count_by_cat ?? {};
                              return (
                                <div key={code} className="flex items-center gap-3">
                                  <label className="text-xs text-gray-400 w-24">{label}</label>
                                  <input
                                    type="number" min={1}
                                    value={map[String(code)] ?? ""}
                                    placeholder={String(config.danbooru_new_tag_min_count ?? 200)}
                                    onChange={(e) => {
                                      const next: Record<string, number> = { ...(config.danbooru_new_tag_min_count_by_cat ?? {}) };
                                      const v = parseInt(e.target.value);
                                      if (Number.isFinite(v) && v > 0) next[String(code)] = v;
                                      else delete next[String(code)];
                                      setField("danbooru_new_tag_min_count_by_cat", next);
                                    }}
                                    className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                                  />
                                </div>
                              );
                            })}
                        </div>
                      </div>
                    )}

                    {/* Co-occurrence discovery (created-at independent) */}
                    <div className="pt-2 border-t border-gray-700/50">
                      <label className="flex items-center gap-3 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={!!config.danbooru_cooc_expand_enable}
                          onChange={(e) => setField("danbooru_cooc_expand_enable", e.target.checked)}
                          className="w-4 h-4 rounded"
                        />
                        <span className="text-sm text-gray-300">
                          Co-occurrence discovery (add vocab-absent tags seen in collected posts)
                        </span>
                      </label>
                      {config.danbooru_cooc_expand_enable && (
                        <div className="mt-2 ml-7 space-y-2">
                          <p className="text-xs text-gray-500">
                            Adds tags that co-occur in collected posts at least N times, filtered to its own
                            categories below (independent of the surveyor&apos;s). Unlike the surveyor it ignores
                            creation date, so it also catches older tags missing from your vocabulary (e.g. a
                            new character&apos;s copyright — added even if Copyright is off in the surveyor above).
                          </p>
                          <div className="flex items-center gap-3">
                            <label className="text-xs text-gray-400 w-48">Min co-occurrence count</label>
                            <input
                              type="number" min={1}
                              value={config.danbooru_cooc_min_count ?? 50}
                              onChange={(e) => setField("danbooru_cooc_min_count", parseInt(e.target.value) || 1)}
                              className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                            />
                            <span className="text-xs text-gray-500">Times a tag must appear before it is added.</span>
                          </div>
                          <div className="flex items-start gap-3">
                            <label className="text-xs text-gray-400 w-48 pt-1">Categories</label>
                            <div className="flex flex-wrap gap-3">
                              {[
                                { code: 0, label: "General" },
                                { code: 4, label: "Character" },
                                { code: 3, label: "Copyright" },
                                { code: 1, label: "Artist" },
                                { code: 5, label: "Meta" },
                              ].map(({ code, label }) => {
                                const cats = config.danbooru_cooc_categories ?? [0, 3, 4];
                                return (
                                  <label key={code} className="flex items-center gap-1.5 cursor-pointer">
                                    <input
                                      type="checkbox"
                                      checked={cats.includes(code)}
                                      onChange={(e) => {
                                        const cur = config.danbooru_cooc_categories ?? [0, 3, 4];
                                        const next = e.target.checked
                                          ? [...cur, code].sort((a, b) => a - b)
                                          : cur.filter((c) => c !== code);
                                        setField("danbooru_cooc_categories", next);
                                      }}
                                      className="w-3.5 h-3.5 rounded accent-blue-500"
                                    />
                                    <span className="text-xs text-gray-300">{label}</span>
                                  </label>
                                );
                              })}
                            </div>
                          </div>
                          {/* Active collection of promoted cooc tags */}
                          <div className="pt-2 border-t border-gray-700/50 space-y-2">
                            <p className="text-xs text-gray-500">
                              Active collection: keep collecting promoted cooc tags across epochs
                              (order:random, lightly), so they get trained — not just expanded.
                            </p>
                            <div className="flex items-center gap-3">
                              <label className="text-xs text-gray-400 w-48">Active weight</label>
                              <input
                                type="number" min={0} step={0.05}
                                value={config.danbooru_query_weight_cooc ?? 0.1}
                                onChange={(e) => setField("danbooru_query_weight_cooc", parseFloat(e.target.value) || 0)}
                                className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                              />
                              <span className="text-xs text-gray-500">vs new-tag 1.0. 0 = expand vocab only (no active collection).</span>
                            </div>
                            <div className="flex items-center gap-3">
                              <label className="text-xs text-gray-400 w-48">Collect per epoch (per tag)</label>
                              <input
                                type="number" min={0}
                                value={config.danbooru_cooc_collect_per_epoch ?? 50}
                                onChange={(e) => setField("danbooru_cooc_collect_per_epoch", parseInt(e.target.value) || 0)}
                                className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                              />
                              <span className="text-xs text-gray-500">Balanced quota: max images per cooc tag per epoch.</span>
                            </div>
                            <label className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={config.danbooru_cooc_order_random ?? true}
                                onChange={(e) => setField("danbooru_cooc_order_random", e.target.checked)}
                                className="w-3.5 h-3.5 rounded accent-blue-500"
                              />
                              <span className="text-xs text-gray-300">order:random (diverse sampling; weakens spurious companion co-occurrence)</span>
                            </label>
                          </div>
                        </div>
                      )}
                    </div>

                  </div>
                )}
              </div>

              {/* Low-F1 deficiency collection */}
              <div className="pt-2 border-t border-gray-700">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!config.danbooru_low_f1_enable}
                    onChange={(e) => setField("danbooru_low_f1_enable", e.target.checked)}
                    className="w-4 h-4 rounded"
                  />
                  <span className="text-sm text-gray-300">
                    Low-F1 Deficiency Collection (collect existing vocab tags with low per-tag F1)
                  </span>
                </label>
                {config.danbooru_low_f1_enable && (
                  <div className="mt-2 ml-7 space-y-2">
                    <p className="text-xs text-gray-500">
                      Requires training F1 metrics enabled. Targets are recomputed at the threshold-search
                      interval; existing vocab tags with a valid F1 below the threshold are collected
                      (untrained tags with no F1 yet are excluded).
                    </p>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">F1 threshold</label>
                      <input
                        type="number" min={0} max={1} step={0.01}
                        value={config.danbooru_low_f1_threshold ?? 0.5}
                        onChange={(e) => setField("danbooru_low_f1_threshold", parseFloat(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Tags below this F1 are targeted.</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Top-K tags</label>
                      <input
                        type="number" min={1} step={50}
                        value={config.danbooru_low_f1_top_k ?? 500}
                        onChange={(e) => setField("danbooru_low_f1_top_k", parseInt(e.target.value) || 1)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Max worst-F1 tags to collect.</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Min Danbooru posts</label>
                      <input
                        type="number" min={1} max={200} step={10}
                        value={config.danbooru_low_f1_min_posts ?? 50}
                        onChange={(e) => setField("danbooru_low_f1_min_posts", parseInt(e.target.value) || 1)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Skip tags with fewer page-1 posts than this (≤200).</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Posts/tag/epoch (0=∞)</label>
                      <input
                        type="number" min={0}
                        value={config.danbooru_low_f1_collect_per_epoch ?? 0}
                        onChange={(e) => setField("danbooru_low_f1_collect_per_epoch", parseInt(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">
                        Per-tag per-epoch collection cap so a high-post_count low-F1 tag does not
                        monopolise the injected batches. Reset each epoch.
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Train-count deficiency collection */}
              <div className="pt-2 border-t border-gray-700">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!config.danbooru_train_count_enable}
                    onChange={(e) => setField("danbooru_train_count_enable", e.target.checked)}
                    className="w-4 h-4 rounded"
                  />
                  <span className="text-sm text-gray-300">
                    Train-count Deficiency Collection (rebalance under-exposed tags by cumulative training count)
                  </span>
                </label>

                {config.danbooru_train_count_enable && (
                  <div className="pl-6 space-y-2 mt-2">
                    <p className="text-xs text-gray-500">
                      Targets tags whose cumulative training-exposure count is well below what their
                      <em> current</em> per-epoch rate implies — i.e. characters/tags added or grown
                      mid-run (e.g. a newly viral character). Genuinely-rare-but-stable tags are excluded
                      (the Low-F1 path covers those). Activates after ≥2 completed epochs. Keeps the
                      per-tag count accumulating even if training-F1 metrics are off.
                    </p>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Min deficit ratio</label>
                      <input
                        type="number" min={0} max={1} step={0.05}
                        value={config.danbooru_train_count_min_deficit_ratio ?? 0.3}
                        onChange={(e) => setField("danbooru_train_count_min_deficit_ratio", parseFloat(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Target tags ≥ this fraction under-exposed (0–1).</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Top-K targeted</label>
                      <input
                        type="number" min={1}
                        value={config.danbooru_train_count_top_k ?? 500}
                        onChange={(e) => setField("danbooru_train_count_top_k", parseInt(e.target.value) || 1)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Cap on number of worst-deficit tags collected.</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Min exposures/epoch</label>
                      <input
                        type="number" min={1}
                        value={config.danbooru_train_count_min_per_epoch ?? 10}
                        onChange={(e) => setField("danbooru_train_count_min_per_epoch", parseInt(e.target.value) || 1)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Noise floor: ignore tags with fewer positives/epoch.</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Min Danbooru posts</label>
                      <input
                        type="number" min={1} max={200} step={10}
                        value={config.danbooru_train_count_min_posts ?? 50}
                        onChange={(e) => setField("danbooru_train_count_min_posts", parseInt(e.target.value) || 1)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">Skip tags with fewer page-1 posts than this (≤200).</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <label className="text-xs text-gray-400 w-48">Posts/tag/epoch (0=∞)</label>
                      <input
                        type="number" min={0}
                        value={config.danbooru_train_count_collect_per_epoch ?? 0}
                        onChange={(e) => setField("danbooru_train_count_collect_per_epoch", parseInt(e.target.value) || 0)}
                        className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
                      />
                      <span className="text-xs text-gray-500">
                        Per-tag per-epoch collection cap. Reset each epoch.
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Score-based quality tag (applies to every collected post) */}
              <div className="pt-2 border-t border-gray-700 space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!config.danbooru_quality_tag_enable}
                    onChange={(e) => setField("danbooru_quality_tag_enable", e.target.checked)}
                  />
                  <span className="text-sm text-gray-300">Add quality tag from Danbooru score</span>
                </label>
                <p className="text-xs text-gray-500">
                  Derive a quality tag from each collected post&apos;s Danbooru score and add it to the
                  label set. It trains only for tiers present in the vocabulary (others are ignored).
                </p>
                {config.danbooru_quality_tag_enable && (
                  <div className="space-y-2 pl-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={!!config.danbooru_quality_tag_attach_negative}
                        onChange={(e) => setField("danbooru_quality_tag_attach_negative", e.target.checked)}
                      />
                      <span className="text-xs text-gray-300">Also attach low/worst-quality tiers</span>
                    </label>
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">
                        Thresholds — one &quot;&lt;min_score&gt; &lt;tag&gt;&quot; per line (empty = Animagine XL 3.0 default)
                      </label>
                      <textarea
                        value={config.danbooru_quality_tag_thresholds ?? ""}
                        onChange={(e) => setField("danbooru_quality_tag_thresholds", e.target.value)}
                        rows={4}
                        placeholder={"151 masterpiece\n100 best quality\n75 high quality\n25 medium quality\n0 normal quality\n-5 low quality\n-1000000 worst quality"}
                        className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-blue-500"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Collection path weights (last: balances the enabled paths above) */}
              <div className="pt-2 border-t border-gray-700 space-y-2">
                <div className="text-sm text-gray-300">Collection Path Weights</div>
                <p className="text-xs text-gray-500">
                  Weighted random selection among the enabled collection paths. A path with no
                  available queries is skipped and the remaining weights renormalize. A disabled
                  path (New-tag / Low-F1 unchecked above) is forced to weight 0.
                </p>
                <div className="flex items-center gap-3">
                  <label className="text-xs text-gray-400 w-48">Query weight</label>
                  <input
                    type="number" min={0} step={0.1}
                    disabled={!config.danbooru_query_enable}
                    value={config.danbooru_query_enable ? (config.danbooru_query_weight_static ?? 1.0) : 0}
                    onChange={(e) => setField("danbooru_query_weight_static", parseFloat(e.target.value) || 0)}
                    className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-40"
                  />
                  <span className="text-xs text-gray-500">User queries (per-tag when Expand is on, else per-string).</span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="text-xs text-gray-400 w-48">New-tag weight</label>
                  <input
                    type="number" min={0} step={0.1}
                    disabled={!config.danbooru_vocab_expand}
                    value={config.danbooru_vocab_expand ? (config.danbooru_query_weight_new_tag ?? 1.0) : 0}
                    onChange={(e) => setField("danbooru_query_weight_new_tag", parseFloat(e.target.value) || 0)}
                    className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <span className="text-xs text-gray-500">Discovered new tags (0 unless Vocab Expansion is enabled).</span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="text-xs text-gray-400 w-48">Low-F1 weight</label>
                  <input
                    type="number" min={0} step={0.1}
                    disabled={!config.danbooru_low_f1_enable}
                    value={config.danbooru_low_f1_enable ? (config.danbooru_query_weight_low_f1 ?? 1.0) : 0}
                    onChange={(e) => setField("danbooru_query_weight_low_f1", parseFloat(e.target.value) || 0)}
                    className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <span className="text-xs text-gray-500">Deficiency collection (0 unless Low-F1 is enabled).</span>
                </div>
                <div className="flex items-center gap-3">
                  <label className="text-xs text-gray-400 w-48">Train-count weight</label>
                  <input
                    type="number" min={0} step={0.1}
                    disabled={!config.danbooru_train_count_enable}
                    value={config.danbooru_train_count_enable ? (config.danbooru_query_weight_train_count ?? 1.0) : 0}
                    onChange={(e) => setField("danbooru_query_weight_train_count", parseFloat(e.target.value) || 0)}
                    className="w-24 bg-gray-800 border border-gray-600 rounded px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <span className="text-xs text-gray-500">Exposure balancing (0 unless Train-count is enabled).</span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Error */}
        {error && (
          <div className="p-3 bg-red-900/30 border border-red-700 rounded text-sm text-red-400">
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
          disabled={saving}
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
