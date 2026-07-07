/**
 * Loop Generation setting inheritance — shared source of truth.
 *
 * Loop-generation steps inherit their parameters from the "main" generation
 * parameters. Inheritance is organized into genres. Each genre either:
 *   - is copied wholesale from the main params (genre toggle ON), or
 *   - falls back to the CURRENT MAIN VALUES per-field (genre toggle OFF),
 *     with any explicit per-step override still winning.
 *
 * This module is the single place where the genre -> field mapping lives, so
 * that new fields are added once and every panel stays consistent.
 *
 * IMPORTANT: hardcoded literal fallbacks (steps||20, cfg||7, nag_scale??5.0 ...)
 * must NOT be reintroduced in the panels. When a genre toggle is OFF, the
 * fallback is always the corresponding main value (mainParams.X).
 */

import type { LoopGenerationConfig, LoopGenerationStep } from "@/components/generation/LoopGenerationPanel";

/**
 * Genre -> list of stepParams keys it governs.
 *
 * - sampling / cfgSchedule / nag have per-step override UI in LoopGenerationPanel
 *   and are gated by their own genre toggle (use_main_sampling, ...).
 * - acceleration / postProcess currently have NO per-step override UI, so they
 *   are always inherited from main. They are listed here for documentation and
 *   so future per-step fields can be added in exactly one place.
 * - modelEnvironment params are model-global (mutate the pipeline, not a single
 *   image). They are ALWAYS inherited from main implicitly (no toggle).
 */
export const LOOP_GENRE_FIELDS = {
  sampling: [
    "steps",
    "cfg_scale",
    "sampler",
    "schedule_type",
    "seed",
    "ancestral_seed",
  ],
  cfgSchedule: [
    "cfg_schedule_type",
    "cfg_schedule_min",
    "cfg_schedule_max",
    "cfg_schedule_power",
    "cfg_rescale_snr_alpha",
    "dynamic_threshold_percentile",
    "dynamic_threshold_mimic_scale",
  ],
  nag: [
    "nag_enable",
    "nag_scale",
    "nag_tau",
    "nag_alpha",
    "nag_sigma_end",
    "nag_negative_prompt",
  ],
  acceleration: [
    "spectrum_enable",
    "spectrum_w",
    "spectrum_w_decay",
    "spectrum_delta_cap",
    "spectrum_m",
    "spectrum_lam",
    "spectrum_warmup_steps",
    "spectrum_window_size",
    "spectrum_flex_window",
    "spectrum_tail",
    "spectrum_feature_mode",
    "spectrum_cache_branch",
    "spectrum_max_cache",
    "fbcache_enable",
    "fbcache_threshold",
    "fbcache_warmup_steps",
  ],
  postProcess: [
    "color_flatten_strength",
    "flatten_in_loop",
    "flatten_in_loop_last_steps",
    "flatten_in_loop_min_region",
    "vae_drift_correction", // img2img / inpaint only
  ],
  /**
   * Model / Environment (model-global). Always inherited from main; never
   * varies per loop step. Block swap + text_encoder_quantization were
   * previously NOT inherited (coverage gaps) and are now included here.
   */
  modelEnvironment: [
    "unet_quantization",
    "text_encoder_quantization",
    "cpu_text_encoding",
    "use_torch_compile",
    "vae_tiling",
    "vae_tile_threshold",
    "attention_type",
    "enable_block_swap",
    "blocks_to_swap",
    "block_swap_h2d_only",
    "block_swap_ring_size",
  ],
} as const;

/**
 * Copy every model-global (Model/Environment) field from mainParams into target.
 * These are inherited implicitly by loop generation (no per-step variation),
 * including the previously-missing text_encoder_quantization + block swap group.
 */
export function applyModelEnvironmentInheritance(
  target: Record<string, any>,
  mainParams: Record<string, any>,
): void {
  for (const key of LOOP_GENRE_FIELDS.modelEnvironment) {
    target[key] = mainParams[key];
  }
}

/**
 * Backward-compatible migration for a stored LoopGenerationConfig.
 *
 * Older configs only had a single `useMainSettings` boolean per step, which
 * governed sampling + Advanced CFG + NAG together. New configs use independent
 * genre toggles (use_main_sampling / use_main_cfg_schedule / use_main_nag).
 *
 * Migration rule: if a genre toggle is missing on a step, seed it from the
 * legacy `useMainSettings` value (defaulting to true when that is also absent).
 * This preserves the exact behavior existing users currently see.
 */
export function migrateLoopGenerationConfig(
  config: LoopGenerationConfig,
): LoopGenerationConfig {
  if (!config || !Array.isArray(config.steps)) {
    return config;
  }

  const migrated: LoopGenerationConfig = {
    ...config,
    steps: config.steps.map((step) => migrateLoopGenerationStep(step)),
  };
  return migrated;
}

export function migrateLoopGenerationStep(
  step: LoopGenerationStep,
): LoopGenerationStep {
  // Legacy single flag; default to true (old default) when absent.
  const legacy = step.useMainSettings ?? true;

  return {
    ...step,
    useMainSettings: legacy,
    use_main_sampling: step.use_main_sampling ?? legacy,
    use_main_cfg_schedule: step.use_main_cfg_schedule ?? legacy,
    use_main_nag: step.use_main_nag ?? legacy,
  };
}
