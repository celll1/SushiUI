"use client";

import type { TrainingRunCreateRequest } from "@/utils/api";
import { NumField } from "./trainingConfigDefinitions";

type UpdateParam = <K extends keyof TrainingRunCreateRequest>(
  key: K,
  value: TrainingRunCreateRequest[K]
) => void;

interface DanbooruAugmentationSectionProps {
  params: TrainingRunCreateRequest;
  updateParam: UpdateParam;
}

export default function DanbooruAugmentationSection({
  params,
  updateParam,
}: DanbooruAugmentationSectionProps) {
  return (
    <div className="border border-gray-700 rounded p-4 space-y-3">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={!!params.danbooru_aug_enable}
          onChange={(e) => updateParam("danbooru_aug_enable", e.target.checked)}
        />
        <h3 className="text-sm font-medium text-gray-300">Online Danbooru Augmentation</h3>
      </label>
      <p className="text-xs text-gray-500">
        Fetch extra training images from Danbooru during training and inject them as samples
        (interrupt-batch). No vocabulary expansion. Requires Latent Encoding Mode
        swap_onthefly or onthefly_gpu.
      </p>

      {params.danbooru_aug_enable && (
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Static queries (one per line)</label>
            <textarea
              value={params.danbooru_aug_queries ?? ""}
              onChange={(e) => updateParam("danbooru_aug_queries", e.target.value)}
              rows={3}
              placeholder={"1girl solo score:>=50\nhatsune_miku"}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm font-mono focus:outline-none focus:border-blue-500"
            />
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={!!params.danbooru_aug_deficiency_enable}
              onChange={(e) => updateParam("danbooru_aug_deficiency_enable", e.target.checked)}
            />
            <span className="text-xs text-gray-300">
              Auto-collect under-represented tags (dataset frequency based)
            </span>
          </label>

          {params.danbooru_aug_deficiency_enable && (
            <div className="grid grid-cols-2 gap-3">
              <NumField label="Deficiency min count" value={params.danbooru_aug_deficiency_min_count}
                onChange={(v) => updateParam("danbooru_aug_deficiency_min_count", v)} step={1} />
              <NumField label="Deficiency top-K" value={params.danbooru_aug_deficiency_top_k}
                onChange={(v) => updateParam("danbooru_aug_deficiency_top_k", v)} step={1} />
            </div>
          )}

          <div>
            <label className="block text-xs text-gray-400 mb-1">Manual deficiency tags (comma or newline)</label>
            <textarea
              value={params.danbooru_aug_deficiency_manual ?? ""}
              onChange={(e) => updateParam("danbooru_aug_deficiency_manual", e.target.value)}
              rows={2}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm font-mono focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <NumField label="Weight: static" value={params.danbooru_aug_weight_static}
              onChange={(v) => updateParam("danbooru_aug_weight_static", v)} step={0.1} />
            <NumField label="Weight: deficiency" value={params.danbooru_aug_weight_deficiency}
              onChange={(v) => updateParam("danbooru_aug_weight_deficiency", v)} step={0.1} />
            <NumField label="Injection interval (batches)" value={params.danbooru_aug_injection_interval}
              onChange={(v) => updateParam("danbooru_aug_injection_interval", v)} step={1} />
            <NumField label="Injection ratio (x batch)" value={params.danbooru_aug_injection_ratio}
              onChange={(v) => updateParam("danbooru_aug_injection_ratio", v)} step={0.1} />
            <NumField label="Min score" value={params.danbooru_aug_min_score}
              onChange={(v) => updateParam("danbooru_aug_min_score", v)} step={1} />
            <NumField label="Max posts / query" value={params.danbooru_aug_max_posts_per_query}
              onChange={(v) => updateParam("danbooru_aug_max_posts_per_query", v)} step={1} />
            <NumField label="API interval (s)" value={params.danbooru_aug_api_interval}
              onChange={(v) => updateParam("danbooru_aug_api_interval", v)} step={0.1} />
            <NumField label="DL speed (KB/s)" value={params.danbooru_aug_dl_speed_kbps}
              onChange={(v) => updateParam("danbooru_aug_dl_speed_kbps", v)} step={1} />
            <div>
              <label className="block text-xs text-gray-400 mb-1">Buffer size (blank=auto)</label>
              <input
                type="number"
                step={1}
                value={params.danbooru_aug_buffer_size ?? ""}
                onChange={(e) =>
                  updateParam("danbooru_aug_buffer_size", e.target.value === "" ? null : parseInt(e.target.value, 10))
                }
                className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500"
              />
            </div>
            <NumField label="Max caption tags (0=all)" value={params.danbooru_aug_max_caption_tags}
              onChange={(v) => updateParam("danbooru_aug_max_caption_tags", v)} step={1} />
          </div>

          {/* Download-speed safety (throttle/ban avoidance) */}
          <div className="border-t border-gray-700 pt-3 mt-1 space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!params.danbooru_speed_check_enable}
                onChange={(e) => updateParam("danbooru_speed_check_enable", e.target.checked)}
              />
              <span className="text-sm text-gray-300">Download-speed safety (throttle/ban avoidance)</span>
            </label>
            <p className="text-xs text-gray-500">
              Pause collection when download speed stays degraded (Danbooru often throttles before a
              hard ban). Robust to transient dips — a sustained slow streak is required. Live speed and
              manual resume are in the metrics panel.
            </p>
            {params.danbooru_speed_check_enable && (
              <div className="grid grid-cols-2 gap-2">
                <NumField label="Degraded below (KB/s)" value={params.danbooru_speed_degraded_kbps}
                  onChange={(v) => updateParam("danbooru_speed_degraded_kbps", v)} step={1} />
                <NumField label="Slow streak to trip" value={params.danbooru_speed_min_slow_streak}
                  onChange={(v) => updateParam("danbooru_speed_min_slow_streak", v)} step={1} />
                <NumField label="Sustained at least (s)" value={params.danbooru_speed_min_slow_seconds}
                  onChange={(v) => updateParam("danbooru_speed_min_slow_seconds", v)} step={1} />
                <NumField label="Cooldown (s)" value={params.danbooru_speed_cooldown_seconds}
                  onChange={(v) => updateParam("danbooru_speed_cooldown_seconds", v)} step={1} />
              </div>
            )}
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={!!params.danbooru_aug_include_rating_tag}
              onChange={(e) => updateParam("danbooru_aug_include_rating_tag", e.target.checked)}
            />
            <span className="text-xs text-gray-300">Include rating word in caption (general/sensitive/…)</span>
          </label>

          {/* Score-based quality tag */}
          <div className="border-t border-gray-700 pt-3 mt-1 space-y-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!params.danbooru_quality_tag_enable}
                onChange={(e) => updateParam("danbooru_quality_tag_enable", e.target.checked)}
              />
              <span className="text-xs text-gray-300">Add quality tag from Danbooru score</span>
            </label>
            {params.danbooru_quality_tag_enable && (
              <div className="space-y-2 pl-6">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={!!params.danbooru_quality_tag_attach_negative}
                    onChange={(e) => updateParam("danbooru_quality_tag_attach_negative", e.target.checked)}
                  />
                  <span className="text-xs text-gray-300">Also attach low/worst-quality tiers</span>
                </label>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">
                    Thresholds — one <code>&lt;min_score&gt; &lt;tag&gt;</code> per line (empty = Animagine XL 3.0 default)
                  </label>
                  <textarea
                    value={params.danbooru_quality_tag_thresholds ?? ""}
                    onChange={(e) => updateParam("danbooru_quality_tag_thresholds", e.target.value)}
                    rows={4}
                    placeholder={"151 masterpiece\n100 best quality\n75 high quality\n25 medium quality\n0 normal quality\n-5 low quality\n-1000000 worst quality"}
                    className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-xs font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Caption tag shuffle / dropout (dedicated — independent of the
              per-dataset caption processing). */}
          <div className="border-t border-gray-700 pt-3 mt-1 space-y-3">
            <p className="text-xs text-gray-400">
              Caption tag shuffle / dropout for injected samples (separate from per-dataset caption processing)
            </p>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!params.danbooru_aug_shuffle_tags}
                onChange={(e) => updateParam("danbooru_aug_shuffle_tags", e.target.checked)}
              />
              <span className="text-xs text-gray-300">Shuffle tags within each category (per-epoch)</span>
            </label>
            <div className="grid grid-cols-2 gap-3">
              <NumField label="Shuffle keep first N" value={params.danbooru_aug_shuffle_keep_first_n}
                onChange={(v) => updateParam("danbooru_aug_shuffle_keep_first_n", v)} step={1} />
              <NumField label="Keep tokens (token dropout)" value={params.danbooru_aug_keep_tokens}
                onChange={(v) => updateParam("danbooru_aug_keep_tokens", v)} step={1} />
              <NumField label="Tag dropout rate (0-1)" value={params.danbooru_aug_tag_dropout_rate}
                onChange={(v) => updateParam("danbooru_aug_tag_dropout_rate", v)} step={0.05} />
              <NumField label="Tag dropout keep first N" value={params.danbooru_aug_tag_dropout_keep_first_n}
                onChange={(v) => updateParam("danbooru_aug_tag_dropout_keep_first_n", v)} step={1} />
              <NumField label="Caption dropout rate (0-1)" value={params.danbooru_aug_caption_dropout_rate}
                onChange={(v) => updateParam("danbooru_aug_caption_dropout_rate", v)} step={0.05} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

