import { TrainingRunCreateRequest } from "@/utils/api";

type Props = {
  params: TrainingRunCreateRequest;
  defaults: Record<string, unknown> | null;
  onChange: (patch: Partial<TrainingRunCreateRequest>) => void;
};

export default function QwenTrainingObjective({ params, defaults, onChange }: Props) {
  const low = params.qwen_guidance_loss_weight ?? Number(defaults?.qwen_guidance_loss_weight ?? 0);
  const schedule = params.qwen_guidance_loss_weight_schedule ??
    (defaults?.qwen_guidance_loss_weight_schedule as "constant" | "high_noise_smoothstep" | undefined) ?? "constant";
  const high = params.qwen_guidance_loss_high_noise_weight ?? Number(defaults?.qwen_guidance_loss_high_noise_weight ?? 1);
  const objective = schedule === "constant" || low === high
    ? (low === 0 ? "mse" : low === 1 ? "contrastive" : "mixed")
    : "mixed";
  const mode = params.qwen_guidance_loss_mix_mode ??
    (defaults?.qwen_guidance_loss_mix_mode as "stochastic" | "blend" | undefined) ?? "stochastic";

  return (
    <div className="break-inside-avoid border border-amber-800/50 rounded p-4 space-y-3">
      <h3 className="text-sm font-medium text-amber-300">Training Objective · Qwen-Image 2.1</h3>
      <label className="block text-xs text-gray-300">Objective
        <select
          value={objective}
          onChange={(event) => {
            const choice = event.target.value;
            onChange(choice === "mse"
              ? { qwen_guidance_loss_weight: 0, qwen_guidance_loss_weight_schedule: "constant",
                  qwen_cfg_null_sigma_schedule: false }
              : choice === "contrastive"
                ? { qwen_guidance_loss_weight: 1, qwen_guidance_loss_weight_schedule: "constant",
                    qwen_cfg_null_sigma_schedule: false }
                : { qwen_guidance_loss_weight: 0.5, qwen_guidance_loss_weight_schedule: "constant",
                    qwen_guidance_loss_mix_mode: "stochastic",
                    qwen_cfg_null_sigma_schedule: (params.cfg_uncond_drop_rate ?? 0) > 0 });
          }}
          className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200"
        >
          <option value="mse">Ordinary MSE (0% guided)</option>
          <option value="contrastive">Guidance / contrastive (100%)</option>
          <option value="mixed">Custom mixture</option>
        </select>
      </label>
      <p className="text-xs text-gray-400">
        Qwen defaults to 100% guidance-target loss. MSE is available for ordinary training.
        Guided selections reuse a no-gradient empty-prompt prediction; the branch shares LoRA weights.
      </p>
      {objective !== "mse" && (
        <div className="grid grid-cols-2 gap-2">
          <label className="text-xs text-gray-400">Target CFG
            <input type="number" min={1} max={10} step={0.25}
              value={params.qwen_guidance_loss_scale ?? Number(defaults?.qwen_guidance_loss_scale ?? 3)}
              onChange={(event) => onChange({ qwen_guidance_loss_scale: Number(event.target.value) })}
              className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200" />
          </label>
          <label className="text-xs text-gray-400">Target CFG schedule
            <select value={params.qwen_guidance_loss_schedule ?? String(defaults?.qwen_guidance_loss_schedule ?? "sigma")}
              onChange={(event) => onChange({ qwen_guidance_loss_schedule: event.target.value as "constant" | "sigma" })}
              className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200">
              <option value="sigma">Sigma tapered</option>
              <option value="constant">Constant</option>
            </select>
          </label>
        </div>
      )}
      {objective === "mixed" && (
        <div className="space-y-2 border-t border-gray-700 pt-2">
          <label className="block text-xs text-gray-400">How to combine losses
            <select value={mode}
              onChange={(event) => onChange({
                qwen_guidance_loss_mix_mode: event.target.value as "stochastic" | "blend",
                ...(event.target.value === "blend" ? { qwen_cfg_null_sigma_schedule: false } : {}),
              })}
              className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200">
              <option value="stochastic">Choose one loss per image (default)</option>
              <option value="blend">Weighted blend (legacy)</option>
            </select>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-xs text-gray-400">Low-σ guided probability / weight
              <input type="number" min={0} max={1} step={0.05} value={low}
                onChange={(event) => onChange({ qwen_guidance_loss_weight: Number(event.target.value) })}
                className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200" />
            </label>
            <label className="text-xs text-gray-400">Probability / weight schedule
              <select value={schedule}
                onChange={(event) => onChange({ qwen_guidance_loss_weight_schedule: event.target.value as "constant" | "high_noise_smoothstep" })}
                className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200">
                <option value="constant">Constant</option>
                <option value="high_noise_smoothstep">High-noise ramp</option>
              </select>
            </label>
          </div>
          {schedule === "high_noise_smoothstep" && (
            <div className="grid grid-cols-3 gap-2">
              <label className="text-xs text-gray-400">High-σ weight
                <input type="number" min={low} max={1} step={0.05} value={high}
                  onChange={(event) => onChange({ qwen_guidance_loss_high_noise_weight: Number(event.target.value) })}
                  className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200" />
              </label>
              <label className="text-xs text-gray-400">Ramp starts at σ
                <input type="number" min={0} max={1} step={0.05}
                  value={params.qwen_guidance_loss_ramp_start ?? Number(defaults?.qwen_guidance_loss_ramp_start ?? 0.5)}
                  onChange={(event) => onChange({ qwen_guidance_loss_ramp_start: Number(event.target.value) })}
                  className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200" />
              </label>
              <label className="text-xs text-gray-400">Ramp ends at σ
                <input type="number" min={0} max={1} step={0.05}
                  value={params.qwen_guidance_loss_ramp_end ?? Number(defaults?.qwen_guidance_loss_ramp_end ?? 0.8)}
                  onChange={(event) => onChange({ qwen_guidance_loss_ramp_end: Number(event.target.value) })}
                  className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-xs text-gray-200" />
              </label>
            </div>
          )}
          {mode === "stochastic" && (
            <>
              <label className="flex items-center gap-2 text-xs text-gray-300">
                <input type="checkbox" checked={(params.cfg_uncond_drop_rate ?? 0) > 0 && !!params.qwen_cfg_null_sigma_schedule}
                  disabled={(params.cfg_uncond_drop_rate ?? 0) <= 0}
                  onChange={(event) => onChange({ qwen_cfg_null_sigma_schedule: event.target.checked })}
                  className="w-3.5 h-3.5" />
                Apply CFG-null drop only after ordinary MSE is selected
              </label>
              <p className="text-xs text-gray-500">
                Effective drop probability is configured drop rate × (1 − guided probability).
                The same per-image/timestep draw chooses MSE before testing its null-drop rate.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
