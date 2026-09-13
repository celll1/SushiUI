"use client";

import Card from "../common/Card";
import Input from "../common/Input";
import NumberInput from "../common/NumberInput";
import Select from "../common/Select";
import Textarea from "../common/Textarea";
import { GenerationParams, Txt2AudParams } from "@/utils/api";

type Props = {
  params: GenerationParams;
  defaults: Partial<Txt2AudParams>;
  onChange: (params: GenerationParams) => void;
};

export default function Yue2AudioControls({ params, defaults, onChange }: Props) {
  const numeric = [
    ["audio_duration", "Duration upper bound (seconds)", 0.04, undefined, "float"],
    ["yue2_abc_max_tokens", "ABC token budget", 1, 24576, "int"],
    ["temperature", "Semantic temperature", 0, 5, "float"],
    ["top_p", "Semantic top-p", 0.01, 1, "float"],
    ["top_k", "Semantic top-k (0 disables)", 0, 32768, "int"],
    ["repetition_penalty", "Repetition penalty", 0.01, undefined, "float"],
    ["vae_tile_frames", "VAE tile core frames", 1, undefined, "int"],
  ] as const;
  return (
    <Card title="Audio Settings (YuE2)">
      <p className="text-xs text-gray-400 mb-3">
        Duration is an upper bound; the song can end earlier and prompt length limits available context.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Select label="Symbolic planning" value={params.yue2_cot ?? defaults.yue2_cot}
          onChange={(e) => onChange({ ...params, yue2_cot: e.target.value as Txt2AudParams["yue2_cot"] })}
          options={[{ value: "full", label: "Melody + chords" }, { value: "melody", label: "Melody" }, { value: "off", label: "Off" }]} />
        <Select label="VAE decode" value={params.vae_decode_mode ?? defaults.vae_decode_mode}
          onChange={(e) => onChange({ ...params, vae_decode_mode: e.target.value as Txt2AudParams["vae_decode_mode"] })}
          options={[{ value: "tiled", label: "Tiled" }, { value: "full", label: "Full" }]} />
        {numeric.map(([key, label, min, max, parse]) => (
          <label key={key} className="block text-xs font-medium text-gray-400">
            {label}
            <NumberInput label={label} value={(params[key] ?? defaults[key]) as number}
              onCommit={(value) => onChange({ ...params, [key]: value })}
              min={min} max={max} parse={parse} step={parse === "int" ? 1 : "any"} className="mt-1 w-full" />
          </label>
        ))}
        <Input type="number" label="Semantic guidance (blank = automatic)" step="any" min={0.01} max={20}
          value={params.yue2_guidance_scale ?? ""}
          onChange={(e) => onChange({ ...params, yue2_guidance_scale: e.target.value === "" ? undefined : Number(e.target.value) })} />
        <Input type="number" label="Seed" value={params.seed}
          onChange={(e) => { if (e.target.value !== "") onChange({ ...params, seed: Number(e.target.value) }); }} />
      </div>
      <Textarea label="ABC score (optional)" rows={5} value={params.yue2_abc ?? defaults.yue2_abc ?? ""}
        onChange={(e) => onChange({ ...params, yue2_abc: e.target.value })}
        placeholder="Leave blank to generate a plan. Supplied ABC requires planning to be enabled." />
      <p className="text-xs text-gray-500 mt-2">Acoustic synthesis uses the fixed official midpoint recipe. Model weights are CC BY-NC 4.0.</p>
    </Card>
  );
}
