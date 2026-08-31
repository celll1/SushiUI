"use client";

import { useState, useEffect } from "react";
import { getGPUStats, GPUStats } from "@/utils/api";

interface GpuSelectProps {
  value: number | null;
  onChange: (value: number | null) => void;
  disabled?: boolean;
  label?: string;
}

export default function GpuSelect({ value, onChange, disabled, label = "GPU" }: GpuSelectProps) {
  const [gpus, setGpus] = useState<GPUStats[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    getGPUStats()
      .then((data) => {
        // cuda_index is what the backend accepts as gpu_index; entries without
        // one are not addressable by torch and cannot be selected.
        setGpus(data.available ? (data.gpus || []).filter((g) => g.cuda_index !== null) : []);
      })
      .catch(() => setGpus([]))
      .finally(() => setLoaded(true));
  }, []);

  const selectorDisabled = disabled || gpus.length === 0;
  // An already-pinned run must not silently read as "Auto" while it still sends
  // its index on save.
  const valueMissing = value !== null && !gpus.some((g) => g.cuda_index === value);

  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      <select
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        disabled={selectorDisabled && !valueMissing}
        className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-white disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <option value="">Auto (default)</option>
        {gpus.map((gpu) => (
          <option key={gpu.cuda_index} value={gpu.cuda_index as number}>
            {gpu.cuda_index}: {gpu.name} ({gpu.vram_total_gb.toFixed(1)} GB)
          </option>
        ))}
        {valueMissing && <option value={value as number}>{value}: not listed</option>}
      </select>
      <p className="text-xs text-gray-500 mt-1">
        Runs this training on the selected GPU. Auto uses the default device.
        {loaded && gpus.length === 1 && " Only one GPU was detected."}
        {loaded && gpus.length === 0 && " No selectable GPU was reported by the backend."}
        {valueMissing && " The saved GPU is not among those currently reported."}
      </p>
    </div>
  );
}
