"use client";

import { useState, useEffect } from "react";
import { getGPUStats, GPUStats } from "@/utils/api";

export default function GPUMonitor() {
  const [gpuStats, setGpuStats] = useState<GPUStats[]>([]);
  const [available, setAvailable] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getGPUStats();
        setAvailable(data.available);
        setGpuStats(data.gpus || []);
        setError(data.error || null);
      } catch (err) {
        console.error("Error fetching GPU stats:", err);
        setError("Failed to fetch GPU stats");
      }
    };

    // Fetch immediately
    fetchStats();

    // Update every 2 seconds
    const interval = setInterval(fetchStats, 2000);

    return () => clearInterval(interval);
  }, []);

  if (!available || error) {
    return null; // Don't show anything if GPU monitoring is not available
  }

  if (gpuStats.length === 0) {
    return null;
  }

  // If only one GPU and not expanded, show compact view
  if (gpuStats.length === 1 && !isExpanded) {
    const gpu = gpuStats[0];
    return (
      <div
        className="fixed bottom-4 right-4 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-xs cursor-pointer hover:bg-gray-750 transition-colors"
        onClick={() => setIsExpanded(true)}
        title="Click to expand"
      >
        <div className="flex items-center gap-3">
          {/* VRAM */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400">VRAM</span>
            <div className="flex items-center gap-1">
              <div
                className="h-2 w-16 bg-gray-700 rounded-full overflow-hidden"
                title={`VRAM: ${gpu.vram_used_gb}GB / ${gpu.vram_total_gb}GB`}
              >
                <div
                  className={`h-full transition-all ${
                    gpu.vram_percent > 90
                      ? "bg-red-500"
                      : gpu.vram_percent > 70
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                  style={{ width: `${gpu.vram_percent}%` }}
                />
              </div>
              <span className="text-gray-300 min-w-[3.5rem]">
                {gpu.vram_used_gb}/{gpu.vram_total_gb}GB
              </span>
            </div>
          </div>

          {/* GPU Utilization */}
          <div className="flex items-center gap-1">
            <span className="text-gray-400">GPU</span>
            <span className="text-gray-300 min-w-[2rem]">
              {gpu.gpu_utilization !== null ? `${gpu.gpu_utilization}%` : "N/A"}
            </span>
          </div>

          {/* Temperature */}
          <div className="flex items-center gap-1">
            <span className="text-gray-400">🌡️</span>
            <span
              className={`min-w-[2.5rem] ${
                gpu.temperature !== null
                  ? gpu.temperature > 80
                    ? "text-red-400"
                    : gpu.temperature > 70
                    ? "text-yellow-400"
                    : "text-green-400"
                  : "text-gray-500"
              }`}
            >
              {gpu.temperature !== null ? `${gpu.temperature}°C` : "N/A"}
            </span>
          </div>
        </div>
      </div>
    );
  }

  // Expanded view or multiple GPUs
  return (
    <div className="fixed bottom-4 right-4 bg-gray-800 border border-gray-700 rounded p-3 text-xs min-w-[300px]">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-gray-300 font-semibold">GPU Monitor</h3>
        {gpuStats.length === 1 && (
          <button
            onClick={() => setIsExpanded(false)}
            className="text-gray-400 hover:text-gray-200 text-xs"
          >
            ✕
          </button>
        )}
      </div>

      <div className="space-y-3">
        {gpuStats.map((gpu) => (
          <div key={gpu.index} className="space-y-1">
            <div className="text-gray-400 text-[10px]">
              GPU {gpu.index}: {gpu.name}
            </div>

            {/* VRAM */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">VRAM</span>
                <span className="text-gray-300">
                  {gpu.vram_used_gb}GB / {gpu.vram_total_gb}GB ({gpu.vram_percent}%)
                </span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    gpu.vram_percent > 90
                      ? "bg-red-500"
                      : gpu.vram_percent > 70
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                  style={{ width: `${gpu.vram_percent}%` }}
                />
              </div>
            </div>

            {/* GPU Utilization */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">Utilization</span>
                <span className="text-gray-300">
                  {gpu.gpu_utilization !== null ? `${gpu.gpu_utilization}%` : "N/A"}
                </span>
              </div>
              {gpu.gpu_utilization !== null && (
                <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 transition-all"
                    style={{ width: `${gpu.gpu_utilization}%` }}
                  />
                </div>
              )}
            </div>

            {/* Temperature & Power */}
            <div className="flex items-center justify-between text-[10px]">
              <div className="flex items-center gap-1">
                <span className="text-gray-400">Temp:</span>
                <span
                  className={`${
                    gpu.temperature !== null
                      ? gpu.temperature > 80
                        ? "text-red-400"
                        : gpu.temperature > 70
                        ? "text-yellow-400"
                        : "text-green-400"
                      : "text-gray-500"
                  }`}
                >
                  {gpu.temperature !== null ? `${gpu.temperature}°C` : "N/A"}
                </span>
              </div>
              <div className="flex items-center gap-1">
                <span className="text-gray-400">Power:</span>
                <span className="text-gray-300">
                  {gpu.power_watts !== null ? `${gpu.power_watts}W` : "N/A"}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
