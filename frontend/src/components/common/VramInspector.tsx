"use client";

import { useState } from "react";
import { debugVramInspection, debugVramForceRelease } from "@/utils/api";
import Button from "./Button";

interface VramTensor {
  shape: number[];
  dtype: string;
  count: number;
  total_mb: number;
  referrers: string[];
}

interface VramData {
  memory: {
    allocated_mb: number;
    reserved_mb: number;
    max_allocated_mb: number;
    max_reserved_mb: number;
    total_tensor_mb: number;
  };
  tensor_count: number;
  unique_shapes: number;
  tensors: VramTensor[];
  components: Record<string, string>;
}

export default function VramInspector() {
  const [data, setData] = useState<VramData | null>(null);
  const [loading, setLoading] = useState(false);
  const [releaseResult, setReleaseResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inspect = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await debugVramInspection();
      setData(result);
    } catch (e: any) {
      setError(e.message || "Failed to inspect VRAM");
    } finally {
      setLoading(false);
    }
  };

  const forceRelease = async () => {
    setReleaseResult(null);
    try {
      const result = await debugVramForceRelease();
      setReleaseResult(`Freed ${result.freed_mb} MB (${result.before.reserved_mb} -> ${result.after.reserved_mb} MB reserved)`);
      // Auto-refresh inspection
      await inspect();
    } catch (e: any) {
      setError(e.message || "Failed to release VRAM");
    }
  };

  return (
    <div className="mt-3 p-3 bg-gray-800 rounded-lg">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-medium text-gray-300">VRAM Inspector</span>
        <Button onClick={inspect} disabled={loading} variant="secondary" size="sm">
          {loading ? "Scanning..." : "Inspect VRAM"}
        </Button>
        <Button onClick={forceRelease} disabled={loading} variant="secondary" size="sm">
          Force Release
        </Button>
      </div>
      {releaseResult && <div className="text-green-400 text-xs mb-2">{releaseResult}</div>}

      {error && <div className="text-red-400 text-xs">{error}</div>}

      {data && (
        <div className="space-y-3 text-xs font-mono">
          {/* Memory Summary */}
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <div className="text-gray-400">Allocated:</div>
            <div className="text-yellow-300">{data.memory.allocated_mb} MB</div>
            <div className="text-gray-400">Reserved:</div>
            <div className="text-yellow-300">{data.memory.reserved_mb} MB</div>
            <div className="text-gray-400">Max Allocated:</div>
            <div className="text-gray-500">{data.memory.max_allocated_mb} MB</div>
            <div className="text-gray-400">Max Reserved:</div>
            <div className="text-gray-500">{data.memory.max_reserved_mb} MB</div>
            <div className="text-gray-400">GC Tensor Total:</div>
            <div className="text-yellow-300">{data.memory.total_tensor_mb} MB</div>
            <div className="text-gray-400">Tensor Count:</div>
            <div className="text-gray-300">{data.tensor_count} ({data.unique_shapes} unique)</div>
          </div>

          {/* Component Devices */}
          {Object.keys(data.components).length > 0 && (
            <div>
              <div className="text-gray-400 mb-1 font-semibold">Components:</div>
              <div className="space-y-0.5">
                {Object.entries(data.components).map(([name, device]) => (
                  <div key={name} className="flex gap-2">
                    <span className="text-gray-500 truncate" style={{ maxWidth: "200px" }}>{name}:</span>
                    <span className={device.includes("cuda") ? "text-red-400" : "text-green-400"}>
                      {device}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tensor List */}
          {data.tensors.length > 0 && (
            <div>
              <div className="text-gray-400 mb-1 font-semibold">GPU Tensors (by size):</div>
              <div className="max-h-64 overflow-y-auto space-y-1">
                {data.tensors.map((t, i) => (
                  <div key={i} className="p-1.5 bg-gray-900 rounded">
                    <div className="flex justify-between">
                      <span className="text-blue-300">[{t.shape.join(", ")}]</span>
                      <span className="text-yellow-300">{t.total_mb} MB</span>
                    </div>
                    <div className="flex justify-between text-gray-500">
                      <span>{t.dtype} x{t.count}</span>
                      <span className="truncate ml-2" style={{ maxWidth: "200px" }}>
                        {t.referrers.length > 0 ? t.referrers.join(", ") : "?"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {data.tensors.length === 0 && (
            <div className="text-green-400">No CUDA tensors found in GC</div>
          )}
        </div>
      )}
    </div>
  );
}
