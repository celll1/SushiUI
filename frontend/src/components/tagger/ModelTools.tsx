"use client";

import { useState } from "react";
import { mergeSigLIP2LoRA, exportSigLIP2ONNX } from "@/utils/api";

interface ModelToolsProps {
  modelLoaded: boolean;
  modelType: string;
}

export default function ModelTools({ modelLoaded, modelType }: ModelToolsProps) {
  const [mergeOutput,    setMergeOutput]    = useState("");
  const [onnxOutput,     setOnnxOutput]     = useState("");
  const [maxPatches,     setMaxPatches]     = useState(256);
  const [merging,        setMerging]        = useState(false);
  const [exporting,      setExporting]      = useState(false);
  const [mergeResult,    setMergeResult]    = useState<string | null>(null);
  const [onnxResult,     setOnnxResult]     = useState<{onnx: string; vocab: string} | null>(null);
  const [mergeError,     setMergeError]     = useState<string | null>(null);
  const [onnxError,      setOnnxError]      = useState<string | null>(null);

  const inputCls = "w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-xs text-white focus:outline-none focus:border-blue-500";
  const labelCls = "block text-xs text-gray-400 mb-1";
  const disabledNote = "text-xs text-gray-600 italic";

  const handleMerge = async () => {
    setMerging(true);
    setMergeError(null);
    setMergeResult(null);
    try {
      const result = await mergeSigLIP2LoRA(mergeOutput);
      setMergeResult(result.saved_path);
    } catch (e: any) {
      setMergeError(e?.response?.data?.detail ?? e?.message ?? "Merge failed");
    } finally {
      setMerging(false);
    }
  };

  const handleExportONNX = async () => {
    setExporting(true);
    setOnnxError(null);
    setOnnxResult(null);
    try {
      const result = await exportSigLIP2ONNX(onnxOutput, maxPatches);
      setOnnxResult({ onnx: result.saved_path, vocab: result.vocab_path });
    } catch (e: any) {
      setOnnxError(e?.response?.data?.detail ?? e?.message ?? "Export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="space-y-4 p-3 border-t border-gray-700">
      <h3 className="text-sm font-semibold text-gray-200">Model Tools</h3>

      {/* ── Merge LoRA ── */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-gray-300">Merge LoRA → Full Model</h4>
        {modelType !== "lora" && (
          <p className={disabledNote}>Only available for LoRA models</p>
        )}
        <div>
          <label className={labelCls}>Output Path (.safetensors)</label>
          <input
            type="text"
            value={mergeOutput}
            onChange={(e) => setMergeOutput(e.target.value)}
            placeholder="D:\...\merged_model.safetensors"
            disabled={!modelLoaded || modelType !== "lora"}
            className={inputCls}
          />
        </div>
        {mergeError && (
          <p className="text-xs text-red-400 break-all">{mergeError}</p>
        )}
        {mergeResult && (
          <p className="text-xs text-green-400 break-all">Saved → {mergeResult}</p>
        )}
        <button
          onClick={handleMerge}
          disabled={!modelLoaded || modelType !== "lora" || !mergeOutput || merging}
          className="w-full py-1.5 rounded text-xs font-medium bg-purple-700 hover:bg-purple-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          {merging ? "Merging…" : "Merge & Save"}
        </button>
      </div>

      {/* ── Export ONNX ── */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-gray-300">Export ONNX</h4>
        <div>
          <label className={labelCls}>Output Path (.onnx)</label>
          <input
            type="text"
            value={onnxOutput}
            onChange={(e) => setOnnxOutput(e.target.value)}
            placeholder="D:\...\model.onnx"
            disabled={!modelLoaded}
            className={inputCls}
          />
        </div>
        <div>
          <label className={labelCls}>Max Num Patches</label>
          <input
            type="number"
            min={64}
            max={1024}
            step={64}
            value={maxPatches}
            onChange={(e) => setMaxPatches(parseInt(e.target.value) || 256)}
            disabled={!modelLoaded}
            className={inputCls}
          />
        </div>
        {onnxError && (
          <p className="text-xs text-red-400 break-all">{onnxError}</p>
        )}
        {onnxResult && (
          <div className="text-xs text-green-400 space-y-0.5">
            <p className="break-all">ONNX → {onnxResult.onnx}</p>
            <p className="break-all">Vocab → {onnxResult.vocab}</p>
          </div>
        )}
        <button
          onClick={handleExportONNX}
          disabled={!modelLoaded || !onnxOutput || exporting}
          className="w-full py-1.5 rounded text-xs font-medium bg-teal-700 hover:bg-teal-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          {exporting ? "Exporting…" : "Export ONNX"}
        </button>
      </div>
    </div>
  );
}
