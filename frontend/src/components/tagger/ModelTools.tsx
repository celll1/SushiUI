"use client";

import { useState } from "react";
import {
  mergeSigLIP2LoRA,
  exportSigLIP2ONNX,
  extractSigLIP2Encoder,
  SigLIP2ExtractEncoderResponse,
} from "@/utils/api";

interface ModelToolsProps {
  modelLoaded: boolean;
  modelType: string;
}

export default function ModelTools({ modelLoaded, modelType }: ModelToolsProps) {
  const [mergeOutput,    setMergeOutput]    = useState("");
  const [onnxOutput,     setOnnxOutput]     = useState("");
  const [maxPatches,     setMaxPatches]     = useState(256);
  const [stripUnknown,   setStripUnknown]   = useState(false);
  const [alsoSplit,      setAlsoSplit]      = useState(false);
  const [useModelStem,   setUseModelStem]   = useState(false);
  const [merging,        setMerging]        = useState(false);
  const [exporting,      setExporting]      = useState(false);
  const [mergeResult,    setMergeResult]    = useState<string | null>(null);
  const [onnxResult,     setOnnxResult]     = useState<{onnx: string; vocab: string} | null>(null);
  const [mergeError,     setMergeError]     = useState<string | null>(null);
  const [onnxError,      setOnnxError]      = useState<string | null>(null);

  const [extractRepoId,  setExtractRepoId]  = useState("google/siglip2-so400m-patch16-naflex");
  const [extractOutPath, setExtractOutPath] = useState("");
  const [extractType,    setExtractType]    = useState<"vision" | "text">("vision");
  const [extracting,     setExtracting]     = useState(false);
  const [extractResult,  setExtractResult]  = useState<SigLIP2ExtractEncoderResponse | null>(null);
  const [extractError,   setExtractError]   = useState<string | null>(null);

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
      const result = await exportSigLIP2ONNX(onnxOutput, maxPatches, stripUnknown, alsoSplit, useModelStem);
      setOnnxResult({ onnx: result.saved_path, vocab: result.vocab_path });
    } catch (e: any) {
      setOnnxError(e?.response?.data?.detail ?? e?.message ?? "Export failed");
    } finally {
      setExporting(false);
    }
  };

  const handleExtract = async () => {
    setExtracting(true);
    setExtractError(null);
    setExtractResult(null);
    try {
      const result = await extractSigLIP2Encoder(extractRepoId, extractOutPath, extractType);
      setExtractResult(result);
    } catch (e: any) {
      setExtractError(e?.response?.data?.detail ?? e?.message ?? "Extraction failed");
    } finally {
      setExtracting(false);
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
          <label className={labelCls}>Output Directory <span className="text-gray-600">— optional</span></label>
          <input
            type="text"
            value={mergeOutput}
            onChange={(e) => setMergeOutput(e.target.value)}
            placeholder="Auto: {checkpoint_dir}/merged/"
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
          disabled={!modelLoaded || modelType !== "lora" || merging}
          className="w-full py-1.5 rounded text-xs font-medium bg-purple-700 hover:bg-purple-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          {merging ? "Merging…" : "Merge & Save"}
        </button>
      </div>

      {/* ── Export ONNX ── */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-gray-300">Export ONNX</h4>
        {modelType === "onnx" && (
          <p className={disabledNote}>Already an ONNX model</p>
        )}
        <div>
          <label className={labelCls}>Output Path (.onnx) <span className="text-gray-600">— optional</span></label>
          <input
            type="text"
            value={onnxOutput}
            onChange={(e) => setOnnxOutput(e.target.value)}
            placeholder="Auto: {checkpoint_dir}/onnx/{name}.onnx"
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
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={stripUnknown}
            onChange={(e) => setStripUnknown(e.target.checked)}
            disabled={!modelLoaded}
            className="w-3.5 h-3.5 accent-teal-500"
          />
          <span className="text-xs text-gray-300">Strip Unknown-category tags from head</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={alsoSplit}
            onChange={(e) => setAlsoSplit(e.target.checked)}
            disabled={!modelLoaded}
            className="w-3.5 h-3.5 accent-teal-500"
          />
          <span className="text-xs text-gray-300">
            Also export WebGPU split version (sub-models under 2GB, in <code>_split_files/</code>)
          </span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={useModelStem}
            onChange={(e) => setUseModelStem(e.target.checked)}
            disabled={!modelLoaded}
            className="w-3.5 h-3.5 accent-teal-500"
          />
          <span className="text-xs text-gray-300">
            Name output <code>model.onnx</code> (+ <code>model_*.json</code>) instead of the checkpoint stem
          </span>
        </label>
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
          disabled={!modelLoaded || exporting || modelType === "onnx"}
          className="w-full py-1.5 rounded text-xs font-medium bg-teal-700 hover:bg-teal-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          {exporting ? "Exporting…" : "Export ONNX"}
        </button>
      </div>

      {/* ── Extract Encoder from HuggingFace ── */}
      <div className="space-y-2">
        <h4 className="text-xs font-medium text-gray-300">Extract Encoder from HuggingFace</h4>
        <div>
          <label className={labelCls}>HuggingFace Repo ID</label>
          <input
            type="text"
            value={extractRepoId}
            onChange={(e) => setExtractRepoId(e.target.value)}
            placeholder="google/siglip2-so400m-patch16-naflex"
            className={inputCls}
          />
        </div>
        <div>
          <label className={labelCls}>Output Path (.safetensors)</label>
          <input
            type="text"
            value={extractOutPath}
            onChange={(e) => setExtractOutPath(e.target.value)}
            placeholder="D:\...\siglip2_so400m_vision_encoder.safetensors"
            className={inputCls}
          />
        </div>
        <div>
          <label className={labelCls}>Encoder Type</label>
          <select
            value={extractType}
            onChange={(e) => setExtractType(e.target.value as "vision" | "text")}
            className={inputCls}
          >
            <option value="vision">Vision</option>
            <option value="text">Text</option>
          </select>
        </div>
        {extractError && (
          <p className="text-xs text-red-400 break-all">{extractError}</p>
        )}
        {extractResult && (
          <div className="text-xs text-green-400 space-y-0.5">
            <p className="break-all">Saved → {extractResult.output_path}</p>
            <p>{extractResult.num_params.toLocaleString()} params · hidden={extractResult.hidden_size} · layers={extractResult.num_layers}</p>
          </div>
        )}
        <button
          onClick={handleExtract}
          disabled={!extractRepoId || !extractOutPath || extracting}
          className="w-full py-1.5 rounded text-xs font-medium bg-indigo-700 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
        >
          {extracting ? "Extracting…" : "Extract & Save"}
        </button>
      </div>
    </div>
  );
}
