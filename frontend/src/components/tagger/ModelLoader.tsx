"use client";

import { useState, useEffect, useRef } from "react";
import {
  loadSigLIP2Model,
  unloadSigLIP2Model,
  getSigLIP2Status,
  getSigLIP2CheckpointMeta,
  SigLIP2StatusResponse,
} from "@/utils/api";
import VocabularyBrowser from "./VocabularyBrowser";

interface ModelLoaderProps {
  onStatusChange: (status: SigLIP2StatusResponse) => void;
}

export default function ModelLoader({ onStatusChange }: ModelLoaderProps) {
  const [modelType,          setModelType]          = useState<"full" | "lora">("lora");
  const [checkpointPath,     setCheckpointPath]     = useState("");
  const [visionEncoderPath,  setVisionEncoderPath]  = useState("");
  const [vocabPath,          setVocabPath]          = useState("");
  const [loraRank,           setLoraRank]           = useState(32);
  const [loraAlpha,          setLoraAlpha]          = useState(16.0);
  const [loading,            setLoading]            = useState(false);
  const [error,              setError]              = useState<string | null>(null);
  const [status,             setStatus]             = useState<SigLIP2StatusResponse | null>(null);
  // null = not fetched, "found" = meta loaded, "not_found" = no meta
  const [metaStatus,         setMetaStatus]         = useState<"found" | "not_found" | null>(null);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch status on mount
  useEffect(() => {
    getSigLIP2Status()
      .then((s) => { setStatus(s); onStatusChange(s); })
      .catch(() => {});
  }, []);

  // Auto-complete vocab path when checkpoint path changes, and fetch metadata
  const handleCheckpointChange = (val: string) => {
    setCheckpointPath(val);
    setMetaStatus(null);

    if (!vocabPath) {
      const dir = val.replace(/[/\\][^/\\]*$/, "");
      if (dir) setVocabPath(dir + "/vocabulary.json");
    }

    // Debounce metadata fetch (500ms)
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.endsWith(".safetensors")) {
      debounceRef.current = setTimeout(async () => {
        try {
          const meta = await getSigLIP2CheckpointMeta(val);
          if (meta.lora_rank !== undefined) {
            setLoraRank(meta.lora_rank);
            setModelType("lora");
          }
          if (meta.lora_alpha !== undefined) {
            setLoraAlpha(meta.lora_alpha);
          }
          if (meta.training_method === "full") {
            setModelType("full");
          }
          setMetaStatus("found");
        } catch {
          setMetaStatus("not_found");
        }
      }, 500);
    }
  };

  const handleLoad = async () => {
    setLoading(true);
    setError(null);
    try {
      await loadSigLIP2Model({
        checkpoint_path:     checkpointPath,
        vision_encoder_path: visionEncoderPath,
        vocab_path:          vocabPath,
        lora_rank:           loraRank,
        lora_alpha:          loraAlpha,
      });
      const s = await getSigLIP2Status();
      setStatus(s);
      onStatusChange(s);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Load failed");
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async () => {
    setLoading(true);
    try {
      await unloadSigLIP2Model();
      const s = await getSigLIP2Status();
      setStatus(s);
      onStatusChange(s);
    } finally {
      setLoading(false);
    }
  };

  const inputCls = "w-full bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500";
  const labelCls = "block text-sm text-gray-400 mb-1";

  return (
    <div className="space-y-3 p-3">
      <h3 className="text-sm font-semibold text-gray-200">Model</h3>

      {/* Status badge */}
      {status && (
        <div className={`text-sm px-2 py-1 rounded ${status.loaded ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-400"}`}>
          {status.loaded
            ? `Loaded · ${status.model_type} · ${status.num_tags.toLocaleString()} tags`
            : "Not loaded"}
        </div>
      )}

      {/* Vocabulary browser (only when model is loaded) */}
      {status?.loaded && (
        <VocabularyBrowser useLoadedModel />
      )}

      {/* Model type */}
      <div>
        <label className={labelCls}>Model Type</label>
        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value as "full" | "lora")}
          className={inputCls}
        >
          <option value="lora">LoRA (compact)</option>
          <option value="full">Full model</option>
        </select>
      </div>

      {/* Checkpoint */}
      <div>
        <label className={labelCls}>Checkpoint Path (.safetensors)</label>
        <input
          type="text"
          value={checkpointPath}
          onChange={(e) => handleCheckpointChange(e.target.value)}
          placeholder="D:\tagger_models\...\latest.safetensors"
          className={inputCls}
        />
        {/* Metadata indicator */}
        {metaStatus === "found" && (
          <p className="text-xs text-green-400 mt-0.5">✓ Metadata loaded — rank/alpha auto-filled</p>
        )}
        {metaStatus === "not_found" && (
          <p className="text-xs text-gray-500 mt-0.5">No metadata file found alongside checkpoint</p>
        )}
      </div>

      {/* Vision encoder (always required) */}
      <div>
        <label className={labelCls}>Vision Encoder Path (.safetensors)</label>
        <input
          type="text"
          value={visionEncoderPath}
          onChange={(e) => setVisionEncoderPath(e.target.value)}
          placeholder="D:\...\siglip2_so400m_vision_encoder.safetensors"
          className={inputCls}
        />
      </div>

      {/* Vocabulary */}
      <div>
        <label className={labelCls}>Vocabulary Path (vocabulary.json)</label>
        <input
          type="text"
          value={vocabPath}
          onChange={(e) => setVocabPath(e.target.value)}
          placeholder="Auto-filled from checkpoint directory"
          className={inputCls}
        />
      </div>

      {/* LoRA params */}
      {modelType === "lora" && (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className={labelCls}>LoRA Rank</label>
            <input
              type="number"
              min={1}
              value={loraRank}
              onChange={(e) => setLoraRank(parseInt(e.target.value) || 32)}
              className={inputCls}
            />
          </div>
          <div>
            <label className={labelCls}>LoRA Alpha</label>
            <input
              type="number"
              min={0.1}
              step={0.5}
              value={loraAlpha}
              onChange={(e) => setLoraAlpha(parseFloat(e.target.value) || 16)}
              className={inputCls}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-sm text-red-400 bg-red-900/30 rounded px-2 py-1 break-all">
          {error}
        </div>
      )}

      {/* Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleLoad}
          disabled={loading || !checkpointPath}
          className="flex-1 py-1.5 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
        >
          {loading ? "Loading…" : "Load"}
        </button>
        {status?.loaded && (
          <button
            onClick={handleUnload}
            disabled={loading}
            className="px-3 py-1.5 rounded text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
          >
            Unload
          </button>
        )}
      </div>
    </div>
  );
}
