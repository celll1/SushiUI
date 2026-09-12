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

const HISTORY_KEY = "tagger_model_loader_history";
const MAX_HISTORY = 5;
type DetectedModelType = "" | "full" | "lora" | "onnx";

interface LoaderHistory {
  checkpointPath: string;
  visionEncoderPath: string;
  vocabPath: string;
  modelType?: DetectedModelType;
  loraRank: number;
  loraAlpha: number;
}

function derivedVocabularyPath(checkpointPath: string): string {
  if (checkpointPath.toLowerCase().endsWith(".onnx")) {
    return checkpointPath.replace(/\.onnx$/i, "_vocabulary.json");
  }
  const directory = checkpointPath.replace(/[/\\][^/\\]*$/, "");
  return directory ? `${directory}/vocabulary.json` : "";
}

function loadHistory(): LoaderHistory[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entry: LoaderHistory) {
  const prev = loadHistory().filter((h) => h.checkpointPath !== entry.checkpointPath);
  const next = [entry, ...prev].slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
}

interface ModelLoaderProps {
  onStatusChange: (status: SigLIP2StatusResponse) => void;
}

export default function ModelLoader({ onStatusChange }: ModelLoaderProps) {
  const [detectedModelType,  setDetectedModelType]  = useState<DetectedModelType>("");
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
  const [history,            setHistory]            = useState<LoaderHistory[]>([]);
  const [showHistory,        setShowHistory]        = useState(false);

  const debounceRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const metadataRequestRef = useRef(0);
  const vocabIsAutomaticRef = useRef(true);
  const historyRef   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getSigLIP2Status()
      .then((s) => { setStatus(s); onStatusChange(s); })
      .catch((e: any) => setError(e?.message ?? "モデル状態を取得できませんでした"));

    const hist = loadHistory();
    setHistory(hist);
    if (hist.length > 0) {
      const last = hist[0];
      setCheckpointPath(last.checkpointPath);
      setVisionEncoderPath(last.visionEncoderPath);
      setVocabPath(last.vocabPath);
      setDetectedModelType(last.modelType ?? (last.checkpointPath.toLowerCase().endsWith(".onnx") ? "onnx" : ""));
      setLoraRank(last.loraRank);
      setLoraAlpha(last.loraAlpha);
      vocabIsAutomaticRef.current = last.vocabPath === derivedVocabularyPath(last.checkpointPath);
    }
    return () => {
      metadataRequestRef.current += 1;
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [onStatusChange]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const applyHistoryEntry = (entry: LoaderHistory) => {
    setCheckpointPath(entry.checkpointPath);
    setVisionEncoderPath(entry.visionEncoderPath);
    setVocabPath(entry.vocabPath);
    vocabIsAutomaticRef.current = entry.vocabPath === derivedVocabularyPath(entry.checkpointPath);
    setDetectedModelType(entry.modelType ?? (entry.checkpointPath.toLowerCase().endsWith(".onnx") ? "onnx" : ""));
    setLoraRank(entry.loraRank);
    setLoraAlpha(entry.loraAlpha);
    setShowHistory(false);
    setMetaStatus(null);
  };

  // Auto-complete vocab path when checkpoint path changes, and fetch metadata
  const handleCheckpointChange = (val: string) => {
    metadataRequestRef.current += 1;
    setCheckpointPath(val);
    setMetaStatus(null);
    setDetectedModelType(val.toLowerCase().endsWith(".onnx") ? "onnx" : "");

    if (vocabIsAutomaticRef.current) {
      setVocabPath(derivedVocabularyPath(val));
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (val.toLowerCase().endsWith(".onnx")) {
      return;
    }

    if (val.toLowerCase().endsWith(".safetensors")) {
      const requestId = metadataRequestRef.current;
      debounceRef.current = setTimeout(async () => {
        try {
          const meta = await getSigLIP2CheckpointMeta(val);
          if (requestId !== metadataRequestRef.current) return;
          if (meta.lora_rank !== undefined) {
            setLoraRank(meta.lora_rank);
          }
          if (meta.lora_alpha !== undefined) {
            setLoraAlpha(meta.lora_alpha);
          }
          setDetectedModelType(meta.training_method === "full" ? "full" : "lora");
          setMetaStatus("found");
        } catch {
          if (requestId !== metadataRequestRef.current) return;
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
      setDetectedModelType(s.model_type as DetectedModelType);
      const entry: LoaderHistory = {
        checkpointPath, visionEncoderPath, vocabPath,
        modelType: s.model_type as DetectedModelType, loraRank, loraAlpha,
      };
      saveHistory(entry);
      setHistory(loadHistory());
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "モデルを読み込めませんでした");
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async () => {
    setLoading(true);
    setError(null);
    try {
      await unloadSigLIP2Model();
      const s = await getSigLIP2Status();
      setStatus(s);
      onStatusChange(s);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "モデルを解放できませんでした");
    } finally {
      setLoading(false);
    }
  };

  const inputCls = "h-8 w-full rounded-md border border-gray-700 bg-gray-800 px-2 text-xs text-white focus:border-violet-500 focus:outline-none";
  const labelCls = "mb-1 block text-xs text-gray-400";

  return (
    <div className="space-y-3 p-3">
      <h3 className="text-sm font-semibold text-gray-200">モデル</h3>

      {status && (
        <div className={`text-sm px-2 py-1 rounded ${status.loaded ? "bg-green-900 text-green-300" : "bg-gray-800 text-gray-400"}`}>
          {status.loaded
            ? `読込済み · ${status.model_type} · ${status.num_tags.toLocaleString()}タグ`
            : "未読込"}
        </div>
      )}

      {status?.loaded && (
        <VocabularyBrowser useLoadedModel />
      )}

      <div className="text-xs text-gray-500">
        形式: {detectedModelType === "full" ? "Full" : detectedModelType === "lora" ? "LoRA" : detectedModelType === "onnx" ? "ONNX" : "読込時に自動判定"}
      </div>

      <div>
        <div className="flex items-center justify-between mb-1">
          <label className={labelCls.replace(" mb-1", "")}>チェックポイント</label>
          {history.length > 0 && (
            <div className="relative" ref={historyRef}>
              <button
                type="button"
                onClick={() => setShowHistory((v) => !v)}
                className="text-xs text-blue-400 hover:text-blue-300 px-1.5 py-0.5 rounded border border-gray-600 hover:border-gray-500 transition-colors"
              >
                履歴 ▾
              </button>
              {showHistory && (
                <div className="absolute right-0 top-full mt-1 z-50 w-max max-w-xs bg-gray-800 border border-gray-600 rounded shadow-lg overflow-hidden">
                  {history.map((h, i) => (
                    <button
                      key={i}
                      type="button"
                      onClick={() => applyHistoryEntry(h)}
                      className="w-full text-left px-3 py-2 text-xs text-gray-200 hover:bg-gray-700 border-b border-gray-700 last:border-0"
                    >
                      <div className="truncate max-w-xs font-mono">{h.checkpointPath.replace(/.*[/\\]/, "")}</div>
                      <div className="text-gray-500 truncate max-w-xs">{h.checkpointPath}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
        <input
          type="text"
          value={checkpointPath}
          onChange={(e) => handleCheckpointChange(e.target.value)}
          placeholder="<MODEL_ROOT>/latest.safetensors"
          className={inputCls}
        />
        {metaStatus === "found" && (
          <p className="text-xs text-green-400 mt-0.5">✓ メタデータから形式と設定を取得しました</p>
        )}
        {metaStatus === "not_found" && (
          <p className="text-xs text-gray-500 mt-0.5">メタデータなし — 読込時に重みから判定します</p>
        )}
      </div>

      {detectedModelType !== "full" && detectedModelType !== "onnx" && (
        <div>
          <label className={labelCls}>Vision Encoder <span className="text-gray-600">— LoRAのみ・省略可</span></label>
          <input
            type="text"
            value={visionEncoderPath}
            onChange={(e) => setVisionEncoderPath(e.target.value)}
            placeholder="メタデータから自動取得"
            className={inputCls}
          />
        </div>
      )}

      <div>
        <div className="mb-1 flex items-center justify-between">
          <label className={labelCls.replace(" mb-1", "")}>語彙 (vocabulary.json)</label>
          <button
            type="button"
            onClick={() => {
              vocabIsAutomaticRef.current = true;
              setVocabPath(derivedVocabularyPath(checkpointPath));
            }}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            自動設定に戻す
          </button>
        </div>
        <input
          type="text"
          value={vocabPath}
          onChange={(e) => {
            vocabIsAutomaticRef.current = false;
            setVocabPath(e.target.value);
          }}
          placeholder="チェックポイントと同じ場所から自動設定"
          className={inputCls}
        />
      </div>

      {detectedModelType === "lora" && (
        <div className="text-xs text-gray-500 bg-gray-800/50 rounded px-2 py-1.5">
          {metaStatus === "found" ? (
            <>LoRA rank <span className="text-gray-300 font-mono">{loraRank}</span> · alpha{" "}
              <span className="text-gray-300 font-mono">{loraAlpha}</span>{" "}
              <span className="text-gray-600">(メタデータ)</span></>
          ) : (
            <>rank / alpha は読込時に重みから自動判定します。</>
          )}
        </div>
      )}

      {error && (
        <div className="text-sm text-red-400 bg-red-900/30 rounded px-2 py-1 break-all">
          {error}
        </div>
      )}

      <div className="flex gap-2">
        <button
          onClick={handleLoad}
          disabled={loading || !checkpointPath}
          className="flex-1 rounded-md border border-violet-400/30 bg-violet-600 py-1.5 text-xs font-medium text-white transition-colors hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "処理中…" : "読み込む"}
        </button>
        {status?.loaded && (
          <button
            onClick={handleUnload}
            disabled={loading}
            className="px-3 py-1.5 rounded text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
          >
            解放
          </button>
        )}
      </div>
    </div>
  );
}
