"use client";

import { useEffect, useState } from "react";
import MiniMaxH3TextEncoderSelector from "./MiniMaxH3TextEncoderSelector";
import MiniMaxH3HybridSelector, { HYBRID_CHECK_PENDING } from "./MiniMaxH3HybridSelector";
import type { MiniMaxH3HybridLoadRequest } from "@/utils/api";

// MiniMax-H3 load-time component choices. The state lives here rather than in
// ModelSelector because the controls render in the Components tab while the
// Load button that sends them renders in the Model tab.
export interface MiniMaxH3LoadOptions {
  // The base path every field below was restored for. The consumer compares it
  // with its own selection: this state reaches the host through an effect, so
  // it can describe the checkpoint that was selected one commit ago.
  keyedPath: string;
  // -> text_encoder_file / clip_projection_file on POST /models/load.
  textEncoderFile: string | null;
  clipProjectionFile: string | null;
  // -> overlay_file + the four hybrid_* fields; null = single-checkpoint load.
  hybrid: MiniMaxH3HybridLoadRequest | null;
  // Non-null only while an overlay is chosen that must not be sent. Says
  // nothing about the architecture -- only the consumer holds a fresh one.
  loadBlockedReason: string | null;
  setEncoderChoice: (textEncoder: string | null, clipProjection: string | null) => void;
  setHybrid: (next: MiniMaxH3HybridLoadRequest | null) => void;
  setHybridBlocked: (reason: string | null) => void;
}

export function useMiniMaxH3LoadOptions(modelPath: string): MiniMaxH3LoadOptions {
  const [textEncoderFile, setTextEncoderFile] = useState<string | null>(null);
  const [clipProjectionFile, setClipProjectionFile] = useState<string | null>(null);
  const [hybrid, setHybridState] = useState<MiniMaxH3HybridLoadRequest | null>(null);
  const [hybridBlocked, setHybridBlocked] = useState<string | null>(null);

  const encoderStorageKey = modelPath ? `minimax_h3_te_choice_${modelPath}` : "";
  const hybridStorageKey = modelPath ? `minimax_h3_hybrid_choice_${modelPath}` : "";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = encoderStorageKey ? localStorage.getItem(encoderStorageKey) : null;
    if (!saved) {
      setTextEncoderFile(null);
      setClipProjectionFile(null);
      return;
    }
    try {
      const parsed = JSON.parse(saved);
      setTextEncoderFile(typeof parsed?.text_encoder === "string" ? parsed.text_encoder : null);
      setClipProjectionFile(
        typeof parsed?.clip_projection === "string" ? parsed.clip_projection : null
      );
    } catch {
      setTextEncoderFile(null);
      setClipProjectionFile(null);
    }
  }, [encoderStorageKey]);

  // Written here rather than in an effect: an effect would fire once with the
  // previous model's choice already under the new model's key.
  const setEncoderChoice = (textEncoder: string | null, clipProjection: string | null) => {
    setTextEncoderFile(textEncoder);
    setClipProjectionFile(clipProjection);
    if (typeof window === "undefined" || !encoderStorageKey) return;
    if (!textEncoder && !clipProjection) {
      localStorage.removeItem(encoderStorageKey);
    } else {
      localStorage.setItem(
        encoderStorageKey,
        JSON.stringify({ text_encoder: textEncoder, clip_projection: clipProjection })
      );
    }
  };

  // A restored overlay starts BLOCKED: the selector has not answered for it
  // yet, and until it does the load button must not offer an unchecked pair.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = hybridStorageKey ? localStorage.getItem(hybridStorageKey) : null;
    let restored: MiniMaxH3HybridLoadRequest | null = null;
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (typeof parsed?.overlay_file === "string" && parsed.overlay_file) {
          restored = parsed as MiniMaxH3HybridLoadRequest;
        }
      } catch {
        restored = null;
      }
    }
    setHybridState(restored);
    setHybridBlocked(restored ? HYBRID_CHECK_PENDING : null);
  }, [hybridStorageKey]);

  const setHybrid = (next: MiniMaxH3HybridLoadRequest | null) => {
    setHybridState(next);
    if (typeof window === "undefined" || !hybridStorageKey) return;
    if (!next?.overlay_file) {
      localStorage.removeItem(hybridStorageKey);
    } else {
      localStorage.setItem(hybridStorageKey, JSON.stringify(next));
    }
  };

  // Only a chosen overlay can block: without one the request is the
  // single-checkpoint load, which no compatibility check applies to.
  const loadBlockedReason = hybrid?.overlay_file ? hybridBlocked : null;

  return {
    keyedPath: modelPath,
    textEncoderFile,
    clipProjectionFile,
    hybrid,
    loadBlockedReason,
    setEncoderChoice,
    setHybrid,
    setHybridBlocked,
  };
}

interface MiniMaxH3LoadOptionsGroupProps {
  // The BASE checkpoint selected in the Model tab's dropdown.
  modelPath: string;
  options: MiniMaxH3LoadOptions;
  // True while a load is in flight.
  disabled?: boolean;
}

// The load-time group: a third category next to the resident-component rows and
// the generation-only overrides, so what each control acts on stays readable.
export default function MiniMaxH3LoadOptionsGroup({
  modelPath,
  options,
  disabled = false,
}: MiniMaxH3LoadOptionsGroupProps) {
  return (
    <div className="space-y-2 rounded-md border border-gray-800 p-2">
      <p className="text-xs font-medium text-gray-400">Load-time components (MiniMax-H3)</p>
      <p className="text-[11px] text-gray-500">
        These are read when you press Load Model on the Model tab, which rebuilds the model. They do
        not change the resident components above.
      </p>
      <MiniMaxH3TextEncoderSelector
        modelPath={modelPath}
        textEncoderPath={options.textEncoderFile}
        clipProjectionPath={options.clipProjectionFile}
        onChange={options.setEncoderChoice}
        disabled={disabled}
      />
      {/* The Model tab's dropdown picks the base; this picks the second
          checkpoint merged into it. */}
      <MiniMaxH3HybridSelector
        modelPath={modelPath}
        value={options.hybrid}
        onChange={options.setHybrid}
        onBlockedChange={options.setHybridBlocked}
        disabled={disabled}
      />
    </div>
  );
}
