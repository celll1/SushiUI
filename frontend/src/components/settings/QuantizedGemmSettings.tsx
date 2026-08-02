"use client";

import { useCallback, useEffect, useState } from "react";
import Button from "@/components/common/Button";
import {
  getFp8ScaledMm,
  setFp8ScaledMm,
  getInt8Mm,
  setInt8Mm,
  type Fp8ScaledMmState,
  type Int8MmState,
} from "@/utils/api";

type ResolvedModes = Record<string, string | null>;

interface ModeState {
  enabled: boolean;
  origin: "default" | "env" | "api";
  resolved_modes: ResolvedModes;
}

const ORIGIN_TEXT: Record<ModeState["origin"], string> = {
  default: "process default (no environment variable at import)",
  env: "environment variable at import",
  api: "set through this API in the current process",
};

function extractDetail(error: any, fallback: string): string {
  const status = error?.response?.status;
  const detail = error?.response?.data?.detail;
  if (status === 409) {
    return (
      detail ||
      "Cannot change this while a generation or training run is active."
    );
  }
  return detail || error?.message || fallback;
}

interface ModeRowProps {
  title: string;
  mechanism: string;
  state: ModeState | null;
  loadError: string | null;
  busy: boolean;
  probeKeyLabel: string;
  onToggle: (enabled: boolean) => void;
}

function ModeRow({
  title,
  mechanism,
  state,
  loadError,
  busy,
  probeKeyLabel,
  onToggle,
}: ModeRowProps) {
  const entries = state ? Object.entries(state.resolved_modes) : [];
  const allFellBack =
    entries.length > 0 && entries.every(([, mode]) => mode === null);

  return (
    <div className="p-4 bg-gray-800 rounded-lg space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-white">{title}</h3>
          <p className="text-xs text-gray-400 mt-1 font-mono">{mechanism}</p>
        </div>
        <label className="flex items-center gap-2 shrink-0 cursor-pointer">
          <input
            type="checkbox"
            checked={state?.enabled ?? false}
            disabled={!state || busy}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span className="text-sm text-gray-300">
            {state ? (state.enabled ? "On" : "Off") : "Unknown"}
          </span>
        </label>
      </div>

      {loadError && (
        <p className="text-xs text-red-400">
          State could not be read from the backend: {loadError}
        </p>
      )}

      {state && (
        <div className="space-y-2 text-xs text-gray-400">
          <div>
            <span className="text-gray-500">Origin:</span>{" "}
            {ORIGIN_TEXT[state.origin] ?? state.origin}
          </div>

          <div>
            <div className="text-gray-500 mb-1">
              Resolved path per {probeKeyLabel}:
            </div>
            {entries.length === 0 ? (
              <div className="pl-2">
                No probe result in this process yet. The probe runs on the first
                forward pass through a quantized Linear layer; toggling this
                setting clears the cached probe result.
              </div>
            ) : (
              <ul className="pl-2 space-y-0.5 font-mono">
                {entries.map(([key, mode]) => (
                  <li key={key}>
                    {key} &rarr;{" "}
                    {mode === null ? (
                      <span className="text-yellow-400">
                        none (dequantized matmul)
                      </span>
                    ) : (
                      <span className="text-white">{mode}</span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {state.enabled && allFellBack && (
            <p className="text-yellow-400">
              This mode is on, but the probe accepted no variant on this
              hardware/build, so those layers run the dequantized matmul.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function QuantizedGemmSettings() {
  const [fp8, setFp8] = useState<Fp8ScaledMmState | null>(null);
  const [int8, setInt8] = useState<Int8MmState | null>(null);
  const [fp8Error, setFp8Error] = useState<string | null>(null);
  const [int8Error, setInt8Error] = useState<string | null>(null);
  const [message, setMessage] = useState<
    { type: "success" | "error"; text: string } | null
  >(null);
  const [busy, setBusy] = useState(false);

  const loadStates = useCallback(async () => {
    try {
      setFp8(await getFp8ScaledMm());
      setFp8Error(null);
    } catch (error: any) {
      setFp8(null);
      setFp8Error(extractDetail(error, "Failed to read the FP8 GEMM state"));
    }
    try {
      setInt8(await getInt8Mm());
      setInt8Error(null);
    } catch (error: any) {
      setInt8(null);
      setInt8Error(extractDetail(error, "Failed to read the INT8 GEMM state"));
    }
  }, []);

  useEffect(() => {
    loadStates();
  }, [loadStates]);

  const handleToggleFp8 = async (enabled: boolean) => {
    setBusy(true);
    setMessage(null);
    try {
      setFp8(await setFp8ScaledMm(enabled));
      setFp8Error(null);
      setMessage({
        type: "success",
        text: `FP8 W8A8 GEMM path turned ${enabled ? "on" : "off"} for this backend process.`,
      });
    } catch (error: any) {
      setMessage({
        type: "error",
        text: extractDetail(error, "Failed to change the FP8 GEMM path"),
      });
      await loadStates();
    } finally {
      setBusy(false);
    }
  };

  const handleToggleInt8 = async (enabled: boolean) => {
    setBusy(true);
    setMessage(null);
    try {
      setInt8(await setInt8Mm(enabled));
      setInt8Error(null);
      setMessage({
        type: "success",
        text: `INT8 W8A8 GEMM path turned ${enabled ? "on" : "off"} for this backend process.`,
      });
    } catch (error: any) {
      setMessage({
        type: "error",
        text: extractDetail(error, "Failed to change the INT8 GEMM path"),
      });
      await loadStates();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-gray-400 text-sm">
        Select how quantized Linear layers compute their matrix product: W8A8
        (the activation is quantized and the product is computed in the
        quantized type) or dequantize-then-matmul. The two paths are
        numerically different. Both settings apply to the whole backend process,
        are independent of each other, are not stored with a generation, and
        reset to their environment value when the backend restarts.
      </p>

      {message && (
        <div
          className={`p-3 rounded text-sm ${
            message.type === "success"
              ? "bg-green-900/30 text-green-400"
              : "bg-red-900/30 text-red-400"
          }`}
        >
          {message.text}
        </div>
      )}

      <ModeRow
        title="FP8 W8A8 GEMM"
        mechanism="torch._scaled_mm"
        state={fp8}
        loadError={fp8Error}
        busy={busy}
        probeKeyLabel="device / activation dtype"
        onToggle={handleToggleFp8}
      />

      <ModeRow
        title="INT8 W8A8 GEMM"
        mechanism="torch._int_mm"
        state={int8}
        loadError={int8Error}
        busy={busy}
        probeKeyLabel="device"
        onToggle={handleToggleInt8}
      />

      <div className="flex gap-3">
        <Button onClick={loadStates} variant="secondary" disabled={busy}>
          Refresh
        </Button>
      </div>

      <div className="p-4 bg-gray-800 rounded-lg">
        <h3 className="text-sm font-semibold mb-2">Notes:</h3>
        <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
          <li>
            These settings only affect checkpoints that carry quantized weights
            (Ideogram 4 / Krea 2 FP8 checkpoints, and the INT8 checkpoint
            format). A model whose weights are not quantized, such as SDXL, has
            no quantized Linear layers, so neither setting changes anything for
            it.
          </li>
          <li>
            The path each image actually ran is recorded per generation and
            shown as <span className="font-mono">fp8_gemm</span> in the gallery
            image details.
          </li>
          <li>
            The backend refuses a change while an image generation or a training
            run is active, because the two paths are numerically different and a
            mid-run change would make the recorded metadata describe a path that
            produced only part of the result.
          </li>
        </ul>
      </div>
    </div>
  );
}
