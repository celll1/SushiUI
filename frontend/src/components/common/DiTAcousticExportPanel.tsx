"use client";

import { useState } from "react";
import { exportDiTRewiring } from "@/utils/api";
import type { DiTRewiringSpec } from "@/utils/api";
import DiTRewiringEditor from "../training/DiTRewiringEditor";

export default function DiTAcousticExportPanel({ modelPath, onExported }: {
  modelPath: string; onExported: (path: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"inherit" | "configure">("configure");
  const [spec, setSpec] = useState<DiTRewiringSpec | null>(null);
  const [validated, setValidated] = useState<DiTRewiringSpec | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const save = async () => {
    if (!validated || !name.trim() || busy) return;
    setBusy(true); setError(null); setSaved(null);
    try {
      const result = await exportDiTRewiring(modelPath, name.trim(), validated);
      setSaved(result.model_path); onExported(result.model_path);
    } catch (err) {
      const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : err instanceof Error ? err.message : "Checkpoint export failed");
    } finally { setBusy(false); }
  };
  return <details className="mt-2 w-full rounded border border-gray-700 p-2 text-xs" onToggle={e => {
    setOpen(e.currentTarget.open);
    if (!e.currentTarget.open) setValidated(null);
  }}>
    <summary className="cursor-pointer text-sky-300">Configure acoustic U-Net checkpoint</summary>
    <p className="my-2 text-gray-400">Choose the layers, time scale and residual skips, then write a new checkpoint beside this file. Tensor values stay unchanged; uncalibrated quality may degrade. Supports complete YuE2 and flat MiniMax Music 3 safetensors.</p>
    <fieldset disabled={busy} className={busy ? "pointer-events-none opacity-60" : ""}>
      {open && <DiTRewiringEditor modelPath={modelPath} resolution={1} mode={mode} spec={spec}
        onValidated={setValidated} onChange={(nextMode, nextSpec) => {
          setMode(nextMode); setSpec(nextSpec); setValidated(null); setSaved(null); setError(null);
        }} />}
    </fieldset>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <label>New filename
        <input aria-label="U-Net checkpoint filename" value={name} disabled={busy}
          placeholder="New .safetensors filename" className="ml-2 rounded bg-gray-800 p-2"
          onChange={e => { setName(e.target.value); setSaved(null); }} />
      </label>
      <button type="button" disabled={busy || !validated || !name.trim()} onClick={save}
        className="rounded bg-sky-800 px-3 py-2 disabled:opacity-50">{busy ? "Exporting…" : "Export U-Net checkpoint"}</button>
    </div>
    {saved && <p className="mt-2 break-all text-emerald-300">Saved: {saved}. Select this file in the model list and load it to generate with the saved topology.</p>}
    {error && <p role="alert" className="mt-2 text-red-400">{error}</p>}
  </details>;
}
