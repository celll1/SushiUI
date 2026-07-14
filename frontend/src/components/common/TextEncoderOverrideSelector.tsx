"use client";

import { useState, useEffect } from "react";
import Select from "./Select";
import { fetchTextEncoders, TextEncoderEntry } from "@/utils/api";

interface TextEncoderOverrideSelectorProps {
  value: string | null;
  onChange: (path: string | null) => void;
  label?: string;
  className?: string;
  disabled?: boolean;
  // When true, hide candidates whose arch doesn't match the loaded model's
  // arch (see isTeCompatible below). Defaults to false (show everything) so
  // existing callers that don't pass loaded-model info are unaffected.
  compatibleOnly?: boolean;
  // Loaded model's arch (modelInfo.type), used only when compatibleOnly is true.
  loadedArch?: string | null;
}

// A candidate is compatible when its arch matches the loaded model's arch
// family. If either arch is unknown, don't hide the candidate (avoid false
// negatives) — the existing `disabled` prop already restricts TE overrides
// to sd15/sdxl archs server-side, this only narrows the visible list.
export function isTeCompatible(
  te: TextEncoderEntry,
  loadedArch: string | null | undefined
): boolean {
  if (!loadedArch || !te.arch) return true;
  return te.arch.toLowerCase() === loadedArch.toLowerCase();
}

export default function TextEncoderOverrideSelector({
  value,
  onChange,
  label = "Text encoder override",
  className = "",
  disabled = false,
  compatibleOnly = false,
  loadedArch = null,
}: TextEncoderOverrideSelectorProps) {
  const [encoders, setEncoders] = useState<TextEncoderEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchTextEncoders()
      .then((data) => {
        setEncoders(data.text_encoders || []);
      })
      .catch(() => {
        setEncoders([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const visibleEncoders = compatibleOnly
    ? encoders.filter((te) => isTeCompatible(te, loadedArch))
    : encoders;

  const options = [
    { value: "", label: "Default (model's text encoder)" },
    ...visibleEncoders.map((te) => {
      const dims: string[] = [];
      if (te.out_dim != null) dims.push(`${te.out_dim}d`);
      if (te.te_type) dims.push(te.te_type);
      const suffix = dims.length > 0 ? ` (${dims.join(", ")})` : "";
      const label = te.arch
        ? `${te.name} — ${te.arch}${suffix}`
        : `${te.name}${suffix}`;
      return { value: te.path, label };
    }),
  ];

  return (
    <div className={`space-y-1 ${className}`}>
      <Select
        label={label}
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        options={options}
        disabled={loading || disabled}
      />
      {loading && (
        <p className="text-xs text-gray-500">Scanning model directory...</p>
      )}
      {!loading && encoders.length === 0 && (
        <p className="text-xs text-gray-500">
          No standalone text encoders found in model directory.
        </p>
      )}
      {!loading && encoders.length > 0 && visibleEncoders.length === 0 && (
        <p className="text-xs text-gray-500">
          No compatible text encoder found — untick &quot;Show only compatible with loaded model&quot; to show all.
        </p>
      )}
    </div>
  );
}
