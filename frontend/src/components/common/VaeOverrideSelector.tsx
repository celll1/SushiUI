"use client";

import { useState, useEffect } from "react";
import Select from "./Select";
import { fetchVaes, VaeEntry } from "@/utils/api";

interface VaeOverrideSelectorProps {
  value: string | null;
  onChange: (path: string | null) => void;
  // Reports the "kind" of the currently-selected candidate ("autoencoder" |
  // "pid_decoder" | null for no override / unknown). Fires on mount (once the
  // candidate list has loaded) and whenever the selection changes, so callers
  // can react to a PiD-decoder override without duplicating the /models/vaes
  // fetch.
  onKindChange?: (kind: string | null) => void;
  label?: string;
  className?: string;
}

export default function VaeOverrideSelector({
  value,
  onChange,
  onKindChange,
  label = "VAE override",
  className = "",
}: VaeOverrideSelectorProps) {
  const [vaes, setVaes] = useState<VaeEntry[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchVaes()
      .then((data) => {
        setVaes(data.vaes || []);
      })
      .catch(() => {
        setVaes([]);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!onKindChange) return;
    if (!value) {
      onKindChange(null);
      return;
    }
    const selected = vaes.find((v) => v.path === value);
    onKindChange(selected?.kind ?? null);
  }, [value, vaes, onKindChange]);

  const options = [
    { value: "", label: "Default (model's VAE)" },
    ...vaes.map((v) => {
      const dims: string[] = [];
      if (v.latent_channels != null) dims.push(`${v.latent_channels}ch`);
      if (v.vae_class) dims.push(v.vae_class);
      const suffix = dims.length > 0 ? ` (${dims.join(", ")})` : "";
      const label = v.arch
        ? `${v.name} — ${v.arch}${suffix}`
        : `${v.name}${suffix}`;
      return { value: v.path, label };
    }),
  ];

  return (
    <div className={`space-y-1 ${className}`}>
      <Select
        label={label}
        value={value || ""}
        onChange={(e) => onChange(e.target.value || null)}
        options={options}
        disabled={loading}
      />
      {loading && (
        <p className="text-xs text-gray-500">Scanning model directory...</p>
      )}
      {!loading && vaes.length === 0 && (
        <p className="text-xs text-gray-500">
          No standalone VAEs found in model directory.
        </p>
      )}
    </div>
  );
}
