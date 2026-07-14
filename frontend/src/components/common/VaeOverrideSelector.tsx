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
  // When true, hide candidates that don't match the loaded model (see
  // isVaeCompatible below). Defaults to false (show everything) so existing
  // callers that don't pass loaded-model info are unaffected.
  compatibleOnly?: boolean;
  // Loaded model's arch (modelInfo.type) and latent channel count, used only
  // when compatibleOnly is true.
  loadedArch?: string | null;
  loadedLatentChannels?: number | null;
}

// A candidate is compatible when its latent_channels matches the loaded
// model's latent_channels (the most reliable signal — a VAE with a different
// channel count cannot decode the model's latents at all). If either side's
// latent_channels is unknown, fall back to an arch-name match. If both
// signals are unknown, don't hide the candidate (avoid false negatives).
export function isVaeCompatible(
  v: VaeEntry,
  loadedLatentChannels: number | null | undefined,
  loadedArch: string | null | undefined
): boolean {
  if (loadedLatentChannels != null && v.latent_channels != null) {
    return v.latent_channels === loadedLatentChannels;
  }
  if (loadedArch && v.arch) {
    return v.arch.toLowerCase() === loadedArch.toLowerCase();
  }
  return true;
}

export default function VaeOverrideSelector({
  value,
  onChange,
  onKindChange,
  label = "VAE override",
  className = "",
  compatibleOnly = false,
  loadedArch = null,
  loadedLatentChannels = null,
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

  const visibleVaes = compatibleOnly
    ? vaes.filter((v) => isVaeCompatible(v, loadedLatentChannels, loadedArch))
    : vaes;

  const options = [
    { value: "", label: "Default (model's VAE)" },
    ...visibleVaes.map((v) => {
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
      {!loading && vaes.length > 0 && visibleVaes.length === 0 && (
        <p className="text-xs text-gray-500">
          No compatible VAE found — untick &quot;Show only compatible with loaded model&quot; to show all.
        </p>
      )}
    </div>
  );
}
