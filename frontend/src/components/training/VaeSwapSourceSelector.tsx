"use client";

import { useEffect, useMemo, useState } from "react";
import { listTrainingVaeSources, VaeSwapCandidate } from "@/utils/api";

interface VaeSwapSourceSelectorProps {
  /** Current `vae_swap_source`; "" means no swap. */
  value: string;
  onChange: (value: string) => void;
  /** Architecture of the selected base model; candidates and their
   *  compatibility are answered per architecture by the backend. */
  arch: string | null;
  disabled?: boolean;
}

const GROUPS: { key: keyof VaeSwapSourcesGroups; label: string }[] = [
  { key: "registry", label: "Shared VAE table" },
  { key: "standalone", label: "Standalone VAE files" },
  { key: "extract_from_model", label: "From another checkpoint" },
];

type VaeSwapSourcesGroups = {
  registry: VaeSwapCandidate[];
  standalone: VaeSwapCandidate[];
  extract_from_model: VaeSwapCandidate[];
};

const EMPTY: VaeSwapSourcesGroups = {
  registry: [], standalone: [], extract_from_model: [],
};

const facts = (candidate: VaeSwapCandidate): string => {
  const parts: string[] = [];
  if (candidate.latent_channels != null) parts.push(`${candidate.latent_channels}ch`);
  if (candidate.scale_factor != null) parts.push(`${candidate.scale_factor}x`);
  if (candidate.ndim != null && candidate.ndim !== 4) parts.push(`${candidate.ndim}-D`);
  if (candidate.norm) parts.push(candidate.norm);
  // SenseNova: the token width and the resolution band that moves with it
  // (VAE_SWAP_MIGRATION_DESIGN.md §10.2 — a VAE choice must not move the band
  // without saying so).
  if (candidate.token_pixel_width != null) {
    parts.push(`${candidate.token_pixel_width}px/token`);
  }
  if (candidate.resolution_band_px && candidate.resolution_band_px.length === 2) {
    const mp = candidate.resolution_band_px.map((px) => (px / 1e6).toFixed(1));
    parts.push(`${mp[0]}-${mp[1]} MP`);
  }
  return parts.join(", ");
};

export default function VaeSwapSourceSelector({
  value, onChange, arch, disabled,
}: VaeSwapSourceSelectorProps) {
  const [groups, setGroups] = useState<VaeSwapSourcesGroups>(EMPTY);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!arch) {
      setGroups(EMPTY);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    listTrainingVaeSources(arch)
      .then((data) => {
        if (!cancelled) setGroups({ ...EMPTY, ...(data.sources || {}) });
      })
      .catch((e) => {
        if (!cancelled) {
          setGroups(EMPTY);
          setError(e?.response?.data?.detail || e?.message || "could not be listed");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, [arch]);

  const selected = useMemo(() => {
    for (const group of GROUPS) {
      const hit = groups[group.key].find((c) => c.source === value);
      if (hit) return hit;
    }
    return null;
  }, [groups, value]);

  // A run being edited names a VAE this scan no longer offers (moved file, or a
  // directory that is not scanned any more). Keeping it selectable is the only
  // way the form can save that run without silently dropping its VAE.
  const valueMissing = !!value && !selected;

  // A refused candidate cannot be selected, so its reason has nowhere else to
  // appear -- and the reasons are the interesting part: an extraction from a
  // stock third-party checkpoint is refused when its scaling factor is neither
  // observable nor declared, which is otherwise indistinguishable from the VAE
  // simply not being offered.
  const refused = useMemo(
    () => GROUPS.flatMap(({ key }) => groups[key])
      .filter((candidate) => candidate.compatible === false),
    [groups]);

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">
        VAE Swap (train into another VAE&apos;s latent space)
      </label>
      <select
        value={value || ""}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="w-full px-2 py-1.5 bg-gray-900 border border-gray-700 rounded text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
      >
        <option value="">None (keep this model&apos;s own VAE)</option>
        {GROUPS.map(({ key, label }) => (
          groups[key].length > 0 && (
            <optgroup key={key} label={label}>
              {groups[key].map((candidate) => (
                <option
                  key={candidate.source}
                  value={candidate.source}
                  disabled={candidate.compatible === false}
                >
                  {candidate.name || candidate.source}
                  {facts(candidate) ? ` — ${facts(candidate)}` : ""}
                  {candidate.compatible === false ? " (incompatible)" : ""}
                </option>
              ))}
            </optgroup>
          )
        ))}
        {valueMissing && <option value={value}>{value} (not listed)</option>}
      </select>
      <p className="text-xs text-gray-500 mt-1">
        Replaces the VAE and resizes the backbone&apos;s latent input/output
        layers to the new channel count, copying the channels the two VAEs share
        and zeroing the rest. The result is a checkpoint that only loads with
        this VAE; it is bundled into every save.
        {loading && " Listing candidates…"}
        {error && ` Candidates could not be listed: ${error}`}
        {!loading && !error && arch
          && groups.registry.length + groups.standalone.length
             + groups.extract_from_model.length === 0
          && " No candidate VAE was found in the scanned directories."}
        {valueMissing && " The saved source is not among the listed candidates."}
      </p>
      {selected && selected.compatible === false && (
        <p className="text-xs text-red-400 mt-1">
          This VAE cannot drive the selected model: {selected.reason}
        </p>
      )}
      {refused.length > 0 && (
        <details className="mt-1">
          <summary className="text-xs text-gray-500 cursor-pointer">
            {refused.length} candidate{refused.length === 1 ? "" : "s"} not
            available for this model
          </summary>
          <ul className="mt-1 max-h-32 overflow-y-auto space-y-0.5">
            {refused.map((candidate) => (
              <li key={candidate.source} className="text-xs text-gray-500">
                <span className="text-gray-400">
                  {candidate.name || candidate.source}
                </span>
                {candidate.reason ? `: ${candidate.reason}` : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
