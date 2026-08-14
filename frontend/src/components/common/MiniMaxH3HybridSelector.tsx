"use client";

import { useEffect, useState } from "react";
import Select from "./Select";
import NumberInput from "./NumberInput";
import {
  fetchMiniMaxH3HybridOverlays,
  MiniMaxH3HybridLoadRequest,
  MiniMaxH3HybridOverlaysResponse,
} from "@/utils/api";

// What the parent shows before this component has answered for a restored
// choice, so the load button is never briefly enabled on an unchecked overlay.
export const HYBRID_CHECK_PENDING = "The overlay compatibility check has not finished.";

interface MiniMaxH3HybridSelectorProps {
  // The BASE partition: the DiT file picked in the model dropdown.
  modelPath: string;
  value: MiniMaxH3HybridLoadRequest | null;
  onChange: (next: MiniMaxH3HybridLoadRequest | null) => void;
  // Non-null while this selection must not be sent, carrying the reason.
  onBlockedChange: (reason: string | null) => void;
  disabled?: boolean;
  className?: string;
}

function baseName(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function formatSize(bytes: number): string {
  if (!bytes || bytes <= 0) return "";
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

// The endpoint's `defaults.presets` is the loader's implemented set; the union
// in the request type names only the one that exists today.
const asPreset = (preset: string): MiniMaxH3HybridLoadRequest["preset"] =>
  preset as MiniMaxH3HybridLoadRequest["preset"];

export default function MiniMaxH3HybridSelector({
  modelPath,
  value,
  onChange,
  onBlockedChange,
  disabled = false,
  className = "",
}: MiniMaxH3HybridSelectorProps) {
  // Tagged with the base it answers for. An effect below decides whether the
  // load may proceed, and effects see the PREVIOUS render's values: on a model
  // switch it would otherwise reach a verdict from the old base's listing,
  // which in a tree whose two bases list each other is a plausible verdict.
  const [fetched, setFetched] = useState<{
    path: string;
    data: MiniMaxH3HybridOverlaysResponse | null;
    error: string | null;
  } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!modelPath) return;
    let cancelled = false;
    setLoading(true);
    fetchMiniMaxH3HybridOverlays(modelPath)
      .then((data) => {
        if (!cancelled) setFetched({ path: modelPath, data, error: null });
      })
      .catch((err: any) => {
        if (cancelled) return;
        const detail = err?.response?.data?.detail;
        setFetched({
          path: modelPath,
          data: null,
          error:
            (typeof detail === "string" && detail) ||
            err?.message ||
            "The overlay listing could not be read.",
        });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelPath]);

  const current = fetched && fetched.path === modelPath ? fetched : null;
  const listing = current?.data ?? null;
  const error = current?.error ?? null;

  const overlayPath = value?.overlay_file || "";
  const defaults = listing?.defaults ?? null;
  const overlays = listing?.overlays ?? [];
  const candidate = overlays.find((o) => o.path === overlayPath) ?? null;
  const numBlocks = listing?.base.num_blocks ?? 0;
  const lastBlock = Math.max(numBlocks - 1, 0);

  const preset = value?.preset ?? defaults?.preset ?? "";
  const rangeStart = value?.block_range_start ?? defaults?.block_range_start ?? 0;
  const rangeEnd = value?.block_range_end ?? defaults?.block_range_end ?? 0;
  const finalAdaln = value?.final_adaln_from_overlay ?? defaults?.final_adaln_from_overlay ?? false;
  const emptyRange = rangeStart > rangeEnd;
  // Only meaningful against a listing; a restored range never passes through
  // NumberInput's clamp, so it can name a block this base does not have.
  const overRange = !!listing && rangeEnd > lastBlock;

  // A stored overlay that this listing does not offer, or offers as
  // incompatible, is REPORTED rather than cleared: clearing it would turn a
  // hybrid the user asked for into a base-only load that reports success. Every
  // one of these states leaves the overlay Select enabled, so "None" is always
  // one click away.
  useEffect(() => {
    if (!overlayPath) {
      onBlockedChange(null);
      return;
    }
    if (loading || !current) {
      onBlockedChange(HYBRID_CHECK_PENDING);
      return;
    }
    if (error || !listing) {
      onBlockedChange(`The overlay compatibility check did not run: ${error}`);
      return;
    }
    if (!candidate) {
      onBlockedChange(
        `${baseName(overlayPath)} is not one of the overlay candidates listed for this base.`
      );
      return;
    }
    if (!candidate.compatible) {
      onBlockedChange(
        `${candidate.name} cannot be merged with this base: ${candidate.reason ?? "no reason reported"}`
      );
      return;
    }
    if (emptyRange) {
      onBlockedChange(
        `Blocks ${rangeStart}..${rangeEnd} select no blocks; the last block cannot be below the first.`
      );
      return;
    }
    if (overRange) {
      onBlockedChange(
        `Blocks ${rangeStart}..${rangeEnd} run past this base's last block (${lastBlock}).`
      );
      return;
    }
    onBlockedChange(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayPath, loading, current, error, listing, candidate, emptyRange, overRange,
      rangeStart, rangeEnd, lastBlock]);

  const selectOverlay = (path: string) => {
    if (!path) {
      onChange(null);
      return;
    }
    // Unreachable: every enabled non-empty option comes from the listing that
    // carries these defaults. Not a silent onChange(null) -- clearing a stored
    // overlay must stay something the user asked for by picking "None".
    if (!defaults) return;
    onChange({
      overlay_file: path,
      preset: asPreset(defaults.preset),
      block_range_start: defaults.block_range_start,
      block_range_end: defaults.block_range_end,
      final_adaln_from_overlay: defaults.final_adaln_from_overlay,
    });
  };

  const patch = (fields: Partial<MiniMaxH3HybridLoadRequest>) => {
    if (!overlayPath) return;
    onChange({
      overlay_file: overlayPath,
      preset: asPreset(preset),
      block_range_start: rangeStart,
      block_range_end: rangeEnd,
      final_adaln_from_overlay: finalAdaln,
      ...fields,
    });
  };

  // A stored overlay this listing does not name still gets an option, disabled:
  // without one the box renders empty above amber text about a file it does not
  // show, and the same reasoning applies to the preset below.
  const overlayOptions = [
    { value: "", label: "None (load the selected checkpoint on its own)" },
    ...(overlayPath && !candidate
      ? [
          {
            value: overlayPath,
            label: `${baseName(overlayPath)} — stored for this base, not in the current listing`,
            disabled: true,
          },
        ]
      : []),
    ...overlays.map((o) => {
      const facts: string[] = [];
      if (o.variant) facts.push(o.variant);
      const size = formatSize(o.size_bytes);
      if (size) facts.push(size);
      if (o.quantization_format) facts.push(o.quantization_format);
      const suffix = facts.length > 0 ? ` (${facts.join(", ")})` : "";
      return {
        value: o.path,
        label: o.compatible
          ? `${o.name}${suffix}`
          : `${o.name}${suffix} — cannot be merged with this base: ${o.reason ?? ""}`,
        disabled: !o.compatible,
        title: o.refusal_code ? `[${o.refusal_code}] ${o.reason ?? ""}` : undefined,
      };
    }),
  ];

  // A stored preset the endpoint no longer lists is kept as an option: dropping
  // it would leave the select showing one value while the request sends another.
  const presetChoices = defaults?.presets ?? [];
  const presetOptions = (preset && !presetChoices.includes(preset)
    ? [...presetChoices, preset]
    : presetChoices
  ).map((p) => ({ value: p, label: p }));

  return (
    <div className={`space-y-1.5 ${className}`}>
      <Select
        label="MiniMax-H3 overlay checkpoint"
        value={overlayPath}
        onChange={(e) => selectOverlay(e.target.value)}
        options={overlayOptions}
        // Enabled even when the listing failed: a stored overlay blocks the
        // load, and this Select is the only control that can clear it. Disabling
        // it there strands the user with no way to load the base alone.
        disabled={disabled || loading}
      />

      {loading && (
        <p className="text-xs text-gray-500">
          Reading which checkpoints in this tree can be merged with this base...
        </p>
      )}

      {!loading && error && (
        <p className="text-xs text-red-300">
          Overlay listing failed: {error}
          {overlayPath ? " Select None above to load this checkpoint on its own." : ""}
        </p>
      )}

      {!loading && !error && listing && overlays.length === 0 && (
        <p className="text-xs text-gray-500">
          This tree holds no second MiniMax-H3 checkpoint to merge with.
        </p>
      )}

      {!loading && !error && listing && overlays.length > 0 && (
        <p className="text-xs text-gray-500">
          Base: {listing.base.name}
          {listing.base.variant ? ` (${listing.base.variant})` : ""}, {numBlocks} blocks. Each
          candidate was checked over blocks{" "}
          {`${listing.checked_block_range[0]}..${listing.checked_block_range[1]}`}, so a candidate
          listed as compatible stays compatible for any range selected below.
        </p>
      )}

      {/* The recipe controls need the listing: their bounds are the base's own
          block count, and offering a range against a placeholder would clamp
          typed values to it. */}
      {overlayPath && defaults && (
        <div className="space-y-1.5 rounded border border-gray-700 bg-gray-800/60 px-2 py-2">
          {presetOptions.length > 0 && (
            <Select
              label="Overlay preset"
              value={preset}
              onChange={(e) => patch({ preset: asPreset(e.target.value) })}
              options={presetOptions}
              disabled={disabled}
            />
          )}

          <div className="flex flex-wrap items-end gap-3">
            <label className="flex items-center gap-1.5 text-xs text-gray-400">
              First block
              <NumberInput
                label="First overlaid block"
                value={rangeStart}
                onCommit={(v) => patch({ block_range_start: v })}
                min={0}
                max={lastBlock}
                step={1}
                parse="int"
                className="w-16"
                disabled={disabled}
              />
            </label>
            <label className="flex items-center gap-1.5 text-xs text-gray-400">
              Last block (inclusive)
              <NumberInput
                label="Last overlaid block, inclusive"
                value={rangeEnd}
                onCommit={(v) => patch({ block_range_end: v })}
                min={0}
                max={lastBlock}
                step={1}
                parse="int"
                className="w-16"
                disabled={disabled}
              />
            </label>
          </div>

          {emptyRange ? (
            <p className="text-xs text-amber-300">
              Blocks {rangeStart}..{rangeEnd} select no blocks; the last block cannot be below the
              first.
            </p>
          ) : overRange ? (
            <p className="text-xs text-amber-300">
              Blocks {rangeStart}..{rangeEnd} run past this base&apos;s last block ({lastBlock}).
            </p>
          ) : (
            <p className="text-[11px] leading-relaxed text-gray-400">
              Blocks {rangeStart}..{rangeEnd} of {numBlocks} take their AdaLN projection from{" "}
              {candidate?.name ?? baseName(overlayPath)}. Every other tensor is read from the base.
            </p>
          )}

          <details className="text-xs text-gray-400">
            <summary className="cursor-pointer select-none">Advanced</summary>
            <label className="mt-2 flex cursor-pointer items-start gap-2">
              <input
                type="checkbox"
                checked={finalAdaln}
                onChange={(e) => patch({ final_adaln_from_overlay: e.target.checked })}
                disabled={disabled}
                className="mt-0.5 h-4 w-4 rounded border-gray-600 bg-gray-700 text-blue-500 focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-gray-300">
                Also take the final-layer AdaLN projection from the overlay
              </span>
            </label>
            <p className="mt-1 text-[11px] leading-relaxed text-gray-500">
              The compatibility check above was taken with this off, and it selects a tensor that
              check never read. With it on the load can still be refused, and the refusal arrives
              when you press Load.
            </p>
          </details>

          <p className="text-[11px] leading-relaxed text-amber-300">
            A merged checkpoint loads and can be inspected, and every generation endpoint refuses
            it. The A/B measurement that would release generation for it has not been run.
          </p>
        </div>
      )}
    </div>
  );
}
