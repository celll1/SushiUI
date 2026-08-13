"use client";

import { useEffect, useState } from "react";
import Select from "./Select";
import {
  fetchMiniMaxH3TextEncoders,
  MiniMaxH3ClipProjectionEntry,
  MiniMaxH3TextEncoderEntry,
  MiniMaxH3TextEncodersResponse,
} from "@/utils/api";

interface MiniMaxH3TextEncoderSelectorProps {
  // DiT file or model tree root; the listing is re-read when it changes.
  modelPath: string;
  textEncoderPath: string | null;
  clipProjectionPath: string | null;
  onChange: (textEncoderPath: string | null, clipProjectionPath: string | null) => void;
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

// A converted encoder is usable only through a projection trained for its exact
// hidden size, so an unmatched pairing is never offered.
function projectionsFor(
  encoder: MiniMaxH3TextEncoderEntry | undefined,
  projections: MiniMaxH3ClipProjectionEntry[]
): MiniMaxH3ClipProjectionEntry[] {
  if (!encoder || encoder.hidden_size == null) return [];
  return projections.filter((p) => p.d_in === encoder.hidden_size);
}

export default function MiniMaxH3TextEncoderSelector({
  modelPath,
  textEncoderPath,
  clipProjectionPath,
  onChange,
  disabled = false,
  className = "",
}: MiniMaxH3TextEncoderSelectorProps) {
  const [listing, setListing] = useState<MiniMaxH3TextEncodersResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [staleNotice, setStaleNotice] = useState<string | null>(null);

  useEffect(() => {
    if (!modelPath) {
      setListing(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setStaleNotice(null);
    fetchMiniMaxH3TextEncoders(modelPath)
      .then((data) => {
        if (cancelled) return;
        setListing(data);
      })
      .catch((err: any) => {
        if (cancelled) return;
        setListing(null);
        const detail = err?.response?.data?.detail;
        setError(
          (typeof detail === "string" && detail) ||
            err?.message ||
            "The text encoder listing could not be read."
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelPath]);

  const encoders = listing?.text_encoders ?? [];
  const projections = listing?.clip_projections ?? [];
  const chosenEncoder = encoders.find((te) => te.path === textEncoderPath);
  const usableProjections = projectionsFor(chosenEncoder, projections);
  const needsProjection = chosenEncoder?.requires_projection === true;

  // A stored choice can name a file that is gone, or one this listing marks
  // unusable. Drop it rather than sending a dead path at load time.
  useEffect(() => {
    if (!listing) return;
    if (textEncoderPath) {
      const entry = encoders.find((te) => te.path === textEncoderPath);
      if (!entry) {
        setStaleNotice(
          `${baseName(textEncoderPath)} is not in this model's listing. Falling back to the loader's default.`
        );
        onChange(null, null);
        return;
      }
      if (!entry.compatible) {
        setStaleNotice(
          `${entry.name} cannot be used with this model: ${entry.reason} Falling back to the loader's default.`
        );
        onChange(null, null);
        return;
      }
    }
    if (clipProjectionPath) {
      const stillUsable = projectionsFor(
        encoders.find((te) => te.path === textEncoderPath),
        projections
      ).some((p) => p.path === clipProjectionPath);
      if (!stillUsable) {
        setStaleNotice(
          `${baseName(clipProjectionPath)} does not pair with the selected text encoder. Falling back to auto-discovery.`
        );
        onChange(textEncoderPath, null);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [listing, textEncoderPath, clipProjectionPath]);

  const encoderOptions = [
    { value: "", label: "Default (loader's preference order)" },
    ...encoders.map((te) => {
      const facts: string[] = [];
      if (te.variant) facts.push(te.variant);
      const size = formatSize(te.size_bytes);
      if (size) facts.push(size);
      if (te.hidden_size != null) facts.push(`hidden ${te.hidden_size}`);
      if (te.num_hidden_layers != null) facts.push(`${te.num_hidden_layers} layers`);
      const suffix = facts.length > 0 ? ` (${facts.join(", ")})` : "";
      return {
        value: te.path,
        label: te.compatible
          ? `${te.name}${suffix}`
          : `${te.name}${suffix} — unusable: ${te.reason}`,
        disabled: !te.compatible,
        title: te.reason || undefined,
      };
    }),
  ];

  const projectionOptions = [
    { value: "", label: "Auto-discovery in clip_projections/" },
    ...usableProjections.map((p) => ({
      value: p.path,
      label: `${p.name} (d_in ${p.d_in} → d_out ${p.d_out}, tap ${p.tap})`,
    })),
  ];

  // The measurement is keyed by the (encoder, projection) PAIR, so it only
  // describes what is about to be loaded while the selected projection is the
  // one it was measured with. Auto-discovery resolves to exactly that
  // projection, so the common case shows it; picking a different one must not
  // inherit the number.
  const rawAgreement = chosenEncoder?.agreement ?? null;
  const effectiveProjectionName = clipProjectionPath
    ? usableProjections.find((p) => p.path === clipProjectionPath)?.name ?? null
    : rawAgreement?.projection ?? null;
  const agreementCoversSelection =
    rawAgreement != null
    && effectiveProjectionName != null
    && effectiveProjectionName.toLowerCase() === rawAgreement.projection.toLowerCase();
  const agreement = agreementCoversSelection ? rawAgreement : null;

  return (
    <div className={`space-y-1.5 ${className}`}>
      <Select
        label="MiniMax-H3 text encoder"
        value={textEncoderPath || ""}
        onChange={(e) => onChange(e.target.value || null, null)}
        options={encoderOptions}
        disabled={disabled || loading || !!error}
      />

      {loading && <p className="text-xs text-gray-500">Reading text encoders for this model...</p>}

      {!loading && error && (
        <p className="text-xs text-red-300">Text encoder listing failed: {error}</p>
      )}

      {/* A failed listing cannot tell a live path from a dead one, so the stored
          choice is neither validated nor silently dropped. */}
      {!loading && error && textEncoderPath && (
        <p className="text-xs text-amber-300">
          {baseName(textEncoderPath)} is still stored for this model and will be sent unchecked.
        </p>
      )}

      {!loading && !error && listing && encoders.length === 0 && (
        <p className="text-xs text-gray-500">No text encoders found for this model.</p>
      )}

      {!loading && !error && listing && !textEncoderPath && (
        <p className="text-xs text-gray-500">
          {listing.selected
            ? `Default: ${baseName(listing.selected)} — ${listing.selected_reason}`
            : listing.selected_reason}
        </p>
      )}

      {staleNotice && <p className="text-xs text-amber-300">{staleNotice}</p>}

      {needsProjection && (
        <>
          <Select
            label="Hidden-state projection"
            value={clipProjectionPath || ""}
            onChange={(e) => onChange(textEncoderPath, e.target.value || null)}
            options={projectionOptions}
            disabled={disabled || loading}
          />
          {usableProjections.length === 0 && (
            <p className="text-xs text-amber-300">
              No projection in clip_projections/ has d_in{" "}
              {chosenEncoder?.hidden_size ?? "?"}, which this encoder&apos;s hidden state requires.
            </p>
          )}
        </>
      )}

      {chosenEncoder && (
        <div className="rounded border border-gray-700 bg-gray-800/60 px-2 py-1.5 text-[11px] leading-relaxed text-gray-300">
          {agreement ? (
            <>
              <p>
                {chosenEncoder.name} through {agreement.projection}, measured against{" "}
                {agreement.reference} on {agreement.presentations} prompt-only presentations,
                post-token_refiner.
              </p>
              <p>
                Mean-removed cosine {agreement.cosine}; relative RMS {agreement.rel_rms}, against{" "}
                {agreement.rel_rms_floor} for that same encoder in another quantization.
              </p>
            </>
          ) : rawAgreement && effectiveProjectionName ? (
            <p>
              {chosenEncoder.name} has a measurement recorded through{" "}
              {rawAgreement.projection}, not through {effectiveProjectionName}. No agreement is
              recorded for the pair selected here.
            </p>
          ) : (
            <p>
              No agreement with a released encoder is recorded for {chosenEncoder.name}.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
