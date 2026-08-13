"use client";

import Select from "./Select";
import { useEffect, useState, type ReactNode } from "react";
import { ComponentCandidate, ComponentOrigin, ComponentSlotState, MiniMaxH3ProjectionCandidate } from "@/utils/api";
import { agreementCoversProjection } from "@/utils/minimaxH3Projection";

const ORIGIN_LABELS: Record<ComponentOrigin, string> = {
  embedded_checkpoint: "embedded",
  model_tree: "model tree",
  architecture_default: "default external",
  selected_external: "selected external",
  unused: "not used",
  unavailable: "unavailable",
};

function sizeLabel(bytes?: number | null): string {
  if (!bytes) return "";
  const gib = bytes / 1024 ** 3;
  return gib >= 0.1 ? ` · ${gib.toFixed(gib >= 10 ? 1 : 2)} GiB` : ` · ${(bytes / 1024 ** 2).toFixed(0)} MiB`;
}

// A candidate that needs a projection cannot be switched to until one is named,
// so selecting it stages the switch instead of starting it.
function needsProjectionChoice(candidate: ComponentCandidate | undefined): boolean {
  return !!candidate?.requires_projection && (candidate.projection_candidates?.length ?? 0) > 0;
}

function projectionLabel(projection: MiniMaxH3ProjectionCandidate): string {
  const facts = `d_in ${projection.d_in} → d_out ${projection.d_out}, tap ${projection.tap}`;
  return projection.usable
    ? `${projection.name} (${facts})`
    : `${projection.name} (${facts}) — unusable: ${projection.reason}`;
}

interface LoadedComponentSelectorProps {
  label: string;
  state?: ComponentSlotState;
  compatibleOnly: boolean;
  switching: boolean;
  onSwitch: (candidateId: string, projectionPath?: string | null) => Promise<void>;
  children?: ReactNode;
}

export default function LoadedComponentSelector({
  label,
  state,
  compatibleOnly,
  switching,
  onSwitch,
  children,
}: LoadedComponentSelectorProps) {
  const currentId = state?.current?.candidate_id ?? "";
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [pendingProjection, setPendingProjection] = useState<string>("");

  // A completed switch (or any refresh that moves the slot) invalidates a
  // staged pair; keeping it would show a pick for an encoder that already left.
  useEffect(() => {
    setPendingId(null);
    setPendingProjection("");
  }, [currentId]);

  if (!state?.visible) return null;
  const candidates = state.candidates.filter((candidate) => (
    !compatibleOnly || candidate.is_current || candidate.compatibility === "compatible"
  ));
  const currentCandidate = candidates.find((candidate) => candidate.is_current);
  const options = candidates.map((candidate) => {
    const reason = candidate.switch_reason || candidate.compatibility_reason || undefined;
    const disabled = !candidate.is_current && (!candidate.switchable || candidate.compatibility !== "compatible");
    const tags = [ORIGIN_LABELS[candidate.origin]];
    if (candidate.compatibility !== "compatible") tags.push(candidate.compatibility);
    // Which encoder is which matters before the switch, not after it.
    if (candidate.requires_projection) {
      tags.push(candidate.projection ? `via ${candidate.projection}` : "no projection resolved");
    }
    return {
      value: candidate.candidate_id,
      label: `${candidate.display_name} — ${tags.join(", ")}${sizeLabel(candidate.container_size_bytes)}`,
      disabled,
      title: reason,
    };
  });

  // Looked up in the FILTERED list, so hiding a row also drops a pick staged for it.
  const pending = pendingId ? candidates.find((candidate) => candidate.candidate_id === pendingId) : undefined;
  const pendingProjections = pending?.projection_candidates ?? [];
  const usableProjections = pendingProjections.filter((projection) => projection.usable);
  const pendingAgreement = pending?.agreement ?? null;
  const pendingAgreementCovers = agreementCoversProjection(pendingAgreement, pendingProjection);

  const selectCandidate = (value: string) => {
    if (value === currentId) {
      setPendingId(null);
      setPendingProjection("");
      return;
    }
    const candidate = candidates.find((item) => item.candidate_id === value);
    if (needsProjectionChoice(candidate)) {
      const usable = (candidate?.projection_candidates ?? []).filter((projection) => projection.usable);
      setPendingId(value);
      setPendingProjection(usable.length === 1 ? usable[0].path : "");
      return;
    }
    setPendingId(null);
    setPendingProjection("");
    void onSwitch(value).catch(() => undefined);
  };

  return (
    <div className="space-y-1 rounded-md border border-gray-800 bg-gray-900/40 p-2">
      <Select
        label={label}
        value={pending ? pending.candidate_id : currentId}
        disabled={switching || !state.current}
        aria-busy={switching}
        onChange={(event) => selectCandidate(event.target.value)}
        options={options.length ? options : [{ value: currentId, label: "No scanned candidates" }]}
      />
      {state.current && (
        <p className="text-[11px] text-gray-500">
          {ORIGIN_LABELS[state.current.origin]} · {state.current.residency}
          {state.reason ? ` · ${state.reason}` : ""}
        </p>
      )}
      {pending && (
        <div className="space-y-1 rounded border border-violet-700 bg-violet-950/30 p-2">
          <Select
            label="Hidden-state projection"
            value={pendingProjection}
            disabled={switching}
            onChange={(event) => setPendingProjection(event.target.value)}
            options={[
              ...(usableProjections.length === 1
                ? []
                : [{ value: "", label: "Select a projection", disabled: true }]),
              ...pendingProjections.map((projection) => ({
                value: projection.path,
                label: projectionLabel(projection),
                disabled: !projection.usable,
                title: projection.reason || undefined,
              })),
            ]}
          />
          <p className="text-[11px] text-gray-400">
            {usableProjections.length === 0
              ? "No file in clip_projections/ passes every pairing gate for this encoder."
              : usableProjections.length === 1
                ? `${usableProjections[0].name} is the only file in clip_projections/ declaring d_in ${usableProjections[0].d_in}; it was resolved automatically.`
                : `${usableProjections.length} projections declare d_in ${usableProjections[0].d_in}. Which one was trained for ${pending.display_name} is not derivable from the files; select one.`}
          </p>
          {pendingProjection && (
            <p className="text-[11px] text-gray-400">
              {pendingAgreementCovers && pendingAgreement
                ? `Measured against ${pendingAgreement.reference} on ${pendingAgreement.presentations} prompt-only presentations, post-token_refiner: mean-removed cosine ${pendingAgreement.cosine}; relative RMS ${pendingAgreement.rel_rms}, against ${pendingAgreement.rel_rms_floor} for that same encoder in another quantization.`
                : pendingAgreement
                  ? `A measurement is recorded for this encoder through ${pendingAgreement.projection}, not through the projection selected here.`
                  : "No agreement with a released encoder is recorded for the pair selected here."}
            </p>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              className="rounded bg-violet-600 px-2 py-1 text-[11px] text-white disabled:opacity-50"
              disabled={switching || !pendingProjection}
              onClick={() => void onSwitch(pending.candidate_id, pendingProjection).catch(() => undefined)}
            >
              Switch to this pair
            </button>
            <button
              type="button"
              className="rounded border border-gray-600 px-2 py-1 text-[11px] text-gray-300"
              disabled={switching}
              onClick={() => {
                setPendingId(null);
                setPendingProjection("");
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
      {currentCandidate?.requires_projection && currentCandidate.projection && (
        <div className="rounded border border-gray-700 bg-gray-800/60 px-2 py-1.5 text-[11px] leading-relaxed text-gray-300">
          <p>
            {currentCandidate.display_name} conditions this model through {currentCandidate.projection},
            not through a released Qwen3-VL-32B text encoder.
          </p>
          {currentCandidate.agreement ? (
            <p>
              Measured against {currentCandidate.agreement.reference} on{" "}
              {currentCandidate.agreement.presentations} prompt-only presentations,
              post-token_refiner: mean-removed cosine {currentCandidate.agreement.cosine}; relative
              RMS {currentCandidate.agreement.rel_rms}, against{" "}
              {currentCandidate.agreement.rel_rms_floor} for that same encoder in another
              quantization.
            </p>
          ) : (
            <p>No agreement with a released encoder is recorded for this pair.</p>
          )}
        </div>
      )}
      {state.runtime_override && (
        <p className="text-[11px] text-amber-400">
          Generation-only override: {state.runtime_override.display_name}. Resident selection is unchanged.
        </p>
      )}
      {children}
    </div>
  );
}
