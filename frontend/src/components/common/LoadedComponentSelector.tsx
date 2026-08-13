"use client";

import Select from "./Select";
import type { ReactNode } from "react";
import { ComponentOrigin, ComponentSlotState } from "@/utils/api";

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

interface LoadedComponentSelectorProps {
  label: string;
  state?: ComponentSlotState;
  compatibleOnly: boolean;
  switching: boolean;
  onSwitch: (candidateId: string) => Promise<void>;
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
  if (!state?.visible) return null;
  const currentId = state.current?.candidate_id ?? "";
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

  return (
    <div className="space-y-1 rounded-md border border-gray-800 bg-gray-900/40 p-2">
      <Select
        label={label}
        value={currentId}
        disabled={switching || !state.current}
        aria-busy={switching}
        onChange={(event) => {
          if (event.target.value !== currentId) void onSwitch(event.target.value).catch(() => undefined);
        }}
        options={options.length ? options : [{ value: currentId, label: "No scanned candidates" }]}
      />
      {state.current && (
        <p className="text-[11px] text-gray-500">
          {ORIGIN_LABELS[state.current.origin]} · {state.current.residency}
          {state.reason ? ` · ${state.reason}` : ""}
        </p>
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
