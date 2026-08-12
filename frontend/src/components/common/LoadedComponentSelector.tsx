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
  const options = candidates.map((candidate) => {
    const reason = candidate.switch_reason || candidate.compatibility_reason || undefined;
    const disabled = !candidate.is_current && (!candidate.switchable || candidate.compatibility !== "compatible");
    const tags = [ORIGIN_LABELS[candidate.origin]];
    if (candidate.compatibility !== "compatible") tags.push(candidate.compatibility);
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
      {state.runtime_override && (
        <p className="text-[11px] text-amber-400">
          Generation-only override: {state.runtime_override.display_name}. Resident selection is unchanged.
        </p>
      )}
      {children}
    </div>
  );
}
