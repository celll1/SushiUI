"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import {
  ComponentSlotId,
  CurrentComponentsResponse,
  getCurrentModelComponents,
  switchCurrentModelComponent,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";

interface ModelComponentsContextValue {
  snapshot: CurrentComponentsResponse | null;
  loading: boolean;
  switchingSlot: ComponentSlotId | null;
  error: string | null;
  refresh: () => Promise<void>;
  switchComponent: (slot: ComponentSlotId, candidateId: string) => Promise<void>;
  clearError: () => void;
}

const ModelComponentsContext = createContext<ModelComponentsContextValue | null>(null);

function errorMessage(error: unknown): string {
  const candidate = error as { response?: { data?: { detail?: string } }; message?: string };
  return candidate.response?.data?.detail || candidate.message || "Component operation failed.";
}

export function ModelComponentsProvider({ children }: { children: React.ReactNode }) {
  const { modelInfoVersion } = useStartup();
  const [snapshot, setSnapshot] = useState<CurrentComponentsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [switchingSlot, setSwitchingSlot] = useState<ComponentSlotId | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSnapshot(await getCurrentModelComponents());
    } catch (nextError) {
      setError(errorMessage(nextError));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [modelInfoVersion, refresh]);

  const switchComponent = useCallback(async (slot: ComponentSlotId, candidateId: string) => {
    if (!snapshot) return;
    setSwitchingSlot(slot);
    setError(null);
    try {
      const result = await switchCurrentModelComponent(
        slot,
        candidateId,
        snapshot.model_revision,
        snapshot.component_revision,
      );
      setSnapshot(result.components);
    } catch (nextError) {
      setError(errorMessage(nextError));
      await refresh();
      throw nextError;
    } finally {
      setSwitchingSlot(null);
    }
  }, [refresh, snapshot]);

  const value = useMemo(() => ({
    snapshot,
    loading,
    switchingSlot,
    error,
    refresh,
    switchComponent,
    clearError: () => setError(null),
  }), [snapshot, loading, switchingSlot, error, refresh, switchComponent]);

  return <ModelComponentsContext.Provider value={value}>{children}</ModelComponentsContext.Provider>;
}

export function useModelComponents(): ModelComponentsContextValue {
  const value = useContext(ModelComponentsContext);
  if (!value) throw new Error("useModelComponents must be used inside ModelComponentsProvider");
  return value;
}
