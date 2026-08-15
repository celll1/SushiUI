"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  ComponentSlotId,
  CurrentComponentsResponse,
  getCurrentModelComponents,
  switchCurrentModelComponent,
} from "@/utils/api";

interface ModelComponentsContextValue {
  snapshot: CurrentComponentsResponse | null;
  loading: boolean;
  switchingSlot: ComponentSlotId | null;
  error: string | null;
  refresh: (expectedModelRevision?: number) => Promise<void>;
  // projectionPath: MiniMax-H3 text encoders only (see switchCurrentModelComponent).
  switchComponent: (slot: ComponentSlotId, candidateId: string, projectionPath?: string | null) => Promise<void>;
  clearError: () => void;
}

const ModelComponentsContext = createContext<ModelComponentsContextValue | null>(null);

function errorMessage(error: unknown): string {
  const candidate = error as { response?: { data?: { detail?: string } }; message?: string };
  return candidate.response?.data?.detail || candidate.message || "Component operation failed.";
}

export function ModelComponentsProvider({ children }: { children: React.ReactNode }) {
  const [snapshot, setSnapshot] = useState<CurrentComponentsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [switchingSlot, setSwitchingSlot] = useState<ComponentSlotId | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);
  const mountedRef = useRef(true);
  const refreshInFlightRef = useRef<{ key: string; promise: Promise<void> } | null>(null);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = useCallback((expectedModelRevision?: number) => {
    const key = expectedModelRevision == null ? "current" : String(expectedModelRevision);
    const existing = refreshInFlightRef.current;
    if (existing?.key === key) return existing.promise;

    const requestId = ++requestIdRef.current;
    setLoading(true);
    const promise = (async () => {
      try {
        const next = await getCurrentModelComponents();
        if (!mountedRef.current || requestId !== requestIdRef.current) return;
        setSnapshot(next);
        // A successful fetch retires whatever the last failure said. Without
        // this the banner from one backend restart stays up for the rest of the
        // session, until someone closes it by hand.
        setError(null);
      } catch (nextError) {
        if (!mountedRef.current || requestId !== requestIdRef.current) return;
        setError(errorMessage(nextError));
      } finally {
        if (mountedRef.current && requestId === requestIdRef.current) setLoading(false);
      }
    })();
    const record = { key, promise };
    refreshInFlightRef.current = record;
    void promise.finally(() => {
      if (refreshInFlightRef.current === record) refreshInFlightRef.current = null;
    });
    return promise;
  }, []);

  const switchComponent = useCallback(async (
    slot: ComponentSlotId,
    candidateId: string,
    projectionPath?: string | null,
  ) => {
    if (!snapshot) return;
    const requestId = ++requestIdRef.current;
    setSwitchingSlot(slot);
    setError(null);
    try {
      const result = await switchCurrentModelComponent(
        slot,
        candidateId,
        snapshot.model_revision,
        snapshot.component_revision,
        projectionPath,
      );
      if (mountedRef.current && requestId === requestIdRef.current) setSnapshot(result.components);
    } catch (nextError) {
      if (mountedRef.current && requestId === requestIdRef.current) {
        setError(errorMessage(nextError));
        await refresh();
      }
      throw nextError;
    } finally {
      if (mountedRef.current) setSwitchingSlot(null);
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
