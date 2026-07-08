"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import {
  fetchGenerationDefaults,
  fetchTrainingDefaults,
  fetchTaggerTrainingDefaults,
  fetchTimestepDefaultsByArch,
  fetchBundleVaeDefaultsByArch,
  GenerationDefaultsResponse,
} from "@/utils/api";

// Loaded-model info object returned by GET /models/current -> model_info.
// Single source of truth for the currently loaded model across generation panels.
export interface ModelInfo {
  type?: string;
  is_video?: boolean;
  source?: string;
  is_v_prediction?: boolean;
  [key: string]: unknown;
}

interface StartupContextType {
  isBackendReady: boolean;
  modelLoaded: boolean;
  // Currently loaded model info (model_info from GET /models/current), or null.
  modelInfo: ModelInfo | null;
  // Derived convenience flag: modelInfo?.is_video === true.
  isVideo: boolean;
  // Re-fetch modelInfo (call after a (re)load so the shared source updates).
  refreshModelInfo: () => Promise<void>;
  generationDefaults: GenerationDefaultsResponse | null;
  trainingDefaults: Record<string, unknown> | null;
  taggerTrainingDefaults: Record<string, unknown> | null;
  timestepDefaultsByArch: Record<string, Record<string, unknown>> | null;
  bundleVaeDefaultsByArch: Record<string, boolean> | null;
}

const StartupContext = createContext<StartupContextType>({
  isBackendReady: false,
  modelLoaded: false,
  modelInfo: null,
  isVideo: false,
  refreshModelInfo: async () => {},
  generationDefaults: null,
  trainingDefaults: null,
  taggerTrainingDefaults: null,
  timestepDefaultsByArch: null,
  bundleVaeDefaultsByArch: null,
});

export const useStartup = () => useContext(StartupContext);

interface StartupProviderProps {
  children: ReactNode;
}

// Global flag to prevent duplicate polling (across re-mounts in dev mode)
let globalPollingStarted = false;

export function StartupProvider({ children }: StartupProviderProps) {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [hasShownAlert, setHasShownAlert] = useState(false);
  const [generationDefaults, setGenerationDefaults] = useState<GenerationDefaultsResponse | null>(null);
  const [trainingDefaults, setTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [taggerTrainingDefaults, setTaggerTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [timestepDefaultsByArch, setTimestepDefaultsByArch] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [bundleVaeDefaultsByArch, setBundleVaeDefaultsByArch] = useState<Record<string, boolean> | null>(null);

  // Re-fetch the currently loaded model info. Panels call this after a
  // (re)load so modelInfo/isVideo stay the single source of truth.
  const refreshModelInfo = async () => {
    try {
      const response = await fetch("/api/models/current");
      const data = await response.json();
      if (data.loaded) {
        setModelInfo((data.model_info as ModelInfo) ?? null);
      } else {
        setModelInfo(null);
      }
    } catch (error) {
      console.warn("[StartupContext] Failed to refresh modelInfo", error);
    }
  };

  useEffect(() => {
    // Prevent duplicate polling if already started
    if (globalPollingStarted) {
      console.log("[StartupContext] Already polling, skipping duplicate mount");
      return;
    }

    globalPollingStarted = true;
    console.log("[StartupContext] Initializing...");

    // Poll backend for model load status (always poll, don't use sessionStorage)
    const pollInterval = setInterval(async () => {
      try {
        console.log("[StartupContext] Polling /api/models/current...");
        const response = await fetch("/api/models/current");
        const data = await response.json();
        console.log("[StartupContext] Response:", data);

        if (data.loaded) {
          clearInterval(pollInterval);
          console.log("[StartupContext] Model loaded! Updating state...");
          setIsBackendReady(true);
          setModelLoaded(true);
          // Seed the shared model-info source from the same poll response.
          setModelInfo((data.model_info as ModelInfo) ?? null);

          // Fetch param schema defaults from backend (single source of truth)
          try {
            const [genDef, trainDef, taggerDef, tsByArch, bvByArch] = await Promise.all([
              fetchGenerationDefaults(),
              fetchTrainingDefaults(),
              fetchTaggerTrainingDefaults(),
              fetchTimestepDefaultsByArch(),
              fetchBundleVaeDefaultsByArch(),
            ]);
            setGenerationDefaults(genDef);
            setTrainingDefaults(trainDef);
            setTaggerTrainingDefaults(taggerDef);
            setTimestepDefaultsByArch(tsByArch);
            setBundleVaeDefaultsByArch(bvByArch);
            console.log("[StartupContext] Param defaults loaded from backend");
          } catch (e) {
            console.warn("[StartupContext] Failed to fetch param defaults, using hardcoded fallbacks", e);
          }
        }
      } catch (error) {
        // Backend not ready yet, will retry
        console.log("[StartupContext] Waiting for backend to start...", error);
      }
    }, 1000);

    // Stop polling after 60 seconds
    const timeout = setTimeout(() => {
      console.log("[StartupContext] Polling timeout reached");
      clearInterval(pollInterval);
      globalPollingStarted = false; // Allow retry on timeout
    }, 60000);

    return () => {
      console.log("[StartupContext] Cleanup");
      clearInterval(pollInterval);
      clearTimeout(timeout);
    };
  }, []); // Empty dependency array - only run once on mount

  return (
    <StartupContext.Provider value={{
      isBackendReady,
      modelLoaded,
      modelInfo,
      isVideo: modelInfo?.is_video === true,
      refreshModelInfo,
      generationDefaults,
      trainingDefaults,
      taggerTrainingDefaults,
      timestepDefaultsByArch,
      bundleVaeDefaultsByArch,
    }}>
      {children}
    </StartupContext.Provider>
  );
}
