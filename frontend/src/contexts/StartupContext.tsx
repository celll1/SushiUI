"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import {
  fetchGenerationDefaults,
  fetchTrainingDefaults,
  fetchTaggerTrainingDefaults,
  fetchTimestepDefaultsByArch,
  GenerationDefaultsResponse,
} from "@/utils/api";

interface StartupContextType {
  isBackendReady: boolean;
  modelLoaded: boolean;
  generationDefaults: GenerationDefaultsResponse | null;
  trainingDefaults: Record<string, unknown> | null;
  taggerTrainingDefaults: Record<string, unknown> | null;
  timestepDefaultsByArch: Record<string, Record<string, unknown>> | null;
}

const StartupContext = createContext<StartupContextType>({
  isBackendReady: false,
  modelLoaded: false,
  generationDefaults: null,
  trainingDefaults: null,
  taggerTrainingDefaults: null,
  timestepDefaultsByArch: null,
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
  const [hasShownAlert, setHasShownAlert] = useState(false);
  const [generationDefaults, setGenerationDefaults] = useState<GenerationDefaultsResponse | null>(null);
  const [trainingDefaults, setTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [taggerTrainingDefaults, setTaggerTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [timestepDefaultsByArch, setTimestepDefaultsByArch] = useState<Record<string, Record<string, unknown>> | null>(null);

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

          // Fetch param schema defaults from backend (single source of truth)
          try {
            const [genDef, trainDef, taggerDef, tsByArch] = await Promise.all([
              fetchGenerationDefaults(),
              fetchTrainingDefaults(),
              fetchTaggerTrainingDefaults(),
              fetchTimestepDefaultsByArch(),
            ]);
            setGenerationDefaults(genDef);
            setTrainingDefaults(trainDef);
            setTaggerTrainingDefaults(taggerDef);
            setTimestepDefaultsByArch(tsByArch);
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
      generationDefaults,
      trainingDefaults,
      taggerTrainingDefaults,
      timestepDefaultsByArch,
    }}>
      {children}
    </StartupContext.Provider>
  );
}
