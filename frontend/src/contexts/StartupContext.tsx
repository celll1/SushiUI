"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface StartupContextType {
  isBackendReady: boolean;
  modelLoaded: boolean;
}

const StartupContext = createContext<StartupContextType>({
  isBackendReady: false,
  modelLoaded: false,
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

  useEffect(() => {
    // Prevent duplicate polling if already started
    if (globalPollingStarted) {
      console.log("[StartupContext] Already polling, skipping duplicate mount");
      return;
    }

    globalPollingStarted = true;
    console.log("[StartupContext] Initializing...");
    let hasShownAlertInSession = false;

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

          // Show alert only once per session
          // if (!hasShownAlertInSession) {
          //   hasShownAlertInSession = true;
          //   alert("Model loaded successfully!");
          // }
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
    <StartupContext.Provider value={{ isBackendReady, modelLoaded }}>
      {children}
    </StartupContext.Provider>
  );
}
