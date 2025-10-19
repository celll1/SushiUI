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

export function StartupProvider({ children }: StartupProviderProps) {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [hasShownAlert, setHasShownAlert] = useState(false);

  useEffect(() => {
    // Check if we already checked in this session
    const alreadyChecked = sessionStorage.getItem("startup_model_checked") === "true";

    if (alreadyChecked) {
      setIsBackendReady(true);
      setModelLoaded(true);
      setHasShownAlert(true);
      return;
    }

    // Poll backend for model load status
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch("/api/models/current");
        const data = await response.json();

        if (data.loaded) {
          clearInterval(pollInterval);
          setIsBackendReady(true);
          setModelLoaded(true);
          sessionStorage.setItem("startup_model_checked", "true");

          // Show alert only once
          if (!hasShownAlert) {
            setHasShownAlert(true);
            alert("Model loaded successfully!");
          }
        }
      } catch (error) {
        // Backend not ready yet, will retry
        console.log("Waiting for backend to start...");
      }
    }, 1000);

    // Stop polling after 60 seconds
    setTimeout(() => clearInterval(pollInterval), 60000);

    return () => clearInterval(pollInterval);
  }, [hasShownAlert]);

  return (
    <StartupContext.Provider value={{ isBackendReady, modelLoaded }}>
      {children}
    </StartupContext.Provider>
  );
}
