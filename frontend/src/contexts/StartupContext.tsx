"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef, ReactNode } from "react";
import {
  fetchGenerationDefaults,
  fetchTrainingDefaults,
  fetchTaggerTrainingDefaults,
  fetchVaeTrainingDefaults,
  fetchTimestepDefaultsByArch,
  fetchBundleVaeDefaultsByArch,
  fetchArchCapabilities,
  fetchGenerationSettings,
  getCurrentModel,
  GenerationDefaultsResponse,
  ArchCapabilities,
} from "@/utils/api";

// Loaded-model info object returned by GET /models/current -> model_info.
// Single source of truth for the currently loaded model across generation panels.
export interface ModelInfo {
  type?: string;
  is_video?: boolean;
  is_audio?: boolean;
  // MiniMax-H3 only: which of the two released transformer partitions is
  // loaded ("fl2va" | "ref2va"). They share every other component and are
  // otherwise indistinguishable, and which one is loaded decides which
  // workflows the model can serve -- ref2va is the only one that reads
  // reference rows (/generate/ref2vid).
  variant?: string;
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
  // Derived convenience flag: modelInfo?.is_audio === true.
  isAudio: boolean;
  // Re-fetch modelInfo (call after a (re)load so the shared source updates).
  // Resolves to the freshly fetched info, or the last known info if the
  // request failed.
  refreshModelInfo: () => Promise<ModelInfo | null>;
  // Authoritative modality check for dispatch time. Re-fetches
  // GET /models/current and returns the modality of what is ACTUALLY loaded,
  // so a panel never routes an image request to a video model (or vice versa)
  // on the strength of a cached flag. `fresh` is false when the request failed
  // and the cached flags were used instead.
  resolveModality: () => Promise<{ isVideo: boolean; isAudio: boolean; fresh: boolean; modelInfo: ModelInfo | null }>;
  // Increments whenever the loaded model's identity changes (arch/source/
  // variant/modality). Components holding their own copy of GET /models/current
  // can use it as an effect dependency to re-fetch.
  modelInfoVersion: number;
  generationDefaults: GenerationDefaultsResponse | null;
  trainingDefaults: Record<string, unknown> | null;
  taggerTrainingDefaults: Record<string, unknown> | null;
  vaeTrainingDefaults: Record<string, unknown> | null;
  timestepDefaultsByArch: Record<string, Record<string, unknown>> | null;
  bundleVaeDefaultsByArch: Record<string, boolean> | null;
  // Per-architecture capability matrix (GET /schema/arch-capabilities).
  // null until fetched; archSupportsFeature() treats null as "supported" so a
  // control is never hidden just because the matrix has not arrived.
  archCapabilities: ArchCapabilities | null;
  // Upper bound for the video frame-count SLIDER TRACK (backend UserSettings.
  // video_frame_slider_max, GET /settings/generation). null = unset = the
  // slider's own built-in track reach (VideoFrameCountSlider's constants).
  // Never bounds the paired number box.
  videoFrameSliderMax: number | null;
}

const StartupContext = createContext<StartupContextType>({
  isBackendReady: false,
  modelLoaded: false,
  modelInfo: null,
  isVideo: false,
  isAudio: false,
  refreshModelInfo: async () => null,
  resolveModality: async () => ({ isVideo: false, isAudio: false, fresh: false, modelInfo: null }),
  modelInfoVersion: 0,
  generationDefaults: null,
  trainingDefaults: null,
  taggerTrainingDefaults: null,
  vaeTrainingDefaults: null,
  timestepDefaultsByArch: null,
  bundleVaeDefaultsByArch: null,
  archCapabilities: null,
  videoFrameSliderMax: null,
});

export const useStartup = () => useContext(StartupContext);

interface StartupProviderProps {
  children: ReactNode;
}

// Startup readiness poll: fast, but only until the backend answers with a
// loaded model (or the cap below is reached). This is a busy-wait for the
// backend process to come up, not a steady-state mechanism.
const READY_POLL_MS = 1000;
const READY_POLL_TIMEOUT_MS = 60000;
// Steady-state refresh of GET /models/current. Deliberately slow: the model can
// change without this page doing anything (API call, backend restart, another
// tab, an agent), and a page that never re-checks routes generation requests at
// a model that is no longer loaded. Skipped entirely while the tab is hidden;
// a tab returning to the foreground refreshes immediately instead.
const MODEL_SYNC_MS = 20000;

// Identity of a loaded model, for change detection. Deliberately narrow: these
// are the fields that change what the UI must render or where it must dispatch.
function modelIdentity(info: ModelInfo | null): string {
  if (!info) return "";
  return JSON.stringify([info.type ?? null, info.source ?? null, info.variant ?? null,
                         info.is_video === true, info.is_audio === true]);
}

export function StartupProvider({ children }: StartupProviderProps) {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [hasShownAlert, setHasShownAlert] = useState(false);
  const [generationDefaults, setGenerationDefaults] = useState<GenerationDefaultsResponse | null>(null);
  const [trainingDefaults, setTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [taggerTrainingDefaults, setTaggerTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [vaeTrainingDefaults, setVaeTrainingDefaults] = useState<Record<string, unknown> | null>(null);
  const [timestepDefaultsByArch, setTimestepDefaultsByArch] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [bundleVaeDefaultsByArch, setBundleVaeDefaultsByArch] = useState<Record<string, boolean> | null>(null);
  const [archCapabilities, setArchCapabilities] = useState<ArchCapabilities | null>(null);
  const [videoFrameSliderMax, setVideoFrameSliderMax] = useState<number | null>(null);

  const [modelInfoVersion, setModelInfoVersion] = useState(0);
  // Last known info, kept in a ref so a failed refresh can fall back to it
  // without making every caller depend on render timing.
  const modelInfoRef = useRef<ModelInfo | null>(null);
  const modelIdentityRef = useRef<string>("");
  // Startup payload fetch state. A ref, not component state, because it guards
  // an in-flight request: two callers in the same tick must share one fetch.
  const payloadsLoadedRef = useRef(false);
  const payloadsInFlightRef = useRef<Promise<void> | null>(null);

  // Fetch the backend-owned startup payloads (param defaults + capability
  // matrix). Idempotent and re-runnable: a call after success is a no-op, and
  // concurrent calls share one in-flight request, so it is safe to invoke from
  // every place that learns the backend is up.
  const fetchStartupPayloads = useCallback(async (): Promise<void> => {
    if (payloadsLoadedRef.current) return;
    if (payloadsInFlightRef.current) return payloadsInFlightRef.current;

    const inFlight = (async () => {
      try {
        const [genDef, trainDef, taggerDef, vaeDef, tsByArch, bvByArch, archCaps, genSettings] = await Promise.all([
          fetchGenerationDefaults(),
          fetchTrainingDefaults(),
          fetchTaggerTrainingDefaults(),
          fetchVaeTrainingDefaults(),
          fetchTimestepDefaultsByArch(),
          fetchBundleVaeDefaultsByArch(),
          fetchArchCapabilities(),
          fetchGenerationSettings(),
        ]);
        setGenerationDefaults(genDef);
        setTrainingDefaults(trainDef);
        setTaggerTrainingDefaults(taggerDef);
        setVaeTrainingDefaults(vaeDef);
        setTimestepDefaultsByArch(tsByArch);
        setBundleVaeDefaultsByArch(bvByArch);
        setArchCapabilities(archCaps);
        setVideoFrameSliderMax(genSettings.video_frame_slider_max ?? null);
        payloadsLoadedRef.current = true;
        console.log("[StartupContext] Param defaults + arch capabilities loaded from backend");
      } catch (e) {
        // Left unset so a later attempt retries; consumers fall back meanwhile.
        console.warn("[StartupContext] Failed to fetch param defaults, using hardcoded fallbacks", e);
      } finally {
        payloadsInFlightRef.current = null;
      }
    })();

    payloadsInFlightRef.current = inFlight;
    return inFlight;
  }, []);

  // Single fetch path for GET /models/current. Every caller (startup poll,
  // background sync, ModelLoadSection, dispatch-time modality check) goes
  // through this so the shared state can never disagree with itself.
  const syncModelInfo = useCallback(async (): Promise<{ ok: boolean; modelInfo: ModelInfo | null }> => {
    try {
      // Via the shared client (baseURL "/api/v1"): the unversioned
      // "/api/models/current" answers with a 308 to the versioned path, so a
      // raw fetch of it costs two round trips per sync.
      const data = await getCurrentModel();
      const info: ModelInfo | null = data.loaded ? ((data.model_info as ModelInfo) ?? null) : null;

      setIsBackendReady(true);
      setModelLoaded(data.loaded === true);
      setModelInfo(info);
      modelInfoRef.current = info;

      const identity = modelIdentity(info);
      if (identity !== modelIdentityRef.current) {
        modelIdentityRef.current = identity;
        // Bump only on a real change, so consumers keyed on this do not
        // re-fetch every background tick.
        setModelInfoVersion((v) => v + 1);
      }

      void fetchStartupPayloads();
      return { ok: true, modelInfo: info };
    } catch (error) {
      // Backend down / restarting: keep the last known info rather than
      // claiming nothing is loaded.
      console.warn("[StartupContext] /models/current failed; keeping last known model info", error);
      return { ok: false, modelInfo: modelInfoRef.current };
    }
  }, [fetchStartupPayloads]);

  // Re-fetch the currently loaded model info. Panels call this after a
  // (re)load so modelInfo/isVideo stay the single source of truth.
  const refreshModelInfo = useCallback(async (): Promise<ModelInfo | null> => {
    const { modelInfo: info } = await syncModelInfo();
    return info;
  }, [syncModelInfo]);

  // Dispatch-time modality check. Panels call this immediately before choosing
  // between the image / video / audio endpoint: a cached flag can be wrong for
  // as long as the sync interval, and a wrong route is a 400 with a confusing
  // message, whereas this check is one cheap request.
  const resolveModality = useCallback(async () => {
    const { ok, modelInfo: info } = await syncModelInfo();
    return {
      isVideo: info?.is_video === true,
      isAudio: info?.is_audio === true,
      fresh: ok,
      // The whole record, so a caller that needs more than the modality (e.g.
      // MiniMax-H3's fl2va/ref2va `variant`) reads it from the same fresh
      // fetch rather than from its own possibly-lagging copy.
      modelInfo: info,
    };
  }, [syncModelInfo]);

  // Startup readiness poll. StrictMode-safe: the only guard is per-effect-run
  // (`cancelled`), so the dev-mode mount/unmount/mount cycle re-arms the poll
  // instead of latching a module-level flag that outlives the interval it was
  // guarding -- which is what previously left modelInfo/archCapabilities/all
  // the *Defaults null for the entire life of the page.
  useEffect(() => {
    let cancelled = false;
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    const stopPolling = () => {
      if (pollInterval !== null) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    };

    const tick = async () => {
      if (cancelled) return;
      const { ok, modelInfo: info } = await syncModelInfo();
      if (cancelled) return;
      // Stop the fast poll once the backend answers AND has a model loaded;
      // steady state is the slow sync effect below.
      if (ok && info) stopPolling();
    };

    void tick(); // first attempt immediately, not after READY_POLL_MS
    pollInterval = setInterval(() => { void tick(); }, READY_POLL_MS);
    const timeout = setTimeout(stopPolling, READY_POLL_TIMEOUT_MS);

    return () => {
      cancelled = true;
      stopPolling();
      clearTimeout(timeout);
    };
  }, [syncModelInfo]);

  // Steady-state sync: the loaded model can change without this page acting
  // (API call, backend restart, second tab, agent). Slow, and suspended while
  // the tab is hidden; a tab regaining focus refreshes at once.
  useEffect(() => {
    const syncIfVisible = () => {
      if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
      void syncModelInfo();
    };

    const interval = setInterval(syncIfVisible, MODEL_SYNC_MS);
    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") void syncModelInfo();
    };
    window.addEventListener("focus", syncIfVisible);
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      clearInterval(interval);
      window.removeEventListener("focus", syncIfVisible);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [syncModelInfo]);

  return (
    <StartupContext.Provider value={{
      isBackendReady,
      modelLoaded,
      modelInfo,
      isVideo: modelInfo?.is_video === true,
      isAudio: modelInfo?.is_audio === true,
      refreshModelInfo,
      resolveModality,
      modelInfoVersion,
      generationDefaults,
      trainingDefaults,
      taggerTrainingDefaults,
      vaeTrainingDefaults,
      timestepDefaultsByArch,
      bundleVaeDefaultsByArch,
      archCapabilities,
      videoFrameSliderMax,
    }}>
      {children}
    </StartupContext.Provider>
  );
}
