"use client";

import { useState, useEffect, useRef } from "react";
import Sidebar from "@/components/common/Sidebar";
import Card from "@/components/common/Card";
import Button from "@/components/common/Button";
import DirectorySettings from "@/components/settings/DirectorySettings";
import GenerationSettings from "@/components/settings/GenerationSettings";
import QuantizedGemmSettings from "@/components/settings/QuantizedGemmSettings";
import ProtectedRoute from "@/components/common/ProtectedRoute";
import { restartBackend, restartFrontend, restartBoth, saveVideoFrameSliderMax, saveSliderBounds } from "@/utils/api";
import NumberInput from "@/components/common/NumberInput";
import { useStartup } from "@/contexts/StartupContext";
import { isAboveBuiltin } from "@/utils/paramBounds";
import {
  readGlobalAttentionImpl,
  readGlobalAttentionType,
  type AttentionImplementation,
  type InferenceAttentionType,
} from "@/utils/attentionSettings";

// Default presets
const DEFAULT_ASPECT_RATIO_PRESETS = [
  { label: "1:1", ratio: 1 / 1 },
  { label: "4:3", ratio: 4 / 3 },
  { label: "3:4", ratio: 3 / 4 },
  { label: "16:9", ratio: 16 / 9 },
  { label: "9:16", ratio: 9 / 16 },
  { label: "21:9", ratio: 21 / 9 },
  { label: "9:21", ratio: 9 / 21 },
  { label: "3:2", ratio: 3 / 2 },
  { label: "2:3", ratio: 2 / 3 },
  { label: "5:4", ratio: 5 / 4 },
];

const DEFAULT_FIXED_RESOLUTION_PRESETS = [
  { width: 768, height: 1152 },
  { width: 1152, height: 768 },
  { width: 1248, height: 720 },
  { width: 720, height: 1248 },
  { width: 960, height: 1344 },
  { width: 1344, height: 960 },
  { width: 1024, height: 1152 },
  { width: 1152, height: 1024 },
  { width: 1024, height: 1024 },
  { width: 896, height: 1152 },
  { width: 1152, height: 896 },
  { width: 832, height: 1216 },
  { width: 1216, height: 832 },
  { width: 640, height: 1536 },
  { width: 1536, height: 640 },
  { width: 512, height: 512 },
];

export default function SettingsPage() {
  // videoFrameSliderMax: the live value panels read (StartupContext,
  // sourced from GET /settings/generation at startup).
  // setVideoFrameSliderMax: the context setter this page calls right after a
  // successful write, so the new bound applies without a reload (the fix for
  // "saving this setting did not apply it").
  // generationDefaults: source of the checkbox's seed value (see
  // videoFrameSliderMaxSeed below) -- never a bare literal per param_defaults.py.
  // sliderBounds/setSliderBounds: the general PARAM_BOUNDS override
  // mechanism (backend UserSettings.slider_bounds) -- see paramBounds.ts's
  // resolveBound() for how panel controls consume it, and the "Slider
  // Bounds" card below for where it is edited.
  const {
    videoFrameSliderMax: liveVideoFrameSliderMax,
    setVideoFrameSliderMax: setLiveVideoFrameSliderMax,
    sliderBounds: liveSliderBounds,
    setSliderBounds: setLiveSliderBounds,
    generationDefaults,
  } = useStartup();
  // backend/api/param_defaults.py PARAM_BOUNDS, served via
  // GET /schema/generation-defaults's `param_bounds` field. {} until fetched
  // -- the card below simply renders no rows until then.
  const paramBounds = generationDefaults?.param_bounds ?? {};

  const [isRestarting, setIsRestarting] = useState(false);
  const [storageInfo, setStorageInfo] = useState({ used: 0, quota: 0 });
  const [restoreOnCancel, setRestoreOnCancel] = useState(false);
  const [resolutionStep, setResolutionStep] = useState(64);
  const [aspectRatioPresets, setAspectRatioPresets] = useState(DEFAULT_ASPECT_RATIO_PRESETS);
  const [fixedResolutionPresets, setFixedResolutionPresets] = useState(DEFAULT_FIXED_RESOLUTION_PRESETS);
  const [includeMetadataInDownloads, setIncludeMetadataInDownloads] = useState(false);

  // Tag suggestion / floating gallery settings
  const [tagSuggestionMinCount, setTagSuggestionMinCount] = useState(50);
  const [floatingGalleryMaxImages, setFloatingGalleryMaxImages] = useState(30);

  // Send size mode settings
  const [sendSizeMode, setSendSizeMode] = useState<"absolute" | "scale">("absolute");
  const [sendDefaultScale, setSendDefaultScale] = useState(1.0);

  // Developer mode
  const [developerMode, setDeveloperMode] = useState(false);

  // Advanced CFG settings visibility
  const [showAdvancedCFG, setShowAdvancedCFG] = useState(false);

  // Attention type
  const [attentionType, setAttentionType] = useState<InferenceAttentionType>("normal");
  const [attentionImpl, setAttentionImpl] = useState<AttentionImplementation>("conduit");

  // Video frame-count slider track max (server-persisted UserSettings row,
  // GET/POST /settings/generation) -- unlike the other controls in this card
  // it is NOT localStorage; it commits to the backend on NumberInput's
  // onCommit / the checkbox's onChange, same trigger points as its
  // localStorage-backed siblings, just backed by a network write instead.
  // `null` = unset = the slider's own built-in track reach; bounds the
  // TRACK only, never the paired number box.
  const [videoFrameSliderMaxEnabled, setVideoFrameSliderMaxEnabled] = useState(false);
  const [videoFrameSliderMaxValue, setVideoFrameSliderMaxValue] = useState(241);
  const [videoFrameSliderMaxSaving, setVideoFrameSliderMaxSaving] = useState(false);
  const [videoFrameSliderMaxMessage, setVideoFrameSliderMaxMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  // backend/api/param_defaults.py VIDEO_GEN_DEFAULTS.video_frame_slider_max_seed
  // (fetched via generationDefaults.txt2vid, which resolves it like every
  // other video-only key). 241 mirrors that same value as the pre-fetch
  // fallback, per this repo's convention for DEFAULT_PARAMS-style fallbacks
  // (see param_defaults.py's comment on that constant for why 241 specifically).
  const videoFrameSliderMaxSeed =
    (generationDefaults?.txt2vid?.video_frame_slider_max_seed as number | undefined) ?? 241;

  // Mirror the live context value into local editing state. Runs on the
  // initial startup fetch AND after this page's own successful write (the
  // context echoes the saved value back), so it never fights an edit that is
  // still in flight -- it is not consulted again until the next write.
  useEffect(() => {
    setVideoFrameSliderMaxEnabled(liveVideoFrameSliderMax != null);
    if (liveVideoFrameSliderMax != null) {
      setVideoFrameSliderMaxValue(liveVideoFrameSliderMax);
    }
  }, [liveVideoFrameSliderMax]);

  // NumberInput's onCommit fires on every keystroke that parses to a valid
  // number (see NumberInput.tsx's onChange), not only on blur -- so wiring
  // the POST straight to onCommit would write on every digit typed. This
  // debounces the actual network write to once per pause in typing (plus the
  // final settle on blur, which re-fires onCommit with the same value and so
  // collapses into the same pending timer): the *local* value still updates
  // immediately for a responsive field, only the backend round trip waits.
  const videoFrameSliderMaxDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // The value a pending timer is going to write, kept so unmount can FLUSH it
  // rather than drop it. Cancelling the timer on unmount would mean a value
  // typed and then navigated away from within the debounce window is silently
  // never saved -- which the help text ("Applies immediately and is held on
  // the server") would then be lying about. The flush is fire-and-forget: any
  // state it would set belongs to a page that is going away, and the write
  // itself is what matters.
  const videoFrameSliderMaxPendingRef = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (videoFrameSliderMaxDebounceRef.current) {
        clearTimeout(videoFrameSliderMaxDebounceRef.current);
        videoFrameSliderMaxDebounceRef.current = null;
        const pending = videoFrameSliderMaxPendingRef.current;
        if (pending != null) {
          void saveVideoFrameSliderMax(pending).catch((error) => {
            console.error("Failed to flush video frame slider max on unmount:", error);
          });
        }
      }
    };
  }, []);

  // Commit-time write (checkbox onChange directly; NumberInput onCommit via
  // the debounce above) -- never per keystroke. On success, updates both
  // this page's local state and the live StartupContext value, so panels see
  // the new bound immediately. On failure, reverts the local UI to the last
  // known-good (live) value and surfaces an error, so the user is never left
  // believing an unsaved value took effect -- same honesty contract
  // QuantizedGemmSettings.tsx uses for its own backend-persisted toggles
  // (show the error, then reload actual state instead of trusting the
  // optimistic local edit).
  const commitVideoFrameSliderMax = async (value: number | null) => {
    setVideoFrameSliderMaxSaving(true);
    setVideoFrameSliderMaxMessage(null);
    try {
      const saved = await saveVideoFrameSliderMax(value);
      setLiveVideoFrameSliderMax(saved.video_frame_slider_max ?? null);
      setVideoFrameSliderMaxEnabled(saved.video_frame_slider_max != null);
      if (saved.video_frame_slider_max != null) {
        setVideoFrameSliderMaxValue(saved.video_frame_slider_max);
      }
    } catch (error) {
      console.error("Failed to save video frame slider max:", error);
      setVideoFrameSliderMaxMessage({
        type: "error",
        text: "Failed to save the video frame-count slider track max. The previous value is still in effect; please check the console and try again.",
      });
      // Revert to the last known-good value rather than leave the just-typed
      // value looking applied when it was not saved.
      setVideoFrameSliderMaxEnabled(liveVideoFrameSliderMax != null);
      setVideoFrameSliderMaxValue(liveVideoFrameSliderMax ?? videoFrameSliderMaxSeed);
    } finally {
      setVideoFrameSliderMaxSaving(false);
    }
  };

  // NumberInput onCommit target: updates the field immediately, defers the
  // actual save (see the debounce comment above `commitVideoFrameSliderMax`).
  const handleVideoFrameSliderMaxNumberCommit = (v: number) => {
    setVideoFrameSliderMaxValue(v);
    if (videoFrameSliderMaxDebounceRef.current) {
      clearTimeout(videoFrameSliderMaxDebounceRef.current);
    }
    videoFrameSliderMaxPendingRef.current = v;
    videoFrameSliderMaxDebounceRef.current = setTimeout(() => {
      videoFrameSliderMaxDebounceRef.current = null;
      videoFrameSliderMaxPendingRef.current = null;
      void commitVideoFrameSliderMax(v);
    }, 600);
  };

  // ---------------------------------------------------------------------
  // Slider Bounds card: one row per PARAM_BOUNDS registry entry
  // (backend/api/param_defaults.py). Generic over `boundName` rather than a
  // state variable per bound, so a new registry entry needs no new state
  // here -- it just appears as a row (see "Adding a new overridable bound"
  // in PARAM_BOUNDS's own docstring). Same commit-time-write / debounce /
  // revert-on-failure contract as the video_frame_slider_max control above,
  // generalized to a per-row map instead of one set of variables each.
  // ---------------------------------------------------------------------
  const [sliderBoundEnabled, setSliderBoundEnabled] = useState<Record<string, boolean>>({});
  const [sliderBoundValue, setSliderBoundValue] = useState<Record<string, number>>({});
  const [sliderBoundSaving, setSliderBoundSaving] = useState<Record<string, boolean>>({});
  const [sliderBoundsMessage, setSliderBoundsMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Mirror the live override map into per-row local editing state whenever
  // it changes (initial fetch, this page's own successful write, or a
  // fresh registry arriving) -- same "mirror the context value" pattern as
  // the video_frame_slider_max useEffect above.
  useEffect(() => {
    const nextEnabled: Record<string, boolean> = {};
    const nextValue: Record<string, number> = {};
    for (const boundName of Object.keys(paramBounds)) {
      const overridden = liveSliderBounds[boundName];
      nextEnabled[boundName] = overridden != null;
      nextValue[boundName] = overridden ?? paramBounds[boundName].builtin;
    }
    setSliderBoundEnabled(nextEnabled);
    setSliderBoundValue(nextValue);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveSliderBounds, generationDefaults]);

  // Per-bound debounce timers/pending values, keyed the same way as the
  // enabled/value state above. Flushed on unmount so a value typed and then
  // navigated away from within the debounce window is not silently dropped
  // (same reasoning as the video_frame_slider_max flush effect).
  const sliderBoundDebounceRef = useRef<Record<string, ReturnType<typeof setTimeout> | null>>({});
  const sliderBoundPendingRef = useRef<Record<string, number | null>>({});
  useEffect(() => {
    return () => {
      const timers = sliderBoundDebounceRef.current;
      for (const boundName of Object.keys(timers)) {
        const timer = timers[boundName];
        if (timer) {
          clearTimeout(timer);
          const pending = sliderBoundPendingRef.current[boundName];
          if (pending != null) {
            void saveSliderBounds({ [boundName]: pending }).catch((error) => {
              console.error(`Failed to flush slider bound ${boundName} on unmount:`, error);
            });
          }
        }
      }
    };
  }, []);

  const commitSliderBound = async (boundName: string, value: number | null) => {
    setSliderBoundSaving((prev) => ({ ...prev, [boundName]: true }));
    setSliderBoundsMessage(null);
    try {
      const saved = await saveSliderBounds({ [boundName]: value });
      setLiveSliderBounds(saved.slider_bounds ?? {});
    } catch (error) {
      console.error(`Failed to save slider bound ${boundName}:`, error);
      const label = paramBounds[boundName]?.label ?? boundName;
      setSliderBoundsMessage({
        type: "error",
        text: `Failed to save the "${label}" bound. The previous value is still in effect; please check the console and try again.`,
      });
      // Revert this one row to the last known-good (live) value, same
      // honesty contract as commitVideoFrameSliderMax's own catch block.
      setSliderBoundEnabled((prev) => ({ ...prev, [boundName]: liveSliderBounds[boundName] != null }));
      setSliderBoundValue((prev) => ({
        ...prev,
        [boundName]: liveSliderBounds[boundName] ?? paramBounds[boundName]?.builtin ?? prev[boundName],
      }));
    } finally {
      setSliderBoundSaving((prev) => ({ ...prev, [boundName]: false }));
    }
  };

  const handleSliderBoundNumberCommit = (boundName: string, v: number) => {
    setSliderBoundValue((prev) => ({ ...prev, [boundName]: v }));
    const timers = sliderBoundDebounceRef.current;
    if (timers[boundName]) clearTimeout(timers[boundName]!);
    sliderBoundPendingRef.current[boundName] = v;
    timers[boundName] = setTimeout(() => {
      timers[boundName] = null;
      sliderBoundPendingRef.current[boundName] = null;
      void commitSliderBound(boundName, v);
    }, 600);
  };

  // The per-row checkbox IS the per-item reset (unchecking commits `null`,
  // which the backend removes from the stored map -- see
  // save_generation_settings's slider_bounds handling). This is the one
  // "reset all" action: clears every currently-set override in a single
  // request rather than one round trip per row.
  const resetAllSliderBounds = async () => {
    const overrides: Record<string, number | null> = {};
    for (const boundName of Object.keys(liveSliderBounds)) {
      overrides[boundName] = null;
    }
    if (Object.keys(overrides).length === 0) return;
    setSliderBoundsMessage(null);
    try {
      const saved = await saveSliderBounds(overrides);
      setLiveSliderBounds(saved.slider_bounds ?? {});
    } catch (error) {
      console.error("Failed to reset slider bounds:", error);
      setSliderBoundsMessage({
        type: "error",
        text: "Failed to reset slider bounds. The previous overrides are still in effect; please check the console and try again.",
      });
    }
  };

  const SLIDER_BOUND_FAMILY_LABELS: Record<string, string> = {
    canvas: "Canvas",
    sampling: "Sampling",
    video: "Video",
    upscale: "Upscale",
  };
  const sliderBoundFamilies = Array.from(new Set(Object.values(paramBounds).map((spec) => spec.family)));

  // Font size (mobile UI scaling)
  const [fontSize, setFontSize] = useState(100); // 100 = 100% (default)

  // Panel visibility settings
  const [txt2imgVisibility, setTxt2imgVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });
  const [img2imgVisibility, setImg2imgVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });
  const [inpaintVisibility, setInpaintVisibility] = useState({
    lora: true,
    controlnet: true,
    aspectRatioPresets: true,
    fixedResolutionPresets: true,
  });

  const updateStorageInfo = () => {
    if (typeof window !== 'undefined' && 'storage' in navigator && 'estimate' in navigator.storage) {
      navigator.storage.estimate().then(estimate => {
        setStorageInfo({
          used: estimate.usage || 0,
          quota: estimate.quota || 0,
        });
      });
    }
  };

  const handleClearLocalStorage = () => {
    if (!confirm("Are you sure you want to clear all localStorage data? This will reset all saved settings, images, and panel states.")) {
      return;
    }

    try {
      localStorage.clear();
      alert("localStorage cleared successfully! The page will reload.");
      window.location.reload();
    } catch (error) {
      console.error("Failed to clear localStorage:", error);
      alert("Failed to clear localStorage. Please check the console.");
    }
  };

  const handleClearTempImages = async () => {
    if (!confirm("Are you sure you want to clear all temporary images? This will remove all saved input images and ControlNet references.")) {
      return;
    }

    try {
      const { cleanupTempImages } = await import("@/utils/api");
      const deletedCount = await cleanupTempImages(0); // Delete all images (max age 0 hours)
      alert(`Successfully deleted ${deletedCount} temporary images.`);
      updateStorageInfo();
    } catch (error) {
      console.error("Failed to clear temp images:", error);
      alert("Failed to clear temp images. Please check the console.");
    }
  };

  const handleRestartBackend = async () => {
    if (!confirm("Are you sure you want to restart the backend server?")) {
      return;
    }

    setIsRestarting(true);
    try {
      const result = await restartBackend();
      console.log("Backend restart response:", result);
      alert("Backend restart scheduled. The backend will restart in a moment. You may need to refresh the page in a few seconds.");
    } catch (error: any) {
      console.error("Failed to restart backend:", error);
      console.error("Error details:", error.response?.data);
      const errorMsg = error.response?.data?.detail || error.message || "Unknown error";
      alert(`Failed to restart backend: ${errorMsg}\n\nPlease check the backend console for details.`);
    } finally {
      // Keep the button disabled for a few seconds
      setTimeout(() => {
        setIsRestarting(false);
      }, 5000);
    }
  };

  const handleRestartFrontend = () => {
    if (!confirm("Are you sure you want to restart the frontend? The page will reload.")) {
      return;
    }

    restartFrontend();
  };

  const handleRestartBoth = async () => {
    if (!confirm("Are you sure you want to restart both servers? The page will reload after backend restarts.")) {
      return;
    }

    setIsRestarting(true);
    try {
      await restartBoth();
    } catch (error) {
      console.error("Failed to restart servers:", error);
      alert("Failed to restart servers. Please check the console.");
      setIsRestarting(false);
    }
  };

  useEffect(() => {
    updateStorageInfo();
    // Load settings from localStorage
    if (typeof window !== 'undefined') {
      setRestoreOnCancel(localStorage.getItem('restore_image_on_cancel') === 'true');
      setIncludeMetadataInDownloads(localStorage.getItem('include_metadata_in_downloads') === 'true');

      const savedTagMinCount = localStorage.getItem('tag_suggestion_min_count');
      if (savedTagMinCount !== null) {
        setTagSuggestionMinCount(parseInt(savedTagMinCount));
      }

      const savedGalleryMaxImages = localStorage.getItem('floating_gallery_max_images');
      if (savedGalleryMaxImages !== null) {
        setFloatingGalleryMaxImages(parseInt(savedGalleryMaxImages));
      }

      const savedAttentionType = readGlobalAttentionType();
      if (savedAttentionType) {
        setAttentionType(savedAttentionType);
      }

      const savedAttentionImpl = readGlobalAttentionImpl();
      if (savedAttentionImpl) {
        setAttentionImpl(savedAttentionImpl);
      }

      const savedResolutionStep = localStorage.getItem('resolution_step');
      if (savedResolutionStep) {
        setResolutionStep(parseInt(savedResolutionStep));
      }

      // Load custom presets
      const savedAspectRatioPresets = localStorage.getItem('aspect_ratio_presets');
      if (savedAspectRatioPresets) {
        try {
          setAspectRatioPresets(JSON.parse(savedAspectRatioPresets));
        } catch (e) {
          console.error('Failed to parse aspect ratio presets:', e);
        }
      }

      const savedFixedResolutionPresets = localStorage.getItem('fixed_resolution_presets');
      if (savedFixedResolutionPresets) {
        try {
          setFixedResolutionPresets(JSON.parse(savedFixedResolutionPresets));
        } catch (e) {
          console.error('Failed to parse fixed resolution presets:', e);
        }
      }

      // Load panel visibility settings
      const savedTxt2imgVisibility = localStorage.getItem('txt2img_visibility');
      if (savedTxt2imgVisibility) {
        try {
          setTxt2imgVisibility(JSON.parse(savedTxt2imgVisibility));
        } catch (e) {
          console.error('Failed to parse txt2img visibility:', e);
        }
      }

      const savedImg2imgVisibility = localStorage.getItem('img2img_visibility');
      if (savedImg2imgVisibility) {
        try {
          setImg2imgVisibility(JSON.parse(savedImg2imgVisibility));
        } catch (e) {
          console.error('Failed to parse img2img visibility:', e);
        }
      }

      const savedInpaintVisibility = localStorage.getItem('inpaint_visibility');
      if (savedInpaintVisibility) {
        try {
          setInpaintVisibility(JSON.parse(savedInpaintVisibility));
        } catch (e) {
          console.error('Failed to parse inpaint visibility:', e);
        }
      }

      // Load send size mode settings
      const savedSendSizeMode = localStorage.getItem('send_size_mode');
      if (savedSendSizeMode && (savedSendSizeMode === 'absolute' || savedSendSizeMode === 'scale')) {
        setSendSizeMode(savedSendSizeMode);
      }

      const savedSendDefaultScale = localStorage.getItem('send_default_scale');
      if (savedSendDefaultScale) {
        const scale = parseFloat(savedSendDefaultScale);
        if (!isNaN(scale) && scale > 0) {
          setSendDefaultScale(scale);
        }
      }

      // Load developer mode setting
      const savedDeveloperMode = localStorage.getItem('developer_mode');
      if (savedDeveloperMode === 'true') {
        setDeveloperMode(true);
      }

      // Load font size setting
      const savedFontSize = localStorage.getItem('ui_font_size');
      if (savedFontSize) {
        const size = parseInt(savedFontSize);
        if (!isNaN(size) && size >= 50 && size <= 200) {
          setFontSize(size);
          document.documentElement.style.setProperty('--ui-font-size', `${size}%`);
        }
      }

      // Load advanced CFG settings visibility
      const savedShowAdvancedCFG = localStorage.getItem('show_advanced_cfg');
      if (savedShowAdvancedCFG === 'true') {
        setShowAdvancedCFG(true);
      }
    }
  }, []);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Aspect ratio preset management
  const handleAddAspectRatioPreset = () => {
    const label = prompt("Enter aspect ratio label (e.g., '16:9'):");
    if (!label) return;

    const ratioStr = prompt("Enter aspect ratio as width:height (e.g., '16:9'):");
    if (!ratioStr) return;

    const [w, h] = ratioStr.split(':').map(n => parseFloat(n.trim()));
    if (isNaN(w) || isNaN(h) || w <= 0 || h <= 0) {
      alert("Invalid ratio format. Please use format like '16:9'");
      return;
    }

    const newPresets = [...aspectRatioPresets, { label, ratio: w / h }];
    setAspectRatioPresets(newPresets);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(newPresets));
  };

  const handleRemoveAspectRatioPreset = (index: number) => {
    const newPresets = aspectRatioPresets.filter((_, i) => i !== index);
    setAspectRatioPresets(newPresets);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(newPresets));
  };

  const handleRestoreAspectRatioDefaults = () => {
    if (!confirm("Restore default aspect ratio presets?")) return;
    setAspectRatioPresets(DEFAULT_ASPECT_RATIO_PRESETS);
    localStorage.setItem('aspect_ratio_presets', JSON.stringify(DEFAULT_ASPECT_RATIO_PRESETS));
  };

  // Fixed resolution preset management
  const handleAddFixedResolutionPreset = () => {
    const widthStr = prompt("Enter width (must be multiple of 8):");
    if (!widthStr) return;
    let width = parseInt(widthStr);
    if (isNaN(width) || width < 8) {
      alert("Invalid width");
      return;
    }
    // Round to nearest multiple of 8
    width = Math.round(width / 8) * 8;

    const heightStr = prompt("Enter height (must be multiple of 8):");
    if (!heightStr) return;
    let height = parseInt(heightStr);
    if (isNaN(height) || height < 8) {
      alert("Invalid height");
      return;
    }
    // Round to nearest multiple of 8
    height = Math.round(height / 8) * 8;

    const newPresets = [...fixedResolutionPresets, { width, height }];
    setFixedResolutionPresets(newPresets);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(newPresets));
  };

  const handleRemoveFixedResolutionPreset = (index: number) => {
    const newPresets = fixedResolutionPresets.filter((_, i) => i !== index);
    setFixedResolutionPresets(newPresets);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(newPresets));
  };

  const handleRestoreFixedResolutionDefaults = () => {
    if (!confirm("Restore default fixed resolution presets?")) return;
    setFixedResolutionPresets(DEFAULT_FIXED_RESOLUTION_PRESETS);
    localStorage.setItem('fixed_resolution_presets', JSON.stringify(DEFAULT_FIXED_RESOLUTION_PRESETS));
  };

  return (
    <ProtectedRoute>
      <div className="app-shell">
        <Sidebar />
        <main className="app-main compact-workspace flex flex-col overflow-hidden">
          <header className="app-topbar">
            <div>
              <p className="app-kicker">System</p>
              <h1 className="app-title">Settings</h1>
            </div>
          </header>

          <div className="app-content flex-1 overflow-auto">
            <div className="columns-1 gap-3 xl:columns-2 [&>section]:mb-3 [&>section]:break-inside-avoid">
          <Card title="Server Control">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Restart the backend or frontend servers without manually stopping them.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Button
                  onClick={handleRestartBackend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Backend"}
                </Button>

                <Button
                  onClick={handleRestartFrontend}
                  disabled={isRestarting}
                  variant="secondary"
                  className="w-full"
                >
                  Restart Frontend
                </Button>

                <Button
                  onClick={handleRestartBoth}
                  disabled={isRestarting}
                  className="w-full"
                >
                  {isRestarting ? "Restarting..." : "Restart Both"}
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">Notes:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>Backend:</strong> Restarts the Python FastAPI server. Use this after code changes in backend/.</li>
                  <li><strong>Frontend:</strong> Reloads the page. Use this to refresh the UI state.</li>
                  <li><strong>Both:</strong> Restarts backend first, then reloads the page after 2 seconds.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Storage Management">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Manage browser storage and temporary files to free up space.
              </p>

              {storageInfo.quota > 0 && (
                <div className="p-4 bg-gray-800 rounded-lg mb-4">
                  <h3 className="text-sm font-semibold mb-2">Storage Usage</h3>
                  <div className="space-y-2 text-sm text-gray-400">
                    <div className="flex justify-between">
                      <span>Used:</span>
                      <span className="font-mono">{formatBytes(storageInfo.used)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Quota:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Available:</span>
                      <span className="font-mono">{formatBytes(storageInfo.quota - storageInfo.used)}</span>
                    </div>
                    <div className="mt-2 bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full"
                        style={{ width: `${(storageInfo.used / storageInfo.quota) * 100}%` }}
                      />
                    </div>
                    <div className="text-xs text-center text-gray-500">
                      {((storageInfo.used / storageInfo.quota) * 100).toFixed(1)}% used
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button
                  onClick={handleClearLocalStorage}
                  variant="secondary"
                  className="w-full"
                >
                  Clear localStorage
                </Button>

                <Button
                  onClick={handleClearTempImages}
                  variant="secondary"
                  className="w-full"
                >
                  Clear Temp Images
                </Button>
              </div>

              <div className="mt-4 p-4 bg-gray-800 rounded-lg">
                <h3 className="text-sm font-semibold mb-2">What gets cleared:</h3>
                <ul className="text-sm text-gray-400 space-y-1 list-disc list-inside">
                  <li><strong>localStorage:</strong> All saved settings, prompts, parameters, and image references. The page will reload after clearing.</li>
                  <li><strong>Temp Images:</strong> All temporary images stored on the server (input images, ControlNet references). References in localStorage will become invalid.</li>
                </ul>
              </div>
            </div>
          </Card>

          <Card title="Model Directories">
            <DirectorySettings />
          </Card>

          <Card title="Generation Settings">
            <GenerationSettings />
          </Card>

          <Card title="Quantized GEMM Paths">
            <QuantizedGemmSettings />
          </Card>

          <Card title="Tag Suggestions">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure tag autocompletion behavior in prompt fields.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Minimum Tag Count
                  </label>
                  <NumberInput
                    min={0}
                    max={10000}
                    value={tagSuggestionMinCount}
                    defaultValue={0}
                    onCommit={(v) => {
                      setTagSuggestionMinCount(v);
                      localStorage.setItem('tag_suggestion_min_count', String(v));
                    }}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Only show tags that appear at least this many times in the dataset. Lower values show more tags but may include uncommon or misspelled tags. Default: 50
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Generation Gallery">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure the floating gallery that shows recent generated images.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Maximum Images in Gallery
                  </label>
                  <NumberInput
                    min={5}
                    max={100}
                    value={floatingGalleryMaxImages}
                    defaultValue={30}
                    onCommit={(v) => {
                      setFloatingGalleryMaxImages(v);
                      localStorage.setItem('floating_gallery_max_images', String(v));
                    }}
                    className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Maximum number of recent images to keep in the floating gallery. Older images will be removed automatically. Default: 30
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Generation Behavior">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure how the UI behaves during and after generation.
              </p>

              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="restore_on_cancel"
                    checked={restoreOnCancel}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setRestoreOnCancel(newValue);
                      localStorage.setItem('restore_image_on_cancel', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="restore_on_cancel" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Restore previous image on generation cancel
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      When enabled, cancelling a generation will restore the previously completed image instead of showing the intermediate TAESD preview. Disable this if you want to see the generation progress at the point of cancellation.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <label htmlFor="resolution_step" className="text-sm font-medium text-gray-300">
                    Resolution slider step size
                  </label>
                  <div className="flex items-center space-x-4">
                    <NumberInput
                      id="resolution_step"
                      value={resolutionStep}
                      defaultValue={64}
                      onCommit={(v) => {
                        let value = v;
                        // Ensure it's a multiple of 8
                        if (value < 8) value = 8;
                        if (value % 8 !== 0) {
                          value = Math.round(value / 8) * 8;
                        }
                        setResolutionStep(value);
                        localStorage.setItem('resolution_step', value.toString());
                      }}
                      min={8}
                      step={8}
                      className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                    />
                    <span className="text-sm text-gray-400">pixels (must be multiple of 8)</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Controls the step size for width and height sliders in generation panels. Default is 64.
                  </p>
                </div>

                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="include_metadata_in_downloads"
                    checked={includeMetadataInDownloads}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setIncludeMetadataInDownloads(newValue);
                      localStorage.setItem('include_metadata_in_downloads', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="include_metadata_in_downloads" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Include metadata in manual downloads
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      When enabled, images downloaded using the download button will include generation metadata (prompt, parameters, etc.). Note: Images automatically saved to the output folder always include metadata regardless of this setting.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="developer_mode"
                    checked={developerMode}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setDeveloperMode(newValue);
                      localStorage.setItem('developer_mode', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="developer_mode" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Developer Mode
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Enable developer features including CFG metrics visualization during generation. Shows noise prediction magnitudes, guidance strength, and other diagnostic information below the preview panel.
                    </p>
                  </div>
                </div>

                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="show_advanced_cfg"
                    checked={showAdvancedCFG}
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setShowAdvancedCFG(newValue);
                      localStorage.setItem('show_advanced_cfg', newValue.toString());
                    }}
                    className="mt-1 w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <div>
                    <label htmlFor="show_advanced_cfg" className="text-sm font-medium text-gray-300 cursor-pointer">
                      Show Advanced CFG Settings
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Show advanced CFG (Classifier-Free Guidance) settings in generation panels. Includes Dynamic CFG Schedule (sigma-based, SNR-based), Dynamic Thresholding, and related parameters. When disabled, all advanced CFG features are turned off.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <label htmlFor="attention_type" className="block text-sm font-medium text-gray-300">
                    Attention Type
                  </label>
                  <select
                    id="attention_type"
                    value={attentionType}
                    onChange={(e) => {
                      const newValue = e.target.value as InferenceAttentionType;
                      setAttentionType(newValue);
                      localStorage.setItem('attention_type', newValue);
                    }}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="normal">Normal (PyTorch SDPA + Auto Flash Attention)</option>
                    <option value="sage">SageAttention (2-5x faster, quantized)</option>
                    <option value="flash">FlashAttention (explicit FA2)</option>
                    <option value="tq">TQ (Triton-Quantized)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Choose attention acceleration method. <strong>Normal</strong>: PyTorch 2.0+ automatically uses Flash Attention when available. <strong>SageAttention</strong>: INT8 quantized attention for 2-5x speedup (requires <code>pip install sageattention</code>). <strong>FlashAttention</strong>: Explicit Flash Attention 2 (requires <code>pip install flash-attn</code>). <strong>TQ</strong>: Triton-Quantized attention; applies to Z-Image, Lens, MiniT2I, Anima, and SDXL inference. Other architectures fall back to native. Changes take effect immediately on next generation.
                  </p>
                </div>

                <div>
                  <label htmlFor="attention_impl" className="block text-sm font-medium text-gray-300">
                    Attention Implementation (FLUX.2)
                  </label>
                  <select
                    id="attention_impl"
                    value={attentionImpl}
                    onChange={(e) => {
                      const newValue = e.target.value as AttentionImplementation;
                      setAttentionImpl(newValue);
                      localStorage.setItem('attention_impl', newValue);
                    }}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="conduit">Conduit (unified dispatch; enables TQ on FLUX.2)</option>
                    <option value="diffusers">Diffusers (legacy registry; byte-identical fallback)</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Selects which implementation runs the FLUX.2 attention kernel. <strong>Conduit</strong> routes through SushiUI&apos;s unified dispatch so the Attention Type above (including TQ) applies to FLUX.2. <strong>Diffusers</strong> keeps diffusers&apos; own registry (reproduces the previous behavior). Native output is identical either way; other architectures ignore this.
                  </p>
                </div>

              </div>
            </div>
          </Card>

          <Card title="Slider Bounds">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Raises the slider/number-input range for the settings below;
                does not change model or hardware limits.
              </p>

              {sliderBoundsMessage && (
                <div className={`p-3 rounded text-sm ${sliderBoundsMessage.type === "success" ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"}`}>
                  {sliderBoundsMessage.text}
                </div>
              )}

              <div className="flex justify-end">
                <Button onClick={() => void resetAllSliderBounds()} variant="secondary" size="sm">
                  Reset All
                </Button>
              </div>

              <div className="space-y-6">
                {sliderBoundFamilies.map((family) => (
                  <div key={family} className="space-y-4">
                    <h4 className="text-sm font-semibold text-gray-300 border-b border-gray-700 pb-1">
                      {SLIDER_BOUND_FAMILY_LABELS[family] ?? family}
                    </h4>
                    {Object.entries(paramBounds)
                      .filter(([, spec]) => spec.family === family)
                      .map(([boundName, spec]) => {
                        const enabled = sliderBoundEnabled[boundName] ?? false;
                        const value = sliderBoundValue[boundName] ?? spec.builtin;
                        const saving = sliderBoundSaving[boundName] ?? false;
                        return (
                          <div key={boundName} className="space-y-2">
                            <label className="flex items-center gap-2 text-sm font-medium text-gray-300 cursor-pointer">
                              <input
                                type="checkbox"
                                id={`slider_bound_${boundName}_enabled`}
                                checked={enabled}
                                disabled={saving}
                                onChange={(e) => {
                                  const checked = e.target.checked;
                                  setSliderBoundEnabled((prev) => ({ ...prev, [boundName]: checked }));
                                  void commitSliderBound(boundName, checked ? (sliderBoundValue[boundName] ?? spec.builtin) : null);
                                }}
                                className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                              />
                              {spec.label}
                            </label>
                            {enabled && (
                              <NumberInput
                                id={`slider_bound_${boundName}`}
                                label={spec.label}
                                value={value}
                                onCommit={(v) => handleSliderBoundNumberCommit(boundName, v)}
                                min={spec.floor}
                                max={spec.ceiling}
                                parse="int"
                                disabled={saving}
                                className="w-28 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                              />
                            )}
                            <p className="text-xs text-gray-500">
                              Effective value: {enabled ? value : spec.builtin} ({enabled ? "your override" : "built-in"}
                              {enabled ? `, built-in is ${spec.builtin}` : ""}).
                            </p>
                            {isAboveBuiltin(boundName, paramBounds, enabled ? value : spec.builtin) && (
                              <p className="text-xs text-amber-400">
                                This raises {spec.label.toLowerCase()} above the built-in default ({spec.builtin});
                                territory beyond the built-in default is untested.
                              </p>
                            )}
                          </div>
                        );
                      })}
                  </div>
                ))}

                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-gray-300 border-b border-gray-700 pb-1">Video</h4>
                  <div className="space-y-2">
                    {videoFrameSliderMaxMessage && (
                      <div className={`p-3 rounded text-sm ${videoFrameSliderMaxMessage.type === "success" ? "bg-green-900/30 text-green-400" : "bg-red-900/30 text-red-400"}`}>
                        {videoFrameSliderMaxMessage.text}
                      </div>
                    )}
                    <label className="flex items-center gap-2 text-sm font-medium text-gray-300 cursor-pointer">
                      <input
                        type="checkbox"
                        id="video_frame_slider_max_enabled"
                        checked={videoFrameSliderMaxEnabled}
                        disabled={videoFrameSliderMaxSaving}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setVideoFrameSliderMaxEnabled(checked);
                          void commitVideoFrameSliderMax(checked ? (videoFrameSliderMaxValue ?? videoFrameSliderMaxSeed) : null);
                        }}
                        className="w-4 h-4 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                      />
                      Video Frame Count Slider Track Max
                    </label>
                    {videoFrameSliderMaxEnabled && (
                      <div className="flex items-center space-x-4">
                        <NumberInput
                          id="video_frame_slider_max"
                          label="Video Frame Count Slider Track Max"
                          value={videoFrameSliderMaxValue}
                          onCommit={handleVideoFrameSliderMaxNumberCommit}
                          min={1}
                          parse="int"
                          disabled={videoFrameSliderMaxSaving}
                          className="w-28 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                        />
                      </div>
                    )}
                    <p className="text-xs text-gray-500 mt-1">
                    Sets how far the video frame-count slider&apos;s track reaches on an
                    architecture that does not impose a hard per-request frame limit.
                    The number box next to the slider is not bounded by this setting
                    and always accepts a value above it. Unchecked uses the
                    slider&apos;s own built-in track reach. The value is snapped onto
                    the loaded architecture&apos;s frame grid where the slider is used.
                    Applies immediately and is held on the server.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Send Settings">
            <div className="space-y-4">
              <p className="text-gray-400 text-sm mb-4">
                Configure default size mode when sending images between panels (txt2img → img2img/inpaint, etc.)
              </p>

              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300">
                    Default Size Mode
                  </label>
                  <div className="flex gap-4">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="send_size_mode"
                        value="absolute"
                        checked={sendSizeMode === "absolute"}
                        onChange={(e) => {
                          setSendSizeMode("absolute");
                          localStorage.setItem('send_size_mode', "absolute");
                        }}
                        className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-300">Absolute</span>
                    </label>
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="radio"
                        name="send_size_mode"
                        value="scale"
                        checked={sendSizeMode === "scale"}
                        onChange={(e) => {
                          setSendSizeMode("scale");
                          localStorage.setItem('send_size_mode', "scale");
                        }}
                        className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 focus:ring-blue-500"
                      />
                      <span className="text-sm text-gray-300">Scale</span>
                    </label>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    When set to &quot;Scale&quot;, receiving panels will use scale mode with the default scale value below. When &quot;Absolute&quot;, the exact pixel dimensions are used.
                  </p>
                </div>

                <div className="space-y-2">
                  <label htmlFor="send_default_scale" className="text-sm font-medium text-gray-300">
                    Default Scale Value (for Scale mode)
                  </label>
                  <div className="flex items-center space-x-4">
                    <NumberInput
                      id="send_default_scale"
                      value={sendDefaultScale}
                      defaultValue={1.0}
                      parse="float"
                      onCommit={(v) => {
                        setSendDefaultScale(v);
                        localStorage.setItem('send_default_scale', v.toString());
                      }}
                      min={0.1}
                      max={4.0}
                      step="any"
                      className="w-24 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-gray-100 focus:ring-blue-500 focus:border-blue-500"
                    />
                    <span className="text-sm text-gray-400">×</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Default scale multiplier when receiving images in scale mode. Default is 1.0 (same size as source).
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Resolution Presets">
            <div className="space-y-6">
              <p className="text-gray-400 text-sm mb-4">
                Customize aspect ratio and fixed resolution presets shown in generation panels.
              </p>

              {/* Aspect Ratio Presets */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-200">Aspect Ratio Presets</h3>
                  <div className="flex gap-2">
                    <Button onClick={handleAddAspectRatioPreset} size="sm" variant="secondary">
                      Add
                    </Button>
                    <Button onClick={handleRestoreAspectRatioDefaults} size="sm" variant="secondary">
                      Restore Defaults
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
                  {aspectRatioPresets.map((preset, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded border border-gray-700"
                    >
                      <span className="text-sm text-gray-300 flex-1">{preset.label}</span>
                      <button
                        onClick={() => handleRemoveAspectRatioPreset(index)}
                        className="text-red-400 hover:text-red-300 text-xs"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fixed Resolution Presets */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-200">Fixed Resolution Presets</h3>
                  <div className="flex gap-2">
                    <Button onClick={handleAddFixedResolutionPreset} size="sm" variant="secondary">
                      Add
                    </Button>
                    <Button onClick={handleRestoreFixedResolutionDefaults} size="sm" variant="secondary">
                      Restore Defaults
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
                  {fixedResolutionPresets.map((preset, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-2 px-2 py-2 bg-gray-800 rounded border border-gray-700"
                    >
                      <span className="text-xs text-gray-300 flex-1">
                        {preset.width}×{preset.height}
                      </span>
                      <button
                        onClick={() => handleRemoveFixedResolutionPreset(index)}
                        className="text-red-400 hover:text-red-300 text-xs"
                        title="Remove"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <Card title="Panel Visibility Settings">
            <div className="space-y-6">
              <p className="text-gray-400 text-sm mb-4">
                Control which features are visible in each generation panel.
              </p>

              {/* Txt2Img Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Text to Image Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, lora: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, controlnet: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, aspectRatioPresets: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={txt2imgVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...txt2imgVisibility, fixedResolutionPresets: e.target.checked };
                        setTxt2imgVisibility(newVisibility);
                        localStorage.setItem('txt2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>

              {/* Img2Img Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Image to Image Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, lora: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, controlnet: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, aspectRatioPresets: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={img2imgVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...img2imgVisibility, fixedResolutionPresets: e.target.checked };
                        setImg2imgVisibility(newVisibility);
                        localStorage.setItem('img2img_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>

              {/* Inpaint Panel */}
              <div className="space-y-3">
                <h3 className="text-sm font-semibold text-gray-200">Inpaint Panel</h3>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.lora}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, lora: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show LoRA</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.controlnet}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, controlnet: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show ControlNet</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.aspectRatioPresets}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, aspectRatioPresets: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Aspect Ratio Presets</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={inpaintVisibility.fixedResolutionPresets}
                      onChange={(e) => {
                        const newVisibility = { ...inpaintVisibility, fixedResolutionPresets: e.target.checked };
                        setInpaintVisibility(newVisibility);
                        localStorage.setItem('inpaint_visibility', JSON.stringify(newVisibility));
                      }}
                      className="w-4 h-4"
                    />
                    <span className="text-sm text-gray-300">Show Fixed Resolution Presets</span>
                  </label>
                </div>
              </div>
            </div>
          </Card>

          <Card title="Other Settings">
            {/* Font Size Slider (Mobile UI Scaling) */}
            <div className="mb-6">
              <label className="block text-sm font-medium mb-2">
                UI Font Size (Mobile): {fontSize}%
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="50"
                  max="200"
                  step="5"
                  value={fontSize}
                  onChange={(e) => {
                    const newSize = parseInt(e.target.value);
                    setFontSize(newSize);
                    localStorage.setItem('ui_font_size', newSize.toString());
                    document.documentElement.style.setProperty('--ui-font-size', `${newSize}%`);
                  }}
                  className="flex-1"
                />
                <Button
                  onClick={() => {
                    setFontSize(100);
                    localStorage.setItem('ui_font_size', '100');
                    document.documentElement.style.setProperty('--ui-font-size', '100%');
                  }}
                  variant="secondary"
                >
                  Reset
                </Button>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Adjust the overall font size for mobile devices. This affects all UI elements. Default is 100%.
              </p>
            </div>
          </Card>
            </div>
          </div>
      </main>
    </div>
    </ProtectedRoute>
  );
}
