"use client";

import {
  ChangeEvent,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import NextImage from "next/image";
import {
  Archive,
  AudioLines,
  Check,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Clock3,
  Download,
  Eye,
  EyeOff,
  Film,
  FolderOpen,
  Hand,
  Image as ImageIcon,
  ImagePlus,
  Link2,
  Lock,
  Magnet,
  Maximize2,
  Menu,
  MousePointer2,
  Pause,
  Play,
  Plus,
  Redo2,
  RotateCcw,
  Scissors,
  Search,
  SlidersHorizontal,
  Sparkles,
  MousePointerSquareDashed,
  Trash2,
  Undo2,
  Unlock,
  Upload,
  Volume2,
  VolumeX,
  Wand2,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  GeneratedImage,
  archSupportsFeature,
  cancelGeneration,
  generateImg2Img,
  generateImg2Vid,
  generateInpaint,
  generateInpaintVideo,
  generateOutpaintVideo,
  generateRef2Vid,
  generateTxt2Img,
  generateTxt2Vid,
  getImage,
  getImages,
  getResultFilename,
  getResultPlaybackFilename,
  videoFrameOptions,
} from "@/utils/api";
import type { GenerationParams, Img2ImgParams, InpaintParams, InpaintVideoParams, MiniMaxH3References, OutpaintVideoParams, Ref2VidParams } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { formatTimecode } from "@/utils/timecode";
import ImageEditor from "../common/ImageEditor";
import { loadStudioProject, saveImportedMedia, saveStudioProject } from "./studioStorage";
import { resolveStudioTransferUrl, takeStudioTransfer, type StudioTransferPayload } from "./studioTransfer";
import {
  StudioAsset,
  StudioClip,
  StudioGenerationMode,
  StudioInputRole,
  StudioJob,
  StudioMode,
  StudioPane,
  StudioProject,
  StudioRange,
  StudioTool,
  createStudioProject,
} from "./types";
import { clipEnd, frameDuration, maxTimelineDuration, planStudioGeneration } from "./studioTimeline";
import {
  frameTimeForClip,
  frameIndexForClipTime,
  sourceTrimFrames,
  studioAssetFromGeneration,
  videoInpaintFrames,
  videoOutpaintPlacement,
} from "./studioGeneration";
import styles from "./studio.module.css";

interface StudioFormState {
  prompt: string;
  negativePrompt: string;
  width?: number;
  height?: number;
  numFrames?: number;
  frameRate?: number;
  steps?: number;
  guidance?: number;
  sampler?: string;
  scheduleType?: string;
  denoisingStrength?: number;
  seed?: number;
  audioEnable?: boolean;
}

type MediaFilter = "all" | "image" | "video" | "audio";
type AssetScope = "all" | "gallery" | "import" | "generation";
type RangeTarget = "output" | "inpaint";

interface AssetFilters {
  scope: AssetScope;
  dateFrom: string;
  dateTo: string;
  widthMin: string;
  widthMax: string;
  heightMin: string;
  heightMax: string;
}

const EMPTY_FORM: StudioFormState = { prompt: "", negativePrompt: "" };
const EMPTY_ASSET_FILTERS: AssetFilters = {
  scope: "all",
  dateFrom: "",
  dateTo: "",
  widthMin: "",
  widthMax: "",
  heightMin: "",
  heightMax: "",
};
const MAX_HISTORY = 60;
const GALLERY_PAGE_SIZE = 80;

const numeric = (value: unknown): number | undefined => {
  if (typeof value !== "number" && (typeof value !== "string" || !value.trim())) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const booleanValue = (value: unknown): boolean | undefined =>
  typeof value === "boolean" ? value : undefined;

const sourceDurationForAsset = (asset: StudioAsset): number | undefined => {
  if (asset.kind === "image") return undefined;
  const duration = numeric(asset.duration);
  const frames = numeric(asset.parameters?.num_frames) ?? numeric(asset.parameters?.frames);
  const frameRate = numeric(asset.parameters?.frame_rate) ?? numeric(asset.parameters?.fps);
  const frameDuration = frames && frameRate && frames > 0 && frameRate > 0 ? frames / frameRate : undefined;
  if (duration != null && duration > 0 && frameDuration != null) return Math.min(duration, frameDuration);
  return duration != null && duration > 0 ? duration : frameDuration;
};

const frameDurationFor = (fps: number): number => 1 / Math.max(1, fps);

const clampTime = (value: number, duration: number): number =>
  Math.max(0, Math.min(duration, Number.isFinite(value) ? value : 0));

const defaultClipDurationForAsset = (
  asset: StudioAsset,
  fps: number,
  remaining: number,
  holdStill = false,
): number => {
  const frameDuration = frameDurationFor(fps);
  if (asset.kind === "image") return holdStill ? remaining : Math.min(frameDuration, remaining);
  const sourceDuration = sourceDurationForAsset(asset) || remaining;
  return Math.min(remaining, Math.max(frameDuration, sourceDuration));
};

const safeModelLabel = (value: unknown): string => {
  const raw = String(value || "No model loaded");
  return raw.split(/[\\/]/).filter(Boolean).at(-1) || "No model loaded";
};

const assetKind = (image: GeneratedImage): StudioAsset["kind"] => {
  if (image.is_video || /\.(mp4|webm|mkv)$/i.test(image.filename)) return "video";
  if (image.is_audio || /\.(flac|wav|mp3|ogg|m4a)$/i.test(image.filename)) return "audio";
  return "image";
};

const galleryAsset = (image: GeneratedImage): StudioAsset => {
  const kind = assetKind(image);
  const baseName = image.filename.replace(/\.[^/.]+$/, "");
  const parsedDuration = Number(image.duration);
  return {
    id: `gallery-${image.id}`,
    galleryId: image.id,
    name: image.filename,
    kind,
    url: `/outputs/${image.preview_filename || image.filename}`,
    masterUrl: `/outputs/${image.filename}`,
    thumbnailUrl: `/thumbnails/${baseName}.png`,
    duration: kind === "image" ? 0 : Number.isFinite(parsedDuration) && parsedDuration > 0 ? parsedDuration : 6,
    width: image.width,
    height: image.height,
    source: "gallery",
    prompt: image.prompt,
    negativePrompt: image.negative_prompt || undefined,
    createdAt: image.created_at,
    generationType: image.generation_type,
    modelName: image.model_name,
    seed: image.seed,
    parameters: image.parameters,
  };
};

const galleryTypesFor = (filter: MediaFilter): string | undefined => {
  if (filter === "video") return "txt2vid,img2vid,ref2vid,inpaint_vid,outpaint_vid";
  if (filter === "audio") return "txt2aud,aud2aud,repaint,outpaint_aud";
  if (filter === "image") return "txt2img,img2img,inpaint,outpaint,upscale";
  return undefined;
};

const numberFilter = (value: string): number | undefined => {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
};

const canonicalAssetKey = (asset: StudioAsset): string => {
  return asset.blobKey || asset.masterUrl || (asset.galleryId != null ? `gallery:${asset.galleryId}` : asset.url || asset.id);
};

const readMediaMetadata = (file: File, url: string): Promise<Pick<StudioAsset, "duration" | "width" | "height">> =>
  new Promise((resolve) => {
    if (file.type.startsWith("image/")) {
      const image = new window.Image();
      image.onload = () => resolve({ duration: 0, width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => resolve({ duration: 0 });
      image.src = url;
      return;
    }

    const media = document.createElement(file.type.startsWith("audio/") ? "audio" : "video");
    media.preload = "metadata";
    media.onloadedmetadata = () =>
      resolve({
        duration: Number.isFinite(media.duration) ? media.duration : 6,
        width: media instanceof HTMLVideoElement ? media.videoWidth : undefined,
        height: media instanceof HTMLVideoElement ? media.videoHeight : undefined,
      });
    media.onerror = () => resolve({ duration: 6 });
    media.src = url;
  });

const captureVideoFrameAsset = async (asset: StudioAsset, time: number): Promise<StudioAsset | null> => {
  if (asset.kind === "image") return asset;
  if (asset.kind !== "video" || !asset.url) return null;

  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = asset.url;
  try {
    await new Promise<void>((resolve, reject) => {
      const loaded = () => { cleanup(); resolve(); };
      const failed = () => { cleanup(); reject(new Error("Could not load the video frame.")); };
      const cleanup = () => {
        video.removeEventListener("loadedmetadata", loaded);
        video.removeEventListener("error", failed);
      };
      video.addEventListener("loadedmetadata", loaded, { once: true });
      video.addEventListener("error", failed, { once: true });
      if (video.readyState >= 1) loaded();
    });

    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : sourceDurationForAsset(asset) || 0;
    const target = clampTime(time, Math.max(0, duration - 0.001));
    if (Math.abs(video.currentTime - target) > 0.001 || video.readyState < 2) {
      await new Promise<void>((resolve, reject) => {
        const seeked = () => { cleanup(); resolve(); };
        const failed = () => { cleanup(); reject(new Error("Could not seek to the video frame.")); };
        const cleanup = () => {
          video.removeEventListener("seeked", seeked);
          video.removeEventListener("error", failed);
        };
        video.addEventListener("seeked", seeked, { once: true });
        video.addEventListener("error", failed, { once: true });
        video.currentTime = target;
      });
    }

    if (!video.videoWidth || !video.videoHeight) return null;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const url = canvas.toDataURL("image/png");
    return {
      id: `frame-${asset.id}-${Math.round(target * 1000)}`,
      name: `${asset.name} · frame ${target.toFixed(2)}s`,
      kind: "image",
      url,
      masterUrl: url,
      thumbnailUrl: url,
      duration: 0,
      width: canvas.width,
      height: canvas.height,
      source: asset.source,
      prompt: asset.prompt,
      negativePrompt: asset.negativePrompt,
      createdAt: new Date().toISOString(),
      generationType: asset.generationType,
      modelName: asset.modelName,
      seed: asset.seed,
      parameters: {
        ...(asset.parameters || {}),
        source_asset_id: asset.id,
        source_time: target,
      },
    };
  } finally {
    video.pause();
    video.removeAttribute("src");
    video.load();
  }
};

const mediaFileForUpload = async (asset: StudioAsset): Promise<File> => {
  const source = asset.masterUrl || asset.url;
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Could not read ${asset.name}.`);
  const blob = await response.blob();
  return new File([blob], asset.name || "studio-media", { type: blob.type || "application/octet-stream" });
};

export default function StudioWorkspace() {
  const [project, setProject] = useState<StudioProject>(() => createStudioProject());
  const [restored, setRestored] = useState(false);
  const [undoStack, setUndoStack] = useState<StudioProject[]>([]);
  const [redoStack, setRedoStack] = useState<StudioProject[]>([]);
  const [galleryAssets, setGalleryAssets] = useState<StudioAsset[]>([]);
  const [galleryTotal, setGalleryTotal] = useState(0);
  const [mediaFilter, setMediaFilter] = useState<MediaFilter>("all");
  const [mediaQuery, setMediaQuery] = useState("");
  const [assetFilters, setAssetFilters] = useState<AssetFilters>(EMPTY_ASSET_FILTERS);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [pendingTransfer, setPendingTransfer] = useState<StudioTransferPayload | null>(null);
  const [defaultsIdentity, setDefaultsIdentity] = useState<string | null>(null);
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [tool, setTool] = useState<StudioTool>("select");
  const [rightPane, setRightPane] = useState<StudioPane>("generate");
  const [mode, setMode] = useState<StudioMode>("t2v");
  const [form, setForm] = useState<StudioFormState>(EMPTY_FORM);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState(18);
  const [range, setRange] = useState<StudioRange | null>(null);
  const [inpaintRange, setInpaintRange] = useState<StudioRange | null>(null);
  const [rangeTarget, setRangeTarget] = useState<RangeTarget>("output");
  const [imageEditorState, setImageEditorState] = useState<{ assetId: string; mode: "edit" | "inpaint" } | null>(null);
  const [imageInputMode, setImageInputMode] = useState<"i2i" | "inpaint">("i2i");
  const [referenceAssetIds, setReferenceAssetIds] = useState<string[]>([]);
  const [generationDropActive, setGenerationDropActive] = useState(false);
  const [frameDropLoading, setFrameDropLoading] = useState(false);
  const [jobs, setJobs] = useState<StudioJob[]>([]);
  const [resultAssetIds, setResultAssetIds] = useState<string[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(true);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playStartedRef = useRef({ at: 0, playhead: 0 });
  const initializedDefaultsForArchRef = useRef<string | null>(null);
  const galleryHydrationRef = useRef(new Map<string, Promise<StudioAsset>>());
  const galleryRequestRef = useRef(0);
  const timelineGestureCleanupRef = useRef<(() => void) | null>(null);
  const suppressClipClickRef = useRef<string | null>(null);
  const pendingImageMaskRef = useRef<string | undefined>(undefined);
  const {
    isBackendReady,
    isVideo,
    modelInfo,
    generationDefaults,
    archCapabilities,
    resolveModality,
  } = useStartup();

  const commit = useCallback((updater: (current: StudioProject) => StudioProject) => {
    setProject((current) => {
      setUndoStack((history) => [...history, current].slice(-MAX_HISTORY));
      setRedoStack([]);
      return { ...updater(current), revision: current.revision + 1, updatedAt: new Date().toISOString() };
    });
  }, []);

  const undo = useCallback(() => {
    setUndoStack((history) => {
      const previous = history.at(-1);
      if (!previous) return history;
      setProject((current) => {
        setRedoStack((redo) => [...redo, current].slice(-MAX_HISTORY));
        return { ...previous, jobs: current.jobs };
      });
      return history.slice(0, -1);
    });
  }, []);

  const redo = useCallback(() => {
    setRedoStack((history) => {
      const next = history.at(-1);
      if (!next) return history;
      setProject((current) => {
        setUndoStack((undoHistory) => [...undoHistory, current].slice(-MAX_HISTORY));
        return { ...next, jobs: current.jobs };
      });
      return history.slice(0, -1);
    });
  }, []);

  useEffect(() => {
    loadStudioProject()
      .then((saved) => {
        if (saved) {
          const restoredJobs = (saved.jobs || []).map((job): StudioJob => job.status === "running"
            ? { ...job, status: "failed", error: "Studio closed before this job returned. Check Gallery before retrying." }
            : job);
          setProject({ ...saved, jobs: restoredJobs });
          setRange(saved.outputRange ?? null);
          setInpaintRange(saved.inpaintRange ?? null);
          setReferenceAssetIds(saved.referenceAssetIds ?? []);
          setJobs(restoredJobs);
          setResultAssetIds(restoredJobs.flatMap((job) => job.assetId ? [job.assetId] : []));
        }
      })
      .finally(() => setRestored(true));
  }, []);

  useEffect(() => {
    setPendingTransfer(takeStudioTransfer());
  }, []);

  useEffect(() => () => {
    timelineGestureCleanupRef.current?.();
  }, []);

  useEffect(() => {
    const expectedDefaultsIdentity = modelInfo?.type ? `${modelInfo.type}:${modelInfo.variant || ""}` : null;
    if (!restored || !pendingTransfer || !isBackendReady || !generationDefaults
      || (expectedDefaultsIdentity && defaultsIdentity !== expectedDefaultsIdentity)) return;
    let cancelled = false;
    const applyTransfer = async () => {
      const parameters = pendingTransfer.parameters || {};
      let asset: StudioAsset | null = null;
      if (pendingTransfer.media) {
        const media = pendingTransfer.media;
        const url = await resolveStudioTransferUrl(media);
        const fallbackName = (media.masterUrl || media.url).split("/").pop()?.split("?")[0];
        const duration = Number(media.duration);
        asset = {
          id: media.galleryId != null ? `gallery-${media.galleryId}` : `transfer-${pendingTransfer.id}`,
          galleryId: media.galleryId,
          name: media.name || fallbackName || `Studio ${media.kind}`,
          kind: media.kind,
          url,
          masterUrl: media.blobKey ? undefined : media.masterUrl || url,
          thumbnailUrl: media.thumbnailUrl || (media.kind === "image"
            ? url
            : media.masterUrl?.startsWith("/outputs/")
              ? `/thumbnails/${(media.masterUrl.split("/").pop() || "").replace(/\.[^/.]+$/, "")}.png`
              : undefined),
          duration: media.kind === "image" ? 0 : Number.isFinite(duration) && duration > 0 ? duration : 6,
          width: media.width,
          height: media.height,
          source: pendingTransfer.source === "gallery" ? "gallery" : "generation",
          blobKey: media.blobKey,
          prompt: pendingTransfer.prompt || (typeof parameters.prompt === "string" ? parameters.prompt : undefined),
          negativePrompt: pendingTransfer.negativePrompt || (typeof parameters.negative_prompt === "string" ? parameters.negative_prompt : undefined),
          createdAt: media.createdAt || pendingTransfer.createdAt,
          generationType: media.generationType,
          modelName: media.modelName,
          seed: media.seed,
          parameters,
        };
      }
      if (cancelled) return;

      if (asset) {
        const incoming = asset;
        setProject((current) => {
          const existing = current.assets.find((item) => canonicalAssetKey(item) === canonicalAssetKey(incoming));
          const selected = existing || incoming;
          setSelectedAssetId(selected.id);
          if (selected.kind === "image") setMode("i2v");
          if (existing) {
            return current;
          }
          return {
            ...current,
            assets: [...current.assets, incoming],
            revision: current.revision + 1,
            updatedAt: new Date().toISOString(),
          };
        });
      }

      setForm((current) => {
        const next = { ...current };
        if (pendingTransfer.prompt !== undefined) next.prompt = pendingTransfer.prompt;
        if (pendingTransfer.negativePrompt !== undefined) next.negativePrompt = pendingTransfer.negativePrompt;
        const width = numeric(parameters.width);
        const height = numeric(parameters.height);
        const numFrames = numeric(parameters.num_frames);
        const frameRate = numeric(parameters.frame_rate) ?? numeric(parameters.fps);
        const steps = numeric(parameters.num_inference_steps) ?? numeric(parameters.inference_steps) ?? numeric(parameters.steps);
        const guidance = numeric(parameters.guidance_scale) ?? numeric(parameters.cfg_scale);
        const seed = numeric(parameters.seed);
        const audioEnable = booleanValue(parameters.audio_enable);
        if (width !== undefined) next.width = width;
        if (height !== undefined) next.height = height;
        if (numFrames !== undefined) next.numFrames = numFrames;
        if (frameRate !== undefined) next.frameRate = frameRate;
        if (steps !== undefined) next.steps = steps;
        if (guidance !== undefined) next.guidance = guidance;
        if (seed !== undefined) next.seed = seed;
        if (audioEnable !== undefined) next.audioEnable = audioEnable;
        return next;
      });
      setSelectedClipId(null);
      setRightPane("generate");
      setNotice(`Received ${asset?.name || "generation settings"} from ${pendingTransfer.source}.`);
      setPendingTransfer(null);
    };
    void applyTransfer().catch((error) => {
      console.error("[Studio] Failed to receive transferred media", error);
      if (!cancelled) {
        setNotice("Studio could not receive the transferred media. Please send it again.");
        setPendingTransfer(null);
      }
    });
    return () => { cancelled = true; };
  }, [defaultsIdentity, generationDefaults, isBackendReady, modelInfo?.type, modelInfo?.variant, pendingTransfer, restored]);

  useEffect(() => {
    if (!restored) return;
    setProject((current) => current.jobs === jobs ? current : { ...current, jobs });
  }, [jobs, restored]);

  useEffect(() => {
    if (!restored) return;
    const timer = window.setTimeout(() => saveStudioProject({ ...project, outputRange: range, inpaintRange, referenceAssetIds }), 350);
    return () => window.clearTimeout(timer);
  }, [inpaintRange, project, range, referenceAssetIds, restored]);

  useEffect(() => {
    if (!restored) return;
    const saveOnExit = () => saveStudioProject({ ...project, outputRange: range, inpaintRange, referenceAssetIds });
    window.addEventListener("pagehide", saveOnExit);
    return () => window.removeEventListener("pagehide", saveOnExit);
  }, [inpaintRange, project, range, referenceAssetIds, restored]);

  useEffect(() => {
    if (!generationDefaults || !modelInfo?.type) return;
    const identity = `${modelInfo.type}:${modelInfo.variant || ""}`;
    if (initializedDefaultsForArchRef.current === identity) return;
    initializedDefaultsForArchRef.current = identity;
    setDefaultsIdentity(identity);
    const base = isVideo ? generationDefaults.txt2vid : generationDefaults.txt2img;
    const resolved = {
      ...(base || {}),
      ...(isVideo ? (generationDefaults.video_arch_overlays?.[modelInfo.type] || {}) : {}),
    };
    setForm((current) => ({
      ...current,
      width: numeric(resolved.width),
      height: numeric(resolved.height),
      numFrames: numeric(resolved.num_frames),
      frameRate: numeric(resolved.frame_rate),
      steps: numeric(resolved.num_inference_steps),
      guidance: numeric(resolved.guidance_scale) ?? numeric(resolved.cfg_scale),
      sampler: typeof resolved.sampler === "string" ? resolved.sampler : undefined,
      scheduleType: typeof resolved.schedule_type === "string" ? resolved.schedule_type : undefined,
      denoisingStrength: numeric(resolved.denoising_strength),
      seed: numeric(resolved.seed),
      audioEnable: booleanValue(resolved.audio_enable),
    }));
  }, [generationDefaults, isVideo, modelInfo?.type, modelInfo?.variant]);

  const loadGalleryPage = useCallback(async (skip = 0) => {
    const requestId = ++galleryRequestRef.current;
    const append = skip > 0;
    append ? setLoadingMore(true) : setLibraryLoading(true);
    try {
      const result = await getImages({
        skip,
        limit: GALLERY_PAGE_SIZE,
        search: mediaQuery.trim() || undefined,
        generation_types: galleryTypesFor(mediaFilter),
        date_from: assetFilters.dateFrom ? `${assetFilters.dateFrom}T00:00:00` : undefined,
        date_to: assetFilters.dateTo ? `${assetFilters.dateTo}T23:59:59.999` : undefined,
        width_min: numberFilter(assetFilters.widthMin),
        width_max: numberFilter(assetFilters.widthMax),
        height_min: numberFilter(assetFilters.heightMin),
        height_max: numberFilter(assetFilters.heightMax),
      });
      if (requestId !== galleryRequestRef.current) return;
      const incoming = (result.images || []).map(galleryAsset);
      setGalleryTotal(result.total || 0);
      setGalleryAssets((current) => {
        if (!append) return incoming;
        const known = new Set(current.map((asset) => asset.id));
        return [...current, ...incoming.filter((asset) => !known.has(asset.id))];
      });
    } catch (error) {
      if (requestId === galleryRequestRef.current) console.error("[Studio] Failed to load media library", error);
    } finally {
      if (requestId === galleryRequestRef.current) {
        setLibraryLoading(false);
        setLoadingMore(false);
      }
    }
  }, [assetFilters, mediaFilter, mediaQuery]);

  const refreshLibrary = useCallback(() => {
    void loadGalleryPage(0);
  }, [loadGalleryPage]);

  useEffect(() => {
    const timer = window.setTimeout(refreshLibrary, 250);
    return () => window.clearTimeout(timer);
  }, [refreshLibrary]);

  const allAssets = useMemo(() => {
    const known = new Set(project.assets.map(canonicalAssetKey));
    return [...project.assets, ...galleryAssets.filter((asset) => !known.has(canonicalAssetKey(asset)))];
  }, [galleryAssets, project.assets]);

  const galleryAssetKeys = useMemo(() => new Set(galleryAssets.map(canonicalAssetKey)), [galleryAssets]);

  const filteredAssets = useMemo(() => {
    const query = mediaQuery.trim().toLowerCase();
    const widthMin = numberFilter(assetFilters.widthMin);
    const widthMax = numberFilter(assetFilters.widthMax);
    const heightMin = numberFilter(assetFilters.heightMin);
    const heightMax = numberFilter(assetFilters.heightMax);
    const from = assetFilters.dateFrom ? new Date(`${assetFilters.dateFrom}T00:00:00`).getTime() : null;
    const to = assetFilters.dateTo ? new Date(`${assetFilters.dateTo}T23:59:59.999`).getTime() : null;
    return allAssets.filter((asset) => {
      const matchesMedia = mediaFilter === "all" || asset.kind === mediaFilter;
      const matchesScope = assetFilters.scope === "all"
        || (assetFilters.scope === "gallery"
          ? asset.galleryId != null || asset.source === "gallery" || galleryAssetKeys.has(canonicalAssetKey(asset))
          : asset.source === assetFilters.scope);
      const matchesQuery = !query || [asset.name, asset.prompt, asset.negativePrompt]
        .some((value) => value?.toLowerCase().includes(query));
      const created = asset.createdAt ? new Date(asset.createdAt).getTime() : null;
      const matchesDate = (from == null || (created != null && created >= from))
        && (to == null || (created != null && created <= to));
      const matchesResolution = (widthMin == null || (asset.width != null && asset.width >= widthMin))
        && (widthMax == null || (asset.width != null && asset.width <= widthMax))
        && (heightMin == null || (asset.height != null && asset.height >= heightMin))
        && (heightMax == null || (asset.height != null && asset.height <= heightMax));
      return matchesMedia && matchesScope && matchesQuery && matchesDate && matchesResolution;
    });
  }, [allAssets, assetFilters, galleryAssetKeys, mediaFilter, mediaQuery]);

  const activeFilterCount = useMemo(() => Object.entries(assetFilters)
    .filter(([key, value]) => key === "scope" ? value !== "all" : Boolean(value)).length, [assetFilters]);

  const selectedClip = project.clips.find((clip) => clip.id === selectedClipId) || null;
  const selectedAsset =
    allAssets.find((asset) => asset.id === (selectedClip?.assetId || selectedAssetId)) || null;
  const activeClips = project.clips.filter((clip) => clip.activeTake !== false);
  const timelinePreviewClip = [...activeClips]
    .sort((left, right) => project.tracks.findIndex((track) => track.id === left.trackId) - project.tracks.findIndex((track) => track.id === right.trackId))
    .find((clip) => {
      const track = project.tracks.find((item) => item.id === clip.trackId);
      return track?.visible && playhead >= clip.start && playhead < clip.start + clip.duration;
    }) || null;
  const timelinePreviewAsset = allAssets.find((asset) => asset.id === timelinePreviewClip?.assetId) || null;
  const previewClip = playing ? timelinePreviewClip : selectedClip || timelinePreviewClip;
  const previewAsset = playing ? timelinePreviewAsset : selectedAsset || timelinePreviewAsset;
  const previewTrack = project.tracks.find((track) => track.id === previewClip?.trackId) || null;
  const loadedArch = modelInfo?.type;
  const isVideoModel = isVideo;
  const currentModelName = safeModelLabel(modelInfo?.name || modelInfo?.source);
  const frameOptions = videoFrameOptions(archCapabilities, loadedArch, form.numFrames);
  const supportsNegativePrompt = archSupportsFeature(archCapabilities, loadedArch, "negative_prompt");
  const supportsGuidance = archSupportsFeature(archCapabilities, loadedArch, "cfg");

  const hydrateGalleryAsset = useCallback(async (asset: StudioAsset): Promise<StudioAsset> => {
    if (asset.source !== "gallery" || asset.galleryId == null) return asset;
    const existing = galleryHydrationRef.current.get(asset.id);
    if (existing) return existing;
    const request = (async () => {
      try {
        const detail = await getImage(asset.galleryId!) as GeneratedImage;
        const hydrated = galleryAsset(detail);
        setGalleryAssets((current) => current.map((item) => item.id === hydrated.id ? hydrated : item));
        setProject((current) => ({
          ...current,
          assets: current.assets.map((item) => item.id === hydrated.id ? hydrated : item),
        }));
        return hydrated;
      } catch (error) {
        console.warn("[Studio] Failed to resolve media details", error);
        galleryHydrationRef.current.delete(asset.id);
        return asset;
      }
    })();
    galleryHydrationRef.current.set(asset.id, request);
    return request;
  }, []);

  const selectAsset = useCallback((asset: StudioAsset) => {
    setSelectedAssetId(asset.id);
    void hydrateGalleryAsset(asset);
  }, [hydrateGalleryAsset]);

  const outputDuration = useMemo(() => {
    if (!form.numFrames || !form.frameRate) return 0;
    return form.numFrames / form.frameRate;
  }, [form.frameRate, form.numFrames]);

  const addAssetToTimeline = useCallback((asset: StudioAsset, start?: number, trackId?: string, holdStill = false) => {
    const targetTrack =
      project.tracks.find((track) => track.id === trackId && track.kind === (asset.kind === "audio" ? "audio" : "video")) ||
      project.tracks.find((track) => track.kind === (asset.kind === "audio" ? "audio" : "video"));
    if (!targetTrack || targetTrack.locked) return;

    const trackEnd = activeClips
      .filter((clip) => clip.trackId === targetTrack.id)
      .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
    const requestedStart = clampTime(start ?? trackEnd, project.duration);
    const initialDuration = defaultClipDurationForAsset(asset, project.fps, project.duration - requestedStart, holdStill);
    const clipStart = clampTime(requestedStart, Math.max(0, project.duration - initialDuration));
    const duration = defaultClipDurationForAsset(asset, project.fps, project.duration - clipStart, holdStill);
    const sourceDuration = sourceDurationForAsset(asset);
    const clip: StudioClip = {
      id: crypto.randomUUID(),
      assetId: asset.id,
      trackId: targetTrack.id,
      name: asset.name,
      start: clipStart,
      duration,
      sourceIn: 0,
      presentation: asset.kind === "image" ? (holdStill ? "hold" : "frame") : "clip",
      ...(sourceDuration != null ? { sourceDuration } : {}),
    };
    commit((current) => ({
      ...current,
      assets: current.assets.some((item) => item.id === asset.id) ? current.assets : [...current.assets, asset],
      clips: [...current.clips, clip],
    }));
    setSelectedAssetId(asset.id);
    setSelectedClipId(clip.id);
  }, [activeClips, commit, project.duration, project.fps, project.tracks]);

  const deleteSelectedClip = useCallback(() => {
    if (!selectedClipId) return;
    const target = project.clips.find((clip) => clip.id === selectedClipId);
    const track = project.tracks.find((item) => item.id === target?.trackId);
    if (!target || track?.locked) {
      if (track?.locked) setNotice(`Unlock ${track.name} before deleting this clip.`);
      return;
    }
    const replacement = target.takeGroupId
      ? project.clips.find((clip) => clip.takeGroupId === target.takeGroupId && clip.id !== target.id)
      : undefined;
    commit((current) => ({
      ...current,
      clips: current.clips
        .filter((clip) => clip.id !== selectedClipId)
        .map((clip) => replacement && clip.id === replacement.id ? { ...clip, activeTake: true } : clip),
    }));
    setSelectedClipId(replacement?.id || null);
    if (replacement) setSelectedAssetId(replacement.assetId);
  }, [commit, project.clips, project.tracks, selectedClipId]);

  const splitSelectedClip = useCallback((targetClip?: StudioClip | null) => {
    const clipToSplit = targetClip || selectedClip;
    const track = project.tracks.find((item) => item.id === clipToSplit?.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before splitting this clip.`);
      return;
    }
    if (!clipToSplit || playhead <= clipToSplit.start + 0.05 || playhead >= clipToSplit.start + clipToSplit.duration - 0.05) {
      setNotice("Move the playhead inside the selected clip before splitting.");
      return;
    }
    const leftDuration = playhead - clipToSplit.start;
    const right: StudioClip = {
      ...clipToSplit,
      id: crypto.randomUUID(),
      start: playhead,
      duration: clipToSplit.duration - leftDuration,
      sourceIn: clipToSplit.sourceIn + leftDuration,
    };
    commit((current) => ({
      ...current,
      clips: current.clips.flatMap((clip) =>
        clip.id === clipToSplit.id ? [{ ...clip, duration: leftDuration }, right] : [clip],
      ),
    }));
    setSelectedClipId(right.id);
  }, [commit, playhead, project.tracks, selectedClip]);

  const moveClip = useCallback((clipId: string, trackId: string, start: number) => {
    const targetTrack = project.tracks.find((track) => track.id === trackId);
    const clip = project.clips.find((item) => item.id === clipId);
    if (!targetTrack || !clip || targetTrack.locked) return;
    const asset = allAssets.find((item) => item.id === clip.assetId);
    if (!asset || targetTrack.kind !== (asset.kind === "audio" ? "audio" : "video")) return;
    const snapped = snapEnabled ? Math.round(start * project.fps) / project.fps : start;
    commit((current) => ({
      ...current,
      clips: current.clips.map((item) => item.id === clipId
        ? { ...item, trackId, start: Math.max(0, Math.min(snapped, current.duration - item.duration)) }
        : item),
    }));
  }, [allAssets, commit, project.clips, project.fps, project.tracks, snapEnabled]);

  const handleTrackDrop = async (event: DragEvent<HTMLDivElement>, trackId: string) => {
    event.preventDefault();
    const bounds = event.currentTarget.getBoundingClientRect();
    const start = clampTime((event.clientX - bounds.left) / zoom, project.duration);
    const clipId = event.dataTransfer.getData("application/x-studio-clip");
    if (clipId) {
      moveClip(clipId, trackId, start);
      return;
    }
    const frameAssetId = event.dataTransfer.getData("application/x-studio-frame");
    if (frameAssetId) {
      const source = allAssets.find((item) => item.id === frameAssetId);
      if (!source) return;
      const hydrated = await hydrateGalleryAsset(source);
      if (hydrated.kind === "audio") {
        setNotice("Audio clips do not have video frames to extract.");
        return;
      }
      const frame = await captureVideoFrameAsset(hydrated, numeric(event.dataTransfer.getData("application/x-studio-frame-time")) ?? playhead);
      if (frame) addAssetToTimeline(frame, start, trackId);
      return;
    }
    const assetId = event.dataTransfer.getData("application/x-studio-asset");
    const asset = allAssets.find((item) => item.id === assetId);
    if (asset) {
      const holdStill = event.dataTransfer.getData("application/x-studio-hold-still") === "1";
      addAssetToTimeline(await hydrateGalleryAsset(asset), start, trackId, holdStill);
    }
  };

  const handleImport = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    for (const file of files) {
      const kind: StudioAsset["kind"] = file.type.startsWith("image/")
        ? "image"
        : file.type.startsWith("audio/")
          ? "audio"
          : "video";
      const id = crypto.randomUUID();
      const blobKey = `media-${id}`;
      const url = URL.createObjectURL(file);
      const metadata = await readMediaMetadata(file, url);
      await saveImportedMedia(blobKey, file);
      const asset: StudioAsset = {
        id,
        name: file.name,
        kind,
        url,
        thumbnailUrl: kind === "image" ? url : undefined,
        duration: metadata.duration,
        width: metadata.width,
        height: metadata.height,
        source: "import",
        blobKey,
        createdAt: new Date(file.lastModified || Date.now()).toISOString(),
      };
      commit((current) => ({ ...current, assets: [...current.assets, asset] }));
      setSelectedAssetId(asset.id);
    }
    event.target.value = "";
  };

  const handleTimelineInput = useCallback(async (clip: StudioClip, asFrame: boolean) => {
    const source = allAssets.find((asset) => asset.id === clip.assetId);
    if (!source) return;
    let inputAsset = source;
    if (asFrame) {
      if (source.kind === "audio") {
        setNotice("Audio clips can be selected as timeline context, but they do not provide an image frame.");
        return;
      }
      const sourceTime = clip.sourceIn + clampTime(playhead - clip.start, clip.duration);
      const frame = await captureVideoFrameAsset(source, sourceTime);
      if (!frame) {
        setNotice("Could not capture a frame from this clip.");
        return;
      }
      inputAsset = frame;
      commit((current) => current.assets.some((asset) => asset.id === frame.id)
        ? current
        : { ...current, assets: [...current.assets, frame] });
    }
    setSelectedClipId(clip.id);
    setSelectedAssetId(inputAsset.id);
    if (inputAsset.kind === "image") setMode("i2v");
    setRightPane("generate");
  }, [allAssets, commit, playhead]);

  const handleRightPaneDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const clipId = event.dataTransfer.getData("application/x-studio-clip");
    if (clipId) {
      const clip = project.clips.find((item) => item.id === clipId);
      if (clip) await handleTimelineInput(clip, event.shiftKey || event.dataTransfer.getData("application/x-studio-input-mode") === "frame");
      return;
    }

    const frameAssetId = event.dataTransfer.getData("application/x-studio-frame");
    const assetId = frameAssetId || event.dataTransfer.getData("application/x-studio-asset");
    const asset = allAssets.find((item) => item.id === assetId);
    if (!asset) return;
    const hydrated = await hydrateGalleryAsset(asset);
    if (frameAssetId) {
      if (hydrated.kind === "audio") {
        setNotice("Audio clips do not have a still frame to use as an image input.");
        return;
      }
      const frame = await captureVideoFrameAsset(hydrated, numeric(event.dataTransfer.getData("application/x-studio-frame-time")) ?? playhead);
      if (!frame) {
        setNotice("Could not capture a frame from this media.");
        return;
      }
      commit((current) => current.assets.some((item) => item.id === frame.id)
        ? current
        : { ...current, assets: [...current.assets, frame] });
      setSelectedAssetId(frame.id);
      setSelectedClipId(null);
      setMode("i2v");
      return;
    }
    setSelectedAssetId(hydrated.id);
    setSelectedClipId(null);
    if (hydrated.kind === "image") setMode("i2v");
  };

  const toggleClipInputRole = (clipId: string, role: StudioInputRole) => {
    commit((current) => ({
      ...current,
      clips: current.clips.map((clip) => {
        if (clip.id !== clipId) return clip;
        const roles = new Set(clip.inputRoles || []);
        if (roles.has(role)) roles.delete(role);
        else roles.add(role);
        return { ...clip, inputRoles: [...roles] };
      }),
    }));
  };

  const openImageEditor = (asset: StudioAsset, editorMode: "edit" | "inpaint" = "edit") => {
    if (asset.kind !== "image" || !asset.url) return;
    pendingImageMaskRef.current = asset.maskUrl;
    setImageEditorState({ assetId: asset.id, mode: editorMode });
  };

  const saveStudioEditedImage = (editedImageUrl: string) => {
    const source = imageEditorState ? allAssets.find((asset) => asset.id === imageEditorState.assetId) : null;
    if (!source) return;
    const derived: StudioAsset = {
      ...source,
      id: `studio-image-${crypto.randomUUID()}`,
      galleryId: undefined,
      name: `${source.name.replace(/\.[^/.]+$/, "")} · edited`,
      url: editedImageUrl,
      masterUrl: editedImageUrl,
      thumbnailUrl: editedImageUrl,
      source: "generation",
      maskUrl: pendingImageMaskRef.current,
      createdAt: new Date().toISOString(),
      parameters: {
        ...(source.parameters || {}),
        studio_derived_from: source.galleryId ?? source.id,
        studio_edit: true,
      },
    };
    commit((current) => ({ ...current, assets: [...current.assets, derived] }));
    setSelectedAssetId(derived.id);
    setSelectedClipId(null);
    setImageEditorState(null);
    pendingImageMaskRef.current = undefined;
  };

  const togglePlayback = useCallback(() => {
    setPlaying((current) => {
      const next = !current;
      if (next) playStartedRef.current = { at: performance.now(), playhead };
      return next;
    });
  }, [playhead]);

  useEffect(() => {
    if (!playing) {
      videoRef.current?.pause();
      audioRef.current?.pause();
      return;
    }
    const timelineNow = playStartedRef.current.playhead + (performance.now() - playStartedRef.current.at) / 1000;
    const localTime = previewClip ? previewClip.sourceIn + Math.max(0, timelineNow - previewClip.start) : 0;
    if (videoRef.current && Number.isFinite(videoRef.current.duration)) videoRef.current.currentTime = localTime;
    if (audioRef.current && Number.isFinite(audioRef.current.duration)) audioRef.current.currentTime = localTime;
    videoRef.current?.play().catch(() => undefined);
    audioRef.current?.play().catch(() => undefined);
    let animation = 0;
    const tick = (now: number) => {
      const next = playStartedRef.current.playhead + (now - playStartedRef.current.at) / 1000;
      if (next >= project.duration) {
        setPlayhead(0);
        setPlaying(false);
        return;
      }
      setPlayhead(next);
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [playing, previewAsset?.url, previewClip, project.duration]);

  useEffect(() => {
    if (playing || !previewAsset) return;
    const localTime = previewClip
      ? previewClip.sourceIn + Math.max(0, playhead - previewClip.start)
      : playhead;
    const bounded = Math.max(0, Math.min(previewAsset.duration || localTime, localTime));
    if (videoRef.current && Number.isFinite(videoRef.current.duration)) videoRef.current.currentTime = bounded;
    if (audioRef.current && Number.isFinite(audioRef.current.duration)) audioRef.current.currentTime = bounded;
  }, [playhead, playing, previewAsset, previewClip]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.matches("input, textarea, select, [contenteditable='true']")) return;
      if (event.code === "Space") {
        event.preventDefault();
        togglePlayback();
      } else if (event.key.toLowerCase() === "s") {
        splitSelectedClip();
      } else if (event.key === "Delete" || event.key === "Backspace") {
        deleteSelectedClip();
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        const frames = event.shiftKey ? 10 : 1;
        const direction = event.key === "ArrowRight" ? 1 : -1;
        setPlayhead((current) => Math.max(0, Math.min(project.duration, current + direction * frames / project.fps)));
      } else if (event.key.toLowerCase() === "k") {
        setPlaying(false);
      } else if (event.key.toLowerCase() === "l") {
        if (!playing) togglePlayback();
      } else if (event.key.toLowerCase() === "i" || event.key.toLowerCase() === "o") {
        const isStart = event.key.toLowerCase() === "i";
        const currentRange = event.altKey ? inpaintRange : range;
        const frame = Math.round(playhead * project.fps) / project.fps;
        const next = currentRange
          ? { start: isStart ? frame : currentRange.start, end: isStart ? currentRange.end : frame }
          : { start: isStart ? frame : 0, end: isStart ? project.duration : frame };
        const normalized = { start: Math.min(next.start, next.end), end: Math.max(next.start, next.end) };
        event.preventDefault();
        if (event.altKey) setInpaintRange(normalized);
        else setRange(normalized);
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteSelectedClip, inpaintRange, playing, project.duration, project.fps, range, redo, splitSelectedClip, togglePlayback, undo]);

  const beginRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool !== "range") {
      const bounds = event.currentTarget.getBoundingClientRect();
      setPlayhead(clampTime((event.clientX - bounds.left) / zoom, project.duration));
      setSelectedClipId(null);
      setSelectedAssetId(null);
      return;
    }
    event.preventDefault();
    timelineGestureCleanupRef.current?.();
    const element = event.currentTarget;
    const bounds = element.getBoundingClientRect();
    const start = clampTime((event.clientX - bounds.left) / zoom, project.duration);
    const updateRange = (current: number) => {
      const next = { start: Math.min(start, current), end: Math.max(start, current) };
      if (rangeTarget === "output") setRange(next);
      else setInpaintRange(next);
    };
    updateRange(start);
    const move = (pointerEvent: PointerEvent) => {
      updateRange(clampTime((pointerEvent.clientX - bounds.left) / zoom, project.duration));
    };
    const finish = () => {
      element.removeEventListener("pointermove", move);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      setRightPane("generate");
    };
    const cancel = () => {
      finish();
      const next = { start, end: Math.min(project.duration, start + frameDurationFor(project.fps)) };
      if (rangeTarget === "output") setRange(next);
      else setInpaintRange(next);
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    element.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
  };

  const beginClipMove = (event: ReactPointerEvent<HTMLDivElement>, clip: StudioClip) => {
    if (event.button !== 0 || tool !== "select") return;
    event.preventDefault();
    event.stopPropagation();
    const track = project.tracks.find((item) => item.id === clip.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before moving this clip.`);
      return;
    }
    timelineGestureCleanupRef.current?.();
    setSelectedClipId(clip.id);
    setSelectedAssetId(clip.assetId);
    const originX = event.clientX;
    const initialStart = clip.start;
    let changed = false;
    const move = (pointerEvent: PointerEvent) => {
      const delta = (pointerEvent.clientX - originX) / zoom;
      const raw = snapEnabled ? Math.round((initialStart + delta) * project.fps) / project.fps : initialStart + delta;
      const nextStart = clampTime(raw, Math.max(0, project.duration - clip.duration));
      if (Math.abs(nextStart - initialStart) < 0.0001) return;
      changed = true;
      setProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id ? { ...item, start: nextStart } : item),
      }));
    };
    const restore = () => {
      setProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id ? { ...item, trackId: clip.trackId, start: clip.start } : item),
      }));
    };
    const up = (pointerEvent: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;

      const generationTarget = document.elementFromPoint(pointerEvent.clientX, pointerEvent.clientY)
        ?.closest("[data-studio-generation-drop]");
      if (generationTarget) {
        if (changed) suppressClipClickRef.current = clip.id;
        restore();
        void handleTimelineInput(clip, pointerEvent.shiftKey);
        return;
      }

      const lane = document.elementFromPoint(pointerEvent.clientX, pointerEvent.clientY)
        ?.closest<HTMLElement>("[data-studio-track-id]");
      const targetTrackId = lane?.dataset.trackId || clip.trackId;
      const targetTrack = project.tracks.find((item) => item.id === targetTrackId);
      const asset = allAssets.find((item) => item.id === clip.assetId);
      const canDropOnTrack = Boolean(targetTrack && asset && !targetTrack.locked
        && targetTrack.kind === (asset.kind === "audio" ? "audio" : "video"));
      const finalTrackId = canDropOnTrack ? targetTrackId : clip.trackId;
      const delta = (pointerEvent.clientX - originX) / zoom;
      const raw = snapEnabled ? Math.round((initialStart + delta) * project.fps) / project.fps : initialStart + delta;
      const finalStart = clampTime(raw, Math.max(0, project.duration - clip.duration));
      const didChange = finalTrackId !== clip.trackId || Math.abs(finalStart - initialStart) >= 0.0001;
      if (didChange) {
        setUndoStack((history) => [...history, project].slice(-MAX_HISTORY));
        setRedoStack([]);
        setProject((current) => ({
          ...current,
          clips: current.clips.map((item) => item.id === clip.id ? { ...item, trackId: finalTrackId, start: finalStart } : item),
          revision: current.revision + 1,
          updatedAt: new Date().toISOString(),
        }));
        suppressClipClickRef.current = clip.id;
      } else if (changed) {
        restore();
      }
    };
    const cancel = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      restore();
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
  };

  const beginTrim = (event: ReactPointerEvent<HTMLButtonElement>, clip: StudioClip, edge: "start" | "end") => {
    event.preventDefault();
    event.stopPropagation();
    const track = project.tracks.find((item) => item.id === clip.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before trimming this clip.`);
      return;
    }
    timelineGestureCleanupRef.current?.();
    const asset = allAssets.find((item) => item.id === clip.assetId);
    if (!asset) return;
    const frameDuration = frameDurationFor(project.fps);
    const sourceDuration = clip.sourceDuration ?? sourceDurationForAsset(asset);
    const initialHold = clip.presentation === "hold" || (asset.kind === "image" && clip.duration > frameDuration + 0.0001);
    const modifierHeld = event.ctrlKey || event.metaKey || event.altKey;
    const originX = event.clientX;
    const initialEnd = clip.start + clip.duration;
    const snapshot = project;
    let changed = false;
    let stillNoticeShown = false;

    const update = (pointerEvent: PointerEvent) => {
      const delta = (pointerEvent.clientX - originX) / zoom;
      let nextStart = clip.start;
      let nextEnd = initialEnd;
      let nextSourceIn = clip.sourceIn;
      let nextPresentation = clip.presentation || (asset.kind === "image" ? "frame" : "clip");
      if (edge === "start") {
        const rawStart = snapEnabled ? Math.round((clip.start + delta) * project.fps) / project.fps : clip.start + delta;
        const minimumStart = Math.max(0, clip.start - clip.sourceIn);
        const sourceStartLimit = asset.kind === "image" || sourceDuration == null
          ? Number.POSITIVE_INFINITY
          : clip.start + Math.max(0, sourceDuration - clip.sourceIn - frameDuration);
        const maximumStart = Math.min(initialEnd - frameDuration, sourceStartLimit);
        nextStart = Math.max(minimumStart, Math.min(maximumStart, rawStart));
        if (asset.kind === "image" && nextStart < clip.start && !modifierHeld) {
          nextStart = clip.start;
          if (!stillNoticeShown) {
            setNotice("Hold Ctrl/Cmd while dragging a still edge to extend its hold.");
            stillNoticeShown = true;
          }
        }
        nextEnd = initialEnd;
        nextSourceIn = asset.kind === "image" ? 0 : clip.sourceIn + (nextStart - clip.start);
        if (asset.kind === "image" && nextEnd - nextStart > frameDuration + 0.0001) nextPresentation = "hold";
      } else {
        const rawEnd = snapEnabled ? Math.round((initialEnd + delta) * project.fps) / project.fps : initialEnd + delta;
        let maximumEnd = project.duration;
        if (asset.kind !== "image" && sourceDuration != null) maximumEnd = Math.min(maximumEnd, clip.start + sourceDuration - clip.sourceIn);
        if (asset.kind === "image" && !initialHold && !modifierHeld) {
          maximumEnd = Math.min(maximumEnd, clip.start + frameDuration);
          if (rawEnd > maximumEnd && !stillNoticeShown) {
            setNotice("Hold Ctrl/Cmd while dragging a still edge to extend its hold.");
            stillNoticeShown = true;
          }
        }
        nextEnd = Math.max(clip.start + frameDuration, Math.min(maximumEnd, rawEnd));
        if (asset.kind === "image") nextPresentation = nextEnd - nextStart > frameDuration + 0.0001 ? "hold" : "frame";
      }
      const nextDuration = Math.max(frameDuration, nextEnd - nextStart);
      const same = Math.abs(nextStart - clip.start) < 0.0001
        && Math.abs(nextDuration - clip.duration) < 0.0001
        && Math.abs(nextSourceIn - clip.sourceIn) < 0.0001
        && nextPresentation === clip.presentation;
      if (same) return;
      changed = true;
      setProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id
          ? { ...item, start: nextStart, duration: nextDuration, sourceIn: nextSourceIn, presentation: nextPresentation as StudioClip["presentation"] }
          : item),
      }));
    };
    const finish = () => {
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (!changed) return;
      setUndoStack((history) => [...history, snapshot].slice(-MAX_HISTORY));
      setRedoStack([]);
      setProject((current) => ({ ...current, revision: current.revision + 1, updatedAt: new Date().toISOString() }));
    };
    const cancel = () => {
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      setProject(snapshot);
    };
    const cleanup = () => {
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    window.addEventListener("pointermove", update);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
  };

  const generateClip = async () => {
    setNotice(null);
    const modality = await resolveModality();
    if (modality.modelInfo?.type !== loadedArch || modality.modelInfo?.variant !== modelInfo?.variant) {
      setNotice("The loaded model changed. Studio refreshed its capability defaults; review them and generate again.");
      return;
    }
    if (!form.prompt.trim() || !form.width || !form.height || form.steps == null || form.guidance == null || form.seed == null) {
      setNotice("Prompt and generation schema values are required.");
      return;
    }
    if (modality.isVideo && (!form.numFrames || !form.frameRate)) {
      setNotice("Video frame count and frame rate are required.");
      return;
    }

    const plan = planStudioGeneration({
      isVideoModel: modality.isVideo,
      fps: form.frameRate || project.fps,
      projectDuration: project.duration,
      playhead,
      outputRange: range,
      inpaintRange,
      selectedClipId,
      clips: activeClips,
      assets: allAssets,
    });
    if (!modality.isVideo && plan.hasVideoInput) {
      setNotice("An image model cannot use a video clip as its image input. Shift-drag a frame first.");
      return;
    }
    const referenceIds = Array.from(new Set([
      ...referenceAssetIds,
      ...activeClips.filter((clip) => clip.inputRoles?.includes("reference")).map((clip) => clip.assetId),
    ]));
    const inferredMode: StudioGenerationMode = modality.isVideo && !plan.hasVideoInput
      && (plan.hasImageInput || selectedAsset?.kind === "image")
      ? "i2v"
      : plan.mode;
    const planMode: StudioGenerationMode = modality.isVideo && modelInfo?.variant === "ref2va" && referenceIds.length
      ? "ref2v"
      : inferredMode;
    if (planMode === "inpaint" && loadedArch !== "minimax_h3") {
      setNotice("Temporal inpaint currently requires the MiniMax-H3 fl2va model.");
      return;
    }
    if (planMode === "ref2v" && (loadedArch !== "minimax_h3" || modelInfo?.variant !== "ref2va")) {
      setNotice("Explicit reference video generation requires MiniMax-H3 ref2va.");
      return;
    }
    const imageInput = selectedAsset?.kind === "image"
      ? selectedAsset
      : plan.imageClip
        ? allAssets.find((asset) => asset.id === plan.imageClip?.assetId) || null
        : null;
    const videoInput = plan.videoClip
      ? allAssets.find((asset) => asset.id === plan.videoClip?.assetId) || null
      : null;
    const keyframeClips = activeClips.filter((clip) => clip.inputRoles?.includes("keyframe"));
    const keyframeAssets = await Promise.all(keyframeClips.map(async (clip) => {
      const asset = allAssets.find((item) => item.id === clip.assetId);
      if (!asset?.url) return null;
      const image = asset.kind === "image"
        ? asset
        : await captureVideoFrameAsset(asset, frameTimeForClip(clip, playhead));
      return image?.url ? { image: image.url, frame_index: frameIndexForClipTime(clip, playhead, form.frameRate || project.fps) } : null;
    }));
    const keyframes = keyframeAssets.filter((item): item is { image: string; frame_index: number } => !!item);
    const firstKeyframe = keyframes[0] || (imageInput?.url ? {
      image: imageInput.url,
      frame_index: imageInput === selectedAsset ? frameIndexForClipTime(plan.imageClip || {
        id: "studio-input",
        assetId: imageInput.id,
        trackId: "video-1",
        name: imageInput.name,
        start: playhead,
        duration: frameDuration(form.frameRate || project.fps),
        sourceIn: 0,
      }, playhead, form.frameRate || project.fps) : 0,
    } : null);

    const jobId = crypto.randomUUID();
    const resolvedModelName = safeModelLabel(modality.modelInfo?.name || modality.modelInfo?.source || currentModelName);
    const recipe: Record<string, unknown> = {
      model: resolvedModelName,
      architecture: modality.modelInfo?.type,
      model_variant: modality.modelInfo?.variant,
      mode: planMode,
      prompt: form.prompt,
      negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
      width: form.width,
      height: form.height,
      num_frames: form.numFrames,
      frame_rate: form.frameRate,
      num_inference_steps: form.steps,
      guidance_scale: form.guidance,
      cfg_scale: form.guidance,
      sampler: form.sampler,
      schedule_type: form.scheduleType,
      denoising_strength: form.denoisingStrength,
      seed: form.seed,
      audio_enable: form.audioEnable,
      output_range: range,
      inpaint_range: inpaintRange,
      source_clip_id: plan.videoClip?.id,
      keyframe_asset_id: imageInput?.id,
      reference_asset_ids: referenceIds,
    };
    setMode(planMode);
    setJobs((current) => [{ id: jobId, mode: planMode, prompt: form.prompt, status: "running", startedAt: Date.now(), recipe }, ...current]);
    setRightPane("jobs");

    try {
      const baseVideoParameters = {
        prompt: form.prompt,
        negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
        width: form.width,
        height: form.height,
        num_frames: form.numFrames,
        frame_rate: form.frameRate,
        num_inference_steps: form.steps,
        guidance_scale: form.guidance,
        seed: form.seed,
        audio_enable: form.audioEnable,
      };
      let result: any;
      if (!modality.isVideo) {
        const imageParameters: GenerationParams = {
          prompt: form.prompt,
          negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
          width: form.width,
          height: form.height,
          steps: form.steps,
          cfg_scale: form.guidance,
          sampler: form.sampler,
          schedule_type: form.scheduleType,
          seed: form.seed,
        };
        if (planMode === "image-inpaint" || imageInputMode === "inpaint") {
          if (!imageInput?.maskUrl) {
            throw new Error("Open the image editor and draw a mask before using image inpaint.");
          }
          result = await generateInpaint({
            ...imageParameters,
            denoising_strength: form.denoisingStrength,
          } as InpaintParams, imageInput.url, imageInput.maskUrl);
        } else if (imageInput) {
          result = await generateImg2Img({
            ...imageParameters,
            denoising_strength: form.denoisingStrength,
          } as Img2ImgParams, imageInput.url);
        } else {
          result = await generateTxt2Img(imageParameters);
        }
      } else if (planMode === "ref2v") {
        if (!referenceIds.length) throw new Error("Select at least one explicit reference in the right pane.");
        const references: MiniMaxH3References = { images: [], videos: [], videoAudios: [], audios: [] };
        for (const assetId of referenceIds) {
          const asset = allAssets.find((item) => item.id === assetId);
          if (!asset) continue;
          const file = await mediaFileForUpload(asset);
          if (asset.kind === "image") references.images.push(file);
          else if (asset.kind === "video") {
            references.videos.push(file);
            references.videoAudios.push(null);
          } else references.audios.push(file);
        }
        if (!references.images.length && !references.videos.length) throw new Error("REF2VA requires an image or video reference.");
        result = await generateRef2Vid({ ...baseVideoParameters, keyframes } as Ref2VidParams, references);
      } else if (planMode === "inpaint") {
        if (!videoInput || !plan.videoClip) throw new Error("Select a video clip to inpaint.");
        if (!plan.inpaintRange) throw new Error("Select an Edit / inpaint range inside the video clip.");
        const edit = plan.inpaintRange;
        const sourceDuration = sourceDurationForAsset(videoInput) || videoInput.duration;
        const editFrames = videoInpaintFrames(plan.videoClip, edit, form.frameRate || project.fps);
        const trim = sourceTrimFrames(plan.videoClip, sourceDuration, form.frameRate || project.fps);
        result = await generateInpaintVideo({
          ...baseVideoParameters,
          regenerate_start_frame: editFrames.start,
          regenerate_end_frame: Math.max(editFrames.start + 1, editFrames.end),
          input_trim_start_frames: trim.start,
          input_trim_end_frames: trim.end,
        } as InpaintVideoParams, videoInput.masterUrl || videoInput.url);
      } else if (planMode === "outpaint") {
        if (!videoInput || !plan.videoClip) throw new Error("Select a video clip to outpaint.");
        const output = range || plan.outputRange;
        const sourceDuration = sourceDurationForAsset(videoInput) || videoInput.duration;
        const placement = videoOutpaintPlacement(plan.videoClip, output, sourceDuration, form.frameRate || project.fps);
        result = await generateOutpaintVideo({
          ...baseVideoParameters,
          total_frames: placement.totalFrames,
          input_offset_frames: placement.inputOffsetFrames,
          input_trim_start_frames: placement.inputTrimStartFrames,
          input_trim_end_frames: placement.inputTrimEndFrames,
        } as OutpaintVideoParams, videoInput.masterUrl || videoInput.url);
      } else {
        result = planMode === "i2v"
          ? await generateImg2Vid({ ...baseVideoParameters, keyframes, input_image_frame_index: firstKeyframe?.frame_index ?? 0 }, firstKeyframe?.image || imageInput?.url || null)
          : await generateTxt2Vid(baseVideoParameters);
      }
      const filename = getResultPlaybackFilename(result) || getResultFilename(result);
      const masterFilename = getResultFilename(result) || filename;
      if (!filename || !masterFilename) throw new Error("Generation completed without an output filename.");
      const generatedKind: StudioAsset["kind"] = modality.isVideo ? "video" : "image";
      const fallbackDuration = modality.isVideo
        ? (form.numFrames || 1) / (form.frameRate || project.fps)
        : 0;
      const asset = studioAssetFromGeneration(result, {
        id: `generation-${jobId}`,
        filename: masterFilename,
        kind: generatedKind,
        url: `/outputs/${filename}`,
        masterUrl: `/outputs/${masterFilename}`,
        thumbnailUrl: generatedKind === "image" ? `/thumbnails/${masterFilename.replace(/\.[^/.]+$/, "")}.png` : undefined,
        duration: fallbackDuration,
        width: form.width,
        height: form.height,
        source: "generation",
        prompt: form.prompt,
        negativePrompt: supportsNegativePrompt ? form.negativePrompt : "",
        generationType: planMode,
        modelName: resolvedModelName,
        seed: form.seed,
        parameters: recipe,
      });
      const targetStart = range?.start ?? playhead;
      const targetDuration = range ? Math.max(frameDuration(project.fps), range.end - range.start) : (asset.duration || fallbackDuration || frameDuration(project.fps));
      const takeGroupId = selectedClip?.takeGroupId || (selectedClip ? crypto.randomUUID() : undefined);
      const clip: StudioClip = {
        id: crypto.randomUUID(),
        assetId: asset.id,
        trackId: selectedClip?.trackId || "video-1",
        name: filename,
        start: selectedClip?.start ?? targetStart,
        duration: generatedKind === "image"
          ? frameDuration(project.fps)
          : Math.min(selectedClip?.duration ?? targetDuration, asset.duration || targetDuration),
        sourceIn: 0,
        presentation: generatedKind === "image" ? "frame" : "clip",
        sourceDuration: asset.duration || targetDuration,
        takeGroupId,
        activeTake: false,
        generated: true,
      };
      commit((current) => ({
        ...current,
        assets: current.assets.some((item) => item.id === asset.id) ? current.assets : [...current.assets, asset],
        clips: [...current.clips.map((item) => item.id === selectedClip?.id ? { ...item, takeGroupId } : item), clip],
      }));
      setSelectedAssetId(asset.id);
      setSelectedClipId(clip.id);
      setResultAssetIds((current) => [asset.id, ...current]);
      setJobs((current) => current.map((job) => job.id === jobId ? { ...job, status: "review" as const, assetId: asset.id } : job));
      setRightPane("generate");
      refreshLibrary();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Generation failed";
      setJobs((current) => current.map((job) => job.id === jobId ? { ...job, status: "failed" as const, error: message } : job));
      setNotice(message);
    }
  };

  const restoreRecipe = (job: StudioJob) => {
    const recipe = job.recipe;
    setMode(job.mode);
    const recipeMatchesModel = recipe.architecture === loadedArch && recipe.model_variant === modelInfo?.variant;
    if (!recipeMatchesModel) {
      setForm((current) => ({ ...current, prompt: String(recipe.prompt || ""), negativePrompt: "" }));
      setNotice("This recipe belongs to another model variant. Its prompt was restored, but current capability defaults were kept.");
      setRightPane("generate");
      return;
    }
    setForm({
      prompt: String(recipe.prompt || ""),
      negativePrompt: String(recipe.negative_prompt || ""),
      width: numeric(recipe.width),
      height: numeric(recipe.height),
      numFrames: numeric(recipe.num_frames),
      frameRate: numeric(recipe.frame_rate),
      steps: numeric(recipe.num_inference_steps),
      guidance: numeric(recipe.guidance_scale),
      sampler: typeof recipe.sampler === "string" ? recipe.sampler : undefined,
      scheduleType: typeof recipe.schedule_type === "string" ? recipe.schedule_type : undefined,
      denoisingStrength: numeric(recipe.denoising_strength),
      seed: numeric(recipe.seed),
      audioEnable: booleanValue(recipe.audio_enable),
    });
    if (recipe.output_range && typeof recipe.output_range === "object") setRange(recipe.output_range as StudioRange);
    if (recipe.inpaint_range && typeof recipe.inpaint_range === "object") setInpaintRange(recipe.inpaint_range as StudioRange);
    setRightPane("generate");
  };

  const cancelJob = async (jobId: string) => {
    try {
      await cancelGeneration();
      setJobs((current) => current.map((job) => job.id === jobId
        ? { ...job, status: "failed" as const, error: "Cancelled by user." }
        : job));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not cancel generation.");
    }
  };

  const applyJob = (job: StudioJob) => {
    if (!job.assetId) return;
    const clip = project.clips.find((item) => item.assetId === job.assetId);
    if (!clip) return;
    const track = project.tracks.find((item) => item.id === clip.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before applying this take.`);
      return;
    }
    if (clip.takeGroupId) {
      if (!activateTake(clip.id)) return;
    } else {
      commit((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id ? { ...item, activeTake: true } : item),
      }));
      setSelectedClipId(clip.id);
      setSelectedAssetId(clip.assetId);
    }
    setJobs((current) => current.map((item) => item.id === job.id ? { ...item, status: "applied" as const } : item));
  };

  const exportProject = () => {
    const manifest = new Blob([JSON.stringify({ ...project, jobs }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(manifest);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${project.name.replace(/[^a-z0-9-_]+/gi, "_") || "studio-project"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setNotice("Project manifest exported. Sequence media rendering needs a backend render endpoint.");
  };

  const updateSelectedClip = (changes: Partial<StudioClip>) => {
    if (!selectedClipId) return;
    const track = project.tracks.find((item) => item.id === selectedClip?.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before editing this clip.`);
      return;
    }
    const asset = selectedClip ? allAssets.find((item) => item.id === selectedClip.assetId) : null;
    if (!asset || !selectedClip) return;
    const next = { ...selectedClip, ...changes };
    const minimum = frameDuration(project.fps);
    if (asset.kind !== "image" && next.sourceDuration != null) {
      next.sourceIn = clampTime(next.sourceIn, Math.max(0, next.sourceDuration - minimum));
    }
    const maxDuration = maxTimelineDuration(next, asset, project.duration);
    next.start = clampTime(next.start, Math.max(0, project.duration - minimum));
    next.duration = Math.max(minimum, Math.min(next.duration, Math.max(minimum, maxDuration)));
    next.start = Math.min(next.start, Math.max(0, project.duration - next.duration));
    if (asset.kind === "image") {
      next.sourceIn = 0;
      next.presentation = next.duration > minimum + 0.0001 ? "hold" : "frame";
    }
    commit((current) => ({
      ...current,
      clips: current.clips.map((clip) => clip.id === selectedClipId ? next : clip),
    }));
  };

  const toggleTrack = (trackId: string, field: "muted" | "locked" | "visible") => {
    commit((current) => ({
      ...current,
      tracks: current.tracks.map((track) => track.id === trackId ? { ...track, [field]: !track[field] } : track),
    }));
  };

  const takeCount = selectedClip?.takeGroupId
    ? project.clips.filter((clip) => clip.takeGroupId === selectedClip.takeGroupId).length
    : 1;
  const takeAlternatives = selectedClip?.takeGroupId
    ? project.clips.filter((clip) => clip.takeGroupId === selectedClip.takeGroupId)
    : selectedClip ? [selectedClip] : [];

  const resolvedPlan = planStudioGeneration({
    isVideoModel,
    fps: form.frameRate || project.fps,
    projectDuration: project.duration,
    playhead,
    outputRange: range,
    inpaintRange,
    selectedClipId,
    clips: activeClips,
    assets: allAssets,
  });
  const hasReferenceInput = referenceAssetIds.length > 0 || activeClips.some((clip) => clip.inputRoles?.includes("reference"));
  const resolvedMode: StudioGenerationMode = isVideoModel && modelInfo?.variant === "ref2va" && hasReferenceInput
    ? "ref2v"
    : !isVideoModel && selectedAsset?.kind === "image"
      ? (imageInputMode === "inpaint" || inpaintRange ? "image-inpaint" : "i2i")
      : isVideoModel && !resolvedPlan.hasVideoInput && selectedAsset?.kind === "image"
        ? "i2v"
        : resolvedPlan.mode;
  const resolvedModeLabel: Record<StudioGenerationMode, string> = {
    t2v: "T2VA · text to video",
    i2v: "I2VA · image to video",
    inpaint: "Temporal inpaint",
    outpaint: "Temporal outpaint",
    ref2v: "REF2VA · explicit references",
    t2i: "T2I · text to image",
    i2i: "I2I · image to image",
    "image-inpaint": "Image inpaint",
  };

  const activateTake = (clipId: string): boolean => {
    const target = project.clips.find((clip) => clip.id === clipId);
    if (!target) return false;
    const track = project.tracks.find((item) => item.id === target.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before changing takes.`);
      return false;
    }
    commit((current) => ({
      ...current,
      clips: current.clips.map((clip) => target.takeGroupId
        ? clip.takeGroupId === target.takeGroupId ? { ...clip, activeTake: clip.id === clipId } : clip
        : clip.id === clipId ? { ...clip, activeTake: true } : clip),
    }));
    setSelectedClipId(target.id);
    setSelectedAssetId(target.assetId);
    return true;
  };

  return (
    <main className={styles.studio}>
      <header className={styles.topbar}>
        <div className={styles.projectIdentity}>
          <Menu size={17} />
          <span className={styles.studioLabel}>Studio</span>
          <span className={styles.divider} />
          <input
            aria-label="Project name"
            value={project.name}
            onChange={(event) => setProject((current) => ({ ...current, name: event.target.value, revision: current.revision + 1, updatedAt: new Date().toISOString() }))}
            className={styles.projectName}
          />
          <ChevronDown size={14} />
        </div>
        <div className={styles.historyControls}>
          <button onClick={undo} disabled={!undoStack.length} aria-label="Undo"><Undo2 size={17} /></button>
          <button onClick={redo} disabled={!redoStack.length} aria-label="Redo"><Redo2 size={17} /></button>
          <span className={styles.savedState}>Saved locally · {new Date(project.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
        </div>
        <div className={styles.exportControls}>
          <span className={styles.sequenceBadge}>{project.width}×{project.height} · {project.fps} fps</span>
          <button className={styles.exportButton} onClick={exportProject}><Upload size={16} /> Export project</button>
        </div>
      </header>

      <div className={styles.workbench}>
        <aside className={styles.mediaPane} aria-label="Media library">
          <div className={styles.mediaHeader}>
           <div>
              <span className={styles.eyebrow}>ASSETS</span>
              <strong>{filteredAssets.length} shown</strong>
            </div>
            <button onClick={refreshLibrary} aria-label="Refresh Gallery assets" title="Refresh Gallery"><RotateCcw size={15} /></button>
            <button onClick={() => fileInputRef.current?.click()} aria-label="Import media"><Plus size={17} /></button>
            <input ref={fileInputRef} type="file" accept="image/*,video/*,audio/*" multiple hidden onChange={handleImport} />
          </div>
          <div className={styles.searchRow}>
            <div className={styles.searchBox}><Search size={14} /><input value={mediaQuery} onChange={(event) => setMediaQuery(event.target.value)} placeholder="Search prompt or name" /></div>
            <button className={`${styles.filterButton} ${filtersOpen || activeFilterCount ? styles.activeFilterButton : ""}`} onClick={() => setFiltersOpen((open) => !open)} aria-expanded={filtersOpen}>
              <SlidersHorizontal size={14} />{activeFilterCount > 0 && <span>{activeFilterCount}</span>}
            </button>
          </div>
          <div className={styles.mediaFilters}>
            {(["all", "video", "image", "audio"] as MediaFilter[]).map((filter) => (
              <button key={filter} className={mediaFilter === filter ? styles.activeFilter : ""} onClick={() => setMediaFilter(filter)}>
                {filter === "all" ? "All" : filter[0].toUpperCase() + filter.slice(1)}
              </button>
            ))}
          </div>
          {filtersOpen && (
            <div className={styles.assetFilterPanel}>
              <label className={styles.mobileAssetSearch}>Search<input type="search" value={mediaQuery} onChange={(event) => setMediaQuery(event.target.value)} placeholder="Prompt or filename" /></label>
              <label>Media<select value={mediaFilter} onChange={(event) => setMediaFilter(event.target.value as MediaFilter)}><option value="all">All media</option><option value="image">Images</option><option value="video">Video</option><option value="audio">Audio</option></select></label>
              <label>Source<select value={assetFilters.scope} onChange={(event) => setAssetFilters((current) => ({ ...current, scope: event.target.value as AssetScope }))}><option value="all">All sources</option><option value="gallery">Gallery</option><option value="import">Imported</option><option value="generation">Studio takes</option></select></label>
              <div className={styles.filterPair}><label>From<input type="date" value={assetFilters.dateFrom} onChange={(event) => setAssetFilters((current) => ({ ...current, dateFrom: event.target.value }))} /></label><label>To<input type="date" value={assetFilters.dateTo} onChange={(event) => setAssetFilters((current) => ({ ...current, dateTo: event.target.value }))} /></label></div>
              <span className={styles.filterGroupLabel}>Resolution</span>
              <div className={styles.filterPair}><label>Min W<input type="number" min="0" value={assetFilters.widthMin} onChange={(event) => setAssetFilters((current) => ({ ...current, widthMin: event.target.value }))} /></label><label>Max W<input type="number" min="0" value={assetFilters.widthMax} onChange={(event) => setAssetFilters((current) => ({ ...current, widthMax: event.target.value }))} /></label></div>
              <div className={styles.filterPair}><label>Min H<input type="number" min="0" value={assetFilters.heightMin} onChange={(event) => setAssetFilters((current) => ({ ...current, heightMin: event.target.value }))} /></label><label>Max H<input type="number" min="0" value={assetFilters.heightMax} onChange={(event) => setAssetFilters((current) => ({ ...current, heightMax: event.target.value }))} /></label></div>
              <button className={styles.resetFilters} onClick={() => setAssetFilters(EMPTY_ASSET_FILTERS)} disabled={!activeFilterCount}><RotateCcw size={13} /> Reset filters</button>
            </div>
          )}
          <div className={styles.assetGrid}>
            {libraryLoading && !filteredAssets.length && <div className={styles.emptyLibrary}>Loading library…</div>}
            {!libraryLoading && !filteredAssets.length && (
              <button className={styles.emptyLibrary} onClick={() => fileInputRef.current?.click()}>
                <FolderOpen size={24} /><span>Import media or generate a clip to begin.</span>
              </button>
            )}
            {filteredAssets.map((asset) => (
              <button
                key={asset.id}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = "copy";
                  event.dataTransfer.setData("application/x-studio-asset", asset.id);
                  event.dataTransfer.setData("application/x-studio-frame-time", String(playhead));
                  if (event.shiftKey && asset.kind !== "audio") {
                    event.dataTransfer.setData("application/x-studio-frame", asset.id);
                    event.dataTransfer.setData("application/x-studio-input-mode", "frame");
                  } else if ((event.ctrlKey || event.metaKey || event.altKey) && asset.kind === "image") {
                    event.dataTransfer.setData("application/x-studio-hold-still", "1");
                  }
                }}
                onClick={() => { selectAsset(asset); setSelectedClipId(null); }}
                onDoubleClick={() => { void hydrateGalleryAsset(asset).then((hydrated) => addAssetToTimeline(hydrated)); }}
                className={`${styles.assetCard} ${selectedAssetId === asset.id && !selectedClipId ? styles.selectedAsset : ""}`}
                title={`${asset.name} — double-click to add to timeline`}
              >
                <span className={styles.assetThumb}>
                  {asset.thumbnailUrl ? <NextImage src={asset.thumbnailUrl} alt="" fill sizes="110px" unoptimized /> : asset.kind === "audio" ? <AudioLines size={24} /> : <Film size={24} />}
                  <span className={styles.assetKind}>{asset.kind === "video" ? <Film size={11} /> : asset.kind === "image" ? <ImageIcon size={11} /> : <AudioLines size={11} />}</span>
                  <span className={styles.assetDuration}>{asset.kind === "image" ? "STILL · 1F" : `${asset.duration.toFixed(1)}s`}</span>
                </span>
                <span className={styles.assetName}>{asset.name}</span>
              </button>
            ))}
          </div>
          {(assetFilters.scope === "all" || assetFilters.scope === "gallery") && galleryAssets.length < galleryTotal && (
            <button className={styles.loadMoreButton} onClick={() => void loadGalleryPage(galleryAssets.length)} disabled={loadingMore}>{loadingMore ? "Loading…" : `Load more · ${galleryAssets.length} of ${galleryTotal}`}</button>
          )}
          <button className={styles.importButton} onClick={() => fileInputRef.current?.click()}><Download size={15} /> Import media</button>
        </aside>

        <section className={styles.center}>
          <div className={styles.previewShell}>
            <div className={styles.previewToolbar}>
              <span>{previewAsset ? previewAsset.name : "Sequence preview"}</span>
              <div><button>Fit</button><button>Proxy</button><button aria-label="Fullscreen preview"><Maximize2 size={14} /></button></div>
            </div>
            <div className={styles.preview} onClick={togglePlayback}>
              {previewAsset?.url && previewAsset.kind === "video" ? (
                <video ref={videoRef} src={previewAsset.url} muted={previewTrack?.muted ?? false} playsInline onLoadedMetadata={(event) => { if (previewClip) event.currentTarget.currentTime = previewClip.sourceIn + Math.max(0, playhead - previewClip.start); if (playing) void event.currentTarget.play(); }} />
              ) : previewAsset?.url && previewAsset.kind === "image" ? (
                <NextImage src={previewAsset.url} alt={previewAsset.name} fill sizes="(max-width: 1180px) 55vw, 45vw" unoptimized />
              ) : previewAsset?.url && previewAsset.kind === "audio" ? (
                <div className={styles.audioPreview}><AudioLines size={50} /><span>{previewAsset.name}</span><audio ref={audioRef} src={previewAsset.url} muted={previewTrack?.muted ?? false} onLoadedMetadata={(event) => { if (previewClip) event.currentTarget.currentTime = previewClip.sourceIn + Math.max(0, playhead - previewClip.start); if (playing) void event.currentTarget.play(); }} /></div>
              ) : (
                <div className={styles.emptyPreview}>
                  <Wand2 size={34} />
                  <strong>Build your sequence</strong>
                  <span>Drag media to the timeline or generate a new clip.</span>
                </div>
              )}
              {range && <div className={styles.rangeReadout}>OUTPUT {formatTimecode(range.start, project.fps)} — {formatTimecode(range.end, project.fps)}</div>}
            </div>
            <div className={styles.transport}>
              <span>{formatTimecode(playhead, project.fps)} <small>/ {formatTimecode(project.duration, project.fps)}</small></span>
              <div>
                <button onClick={() => setPlayhead(0)} aria-label="Go to start"><RotateCcw size={15} /></button>
                <button onClick={(event) => { event.stopPropagation(); togglePlayback(); }} className={styles.playButton} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={19} /> : <Play size={19} fill="currentColor" />}</button>
                <button onClick={() => setPlayhead(project.duration)} aria-label="Go to end"><ChevronRight size={17} /></button>
              </div>
              <div className={styles.volume}><Volume2 size={15} /><span /><Maximize2 size={15} /></div>
            </div>
          </div>

          <div className={styles.timelineShell}>
            <div className={styles.timelineToolbar}>
              <div className={styles.toolGroup}>
                {([
                  ["select", MousePointer2, "Select (V)"],
                  ["blade", Scissors, "Blade (S)"],
                  ["hand", Hand, "Hand"],
                  ["range", MousePointerSquareDashed, "Range"],
                ] as const).map(([name, Icon, label]) => (
                  <button key={name} className={tool === name ? styles.activeTool : ""} onClick={() => setTool(name)} title={label} aria-label={label}><Icon size={16} /></button>
                ))}
                <button disabled title="Linked A/V editing needs stream-aware media import" aria-label="Link clips (not available yet)"><Link2 size={16} /></button>
                <span className={styles.toolbarDivider} />
                <button onClick={() => splitSelectedClip()} title="Split selected clip"><Scissors size={15} /></button>
                <button onClick={deleteSelectedClip} disabled={!selectedClipId} title="Delete selected clip"><Trash2 size={15} /></button>
                <button className={snapEnabled ? styles.snapActive : ""} onClick={() => setSnapEnabled((value) => !value)} title="Toggle snapping"><Magnet size={15} /></button>
              </div>
              <div className={styles.zoomControls}>
                <ZoomOut size={14} /><input aria-label="Timeline zoom" type="range" min="8" max="48" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><ZoomIn size={14} />
              </div>
            </div>

            <div className={styles.timeline}>
              <div className={styles.trackHeaders}>
                <div className={styles.rulerHeader}>TRACKS</div>
                {project.tracks.map((track) => (
                  <div className={styles.trackHeader} key={track.id}>
                    <button onClick={() => toggleTrack(track.id, "visible")} aria-label={`Toggle ${track.name} visibility`}>{track.visible ? <Eye size={14} /> : <EyeOff size={14} />}</button>
                    <strong>{track.name}</strong>
                    <button onClick={() => toggleTrack(track.id, "muted")} aria-label={`Mute ${track.name}`}>{track.muted ? <VolumeX size={13} /> : <Volume2 size={13} />}</button>
                    <button onClick={() => toggleTrack(track.id, "locked")} aria-label={`Lock ${track.name}`}>{track.locked ? <Lock size={13} /> : <Unlock size={13} />}</button>
                  </div>
                ))}
              </div>
              <div className={styles.timelineScroll}>
                <div className={styles.timelineContent} style={{ width: Math.max(project.duration * zoom, 920) }}>
                  <div className={styles.ruler} onPointerDown={beginRange}>
                    {Array.from({ length: Math.ceil(project.duration / 5) + 1 }, (_, index) => (
                      <span key={index} style={{ left: index * 5 * zoom }}>{formatTimecode(index * 5, project.fps).slice(3, 8)}</span>
                    ))}
                    {range && (
                      <div className={styles.outputRange} style={{ left: range.start * zoom, width: Math.max(2, (range.end - range.start) * zoom) }}>
                        <span>OUTPUT RANGE</span>
                      </div>
                    )}
                    {inpaintRange && (
                      <div className={styles.inpaintRange} style={{ left: inpaintRange.start * zoom, width: Math.max(2, (inpaintRange.end - inpaintRange.start) * zoom) }}>
                        <span>EDIT RANGE</span>
                      </div>
                    )}
                  </div>
                  {project.tracks.map((track) => (
                    <div
                      key={track.id}
                      data-studio-track-id={track.id}
                      className={`${styles.trackLane} ${track.kind === "audio" ? styles.audioLane : ""}`}
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => handleTrackDrop(event, track.id)}
                      onClick={(event) => {
                        if (event.target === event.currentTarget) {
                          const bounds = event.currentTarget.getBoundingClientRect();
                          setPlayhead(Math.max(0, Math.min(project.duration, (event.clientX - bounds.left) / zoom)));
                          setSelectedClipId(null);
                          setSelectedAssetId(null);
                        }
                      }}
                    >
                      {track.visible && activeClips.filter((clip) => clip.trackId === track.id).map((clip) => {
                        const asset = allAssets.find((item) => item.id === clip.assetId);
                        return (
                          <div
                            key={clip.id}
                            role="button"
                            tabIndex={0}
                            aria-label={`${clip.name}, ${track.name}, ${formatTimecode(clip.start, project.fps)} to ${formatTimecode(clip.start + clip.duration, project.fps)}${clip.generated ? ", generated take" : ""}`}
                            onPointerDown={(event) => beginClipMove(event, clip)}
                            onMouseEnter={() => setHoveredClipId(clip.id)}
                            onMouseLeave={() => setHoveredClipId((current) => current === clip.id ? null : current)}
                            onDoubleClick={(event) => {
                              event.stopPropagation();
                              if (asset?.kind === "image") openImageEditor(asset, imageInputMode === "inpaint" ? "inpaint" : "edit");
                            }}
                            onClick={(event) => {
                              event.stopPropagation();
                              if (suppressClipClickRef.current === clip.id) {
                                suppressClipClickRef.current = null;
                                return;
                              }
                              setSelectedClipId(clip.id);
                              setSelectedAssetId(clip.assetId);
                              setRightPane("inspector");
                              if (tool === "blade") splitSelectedClip(clip);
                            }}
                            onKeyDown={(event) => {
                              if (event.key !== "Enter") return;
                              event.preventDefault();
                              setSelectedClipId(clip.id);
                              setSelectedAssetId(clip.assetId);
                              setRightPane("inspector");
                            }}
                            className={`${styles.timelineClip} ${asset?.kind === "audio" ? styles.audioClip : ""} ${clip.generated ? styles.generatedClip : ""} ${selectedClipId === clip.id ? styles.selectedClip : ""} ${selectedClipId && selectedClipId !== clip.id ? styles.dimmedClip : ""} ${clip.presentation === "frame" ? styles.stillClip : ""}`}
                            style={{ left: clip.start * zoom, width: Math.max(18, clip.duration * zoom), backgroundImage: asset?.thumbnailUrl && asset.kind !== "audio" ? `linear-gradient(90deg, rgba(8,12,18,.42), rgba(8,12,18,.08)), url(${asset.thumbnailUrl})` : undefined }}
                          >
                            <button className={styles.trimStart} onPointerDown={(event) => beginTrim(event, clip, "start")} aria-label={`Trim start of ${clip.name}`} />
                            <span className={styles.clipName}>{clip.name}</span>
                            <span className={styles.clipInputControls} onPointerDown={(event) => event.stopPropagation()}>
                              {asset && asset.kind !== "audio" && (
                                <label title="Use as keyframe">
                                  <input type="checkbox" checked={clip.inputRoles?.includes("keyframe") || false} onChange={() => toggleClipInputRole(clip.id, "keyframe")} />
                                  <span>K</span>
                                </label>
                              )}
                              <label title="Use as explicit reference">
                                <input type="checkbox" checked={clip.inputRoles?.includes("reference") || false} onChange={() => toggleClipInputRole(clip.id, "reference")} />
                                <span>R</span>
                              </label>
                            </span>
                            <span
                              className={styles.clipSourceHandle}
                              draggable={!track.locked}
                              onPointerDown={(event) => event.stopPropagation()}
                              onDragStart={(event) => {
                                event.dataTransfer.effectAllowed = "copy";
                                event.dataTransfer.setData("application/x-studio-clip", clip.id);
                                event.dataTransfer.setData("application/x-studio-frame-time", String(frameTimeForClip(clip, playhead)));
                                if (event.shiftKey) event.dataTransfer.setData("application/x-studio-input-mode", "frame");
                              }}
                              title="Drag to Generate; Shift: current frame"
                            ><ImagePlus size={10} /></span>
                            {clip.linkGroupId && <Link2 size={11} className={styles.linkBadge} />}
                            {clip.generated && <Sparkles size={11} className={styles.generationBadge} />}
                            {asset?.kind === "audio" && <span className={styles.waveform} />}
                            <button className={styles.trimEnd} onPointerDown={(event) => beginTrim(event, clip, "end")} aria-label={`Trim end of ${clip.name}`} />
                            {hoveredClipId === clip.id && asset?.kind === "image" && (
                              <span className={styles.stillPopover} role="tooltip">
                                <img src={asset.url || asset.thumbnailUrl || ""} alt="" />
                                <strong>{asset.name}</strong>
                                <small>{asset.width && asset.height ? `${asset.width}×${asset.height}` : "Still image"} · 1 frame</small>
                              </span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}
                  <div className={styles.playhead} style={{ left: playhead * zoom }}><span /></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <aside className={styles.rightPane}>
          <div className={styles.paneTabs}>
            {(["generate", "inspector", "jobs"] as StudioPane[]).map((pane) => (
              <button key={pane} className={rightPane === pane ? styles.activePane : ""} onClick={() => setRightPane(pane)}>
                {pane}{pane === "jobs" && jobs.some((job) => job.status === "running") && <span className={styles.liveDot} />}
              </button>
            ))}
          </div>

          {rightPane === "generate" && (
            <div
              className={`${styles.paneBody} ${generationDropActive ? styles.generationDropActive : ""}`}
              data-studio-generation-drop
              onDragOver={(event) => { event.preventDefault(); setGenerationDropActive(true); }}
              onDragLeave={() => setGenerationDropActive(false)}
              onDrop={async (event) => { event.preventDefault(); setGenerationDropActive(false); setFrameDropLoading(true); try { await handleRightPaneDrop(event); } finally { setFrameDropLoading(false); } }}
            >
              <section className={styles.modelCard}>
                <span className={styles.eyebrow}>{isVideoModel ? "VIDEO MODEL" : "IMAGE MODEL"}</span>
                <div className={styles.modelLine}><strong>{currentModelName}</strong><span className={isBackendReady ? styles.ready : styles.unavailable}>{isBackendReady ? "READY" : "OFFLINE"}</span></div>
                <small>{loadedArch || "No architecture"}</small>
              </section>
              <section className={styles.resolvedModeCard}>
                <span className={styles.eyebrow}>RESOLVED WORKFLOW</span>
                <strong>{resolvedModeLabel[resolvedMode]}</strong>
                <small>Selected from ranges and explicit timeline inputs.</small>
              </section>
              <section className={styles.rangeControls}>
                <div className={styles.sectionTitle}><span>Generation ranges</span><small>I/O · Alt+I/O</small></div>
                <div className={styles.rangeButtons}>
                  <button className={rangeTarget === "output" ? styles.activeRangeTarget : ""} onClick={() => { setRangeTarget("output"); setTool("range"); }}><span>Output</span><small>{range ? `${formatTimecode(range.start, project.fps)} – ${formatTimecode(range.end, project.fps)}` : "Unset"}</small></button>
                  <button className={rangeTarget === "inpaint" ? styles.activeRangeTarget : ""} onClick={() => { setRangeTarget("inpaint"); setTool("range"); }}><span>Edit / inpaint</span><small>{inpaintRange ? `${formatTimecode(inpaintRange.start, project.fps)} – ${formatTimecode(inpaintRange.end, project.fps)}` : "Unset"}</small></button>
                </div>
                {(range || inpaintRange) && <button className={styles.clearRanges} onClick={() => { setRange(null); setInpaintRange(null); }}>Clear ranges</button>}
              </section>
              <section>
                <label className={styles.fieldLabel} htmlFor="studio-prompt">Prompt</label>
                <div className={styles.promptBox}>
                  <textarea id="studio-prompt" value={form.prompt} onChange={(event) => setForm((current) => ({ ...current, prompt: event.target.value }))} placeholder="Describe the shot, camera motion, lighting, and continuity…" maxLength={1000} />
                  <span>{form.prompt.length}/1000</span>
                </div>
              </section>
              {isVideoModel && (
                <section>
                  <label className={styles.fieldLabel}>Timeline inputs</label>
                  {activeClips.filter((clip) => clip.inputRoles?.includes("keyframe")).map((clip) => <span key={clip.id}><Check size={11} /> {clip.name}</span>)}
                  {selectedAsset?.kind !== "image" && !activeClips.some((clip) => clip.inputRoles?.includes("keyframe")) && <small>Tick K on a timeline clip to use it as a keyframe.</small>}
                  {selectedAsset?.kind === "image" && false ? (
                    <button className={styles.keyframeSlot} onClick={() => setSelectedAssetId(null)}>
                      <NextImage src={selectedAsset?.thumbnailUrl || selectedAsset?.url || ""} alt="" width={74} height={48} unoptimized />
                      <span><strong>{selectedAsset.name}</strong><small>Selected image · click to clear</small></span>
                    </button>
                  ) : (
                    <button className={styles.emptySlot} onClick={() => { setMediaFilter("image"); setNotice("Select an image from Media."); }}><Plus size={18} /> Select image</button>
                  )}
                </section>
              )}
              {selectedAsset?.kind === "image" && (
                <section className={styles.inputCard}>
                  <div className={styles.sectionTitle}><span>{isVideoModel ? "Image keyframe" : "Input image"}</span><button onClick={() => { setSelectedAssetId(null); setImageInputMode("i2i"); }}><X size={12} /></button></div>
                  <div className={styles.keyframeSlot}><NextImage src={selectedAsset.thumbnailUrl || selectedAsset.url} alt="" width={74} height={48} unoptimized /><span><strong>{selectedAsset.name}</strong><small>{isVideoModel ? "I2VA anchor" : imageInputMode === "inpaint" ? "Mask enabled" : "I2I input"}</small></span></div>
                  {!isVideoModel && <div className={styles.inputActions}><button onClick={() => openImageEditor(selectedAsset, "edit")}><ImagePlus size={13} /> Edit image</button><button onClick={() => { setImageInputMode("inpaint"); openImageEditor(selectedAsset, "inpaint"); }}>Mask / inpaint</button></div>}
                </section>
              )}
              <section className={styles.referenceDropCard}>
                <div className={styles.sectionTitle}><span>Explicit references</span><small>drag here or use R</small></div>
                <div className={styles.referenceList}>{referenceAssetIds.map((assetId) => { const asset = allAssets.find((item) => item.id === assetId); return asset ? <button key={assetId} onClick={() => setReferenceAssetIds((current) => current.filter((id) => id !== assetId))}>{asset.name}<X size={11} /></button> : null; })}{!referenceAssetIds.length && <small>References are never inferred from clips.</small>}</div>
                {selectedAsset && <button className={styles.addReferenceButton} onClick={() => setReferenceAssetIds((current) => current.includes(selectedAsset.id) ? current : [...current, selectedAsset.id])}>Add selected media as reference</button>}
              </section>
              <details className={styles.advancedPrompt}>
                <summary><ChevronRight size={14} /> Negative prompt {!supportsNegativePrompt && <small>Not supported by {loadedArch}</small>}</summary>
                <textarea disabled={!supportsNegativePrompt} value={form.negativePrompt} onChange={(event) => setForm((current) => ({ ...current, negativePrompt: event.target.value }))} />
              </details>
              {range && (
                <section className={styles.rangeCard}>
                  <div><MousePointerSquareDashed size={15} /><strong>Timeline output range</strong></div>
                  <span>{formatTimecode(range.start, project.fps)} — {formatTimecode(range.end, project.fps)}</span>
                  <button onClick={() => setRange(null)}>Clear</button>
                </section>
              )}
              <div className={styles.settingsGrid}>
                <label>Width<input type="number" value={form.width ?? ""} onChange={(event) => setForm((current) => ({ ...current, width: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                <label>Height<input type="number" value={form.height ?? ""} onChange={(event) => setForm((current) => ({ ...current, height: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                {isVideoModel && <label>Clip length<select value={form.numFrames ?? ""} onChange={(event) => setForm((current) => ({ ...current, numFrames: Number(event.target.value) }))} disabled={!generationDefaults}>{frameOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select></label>}
                {isVideoModel && <label>Frame rate<input type="number" value={form.frameRate ?? ""} onChange={(event) => setForm((current) => ({ ...current, frameRate: Number(event.target.value) }))} disabled={!generationDefaults} /></label>}
                <label>Seed<input type="number" value={form.seed ?? ""} onChange={(event) => setForm((current) => ({ ...current, seed: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                <label>Steps<input type="number" value={form.steps ?? ""} onChange={(event) => setForm((current) => ({ ...current, steps: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                {!isVideoModel && <label>Denoise<input type="number" min="0" max="1" step="0.05" value={form.denoisingStrength ?? 0.75} onChange={(event) => setForm((current) => ({ ...current, denoisingStrength: Number(event.target.value) }))} /></label>}
                {!isVideoModel && <label>Sampler<input value={form.sampler ?? "euler"} onChange={(event) => setForm((current) => ({ ...current, sampler: event.target.value }))} /></label>}
              </div>
              <label className={styles.sliderField}><span>Guidance <strong>{supportsGuidance ? form.guidance ?? "—" : `Fixed by ${loadedArch}`}</strong></span><input type="range" min="0" max="20" step="0.1" value={form.guidance ?? 0} onChange={(event) => setForm((current) => ({ ...current, guidance: Number(event.target.value) }))} disabled={!generationDefaults || !supportsGuidance} /></label>
              {isVideoModel && <label className={styles.toggleField}><span><AudioLines size={15} /> Generate audio jointly</span><input type="checkbox" checked={form.audioEnable ?? false} onChange={(event) => setForm((current) => ({ ...current, audioEnable: event.target.checked }))} disabled={!generationDefaults} /></label>}
              {(notice || (!isBackendReady ? "Generation schema is unavailable. Start the backend to enable AI generation." : null)) && <div className={styles.notice}><AlertCircle size={15} /><span>{notice || "Generation schema is unavailable. Start the backend to enable AI generation."}</span><button onClick={() => setNotice(null)}>×</button></div>}
              <button className={styles.generateButton} onClick={generateClip} disabled={jobs.some((job) => job.status === "running")}><Sparkles size={17} /> Generate {isVideoModel ? "video" : "image"} {outputDuration > 0 && isVideoModel && <small>{outputDuration.toFixed(1)}s</small>}</button>
              <section className={styles.resultsShelf}>
                <div className={styles.sectionTitle}><span>Generation results</span><button onClick={() => setMediaFilter("generation")}>See all</button></div>
                <div className={styles.resultGrid}>
                  {resultAssetIds.length === 0 && <div className={styles.emptyResults}>New takes appear here and remain reusable.</div>}
                  {resultAssetIds.map((assetId) => {
                    const asset = project.assets.find((item) => item.id === assetId);
                    if (!asset) return null;
                    return <button key={asset.id} onClick={() => selectAsset(asset)} onDoubleClick={() => void hydrateGalleryAsset(asset).then((hydrated) => addAssetToTimeline(hydrated))}><NextImage src={asset.thumbnailUrl || asset.url} alt="" fill sizes="105px" unoptimized /><span>{asset.duration.toFixed(1)}s</span></button>;
                  })}
                </div>
              </section>
            </div>
          )}

          {rightPane === "inspector" && (
            <div className={styles.paneBody}>
              {selectedClip ? (
                <>
                  <section className={styles.inspectorTitle}><span className={styles.eyebrow}>SELECTED CLIP</span><strong>{selectedClip.name}</strong><small>{takeCount} take{takeCount === 1 ? "" : "s"} · {selectedClip.generated ? "Generated" : "Source"}</small></section>
                  {takeAlternatives.length > 1 && (
                    <section className={styles.takeSection}>
                      <div className={styles.sectionTitle}><span>Non-destructive takes</span><small>{takeAlternatives.length}</small></div>
                      <div className={styles.takeList}>
                        {takeAlternatives.map((take, index) => (
                          <button key={take.id} className={take.activeTake !== false ? styles.activeTake : ""} onClick={() => activateTake(take.id)}>
                            <span>{take.generated ? `Take ${index}` : "Source"}</span>
                            <small>{take.duration.toFixed(1)}s</small>
                          </button>
                        ))}
                      </div>
                    </section>
                  )}
                  <div className={styles.inspectorFields}>
                    <label>Timeline start<input type="number" step="0.04" min="0" max={project.duration - selectedClip.duration} value={selectedClip.start} onChange={(event) => updateSelectedClip({ start: Number(event.target.value) })} /></label>
                    <label>Duration<input type="number" step="0.04" min="0.1" max={project.duration - selectedClip.start} value={selectedClip.duration} onChange={(event) => updateSelectedClip({ duration: Number(event.target.value) })} /></label>
                    <label>Source in<input type="number" step="0.04" min="0" value={selectedClip.sourceIn} onChange={(event) => updateSelectedClip({ sourceIn: Number(event.target.value) })} /></label>
                  </div>
                  <section className={styles.inspectorSection}><div><strong>Link group</strong><small>Stream-aware linked A/V editing is planned for the backend media-import phase.</small></div><button disabled><Link2 size={14} /> Unavailable</button></section>
                  <section className={styles.inspectorSection}><div><strong>Generation context</strong><small>Use this clip as the next shot&apos;s visual context.</small></div><button onClick={() => { setMode(selectedAsset?.kind === "image" ? "i2v" : "t2v"); setRightPane("generate"); }}>Generate from clip</button></section>
                  <button className={styles.dangerButton} onClick={deleteSelectedClip}><Trash2 size={15} /> Delete clip</button>
                </>
              ) : (
                <div className={styles.emptyInspector}><MousePointer2 size={28} /><strong>No clip selected</strong><span>Select a timeline clip to edit timing, link state, and generation context.</span></div>
              )}
            </div>
          )}

          {rightPane === "jobs" && (
            <div className={styles.paneBody} aria-live="polite">
              <div className={styles.sectionTitle}><span>Generation jobs</span><small>{jobs.length}</small></div>
              {!jobs.length && <div className={styles.emptyInspector}><Archive size={28} /><strong>No Studio jobs yet</strong><span>Generation runs continue while you edit the timeline.</span></div>}
              <div className={styles.jobList}>
                {jobs.map((job) => (
                  <article key={job.id} className={styles.jobCard}>
                    <div className={styles.jobHeader}><span className={`${styles.jobStatus} ${styles[job.status]}`}>{job.status}</span><small><Clock3 size={12} /> {new Date(job.startedAt).toLocaleTimeString()}</small></div>
                    <strong>{job.mode.toUpperCase()} · {job.prompt}</strong>
                    {job.status === "running" && <div className={styles.indeterminate}><span /></div>}
                    {job.error && <p>{job.error}</p>}
                    <div className={styles.jobActions}>
                      {job.status === "review" && <button onClick={() => applyJob(job)}>Apply take</button>}
                      <button onClick={() => restoreRecipe(job)}>Restore recipe</button>
                      {job.status === "running" && <button onClick={() => cancelJob(job.id)}>Cancel</button>}
                    </div>
                  </article>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
      {imageEditorState && (() => {
        const asset = allAssets.find((item) => item.id === imageEditorState.assetId);
        if (!asset) return null;
        return (
          <div className={styles.editorOverlay} role="dialog" aria-modal="true" aria-label="Studio image editor">
            <ImageEditor
              imageUrl={asset.url}
              mode={imageEditorState.mode}
              initialMaskUrl={asset.maskUrl}
              onSave={saveStudioEditedImage}
              onSaveMask={(maskUrl) => { pendingImageMaskRef.current = maskUrl; }}
              onClose={() => { setImageEditorState(null); pendingImageMaskRef.current = undefined; }}
            />
          </div>
        );
      })()}
    </main>
  );
}
