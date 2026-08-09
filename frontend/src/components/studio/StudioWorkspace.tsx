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
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import {
  GeneratedImage,
  archSupportsFeature,
  cancelGeneration,
  generateImg2Vid,
  generateTxt2Vid,
  getImage,
  getImages,
  getResultFilename,
  getResultPlaybackFilename,
  videoFrameOptions,
} from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { loadStudioProject, saveImportedMedia, saveStudioProject } from "./studioStorage";
import { resolveStudioTransferUrl, takeStudioTransfer, type StudioTransferPayload } from "./studioTransfer";
import {
  StudioAsset,
  StudioClip,
  StudioJob,
  StudioMode,
  StudioPane,
  StudioProject,
  StudioRange,
  StudioTool,
  createStudioProject,
} from "./types";
import styles from "./studio.module.css";

interface VideoFormState {
  prompt: string;
  negativePrompt: string;
  width?: number;
  height?: number;
  numFrames?: number;
  frameRate?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  audioEnable?: boolean;
}

type MediaFilter = "all" | "image" | "video" | "audio";
type AssetScope = "all" | "gallery" | "import" | "generation";

interface AssetFilters {
  scope: AssetScope;
  dateFrom: string;
  dateTo: string;
  widthMin: string;
  widthMax: string;
  heightMin: string;
  heightMax: string;
}

const EMPTY_FORM: VideoFormState = { prompt: "", negativePrompt: "" };
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

const safeModelLabel = (value: unknown): string => {
  const raw = String(value || "No model loaded");
  return raw.split(/[\\/]/).filter(Boolean).at(-1) || "No model loaded";
};

const formatTimecode = (seconds: number, fps = 24) => {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const wholeSeconds = Math.floor(safe % 60);
  const frames = Math.floor((safe - Math.floor(safe)) * fps);
  return [hours, minutes, wholeSeconds, frames]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
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
    duration: Number.isFinite(parsedDuration) && parsedDuration > 0 ? parsedDuration : (kind === "image" ? 5 : 6),
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
      image.onload = () => resolve({ duration: 5, width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => resolve({ duration: 5 });
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
  const [form, setForm] = useState<VideoFormState>(EMPTY_FORM);
  const [playhead, setPlayhead] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [zoom, setZoom] = useState(18);
  const [range, setRange] = useState<StudioRange | null>(null);
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
          setJobs(restoredJobs);
          setResultAssetIds(restoredJobs.flatMap((job) => job.assetId ? [job.assetId] : []));
        }
      })
      .finally(() => setRestored(true));
  }, []);

  useEffect(() => {
    setPendingTransfer(takeStudioTransfer());
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
          duration: Number.isFinite(duration) && duration > 0 ? duration : (media.kind === "image" ? 5 : 6),
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
    const timer = window.setTimeout(() => saveStudioProject(project), 350);
    return () => window.clearTimeout(timer);
  }, [project, restored]);

  useEffect(() => {
    if (!restored) return;
    const saveOnExit = () => saveStudioProject(project);
    window.addEventListener("pagehide", saveOnExit);
    return () => window.removeEventListener("pagehide", saveOnExit);
  }, [project, restored]);

  useEffect(() => {
    if (!generationDefaults || !modelInfo?.type) return;
    const identity = `${modelInfo.type}:${modelInfo.variant || ""}`;
    if (initializedDefaultsForArchRef.current === identity) return;
    initializedDefaultsForArchRef.current = identity;
    setDefaultsIdentity(identity);
    const resolved = {
      ...(generationDefaults.txt2vid || {}),
      ...(generationDefaults.video_arch_overlays?.[modelInfo.type] || {}),
    };
    setForm((current) => ({
      ...current,
      width: numeric(resolved.width),
      height: numeric(resolved.height),
      numFrames: numeric(resolved.num_frames),
      frameRate: numeric(resolved.frame_rate),
      steps: numeric(resolved.num_inference_steps),
      guidance: numeric(resolved.guidance_scale),
      seed: numeric(resolved.seed),
      audioEnable: booleanValue(resolved.audio_enable),
    }));
  }, [generationDefaults, modelInfo?.type, modelInfo?.variant]);

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
  const studioModesAvailable = !(loadedArch === "minimax_h3" && modelInfo?.variant === "ref2va");
  const frameOptions = videoFrameOptions(archCapabilities, loadedArch, form.numFrames);
  const supportsNegativePrompt = archSupportsFeature(archCapabilities, loadedArch, "negative_prompt");
  const supportsGuidance = archSupportsFeature(archCapabilities, loadedArch, "cfg");

  const hydrateGalleryAsset = useCallback(async (asset: StudioAsset): Promise<StudioAsset> => {
    if (asset.source !== "gallery" || asset.kind === "image" || asset.galleryId == null) return asset;
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

  const addAssetToTimeline = useCallback((asset: StudioAsset, start?: number, trackId?: string) => {
    const targetTrack =
      project.tracks.find((track) => track.id === trackId && track.kind === (asset.kind === "audio" ? "audio" : "video")) ||
      project.tracks.find((track) => track.kind === (asset.kind === "audio" ? "audio" : "video"));
    if (!targetTrack || targetTrack.locked) return;

    const trackEnd = activeClips
      .filter((clip) => clip.trackId === targetTrack.id)
      .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
    const clipStart = Math.max(0, Math.min(start ?? trackEnd, project.duration - 0.1));
    const duration = Math.max(0.1, Math.min(asset.duration || 5, project.duration - clipStart));
    const clip: StudioClip = {
      id: crypto.randomUUID(),
      assetId: asset.id,
      trackId: targetTrack.id,
      name: asset.name,
      start: clipStart,
      duration,
      sourceIn: 0,
    };
    commit((current) => ({
      ...current,
      assets: current.assets.some((item) => item.id === asset.id) ? current.assets : [...current.assets, asset],
      clips: [...current.clips, clip],
    }));
    setSelectedAssetId(asset.id);
    setSelectedClipId(clip.id);
  }, [activeClips, commit, project.duration, project.tracks]);

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
    const start = (event.clientX - bounds.left) / zoom;
    const clipId = event.dataTransfer.getData("application/x-studio-clip");
    if (clipId) {
      moveClip(clipId, trackId, start);
      return;
    }
    const assetId = event.dataTransfer.getData("application/x-studio-asset");
    const asset = allAssets.find((item) => item.id === assetId);
    if (asset) addAssetToTimeline(await hydrateGalleryAsset(asset), start, trackId);
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
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteSelectedClip, playing, project.duration, project.fps, redo, splitSelectedClip, togglePlayback, undo]);

  const beginRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool !== "range") {
      const bounds = event.currentTarget.getBoundingClientRect();
      setPlayhead(Math.max(0, Math.min(project.duration, (event.clientX - bounds.left) / zoom)));
      setSelectedClipId(null);
      setSelectedAssetId(null);
      return;
    }
    const element = event.currentTarget;
    const bounds = element.getBoundingClientRect();
    const start = Math.max(0, Math.min(project.duration, (event.clientX - bounds.left) / zoom));
    setRange({ start, end: start });
    element.setPointerCapture(event.pointerId);
    const move = (pointerEvent: PointerEvent) => {
      const current = Math.max(0, Math.min(project.duration, (pointerEvent.clientX - bounds.left) / zoom));
      setRange({ start: Math.min(start, current), end: Math.max(start, current) });
    };
    const up = () => {
      element.removeEventListener("pointermove", move);
      element.removeEventListener("pointerup", up);
      setRightPane("generate");
    };
    element.addEventListener("pointermove", move);
    element.addEventListener("pointerup", up);
  };

  const beginTrim = (event: ReactPointerEvent, clip: StudioClip, edge: "start" | "end") => {
    event.stopPropagation();
    const track = project.tracks.find((item) => item.id === clip.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before trimming this clip.`);
      return;
    }
    const originX = event.clientX;
    const snapshot = project;
    setUndoStack((history) => [...history, snapshot].slice(-MAX_HISTORY));
    setRedoStack([]);
    const move = (pointerEvent: PointerEvent) => {
      const delta = (pointerEvent.clientX - originX) / zoom;
      setProject((current) => ({
        ...current,
        clips: current.clips.map((item) => {
          if (item.id !== clip.id) return item;
          if (edge === "start") {
            const nextStart = Math.max(0, Math.min(clip.start + clip.duration - 0.1, clip.start + delta));
            return {
              ...item,
              start: nextStart,
              duration: clip.duration - (nextStart - clip.start),
              sourceIn: clip.sourceIn + (nextStart - clip.start),
            };
          }
          return { ...item, duration: Math.max(0.1, Math.min(project.duration - clip.start, clip.duration + delta)) };
        }),
        updatedAt: new Date().toISOString(),
      }));
    };
    const up = () => {
      setProject((current) => ({ ...current, revision: current.revision + 1, updatedAt: new Date().toISOString() }));
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const generateClip = async () => {
    setNotice(null);
    const modality = await resolveModality();
    if (!modality.isVideo) {
      setNotice("Load a video architecture in Generate before starting a Studio job.");
      return;
    }
    if (modality.modelInfo?.type !== loadedArch || modality.modelInfo?.variant !== modelInfo?.variant) {
      setNotice("The loaded model changed. Studio refreshed its capability defaults; review them and generate again.");
      return;
    }
    if (!studioModesAvailable) {
      setNotice("The loaded MiniMax-H3 ref2va variant requires the REF2VA workflow, which is planned for the next Studio phase.");
      return;
    }
    if (!form.prompt.trim() || !form.width || !form.height || !form.numFrames || !form.frameRate || form.steps == null || form.guidance == null || form.seed == null) {
      setNotice("Prompt and generation schema values are required.");
      return;
    }
    const keyframe = mode === "i2v"
      ? allAssets.find((asset) => asset.id === selectedAssetId && asset.kind === "image") ||
        (selectedAsset?.kind === "image" ? selectedAsset : null)
      : null;
    if (mode === "i2v" && !keyframe?.url) {
      setNotice("Select an image in Media or an image clip in the timeline for I2V(A).");
      return;
    }

    const jobId = crypto.randomUUID();
    const resolvedModelName = safeModelLabel(modality.modelInfo?.name || modality.modelInfo?.source || currentModelName);
    const recipe = {
      model: resolvedModelName,
      architecture: modality.modelInfo?.type,
      model_variant: modality.modelInfo?.variant,
      mode,
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
      output_range: range,
      keyframe_asset_id: keyframe?.id,
    };
    setJobs((current) => [{ id: jobId, mode, prompt: form.prompt, status: "running", startedAt: Date.now(), recipe }, ...current]);
    setRightPane("jobs");

    try {
      const parameters = {
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
      const result = mode === "t2v"
        ? await generateTxt2Vid(parameters)
        : await generateImg2Vid(parameters, keyframe!.url);
      const filename = getResultPlaybackFilename(result);
      const masterFilename = getResultFilename(result);
      if (!filename || !masterFilename) throw new Error("Generation completed without an output filename.");
      const galleryId = numeric(result?.image?.id);
      const assetId = galleryId != null ? `gallery-${galleryId}` : `generation-${jobId}`;
      const baseName = filename.replace(/\.[^/.]+$/, "");
      const duration = numeric(result?.image?.duration) ?? form.numFrames / form.frameRate;
      const asset: StudioAsset = {
        id: assetId,
        galleryId,
        name: filename,
        kind: "video",
        url: `/outputs/${filename}`,
        masterUrl: `/outputs/${masterFilename}`,
        thumbnailUrl: `/thumbnails/${baseName}.png`,
        duration,
        width: form.width,
        height: form.height,
        source: "generation",
        prompt: form.prompt,
        negativePrompt: supportsNegativePrompt ? form.negativePrompt : "",
        createdAt: result?.image?.created_at || new Date().toISOString(),
        generationType: result?.image?.generation_type || (mode === "t2v" ? "txt2vid" : "img2vid"),
        modelName: resolvedModelName,
        seed: numeric(result?.image?.seed) ?? form.seed,
        parameters: recipe,
      };
      const targetStart = range?.start ?? playhead;
      const targetDuration = range ? Math.max(0.1, range.end - range.start) : duration;
      const takeGroupId = selectedClip?.takeGroupId || (selectedClip ? crypto.randomUUID() : undefined);
      const clip: StudioClip = {
        id: crypto.randomUUID(),
        assetId,
        trackId: selectedClip?.trackId || "video-1",
        name: filename,
        start: selectedClip?.start ?? targetStart,
        duration: Math.min(selectedClip?.duration ?? targetDuration, duration),
        sourceIn: 0,
        takeGroupId,
        activeTake: false,
        generated: true,
      };
      commit((current) => ({
        ...current,
        assets: [...current.assets, asset],
        clips: [...current.clips.map((item) => item.id === selectedClip?.id ? { ...item, takeGroupId } : item), clip],
      }));
      setSelectedAssetId(assetId);
      setSelectedClipId(null);
      setResultAssetIds((current) => [assetId, ...current]);
      setJobs((current) => current.map((job) => job.id === jobId ? { ...job, status: "review" as const, assetId } : job));
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
      seed: numeric(recipe.seed),
      audioEnable: booleanValue(recipe.audio_enable),
    });
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
    commit((current) => ({
      ...current,
      clips: current.clips.map((clip) => clip.id === selectedClipId ? { ...clip, ...changes } : clip),
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
                onDragStart={(event) => event.dataTransfer.setData("application/x-studio-asset", asset.id)}
                onClick={() => { selectAsset(asset); setSelectedClipId(null); }}
                onDoubleClick={() => { void hydrateGalleryAsset(asset).then((hydrated) => addAssetToTimeline(hydrated)); }}
                className={`${styles.assetCard} ${selectedAssetId === asset.id && !selectedClipId ? styles.selectedAsset : ""}`}
                title={`${asset.name} — double-click to add to timeline`}
              >
                <span className={styles.assetThumb}>
                  {asset.thumbnailUrl ? <NextImage src={asset.thumbnailUrl} alt="" fill sizes="110px" unoptimized /> : asset.kind === "audio" ? <AudioLines size={24} /> : <Film size={24} />}
                  <span className={styles.assetKind}>{asset.kind === "video" ? <Film size={11} /> : asset.kind === "image" ? <ImageIcon size={11} /> : <AudioLines size={11} />}</span>
                  <span className={styles.assetDuration}>{asset.kind === "image" ? "STILL" : `${asset.duration.toFixed(1)}s`}</span>
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
                  </div>
                  {project.tracks.map((track) => (
                    <div
                      key={track.id}
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
                            draggable={!track.locked}
                            onDragStart={(event) => event.dataTransfer.setData("application/x-studio-clip", clip.id)}
                            onClick={(event) => { event.stopPropagation(); setSelectedClipId(clip.id); setSelectedAssetId(clip.assetId); setRightPane("inspector"); if (tool === "blade") splitSelectedClip(clip); }}
                            onKeyDown={(event) => {
                              if (event.key !== "Enter") return;
                              event.preventDefault();
                              setSelectedClipId(clip.id);
                              setSelectedAssetId(clip.assetId);
                              setRightPane("inspector");
                            }}
                            className={`${styles.timelineClip} ${asset?.kind === "audio" ? styles.audioClip : ""} ${clip.generated ? styles.generatedClip : ""} ${selectedClipId === clip.id ? styles.selectedClip : ""}`}
                            style={{ left: clip.start * zoom, width: Math.max(18, clip.duration * zoom), backgroundImage: asset?.thumbnailUrl && asset.kind !== "audio" ? `linear-gradient(90deg, rgba(8,12,18,.42), rgba(8,12,18,.08)), url(${asset.thumbnailUrl})` : undefined }}
                          >
                            <button className={styles.trimStart} onPointerDown={(event) => beginTrim(event, clip, "start")} aria-label={`Trim start of ${clip.name}`} />
                            <span className={styles.clipName}>{clip.name}</span>
                            {clip.linkGroupId && <Link2 size={11} className={styles.linkBadge} />}
                            {clip.generated && <Sparkles size={11} className={styles.generationBadge} />}
                            {asset?.kind === "audio" && <span className={styles.waveform} />}
                            <button className={styles.trimEnd} onPointerDown={(event) => beginTrim(event, clip, "end")} aria-label={`Trim end of ${clip.name}`} />
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
            <div className={styles.paneBody}>
              <section className={styles.modelCard}>
                <span className={styles.eyebrow}>VIDEO MODEL</span>
                <div className={styles.modelLine}><strong>{currentModelName}</strong><span className={isVideoModel && studioModesAvailable ? styles.ready : styles.unavailable}>{isVideoModel && studioModesAvailable ? "READY" : isVideoModel ? "VARIANT" : "UNAVAILABLE"}</span></div>
                <small>{loadedArch || "No architecture"}</small>
              </section>
              <section>
                <label className={styles.fieldLabel}>Mode</label>
                <div className={styles.modeTabs}>
                  <button disabled={!studioModesAvailable} className={mode === "t2v" ? styles.activeMode : ""} onClick={() => setMode("t2v")}>T2V(A)</button>
                  <button disabled={!studioModesAvailable} className={mode === "i2v" ? styles.activeMode : ""} onClick={() => setMode("i2v")}>I2V(A)</button>
                  <button disabled title="Planned: audio-conditioned generation">A2V(A)</button>
                  <button disabled title="Planned: reference workflow">REF2VA</button>
                </div>
              </section>
              <section>
                <label className={styles.fieldLabel} htmlFor="studio-prompt">Prompt</label>
                <div className={styles.promptBox}>
                  <textarea id="studio-prompt" value={form.prompt} onChange={(event) => setForm((current) => ({ ...current, prompt: event.target.value }))} placeholder="Describe the shot, camera motion, lighting, and continuity…" maxLength={1000} />
                  <span>{form.prompt.length}/1000</span>
                </div>
              </section>
              {mode === "i2v" && (
                <section>
                  <label className={styles.fieldLabel}>Start keyframe</label>
                  {selectedAsset?.kind === "image" ? (
                    <button className={styles.keyframeSlot} onClick={() => setSelectedAssetId(null)}>
                      <NextImage src={selectedAsset.thumbnailUrl || selectedAsset.url} alt="" width={74} height={48} unoptimized />
                      <span><strong>{selectedAsset.name}</strong><small>Selected image · click to clear</small></span>
                    </button>
                  ) : (
                    <button className={styles.emptySlot} onClick={() => { setMediaFilter("image"); setNotice("Select an image from Media."); }}><Plus size={18} /> Select image</button>
                  )}
                </section>
              )}
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
                <label>Clip length<select value={form.numFrames ?? ""} onChange={(event) => setForm((current) => ({ ...current, numFrames: Number(event.target.value) }))} disabled={!generationDefaults}>{frameOptions.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}</select></label>
                <label>Frame rate<input type="number" value={form.frameRate ?? ""} onChange={(event) => setForm((current) => ({ ...current, frameRate: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                <label>Seed<input type="number" value={form.seed ?? ""} onChange={(event) => setForm((current) => ({ ...current, seed: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
                <label>Steps<input type="number" value={form.steps ?? ""} onChange={(event) => setForm((current) => ({ ...current, steps: Number(event.target.value) }))} disabled={!generationDefaults} /></label>
              </div>
              <label className={styles.sliderField}><span>Guidance <strong>{supportsGuidance ? form.guidance ?? "—" : `Fixed by ${loadedArch}`}</strong></span><input type="range" min="0" max="20" step="0.1" value={form.guidance ?? 0} onChange={(event) => setForm((current) => ({ ...current, guidance: Number(event.target.value) }))} disabled={!generationDefaults || !supportsGuidance} /></label>
              <label className={styles.toggleField}><span><AudioLines size={15} /> Generate audio jointly</span><input type="checkbox" checked={form.audioEnable ?? false} onChange={(event) => setForm((current) => ({ ...current, audioEnable: event.target.checked }))} disabled={!generationDefaults} /></label>
              {(notice || (!isBackendReady ? "Generation schema is unavailable. Start the backend to enable AI generation." : null)) && <div className={styles.notice}><AlertCircle size={15} /><span>{notice || "Generation schema is unavailable. Start the backend to enable AI generation."}</span><button onClick={() => setNotice(null)}>×</button></div>}
              <button className={styles.generateButton} onClick={generateClip} disabled={!studioModesAvailable || jobs.some((job) => job.status === "running")}><Sparkles size={17} /> Generate clip {outputDuration > 0 && <small>{outputDuration.toFixed(1)}s</small>}</button>
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
    </main>
  );
}
