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
  FastForward,
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
  Rewind,
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
  cancelStudioRenderJob,
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
  getStudioRenderJob,
  getResultFilename,
  getResultPlaybackFilename,
  isValidVideoFrameCount,
  renderStudioProject,
  videoFrameOptions,
} from "@/utils/api";
import type { GenerationParams, H3PromptMode, Img2ImgParams, InpaintParams, InpaintVideoParams, MiniMaxH3References, OutpaintVideoParams, Ref2VidParams } from "@/utils/api";
import { useStartup } from "@/contexts/StartupContext";
import { formatTimecode } from "@/utils/timecode";
import { newId } from "@/utils/id";
import ImageEditor from "../common/ImageEditor";
import H3PromptAssist from "../common/H3PromptAssist";
import ModelLoadSection from "../common/ModelLoadSection";
import { loadImportedMedia, loadStudioProject, saveImportedMedia, saveStudioProject } from "./studioStorage";
import { resolveStudioTransferUrl, takeStudioTransfer, type StudioTransferPayload } from "./studioTransfer";
import {
  StudioAsset,
  StudioClip,
  StudioClipFitMode,
  StudioGenerationMode,
  StudioInputRole,
  StudioJob,
  StudioPane,
  StudioProject,
  StudioRange,
  StudioTool,
  StudioTrack,
  createStudioProject,
} from "./types";
import { clipEnd, frameDuration, frameIndexAt, maxTimelineDuration, planStudioGeneration } from "./studioTimeline";
import {
  frameTimeForClip,
  sourceTrimFrames,
  studioAssetFromGeneration,
  videoInpaintFrames,
  videoOutpaintPlacement,
} from "./studioGeneration";
import { createH3ReferenceInventory, maybeTransformH3PromptForGeneration } from "@/utils/h3PromptAssist";
import {
  parseStudioProjectFile,
  projectFileName,
  readRecentProjects,
  rememberRecentProject,
  serializeStudioProject,
  type StudioRecentProject,
} from "./studioProjectFile";
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
type SeekRepeatState = {
  timer: number | null;
  frame: number | null;
  direction: -1 | 1;
  started: boolean;
  startedAt: number;
  lastAt: number;
};

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

const stringOrUndefined = (value: unknown): string | undefined =>
  typeof value === "string" && value ? value : undefined;

// Normalize imported numeric fields before timeline arithmetic.
const normalizeImportedAsset = (raw: unknown): StudioAsset => {
  const asset = (raw && typeof raw === "object" ? raw : {}) as Partial<StudioAsset>;
  return {
    id: stringOrUndefined(asset.id) || newId(),
    galleryId: numeric(asset.galleryId),
    name: stringOrUndefined(asset.name) || "Untitled asset",
    kind: asset.kind === "video" || asset.kind === "audio" ? asset.kind : "image",
    url: typeof asset.url === "string" ? asset.url : "",
    masterUrl: stringOrUndefined(asset.masterUrl),
    thumbnailUrl: stringOrUndefined(asset.thumbnailUrl),
    maskUrl: stringOrUndefined(asset.maskUrl),
    duration: numeric(asset.duration) ?? 0,
    width: numeric(asset.width),
    height: numeric(asset.height),
    source: asset.source === "gallery" || asset.source === "import" ? asset.source : "generation",
    blobKey: stringOrUndefined(asset.blobKey),
    prompt: stringOrUndefined(asset.prompt),
    negativePrompt: stringOrUndefined(asset.negativePrompt),
    createdAt: stringOrUndefined(asset.createdAt),
    generationType: stringOrUndefined(asset.generationType),
    modelName: stringOrUndefined(asset.modelName),
    seed: numeric(asset.seed),
    parameters: asset.parameters && typeof asset.parameters === "object" ? asset.parameters as Record<string, unknown> : undefined,
    missing: booleanValue(asset.missing) ?? (!asset.url && asset.galleryId == null && !asset.blobKey),
    sourceRef: asset.sourceRef && typeof asset.sourceRef === "object"
      ? {
        name: stringOrUndefined(asset.sourceRef.name),
        size: numeric(asset.sourceRef.size),
        lastModified: numeric(asset.sourceRef.lastModified),
      }
      : undefined,
  };
};

const normalizeImportedTrack = (raw: unknown): StudioTrack => {
  const track = (raw && typeof raw === "object" ? raw : {}) as Partial<StudioTrack>;
  return {
    id: stringOrUndefined(track.id) || newId(),
    name: stringOrUndefined(track.name) || "Track",
    kind: track.kind === "audio" ? "audio" : "video",
    muted: booleanValue(track.muted) ?? false,
    locked: booleanValue(track.locked) ?? false,
    visible: booleanValue(track.visible) ?? true,
  };
};

// Drop clips without valid asset/track references and clamp timeline values.
const normalizeImportedClip = (raw: unknown, fps: number): StudioClip | null => {
  const clip = (raw && typeof raw === "object" ? raw : {}) as Partial<StudioClip>;
  const assetId = stringOrUndefined(clip.assetId);
  const trackId = stringOrUndefined(clip.trackId);
  if (!assetId || !trackId) return null;
  const minDuration = 1 / Math.max(1, fps);
  return {
    id: stringOrUndefined(clip.id) || newId(),
    assetId,
    trackId,
    name: stringOrUndefined(clip.name) || "Clip",
    start: Math.max(0, numeric(clip.start) ?? 0),
    duration: Math.max(minDuration, numeric(clip.duration) ?? 0),
    sourceIn: Math.max(0, numeric(clip.sourceIn) ?? 0),
    presentation: clip.presentation === "hold" || clip.presentation === "frame" || clip.presentation === "clip" ? clip.presentation : undefined,
    sourceDuration: numeric(clip.sourceDuration),
    fitMode: clip.fitMode === "contain" || clip.fitMode === "cover" ? clip.fitMode : undefined,
    inputRoles: Array.isArray(clip.inputRoles) ? clip.inputRoles.filter((role) => role === "keyframe") : undefined,
    linkGroupId: stringOrUndefined(clip.linkGroupId),
    takeGroupId: stringOrUndefined(clip.takeGroupId),
    activeTake: booleanValue(clip.activeTake),
    generated: booleanValue(clip.generated),
  };
};

const normalizeImportedRange = (raw: unknown): StudioRange | null => {
  if (!raw || typeof raw !== "object") return null;
  const start = numeric((raw as Partial<StudioRange>).start);
  const end = numeric((raw as Partial<StudioRange>).end);
  if (start == null || end == null) return null;
  return { start: Math.min(start, end), end: Math.max(start, end) };
};

const JOB_STATUSES: StudioJob["status"][] = ["running", "review", "failed", "applied"];
const JOB_MODES: StudioGenerationMode[] = ["t2v", "i2v", "inpaint", "outpaint", "ref2v", "t2i", "i2i", "image-inpaint"];

// Normalize imported jobs before the Jobs pane renders them.
const normalizeImportedJob = (raw: unknown): StudioJob => {
  const job = (raw && typeof raw === "object" ? raw : {}) as Partial<StudioJob>;
  return {
    id: stringOrUndefined(job.id) || newId(),
    mode: JOB_MODES.includes(job.mode as StudioGenerationMode) ? (job.mode as StudioGenerationMode) : "t2i",
    prompt: typeof job.prompt === "string" ? job.prompt : String(job.prompt ?? ""),
    status: JOB_STATUSES.includes(job.status as StudioJob["status"]) ? (job.status as StudioJob["status"]) : "failed",
    startedAt: numeric(job.startedAt) ?? Date.now(),
    error: stringOrUndefined(job.error),
    assetId: stringOrUndefined(job.assetId),
    recipe: job.recipe && typeof job.recipe === "object" ? job.recipe as Record<string, unknown> : {},
  };
};

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

const clampTimelineZoom = (value: number): number =>
  Math.max(8, Math.min(48, Math.round(Number.isFinite(value) ? value : 18)));

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

function h3PromptModeForStudio(mode: StudioGenerationMode): H3PromptMode {
  if (mode === "ref2v") return "ref2va";
  if (mode === "i2v") return "i2va";
  if (mode === "inpaint" || mode === "outpaint") return "fl2va";
  return "t2va";
}

const normalizeCanvasDimension = (value: number, fallback: number): number => {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(64, Math.min(8192, Math.round(value / 16) * 16));
};

const assetNeedsCanvasFit = (asset: StudioAsset, width: number, height: number): boolean =>
  asset.kind !== "audio" && !!asset.width && !!asset.height && (asset.width !== width || asset.height !== height);

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
  if (filter === "video") return "txt2vid,img2vid,ref2vid,inpaint_vid,outpaint_vid,studio_render";
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
  video.src = asset.masterUrl || asset.url;
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
    const id = `frame-${asset.id}-${Math.round(target * 1000)}`;
    // Persist the captured frame to IndexedDB (same store used for imported
    // media) rather than embedding a data URL in the project. Studio's
    // project manifest is persisted to localStorage, which has a small
    // quota that a full-resolution PNG data URL can exceed on its own.
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    const blobKey = `media-${id}`;
    let url: string;
    if (blob) {
      await saveImportedMedia(blobKey, blob);
      url = URL.createObjectURL(blob);
    } else {
      url = canvas.toDataURL("image/png");
    }
    return {
      id,
      name: `${asset.name} · frame ${target.toFixed(2)}s`,
      kind: "image",
      url,
      masterUrl: url,
      thumbnailUrl: url,
      blobKey: blob ? blobKey : undefined,
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

// Everything Ctrl+Z should be able to restore. `range`/`inpaintRange` and
// `referenceAssetIds` live in their own state hooks (see below) so that most
// of the component can read them without going through `project`, but an
// undo/redo step still needs to roll all of them back together or the
// ranges drift out of sync with the clip edit that was undone.
interface StudioHistoryEntry {
  project: StudioProject;
  range: StudioRange | null;
  inpaintRange: StudioRange | null;
  referenceAssetIds: string[];
}

interface PendingPlacement {
  asset: StudioAsset;
  start?: number;
  trackId?: string;
  holdStill: boolean;
}

interface ClipDragPreview {
  clips: Array<{ clipId: string; trackId: string; start: number; duration: number }>;
  valid: boolean;
}

export default function StudioWorkspace() {
  const [project, setProject] = useState<StudioProject>(() => createStudioProject());
  const [restored, setRestored] = useState(false);
  const [undoStack, setUndoStack] = useState<StudioHistoryEntry[]>([]);
  const [redoStack, setRedoStack] = useState<StudioHistoryEntry[]>([]);
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
  const [selectedClipIds, setSelectedClipIds] = useState<string[]>([]);
  const [tool, setTool] = useState<StudioTool>("select");
  const [rightPane, setRightPane] = useState<StudioPane>("generate");
  const [form, setForm] = useState<StudioFormState>(EMPTY_FORM);
  const [studioVaePath, setStudioVaePath] = useState<string | null>(null);
  const [studioTextEncoderPath, setStudioTextEncoderPath] = useState<string | null>(null);
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
  const [rendering, setRendering] = useState(false);
  const [renderJobId, setRenderJobId] = useState<string | null>(null);
  const [renderProgress, setRenderProgress] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [libraryLoading, setLibraryLoading] = useState(true);
  const [snapEnabled, setSnapEnabled] = useState(true);
  // Which clip the pointer is over, for the still-image preview popover on
  // the timeline. Nothing else reads it, so it is deliberately not part of
  // the undoable project state.
  const [hoveredClipId, setHoveredClipId] = useState<string | null>(null);
  const [pendingPlacement, setPendingPlacement] = useState<PendingPlacement | null>(null);
  const [clipDragPreview, setClipDragPreview] = useState<ClipDragPreview | null>(null);
  const [projectSettingsOpen, setProjectSettingsOpen] = useState(false);
  const [canvasAspectLocked, setCanvasAspectLocked] = useState(true);
  const [recentProjectsOpen, setRecentProjectsOpen] = useState(false);
  const [recentProjects, setRecentProjects] = useState<StudioRecentProject[]>([]);
  const [canvasDraft, setCanvasDraft] = useState({ width: String(project.width), height: String(project.height) });
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const projectFileInputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const playStartedRef = useRef({ at: 0, playhead: 0 });
  const playheadRef = useRef(0);
  const seekRepeatRef = useRef<SeekRepeatState>({ timer: null, frame: null, direction: 1, started: false, startedAt: 0, lastAt: 0 });
  const initializedDefaultsForArchRef = useRef<string | null>(null);
  const galleryHydrationRef = useRef(new Map<string, Promise<StudioAsset>>());
  const galleryRequestRef = useRef(0);
  const timelineGestureCleanupRef = useRef<(() => void) | null>(null);
  const timelineGestureCancelRef = useRef<(() => void) | null>(null);
  const timelineScrollRef = useRef<HTMLDivElement | null>(null);
  const timelinePointersRef = useRef(new Map<number, { x: number; y: number }>());
  const timelinePinchRef = useRef<{ distance: number; zoom: number; centerX: number; centerTime: number } | null>(null);
  const assetPressRef = useRef<{ assetId: string; timer: number; x: number; y: number } | null>(null);
  const renderControllersRef = useRef(new Map<string, AbortController>());
  const studioUnmountedRef = useRef(false);
  const suppressClipClickRef = useRef<string | null>(null);
  const selectedClipIdsRef = useRef<string[]>([]);
  const pendingImageMaskRef = useRef<string | undefined>(undefined);
  // Keep edit callbacks independent from high-frequency state changes.
  const projectRef = useRef(project);
  useEffect(() => { projectRef.current = project; }, [project]);
  useEffect(() => { playheadRef.current = playhead; }, [playhead]);
  useEffect(() => { selectedClipIdsRef.current = selectedClipIds; }, [selectedClipIds]);
  useEffect(() => {
    setCanvasDraft({ width: String(project.width), height: String(project.height) });
  }, [project.height, project.width]);
  const rangeRef = useRef(range);
  useEffect(() => { rangeRef.current = range; }, [range]);
  const inpaintRangeRef = useRef(inpaintRange);
  useEffect(() => { inpaintRangeRef.current = inpaintRange; }, [inpaintRange]);
  const referenceAssetIdsRef = useRef(referenceAssetIds);
  useEffect(() => { referenceAssetIdsRef.current = referenceAssetIds; }, [referenceAssetIds]);
  const undoStackRef = useRef(undoStack);
  useEffect(() => { undoStackRef.current = undoStack; }, [undoStack]);
  const redoStackRef = useRef(redoStack);
  useEffect(() => { redoStackRef.current = redoStack; }, [redoStack]);
  const {
    isBackendReady,
    isVideo,
    modelInfo,
    refreshModelInfo,
    generationDefaults,
    archCapabilities,
    resolveModality,
  } = useStartup();

  // Keep undo stack refs synchronized for gesture edits.
  const pushHistoryEntry = useCallback((entry: StudioHistoryEntry) => {
    const nextUndoStack = [...undoStackRef.current, entry].slice(-MAX_HISTORY);
    setUndoStack(nextUndoStack);
    setRedoStack([]);
    undoStackRef.current = nextUndoStack;
    redoStackRef.current = [];
  }, []);

  // Keep the project ref synchronous; all project writes go through here.
  const applyProject = useCallback((next: StudioProject | ((current: StudioProject) => StudioProject)) => {
    const resolved = typeof next === "function" ? (next as (current: StudioProject) => StudioProject)(projectRef.current) : next;
    projectRef.current = resolved;
    setProject(resolved);
  }, []);

  // Commit one undo entry and one synchronous project update.
  const commit = useCallback((updater: (current: StudioProject) => StudioProject) => {
    const current = projectRef.current;
    pushHistoryEntry({
      project: current,
      range: rangeRef.current,
      inpaintRange: inpaintRangeRef.current,
      referenceAssetIds: referenceAssetIdsRef.current,
    });
    applyProject((latest) => ({ ...updater(latest), revision: latest.revision + 1, updatedAt: new Date().toISOString() }));
  }, [applyProject, pushHistoryEntry]);

  const commitCanvasSize = useCallback(() => {
    const width = normalizeCanvasDimension(Number(canvasDraft.width), projectRef.current.width);
    const height = normalizeCanvasDimension(Number(canvasDraft.height), projectRef.current.height);
    setCanvasDraft({ width: String(width), height: String(height) });
    if (width === projectRef.current.width && height === projectRef.current.height) return;
    commit((current) => ({ ...current, width, height }));
  }, [canvasDraft.height, canvasDraft.width, commit]);

  const updateCanvasDraft = useCallback((field: "width" | "height", value: string) => {
    if (!canvasAspectLocked) {
      setCanvasDraft((current) => ({ ...current, [field]: value }));
      return;
    }
    const numericValue = Number(value);
    const currentWidth = Number(canvasDraft.width) || projectRef.current.width;
    const currentHeight = Number(canvasDraft.height) || projectRef.current.height;
    const ratio = field === "width" ? currentHeight / currentWidth : currentWidth / currentHeight;
    const paired = Number.isFinite(numericValue) && numericValue > 0
      ? Math.round((field === "width" ? numericValue * ratio : numericValue * ratio))
      : field === "width" ? currentHeight : currentWidth;
    setCanvasDraft(field === "width"
      ? { width: value, height: String(paired) }
      : { width: String(paired), height: value });
  }, [canvasAspectLocked, canvasDraft.height, canvasDraft.width]);

  const clearClipSelection = useCallback(() => {
    selectedClipIdsRef.current = [];
    setSelectedClipIds([]);
    setSelectedClipId(null);
  }, []);

  const selectClip = useCallback((clipId: string, additive = false) => {
    const current = selectedClipIdsRef.current;
    const next = additive
      ? current.includes(clipId)
        ? current.filter((id) => id !== clipId)
        : [...current, clipId]
      : [clipId];
    selectedClipIdsRef.current = next;
    setSelectedClipIds(next);
    setSelectedClipId(next.at(-1) || null);
  }, []);

  const undo = useCallback(() => {
    const previous = undoStackRef.current.at(-1);
    if (!previous) return;
    const current = projectRef.current;
    const restored = { ...previous.project, jobs: current.jobs };
    const nextUndoStack = undoStackRef.current.slice(0, -1);
    const nextRedoStack = [...redoStackRef.current, {
      project: current,
      range: rangeRef.current,
      inpaintRange: inpaintRangeRef.current,
      referenceAssetIds: referenceAssetIdsRef.current,
    }].slice(-MAX_HISTORY);
    setUndoStack(nextUndoStack);
    setRedoStack(nextRedoStack);
    applyProject(restored);
    setRange(previous.range);
    setInpaintRange(previous.inpaintRange);
    setReferenceAssetIds(previous.referenceAssetIds);
    undoStackRef.current = nextUndoStack;
    redoStackRef.current = nextRedoStack;
    rangeRef.current = previous.range;
    inpaintRangeRef.current = previous.inpaintRange;
    referenceAssetIdsRef.current = previous.referenceAssetIds;
  }, [applyProject]);

  const redo = useCallback(() => {
    const next = redoStackRef.current.at(-1);
    if (!next) return;
    const current = projectRef.current;
    const restored = { ...next.project, jobs: current.jobs };
    const nextRedoStack = redoStackRef.current.slice(0, -1);
    const nextUndoStack = [...undoStackRef.current, {
      project: current,
      range: rangeRef.current,
      inpaintRange: inpaintRangeRef.current,
      referenceAssetIds: referenceAssetIdsRef.current,
    }].slice(-MAX_HISTORY);
    setRedoStack(nextRedoStack);
    setUndoStack(nextUndoStack);
    applyProject(restored);
    setRange(next.range);
    setInpaintRange(next.inpaintRange);
    setReferenceAssetIds(next.referenceAssetIds);
    redoStackRef.current = nextRedoStack;
    undoStackRef.current = nextUndoStack;
    rangeRef.current = next.range;
    inpaintRangeRef.current = next.inpaintRange;
    referenceAssetIdsRef.current = next.referenceAssetIds;
  }, [applyProject]);

  useEffect(() => {
    setRecentProjects(readRecentProjects());
    loadStudioProject()
      .then((saved) => {
        if (saved) {
          const restoredJobs = (Array.isArray(saved.jobs) ? saved.jobs : []).map(normalizeImportedJob).map((job): StudioJob => job.status === "running"
            ? { ...job, status: "failed", error: "Studio closed before this job returned. Check Gallery before retrying." }
            : job);
          applyProject({ ...saved, jobs: restoredJobs });
          setRange(saved.outputRange ?? null);
          setInpaintRange(saved.inpaintRange ?? null);
          setReferenceAssetIds(saved.referenceAssetIds ?? []);
          setJobs(restoredJobs);
          setResultAssetIds(restoredJobs.flatMap((job) => job.assetId ? [job.assetId] : []));
          if (saved.renderJobId) {
            setRenderJobId(saved.renderJobId);
            setRendering(true);
          }
        }
      })
      .finally(() => setRestored(true));
  }, []);

  useEffect(() => {
    setPendingTransfer(takeStudioTransfer());
  }, []);

  useEffect(() => {
    studioUnmountedRef.current = false;
    return () => {
      studioUnmountedRef.current = true;
      renderControllersRef.current.forEach((controller) => controller.abort());
      renderControllersRef.current.clear();
      timelineGestureCleanupRef.current?.();
      timelineGestureCancelRef.current = null;
      timelinePointersRef.current.clear();
      timelinePinchRef.current = null;
      if (assetPressRef.current) window.clearTimeout(assetPressRef.current.timer);
      assetPressRef.current = null;
    };
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
        // `setSelectedAssetId` used to be called from inside the
        // `setProject` updater above -- a second setState nested inside
        // another's updater, the same anti-pattern `commit` used to have.
        // Read `projectRef.current` (kept synchronously current by
        // `applyProject`) instead of reaching for a stale `current` via a
        // functional updater just to decide whether the asset already
        // exists.
        const existing = projectRef.current.assets.find((item) => canonicalAssetKey(item) === canonicalAssetKey(incoming));
        setSelectedAssetId((existing || incoming).id);
        if (!existing) {
          applyProject((current) => ({
            ...current,
            assets: [...current.assets, incoming],
            revision: current.revision + 1,
            updatedAt: new Date().toISOString(),
          }));
        }
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
      clearClipSelection();
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
    applyProject((current) => current.jobs === jobs ? current : { ...current, jobs });
  }, [applyProject, jobs, restored]);

  useEffect(() => {
    if (!restored) return;
    const timer = window.setTimeout(() => {
      const snapshot = { ...project, jobs, outputRange: range, inpaintRange, referenceAssetIds };
      const result = saveStudioProject(snapshot);
      setRecentProjects(rememberRecentProject(snapshot));
      if (!result.ok) setNotice("Could not save the project locally (browser storage is full). Recent edits may be lost on reload.");
    }, 350);
    return () => window.clearTimeout(timer);
  }, [inpaintRange, jobs, project, range, referenceAssetIds, restored]);

  useEffect(() => {
    if (!restored) return;
    const saveOnExit = () => saveStudioProject({ ...project, jobs, outputRange: range, inpaintRange, referenceAssetIds });
    window.addEventListener("pagehide", saveOnExit);
    return () => window.removeEventListener("pagehide", saveOnExit);
  }, [inpaintRange, jobs, project, range, referenceAssetIds, restored]);

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
    allAssets.find((asset) => asset.id === (selectedAssetId || selectedClip?.assetId)) || null;
  const activeClips = useMemo(() => project.clips.filter((clip) => clip.activeTake !== false), [project.clips]);
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
        applyProject((current) => ({
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
  }, [applyProject]);

  const selectAsset = useCallback((asset: StudioAsset) => {
    setSelectedAssetId(asset.id);
    void hydrateGalleryAsset(asset);
  }, [hydrateGalleryAsset]);

  const outputDuration = useMemo(() => {
    if (isVideoModel && range) return Math.max(0, range.end - range.start);
    if (!form.numFrames || !form.frameRate) return 0;
    return form.numFrames / form.frameRate;
  }, [form.frameRate, form.numFrames, isVideoModel, range]);

  const placeAssetOnTimeline = useCallback((asset: StudioAsset, start?: number, trackId?: string, holdStill = false, requestedFitMode?: StudioClipFitMode): boolean => {
    const targetTrack =
      project.tracks.find((track) => track.id === trackId && track.kind === (asset.kind === "audio" ? "audio" : "video")) ||
      project.tracks.find((track) => track.kind === (asset.kind === "audio" ? "audio" : "video"));
    if (!targetTrack) return false;
    if (targetTrack.locked) {
      setNotice(`Unlock ${targetTrack.name} before adding a clip.`);
      return false;
    }

    const trackEnd = activeClips
      .filter((clip) => clip.trackId === targetTrack.id)
      .reduce((end, clip) => Math.max(end, clip.start + clip.duration), 0);
    const requestedStart = clampTime(start ?? trackEnd, project.duration);
    const initialDuration = defaultClipDurationForAsset(asset, project.fps, project.duration - requestedStart, holdStill);
    const clipStart = clampTime(requestedStart, Math.max(0, project.duration - initialDuration));
    const duration = defaultClipDurationForAsset(asset, project.fps, project.duration - clipStart, holdStill);
    // No room for even one frame: refuse rather than create a 0-duration
    // clip, which the backend's manifest validation rejects outright.
    if (duration <= frameDurationFor(project.fps) / 2) {
      setNotice("There is no room left at the end of the timeline for this clip. Extend the timeline duration first.");
      return false;
    }
    const sourceDuration = sourceDurationForAsset(asset);
    const clip: StudioClip = {
      id: newId(),
      assetId: asset.id,
      trackId: targetTrack.id,
      name: asset.name,
      start: clipStart,
      duration,
      sourceIn: 0,
      presentation: asset.kind === "image" ? (holdStill ? "hold" : "frame") : "clip",
      ...(sourceDuration != null ? { sourceDuration } : {}),
      ...(asset.kind !== "audio" && requestedFitMode ? { fitMode: requestedFitMode } : {}),
    };
    commit((current) => ({
      ...current,
      assets: current.assets.some((item) => item.id === asset.id) ? current.assets : [...current.assets, asset],
      clips: [...current.clips, clip],
    }));
    setSelectedAssetId(asset.id);
    selectClip(clip.id);
    return true;
  }, [activeClips, commit, project.duration, project.fps, project.tracks]);

  const addAssetToTimeline = useCallback((asset: StudioAsset, start?: number, trackId?: string, holdStill = false, fitMode?: StudioClipFitMode): boolean => {
    if (asset.kind === "image" && assetNeedsCanvasFit(asset, project.width, project.height) && !fitMode) {
      setPendingPlacement({ asset, start, trackId, holdStill });
      setSelectedAssetId(asset.id);
      return false;
    }
    return placeAssetOnTimeline(asset, start, trackId, holdStill, fitMode || (asset.kind === "video" ? "cover" : undefined));
  }, [placeAssetOnTimeline, project.height, project.width]);

  const confirmPendingPlacement = useCallback((fitMode: StudioClipFitMode) => {
    const pending = pendingPlacement;
    if (!pending) return;
    setPendingPlacement(null);
    placeAssetOnTimeline(pending.asset, pending.start, pending.trackId, pending.holdStill, fitMode);
  }, [pendingPlacement, placeAssetOnTimeline]);

  const cancelAssetPress = () => {
    const press = assetPressRef.current;
    if (!press) return;
    window.clearTimeout(press.timer);
    assetPressRef.current = null;
  };

  const beginAssetPress = (event: ReactPointerEvent<HTMLButtonElement>, asset: StudioAsset) => {
    if (event.pointerType !== "touch") return;
    cancelAssetPress();
    const timer = window.setTimeout(() => {
      if (assetPressRef.current?.assetId !== asset.id) return;
      assetPressRef.current = null;
      void hydrateGalleryAsset(asset).then((hydrated) => {
        if (addAssetToTimeline(hydrated)) setNotice(`Added ${asset.name} to the timeline.`);
      });
    }, 420);
    assetPressRef.current = { assetId: asset.id, timer, x: event.clientX, y: event.clientY };
  };

  const moveAssetPress = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const press = assetPressRef.current;
    if (!press || Math.hypot(event.clientX - press.x, event.clientY - press.y) > 10) cancelAssetPress();
  };

  const finishAssetPress = () => cancelAssetPress();

  const deleteSelectedClip = useCallback(() => {
    const selectedIds = new Set(selectedClipIdsRef.current.length ? selectedClipIdsRef.current : selectedClipId ? [selectedClipId] : []);
    if (!selectedIds.size) return;
    const locked = project.clips.find((clip) => selectedIds.has(clip.id)
      && project.tracks.find((track) => track.id === clip.trackId)?.locked);
    if (locked) {
      setNotice(`Unlock ${project.tracks.find((track) => track.id === locked.trackId)?.name || "the track"} before deleting this clip.`);
      return;
    }
    const replacements = new Map(project.clips
      .filter((clip) => selectedIds.has(clip.id) && clip.takeGroupId)
      .map((clip) => [clip.takeGroupId!, project.clips.find((candidate) => candidate.takeGroupId === clip.takeGroupId && !selectedIds.has(candidate.id))]));
    commit((current) => ({
      ...current,
      clips: current.clips
        .filter((clip) => !selectedIds.has(clip.id))
        .map((clip) => [...replacements.values()].includes(clip) ? { ...clip, activeTake: true } : clip),
    }));
    clearClipSelection();
  }, [clearClipSelection, commit, project.clips, project.tracks, selectedClipId]);

  const splitSelectedClip = useCallback((targetClip?: StudioClip | null, splitTime = playhead) => {
    const selectedIds = new Set(selectedClipIdsRef.current.length ? selectedClipIdsRef.current : selectedClipId ? [selectedClipId] : []);
    const candidates = (targetClip && !selectedIds.has(targetClip.id)
      ? [targetClip]
      : project.clips.filter((clip) => selectedIds.has(clip.id)))
      .filter((clip) => splitTime > clip.start + frameDurationFor(project.fps)
        && splitTime < clip.start + clip.duration - frameDurationFor(project.fps));
    if (!candidates.length) {
      setNotice("Move the playhead inside the selected clip before splitting.");
      return;
    }
    const locked = candidates.find((clip) => project.tracks.find((track) => track.id === clip.trackId)?.locked);
    if (locked) {
      setNotice(`Unlock ${project.tracks.find((track) => track.id === locked.trackId)?.name || "the track"} before splitting this clip.`);
      return;
    }
    const rightIds: string[] = [];
    commit((current) => ({
      ...current,
      clips: current.clips.flatMap((clip) => {
        const match = candidates.find((candidate) => candidate.id === clip.id);
        if (!match) return [clip];
        const leftDuration = splitTime - match.start;
        const right = { ...match, id: newId(), start: splitTime, duration: match.duration - leftDuration, sourceIn: match.sourceIn + leftDuration };
        rightIds.push(right.id);
        return [{ ...match, duration: leftDuration }, right];
      }),
    }));
    selectedClipIdsRef.current = rightIds;
    setSelectedClipIds(rightIds);
    setSelectedClipId(rightIds.at(-1) || null);
  }, [commit, playhead, project.clips, project.fps, project.tracks, selectedClipId]);

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
    const start = clampTime(
      ((event.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
      project.duration,
    );
    const clipId = event.dataTransfer.getData("application/x-studio-clip");
    if (clipId) {
      moveClip(clipId, trackId, start);
      return;
    }
    const frameAssetId = event.dataTransfer.getData("application/x-studio-frame");
    if (frameAssetId) {
      const frameTime = numeric(event.dataTransfer.getData("application/x-studio-frame-time")) ?? playhead;
      const source = allAssets.find((item) => item.id === frameAssetId);
      if (!source) return;
      const hydrated = await hydrateGalleryAsset(source);
      if (hydrated.kind === "audio") {
        setNotice("Audio clips do not have video frames to extract.");
        return;
      }
      const frame = await captureVideoFrameAsset(hydrated, frameTime);
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
      const id = newId();
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

  const handleProjectImport = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const parsed = parseStudioProjectFile(JSON.parse(await file.text()));
      const defaults = createStudioProject();
      const fps = (numeric(parsed.fps) || 0) > 0 ? Number(parsed.fps) : defaults.fps;
      const normalizedClips = (parsed.clips as unknown[])
        .map((raw) => normalizeImportedClip(raw, fps))
        .filter((clip): clip is StudioClip => clip !== null);
      // A hand-edited or concatenated manifest can carry duplicate clip ids;
      // reassign any repeat rather than let two clips silently collide on
      // React's `key` and on every `clips.map((clip) => clip.id === id ...)`
      // lookup the rest of this component relies on to touch a single clip.
      const seenClipIds = new Set<string>();
      const dedupedClips = normalizedClips.map((clip) => {
        if (!seenClipIds.has(clip.id)) {
          seenClipIds.add(clip.id);
          return clip;
        }
        const reassigned = { ...clip, id: newId() };
        seenClipIds.add(reassigned.id);
        return reassigned;
      });
      const imported: StudioProject = {
        ...defaults,
        ...parsed,
        schemaVersion: 4,
        name: stringOrUndefined(parsed.name) || defaults.name,
        createdAt: stringOrUndefined(parsed.createdAt) || defaults.createdAt,
        updatedAt: stringOrUndefined(parsed.updatedAt) || defaults.updatedAt,
        renderJobId: stringOrUndefined(parsed.renderJobId),
        revision: numeric(parsed.revision) ?? 0,
        duration: numeric(parsed.duration) ?? defaults.duration,
        fps,
        width: normalizeCanvasDimension(numeric(parsed.width) ?? defaults.width, defaults.width),
        height: normalizeCanvasDimension(numeric(parsed.height) ?? defaults.height, defaults.height),
        assets: (parsed.assets as unknown[]).map(normalizeImportedAsset),
        tracks: (parsed.tracks as unknown[]).map(normalizeImportedTrack),
        clips: dedupedClips,
        jobs: Array.isArray(parsed.jobs) ? parsed.jobs.map(normalizeImportedJob) : [],
        outputRange: normalizeImportedRange(parsed.outputRange),
        inpaintRange: normalizeImportedRange(parsed.inpaintRange),
        referenceAssetIds: Array.isArray(parsed.referenceAssetIds)
          ? parsed.referenceAssetIds.filter((id): id is string => typeof id === "string")
          : [],
      };
      const imageAssetIds = new Set(imported.assets.filter((asset) => asset.kind === "image").map((asset) => asset.id));
      const clips = imported.clips.map((clip) => {
        const inputRoles = clip.inputRoles?.filter((role) => role === "keyframe");
        if (!imageAssetIds.has(clip.assetId) || clip.presentation) return { ...clip, inputRoles };
        const held = Number(clip.duration) > (1 / imported.fps) + 0.0001;
        return {
          ...clip,
          inputRoles,
          duration: held ? clip.duration : 1 / imported.fps,
          sourceIn: 0,
          presentation: held ? "hold" as const : "frame" as const,
          sourceDuration: 0,
        };
      });
      const normalized = { ...imported, clips };
      const assets = await Promise.all(normalized.assets.map(async (asset) => {
        if (!asset.blobKey) return asset;
        const blob = await loadImportedMedia(asset.blobKey);
        if (!blob) return { ...asset, url: "", thumbnailUrl: undefined, missing: true };
        const url = URL.createObjectURL(blob);
        return { ...asset, url, thumbnailUrl: asset.kind === "image" ? url : undefined, missing: false };
      }));
      const restored = { ...normalized, assets };
      // The project currently in memory may hold `blob:` URLs created by
      // captureVideoFrameAsset/import/edit that nothing else references
      // once it is replaced below; without this they leak for the rest of
      // the tab's lifetime.
      const staleBlobUrls = project.assets.filter((asset) => asset.blobKey && asset.url).map((asset) => asset.url);
      const importedRange = restored.outputRange ?? null;
      const importedInpaintRange = restored.inpaintRange ?? null;
      const importedReferenceAssetIds = restored.referenceAssetIds || [];
      applyProject(restored);
      setRange(importedRange);
      setInpaintRange(importedInpaintRange);
      setReferenceAssetIds(importedReferenceAssetIds);
      setJobs(restored.jobs || []);
      setResultAssetIds((restored.jobs || []).flatMap((job) => job.assetId ? [job.assetId] : []));
      clearClipSelection();
      setSelectedAssetId(null);
      setUndoStack([]);
      setRedoStack([]);
      // An imported project is a fresh document; its own history starts
      // empty, and none of the state it replaced should still be reachable
      // through a stale ref a moment later.
      rangeRef.current = importedRange;
      inpaintRangeRef.current = importedInpaintRange;
      referenceAssetIdsRef.current = importedReferenceAssetIds;
      undoStackRef.current = [];
      redoStackRef.current = [];
      staleBlobUrls.forEach((url) => { try { URL.revokeObjectURL(url); } catch { /* already revoked */ } });
      const missingLocalMedia = assets.filter((asset) => asset.missing || (asset.blobKey && !asset.url)).length;
      setNotice(missingLocalMedia
        ? `Imported ${restored.name}; ${missingLocalMedia} local media item(s) need to be re-imported.`
        : `Imported ${restored.name}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not import the Studio project.");
    } finally {
      event.target.value = "";
    }
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
    selectClip(clip.id);
    setSelectedAssetId(inputAsset.id);
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
    const frameTime = numeric(event.dataTransfer.getData("application/x-studio-frame-time")) ?? playhead;
    const assetId = frameAssetId || event.dataTransfer.getData("application/x-studio-asset");
    const asset = allAssets.find((item) => item.id === assetId);
    if (!asset) return;
    const hydrated = await hydrateGalleryAsset(asset);
    if (frameAssetId) {
      if (hydrated.kind === "audio") {
        setNotice("Audio clips do not have a still frame to use as an image input.");
        return;
      }
      const frame = await captureVideoFrameAsset(hydrated, frameTime);
      if (!frame) {
        setNotice("Could not capture a frame from this media.");
        return;
      }
      commit((current) => current.assets.some((item) => item.id === frame.id)
        ? current
        : { ...current, assets: [...current.assets, frame] });
      setSelectedAssetId(frame.id);
      clearClipSelection();
      return;
    }
    setSelectedAssetId(hydrated.id);
    clearClipSelection();
  };

  const handleReferenceDrop = async (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    const clipId = event.dataTransfer.getData("application/x-studio-clip");
    if (clipId) {
      setNotice("References are chosen explicitly from Media; drop a clip on Generate to use its input.");
      return;
    }
    const frameAssetId = event.dataTransfer.getData("application/x-studio-frame");
    const frameTime = numeric(event.dataTransfer.getData("application/x-studio-frame-time")) ?? playhead;
    const assetId = frameAssetId || event.dataTransfer.getData("application/x-studio-asset");
    const asset = allAssets.find((item) => item.id === assetId);
    if (!asset) return;
    const hydrated = await hydrateGalleryAsset(asset);
    if (frameAssetId) {
      if (hydrated.kind === "audio") {
        setNotice("Audio clips do not have a still frame to use as a reference.");
        return;
      }
      const frame = await captureVideoFrameAsset(hydrated, frameTime);
      if (!frame) {
        setNotice("Could not capture a frame from this media.");
        return;
      }
      // One user gesture (the drop) both creates the frame asset and
      // registers it as a reference; a single history entry keeps a later
      // Ctrl+Z undoing both together instead of removing the reference but
      // leaving its now-orphaned frame asset behind.
      pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
      applyProject((current) => current.assets.some((item) => item.id === frame.id)
        ? current
        : { ...current, assets: [...current.assets, frame], revision: current.revision + 1, updatedAt: new Date().toISOString() });
      setReferenceAssetIds((current) => current.includes(frame.id) ? current : [...current, frame.id]);
      return;
    }
    if (!referenceAssetIds.includes(hydrated.id)) {
      pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
      setReferenceAssetIds((current) => [...current, hydrated.id]);
    }
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

  const saveStudioEditedImage = async (editedImageUrl: string) => {
    const source = imageEditorState ? allAssets.find((asset) => asset.id === imageEditorState.assetId) : null;
    if (!source) return;
    const id = `studio-image-${newId()}`;
    // Persist the edited image to IndexedDB rather than embedding a data URL
    // in the project (see captureVideoFrameAsset for the same reasoning).
    let url = editedImageUrl;
    let blobKey: string | undefined;
    try {
      const blob = await (await fetch(editedImageUrl)).blob();
      blobKey = `media-${id}`;
      await saveImportedMedia(blobKey, blob);
      url = URL.createObjectURL(blob);
    } catch (error) {
      console.error("[Studio] Failed to persist edited image to IndexedDB, keeping inline data URL", error);
    }
    const derived: StudioAsset = {
      ...source,
      id,
      galleryId: undefined,
      name: `${source.name.replace(/\.[^/.]+$/, "")} · edited`,
      url,
      masterUrl: url,
      thumbnailUrl: url,
      blobKey,
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
    clearClipSelection();
    setImageEditorState(null);
    pendingImageMaskRef.current = undefined;
  };

  const setTimelinePlayhead = useCallback((time: number) => {
    const next = clampTime(time, projectRef.current.duration);
    playheadRef.current = next;
    setPlayhead(next);
  }, []);

  const seekTimeline = useCallback((time: number) => {
    const next = clampTime(time, projectRef.current.duration);
    playStartedRef.current = { at: performance.now(), playhead: next };
    setTimelinePlayhead(next);
    if (next >= projectRef.current.duration) setPlaying(false);
    const localTime = previewClip
      ? previewClip.sourceIn + Math.max(0, next - previewClip.start)
      : next;
    if (videoRef.current && Number.isFinite(videoRef.current.duration)) videoRef.current.currentTime = localTime;
    if (audioRef.current && Number.isFinite(audioRef.current.duration)) audioRef.current.currentTime = localTime;
  }, [previewClip, setTimelinePlayhead]);

  const seekBy = useCallback((seconds: number) => {
    seekTimeline(playheadRef.current + seconds);
  }, [seekTimeline]);

  const togglePlayback = useCallback(() => {
    setPlaying((current) => {
      if (current) return false;
      const start = playheadRef.current >= projectRef.current.duration ? 0 : playheadRef.current;
      playStartedRef.current = { at: performance.now(), playhead: start };
      if (start !== playheadRef.current) setTimelinePlayhead(start);
      return true;
    });
  }, [setTimelinePlayhead]);

  const stopSeekRepeat = useCallback(() => {
    const repeat = seekRepeatRef.current;
    if (repeat.timer != null) window.clearTimeout(repeat.timer);
    if (repeat.frame != null) window.cancelAnimationFrame(repeat.frame);
    repeat.timer = null;
    repeat.frame = null;
    repeat.started = false;
  }, []);

  const beginSeekRepeat = useCallback((event: ReactPointerEvent<HTMLButtonElement>, direction: -1 | 1) => {
    event.preventDefault();
    stopSeekRepeat();
    const repeat = seekRepeatRef.current;
    repeat.direction = direction;
    event.currentTarget.setPointerCapture(event.pointerId);
    repeat.timer = window.setTimeout(() => {
      const active = seekRepeatRef.current;
      active.timer = null;
      active.started = true;
      active.startedAt = performance.now();
      active.lastAt = active.startedAt;
      const tick = (now: number) => {
        const current = seekRepeatRef.current;
        if (!current.started || current.direction !== direction) return;
        const delta = Math.min(0.05, Math.max(0, (now - current.lastAt) / 1000));
        current.lastAt = now;
        const heldSeconds = (now - current.startedAt) / 1000;
        const speed = Math.min(24, 4 + heldSeconds * 12);
        seekBy(direction * delta * speed);
        current.frame = window.requestAnimationFrame(tick);
      };
      active.frame = window.requestAnimationFrame(tick);
    }, 350);
  }, [seekBy, stopSeekRepeat]);

  const finishSeekRepeat = useCallback((direction: -1 | 1) => {
    const repeat = seekRepeatRef.current;
    const wasClick = repeat.timer != null && !repeat.started;
    stopSeekRepeat();
    if (wasClick) seekBy(direction * 5);
  }, [seekBy, stopSeekRepeat]);

  useEffect(() => stopSeekRepeat, [stopSeekRepeat]);

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
        setTimelinePlayhead(0);
        setPlaying(false);
        return;
      }
      setTimelinePlayhead(next);
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [playing, previewAsset?.url, previewClip, project.duration, setTimelinePlayhead]);

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
      } else if (event.key.toLowerCase() === "v" && !event.ctrlKey && !event.metaKey) {
        setTool("select");
      } else if ((event.key.toLowerCase() === "b" || event.key.toLowerCase() === "c") && !event.ctrlKey && !event.metaKey) {
        setTool("blade");
      } else if (event.key.toLowerCase() === "h" && !event.ctrlKey && !event.metaKey) {
        setTool("hand");
      } else if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) {
        splitSelectedClip();
      } else if (event.key === "Delete" || event.key === "Backspace") {
        deleteSelectedClip();
      } else if (event.key === "Home") {
        event.preventDefault();
        seekTimeline(0);
      } else if (event.key === "End") {
        event.preventDefault();
        seekTimeline(project.duration);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        const frames = event.shiftKey ? 10 : 1;
        const direction = event.key === "ArrowRight" ? 1 : -1;
        event.preventDefault();
        seekBy(direction * frames / project.fps);
      } else if (event.key.toLowerCase() === "i" || event.key.toLowerCase() === "o") {
        const isStart = event.key.toLowerCase() === "i";
        const currentRange = event.altKey ? inpaintRange : range;
        const frame = Math.round(playhead * project.fps) / project.fps;
        const next = currentRange
          ? { start: isStart ? frame : currentRange.start, end: isStart ? currentRange.end : frame }
          : { start: isStart ? frame : 0, end: isStart ? project.duration : frame };
        const normalized = { start: Math.min(next.start, next.end), end: Math.max(next.start, next.end) };
        event.preventDefault();
        pushHistoryEntry({ project: projectRef.current, range, inpaintRange, referenceAssetIds });
        if (event.altKey) setInpaintRange(normalized);
        else setRange(normalized);
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? redo() : undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [deleteSelectedClip, inpaintRange, project.duration, project.fps, pushHistoryEntry, range, redo, referenceAssetIds, seekBy, seekTimeline, splitSelectedClip, togglePlayback, undo]);

  const handleTimelinePointerDownCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "touch") return;
    const pointers = timelinePointersRef.current;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size !== 2) return;
    const [first, second] = [...pointers.values()];
    timelineGestureCancelRef.current?.();
    timelineGestureCleanupRef.current?.();
    timelineGestureCleanupRef.current = null;
    timelineGestureCancelRef.current = null;
    const centerX = (first.x + second.x) / 2;
    const scroll = timelineScrollRef.current;
    const bounds = scroll?.getBoundingClientRect();
    timelinePinchRef.current = {
      distance: Math.max(1, Math.hypot(first.x - second.x, first.y - second.y)),
      zoom,
      centerX,
      centerTime: bounds && scroll ? (scroll.scrollLeft + centerX - bounds.left) / zoom : centerX / zoom,
    };
    event.preventDefault();
  };

  const handleTimelinePointerMoveCapture = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "touch") return;
    const pointer = timelinePointersRef.current.get(event.pointerId);
    if (!pointer) return;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    const pinch = timelinePinchRef.current;
    if (!pinch || timelinePointersRef.current.size < 2) return;
    const [first, second] = [...timelinePointersRef.current.values()];
    const distance = Math.max(1, Math.hypot(first.x - second.x, first.y - second.y));
    const nextZoom = clampTimelineZoom(pinch.zoom * distance / pinch.distance);
    setZoom(nextZoom);
    const scroll = timelineScrollRef.current;
    const bounds = scroll?.getBoundingClientRect();
    if (scroll && bounds) {
      window.requestAnimationFrame(() => {
        if (timelinePinchRef.current !== pinch) return;
        scroll.scrollLeft = Math.max(0, pinch.centerTime * nextZoom - (pinch.centerX - bounds.left));
      });
    }
    event.preventDefault();
  };

  const finishTimelinePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.pointerType !== "touch") return;
    timelinePointersRef.current.delete(event.pointerId);
    if (timelinePointersRef.current.size < 2) timelinePinchRef.current = null;
  };

  const beginTimelinePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (tool !== "hand" || event.button !== 0 || event.pointerType === "touch") return;
    const scroll = timelineScrollRef.current;
    if (!scroll) return;
    event.preventDefault();
    const originX = event.clientX;
    const originY = event.clientY;
    const initialLeft = scroll.scrollLeft;
    const initialTop = scroll.scrollTop;
    const move = (pointerEvent: PointerEvent) => {
      scroll.scrollLeft = initialLeft - (pointerEvent.clientX - originX);
      scroll.scrollTop = initialTop - (pointerEvent.clientY - originY);
    };
    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  const beginRange = (event: ReactPointerEvent<HTMLDivElement>) => {
    const element = event.currentTarget;
    const bounds = element.getBoundingClientRect();
    const start = clampTime(
      ((event.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
      project.duration,
    );
    // Capture the pre-gesture state for one undo entry.
    const pushRangeHistory = () => pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
    if (tool !== "range") {
      seekTimeline(start);
      clearClipSelection();
      setSelectedAssetId(null);
      if (event.pointerType !== "touch") {
        // Ruler dragging scrubs unless range mode is active.
        event.preventDefault();
        timelineGestureCleanupRef.current?.();
        const move = (pointerEvent: PointerEvent) => {
          seekTimeline(clampTime(((pointerEvent.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom, project.duration));
        };
        const finish = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", finish);
          window.removeEventListener("pointercancel", finish);
          if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
          if (timelineGestureCancelRef.current === cleanup) timelineGestureCancelRef.current = null;
        };
        const cleanup = finish;
        timelineGestureCleanupRef.current = cleanup;
        timelineGestureCancelRef.current = cleanup;
        element.setPointerCapture(event.pointerId);
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", finish);
        window.addEventListener("pointercancel", finish);
        return;
      }

      event.preventDefault();
      timelineGestureCleanupRef.current?.();
      let rangeArmed = false;
      const longPress = window.setTimeout(() => {
        rangeArmed = true;
        setTool("range");
        const next = { start, end: Math.min(project.duration, start + frameDurationFor(project.fps)) };
        if (rangeTarget === "output") setRange(next);
        else setInpaintRange(next);
      }, 420);
      const move = (pointerEvent: PointerEvent) => {
        const current = clampTime(
          ((pointerEvent.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
          project.duration,
        );
        if (rangeArmed) {
          const next = { start: Math.min(start, current), end: Math.max(start, current) };
          if (rangeTarget === "output") setRange(next);
          else setInpaintRange(next);
        } else {
          seekTimeline(current);
        }
      };
      const finish = () => {
        window.clearTimeout(longPress);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", finish);
        window.removeEventListener("pointercancel", cancel);
        if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
        if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
        if (rangeArmed) {
          pushRangeHistory();
          setRightPane("generate");
          setTool("select");
        }
      };
      const cancel = () => {
        finish();
        if (rangeArmed) {
          const next = { start, end: Math.min(project.duration, start + frameDurationFor(project.fps)) };
          if (rangeTarget === "output") setRange(next);
          else setInpaintRange(next);
        }
      };
      const cleanup = () => {
        window.clearTimeout(longPress);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", finish);
        window.removeEventListener("pointercancel", cancel);
      };
      timelineGestureCleanupRef.current = cleanup;
      timelineGestureCancelRef.current = cancel;
      element.setPointerCapture(event.pointerId);
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", finish);
      window.addEventListener("pointercancel", cancel);
      return;
    }

    event.preventDefault();
    timelineGestureCleanupRef.current?.();
    const updateRange = (current: number) => {
      const next = { start: Math.min(start, current), end: Math.max(start, current) };
      if (rangeTarget === "output") setRange(next);
      else setInpaintRange(next);
    };
    updateRange(start);
    const move = (pointerEvent: PointerEvent) => {
      updateRange(clampTime(
        ((pointerEvent.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
        project.duration,
      ));
    };
    const finish = () => {
      element.removeEventListener("pointermove", move);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
      pushRangeHistory();
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
    timelineGestureCancelRef.current = cancel;
    element.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
  };

  const beginTrackScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || event.target !== event.currentTarget) return;
    event.preventDefault();
    timelineGestureCleanupRef.current?.();
    const element = event.currentTarget;
    const bounds = element.getBoundingClientRect();
    const timeAt = (clientX: number) => clampTime(
      ((clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
      project.duration,
    );
    seekTimeline(timeAt(event.clientX));
    clearClipSelection();
    setSelectedAssetId(null);
    const move = (pointerEvent: PointerEvent) => seekTimeline(timeAt(pointerEvent.clientX));
    const finish = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cleanup) timelineGestureCancelRef.current = null;
    };
    const cleanup = finish;
    timelineGestureCleanupRef.current = cleanup;
    timelineGestureCancelRef.current = cleanup;
    element.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
  };

  const beginClipMoveLegacy = (event: ReactPointerEvent<HTMLDivElement>, clip: StudioClip) => {
    if (event.button !== 0 || tool !== "select") return;
    if (event.pointerType === "touch" && timelinePointersRef.current.size > 1) return;
    event.preventDefault();
    event.stopPropagation();
    const track = project.tracks.find((item) => item.id === clip.trackId);
    if (track?.locked) {
      setNotice(`Unlock ${track.name} before moving this clip.`);
      return;
    }
    timelineGestureCleanupRef.current?.();
    selectClip(clip.id);
    setSelectedAssetId(clip.assetId);
    const asset = allAssets.find((item) => item.id === clip.assetId);
    const isTouch = event.pointerType === "touch";
    let touchState: "pending" | "moving" | "editor" = isTouch ? "pending" : "moving";
    let longPressTimer: number | null = null;
    const beginTouchMove = () => {
      if (touchState === "pending") touchState = "moving";
    };
    if (isTouch) {
      longPressTimer = window.setTimeout(() => {
        if (touchState !== "pending") return;
        if (asset?.kind === "image") {
          touchState = "editor";
          suppressClipClickRef.current = clip.id;
          openImageEditor(asset, imageInputMode === "inpaint" ? "inpaint" : "edit");
        } else {
          beginTouchMove();
        }
      }, 420);
    }
    const originX = event.clientX;
    const initialStart = clip.start;
    let changed = false;
    const move = (pointerEvent: PointerEvent) => {
      if (isTouch && touchState === "pending") {
        if (Math.hypot(pointerEvent.clientX - event.clientX, pointerEvent.clientY - event.clientY) <= 8) return;
        if (longPressTimer != null) window.clearTimeout(longPressTimer);
        longPressTimer = null;
        beginTouchMove();
      }
      if (touchState !== "moving") return;
      const delta = (pointerEvent.clientX - originX) / zoom;
      const raw = snapEnabled ? Math.round((initialStart + delta) * project.fps) / project.fps : initialStart + delta;
      const nextStart = clampTime(raw, Math.max(0, project.duration - clip.duration));
      if (Math.abs(nextStart - initialStart) < 0.0001) return;
      changed = true;
      applyProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id ? { ...item, start: nextStart } : item),
      }));
    };
    const restore = () => {
      applyProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id ? { ...item, trackId: clip.trackId, start: clip.start } : item),
      }));
    };
    const up = (pointerEvent: PointerEvent) => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
      if (touchState !== "moving") return;

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
      const targetTrackId = lane?.dataset.studioTrackId || clip.trackId;
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
        pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
        applyProject((current) => ({
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
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
      restore();
    };
    const cleanup = () => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    timelineGestureCancelRef.current = cancel;
    event.currentTarget.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
  };

  const beginClipMove = (event: ReactPointerEvent<HTMLDivElement>, clip: StudioClip) => {
    if (event.button !== 0 || tool !== "select") return;
    if (event.pointerType === "touch" && timelinePointersRef.current.size > 1) return;
    event.preventDefault();
    event.stopPropagation();
    const currentSelection = selectedClipIdsRef.current;
    const dragIds = currentSelection.includes(clip.id)
      ? currentSelection
      : event.shiftKey ? [...currentSelection, clip.id] : [clip.id];
    const dragClips = project.clips.filter((item) => dragIds.includes(item.id));
    const locked = dragClips.find((item) => project.tracks.find((track) => track.id === item.trackId)?.locked);
    if (locked) {
      setNotice(`Unlock ${project.tracks.find((track) => track.id === locked.trackId)?.name || "the track"} before moving this clip.`);
      return;
    }
    timelineGestureCleanupRef.current?.();
    if (!currentSelection.includes(clip.id)) selectClip(clip.id, event.shiftKey);
    setSelectedAssetId(clip.assetId);
    const asset = allAssets.find((item) => item.id === clip.assetId);
    const isTouch = event.pointerType === "touch";
    let touchState: "pending" | "moving" | "editor" = isTouch ? "pending" : "moving";
    let longPressTimer: number | null = null;
    const beginTouchMove = () => { if (touchState === "pending") touchState = "moving"; };
    if (isTouch) {
      longPressTimer = window.setTimeout(() => {
        if (touchState !== "pending") return;
        if (asset?.kind === "image" && dragClips.length === 1) {
          touchState = "editor";
          suppressClipClickRef.current = clip.id;
          openImageEditor(asset, imageInputMode === "inpaint" ? "inpaint" : "edit");
        } else beginTouchMove();
      }, 420);
    }
    const originX = event.clientX;
    let changed = false;
    const buildPreview = (pointerEvent: PointerEvent): ClipDragPreview => {
      const lane = document.elementFromPoint(pointerEvent.clientX, pointerEvent.clientY)?.closest<HTMLElement>("[data-studio-track-id]");
      const targetTrackId = lane?.dataset.studioTrackId || clip.trackId;
      const targetTrack = project.tracks.find((item) => item.id === targetTrackId);
      const valid = Boolean(targetTrack && !targetTrack.locked && dragClips.every((item) => {
        const source = allAssets.find((candidate) => candidate.id === item.assetId);
        return source && targetTrack.kind === (source.kind === "audio" ? "audio" : "video");
      }));
      const delta = (pointerEvent.clientX - originX) / zoom;
      return {
        valid,
        clips: dragClips.map((item) => {
          const raw = snapEnabled ? Math.round((item.start + delta) * project.fps) / project.fps : item.start + delta;
          return { clipId: item.id, trackId: valid ? targetTrackId : item.trackId, start: clampTime(raw, Math.max(0, project.duration - item.duration)), duration: item.duration };
        }),
      };
    };
    const move = (pointerEvent: PointerEvent) => {
      if (isTouch && touchState === "pending") {
        if (Math.hypot(pointerEvent.clientX - event.clientX, pointerEvent.clientY - event.clientY) <= 8) return;
        if (longPressTimer != null) window.clearTimeout(longPressTimer);
        longPressTimer = null;
        beginTouchMove();
      }
      if (touchState !== "moving") return;
      const preview = buildPreview(pointerEvent);
      changed = preview.clips.some((candidate) => {
        const original = dragClips.find((item) => item.id === candidate.clipId);
        return original && (original.trackId !== candidate.trackId || Math.abs(original.start - candidate.start) >= 0.0001);
      });
      setClipDragPreview(preview);
    };
    const finishGesture = () => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
    };
    const up = (pointerEvent: PointerEvent) => {
      finishGesture();
      if (touchState !== "moving") { setClipDragPreview(null); return; }
      const generationTarget = document.elementFromPoint(pointerEvent.clientX, pointerEvent.clientY)?.closest("[data-studio-generation-drop]");
      if (generationTarget) {
        setClipDragPreview(null);
        if (changed) suppressClipClickRef.current = clip.id;
        void handleTimelineInput(clip, pointerEvent.shiftKey);
        return;
      }
      const preview = buildPreview(pointerEvent);
      setClipDragPreview(null);
      if (!preview.valid) { setNotice("This clip cannot be moved to the target track."); return; }
      const didChange = preview.clips.some((candidate) => {
        const original = dragClips.find((item) => item.id === candidate.clipId);
        return original && (original.trackId !== candidate.trackId || Math.abs(original.start - candidate.start) >= 0.0001);
      });
      if (!didChange) return;
      pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
      applyProject((current) => ({
        ...current,
        clips: current.clips.map((item) => {
          const candidate = preview.clips.find((next) => next.clipId === item.id);
          return candidate ? { ...item, trackId: candidate.trackId, start: candidate.start } : item;
        }),
        revision: current.revision + 1,
        updatedAt: new Date().toISOString(),
      }));
      suppressClipClickRef.current = clip.id;
    };
    const cancel = () => { finishGesture(); setClipDragPreview(null); };
    const cleanup = () => { if (longPressTimer != null) window.clearTimeout(longPressTimer); longPressTimer = null; setClipDragPreview(null); };
    timelineGestureCleanupRef.current = cleanup;
    timelineGestureCancelRef.current = cancel;
    event.currentTarget.setPointerCapture(event.pointerId);
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
    const isTouch = event.pointerType === "touch";
    let modifierHeld = event.ctrlKey || event.metaKey;
    let longPressTimer: number | null = null;
    if (isTouch) {
      longPressTimer = window.setTimeout(() => {
        modifierHeld = true;
      }, 360);
    }
    const originX = event.clientX;
    const initialEnd = clip.start + clip.duration;
    const snapshot = project;
    let changed = false;
    let stillNoticeShown = false;

    const update = (pointerEvent: PointerEvent) => {
      if (!isTouch) modifierHeld = pointerEvent.ctrlKey || pointerEvent.metaKey;
      if (isTouch && longPressTimer != null
        && Math.hypot(pointerEvent.clientX - event.clientX, pointerEvent.clientY - event.clientY) > 8) {
        window.clearTimeout(longPressTimer);
        longPressTimer = null;
      }
      const delta = (pointerEvent.clientX - originX) / zoom;
      let nextStart = clip.start;
      let nextEnd = initialEnd;
      let nextSourceIn = clip.sourceIn;
      let nextPresentation = clip.presentation || (asset.kind === "image" ? "frame" : "clip");
      if (edge === "start") {
        const rawStart = snapEnabled ? Math.round((clip.start + delta) * project.fps) / project.fps : clip.start + delta;
        const minimumStart = asset.kind === "image" && modifierHeld
          ? 0
          : Math.max(0, clip.start - clip.sourceIn);
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
      applyProject((current) => ({
        ...current,
        clips: current.clips.map((item) => item.id === clip.id
          ? { ...item, start: nextStart, duration: nextDuration, sourceIn: nextSourceIn, presentation: nextPresentation as StudioClip["presentation"] }
          : item),
      }));
    };
    const finish = () => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
      if (!changed) return;
      pushHistoryEntry({ project: snapshot, range, inpaintRange, referenceAssetIds });
      applyProject((current) => ({ ...current, revision: current.revision + 1, updatedAt: new Date().toISOString() }));
    };
    const cancel = () => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
      applyProject(snapshot);
    };
    const cleanup = () => {
      if (longPressTimer != null) window.clearTimeout(longPressTimer);
      longPressTimer = null;
      window.removeEventListener("pointermove", update);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    timelineGestureCancelRef.current = cancel;
    event.currentTarget.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", update);
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", cancel);
  };

  const beginClipSourcePress = (event: ReactPointerEvent<HTMLSpanElement>, clip: StudioClip) => {
    if (event.pointerType !== "touch") {
      event.stopPropagation();
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const originX = event.clientX;
    const originY = event.clientY;
    let timer: number | null = window.setTimeout(() => {
      timer = null;
      suppressClipClickRef.current = clip.id;
      void handleTimelineInput(clip, true);
    }, 420);
    const move = (pointerEvent: PointerEvent) => {
      if (Math.hypot(pointerEvent.clientX - originX, pointerEvent.clientY - originY) <= 8) return;
      cancel();
    };
    const finish = () => {
      if (timer != null) window.clearTimeout(timer);
      timer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
      if (timelineGestureCleanupRef.current === cleanup) timelineGestureCleanupRef.current = null;
      if (timelineGestureCancelRef.current === cancel) timelineGestureCancelRef.current = null;
    };
    const cancel = () => finish();
    const cleanup = () => {
      if (timer != null) window.clearTimeout(timer);
      timer = null;
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", cancel);
    };
    timelineGestureCleanupRef.current = cleanup;
    timelineGestureCancelRef.current = cancel;
    event.currentTarget.setPointerCapture(event.pointerId);
    window.addEventListener("pointermove", move);
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
    if (!modality.isVideo && plan.hasVideoInput && selectedAsset?.kind !== "image") {
      setNotice("An image model cannot use a video clip as its image input. Shift-drag a frame first.");
      return;
    }
    const referenceIds = Array.from(new Set(referenceAssetIds));
    const inferredMode: StudioGenerationMode = modality.isVideo
      && (selectedAsset?.kind === "image" || (!plan.hasVideoInput && plan.hasImageInput))
      ? "i2v"
      : plan.mode;
    const planMode: StudioGenerationMode = modality.isVideo && modelInfo?.variant === "ref2va" && referenceIds.length
      ? "ref2v"
      : inferredMode;
    if (modality.isVideo && referenceIds.length && modelInfo?.variant !== "ref2va") {
      setNotice("Explicit references require the MiniMax-H3 ref2va model; they are not inferred for fl2va or LTX.");
      return;
    }
    if (modality.isVideo && modelInfo?.variant === "ref2va" && !referenceIds.length) {
      setNotice("MiniMax-H3 ref2va requires at least one explicit image or video reference.");
      return;
    }
    const videoFrameRate = form.frameRate || project.fps;
    const rangeFrameCount = range
      ? Math.max(1, Math.round((plan.outputRange.end - plan.outputRange.start) * videoFrameRate))
      : null;
    const generatedFrameCount = modality.isVideo && range && planMode !== "inpaint" && planMode !== "outpaint"
      ? rangeFrameCount ?? undefined
      : form.numFrames;
    const videoConstraintsKnown = Boolean(loadedArch && archCapabilities?.video_constraints?.[loadedArch]);
    if (modality.isVideo && range && planMode !== "inpaint" && planMode !== "outpaint"
      && videoConstraintsKnown && !isValidVideoFrameCount(archCapabilities, loadedArch, generatedFrameCount)) {
      setNotice(`Output range resolves to ${generatedFrameCount} frames, which ${loadedArch} does not accept. Drag the range to a supported clip length.`);
      return;
    }
    if (planMode === "inpaint" && (loadedArch !== "minimax_h3" || (modelInfo?.variant != null && modelInfo.variant !== "fl2va"))) {
      setNotice("Temporal inpaint currently requires the MiniMax-H3 fl2va model.");
      return;
    }
    if (planMode === "outpaint" && modelInfo?.variant === "ref2va" && plan.videoClip
      && Math.abs(plan.outputRange.start - plan.videoClip.start) > 1 / (form.frameRate || project.fps)) {
      setNotice("MiniMax-H3 ref2va outpaint only supports extending forward from the clip start.");
      return;
    }
    if (planMode === "ref2v" && (loadedArch !== "minimax_h3" || modelInfo?.variant !== "ref2va")) {
      setNotice("Explicit reference video generation requires MiniMax-H3 ref2va.");
      return;
    }
    let generationPrompt = form.prompt;
    if (modality.isVideo && modality.modelInfo?.type === "minimax_h3") {
      try {
        const assisted = await maybeTransformH3PromptForGeneration({
          prompt: generationPrompt,
          mode: h3PromptModeForStudio(planMode),
          durationSeconds: Math.max(frameDuration(project.fps), plan.outputRange.end - plan.outputRange.start),
          references: studioH3References,
        });
        generationPrompt = assisted.prompt;
      } catch (error: any) {
        setNotice(error?.message || "MiniMax H3 Prompt Assist failed.");
        return;
      }
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
      const keyframeTimelineTime = Math.max(clip.start, Math.min(clipEnd(clip), playhead));
      const image = asset.kind === "image"
        ? asset
        : await captureVideoFrameAsset(asset, frameTimeForClip(clip, keyframeTimelineTime));
      return image?.url
        ? { image: image.url, frame_index: frameIndexAt(Math.max(0, keyframeTimelineTime - plan.outputRange.start), form.frameRate || project.fps) }
        : null;
    }));
    const keyframes = keyframeAssets.filter((item): item is { image: string; frame_index: number } => !!item);
    const firstKeyframe = keyframes[0] || (imageInput?.url ? {
      image: imageInput.url,
      frame_index: imageInput === selectedAsset
        ? frameIndexAt(Math.max(0, playhead - plan.outputRange.start), form.frameRate || project.fps)
        : 0,
    } : null);

    const jobId = newId();
    const resolvedModelName = safeModelLabel(modality.modelInfo?.name || modality.modelInfo?.source || currentModelName);
    const recipe: Record<string, unknown> = {
      model: resolvedModelName,
      architecture: modality.modelInfo?.type,
      model_variant: modality.modelInfo?.variant,
      mode: planMode,
      prompt: generationPrompt,
      source_prompt: form.prompt,
      negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
      width: form.width,
      height: form.height,
      num_frames: generatedFrameCount,
      frame_rate: form.frameRate,
      num_inference_steps: form.steps,
      guidance_scale: form.guidance,
      cfg_scale: form.guidance,
      sampler: form.sampler,
      schedule_type: form.scheduleType,
      denoising_strength: form.denoisingStrength,
      seed: form.seed,
      audio_enable: form.audioEnable,
      vae_path: studioVaePath,
      text_encoder_path: studioTextEncoderPath,
      output_range: range,
      inpaint_range: inpaintRange,
      source_clip_id: plan.videoClip?.id,
      keyframe_asset_id: imageInput?.id,
      reference_asset_ids: referenceIds,
    };
    setJobs((current) => [{ id: jobId, mode: planMode, prompt: generationPrompt, status: "running", startedAt: Date.now(), recipe }, ...current]);
    setRightPane("jobs");

    try {
      const baseVideoParameters = {
        prompt: generationPrompt,
        negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
        width: form.width,
        height: form.height,
        num_frames: generatedFrameCount,
        frame_rate: form.frameRate,
        num_inference_steps: form.steps,
        guidance_scale: form.guidance,
        seed: form.seed,
        audio_enable: form.audioEnable,
        vae_path: studioVaePath,
        text_encoder_path: studioTextEncoderPath,
      };
      let result: any;
      if (!modality.isVideo) {
        const imageParameters: GenerationParams = {
          prompt: generationPrompt,
          negative_prompt: supportsNegativePrompt ? form.negativePrompt : "",
          width: form.width,
          height: form.height,
          steps: form.steps,
          cfg_scale: form.guidance,
          sampler: form.sampler,
          schedule_type: form.scheduleType,
          seed: form.seed,
          vae_path: studioVaePath,
          text_encoder_path: studioTextEncoderPath,
        };
        if (planMode === "image-inpaint" || imageInputMode === "inpaint" || !!imageInput?.maskUrl || (!!inpaintRange && !!imageInput)) {
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
        ? (generatedFrameCount || 1) / (form.frameRate || project.fps)
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
        prompt: generationPrompt,
        negativePrompt: supportsNegativePrompt ? form.negativePrompt : "",
        generationType: planMode,
        modelName: resolvedModelName,
        seed: form.seed,
        parameters: recipe,
      });
      const targetStart = planMode === "outpaint"
        ? plan.outputRange.start
        : range?.start ?? playhead;
      const targetDuration = planMode === "outpaint"
        ? Math.max(frameDuration(project.fps), plan.outputRange.end - plan.outputRange.start)
        : range
          ? Math.max(frameDuration(project.fps), range.end - range.start)
          : (asset.duration || fallbackDuration || frameDuration(project.fps));
      const takeGroupId = selectedClip?.takeGroupId || (selectedClip ? newId() : undefined);
      const selectedTrack = selectedClip ? project.tracks.find((track) => track.id === selectedClip.trackId) : null;
      const clip: StudioClip = {
        id: newId(),
        assetId: asset.id,
        trackId: selectedTrack?.kind === "video" ? selectedTrack.id : "video-1",
        name: filename,
        start: planMode === "outpaint" ? targetStart : selectedClip?.start ?? targetStart,
        duration: generatedKind === "image"
          ? frameDuration(project.fps)
          : Math.min(
            planMode === "outpaint" ? targetDuration : selectedClip?.duration ?? targetDuration,
            asset.duration || targetDuration,
          ),
        sourceIn: 0,
        presentation: generatedKind === "image" ? "frame" : "clip",
        sourceDuration: asset.duration || targetDuration,
        ...(generatedKind !== "audio" && form.width && form.height && (form.width !== project.width || form.height !== project.height)
          ? { fitMode: "cover" as const }
          : {}),
        takeGroupId,
        // No clip to review against yet, so make it active immediately;
        // a take that replaces a selected clip still waits for review.
        activeTake: !selectedClip,
        generated: true,
      };
      commit((current) => ({
        ...current,
        assets: current.assets.some((item) => item.id === asset.id) ? current.assets : [...current.assets, asset],
        clips: [...current.clips.map((item) => item.id === selectedClip?.id ? { ...item, takeGroupId } : item), clip],
      }));
      setSelectedAssetId(asset.id);
      selectClip(clip.id);
      setResultAssetIds((current) => [asset.id, ...current]);
      setJobs((current) => current.map((job) => job.id === jobId
        ? { ...job, status: (selectedClip ? "review" : "applied") as const, assetId: asset.id }
        : job));
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
    setStudioVaePath(typeof recipe.vae_path === "string" ? recipe.vae_path : null);
    setStudioTextEncoderPath(typeof recipe.text_encoder_path === "string" ? recipe.text_encoder_path : null);
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
    const nextOutputRange = recipe.output_range && typeof recipe.output_range === "object" ? recipe.output_range as StudioRange : range;
    const nextInpaintRange = recipe.inpaint_range && typeof recipe.inpaint_range === "object" ? recipe.inpaint_range as StudioRange : inpaintRange;
    if (nextOutputRange !== range || nextInpaintRange !== inpaintRange) {
      pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
      setRange(nextOutputRange);
      setInpaintRange(nextInpaintRange);
    }
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
      selectClip(clip.id);
      setSelectedAssetId(clip.assetId);
    }
    setJobs((current) => current.map((item) => item.id === job.id ? { ...item, status: "applied" as const } : item));
  };

  const pollRenderJob = useCallback(async (jobId: string) => {
    if (renderControllersRef.current.has(jobId)) return;
    const controller = new AbortController();
    renderControllersRef.current.set(jobId, controller);
    setRendering(true);
    let transientFailures = 0;
    try {
      while (!controller.signal.aborted && !studioUnmountedRef.current) {
        let status: any;
        try {
          status = await getStudioRenderJob(jobId, controller.signal);
          transientFailures = 0;
        } catch (error) {
          if (controller.signal.aborted || studioUnmountedRef.current) return;
          const responseStatus = error && typeof error === "object" && "response" in error
            ? Number((error as { response?: { status?: number } }).response?.status)
            : 0;
          if (responseStatus === 404) throw new Error("Studio render job was not found.");
          transientFailures += 1;
          setNotice("Render status is temporarily unavailable; retrying…");
          await new Promise<void>((resolve, reject) => {
            const timer = window.setTimeout(resolve, Math.min(5000, 1000 * transientFailures));
            controller.signal.addEventListener("abort", () => {
              window.clearTimeout(timer);
              reject(new DOMException("Render polling cancelled", "AbortError"));
            }, { once: true });
          });
          continue;
        }
        if (controller.signal.aborted || studioUnmountedRef.current) return;
        setRenderProgress(Math.max(0, Math.min(1, Number(status.progress) || 0)));
        if (status.state === "completed") {
          const image = status.image;
          if (!image) throw new Error("Render completed without a Gallery result.");
          const fallbackFilename = String(status.filename || image.filename || `studio-render-${jobId}.mp4`);
          const asset = studioAssetFromGeneration({ image }, {
            id: `studio-render-${jobId}`,
            filename: fallbackFilename,
            kind: "video",
            url: `/outputs/${image.preview_filename || fallbackFilename}`,
            masterUrl: `/outputs/${fallbackFilename}`,
            duration: project.duration,
            width: project.width,
            height: project.height,
            source: "generation",
            generationType: "studio_render",
            modelName: "Studio timeline renderer",
            seed: -1,
            parameters: { studio_project_id: project.id, studio_render_job_id: jobId },
          });
          applyProject((current) => current.assets.some((item) => item.id === asset.id)
            ? { ...current, renderJobId: undefined }
            : { ...current, renderJobId: undefined, assets: [...current.assets, asset], revision: current.revision + 1, updatedAt: new Date().toISOString() });
          setResultAssetIds((current) => [asset.id, ...current.filter((id) => id !== asset.id)]);
          setSelectedAssetId(asset.id);
          const renderWarnings: string[] = Array.isArray(status.warnings)
            ? status.warnings.filter((entry: unknown): entry is string => typeof entry === "string")
            : [];
          setNotice(`Timeline rendered and registered in Gallery as ${asset.name}.${renderWarnings.length ? ` ${renderWarnings.join(" ")}` : ""}`);
          refreshLibrary();
          return;
        }
        if (status.state === "failed") throw new Error(status.error || "Studio render failed.");
        if (status.state === "cancelled") throw new Error("Studio render cancelled.");
        await new Promise<void>((resolve, reject) => {
          const timer = window.setTimeout(resolve, 1000);
          controller.signal.addEventListener("abort", () => {
            window.clearTimeout(timer);
            reject(new DOMException("Render polling cancelled", "AbortError"));
          }, { once: true });
        });
      }
    } catch (error) {
      if (!controller.signal.aborted && !studioUnmountedRef.current) {
        applyProject((current) => ({ ...current, renderJobId: undefined }));
        setNotice(error instanceof Error ? error.message : "Studio render failed.");
      }
    } finally {
      renderControllersRef.current.delete(jobId);
      if (!studioUnmountedRef.current) {
        setRendering(false);
        setRenderJobId((current) => current === jobId ? null : current);
      }
    }
    // Deliberately depends on only the project fields this reads, not the
    // whole `project` object, or every clip edit during a render would
    // recreate this poller and restart its abort-controller bookkeeping.
  }, [applyProject, project.duration, project.height, project.id, project.width, refreshLibrary]);

  useEffect(() => {
    if (!restored || !renderJobId) return;
    void pollRenderJob(renderJobId);
  }, [pollRenderJob, renderJobId, restored]);

  const renderTimeline = async () => {
    if (rendering) return;
    setNotice(null);
    // Preserve muted video pictures while excluding muted audio tracks.
    const renderTracks = project.tracks.filter((track) => track.visible);
    const renderTrackIds = new Set(renderTracks.map((track) => track.id));
    const mutedAudioTrackIds = new Set(
      project.tracks.filter((track) => track.kind === "audio" && track.muted).map((track) => track.id),
    );
    const renderClips = project.clips.filter((clip) =>
      clip.activeTake !== false && renderTrackIds.has(clip.trackId) && !mutedAudioTrackIds.has(clip.trackId),
    );
    if (!renderClips.length) {
      setNotice("Add a visible, active clip before rendering the timeline.");
      return;
    }

    setRendering(true);
    setRenderProgress(0);
    try {
      const assetMap = new Map(allAssets.map((asset) => [asset.id, asset]));
      const requiredAssets = Array.from(new Set(renderClips.map((clip) => clip.assetId)))
        .map((assetId) => assetMap.get(assetId))
        .filter((asset): asset is StudioAsset => !!asset);
      if (requiredAssets.length !== new Set(renderClips.map((clip) => clip.assetId)).size) {
        throw new Error("A timeline clip refers to a missing media asset.");
      }

      const hydratedAssets = await Promise.all(requiredAssets.map((asset) => hydrateGalleryAsset(asset)));
      const uploads = [] as { assetId: string; file: File }[];
      const manifestAssets = [] as Record<string, unknown>[];
      for (const asset of hydratedAssets) {
        if (asset.galleryId == null) {
          uploads.push({ assetId: asset.id, file: await mediaFileForUpload(asset) });
        }
        manifestAssets.push({
          id: asset.id,
          name: asset.name,
          kind: asset.kind,
          galleryId: asset.galleryId,
          duration: asset.duration,
          width: asset.width,
          height: asset.height,
        });
      }

      const manifest: Record<string, unknown> = {
        schemaVersion: project.schemaVersion,
        project: {
          id: project.id,
          revision: project.revision,
          name: project.name,
          duration: project.duration,
          fps: project.fps,
          width: project.width,
          height: project.height,
        },
        render: { audio_enabled: true, fit_mode: "cover" },
        assets: manifestAssets,
        tracks: renderTracks.map((track) => ({
          id: track.id,
          name: track.name,
          kind: track.kind,
          muted: track.muted,
          visible: track.visible,
        })),
        clips: renderClips.map((clip) => ({
          id: clip.id,
          assetId: clip.assetId,
          trackId: clip.trackId,
          start: clip.start,
          duration: clip.duration,
          sourceIn: clip.sourceIn,
          presentation: clip.presentation,
          fitMode: clip.fitMode,
          activeTake: true,
        })),
      };

      const queued = await renderStudioProject(manifest, uploads);
      const jobId = String(queued.job_id || "");
      if (!jobId) throw new Error("The server did not return a Studio render job id.");
      setRenderJobId(jobId);
      applyProject((current) => ({ ...current, renderJobId: jobId, revision: current.revision + 1, updatedAt: new Date().toISOString() }));
    } catch (error) {
      setRendering(false);
      setNotice(error instanceof Error ? error.message : "Studio render failed.");
    }
  };

  const cancelTimelineRender = async () => {
    if (!renderJobId) return;
    try {
      await cancelStudioRenderJob(renderJobId);
      setNotice("Cancelling Studio render...");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not cancel Studio render.");
    }
  };

  const saveProjectFile = () => {
    const snapshot = { ...projectRef.current, jobs, outputRange: range, inpaintRange, referenceAssetIds };
    const manifest = new Blob([serializeStudioProject(snapshot)], { type: "application/json" });
    const url = URL.createObjectURL(manifest);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = projectFileName(snapshot.name);
    anchor.click();
    // Revoking synchronously after click() works in Chromium today, but the
    // spec only guarantees the URL is valid for as long as the download is
    // in flight; deferring the revoke avoids racing that on other engines.
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setRecentProjects(rememberRecentProject(snapshot));
    setNotice("Project saved.");
  };

  const createNewProject = () => {
    if ((project.clips.length || project.assets.length) && !window.confirm("Start a new Studio project? The current project will remain in Recent.")) return;
    const currentSnapshot = { ...projectRef.current, jobs, outputRange: range, inpaintRange, referenceAssetIds };
    const next = createStudioProject();
    setRecentProjects(rememberRecentProject(currentSnapshot));
    applyProject(next);
    setRange(null);
    setInpaintRange(null);
    setReferenceAssetIds([]);
    rangeRef.current = null;
    inpaintRangeRef.current = null;
    referenceAssetIdsRef.current = [];
    setJobs([]);
    setResultAssetIds([]);
    clearClipSelection();
    setSelectedAssetId(null);
    setUndoStack([]);
    setRedoStack([]);
    setRecentProjects(rememberRecentProject(next));
    setNotice("New Studio project created.");
  };

  const openRecentProject = async (entry: StudioRecentProject) => {
    const assets = await Promise.all(entry.manifest.assets.map(async (asset) => {
      if (!asset.blobKey) return asset;
      const blob = await loadImportedMedia(asset.blobKey);
      if (!blob) return { ...asset, url: "", thumbnailUrl: undefined, missing: true };
      const url = URL.createObjectURL(blob);
      return { ...asset, url, thumbnailUrl: asset.kind === "image" ? url : undefined, missing: false };
    }));
    const restored = { ...entry.manifest, assets };
    applyProject(restored);
    setRange(restored.outputRange ?? null);
    setInpaintRange(restored.inpaintRange ?? null);
    setReferenceAssetIds(restored.referenceAssetIds ?? []);
    rangeRef.current = restored.outputRange ?? null;
    inpaintRangeRef.current = restored.inpaintRange ?? null;
    referenceAssetIdsRef.current = restored.referenceAssetIds ?? [];
    setJobs(restored.jobs ?? []);
    setResultAssetIds((restored.jobs ?? []).flatMap((job) => job.assetId ? [job.assetId] : []));
    clearClipSelection();
    setSelectedAssetId(null);
    setUndoStack([]);
    setRedoStack([]);
    setRecentProjects(rememberRecentProject(restored));
    setRecentProjectsOpen(false);
    setNotice(`Opened ${restored.name}.`);
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

  // Recomputed on every playhead tick during playback (up to 60fps), so this
  // has to skip re-scanning every clip/asset unless something that could
  // change the resolved plan actually changed.
  const resolvedPlan = useMemo(() => planStudioGeneration({
    isVideoModel,
    fps: form.frameRate || project.fps,
    projectDuration: project.duration,
    playhead,
    outputRange: range,
    inpaintRange,
    selectedClipId,
    clips: activeClips,
    assets: allAssets,
  }), [activeClips, allAssets, form.frameRate, inpaintRange, isVideoModel, playhead, project.duration, project.fps, range, selectedClipId]);
  const hasReferenceInput = referenceAssetIds.length > 0;
  const resolvedMode: StudioGenerationMode = isVideoModel && modelInfo?.variant === "ref2va" && hasReferenceInput
    ? "ref2v"
    : !isVideoModel && selectedAsset?.kind === "image"
      ? (imageInputMode === "inpaint" || !!selectedAsset?.maskUrl || inpaintRange ? "image-inpaint" : "i2i")
      : isVideoModel && selectedAsset?.kind === "image"
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
  const studioH3Mode = h3PromptModeForStudio(resolvedMode);
  const studioH3References = useMemo(() => {
    let pictures = 0;
    let videos = 0;
    let audios = 0;
    const countAsset = (assetId: string) => {
      const asset = allAssets.find((item) => item.id === assetId);
      if (asset?.kind === "image") pictures += 1;
      else if (asset?.kind === "video") videos += 1;
      else if (asset?.kind === "audio") audios += 1;
    };
    activeClips
      .filter((clip) => clip.inputRoles?.includes("keyframe"))
      .forEach((clip) => countAsset(clip.assetId));
    referenceAssetIds.forEach(countAsset);
    return createH3ReferenceInventory({ pictures, videos, audios });
  }, [activeClips, allAssets, referenceAssetIds]);
  const studioH3Duration = Math.max(frameDuration(project.fps), outputDuration || project.duration);

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
    selectClip(target.id);
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
            onChange={(event) => applyProject((current) => ({ ...current, name: event.target.value, revision: current.revision + 1, updatedAt: new Date().toISOString() }))}
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
          <button className={styles.sequenceBadge} onClick={() => setProjectSettingsOpen((open) => !open)} aria-expanded={projectSettingsOpen} title="Open project settings">{project.width}×{project.height} · {project.fps} fps</button>
          <button className={styles.projectImportButton} onClick={createNewProject}><Plus size={14} /> New</button>
          <button className={styles.projectImportButton} onClick={() => projectFileInputRef.current?.click()}><FolderOpen size={14} /> Open</button>
          <button className={styles.projectImportButton} onClick={() => setRecentProjectsOpen((open) => !open)} aria-expanded={recentProjectsOpen}><Clock3 size={14} /> Recent</button>
          <input ref={projectFileInputRef} type="file" accept=".sushistudio,.json,application/json" hidden onChange={handleProjectImport} />
          <button className={styles.exportButton} onClick={saveProjectFile}><Upload size={16} /> Save</button>
          <button className={styles.renderButton} onClick={renderTimeline} disabled={rendering}>
            <Film size={15} /> {rendering ? `Rendering ${Math.round(renderProgress * 100)}%` : "Render video"}
          </button>
          {rendering && <button className={styles.cancelRenderButton} onClick={cancelTimelineRender}>Cancel</button>}
        </div>
      </header>

      {projectSettingsOpen && (
        <section className={styles.projectSettingsPopover} role="dialog" aria-label="Project settings">
          <div className={styles.sectionTitle}><strong>Project settings</strong><button onClick={() => setProjectSettingsOpen(false)} aria-label="Close project settings"><X size={14} /></button></div>
          <p className={styles.projectSettingsNote}>Canvas is the final preview/render surface. Generate Width/Height are model output settings.</p>
          <div className={styles.canvasFields}>
            <label>Width<input type="number" min={64} max={8192} step={16} value={canvasDraft.width} onChange={(event) => updateCanvasDraft("width", event.target.value)} /></label>
            <label>Height<input type="number" min={64} max={8192} step={16} value={canvasDraft.height} onChange={(event) => updateCanvasDraft("height", event.target.value)} /></label>
          </div>
          <div className={styles.canvasSliders}>
            <label>Width<input type="range" min={64} max={8192} step={16} value={Number(canvasDraft.width) || project.width} onChange={(event) => updateCanvasDraft("width", event.target.value)} /></label>
            <label>Height<input type="range" min={64} max={8192} step={16} value={Number(canvasDraft.height) || project.height} onChange={(event) => updateCanvasDraft("height", event.target.value)} /></label>
          </div>
          <label className={styles.toggleField}><span>Lock aspect ratio</span><input type="checkbox" checked={canvasAspectLocked} onChange={(event) => setCanvasAspectLocked(event.target.checked)} /></label>
          <div className={styles.canvasPresets}>
            {[{ label: "16:9", width: 1920, height: 1080 }, { label: "9:16", width: 1080, height: 1920 }, { label: "1:1", width: 1080, height: 1080 }].map((preset) => (
              <button key={preset.label} onClick={() => { setCanvasAspectLocked(false); setCanvasDraft({ width: String(preset.width), height: String(preset.height) }); }}>{preset.label}</button>
            ))}
          </div>
          <button className={styles.applySettingsButton} onClick={() => { commitCanvasSize(); setProjectSettingsOpen(false); }}>Apply canvas size</button>
        </section>
      )}
      {recentProjectsOpen && (
        <section className={styles.recentProjectsPopover} role="dialog" aria-label="Recent projects">
          <div className={styles.sectionTitle}><strong>Recent projects</strong><button onClick={() => setRecentProjectsOpen(false)} aria-label="Close recent projects"><X size={14} /></button></div>
          {!recentProjects.length && <small className={styles.emptyRecent}>No recent projects yet.</small>}
          {recentProjects.map((entry) => (
            <button key={entry.id} className={styles.recentProjectItem} onClick={() => void openRecentProject(entry)}>
              <span><strong>{entry.name}</strong><small>{entry.width}×{entry.height} · {entry.assetCount} assets</small></span>
              <time>{new Date(entry.updatedAt).toLocaleDateString()}</time>
            </button>
          ))}
        </section>
      )}

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
                onPointerDown={(event) => beginAssetPress(event, asset)}
                onPointerMove={moveAssetPress}
                onPointerUp={finishAssetPress}
                onPointerCancel={finishAssetPress}
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
                onClick={() => { selectAsset(asset); clearClipSelection(); }}
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
              <div className={styles.previewCanvas} style={{ aspectRatio: `${project.width} / ${project.height}` }}>
              {previewAsset?.url && previewAsset.kind === "video" ? (
                <video ref={videoRef} src={previewAsset.url} muted={previewTrack?.muted ?? false} playsInline style={{ objectFit: previewClip?.fitMode === "contain" ? "contain" : "cover" }} onLoadedMetadata={(event) => { if (previewClip) event.currentTarget.currentTime = previewClip.sourceIn + Math.max(0, playhead - previewClip.start); if (playing) void event.currentTarget.play(); }} />
              ) : previewAsset?.url && previewAsset.kind === "image" ? (
                <NextImage src={previewAsset.url} alt={previewAsset.name} fill sizes="(max-width: 1180px) 55vw, 45vw" unoptimized style={{ objectFit: previewClip?.fitMode === "contain" ? "contain" : "cover" }} />
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
            </div>
            <div className={styles.transport}>
              <span>{formatTimecode(playhead, project.fps)} <small>/ {formatTimecode(project.duration, project.fps)}</small></span>
              <div>
                <button className={styles.seekButton} onPointerDown={(event) => beginSeekRepeat(event, -1)} onPointerUp={() => finishSeekRepeat(-1)} onPointerCancel={stopSeekRepeat} onClick={(event) => { if (event.detail === 0) seekBy(-5); }} aria-label="Rewind five seconds" title="Rewind 5 seconds; hold to continue"><Rewind size={15} /></button>
                <button className={styles.seekButton} onClick={() => seekTimeline(0)} aria-label="Go to start" title="Go to start (Home)"><RotateCcw size={15} /></button>
                <button onClick={(event) => { event.stopPropagation(); togglePlayback(); }} className={styles.playButton} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={19} /> : <Play size={19} fill="currentColor" />}</button>
                <button className={styles.seekButton} onPointerDown={(event) => beginSeekRepeat(event, 1)} onPointerUp={() => finishSeekRepeat(1)} onPointerCancel={stopSeekRepeat} onClick={(event) => { if (event.detail === 0) seekBy(5); }} aria-label="Fast-forward five seconds" title="Fast-forward 5 seconds; hold to continue"><FastForward size={15} /></button>
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
              <span className={styles.touchHint}>Tap scrub / swipe move / hold edit / pinch zoom</span>
              <div className={styles.zoomControls}>
                <ZoomOut size={14} /><input aria-label="Timeline zoom" type="range" min="8" max="48" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><ZoomIn size={14} />
              </div>
            </div>

            <div className={`${styles.timeline} ${tool === "hand" ? styles.handTool : ""}`}>
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
              <div
                ref={timelineScrollRef}
                className={styles.timelineScroll}
                onPointerDown={beginTimelinePan}
                onPointerDownCapture={handleTimelinePointerDownCapture}
                onPointerMoveCapture={handleTimelinePointerMoveCapture}
                onPointerUpCapture={finishTimelinePointer}
                onPointerCancelCapture={finishTimelinePointer}
              >
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
                      onPointerDown={beginTrackScrub}
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => handleTrackDrop(event, track.id)}
                      onClick={(event) => {
                        if (event.target === event.currentTarget) {
                          const bounds = event.currentTarget.getBoundingClientRect();
                          seekTimeline(Math.max(0, Math.min(
                            project.duration,
                            ((event.clientX - bounds.left) + (timelineScrollRef.current?.scrollLeft || 0)) / zoom,
                          )));
                          clearClipSelection();
                          setSelectedAssetId(null);
                        }
                      }}
                    >
                      {track.visible && activeClips.filter((clip) => clip.trackId === track.id).map((clip) => {
                        const asset = allAssets.find((item) => item.id === clip.assetId);
                        const clipFrameCount = Math.max(1, Math.round(clip.duration * project.fps));
                        const dragCandidate = clipDragPreview?.clips.find((item) => item.clipId === clip.id);
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
                              if (tool === "blade") {
                                const bounds = event.currentTarget.getBoundingClientRect();
                                const splitTime = clampTime(clip.start + (event.clientX - bounds.left) / zoom, project.duration);
                                splitSelectedClip(clip, splitTime);
                                return;
                              }
                              selectClip(clip.id, event.shiftKey);
                              setSelectedAssetId(clip.assetId);
                              setRightPane("inspector");
                            }}
                            onKeyDown={(event) => {
                              if (event.key !== "Enter") return;
                              event.preventDefault();
                              selectClip(clip.id, event.shiftKey);
                              setSelectedAssetId(clip.assetId);
                              setRightPane("inspector");
                            }}
                            className={`${styles.timelineClip} ${asset?.kind === "audio" ? styles.audioClip : ""} ${asset?.missing ? styles.missingClip : ""} ${clip.generated ? styles.generatedClip : ""} ${selectedClipIds.includes(clip.id) ? styles.selectedClip : ""} ${selectedClipIds.length > 0 && !selectedClipIds.includes(clip.id) ? styles.dimmedClip : ""} ${dragCandidate ? styles.draggingClip : ""} ${clip.presentation === "frame" ? styles.stillClip : ""}`}
                            style={{ left: clip.start * zoom, width: Math.max(18, clip.duration * zoom), backgroundImage: asset?.thumbnailUrl && asset.kind !== "audio" ? `linear-gradient(90deg, rgba(8,12,18,.42), rgba(8,12,18,.08)), url(${asset.thumbnailUrl})` : undefined, backgroundSize: clip.fitMode === "contain" ? "contain" : "cover", backgroundRepeat: "no-repeat" }}
                          >
                            <button className={styles.trimStart} onPointerDown={(event) => beginTrim(event, clip, "start")} aria-label={`Trim start of ${clip.name}`} title={asset?.kind === "image" && clip.presentation === "frame" ? "Ctrl/Cmd + drag to extend still" : `Trim start of ${clip.name}`} />
                            <span className={styles.clipName}>{clip.name}</span>
                            {asset?.missing && <span className={styles.missingClipLabel}>Missing media</span>}
                            <span className={styles.clipInputControls} onPointerDown={(event) => event.stopPropagation()}>
                              {asset && asset.kind !== "audio" && (
                                <label title="Use as keyframe">
                                  <input type="checkbox" checked={clip.inputRoles?.includes("keyframe") || false} onChange={() => toggleClipInputRole(clip.id, "keyframe")} />
                                  <span>K</span>
                                </label>
                              )}
                            </span>
                              <span
                                className={styles.clipSourceHandle}
                                draggable={!track.locked}
                                onPointerDown={(event) => beginClipSourcePress(event, clip)}
                              onDragStart={(event) => {
                                event.dataTransfer.effectAllowed = "copy";
                                event.dataTransfer.setData("application/x-studio-clip", clip.id);
                                event.dataTransfer.setData("application/x-studio-frame-time", String(frameTimeForClip(clip, playhead)));
                                if (event.shiftKey) event.dataTransfer.setData("application/x-studio-input-mode", "frame");
                              }}
                               title="Drag to Generate; Shift or long-press: current frame"
                            ><ImagePlus size={10} /></span>
                            {clip.linkGroupId && <Link2 size={11} className={styles.linkBadge} />}
                            {clip.generated && <Sparkles size={11} className={styles.generationBadge} />}
                            {asset?.kind === "audio" && <span className={styles.waveform} />}
                            <button className={styles.trimEnd} onPointerDown={(event) => beginTrim(event, clip, "end")} aria-label={`Trim end of ${clip.name}`} title={asset?.kind === "image" && clip.presentation === "frame" ? "Ctrl/Cmd + drag to extend still" : `Trim end of ${clip.name}`} />
                            {hoveredClipId === clip.id && asset?.kind === "image" && (
                              <span className={styles.stillPopover} role="tooltip">
                                {(asset.url || asset.thumbnailUrl)
                                  ? <NextImage src={asset.url || asset.thumbnailUrl || ""} alt="" width={164} height={92} unoptimized />
                                  : <span className={styles.stillPopoverMissingThumb}><ImageIcon size={18} /></span>}
                                <strong>{asset.name}</strong>
                                <small>{asset.width && asset.height ? `${asset.width}×${asset.height}` : "Still image"} · {clipFrameCount === 1 ? "1 frame" : `${clipFrameCount} frames`}</small>
                              </span>
                            )}
                          </div>
                        );
                      })}
                      {track.visible && clipDragPreview?.clips.filter((candidate) => candidate.trackId === track.id).map((candidate) => {
                        const sourceClip = project.clips.find((item) => item.id === candidate.clipId);
                        const asset = sourceClip ? allAssets.find((item) => item.id === sourceClip.assetId) : null;
                        if (!sourceClip) return null;
                        return <div key={`drag-ghost-${candidate.clipId}`} className={`${styles.timelineClip} ${styles.dragGhost} ${!clipDragPreview.valid ? styles.dragGhostInvalid : ""}`} style={{ left: candidate.start * zoom, width: Math.max(18, candidate.duration * zoom), backgroundImage: asset?.thumbnailUrl ? `url(${asset.thumbnailUrl})` : undefined }}><span className={styles.clipName}>{sourceClip.name}</span></div>;
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
              onDragLeave={(event) => {
                // A drag over a child element fires leave+enter on the
                // parent too; only clear the highlight once the pointer has
                // actually left this container (not just moved to a child).
                if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
                setGenerationDropActive(false);
              }}
              onDrop={async (event) => { event.preventDefault(); setGenerationDropActive(false); setFrameDropLoading(true); try { await handleRightPaneDrop(event); } finally { setFrameDropLoading(false); } }}
            >
              <details className={styles.modelLoadDisclosure}>
                <summary><span>Model &amp; components</span><small>{currentModelName}</small></summary>
                <ModelLoadSection
                  storageKeyPrefix="studio"
                  vaePath={studioVaePath}
                  onVaePathChange={setStudioVaePath}
                  textEncoderPath={studioTextEncoderPath}
                  onTextEncoderChange={setStudioTextEncoderPath}
                  onModelLoad={async () => {
                    await refreshModelInfo();
                    setNotice("Studio model updated. Review the generation settings before submitting.");
                  }}
                />
              </details>
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
                {(range || inpaintRange) && <button className={styles.clearRanges} onClick={() => {
                  pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds });
                  setRange(null);
                  setInpaintRange(null);
                }}>Clear ranges</button>}
              </section>
              <section>
                <label className={styles.fieldLabel} htmlFor="studio-prompt">Prompt</label>
                <div className={styles.promptBox}>
                  <textarea id="studio-prompt" value={form.prompt} onChange={(event) => setForm((current) => ({ ...current, prompt: event.target.value }))} placeholder="Describe the shot, camera motion, lighting, and continuity…" maxLength={1000} />
                  <span>{form.prompt.length}/1000</span>
                </div>
              </section>
              {loadedArch === "minimax_h3" && isVideoModel && (
                <H3PromptAssist
                  prompt={form.prompt}
                  onApply={(prompt) => setForm((current) => ({ ...current, prompt }))}
                  suggestedMode={studioH3Mode}
                  durationSeconds={studioH3Duration}
                  references={studioH3References}
                />
              )}
              {isVideoModel && (
                <section>
                  <label className={styles.fieldLabel}>Timeline inputs</label>
                  <div className={styles.inputSummary}>
                    {activeClips.filter((clip) => clip.inputRoles?.includes("keyframe")).map((clip) => <span key={clip.id}><Check size={11} /> {clip.name}</span>)}
                    {!activeClips.some((clip) => clip.inputRoles?.includes("keyframe")) && <small>Tick K on a timeline clip to use it as a keyframe.</small>}
                  </div>
                </section>
              )}
              {selectedAsset?.kind === "image" && (
                <section className={styles.inputCard}>
                  <div className={styles.sectionTitle}><span>{isVideoModel ? "Image keyframe" : "Input image"}</span><button onClick={() => { setSelectedAssetId(null); clearClipSelection(); setImageInputMode("i2i"); }}><X size={12} /></button></div>
                  <div className={styles.keyframeSlot}>
                    {(selectedAsset.thumbnailUrl || selectedAsset.url)
                      ? <NextImage src={selectedAsset.thumbnailUrl || selectedAsset.url} alt="" width={74} height={48} unoptimized />
                      : <span className={styles.missingThumb}><ImageIcon size={18} /></span>}
                    <span><strong>{selectedAsset.name}</strong><small>{isVideoModel ? "I2VA anchor" : imageInputMode === "inpaint" ? "Mask enabled" : "I2I input"}</small></span>
                  </div>
                  {!isVideoModel && <div className={styles.inputActions}><button onClick={() => openImageEditor(selectedAsset, "edit")}><ImagePlus size={13} /> Edit image</button><button onClick={() => { setImageInputMode("inpaint"); openImageEditor(selectedAsset, "inpaint"); }}>Mask / inpaint</button></div>}
                </section>
              )}
              {isVideoModel && (
                <section
                  className={styles.referenceDropCard}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={handleReferenceDrop}
                >
                  <div className={styles.sectionTitle}><span>Explicit references</span><small>Media only · drag here</small></div>
                  <div className={styles.referenceList}>{referenceAssetIds.map((assetId) => { const asset = allAssets.find((item) => item.id === assetId); return asset ? <button key={assetId} onClick={() => { pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds }); setReferenceAssetIds((current) => current.filter((id) => id !== assetId)); }}>{asset.name}<X size={11} /></button> : null; })}{!referenceAssetIds.length && <small>References are never inferred from clips.</small>}</div>
                  {selectedAsset && <button className={styles.addReferenceButton} onClick={() => { if (referenceAssetIds.includes(selectedAsset.id)) return; pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds }); setReferenceAssetIds((current) => [...current, selectedAsset.id]); }}>Add selected media as reference</button>}
                  {frameDropLoading && <small className={styles.dropHint}>Capturing the requested frame…</small>}
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
                  <button onClick={() => { pushHistoryEntry({ project, range, inpaintRange, referenceAssetIds }); setRange(null); }}>Clear</button>
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
                <div className={styles.sectionTitle}><span>Generation results</span><button onClick={() => { setAssetFilters((current) => ({ ...current, scope: "generation" })); setMediaFilter("all"); setFiltersOpen(true); }}>See all</button></div>
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
                    {selectedAsset?.kind !== "audio" && <label>Canvas fit<select value={selectedClip.fitMode || "cover"} onChange={(event) => updateSelectedClip({ fitMode: event.target.value as StudioClipFitMode })}><option value="cover">Fill and crop</option><option value="contain">Fit and letterbox</option></select></label>}
                  </div>
                  <section className={styles.inspectorSection}><div><strong>Link group</strong><small>Stream-aware linked A/V editing is planned for the backend media-import phase.</small></div><button disabled><Link2 size={14} /> Unavailable</button></section>
                  <section className={styles.inspectorSection}><div><strong>Generation context</strong><small>Use this clip as the next shot&apos;s visual context.</small></div><button onClick={() => setRightPane("generate")}>Generate from clip</button></section>
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
      {pendingPlacement && (
        <div className={styles.canvasFitOverlay} role="dialog" aria-modal="true" aria-label="Fit media to project canvas">
          <div className={styles.canvasFitDialog}>
            {pendingPlacement.asset.url && <NextImage className={styles.canvasFitPreview} src={pendingPlacement.asset.url} alt="" width={358} height={150} unoptimized />}
            <h2>Fit “{pendingPlacement.asset.name}” to the project canvas?</h2>
            <p>Project canvas: {project.width} × {project.height}. The source is {pendingPlacement.asset.width} × {pendingPlacement.asset.height}. Choose whether to crop the edges or preserve the whole image with letterbox space.</p>
            <div className={styles.canvasFitActions}>
              <button onClick={() => confirmPendingPlacement("cover")}>Fill and crop</button>
              <button onClick={() => confirmPendingPlacement("contain")}>Fit and letterbox</button>
              <button onClick={() => setPendingPlacement(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
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
