export type StudioAssetKind = "image" | "video" | "audio";

export type StudioInputRole = "keyframe";
export type StudioClipPresentation = "frame" | "hold" | "clip";

export interface StudioAsset {
  id: string;
  galleryId?: number;
  name: string;
  kind: StudioAssetKind;
  url: string;
  masterUrl?: string;
  thumbnailUrl?: string;
  maskUrl?: string;
  duration: number;
  width?: number;
  height?: number;
  source: "gallery" | "import" | "generation";
  blobKey?: string;
  prompt?: string;
  negativePrompt?: string;
  createdAt?: string;
  generationType?: string;
  modelName?: string;
  seed?: number;
  parameters?: Record<string, unknown>;
}

export type StudioTrackKind = "video" | "audio";

export interface StudioTrack {
  id: string;
  name: string;
  kind: StudioTrackKind;
  muted: boolean;
  locked: boolean;
  visible: boolean;
}

export interface StudioClip {
  id: string;
  assetId: string;
  trackId: string;
  name: string;
  start: number;
  duration: number;
  sourceIn: number;
  /** A still is one frame by default; only an explicit hold may exceed it. */
  presentation?: StudioClipPresentation;
  /** Source media duration in seconds at insertion time, for safe trim limits. */
  sourceDuration?: number;
  /** Timeline inputs are opt-in; references never come from an implicit clip. */
  inputRoles?: StudioInputRole[];
  linkGroupId?: string;
  takeGroupId?: string;
  activeTake?: boolean;
  generated?: boolean;
}

export interface StudioRange {
  start: number;
  end: number;
}

export interface StudioProject {
  schemaVersion: number;
  revision: number;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  assets: StudioAsset[];
  tracks: StudioTrack[];
  clips: StudioClip[];
  jobs: StudioJob[];
  /** Server-side render job to resume polling after Studio is reopened. */
  renderJobId?: string;
  outputRange?: StudioRange | null;
  inpaintRange?: StudioRange | null;
  referenceAssetIds?: string[];
}

export type StudioTool = "select" | "blade" | "hand" | "range" | "link";
export type StudioGenerationMode = "t2v" | "i2v" | "inpaint" | "outpaint" | "ref2v" | "t2i" | "i2i" | "image-inpaint";
export type StudioMode = StudioGenerationMode;
export type StudioPane = "generate" | "inspector" | "jobs";

export interface StudioJob {
  id: string;
  mode: StudioMode;
  prompt: string;
  status: "running" | "review" | "failed" | "applied";
  startedAt: number;
  error?: string;
  assetId?: string;
  recipe: Record<string, unknown>;
}

export const createStudioProject = (): StudioProject => ({
  schemaVersion: 2,
  revision: 0,
  id: crypto.randomUUID(),
  name: "Untitled Studio Project",
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  duration: 60,
  fps: 24,
  width: 1920,
  height: 1080,
  assets: [],
  tracks: [
    { id: "video-1", name: "VIDEO 1", kind: "video", muted: false, locked: false, visible: true },
    { id: "video-2", name: "VIDEO 2", kind: "video", muted: false, locked: false, visible: true },
    { id: "audio-1", name: "AUDIO 1", kind: "audio", muted: false, locked: false, visible: true },
    { id: "audio-2", name: "AUDIO 2", kind: "audio", muted: false, locked: false, visible: true },
  ],
  clips: [],
  jobs: [],
});
