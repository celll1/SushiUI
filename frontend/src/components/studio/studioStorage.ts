import { openDB } from "idb";
import { resolveStudioCanvasMode, type StudioAsset, type StudioProject } from "./types";

const PROJECT_KEY = "sushiui_studio_project_v1";
const DATABASE_NAME = "sushiui-studio";
const MEDIA_STORE = "media";

const openStudioDatabase = () =>
  openDB(DATABASE_NAME, 1, {
    upgrade(database) {
      if (!database.objectStoreNames.contains(MEDIA_STORE)) {
        database.createObjectStore(MEDIA_STORE);
      }
    },
  });

export const saveImportedMedia = async (key: string, file: Blob) => {
  const database = await openStudioDatabase();
  await database.put(MEDIA_STORE, file, key);
};

export const loadImportedMedia = async (key: string): Promise<Blob | undefined> => {
  const database = await openStudioDatabase();
  return database.get(MEDIA_STORE, key);
};

export const loadStudioProject = async (): Promise<StudioProject | null> => {
  const raw = localStorage.getItem(PROJECT_KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<StudioProject>;
    if (!parsed.id || !parsed.tracks || !parsed.clips || !parsed.assets) return null;
    const fps = Number.isFinite(parsed.fps) && parsed.fps! > 0 ? parsed.fps! : 24;
    const imageAssetIds = new Set((parsed.assets || []).filter((asset) => asset.kind === "image").map((asset) => asset.id));
    const migratedClips = (parsed.clips || []).map((clip) => {
      const inputRoles = clip.inputRoles?.filter((role) => role === "keyframe");
      if (!imageAssetIds.has(clip.assetId) || clip.presentation) return { ...clip, inputRoles };
      const wasHeld = Number(clip.duration) > (1 / fps) + 0.0001;
      return {
        ...clip,
        inputRoles,
        duration: wasHeld ? clip.duration : 1 / fps,
        sourceIn: 0,
        presentation: wasHeld ? "hold" as const : "frame" as const,
        sourceDuration: 0,
      };
    });
    const project: StudioProject = {
      ...parsed,
      schemaVersion: 4,
      width: Number.isFinite(parsed.width) && parsed.width! > 0
        ? Math.max(64, Math.min(8192, Math.round(parsed.width! / 16) * 16))
        : 1920,
      height: Number.isFinite(parsed.height) && parsed.height! > 0
        ? Math.max(64, Math.min(8192, Math.round(parsed.height! / 16) * 16))
        : 1080,
      canvasMode: resolveStudioCanvasMode(parsed.canvasMode, migratedClips, parsed.assets || []),
      fps,
      clips: migratedClips,
      revision: Number.isFinite(parsed.revision) ? parsed.revision! : 0,
      jobs: parsed.jobs || [],
    } as StudioProject;
    const assets = await Promise.all(
      project.assets.map(async (asset): Promise<StudioAsset> => {
        if (!asset.blobKey) return asset;
        const blob = await loadImportedMedia(asset.blobKey);
        if (!blob) return { ...asset, url: "", thumbnailUrl: undefined, missing: true };
        const url = URL.createObjectURL(blob);
        return { ...asset, url, thumbnailUrl: asset.kind === "image" ? url : undefined, missing: false };
      }),
    );
    return { ...project, assets };
  } catch (error) {
    console.error("[Studio] Failed to restore project", error);
    return null;
  }
};

// localStorage has a small quota (commonly ~5MB) shared by the whole origin.
// Assets backed by a `blobKey` live in IndexedDB (see `saveImportedMedia`
// above) and are excluded from the serialized payload here, but any asset
// that still carries an inline data URL (e.g. a caller that has not yet
// been migrated to the blobKey pattern) is written verbatim and can exceed
// the quota. `localStorage.setItem` throws synchronously in that case, so
// callers must check the returned result rather than assume this always
// succeeds.
export const saveStudioProject = (project: StudioProject): { ok: true } | { ok: false; error: unknown } => {
  const serializable = {
    ...project,
    assets: project.assets.map((asset) =>
      asset.blobKey ? { ...asset, url: "", thumbnailUrl: undefined } : asset,
    ),
  };
  try {
    localStorage.setItem(PROJECT_KEY, JSON.stringify(serializable));
    return { ok: true };
  } catch (error) {
    console.error("[Studio] Failed to persist project", error);
    return { ok: false, error };
  }
};
