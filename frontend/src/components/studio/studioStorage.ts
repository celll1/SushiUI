import { openDB } from "idb";
import type { StudioAsset, StudioProject } from "./types";

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
      if (!imageAssetIds.has(clip.assetId) || clip.presentation) return clip;
      const wasHeld = Number(clip.duration) > (1 / fps) + 0.0001;
      return {
        ...clip,
        duration: wasHeld ? clip.duration : 1 / fps,
        sourceIn: 0,
        presentation: wasHeld ? "hold" as const : "frame" as const,
        sourceDuration: 0,
      };
    });
    const project: StudioProject = {
      ...parsed,
      schemaVersion: 2,
      fps,
      clips: migratedClips,
      revision: Number.isFinite(parsed.revision) ? parsed.revision! : 0,
      jobs: parsed.jobs || [],
    } as StudioProject;
    const assets = await Promise.all(
      project.assets.map(async (asset): Promise<StudioAsset> => {
        if (!asset.blobKey) return asset;
        const blob = await loadImportedMedia(asset.blobKey);
        if (!blob) return { ...asset, url: "", thumbnailUrl: undefined };
        const url = URL.createObjectURL(blob);
        return { ...asset, url, thumbnailUrl: asset.kind === "image" ? url : undefined };
      }),
    );
    return { ...project, assets };
  } catch (error) {
    console.error("[Studio] Failed to restore project", error);
    return null;
  }
};

export const saveStudioProject = (project: StudioProject) => {
  const serializable = {
    ...project,
    assets: project.assets.map((asset) =>
      asset.blobKey ? { ...asset, url: "", thumbnailUrl: undefined } : asset,
    ),
  };
  localStorage.setItem(PROJECT_KEY, JSON.stringify(serializable));
};
