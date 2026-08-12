import type { StudioProject } from "./types";

export const STUDIO_PROJECT_EXTENSION = ".sushistudio";
export const STUDIO_PROJECT_FORMAT = "sushiui-studio";
export const STUDIO_PROJECT_VERSION = 5;
const RECENT_PROJECTS_KEY = "sushiui_studio_recent_v1";
const MAX_RECENT_PROJECTS = 12;

export interface StudioProjectFile {
  format: typeof STUDIO_PROJECT_FORMAT;
  version: number;
  exportedAt: string;
  project: StudioProject;
}

export interface StudioRecentProject {
  id: string;
  name: string;
  updatedAt: string;
  width: number;
  height: number;
  duration: number;
  assetCount: number;
  manifest: StudioProject;
}

export const projectFileName = (name: string): string => {
  const base = name.trim().replace(/[^a-z0-9-_]+/gi, "_").replace(/^_+|_+$/g, "") || "studio-project";
  return `${base}${STUDIO_PROJECT_EXTENSION}`;
};

export const serializeStudioProject = (project: StudioProject): string => JSON.stringify({
  format: STUDIO_PROJECT_FORMAT,
  version: STUDIO_PROJECT_VERSION,
  exportedAt: new Date().toISOString(),
  project: {
    ...project,
    schemaVersion: STUDIO_PROJECT_VERSION,
    assets: project.assets.map((asset) => asset.blobKey
      ? { ...asset, url: "", thumbnailUrl: undefined }
      : asset),
  },
}, null, 2);

export const parseStudioProjectFile = (raw: unknown): Partial<StudioProject> => {
  if (!raw || typeof raw !== "object") throw new Error("The selected file is not a Studio project.");
  const value = raw as Partial<StudioProjectFile> & Partial<StudioProject>;
  if (value.format === STUDIO_PROJECT_FORMAT && value.project && typeof value.project === "object") {
    return value.project;
  }
  if (!value.id || !Array.isArray(value.assets) || !Array.isArray(value.tracks) || !Array.isArray(value.clips)) {
    throw new Error("The selected file is not a Studio project manifest.");
  }
  return value;
};

export const readRecentProjects = (): StudioRecentProject[] => {
  if (typeof window === "undefined") return [];
  try {
    const value = JSON.parse(localStorage.getItem(RECENT_PROJECTS_KEY) || "[]");
    return Array.isArray(value) ? value.filter((item): item is StudioRecentProject => !!item && typeof item.id === "string" && !!item.manifest) : [];
  } catch {
    return [];
  }
};

export const rememberRecentProject = (project: StudioProject): StudioRecentProject[] => {
  const entry: StudioRecentProject = {
    id: project.id,
    name: project.name,
    updatedAt: project.updatedAt,
    width: project.width,
    height: project.height,
    duration: project.duration,
    assetCount: project.assets.length,
    manifest: JSON.parse(JSON.stringify({
      ...project,
      schemaVersion: STUDIO_PROJECT_VERSION,
      assets: project.assets.map((asset) => asset.blobKey
        ? { ...asset, url: "", thumbnailUrl: undefined }
        : asset),
    })) as StudioProject,
  };
  const next = [entry, ...readRecentProjects().filter((item) => item.id !== project.id)].slice(0, MAX_RECENT_PROJECTS);
  try { localStorage.setItem(RECENT_PROJECTS_KEY, JSON.stringify(next)); } catch { /* recent recovery is best effort */ }
  return next;
};

export const removeRecentProject = (id: string): StudioRecentProject[] => {
  const next = readRecentProjects().filter((item) => item.id !== id);
  try { localStorage.setItem(RECENT_PROJECTS_KEY, JSON.stringify(next)); } catch { /* best effort */ }
  return next;
};
