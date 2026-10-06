import type { ArchCapabilities, VideoConstraints } from "./api";

export const videoChainGeometryKey = (base: ArchCapabilities | null | undefined, arch: string | null | undefined): string =>
  JSON.stringify(arch ? base?.video_constraints?.[arch] ?? null : null);

export const overlayLoadedVideoConstraints = (
  base: ArchCapabilities | null,
  model: { type?: string; loaded_video_constraints?: VideoConstraints } | null | undefined,
): ArchCapabilities | null => {
  const arch = model?.type;
  const geometry = model?.loaded_video_constraints;
  if (!base || !arch || !geometry) return base;
  return { ...base, video_constraints: { ...base.video_constraints, [arch]: geometry } };
};
