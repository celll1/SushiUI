import type { MiniMaxH3References } from "@/utils/api";
import {
  deleteMediaInput,
  loadMediaInput,
  saveMediaInput,
} from "@/utils/mediaInputStorage";

type ReferenceKind = "images" | "videos" | "videoAudios" | "audios";

interface H3ReferenceManifest {
  version: 1;
  imageCount: number;
  videoCount: number;
  videoAudioCount: number;
  audioCount: number;
  referenceImageSize: "max" | "match";
}

const pendingWrites = new Map<string, Promise<void>>();

function manifestKey(storageKey: string): string {
  return `${storageKey}:manifest`;
}

function mediaKey(storageKey: string, kind: ReferenceKind, index: number): string {
  return `${storageKey}:${kind}:${index}`;
}

function readManifest(storageKey: string): H3ReferenceManifest | null {
  try {
    const raw = localStorage.getItem(manifestKey(storageKey));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<H3ReferenceManifest>;
    if (
      parsed.version !== 1 ||
      !Number.isInteger(parsed.imageCount) || parsed.imageCount < 0 ||
      !Number.isInteger(parsed.videoCount) || parsed.videoCount < 0 ||
      !Number.isInteger(parsed.videoAudioCount) || parsed.videoAudioCount < 0 ||
      !Number.isInteger(parsed.audioCount) || parsed.audioCount < 0 ||
      (parsed.referenceImageSize !== "max" && parsed.referenceImageSize !== "match")
    ) {
      return null;
    }
    return parsed as H3ReferenceManifest;
  } catch {
    return null;
  }
}

function enqueueWrite(storageKey: string, operation: () => Promise<void>): Promise<void> {
  const previous = pendingWrites.get(storageKey) ?? Promise.resolve();
  const current = previous.catch(() => undefined).then(operation);
  pendingWrites.set(storageKey, current);
  const clear = () => {
    if (pendingWrites.get(storageKey) === current) pendingWrites.delete(storageKey);
  };
  void current.then(clear, clear);
  return current;
}

async function saveSlot(storageKey: string, kind: ReferenceKind, index: number, file: File | null): Promise<void> {
  const key = mediaKey(storageKey, kind, index);
  if (file) {
    await saveMediaInput(key, file);
  } else {
    await deleteMediaInput(key);
  }
}

async function deleteSlots(storageKey: string, kind: ReferenceKind, from: number, to: number): Promise<void> {
  const removals: Promise<void>[] = [];
  for (let index = from; index < to; index += 1) {
    removals.push(deleteMediaInput(mediaKey(storageKey, kind, index)));
  }
  await Promise.all(removals);
}

export function persistH3References(
  storageKey: string,
  references: MiniMaxH3References,
  referenceImageSize: "max" | "match",
): Promise<void> {
  return enqueueWrite(storageKey, async () => {
    const previous = readManifest(storageKey);
    const manifest: H3ReferenceManifest = {
      version: 1,
      imageCount: references.images.length,
      videoCount: references.videos.length,
      videoAudioCount: references.videoAudios.length,
      audioCount: references.audios.length,
      referenceImageSize,
    };

    await Promise.all([
      ...references.images.map((file, index) => saveSlot(storageKey, "images", index, file)),
      ...references.videos.map((file, index) => saveSlot(storageKey, "videos", index, file)),
      ...references.videoAudios.map((file, index) => saveSlot(storageKey, "videoAudios", index, file)),
      ...references.audios.map((file, index) => saveSlot(storageKey, "audios", index, file)),
      previous ? deleteSlots(storageKey, "images", manifest.imageCount, previous.imageCount) : Promise.resolve(),
      previous ? deleteSlots(storageKey, "videos", manifest.videoCount, previous.videoCount) : Promise.resolve(),
      previous ? deleteSlots(storageKey, "videoAudios", manifest.videoAudioCount, previous.videoAudioCount) : Promise.resolve(),
      previous ? deleteSlots(storageKey, "audios", manifest.audioCount, previous.audioCount) : Promise.resolve(),
    ]);

    localStorage.setItem(manifestKey(storageKey), JSON.stringify(manifest));
  });
}

export async function restoreH3References(
  storageKey: string,
): Promise<{ references: MiniMaxH3References; referenceImageSize: "max" | "match" } | null> {
  await pendingWrites.get(storageKey);
  const manifest = readManifest(storageKey);
  if (!manifest) return null;

  const [images, videos, videoAudios, audios] = await Promise.all([
    Promise.all(Array.from({ length: manifest.imageCount }, (_, index) => loadMediaInput(mediaKey(storageKey, "images", index))),),
    Promise.all(Array.from({ length: manifest.videoCount }, (_, index) => loadMediaInput(mediaKey(storageKey, "videos", index))),),
    Promise.all(Array.from({ length: manifest.videoAudioCount }, (_, index) => loadMediaInput(mediaKey(storageKey, "videoAudios", index))),),
    Promise.all(Array.from({ length: manifest.audioCount }, (_, index) => loadMediaInput(mediaKey(storageKey, "audios", index))),),
  ]);

  const restoredVideos: File[] = [];
  const restoredVideoAudios: (File | null)[] = [];
  videos.forEach((video, index) => {
    if (!video) return;
    restoredVideos.push(video);
    restoredVideoAudios.push(videoAudios[index] ?? null);
  });

  return {
    references: {
      images: images.filter((file): file is File => file !== null),
      videos: restoredVideos,
      videoAudios: restoredVideoAudios,
      audios: audios.filter((file): file is File => file !== null),
    },
    referenceImageSize: manifest.referenceImageSize,
  };
}
