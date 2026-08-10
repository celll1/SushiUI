import { DBSchema, IDBPDatabase, openDB } from "idb";

interface StoredMediaInput {
  key: string;
  blob: Blob;
  name: string;
  type: string;
  lastModified: number;
}

interface MediaInputDB extends DBSchema {
  media: {
    key: string;
    value: StoredMediaInput;
  };
}

const DB_NAME = "sushiui_media_inputs";
const DB_VERSION = 1;
const STORE_NAME = "media";

export const INPAINT_VIDEO_INPUT_KEY = "inpaint_input_video_file";
export const OUTPAINT_VIDEO_INPUT_KEY = "outpaint_input_video_file";
export const INPAINT_VIDEO_PENDING_KEY = "inpaint_input_video_pending";
export const OUTPAINT_VIDEO_PENDING_KEY = "outpaint_input_video_pending";

let dbPromise: Promise<IDBPDatabase<MediaInputDB>> | null = null;
const pendingOperations = new Map<string, Promise<void>>();

function getDB(): Promise<IDBPDatabase<MediaInputDB>> {
  if (!dbPromise) {
    dbPromise = openDB<MediaInputDB>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: "key" });
        }
      },
    });
  }
  return dbPromise;
}

function enqueue(key: string, operation: () => Promise<void>): Promise<void> {
  const previous = pendingOperations.get(key) ?? Promise.resolve();
  const current = previous.catch(() => undefined).then(operation);
  pendingOperations.set(key, current);
  const clear = () => {
    if (pendingOperations.get(key) === current) pendingOperations.delete(key);
  };
  void current.then(clear, clear);
  return current;
}

export async function saveMediaInput(key: string, file: File): Promise<void> {
  await enqueue(key, async () => {
    const db = await getDB();
    await db.put(STORE_NAME, {
      key,
      blob: file,
      name: file.name,
      type: file.type,
      lastModified: file.lastModified,
    });
  });
}

export async function loadMediaInput(key: string): Promise<File | null> {
  await pendingOperations.get(key);
  const db = await getDB();
  const stored = await db.get(STORE_NAME, key);
  if (!stored) return null;
  return new File([stored.blob], stored.name, {
    type: stored.type || stored.blob.type,
    lastModified: stored.lastModified,
  });
}

export async function deleteMediaInput(key: string): Promise<void> {
  await enqueue(key, async () => {
    const db = await getDB();
    await db.delete(STORE_NAME, key);
  });
}
