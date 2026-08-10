export const INFERENCE_ATTENTION_TYPES = ["normal", "sage", "flash", "tq"] as const;
export type InferenceAttentionType = (typeof INFERENCE_ATTENTION_TYPES)[number];

export const ATTENTION_IMPLEMENTATIONS = ["conduit", "diffusers"] as const;
export type AttentionImplementation = (typeof ATTENTION_IMPLEMENTATIONS)[number];

export const isInferenceAttentionType = (value: unknown): value is InferenceAttentionType =>
  typeof value === "string" && INFERENCE_ATTENTION_TYPES.includes(value as InferenceAttentionType);

export const isAttentionImplementation = (value: unknown): value is AttentionImplementation =>
  typeof value === "string" && ATTENTION_IMPLEMENTATIONS.includes(value as AttentionImplementation);

const readStorage = (key: string): string | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

export const readGlobalAttentionType = (): InferenceAttentionType | null => {
  const value = readStorage("attention_type");
  return isInferenceAttentionType(value) ? value : null;
};

export const readGlobalAttentionImpl = (): AttentionImplementation | null => {
  const value = readStorage("attention_impl");
  return isAttentionImplementation(value) ? value : null;
};

export const resolveGlobalAttentionType = (fallback?: string | null): string =>
  readGlobalAttentionType() ?? fallback ?? "normal";

export const resolveGlobalAttentionImpl = (fallback?: string | null): string =>
  readGlobalAttentionImpl() ?? fallback ?? "conduit";
