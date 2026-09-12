export interface AspectRatioPreset {
  label: string;
  ratio: number;
}

export interface FixedResolutionPreset {
  width: number;
  height: number;
}

export const DEFAULT_ASPECT_RATIO_PRESETS: AspectRatioPreset[] = [
  { label: "1:1", ratio: 1 / 1 },
  { label: "4:3", ratio: 4 / 3 },
  { label: "3:4", ratio: 3 / 4 },
  { label: "16:9", ratio: 16 / 9 },
  { label: "9:16", ratio: 9 / 16 },
  { label: "21:9", ratio: 21 / 9 },
  { label: "9:21", ratio: 9 / 21 },
  { label: "3:2", ratio: 3 / 2 },
  { label: "2:3", ratio: 2 / 3 },
  { label: "5:4", ratio: 5 / 4 },
];

export const DEFAULT_FIXED_RESOLUTION_PRESETS: FixedResolutionPreset[] = [
  { width: 768, height: 1152 },
  { width: 1152, height: 768 },
  { width: 1248, height: 720 },
  { width: 720, height: 1248 },
  { width: 960, height: 1344 },
  { width: 1344, height: 960 },
  { width: 1024, height: 1152 },
  { width: 1152, height: 1024 },
  { width: 1024, height: 1024 },
  { width: 896, height: 1152 },
  { width: 1152, height: 896 },
  { width: 832, height: 1216 },
  { width: 1216, height: 832 },
  { width: 640, height: 1536 },
  { width: 1536, height: 640 },
  { width: 512, height: 512 },
];

export interface StoredGenerationDisplaySettings {
  resolutionStep?: number;
  developerMode: boolean;
  showAdvancedCFG: boolean;
  aspectRatioPresets?: AspectRatioPreset[];
  fixedResolutionPresets?: FixedResolutionPreset[];
}

function readJson<T>(key: string): T | undefined {
  const saved = localStorage.getItem(key);
  if (!saved) return undefined;
  try {
    return JSON.parse(saved) as T;
  } catch (error) {
    console.error(`Failed to parse ${key}:`, error);
    return undefined;
  }
}

export function readStoredGenerationDisplaySettings(): StoredGenerationDisplaySettings {
  const savedResolutionStep = localStorage.getItem("resolution_step");
  return {
    resolutionStep: savedResolutionStep ? parseInt(savedResolutionStep) : undefined,
    developerMode: localStorage.getItem("developer_mode") === "true",
    showAdvancedCFG: localStorage.getItem("show_advanced_cfg") === "true",
    aspectRatioPresets: readJson<AspectRatioPreset[]>("aspect_ratio_presets"),
    fixedResolutionPresets: readJson<FixedResolutionPreset[]>("fixed_resolution_presets"),
  };
}
