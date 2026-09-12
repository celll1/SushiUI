export const TAG_CATEGORY_ORDER = [
  "General",
  "Character",
  "Copyright",
  "Artist",
  "Meta",
  "Rating",
  "Quality",
  "Model",
  "Unknown",
] as const;

export const TAG_EDITOR_CATEGORY_ORDER = [
  "Copyright",
  "Character",
  "Artist",
  "General",
  "Meta",
  "Quality",
  "Rating",
  "Unknown",
] as const;

export const TAG_RESULT_CATEGORY_ORDER = [
  "Quality",
  "Rating",
  "Character",
  "Copyright",
  "General",
  "Artist",
  "Meta",
  "Unknown",
] as const;

export const TAG_ANALYSIS_CATEGORY_ORDER = [
  "General",
  "Character",
  "Copyright",
  "Meta",
  "Quality",
  "Rating",
  "Artist",
  "Unknown",
] as const;

export type TagEditorCategory = typeof TAG_EDITOR_CATEGORY_ORDER[number];

export const TAG_CATEGORY_HEX: Record<string, string> = {
  Quality: "#facc15",
  Rating: "#fb923c",
  Character: "#60a5fa",
  Copyright: "#c084fc",
  General: "#4ade80",
  Artist: "#f472b6",
  Meta: "#9ca3af",
  Model: "#22d3ee",
  Unknown: "#6b7280",
};

export const TAG_CATEGORY_CHART_HEX: Record<string, string> = {
  Quality: "#eab308",
  Rating: "#f97316",
  Character: "#3b82f6",
  Copyright: "#a855f7",
  General: "#16a34a",
  Artist: "#ec4899",
  Meta: "#9ca3af",
  Model: "#0891b2",
  Unknown: "#4b5563",
};

export const TAG_CATEGORY_TEXT_CLASS: Record<string, string> = {
  Quality: "text-yellow-400",
  Rating: "text-orange-400",
  Character: "text-blue-400",
  Copyright: "text-purple-400",
  General: "text-green-400",
  Artist: "text-pink-400",
  Meta: "text-gray-400",
  Model: "text-cyan-400",
  Unknown: "text-gray-500",
};

export const TAG_CATEGORY_BAR_CLASS: Record<string, string> = {
  Quality: "bg-yellow-500",
  Rating: "bg-orange-500",
  Character: "bg-blue-500",
  Copyright: "bg-purple-500",
  General: "bg-green-600",
  Artist: "bg-pink-500",
  Meta: "bg-gray-400",
  Model: "bg-cyan-600",
  Unknown: "bg-gray-600",
};
