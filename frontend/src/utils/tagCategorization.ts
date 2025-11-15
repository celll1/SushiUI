/**
 * Tag categorization utilities for reordering prompts by category
 */

import { loadAllTags } from "./tagSuggestions";

interface CategorizedTag {
  tag: string;
  category: string;
}

// Special tags mapping
const SPECIAL_TAGS: Record<string, string> = {
  // Rating
  sensitive: "rating",
  explicit: "rating",
  questionable: "rating",
  general: "rating",
  // Quality
  best_quality: "quality",
  normal_quality: "quality",
  bad_quality: "quality",
  worst_quality: "quality",
  masterpiece: "quality",
  // Count (character count tags)
  "1girl": "count",
  "2girls": "count",
  "3girls": "count",
  "4girls": "count",
  "5girls": "count",
  "6+girls": "count",
  "1boy": "count",
  "2boys": "count",
  "3boys": "count",
  "4boys": "count",
  "5boys": "count",
  "6+boys": "count",
  "1other": "count",
  "2others": "count",
  "3others": "count",
  "4others": "count",
  "5others": "count",
  "6+others": "count",
  "solo": "count",
  "solo_focus": "count",
  "multiple_girls": "count",
  "multiple_boys": "count",
  "multiple_others": "count",
  "male_focus": "count",
  "female_focus": "count",
  "other_focus": "count",
};

// Cache for tag-to-category mapping
let tagCategoryCache: Record<string, string> | null = null;

/**
 * Build tag-to-category mapping from loaded tag lists
 */
async function buildTagCategoryCache(): Promise<Record<string, string>> {
  if (tagCategoryCache) {
    return tagCategoryCache;
  }

  // Ensure all tag lists are loaded
  await loadAllTags();

  tagCategoryCache = {};

  // Add special tags
  for (const [tag, category] of Object.entries(SPECIAL_TAGS)) {
    tagCategoryCache[tag] = category;
  }

  // Load tag lists from backend (already cached in tagSuggestions)
  const categories = ["general", "character", "copyright", "artist", "meta", "model"];

  for (const category of categories) {
    try {
      const response = await fetch(`http://localhost:8000/api/taglist/${category}`);
      if (response.ok) {
        const tagData: Record<string, number> = await response.json();
        for (const tag of Object.keys(tagData)) {
          const normalizedTag = tag.toLowerCase().replace(/[_\s]/g, "_");
          // Don't overwrite special tags
          if (!tagCategoryCache[normalizedTag]) {
            tagCategoryCache[normalizedTag] = category;
          }
        }
      }
    } catch (error) {
      console.error(`Failed to load ${category} tags:`, error);
    }
  }

  return tagCategoryCache;
}

/**
 * Determine the category of a tag (fast, uses cache)
 */
function determineTagCategory(tag: string, cache: Record<string, string>): string {
  const normalizedTag = tag.toLowerCase().replace(/[_\s]/g, "_");
  return cache[normalizedTag] || "unknown";
}

/**
 * Parse prompt into individual tags
 */
function parseTags(prompt: string): string[] {
  return prompt
    .split(/[,\n]/)
    .map(tag => tag.trim())
    .filter(tag => tag.length > 0);
}

/**
 * Categorize all tags in a prompt
 */
async function categorizeTags(prompt: string): Promise<CategorizedTag[]> {
  const cache = await buildTagCategoryCache();
  const tags = parseTags(prompt);
  const categorized: CategorizedTag[] = [];

  for (const tag of tags) {
    const category = determineTagCategory(tag, cache);
    categorized.push({ tag, category });
  }

  return categorized;
}

/**
 * Shuffle array (Fisher-Yates algorithm)
 */
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Sort count tags: numeric tags first (by number), then non-numeric tags
 * e.g., "1girl", "2girls", "solo" -> "1girl", "solo", "2girls"
 */
function sortCountTags(tags: string[]): string[] {
  return [...tags].sort((a, b) => {
    // Extract leading number from tag
    const numA = a.match(/^(\d+)/);
    const numB = b.match(/^(\d+)/);

    // Both have numbers: sort by number
    if (numA && numB) {
      return parseInt(numA[1]) - parseInt(numB[1]);
    }

    // Only a has number: a comes first
    if (numA && !numB) {
      return -1;
    }

    // Only b has number: b comes first
    if (!numA && numB) {
      return 1;
    }

    // Neither has number: keep original order
    return 0;
  });
}

/**
 * Reorder tags based on category order
 */
function reorderTagsByCategory(
  categorizedTags: CategorizedTag[],
  categoryOrder: Array<{ id: string; enabled: boolean; randomize?: boolean }>
): string {
  // Group tags by category
  const tagsByCategory: Record<string, string[]> = {};

  for (const { tag, category } of categorizedTags) {
    if (!tagsByCategory[category]) {
      tagsByCategory[category] = [];
    }
    tagsByCategory[category].push(tag);
  }

  // Reorder based on category order
  const orderedTags: string[] = [];

  for (const { id, enabled, randomize } of categoryOrder) {
    if (enabled && tagsByCategory[id]) {
      let tags = tagsByCategory[id];

      // Special handling for "count" category
      if (id === "count" && !randomize) {
        tags = sortCountTags(tags);
      } else if (randomize) {
        tags = shuffleArray(tags);
      }

      orderedTags.push(...tags);
    }
  }

  // Add any remaining tags from categories not in the order
  for (const [category, tags] of Object.entries(tagsByCategory)) {
    const isInOrder = categoryOrder.some(c => c.id === category);
    if (!isInOrder) {
      orderedTags.push(...tags);
    }
  }

  return orderedTags.join(", ");
}

/**
 * Reorder prompt by category order
 */
export async function reorderPromptByCategory(
  prompt: string,
  categoryOrder: Array<{ id: string; enabled: boolean; randomize?: boolean }>
): Promise<string> {
  if (!prompt.trim()) {
    return prompt;
  }

  const categorizedTags = await categorizeTags(prompt);
  return reorderTagsByCategory(categorizedTags, categoryOrder);
}
