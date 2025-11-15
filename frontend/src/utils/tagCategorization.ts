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
  return cache[normalizedTag] || "general";
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
 * Reorder tags based on category order
 */
function reorderTagsByCategory(
  categorizedTags: CategorizedTag[],
  categoryOrder: Array<{ id: string; enabled: boolean }>
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

  for (const { id, enabled } of categoryOrder) {
    if (enabled && tagsByCategory[id]) {
      orderedTags.push(...tagsByCategory[id]);
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
  categoryOrder: Array<{ id: string; enabled: boolean }>
): Promise<string> {
  if (!prompt.trim()) {
    return prompt;
  }

  const categorizedTags = await categorizeTags(prompt);
  return reorderTagsByCategory(categorizedTags, categoryOrder);
}
