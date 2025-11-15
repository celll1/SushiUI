/**
 * Tag categorization utilities for reordering prompts by category
 */

import { TagData } from "./tagSuggestions";

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
  // Add more special tags as needed
};

/**
 * Load tag category data from backend
 */
async function loadCategoryData(category: string): Promise<TagData> {
  try {
    const response = await fetch(`http://localhost:8000/api/taglist/${category}`);
    if (response.ok) {
      return await response.json();
    }
  } catch (error) {
    console.error(`Failed to load ${category} tags:`, error);
  }
  return {};
}

/**
 * Determine the category of a tag
 */
async function determineTagCategory(tag: string): Promise<string> {
  const normalizedTag = tag.toLowerCase().replace(/[_\s]/g, "_");

  // Check special tags first
  if (SPECIAL_TAGS[normalizedTag]) {
    return SPECIAL_TAGS[normalizedTag];
  }

  // Check each category
  const categories = ["general", "character", "copyright", "artist", "meta", "model"];

  for (const category of categories) {
    const tagData = await loadCategoryData(category);

    // Check if tag exists in this category
    for (const categoryTag of Object.keys(tagData)) {
      if (categoryTag.toLowerCase() === normalizedTag) {
        return category;
      }
    }
  }

  // Default to general if not found
  return "general";
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
export async function categorizeTags(prompt: string): Promise<CategorizedTag[]> {
  const tags = parseTags(prompt);
  const categorized: CategorizedTag[] = [];

  for (const tag of tags) {
    const category = await determineTagCategory(tag);
    categorized.push({ tag, category });
  }

  return categorized;
}

/**
 * Reorder tags based on category order
 */
export function reorderTagsByCategory(
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
