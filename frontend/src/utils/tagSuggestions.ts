/**
 * Tag suggestion utility for prompt input
 * Loads tag lists from JSON files and provides search functionality
 */

export interface TagData {
  [tag: string]: number; // tag -> count
}

interface TagCategory {
  name: string;
  tags: TagData;
  loaded: boolean;
}

const categories: Record<string, TagCategory> = {
  general: { name: "General", tags: {}, loaded: false },
  character: { name: "Character", tags: {}, loaded: false },
  artist: { name: "Artist", tags: {}, loaded: false },
  copyright: { name: "Copyright", tags: {}, loaded: false },
  meta: { name: "Meta", tags: {}, loaded: false },
  model: { name: "Model", tags: {}, loaded: false },
};

/**
 * Normalize tag format for comparison
 * Converts both underscore and space formats to a common format
 */
function normalizeTag(tag: string): string {
  return tag.toLowerCase().replace(/[_\s]/g, "");
}

/**
 * Convert tag from storage format (aaaa_bbbb_(cccc)) to display format (aaaa bbbb \(cccc\))
 */
export function formatTagForDisplay(tag: string): string {
  // Replace underscores with spaces, except those in parentheses
  // Escape parentheses with backslash
  return tag
    .replace(/_/g, " ")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
}

/**
 * Load tags from a specific category
 */
async function loadCategory(category: keyof typeof categories): Promise<void> {
  if (categories[category].loaded) {
    return;
  }

  try {
    console.log(`[TagSuggestions] Loading ${category} tags from API`);
    const response = await fetch(`http://localhost:8000/api/taglist/${category}`);
    if (response.ok) {
      const data: TagData = await response.json();
      categories[category].tags = data;
      categories[category].loaded = true;
      console.log(`[TagSuggestions] Loaded ${Object.keys(data).length} tags for ${category}`);
    } else {
      console.error(`[TagSuggestions] Failed to load ${category} tags: HTTP ${response.status}`);
    }
  } catch (error) {
    console.error(`Failed to load ${category} tags:`, error);
  }
}

/**
 * Load all tag categories
 */
export async function loadAllTags(): Promise<void> {
  await Promise.all(
    Object.keys(categories).map((cat) => loadCategory(cat as keyof typeof categories))
  );
}

// Special tags that should always be available
const SPECIAL_TAGS = {
  rating: [
    { tag: "sensitive", category: "Rating Tag" },
    { tag: "explicit", category: "Rating Tag" },
    { tag: "questionable", category: "Rating Tag" },
    { tag: "general", category: "Rating Tag" },
  ],
  quality: [
    { tag: "best_quality", category: "Quality Tag" },
    { tag: "normal_quality", category: "Quality Tag" },
    { tag: "bad_quality", category: "Quality Tag" },
    { tag: "worst_quality", category: "Quality Tag" },
    { tag: "masterpiece", category: "Quality Tag" },
  ],
};

/**
 * Search for tags matching the input
 * @param input - User input (can be in any format: aaaa_bbbb_(c, aaaa bbbb \(c, etc.)
 * @param limit - Maximum number of results
 * @returns Array of matching tags with their counts, sorted by count (descending)
 */
export async function searchTags(
  input: string,
  limit: number = 20
): Promise<Array<{ tag: string; count: number; category: string }>> {
  if (!input.trim()) {
    return [];
  }

  // Get minimum count from localStorage (default: 50)
  const minCount = typeof window !== 'undefined'
    ? parseInt(localStorage.getItem('tag_suggestion_min_count') || '50')
    : 50;

  // Ensure tags are loaded
  await loadAllTags();

  const normalizedInput = normalizeTag(input);
  console.log(`[TagSuggestions] Searching with normalized input: "${normalizedInput}", min count: ${minCount}`);
  const results: Array<{ tag: string; count: number; category: string }> = [];

  // Search special tags first
  const allSpecialTags = [...SPECIAL_TAGS.rating, ...SPECIAL_TAGS.quality];
  for (const specialTag of allSpecialTags) {
    const normalizedTag = normalizeTag(specialTag.tag);
    if (normalizedTag.startsWith(normalizedInput)) {
      results.push({
        tag: specialTag.tag,
        count: -1, // Special marker for special tags
        category: specialTag.category,
      });
    }
  }

  // Search in all categories
  for (const [categoryKey, category] of Object.entries(categories)) {
    if (!category.loaded) {
      console.log(`[TagSuggestions] Category ${categoryKey} not loaded, skipping`);
      continue;
    }

    for (const [tag, count] of Object.entries(category.tags)) {
      // Skip tags below minimum count (except count 0 which are deprecated)
      if (count < minCount && count !== 0) {
        continue;
      }

      // Skip deprecated tags (count = 0)
      if (count === 0) {
        continue;
      }

      const normalizedTag = normalizeTag(tag);

      // Check if the normalized tag starts with the normalized input
      if (normalizedTag.startsWith(normalizedInput)) {
        results.push({
          tag,
          count,
          category: category.name,
        });
      }
    }
  }

  console.log(`[TagSuggestions] Found ${results.length} matches (min count: ${minCount})`);

  // Sort: special tags first, then by count (descending)
  const sorted = results.sort((a, b) => {
    // Special tags (count = -1) come first
    if (a.count === -1 && b.count !== -1) return -1;
    if (a.count !== -1 && b.count === -1) return 1;
    if (a.count === -1 && b.count === -1) return 0;

    // Otherwise sort by count
    return b.count - a.count;
  });

  return sorted.slice(0, limit);
}

/**
 * Get the current tag being typed at cursor position
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @returns The tag being typed, or empty string
 */
export function getCurrentTag(text: string, cursorPos: number): string {
  // Find the start of the current tag (after the last comma or start of string)
  let start = text.lastIndexOf(",", cursorPos - 1) + 1;

  // Find the end of the current tag (before the next comma or end of string)
  let end = text.indexOf(",", cursorPos);
  if (end === -1) {
    end = text.length;
  }

  // Extract and trim the tag
  const currentTag = text.substring(start, end).trim();

  // Only return if cursor is actually within/at the end of this tag
  const tagStart = start + text.substring(start, end).indexOf(currentTag);
  const tagEnd = tagStart + currentTag.length;

  if (cursorPos >= tagStart && cursorPos <= tagEnd) {
    return currentTag;
  }

  return "";
}

/**
 * Replace the current tag at cursor position with a new tag
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @param newTag - New tag to insert (will be formatted for display)
 * @returns Object with new text and new cursor position
 */
export function replaceCurrentTag(
  text: string,
  cursorPos: number,
  newTag: string
): { text: string; cursorPos: number } {
  // Find the start of the current tag
  let start = text.lastIndexOf(",", cursorPos - 1) + 1;

  // Find the end of the current tag
  let end = text.indexOf(",", cursorPos);
  if (end === -1) {
    end = text.length;
  }

  // Get the current tag position
  const beforeTag = text.substring(0, start);
  const afterTag = text.substring(end);

  // Format the tag for display
  const formattedTag = formatTagForDisplay(newTag);

  // Build new text with proper spacing
  const trimmedBefore = beforeTag.trimEnd();
  const needsSpaceBefore = trimmedBefore.length > 0 && !trimmedBefore.endsWith(",");
  const prefix = needsSpaceBefore ? trimmedBefore + ", " : trimmedBefore + (trimmedBefore.length > 0 ? " " : "");

  const newText = prefix + formattedTag + afterTag;
  const newCursorPos = prefix.length + formattedTag.length;

  return {
    text: newText,
    cursorPos: newCursorPos,
  };
}

/**
 * Delete the tag at cursor position (Ctrl+Backspace functionality)
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @returns Object with new text and new cursor position
 */
export function deleteTagAtCursor(
  text: string,
  cursorPos: number
): { text: string; cursorPos: number } {
  // Find the start of the current tag
  let start = text.lastIndexOf(",", cursorPos - 1);
  if (start === -1) {
    start = 0;
  } else {
    start += 1; // Move past the comma
  }

  // Find the end of the current tag (including the trailing comma if present)
  let end = text.indexOf(",", cursorPos);
  if (end === -1) {
    end = text.length;
  } else {
    end += 1; // Include the comma
  }

  // Extract before and after
  const before = text.substring(0, start).trimEnd();
  const after = text.substring(end).trimStart();

  // Build new text
  const newText = before + (before.length > 0 && after.length > 0 ? ", " : "") + after;
  const newCursorPos = before.length + (before.length > 0 && after.length > 0 ? 2 : 0);

  return {
    text: newText,
    cursorPos: newCursorPos,
  };
}
