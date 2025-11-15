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
 * Handles tags separated by commas, newlines, or spaces between tags
 * Special handling: if user is inserting a new tag between existing tags (e.g., "whi red eyes"),
 * only the part before the space is considered as the current tag
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @returns The tag being typed, or empty string
 */
export function getCurrentTag(text: string, cursorPos: number): string {
  // If text is empty, return empty string
  if (!text || text.trim().length === 0) {
    return "";
  }

  // Find the start: search backwards for comma, newline, or start of string
  let start = cursorPos - 1;
  while (start >= 0) {
    const char = text[start];
    if (char === ',' || char === '\n') {
      start++; // Move past the delimiter
      break;
    }
    start--;
  }
  if (start < 0) {
    start = 0;
  }

  // Find the end: search forwards from cursor
  // Stop at comma, newline, OR if we encounter a space followed by a non-space
  // (indicating the start of another tag)
  let end = cursorPos;
  let foundSpace = false;
  while (end < text.length) {
    const char = text[end];

    // Always stop at comma or newline
    if (char === ',' || char === '\n') {
      break;
    }

    // Track spaces
    if (char === ' ') {
      foundSpace = true;
    } else if (foundSpace) {
      // Found non-space after space - this is likely start of next tag
      // Go back to the space
      while (end > cursorPos && text[end - 1] === ' ') {
        end--;
      }
      break;
    }

    end++;
  }

  // Extract the segment containing the cursor
  const segment = text.substring(start, end);

  // Trim to get the actual tag
  const currentTag = segment.trim();

  // If the segment is empty or only whitespace, return empty string
  if (currentTag.length === 0) {
    return "";
  }

  // Calculate where the trimmed tag actually starts and ends
  const trimmedStart = start + segment.indexOf(currentTag);
  const trimmedEnd = trimmedStart + currentTag.length;

  // Only return if cursor is actually within the tag (not in whitespace before/after)
  if (cursorPos >= trimmedStart && cursorPos <= trimmedEnd) {
    return currentTag;
  }

  return "";
}

/**
 * Replace the current tag at cursor position with a new tag
 * Handles tags separated by commas, newlines, or spaces between tags
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
  // Find the start: search backwards for comma, newline, or start of string
  let start = cursorPos - 1;
  while (start >= 0) {
    const char = text[start];
    if (char === ',' || char === '\n') {
      start++; // Move past the delimiter
      break;
    }
    start--;
  }
  if (start < 0) {
    start = 0;
  }

  // Find the end: search forwards from cursor
  // Stop at comma, newline, OR if we encounter a space followed by a non-space
  let end = cursorPos;
  let foundSpace = false;
  while (end < text.length) {
    const char = text[end];

    // Always stop at comma or newline
    if (char === ',' || char === '\n') {
      break;
    }

    // Track spaces
    if (char === ' ') {
      foundSpace = true;
    } else if (foundSpace) {
      // Found non-space after space - this is likely start of next tag
      // Go back to the space
      while (end > cursorPos && text[end - 1] === ' ') {
        end--;
      }
      break;
    }

    end++;
  }

  // Get the current tag position
  const beforeTag = text.substring(0, start);
  const afterTag = text.substring(end);

  // Format the tag for display
  const formattedTag = formatTagForDisplay(newTag);

  // Build new text with proper spacing
  const trimmedBefore = beforeTag.trimEnd();
  const trimmedAfter = afterTag.trimStart();

  const needsSpaceBefore = trimmedBefore.length > 0 && !trimmedBefore.endsWith(",") && !trimmedBefore.endsWith("\n");
  const prefix = needsSpaceBefore ? trimmedBefore + ", " : trimmedBefore + (trimmedBefore.length > 0 && trimmedBefore.endsWith(",") ? " " : "");

  // Add comma after tag if there's content after (not at end of string)
  const needsCommaAfter = trimmedAfter.length > 0 && !trimmedAfter.startsWith(",") && !trimmedAfter.startsWith("\n");
  const suffix = needsCommaAfter ? ", " + trimmedAfter : trimmedAfter;

  const newText = prefix + formattedTag + suffix;
  const newCursorPos = prefix.length + formattedTag.length + (needsCommaAfter ? 2 : 0);

  return {
    text: newText,
    cursorPos: newCursorPos,
  };
}

/**
 * Delete the tag at cursor position (Ctrl+Backspace functionality)
 * Handles tags separated by commas, newlines, or spaces between tags
 * Removes the tag and one adjacent delimiter to avoid double commas
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @returns Object with new text and new cursor position
 */
export function deleteTagAtCursor(
  text: string,
  cursorPos: number
): { text: string; cursorPos: number } {
  // Delete ALL content between delimiters (comma or newline)
  // This treats the entire segment as one tag, regardless of spaces
  // Examples:
  // - "novel illustration" → delete entire tag
  // - "whi red eyes" → delete entire segment (even if incomplete)

  // Find the start: search backwards from cursor for comma, newline, or start of string
  let tagSegmentStart = cursorPos - 1;
  while (tagSegmentStart >= 0) {
    const char = text[tagSegmentStart];
    if (char === ',' || char === '\n') {
      tagSegmentStart++; // Move past the delimiter
      break;
    }
    tagSegmentStart--;
  }
  if (tagSegmentStart < 0) {
    tagSegmentStart = 0;
  }

  // Find the end: search forwards from tag start to next delimiter
  // Delete EVERYTHING between delimiters
  let tagSegmentEnd = tagSegmentStart;
  while (tagSegmentEnd < text.length) {
    const char = text[tagSegmentEnd];

    // Only stop at comma or newline
    if (char === ',' || char === '\n') {
      break;
    }

    tagSegmentEnd++;
  }

  // Now determine what to delete including delimiters
  // Look for delimiter before tag segment
  let deleteStart = tagSegmentStart;
  let hasLeadingDelimiter = false;
  if (tagSegmentStart > 0) {
    // Check if there's a comma or newline before this tag
    let checkPos = tagSegmentStart - 1;
    while (checkPos >= 0 && text[checkPos] === ' ') {
      checkPos--;
    }
    if (checkPos >= 0 && (text[checkPos] === ',' || text[checkPos] === '\n')) {
      hasLeadingDelimiter = true;
      deleteStart = checkPos;
    }
  }

  // Look for delimiter after tag segment
  let deleteEnd = tagSegmentEnd;
  let hasTrailingDelimiter = false;
  if (tagSegmentEnd < text.length) {
    let checkPos = tagSegmentEnd;
    while (checkPos < text.length && text[checkPos] === ' ') {
      checkPos++;
    }
    if (checkPos < text.length && (text[checkPos] === ',' || text[checkPos] === '\n')) {
      hasTrailingDelimiter = true;
      deleteEnd = checkPos + 1; // Include the delimiter
    }
  }

  // Deletion strategy:
  // 1. If has trailing delimiter: delete from tag start to after trailing delimiter
  // 2. If has leading delimiter but no trailing: delete from leading delimiter to tag end
  // 3. If no delimiters: delete just the tag
  if (hasTrailingDelimiter) {
    // Delete tag + trailing delimiter + any whitespace after
    deleteStart = tagSegmentStart;
    while (deleteEnd < text.length && text[deleteEnd] === ' ') {
      deleteEnd++;
    }
  } else if (hasLeadingDelimiter) {
    // Delete leading delimiter + tag
    // deleteStart already set above
    deleteEnd = tagSegmentEnd;
  } else {
    // No delimiters, just delete the tag segment
    deleteStart = tagSegmentStart;
    deleteEnd = tagSegmentEnd;
  }

  // Build result
  const before = text.substring(0, deleteStart);
  const after = text.substring(deleteEnd);

  // Clean up: ensure proper spacing after comma if needed
  let finalText = before + after;
  let finalCursorPos = deleteStart;

  // If we deleted a tag and there's a comma at cursor position now,
  // ensure there's a space after it
  if (finalCursorPos > 0 && finalCursorPos < finalText.length) {
    if (finalText[finalCursorPos - 1] === ',' && finalText[finalCursorPos] !== ' ' && finalText[finalCursorPos] !== '\n') {
      finalText = finalText.substring(0, finalCursorPos) + ' ' + finalText.substring(finalCursorPos);
      finalCursorPos++;
    }
  }

  return {
    text: finalText,
    cursorPos: finalCursorPos,
  };
}
