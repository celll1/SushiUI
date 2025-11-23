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
  index?: Map<string, Array<{ tag: string; count: number }>>;
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
 * Build index for fast prefix lookup
 * Groups tags by their first 2 normalized characters
 */
function buildIndex(tags: TagData, categoryName: string): Map<string, Array<{ tag: string; count: number }>> {
  const index = new Map<string, Array<{ tag: string; count: number }>>();

  for (const [tag, count] of Object.entries(tags)) {
    const normalized = normalizeTag(tag);
    // Use first 2 characters as index key (or 1 if tag is very short)
    const prefix = normalized.substring(0, Math.min(2, normalized.length));

    if (!index.has(prefix)) {
      index.set(prefix, []);
    }

    index.get(prefix)!.push({ tag, count });
  }

  console.log(`[TagSuggestions] Built index for ${categoryName}: ${index.size} prefix groups`);
  return index;
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
    const response = await fetch(`/api/taglist/${category}`);
    if (response.ok) {
      const data: TagData = await response.json();
      categories[category].tags = data;
      categories[category].index = buildIndex(data, category);
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
 * Search for tags matching the input (optimized with indexing)
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
  const searchStartTime = performance.now();
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

  // Use index for fast lookup (first 2 characters)
  const searchPrefix = normalizedInput.substring(0, Math.min(2, normalizedInput.length));
  let totalScanned = 0;

  // Search in all categories using index
  for (const [categoryKey, category] of Object.entries(categories)) {
    if (!category.loaded || !category.index) {
      continue;
    }

    // Get all tags that start with the same 2-character prefix
    const candidates = category.index.get(searchPrefix) || [];
    totalScanned += candidates.length;

    for (const { tag, count } of candidates) {
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

  const searchTime = (performance.now() - searchStartTime).toFixed(2);
  console.log(`[TagSuggestions] Found ${results.length} matches in ${searchTime}ms (scanned ${totalScanned} tags, min count: ${minCount})`);

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

/**
 * Get all tags in the text as an array
 * @param text - Full text content
 * @returns Array of tag objects with start/end positions and content
 */
export function getAllTags(text: string): Array<{ start: number; end: number; tag: string }> {
  const tags: Array<{ start: number; end: number; tag: string }> = [];
  let pos = 0;

  while (pos < text.length) {
    // Skip delimiters and spaces
    while (pos < text.length && (text[pos] === ',' || text[pos] === '\n' || text[pos] === ' ')) {
      pos++;
    }

    if (pos >= text.length) break;

    // Found start of a tag
    const start = pos;

    // Find end of tag (next delimiter)
    while (pos < text.length && text[pos] !== ',' && text[pos] !== '\n') {
      pos++;
    }

    const end = pos;
    const segment = text.substring(start, end);
    const tag = segment.trim();

    if (tag.length > 0) {
      tags.push({ start, end, tag });
    }
  }

  return tags;
}

/**
 * Swap tag at cursor position with adjacent tag (left or right)
 * @param text - Full text content
 * @param cursorPos - Cursor position
 * @param direction - 'left' or 'right'
 * @returns Object with new text and cursor position, or null if cannot swap
 */
export function swapTagWithAdjacent(
  text: string,
  cursorPos: number,
  direction: 'left' | 'right'
): { text: string; cursorPos: number } | null {
  // Get all tags
  const tags = getAllTags(text);

  if (tags.length < 2) {
    return null; // Need at least 2 tags to swap
  }

  // Find the tag containing the cursor
  let currentTagIndex = -1;
  for (let i = 0; i < tags.length; i++) {
    const tag = tags[i];
    // Check if cursor is within this tag (including whitespace around it)
    if (cursorPos >= tag.start && cursorPos <= tag.end) {
      currentTagIndex = i;
      break;
    }
  }

  if (currentTagIndex === -1) {
    return null; // Cursor not in a tag
  }

  // Determine the tag to swap with
  const swapIndex = direction === 'left' ? currentTagIndex - 1 : currentTagIndex + 1;

  if (swapIndex < 0 || swapIndex >= tags.length) {
    return null; // No tag to swap with
  }

  const currentTag = tags[currentTagIndex];
  const swapTag = tags[swapIndex];

  // Determine the order of tags in the text
  const firstTag = currentTagIndex < swapIndex ? currentTag : swapTag;
  const secondTag = currentTagIndex < swapIndex ? swapTag : currentTag;

  // Extract parts of the text
  const before = text.substring(0, firstTag.start);
  const between = text.substring(firstTag.end, secondTag.start);
  const after = text.substring(secondTag.end);

  // Build new text with swapped tags
  const newText = before + secondTag.tag + between + firstTag.tag + after;

  // Calculate new cursor position (keep cursor in the same tag, which has moved)
  let newCursorPos: number;
  if (currentTagIndex < swapIndex) {
    // Current tag moved right
    newCursorPos = before.length + secondTag.tag.length + between.length + (cursorPos - currentTag.start);
  } else {
    // Current tag moved left
    newCursorPos = before.length + (cursorPos - currentTag.start);
  }

  return {
    text: newText,
    cursorPos: newCursorPos,
  };
}

/**
 * Jump to the next delimiter position (right after comma/newline)
 * @param text - Full text content
 * @param cursorPos - Current cursor position
 * @returns New cursor position
 */
export function jumpToNextDelimiter(text: string, cursorPos: number): number {
  // Search forward for next delimiter
  for (let i = cursorPos; i < text.length; i++) {
    if (text[i] === ',' || text[i] === '\n') {
      // Found delimiter, move to position after it
      let pos = i + 1;
      // Skip spaces
      while (pos < text.length && text[pos] === ' ') {
        pos++;
      }
      return pos;
    }
  }

  // No delimiter found, go to end
  return text.length;
}

/**
 * Jump to the previous delimiter position (right after comma/newline)
 * @param text - Full text content
 * @param cursorPos - Current cursor position
 * @returns New cursor position
 */
export function jumpToPreviousDelimiter(text: string, cursorPos: number): number {
  // First, check if we're already at the start of a tag (right after delimiter)
  // If so, jump to the previous tag start

  // Skip backwards past any spaces before cursor
  let searchPos = cursorPos - 1;
  while (searchPos > 0 && text[searchPos] === ' ') {
    searchPos--;
  }

  // If we're right after a delimiter, skip it to go to previous tag
  if (searchPos >= 0 && (text[searchPos] === ',' || text[searchPos] === '\n')) {
    searchPos--;
  }

  // Search backward for previous delimiter
  for (let i = searchPos; i >= 0; i--) {
    if (text[i] === ',' || text[i] === '\n') {
      // Found delimiter, move to position after it
      let pos = i + 1;
      // Skip spaces
      while (pos < text.length && text[pos] === ' ') {
        pos++;
      }
      return pos;
    }
  }

  // No delimiter found, go to start
  return 0;
}
