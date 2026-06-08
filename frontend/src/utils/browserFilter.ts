import { BrowserImageEntry } from "@/utils/api";

export interface FilterQuery {
  includeTags: string[];  // AND — all must match
  excludeTags: string[];  // OR — any match causes exclusion
  tagCountMin: number | null;
  tagCountMax: number | null;
  missingCopyright: boolean;
  missingCharacter: boolean;
}

export const EMPTY_FILTER: FilterQuery = {
  includeTags: [],
  excludeTags: [],
  tagCountMin: null,
  tagCountMax: null,
  missingCopyright: false,
  missingCharacter: false,
};

export function isFilterActive(q: FilterQuery): boolean {
  return (
    q.includeTags.length > 0 ||
    q.excludeTags.length > 0 ||
    q.tagCountMin !== null ||
    q.tagCountMax !== null ||
    q.missingCopyright ||
    q.missingCharacter
  );
}

export function needsTagsLoaded(q: FilterQuery): boolean {
  return (
    q.includeTags.length > 0 ||
    q.excludeTags.length > 0 ||
    q.tagCountMin !== null ||
    q.tagCountMax !== null ||
    q.missingCopyright ||
    q.missingCharacter
  );
}

// Convert a glob-like pattern to a RegExp.
// Supports: *xxx, xxx*, *xxx*, exact
// <cat> patterns are handled separately via categoryMap.
function patternToMatcher(pattern: string): (tag: string) => boolean {
  const p = pattern.trim().toLowerCase();
  if (p.startsWith("<") && p.endsWith(">")) {
    // Category pattern — resolved externally; return placeholder always-false
    return () => false;
  }
  if (!p.includes("*")) {
    return (tag) => tag.toLowerCase() === p;
  }
  // Convert glob to regex: escape special chars then replace * with .*
  const escaped = p.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  const re = new RegExp(`^${escaped}$`);
  return (tag) => re.test(tag.toLowerCase());
}

/** Pre-compile all matchers in the FilterQuery. categoryMap maps tag→category. */
export function compileFilter(
  q: FilterQuery,
  categoryMap: Map<string, string>
): (entry: BrowserImageEntry) => boolean {
  if (!isFilterActive(q)) return () => true;

  // Resolve <cat> patterns using categoryMap
  const resolvePatternToCategoryName = (pattern: string): string | null => {
    const p = pattern.trim().toLowerCase();
    if (p.startsWith("<") && p.endsWith(">")) {
      return p.slice(1, -1);
    }
    return null;
  };

  const includeMatchers: Array<(tags: string[], catSet: Set<string>) => boolean> = q.includeTags.map((pat) => {
    const catName = resolvePatternToCategoryName(pat);
    if (catName) return (_tags, catSet) => catSet.has(catName);
    const m = patternToMatcher(pat);
    return (tags) => tags.some(m);
  });

  const excludeMatchers: Array<(tags: string[], catSet: Set<string>) => boolean> = q.excludeTags.map((pat) => {
    const catName = resolvePatternToCategoryName(pat);
    if (catName) return (_tags, catSet) => catSet.has(catName);
    const m = patternToMatcher(pat);
    return (tags) => tags.some(m);
  });

  return (entry: BrowserImageEntry) => {
    const tags = entry.tags ?? [];

    // Tag count filter
    if (q.tagCountMin !== null && tags.length < q.tagCountMin) return false;
    if (q.tagCountMax !== null && tags.length > q.tagCountMax) return false;

    // Build category set for this entry
    const catSet = new Set<string>();
    for (const t of tags) {
      const cat = categoryMap.get(t);
      if (cat) catSet.add(cat.toLowerCase());
    }

    // missingCopyright / missingCharacter
    if (q.missingCopyright && catSet.has("copyright")) return false;
    if (q.missingCharacter && catSet.has("character")) return false;

    // Include: all must match
    for (const m of includeMatchers) {
      if (!m(tags, catSet)) return false;
    }

    // Exclude: any match → exclude
    for (const m of excludeMatchers) {
      if (m(tags, catSet)) return false;
    }

    return true;
  };
}
