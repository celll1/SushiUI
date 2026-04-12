"use client";

import { useState, useEffect, useMemo } from "react";
import { getTaggerRunVocabulary, getSigLIP2LoadedVocabulary, VocabularyData } from "@/utils/api";

const CATEGORY_TEXT_COLOR: Record<string, string> = {
  Quality:   "text-yellow-400",
  Rating:    "text-orange-400",
  Character: "text-blue-400",
  Copyright: "text-purple-400",
  General:   "text-green-400",
  Artist:    "text-pink-400",
  Meta:      "text-gray-400",
  Model:     "text-cyan-400",
  Unknown:   "text-gray-500",
};

const CATEGORY_ORDER = [
  "General", "Character", "Copyright", "Artist", "Meta", "Rating", "Quality", "Model",
];

type FilterMode = "partial" | "wildcard" | "regex";

// Convert fnmatch-style wildcard to RegExp
function wildcardToRegex(pattern: string): RegExp | null {
  try {
    const escaped = pattern
      .replace(/[.+^${}()|[\]\\]/g, "\\$&") // escape regex special chars
      .replace(/\*/g, ".*")
      .replace(/\?/g, ".");
    return new RegExp(`^${escaped}$`, "i");
  } catch {
    return null;
  }
}

function buildMatcher(query: string, mode: FilterMode): ((tag: string) => boolean) | null {
  if (!query) return null;
  if (mode === "partial") {
    const q = query.toLowerCase();
    return (tag) => tag.toLowerCase().includes(q);
  }
  if (mode === "wildcard") {
    const re = wildcardToRegex(query);
    if (!re) return null;
    return (tag) => re.test(tag);
  }
  // regex
  try {
    const re = new RegExp(query, "i");
    return (tag) => re.test(tag);
  } catch {
    return null;
  }
}

interface VocabularyBrowserProps {
  runId?: string;
  useLoadedModel?: boolean;
  defaultOpen?: boolean;
}

const MAX_DISPLAY = 500;

export default function VocabularyBrowser({ runId, useLoadedModel, defaultOpen = false }: VocabularyBrowserProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [vocab, setVocab] = useState<VocabularyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [filterMode, setFilterMode] = useState<FilterMode>("partial");
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [regexError, setRegexError] = useState<string | null>(null);

  // Fetch vocabulary when opened
  useEffect(() => {
    if (!open || vocab) return;
    if (!runId && !useLoadedModel) return;
    setLoading(true);
    setFetchError(null);
    const fetch = runId
      ? getTaggerRunVocabulary(runId)
      : getSigLIP2LoadedVocabulary();
    fetch
      .then(setVocab)
      .catch((e) => setFetchError(e?.response?.data?.detail ?? e?.message ?? "Failed to load vocabulary"))
      .finally(() => setLoading(false));
  }, [open, runId, useLoadedModel]);

  // Build flat tag list
  const allTags = useMemo<Array<{ tag: string; category: string }>>(() => {
    if (!vocab) return [];
    const tagToCategory = vocab.tag_to_category || {};
    return Object.keys(tagToCategory).map((tag) => ({ tag, category: tagToCategory[tag] ?? "Unknown" }));
  }, [vocab]);

  // Category list from vocabulary
  const categories = useMemo(() => {
    if (!vocab) return [];
    const cats = new Set(Object.values(vocab.tag_to_category ?? {}));
    return CATEGORY_ORDER.filter(c => cats.has(c)).concat(
      [...cats].filter(c => !CATEGORY_ORDER.includes(c)).sort()
    );
  }, [vocab]);

  // Category counts
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const { category } of allTags) {
      counts[category] = (counts[category] ?? 0) + 1;
    }
    return counts;
  }, [allTags]);

  // Filtered tags
  const filteredTags = useMemo(() => {
    setRegexError(null);
    let tags = allTags;

    // Category filter
    if (selectedCategory !== "All") {
      tags = tags.filter(t => t.category === selectedCategory);
    }

    // Text filter
    if (query) {
      let matcher: ((tag: string) => boolean) | null;
      if (filterMode === "regex") {
        try {
          new RegExp(query, "i"); // validate
          matcher = buildMatcher(query, "regex");
        } catch (e: any) {
          setRegexError(e.message);
          return tags.slice(0, MAX_DISPLAY);
        }
      } else {
        matcher = buildMatcher(query, filterMode);
      }
      if (matcher) tags = tags.filter(t => matcher!(t.tag));
    }

    return tags;
  }, [allTags, query, filterMode, selectedCategory]);

  const displayed = filteredTags.slice(0, MAX_DISPLAY);

  if (!runId && !useLoadedModel) return null;

  return (
    <div className="border border-gray-700 rounded">
      {/* Header toggle */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium text-gray-300 hover:text-white hover:bg-gray-800 rounded transition-colors"
      >
        <span>
          Vocabulary
          {vocab && <span className="text-gray-500 ml-1 font-normal">({vocab.num_tags.toLocaleString()} tags)</span>}
        </span>
        <span className="text-gray-500 text-xs">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-gray-700 p-3 space-y-3">
          {loading && (
            <div className="text-sm text-gray-400">Loading vocabulary…</div>
          )}
          {fetchError && (
            <div className="text-sm text-red-400">{fetchError}</div>
          )}

          {vocab && (
            <>
              {/* Category stats */}
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-gray-500">
                {categories.map(cat => (
                  <span key={cat} className={CATEGORY_TEXT_COLOR[cat] ?? "text-gray-500"}>
                    {cat} {(categoryCounts[cat] ?? 0).toLocaleString()}
                  </span>
                ))}
              </div>

              {/* Filter row */}
              <div className="flex gap-2 flex-wrap">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={
                    filterMode === "partial" ? "girls → includes girls" :
                    filterMode === "wildcard" ? "*girls → ends with girls" :
                    "regex: ^2.*girls$"
                  }
                  className="flex-1 min-w-0 bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500"
                />
                <div className="flex rounded overflow-hidden border border-gray-600 text-xs shrink-0">
                  {(["partial", "wildcard", "regex"] as FilterMode[]).map(mode => (
                    <button
                      key={mode}
                      onClick={() => { setFilterMode(mode); setRegexError(null); }}
                      className={`px-2 py-1 capitalize ${filterMode === mode ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>
              {regexError && (
                <p className="text-xs text-red-400">Regex error: {regexError}</p>
              )}

              {/* Category filter buttons */}
              <div className="flex flex-wrap gap-1">
                <button
                  onClick={() => setSelectedCategory("All")}
                  className={`px-2 py-0.5 rounded text-xs ${selectedCategory === "All" ? "bg-gray-600 text-white" : "text-gray-400 hover:bg-gray-700"}`}
                >
                  All
                </button>
                {categories.map(cat => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-2 py-0.5 rounded text-xs ${selectedCategory === cat ? "bg-gray-600 text-white" : `${CATEGORY_TEXT_COLOR[cat] ?? "text-gray-400"} hover:bg-gray-700`}`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              {/* Tag list */}
              <div className="max-h-64 overflow-y-auto space-y-0.5 border border-gray-700 rounded p-1">
                {displayed.length === 0 ? (
                  <div className="text-sm text-gray-500 p-2 text-center">No tags match</div>
                ) : (
                  displayed.map(({ tag, category }) => (
                    <div key={tag} className="flex items-center gap-2 px-2 py-0.5 rounded hover:bg-gray-800">
                      <span className="text-sm text-gray-200 flex-1 truncate" title={tag}>{tag}</span>
                      <span className={`text-xs shrink-0 ${CATEGORY_TEXT_COLOR[category] ?? "text-gray-500"}`}>
                        {category}
                      </span>
                    </div>
                  ))
                )}
              </div>

              {/* Count */}
              <div className="text-xs text-gray-500">
                Showing {displayed.length.toLocaleString()} / {filteredTags.length.toLocaleString()} tags
                {filteredTags.length > MAX_DISPLAY && (
                  <span className="ml-1 text-yellow-600">(refine your filter to see more)</span>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
