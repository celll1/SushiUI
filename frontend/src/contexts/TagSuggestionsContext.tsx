"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  loadAllTags,
  searchTags as searchTagsOriginal,
  getCategoriesForTags as getCategoriesForTagsOriginal,
  TagFilterMode,
  getNextFilterMode,
  getPreviousFilterMode,
  getFilterDisplayName,
  getCategoriesLoadStatus,
  onCategoryLoaded,
} from "@/utils/tagSuggestions";

interface TagSuggestionsContextValue {
  isLoading: boolean;
  isLoaded: boolean;
  loadStatus: Record<string, boolean>;
  searchTags: (
    input: string,
    limit?: number,
    filterMode?: TagFilterMode
  ) => Promise<Array<{ tag: string; count: number; category: string; alias?: string }>>;
  getCategoriesForTags: (tags: string[]) => Promise<Map<string, string>>;
  getNextFilterMode: (current: TagFilterMode) => TagFilterMode;
  getPreviousFilterMode: (current: TagFilterMode) => TagFilterMode;
  getFilterDisplayName: (mode: TagFilterMode) => string;
}

const TagSuggestionsContext = createContext<TagSuggestionsContextValue | undefined>(undefined);

// Global flag to prevent duplicate loading (across re-mounts in dev mode)
let globalLoadingStarted = false;

export function TagSuggestionsProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(false); // Don't block UI
  const [isLoaded, setIsLoaded] = useState(false);
  const [loadStatus, setLoadStatus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    // Prevent duplicate loading if already started
    if (globalLoadingStarted) {
      console.log("[TagSuggestionsProvider] Already loading, skipping duplicate mount");
      setLoadStatus(getCategoriesLoadStatus());
      return;
    }

    globalLoadingStarted = true;
    console.log("[TagSuggestionsProvider] Starting to load all tags (background)...");
    setIsLoading(true);

    // Listen to category load events
    const unsubscribe = onCategoryLoaded((category, loaded) => {
      setLoadStatus((prev) => ({ ...prev, [category]: loaded }));
    });

    // Load all tags in background (non-blocking)
    loadAllTags()
      .then(() => {
        console.log("[TagSuggestionsProvider] All tags loaded successfully");
        setIsLoaded(true);
        setIsLoading(false);
        setLoadStatus(getCategoriesLoadStatus());
      })
      .catch((err) => {
        console.error("[TagSuggestionsProvider] Failed to load tags:", err);
        setIsLoading(false);
        globalLoadingStarted = false; // Allow retry on error
      });

    return () => {
      unsubscribe();
    };
  }, []);

  const searchTags = useCallback(
    async (input: string, limit: number = 20, filterMode: TagFilterMode = "all") => {
      return searchTagsOriginal(input, limit, filterMode);
    },
    []
  );

  const getCategoriesForTags = useCallback(
    async (tags: string[]) => {
      return getCategoriesForTagsOriginal(tags);
    },
    []
  );

  const value: TagSuggestionsContextValue = {
    isLoading,
    isLoaded,
    loadStatus,
    searchTags,
    getCategoriesForTags,
    getNextFilterMode,
    getPreviousFilterMode,
    getFilterDisplayName,
  };

  return (
    <TagSuggestionsContext.Provider value={value}>{children}</TagSuggestionsContext.Provider>
  );
}

export function useTagSuggestions() {
  const context = useContext(TagSuggestionsContext);
  if (context === undefined) {
    throw new Error("useTagSuggestions must be used within a TagSuggestionsProvider");
  }
  return context;
}
