"use client";

import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import {
  loadAllTags,
  searchTags as searchTagsOriginal,
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
  getNextFilterMode: (current: TagFilterMode) => TagFilterMode;
  getPreviousFilterMode: (current: TagFilterMode) => TagFilterMode;
  getFilterDisplayName: (mode: TagFilterMode) => string;
}

const TagSuggestionsContext = createContext<TagSuggestionsContextValue | undefined>(undefined);

export function TagSuggestionsProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);
  const [loadStatus, setLoadStatus] = useState<Record<string, boolean>>({});

  useEffect(() => {
    console.log("[TagSuggestionsProvider] Starting to load all tags...");
    setIsLoading(true);

    // Listen to category load events
    const unsubscribe = onCategoryLoaded((category, loaded) => {
      console.log(`[TagSuggestionsProvider] Category loaded: ${category} (${loaded})`);
      setLoadStatus((prev) => ({ ...prev, [category]: loaded }));
    });

    // Load all tags
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

  const value: TagSuggestionsContextValue = {
    isLoading,
    isLoaded,
    loadStatus,
    searchTags,
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
