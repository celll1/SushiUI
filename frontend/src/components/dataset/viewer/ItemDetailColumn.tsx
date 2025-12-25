"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Undo2, Redo2, Copy, Clipboard, Sparkles, Settings } from "lucide-react";
import {
  getDatasetItem,
  DatasetItem,
  updateItemCaption,
  saveItemCaptionToTxt,
  predictTags,
  TaggerPredictionsResponse,
} from "@/utils/api";
import InputWithTagSuggestions from "@/components/common/InputWithTagSuggestions";
import { normalizeTagForMatching } from "@/utils/tagSuggestions";
import { useTagSuggestions } from "@/contexts/TagSuggestionsContext";
import TaggerSettingsDialog, { TaggerSettings } from "./TaggerSettingsDialog";

interface ItemDetailColumnProps {
  item: DatasetItem | null;
  datasetId: number;
  tagCategoryCache: Record<string, string>; // Pre-loaded category map from parent
}

interface EditHistory {
  past: string[][];
  present: string[];
  future: string[][];
}


// Category colors mapping (for tag chips)
const getCategoryColor = (category: string): string => {
  const normalized = category.toLowerCase().replace(/\s+/g, '');
  const colors: Record<string, string> = {
    character: "bg-blue-600 dark:bg-blue-700 hover:bg-blue-500",
    artist: "bg-purple-600 dark:bg-purple-700 hover:bg-purple-500",
    copyright: "bg-pink-600 dark:bg-pink-700 hover:bg-pink-500",
    general: "bg-green-600 dark:bg-green-700 hover:bg-green-500",
    meta: "bg-gray-600 dark:bg-gray-700 hover:bg-gray-500",
    quality: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500",
    qualitytag: "bg-yellow-600 dark:bg-yellow-700 hover:bg-yellow-500", // "Quality Tag"
    rating: "bg-red-600 dark:bg-red-700 hover:bg-red-500",
    ratingtag: "bg-red-600 dark:bg-red-700 hover:bg-red-500", // "Rating Tag"
    model: "bg-indigo-600 dark:bg-indigo-700 hover:bg-indigo-500",
    unknown: "bg-orange-600 dark:bg-orange-700 hover:bg-orange-500",
  };
  return colors[normalized] || "bg-orange-600 dark:bg-orange-700 hover:bg-orange-500";
};

export default function ItemDetailColumn({ item, datasetId, tagCategoryCache }: ItemDetailColumnProps) {
  const tagSuggestionsContext = useTagSuggestions();
  const [detailedItem, setDetailedItem] = useState<DatasetItem | null>(null);
  const [tags, setTags] = useState<string[]>([]);
  const [tagCategories, setTagCategories] = useState<Record<string, string>>({});
  const [newTag, setNewTag] = useState("");
  const [history, setHistory] = useState<EditHistory>({
    past: [],
    present: [],
    future: [],
  });
  const previousItemIdRef = useRef<number | null>(null);
  const [activeFieldType, setActiveFieldType] = useState<string>("tags"); // Current field being displayed
  const [isTaggerSettingsOpen, setIsTaggerSettingsOpen] = useState(false);
  const [isTagging, setIsTagging] = useState(false);
  const [taggerSettings, setTaggerSettings] = useState<TaggerSettings | null>(null);
  const [isImageExpanded, setIsImageExpanded] = useState(false);
  const [expandedImageWidth, setExpandedImageWidth] = useState(800); // Pixels
  const [isResizing, setIsResizing] = useState(false);

  // Initialize tag categories from cache when item loads
  useEffect(() => {
    if (tags.length > 0 && Object.keys(tagCategoryCache).length > 0) {
      const categories: Record<string, string> = {};
      for (const tag of tags) {
        if (tagCategoryCache[tag]) {
          categories[tag] = tagCategoryCache[tag];
        }
      }
      setTagCategories(prev => ({ ...prev, ...categories }));
    }
  }, [tags, tagCategoryCache]);

  const loadItemDetails = useCallback(async () => {
    if (!item) return;

    try {
      const details = await getDatasetItem(datasetId, item.id);
      setDetailedItem(details);

      // Extract tags from captions
      const tagCaption = details.captions?.find(c => c.caption_type === "tags");
      if (tagCaption) {
        const tagList = tagCaption.content.split(",").map(t => t.trim()).filter(Boolean);
        setTags(tagList);
        setHistory({
          past: [],
          present: tagList,
          future: [],
        });

        // Load categories from tag_data if available (fast path)
        if (tagCaption.tag_data && tagCaption.tag_data.length > 0) {
          const categories: Record<string, string> = {};
          for (const item of tagCaption.tag_data) {
            categories[item.tag] = item.category;
          }
          setTagCategories(categories);
          console.log(`[ItemDetailColumn] Loaded ${Object.keys(categories).length} categories from tag_data (fast path)`);
        } else {
          // Fallback: categories will be loaded from cache via useEffect (Line 62-72)
          console.log("[ItemDetailColumn] No tag_data, using tagCategoryCache (legacy path)");
        }
      } else {
        setTags([]);
        setHistory({ past: [], present: [], future: [] });
      }
    } catch (err) {
      console.error("Failed to load item details:", err);
    }
  }, [item, datasetId]);

  useEffect(() => {
    if (item) {
      loadItemDetails();
    }
  }, [item, loadItemDetails]);

  // Update item reference when item changes
  useEffect(() => {
    if (item) {
      previousItemIdRef.current = item.id;
    }
  }, [item?.id]);

  // Keyboard shortcuts for field switching
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const availableFields = detailedItem?.captions?.map(c => c.caption_type) || ["tags"];

      // Tab: cycle through fields
      if (e.key === "Tab") {
        e.preventDefault();
        const currentIndex = availableFields.indexOf(activeFieldType);
        const nextIndex = (currentIndex + 1) % availableFields.length;
        setActiveFieldType(availableFields[nextIndex]);
      }

      // Number keys (1-9): switch to specific field
      const num = parseInt(e.key);
      if (num >= 1 && num <= availableFields.length) {
        e.preventDefault();
        setActiveFieldType(availableFields[num - 1]);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [detailedItem?.captions, activeFieldType]);

  // Resize handling for expanded image panel
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      const newWidth = e.clientX;
      setExpandedImageWidth(Math.max(300, Math.min(newWidth, window.innerWidth - 400)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isResizing]);

  // Build tag_data with categories for backend
  const buildTagData = async (tags: string[]): Promise<Array<{ tag: string; category: string }>> => {
    const tagData: Array<{ tag: string; category: string }> = [];

    for (const tag of tags) {
      // Try to get category from cache first
      const normalizedTag = normalizeTagForMatching(tag);
      let category = tagCategoryCache[normalizedTag];

      // If not in cache, search via tagSuggestions
      if (!category) {
        try {
          const results = await tagSuggestionsContext.searchTags(tag, 1, 'all');
          if (results.length > 0) {
            const normalizedUserTag = normalizeTagForMatching(tag);
            const normalizedResultTag = normalizeTagForMatching(results[0].tag);
            if (normalizedUserTag === normalizedResultTag) {
              category = results[0].category;
            }
          }
        } catch (err) {
          console.error(`[ItemDetailColumn] Failed to get category for tag "${tag}":`, err);
        }
      }

      // Default to "General" if category not found
      tagData.push({
        tag,
        category: category || "General",
      });
    }

    return tagData;
  };

  const saveToFileSystem = async (itemId: number) => {
    try {
      const result = await saveItemCaptionToTxt(itemId);
      if (result.success) {
        console.log("[ItemDetailColumn] Auto-saved to file:", result.message);
      } else {
        console.warn("[ItemDetailColumn] File save failed:", result.message);
      }
    } catch (err) {
      console.error("[ItemDetailColumn] Error auto-saving to file:", err);
    }
  };

  const pushHistory = async (newTags: string[]) => {
    setHistory({
      past: [...history.past, history.present],
      present: newTags,
      future: [],
    });
    setTags(newTags);

    // Immediately save to DB and file
    if (item) {
      try {
        const content = newTags.join(", ");
        const tag_data = await buildTagData(newTags);
        await updateItemCaption(item.id, {
          caption_type: "tags",
          content,
          tag_data,
        });
        console.log("[ItemDetailColumn] Tags saved to DB");

        // Auto-save to txt/json file
        await saveToFileSystem(item.id);
      } catch (err) {
        console.error("[ItemDetailColumn] Failed to save tags:", err);
      }
    }
  };

  const handleUndo = async () => {
    if (history.past.length === 0) return;

    const previous = history.past[history.past.length - 1];
    const newPast = history.past.slice(0, -1);

    setHistory({
      past: newPast,
      present: previous,
      future: [history.present, ...history.future],
    });
    setTags(previous);

    // Immediately save to DB and file
    if (item) {
      try {
        const content = previous.join(", ");
        const tag_data = await buildTagData(previous);
        await updateItemCaption(item.id, {
          caption_type: "tags",
          content,
          tag_data,
        });
        console.log("[ItemDetailColumn] Undo saved to DB");

        // Auto-save to txt/json file
        await saveToFileSystem(item.id);
      } catch (err) {
        console.error("[ItemDetailColumn] Failed to save undo:", err);
      }
    }
  };

  const handleRedo = async () => {
    if (history.future.length === 0) return;

    const next = history.future[0];
    const newFuture = history.future.slice(1);

    setHistory({
      past: [...history.past, history.present],
      present: next,
      future: newFuture,
    });
    setTags(next);

    // Immediately save to DB and file
    if (item) {
      try {
        const content = next.join(", ");
        const tag_data = await buildTagData(next);
        await updateItemCaption(item.id, {
          caption_type: "tags",
          content,
          tag_data,
        });
        console.log("[ItemDetailColumn] Redo saved to DB");

        // Auto-save to txt/json file
        await saveToFileSystem(item.id);
      } catch (err) {
        console.error("[ItemDetailColumn] Failed to save redo:", err);
      }
    }
  };


  const handleRemoveTag = (index: number) => {
    const newTags = tags.filter((_, i) => i !== index);
    pushHistory(newTags);
  };

  const handleCopyTags = () => {
    navigator.clipboard.writeText(tags.join(", "));
  };

  const handlePasteTags = async () => {
    try {
      const text = await navigator.clipboard.readText();
      const pastedTags = text.split(",").map(t => t.trim()).filter(Boolean);
      pushHistory([...tags, ...pastedTags]);
    } catch (err) {
      console.error("Failed to paste tags:", err);
    }
  };

  const handleTagAdd = (tag: string, category: string) => {
    if (tags.includes(tag)) {
      return; // Don't add duplicates
    }

    pushHistory([...tags, tag]);

    // Store category for this tag
    setTagCategories(prev => ({
      ...prev,
      [tag]: category
    }));
  };

  const handleTaggerInference = async () => {
    if (!item || !taggerSettings) {
      // If no settings, open settings dialog first
      setIsTaggerSettingsOpen(true);
      return;
    }

    setIsTagging(true);

    try {
      // Convert image to base64
      const imageResponse = await fetch(`/api/serve-image?path=${encodeURIComponent(item.image_path)}`);
      const imageBlob = await imageResponse.blob();
      const base64 = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const dataUrl = reader.result as string;
          resolve(dataUrl.split(',')[1]);
        };
        reader.readAsDataURL(imageBlob);
      });

      // Build thresholds dict (using addThreshold for prediction)
      const thresholds: { [key: string]: number } = {};
      taggerSettings.categoryThresholds.forEach(cat => {
        if (cat.enabled) {
          thresholds[cat.id] = cat.addThreshold;
        }
      });

      const genThreshold = taggerSettings.categoryThresholds.find(c => c.id === "general")?.addThreshold || 0.45;
      const charThreshold = taggerSettings.categoryThresholds.find(c => c.id === "character")?.addThreshold || 0.45;

      // Predict tags
      const response = await predictTags(
        base64,
        genThreshold,
        charThreshold,
        taggerSettings.modelVersion,
        true, // auto_unload
        thresholds
      );

      // Process predictions: merge with existing tags
      const existingTags = new Set(tags);
      const predictedTags = new Map<string, { confidence: number; category: string }>(); // tag -> {confidence, category}

      // Collect all predicted tags with confidence and category
      Object.entries(response.predictions).forEach(([category, categoryTags]) => {
        categoryTags.forEach(([tag, confidence]) => {
          predictedTags.set(tag, { confidence, category });
        });
      });

      // Build category threshold map for quick lookup
      const removeThresholdMap = new Map<string, number>();
      const addThresholdMap = new Map<string, number>();
      taggerSettings.categoryThresholds.forEach(cat => {
        removeThresholdMap.set(cat.id, cat.removeThreshold);
        addThresholdMap.set(cat.id, cat.addThreshold);
      });

      // Build new tag list
      const newTags: string[] = [];

      // Special handling for Rating and Quality: pick top prediction only
      const ratingEnabled = taggerSettings.categoryThresholds.find(c => c.id === "rating")?.enabled;
      const qualityEnabled = taggerSettings.categoryThresholds.find(c => c.id === "quality")?.enabled;

      const existingRatingTags = Array.from(existingTags).filter(tag => {
        const category = tagCategories[tag] || (predictedTags.get(tag)?.category);
        return category?.toLowerCase() === "rating";
      });
      const existingQualityTags = Array.from(existingTags).filter(tag => {
        const category = tagCategories[tag] || (predictedTags.get(tag)?.category);
        return category?.toLowerCase() === "quality";
      });

      // Get top predicted rating and quality tags
      let topRatingTag: string | null = null;
      let topRatingConfidence = 0;
      let topQualityTag: string | null = null;
      let topQualityConfidence = 0;

      predictedTags.forEach(({ confidence, category }, tag) => {
        if (category === "rating" && confidence > topRatingConfidence) {
          topRatingTag = tag;
          topRatingConfidence = confidence;
        }
        if (category === "quality" && confidence > topQualityConfidence) {
          topQualityTag = tag;
          topQualityConfidence = confidence;
        }
      });

      // 1. Keep existing tags (with special handling for rating/quality)
      existingTags.forEach(tag => {
        const predicted = predictedTags.get(tag);
        if (!predicted) {
          // Tag not in predictions, keep it
          newTags.push(tag);
        } else if (predicted.category === "rating" || predicted.category === "quality") {
          // Rating/Quality: handled separately below
          return;
        } else {
          // Other categories: check category-specific removeThreshold
          const removeThreshold = removeThresholdMap.get(predicted.category) || 0.0;
          if (predicted.confidence >= removeThreshold) {
            newTags.push(tag);
          }
        }
      });

      // 2. Add new predicted tags
      predictedTags.forEach(({ confidence, category }, tag) => {
        if (category === "rating" || category === "quality") {
          // Rating/Quality: handled separately below
          return;
        }
        const addThreshold = addThresholdMap.get(category) || 0.45;
        if (confidence >= addThreshold && !existingTags.has(tag)) {
          newTags.push(tag);
        }
      });

      // 3. Handle Rating: add top prediction only if no existing rating tag
      if (ratingEnabled && topRatingTag && existingRatingTags.length === 0) {
        newTags.push(topRatingTag);
      }

      // 4. Handle Quality: add top prediction only if no existing quality tag
      if (qualityEnabled && topQualityTag && existingQualityTags.length === 0) {
        newTags.push(topQualityTag);
      }

      // Update tags with history
      pushHistory(newTags);

      console.log(`[Tagger] Inference complete: ${tags.length} → ${newTags.length} tags (removed: ${tags.length - newTags.filter(t => existingTags.has(t)).length}, added: ${newTags.filter(t => !existingTags.has(t)).length})`);
    } catch (error) {
      console.error("[Tagger] Inference failed:", error);
      alert("Tagger inference failed. See console for details.");
    } finally {
      setIsTagging(false);
    }
  };

  if (!item) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        Select an item to view details
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header - Compact */}
      <div className="flex-shrink-0 p-2 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold">Item Details</h3>
          <div className="text-[10px] text-gray-400">
            {item.width}×{item.height} • {(item.file_size / 1024).toFixed(1)}KB
          </div>
        </div>
      </div>

      {/* Content - Optimized Layout */}
      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto p-2 space-y-2">
        {/* Image + File Info - Horizontal Layout */}
        <div className="flex-shrink-0 flex gap-2">
          {/* Image Preview - Small Thumbnail */}
          <div className="w-32 h-32 bg-gray-800 rounded overflow-hidden flex-shrink-0">
            <img
              src={`/api/serve-image?path=${encodeURIComponent(item.image_path)}`}
              alt={item.base_name}
              className="w-full h-full object-contain bg-gray-900 cursor-pointer hover:opacity-80 transition-opacity"
              onDoubleClick={() => setIsImageExpanded(true)}
              title="Double-click to expand"
            />
          </div>

          {/* File Info */}
          <div className="flex-1 bg-gray-800 rounded p-2 min-w-0">
            <div className="text-xs font-medium text-gray-200 truncate mb-1" title={item.base_name}>
              {item.base_name}
            </div>
            <div className="text-[10px] text-gray-400 truncate" title={item.image_path}>
              {item.image_path}
            </div>
          </div>
        </div>

        {/* Field Switcher - Category Buttons (tags, text, metadata) */}
        <div className="flex-shrink-0 flex gap-1 flex-wrap">
          {(() => {
            const captions = detailedItem?.captions || [];
            const tagsCaptions = captions.filter(c => c.field_category === 'training' && c.is_tags_format);
            const textCaptions = captions.filter(c => c.field_category === 'training' && !c.is_tags_format);
            const metadataCaptions = captions.filter(c => c.field_category === 'metadata');

            return (
              <>
                {/* Tags button */}
                {tagsCaptions.length > 0 && (
                  <button
                    onClick={() => setActiveFieldType(tagsCaptions[0].caption_type)}
                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                      tagsCaptions.some(c => c.caption_type === activeFieldType)
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                    }`}
                    title="Tags (Danbooru format)"
                  >
                    Tags
                  </button>
                )}

                {/* Text button */}
                {textCaptions.length > 0 && (
                  <button
                    onClick={() => setActiveFieldType(textCaptions[0].caption_type)}
                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                      textCaptions.some(c => c.caption_type === activeFieldType)
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                    }`}
                    title="Natural language text"
                  >
                    Text
                  </button>
                )}

                {/* Metadata button (collapsible) */}
                {metadataCaptions.length > 0 && (
                  <button
                    onClick={() => setActiveFieldType('__metadata__')}
                    className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                      activeFieldType === '__metadata__'
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 hover:bg-gray-600 text-gray-300"
                    }`}
                    title={`Metadata (${metadataCaptions.length} fields)`}
                  >
                    Metadata ({metadataCaptions.length})
                  </button>
                )}
              </>
            );
          })()}
        </div>

        {/* Caption Display Area - Unified */}
        <div className="flex-1 bg-gray-800 rounded-lg p-2 flex flex-col min-h-0">
          {/* Header with field name and actions */}
          <div className="flex-shrink-0 flex items-center justify-between mb-2">
            <h4 className="text-xs font-semibold capitalize">
              {activeFieldType === '__metadata__'
                ? `Metadata (${detailedItem?.captions?.filter(c => c.field_category === 'metadata').length || 0} fields)`
                : activeFieldType.replace(/_/g, " ")
              }
              {activeFieldType === "tags" && ` (${tags.length})`}
            </h4>
            {activeFieldType === "tags" && (
              <div className="flex items-center space-x-0.5">
                <button
                  onClick={handleTaggerInference}
                  disabled={isTagging}
                  className="p-0.5 hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  title="Run Tagger Inference"
                >
                  <Sparkles className={`h-3 w-3 ${isTagging ? 'animate-pulse text-blue-400' : ''}`} />
                </button>
                <button
                  onClick={() => setIsTaggerSettingsOpen(true)}
                  className="p-0.5 hover:bg-gray-700 rounded transition-colors"
                  title="Tagger Settings"
                >
                  <Settings className="h-3 w-3" />
                </button>
                <div className="w-px h-3 bg-gray-600 mx-0.5" />
                <button
                  onClick={handleUndo}
                  disabled={history.past.length === 0}
                  className="p-0.5 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Undo (Ctrl+Z)"
                >
                  <Undo2 className="h-3 w-3" />
                </button>
                <button
                  onClick={handleRedo}
                  disabled={history.future.length === 0}
                  className="p-0.5 hover:bg-gray-700 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="Redo (Ctrl+Y)"
                >
                  <Redo2 className="h-3 w-3" />
                </button>
                <button
                  onClick={handleCopyTags}
                  className="p-0.5 hover:bg-gray-700 rounded transition-colors"
                  title="Copy Tags (Ctrl+C)"
                >
                  <Copy className="h-3 w-3" />
                </button>
                <button
                  onClick={handlePasteTags}
                  className="p-0.5 hover:bg-gray-700 rounded transition-colors"
                  title="Paste Tags (Ctrl+V)"
                >
                  <Clipboard className="h-3 w-3" />
                </button>
              </div>
            )}
          </div>

          {/* Content Area - Tags, Text, or Metadata */}
          {activeFieldType === "tags" ? (
            <>
              {/* Tag List - Scrollable */}
              <div className="flex-1 flex flex-wrap gap-1 content-start bg-gray-900 rounded p-2 overflow-y-auto min-h-0">
                {tags.length === 0 ? (
                  <div className="text-xs text-gray-500 w-full text-center py-2">No tags</div>
                ) : (
                  tags.map((tag, index) => {
                    const category = tagCategories[tag] || "general";
                    const colorClass = getCategoryColor(category);
                    return (
                      <div
                        key={index}
                        className={`flex items-center space-x-1 px-2 py-1 ${colorClass} rounded text-xs transition-colors group h-fit cursor-pointer`}
                        title={`Category: ${category}`}
                        onClick={() => handleRemoveTag(index)}
                      >
                        <span>{tag}</span>
                        <span className="opacity-0 group-hover:opacity-100 transition-opacity text-[10px]">
                          ✕
                        </span>
                      </div>
                    );
                  })
                )}
              </div>

              {/* Add Tag - with Autocomplete */}
              <div className="flex-shrink-0 mt-2">
                <InputWithTagSuggestions
                  value={newTag}
                  onChange={setNewTag}
                  onTagAdd={handleTagAdd}
                  placeholder="Type to search tags..."
                  className="w-full px-2 py-1 bg-gray-900 border border-gray-700 rounded text-xs focus:outline-none focus:border-blue-500"
                  showSuggestionsAbove={true}
                />
              </div>
            </>
          ) : activeFieldType === '__metadata__' ? (
            <>
              {/* Metadata Display - All fields in one textbox */}
              <div className="flex-1 bg-gray-900 rounded p-2 overflow-y-auto min-h-0">
                <textarea
                  readOnly
                  value={(() => {
                    const metadataCaptions = detailedItem?.captions?.filter(c => c.field_category === 'metadata') || [];
                    return metadataCaptions
                      .map(c => `${c.source_field || c.caption_type}: ${c.content}`)
                      .join('\n');
                  })()}
                  className="w-full h-full bg-transparent text-xs text-gray-300 font-mono resize-none focus:outline-none"
                  spellCheck={false}
                />
              </div>
              <div className="flex-shrink-0 mt-2 text-[10px] text-gray-500">
                {detailedItem?.captions?.filter(c => c.field_category === 'metadata').length || 0} metadata fields
              </div>
            </>
          ) : (
            <>
              {/* Read-only Caption Display (Text/Natural Language) */}
              <div className="flex-1 bg-gray-900 rounded p-2 overflow-y-auto min-h-0">
                <p className="text-xs text-gray-300 whitespace-pre-wrap">
                  {detailedItem?.captions?.find(c => c.caption_type === activeFieldType)?.content || "No content"}
                </p>
              </div>
              <div className="flex-shrink-0 mt-2 text-[10px] text-gray-500">
                Source: {detailedItem?.captions?.find(c => c.caption_type === activeFieldType)?.source || "unknown"}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Tagger Settings Dialog */}
      <TaggerSettingsDialog
        isOpen={isTaggerSettingsOpen}
        onClose={() => setIsTaggerSettingsOpen(false)}
        onSave={setTaggerSettings}
      />

      {/* Expanded Image Popup - Slides from left with resizable width */}
      {isImageExpanded && (
        <div
          className="fixed top-0 left-0 bottom-0 bg-gray-900 shadow-2xl flex"
          style={{ width: `${expandedImageWidth}px`, zIndex: 45 }}
        >
          {/* Image Container */}
          <div className="flex-1 flex items-center justify-center p-4">
            <img
              src={`/api/serve-image?path=${encodeURIComponent(item.image_path)}`}
              alt={item.base_name}
              className="max-w-full max-h-full object-contain"
            />
          </div>

          {/* Resize Handle */}
          <div
            className="w-1 bg-gray-700 hover:bg-blue-500 cursor-ew-resize transition-colors"
            onMouseDown={(e) => {
              e.preventDefault();
              setIsResizing(true);
            }}
            title="Drag to resize"
          />

          {/* Close Button */}
          <button
            className="absolute top-2 right-2 bg-gray-800 hover:bg-gray-700 text-white rounded p-2 transition-colors z-10"
            onClick={() => setIsImageExpanded(false)}
            title="Close"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
