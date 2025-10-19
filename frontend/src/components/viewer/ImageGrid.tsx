"use client";

/**
 * ImageGrid - Gallery view with filters and pagination
 *
 * Performance optimizations:
 * - GalleryFilter and ImageList are memoized components
 * - All callbacks are wrapped with useCallback to prevent filter re-renders
 * - Computed values (tagSuggestions, filteredImages) use useMemo
 * - Loading state is handled within ImageList to avoid full re-render
 */

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { getImages, GeneratedImage, ImageFilters } from "@/utils/api";
import Card from "../common/Card";
import Button from "../common/Button";
import GalleryFilter from "./GalleryFilter";
import ImageList from "./ImageList";

export default function ImageGrid() {
  const router = useRouter();
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);

  // Filter states
  const [filterTxt2Img, setFilterTxt2Img] = useState(true);
  const [filterImg2Img, setFilterImg2Img] = useState(true);
  const [filterInpaint, setFilterInpaint] = useState(true);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [widthRange, setWidthRange] = useState<[number, number]>([0, 2048]);
  const [heightRange, setHeightRange] = useState<[number, number]>([0, 2048]);
  // Committed range values (only updated after drag ends)
  const [committedWidthRange, setCommittedWidthRange] = useState<[number, number]>([0, 2048]);
  const [committedHeightRange, setCommittedHeightRange] = useState<[number, number]>([0, 2048]);
  const [tagSearchInput, setTagSearchInput] = useState("");
  const [tagSearchCommitted, setTagSearchCommitted] = useState<string[]>([]);
  const [searchInNegative, setSearchInNegative] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(-1);
  const [excludeRareTags, setExcludeRareTags] = useState(true);

  // UI states
  const [gridColumns, setGridColumns] = useState(6);
  const [showFullSizeImage, setShowFullSizeImage] = useState(false);

  // Pagination states
  const [currentPage, setCurrentPage] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const imagesPerPage = 100;

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [filterTxt2Img, filterImg2Img, filterInpaint, dateFrom, dateTo, committedWidthRange, committedHeightRange]);

  useEffect(() => {
    loadImages();
  }, [filterTxt2Img, filterImg2Img, filterInpaint, dateFrom, dateTo, committedWidthRange, committedHeightRange, currentPage]);

  const loadImages = async () => {
    try {
      setLoading(true);

      // Build generation types filter
      const types: string[] = [];
      if (filterTxt2Img) types.push("txt2img");
      if (filterImg2Img) types.push("img2img");
      if (filterInpaint) types.push("inpaint");

      const filters: ImageFilters = {
        skip: (currentPage - 1) * imagesPerPage,
        limit: imagesPerPage,
        generation_types: types.length > 0 ? types.join(",") : undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        width_min: committedWidthRange[0] > 0 ? committedWidthRange[0] : undefined,
        width_max: committedWidthRange[1] < 2048 ? committedWidthRange[1] : undefined,
        height_min: committedHeightRange[0] > 0 ? committedHeightRange[0] : undefined,
        height_max: committedHeightRange[1] < 2048 ? committedHeightRange[1] : undefined,
      };

      const result = await getImages(filters);
      setImages(result.images);
      setTotalImages(result.total || 0);
    } catch (error) {
      console.error("Failed to load images:", error);
    } finally {
      setLoading(false);
    }
  };

  const findImageByHash = (hash: string): GeneratedImage | undefined => {
    return images.find((img) => img.image_hash === hash);
  };

  const handleSourceImageClick = (sourceHash: string) => {
    const sourceImage = findImageByHash(sourceHash);
    if (sourceImage) {
      setSelectedImage(sourceImage);
    } else {
      alert("Source image not found in current gallery view. Try adjusting filters.");
    }
  };

  // Extract unique tags from all prompts for autocomplete - memoized
  const tagSuggestions = useMemo((): string[] => {
    if (!tagSearchInput || tagSearchInput.length < 2) return [];

    const searchLower = tagSearchInput.toLowerCase();
    const tagCount = new Map<string, number>();

    images.forEach((image) => {
      const promptText = searchInNegative ? image.negative_prompt : image.prompt;
      if (!promptText) return;

      // Split by common delimiters (comma, space, etc.)
      const tags = promptText.split(/[,\n]+/).map(t => t.trim()).filter(t => t.length > 0);

      tags.forEach(tag => {
        if (tag.toLowerCase().includes(searchLower)) {
          tagCount.set(tag, (tagCount.get(tag) || 0) + 1);
        }
      });
    });

    // Filter out tags that appear only once if option is enabled
    const filteredTags = Array.from(tagCount.entries())
      .filter(([_, count]) => !excludeRareTags || count > 1)
      .map(([tag, _]) => tag)
      .slice(0, 10); // Limit to 10 suggestions

    return filteredTags;
  }, [tagSearchInput, images, searchInNegative, excludeRareTags]);

  // Client-side tag filtering (only apply committed search) - AND search with exact match - memoized
  const filteredImages = useMemo(() => {
    return images.filter((image) => {
      if (tagSearchCommitted.length === 0) return true;

      const searchField = searchInNegative ? image.negative_prompt : image.prompt;
      if (!searchField) return false;

      // Split tags by comma and trim
      const imageTags = searchField.split(/[,\n]+/).map(t => t.trim().toLowerCase());

      // AND search: all committed tags must exist as exact matches
      return tagSearchCommitted.every(searchTag =>
        imageTags.includes(searchTag.toLowerCase())
      );
    });
  }, [images, tagSearchCommitted, searchInNegative]);

  // Keyboard navigation for pagination and image navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // If viewing a single image, handle image navigation
      if (selectedImage) {
        const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage.filename);

        if (e.key === 'ArrowLeft' && currentIndex > 0) {
          e.preventDefault();
          setSelectedImage(filteredImages[currentIndex - 1]);
        } else if (e.key === 'ArrowRight' && currentIndex < filteredImages.length - 1) {
          e.preventDefault();
          setSelectedImage(filteredImages[currentIndex + 1]);
        }
      } else {
        // Gallery pagination
        if (e.key === 'ArrowLeft' && currentPage > 1 && !loading) {
          e.preventDefault();
          setCurrentPage(currentPage - 1);
        } else if (e.key === 'ArrowRight' && currentPage * imagesPerPage < totalImages && !loading) {
          e.preventDefault();
          setCurrentPage(currentPage + 1);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPage, totalImages, loading, imagesPerPage, selectedImage, filteredImages]);

  const handleTagSearchSubmit = useCallback(() => {
    if (tagSearchInput.trim() && !tagSearchCommitted.includes(tagSearchInput.trim())) {
      setTagSearchCommitted([...tagSearchCommitted, tagSearchInput.trim()]);
      setTagSearchInput("");
    }
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  }, [tagSearchInput, tagSearchCommitted]);

  const removeTag = useCallback((tagToRemove: string) => {
    setTagSearchCommitted(tagSearchCommitted.filter(tag => tag !== tagToRemove));
  }, [tagSearchCommitted]);

  const clearAllTags = useCallback(() => {
    setTagSearchCommitted([]);
    setTagSearchInput("");
  }, []);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    if (!tagSearchCommitted.includes(suggestion)) {
      setTagSearchCommitted([...tagSearchCommitted, suggestion]);
    }
    setTagSearchInput("");
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  }, [tagSearchCommitted]);

  const handleTagSearchKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      if (selectedSuggestionIndex >= 0 && selectedSuggestionIndex < tagSuggestions.length) {
        handleSuggestionClick(tagSuggestions[selectedSuggestionIndex]);
      } else {
        handleTagSearchSubmit();
      }
      e.preventDefault();
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      setSelectedSuggestionIndex(-1);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedSuggestionIndex(prev =>
        prev < tagSuggestions.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedSuggestionIndex(prev => prev > 0 ? prev - 1 : -1);
    }
  }, [selectedSuggestionIndex, tagSuggestions, handleSuggestionClick, handleTagSearchSubmit]);

  const sendToTxt2Img = (image: GeneratedImage) => {
    // Note: Send image is not applicable for txt2img (no input image)

    // Send prompt if checked
    if (sendPrompt) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.prompt = image.prompt;
      txt2imgParams.negative_prompt = image.negative_prompt;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");
      txt2imgParams.steps = image.steps;
      txt2imgParams.cfg_scale = image.cfg_scale;
      txt2imgParams.sampler = image.parameters?.sampler || "euler";
      txt2imgParams.schedule_type = image.parameters?.schedule_type || "uniform";
      txt2imgParams.seed = image.seed;
      txt2imgParams.width = image.width;
      txt2imgParams.height = image.height;
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
    }

    router.push("/generate");
  };

  const sendToImg2Img = (image: GeneratedImage) => {
    // Send image if checked
    if (sendImage) {
      const imageUrl = `/outputs/${image.filename}`;
      localStorage.setItem("img2img_input_image", imageUrl);
      window.dispatchEvent(new Event("img2img_input_updated"));
    }

    // Send prompt if checked
    if (sendPrompt) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.prompt = image.prompt;
      img2imgParams.negative_prompt = image.negative_prompt;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
      img2imgParams.steps = image.steps;
      img2imgParams.cfg_scale = image.cfg_scale;
      img2imgParams.sampler = image.parameters?.sampler || "euler";
      img2imgParams.schedule_type = image.parameters?.schedule_type || "uniform";
      img2imgParams.seed = image.seed;
      img2imgParams.width = image.width;
      img2imgParams.height = image.height;
      img2imgParams.denoising_strength = 0.75;
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
    }

    router.push("/generate?tab=img2img");
  };

  const sendToInpaint = (image: GeneratedImage) => {
    // Send image if checked
    if (sendImage) {
      const imageUrl = `/outputs/${image.filename}`;
      localStorage.setItem("inpaint_input_image", imageUrl);
      localStorage.removeItem("inpaint_mask_image");
      window.dispatchEvent(new Event("inpaint_input_updated"));
    }

    // Send prompt if checked
    if (sendPrompt) {
      const inpaintParams = JSON.parse(localStorage.getItem("inpaint_params") || "{}");
      inpaintParams.prompt = image.prompt;
      inpaintParams.negative_prompt = image.negative_prompt;
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
    }

    // Send parameters if checked
    if (sendParameters) {
      const inpaintParams = JSON.parse(localStorage.getItem("inpaint_params") || "{}");
      inpaintParams.steps = image.steps;
      inpaintParams.cfg_scale = image.cfg_scale;
      inpaintParams.sampler = image.parameters?.sampler || "euler";
      inpaintParams.schedule_type = image.parameters?.schedule_type || "uniform";
      inpaintParams.seed = image.seed;
      inpaintParams.width = image.width;
      inpaintParams.height = image.height;
      inpaintParams.denoising_strength = 0.75;
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
    }

    router.push("/generate?tab=inpaint");
  };

  return (
    <div>
      {selectedImage ? (
        <div className="h-full">
          <button
            onClick={() => setSelectedImage(null)}
            className="text-blue-400 hover:text-blue-300 mb-4 block"
          >
            ← Back to gallery
          </button>
          <div className="flex gap-4 h-[calc(100vh-12rem)]">
            {/* Left Sidebar - Details */}
            <div className="w-80 flex-shrink-0 flex flex-col">
              {/* Scrollable content area */}
              <div className="flex-1 overflow-y-auto">
              <Card title="Image Details">
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400">Prompt:</span>
                    <p className="text-gray-100">{selectedImage.prompt}</p>
                  </div>
                {selectedImage.negative_prompt && (
                  <div>
                    <span className="text-gray-400">Negative Prompt:</span>
                    <p className="text-gray-100">{selectedImage.negative_prompt}</p>
                  </div>
                )}
                <div className="space-y-2">
                  <div>
                    <span className="text-gray-400">Type:</span> {selectedImage.generation_type}
                  </div>
                  <div>
                    <span className="text-gray-400">Created:</span> {new Date(selectedImage.created_at).toLocaleString()}
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-400">Steps:</span> {selectedImage.steps}
                    </div>
                    <div>
                      <span className="text-gray-400">CFG Scale:</span> {selectedImage.cfg_scale}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-400">Sampler:</span> {selectedImage.parameters?.sampler || selectedImage.sampler}
                    </div>
                    <div>
                      <span className="text-gray-400">Scheduler:</span> {selectedImage.parameters?.schedule_type || 'uniform'}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <span className="text-gray-400">Size:</span> {selectedImage.width}x{selectedImage.height}
                    </div>
                    <div>
                      <span className="text-gray-400">Seed:</span> {selectedImage.seed}
                    </div>
                  </div>
                  {selectedImage.ancestral_seed && selectedImage.ancestral_seed !== -1 && (
                    <div>
                      <span className="text-gray-400">Ancestral Seed:</span> {selectedImage.ancestral_seed}
                    </div>
                  )}
                  {selectedImage.lora_names && (
                    <div>
                      <span className="text-gray-400">LoRA:</span> {selectedImage.lora_names}
                    </div>
                  )}
                  {selectedImage.model_name && (
                    <div>
                      <span className="text-gray-400">Model:</span> {selectedImage.model_name}
                    </div>
                  )}
                  {selectedImage.model_hash && (
                    <div>
                      <span className="text-gray-400">Model Hash:</span>{' '}
                      <span className="text-xs text-gray-100 font-mono" title={selectedImage.model_hash}>
                        {selectedImage.model_hash.substring(0, 16)}...
                      </span>
                    </div>
                  )}
                </div>

                {/* ControlNet Information */}
                {selectedImage.parameters?.controlnet_images && selectedImage.parameters.controlnet_images.length > 0 && (
                  <div className="border-t border-gray-700 pt-3">
                    <span className="text-gray-400 font-medium">ControlNet ({selectedImage.parameters.controlnet_images.length}):</span>
                    <div className="mt-2 space-y-3">
                      {selectedImage.parameters.controlnet_images.map((cn: any, index: number) => (
                        <div key={index} className="bg-gray-800 rounded p-2 text-xs space-y-1 break-words">
                          <div className="break-words">
                            <span className="text-gray-500">Model:</span>{' '}
                            <span className="text-gray-200 break-all">{cn.model_path}</span>
                          </div>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <span className="text-gray-500">Strength:</span> {cn.strength}
                            </div>
                            <div>
                              <span className="text-gray-500">LLLite:</span> {cn.is_lllite ? 'Yes' : 'No'}
                            </div>
                          </div>
                          {(cn.start_step !== 0 || cn.end_step !== 1000) && (
                            <div>
                              <span className="text-gray-500">Step Range:</span> {cn.start_step} - {cn.end_step}
                            </div>
                          )}
                          {cn.image && (
                            <div className="break-words">
                              <span className="text-gray-500">Image Hash:</span>{' '}
                              <button
                                onClick={() => handleSourceImageClick(cn.image)}
                                className="text-blue-400 hover:text-blue-300 font-mono underline break-all"
                                title={`Click to view image\n${cn.image}`}
                              >
                                {typeof cn.image === 'string' ? cn.image.substring(0, 16) : 'N/A'}...
                              </button>
                            </div>
                          )}
                          {cn.prompt && (
                            <div className="break-words">
                              <span className="text-gray-500">Prompt:</span>{' '}
                              <span className="text-gray-300">{cn.prompt}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {selectedImage.image_hash && (
                  <div>
                    <span className="text-gray-400">Image Hash: </span>
                    <span className="text-xs text-gray-100 font-mono" title={selectedImage.image_hash}>
                      {selectedImage.image_hash.substring(0, 16)}...
                    </span>
                  </div>
                )}
                {selectedImage.source_image_hash && (
                  <div>
                    <span className="text-gray-400">Source Image Hash: </span>
                    <button
                      onClick={() => handleSourceImageClick(selectedImage.source_image_hash!)}
                      className="text-xs text-blue-400 hover:text-blue-300 font-mono underline"
                      title={`Click to view source image\n${selectedImage.source_image_hash}`}
                    >
                      {selectedImage.source_image_hash.substring(0, 16)}...
                    </button>
                  </div>
                )}
              </div>
              </Card>
              </div>

              {/* Fixed bottom panel - Send controls */}
              <div className="border-t border-gray-700 bg-gray-900 p-4 space-y-3">
                <div className="space-y-2 text-sm">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendImage}
                      onChange={(e) => setSendImage(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send image</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendPrompt}
                      onChange={(e) => setSendPrompt(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send prompt</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={sendParameters}
                      onChange={(e) => setSendParameters(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-gray-300">Send parameters</span>
                  </label>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <Button
                    onClick={() => sendToTxt2Img(selectedImage)}
                    variant="secondary"
                    size="sm"
                    disabled={!sendPrompt && !sendParameters}
                    title="Send image not applicable for txt2img"
                  >
                    Send to txt2img
                  </Button>
                  <Button
                    onClick={() => sendToImg2Img(selectedImage)}
                    variant="secondary"
                    size="sm"
                    disabled={!sendImage && !sendPrompt && !sendParameters}
                  >
                    Send to img2img
                  </Button>
                  <Button
                    onClick={() => sendToInpaint(selectedImage)}
                    variant="secondary"
                    size="sm"
                    disabled={!sendImage && !sendPrompt && !sendParameters}
                  >
                    Send to inpaint
                  </Button>
                </div>
              </div>
            </div>

            {/* Right Area - Image Display with Navigation */}
            <div className="flex-1 flex items-center justify-center bg-gray-900 rounded-lg overflow-hidden relative">
              {/* Previous Image Button */}
              {(() => {
                const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage.filename);
                return currentIndex > 0 && (
                  <button
                    onClick={() => setSelectedImage(filteredImages[currentIndex - 1])}
                    className="absolute left-4 z-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white text-3xl w-12 h-12 rounded-full flex items-center justify-center transition-all"
                    title="Previous image (← key)"
                  >
                    ‹
                  </button>
                );
              })()}

              <img
                src={`/outputs/${selectedImage.filename}`}
                alt="Generated"
                className="max-w-full max-h-full object-contain cursor-pointer"
                onDoubleClick={() => setShowFullSizeImage(true)}
                title="Double-click to view full size"
              />

              {/* Next Image Button */}
              {(() => {
                const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage.filename);
                return currentIndex < filteredImages.length - 1 && (
                  <button
                    onClick={() => setSelectedImage(filteredImages[currentIndex + 1])}
                    className="absolute right-4 z-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white text-3xl w-12 h-12 rounded-full flex items-center justify-center transition-all"
                    title="Next image (→ key)"
                  >
                    ›
                  </button>
                );
              })()}
            </div>
          </div>

          {/* Full-size image popup */}
          {showFullSizeImage && (
            <div
              className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center p-4"
              onClick={() => setShowFullSizeImage(false)}
            >
              <div className="relative max-w-full max-h-full">
                <button
                  onClick={() => setShowFullSizeImage(false)}
                  className="absolute top-4 right-4 text-white text-2xl bg-black bg-opacity-50 rounded-full w-10 h-10 flex items-center justify-center hover:bg-opacity-75"
                >
                  ×
                </button>
                <img
                  src={`/outputs/${selectedImage.filename}`}
                  alt="Generated - Full Size"
                  className="max-w-full max-h-[90vh] object-contain"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex gap-4">
          {/* Left Sidebar - Filters */}
          <GalleryFilter
            filterTxt2Img={filterTxt2Img}
            setFilterTxt2Img={setFilterTxt2Img}
            filterImg2Img={filterImg2Img}
            setFilterImg2Img={setFilterImg2Img}
            filterInpaint={filterInpaint}
            setFilterInpaint={setFilterInpaint}
            dateFrom={dateFrom}
            setDateFrom={setDateFrom}
            dateTo={dateTo}
            setDateTo={setDateTo}
            widthRange={widthRange}
            setWidthRange={setWidthRange}
            heightRange={heightRange}
            setHeightRange={setHeightRange}
            setCommittedWidthRange={setCommittedWidthRange}
            setCommittedHeightRange={setCommittedHeightRange}
            tagSearchInput={tagSearchInput}
            setTagSearchInput={setTagSearchInput}
            tagSearchCommitted={tagSearchCommitted}
            setTagSearchCommitted={setTagSearchCommitted}
            searchInNegative={searchInNegative}
            setSearchInNegative={setSearchInNegative}
            showSuggestions={showSuggestions}
            setShowSuggestions={setShowSuggestions}
            selectedSuggestionIndex={selectedSuggestionIndex}
            setSelectedSuggestionIndex={setSelectedSuggestionIndex}
            excludeRareTags={excludeRareTags}
            setExcludeRareTags={setExcludeRareTags}
            tagSuggestions={tagSuggestions}
            handleTagSearchSubmit={handleTagSearchSubmit}
            handleTagSearchKeyDown={handleTagSearchKeyDown}
            handleSuggestionClick={handleSuggestionClick}
            removeTag={removeTag}
            clearAllTags={clearAllTags}
            gridColumns={gridColumns}
            setGridColumns={setGridColumns}
            currentPage={currentPage}
            setCurrentPage={setCurrentPage}
            totalImages={totalImages}
            imagesPerPage={imagesPerPage}
            loading={loading}
          />

          {/* Right Area - Image Grid */}
          <ImageList
            images={filteredImages}
            gridColumns={gridColumns}
            onImageClick={setSelectedImage}
            loading={loading}
          />
        </div>
      )}
    </div>
  );
}
