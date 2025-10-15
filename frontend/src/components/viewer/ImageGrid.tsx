"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getImages, GeneratedImage, ImageFilters } from "@/utils/api";
import Card from "../common/Card";
import Button from "../common/Button";
import RangeSlider from "../common/RangeSlider";
import Slider from "../common/Slider";

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

  // UI states
  const [gridColumns, setGridColumns] = useState(6);

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

  // Keyboard navigation for pagination
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      if (e.key === 'ArrowLeft' && currentPage > 1 && !loading) {
        e.preventDefault();
        setCurrentPage(currentPage - 1);
      } else if (e.key === 'ArrowRight' && currentPage * imagesPerPage < totalImages && !loading) {
        e.preventDefault();
        setCurrentPage(currentPage + 1);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentPage, totalImages, loading, imagesPerPage]);

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

  // Extract unique tags from all prompts for autocomplete
  const getTagSuggestions = (): string[] => {
    if (!tagSearchInput || tagSearchInput.length < 2) return [];

    const searchLower = tagSearchInput.toLowerCase();
    const tagSet = new Set<string>();

    images.forEach((image) => {
      const promptText = searchInNegative ? image.negative_prompt : image.prompt;
      if (!promptText) return;

      // Split by common delimiters (comma, space, etc.)
      const tags = promptText.split(/[,\n]+/).map(t => t.trim()).filter(t => t.length > 0);

      tags.forEach(tag => {
        if (tag.toLowerCase().includes(searchLower)) {
          tagSet.add(tag);
        }
      });
    });

    return Array.from(tagSet).slice(0, 10); // Limit to 10 suggestions
  };

  const tagSuggestions = getTagSuggestions();

  // Client-side tag filtering (only apply committed search) - AND search
  const filteredImages = images.filter((image) => {
    if (tagSearchCommitted.length === 0) return true;

    const searchField = searchInNegative ? image.negative_prompt : image.prompt;
    if (!searchField) return false;

    const searchFieldLower = searchField.toLowerCase();

    // AND search: all tags must be present
    return tagSearchCommitted.every(tag =>
      searchFieldLower.includes(tag.toLowerCase())
    );
  });

  const handleTagSearchSubmit = () => {
    if (tagSearchInput.trim() && !tagSearchCommitted.includes(tagSearchInput.trim())) {
      setTagSearchCommitted([...tagSearchCommitted, tagSearchInput.trim()]);
      setTagSearchInput("");
    }
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  };

  const removeTag = (tagToRemove: string) => {
    setTagSearchCommitted(tagSearchCommitted.filter(tag => tag !== tagToRemove));
  };

  const clearAllTags = () => {
    setTagSearchCommitted([]);
    setTagSearchInput("");
  };

  const handleTagSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
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
  };

  const handleSuggestionClick = (suggestion: string) => {
    if (!tagSearchCommitted.includes(suggestion)) {
      setTagSearchCommitted([...tagSearchCommitted, suggestion]);
    }
    setTagSearchInput("");
    setShowSuggestions(false);
    setSelectedSuggestionIndex(-1);
  };

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

  if (loading) {
    return <div className="text-center py-8">Loading images...</div>;
  }

  return (
    <div>
      {selectedImage ? (
        <div className="space-y-4">
          <button
            onClick={() => setSelectedImage(null)}
            className="text-blue-400 hover:text-blue-300"
          >
            ← Back to gallery
          </button>
          <Card title="Image Details">
            <div className="space-y-4">
              <img
                src={`/outputs/${selectedImage.filename}`}
                alt="Generated"
                className="w-full rounded-lg"
              />
              <div className="space-y-2 text-sm">
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
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <span className="text-gray-400">Type:</span> {selectedImage.generation_type}
                  </div>
                  <div>
                    <span className="text-gray-400">Created:</span> {new Date(selectedImage.created_at).toLocaleString()}
                  </div>
                  <div>
                    <span className="text-gray-400">Steps:</span> {selectedImage.steps}
                  </div>
                  <div>
                    <span className="text-gray-400">CFG Scale:</span> {selectedImage.cfg_scale}
                  </div>
                  <div>
                    <span className="text-gray-400">Sampler:</span> {selectedImage.sampler}
                  </div>
                  <div>
                    <span className="text-gray-400">Seed:</span> {selectedImage.seed}
                  </div>
                  <div>
                    <span className="text-gray-400">Size:</span> {selectedImage.width}x{selectedImage.height}
                  </div>
                  {selectedImage.lora_names && (
                    <div className="col-span-2">
                      <span className="text-gray-400">LoRA:</span> {selectedImage.lora_names}
                    </div>
                  )}
                </div>
                {selectedImage.image_hash && (
                  <div>
                    <span className="text-gray-400">Image Hash: </span>
                    <span className="text-xs text-gray-100 font-mono break-all">{selectedImage.image_hash}</span>
                  </div>
                )}
                {selectedImage.source_image_hash && (
                  <div>
                    <span className="text-gray-400">Source Image Hash: </span>
                    <button
                      onClick={() => handleSourceImageClick(selectedImage.source_image_hash!)}
                      className="text-xs text-blue-400 hover:text-blue-300 font-mono break-all underline"
                      title="Click to view source image"
                    >
                      {selectedImage.source_image_hash}
                    </button>
                  </div>
                )}
              </div>
              <div className="space-y-3 mt-4">
                <div className="flex flex-wrap gap-2 text-sm">
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
          </Card>
        </div>
      ) : (
        <div className="flex gap-4">
          {/* Left Sidebar - Filters */}
          <div className="w-80 flex-shrink-0">
            <Card title="Filters" defaultCollapsed={false} storageKey="gallery_filters_collapsed">
              <div className="space-y-3">
                {/* Generation Type Filter */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Generation Type</label>
                  <div className="space-y-1">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={filterTxt2Img}
                        onChange={(e) => setFilterTxt2Img(e.target.checked)}
                        className="rounded"
                      />
                      <span className="text-sm text-gray-300">txt2img</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={filterImg2Img}
                        onChange={(e) => setFilterImg2Img(e.target.checked)}
                        className="rounded"
                      />
                      <span className="text-sm text-gray-300">img2img</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={filterInpaint}
                        onChange={(e) => setFilterInpaint(e.target.checked)}
                        className="rounded"
                      />
                      <span className="text-sm text-gray-300">inpaint</span>
                    </label>
                  </div>
                </div>

                {/* Tag Search Filter */}
                <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-gray-300">Tag Search</label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={searchInNegative}
                      onChange={(e) => setSearchInNegative(e.target.checked)}
                      className="rounded"
                    />
                    <span className="text-xs text-gray-400">Search in negative prompt</span>
                  </label>
                </div>
                <div className="relative">
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={tagSearchInput}
                      onChange={(e) => {
                        setTagSearchInput(e.target.value);
                        setShowSuggestions(e.target.value.length >= 2);
                        setSelectedSuggestionIndex(-1);
                      }}
                      onKeyDown={handleTagSearchKeyDown}
                      onFocus={() => setShowSuggestions(tagSearchInput.length >= 2)}
                      onBlur={() => setTimeout(() => {
                        setShowSuggestions(false);
                        setSelectedSuggestionIndex(-1);
                      }, 200)}
                      placeholder="Enter tag (press Enter to search)"
                      className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                    />
                    <Button
                      onClick={handleTagSearchSubmit}
                      variant="primary"
                      size="sm"
                    >
                      Search
                    </Button>
                  </div>

                  {/* Autocomplete suggestions */}
                  {showSuggestions && tagSuggestions.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {tagSuggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => handleSuggestionClick(suggestion)}
                          className={`w-full px-3 py-2 text-left text-sm text-gray-100 hover:bg-gray-700 focus:outline-none ${
                            index === selectedSuggestionIndex ? 'bg-gray-700' : ''
                          }`}
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                {tagSearchCommitted.length > 0 && (
                  <div className="mt-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">Searching for (AND):</span>
                      <button
                        onClick={clearAllTags}
                        className="text-xs text-red-400 hover:text-red-300"
                      >
                        Clear All
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {tagSearchCommitted.map((tag, index) => (
                        <div
                          key={index}
                          className="inline-flex items-center gap-1 px-2 py-1 bg-blue-600 text-white text-xs rounded border border-blue-500"
                        >
                          <span>{tag}</span>
                          <button
                            onClick={() => removeTag(tag)}
                            className="hover:text-red-300 focus:outline-none"
                            title="Remove tag"
                          >
                            ×
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Date Range Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Date Range</label>
                <div className="space-y-2">
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    placeholder="From"
                    className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-xs cursor-pointer hover:border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-60 [&::-webkit-calendar-picker-indicator]:hover:opacity-100"
                  />
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    placeholder="To"
                    className="w-full px-2 py-1.5 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 text-xs cursor-pointer hover:border-gray-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 [&::-webkit-calendar-picker-indicator]:cursor-pointer [&::-webkit-calendar-picker-indicator]:opacity-60 [&::-webkit-calendar-picker-indicator]:hover:opacity-100"
                  />
                </div>
              </div>

              {/* Size Range Filters */}
              <div>
                <RangeSlider
                  label="Width Range"
                  min={0}
                  max={2048}
                  step={64}
                  value={widthRange}
                  onChange={setWidthRange}
                  onCommit={setCommittedWidthRange}
                />
              </div>

              <div>
                <RangeSlider
                  label="Height Range"
                  min={0}
                  max={2048}
                  step={64}
                  value={heightRange}
                  onChange={setHeightRange}
                  onCommit={setCommittedHeightRange}
                />
              </div>

              {/* Grid Layout Control */}
              <div>
                <Slider
                  label="Images per Row"
                  min={2}
                  max={8}
                  step={1}
                  value={gridColumns}
                  onChange={(e) => setGridColumns(Number(e.target.value))}
                />
              </div>

              {/* Pagination */}
              <div className="border-t border-gray-700 pt-3 mt-3">
                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                  <span>
                    {totalImages > 0 ? (
                      <>
                        {(currentPage - 1) * imagesPerPage + 1}-
                        {Math.min(currentPage * imagesPerPage, totalImages)} of {totalImages}
                      </>
                    ) : (
                      "No images"
                    )}
                  </span>
                </div>
                <div className="flex gap-2">
                  <Button
                    onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                    disabled={currentPage === 1 || loading}
                    variant="secondary"
                    size="sm"
                    className="flex-1"
                  >
                    Previous
                  </Button>
                  <Button
                    onClick={() => setCurrentPage(currentPage + 1)}
                    disabled={currentPage * imagesPerPage >= totalImages || loading}
                    variant="secondary"
                    size="sm"
                    className="flex-1"
                  >
                    Next
                  </Button>
                </div>
                <div className="text-center mt-2 text-xs text-gray-500">
                  Page {currentPage} of {Math.max(1, Math.ceil(totalImages / imagesPerPage))}
                </div>
              </div>
            </div>
          </Card>
          </div>

          {/* Right Area - Image Grid */}
          <div className="flex-1">
          <div
            className="grid gap-4"
            style={{
              gridTemplateColumns: `repeat(${gridColumns}, minmax(0, 1fr))`
            }}
          >
            {filteredImages.map((image) => (
              <div
                key={image.id}
                onClick={() => setSelectedImage(image)}
                className="cursor-pointer group"
              >
                <div className="aspect-square bg-gray-800 rounded-lg overflow-hidden">
                  <img
                    src={`/thumbnails/${image.filename}`}
                    alt={image.prompt}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                  />
                </div>
                <p className="mt-2 text-xs text-gray-400 truncate">{image.prompt}</p>
              </div>
            ))}
          </div>
          </div>
        </div>
      )}
    </div>
  );
}
