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
import { SlidersHorizontal, X, Info, ArrowLeft, Download, Maximize } from "lucide-react";
import { getImages, GeneratedImage, ImageFilters } from "@/utils/api";
import Card from "../common/Card";
import Button from "../common/Button";
import GalleryFilter from "./GalleryFilter";
import ImageList from "./ImageList";
import { saveTempImage } from "@/utils/tempImageStorage";

export default function ImageGrid() {
  const router = useRouter();
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  const [sendImage, setSendImage] = useState(true);
  const [sendPrompt, setSendPrompt] = useState(true);
  const [sendParameters, setSendParameters] = useState(true);
  const [isFilterOpen, setIsFilterOpen] = useState(false);

  // Swipe gesture detection
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

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
  const [isDetailOpen, setIsDetailOpen] = useState(false);

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

  const handleDownload = async (image: GeneratedImage) => {
    try {
      // Get metadata setting from localStorage
      const includeMetadata = localStorage.getItem('include_metadata_in_downloads') === 'true';

      // Use API endpoint for metadata-aware download
      const downloadUrl = `/api/download/${image.filename}?include_metadata=${includeMetadata}`;

      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = image.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
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

  const sendToImg2Img = async (image: GeneratedImage) => {
    // Send image if checked
    if (sendImage) {
      try {
        // Load image from /outputs/ and save to tempStorage
        const imageUrl = `/outputs/${image.filename}`;
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const reader = new FileReader();

        await new Promise((resolve, reject) => {
          reader.onloadend = async () => {
            try {
              const base64data = reader.result as string;
              const tempRef = await saveTempImage(base64data);
              localStorage.setItem("img2img_input_image", tempRef);
              window.dispatchEvent(new Event("img2img_input_updated"));
              resolve(null);
            } catch (error) {
              reject(error);
            }
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.error("[ImageGrid] Failed to send image to img2img:", error);
      }
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

  const sendToInpaint = async (image: GeneratedImage) => {
    // Send image if checked
    if (sendImage) {
      try {
        // Load image from /outputs/ and save to tempStorage
        const imageUrl = `/outputs/${image.filename}`;
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        const reader = new FileReader();

        await new Promise((resolve, reject) => {
          reader.onloadend = async () => {
            try {
              const base64data = reader.result as string;
              const tempRef = await saveTempImage(base64data);
              localStorage.setItem("inpaint_input_image", tempRef);
              localStorage.removeItem("inpaint_mask_image");
              window.dispatchEvent(new Event("inpaint_input_updated"));
              resolve(null);
            } catch (error) {
              reject(error);
            }
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.error("[ImageGrid] Failed to send image to inpaint:", error);
      }
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

  // Swipe gesture handlers for gallery pagination
  const minSwipeDistance = 50; // Minimum distance for a swipe

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null);
    setTouchStart(e.targetTouches[0].clientX);
  };

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX);
  };

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return;

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    // Only handle swipe if not viewing a specific image and not loading
    if (!selectedImage && !loading) {
      if (isLeftSwipe && currentPage * imagesPerPage < totalImages) {
        // Swipe left = next page
        setCurrentPage(currentPage + 1);
      } else if (isRightSwipe && currentPage > 1) {
        // Swipe right = previous page
        setCurrentPage(currentPage - 1);
      }
    }
  };

  // Image detail view swipe handlers
  const handleDetailImageTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setTouchStart(e.touches[0].clientX);
      setTouchEnd(null);
    }
  };

  const handleDetailImageTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 1) {
      setTouchEnd(e.touches[0].clientX);
    }
  };

  const handleDetailImageTouchEnd = () => {
    if (touchStart !== null && touchEnd !== null) {
      const distance = touchStart - touchEnd;
      const isLeftSwipe = distance > minSwipeDistance;
      const isRightSwipe = distance < -minSwipeDistance;

      const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage?.filename);

      if (isLeftSwipe && currentIndex < filteredImages.length - 1) {
        setSelectedImage(filteredImages[currentIndex + 1]);
      } else if (isRightSwipe && currentIndex > 0) {
        setSelectedImage(filteredImages[currentIndex - 1]);
      }
    }
    setTouchStart(null);
    setTouchEnd(null);
  };

  return (
    <div>
      {selectedImage ? (
        <div className="fixed inset-0 lg:relative bg-gray-950 lg:bg-transparent z-30 lg:z-auto">
          {/* Back button - Desktop */}
          <button
            onClick={() => {
              setSelectedImage(null);
              setIsDetailOpen(false);
            }}
            className="hidden lg:flex items-center gap-2 text-blue-400 hover:text-blue-300 mb-4"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>Back to gallery</span>
          </button>

          {/* Back button - Mobile */}
          <button
            onClick={() => {
              setSelectedImage(null);
              setIsDetailOpen(false);
            }}
            className="fixed top-20 left-4 z-50 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg lg:hidden"
            aria-label="Back to gallery"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>

          {/* Mobile: Detail info toggle button */}
          <button
            onClick={() => setIsDetailOpen(!isDetailOpen)}
            className="fixed top-4 right-4 z-50 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg lg:hidden"
            aria-label="Toggle detail info"
          >
            {isDetailOpen ? <X className="h-5 w-5" /> : <Info className="h-5 w-5" />}
          </button>

          {/* Mobile detail overlay */}
          {isDetailOpen && (
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
              onClick={() => setIsDetailOpen(false)}
            />
          )}

          <div className="flex flex-col lg:flex-row gap-4 h-screen lg:h-[calc(100vh-12rem)] lg:p-4">
            {/* Left Sidebar - Details (Desktop always visible, Mobile toggleable) */}
            <div className={`
              fixed lg:relative top-0 left-0 h-full lg:h-auto w-80 max-w-[calc(100vw-5rem)] lg:max-w-none z-50 lg:z-auto
              transform transition-transform duration-200 ease-in-out
              ${isDetailOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
              bg-gray-900 lg:bg-transparent
              p-4 lg:p-0 pt-20 lg:pt-0
              flex-shrink-0 lg:flex lg:flex-col
            `}>
              {/* Scrollable content area */}
              <div className="lg:flex-1 lg:overflow-y-auto lg:mb-4">
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

                {/* img2img/Inpaint Parameters */}
                {(selectedImage.generation_type === 'img2img' || selectedImage.generation_type === 'inpaint') && (
                  <div className="border-t border-gray-700 pt-3">
                    <span className="text-gray-400 font-medium">{selectedImage.generation_type === 'inpaint' ? 'Inpaint' : 'img2img'} Parameters:</span>
                    <div className="mt-2 space-y-2 text-xs">
                      {selectedImage.parameters?.denoising_strength !== undefined && (
                        <div>
                          <span className="text-gray-500">Denoising Strength:</span> {selectedImage.parameters.denoising_strength}
                        </div>
                      )}
                      {selectedImage.parameters?.img2img_fix_steps !== undefined && (
                        <div>
                          <span className="text-gray-500">Fix Steps:</span> {selectedImage.parameters.img2img_fix_steps ? 'Yes' : 'No'}
                        </div>
                      )}
                      {selectedImage.generation_type === 'inpaint' && (
                        <>
                          {selectedImage.parameters?.mask_blur !== undefined && (
                            <div>
                              <span className="text-gray-500">Mask Blur:</span> {selectedImage.parameters.mask_blur}
                            </div>
                          )}
                          {/* Note: Inpaint Full Res is not implemented in backend
                          {selectedImage.parameters?.inpaint_full_res !== undefined && (
                            <div>
                              <span className="text-gray-500">Inpaint Full Res:</span> {selectedImage.parameters.inpaint_full_res ? 'Yes' : 'No'}
                            </div>
                          )}
                          {selectedImage.parameters?.inpaint_full_res_padding !== undefined && selectedImage.parameters.inpaint_full_res && (
                            <div>
                              <span className="text-gray-500">Full Res Padding:</span> {selectedImage.parameters.inpaint_full_res_padding}
                            </div>
                          )}
                          */}
                          {selectedImage.parameters?.inpaint_fill_mode !== undefined && (
                            <div>
                              <span className="text-gray-500">Fill Mode:</span> {selectedImage.parameters.inpaint_fill_mode}
                            </div>
                          )}
                          {selectedImage.parameters?.inpaint_fill_strength !== undefined && selectedImage.parameters.inpaint_fill_mode !== 'original' && (
                            <div>
                              <span className="text-gray-500">Fill Strength:</span> {selectedImage.parameters.inpaint_fill_strength}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}

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

              {/* Fixed Send Options and Buttons - Desktop only */}
              <div className="hidden lg:block lg:flex-shrink-0">
                <Card>
                  <div className="space-y-3">
                    {/* Send options in one line */}
                    <div className="flex items-center gap-3 text-sm">
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={sendImage}
                          onChange={(e) => setSendImage(e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-gray-300">Image</span>
                      </label>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={sendPrompt}
                          onChange={(e) => setSendPrompt(e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-gray-300">Prompt</span>
                      </label>
                      <label className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={sendParameters}
                          onChange={(e) => setSendParameters(e.target.checked)}
                          className="rounded"
                        />
                        <span className="text-gray-300">Params</span>
                      </label>
                    </div>

                    {/* Download and Send buttons */}
                    <div className="flex flex-col gap-2">
                      <Button
                        onClick={() => handleDownload(selectedImage)}
                        variant="primary"
                        size="sm"
                        className="flex items-center justify-center"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download
                      </Button>
                      <div className="grid grid-cols-3 gap-2">
                        <Button
                          onClick={() => sendToTxt2Img(selectedImage)}
                          variant="secondary"
                          size="sm"
                          disabled={!sendPrompt && !sendParameters}
                          title="Send image not applicable for txt2img"
                        >
                          txt2img
                        </Button>
                        <Button
                          onClick={() => sendToImg2Img(selectedImage)}
                          variant="secondary"
                          size="sm"
                          disabled={!sendImage && !sendPrompt && !sendParameters}
                        >
                          img2img
                        </Button>
                        <Button
                          onClick={() => sendToInpaint(selectedImage)}
                          variant="secondary"
                          size="sm"
                          disabled={!sendImage && !sendPrompt && !sendParameters}
                        >
                          inpaint
                        </Button>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Right Area - Image Display with Navigation */}
            <div className="flex-1 flex items-center justify-center bg-gray-900 rounded-lg overflow-hidden relative touch-none">
              {/* Fullscreen Button - Mobile only */}
              <button
                onClick={() => setShowFullSizeImage(true)}
                className="lg:hidden absolute top-4 right-4 z-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white w-10 h-10 rounded-full flex items-center justify-center transition-all"
                title="View fullscreen"
              >
                <Maximize className="h-5 w-5" />
              </button>

              {/* Previous Image Button - Desktop only */}
              {(() => {
                const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage.filename);
                return currentIndex > 0 && (
                  <button
                    onClick={() => setSelectedImage(filteredImages[currentIndex - 1])}
                    className="hidden lg:flex absolute left-4 z-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white text-3xl w-12 h-12 rounded-full items-center justify-center transition-all"
                    title="Previous image (← key)"
                  >
                    ‹
                  </button>
                );
              })()}

              <div
                className="w-full h-full flex items-center justify-center overflow-hidden"
                onTouchStart={handleDetailImageTouchStart}
                onTouchMove={handleDetailImageTouchMove}
                onTouchEnd={handleDetailImageTouchEnd}
              >
                <img
                  src={`/outputs/${selectedImage.filename}`}
                  alt="Generated"
                  className="max-w-full max-h-full object-contain cursor-pointer"
                  onDoubleClick={() => setShowFullSizeImage(true)}
                  title="Double-click to view full size"
                />
              </div>

              {/* Next Image Button - Desktop only */}
              {(() => {
                const currentIndex = filteredImages.findIndex(img => img.filename === selectedImage.filename);
                return currentIndex < filteredImages.length - 1 && (
                  <button
                    onClick={() => setSelectedImage(filteredImages[currentIndex + 1])}
                    className="hidden lg:flex absolute right-4 z-10 bg-black bg-opacity-50 hover:bg-opacity-75 text-white text-3xl w-12 h-12 rounded-full items-center justify-center transition-all"
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
                {/* Download button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownload(selectedImage);
                  }}
                  className="absolute top-4 right-20 text-white bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
                  title="Download"
                >
                  <Download className="h-6 w-6" />
                </button>
                {/* Close button */}
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

          {/* Mobile: Fixed bottom Send Options and Buttons */}
          <div className="fixed bottom-4 portrait:left-1/2 portrait:-translate-x-1/2 landscape:right-4 landscape:translate-x-0 z-30 lg:hidden flex flex-col gap-2 bg-gray-900 bg-opacity-95 p-3 rounded-lg shadow-lg">
            {/* Send options checkboxes */}
            <div className="flex items-center gap-3 text-xs text-white">
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sendImage}
                  onChange={(e) => setSendImage(e.target.checked)}
                  className="rounded"
                />
                <span>Image</span>
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sendPrompt}
                  onChange={(e) => setSendPrompt(e.target.checked)}
                  className="rounded"
                />
                <span>Prompt</span>
              </label>
              <label className="flex items-center gap-1 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sendParameters}
                  onChange={(e) => setSendParameters(e.target.checked)}
                  className="rounded"
                />
                <span>Params</span>
              </label>
            </div>

            {/* Buttons */}
            <div className="flex gap-2">
              <button
                onClick={() => handleDownload(selectedImage)}
                className="px-3 py-2 text-sm bg-blue-600 hover:bg-blue-700 text-white rounded"
                title="Download"
              >
                <Download className="h-4 w-4" />
              </button>
              <button
                onClick={() => sendToTxt2Img(selectedImage)}
                disabled={!sendPrompt && !sendParameters}
                className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Send to txt2img"
              >
                txt2img
              </button>
              <button
                onClick={() => sendToImg2Img(selectedImage)}
                disabled={!sendImage && !sendPrompt && !sendParameters}
                className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Send to img2img"
              >
                img2img
              </button>
              <button
                onClick={() => sendToInpaint(selectedImage)}
                disabled={!sendImage && !sendPrompt && !sendParameters}
                className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Send to inpaint"
              >
                inpaint
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="relative">
          {/* Mobile filter toggle button */}
          <button
            onClick={() => setIsFilterOpen(!isFilterOpen)}
            className="fixed bottom-4 right-4 z-40 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg lg:hidden"
            aria-label="Toggle filters"
          >
            {isFilterOpen ? <X className="h-5 w-5" /> : <SlidersHorizontal className="h-5 w-5" />}
          </button>

          {/* Overlay for mobile filter panel */}
          {isFilterOpen && (
            <div
              className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
              onClick={() => setIsFilterOpen(false)}
            />
          )}

          <div className="flex gap-4">
          {/* Left Sidebar - Filters */}
          <div className={`
            fixed lg:relative top-0 left-0 h-full lg:h-auto w-80 max-w-[calc(100vw-5rem)] lg:max-w-none z-40 lg:z-auto
            transform transition-transform duration-200 ease-in-out
            ${isFilterOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
            bg-gray-900 lg:bg-transparent
            overflow-y-auto lg:overflow-visible
            p-4 lg:p-0 pt-20 lg:pt-0
          `}>
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
          </div>

          {/* Right Area - Image Grid */}
          <div
            className="flex-1 w-full lg:w-auto"
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={onTouchEnd}
          >
          <ImageList
            images={filteredImages}
            gridColumns={gridColumns}
            onImageClick={setSelectedImage}
            loading={loading}
          />
          </div>
        </div>
        </div>
      )}
    </div>
  );
}
