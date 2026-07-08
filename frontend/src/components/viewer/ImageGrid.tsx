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

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { useRouter } from "next/navigation";
import { SlidersHorizontal, X, Info, ArrowLeft, Download, Maximize, Camera } from "lucide-react";
import { getImages, GeneratedImage, ImageFilters } from "@/utils/api";
import Card from "../common/Card";
import Button from "../common/Button";
import GalleryFilter from "./GalleryFilter";
import ImageList from "./ImageList";
import { saveTempImage } from "@/utils/tempImageStorage";
import { sendBase64ImageToImg2Img, sendBase64ImageToImg2Vid, sendImageToImg2Vid } from "@/utils/sendHelpers";
import PostEditControls from "../common/PostEditControls";
import { PostEditState, NEUTRAL_POST_EDIT, isNeutral, applyPostEdit, buildFilterString, editedFilename } from "@/utils/postEdit";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";

export default function ImageGrid() {
  const router = useRouter();
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);
  // Client-side post-edit (brightness/saturation) for the currently selected
  // gallery image. Never sent to the backend; reset to neutral when the
  // selected image changes. Output-folder files are never modified.
  const [postEdit, setPostEdit] = useState<PostEditState>({ ...NEUTRAL_POST_EDIT });
  useEffect(() => {
    setPostEdit({ ...NEUTRAL_POST_EDIT });
  }, [selectedImage?.filename]);
  // Sidebar "Post-edit" section collapse state, persisted across sessions.
  const [postEditCollapsed, setPostEditCollapsed] = useState(true);
  useEffect(() => {
    setPostEditCollapsed(localStorage.getItem("gallery_postedit_collapsed") !== "false");
  }, []);
  const togglePostEditCollapsed = () => {
    setPostEditCollapsed((prev) => {
      localStorage.setItem("gallery_postedit_collapsed", String(!prev));
      return !prev;
    });
  };
  // Color-flatten preview for the selected image (detail + full-size popup).
  // brightness/saturation remain a CSS filter layered on top (below).
  const selectedImageSrc = selectedImage ? `/outputs/${selectedImage.filename}` : undefined;
  const effectiveSelectedSrc = usePostEditPreview(selectedImageSrc, postEdit.flatten);
  // Ref to the <video> element in the video detail view, used for frame-grab.
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Whether the selected item is a video (mp4/webm or is_video flag / video type).
  const isSelectedVideo = !!selectedImage && (
    selectedImage.is_video === true ||
    /\.(mp4|webm)$/i.test(selectedImage.filename) ||
    selectedImage.generation_type === "txt2vid" ||
    selectedImage.generation_type === "img2vid"
  );
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
  const [filterTxt2Vid, setFilterTxt2Vid] = useState(true);
  const [filterImg2Vid, setFilterImg2Vid] = useState(true);
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
  // Collapsed by default so the post-edit strip never covers the enlarged
  // image; purely internal UI state for the full-size popup below.
  const [postEditBarExpanded, setPostEditBarExpanded] = useState(false);
  const [isDetailOpen, setIsDetailOpen] = useState(false);

  // Pagination states
  const [currentPage, setCurrentPage] = useState(1);
  const [totalImages, setTotalImages] = useState(0);
  const imagesPerPage = 100;

  const loadImages = useCallback(async () => {
    try {
      setLoading(true);

      // Build generation types filter
      const types: string[] = [];
      if (filterTxt2Img) types.push("txt2img");
      if (filterImg2Img) types.push("img2img");
      if (filterInpaint) types.push("inpaint");
      if (filterTxt2Vid) types.push("txt2vid");
      if (filterImg2Vid) types.push("img2vid");

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
  }, [currentPage, filterTxt2Img, filterImg2Img, filterInpaint, filterTxt2Vid, filterImg2Vid, dateFrom, dateTo, committedWidthRange, committedHeightRange]);

  // Reset to page 1 when filters change, then load images
  useEffect(() => {
    setCurrentPage(1);
  }, [filterTxt2Img, filterImg2Img, filterInpaint, filterTxt2Vid, filterImg2Vid, dateFrom, dateTo, committedWidthRange, committedHeightRange]);

  // Load images when filters or page change
  useEffect(() => {
    loadImages();
  }, [loadImages]);

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

      let blob = await response.blob();
      let downloadName = image.filename;

      // Bake post-edit adjustments only when non-neutral. Neutral -> original
      // blob unchanged (metadata preserved). Baking re-encodes the PNG and
      // loses embedded metadata (see postEdit.ts).
      if (!isNeutral(postEdit)) {
        blob = await applyPostEdit(blob, postEdit);
        downloadName = editedFilename(image.filename);
      }

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    }
  };

  const handleFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen().catch((err) => {
        console.error('Failed to enter fullscreen:', err);
        alert('Fullscreen mode is not supported on this device.');
      });
    }
  };

  // Carry acceleration / determinism-affecting settings from a stored image
  // into a target params object for reproducible re-generation. These all flow
  // through the generation request today, so restoring them reproduces output.
  const applyAccelParams = (target: any, image: GeneratedImage) => {
    const p = image.parameters || {};
    if (p.spectrum_enable) {
      target.spectrum_enable = true;
      const keys = [
        "spectrum_w", "spectrum_w_decay", "spectrum_delta_cap", "spectrum_m", "spectrum_lam", "spectrum_warmup_steps",
        "spectrum_window_size", "spectrum_flex_window", "spectrum_tail",
        "spectrum_feature_mode", "spectrum_cache_branch", "spectrum_max_cache",
      ];
      keys.forEach((k) => { if (p[k] !== undefined) target[k] = p[k]; });
    } else {
      target.spectrum_enable = false;
    }
    if (p.fbcache_enable) {
      target.fbcache_enable = true;
      ["fbcache_threshold", "fbcache_warmup_steps"].forEach((k) => {
        if (p[k] !== undefined) target[k] = p[k];
      });
    } else {
      target.fbcache_enable = false;
    }
    if (p.prompt_chunking_mode !== undefined) target.prompt_chunking_mode = p.prompt_chunking_mode;
    if (p.max_prompt_chunks !== undefined) target.max_prompt_chunks = p.max_prompt_chunks;
    if (p.text_encoder_quantization !== undefined) target.text_encoder_quantization = p.text_encoder_quantization;
    if (p.nag_negative_prompt !== undefined) target.nag_negative_prompt = p.nag_negative_prompt;
    if (p.original_size_w !== undefined) target.original_size_w = p.original_size_w;
    if (p.original_size_h !== undefined) target.original_size_h = p.original_size_h;
    if (p.original_size_scale !== undefined) target.original_size_scale = p.original_size_scale;
    if (p.use_tipo !== undefined) target.use_tipo = p.use_tipo;
    if (p.tipo_config !== undefined) target.tipo_config = p.tipo_config;
    if (p.color_flatten_strength !== undefined) target.color_flatten_strength = p.color_flatten_strength;
    if (p.vae_drift_correction !== undefined) target.vae_drift_correction = p.vae_drift_correction;
    if (p.flatten_in_loop !== undefined) target.flatten_in_loop = p.flatten_in_loop;
    if (p.flatten_in_loop_last_steps !== undefined) target.flatten_in_loop_last_steps = p.flatten_in_loop_last_steps;
    if (p.flatten_in_loop_min_region !== undefined) target.flatten_in_loop_min_region = p.flatten_in_loop_min_region;
    // attention_type / attention_impl are read from localStorage by the API layer,
    // not from the params object, so restore them there for reproducibility.
    if (typeof window !== 'undefined') {
      if (p.attention_type !== undefined) localStorage.setItem('attention_type', p.attention_type);
      if (p.attention_impl !== undefined) localStorage.setItem('attention_impl', p.attention_impl);
    }
  };

  const sendToTxt2Img = (image: GeneratedImage) => {
    // Note: Send image is not applicable for txt2img (no input image)

    // Build params object by merging prompt and parameters
    const txt2imgParams = JSON.parse(localStorage.getItem("txt2img_params") || "{}");

    // Send prompt if checked
    if (sendPrompt) {
      txt2imgParams.prompt = image.prompt;
      txt2imgParams.negative_prompt = image.negative_prompt;
    }

    // Send parameters if checked
    if (sendParameters) {
      txt2imgParams.steps = image.steps;
      txt2imgParams.cfg_scale = image.cfg_scale;
      txt2imgParams.sampler = image.parameters?.sampler || "euler";
      txt2imgParams.schedule_type = image.parameters?.schedule_type || "uniform";
      txt2imgParams.seed = image.seed;
      txt2imgParams.width = image.width;
      txt2imgParams.height = image.height;

      // Add Advanced CFG parameters (always load, even if constant)
      if (image.cfg_schedule_type) {
        txt2imgParams.cfg_schedule_type = image.cfg_schedule_type;
      }
      if (image.cfg_schedule_min) {
        txt2imgParams.cfg_schedule_min = parseFloat(image.cfg_schedule_min);
      }
      if (image.cfg_schedule_max) {
        txt2imgParams.cfg_schedule_max = parseFloat(image.cfg_schedule_max);
      }
      if (image.cfg_schedule_power) {
        txt2imgParams.cfg_schedule_power = parseFloat(image.cfg_schedule_power);
      }
      if (image.cfg_rescale_snr_alpha) {
        txt2imgParams.cfg_rescale_snr_alpha = parseFloat(image.cfg_rescale_snr_alpha);
      }
      if (image.dynamic_threshold_percentile) {
        txt2imgParams.dynamic_threshold_percentile = parseFloat(image.dynamic_threshold_percentile);
        txt2imgParams.dynamic_threshold_mimic_scale = parseFloat(image.dynamic_threshold_mimic_scale || "7.0");
      }

      // Add NAG parameters
      if (image.nag_enable === 'True') {
        txt2imgParams.nag_enable = true;
        txt2imgParams.nag_scale = parseFloat(image.nag_scale || "5.0");
        txt2imgParams.nag_tau = parseFloat(image.nag_tau || "3.5");
        txt2imgParams.nag_alpha = parseFloat(image.nag_alpha || "0.25");
        txt2imgParams.nag_sigma_end = parseFloat(image.nag_sigma_end || "3.0");
      } else {
        txt2imgParams.nag_enable = false;
      }

      // Restore acceleration / determinism-affecting settings
      applyAccelParams(txt2imgParams, image);
    }

    // Save merged params once (only if sendPrompt or sendParameters is checked)
    if (sendPrompt || sendParameters) {
      localStorage.setItem("txt2img_params", JSON.stringify(txt2imgParams));
      // Dispatch custom event for same-tab localStorage change detection
      window.dispatchEvent(new Event("txt2img_params_updated"));
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

    // Build params object by merging prompt and parameters
    const img2imgParams = JSON.parse(localStorage.getItem("img2img_params") || "{}");
    console.log("[ImageGrid] sendToImg2Img - sendPrompt:", sendPrompt, "sendParameters:", sendParameters);
    console.log("[ImageGrid] sendToImg2Img - image.prompt:", image.prompt);

    // Send prompt if checked
    if (sendPrompt) {
      img2imgParams.prompt = image.prompt;
      img2imgParams.negative_prompt = image.negative_prompt;
      console.log("[ImageGrid] sendToImg2Img - Set prompt to:", img2imgParams.prompt);
    }

    // Send parameters if checked
    if (sendParameters) {
      img2imgParams.steps = image.steps;
      img2imgParams.cfg_scale = image.cfg_scale;
      img2imgParams.sampler = image.parameters?.sampler || "euler";
      img2imgParams.schedule_type = image.parameters?.schedule_type || "uniform";
      img2imgParams.seed = image.seed;
      img2imgParams.width = image.width;
      img2imgParams.height = image.height;
      img2imgParams.denoising_strength = 0.75;

      // Add Advanced CFG parameters (always load, even if constant)
      if (image.cfg_schedule_type) {
        img2imgParams.cfg_schedule_type = image.cfg_schedule_type;
      }
      if (image.cfg_schedule_min) {
        img2imgParams.cfg_schedule_min = parseFloat(image.cfg_schedule_min);
      }
      if (image.cfg_schedule_max) {
        img2imgParams.cfg_schedule_max = parseFloat(image.cfg_schedule_max);
      }
      if (image.cfg_schedule_power) {
        img2imgParams.cfg_schedule_power = parseFloat(image.cfg_schedule_power);
      }
      if (image.cfg_rescale_snr_alpha) {
        img2imgParams.cfg_rescale_snr_alpha = parseFloat(image.cfg_rescale_snr_alpha);
      }
      if (image.dynamic_threshold_percentile) {
        img2imgParams.dynamic_threshold_percentile = parseFloat(image.dynamic_threshold_percentile);
        img2imgParams.dynamic_threshold_mimic_scale = parseFloat(image.dynamic_threshold_mimic_scale || "7.0");
      }

      // Add NAG parameters
      if (image.nag_enable === 'True') {
        img2imgParams.nag_enable = true;
        img2imgParams.nag_scale = parseFloat(image.nag_scale || "5.0");
        img2imgParams.nag_tau = parseFloat(image.nag_tau || "3.5");
        img2imgParams.nag_alpha = parseFloat(image.nag_alpha || "0.25");
        img2imgParams.nag_sigma_end = parseFloat(image.nag_sigma_end || "3.0");
      } else {
        img2imgParams.nag_enable = false;
      }

      // Restore acceleration / determinism-affecting settings
      applyAccelParams(img2imgParams, image);
    }

    // Save merged params once (only if sendPrompt or sendParameters is checked)
    if (sendPrompt || sendParameters) {
      console.log("[ImageGrid] sendToImg2Img - Saving merged params:", img2imgParams);
      localStorage.setItem("img2img_params", JSON.stringify(img2imgParams));
      // Dispatch custom event for same-tab localStorage change detection
      window.dispatchEvent(new Event("img2img_params_updated"));
      console.log("[ImageGrid] sendToImg2Img - Dispatched img2img_params_updated event");
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

    // Build params object by merging prompt and parameters
    const inpaintParams = JSON.parse(localStorage.getItem("inpaint_params") || "{}");

    // Send prompt if checked
    if (sendPrompt) {
      inpaintParams.prompt = image.prompt;
      inpaintParams.negative_prompt = image.negative_prompt;
    }

    // Send parameters if checked
    if (sendParameters) {
      inpaintParams.steps = image.steps;
      inpaintParams.cfg_scale = image.cfg_scale;
      inpaintParams.sampler = image.parameters?.sampler || "euler";
      inpaintParams.schedule_type = image.parameters?.schedule_type || "uniform";
      inpaintParams.seed = image.seed;
      inpaintParams.width = image.width;
      inpaintParams.height = image.height;
      inpaintParams.denoising_strength = 0.75;

      // Add Advanced CFG parameters (always load, even if constant)
      if (image.cfg_schedule_type) {
        inpaintParams.cfg_schedule_type = image.cfg_schedule_type;
      }
      if (image.cfg_schedule_min) {
        inpaintParams.cfg_schedule_min = parseFloat(image.cfg_schedule_min);
      }
      if (image.cfg_schedule_max) {
        inpaintParams.cfg_schedule_max = parseFloat(image.cfg_schedule_max);
      }
      if (image.cfg_schedule_power) {
        inpaintParams.cfg_schedule_power = parseFloat(image.cfg_schedule_power);
      }
      if (image.cfg_rescale_snr_alpha) {
        inpaintParams.cfg_rescale_snr_alpha = parseFloat(image.cfg_rescale_snr_alpha);
      }
      if (image.dynamic_threshold_percentile) {
        inpaintParams.dynamic_threshold_percentile = parseFloat(image.dynamic_threshold_percentile);
        inpaintParams.dynamic_threshold_mimic_scale = parseFloat(image.dynamic_threshold_mimic_scale || "7.0");
      }

      // Add NAG parameters
      if (image.nag_enable === 'True') {
        inpaintParams.nag_enable = true;
        inpaintParams.nag_scale = parseFloat(image.nag_scale || "5.0");
        inpaintParams.nag_tau = parseFloat(image.nag_tau || "3.5");
        inpaintParams.nag_alpha = parseFloat(image.nag_alpha || "0.25");
        inpaintParams.nag_sigma_end = parseFloat(image.nag_sigma_end || "3.0");
      } else {
        inpaintParams.nag_enable = false;
      }

      // Restore acceleration / determinism-affecting settings
      applyAccelParams(inpaintParams, image);
    }

    // Save merged params once (only if sendPrompt or sendParameters is checked)
    if (sendPrompt || sendParameters) {
      localStorage.setItem("inpaint_params", JSON.stringify(inpaintParams));
      // Dispatch custom event for same-tab localStorage change detection
      window.dispatchEvent(new Event("inpaint_params_updated"));
    }

    router.push("/generate?tab=inpaint");
  };

  const sendToUpscale = async (image: GeneratedImage) => {
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
            localStorage.setItem("upscale_input_image", tempRef);
            window.dispatchEvent(new Event("upscale_input_updated"));
            resolve(null);
          } catch (error) {
            reject(error);
          }
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error("[ImageGrid] Failed to send image to upscale:", error);
    }

    // Restore upscale-relevant parameters if this image was itself an upscale result
    if (image.generation_type === 'upscale') {
      const upscaleParams = JSON.parse(localStorage.getItem("upscale_params") || "{}");
      if (image.upscaler_backend) upscaleParams.upscaler_backend = image.upscaler_backend;
      if (image.upscaler_model) upscaleParams.upscaler_model = image.upscaler_model;
      if (image.scale_factor) upscaleParams.scale_factor = parseFloat(image.scale_factor);
      if (image.pil_resample) upscaleParams.pil_resample = image.pil_resample;
      if (image.tile_size) upscaleParams.tile_size = parseInt(image.tile_size);
      if (image.tile_overlap) upscaleParams.tile_overlap = parseInt(image.tile_overlap);
      if (image.rtx_vsr_quality) upscaleParams.rtx_vsr_quality = image.rtx_vsr_quality;
      if (image.upscaler_backend === 'diffusion') {
        if (image.prompt !== undefined) upscaleParams.prompt = image.prompt;
        if (image.negative_prompt !== undefined) upscaleParams.negative_prompt = image.negative_prompt;
        if (image.diffusion_denoising_strength) upscaleParams.diffusion_denoising_strength = parseFloat(image.diffusion_denoising_strength);
        if (image.steps !== undefined) upscaleParams.steps = image.steps;
        if (image.cfg_scale !== undefined) upscaleParams.cfg_scale = image.cfg_scale;
        if (image.parameters?.sampler) upscaleParams.sampler = image.parameters.sampler;
        if (image.parameters?.schedule_type) upscaleParams.schedule_type = image.parameters.schedule_type;
        if (image.seed !== undefined) upscaleParams.seed = image.seed;
        if (image.diffusion_pre_upscale_mode) upscaleParams.diffusion_pre_upscale_mode = image.diffusion_pre_upscale_mode;
      }
      localStorage.setItem("upscale_params", JSON.stringify(upscaleParams));
      window.dispatchEvent(new Event("upscale_params_updated"));
    }

    router.push("/generate?tab=upscale");
  };

  // Send a still image to the img2vid panel as a keyframe.
  const sendToImg2Vid = async (image: GeneratedImage) => {
    try {
      await sendImageToImg2Vid(`/outputs/${image.filename}`);
    } catch (error) {
      console.error("[ImageGrid] Failed to send image to img2vid:", error);
    }
    // img2vid was merged into the img2img panel (dual img->img/vid, driven by the
    // loaded model). The keyframe still rides img2vid_input_image / img2vid_input_updated.
    router.push("/generate?tab=img2img");
  };

  // Grab the current frame of the selected video via a canvas. Same-origin
  // /outputs source means the canvas is not tainted, so toDataURL succeeds.
  const captureCurrentFrame = (): string | null => {
    const video = videoRef.current;
    if (!video) return null;
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height) return null;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, width, height);
    try {
      return canvas.toDataURL("image/png");
    } catch (error) {
      console.error("[ImageGrid] Frame capture failed (canvas tainted?):", error);
      return null;
    }
  };

  const captureFrameToImg2Img = async () => {
    const dataUrl = captureCurrentFrame();
    if (!dataUrl) {
      alert("Could not capture the current frame. Wait for the video to load, then try again.");
      return;
    }
    try {
      await sendBase64ImageToImg2Img(dataUrl);
    } catch (error) {
      console.error("[ImageGrid] Failed to send captured frame to img2img:", error);
    }
    router.push("/generate?tab=img2img");
  };

  const captureFrameToImg2Vid = async () => {
    const dataUrl = captureCurrentFrame();
    if (!dataUrl) {
      alert("Could not capture the current frame. Wait for the video to load, then try again.");
      return;
    }
    try {
      await sendBase64ImageToImg2Vid(dataUrl);
    } catch (error) {
      console.error("[ImageGrid] Failed to send captured frame to img2vid:", error);
    }
    // img2vid merged into img2img panel; keyframe rides img2vid_input_image.
    router.push("/generate?tab=img2img");
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
              flex-shrink-0 flex flex-col
            `}>
              {/* Scrollable content area */}
              <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden mb-4 pb-[env(safe-area-inset-bottom)]">
              <Card title="Image Details">
                <div className="space-y-3 text-sm min-w-0 break-words">
                  <div>
                    <span className="text-gray-400">Prompt:</span>
                    <p className="text-gray-100 break-words">{selectedImage.prompt}</p>
                  </div>
                {selectedImage.negative_prompt && (
                  <div>
                    <span className="text-gray-400">Negative Prompt:</span>
                    <p className="text-gray-100 break-words">{selectedImage.negative_prompt}</p>
                  </div>
                )}
                <div className="space-y-2">
                  <div>
                    <span className="text-gray-400">Type:</span> {selectedImage.generation_type}
                  </div>
                  <div>
                    <span className="text-gray-400">Created:</span> {new Date(selectedImage.created_at).toLocaleString()}
                  </div>
                  {/* Video rows: show video-relevant fields and suppress the
                      image-only steps/cfg/sampler/scheduler (which carry
                      meaningless defaults for video). */}
                  {isSelectedVideo ? (
                    <>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <span className="text-gray-400">Frames:</span>{' '}
                          {selectedImage.num_frames ?? selectedImage.parameters?.num_frames}
                        </div>
                        <div>
                          <span className="text-gray-400">FPS:</span>{' '}
                          {selectedImage.fps ?? selectedImage.parameters?.fps}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <span className="text-gray-400">Duration:</span>{' '}
                          {selectedImage.duration ?? selectedImage.parameters?.duration}s
                        </div>
                        <div>
                          <span className="text-gray-400">Audio:</span>{' '}
                          {(selectedImage.audio_enable ?? selectedImage.parameters?.audio_enable) ? 'on' : 'off'}
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
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
                    </>
                  )}
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
                  {/* Advanced CFG parameters */}
                  {selectedImage.cfg_schedule_type && selectedImage.cfg_schedule_type !== 'constant' && (
                    <div>
                      <span className="text-gray-400">CFG Schedule:</span> {selectedImage.cfg_schedule_type} (min: {selectedImage.cfg_schedule_min}, max: {selectedImage.cfg_schedule_max || selectedImage.cfg_scale})
                      {selectedImage.cfg_schedule_type === 'quadratic' && selectedImage.cfg_schedule_power && ` power: ${selectedImage.cfg_schedule_power}`}
                    </div>
                  )}
                  {selectedImage.cfg_rescale_snr_alpha && parseFloat(selectedImage.cfg_rescale_snr_alpha) > 0 && (
                    <div>
                      <span className="text-gray-400">SNR Alpha:</span> {selectedImage.cfg_rescale_snr_alpha}
                    </div>
                  )}
                  {selectedImage.dynamic_threshold_percentile && parseFloat(selectedImage.dynamic_threshold_percentile) > 0 && (
                    <div>
                      <span className="text-gray-400">Dynamic Threshold:</span> {selectedImage.dynamic_threshold_percentile}% (mimic: {selectedImage.dynamic_threshold_mimic_scale || 7.0})
                    </div>
                  )}
                  {/* NAG parameters */}
                  {selectedImage.nag_enable === 'True' && (
                    <div>
                      <span className="text-gray-400">NAG:</span> scale: {selectedImage.nag_scale || 5.0}, tau: {selectedImage.nag_tau || 3.5}, alpha: {selectedImage.nag_alpha || 0.25}, sigma_end: {selectedImage.nag_sigma_end || 3.0}
                    </div>
                  )}
                  {selectedImage.color_flatten_strength && parseFloat(selectedImage.color_flatten_strength) > 0 && (
                    <div>
                      <span className="text-gray-400">Color Flatten:</span> {selectedImage.color_flatten_strength}
                    </div>
                  )}
                  {selectedImage.vae_drift_correction === 'True' && (
                    <div>
                      <span className="text-gray-400">VAE Drift Correction:</span> enabled
                    </div>
                  )}
                  {selectedImage.flatten_in_loop === 'True' && (
                    <div>
                      <span className="text-gray-400">In-loop Background Flatten:</span> last {selectedImage.flatten_in_loop_last_steps || 3} steps, min region {selectedImage.flatten_in_loop_min_region || 0.02}
                    </div>
                  )}
                  {Array.isArray(selectedImage.parameters?.loras) && selectedImage.parameters.loras.length > 0 ? (
                    <div>
                      <span className="text-gray-400">LoRA:</span>{' '}
                      {selectedImage.parameters.loras
                        .map((l: any) => {
                          const name = l?.path ? l.path.split(/[\\/]/).pop() : (l?.name || '');
                          const weight = l?.strength ?? l?.weight;
                          return weight !== undefined && weight !== null ? `${name} (${weight})` : name;
                        })
                        .filter(Boolean)
                        .join(', ')}
                    </div>
                  ) : selectedImage.lora_names && (
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
                  {selectedImage.vision_encoder_name && (
                    <div>
                      <span className="text-gray-400">Vision Encoder:</span>{' '}
                      <span className="text-xs text-white">{selectedImage.vision_encoder_name}</span>
                    </div>
                  )}
                  {selectedImage.vision_encoder_hash && (
                    <div>
                      <span className="text-gray-400">VE Hash:</span>{' '}
                      <span className="text-xs text-gray-100 font-mono" title={selectedImage.vision_encoder_hash}>
                        {selectedImage.vision_encoder_hash.substring(0, 16)}...
                      </span>
                    </div>
                  )}
                  {selectedImage.vae_name && (
                    <div>
                      <span className="text-gray-400">VAE:</span>{' '}
                      <span className="text-xs text-white font-mono" title={selectedImage.vae_name}>{selectedImage.vae_name}</span>
                    </div>
                  )}
                  {selectedImage.vae_hash && (
                    <div>
                      <span className="text-gray-400">VAE Hash:</span>{' '}
                      <span className="text-xs text-gray-100 font-mono" title={selectedImage.vae_hash}>
                        {selectedImage.vae_hash.substring(0, 16)}...
                      </span>
                    </div>
                  )}
                  {selectedImage.unet_quantization && (
                    <div>
                      <span className="text-gray-400">U-Net Quantization:</span> {selectedImage.unet_quantization}
                    </div>
                  )}
                  {selectedImage.effective_warnings && (
                    <div>
                      <span className="text-gray-400">Effective Warnings:</span>{' '}
                      {(() => {
                        let ws: any = selectedImage.effective_warnings;
                        if (typeof ws === 'string') { try { ws = JSON.parse(ws); } catch { ws = []; } }
                        if (!Array.isArray(ws) || ws.length === 0) return null;
                        return (
                          <ul className="mt-1 list-disc list-inside text-xs text-amber-300">
                            {ws.map((w: any, i: number) => (
                              <li key={i}>{typeof w === 'string' ? w : w.message}</li>
                            ))}
                          </ul>
                        );
                      })()}
                    </div>
                  )}
                </div>

                {/* Acceleration / determinism-affecting settings */}
                {(selectedImage.parameters?.spectrum_enable ||
                  selectedImage.parameters?.fbcache_enable ||
                  selectedImage.parameters?.attention_type ||
                  selectedImage.parameters?.attention_impl ||
                  (selectedImage.parameters?.prompt_chunking_mode && selectedImage.parameters.prompt_chunking_mode !== 'a1111') ||
                  (selectedImage.parameters?.max_prompt_chunks !== undefined && selectedImage.parameters.max_prompt_chunks > 0) ||
                  (selectedImage.parameters?.text_encoder_quantization && selectedImage.parameters.text_encoder_quantization !== 'none') ||
                  selectedImage.parameters?.use_tipo ||
                  (selectedImage.parameters?.original_size_w > 0) ||
                  (selectedImage.parameters?.original_size_h > 0) ||
                  (selectedImage.parameters?.original_size_scale !== undefined && selectedImage.parameters.original_size_scale !== 1.0)) && (
                  <div className="border-t border-gray-700 pt-3">
                    <span className="text-gray-400 font-medium">Acceleration:</span>
                    <div className="mt-2 space-y-2 text-xs">
                      {selectedImage.parameters?.spectrum_enable && (
                        <div>
                          <span className="text-gray-500">Spectrum forecasting:</span> enabled (w: {selectedImage.parameters.spectrum_w}, m: {selectedImage.parameters.spectrum_m}, lam: {selectedImage.parameters.spectrum_lam}, warmup: {selectedImage.parameters.spectrum_warmup_steps}, window: {selectedImage.parameters.spectrum_window_size}, flex: {selectedImage.parameters.spectrum_flex_window}, tail: {selectedImage.parameters.spectrum_tail}, mode: {selectedImage.parameters.spectrum_feature_mode}, cache_branch: {selectedImage.parameters.spectrum_cache_branch}, max_cache: {selectedImage.parameters.spectrum_max_cache})
                        </div>
                      )}
                      {selectedImage.parameters?.fbcache_enable && (
                        <div>
                          <span className="text-gray-500">FBCache:</span> enabled (threshold: {selectedImage.parameters.fbcache_threshold}, warmup: {selectedImage.parameters.fbcache_warmup_steps}{selectedImage.parameters.fbcache_cache_branch !== undefined ? `, cache_branch: ${selectedImage.parameters.fbcache_cache_branch}` : ''})
                        </div>
                      )}
                      {selectedImage.parameters?.attention_type && (
                        <div>
                          <span className="text-gray-500">Attention type:</span> {selectedImage.parameters.attention_type}
                        </div>
                      )}
                      {selectedImage.parameters?.attention_impl && (
                        <div>
                          <span className="text-gray-500">Attention impl:</span> {selectedImage.parameters.attention_impl}
                        </div>
                      )}
                      {selectedImage.parameters?.prompt_chunking_mode && selectedImage.parameters.prompt_chunking_mode !== 'a1111' && (
                        <div>
                          <span className="text-gray-500">Prompt chunking:</span> {selectedImage.parameters.prompt_chunking_mode}{selectedImage.parameters.max_prompt_chunks > 0 ? ` (max: ${selectedImage.parameters.max_prompt_chunks})` : ''}
                        </div>
                      )}
                      {(selectedImage.parameters?.prompt_chunking_mode === 'a1111' || selectedImage.parameters?.prompt_chunking_mode === undefined) && selectedImage.parameters?.max_prompt_chunks > 0 && (
                        <div>
                          <span className="text-gray-500">Max prompt chunks:</span> {selectedImage.parameters.max_prompt_chunks}
                        </div>
                      )}
                      {selectedImage.parameters?.text_encoder_quantization && selectedImage.parameters.text_encoder_quantization !== 'none' && (
                        <div>
                          <span className="text-gray-500">TE Quantization:</span> {selectedImage.parameters.text_encoder_quantization}
                        </div>
                      )}
                      {selectedImage.parameters?.use_tipo && (
                        <div>
                          <span className="text-gray-500">TIPO:</span> enabled
                        </div>
                      )}
                      {(selectedImage.parameters?.original_size_w > 0 || selectedImage.parameters?.original_size_h > 0 || (selectedImage.parameters?.original_size_scale !== undefined && selectedImage.parameters.original_size_scale !== 1.0)) && (
                        <div>
                          <span className="text-gray-500">Original size:</span> {selectedImage.parameters?.original_size_w || 0} x {selectedImage.parameters?.original_size_h || 0}{selectedImage.parameters?.original_size_scale !== undefined && selectedImage.parameters.original_size_scale !== 1.0 ? ` (scale: ${selectedImage.parameters.original_size_scale})` : ''}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Generation timing (informational) */}
                {(selectedImage.parameters?.generation_time !== undefined ||
                  selectedImage.parameters?.time_text_encode !== undefined ||
                  selectedImage.parameters?.time_denoise !== undefined ||
                  selectedImage.parameters?.time_vae_decode !== undefined) && (
                  <div className="border-t border-gray-700 pt-3">
                    <span className="text-gray-400 font-medium">Timing:</span>
                    <div className="mt-2 space-y-2 text-xs">
                      {selectedImage.parameters?.generation_time !== undefined && (
                        <div>
                          <span className="text-gray-500">Total:</span> {selectedImage.parameters.generation_time}s
                        </div>
                      )}
                      {selectedImage.parameters?.time_text_encode !== undefined && (
                        <div>
                          <span className="text-gray-500">Text encode:</span> {selectedImage.parameters.time_text_encode}s
                        </div>
                      )}
                      {selectedImage.parameters?.time_denoise !== undefined && (
                        <div>
                          <span className="text-gray-500">Denoise:</span> {selectedImage.parameters.time_denoise}s
                        </div>
                      )}
                      {selectedImage.parameters?.time_vae_decode !== undefined && (
                        <div>
                          <span className="text-gray-500">VAE decode:</span> {selectedImage.parameters.time_vae_decode}s
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Upscale Parameters */}
                {selectedImage.generation_type === 'upscale' && (
                  <div className="border-t border-gray-700 pt-3">
                    <span className="text-gray-400 font-medium">Upscale Parameters:</span>
                    <div className="mt-2 space-y-2 text-xs">
                      {selectedImage.upscaler_backend && (
                        <div>
                          <span className="text-gray-500">Backend:</span> {selectedImage.upscaler_backend}
                        </div>
                      )}
                      {selectedImage.upscaler_model && (
                        <div>
                          <span className="text-gray-500">Model:</span> {selectedImage.upscaler_model}
                        </div>
                      )}
                      {selectedImage.upscaler_model_hash && (
                        <div>
                          <span className="text-gray-500">Model Hash:</span> {selectedImage.upscaler_model_hash.substring(0, 12)}...
                        </div>
                      )}
                      {selectedImage.scale_factor && (
                        <div>
                          <span className="text-gray-500">Scale Factor:</span> {selectedImage.scale_factor}
                        </div>
                      )}
                      {selectedImage.pil_resample && (
                        <div>
                          <span className="text-gray-500">Resample:</span> {selectedImage.pil_resample}
                        </div>
                      )}
                      {selectedImage.tile_size && (
                        <div>
                          <span className="text-gray-500">Tile Size:</span> {selectedImage.tile_size}
                        </div>
                      )}
                      {selectedImage.tile_overlap && (
                        <div>
                          <span className="text-gray-500">Tile Overlap:</span> {selectedImage.tile_overlap}
                        </div>
                      )}
                      {selectedImage.rtx_vsr_quality && (
                        <div>
                          <span className="text-gray-500">RTX VSR Quality:</span> {selectedImage.rtx_vsr_quality}
                        </div>
                      )}
                      {selectedImage.diffusion_denoising_strength && (
                        <div>
                          <span className="text-gray-500">Denoising Strength:</span> {selectedImage.diffusion_denoising_strength}
                        </div>
                      )}
                      {selectedImage.diffusion_pre_upscale_mode && (
                        <div>
                          <span className="text-gray-500">Pre-upscale Mode:</span> {selectedImage.diffusion_pre_upscale_mode}
                        </div>
                      )}
                      {selectedImage.source_image_hash && (
                        <div>
                          <span className="text-gray-500">Source Image Hash:</span> {selectedImage.source_image_hash.substring(0, 12)}...
                        </div>
                      )}
                    </div>
                  </div>
                )}

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
                          {selectedImage.parameters?.inpaint_blur_strength !== undefined && selectedImage.parameters.inpaint_fill_mode === 'blur' && (
                            <div>
                              <span className="text-gray-500">Blur Strength:</span> {selectedImage.parameters.inpaint_blur_strength}
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
                {selectedImage.ref_images && selectedImage.ref_images.length > 0 && (
                  <div>
                    <span className="text-gray-400">Reference Images ({selectedImage.ref_images.length}): </span>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {selectedImage.ref_images.map((hash: string, index: number) => (
                        <button
                          key={index}
                          onClick={() => handleSourceImageClick(hash)}
                          className="text-xs text-blue-400 hover:text-blue-300 font-mono underline break-all"
                          title={`Click to view reference image ${index + 1}\n${hash}`}
                        >
                          ref{index + 1}: {hash.substring(0, 12)}...
                        </button>
                      ))}
                    </div>
                  </div>
                )}

              </div>
              </Card>
              </div>

              {/* Fixed action panel - Desktop only. Two clearly separated
                  groups: "Send to" (checkboxes select what to carry over, then
                  the 4 destination buttons) and "Post-edit" (brightness /
                  saturation / color-flatten sliders + Download, which bakes the
                  active edits into the saved file). */}
              <div className="hidden lg:block lg:flex-shrink-0">
                <Card>
                  <div className="space-y-4">
                    {/* Send to section */}
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-gray-300">Send to</span>
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
                      <div className="grid grid-cols-2 gap-2">
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
                        <Button
                          onClick={() => sendToUpscale(selectedImage)}
                          variant="secondary"
                          size="sm"
                        >
                          Upscale
                        </Button>
                        <Button
                          onClick={() => sendToImg2Vid(selectedImage)}
                          variant="secondary"
                          size="sm"
                          disabled={isSelectedVideo}
                          title={isSelectedVideo ? "Use Capture frame for videos" : "Send image to img2vid as a keyframe"}
                        >
                          img2vid
                        </Button>
                      </div>

                      {/* Video: capture the current frame and send it onward */}
                      {isSelectedVideo && (
                        <div className="border-t border-gray-700 pt-3 space-y-2">
                          <span className="text-xs font-medium text-gray-300">Capture frame</span>
                          <div className="grid grid-cols-2 gap-2">
                            <Button
                              onClick={captureFrameToImg2Img}
                              variant="secondary"
                              size="sm"
                            >
                              <Camera className="h-4 w-4 mr-1" />
                              img2img
                            </Button>
                            <Button
                              onClick={captureFrameToImg2Vid}
                              variant="secondary"
                              size="sm"
                            >
                              <Camera className="h-4 w-4 mr-1" />
                              img2vid
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Post-edit section (Download bakes these edits into the file) */}
                    <div className="border-t border-gray-700 pt-3 space-y-3">
                      <PostEditControls
                        value={postEdit}
                        onChange={setPostEdit}
                        variant="stacked"
                        collapsed={postEditCollapsed}
                        onToggleCollapsed={togglePostEditCollapsed}
                      />
                      <Button
                        onClick={() => handleDownload(selectedImage)}
                        variant="primary"
                        size="sm"
                        className="w-full flex items-center justify-center"
                        title="Download the image with the post-edit adjustments baked in"
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download
                      </Button>
                    </div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Right Area - Image Display with Navigation */}
            <div className="flex-1 flex items-center justify-center bg-gray-900 rounded-lg overflow-hidden relative touch-none">
              {/* Fullscreen Button - Mobile only */}
              <button
                onClick={handleFullscreen}
                className="lg:hidden fixed top-20 right-4 z-40 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg"
                title="Toggle fullscreen mode"
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
                {isSelectedVideo ? (
                  // Same-origin /outputs path so the canvas frame-grab is not tainted.
                  <video
                    ref={videoRef}
                    src={`/outputs/${selectedImage.filename}`}
                    className="max-w-full max-h-full object-contain"
                    controls
                    loop
                    playsInline
                  />
                ) : (
                  <img
                    src={effectiveSelectedSrc ?? `/outputs/${selectedImage.filename}`}
                    alt="Generated"
                    className="max-w-full max-h-full object-contain cursor-pointer"
                    style={{ filter: buildFilterString(postEdit) }}
                    onDoubleClick={() => setShowFullSizeImage(true)}
                    title="Double-click to view full size"
                  />
                )}
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
                {/* Post-edit toggle: small, unobtrusive, docked with download/close.
                    A dot indicates a non-neutral edit while collapsed. */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setPostEditBarExpanded((prev) => !prev);
                  }}
                  className={`absolute top-4 right-36 text-white bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-10 h-10 flex items-center justify-center ${
                    postEditBarExpanded ? "ring-2 ring-blue-500" : ""
                  }`}
                  title="Adjust brightness/saturation"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  {!postEditBarExpanded && !isNeutral(postEdit) && (
                    <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-blue-500" />
                  )}
                </button>
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
                  src={effectiveSelectedSrc ?? `/outputs/${selectedImage.filename}`}
                  alt="Generated - Full Size"
                  className="max-w-full max-h-[90vh] object-contain"
                  style={{ filter: buildFilterString(postEdit) }}
                  onClick={(e) => e.stopPropagation()}
                />
                {/* Post-edit strip: collapsed by default (just the toggle button
                    above) so it never covers the image; expanding shows one
                    compact row flush to the bottom edge. */}
                {postEditBarExpanded && (
                  <div
                    className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 px-3 py-2"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <PostEditControls value={postEdit} onChange={setPostEdit} />
                  </div>
                )}
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
              <button
                onClick={() => sendToUpscale(selectedImage)}
                className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Send to upscale"
              >
                Upscale
              </button>
              {isSelectedVideo ? (
                <>
                  <button
                    onClick={captureFrameToImg2Img}
                    className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded flex items-center gap-1"
                    title="Capture current frame to img2img"
                  >
                    <Camera className="h-4 w-4" />
                    img2img
                  </button>
                  <button
                    onClick={captureFrameToImg2Vid}
                    className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded flex items-center gap-1"
                    title="Capture current frame to img2vid"
                  >
                    <Camera className="h-4 w-4" />
                    img2vid
                  </button>
                </>
              ) : (
                <button
                  onClick={() => sendToImg2Vid(selectedImage)}
                  className="px-3 py-2 text-sm bg-gray-800 hover:bg-gray-700 text-white rounded"
                  title="Send image to img2vid as a keyframe"
                >
                  img2vid
                </button>
              )}
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
            filterTxt2Vid={filterTxt2Vid}
            setFilterTxt2Vid={setFilterTxt2Vid}
            filterImg2Vid={filterImg2Vid}
            setFilterImg2Vid={setFilterImg2Vid}
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
