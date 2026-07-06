"use client";

import { useEffect, useState } from "react";
import { Download, SlidersHorizontal } from "lucide-react";
import { PostEditState, isNeutral, applyPostEdit, buildFilterString, editedFilename } from "@/utils/postEdit";
import PostEditControls from "./PostEditControls";

interface ImageViewerProps {
  imageUrl: string;
  onClose: () => void;
  onNavigate?: (direction: 'prev' | 'next') => void;
  hasPrev?: boolean;
  hasNext?: boolean;
  // Optional client-side post-edit (brightness/saturation). When both are
  // provided, controls render in the toolbar, the CSS filter is applied to the
  // preview, and downloads bake the adjustments. When absent, ImageViewer
  // behaves exactly as before (e.g. FloatingGallery consumer).
  postEdit?: PostEditState;
  onPostEditChange?: (value: PostEditState) => void;
}

export default function ImageViewer({ imageUrl, onClose, onNavigate, hasPrev, hasNext, postEdit, onPostEditChange }: ImageViewerProps) {
  // Post-edit strip is collapsed by default so it never obscures the image;
  // this is purely internal UI state (not one of the optional postEdit props).
  const [postEditExpanded, setPostEditExpanded] = useState(false);
  const postEditNonNeutral = postEdit ? !isNeutral(postEdit) : false;

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();

    try {
      // Get metadata setting from localStorage
      const includeMetadata = localStorage.getItem('include_metadata_in_downloads') === 'true';

      // Extract filename from imageUrl
      const filename = imageUrl.split('/').pop() || 'image.png';

      // Use API endpoint for metadata-aware download
      const downloadUrl = `/api/download/${filename}?include_metadata=${includeMetadata}`;

      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`);
      }

      let blob = await response.blob();
      let downloadName = filename;

      // Bake post-edit adjustments only when non-neutral. Neutral -> original
      // blob unchanged (metadata preserved). Baking re-encodes the PNG and
      // loses embedded metadata (see postEdit.ts).
      if (postEdit && !isNeutral(postEdit)) {
        blob = await applyPostEdit(blob, postEdit);
        downloadName = editedFilename(filename);
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
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      } else if (e.key === "ArrowLeft" && hasPrev && onNavigate) {
        onNavigate('prev');
      } else if (e.key === "ArrowRight" && hasNext && onNavigate) {
        onNavigate('next');
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, onNavigate, hasPrev, hasNext]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="relative max-w-[95vw] max-h-[95vh] flex items-center">
        {/* Previous button */}
        {hasPrev && onNavigate && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('prev');
            }}
            className="absolute left-4 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center z-10"
            title="Previous (Left Arrow)"
          >
            ‹
          </button>
        )}

        <img
          src={imageUrl}
          alt="Full size preview"
          className="max-w-full max-h-[95vh] object-contain"
          style={postEdit ? { filter: buildFilterString(postEdit) } : undefined}
          onClick={(e) => e.stopPropagation()}
        />

        {/* Post-edit strip: collapsed by default (just the toggle button below)
            so the image is never obscured. Expanding shows one compact row
            flush to the bottom edge, which the user can collapse again. */}
        {postEdit && onPostEditChange && postEditExpanded && (
          <div
            className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 px-3 py-2"
            onClick={(e) => e.stopPropagation()}
          >
            <PostEditControls value={postEdit} onChange={onPostEditChange} />
          </div>
        )}

        {/* Next button */}
        {hasNext && onNavigate && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('next');
            }}
            className="absolute right-4 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center z-10"
            title="Next (Right Arrow)"
          >
            ›
          </button>
        )}

        {/* Post-edit toggle: small, unobtrusive, docked in the toolbar with
            download/close. A dot indicates a non-neutral edit while collapsed. */}
        {postEdit && onPostEditChange && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setPostEditExpanded((prev) => !prev);
            }}
            className={`absolute top-4 right-36 text-white bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center ${
              postEditExpanded ? "ring-2 ring-blue-500" : ""
            }`}
            title="Adjust brightness/saturation"
          >
            <SlidersHorizontal className="h-5 w-5" />
            {!postEditExpanded && postEditNonNeutral && (
              <span className="absolute top-2 right-2 w-2 h-2 rounded-full bg-blue-500" />
            )}
          </button>
        )}

        {/* Download button */}
        <button
          onClick={handleDownload}
          className="absolute top-4 right-20 text-white bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
          title="Download"
        >
          <Download className="h-6 w-6" />
        </button>

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-white text-3xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
          title="Close (Escape)"
        >
          ×
        </button>
      </div>
    </div>
  );
}
