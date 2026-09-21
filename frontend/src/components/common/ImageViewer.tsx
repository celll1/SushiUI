"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, SlidersHorizontal } from "lucide-react";
import { PostEditState, isNeutral, applyPostEdit, buildFilterString, editedFilename } from "@/utils/postEdit";
import PostEditControls from "./PostEditControls";
import { usePostEditPreview } from "@/hooks/usePostEditPreview";

const VIEWER_OPEN_EVENT = "sushiui:image-viewer-open";
let bodyLockCount = 0;
let bodyOverflowBeforeLock = "";

function acquireBodyScrollLock() {
  if (bodyLockCount === 0) {
    bodyOverflowBeforeLock = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  bodyLockCount += 1;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    bodyLockCount = Math.max(0, bodyLockCount - 1);
    if (bodyLockCount === 0) {
      document.body.style.overflow = bodyOverflowBeforeLock;
    }
  };
}

interface ImageViewerProps {
  // For kind="video"/"audio" this is the browser-playable URL to render (the
  // caller resolves playbackUrl vs. url before passing it in).
  imageUrl: string;
  // Defaults to "image" so every existing caller (which omits this prop) keeps
  // today's <img>-only behavior unchanged.
  kind?: "image" | "video" | "audio";
  // Video-only poster frame; ignored for "image"/"audio".
  posterUrl?: string;
  onClose: () => void;
  onNavigate?: (direction: 'prev' | 'next') => void;
  hasPrev?: boolean;
  hasNext?: boolean;
  // Optional client-side post-edit (brightness/saturation). When both are
  // provided, controls render in the toolbar, the CSS filter is applied to the
  // preview, and downloads bake the adjustments. When absent, ImageViewer
  // behaves exactly as before (e.g. FloatingGallery consumer).
  // Image-only: post-edit is a pixel operation on decoded image data and the
  // download button re-encodes a PNG, so neither applies to video/audio and
  // both are hidden whenever kind !== "image" (see render below), regardless
  // of whether these props are supplied.
  postEdit?: PostEditState;
  onPostEditChange?: (value: PostEditState) => void;
  // The download button resolves the filename against the outputs directory
  // (`/api/download/{filename}`), so callers showing images served from
  // elsewhere (e.g. training samples) turn it off.
  showDownload?: boolean;
}

// Arrow keys belong to a focused text field, not to the viewer. A range input
// is not text entry: the viewer takes the key and preventDefault below stops
// the slider behind the overlay moving a second time.
const isTextEntry = (target: EventTarget | null) =>
  target instanceof HTMLTextAreaElement ||
  (target instanceof HTMLInputElement && target.type !== "range");

export default function ImageViewer({ imageUrl, kind = "image", posterUrl, onClose, onNavigate, hasPrev, hasNext, postEdit, onPostEditChange, showDownload = true }: ImageViewerProps) {
  // Post-edit strip is collapsed by default so it never obscures the image;
  // this is purely internal UI state (not one of the optional postEdit props).
  const [postEditExpanded, setPostEditExpanded] = useState(false);
  const [mediaFailed, setMediaFailed] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const instanceIdRef = useRef(Symbol("image-viewer"));
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const postEditNonNeutral = postEdit ? !isNeutral(postEdit) : false;

  // Color-flatten preview: swaps in a processed object URL when flatten>0.
  // brightness/saturation stay as a CSS filter layered on top (below).
  // Only meaningful for images; passing null for video/audio skips the fetch
  // path in usePostEditPreview entirely (flatten<=0 there is already a no-op,
  // this just avoids treating a video/audio URL as decodable image data).
  const effectiveImageUrl = usePostEditPreview(kind === "image" ? imageUrl : null, postEdit?.flatten ?? 0);

  useEffect(() => {
    setMediaFailed(false);
  }, [effectiveImageUrl, imageUrl, kind]);

  useEffect(() => {
    const releaseBodyScrollLock = acquireBodyScrollLock();
    closeButtonRef.current?.focus();
    return releaseBodyScrollLock;
  }, []);

  useEffect(() => {
    const instanceId = instanceIdRef.current;
    const closeSupersededViewer = (event: Event) => {
      if ((event as CustomEvent<symbol>).detail !== instanceId) onCloseRef.current();
    };
    window.addEventListener(VIEWER_OPEN_EVENT, closeSupersededViewer);
    window.dispatchEvent(new CustomEvent(VIEWER_OPEN_EVENT, { detail: instanceId }));
    return () => window.removeEventListener(VIEWER_OPEN_EVENT, closeSupersededViewer);
  }, []);

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();

    try {
      // Get metadata setting from localStorage
      const includeMetadata = localStorage.getItem('include_metadata_in_downloads') === 'true';

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
      } else if (isTextEntry(e.target)) {
        return;
      } else if (e.key === "ArrowLeft" && hasPrev && onNavigate) {
        e.preventDefault();
        onNavigate('prev');
      } else if (e.key === "ArrowRight" && hasNext && onNavigate) {
        e.preventDefault();
        onNavigate('next');
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose, onNavigate, hasPrev, hasNext]);

  if (typeof document === "undefined") return null;

  const handleViewerPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = event.target;
    if (target instanceof Element && target.closest('[data-image-viewer-interactive="true"]')) {
      return;
    }
    // Dismiss in capture phase.  A decoded image can cover almost the entire
    // viewport; if its paint fails, it must not become an invisible click trap.
    event.preventDefault();
    event.stopPropagation();
    onClose();
  };

  return createPortal(
    <div
      className="fixed inset-0 isolate overflow-hidden bg-black bg-opacity-90 pointer-events-auto"
      style={{ zIndex: 2147483647, overscrollBehavior: "contain" }}
      onPointerDownCapture={handleViewerPointerDown}
      role="dialog"
      aria-modal="true"
      aria-label="Full size media preview"
    >
      <div className="relative flex h-full w-full min-h-0 min-w-0 items-center justify-center p-4">
        {/* Previous button */}
        {hasPrev && onNavigate && (
          <button
            data-image-viewer-interactive="true"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('prev');
            }}
            className="absolute left-4 z-20 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center"
            title="Previous (Left Arrow)"
          >
            ‹
          </button>
        )}

        {kind === "video" ? (
          <video
            data-image-viewer-interactive="true"
            src={imageUrl}
            poster={posterUrl}
            className="max-w-full max-h-full object-contain"
            controls
            autoPlay
            playsInline
            onClick={(e) => e.stopPropagation()}
            onError={() => setMediaFailed(true)}
          />
        ) : kind === "audio" ? (
          <audio
            data-image-viewer-interactive="true"
            src={imageUrl}
            className="w-[80vw] max-w-xl"
            controls
            autoPlay
            onClick={(e) => e.stopPropagation()}
            onError={() => setMediaFailed(true)}
          />
        ) : (
          <img
            src={effectiveImageUrl ?? imageUrl}
            alt="Full size preview"
            className={`max-w-full max-h-full object-contain ${mediaFailed ? "hidden" : ""}`}
            style={postEdit ? { filter: buildFilterString(postEdit) } : undefined}
            onLoad={() => setMediaFailed(false)}
            onError={() => setMediaFailed(true)}
          />
        )}

        {mediaFailed && (
          <div
            data-image-viewer-interactive="true"
            className="rounded-lg bg-gray-900 px-6 py-5 text-center text-sm text-gray-200 shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <p>The full-size media could not be loaded.</p>
            <p className="mt-1 text-xs text-gray-400">Close this viewer and refresh the preview.</p>
          </div>
        )}

        {/* Post-edit strip: collapsed by default (just the toggle button below)
            so the image is never obscured. Expanding shows one compact row
            flush to the bottom edge, which the user can collapse again.
            Image-only (see ImageViewerProps.postEdit doc). */}
        {kind === "image" && postEdit && onPostEditChange && postEditExpanded && (
          <div
            data-image-viewer-interactive="true"
            className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 px-3 py-2"
            onClick={(e) => e.stopPropagation()}
          >
            <PostEditControls value={postEdit} onChange={onPostEditChange} />
          </div>
        )}

        {/* Next button */}
        {hasNext && onNavigate && (
          <button
            data-image-viewer-interactive="true"
            onClick={(e) => {
              e.stopPropagation();
              onNavigate('next');
            }}
            className="absolute right-4 z-20 text-white text-4xl font-bold bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-14 h-14 flex items-center justify-center"
            title="Next (Right Arrow)"
          >
            ›
          </button>
        )}

        {/* Post-edit toggle: small, unobtrusive, docked in the toolbar with
            download/close. A dot indicates a non-neutral edit while collapsed.
            Image-only. */}
        {kind === "image" && postEdit && onPostEditChange && (
          <button
            data-image-viewer-interactive="true"
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

        {/* Download button: image-only (re-encodes a PNG via the metadata-aware
            download endpoint, which is not meaningful for video/audio). */}
        {kind === "image" && showDownload && (
          <button
            data-image-viewer-interactive="true"
            onClick={handleDownload}
            className="absolute top-4 right-20 text-white bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full w-12 h-12 flex items-center justify-center"
            title="Download"
          >
            <Download className="h-6 w-6" />
          </button>
        )}

        {/* Close button */}
        <button
          data-image-viewer-interactive="true"
          onClick={onClose}
          ref={closeButtonRef}
          type="button"
          className="fixed top-4 right-4 z-50 text-white text-3xl font-bold bg-black bg-opacity-70 hover:bg-opacity-90 rounded-full w-12 h-12 flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-white"
          title="Close (Escape)"
          aria-label="Close full-size preview"
        >
          ×
        </button>
      </div>
    </div>,
    document.body,
  );
}
