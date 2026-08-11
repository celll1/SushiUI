"use client";

import { useState, useEffect } from "react";
import { Image as ImageIcon } from "lucide-react";
import ImageViewer from "./ImageViewer";
import { isVideoUrl, isAudioUrl, posterUrlForVideo } from "@/utils/previewStorage";

/** One entry in the shared top-right recent-results strip. */
export interface GalleryEntry {
  url: string;
  timestamp: number;
  // When the producing panel knows the kind up front it is passed through
  // directly; when absent (e.g. an older caller) it is inferred from `url`'s
  // extension, same as before.
  kind?: "image" | "video" | "audio";
  // Browser-playable URL, when it differs from `url` (e.g. `url` is a
  // video_lossless FFV1-in-mkv master no browser can decode, and this is its
  // H.264 mp4 proxy). Falls back to `url` when absent.
  playbackUrl?: string;
}

interface FloatingGalleryProps {
  images: GalleryEntry[];
  maxImages: number;
}

export default function FloatingGallery({ images, maxImages }: FloatingGalleryProps) {
  const [viewerImageIndex, setViewerImageIndex] = useState<number | null>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [isEditorOpen, setIsEditorOpen] = useState(false);

  // Limit to most recent results
  const displayImages = images.slice(-maxImages);

  // Monitor editor state changes
  useEffect(() => {
    const checkEditorState = () => {
      const editorOpen =
        document.body.dataset.promptEditorOpen === 'true' ||
        document.body.dataset.imageEditorOpen === 'true';
      setIsEditorOpen(editorOpen);
    };

    // Check immediately
    checkEditorState();

    // Set up a MutationObserver to watch for dataset changes
    const observer = new MutationObserver(checkEditorState);
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['data-prompt-editor-open', 'data-image-editor-open']
    });

    return () => observer.disconnect();
  }, []);

  if (displayImages.length === 0) {
    return null;
  }

  const kindOf = (entry: GalleryEntry): "image" | "video" | "audio" =>
    entry.kind ?? (isVideoUrl(entry.url) ? "video" : isAudioUrl(entry.url) ? "audio" : "image");

  const playbackUrlOf = (entry: GalleryEntry): string => entry.playbackUrl || entry.url;

  const handleNavigate = (direction: 'prev' | 'next') => {
    if (viewerImageIndex === null) return;

    if (direction === 'prev' && viewerImageIndex > 0) {
      setViewerImageIndex(viewerImageIndex - 1);
    } else if (direction === 'next' && viewerImageIndex < displayImages.length - 1) {
      setViewerImageIndex(viewerImageIndex + 1);
    }
  };

  return (
    <>
      {/* Mobile gallery toggle button */}
      {!isEditorOpen && (
        <button
          onClick={() => setIsGalleryOpen(!isGalleryOpen)}
          className="fixed top-4 right-4 z-50 p-3 rounded-lg bg-gray-800 bg-opacity-90 text-white shadow-lg lg:hidden"
          aria-label="Toggle gallery"
        >
          <ImageIcon className="h-5 w-5" />
          {displayImages.length > 0 && (
            <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center">
              {displayImages.length}
            </span>
          )}
        </button>
      )}

      {/* Gallery panel - collapsible on mobile, always visible on desktop */}
      <div className={`
        fixed top-4 right-4 z-40 bg-gray-800 rounded-lg shadow-lg p-2
        transition-all duration-200 ease-in-out
        ${isGalleryOpen ? 'translate-x-0' : 'translate-x-[calc(100%+1rem)]'}
        lg:translate-x-0
        max-w-[80vw] lg:max-w-[60vw]
        ${isEditorOpen ? 'lg:hidden' : ''}
      `}>
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
          {displayImages.map((image, index) => {
            const kind = kindOf(image);
            const playbackUrl = playbackUrlOf(image);
            return (
              <div
                key={`${image.timestamp}-${index}`}
                className="flex-shrink-0 hover:opacity-80 transition-opacity cursor-pointer"
                onDoubleClick={() => setViewerImageIndex(index)}
              >
                {kind === "video" ? (
                  // Static tile: `poster` points at the poster PNG the backend
                  // writes next to every clip (keyed off the master `url`, not
                  // the playback proxy), so a thumbnail shows without fetching
                  // video data (a missing poster degrades to the browser's own
                  // first-frame handling). `src` is the playback URL, since a
                  // video_lossless master (FFV1-in-mkv) is not browser-playable.
                  // Deliberately not autoplaying -- this strip can hold dozens
                  // of results.
                  <video
                    src={playbackUrl}
                    poster={posterUrlForVideo(image.url)}
                    className="h-24 w-auto object-contain rounded border border-gray-700"
                    preload="metadata"
                    muted
                    playsInline
                  />
                ) : kind === "audio" ? (
                  <audio src={playbackUrl} controls className="h-24 w-56 rounded border border-gray-700" />
                ) : (
                  <img
                    src={image.url}
                    alt={`Generated ${index + 1}`}
                    className="h-24 w-auto object-contain rounded border border-gray-700"
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {viewerImageIndex !== null && displayImages[viewerImageIndex] && (
        <ImageViewer
          imageUrl={playbackUrlOf(displayImages[viewerImageIndex])}
          kind={kindOf(displayImages[viewerImageIndex])}
          posterUrl={
            kindOf(displayImages[viewerImageIndex]) === "video"
              ? posterUrlForVideo(displayImages[viewerImageIndex].url)
              : undefined
          }
          onClose={() => setViewerImageIndex(null)}
          onNavigate={handleNavigate}
          hasPrev={viewerImageIndex > 0}
          hasNext={viewerImageIndex < displayImages.length - 1}
        />
      )}
    </>
  );
}
